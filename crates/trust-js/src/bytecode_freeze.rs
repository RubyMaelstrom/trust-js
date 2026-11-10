//! The optimizing tier's frozen view of a chunk, made on the main thread (P3 stage 2, step 12;
//! `opt::view`): a child of `bytecode`, which reads the chunk's fields as its own accessors do.

use super::{named_site_cache, Chunk, Op};
use crate::opt::view::{FrozenChunk, FrozenConst, InlineTargetView};

impl Chunk {
    /// The tier's frozen view of this chunk (see `opt::view`): its immutable data copied, its
    /// lazily built facts forced, the live state the back end bakes in read now, and its cells'
    /// and images' addresses as words. Main thread only: it upgrades the inline targets' pins and
    /// reads the agent heap's allocation sites.
    pub(crate) fn freeze(&self, interns: bool) -> FrozenChunk {
        let ops = self.ops.as_slice();
        let len = ops.len();
        let mut names = String::new();
        let mut name_ends = Vec::with_capacity(self.names.len());
        for name in &self.names {
            names.push_str(name);
            name_ends.push(names.len() as u32);
        }
        let consts = self
            .consts
            .iter()
            .enumerate()
            .map(|(k, _)| {
                let k = k as u32;
                FrozenConst {
                    copyable: self.jit_const_copyable(k),
                    scalar_bits: self.jit_const_scalar_bits(k),
                    is_str: self.jit_const_is_str(k),
                    str_word: self.jit_const_str_word(k),
                }
            })
            .collect();
        let fixed_activation = self.jit_fixed_activation();
        // The captured bindings the cap operations' templates read (`jit_operations::fixed_cap`):
        // their names' slots in the fixed activation's plan.
        let mut cap_bindings = Vec::new();
        if fixed_activation {
            cap_bindings = vec![None; self.names.len()];
            for op in ops {
                if let Op::LoadCap(name)
                | Op::StoreCap(name)
                | Op::StoreCapInit(name)
                | Op::UpdateCap(name, _) = *op
                {
                    if let Some(binding) = cap_bindings.get_mut(name as usize) {
                        *binding = self.jit_cap_binding(name);
                    }
                }
            }
        }
        // The named operations' names' keys, where their atoms exist (none is made here).
        let mut keys = vec![None; self.names.len()];
        for op in ops {
            if let Some(n) = super::named_site_name(op) {
                if let Some(key) = keys.get_mut(n as usize) {
                    if key.is_none() {
                        *key = crate::atom::PropKey::lookup_text(&self.names[n as usize])
                            .map(|key| key.word());
                    }
                }
            }
        }
        // The named sites' ways, which the property templates bake (read once, here).
        #[allow(clippy::type_complexity)]
        let mut sites: Vec<(
            u32,
            Option<super::IcState>,
            [super::IcState; super::PROP_IC_WAYS],
        )> = ops
            .iter()
            .filter_map(named_site_cache)
            .map(|cache| {
                (
                    cache,
                    self.jit_cache_preferred(cache),
                    self.jit_cache_ways(cache),
                )
            })
            .collect();
        sites.sort_unstable_by_key(|&(cache, _, _)| cache);
        sites.dedup_by_key(|&mut (cache, _, _)| cache);
        let mut resolution_sites: Vec<(u32, usize)> = ops
            .iter()
            .filter_map(|op| match *op {
                Op::ResolveNameRef(name, _) => Some((name, self.jit_resolution_cache_ptr(name))),
                _ => None,
            })
            .collect();
        resolution_sites.sort_unstable_by_key(|&(name, _)| name);
        resolution_sites.dedup_by_key(|&mut (name, _)| name);
        let inline_targets = self
            .inline_targets
            .iter()
            .enumerate()
            .map(|(t, target)| InlineTargetView {
                argc: target.argc,
                check_this: target.check_this,
                expected_env: target.expected_env,
                expected_genv: target.expected_genv,
                // The stored word of an `Option<Gc>`, as the first stage's InlineGuard reads it
                // (`opt::a64`'s callee checks).
                callee: target.pin.upgrade().map(|object| {
                    let some: Option<crate::value::Gc> = Some(object);
                    unsafe { *(&some as *const Option<crate::value::Gc> as *const usize) }
                }),
                env: self
                    .inline_target_env(t as u32)
                    .map(|env| crate::interpreter::Env::as_ptr(&env) as usize),
            })
            .collect();
        // The in-line allocation sites of the agent heap, where a literal may use one.
        let literals = !self.feedback.detailed_enabled()
            && ops.iter().any(|op| {
                matches!(
                    op,
                    Op::MakeObject(..) | Op::MakeArray(..) | Op::MakeArrayLiteral(_)
                )
            });
        let mut object_sites = [None; 4];
        let mut words_sites = [None; super::MAX_INLINE_ELEMENTS + 1];
        if literals {
            for (class, site) in object_sites.iter_mut().enumerate() {
                *site = crate::value::object_inline_alloc_site(class as u32);
            }
            for (words, site) in words_sites.iter_mut().enumerate().skip(1) {
                *site = crate::value::words_inline_alloc_site(words);
            }
        }
        let label = format!(
            "fn({})",
            self.slot_names
                .iter()
                .take(4)
                .map(|name| &**name)
                .collect::<Vec<_>>()
                .join(",")
        );
        FrozenChunk {
            ops: ops.into(),
            names: names.into_boxed_str(),
            name_ends: name_ends.into_boxed_slice(),
            keys: keys.into_boxed_slice(),
            interns,
            missing_keys: Default::default(),
            consts,
            n_params: self.n_params,
            n_slots: self.n_slots,
            resumable: self.resumable,
            prepared_entry: self.prepared_entry,
            detailed_feedback: self.feedback.detailed_enabled(),
            second_stage: self.second_stage,
            strict: self.strict,
            needs_activation_state: self.jit_needs_activation_state,
            fixed_activation,
            frame_setup_needed: self.jit_frame_setup_needed(),
            frame_args_needed: self.jit_frame_args_needed(),
            arguments_count_slot: self.jit_arguments_count_slot(),
            inline_activation: self.jit_inline_activation(),
            prologue_copies: self.jit_prologue_copies(),
            mapped_formals: self.jit_mapped_formals(),
            stack_effects: (0..len)
                .map(|pc| {
                    self.jit_stack_effect(pc)
                        .map(|(pops, pushes)| (pops as u32, pushes as u32))
                })
                .collect(),
            cap_initialized: self
                .jit_cap_tdz_known
                .get_or_init(|| super::cap_tdz_known(&self.ops))
                .clone(),
            cap_bindings: cap_bindings.into_boxed_slice(),
            name_statics: self.name_statics.clone(),
            name_globals: self.name_globals.clone(),
            caches_base: self.caches.as_ptr() as usize,
            name_caches_base: self.name_caches.as_ptr() as usize,
            cap_caches_base: self.cap_caches.as_ptr() as usize,
            construct_caches_base: self.construct_caches.as_ptr() as usize,
            call_caches: self
                .call_caches
                .iter()
                .map(|site| site.entries.as_ptr() as usize)
                .collect(),
            sites: sites.into_boxed_slice(),
            call_positions: self.call_positions.clone(),
            construct_positions: self.construct_positions.clone(),
            inline_targets,
            object_images: self
                .obj_images
                .iter()
                .map(|image| std::ptr::from_ref(image) as usize)
                .collect(),
            array_images: self
                .array_images
                .iter()
                .map(|image| std::ptr::from_ref(image) as usize)
                .collect(),
            empty_array_image: std::ptr::from_ref(&self.empty_array_image) as usize,
            object_sites,
            words_sites,
            resolution_sites: resolution_sites.into_boxed_slice(),
            names_base: self.names.as_ptr() as usize,
            name_slot_size: std::mem::size_of::<std::rc::Rc<str>>(),
            attempted_off: self.jit_inline_attempted_off(),
            runs_off: self.jit_runs_off(),
            retry_off: self.jit_inline_retry_at_off(),
            label,
            #[cfg(test)]
            test_this_commits: self.test_this_commits.as_ptr() as usize,
        }
    }
}
