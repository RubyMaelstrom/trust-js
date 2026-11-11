//! The optimizing tier's frozen view of a chunk, made on the main thread (P3 stage 2, step 12;
//! `opt::view`): a child of `bytecode`, which reads the chunk's fields as its own accessors do.

use super::{named_site_cache, Chunk, Op};
use crate::opt::view::{FrozenChunk, FrozenConst, InlineTargetView};

impl Chunk {
    /// The tier's frozen view of this chunk, in a view of its own ([`Chunk::freeze_into`]): the
    /// tests'.
    #[cfg(test)]
    pub(crate) fn freeze(&self) -> FrozenChunk {
        let mut view = FrozenChunk::default();
        self.freeze_into(&mut view);
        view
    }

    /// The tier's frozen view of this chunk (see `opt::view`), made in `view`, whose vectors it
    /// fills again, their capacities kept, so that a job reuses a finished one's memory (P3 stage
    /// 2, step 16, point 5): the chunk's immutable data copied, its lazily built facts forced (but
    /// those of the operations alone, which the compile makes), the live state the back end bakes
    /// in read now, and its cells' and images' addresses as words; the names' keys from the
    /// chunk's table, made now if none was (step 16). Main thread only: it upgrades the inline
    /// targets' pins, reads the agent heap's allocation sites and may make atoms.
    pub(crate) fn freeze_into(&self, view: &mut FrozenChunk) {
        let ops = self.ops.as_slice();
        view.ops.clear();
        view.ops.extend_from_slice(ops);
        view.names.clear();
        view.name_ends.clear();
        view.name_ends.reserve(self.names.len());
        for name in &self.names {
            view.names.push_str(name);
            view.name_ends.push(view.names.len() as u32);
        }
        // The constants the operations push, the only ones a compile reads.
        view.consts.clear();
        view.consts.resize(self.consts.len(), None);
        for op in ops {
            if let Op::Const(k) = *op {
                if let Some(slot @ None) = view.consts.get_mut(k as usize) {
                    *slot = Some(FrozenConst {
                        copyable: self.jit_const_copyable(k),
                        scalar_bits: self.jit_const_scalar_bits(k),
                        is_str: self.jit_const_is_str(k),
                        str_word: self.jit_const_str_word(k),
                    });
                }
            }
        }
        let fixed_activation = self.jit_fixed_activation();
        // The captured bindings the cap operations' templates read (`jit_operations::fixed_cap`):
        // their names' slots in the fixed activation's plan.
        view.cap_bindings.clear();
        if fixed_activation {
            view.cap_bindings.resize(self.names.len(), None);
            for op in ops {
                if let Op::LoadCap(name)
                | Op::StoreCap(name)
                | Op::StoreCapInit(name)
                | Op::UpdateCap(name, _) = *op
                {
                    if let Some(binding) = view.cap_bindings.get_mut(name as usize) {
                        *binding = self.jit_cap_binding(name);
                    }
                }
            }
        }
        // The names' keys, from the chunk's table, made once a chunk (P3 stage 2, step 16): its
        // first freeze makes it where the VM has not, as the VM's first named access would, and a
        // second stage's recompile carries it (`Chunk::carry_name_keys`), as the exemplars' compiles
        // read a name's atom from the bytecode. No compile looks a name up or makes an atom.
        view.keys.clear();
        view.keys
            .extend(self.name_keys().iter().map(|key| key.word()));
        // The named sites' ways, read once, here: the property templates bake them, with each
        // site's preferred way (`Chunk::jit_cache_preferred`), and the snapshot's decode reads
        // them (P3 stage 2, step 16: `feedback_read::CopiedWords`).
        view.sites.clear();
        view.sites
            .extend(ops.iter().filter_map(named_site_cache).map(|cache| {
                let ways = self.jit_cache_ways(cache);
                (cache, super::preferred_way(&ways), ways)
            }));
        view.sites.sort_unstable_by_key(|&(cache, _, _)| cache);
        view.sites.dedup_by_key(|&mut (cache, _, _)| cache);
        view.resolution_sites.clear();
        view.resolution_sites
            .extend(ops.iter().filter_map(|op| match *op {
                Op::ResolveNameRef(name, _) => Some((name, self.jit_resolution_cache_ptr(name))),
                _ => None,
            }));
        view.resolution_sites
            .sort_unstable_by_key(|&(name, _)| name);
        view.resolution_sites.dedup_by_key(|&mut (name, _)| name);
        view.inline_targets.clear();
        view.inline_targets
            .extend(self.inline_targets.iter().enumerate().map(|(t, target)| {
                InlineTargetView {
                    argc: target.argc,
                    check_this: target.check_this,
                    expected_env: target.expected_env,
                    expected_genv: target.expected_genv,
                    // The stored word of an `Option<Gc>`, as the first stage's InlineGuard reads
                    // it (`opt::a64`'s callee checks).
                    callee: target.pin.upgrade().map(|object| {
                        let some: Option<crate::value::Gc> = Some(object);
                        unsafe { *(&some as *const Option<crate::value::Gc> as *const usize) }
                    }),
                    env: self
                        .inline_target_env(t as u32)
                        .map(|env| crate::interpreter::Env::as_ptr(&env) as usize),
                }
            }));
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
        view.label.clear();
        view.label.push_str("fn(");
        for (k, name) in self.slot_names.iter().take(4).enumerate() {
            if k > 0 {
                view.label.push(',');
            }
            view.label.push_str(name);
        }
        view.label.push(')');
        view.name_statics.clear();
        view.name_statics.extend_from_slice(&self.name_statics);
        view.name_globals.clear();
        view.name_globals.extend_from_slice(&self.name_globals);
        view.call_caches.clear();
        view.call_caches.extend(
            self.call_caches
                .iter()
                .map(|site| site.entries.as_ptr() as usize),
        );
        view.call_positions.clear();
        view.call_positions.extend_from_slice(&self.call_positions);
        view.construct_positions.clear();
        view.construct_positions
            .extend_from_slice(&self.construct_positions);
        view.object_images.clear();
        view.object_images.extend(
            self.obj_images
                .iter()
                .map(|image| std::ptr::from_ref(image) as usize),
        );
        view.array_images.clear();
        view.array_images.extend(
            self.array_images
                .iter()
                .map(|image| std::ptr::from_ref(image) as usize),
        );
        view.n_params = self.n_params;
        view.n_slots = self.n_slots;
        view.resumable = self.resumable;
        view.prepared_entry = self.prepared_entry;
        view.detailed_feedback = self.feedback.detailed_enabled();
        view.second_stage = self.second_stage;
        view.strict = self.strict;
        view.needs_activation_state = self.jit_needs_activation_state;
        view.fixed_activation = fixed_activation;
        view.frame_setup_needed = self.jit_frame_setup_needed();
        view.frame_args_needed = self.jit_frame_args_needed();
        view.arguments_count_slot = self.jit_arguments_count_slot();
        view.inline_activation = self.jit_inline_activation();
        view.prologue_copies = self.jit_prologue_copies();
        view.mapped_formals = self.jit_mapped_formals();
        // Made from the operations on the compile's thread, when it first asks.
        view.cap_initialized = std::cell::OnceCell::new();
        view.caches_base = self.caches.as_ptr() as usize;
        view.name_caches_base = self.name_caches.as_ptr() as usize;
        view.cap_caches_base = self.cap_caches.as_ptr() as usize;
        view.construct_caches_base = self.construct_caches.as_ptr() as usize;
        view.empty_array_image = std::ptr::from_ref(&self.empty_array_image) as usize;
        view.object_sites = object_sites;
        view.words_sites = words_sites;
        view.names_base = self.names.as_ptr() as usize;
        view.name_slot_size = std::mem::size_of::<std::rc::Rc<str>>();
        view.attempted_off = self.jit_inline_attempted_off();
        view.runs_off = self.jit_runs_off();
        view.retry_off = self.jit_inline_retry_at_off();
        #[cfg(test)]
        {
            view.test_this_commits = self.test_this_commits.as_ptr() as usize;
        }
    }
}
