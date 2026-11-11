//! The tier's frozen view of a chunk (P3 stage 2, step 12; the plan's "Step 12, compiling off the
//! main thread", point 2): what the builder, the verifier and the back end read of the chunk,
//! copied or forced on the main thread with the snapshot (`Chunk::freeze`), so that a compile
//! reads no chunk state at all and can run on another thread. It holds plain data only: the
//! operations, the names' text, the constants' words, the frame's facts, the lazily built facts
//! forced (the activation plan's, the resolution sites), the live state the back end bakes in
//! read once (the compact templates' preferred ways, the inline targets' callee words and
//! environments with their pins upgraded, the in-line allocation sites of the agent heap), and
//! the addresses of the chunk's cells and images as words the code embeds and nothing reads
//! through. What is a function of the operations alone (the stack effects, the captured
//! bindings' TDZ facts) it makes on the compile's thread. The chunk outlives every code that
//! embeds them: the engine's list of pending compiles keeps it alive until the install.
//!
//! The shared emitters read it through `jit::JitChunk`, as they read a live chunk, so that the
//! template JIT's code and the tier's effects stay one code. Where the tier has no back end the
//! view serves the front end's tests only.
#![cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use crate::bytecode::{IcState, Op, StaticName};
use crate::gc_core::InlineAllocSite;
use crate::interpreter::BindingAttrs;

/// A constant as the builder and the emitters read it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FrozenConst {
    /// `undefined`, `null`, a Boolean or a Number (`Chunk::jit_const_copyable`).
    pub(crate) copyable: bool,
    pub(crate) scalar_bits: Option<u64>,
    pub(crate) is_str: bool,
    /// A String's execution word, `PACK_STR | cell` (`Chunk::jit_const_str_word`).
    pub(crate) str_word: Option<u64>,
}

/// An inline target (a spliced callee's guard data) as the builder and the back end read it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct InlineTargetView {
    pub(crate) argc: u16,
    pub(crate) check_this: bool,
    pub(crate) expected_env: usize,
    pub(crate) expected_genv: usize,
    /// The expected callee's stored word (an `Option<Gc>`'s, as the first stage's InlineGuard
    /// reads it), its pin upgraded on the main thread; None for a dead callee, whose guard always
    /// exits.
    pub(crate) callee: Option<usize>,
    /// Its [[Environment]]'s address (`Chunk::inline_target_env`), None for a dead pin.
    pub(crate) env: Option<usize>,
}

/// The frozen view of a chunk (see the module's doc).
pub(crate) struct FrozenChunk {
    pub(crate) ops: Box<[Op]>,
    /// The names' text, one after the other, and where each ends.
    pub(crate) names: Box<str>,
    pub(crate) name_ends: Box<[u32]>,
    /// The names' property keys' words (`atom::PropKey::word`), by name index: the chunk's key
    /// table's (`Chunk::name_keys`; P3 stage 2, step 16), which a key-checked or creating
    /// property template embeds (`jit_name_key`).
    pub(crate) keys: Box<[u64]>,
    /// The constants the operations push (`Op::Const`), by index; None for the others, which
    /// nothing in a compile reads.
    pub(crate) consts: Box<[Option<FrozenConst>]>,
    pub(crate) n_params: usize,
    pub(crate) n_slots: usize,
    pub(crate) resumable: bool,
    pub(crate) prepared_entry: bool,
    pub(crate) detailed_feedback: bool,
    pub(crate) second_stage: bool,
    pub(crate) strict: bool,
    pub(crate) needs_activation_state: bool,
    pub(crate) fixed_activation: bool,
    pub(crate) frame_setup_needed: bool,
    pub(crate) frame_args_needed: bool,
    pub(crate) arguments_count_slot: Option<u16>,
    pub(crate) inline_activation: Option<crate::bytecode::activation::InlineActivation>,
    #[allow(clippy::type_complexity)]
    pub(crate) prologue_copies: Option<(Vec<(u32, u16)>, Option<u32>)>,
    pub(crate) mapped_formals: Option<(u32, Option<u32>)>,
    /// Each pc's `Chunk::jit_cap_initialized` (`bytecode::cap_tdz_known` of the operations),
    /// made from the operations on the compile's thread when the compile first asks.
    pub(crate) cap_initialized: std::cell::OnceCell<Box<[bool]>>,
    /// Each name's `Chunk::jit_cap_binding` (empty where the chunk's activation is not fixed).
    pub(crate) cap_bindings: Box<[Option<(u32, BindingAttrs)>]>,
    pub(crate) name_statics: Box<[Option<StaticName>]>,
    pub(crate) name_globals: Box<[Option<u16>]>,
    /// The addresses of the first property, name and captured-binding caches (`caches` and the
    /// others never reallocate), and of the construct caches by site.
    pub(crate) caches_base: usize,
    pub(crate) name_caches_base: usize,
    pub(crate) cap_caches_base: usize,
    pub(crate) construct_caches_base: usize,
    /// Each call site's way-1 address (`Chunk::jit_call_cache_ptr`).
    pub(crate) call_caches: Box<[usize]>,
    /// The named sites' preferred ways (`Chunk::jit_cache_preferred`) and ways
    /// (`Chunk::jit_cache_ways`), by cache index, ascending.
    #[allow(clippy::type_complexity)]
    pub(crate) sites: Box<
        [(
            u32,
            Option<IcState>,
            [IcState; crate::bytecode::PROP_IC_WAYS],
        )],
    >,
    pub(crate) call_positions: Box<[u32]>,
    pub(crate) construct_positions: Box<[u32]>,
    pub(crate) inline_targets: Box<[InlineTargetView]>,
    /// The object literals' images' addresses, by template index; the constant array literals'
    /// by site; the MakeArrays' image's.
    pub(crate) object_images: Box<[usize]>,
    pub(crate) array_images: Box<[usize]>,
    pub(crate) empty_array_image: usize,
    /// The agent heap's in-line allocation sites: an object's cell by capacity class (0 to 3,
    /// `value::object_inline_alloc_site`), an element block by its count of words (1 to
    /// `MAX_INLINE_ELEMENTS`, index 0 unused; `value::words_inline_alloc_site`). None where the
    /// chunk makes no literal.
    pub(crate) object_sites: [Option<InlineAllocSite>; 4],
    pub(crate) words_sites: [Option<InlineAllocSite>; crate::bytecode::MAX_INLINE_ELEMENTS + 1],
    /// The resolution sites' addresses of the names the chunk's ResolveNameRefs resolve, by name
    /// index, ascending.
    pub(crate) resolution_sites: Box<[(u32, usize)]>,
    /// The address of the chunk's first name slot (`Chunk::jit_shared_name_ptr`).
    pub(crate) names_base: usize,
    pub(crate) name_slot_size: usize,
    pub(crate) attempted_off: usize,
    pub(crate) runs_off: usize,
    pub(crate) retry_off: usize,
    /// The log's and the exit table's name of the function: `fn(<its first four slot names>)`.
    pub(crate) label: String,
    #[cfg(test)]
    pub(crate) test_this_commits: usize,
}

impl FrozenChunk {
    pub(crate) fn jit_ops(&self) -> &[Op] {
        &self.ops
    }

    pub(crate) fn jit_name(&self, n: u32) -> &str {
        let n = n as usize;
        let start = if n == 0 {
            0
        } else {
            self.name_ends[n - 1] as usize
        };
        &self.names[start..self.name_ends[n] as usize]
    }

    pub(crate) fn jit_frame(&self) -> (usize, usize) {
        (self.n_params, self.n_slots)
    }

    pub(crate) fn jit_is_resumable(&self) -> bool {
        self.resumable
    }

    pub(crate) fn jit_detailed_feedback_enabled(&self) -> bool {
        self.detailed_feedback
    }

    pub(crate) fn jit_second_stage(&self) -> bool {
        self.second_stage
    }

    pub(crate) fn jit_fixed_activation(&self) -> bool {
        self.fixed_activation
    }

    /// (pops, pushes) of the operation at `pc` (`Chunk::jit_stack_effect`), from the operations,
    /// on the compile's thread.
    pub(crate) fn jit_stack_effect(&self, pc: usize) -> Option<(usize, usize)> {
        crate::bytecode::stack_effect(&self.ops, pc, self.resumable, false)
    }

    pub(crate) fn jit_cap_initialized(&self, pc: usize) -> bool {
        self.cap_initialized
            .get_or_init(|| crate::bytecode::cap_tdz_known(&self.ops))
            .get(pc)
            .copied()
            .unwrap_or(false)
    }

    /// Constant `k`, which an `Op::Const` pushes (a compile reads no other); none embeddable
    /// otherwise, which debug builds refuse.
    fn constant(&self, k: u32) -> FrozenConst {
        let constant = self.consts.get(k as usize).copied().flatten();
        debug_assert!(
            constant.is_some(),
            "no Op::Const of the chunk pushes constant {k}"
        );
        constant.unwrap_or_default()
    }

    pub(crate) fn jit_inline_target(&self, t: u32) -> &InlineTargetView {
        &self.inline_targets[t as usize]
    }

    pub(crate) fn jit_const_copyable(&self, k: u32) -> bool {
        self.constant(k).copyable
    }

    pub(crate) fn jit_const_scalar_bits(&self, k: u32) -> Option<u64> {
        self.constant(k).scalar_bits
    }

    pub(crate) fn jit_const_is_str(&self, k: u32) -> bool {
        self.constant(k).is_str
    }

    pub(crate) fn jit_const_str_word(&self, k: u32) -> Option<u64> {
        self.constant(k).str_word
    }

    pub(crate) fn jit_inline_attempted_off(&self) -> usize {
        self.attempted_off
    }

    pub(crate) fn jit_runs_off(&self) -> usize {
        self.runs_off
    }

    pub(crate) fn jit_inline_retry_at_off(&self) -> usize {
        self.retry_off
    }

    /// The Cfg of the chunk's operations and stack effects (`jit_ir::Cfg::build`'s).
    pub(crate) fn cfg(&self) -> Result<crate::jit_ir::Cfg, crate::jit_ir::BuildError> {
        crate::jit_ir::Cfg::build_from(&self.ops, |pc| self.jit_stack_effect(pc))
    }

    /// `fn(<the first four slot names>)`, the log's name of the function.
    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    /// Where named site `idx`'s ways are in `sites`.
    #[cfg_attr(
        not(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        )),
        allow(dead_code)
    )]
    fn site(&self, idx: u32) -> usize {
        self.sites
            .binary_search_by_key(&idx, |&(cache, _, _)| cache)
            .unwrap_or_else(|_| panic!("the frozen view has no ways for cache {idx}"))
    }

    /// Name `n`'s key's word (see `keys`).
    pub(crate) fn jit_name_key(&self, n: u32) -> u64 {
        self.keys[n as usize]
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
impl crate::jit::JitChunk for FrozenChunk {
    fn jit_ops(&self) -> &[Op] {
        &self.ops
    }
    fn jit_name(&self, n: u32) -> &str {
        FrozenChunk::jit_name(self, n)
    }
    fn jit_name_key(&self, n: u32) -> u64 {
        FrozenChunk::jit_name_key(self, n)
    }
    fn jit_frame(&self) -> (usize, usize) {
        (self.n_params, self.n_slots)
    }
    fn jit_is_strict(&self) -> bool {
        self.strict
    }
    fn jit_needs_activation_state(&self) -> bool {
        self.needs_activation_state
    }
    fn jit_frame_setup_needed(&self) -> bool {
        self.frame_setup_needed
    }
    fn jit_frame_args_needed(&self) -> bool {
        self.frame_args_needed
    }
    fn jit_arguments_count_slot(&self) -> Option<u16> {
        self.arguments_count_slot
    }
    fn jit_inline_activation(&self) -> Option<crate::bytecode::activation::InlineActivation> {
        self.inline_activation.clone()
    }
    fn jit_prologue_copies(&self) -> Option<(Vec<(u32, u16)>, Option<u32>)> {
        self.prologue_copies.clone()
    }
    fn jit_mapped_formals(&self) -> Option<(u32, Option<u32>)> {
        self.mapped_formals
    }
    fn jit_cap_binding(&self, n: u32) -> Option<(u32, BindingAttrs)> {
        self.cap_bindings.get(n as usize).copied().flatten()
    }
    fn jit_cap_initialized(&self, pc: usize) -> bool {
        FrozenChunk::jit_cap_initialized(self, pc)
    }
    fn jit_name_static(&self, cache: u32) -> Option<StaticName> {
        self.name_statics.get(cache as usize).copied().flatten()
    }
    fn jit_name_global(&self, cache: u32) -> Option<u16> {
        self.name_globals.get(cache as usize).copied().flatten()
    }
    fn jit_cache_ptr(&self, idx: u32) -> usize {
        self.caches_base + idx as usize * std::mem::size_of::<std::cell::Cell<IcState>>()
    }
    fn jit_cache_preferred(&self, idx: u32) -> Option<IcState> {
        self.sites[self.site(idx)].1
    }
    fn jit_cache_ways(&self, idx: u32) -> [IcState; crate::bytecode::PROP_IC_WAYS] {
        self.sites[self.site(idx)].2
    }
    fn jit_call_cache_ptr(&self, idx: u32) -> usize {
        self.call_caches[idx as usize]
    }
    fn jit_construct_cache_ptr(&self, idx: u32) -> usize {
        self.construct_caches_base
            + idx as usize * std::mem::size_of::<std::cell::Cell<crate::bytecode::ConstructSite>>()
    }
    fn jit_name_cache_ptr(&self, idx: u32) -> usize {
        self.name_caches_base
            + idx as usize * std::mem::size_of::<std::cell::Cell<crate::bytecode::NameIc>>()
    }
    fn jit_cap_cache_ptr(&self, idx: u32) -> usize {
        self.cap_caches_base
            + idx as usize * std::mem::size_of::<std::cell::Cell<crate::bytecode::NameIc>>()
    }
    fn call_position(&self, cache: u32) -> u32 {
        self.call_positions
            .get(cache as usize)
            .copied()
            .unwrap_or(crate::stack_trace::NO_POSITION)
    }
    fn construct_position(&self, cache: u32) -> u32 {
        self.construct_positions
            .get(cache as usize)
            .copied()
            .unwrap_or(crate::stack_trace::NO_POSITION)
    }
    fn jit_const_copyable(&self, k: u32) -> bool {
        FrozenChunk::jit_const_copyable(self, k)
    }
    fn jit_const_scalar_bits(&self, k: u32) -> Option<u64> {
        FrozenChunk::jit_const_scalar_bits(self, k)
    }
    fn jit_const_is_str(&self, k: u32) -> bool {
        FrozenChunk::jit_const_is_str(self, k)
    }
    fn jit_const_str_word(&self, k: u32) -> Option<u64> {
        FrozenChunk::jit_const_str_word(self, k)
    }
    fn jit_object_site(&self, tidx: u32, count: usize) -> Option<(InlineAllocSite, usize)> {
        if self.detailed_feedback {
            return None;
        }
        let image = *self.object_images.get(tidx as usize)?;
        let class = crate::value::cell_class_for(count)?;
        Some((self.object_sites[(class as usize).min(3)]?, image))
    }
    fn jit_array_site(&self, literal: Option<u32>) -> Option<(InlineAllocSite, usize)> {
        if self.detailed_feedback {
            return None;
        }
        let image = match literal {
            Some(site) => *self.array_images.get(site as usize)?,
            None => self.empty_array_image,
        };
        Some((self.object_sites[0]?, image))
    }
    fn jit_array_elements_site(
        &self,
        count: usize,
    ) -> Option<(InlineAllocSite, InlineAllocSite, usize)> {
        if self.detailed_feedback || count == 0 || count > crate::bytecode::MAX_INLINE_ELEMENTS {
            return None;
        }
        Some((
            self.object_sites[0]?,
            self.words_sites[count]?,
            self.empty_array_image,
        ))
    }
    fn inline_target_env_word(&self, t: u32) -> Option<usize> {
        self.inline_targets[t as usize].env
    }
    fn jit_resolution_cache_ptr(&self, name: u32) -> usize {
        let at = self
            .resolution_sites
            .binary_search_by_key(&name, |&(name, _)| name)
            .unwrap_or_else(|_| panic!("the frozen view has no resolution site for name {name}"));
        self.resolution_sites[at].1
    }
    fn jit_shared_name_ptr(&self, name: u32) -> usize {
        self.names_base + name as usize * self.name_slot_size
    }
    fn jit_inline_attempted_off(&self) -> usize {
        self.attempted_off
    }
    fn jit_runs_off(&self) -> usize {
        self.runs_off
    }
    fn jit_inline_retry_at_off(&self) -> usize {
        self.retry_off
    }
    #[cfg(test)]
    fn test_this_commits_ptr(&self) -> usize {
        self.test_this_commits
    }
}
