//! What the operation emitters the template JIT and the optimizing tier share read of a chunk
//! (P3 stage 2, step 12): one trait, [`JitChunk`], which the live [`Chunk`] implements by its own
//! accessors and the tier's frozen view (`opt::view::FrozenChunk`) by the copies and words its
//! main thread took, so that the emitters stay one code while the tier compiles off the main
//! thread. Everything a method returns is plain data: the operations, names' text, constant
//! words, the frame's facts, and the addresses of chunk-owned cells and images as words that
//! generated code embeds and the emitters never read through.
//!
//! The live chunk's answers are its accessors' (each method forwards), so the template JIT's
//! code is what it was; the lazily built facts (the activation plan, the TDZ facts, the
//! resolution sites) are built at the first call as before.

use crate::bytecode::{Chunk, IcState, Op, StaticName};
use crate::gc_core::InlineAllocSite;
use crate::interpreter::BindingAttrs;

/// A chunk as the shared emitters read it (see the module's doc).
pub(crate) trait JitChunk {
    /// The operations.
    fn jit_ops(&self) -> &[Op];
    /// The text of name `n`.
    fn jit_name(&self, n: u32) -> &str;
    /// Name `n`'s property key's word (`atom::PropKey::word`) in the chunk's heap, which a
    /// key-checked or creating property template embeds: the live chunk atomizes it now, as the
    /// templates did; the frozen view gives its chunk's key table's (`Chunk::freeze`).
    fn jit_name_key(&self, n: u32) -> u64;
    /// (parameters, slots).
    fn jit_frame(&self) -> (usize, usize);
    fn jit_is_strict(&self) -> bool;
    fn jit_needs_activation_state(&self) -> bool;
    fn jit_frame_setup_needed(&self) -> bool;
    fn jit_frame_args_needed(&self) -> bool;
    fn jit_arguments_count_slot(&self) -> Option<u16>;
    fn jit_inline_activation(&self) -> Option<crate::bytecode::activation::InlineActivation>;
    #[allow(clippy::type_complexity)]
    fn jit_prologue_copies(&self) -> Option<(Vec<(u32, u16)>, Option<u32>)>;
    fn jit_mapped_formals(&self) -> Option<(u32, Option<u32>)>;
    fn jit_cap_binding(&self, n: u32) -> Option<(u32, BindingAttrs)>;
    fn jit_cap_initialized(&self, pc: usize) -> bool;
    fn jit_name_static(&self, cache: u32) -> Option<StaticName>;
    fn jit_name_global(&self, cache: u32) -> Option<u16>;
    /// The address of property cache `idx`'s first way (a word to embed).
    fn jit_cache_ptr(&self, idx: u32) -> usize;
    /// Property cache `idx`'s preferred way, which a compact template bakes in behind full
    /// guards: the live chunk's now, the frozen view's at its snapshot.
    fn jit_cache_preferred(&self, idx: u32) -> Option<IcState>;
    /// Property cache `idx`'s ways, which a polymorphic site's template bakes and a store's
    /// tests for an empty cache: the live chunk's now, the frozen view's at its snapshot.
    fn jit_cache_ways(&self, idx: u32) -> [IcState; crate::bytecode::PROP_IC_WAYS];
    fn jit_call_cache_ptr(&self, idx: u32) -> usize;
    fn jit_construct_cache_ptr(&self, idx: u32) -> usize;
    fn jit_name_cache_ptr(&self, idx: u32) -> usize;
    fn jit_cap_cache_ptr(&self, idx: u32) -> usize;
    fn call_position(&self, cache: u32) -> u32;
    fn construct_position(&self, cache: u32) -> u32;
    fn jit_const_copyable(&self, k: u32) -> bool;
    fn jit_const_scalar_bits(&self, k: u32) -> Option<u64>;
    fn jit_const_is_str(&self, k: u32) -> bool;
    fn jit_const_str_word(&self, k: u32) -> Option<u64>;
    /// Object-literal site `tidx`'s in-line allocation site and its image's address.
    fn jit_object_site(&self, tidx: u32, count: usize) -> Option<(InlineAllocSite, usize)>;
    /// `[]`'s (`literal` none) or a constant array literal's site and image address.
    fn jit_array_site(&self, literal: Option<u32>) -> Option<(InlineAllocSite, usize)>;
    /// A MakeArray of `count` elements: the array's and its block's sites, the image's address.
    fn jit_array_elements_site(
        &self,
        count: usize,
    ) -> Option<(InlineAllocSite, InlineAllocSite, usize)>;
    /// The address of inline target `t`'s pinned function's [[Environment]] (`Env::as_ptr`),
    /// None for a dead pin.
    fn inline_target_env_word(&self, t: u32) -> Option<usize>;
    fn jit_resolution_cache_ptr(&self, name: u32) -> usize;
    fn jit_shared_name_ptr(&self, name: u32) -> usize;
    fn jit_inline_attempted_off(&self) -> usize;
    fn jit_runs_off(&self) -> usize;
    fn jit_inline_retry_at_off(&self) -> usize;
    /// The address of the chunk's count of `this` commits (tests).
    #[cfg(test)]
    fn test_this_commits_ptr(&self) -> usize;
}

/// The live chunk's mutable state, lazily built facts and heap are the main thread's: a compile
/// off it reads the frozen view (debug builds check, P3 stage 2, step 12).
#[inline(always)]
fn live() {
    debug_assert!(
        !crate::jit::off_main_compile(),
        "a compile off the main thread reads a live chunk"
    );
}

impl JitChunk for Chunk {
    #[inline]
    fn jit_ops(&self) -> &[Op] {
        Chunk::jit_ops(self)
    }
    #[inline]
    fn jit_name(&self, n: u32) -> &str {
        Chunk::jit_name(self, n)
    }
    #[inline]
    fn jit_name_key(&self, n: u32) -> u64 {
        live();
        crate::atom::PropKey::from_text(Chunk::jit_name(self, n)).word()
    }
    #[inline]
    fn jit_frame(&self) -> (usize, usize) {
        Chunk::jit_frame(self)
    }
    #[inline]
    fn jit_is_strict(&self) -> bool {
        Chunk::jit_is_strict(self)
    }
    #[inline]
    fn jit_needs_activation_state(&self) -> bool {
        Chunk::jit_needs_activation_state(self)
    }
    #[inline]
    fn jit_frame_setup_needed(&self) -> bool {
        Chunk::jit_frame_setup_needed(self)
    }
    #[inline]
    fn jit_frame_args_needed(&self) -> bool {
        Chunk::jit_frame_args_needed(self)
    }
    #[inline]
    fn jit_arguments_count_slot(&self) -> Option<u16> {
        Chunk::jit_arguments_count_slot(self)
    }
    #[inline]
    fn jit_inline_activation(&self) -> Option<crate::bytecode::activation::InlineActivation> {
        live();
        Chunk::jit_inline_activation(self)
    }
    #[inline]
    fn jit_prologue_copies(&self) -> Option<(Vec<(u32, u16)>, Option<u32>)> {
        live();
        Chunk::jit_prologue_copies(self)
    }
    #[inline]
    fn jit_mapped_formals(&self) -> Option<(u32, Option<u32>)> {
        live();
        Chunk::jit_mapped_formals(self)
    }
    #[inline]
    fn jit_cap_binding(&self, n: u32) -> Option<(u32, BindingAttrs)> {
        live();
        Chunk::jit_cap_binding(self, n)
    }
    #[inline]
    fn jit_cap_initialized(&self, pc: usize) -> bool {
        live();
        Chunk::jit_cap_initialized(self, pc)
    }
    #[inline]
    fn jit_name_static(&self, cache: u32) -> Option<StaticName> {
        Chunk::jit_name_static(self, cache)
    }
    #[inline]
    fn jit_name_global(&self, cache: u32) -> Option<u16> {
        Chunk::jit_name_global(self, cache)
    }
    #[inline]
    fn jit_cache_ptr(&self, idx: u32) -> usize {
        Chunk::jit_cache_ptr(self, idx)
    }
    #[inline]
    fn jit_cache_preferred(&self, idx: u32) -> Option<IcState> {
        live();
        Chunk::jit_cache_preferred(self, idx)
    }
    #[inline]
    fn jit_cache_ways(&self, idx: u32) -> [IcState; crate::bytecode::PROP_IC_WAYS] {
        live();
        Chunk::jit_cache_ways(self, idx)
    }
    #[inline]
    fn jit_call_cache_ptr(&self, idx: u32) -> usize {
        Chunk::jit_call_cache_ptr(self, idx)
    }
    #[inline]
    fn jit_construct_cache_ptr(&self, idx: u32) -> usize {
        Chunk::jit_construct_cache_ptr(self, idx)
    }
    #[inline]
    fn jit_name_cache_ptr(&self, idx: u32) -> usize {
        Chunk::jit_name_cache_ptr(self, idx)
    }
    #[inline]
    fn jit_cap_cache_ptr(&self, idx: u32) -> usize {
        Chunk::jit_cap_cache_ptr(self, idx)
    }
    #[inline]
    fn call_position(&self, cache: u32) -> u32 {
        Chunk::call_position(self, cache)
    }
    #[inline]
    fn construct_position(&self, cache: u32) -> u32 {
        Chunk::construct_position(self, cache)
    }
    #[inline]
    fn jit_const_copyable(&self, k: u32) -> bool {
        Chunk::jit_const_copyable(self, k)
    }
    #[inline]
    fn jit_const_scalar_bits(&self, k: u32) -> Option<u64> {
        Chunk::jit_const_scalar_bits(self, k)
    }
    #[inline]
    fn jit_const_is_str(&self, k: u32) -> bool {
        Chunk::jit_const_is_str(self, k)
    }
    #[inline]
    fn jit_const_str_word(&self, k: u32) -> Option<u64> {
        Chunk::jit_const_str_word(self, k)
    }
    #[inline]
    fn jit_object_site(&self, tidx: u32, count: usize) -> Option<(InlineAllocSite, usize)> {
        live();
        Chunk::jit_object_site(self, tidx, count)
            .map(|(site, image)| (site, std::ptr::from_ref(image) as usize))
    }
    #[inline]
    fn jit_array_site(&self, literal: Option<u32>) -> Option<(InlineAllocSite, usize)> {
        live();
        Chunk::jit_array_site(self, literal)
            .map(|(site, image)| (site, std::ptr::from_ref(image) as usize))
    }
    #[inline]
    fn jit_array_elements_site(
        &self,
        count: usize,
    ) -> Option<(InlineAllocSite, InlineAllocSite, usize)> {
        live();
        Chunk::jit_array_elements_site(self, count)
            .map(|(cell, block, image)| (cell, block, std::ptr::from_ref(image) as usize))
    }
    #[inline]
    fn inline_target_env_word(&self, t: u32) -> Option<usize> {
        live();
        Chunk::inline_target_env(self, t).map(|env| crate::interpreter::Env::as_ptr(&env) as usize)
    }
    #[inline]
    fn jit_resolution_cache_ptr(&self, name: u32) -> usize {
        live();
        Chunk::jit_resolution_cache_ptr(self, name)
    }
    #[inline]
    fn jit_shared_name_ptr(&self, name: u32) -> usize {
        Chunk::jit_shared_name_ptr(self, name)
    }
    #[inline]
    fn jit_inline_attempted_off(&self) -> usize {
        Chunk::jit_inline_attempted_off(self)
    }
    #[inline]
    fn jit_runs_off(&self) -> usize {
        Chunk::jit_runs_off(self)
    }
    #[inline]
    fn jit_inline_retry_at_off(&self) -> usize {
        Chunk::jit_inline_retry_at_off(self)
    }
    #[cfg(test)]
    fn test_this_commits_ptr(&self) -> usize {
        self.test_this_commits.as_ptr() as usize
    }
}
