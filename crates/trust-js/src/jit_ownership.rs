//! ARM64 ownership emitters: every reference-count sequence the native JIT emits.
//!
//! Generated code keeps a String/Symbol/Object payload alive by holding one strong count per
//! owner (operand-stack and frame-slot words, property words, the wide `this` and binding
//! Values). Templates never address a count themselves: they pass the [`JitOwnership`] read
//! from the probed layout to these emitters, so how a value is kept alive is described in one
//! place. Every emitter emits exactly the instructions its sites always emitted (the call
//! site chooses registers and labels); each contract says what it proves and what it clobbers.
//!
//! Tag guards that only decide whether a word owns anything ([`exec_owner_guard`]) are kept
//! apart from the count work, and the drop helpers ([`drop_helper_call`]) are Rust destructors
//! that some counted sites also need for a word that is not a value owner (a function's
//! deferred prototype, whose word owns an `Rc` in the default build).
//!
//! Not value ownership: a `PropertyLayout` key buffer is an `Rc<PropertyLayoutData>` in every
//! build. Its count changes only through the `layout_rc_*` emitters at the end of this file,
//! which read their own probed offset (`JitRepr::layout_rc_strong_off`) and are emitted in
//! both builds.
//!
//! The few-instruction emitters are `#[inline]`, like the sequences templates used to write in
//! place; the `exec_*` and `frame_*` emitters stay out of line, like the functions they were.
//!
//! Every emitter that takes a [`JitOwnership`] has a `Traced` arm: the gc-heap build, whose
//! values carry no count. The collector traces the cells reachable from the words generated
//! code stores, so acquiring or releasing an owner emits no instruction there, and an execution
//! word owns nothing: every `Traced` arm below emits nothing, the execution-word clone, drop and
//! release included, so none branches to a site's slow path. Logic that sites keep beside these
//! emitters only for counted owners (alias compares, old-value markers, the direct call's
//! surplus counter, BigInt and deferred-prototype exclusions, a checked path or load that only
//! an owner guard used) is emitted only while owners are counted ([`JitOwnership::counted`]).
//! No traced code calls a drop helper: a property store overwrites its old word, a deferred
//! prototype's included (a cell under gc-heap). A property word that can be a function's
//! deferred `prototype` is kept out of execution by [`property_word_guard`]: a
//! materialization boundary, not ownership.

use super::*;
use crate::value::{PACK_BIGINT, PACK_OBJ, PACK_STR, PACK_SYM};

// ---------------------------------------------------------------------------------------------
// Execution words
// ---------------------------------------------------------------------------------------------

/// Branch to `heap` for known shared-Rc execution kinds, reject BigInt's distinct owner
/// representation, otherwise leave scalar owners alone. Both scratch registers and NZCV
/// are clobbered, raw is untouched. Execution storage never contains lazy prototypes, but a
/// property word can: PACK_LAZY_PROTO and every other property-only or unknown tag go to
/// `slow` without inspecting the payload, which the counted clones of property words rely on.
/// Counted ownership only: the `Refcount` arms below emit it.
pub(super) fn exec_owner_guard(
    a: &mut asm::Asm,
    raw: u32,
    ptr: u32,
    count: u32,
    heap: usize,
    scalar: usize,
    slow: usize,
) {
    assert!(raw != ptr && raw != count && ptr != count);
    a.lsr_imm(ptr, raw, 48);
    a.movz(count, (PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(ptr, count);
    a.b_cond(C_EQ, heap);
    a.b_cond(C_HI, slow); // unknown/property-only tag, never guess an owner type.
    a.movz(count, (PACK_BIGINT >> 48) as u32, 0);
    a.cmp_reg_x(ptr, count);
    a.b_cond(C_LO, scalar);
    a.b_cond(C_EQ, slow);
    // The only high lanes strictly above BigInt and at most Symbol are String/Symbol.
    // Lower lanes are scalars (including positive Numbers); lanes above Symbol but
    // below Object are negative Numbers. Preserve the object-first fast path and
    // reject the same unknown tags as before without inspecting/dereferencing payloads.
    const _: () = assert!(PACK_STR == PACK_BIGINT + (1 << 48));
    const _: () = assert!(PACK_SYM == PACK_STR + (1 << 48));
    const _: () = assert!(PACK_SYM < PACK_OBJ);
    a.movz(count, (PACK_SYM >> 48) as u32, 0);
    a.cmp_reg_x(ptr, count);
    a.b_cond(C_LS, heap);
    a.b(scalar);
}

/// Acquire one owner of raw. Guards precede all changes; slow must clone via the checked
/// operation. Preserve raw, clobber exactly ptr/count/NZCV. The word itself is not stored.
/// Traced: emits nothing (an execution word owns nothing, so there is nothing to acquire and
/// `slow` is never taken). A site that clones a property word which can be a function's
/// deferred `prototype` emits [`property_word_guard`] in its place.
pub(super) fn exec_clone(
    a: &mut asm::Asm,
    own: JitOwnership,
    raw: u32,
    ptr: u32,
    count: u32,
    slow: usize,
) {
    if !own.counted() {
        return;
    }
    let heap = a.new_label();
    let done = a.new_label();
    exec_owner_guard(a, raw, ptr, count, heap, done, slow);
    a.bind(heap);
    emit_exec_payload(a, raw, ptr);
    rc_inc(a, own, ptr, count, 1);
    a.bind(done);
}

/// Release one non-final owner. BigInt and last references branch before any decrement;
/// the checked path must run the real destructor. Raw and its physical word are unchanged
/// so the caller can retire/replace the slot only after success. ptr/count/NZCV clobbered.
/// Traced: emits nothing (dropping the word is its release; `slow` is never taken).
pub(super) fn exec_drop_shared(
    a: &mut asm::Asm,
    own: JitOwnership,
    raw: u32,
    ptr: u32,
    count: u32,
    slow: usize,
) {
    if !own.counted() {
        return;
    }
    let heap = a.new_label();
    let done = a.new_label();
    exec_owner_guard(a, raw, ptr, count, heap, done, slow);
    a.bind(heap);
    emit_exec_payload(a, raw, ptr);
    rc_release_shared(a, own, ptr, count, slow);
    a.bind(done);
}

/// Release the packed owner stored at `[base + off]` (a local slot or an operand-stack word).
/// Shared owners are decremented inline; a last owner, a BigInt or any other word whose release
/// needs its destructor goes to `H_DROP_PACKED_AT`, which runs only Rust destruction (no author
/// code, allocation of JS values, or unwinding). The word is left moved-from: the caller must
/// overwrite or pop it. Scratch x9..x11; the helper path clobbers x0..x17 (the frame registers
/// x19..x22 are preserved). Traced: emits nothing, not even the load (overwriting or popping
/// the word is its release).
pub(super) fn exec_release_word(a: &mut asm::Asm, own: JitOwnership, base: u32, off: i32) {
    if !own.counted() {
        return;
    }
    emit_exec_word_load(a, 9, base, off);
    exec_release_loaded(a, own, base, off);
}

/// [`exec_release_word`] after its load: x9 already holds the word at `[base + off]`. Traced:
/// emits nothing.
pub(super) fn exec_release_loaded(a: &mut asm::Asm, own: JitOwnership, base: u32, off: i32) {
    if !own.counted() {
        return;
    }
    let release = a.new_label();
    let done = a.new_label();
    exec_drop_shared(a, own, 9, 10, 11, release);
    a.b(done);
    a.bind(release);
    drop_helper_call(a, H_DROP_PACKED_AT, None, |a| {
        if off >= 0 {
            a.add_imm(2, base, off as u32);
        } else {
            a.sub_imm(2, base, off.unsigned_abs());
        }
    });
    a.bind(done);
}

// ---------------------------------------------------------------------------------------------
// Strong counts of String/Symbol/Object payloads (`ptr` is the untagged payload: the stored
// Rc pointer). None of these emitters classifies a word: the site has already proven that
// `ptr` names a reference-counted owner. `count` is the scratch register for the count.
//
// Traced (gc-heap): there is no count. Every emitter below emits nothing: `ptr` and `count`
// are neither read nor written, NZCV is preserved, and a guard never branches.
// ---------------------------------------------------------------------------------------------

/// Acquire `n` owners of the payload at `ptr`. Clobbers count; flags are preserved.
#[inline]
pub(super) fn rc_inc(a: &mut asm::Asm, own: JitOwnership, ptr: u32, count: u32, n: u32) {
    match own {
        JitOwnership::Refcount { strong_off } => {
            a.ldur(count, ptr, strong_off);
            a.add_imm(count, count, n);
            a.stur(count, ptr, strong_off);
        }
        // Storing the word where the collector traces it is the acquisition.
        JitOwnership::Traced => {}
    }
}

/// Load the payload's count into `count` for a later [`rc_check_nonfinal`] (the site does
/// unrelated work, such as an alias test, between the two). Nothing else changes. Traced:
/// nothing is loaded, so a site must not read `count` for anything but these emitters.
#[inline]
pub(super) fn rc_load_count(a: &mut asm::Asm, own: JitOwnership, ptr: u32, count: u32) {
    match own {
        JitOwnership::Refcount { strong_off } => a.ldur(count, ptr, strong_off),
        JitOwnership::Traced => {}
    }
}

/// Branch to `slow` unless the count loaded into `count` exceeds `spare`, the number of owners
/// the site will release (more than one when operands alias): the releases then never free,
/// so they need no destructor. Clobbers NZCV only. Traced: no release frees, so this never
/// branches and emits nothing.
#[inline]
pub(super) fn rc_check_nonfinal(
    a: &mut asm::Asm,
    own: JitOwnership,
    count: u32,
    spare: u32,
    slow: usize,
) {
    match own {
        JitOwnership::Refcount { .. } => {
            a.cmp_imm_x(count, spare);
            a.b_cond(C_LS, slow);
        }
        JitOwnership::Traced => {}
    }
}

/// The non-final guard: load the payload's count and branch to `slow` unless it exceeds
/// `spare` (see [`rc_check_nonfinal`]). Every guard precedes the site's first mutation. On
/// fall-through `count` holds the loaded count for [`rc_commit_dec`] or a further check.
/// Clobbers count and NZCV. Traced: emits nothing (never branches, `count` not loaded).
#[inline]
pub(super) fn rc_guard_nonfinal(
    a: &mut asm::Asm,
    own: JitOwnership,
    ptr: u32,
    count: u32,
    spare: u32,
    slow: usize,
) {
    rc_load_count(a, own, ptr, count);
    rc_check_nonfinal(a, own, count, spare, slow);
}

/// Release `n` owners by storing the count a guard left in `count`, minus `n`. Nothing may
/// have changed that count since it was loaded (no helper call, no other release of the
/// same payload in between). Clobbers count; flags are preserved. Traced: emits nothing.
#[inline]
pub(super) fn rc_commit_dec(a: &mut asm::Asm, own: JitOwnership, ptr: u32, count: u32, n: u32) {
    match own {
        JitOwnership::Refcount { strong_off } => {
            a.sub_imm(count, count, n);
            a.stur(count, ptr, strong_off);
        }
        // Dropping the word is the release: the collector reclaims what nothing traces.
        JitOwnership::Traced => {}
    }
}

/// Release one owner whose count an earlier guard proved non-final. The count is re-read, so
/// releases of the same payload in between (an aliased operand) are accounted for as long as
/// the guard reserved them. Clobbers count; flags are preserved. Traced: emits nothing.
#[inline]
pub(super) fn rc_dec_nonfinal(a: &mut asm::Asm, own: JitOwnership, ptr: u32, count: u32) {
    rc_load_count(a, own, ptr, count);
    rc_commit_dec(a, own, ptr, count, 1);
}

/// [`rc_dec_nonfinal`] in the scaled LDR/STR encoding (the strong-count offset must be a
/// non-negative multiple of eight). Used where another owner, such as a cache pin, keeps the
/// payload alive without a guard. Clobbers count; flags are preserved. Traced: emits nothing.
#[inline]
pub(super) fn rc_dec_nonfinal_scaled(a: &mut asm::Asm, own: JitOwnership, ptr: u32, count: u32) {
    match own {
        JitOwnership::Refcount { strong_off } => {
            a.ldr_imm(count, ptr, strong_off as u32);
            a.sub_imm(count, count, 1);
            a.str_imm(count, ptr, strong_off as u32);
        }
        JitOwnership::Traced => {}
    }
}

/// Guard and release one owner: a last owner branches to `slow` before any change (the
/// checked path runs the real destructor). Clobbers count and NZCV. Traced: emits nothing
/// (never branches).
#[inline]
pub(super) fn rc_release_shared(
    a: &mut asm::Asm,
    own: JitOwnership,
    ptr: u32,
    count: u32,
    slow: usize,
) {
    rc_guard_nonfinal(a, own, ptr, count, 1, slow);
    rc_commit_dec(a, own, ptr, count, 1);
}

/// Release one owner unless it is the last one (count exactly 1), which branches to `last`
/// unchanged so the site can run the real destructor through a drop helper. Clobbers count
/// and NZCV. Traced: emits nothing (never branches to `last`).
#[inline]
pub(super) fn rc_release_unless_last(
    a: &mut asm::Asm,
    own: JitOwnership,
    ptr: u32,
    count: u32,
    last: usize,
) {
    match own {
        JitOwnership::Refcount { strong_off } => {
            a.ldur(count, ptr, strong_off);
            a.cmp_imm_x(count, 1);
            a.b_cond(C_EQ, last);
            a.sub_imm(count, count, 1);
            a.stur(count, ptr, strong_off);
        }
        JitOwnership::Traced => {}
    }
}

// ---------------------------------------------------------------------------------------------
// Drop helpers
// ---------------------------------------------------------------------------------------------

/// Call a Rust destructor helper, `H_DROP_PACKED_AT` (a packed word) or `H_DROP_AT` (a wide
/// `Value`), on the value whose address `addr` writes into x2 (x0 = the context, x1 = 0). The
/// helpers run only Rust destruction (no author code, allocation of JS values, or unwinding),
/// which may cascade through an object graph; they clobber x0..x17, x30 and NZCV, and the
/// frame registers x19..x22 survive. `spill` pushes that register pair around the call.
/// Counted ownership only: no `Traced` site emits it (`jit_drop_packed_at` asserts so).
#[inline]
pub(super) fn drop_helper_call(
    a: &mut asm::Asm,
    helper: usize,
    spill: Option<(u32, u32)>,
    addr: impl FnOnce(&mut asm::Asm),
) {
    debug_assert!(helper == H_DROP_PACKED_AT || helper == H_DROP_AT);
    if let Some((first, second)) = spill {
        a.stp_pre(first, second, -16);
    }
    a.mov(0, 19);
    a.movz(1, 0, 0);
    addr(a);
    a.ldr_imm(16, 21, (helper * 8) as u32);
    a.blr(16);
    if let Some((first, second)) = spill {
        a.ldp_post(first, second, 16);
    }
}

// ---------------------------------------------------------------------------------------------
// Direct-call frame teardown
//
// Traced (gc-heap): every teardown emitter emits nothing. The surplus arguments, the callee
// operand, the record's `this` and its slots hold execution words and a wide Value, never a
// lazy prototype, and releasing them is dropping them: the caller pops the operand words, and
// the record keeps its slot words until its next entry overwrites them. The sites' own
// stores that restore pooled invariants (Undefined over `this` and `ret`) stay.
// ---------------------------------------------------------------------------------------------

/// Release the w7 over-applied argument words — the last w7 of the call's argument words, just
/// below x20 — that the callee never bound. Shared owners are decremented inline; a last owner
/// (including two surplus words naming one object) or a BigInt runs its destructor through
/// `H_DROP_PACKED_AT` (Rust destruction only). Releasing them after the call rather than at
/// entry is unobservable. w8 (the completion flag) survives; x3..x7 and x16 are clobbered.
/// Traced: emits nothing and reads no w7 (the direct call keeps no surplus count then).
pub(super) fn frame_surplus_release(a: &mut asm::Asm, own: JitOwnership) {
    if !own.counted() {
        return;
    }
    let done = a.new_label();
    let next_word = a.new_label();
    let helper = a.new_label();
    let word = a.new_label();
    a.cbz(7, false, done);
    a.bind(word);
    a.lsl_imm(3, 7, 3);
    a.sub_reg(3, 20, 3); // the next surplus word, oldest first
    a.ldr_imm(5, 3, 0);
    exec_drop_shared(a, own, 5, 6, 4, helper);
    a.b(next_word);
    a.bind(helper);
    drop_helper_call(a, H_DROP_PACKED_AT, Some((7, 8)), |a| a.mov(2, 3));
    a.bind(next_word);
    a.sub_imm(7, 7, 1);
    a.cbnz(7, false, word);
    a.bind(done);
}

/// Release the caller's callee operand (an Object, proven by the call IC) after restoring its
/// activation. w8 carries the completion flag from the finish stub and must survive any
/// last-owner destruction. Clobbers x3, x5 and x6, and x0..x17 on the last-owner path.
/// Traced: emits nothing.
pub(super) fn frame_callee_release(a: &mut asm::Asm, own: JitOwnership, argc: usize) {
    let JitOwnership::Refcount { strong_off } = own else {
        return;
    };
    a.sub_imm(3, 20, ((argc + 1) * 8) as u32);
    a.ldr_imm(5, 3, 0);
    emit_exec_payload(a, 5, 5);
    a.ldur(6, 5, strong_off);
    let last_owner = a.new_label();
    let done = a.new_label();
    a.sub_imm(6, 6, 1);
    a.cbz(6, true, last_owner);
    a.stur(6, 5, strong_off);
    a.b(done);
    a.bind(last_owner);
    // The real Rc drop owns the final decrement. Decrementing here first would make it
    // underflow instead of releasing the callable. The helper may clobber any C scratch
    // register, including the finish stub's w8 normal/throw completion flag.
    drop_helper_call(a, H_DROP_PACKED_AT, Some((8, 31)), |a| {
        a.mov_imm64(6, (argc as u64 + 1) * 8);
        a.sub_reg(2, 20, 6);
    });
    a.bind(done);
}

/// Tag bounds the finish stub's slot scan keeps in w14/w15: BigInt/String/Symbol occupy the
/// three tags from PACK_BIGINT; Object and the property-only tags start at PACK_OBJ.
const LOW_HEAP_TAG: u32 = (PACK_BIGINT >> 48) as u32;
const OBJECT_TAG: u32 = (PACK_OBJ >> 48) as u32;
const _: () =
    assert!(PACK_STR >> 48 == (PACK_BIGINT >> 48) + 1 && PACK_SYM >> 48 == (PACK_BIGINT >> 48) + 2);

/// The finish stub's full drop of the value at x<value_reg> through `helper`. Only x9 (cursor)
/// and x5 (remaining) are spilled around the call; afterwards the record is reloaded into x10
/// from the stub frame ([sp, 16]) and the scan's w14/w15/x16 constants are re-derived.
fn frame_drop_at(a: &mut asm::Asm, value_reg: u32, helper: usize) {
    a.stp_pre(9, 5, -16);
    a.mov(2, value_reg);
    drop_helper_call(a, helper, None, |_| {});
    a.ldp_post(9, 5, 16);
    a.ldur(10, 31, 16);
    a.movz(14, LOW_HEAP_TAG, 0);
    a.movz(15, OBJECT_TAG, 0);
    a.mov_imm64(16, PACK_BIGINT);
}

/// Release a finished direct callee's wide `this` binding (record in x10, binding at
/// `cx_this`): a shared String/Symbol/Object owner is decremented inline; BigInt and a last
/// reference take `H_DROP_AT` (full drop, may cascade), after which x10 is reloaded. The
/// binding is not reset here: `jit_drop_at` destroys the Value in place without overwriting
/// it, so the caller must store Undefined over it on every path. Clobbers x0..x17 and NZCV.
/// Traced: emits nothing (x10 and the binding unchanged); the caller's reset still applies.
pub(super) fn frame_this_release(a: &mut asm::Asm, own: JitOwnership, cx_this: u32) {
    if !own.counted() {
        return;
    }
    let this_done = a.new_label();
    let this_drop = a.new_label();
    a.ldrb_imm(9, 10, cx_this);
    a.cmp_imm_w(9, 5);
    a.b_cond(C_LO, this_done);
    a.b_cond(C_EQ, this_drop); // BigInt → full drop
    a.ldr_imm(11, 10, cx_this + 8);
    rc_release_shared(a, own, 11, 12, this_drop); // last reference → full drop
    a.b(this_done);
    a.bind(this_drop);
    a.add_imm(9, 10, cx_this);
    frame_drop_at(a, 9, H_DROP_AT); // reloads the record into x10
    a.bind(this_done);
}

/// Release every slot of a finished direct callee's record (x10; slot base at `cx_slots`,
/// count at `cx_n_slots`). The strong count is re-read PER VALUE, after all earlier
/// decrements: two slots aliasing one object (`a = b = new X` seeds several slots from one
/// allocation) must route the LAST reference to a real drop — a snapshot-validated bare dec
/// would zero the count without ever running the destructor and leak the whole subgraph
/// (Splay's splay_ dummy node caught exactly that). Bare dec when shared; `H_DROP_PACKED_AT`
/// for a last reference, a BigInt or a property-only tag. Exit: x10 = the record; clobbers
/// x0..x17 and NZCV. Traced: emits nothing (x10 unchanged); the record's slots keep their
/// words, which own nothing once the record is back in the pool.
pub(super) fn frame_slots_release(
    a: &mut asm::Asm,
    own: JitOwnership,
    cx_slots: u32,
    cx_n_slots: u32,
) {
    if !own.counted() {
        return;
    }
    a.ldr_imm(9, 10, cx_slots);
    a.ldr_imm(5, 10, cx_n_slots);
    a.movz(14, LOW_HEAP_TAG, 0);
    a.movz(15, OBJECT_TAG, 0);
    // The classification of one slot at [x9] (x9 and x5 survive; a drop helper re-derives
    // x10, w14, w15 and x16 afterwards).
    let one_slot = |a: &mut asm::Asm| {
        let c_next = a.new_label();
        let c_drop = a.new_label();
        let packed_ref = a.new_label();
        let low_heap = a.new_label();
        let high_heap = a.new_label();
        a.ldur(11, 9, 0);
        a.lsr_imm(13, 11, 48);
        // Scalars (Numbers, Undefined/Null/Boolean/Empty) are the common case: two unsigned
        // range tests classify every tag. BigInt/String/Symbol occupy the three tags from
        // PACK_BIGINT; Object and the property-only tags start at PACK_OBJ. No Number can
        // carry a tag at or above PACK_OBJ because packing canonicalizes NaN.
        a.sub_reg(12, 13, 14);
        a.cmp_imm_w(12, 2);
        a.b_cond(C_LS, low_heap);
        a.cmp_reg_w(13, 15);
        a.b_cond(C_HS, high_heap);
        a.b(c_next);
        a.bind(low_heap);
        a.cbz(12, false, c_drop); // BigInt → full drop
        a.b(packed_ref); // String / Symbol
        a.bind(high_heap);
        a.b_cond(C_HI, c_drop); // property-only tags never get the bare decrement
        a.bind(packed_ref);
        emit_exec_payload(a, 11, 12);
        rc_release_shared(a, own, 12, 13, c_drop); // last reference
        a.b(c_next);
        a.bind(c_drop);
        frame_drop_at(a, 9, H_DROP_PACKED_AT);
        a.bind(c_next);
    };
    // Pairs first. Every word below PACK_BIGINT (non-negative Numbers, Undefined, Empty,
    // Null, Boolean) owns nothing, so one unsigned compare per word clears two slots; a
    // pair holding anything else (a negative Number included) takes the full per-slot test
    // above, then the scan resumes.
    let pair_loop = a.new_label();
    let pair_slow = a.new_label();
    let pair_next = a.new_label();
    let tail = a.new_label();
    let c_done = a.new_label();
    a.mov_imm64(16, PACK_BIGINT);
    a.subs_imm_w(5, 5, 2); // w5 = slots left after this pair
    a.b_cond(C_LO, tail);
    a.bind(pair_loop);
    a.ldp_post_base(11, 12, 9, 16);
    a.cmp_reg_x(11, 16);
    a.ccmp_reg_x(12, 16, 2, C_LO); // x11 >= bound: force HS
    a.b_cond(C_HS, pair_slow);
    a.bind(pair_next);
    a.subs_imm_w(5, 5, 2);
    a.b_cond(C_HS, pair_loop);
    a.bind(tail);
    a.adds_imm_w(5, 5, 2); // 0 or 1 slot left
    a.cbz(5, false, c_done);
    one_slot(a);
    a.b(c_done);
    a.bind(pair_slow);
    a.sub_imm(9, 9, 16);
    one_slot(a);
    a.add_imm(9, 9, 8);
    one_slot(a);
    a.add_imm(9, 9, 8);
    a.b(pair_next);
    a.bind(c_done);
}

// ---------------------------------------------------------------------------------------------
// PropertyLayout key buffers (an `Rc<PropertyLayoutData>` in every build, never a value owner)
// ---------------------------------------------------------------------------------------------

/// The strong-count offset of a stored `PropertyLayout` Rc pointer
/// (`JitRepr::layout_rc_strong_off`): `jit_repr` locates it on a live `Rc<PropertyLayoutData>`
/// in both builds and verifies its count before a layout is reported valid. Sites gate on
/// [`layout_rc_fits`].
#[inline]
fn layout_rc_strong_off(layout: &crate::value::JitLayout) -> i32 {
    layout.repr.layout_rc_strong_off as i32
}

/// Whether the `layout_rc_*` emitters can address the PropertyLayout count with the unscaled
/// 9-bit LDUR/STUR offset.
pub(super) fn layout_rc_fits(layout: &crate::value::JitLayout) -> bool {
    layout.repr.layout_rc_strong_off < 256
}

/// Branch to `miss` unless the PropertyLayout at `ptr` has another owner, so replacing it
/// releases a non-final count (a last owner's destructor stays with Rust). Clobbers count and
/// NZCV.
#[inline]
pub(super) fn layout_rc_guard_shared(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ptr: u32,
    count: u32,
    miss: usize,
) {
    a.ldur(count, ptr, layout_rc_strong_off(layout));
    a.cmp_imm_x(count, 1);
    a.b_cond(C_LS, miss);
}

/// Acquire one owner of the PropertyLayout at `ptr`. Clobbers count; flags are preserved.
#[inline]
pub(super) fn layout_rc_inc(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ptr: u32,
    count: u32,
) {
    let strong = layout_rc_strong_off(layout);
    a.ldur(count, ptr, strong);
    a.add_imm(count, count, 1);
    a.stur(count, ptr, strong);
}

/// Release one owner of the PropertyLayout at `ptr`, proven non-final by
/// [`layout_rc_guard_shared`]. Clobbers count; flags are preserved.
#[inline]
pub(super) fn layout_rc_dec_shared(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ptr: u32,
    count: u32,
) {
    let strong = layout_rc_strong_off(layout);
    a.ldur(count, ptr, strong);
    a.sub_imm(count, count, 1);
    a.stur(count, ptr, strong);
}
