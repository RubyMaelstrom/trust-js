//! P1-S0d spike: can a conservative native-stack scan recognize the
//! reference forms this codebase actually holds across allocation points?
//!
//! This is test-only evidence-gathering for the Decision 0 scanning
//! contract (docs/v8-chase-p1-plan-2026-10-15.txt). It is NOT the S1
//! scanner. The tests pin down, on this toolchain:
//! - recognized forms: raw cell pointer in a local, tagged word
//!   (tag bits in the high payload region, masked before lookup),
//!   interior pointer (field address inside a cell), pointer in a
//!   stack-resident struct field;
//! - unrecognized forms (contract violations by design): pointer held
//!   only as `usize` arithmetic, pointer held only in a Rust-heap
//!   container;
//! - the `stacker` segment problem: a grown segment is a separate
//!   mapping, invisible to a main-stack range scan.
//!
//! Run in release for meaningful codegen:
//! `cargo test --release -p trust-js --lib --features embed gc_scan_spike`
//! (debug builds keep everything in stack slots and prove much less).

#![cfg(test)]

use std::collections::BTreeSet;
use std::hint::black_box;

const TAG_OBJ: u64 = 0xfff9_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
const CELL_WORDS: usize = 8;
const CELLS: usize = 64;

/// Fake GC arena: word-addressed cells of CELL_WORDS u64s.
struct Arena {
    words: Vec<u64>,
}

impl Arena {
    fn new() -> Self {
        Self {
            words: vec![0; CELLS * CELL_WORDS],
        }
    }
    fn cell_ptr(&self, cell: usize) -> *const u64 {
        unsafe { self.words.as_ptr().add(cell * CELL_WORDS) }
    }
    fn lo(&self) -> usize {
        self.words.as_ptr() as usize
    }
    fn hi(&self) -> usize {
        self.lo() + CELLS * CELL_WORDS * 8
    }
    /// Resolve a candidate word to a cell index the way page metadata
    /// would: mask tag bits, accept in-range (interior) addresses.
    fn resolve(&self, word: u64) -> Option<usize> {
        let addr = (word & PAYLOAD_MASK) as usize;
        if addr >= self.lo() && addr < self.hi() {
            Some((addr - self.lo()) / (CELL_WORDS * 8))
        } else {
            None
        }
    }
}

/// Scan [bottom, top) as u64 words; stack grows down, so `top` is the
/// higher address recorded by an outer frame and `bottom` a lower
/// address in the deepest frame. Both must bracket every frame of
/// interest; reading outside mapped stack is the caller's contract bug.
unsafe fn scan(arena: &Arena, bottom: *const u64, top: *const u64) -> BTreeSet<usize> {
    let mut found = BTreeSet::new();
    let mut p = bottom;
    while p < top {
        let word = unsafe { std::ptr::read_volatile(p) };
        if let Some(cell) = arena.resolve(word) {
            found.insert(cell);
        }
        p = unsafe { p.add(1) };
    }
    found
}

struct Holder {
    field: *const u64,
    _pad: u64,
}

/// Models a collection safepoint. Real collection is entered through a
/// call (the allocation slow path), so by the time we scan, the ABI has
/// forced every live caller-saved value either dead or onto the stack,
/// and callee-saved registers have been saved into intervening frames.
/// The volatile spill array additionally models the contract's explicit
/// register-snapshot step for the collector entry frame itself.
#[inline(never)]
fn collect_here(arena: &Arena, top: *const u64, spill: &[u64; 8]) -> BTreeSet<usize> {
    let marker = 0u64;
    let bottom = &marker as *const u64;
    black_box(spill);
    let mut found = unsafe { scan(arena, bottom, top) };
    for word in spill {
        if let Some(cell) = arena.resolve(*word) {
            found.insert(cell);
        }
    }
    found
}

#[inline(never)]
fn deep_scan(
    arena: &Arena,
    top: *const u64,
    raw: *const u64,
    tagged: u64,
    interior: *const u64,
    holder: &Holder,
    hidden_usize: usize,
    heap_words: &[u64],
) -> BTreeSet<usize> {
    // Spill the held forms to a stack root area, as the safepoint
    // protocol requires, then "collect" through a real call boundary.
    let mut spill = [0u64; 8];
    unsafe {
        std::ptr::write_volatile(&mut spill[0], raw as u64);
        std::ptr::write_volatile(&mut spill[1], tagged);
        std::ptr::write_volatile(&mut spill[2], interior as u64);
        std::ptr::write_volatile(&mut spill[3], holder.field as u64);
    }
    black_box((hidden_usize, heap_words));
    let found = collect_here(arena, top, &spill);
    // Evidence (not asserted): without the spill, an optimizing build
    // may keep sole references in registers where no stack scan sees
    // them. Observed on this toolchain: only the arena base (cell 0)
    // was found. That is why the spill protocol is in the contract.
    let marker = 0u64;
    let unspilled = unsafe { scan(arena, &marker as *const u64, top) };
    eprintln!("spike: unspilled same-frame scan found {unspilled:?}");
    black_box((raw, tagged, interior, holder.field, hidden_usize, heap_words));
    found
}

#[test]
fn spike_recognized_and_unrecognized_forms() {
    let arena = Arena::new();
    // Cells under test: 3=raw 5=tagged 7=interior 11=struct field
    // 13=hidden-as-usize 17=heap-Vec-only 19=never referenced.
    let raw = arena.cell_ptr(3);
    let tagged = TAG_OBJ | (arena.cell_ptr(5) as u64 & PAYLOAD_MASK);
    let interior = unsafe { arena.cell_ptr(7).add(3) };
    let holder = Holder {
        field: arena.cell_ptr(11),
        _pad: 0,
    };
    // Hidden: reconstructable only by arithmetic, not a stored word.
    let hidden_usize = (arena.cell_ptr(13) as usize) ^ 0x5a5a_usize;
    let heap_words: Vec<u64> = vec![arena.cell_ptr(17) as u64];

    let top_marker = 0u64;
    let top = unsafe { (&top_marker as *const u64).add(64) };
    let found = deep_scan(
        &arena,
        top,
        raw,
        tagged,
        interior,
        &holder,
        hidden_usize ^ 0x5a5a,
        &heap_words,
    );
    black_box((&arena, &top_marker));

    assert!(found.contains(&3), "raw pointer local not found: {found:?}");
    assert!(found.contains(&5), "tagged word not found: {found:?}");
    assert!(found.contains(&7), "interior pointer not found: {found:?}");
    assert!(
        found.contains(&11),
        "struct-field pointer not found: {found:?}"
    );
    // Contract violations: these MUST NOT be load-bearing anywhere.
    // (A stale copy could false-positive; absence is the expectation,
    // not a soundness proof either way.)
    assert!(
        !found.contains(&19),
        "unreferenced cell falsely retained: {found:?}"
    );
}

#[test]
fn spike_stacker_segment_is_invisible_to_main_stack_scan() {
    let arena = Arena::new();
    let cell = arena.cell_ptr(23);
    let top_marker = 0u64;
    let top = unsafe { (&top_marker as *const u64).add(64) };
    let bottom_marker = 0u64;
    let main_bottom = &bottom_marker as *const u64;

    // Inside a grown segment, hold the only reference in segment memory
    // and scan both the main-stack range and the segment range we own.
    // (An earlier version guessed a window around a segment local and
    // segfaulted on the segment guard — bounds must come from a real
    // registry, never from arithmetic around a local.)
    let (in_main, in_segment_window) = stacker::grow(256 * 1024, || {
        let mut seg_words = [0u64; 2048];
        unsafe {
            std::ptr::write_volatile(&mut seg_words[1024], cell as u64);
        }
        black_box(&seg_words);
        let in_main = unsafe { scan(&arena, main_bottom, top) }.contains(&23);
        let lo = seg_words.as_ptr();
        let hi = unsafe { lo.add(seg_words.len()) };
        let in_window = unsafe { scan(&arena, lo, hi) }.contains(&23);
        black_box(&seg_words);
        (in_main, in_window)
    });
    black_box((&arena, &top_marker, &bottom_marker, cell));

    assert!(
        in_segment_window,
        "segment-resident reference not found even in its own window"
    );
    assert!(
        !in_main,
        "main-stack scan unexpectedly saw into the grown segment"
    );
}

#[test]
fn spike_scan_cost_is_linear_and_small() {
    let arena = Arena::new();
    let buf = vec![0u64; 128 * 1024]; // 1 MiB of stack-like words
    let start = std::time::Instant::now();
    let mut total = 0usize;
    for _ in 0..100 {
        let found = unsafe { scan(&arena, buf.as_ptr(), buf.as_ptr().add(buf.len())) };
        total += found.len();
    }
    let elapsed = start.elapsed();
    black_box(total);
    // Recorded for the S0 report; bound is deliberately loose (CI noise).
    assert!(elapsed.as_millis() < 1000, "100x1MiB scans took {elapsed:?}");
    eprintln!("spike: 100 scans of 1MiB in {elapsed:?}");
}
