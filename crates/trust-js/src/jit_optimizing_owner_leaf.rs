//! Shared scalar single-owner operations. This is not a JavaScript observer.
//!
//! ECMA-262 e28783d5 OrdinaryGet/OrdinarySetWithOwnDescriptor and WeakRef liveness:
//! callers retain every required owner and defer final destruction to checked paths.
//! No allocation, callback, collection, Rust owner clone/drop, or unwind may occur
//! here. Only a validated live allocation's Cell<usize> strong count is touched.

use crate::value::{PACK_BIGINT, PACK_BOOL, PACK_CANON_NAN, PACK_OBJ, PACK_STR, PACK_SYM};
use std::cell::Cell;

pub(super) const SUCCESS: u64 = 1;
pub(super) const TRANSFER: u64 = 2;
pub(super) const GUARD_SHIFT: u32 = 8;
const PAYLOAD: u64 = 0x0000_ffff_ffff_ffff;

#[cfg(test)]
thread_local! {
    pub(super) static TEST_CALLS: Cell<[usize;2]> = const {Cell::new([0;2])};
}

pub(super) fn requested() -> bool {
    std::env::var("TRUST_JS_OPT_JIT_SHARED_OWNERS").as_deref() == Ok("1")
}

fn immediate(word: u64) -> bool {
    word <= (PACK_BOOL | 1) || (word & 0x7fff_0000_0000_0000) <= PACK_CANON_NAN
}

/// # Safety
/// A supported nonzero payload names a live, aligned strong-count Cell<usize>
/// at offset zero, proved by jit_packed_owners_supported on this runtime. A
/// weak-only allocation may keep that header alive with a zero strong count;
/// both operations reject it without accessing its destroyed payload. In
/// ordinary native execution `word` is a canonical owning engine Value.
/// A nonzero count cannot authenticate an arbitrary pointer. Engine ownership is
/// confined to this thread; no collector or callback runs during this call.
/// The caller must commit the matching owner move immediately after success.
unsafe fn change(word: u64, retain: bool) -> u64 {
    let tag = word & !PAYLOAD;
    if !matches!(tag, PACK_OBJ | PACK_STR | PACK_SYM | PACK_BIGINT) {
        return if immediate(word) {
            SUCCESS
        } else {
            1 << GUARD_SHIFT
        };
    }
    let owner = word & PAYLOAD;
    if owner == 0 {
        return 2 << GUARD_SHIFT;
    }
    // Rc and LStr expose interior-mutable counters; never create an exclusive
    // usize reference while the live source owner is borrowed elsewhere.
    let count = &*(owner as *const Cell<usize>);
    let before = count.get();
    let after = if retain {
        let after = before.wrapping_add(1);
        if after <= 1 {
            return 3 << GUARD_SHIFT;
        }
        after
    } else {
        if before <= 1 {
            return 3 << GUARD_SHIFT;
        }
        before - 1
    };
    // Single commit. No guard, observer or destructor follows this mutation.
    count.set(after);
    SUCCESS | TRANSFER | (3 << GUARD_SHIFT)
}

/// Result: bit0 success; bit1 one real heap transfer; bits8+ logical leaf guards
/// attempted. All failures preserve the original count. Release never reaches0.
/// The dedicated scalar C ABI matches Cranelift's native calling convention.
#[inline(never)]
pub(super) unsafe extern "C" fn retain(word: u64) -> u64 {
    #[cfg(test)]
    TEST_CALLS.with(|cell| {
        let mut calls = cell.get();
        calls[0] += 1;
        cell.set(calls)
    });
    change(word, true)
}

#[inline(never)]
pub(super) unsafe extern "C" fn release(word: u64) -> u64 {
    #[cfg(test)]
    TEST_CALLS.with(|cell| {
        let mut calls = cell.get();
        calls[1] += 1;
        cell.set(calls)
    });
    change(word, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tag_and_count_failure_preserves_the_original_header() {
        for tag in [PACK_OBJ, PACK_STR, PACK_SYM, PACK_BIGINT] {
            for initial in [0, 1, 2, usize::MAX - 1, usize::MAX] {
                let header = Cell::new(initial);
                let address = &header as *const Cell<usize> as usize as u64;
                assert_eq!(address & !PAYLOAD, 0);
                let word = tag | address;
                let result = unsafe { retain(word) };
                if initial == 0 || initial == usize::MAX {
                    assert_eq!(result, 3 << GUARD_SHIFT);
                    assert_eq!(header.get(), initial);
                } else {
                    assert_eq!(result, SUCCESS | TRANSFER | (3 << GUARD_SHIFT));
                    assert_eq!(header.get(), initial + 1);
                }
                header.set(initial);
                let result = unsafe { release(word) };
                if initial <= 1 {
                    assert_eq!(result, 3 << GUARD_SHIFT);
                    assert_eq!(header.get(), initial);
                } else {
                    assert_eq!(result, SUCCESS | TRANSFER | (3 << GUARD_SHIFT));
                    assert_eq!(header.get(), initial - 1);
                }
            }
            assert_eq!(unsafe { retain(tag) }, 2 << GUARD_SHIFT);
            assert_eq!(unsafe { release(tag) }, 2 << GUARD_SHIFT);
        }
    }

    #[test]
    fn immediate_and_lazy_words_are_never_dereferenced_or_mutated() {
        for word in [
            crate::value::PACK_UNDEFINED,
            crate::value::PACK_EMPTY,
            crate::value::PACK_NULL,
            PACK_BOOL,
            PACK_BOOL | 1,
            0,
            (-0.0f64).to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            PACK_CANON_NAN,
        ] {
            assert_eq!(unsafe { retain(word) }, SUCCESS);
            assert_eq!(unsafe { release(word) }, SUCCESS);
        }
        let word = crate::value::PACK_LAZY_PROTO | 5;
        assert_eq!(unsafe { retain(word) }, 1 << GUARD_SHIFT);
        assert_eq!(unsafe { release(word) }, 1 << GUARD_SHIFT);
    }

    #[test]
    fn weak_only_allocation_remains_weak_only_after_both_failed_changes() {
        let sample = crate::value::Object::new(None);
        let layout = crate::value::jit_layout(&sample);
        assert!(crate::value::jit_packed_owners_supported(&layout));
        let object = crate::value::Object::new(None);
        let weak = std::rc::Rc::downgrade(&object);
        let header = (std::rc::Rc::as_ptr(&object) as usize).wrapping_sub(layout.gc_data_off);
        drop(object);
        assert_eq!(weak.strong_count(), 0);
        let word = PACK_OBJ | header as u64;
        // Weak keeps the allocation/header alive. Neither operation may revive
        // its destroyed payload, change the strong count, or notify observers.
        assert_eq!(unsafe { retain(word) }, 3 << GUARD_SHIFT);
        assert_eq!(unsafe { release(word) }, 3 << GUARD_SHIFT);
        assert_eq!(weak.strong_count(), 0);
        assert!(weak.upgrade().is_none());
    }
}
