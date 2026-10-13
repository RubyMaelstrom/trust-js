//! Live verification of the packed execution owners' native strong-count header.
//!
//! This supplements the ordinary-object Rc probe in JitLayout. Native code must not infer
//! that an opaque engine wrapper is Rc-shaped from its size or tag alone. No owner or realm
//! is retained in the cache; a changed layout disables these paths instead of guessing.

use super::{JitLayout, PackedValue, SymbolData, Value, PACK_PAYLOAD};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

pub(super) fn supported(layout: &JitLayout) -> bool {
    if !layout.valid
        || layout.rc_strong_off != 0
        || layout.gc_data_off < std::mem::size_of::<usize>()
    {
        return false;
    }
    static PROBE: OnceLock<(usize, usize, bool)> = OnceLock::new();
    let &(data_offset, count_offset, valid) = PROBE.get_or_init(|| {
        (
            layout.gc_data_off,
            layout.rc_strong_off,
            probe(layout.gc_data_off),
        )
    });
    valid && data_offset == layout.gc_data_off && count_offset == layout.rc_strong_off
}

fn probe(rc_data_offset: usize) -> bool {
    let string = crate::lstr::LStr::from("native owner layout");
    let symbol = Rc::new(SymbolData {
        id: 0,
        description: None,
        weak_observers: RefCell::new(None),
    });
    let bigint = crate::bigint::JsBigInt::from_u64(73);
    let samples = [
        (Value::Str(string.clone()), string.as_ptr() as usize),
        (
            Value::Sym(symbol.clone()),
            (Rc::as_ptr(&symbol) as usize).wrapping_sub(rc_data_offset),
        ),
        (
            Value::BigInt(bigint.clone()),
            bigint.allocation_identity().wrapping_sub(rc_data_offset),
        ),
    ];
    for (value, expected_header) in samples {
        let word = PackedValue::pack(value);
        let header = (word.0.get() & PACK_PAYLOAD) as usize;
        // The std Rc data/header delta was already verified on an ordinary object. LStr's
        // own repr(C) header has its strong count at zero. Check that each packed wrapper
        // actually carries that allocation's header BEFORE reading through the encoded word.
        if header != expected_header {
            return false;
        }
        let count = header as *const usize;
        if unsafe { count.read() } != 2 {
            return false;
        }
        let copied = word.clone();
        if copied.0.get() != word.0.get() || unsafe { count.read() } != 3 {
            return false;
        }
        drop(copied);
        if unsafe { count.read() } != 2 {
            return false;
        }
    }
    true
}
