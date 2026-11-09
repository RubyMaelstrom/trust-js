//! A chunk's side of the read API (W4 step 6; `crate::feedback::read`, feedback-schema.txt
//! section 5.4): the views of its records and caches by baseline pc.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;
use crate::feedback::{
    AllocationKind, AllocationView, CallState, CallTarget, CallView, ConstructView, ElementView,
    NamedView, NamedWay, NamedWayKind, SiteView, ValueClass, ValuesView, ELEM_ARRAY,
    ELEM_KEY_INDEX, ELEM_ORDINARY, ELEM_TYPED_ARRAY, RECORD_OPERAND1_SHIFT, RECORD_RESULT_SHIFT,
};

const NUMBER: u32 = ValueClass::NumberInt32.bit() | ValueClass::NumberDouble.bit();
const INT32: u32 = ValueClass::NumberInt32.bit();
/// The element record's receivers (bits 0-6), keys (8-12) and an Array's facts (16-20).
const ELEMENT_RECEIVERS: u32 = 0x7f;
const ELEMENT_KEYS: u32 = 0x1f << 8;
const ELEMENT_FACTS: u32 = 0x1f << 16;

/// The classes an operation's in-line set holds (operand 0, operand 1, result), which generated
/// code takes without a record (feedback-schema.txt section 5.1); None for an operation without
/// a value record.
pub(crate) fn value_in_line_set(op: &Op) -> Option<[u32; 3]> {
    Some(match op {
        Op::Mod => [INT32, INT32, INT32],
        Op::Lt | Op::Gt | Op::Le | Op::Ge => [NUMBER, NUMBER, 0],
        Op::Neg
        | Op::Plus
        | Op::BitNot
        | Op::UpdateLocal(..)
        | Op::UpdateCap(..)
        | Op::UpdateName(..)
        | Op::UpdateNameCached(..)
        | Op::UpdateConst(..)
        | Op::UpdatePrivate(..)
        | Op::SuperUpdate(..)
        | Op::UpdateProp(..)
        | Op::UpdateElem(..) => [NUMBER, 0, NUMBER],
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::UShr
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::GenBin(..) => [NUMBER, NUMBER, NUMBER],
        _ => return None,
    })
}

/// Whether `op` is an element load or store with the element record.
pub(crate) fn is_element_access(op: &Op) -> bool {
    matches!(
        op,
        Op::GetElem
            | Op::GetElemLocal(..)
            | Op::GetMethodElem
            | Op::SetElem
            | Op::SetElemDrop
            | Op::SetElemLocal(..)
            | Op::SetElemLocalDrop(..)
    )
}

impl Chunk {
    /// The feedback of the operation at `pc` (see `crate::feedback::read`).
    pub(crate) fn feedback_view(&self, pc: usize) -> SiteView {
        let Some(op) = self.ops.get(pc) else {
            return SiteView::None;
        };
        let word = self.feedback.record_word(pc);
        if let Some(cache) = named_site_cache(op) {
            let named = self.named_view(cache, writes_property(op));
            return match value_in_line_set(op) {
                Some(in_line) => SiteView::NamedUpdate(named, values_view(word, in_line)),
                None => SiteView::Named(named),
            };
        }
        if let Some(in_line) = value_in_line_set(op) {
            return SiteView::Values(values_view(word, in_line));
        }
        if is_element_access(op) {
            let recorded_receivers = word & ELEMENT_RECEIVERS;
            let recorded_keys = word & ELEMENT_KEYS;
            return SiteView::Element(ElementView {
                receivers: recorded_receivers | ELEM_ARRAY | ELEM_TYPED_ARRAY | ELEM_ORDINARY,
                keys: recorded_keys | ELEM_KEY_INDEX,
                recorded_receivers,
                recorded_keys,
                facts: word & ELEMENT_FACTS,
                generic: word & crate::feedback::ELEM_GENERIC != 0,
                typed_kinds: word & crate::feedback::ELEM_TYPED_KINDS,
            });
        }
        let allocation = |kind, image: bool, template: bool| {
            SiteView::Allocation(AllocationView {
                kind,
                image,
                template,
            })
        };
        match *op {
            Op::Call(_, cache)
            | Op::CallWithThis(_, cache)
            | Op::CallSpread(_, cache)
            | Op::CallSpreadThis(_, cache) => self
                .call_caches
                .get(cache as usize)
                .map_or(SiteView::None, |site| SiteView::Call(call_view(site))),
            Op::New(_, cache) => {
                self.construct_caches
                    .get(cache as usize)
                    .map_or(SiteView::None, |site| {
                        let site = site.get();
                        SiteView::Construct(ConstructView {
                            target: call_target(&site.call),
                            prototype_shape: site.prototype_shape,
                        })
                    })
            }
            Op::MakeObject(_, _, template) => allocation(
                AllocationKind::Object,
                self.obj_images
                    .get(template as usize)
                    .is_some_and(|image| image.ready.get() != 0),
                template != u32::MAX,
            ),
            Op::NewObject => allocation(AllocationKind::Object, false, false),
            Op::MakeArray(_) => allocation(
                AllocationKind::Array,
                self.empty_array_image.ready.get() != 0,
                false,
            ),
            Op::MakeArrayLiteral(site) => allocation(
                AllocationKind::Array,
                self.array_images
                    .get(site as usize)
                    .is_some_and(|image| image.ready.get() != 0),
                self.array_literals
                    .get(site as usize)
                    .is_some_and(|literal| literal.block.get().is_some()),
            ),
            Op::NewArray => allocation(AllocationKind::Array, false, false),
            Op::MakeRegExp(..) => allocation(
                AllocationKind::RegExp,
                false,
                self.regexp_literals
                    .get(pc)
                    .is_some_and(|literal| literal.get().is_some()),
            ),
            Op::MakeClosure(..) => allocation(AllocationKind::Closure, false, false),
            _ => SiteView::None,
        }
    }

    /// The named view of the property cache whose way 0 is `cache`.
    fn named_view(&self, cache: u32, writable: bool) -> NamedView {
        let way = |k: usize| {
            self.caches
                .get(cache as usize + k)
                .map(std::cell::Cell::get)
        };
        NamedView {
            ways: std::array::from_fn(|k| way(k).and_then(named_way)),
            generic: way(0).is_some_and(|state| state.mid_ok & IC_MID_GENERIC != 0),
            writable,
        }
    }
}

/// A value record's view: rule U's sets and the record alone.
fn values_view(word: u32, in_line: [u32; 3]) -> ValuesView {
    let recorded = [
        word & 0x1ff,
        (word >> RECORD_OPERAND1_SHIFT) & 0x1ff,
        (word >> RECORD_RESULT_SHIFT) & 0x1ff,
    ];
    ValuesView {
        operand0: recorded[0] | in_line[0],
        operand1: recorded[1] | in_line[1],
        result: recorded[2] | in_line[2],
        recorded,
    }
}

/// One property cache way's view; None for an empty way.
fn named_way(state: IcState) -> Option<NamedWay> {
    let (kind, depth, slot) = match state.depth {
        IC_EMPTY => return None,
        IC_ABSENT => (NamedWayKind::Absent, state.slot as u8, 0),
        IC_CREATE => (NamedWayKind::Creation, 0, 0),
        // The hop count in the low bits, IC_ACCESSOR and IC_ARR_KEYCHK above them.
        depth if depth & IC_ACCESSOR != 0 => (NamedWayKind::Accessor, depth & 0x1f, state.slot),
        depth if depth & IC_ARR_KEYCHK != 0 => (NamedWayKind::KeyChecked, depth & 0x1f, state.slot),
        depth => (NamedWayKind::Data, depth, state.slot),
    };
    Some(NamedWay {
        receiver_shape: state.recv_shape,
        holder_shape: state.holder_shape,
        primitive: (state.mid_ok & IC_MID_PRIMITIVE) >> IC_MID_PRIMITIVE_SHIFT,
        depth,
        slot,
        location: state.loc,
        kind,
    })
}

/// A call site's view.
fn call_view(site: &CallSite) -> CallView {
    let ways: [Option<CallTarget>; 4] = std::array::from_fn(|k| {
        site.entries
            .get(k)
            .and_then(|entry| call_target(&entry.get()))
    });
    let filled = ways.iter().flatten().count();
    let state = if site.megamorphic() {
        CallState::Megamorphic
    } else {
        match filled {
            0 => CallState::Empty,
            1 => CallState::Monomorphic,
            _ => CallState::Polymorphic,
        }
    };
    CallView { state, ways }
}

/// A call cache entry's target; None for an empty entry.
fn call_target(entry: &CallIc) -> Option<CallTarget> {
    (entry.callee != 0).then(|| CallTarget {
        callee: entry.callee,
        function: entry.func,
        builtin: entry.builtin,
        native: entry.native,
        realm: entry.realm,
        current: entry.epoch == CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
    })
}
