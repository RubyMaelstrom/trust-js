//! The feedback census's view of a chunk (W4 step 2, `architecture-diagnostics` only): each
//! operation's census kind and the production feedback its caches hold (see
//! `crate::feedback_census`).
use super::*;

/// Census kinds: feedback::OperationKind's numbers for its kinds, then 11 comparison, 12 free-name
/// and captured-variable sites, 13 `typeof`, `in`, `instanceof` and `!`; 0 for the rest.
pub(crate) const CENSUS_COMPARISON: u8 = 11;
pub(crate) const CENSUS_NAME: u8 = 12;
pub(crate) const CENSUS_TYPE_OPS: u8 = 13;

pub(crate) fn census_kind(op: &Op, pc: usize) -> u8 {
    use crate::feedback::OperationKind as K;
    (match op {
        Op::GetProp(..) | Op::GetPropThis(..) | Op::GetPropLocal(..) | Op::GetMethod(..) => {
            K::NamedLoad
        }
        Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::SetPropLocalDrop(..)
        | Op::AppendProp(..)
        | Op::UpdateProp(..) => K::NamedStore,
        Op::GetElem | Op::GetElemLocal(..) | Op::GetMethodElem => K::ElementLoad,
        Op::SetElem
        | Op::SetElemDrop
        | Op::SetElemLocal(..)
        | Op::SetElemLocalDrop(..)
        | Op::UpdateElem(..) => K::ElementStore,
        Op::Call(..)
        | Op::CallWithThis(..)
        | Op::CallSpread(..)
        | Op::CallSpreadThis(..)
        | Op::CallArgsArray
        | Op::CallArgsArrayThis
        | Op::EvalCallArgsArray
        | Op::TailEvalCallArgsArray
        | Op::TailCall(..) => K::Call,
        Op::New(..) | Op::NewArgsArray | Op::SuperCallArgsArray => K::Construct,
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Mod
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::UShr
        | Op::GenBin(..)
        | Op::Neg
        | Op::Plus
        | Op::BitNot
        | Op::UpdateLocal(..)
        | Op::UpdateCap(..)
        | Op::UpdateName(..)
        | Op::UpdateNameCached(..)
        | Op::UpdateConst(..)
        | Op::UpdatePrivate(..)
        | Op::SuperUpdate(..) => K::Arithmetic,
        Op::JumpIfFalse(..)
        | Op::JumpIfFalsePeek(..)
        | Op::JumpIfTruePeek(..)
        | Op::JumpIfNotNullishPeek(..) => K::Branch,
        Op::Jump(target) if (*target as usize) <= pc => K::Loop,
        Op::MakeClosure(..)
        | Op::MakeRegExp(..)
        | Op::MakeArray(..)
        | Op::MakeArrayLiteral(_)
        | Op::NewArray
        | Op::MakeObject(..)
        | Op::NewObject => K::Allocation,
        Op::Lt
        | Op::Gt
        | Op::Le
        | Op::Ge
        | Op::EqEq
        | Op::NotEq
        | Op::StrictEq
        | Op::StrictNotEq => return CENSUS_COMPARISON,
        Op::LoadName(..)
        | Op::LoadNameForCall(..)
        | Op::LoadNameIn(..)
        | Op::LoadNameForCallIn(..)
        | Op::StoreName(..)
        | Op::StoreGlobalName(..)
        | Op::StoreNameCached(..)
        | Op::LoadCap(..)
        | Op::StoreCap(..) => return CENSUS_NAME,
        Op::Typeof
        | Op::TypeofIs(..)
        | Op::TypeofName(..)
        | Op::In
        | Op::InstanceOf(..)
        | Op::Not => return CENSUS_TYPE_OPS,
        _ => return 0,
    }) as u8
}

impl Chunk {
    /// Merge this chunk's production feedback, per pc, into the census, at its drop and at the
    /// shell's exit. The state word, by kind: a named site's filled ways in bits 0-2 and its
    /// uncacheable, accessor, absent and creation ways in bits 3-6; a call site's filled ways in
    /// bits 0-2, its megamorphic state in bit 3, a native way in bit 4 and its first builtin id in
    /// bits 16-31; bit 0 for a filled construct, `instanceof` or name cache and for a made
    /// allocation image (bit 1: a literal's template or shared block exists).
    pub(crate) fn feedback_census_snapshot(&self) {
        let Some(census) = self.feedback.census.as_deref() else {
            return;
        };
        let named = |cache: u32| {
            let mut bits = 0u32;
            for way in self.caches.iter().skip(cache as usize).take(PROP_IC_WAYS) {
                let state = way.get();
                if state.depth == IC_EMPTY {
                    continue;
                }
                bits += 1;
                if !state.has_cacheable_shapes() {
                    bits |= 1 << 3;
                }
                match state.depth {
                    IC_ABSENT => bits |= 1 << 5,
                    IC_CREATE => bits |= 1 << 6,
                    depth if depth & IC_ACCESSOR != 0 => bits |= 1 << 4,
                    _ => {}
                }
            }
            bits
        };
        let call = |cache: u32| {
            let Some(site) = self.call_caches.get(cache as usize) else {
                return 0;
            };
            let mut bits = u32::from(site.megamorphic()) << 3;
            let mut builtin = 0u32;
            for entry in &site.entries {
                let entry = entry.get();
                if entry.callee == 0 {
                    continue;
                }
                bits += 1;
                if entry.native != 0 {
                    bits |= 1 << 4;
                }
                if builtin == 0 {
                    builtin = u32::from(entry.builtin);
                }
            }
            bits | (builtin << 16)
        };
        let ops = self
            .ops
            .iter()
            .enumerate()
            .map(|(pc, op)| {
                let state = match *op {
                    Op::GetProp(_, cache)
                    | Op::GetPropThis(_, cache)
                    | Op::GetPropLocal(_, _, cache)
                    | Op::GetMethod(_, cache)
                    | Op::SetProp(_, cache)
                    | Op::SetPropDrop(_, cache)
                    | Op::SetPropThisDrop(_, cache)
                    | Op::SetPropLocalDrop(_, _, cache)
                    | Op::AppendProp(_, cache)
                    | Op::UpdateProp(_, cache, _) => named(cache),
                    Op::Call(_, cache)
                    | Op::CallWithThis(_, cache)
                    | Op::CallSpread(_, cache)
                    | Op::CallSpreadThis(_, cache) => call(cache),
                    Op::New(_, cache) => u32::from(
                        self.construct_caches
                            .get(cache as usize)
                            .is_some_and(|site| site.get().call.callee != 0),
                    ),
                    Op::InstanceOf(cache) => u32::from(
                        self.caches
                            .get(cache as usize)
                            .is_some_and(|way| way.get().depth != IC_EMPTY),
                    ),
                    Op::LoadName(_, cache)
                    | Op::LoadNameForCall(_, cache)
                    | Op::UpdateNameCached(_, cache, _)
                    | Op::StoreNameCached(_, cache) => u32::from(
                        self.name_caches
                            .get(cache as usize)
                            .is_some_and(|ic| ic.get().env != 0),
                    ),
                    Op::MakeObject(_, _, template) if template != u32::MAX => {
                        2 | u32::from(
                            self.obj_images
                                .get(template as usize)
                                .is_some_and(|image| image.ready.get() != 0),
                        )
                    }
                    Op::MakeArray(_) => u32::from(self.empty_array_image.ready.get() != 0),
                    Op::MakeArrayLiteral(site) => {
                        u32::from(
                            self.array_images
                                .get(site as usize)
                                .is_some_and(|image| image.ready.get() != 0),
                        ) | (u32::from(
                            self.array_literals
                                .get(site as usize)
                                .is_some_and(|literal| literal.block.get().is_some()),
                        ) << 1)
                    }
                    Op::MakeRegExp(..) => u32::from(
                        self.regexp_literals
                            .get(pc)
                            .is_some_and(|literal| literal.get().is_some()),
                    ),
                    _ => 0,
                };
                crate::feedback_census::OpSnapshot {
                    op: format!("{op:?}"),
                    kind: census_kind(op, pc),
                    state,
                    record: self.feedback.record_word(pc),
                }
            })
            .collect();
        crate::feedback_census::merge(census, ops, !self.jit.untried());
    }
}
