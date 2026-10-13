//! Effects on PRIVATE packed frame words, not on JavaScript's heap or environments.
//!
//! ECMA-262 e28783d5, FunctionDeclarationInstantiation, Arguments Exotic Objects and
//! PerformEval: captured, mapped-argument and direct-eval-visible bindings must remain
//! observable. `bytecode::compile_inner` homes those bindings in Environment Records;
//! they are NOT the private `Home::Slot` storage described here. Reentrant JavaScript
//! may change those records, object fields, shapes, prototypes, caches and roots even
//! when none of this activation's private packed words changes.
//!
//! All checked entries still publish canonical local/operand OWNERS. This contract only
//! allows the compiler to preserve their non-owning register copies. It relies on the
//! production non-moving Rc heap and the call ABI restoring the caller's frame fields
//! before normal/throw return (including shared-context callees and tail draining).
//! Dynamic completion landings reload from canonical storage independently.

use crate::bytecode::Op;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlotWrites {
    None,
    One(u16),
    Range(u16, u16),
    All,
}

impl SlotWrites {
    pub(super) fn contains(self, slot: u16) -> bool {
        match self {
            Self::None => false,
            Self::One(written) => written == slot,
            Self::Range(start, count) => {
                // Widen before addition: a range ending at physical slot 65535
                // must not wrap its exclusive upper bound to zero.
                u32::from(slot) >= u32::from(start)
                    && u32::from(slot) < u32::from(start) + u32::from(count)
            }
            Self::All => true,
        }
    }
}

/// The helpers selected by this backend's `Lowering::lower` and `slow`, including
/// their guard-miss paths. This is NOT an opcode-wide purity claim applicable to
/// every tier: e.g. template `jit_add_strings` may retire the following store's
/// owner early, whereas this backend's Add uses `jit_exec` / `jit_bin_num`.
/// Changing the selected helper requires reviewing this contract too.
pub(super) fn checked_writes(op: &Op) -> SlotWrites {
    match op {
        Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) | Op::Tdz(slot) => SlotWrites::One(*slot),
        Op::ResetSlots(start, count) => SlotWrites::Range(*start, *count),
        Op::ForInStepL(_, cursor, _) => SlotWrites::One(*cursor),
        Op::DestructureStepL(_, _, done)
        | Op::DestructureRestL(_, _, done)
        | Op::IterCloseIfNotDoneL(_, done)
        | Op::IterAbortIfNotDoneL(_, done) => SlotWrites::One(*done),

        // jit_exec_inner's scalar, environment and ordinary property operations.
        // Getters/proxies/coercions can reenter and mutate referenced objects;
        // only the packed identity in the caller's private slot is preserved.
        Op::Const(_)
        | Op::Undef
        | Op::Dup
        | Op::Dup2
        | Op::Pop
        | Op::LoadLocal(_)
        | Op::LoadCap(_)
        | Op::StoreCap(_)
        | Op::StoreCapInit(_)
        | Op::UpdateCap(..)
        | Op::LoadName(..)
        | Op::LoadNameForCall(..)
        | Op::StoreName(_)
        | Op::StoreNameCached(..)
        | Op::StoreGlobalName(_)
        | Op::UpdateName(..)
        | Op::UpdateNameCached(..)
        // The single-op VM bridge changes activation.env and publishes env_raw;
        // these operations never copy private slots into an Environment Record.
        // CreatePerIterationEnvironment preserves captured binding identities by
        // cloning that record, while unrelated Home::Slot words remain intact.
        | Op::PushLex(_)
        | Op::PushCatchLex(_)
        | Op::CloneLex(_)
        | Op::InitLex(_)
        | Op::PopEnv
        | Op::LoadThis
        | Op::MakeClosure(..)
        | Op::MakeRegExp(..)
        | Op::GetProp(..)
        | Op::GetPropThis(..)
        | Op::GetPropLocal(..)
        | Op::GetMethod(..)
        | Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::SetPropLocalDrop(..)
        | Op::AppendProp(..)
        | Op::GetElem
        | Op::GetElemLocal(_)
        | Op::GetMethodElem
        | Op::SetElem
        | Op::SetElemDrop
        | Op::SetElemLocal(_)
        | Op::SetElemLocalDrop(_)
        | Op::Abstract(_)
        | Op::UpdateProp(..)
        | Op::UpdateElem(_)
        | Op::ToPropKey
        | Op::ToPropKeyLocal(_)
        | Op::ToStr
        | Op::DeleteProp(..)
        | Op::DeleteElem(_)
        | Op::DeleteName(_)
        | Op::DeleteSuper
        | Op::Add
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
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::EqEq
        | Op::NotEq
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::InstanceOf(_)
        | Op::In
        | Op::GenBin(_)
        | Op::Neg
        | Op::Plus
        | Op::Not
        | Op::BitNot
        | Op::Typeof
        | Op::TypeofIs(..)
        | Op::TypeofName(_)
        | Op::Void
        | Op::GetIter
        | Op::ForInKeys
        | Op::IterStepL(..)
        | Op::IterCloseL(_)
        | Op::IterAbortL(_)
        | Op::DestructureGuard
        | Op::DestructureArr(_)
        | Op::NewTarget
        | Op::NewObject
        | Op::ObjectData(_)
        | Op::ObjectDataName(..)
        | Op::ObjectProto
        | Op::ObjectSpread
        | Op::ObjectMethod(..)
        | Op::ObjectMethodName(..)
        | Op::MakeObject(..)
        | Op::MakeArray(_)
        | Op::Call(..)
        | Op::CallWithThis(..)
        | Op::CallSpread(_)
        | Op::CallSpreadThis(_)
        | Op::TailCall(..)
        // Dedicated checked New restores the caller's shared-context fields.
        | Op::New(..)
        // Conditions, handler bookkeeping and static edges cannot write locals.
        | Op::Jump(_)
        | Op::JumpIfFalse(_)
        | Op::JumpIfFalsePeek(_)
        | Op::JumpIfTruePeek(_)
        | Op::JumpIfNotNullishPeek(_)
        | Op::InlineGuard(..)
        | Op::PushHandler(_)
        | Op::PushFinally(..)
        | Op::PushIterator(..)
        | Op::PopHandler => SlotWrites::None,

        // Retained-expression/assignment bridges project slot bindings into a
        // temporary Environment Record and copy back even on throws. Unreviewed
        // bridges, suspension and completion routing keep the full barrier.
        // A new opcode is conservative until its concrete helper is audited.
        _ => SlotWrites::All,
    }
}

/// Whether a selected checked operation can mutate existing property storage,
/// directly or by reentering user code. Canonical publication is still required
/// for ownership and abrupt completion even when this says false. The collector
/// is non-moving; retaining/dropping a frame operand cannot rewrite the contents
/// of a receiver kept alive by another private slot. ToBoolean/typeof do not
/// invoke conversion hooks (ECMA-262 #sec-toboolean, #sec-typeof-operator).
pub(super) fn checked_heap_write(op: &Op) -> bool {
    !matches!(
        op,
        Op::Const(_)
            | Op::Undef
            | Op::Dup
            | Op::Dup2
            | Op::Pop
            | Op::LoadLocal(_)
            | Op::StoreLocal(_)
            | Op::Tdz(_)
            | Op::ResetSlots(..)
            | Op::LoadThis
            | Op::NewTarget
            | Op::Not
            | Op::Typeof
            | Op::TypeofIs(..)
            | Op::Void
            | Op::Jump(_)
            | Op::JumpIfFalse(_)
            | Op::JumpIfFalsePeek(_)
            | Op::JumpIfTruePeek(_)
            | Op::JumpIfNotNullishPeek(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_word_writes_are_distinct_from_heap_and_environment_effects() {
        for op in [
            Op::Call(2, 0),
            Op::CallWithThis(2, 0),
            Op::New(2, 0),
            Op::SetPropLocalDrop(3, 0, 0),
            Op::GetElem,
            Op::StoreCap(0),
            Op::Add,
        ] {
            assert_eq!(checked_writes(&op), SlotWrites::None, "{op:?}");
        }
        for op in [Op::EvalExpr(0), Op::AssignTarget(0), Op::EvalCallArgsArray] {
            assert_eq!(checked_writes(&op), SlotWrites::All, "{op:?}");
        }
    }

    #[test]
    fn iterator_bookkeeping_and_bulk_resets_have_exact_private_write_sets() {
        for op in [
            Op::ForInStepL(1, 3, 2),
            Op::DestructureStepL(1, 2, 3),
            Op::DestructureRestL(1, 2, 3),
            Op::IterCloseIfNotDoneL(1, 3),
            Op::IterAbortIfNotDoneL(1, 3),
        ] {
            assert_eq!(checked_writes(&op), SlotWrites::One(3), "{op:?}");
        }
        let writes = checked_writes(&Op::ResetSlots(u16::MAX - 1, 2));
        assert!(writes.contains(u16::MAX));
        assert!(writes.contains(u16::MAX - 1));
        assert!(!writes.contains(u16::MAX - 2));
        assert!(!writes.contains(0));
        assert!(!checked_writes(&Op::ResetSlots(3, 0)).contains(3));
    }
}
