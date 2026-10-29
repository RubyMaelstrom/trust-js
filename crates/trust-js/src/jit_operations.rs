//! One native operation vocabulary for baseline code and SSA region barriers.
//!
//! The caller publishes canonical operands/affected locals before entering this
//! emitter. It therefore preserves the baseline's guards, operand moves,
//! exact-PC checked fallback and exception landing instead of degrading operations
//! inside a region to generic VM dispatch. Control-flow/state publication remains
//! the caller's responsibility. A declined emission writes no instructions.
//!
//! ECMA-262 e28783d5: #sec-declarative-environment-records (uninitialized
//! bindings still throw on GetBindingValue), #sec-getvalue, #sec-putvalue,
//! #sec-ordinaryget and #sec-evaluatecall. Sharing lowering changes no ordering.

use super::*;
use crate::bytecode::Op;

pub(super) fn emit(
    a: &mut asm::Asm,
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    pc: usize,
    fast: u32,
    array_intrinsics_on: bool,
    function_call_intrinsic_on: bool,
    unwind: usize,
    direct_finish: usize,
) -> bool {
    let op = &chunk.jit_ops()[pc];
    match op {
        Op::TypeofIs(test, negate) if fast & 2 != 0 && layout.valid => {
            use crate::value::TypeofTest;
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 12, 20, -8);
            emit_exec_kind(a, 12, 9, 14, slow);
            a.cmp_imm_w(9, 1);
            a.b_cond(C_EQ, slow);
            if matches!(
                test,
                TypeofTest::Undefined | TypeofTest::Object | TypeofTest::Function
            ) {
                a.cmp_imm_w(9, 8);
                a.b_cond(C_EQ, slow); // live [[Call]] / [[IsHTMLDDA]] checks
            }
            let kind = if *test == TypeofTest::Object {
                2 // null's typeof is "object"; object inputs took the checked path
            } else {
                *test as u32
            };
            a.cmp_imm_w(9, kind);
            a.cset_w(11, if *negate { C_NE } else { C_EQ });
            a.mov_imm64(9, crate::value::PACK_BOOL);
            a.logic_x(1, 9, 9, 11);
            emit_exec_word_store(a, 9, 20, -8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::ResolveNameRef(..) | Op::LoadRef(_) | Op::StoreRef(_) => {
            if fast & 8192 == 0
                || !names::emit_reference_op(a, layout, ilayout, chunk, op, pc as u32, unwind)
            {
                emit_op_helper(a, H_REFERENCE_OP, pc as u32, unwind);
            }
        }
        Op::GetProp(n, cache) if fast & 256 != 0 && get_method_inlinable(layout) => {
            let arr_ok = !chunk
                .jit_name(*n)
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit());
            emit_prop_load_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_cache_preferred(*cache),
                chunk.jit_name(*n),
                pc as u32,
                unwind,
                false,
                arr_ok,
                PropRecv::Stack,
            );
        }
        Op::GetPropThis(n, cache) if fast & 256 != 0 && get_method_inlinable(layout) => {
            let arr_ok = !chunk
                .jit_name(*n)
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit());
            emit_prop_load_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_cache_preferred(*cache),
                chunk.jit_name(*n),
                pc as u32,
                unwind,
                false,
                arr_ok,
                PropRecv::This,
            );
        }
        Op::GetPropLocal(s, n, cache)
            if fast & 256 != 0 && get_method_inlinable(layout) && (*s as u32) * 8 + 8 < 4096 =>
        {
            let arr_ok = !chunk
                .jit_name(*n)
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit());
            emit_prop_load_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_cache_preferred(*cache),
                chunk.jit_name(*n),
                pc as u32,
                unwind,
                false,
                arr_ok,
                PropRecv::Slot(*s as u32 * 8),
            );
        }
        Op::ToPropKey | Op::ToPropKeyLocal(_) if fast & 64 != 0 => {
            // A Num or Str key passes through untouched (the overwhelmingly common case);
            // anything else — real coercion plus the nullish-base check — takes the helper.
            let slow = a.new_label();
            let done = a.new_label();
            let not_number = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_number_guard(a, 9, 0, 10, not_number);
            a.b(done);
            a.bind(not_number);
            emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 10, slow);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::Dup if fast & 64 != 0 && layout.valid => {
            // Copy the top word as it is.
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::LoadThis if fast & 32768 != 0 && layout.valid => {
            // [48] is the packed `this` word, which owns nothing: push a copy.
            a.ldr_imm(12, 19, 48); // ctx.this_word
            emit_exec_word_store(a, 12, 20, 0);
            a.add_imm(20, 20, 8);
            #[cfg(test)]
            {
                a.mov_imm64(9, chunk.test_this_commits.as_ptr() as u64);
                a.ldr_imm(10, 9, 0);
                a.add_imm(10, 10, 1);
                a.str_imm(10, 9, 0);
            }
        }
        Op::LoadName(_, cache) if fast & 8192 != 0 && load_name_inlinable(layout) => {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(*cache),
                pc as u32,
                unwind,
                false,
            );
        }
        Op::LoadNameIn(_, cache, target) | Op::LoadNameForCallIn(_, cache, target)
            if fast & 8192 != 0 && load_name_inlinable(layout) =>
        {
            // A dead pin's guard always takes the generic call; this site is then unreachable
            // and keeps the checked operation.
            let Some(env) = chunk.inline_target_env(*target) else {
                return false;
            };
            emit_load_name_in_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(*cache),
                crate::interpreter::Env::as_ptr(&env) as u64,
                pc as u32,
                unwind,
                matches!(op, Op::LoadNameForCallIn(..)),
            );
        }
        Op::LoadNameForCall(_, cache) if fast & 8192 != 0 && load_name_inlinable(layout) => {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(*cache),
                pc as u32,
                unwind,
                true,
            );
        }
        Op::UpdateNameCached(_, cache, kind)
            if fast & 8192 != 0 && update_name_inlinable(layout) =>
        {
            emit_update_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(*cache),
                *kind,
                pc as u32,
                unwind,
            );
        }
        Op::StoreNameCached(_, cache) if fast & 8192 != 0 && update_name_inlinable(layout) => {
            emit_store_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(*cache),
                pc as u32,
                unwind,
            );
        }
        Op::LoadCap(name) if fast & 8192 != 0 && fixed_cap(a, chunk, *name).is_some() => {
            let (slot, attrs) = fixed_cap(a, chunk, *name).expect("guarded");
            let tdz = attrs.lexical() && !chunk.jit_cap_initialized(pc);
            emit_cap_load(a, cap_slot_offset(slot), tdz, pc as u32, unwind);
        }
        Op::StoreCap(name) | Op::StoreCapInit(name)
            if fast & 8192 != 0
                && fixed_cap(a, chunk, *name).is_some_and(|(_, attrs)| {
                    attrs.mutable() || matches!(op, Op::StoreCapInit(_))
                }) =>
        {
            let (slot, attrs) = fixed_cap(a, chunk, *name).expect("guarded");
            let tdz =
                matches!(op, Op::StoreCap(_)) && attrs.lexical() && !chunk.jit_cap_initialized(pc);
            emit_cap_store(a, cap_slot_offset(slot), tdz, pc as u32, unwind);
        }
        Op::UpdateCap(name, kind)
            if fast & 8192 != 0
                && fixed_cap(a, chunk, *name).is_some_and(|(_, attrs)| attrs.mutable()) =>
        {
            let (slot, _) = fixed_cap(a, chunk, *name).expect("guarded");
            emit_cap_update(a, cap_slot_offset(slot), *kind, pc as u32, unwind);
        }
        Op::LoadCap(name)
            if fast & 8192 != 0
                && load_name_inlinable(layout)
                && !chunk.jit_needs_activation_state() =>
        {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_cap_cache_ptr(*name),
                pc as u32,
                unwind,
                false,
            );
        }
        Op::StoreCap(name)
            if fast & 8192 != 0
                && update_name_inlinable(layout)
                && !chunk.jit_needs_activation_state() =>
        {
            emit_store_name_inline(a, layout, chunk.jit_cap_cache_ptr(*name), pc as u32, unwind);
        }
        Op::MakeRegExp(..) => {
            emit_op_helper(a, H_MAKE_REGEXP, pc as u32, unwind);
        }
        Op::GetElem if fast & 1024 != 0 && get_elem_inlinable(layout) => {
            emit_get_elem_inline(a, layout, pc as u32, unwind);
        }
        Op::In if fast & 1024 != 0 && get_elem_inlinable(layout) => {
            emit_in_inline(a, layout, pc as u32, unwind);
        }
        Op::SetElemDrop if fast & 2048 != 0 && elem_inlinable(layout) => {
            emit_set_elem_inline(a, layout, pc as u32, unwind, false);
        }
        Op::SetElem if fast & 4096 != 0 && elem_inlinable(layout) => {
            emit_set_elem_inline(a, layout, pc as u32, unwind, true);
        }
        Op::GetElemLocal(slot)
            if fast & 1024 != 0 && get_elem_inlinable(layout) && (*slot as u32) * 8 + 8 < 4096 =>
        {
            emit_elem_local_inline(
                a,
                layout,
                *slot as u32 * 8,
                pc as u32,
                unwind,
                ElemLocalKind::Get,
            );
        }
        Op::SetElemLocalDrop(slot)
            if fast & 2048 != 0
                    && elem_inlinable(layout)
                    // Packed local stores are enabled with the complete numeric-loop pipeline.
                    // In reduced diagnostic masks, mixing this baseline store with helper-side
                    // name/element state can violate Navier's aliasing checksum; the old wide
                    // property layout remains independently safe.
                    && (layout.entry_accessor != layout.entry_value + 8
                        || fast & (1024 | 8192 | 32768 | 262144)
                            == (1024 | 8192 | 32768 | 262144))
                    && (*slot as u32) * 8 + 8 < 4096 =>
        {
            emit_elem_local_inline(
                a,
                layout,
                *slot as u32 * 8,
                pc as u32,
                unwind,
                ElemLocalKind::SetDrop,
            );
        }
        Op::SetElemLocal(slot)
            if fast & 4096 != 0 && elem_inlinable(layout) && (*slot as u32) * 8 + 8 < 4096 =>
        {
            emit_elem_local_inline(
                a,
                layout,
                *slot as u32 * 8,
                pc as u32,
                unwind,
                ElemLocalKind::SetKeep,
            );
        }
        Op::GetMethod(n, cache) if fast & 512 != 0 && get_method_inlinable(layout) => {
            let arr_ok = !chunk
                .jit_name(*n)
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit());
            emit_prop_load_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_cache_preferred(*cache),
                chunk.jit_name(*n),
                pc as u32,
                unwind,
                true,
                arr_ok,
                PropRecv::Stack,
            );
        }
        Op::Add if fast & 1 != 0 => {
            let strings = a.new_label();
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_word_load(a, 10, 20, -8);
            emit_exec_number_guard(a, 9, 0, 11, strings);
            emit_exec_number_guard(a, 10, 1, 11, slow);
            a.f_arith(0, 0, 0, 1);
            emit_exec_number_store(a, 0, 20, -16, 11);
            a.sub_imm(20, 20, 8);
            a.b(done);
            a.bind(strings);
            emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 11, slow);
            emit_exec_tag_guard(a, 10, crate::value::PACK_STR, 11, slow);
            emit_op_helper(a, H_ADD_STRINGS, pc as u32, unwind);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::Sub | Op::Mul | Op::Div if fast & 1 != 0 => {
            let f_op = match op {
                Op::Sub => 1,
                Op::Mul => 2,
                _ => 3,
            };
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_word_load(a, 10, 20, -8);
            emit_exec_number_guard(a, 9, 0, 11, slow);
            emit_exec_number_guard(a, 10, 1, 11, slow);
            a.f_arith(f_op, 0, 0, 1);
            emit_exec_number_store(a, 0, 20, -16, 11);
            a.sub_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::Mod if fast & 1 != 0 => {
            // Number::remainder (ECMA-262 §6.1.6.1.6) of Int32-valued operands is the truncated
            // integer remainder, except that a zero result takes the dividend's sign. Zero
            // divisors (NaN), ±0 dividends (the result is the dividend itself), negative exact
            // multiples (-0) and every non-Int32 operand keep the checked operation.
            let slow = a.new_label();
            let done = a.new_label();
            let nonzero = a.new_label();
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_word_load(a, 10, 20, -8);
            emit_exec_number_guard(a, 9, 0, 11, slow);
            emit_exec_number_guard(a, 10, 1, 11, slow);
            a.fcvtzs_w_d(9, 0);
            a.scvtf_d_w(2, 9);
            a.fcmp(0, 2);
            a.b_cond(C_NE, slow);
            a.fcvtzs_w_d(10, 1);
            a.scvtf_d_w(3, 10);
            a.fcmp(1, 3);
            a.b_cond(C_NE, slow);
            a.cbz(10, false, slow);
            a.cbz(9, false, slow);
            a.sdiv_w(11, 9, 10);
            a.msub_w(12, 11, 10, 9);
            a.cbnz(12, false, nonzero);
            a.cmp_imm_w(9, 0);
            a.b_cond(C_MI, slow);
            a.bind(nonzero);
            a.scvtf_d_w(0, 12);
            a.stur_d(0, 20, -16);
            a.sub_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::BitNot if fast & 1 != 0 => {
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_number_guard(a, 9, 0, 10, slow);
            a.fcvtzs_x_d(9, 0);
            a.scvtf_d_x(1, 9);
            a.frintz(2, 0);
            a.fcmp(1, 2);
            a.b_cond(C_NE, slow);
            a.cmn_imm_x(9, 1);
            a.b_cond(C_VS, slow);
            a.mvn_w(10, 9);
            a.scvtf_d_w(0, 10);
            a.stur_d(0, 20, -8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq
            if fast & 2 != 0 && eq_inlinable(layout) =>
        {
            emit_eq_inline(
                a,
                layout,
                pc as u32,
                unwind,
                matches!(op, Op::StrictEq | Op::StrictNotEq),
                matches!(op, Op::NotEq | Op::StrictNotEq),
                None,
            );
        }
        Op::InstanceOf(cache) if layout.valid && instanceof_inlinable(layout, ilayout) => {
            emit_instanceof_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(*cache),
                pc as u32,
                unwind,
            );
        }
        Op::Not if fast & 131072 != 0 && eq_inlinable(layout) => {
            emit_not_inline(a, layout, pc as u32, unwind);
        }
        Op::Neg if fast & 1 != 0 && layout.valid => {
            // ECMA-262 §13.5.5: the negation of a Number operand is the exact Number (the
            // store canonicalizes a NaN); every other operand takes the checked helper.
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_number_guard(a, 9, 0, 11, slow);
            a.fneg(0, 0);
            emit_exec_number_store(a, 0, 20, -8, 9);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::SetProp(_, cache) | Op::SetPropDrop(_, cache)
            if fast & 65536 != 0 && layout.valid && set_prop_inlinable(layout) =>
        {
            emit_set_prop_inline(
                a,
                layout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_name(match op {
                    Op::SetProp(n, _) | Op::SetPropDrop(n, _) => *n,
                    _ => unreachable!(),
                }),
                pc as u32,
                unwind,
                PropRecv::Stack,
                matches!(op, Op::SetProp(..)),
            );
        }
        Op::SetPropThisDrop(_, cache)
            if fast & 65536 != 0 && layout.valid && set_prop_inlinable(layout) =>
        {
            emit_set_prop_inline(
                a,
                layout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_name(match op {
                    Op::SetPropThisDrop(n, _) => *n,
                    _ => unreachable!(),
                }),
                pc as u32,
                unwind,
                PropRecv::This,
                false,
            );
        }
        Op::SetPropLocalDrop(s, _, cache)
            if fast & 65536 != 0
                && layout.valid
                && set_prop_inlinable(layout)
                && (*s as u32) * 8 + 8 < 4096 =>
        {
            emit_set_prop_inline(
                a,
                layout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_name(match op {
                    Op::SetPropLocalDrop(_, n, _) => *n,
                    _ => unreachable!(),
                }),
                pc as u32,
                unwind,
                PropRecv::Slot(*s as u32 * 8),
                false,
            );
        }
        // `o.k += v` as a statement (ECMA-262 AssignmentExpression evaluation step 2: GetValue,
        // ApplyStringOrNumericBinaryOperator, PutValue), its read made by the GetProp before:
        // two Numbers add in line, which runs no author code, and the sum is stored through the
        // store template under the operation's own store cache, whose helper (and its setter
        // call) takes the operation as a `SetPropDrop` of [receiver, sum]. Strings, which append
        // in place, and every other pair keep the generic operation, with nothing changed.
        Op::AppendProp(n, cache)
            if fast & 1 != 0 && fast & 65536 != 0 && layout.valid && set_prop_inlinable(layout) =>
        {
            let generic = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_word_load(a, 10, 20, -8);
            emit_exec_number_guard(a, 9, 0, 11, generic);
            emit_exec_number_guard(a, 10, 1, 11, generic);
            a.f_arith(0, 0, 0, 1);
            emit_exec_number_store(a, 0, 20, -16, 11);
            a.sub_imm(20, 20, 8);
            emit_set_prop_inline(
                a,
                layout,
                chunk.jit_cache_ptr(*cache),
                chunk.jit_name(*n),
                pc as u32,
                unwind,
                PropRecv::Stack,
                false,
            );
            a.b(done);
            a.bind(generic);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::UpdateProp(_, cache, kind)
            if fast & 65536 != 0 && layout.valid && set_prop_inlinable(layout) =>
        {
            emit_update_prop_inline(
                a,
                layout,
                chunk.jit_cache_ptr(*cache),
                *kind,
                pc as u32,
                unwind,
            );
        }
        Op::UpdateElem(kind)
            if fast & 2048 != 0
                && fast & 65536 != 0
                && layout.valid
                && elem_inlinable(layout)
                && layout.entry_accessor == layout.entry_value + 8 =>
        {
            emit_update_elem_inline(a, layout, *kind, pc as u32, unwind);
        }
        Op::Lt
        | Op::Gt
        | Op::Le
        | Op::Ge
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::EqEq
        | Op::NotEq
            if fast & 2 != 0 =>
        {
            // Number-number compare: FCMP + CSET with IEEE-correct conditions (unordered
            // yields false for the ordered relations, true only for !=).
            let cond = match op {
                Op::Lt => C_MI,
                Op::Gt => C_GT,
                Op::Le => C_LS,
                Op::Ge => C_GE,
                Op::StrictEq | Op::EqEq => C_EQ,
                _ => C_NE,
            };
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_word_load(a, 10, 20, -8);
            emit_exec_number_guard(a, 9, 0, 11, slow);
            emit_exec_number_guard(a, 10, 1, 11, slow);
            a.fcmp(0, 1);
            a.cset_w(9, cond);
            a.mov_imm64(10, crate::value::PACK_BOOL);
            a.logic_x(1, 9, 9, 10);
            emit_exec_word_store(a, 9, 20, -16);
            a.sub_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::LoadLocal(slot) if fast & 8 != 0 && layout.valid && (*slot as u32) * 8 + 8 < 4096 => {
            let off = *slot as i32 * 8;
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 22, off);
            a.mov_imm64(10, crate::value::PACK_EMPTY);
            a.cmp_reg_x(9, 10); // Empty = TDZ throw → slow
            a.b_cond(C_EQ, slow);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::StoreLocal(slot) if fast & 16 != 0 && (*slot as u32) * 8 + 8 < 4096 => {
            emit_store_local(
                a,
                *slot as u32 * 8,
                &[pc as u32],
                unwind,
                false,
                layout.valid,
            );
        }
        Op::UpdateLocal(slot, kind) if fast & 32 != 0 && (*slot as u32) * 8 + 8 < 4096 => {
            emit_update_local(a, *slot, *kind, pc as u32, unwind);
        }
        Op::Pop if fast & 64 != 0 && layout.valid => {
            a.sub_imm(20, 20, 8);
        }
        Op::DestructureGuard if fast & 1024 != 0 => {
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            a.mov_imm64(10, crate::value::PACK_UNDEFINED);
            a.cmp_reg_x(9, 10);
            a.b_cond(C_EQ, slow);
            a.mov_imm64(10, crate::value::PACK_NULL);
            a.cmp_reg_x(9, 10);
            a.b_cond(C_NE, done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::ToStr if fast & 1024 != 0 => {
            // ToString(String) returns the existing value. Keep its canonical owner
            // in place; observable conversions and Symbol failures remain checked.
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 10, slow);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::Void if fast & (64 | 128) == (64 | 128) && layout.valid => {
            // GetValue/effects already ran: the operand becomes undefined.
            a.mov_imm64(9, crate::value::PACK_UNDEFINED);
            emit_exec_word_store(a, 9, 20, -8);
        }
        Op::Undef if fast & 128 != 0 => {
            a.mov_imm64(9, crate::value::PACK_UNDEFINED);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::Const(k) if fast & 128 != 0 && chunk.jit_const_copyable(*k) => {
            let Some(bits) = chunk.jit_const_scalar_bits(*k) else {
                return false;
            };
            a.mov_imm64(9, bits);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        // The constant's word owns nothing and names a cell the chunk's pool roots while this
        // code can run, so it is an immediate (see `jit_const_str_word`).
        Op::Const(k) if fast & 128 != 0 && chunk.jit_const_is_str(*k) => {
            let Some(word) = chunk.jit_const_str_word(*k) else {
                return false;
            };
            a.mov_imm64(9, word);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::DestructureGuard => {
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            a.mov_imm64(10, crate::value::PACK_UNDEFINED);
            a.cmp_reg_x(9, 10);
            a.b_cond(C_EQ, slow);
            a.mov_imm64(10, crate::value::PACK_NULL);
            a.cmp_reg_x(9, 10);
            a.b_cond(C_NE, done); // anything but Null
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        // Block entry puts the slot in its temporal dead zone: the store overwrites the previous
        // iteration's value (no generic dispatch).
        Op::Tdz(slot) if fast & 16 != 0 && layout.valid && (*slot as u32) * 8 + 8 < 4096 => {
            let off = *slot as i32 * 8;
            a.mov_imm64(9, crate::value::PACK_EMPTY);
            emit_exec_word_store(a, 9, 22, off);
        }
        Op::ResetSlots(start, count)
            if layout.valid && (*start as u32 + *count as u32) * 8 < 4096 =>
        {
            for k in *start..*start + *count {
                let off = k as i32 * 8;
                a.mov_imm64(9, crate::value::PACK_UNDEFINED);
                emit_exec_word_store(a, 9, 22, off);
            }
        }
        Op::Call(..) | Op::CallWithThis(..) => {
            emit_call_inline(
                a,
                chunk,
                layout,
                ilayout,
                pc,
                fast,
                array_intrinsics_on,
                function_call_intrinsic_on,
                unwind,
                direct_finish,
            );
        }
        Op::MakeObject(..) => {
            emit_op_helper(a, H_MAKE_OBJECT, pc as u32, unwind);
        }
        Op::MakeClosure(..) => {
            emit_op_helper(a, H_MAKE_CLOSURE, pc as u32, unwind);
        }
        Op::NewObject
        | Op::ObjectData(_)
        | Op::ObjectDataName(..)
        | Op::ObjectProto
        | Op::ObjectSpread
        | Op::ObjectMethod(..)
        | Op::ObjectMethodName(..) => {
            emit_op_helper(a, H_OBJECT_LITERAL, pc as u32, unwind);
        }
        Op::MakeArray(..) => {
            emit_op_helper(a, H_MAKE_ARRAY, pc as u32, unwind);
        }
        Op::Abstract(crate::bytecode::AbstractOp::CreateDataPropertyOrThrow)
            if fast & 4096 != 0 && define_elem_inlinable(layout) =>
        {
            emit_define_elem_inline(a, layout, pc as u32, unwind);
        }
        Op::Abstract(_) => {
            emit_op_helper(a, H_ABSTRACT, pc as u32, unwind);
        }
        Op::New(argc, cache) => {
            // A site whose construct cache holds its constructor constructs through the direct
            // sequence; a miss, or a gate the sequence declines, takes the helper.
            let slow = a.new_label();
            let done = a.new_label();
            emit_direct_construct(
                a,
                chunk,
                layout,
                ilayout,
                *argc as usize,
                *cache,
                fast,
                slow,
                unwind,
                done,
                direct_finish,
            );
            a.bind(slow);
            // H_NEW needs only the pc for optional diagnostics and the statically encoded
            // arity. Pack both so the million-call constructor path does not reload/decode
            // its bytecode op in Rust.
            a.mov(0, 19);
            a.movz(1, pc as u32, 0);
            a.movk(1, *argc as u32, 1);
            a.mov(2, 20);
            a.ldr_imm(16, 21, (H_NEW * 8) as u32);
            a.blr(16);
            a.mov(20, 0);
            a.cbnz(1, false, unwind);
            // Either form lands here with its result pushed. A native constructor allocates
            // after the call's last safepoint.
            a.bind(done);
            emit_helper_end_poll(a, unwind);
        }
        Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::SetPropLocalDrop(..) => {
            emit_op_helper(a, H_SET_PROP, pc as u32, unwind);
        }
        Op::GetProp(..) | Op::GetPropThis(..) | Op::GetPropLocal(..) | Op::GetMethod(..) => {
            emit_op_helper(a, H_GET_PROP, pc as u32, unwind);
        }
        Op::GetMethodElem => {
            if fast & 1024 != 0 && get_method_inlinable(layout) && ilayout.valid {
                emit_computed_method_inline(a, layout, ilayout, pc as u32, unwind);
            } else {
                emit_op_helper(a, H_GET_METHOD_ELEM, pc as u32, unwind);
            }
        }
        Op::GetElem | Op::GetElemLocal(_) => {
            emit_op_helper(a, H_GET_ELEM, pc as u32, unwind);
        }
        Op::DestructureArr(_)
        | Op::TryArrayBinding(..)
        | Op::ArrayBindingStep(..)
        | Op::IterStepL(..) => {
            emit_op_helper(a, H_ITERATION, pc as u32, unwind);
        }
        // Decision 20e's elided `arguments` and rest reads: in a fresh frame the slot holds
        // Undefined and the actual arguments are the record's (`JitCtx::argc`, `JitCtx::args`).
        // A filled slot (a borrowed frame's, or a materialized object) and every key that is
        // not an in-range array index take the operation's helper (`jit_exec`).
        Op::ArgLength(slot, ..) | Op::RestLength(slot, ..) => {
            let skip = if matches!(op, Op::RestLength(..)) {
                chunk.jit_frame().0 as u32
            } else {
                0
            };
            emit_elided_length(a, *slot, skip, pc as u32, unwind);
        }
        Op::ArgElem(slot) | Op::RestElem(slot) => {
            let skip = if matches!(op, Op::RestElem(_)) {
                chunk.jit_frame().0 as u32
            } else {
                0
            };
            // A mapped body's index below its formals' count reads its formal's binding
            // (Decision 20f): at its fixed slot in a fresh frame, else through the helper.
            let mapped = match op {
                Op::ArgElem(_) => chunk
                    .jit_mapped_formals()
                    .map(|(count, first)| (count, first.filter(|_| a.fixed_activation()))),
                _ => None,
            };
            emit_elided_element(a, *slot, skip, mapped, pc as u32, unwind);
        }
        _ => return false,
    }
    true
}

/// Whether the elided value's slot, at `slot`, still holds the fresh frame's Undefined (x9 and
/// x10 scratch); otherwise branch to `slow`.
fn emit_elided_slot_check(a: &mut asm::Asm, slot: u16, slow: usize) {
    a.ldr_imm(9, 22, u32::from(slot) * 8);
    a.mov_imm64(10, crate::value::PACK_UNDEFINED);
    a.cmp_reg_x(9, 10);
    a.b_cond(C_NE, slow);
}

/// [`Op::ArgLength`] and [`Op::RestLength`]: the actual count less the formals a rest
/// parameter follows (`skip`, at least 0), pushed as a Number.
fn emit_elided_length(a: &mut asm::Asm, slot: u16, skip: u32, pc: u32, unwind: usize) {
    let slow = a.new_label();
    let done = a.new_label();
    emit_elided_slot_check(a, slot, slow);
    a.ldr_imm(9, 19, CTX_ARGC_OFF as u32);
    if skip != 0 {
        a.subs_imm_w(9, 9, skip);
        a.csel_x(9, 31, 9, C_MI); // fewer arguments than formals: none
    }
    a.ucvtf_d_w(0, 9);
    a.fmov_x_d(9, 0);
    emit_exec_word_store(a, 9, 20, 0);
    a.add_imm(20, 20, 8);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, unwind);
    a.bind(done);
}

/// [`Op::ArgElem`] and [`Op::RestElem`]: a key that is a Number naming an int32 index `k >= 0`
/// with `k + skip` below the count reads argument `k + skip` in place of the key; anything else
/// (another key, an index past the end, a filled slot) takes the helper. `mapped`, a mapped
/// body's (`Chunk::jit_mapped_formals`): an index below its formals' count reads the formal's
/// binding instead, at the fixed slots from the given first one, or without one through the
/// helper.
fn emit_elided_element(
    a: &mut asm::Asm,
    slot: u16,
    skip: u32,
    mapped: Option<(u32, Option<u32>)>,
    pc: u32,
    unwind: usize,
) {
    let slow = a.new_label();
    let done = a.new_label();
    emit_elided_slot_check(a, slot, slow);
    // An int32 Number: its truncation converts back exactly (every other execution word is a
    // NaN, which compares unordered); -0 is index 0.
    a.ldur_d(0, 20, -8);
    a.fcvtzs_w_d(11, 0);
    a.scvtf_d_w(1, 11);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    a.cmp_imm_w(11, 0);
    a.b_cond(C_MI, slow);
    if skip != 0 {
        a.add_imm(11, 11, skip);
    }
    a.ldr_imm(12, 19, CTX_ARGC_OFF as u32);
    a.cmp_reg_w(11, 12);
    a.b_cond(C_HS, slow);
    if let Some((count, first)) = mapped {
        let unmapped = a.new_label();
        a.cmp_imm_w(11, count);
        a.b_cond(C_HS, unmapped);
        match first {
            Some(first) => {
                // The formal's word: its binding at slot `first + k` of the activation.
                a.ldr_imm(13, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
                a.add_imm(13, 13, cap_slot_offset(first));
                a.ldr_x_lsl3(14, 13, 11);
                emit_exec_word_store(a, 14, 20, -8);
                a.b(done);
            }
            None => a.b(slow),
        }
        a.bind(unmapped);
    }
    a.ldr_imm(13, 19, CTX_ARGC_OFF as u32 + 8); // the arguments' address
    a.ldr_x_lsl3(14, 13, 11);
    emit_exec_word_store(a, 14, 20, -8);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, unwind);
    a.bind(done);
}

/// The fixed slot and attributes of the captured binding name `n` names, when the code being
/// emitted runs only in activations its chunk's plan made (`Asm::fixed_activation`; a borrowed
/// frame's may be another tier's) and the chunk keeps it at a fixed slot
/// (`Chunk::jit_cap_binding`).
fn fixed_cap(
    a: &asm::Asm,
    chunk: &Chunk,
    n: u32,
) -> Option<(u32, crate::interpreter::BindingAttrs)> {
    a.fixed_activation()
        .then(|| chunk.jit_cap_binding(n))
        .flatten()
}

/// A captured binding's fixed offset from its activation (`env_raw`): the record's header, then
/// one word a slot (W3 item 3's step 5).
fn cap_slot_offset(slot: u32) -> u32 {
    (crate::interpreter::RECORD_HEADER + 8 * slot as usize) as u32
}

/// LoadCap from a fixed slot of the running activation (`Chunk::jit_cap_binding`): its word,
/// tested for TDZ (the empty word) unless `tdz` is false (a binding never in TDZ, or one its
/// basic block checked already, `Chunk::jit_cap_initialized`). A binding's word is never a
/// deferred prototype, which only a property holds.
fn emit_cap_load(a: &mut asm::Asm, offset: u32, tdz: bool, pc: u32, unwind: usize) {
    a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
    a.ldr_imm(12, 9, offset);
    if tdz {
        let slow = a.new_label();
        let done = a.new_label();
        a.movz(13, 0x7ffa, 3); // PACK_EMPTY
        a.cmp_reg_x(12, 13);
        a.b_cond(C_EQ, slow);
        emit_exec_word_store(a, 12, 20, 0);
        a.add_imm(20, 20, 8);
        #[cfg(test)]
        names::record_name_commit(a, 0);
        a.b(done);
        a.bind(slow);
        emit_exec(a, pc, unwind);
        a.bind(done);
    } else {
        emit_exec_word_store(a, 12, 20, 0);
        a.add_imm(20, 20, 8);
        #[cfg(test)]
        names::record_name_commit(a, 0);
    }
}

/// StoreCap (or StoreCapInit, `tdz` false) into a fixed slot of the running activation: the
/// operand's word, after the TDZ test of the old word when `tdz`, with the card mark of the
/// record's start unless the word is a Number (as `bytecode::Chunk::store_cap_ic` and the name
/// stores do). The binding is mutable (a const's stores are StoreConstCap) or being initialized.
fn emit_cap_store(a: &mut asm::Asm, offset: u32, tdz: bool, pc: u32, unwind: usize) {
    let slow = a.new_label();
    let done = a.new_label();
    let number = a.new_label();
    a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
    if tdz {
        a.ldr_imm(12, 9, offset);
        a.movz(13, 0x7ffa, 3); // PACK_EMPTY
        a.cmp_reg_x(12, 13);
        a.b_cond(C_EQ, slow);
    }
    emit_exec_word_load(a, 8, 20, -8);
    // Property-only and reserved words take the checked path, as in the name stores.
    property_word_guard(a, 8, 10, slow);
    a.str_imm(8, 9, offset);
    a.fmov_d_x(0, 8);
    a.fcmp(0, 0);
    a.b_cond(7, number); // VC: a Number word needs no young-edge barrier
    a.mov_imm64(10, crate::value::PACK_CANON_NAN);
    a.cmp_reg_x(8, 10);
    a.b_cond(C_EQ, number);
    emit_card_mark(a, 9, 15, 16);
    a.bind(number);
    a.sub_imm(20, 20, 8);
    #[cfg(test)]
    names::record_name_commit(a, 1);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, unwind);
    a.bind(done);
}

/// UpdateCap (`++`/`--`) on a fixed slot of the running activation holding a Number: the empty
/// word (TDZ) and every other non-Number are NaN words, which take the checked path, as do
/// numeric NaNs (their canonicalization). The binding is mutable.
fn emit_cap_update(a: &mut asm::Asm, offset: u32, kind: UpdKind, pc: u32, unwind: usize) {
    let slow = a.new_label();
    let done = a.new_label();
    a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
    a.ldr_d_imm(0, 9, offset);
    a.fcmp(0, 0);
    a.b_cond(C_VS, slow);
    a.fmov_one(1);
    let dec = matches!(
        kind,
        UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
    );
    a.f_arith(if dec { 1 } else { 0 }, 2, 0, 1);
    a.str_d_imm(2, 9, offset);
    match kind {
        UpdKind::PreInc | UpdKind::PreDec => {
            emit_exec_number_store(a, 2, 20, 0, 10);
            a.add_imm(20, 20, 8);
        }
        UpdKind::PostInc | UpdKind::PostDec => {
            emit_exec_number_store(a, 0, 20, 0, 10);
            a.add_imm(20, 20, 8);
        }
        UpdKind::IncDiscard | UpdKind::DecDiscard => {}
    }
    #[cfg(test)]
    names::record_name_commit(a, 2);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, unwind);
    a.bind(done);
}
