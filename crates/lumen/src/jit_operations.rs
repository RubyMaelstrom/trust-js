//! One native operation vocabulary for baseline code and SSA region barriers.
//!
//! The caller publishes canonical operands/affected locals before entering this
//! emitter. It therefore preserves the baseline's guards, ownership transfers,
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
    let rc_ok = layout.valid && layout.rc_strong_off < 256;
    let op = &chunk.jit_ops()[pc];
    match op {
        Op::TypeofIs(test, negate) if fast & 2 != 0 && rc_ok && layout.rc_strong_off == 0 => {
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
            // Every declining owner guard precedes the decrement. Last owners and
            // BigInt retain the real destructor through the checked operation.
            emit_exec_drop_shared(a, layout, 12, 13, 14, slow);
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
        Op::Dup if fast & 64 != 0 && rc_ok => {
            // Copy the top value; refcounted payloads bump inline, BigInt takes the helper.
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_clone(a, layout, 9, 10, 11, slow);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::LoadThis if fast & 32768 != 0 && rc_ok => {
            // `this_raw` is deliberately a wide host-boundary Value; encode before
            // acquiring the compact operand owner. BigInt uses the checked helper.
            let slow = a.new_label();
            let done = a.new_label();
            a.ldr_imm(9, 19, 48); // ctx.this_raw
            a.ldr_imm(10, 9, 0);
            a.ldr_imm(11, 9, 8);
            emit_exec_encode_wide(a, 10, 11, 12, 13, 14, 0, slow);
            emit_exec_clone(a, layout, 12, 13, 14, slow);
            emit_exec_word_store(a, 12, 20, 0);
            a.add_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc as u32, unwind);
            a.bind(done);
        }
        Op::LoadName(_, cache) if fast & 8192 != 0 && load_name_inlinable(layout) => {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(*cache),
                chunk.jit_name_number(*cache),
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
                chunk.jit_name_number(*cache),
                std::rc::Rc::as_ptr(&env) as u64,
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
                chunk.jit_name_number(*cache),
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
        Op::LoadCap(name)
            if fast & 8192 != 0
                && load_name_inlinable(layout)
                && !chunk.jit_needs_activation_state() =>
        {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_cap_cache_ptr(*name),
                None,
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
        Op::InstanceOf(cache) if rc_ok && instanceof_inlinable(layout, ilayout) => {
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
            if fast & 65536 != 0 && rc_ok && set_prop_inlinable(layout) =>
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
            if fast & 65536 != 0 && rc_ok && set_prop_inlinable(layout) =>
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
                && rc_ok
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
        Op::UpdateProp(_, cache, kind)
            if fast & 65536 != 0 && rc_ok && set_prop_inlinable(layout) =>
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
        Op::LoadLocal(slot) if fast & 8 != 0 && rc_ok && (*slot as u32) * 8 + 8 < 4096 => {
            let off = *slot as i32 * 8;
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 22, off);
            a.mov_imm64(10, crate::value::PACK_EMPTY);
            a.cmp_reg_x(9, 10); // Empty = TDZ throw → slow
            a.b_cond(C_EQ, slow);
            emit_exec_clone(a, layout, 9, 10, 11, slow);
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
                layout,
                *slot as u32 * 8,
                &[pc as u32],
                unwind,
                false,
                rc_ok,
            );
        }
        Op::UpdateLocal(slot, kind) if fast & 32 != 0 && (*slot as u32) * 8 + 8 < 4096 => {
            emit_update_local(a, *slot, *kind, pc as u32, unwind);
        }
        Op::Pop if fast & 64 != 0 && rc_ok => {
            emit_exec_release_word(a, layout, 20, -8);
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
        Op::Void if fast & (64 | 128) == (64 | 128) && rc_ok => {
            // GetValue/effects already ran. Release the consumed owner exactly as Pop (a last
            // owner's destructor runs through the drop helper), then return undefined.
            emit_exec_release_word(a, layout, 20, -8);
            a.mov_imm64(9, crate::value::PACK_UNDEFINED);
            emit_exec_word_store(a, 9, 20, -8);
        }
        Op::Undef if fast & 128 != 0 => {
            a.mov_imm64(9, crate::value::PACK_UNDEFINED);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::Const(k) if fast & 128 != 0 && chunk.jit_const_copyable(*k) => {
            let Some(bits) = chunk.jit_const_packed_bits(*k) else {
                return false;
            };
            a.mov_imm64(9, bits);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::Const(k)
            if fast & 128 != 0
                && rc_ok
                && layout.rc_strong_off == 0
                && chunk.jit_const_is_str(*k) =>
        {
            a.mov_imm64(9, chunk.jit_const_ptr(*k) as u64);
            a.ldr_imm(11, 9, 8);
            a.mov_imm64(10, crate::value::PACK_STR);
            a.logic_x(1, 10, 10, 11);
            emit_exec_word_store(a, 10, 20, 0);
            a.ldur(13, 11, 0); // strong (payload+0)
            a.add_imm(13, 13, 1);
            a.stur(13, 11, 0);
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
        // Block entry puts the slot in its temporal dead zone. The previous iteration's value
        // is commonly the slot's last owner; release it in place (no generic dispatch).
        Op::Tdz(slot) if fast & 16 != 0 && rc_ok && (*slot as u32) * 8 + 8 < 4096 => {
            let off = *slot as i32 * 8;
            emit_exec_release_word(a, layout, 22, off);
            a.mov_imm64(9, crate::value::PACK_EMPTY);
            emit_exec_word_store(a, 9, 22, off);
        }
        Op::ResetSlots(start, count) if rc_ok && (*start as u32 + *count as u32) * 8 < 4096 => {
            for k in *start..*start + *count {
                let off = k as i32 * 8;
                emit_exec_release_word(a, layout, 22, off);
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
        Op::New(argc, _) => {
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
        _ => return false,
    }
    true
}
