//! ARM64 ordinary numeric typed-array templates. Receiver/index guards are supplied by the
//! enclosing element template. All guards precede stores; no helper, allocation or call occurs
//! between the fresh backing-store checks and the access. X10 and D0 remain intact for receiver
//! ownership and deferred local-key updates. Scratch: X9, X12..X17, D1..D2.
//!
//! ECMA-262 TypedArrayGetElement/TypedArraySetElement (§10.4.5.15-16), IsValidIntegerIndex and
//! NumericToRawBytes (§25.1.3.17), local snapshot e28783d5fc9d. The template serves fixed-length
//! views: an index below the element count of a view that fits the live buffer is a valid integer
//! index, so its byte address is `data + byteOffset + (index << shift)` and nothing else needs
//! checking. A length-tracking view registers an element count of zero and an end of `usize::MAX`,
//! so it always takes the checked operation, as do detached buffers, out-of-bounds views (a
//! resizable buffer shrank below the view's end), non-integral, negative or large keys, and
//! coercing stores.

use super::{
    asm, emit_exec_number_guard, emit_exec_word_load, jscvt_available, C_EQ, C_GE, C_HS, C_LO,
    C_LS, C_VS,
};
use crate::native_typed_array::{
    Layout as NativeLayout, NativeBuffer as Buffer, NativeTypedArray as View,
};
use crate::value::JitLayout;
use std::mem::offset_of;

#[derive(Clone, Copy)]
pub(super) enum WriteValue {
    Stack,
    Register(u32),
}

// Element kinds, as registered by `Interp::register_native_typed_array`.
const U8: u32 = 0;
const I8: u32 = 1;
const U16: u32 = 2;
const I16: u32 = 3;
const U32: u32 = 4;
const I32: u32 = 5;
const F32: u32 = 6;
const F64: u32 = 7;
const U8_CLAMPED: u32 = 8;

/// X11 = Object, X9 = exact unsigned index. On read, X12 is a packed Number and D1 its
/// numeric value. The caller chooses the ownership/stack commit at `hit`.
pub(super) fn emit(
    a: &mut asm::Asm,
    layout: &JitLayout,
    write: Option<WriteValue>,
    hit: usize,
    slow: usize,
) {
    let Some(native) = layout.native_typed_array else {
        a.b(slow);
        return;
    };
    a.mark_push(if write.is_some() {
        "typed-store"
    } else {
        "typed-load"
    });
    match write {
        Some(WriteValue::Stack) => {
            emit_exec_word_load(a, 14, 20, -8);
            emit_exec_number_guard(a, 14, 1, 16, slow);
        }
        Some(WriteValue::Register(register)) => a.fmov_d_d(1, register),
        None => {}
    }
    emit_access(a, &native, write.is_some(), hit, slow);
    a.mark_pop();
}

/// The element access proper: X11 = Object, X9 = exact unsigned index (and D1 the value to
/// store). Every guard branches to `slow` before the first write; the access ends at `hit`.
///
/// After the receiver/index guards the template loads the live view, buffer and backing Vec,
/// leaving X15 at the address of element zero (the Vec's data plus the view's byte offset), X13
/// at the buffer record, X14 at the view's byte offset and X17 at the element kind; every
/// element then addresses `[X15, X9, lsl #shift]` directly. Scratch: X9, X12..X17, D1..D2.
pub(super) fn emit_access(
    a: &mut asm::Asm,
    layout: &NativeLayout,
    write: bool,
    hit: usize,
    slow: usize,
) {
    a.ldr_imm(12, 11, layout.object_view as u32);
    a.cbz(12, true, slow); // proxies, namespaces and unsupported typed-array kinds
    a.ldr_imm(13, 12, offset_of!(View, buffer_ptr) as u32);
    a.ldr_imm(14, 12, offset_of!(View, limit) as u32);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow); // also every length-tracking view (element count zero)
    a.ldrb_imm(14, 13, offset_of!(Buffer, attached) as u32);
    a.cbz(14, false, slow);
    if write {
        a.ldrb_imm(14, 13, offset_of!(Buffer, writable) as u32);
        a.cbz(14, false, slow);
    }
    a.ldr_imm(16, 13, offset_of!(Buffer, storage_ptr) as u32);
    a.ldr_imm(14, 16, 0); // validated RefCell borrow flag: raw access requires no Rust borrow
    a.cbnz(14, true, slow);
    a.ldr_imm(15, 16, (layout.refcell_value + layout.vec_len) as u32);
    a.ldr_imm(14, 12, offset_of!(View, end) as u32);
    a.cmp_reg_x(15, 14);
    a.b_cond(C_LO, slow); // an out-of-bounds fixed view is invalid in its entirety
    a.ldr_imm(15, 16, (layout.refcell_value + layout.vec_ptr) as u32);
    a.ldr_imm(14, 12, offset_of!(View, offset) as u32);
    a.add_shifted(15, 15, 14, 0);
    a.ldrb_imm(17, 12, offset_of!(View, kind) as u32);
    if write {
        emit_store(a, hit, slow);
    } else {
        emit_load(a, hit);
    }
}

/// Packed Number for a floating-point load: any NaN becomes the canonical word.
fn emit_float_result(a: &mut asm::Asm, hit: usize) {
    a.fmov_x_d(12, 1);
    a.fcmp(1, 1);
    a.b_cond(C_VS ^ 1, hit);
    a.mov_imm64(12, crate::value::PACK_CANON_NAN);
    a.b(hit);
}

/// Element kinds are tested in the order DOS-style and canvas code uses them (Int32, Uint8,
/// Float64, the 16-bit pair, Uint32, Int8, Float32); Uint8Clamped reads like Uint8 and is the
/// fall-through.
fn emit_load(a: &mut asm::Asm, hit: usize) {
    let order = [I32, U8, F64, U16, I16, U32, I8, F32];
    let labels: Vec<usize> = order.iter().map(|_| a.new_label()).collect();
    for (&kind, &label) in order.iter().zip(&labels) {
        a.cmp_imm_w(17, kind);
        a.b_cond(C_EQ, label);
    }
    // Uint8 (and Uint8Clamped): the byte, zero extended
    let u8_label = labels[1];
    a.bind(u8_label);
    a.ldrb_reg(14, 15, 9);
    a.ucvtf_d_w(1, 14);
    a.fmov_x_d(12, 1);
    a.b(hit);
    for (&kind, &label) in order.iter().zip(&labels) {
        if kind == U8 {
            continue;
        }
        a.bind(label);
        match kind {
            I32 => {
                a.ldr_w_lsl2(14, 15, 9);
                a.scvtf_d_w(1, 14);
            }
            U32 => {
                a.ldr_w_lsl2(14, 15, 9);
                a.ucvtf_d_w(1, 14);
            }
            U16 => {
                a.ldrh_lsl1(14, 15, 9);
                a.ucvtf_d_w(1, 14);
            }
            I16 => {
                a.ldrsh_lsl1(14, 15, 9);
                a.scvtf_d_w(1, 14);
            }
            I8 => {
                a.ldrsb_reg(14, 15, 9);
                a.scvtf_d_w(1, 14);
            }
            F64 => {
                a.ldr_d_lsl3(1, 15, 9);
                emit_float_result(a, hit);
                continue;
            }
            F32 => {
                a.ldr_s_lsl2(1, 15, 9);
                a.fcvt_float_width(1, 1, false); // FCVT D1, S1 is exact
                emit_float_result(a, hit);
                continue;
            }
            _ => unreachable!(),
        }
        // Integer elements cannot be NaN: the converted value is already canonical.
        a.fmov_x_d(12, 1);
        a.b(hit);
    }
}

/// ToInt8/ToUint8/ToInt16/ToUint16/ToInt32/ToUint32 of the number in D1 into W16: each is the
/// value truncated, modulo 2^N, and 2^N divides 2^32, so the low bits of ToInt32 serve every
/// width. FJCVTZS is exactly ToInt32 for every Number (NaN and the infinities give zero), so
/// no value declines. Without FEAT_JSCVT the values FCVTZS X converts without saturation,
/// [-2^31, 2^32), take the instruction and every other Number the checked operation.
fn emit_to_integer(a: &mut asm::Asm, slow: usize) {
    if jscvt_available() {
        a.fjcvtzs_w_d(16, 1);
        return;
    }
    a.mov_imm64(12, (-2147483648.0_f64).to_bits());
    a.fmov_d_x(2, 12);
    a.fcmp(1, 2);
    a.b_cond(C_GE ^ 1, slow); // signed LT also rejects unordered (NaN)
    a.mov_imm64(12, 4294967296.0_f64.to_bits());
    a.fmov_d_x(2, 12);
    a.fcmp(1, 2);
    a.b_cond(C_GE, slow);
    a.fcvtzs_x_d(16, 1);
}

/// The element has been written. X9 = element index, X14 = the view's byte offset: record
/// `[start, end)` and one write in the buffer's bookkeeping. The bytes are authoritative
/// immediately, including when a later instruction throws. A conservative interval and
/// generation are folded into host metadata on demand.
fn emit_dirty(a: &mut asm::Asm, hit: usize, shift: u32, width: u32) {
    a.add_shifted(9, 14, 9, shift); // start = byteOffset + (index << shift)
    a.add_imm(16, 9, width); // end
    let start_done = a.new_label();
    a.ldr_imm(14, 13, offset_of!(Buffer, dirty_start) as u32);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, start_done);
    a.str_imm(9, 13, offset_of!(Buffer, dirty_start) as u32);
    a.bind(start_done);
    let end_done = a.new_label();
    a.ldr_imm(14, 13, offset_of!(Buffer, dirty_end) as u32);
    a.cmp_reg_x(16, 14);
    a.b_cond(C_LS, end_done);
    a.str_imm(16, 13, offset_of!(Buffer, dirty_end) as u32);
    a.bind(end_done);
    a.ldr_imm(14, 13, offset_of!(Buffer, writes) as u32);
    a.add_imm(14, 14, 1);
    a.str_imm(14, 13, offset_of!(Buffer, writes) as u32);
    a.b(hit);
}

/// Stores the number in D1. Every guard that can decline precedes the first store. Kinds are
/// grouped by width; a width shares its code (all integer kinds store the same low bits).
fn emit_store(a: &mut asm::Asm, hit: usize, slow: usize) {
    let word = a.new_label();
    let byte = a.new_label();
    let half = a.new_label();
    let float32 = a.new_label();
    let float64 = a.new_label();
    a.sub_imm(16, 17, U32);
    a.cmp_imm_w(16, I32 - U32);
    a.b_cond(C_LS, word); // Uint32, Int32
    a.cmp_imm_w(17, I8);
    a.b_cond(C_LS, byte); // Uint8, Int8
    a.cmp_imm_w(17, I16);
    a.b_cond(C_LS, half); // Uint16, Int16
    a.cmp_imm_w(17, F64);
    a.b_cond(C_EQ, float64);
    a.cmp_imm_w(17, F32);
    a.b_cond(C_EQ, float32);
    debug_assert_eq!(U8_CLAMPED, 8);
    // Uint8ClampedArray (ToUint8Clamp): round to nearest, ties to even, then clamp to
    // [0, 255]. FCVTNS saturates and maps NaN to 0, so no value declines.
    let non_negative = a.new_label();
    let store_clamped = a.new_label();
    a.fcvtns_w_d(16, 1);
    a.cmp_imm_w(16, 0);
    a.b_cond(C_GE, non_negative);
    a.movz(16, 0, 0);
    a.b(store_clamped);
    a.bind(non_negative);
    a.cmp_imm_w(16, 255);
    a.b_cond(C_LS, store_clamped);
    a.movz(16, 255, 0);
    a.bind(store_clamped);
    a.strb_reg(16, 15, 9);
    emit_dirty(a, hit, 0, 1);

    a.bind(word);
    emit_to_integer(a, slow);
    a.str_w_lsl2(16, 15, 9);
    emit_dirty(a, hit, 2, 4);

    a.bind(byte);
    emit_to_integer(a, slow);
    a.strb_reg(16, 15, 9);
    emit_dirty(a, hit, 0, 1);

    a.bind(half);
    emit_to_integer(a, slow);
    a.strh_lsl1(16, 15, 9);
    emit_dirty(a, hit, 1, 2);

    a.bind(float32);
    a.fcvt_float_width(2, 1, true); // FCVT S2, D1 (roundTiesToEven)
    a.str_s_lsl2(2, 15, 9);
    emit_dirty(a, hit, 2, 4);

    a.bind(float64);
    a.str_d_lsl3(1, 15, 9);
    emit_dirty(a, hit, 3, 8);
}

#[cfg(test)]
mod tests {
    #[test]
    fn typed_element_encodings_match_the_assembler() {
        // `llvm-mc --triple=aarch64 --show-encoding`, registers x15/x9 and w14/s1/s2/d1.
        let cases: [(&str, u32, fn(&mut super::asm::Asm)); 10] = [
            ("ldr w14, [x15, x9, lsl #2]", 0xb86979ee, |a| {
                a.ldr_w_lsl2(14, 15, 9)
            }),
            ("ldrsb w14, [x15, x9]", 0x38e969ee, |a| {
                a.ldrsb_reg(14, 15, 9)
            }),
            ("ldrh w14, [x15, x9, lsl #1]", 0x786979ee, |a| {
                a.ldrh_lsl1(14, 15, 9)
            }),
            ("ldrsh w14, [x15, x9, lsl #1]", 0x78e979ee, |a| {
                a.ldrsh_lsl1(14, 15, 9)
            }),
            ("ldr s1, [x15, x9, lsl #2]", 0xbc6979e1, |a| {
                a.ldr_s_lsl2(1, 15, 9)
            }),
            ("strb w14, [x15, x9]", 0x382969ee, |a| a.strb_reg(14, 15, 9)),
            ("strh w14, [x15, x9, lsl #1]", 0x782979ee, |a| {
                a.strh_lsl1(14, 15, 9)
            }),
            ("str w14, [x15, x9, lsl #2]", 0xb82979ee, |a| {
                a.str_w_lsl2(14, 15, 9)
            }),
            ("str s2, [x15, x9, lsl #2]", 0xbc2979e2, |a| {
                a.str_s_lsl2(2, 15, 9)
            }),
            ("ldrb w14, [x15, x9]", 0x386969ee, |a| a.ldrb_reg(14, 15, 9)),
        ];
        for (text, word, emit) in cases {
            let mut a = super::asm::Asm::new();
            emit(&mut a);
            assert_eq!(a.finish(), vec![word], "{text}");
        }
    }
}
