//! ARM64 ordinary numeric typed-array templates. Receiver/index guards are supplied by the
//! enclosing element template. All guards precede stores; no helper, allocation or call occurs
//! between the fresh backing-store checks and the access. X10 and D0 remain intact for receiver
//! ownership and deferred local-key updates. Scratch: X9, X12..X17, D1..D2.

use super::{
    asm, emit_exec_number_guard, emit_exec_word_load, C_EQ, C_GE, C_HI, C_HS, C_LO, C_LS, C_VS,
};
use crate::native_typed_array::{NativeBuffer as Buffer, NativeTypedArray as View};
use crate::value::JitLayout;
use std::mem::offset_of;

#[derive(Clone, Copy)]
pub(super) enum WriteValue {
    Stack,
    Register(u32),
}

/// X11 = Object, X9 = exact unsigned index. On read, X12 is a packed Number and D1 its
/// numeric value. The caller chooses the ownership/stack commit at `hit`.
pub(super) fn emit(
    a: &mut asm::Asm,
    layout: &JitLayout,
    write: Option<WriteValue>,
    hit: usize,
    slow: usize,
) {
    let Some(layout) = layout.native_typed_array else {
        a.b(slow);
        return;
    };
    a.mark_push(if write.is_some() {
        "typed-store"
    } else {
        "typed-load"
    });
    a.ldr_imm(12, 11, layout.object_view as u32);
    a.cbz(12, true, slow); // proxies, namespaces and unsupported typed-array kinds
    if let Some(value) = write {
        match value {
            WriteValue::Stack => {
                emit_exec_word_load(a, 14, 20, -8);
                emit_exec_number_guard(a, 14, 1, 16, slow);
            }
            WriteValue::Register(register) => a.fmov_d_d(1, register),
        }
    }
    a.ldr_imm(13, 12, offset_of!(View, buffer_ptr) as u32);
    a.ldrb_imm(14, 13, offset_of!(Buffer, attached) as u32);
    a.cbz(14, false, slow);
    if write.is_some() {
        a.ldrb_imm(14, 13, offset_of!(Buffer, writable) as u32);
        a.cbz(14, false, slow);
    }
    a.ldr_imm(14, 12, offset_of!(View, limit) as u32);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    a.ldr_imm(16, 13, offset_of!(Buffer, storage_ptr) as u32);
    a.ldr_imm(14, 16, 0); // validated RefCell borrow flag: raw access requires no Rust borrow
    a.cbnz(14, true, slow);
    a.ldr_imm(15, 16, (layout.refcell_value + layout.vec_len) as u32);
    a.ldr_imm(14, 12, offset_of!(View, end) as u32);
    a.cmp_reg_x(15, 14);
    a.b_cond(C_LO, slow); // an out-of-bounds fixed view is invalid in its entirety
    a.ldr_imm(14, 12, offset_of!(View, offset) as u32);
    a.sub_reg(15, 15, 14); // remaining live bytes, after the view's byteOffset
    a.ldrb_imm(17, 12, offset_of!(View, shift) as u32);
    // LSLV X9, X9, X17. The input is a u32 and shift <= 3, so multiplication cannot overflow.
    a.lsl_reg(9, 9, 17);
    a.movz(14, 1, 0);
    a.lsl_reg(14, 14, 17); // element byte width
    a.cmp_reg_x(15, 14);
    a.b_cond(C_LO, slow);
    a.sub_reg(15, 15, 14);
    a.cmp_reg_x(9, 15);
    a.b_cond(C_HI, slow); // length-tracking views include only whole elements
    a.ldr_imm(17, 12, offset_of!(View, offset) as u32);
    a.add_shifted(9, 9, 17, 0); // absolute byte offset; bounded above by the live Vec length
    a.add_shifted(15, 9, 14, 0); // exclusive end for dirty tracking
    a.ldr_imm(16, 16, (layout.refcell_value + layout.vec_ptr) as u32);
    a.add_shifted(16, 16, 9, 0);
    a.ldrb_imm(17, 12, offset_of!(View, kind) as u32);
    if write.is_some() {
        emit_store(a, slow);
        // The bytes are authoritative immediately, including when a later instruction throws.
        // A conservative interval and generation are folded into host metadata on demand.
        let start_done = a.new_label();
        a.ldr_imm(14, 13, offset_of!(Buffer, dirty_start) as u32);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, start_done);
        a.str_imm(9, 13, offset_of!(Buffer, dirty_start) as u32);
        a.bind(start_done);
        let end_done = a.new_label();
        a.ldr_imm(14, 13, offset_of!(Buffer, dirty_end) as u32);
        a.cmp_reg_x(15, 14);
        a.b_cond(C_LO, end_done);
        a.str_imm(15, 13, offset_of!(Buffer, dirty_end) as u32);
        a.bind(end_done);
        a.ldr_imm(14, 13, offset_of!(Buffer, writes) as u32);
        a.add_imm(14, 14, 1);
        a.str_imm(14, 13, offset_of!(Buffer, writes) as u32);
    } else {
        emit_load(a);
        a.fmov_x_d(12, 1);
        a.fcmp(1, 1);
        let number = a.new_label();
        a.b_cond(C_VS ^ 1, number);
        a.mov_imm64(12, crate::value::PACK_CANON_NAN);
        a.bind(number);
    }
    a.mark_pop();
    a.b(hit);
}

fn emit_load(a: &mut asm::Asm) {
    let labels: Vec<_> = (0..9).map(|_| a.new_label()).collect();
    let done = a.new_label();
    // Kind 7 (Float64) is the fall-through; Uint8Clamped (8) reads as Uint8.
    for kind in [0, 1, 2, 3, 4, 5, 6, 8] {
        a.cmp_imm_w(17, kind);
        a.b_cond(super::C_EQ, labels[kind as usize]);
    }
    a.b(labels[7]);
    for (kind, &label) in labels.iter().enumerate() {
        a.bind(label);
        match kind {
            0 | 8 => {
                a.ldrb_imm(14, 16, 0);
                a.ucvtf_d_w(1, 14);
            }
            1 => {
                // LDRSB W14, [X16], sign-extended to 32 bits
                a.ldr_signed_packed(14, 16, false);
                a.scvtf_d_w(1, 14);
            }
            2 => {
                a.ldrh_imm(14, 16, 0);
                a.ucvtf_d_w(1, 14);
            }
            3 => {
                // LDRSH W14, [X16]
                a.ldr_signed_packed(14, 16, true);
                a.scvtf_d_w(1, 14);
            }
            4 => {
                a.ldr_w_imm(14, 16, 0);
                a.ucvtf_d_w(1, 14);
            }
            5 => {
                a.ldr_w_imm(14, 16, 0);
                a.scvtf_d_w(1, 14);
            }
            6 => {
                // LDR S1, [X16]; FCVT D1, S1
                a.float32_memory(1, 16, false);
                a.fcvt_float_width(1, 1, false);
            }
            7 => a.ldr_d_imm(1, 16, 0),
            _ => unreachable!(),
        }
        a.b(done);
    }
    a.bind(done);
}

fn emit_store(a: &mut asm::Asm, slow: usize) {
    let float32 = a.new_label();
    let float64 = a.new_label();
    let half = a.new_label();
    let word = a.new_label();
    let done = a.new_label();
    let clamped = a.new_label();
    let store_byte = a.new_label();
    a.cmp_imm_w(17, 8);
    a.b_cond(C_EQ, clamped);
    a.cmp_imm_w(17, 6);
    a.b_cond(super::C_EQ, float32);
    a.b_cond(C_HI, float64);
    // ToInt32/ToUint32: truncation followed by modulo 2^32. FCVTZS X accepts the common
    // range [-2^31, 2^32) without saturation; the packed stores discard the high bits.
    // NaN/infinities and larger Numbers use the complete checked conversion.
    a.mov_imm64(14, (-2147483648.0_f64).to_bits());
    a.fmov_d_x(2, 14);
    a.fcmp(1, 2);
    a.b_cond(C_GE ^ 1, slow); // signed LT also rejects unordered (NaN)
    a.mov_imm64(14, 4294967296.0_f64.to_bits());
    a.fmov_d_x(2, 14);
    a.fcmp(1, 2);
    a.b_cond(C_GE, slow);
    a.fcvtzs_x_d(14, 1);
    a.cmp_imm_w(17, 4);
    a.b_cond(C_HS, word);
    a.cmp_imm_w(17, 2);
    a.b_cond(C_HS, half);
    a.strb_imm(14, 16, 0);
    a.b(done);
    a.bind(half);
    a.strh(14, 16);
    a.b(done);
    a.bind(word);
    a.str_w_imm(14, 16, 0);
    a.b(done);
    a.bind(float32);
    a.fcvt_float_width(2, 1, true); // FCVT S2, D1 (roundTiesToEven)
    a.float32_memory(2, 16, true); // STR S2, [X16]
    a.b(done);
    a.bind(float64);
    a.str_d_imm(1, 16, 0);
    a.b(done);
    // Uint8ClampedArray (ECMA-262 ToUint8Clamp): round to nearest, ties to even, then clamp
    // to [0, 255]. FCVTNS saturates and maps NaN to 0, so no value declines.
    a.bind(clamped);
    a.fcvtns_w_d(14, 1);
    a.cmp_imm_w(14, 0);
    let non_negative = a.new_label();
    a.b_cond(C_GE, non_negative);
    a.movz(14, 0, 0);
    a.b(store_byte);
    a.bind(non_negative);
    a.cmp_imm_w(14, 255);
    a.b_cond(C_LS, store_byte);
    a.movz(14, 255, 0);
    a.bind(store_byte);
    a.strb_imm(14, 16, 0);
    a.bind(done);
}
