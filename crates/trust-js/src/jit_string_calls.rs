//! Fused String method calls (Djot's gap, candidate 4, step 2): `str.codePointAt(i)`, its
//! `GetMethod`, its index and its `CallWithThis(1)`, as one step while the active Realm's
//! String.prototype still holds the native the call names (the design note's fork 2, decided:
//! SpiderMonkey's Baseline attaches such a stub at the call for a primitive receiver, and JSC's
//! thunk and V8's optimizing tiers choose at the call too). The guard is String.prototype's fuse
//! (`crate::object_fuse`, SpiderMonkey's GuardObjectFuseProperty): a primitive String receiver,
//! the site's fuse being the active Realm's, the fuse's word or else the method's own state, and
//! a Number index. Any miss runs the unfused templates from the `GetMethod`, which nothing the
//! step did has changed: the index operations it reads ahead (a local, a Number constant, a
//! local plus or minus a Number constant) run no author code and cannot throw while the local
//! holds a Number, and the step reads the local only to test that.
//!
//! The step is chosen where the call's cache holds the native, or holds nothing (the VM fills
//! no native entry, and a fused call fills none). In a loop the step is in line; elsewhere the
//! site computes the index and calls the shared stub ([`SharedStub::StringCall`]), as the
//! compact templates keep straight-line code small.
//!
//! ECMA-262 e28783d5: EvaluateCall's GetValue of the member reference (a primitive String base's
//! lookup starts at its prototype: a String's own properties are its indices and `length`),
//! String.prototype.codePointAt (spec.html:36270) and CodePointAt (:16721).

use super::*;
use crate::bytecode::Op;
use crate::object_fuse::{ObjectFuse, StringCallSite};

#[cfg(test)]
thread_local! {
    /// Fused calls finished, for tests.
    static FUSED_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn fused_calls() -> usize {
    FUSED_CALLS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) extern "C" fn record_fused_call() {
    FUSED_CALLS.with(|count| count.set(count.get() + 1));
}

/// The index operand a fused call computes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Index {
    /// `LoadLocal`: the frame slot's byte offset.
    Local(u32),
    /// A Number `Const`, an exact unsigned one: its bits.
    Number(u64),
    /// `LoadLocal; Const; Add` or `Sub`, the constant an integer Number.
    Offset { local: u32, bits: u64, sub: bool },
}

/// A call the step fuses: its method's call-cache class, its index and the pc of its
/// `CallWithThis(1)`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub(super) intrinsic: u8,
    pub(super) index: Index,
    pub(super) call_pc: usize,
}

impl Plan {
    /// The fused call whose `GetMethod` is at `pc`, if the step takes it: nothing enters the
    /// operations it reads ahead, and the call's cache admits the native.
    pub(super) fn at(
        chunk: &Chunk,
        ops: &[Op],
        pc: usize,
        targeted: &[bool],
        fast: u32,
    ) -> Option<Plan> {
        let Op::GetMethod(name, _) = ops[pc] else {
            return None;
        };
        // The GetMethod and call templates' own ablation bits; detailed feedback observes every
        // operation in its helper.
        if fast & 512 == 0 || fast & 524288 == 0 || chunk.jit_detailed_feedback_enabled() {
            return None;
        }
        let intrinsic = match chunk.jit_name(name) {
            "codePointAt" => crate::bytecode::INTRINSIC_CODE_POINT_AT,
            _ => return None,
        };
        let local = |slot: u16| {
            let offset = u32::from(slot) * 8;
            (offset + 8 < 4096).then_some(offset)
        };
        // A constant index the read takes (an exact unsigned one), and an integer offset: a
        // constant the read would never take only adds a test.
        let exact = |k: u32| {
            chunk
                .jit_const_num(k)
                .filter(|&bits| f64::from_bits(bits) == f64::from(f64::from_bits(bits) as u32))
        };
        let integer = |k: u32| {
            chunk.jit_const_num(k).filter(|&bits| {
                let value = f64::from_bits(bits);
                value == f64::from(value as i32)
            })
        };
        let (index, call_pc) = match (ops.get(pc + 1), ops.get(pc + 2), ops.get(pc + 3)) {
            (Some(&Op::LoadLocal(slot)), Some(&Op::Const(k)), Some(op @ (Op::Add | Op::Sub)))
                if integer(k).is_some() =>
            {
                let index = Index::Offset {
                    local: local(slot)?,
                    bits: integer(k)?,
                    sub: matches!(op, Op::Sub),
                };
                (index, pc + 4)
            }
            (Some(&Op::LoadLocal(slot)), ..) => (Index::Local(local(slot)?), pc + 2),
            (Some(&Op::Const(k)), ..) => (Index::Number(exact(k)?), pc + 2),
            _ => return None,
        };
        let Some(&Op::CallWithThis(1, cache)) = ops.get(call_pc) else {
            return None;
        };
        if call_pc + 1 >= ops.len()
            || targeted[pc + 1..=call_pc].iter().any(|&entered| entered)
            || !chunk.jit_call_site_admits(cache, intrinsic)
        {
            return None;
        }
        Some(Plan {
            intrinsic,
            index,
            call_pc,
        })
    }
}

/// d0 = the index, or a branch to `miss` when its local holds no Number. Clobbers x9, x16 and d1.
fn emit_index(a: &mut asm::Asm, index: Index, miss: usize) {
    match index {
        Index::Local(local) => {
            emit_exec_word_load(a, 9, 22, local as i32);
            emit_exec_number_guard(a, 9, 0, 16, miss);
        }
        Index::Number(bits) => {
            a.mov_imm64(9, bits);
            a.fmov_d_x(0, 9);
        }
        Index::Offset { local, bits, sub } => {
            emit_exec_word_load(a, 9, 22, local as i32);
            emit_exec_number_guard(a, 9, 0, 16, miss);
            a.mov_imm64(9, bits);
            a.fmov_d_x(1, 9);
            a.f_arith(u32::from(sub), 0, 0, 1);
        }
    }
}

/// The fused step at the `GetMethod` of `plan`, ahead of its unfused templates; `after` is the
/// label past the call.
pub(super) fn emit_site(
    a: &mut asm::Asm,
    chunk: &Chunk,
    ilayout: &crate::interpreter::InterpLayout,
    pc: usize,
    plan: &Plan,
    after: usize,
) {
    let unfused = a.new_label();
    let site = chunk.jit_string_call_site(pc as u32, plan.intrinsic);
    emit_index(a, plan.index, unfused);
    // The site is the chunk's, which owns this code: its address may be embedded here.
    a.mov_imm64(12, site as u64);
    if use_shared_stub(a) {
        let stub = a.shared_stub(
            SharedStub::StringCall {
                intrinsic: plan.intrinsic,
            }
            .key(),
        );
        a.bl_label(stub);
        a.cbnz(9, false, unfused);
        a.b(after);
    } else {
        emit_body(
            a,
            ilayout.string_proto_fuse,
            plan.intrinsic,
            false,
            unfused,
            after,
        );
    }
    a.bind(unfused);
}

/// The shared stub of [`SharedStub::StringCall`]. Process-wide: it reads the site from x12 and
/// the active Realm's fuse through ctx.interp, and embeds no address.
pub(super) fn emit_stub(
    a: &mut asm::Asm,
    ilayout: Option<&crate::interpreter::InterpLayout>,
    intrinsic: u8,
) {
    let miss = a.new_label();
    let done = a.new_label();
    match ilayout.filter(|ilayout| ilayout.valid) {
        Some(ilayout) => emit_body(a, ilayout.string_proto_fuse, intrinsic, true, miss, done),
        None => a.b(miss),
    }
    a.bind(done);
    a.movz(9, 0, 0);
    a.ret();
    a.bind(miss);
    a.movz(9, 1, 0);
    a.ret();
}

/// The step's guards and read, the receiver at `[x20 - 8]`, x12 = the site and d0 = the index (a
/// Number): the result replaces the receiver and the code branches to `done`, or it branches to
/// `miss` with nothing written. `fuse_off` is `InterpLayout::string_proto_fuse`. The fill is a
/// call: in a stub (`stub`), whose fast path keeps the site's return address in x30 alone, it
/// keeps x30 in a frame around it. Clobbers x0..x17, x30 on the fill, d0 and d1.
fn emit_body(
    a: &mut asm::Asm,
    fuse_off: usize,
    intrinsic: u8,
    stub: bool,
    miss: usize,
    done: usize,
) {
    let fill = a.new_label();
    let constant = a.new_label();
    let read = a.new_label();
    // A primitive String receiver.
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, miss);
    // x13 = the site's fuse, x10 = the active Realm's.
    a.ldr_imm(13, 12, StringCallSite::FUSE_OFF as u32);
    a.ldr_imm(10, 19, 72); // ctx.interp
    a.ldr_imm(10, 10, fuse_off as u32);
    a.cmp_reg_x(13, 10);
    a.b_cond(C_NE, fill);
    // The fuse's word: 0 while no Constant property was written or removed.
    a.ldr_w_imm(14, 13, ObjectFuse::INVALIDATED_OFF as u32);
    a.cbnz(14, false, constant);
    a.bind(read);
    match intrinsic {
        crate::bytecode::INTRINSIC_CODE_POINT_AT => {
            emit_code_point_read(a, -8, 0, miss, done, ReadCommit::Fused)
        }
        _ => unreachable!("a fused call's method has a read"),
    }
    // Else the generation the site recorded, and the method's NotConstant bit clear. Another
    // generation (a tracked property was removed) refills the site.
    a.bind(constant);
    a.ldr_w_imm(14, 13, ObjectFuse::GENERATION_OFF as u32);
    a.ldr_w_imm(16, 12, StringCallSite::GENERATION_OFF as u32);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, fill);
    a.ldr_imm(14, 13, ObjectFuse::WORDS_OFF as u32);
    a.ldr_w_imm(16, 12, StringCallSite::INDEX_OFF as u32);
    a.add_shifted(14, 14, 16, 2);
    a.ldr_w_imm(14, 14, 0);
    a.ldr_w_imm(16, 12, StringCallSite::MASK_OFF as u32);
    a.logic_w(0, 14, 14, 16);
    a.cbz(14, false, read);
    a.b(miss);
    // The fill: none while the Realm has no fuse (its built-ins are being installed), and none
    // for a site refused in this Realm (its fuse's address | 1). This call runs unfused.
    a.bind(fill);
    a.cbz(10, true, miss);
    a.sub_imm(16, 13, 1);
    a.cmp_reg_x(16, 10);
    a.b_cond(C_EQ, miss);
    if stub {
        a.stp_pre(29, 30, -16);
    }
    a.mov(0, 19);
    a.mov(1, 12);
    a.ldr_imm(16, 21, (H_STRING_CALL_FILL * 8) as u32);
    a.blr(16);
    if stub {
        a.ldp_post(29, 30, 16);
    }
    a.b(miss);
}
