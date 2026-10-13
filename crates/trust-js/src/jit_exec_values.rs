//! ARM64 compact execution-word primitives. Canonical local/operand owners are one
//! PackedValue; wide `this`, return fields and lexical Binding cells are separate ABIs.
//! ECMA-262 §6.1.6.1 (snapshot e28783d5): preserve signed zero/infinities; NaN payload
//! canonicalization is allowed and prevents numeric bits from aliasing internal tags.

use super::*;
use crate::value::{
    PACK_BIGINT, PACK_BOOL, PACK_CANON_NAN, PACK_EMPTY, PACK_NULL, PACK_OBJ, PACK_STR, PACK_SYM,
    PACK_UNDEFINED,
};

/// Decode the semantic discriminant (wide Value ABI 0..=8) without acquiring an
/// owner. Preserve raw, clobber kind/scratch/NZCV, reject property-only tags.
pub(super) fn emit_exec_kind(a: &mut asm::Asm, raw: u32, kind: u32, scratch: u32, slow: usize) {
    assert!(raw != kind && raw != scratch && kind != scratch);
    let object = a.new_label();
    let number = a.new_label();
    let done = a.new_label();
    a.lsr_imm(kind, raw, 48);
    a.movz(scratch, (PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(kind, scratch);
    a.b_cond(C_EQ, object);
    a.b_cond(C_HI, slow);
    a.movz(scratch, (PACK_UNDEFINED >> 48) as u32, 0);
    a.cmp_reg_x(kind, scratch);
    a.b_cond(C_LO, number);
    a.movz(scratch, (PACK_SYM >> 48) as u32, 0);
    a.cmp_reg_x(kind, scratch);
    a.b_cond(C_HI, number);
    a.movz(scratch, (PACK_UNDEFINED >> 48) as u32, 0);
    a.sub_reg(kind, kind, scratch);
    // Positive tags map to 0..3,5..7; Number occupies discriminant 4.
    a.cmp_imm_w(kind, 4);
    a.b_cond(C_LO, done);
    a.add_imm(kind, kind, 1);
    a.b(done);
    a.bind(number);
    a.movz(kind, 4, 0);
    a.b(done);
    a.bind(object);
    a.movz(kind, 8, 0);
    a.bind(done);
}

/// Produce a borrowed wide pair for a compact word: the discriminant in tag_word and every
/// payload, including Bool's 0/1, in payload; refs are untagged. No retain/drop.
/// Raw is preserved; both outputs, kind/scratch and NZCV are clobbered.
pub(super) fn emit_exec_decode_wide(
    a: &mut asm::Asm,
    raw: u32,
    tag_word: u32,
    payload: u32,
    kind: u32,
    scratch: u32,
    slow: usize,
) {
    let regs = [raw, tag_word, payload, kind, scratch];
    for i in 0..regs.len() {
        for j in i + 1..regs.len() {
            assert_ne!(regs[i], regs[j]);
        }
    }
    emit_exec_kind(a, raw, kind, scratch, slow);
    a.mov(tag_word, kind);
    a.movz(payload, 0, 0);
    let number = a.new_label();
    let reference = a.new_label();
    let done = a.new_label();
    a.cmp_imm_w(kind, 4);
    a.b_cond(C_EQ, number);
    a.b_cond(C_HI, reference);
    a.cmp_imm_w(kind, 3);
    a.b_cond(C_NE, done);
    a.movz(scratch, 1, 0);
    a.logic_x(0, payload, raw, scratch);
    a.b(done);
    a.bind(number);
    a.mov(payload, raw);
    a.b(done);
    a.bind(reference);
    emit_exec_payload(a, raw, payload);
    a.bind(done);
}

/// No scratch registers or flags changed. Far-negative addresses need a caller-computed base.
pub(super) fn emit_exec_word_load(a: &mut asm::Asm, dst: u32, base: u32, off: i32) {
    if off >= 0 && off & 7 == 0 {
        a.ldr_imm(dst, base, off as u32);
    } else {
        assert!((-256..256).contains(&off));
        a.ldur(dst, base, off);
    }
}

pub(super) fn emit_exec_word_store(a: &mut asm::Asm, src: u32, base: u32, off: i32) {
    if off >= 0 && off & 7 == 0 {
        a.str_imm(src, base, off as u32);
    } else {
        assert!((-256..256).contains(&off));
        a.stur(src, base, off);
    }
}

/// Borrow a Number from an encoded word. Output dreg is live on success; scratch and
/// NZCV are clobbered. All tagged words are NaNs, but the canonical numeric NaN is valid.
pub(super) fn emit_exec_number_guard(
    a: &mut asm::Asm,
    raw: u32,
    dreg: u32,
    scratch: u32,
    fail: usize,
) {
    assert_ne!(raw, scratch);
    let number = a.new_label();
    a.fmov_d_x(dreg, raw);
    a.fcmp(dreg, dreg);
    a.b_cond(7, number); // VC: every finite value, signed zero and infinity.
    a.mov_imm64(scratch, PACK_CANON_NAN);
    a.cmp_reg_x(raw, scratch);
    a.b_cond(C_NE, fail);
    a.bind(number);
}

/// Compare only the encoded high tag; preserve raw and clobber exactly scratch (no NZCV).
pub(super) fn emit_exec_tag_guard(a: &mut asm::Asm, raw: u32, tag: u64, scratch: u32, fail: usize) {
    assert_ne!(raw, scratch);
    assert_eq!(tag & 0x0000_ffff_ffff_ffff, 0);
    a.mov_imm64(scratch, tag);
    a.logic_x(2, scratch, raw, scratch);
    a.lsr_imm(scratch, scratch, 48);
    a.cbnz(scratch, true, fail);
}

/// Extract a borrowed 48-bit heap payload; raw==dst is allowed, flags are preserved.
pub(super) fn emit_exec_payload(a: &mut asm::Asm, raw: u32, dst: u32) {
    a.ubfx(dst, raw, 0, 48);
}

/// Store a Number as one owned execution word. No ownership traffic. Canonicalize
/// *all* numerical NaNs, including hostile DataView payloads and negative NaN.
/// Only scratch and NZCV are clobbered; dreg is preserved. Callers with a live branch
/// condition must compare after materialization, not across this operation.
pub(super) fn emit_exec_number_store(
    a: &mut asm::Asm,
    dreg: u32,
    base: u32,
    off: i32,
    scratch: u32,
) {
    assert_ne!(base, scratch);
    let number = a.new_label();
    let done = a.new_label();
    a.fcmp(dreg, dreg);
    a.b_cond(7, number);
    a.mov_imm64(scratch, PACK_CANON_NAN);
    emit_exec_word_store(a, scratch, base, off);
    a.b(done);
    a.bind(number);
    if off >= 0 && off & 7 == 0 {
        a.str_d_imm(dreg, base, off as u32);
    } else {
        assert!((-256..256).contains(&off));
        a.stur_d(dreg, base, off);
    }
    a.bind(done);
}

/// Encode a borrowed/moved wide Value pair; ownership does not change here. The low
/// byte of tag_word is its discriminant and every payload uses the second word; only bit 0
/// of a Bool's payload word is meaningful (the rest of that word is padding). BigInt is a thin stored handle, exactly as PackedValue::pack.
/// Inputs are preserved; packed/tag_scratch/const_scratch/fp_scratch and NZCV clobbered.
pub(super) fn emit_exec_encode_wide(
    a: &mut asm::Asm,
    tag_word: u32,
    payload: u32,
    packed: u32,
    tag_scratch: u32,
    const_scratch: u32,
    fp_scratch: u32,
    slow: usize,
) {
    let regs = [tag_word, payload, packed, tag_scratch, const_scratch];
    for i in 0..regs.len() {
        for j in i + 1..regs.len() {
            assert_ne!(regs[i], regs[j]);
        }
    }
    a.logic_imm_w(0, tag_scratch, tag_word, asm::logical_imm_w(0xff).unwrap());
    let number = a.new_label();
    let boolean = a.new_label();
    let done = a.new_label();
    a.cmp_imm_w(tag_scratch, 4);
    a.b_cond(C_EQ, number);
    a.cmp_imm_w(tag_scratch, 3);
    a.b_cond(C_EQ, boolean);
    for (tag, bits, reference) in [
        (8, PACK_OBJ, true),
        (0, PACK_UNDEFINED, false),
        (6, PACK_STR, true),
        (7, PACK_SYM, true),
        (5, PACK_BIGINT, true),
        (1, PACK_EMPTY, false),
        (2, PACK_NULL, false),
    ] {
        let next = a.new_label();
        a.cmp_imm_w(tag_scratch, tag);
        a.b_cond(C_NE, next);
        a.mov_imm64(packed, bits);
        if reference {
            a.logic_x(1, packed, packed, payload);
        }
        a.b(done);
        a.bind(next);
    }
    a.b(slow);
    a.bind(boolean);
    a.logic_imm_w(0, packed, payload, asm::logical_imm_w(1).unwrap());
    a.mov_imm64(const_scratch, PACK_BOOL);
    a.logic_x(1, packed, packed, const_scratch);
    a.b(done);
    a.bind(number);
    a.fmov_d_x(fp_scratch, payload);
    a.fcmp(fp_scratch, fp_scratch);
    a.mov(packed, payload);
    a.b_cond(7, done);
    a.mov_imm64(packed, PACK_CANON_NAN);
    a.bind(done);
}

/// Branch to heap for known shared-Rc execution kinds, reject BigInt's distinct owner
/// representation, otherwise leave scalar owners alone. Both scratch registers and NZCV
/// are clobbered, raw is untouched. Execution storage never contains lazy prototypes.
fn classify_owner(
    a: &mut asm::Asm,
    raw: u32,
    ptr: u32,
    count: u32,
    heap: usize,
    scalar: usize,
    slow: usize,
) {
    assert!(raw != ptr && raw != count && ptr != count);
    a.lsr_imm(ptr, raw, 48);
    a.movz(count, (PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(ptr, count);
    a.b_cond(C_EQ, heap);
    a.b_cond(C_HI, slow); // unknown/property-only tag, never guess an owner type.
    a.movz(count, (PACK_BIGINT >> 48) as u32, 0);
    a.cmp_reg_x(ptr, count);
    a.b_cond(C_LO, scalar);
    a.b_cond(C_EQ, slow);
    // The only high lanes strictly above BigInt and at most Symbol are String/Symbol.
    // Lower lanes are scalars (including positive Numbers); lanes above Symbol but
    // below Object are negative Numbers. Preserve the object-first fast path and
    // reject the same unknown tags as before without inspecting/dereferencing payloads.
    const _: () = assert!(PACK_STR == PACK_BIGINT + (1 << 48));
    const _: () = assert!(PACK_SYM == PACK_STR + (1 << 48));
    const _: () = assert!(PACK_SYM < PACK_OBJ);
    a.movz(count, (PACK_SYM >> 48) as u32, 0);
    a.cmp_reg_x(ptr, count);
    a.b_cond(C_LS, heap);
    a.b(scalar);
}

/// Acquire one owner of raw. Guards precede all changes; slow must clone via the checked
/// operation. Preserve raw, clobber exactly ptr/count/NZCV. The word itself is not stored.
pub(super) fn emit_exec_clone(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    raw: u32,
    ptr: u32,
    count: u32,
    slow: usize,
) {
    let heap = a.new_label();
    let done = a.new_label();
    classify_owner(a, raw, ptr, count, heap, done, slow);
    a.bind(heap);
    emit_exec_payload(a, raw, ptr);
    a.ldur(count, ptr, layout.rc_strong_off as i32);
    a.add_imm(count, count, 1);
    a.stur(count, ptr, layout.rc_strong_off as i32);
    a.bind(done);
}

/// Release one non-final owner. BigInt and last references branch before any decrement;
/// the checked path must run the real destructor. Raw and its physical word are unchanged
/// so the caller can retire/replace the slot only after success. ptr/count/NZCV clobbered.
pub(super) fn emit_exec_drop_shared(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    raw: u32,
    ptr: u32,
    count: u32,
    slow: usize,
) {
    let heap = a.new_label();
    let done = a.new_label();
    classify_owner(a, raw, ptr, count, heap, done, slow);
    a.bind(heap);
    emit_exec_payload(a, raw, ptr);
    a.ldur(count, ptr, layout.rc_strong_off as i32);
    a.cmp_imm_x(count, 1);
    a.b_cond(C_LS, slow);
    a.sub_imm(count, count, 1);
    a.stur(count, ptr, layout.rc_strong_off as i32);
    a.bind(done);
}

/// Release the packed owner stored at `[base + off]` (a local slot or an operand-stack word).
/// Shared owners are decremented inline; a last owner, a BigInt or any other word whose release
/// needs its destructor goes to `H_DROP_PACKED_AT`, which runs only Rust destruction (no author
/// code, allocation of JS values, or unwinding). The word is left moved-from: the caller must
/// overwrite or pop it. Scratch x9..x11; the helper path clobbers x0..x17 (the frame registers
/// x19..x22 are preserved).
pub(super) fn emit_exec_release_word(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    base: u32,
    off: i32,
) {
    let release = a.new_label();
    let done = a.new_label();
    emit_exec_word_load(a, 9, base, off);
    emit_exec_drop_shared(a, layout, 9, 10, 11, release);
    a.b(done);
    a.bind(release);
    a.mov(0, 19);
    a.movz(1, 0, 0);
    if off >= 0 {
        a.add_imm(2, base, off as u32);
    } else {
        a.sub_imm(2, base, off.unsigned_abs());
    }
    a.ldr_imm(16, 21, (super::H_DROP_PACKED_AT * 8) as u32);
    a.blr(16);
    a.bind(done);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executable(a: asm::Asm) -> ExecutableBuffer {
        let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
        ExecutableBuffer::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn bitfield_extract_encodes_every_range_and_register_pair() {
        let mut a = asm::Asm::new();
        for lsb in 0..64 {
            for width in 1..=64 - lsb {
                for rn in 0..32 {
                    for rd in 0..32 {
                        a.buf.clear();
                        a.ubfx(rd, rn, lsb, width);
                        assert_eq!(a.buf.len(), 1);
                        let word = a.buf[0];
                        // Independently check every architected field of 64-bit UBFM.
                        assert_eq!(word >> 31, 1); // sf
                        assert_eq!((word >> 29) & 3, 2); // opc: unsigned, no flag update
                        assert_eq!((word >> 23) & 0x3f, 0b100110);
                        assert_eq!((word >> 22) & 1, 1); // N
                        assert_eq!((word >> 16) & 63, lsb); // immr
                        assert_eq!((word >> 10) & 63, lsb + width - 1); // imms
                        assert_eq!((word >> 5) & 31, rn);
                        assert_eq!(word & 31, rd);
                    }
                }
            }
        }
    }

    #[test]
    fn bitfield_extract_rejects_invalid_fields_before_emitting() {
        for (rd, rn, lsb, width) in [
            (32, 0, 0, 1),
            (0, 32, 0, 1),
            (0, 0, 64, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 65),
            (0, 0, 63, 2),
            (0, 0, u32::MAX, 1),
            (0, 0, 1, u32::MAX),
        ] {
            let mut a = asm::Asm::new();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || a.ubfx(rd, rn, lsb, width)
            ))
            .is_err());
            assert!(a.buf.is_empty());
        }
    }

    // x0=input, x1/x2=flag-generating operands, x3=output. x4 is an optional
    // separate destination, x5 records NZCV via CSET (which does not alter flags).
    // Keep all stubs in one executable allocation; register31 is tested as XZR.
    fn append_extract_probe(
        a: &mut asm::Asm,
        rn: u32,
        rd: u32,
        lsb: u32,
        width: u32,
        payload: bool,
    ) -> usize {
        let offset = a.buf.len() * 4;
        a.cmp_reg_x(1, 2);
        for (index, condition) in [C_EQ, C_HS, C_MI, C_VS].into_iter().enumerate() {
            a.cset_w(5, condition);
            a.str_imm(5, 3, (3 + index as u32) * 8);
        }
        let start = a.buf.len();
        if payload {
            assert_eq!((lsb, width), (0, 48));
            emit_exec_payload(a, rn, rd);
        } else {
            a.ubfx(rd, rn, lsb, width);
        }
        assert_eq!(a.buf.len() - start, 1, "extraction must be one instruction");
        a.str_imm(rd, 3, 0);
        a.str_imm(0, 3, 8);
        a.str_imm(1, 3, 16);
        for (index, condition) in [C_EQ, C_HS, C_MI, C_VS].into_iter().enumerate() {
            a.cset_w(5, condition);
            a.str_imm(5, 3, (7 + index as u32) * 8);
        }
        a.ret();
        offset
    }

    const FLAG_INPUTS: [(u64, u64); 6] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1 << 63, 1),
        ((1 << 63) - 1, u64::MAX),
        (1 << 63, 0),
    ];

    fn check_extract_probe(
        code: &ExecutableBuffer,
        offset: usize,
        raw: u64,
        rn: u32,
        rd: u32,
        lsb: u32,
        width: u32,
        flag_operands: (u64, u64),
    ) {
        let run: unsafe extern "C" fn(u64, u64, u64, *mut u64) =
            unsafe { std::mem::transmute(code.as_ptr().add(offset)) };
        let (left, right) = flag_operands;
        let mut output = [u64::MAX; 11];
        unsafe { run(raw, left, right, output.as_mut_ptr()) };
        let source = if rn == 31 { 0 } else { raw };
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let expected = if rd == 31 { 0 } else { (source >> lsb) & mask };
        assert_eq!(
            output[0], expected,
            "extract {raw:016x}, {lsb}:{width}, x{rn}→x{rd}"
        );
        assert_eq!(
            output[1],
            if rd == 0 { expected } else { raw },
            "source preservation"
        );
        assert_eq!(output[2], left, "unrelated register preservation");
        let difference = left.wrapping_sub(right);
        let flags = [
            u64::from(difference == 0),
            u64::from(left >= right),
            difference >> 63,
            ((left ^ right) & (left ^ difference)) >> 63,
        ];
        assert_eq!(output[3..7], flags, "probe's incoming CMP flags");
        assert_eq!(
            output[7..11],
            flags,
            "extract must preserve every NZCV flag"
        );
    }

    #[test]
    fn native_bitfield_extract_all_ranges_preserve_source_flags_and_aliasing() {
        let mut a = asm::Asm::new();
        let mut cases = Vec::new();
        for lsb in 0..64 {
            for width in 1..=64 - lsb {
                for (rn, rd) in [(0, 4), (0, 0), (31, 4), (0, 31)] {
                    let offset = append_extract_probe(&mut a, rn, rd, lsb, width, false);
                    cases.push((offset, rn, rd, lsb, width));
                }
            }
        }
        let code = executable(a);
        let mut random = 0x1319_8a2e_0370_7344u64;
        for (offset, rn, rd, lsb, width) in cases {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            for raw in [
                0,
                u64::MAX,
                1 << lsb,
                1 << (lsb + width - 1),
                0x5555_5555_5555_5555,
                random,
            ] {
                for flags in FLAG_INPUTS {
                    check_extract_probe(&code, offset, raw, rn, rd, lsb, width, flags);
                }
            }
        }
    }

    #[test]
    fn native_payload_extract_exhausts_tags_and_preserves_flags() {
        let mut a = asm::Asm::new();
        let separate = append_extract_probe(&mut a, 0, 4, 0, 48, true);
        let in_place = append_extract_probe(&mut a, 0, 0, 0, 48, true);
        let code = executable(a);
        let mut random = 0xa409_3822_299f_31d0u64;
        for high in 0..=u16::MAX as u64 {
            for (index, payload) in [0, 1, 0x0000_ffff_ffff_ffff, random]
                .into_iter()
                .enumerate()
            {
                let raw = (high << 48) | (payload & 0x0000_ffff_ffff_ffff);
                let flags = FLAG_INPUTS[(high as usize + index) % FLAG_INPUTS.len()];
                check_extract_probe(&code, separate, raw, 0, 4, 0, 48, flags);
                check_extract_probe(&code, in_place, raw, 0, 0, 0, 48, flags);
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
        }
    }

    fn owner_classifier() -> (asm::Asm, [(usize, u64); 3]) {
        let mut a = asm::Asm::new();
        let scalar = a.new_label();
        let heap = a.new_label();
        let slow = a.new_label();
        classify_owner(&mut a, 0, 3, 4, heap, scalar, slow);
        assert_eq!(a.buf.len(), 13, "classifier code-size regression");
        let mut destinations = [(0, 0); 3];
        for (kind, label) in [scalar, heap, slow].into_iter().enumerate() {
            destinations[kind] = (a.buf.len(), kind as u64);
            a.bind(label);
            a.str_imm(0, 1, 0); // raw must survive classification unchanged.
            a.str_imm(2, 1, 8); // unrelated caller-saved register must survive too.
            a.movz(0, kind as u32, 0);
            a.ret();
        }
        (a, destinations)
    }

    #[test]
    fn native_owner_classifier_exhausts_high_lanes_and_random_payloads() {
        let (a, _) = owner_classifier();
        let code = executable(a);
        let run: unsafe extern "C" fn(u64, *mut u64, u64) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        let mut random = 0x243f_6a88_85a3_08d3u64;
        for high in 0..=u16::MAX as u64 {
            // Independent exact-tag oracle, including noncanonical Numbers and every
            // invalid/property-only high lane. No payload is dereferenced by this stub.
            let tag = high << 48;
            let expected = if [PACK_OBJ, PACK_STR, PACK_SYM].contains(&tag) {
                1
            } else if tag == PACK_BIGINT || tag > PACK_OBJ {
                2
            } else {
                0
            };
            for payload in [0, 1, 0x0000_ffff_ffff_ffff, random] {
                let raw = tag | (payload & 0x0000_ffff_ffff_ffff);
                let sentinel = raw.rotate_left(17) ^ 0xdead_beef_badc_0ffe;
                let mut output = [!raw, !sentinel];
                assert_eq!(
                    unsafe { run(raw, output.as_mut_ptr(), sentinel) },
                    expected,
                    "classification for {raw:016x}"
                );
                assert_eq!(
                    output,
                    [raw, sentinel],
                    "register preservation for {raw:016x}"
                );
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
        }
    }

    #[test]
    fn native_owner_classifier_instruction_paths_preserve_object_cost() {
        let (a, destinations) = owner_classifier();
        let words = a.finish();
        // Decode the actual emitted instructions, not a second copy of the classifier.
        // Only x3/x4 writes and comparisons are accepted before a terminal destination.
        let count_path = |raw: u64| {
            let (mut pc, mut steps, mut high, mut constant) = (0, 0, 0, 0);
            let (mut equal, mut lower) = (false, false);
            loop {
                if let Some((_, kind)) = destinations.iter().find(|(target, _)| *target == pc) {
                    break (*kind, steps);
                }
                assert!(steps < 20, "classifier must terminate");
                steps += 1;
                let word = words[pc];
                if word == 0xd370_fc03 {
                    // LSR x3, x0, #48
                    high = raw >> 48;
                } else if word & 0xffe0_001f == 0xd280_0004 {
                    // MOVZ x4, imm16
                    constant = ((word >> 5) & 0xffff) as u64;
                } else if word == 0xeb04_007f {
                    // CMP x3, x4
                    equal = high == constant;
                    lower = high < constant;
                } else if word & 0xff00_0010 == 0x5400_0000 {
                    // B.cond
                    let take = match word & 15 {
                        C_EQ => equal,
                        C_LO => lower,
                        C_LS => lower || equal,
                        C_HI => !lower && !equal,
                        condition => panic!("unexpected branch condition {condition}"),
                    };
                    if take {
                        let offset = ((((word >> 5) & 0x7ffff) as i32) << 13) >> 13;
                        pc = (pc as i32 + offset) as usize;
                        continue;
                    }
                } else if word & 0xfc00_0000 == 0x1400_0000 {
                    // B
                    let offset = (((word & 0x03ff_ffff) as i32) << 6) >> 6;
                    pc = (pc as i32 + offset) as usize;
                    continue;
                } else {
                    panic!("unexpected classifier instruction {word:08x}");
                }
                pc += 1;
            }
        };
        for (raw, kind, count) in [
            (PACK_OBJ, 1, 4),
            (crate::value::PACK_LAZY_PROTO, 2, 5),
            (0, 0, 8),
            (PACK_CANON_NAN, 0, 8),
            (PACK_UNDEFINED, 0, 8),
            (PACK_BOOL | 1, 0, 8),
            (PACK_BIGINT, 2, 9),
            (PACK_STR, 1, 12),
            (PACK_SYM, 1, 12),
            ((-0.0f64).to_bits(), 0, 13),
            (f64::NEG_INFINITY.to_bits(), 0, 13),
        ] {
            assert_eq!(count_path(raw), (kind, count), "path {raw:016x}");
        }
    }

    fn owner_operation(layout: &crate::value::JitLayout, clone: bool) -> ExecutableBuffer {
        assert!(layout.valid);
        let mut a = asm::Asm::new();
        let slow = a.new_label();
        if clone {
            emit_exec_clone(&mut a, layout, 0, 2, 3, slow);
        } else {
            emit_exec_drop_shared(&mut a, layout, 0, 2, 3, slow);
        }
        a.str_imm(0, 1, 0);
        a.movz(0, 1, 0);
        a.ret();
        a.bind(slow);
        a.movz(0, 0, 0);
        a.ret();
        executable(a)
    }

    fn native_clone(code: &ExecutableBuffer, source: &PackedValue) -> Option<PackedValue> {
        let run: unsafe extern "C" fn(u64, *mut PackedValue) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        let mut output = std::mem::MaybeUninit::<PackedValue>::uninit();
        // Success acquired exactly one owner and initialized its repr(transparent) word.
        (unsafe { run(source.bits(), output.as_mut_ptr()) } == 1)
            .then(|| unsafe { output.assume_init() })
    }

    fn native_drop(code: &ExecutableBuffer, value: PackedValue) -> Result<(), PackedValue> {
        let run: unsafe extern "C" fn(u64, *mut u64) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        let raw = value.bits();
        let mut output = !raw;
        if unsafe { run(raw, &mut output) } == 1 {
            // The native operation released this owner; its physical slot is now retired.
            std::mem::forget(value);
            assert_eq!(output, raw);
            Ok(())
        } else {
            assert_eq!(output, !raw, "slow path wrote its destination");
            Err(value)
        }
    }

    fn reference_count(value: &Value) -> usize {
        match value {
            Value::Obj(object) => Rc::strong_count(object),
            Value::Str(string) => string.strong_count(),
            Value::Sym(symbol) => Rc::strong_count(symbol),
            _ => panic!("expected a shared reference"),
        }
    }

    #[test]
    fn native_owner_clone_drop_all_kinds_and_last_owner_fallback() {
        let mut interp = crate::interpreter::Interp::new();
        let layout = crate::value::jit_layout(&interp.object_proto);
        let clone = owner_operation(&layout, true);
        let release = owner_operation(&layout, false);
        for value in [
            Value::Undefined,
            Value::Empty,
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::Num(0.0),
            Value::Num(-0.0),
            Value::Num(42.5),
            Value::Num(-42.5),
            Value::Num(f64::INFINITY),
            Value::Num(f64::NEG_INFINITY),
            Value::Num(f64::NAN),
        ] {
            let packed = PackedValue::pack(value);
            let copied = native_clone(&clone, &packed).expect("scalar clone");
            assert_eq!(copied.bits(), packed.bits());
            assert!(native_drop(&release, copied).is_ok());
            assert!(native_drop(&release, packed).is_ok());
        }
        for value in [
            Value::Obj(crate::value::Object::new(None)),
            Value::Str(crate::lstr::LStr::from("native owner λ")),
            interp.new_symbol(Some("native owner".into())),
        ] {
            let object_weak = value.as_obj().map(|object| Rc::downgrade(object));
            let symbol_weak = if let Value::Sym(symbol) = &value {
                Some(Rc::downgrade(symbol))
            } else {
                None
            };
            assert_eq!(reference_count(&value), 1);
            let packed = PackedValue::pack(value.clone());
            assert_eq!(reference_count(&value), 2);
            let copied = native_clone(&clone, &packed).expect("reference clone");
            assert_eq!(reference_count(&value), 3);
            assert_eq!(copied.bits(), packed.bits());
            assert!(native_drop(&release, copied).is_ok());
            assert_eq!(reference_count(&value), 2);
            drop(value);
            let packed = native_drop(&release, packed).expect_err("last owner needs destructor");
            let value = packed.into_value();
            assert_eq!(
                reference_count(&value),
                1,
                "last-owner guard changed the count"
            );
            drop(value); // checked destructor, not a decrement to an undestroyed zero owner.
            assert!(object_weak.is_none_or(|weak| weak.upgrade().is_none()));
            assert!(symbol_weak.is_none_or(|weak| weak.upgrade().is_none()));
        }
        let integer = crate::bigint::JsBigInt::parse_dec("123456789012345678901234567890").unwrap();
        let packed = PackedValue::pack(Value::BigInt(integer.clone()));
        assert!(
            native_clone(&clone, &packed).is_none(),
            "BigInt must use checked clone"
        );
        let copied = packed.clone();
        let packed = native_drop(&release, packed).expect_err("BigInt needs checked drop");
        drop(packed);
        drop(integer);
        assert!(
            native_clone(&clone, &copied).is_none(),
            "sole BigInt owner needs checked clone"
        );
        let copied =
            native_drop(&release, copied).expect_err("sole BigInt owner needs checked drop");
        assert!(matches!(copied.into_value(), Value::BigInt(value)
            if value.to_string() == "123456789012345678901234567890"));
        // Rejected words must not inspect even a null/invalid heap payload in either operation.
        for code in [&clone, &release] {
            let run: unsafe extern "C" fn(u64, *mut u64) -> u64 =
                unsafe { std::mem::transmute(code.as_ptr()) };
            for raw in [PACK_BIGINT, crate::value::PACK_LAZY_PROTO, u64::MAX] {
                let mut output = 123;
                assert_eq!(unsafe { run(raw, &mut output) }, 0);
                assert_eq!(output, 123);
            }
        }
    }

    #[test]
    fn native_owner_clone_roots_cycles_until_last_execution_owner_is_released() {
        let mut interp = crate::interpreter::Interp::new();
        interp.activate_gc_heap();
        let layout = crate::value::jit_layout(&interp.object_proto);
        let clone = owner_operation(&layout, true);
        let release = owner_operation(&layout, false);
        let object = crate::value::Object::new(None);
        object.borrow_mut().props.insert(
            "self",
            crate::value::Property::plain(Value::Obj(object.clone())),
        );
        let weak = Rc::downgrade(&object);
        let packed = PackedValue::pack(Value::Obj(object));
        let copied = native_clone(&clone, &packed).expect("object clone");
        assert!(native_drop(&release, packed).is_ok());
        interp.unstable_collect_young_for_host_tests();
        interp.collect_garbage_for_host();
        assert_eq!(
            weak.strong_count(),
            2,
            "execution owner plus cycle must remain live"
        );
        assert!(native_drop(&release, copied).is_ok());
        assert_eq!(
            weak.strong_count(),
            1,
            "only the unrooted self-cycle remains"
        );
        interp.collect_garbage_for_host();
        assert!(
            weak.upgrade().is_none(),
            "retired execution owner must not leak its cycle"
        );
    }

    #[test]
    fn native_execution_number_guard_rejects_tags_and_preserves_numeric_bits() {
        let mut a = asm::Asm::new();
        let fail = a.new_label();
        emit_exec_number_guard(&mut a, 0, 0, 2, fail);
        emit_exec_number_store(&mut a, 0, 1, 0, 3);
        a.movz(0, 1, 0);
        a.ret();
        a.bind(fail);
        a.movz(0, 0, 0);
        a.ret();
        let code = executable(a);
        let run: unsafe extern "C" fn(u64, *mut u64) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        for number in [
            0.0f64,
            -0.0,
            1.5,
            -1.5,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let raw = PackedValue::pack(Value::Num(number)).bits();
            let mut output = u64::MAX;
            assert_eq!(unsafe { run(raw, &mut output) }, 1);
            assert_eq!(output, raw);
        }
        for tag in [
            PACK_UNDEFINED,
            PACK_EMPTY,
            PACK_NULL,
            PACK_BOOL,
            PACK_BIGINT,
            PACK_STR,
            PACK_SYM,
            PACK_OBJ,
            crate::value::PACK_LAZY_PROTO,
        ] {
            let mut output = 123;
            assert_eq!(unsafe { run(tag, &mut output) }, 0);
            assert_eq!(output, 123, "failed guard changed canonical memory");
        }
    }

    #[test]
    fn native_execution_number_store_canonicalizes_hostile_nan_payloads() {
        let mut a = asm::Asm::new();
        a.fmov_d_x(0, 0);
        emit_exec_number_store(&mut a, 0, 1, 0, 2);
        a.ret();
        let code = executable(a);
        let run: unsafe extern "C" fn(u64, *mut u64) =
            unsafe { std::mem::transmute(code.as_ptr()) };
        for raw in [
            0x7ff9_0000_0000_1234,
            0xfff9_0000_0000_1234,
            0xffff_ffff_ffff_ffff,
            0x7ff0_0000_0000_0001,
            (-f64::NAN).to_bits(),
        ] {
            let mut output = 0;
            unsafe { run(raw, &mut output) };
            assert_eq!(output, PACK_CANON_NAN);
        }
    }

    #[test]
    fn native_execution_boundary_codec_round_trips_every_kind_without_ownership_changes() {
        let mut a = asm::Asm::new();
        let fail = a.new_label();
        // Packed raw x0, destination x1. Both transforms borrow the owners.
        emit_exec_decode_wide(&mut a, 0, 2, 3, 4, 5, fail);
        emit_exec_encode_wide(&mut a, 2, 3, 6, 7, 8, 0, fail);
        a.str_imm(6, 1, 0);
        a.movz(0, 1, 0);
        a.ret();
        a.bind(fail);
        a.movz(0, 0, 0);
        a.ret();
        let code = executable(a);
        let run: unsafe extern "C" fn(u64, *mut u64) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        // Payloads need not point at allocations: encoding/decoding must not dereference
        // them or change a reference count. Include nonzero low payload bytes.
        let mut words = vec![
            0.0f64.to_bits(),
            (-0.0f64).to_bits(),
            1.5f64.to_bits(),
            f64::MAX.to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            PACK_CANON_NAN,
            PACK_UNDEFINED,
            PACK_EMPTY,
            PACK_NULL,
            PACK_BOOL,
            PACK_BOOL | 1,
        ];
        for tag in [PACK_BIGINT, PACK_STR, PACK_SYM, PACK_OBJ] {
            words.push(tag | 0x1234_5678_90ab);
        }
        for raw in words {
            let mut output = u64::MAX;
            assert_eq!(unsafe { run(raw, &mut output) }, 1, "raw {raw:x}");
            assert_eq!(output, raw, "raw {raw:x}");
        }
        let mut output = 123;
        assert_eq!(
            unsafe { run(crate::value::PACK_LAZY_PROTO, &mut output) },
            0
        );
        assert_eq!(output, 123);
    }
}
