//! P1 S6 step 7b: the x64 storage accessors read the GC heap's layouts from machine code, as
//! `jit_storage_tests.rs` checks ARM64's. Small x86-64 functions built from the emitters run
//! against synthetic sequences of either kind, against live objects whose named fields are in
//! place and spilled (the fifth field spills them) and against live scopes, whose entries spill
//! past the first binding. A spilled sequence's capacity word is nonzero, so a length loaded as a
//! 64-bit word reads past the count, and these tests fail.

use super::*;
use crate::jit::ExecutableBuffer;
use crate::value::{JitLayout, JitVecLayout, Object, Property, Value};

const SI: u8 = 6;
const DI: u8 = 7;

/// A function of two word arguments (rdi, rsi) returning rax, whose body `body` emits. The buffer
/// keeps the code alive. It follows System V on every x86-64 target (Win64 would pass its
/// arguments in rcx and rdx).
fn assemble(
    body: impl FnOnce(&mut Asm),
) -> (ExecutableBuffer, extern "sysv64" fn(usize, usize) -> usize) {
    let mut a = Asm::new();
    body(&mut a);
    a.bytes(&[0xc3]); // ret
    let code = ExecutableBuffer::from_bytes(&a.finish()).expect("executable memory");
    // SAFETY: the body only reads through its arguments and returns in rax.
    let entry = unsafe {
        std::mem::transmute::<*const u8, extern "sysv64" fn(usize, usize) -> usize>(code.as_ptr())
    };
    (code, entry)
}

/// `none`: return all ones.
fn ret_none(a: &mut Asm, none: usize) {
    let done = a.label();
    a.jmp(done);
    a.bind(none);
    a.nr_imm(AX, u64::MAX);
    a.bind(done);
}

fn layout(engine: &crate::Engine) -> JitLayout {
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(layout.valid);
    layout
}

/// The capacity word of the `InlineVec` described by `seq`, `off` bytes into `base`: nonzero
/// exactly while its elements are spilled.
fn capacity_word(seq: JitVecLayout, base: usize, off: usize) -> u32 {
    let JitVecLayout::Inline { cap, .. } = seq else {
        panic!("gc-heap's sequences are InlineVecs")
    };
    // SAFETY: `base + off` is a live sequence of that layout.
    unsafe { ((base + off + cap) as *const u32).read() }
}

/// A synthetic sequence: four in-place words, then the u32 length and capacity words.
#[repr(C)]
struct InPlace {
    data: [u64; 4],
    len: u32,
    cap: u32,
}

#[test]
fn sequence_emitters_follow_both_layouts() {
    let inline = JitVecLayout::Inline {
        len: std::mem::offset_of!(InPlace, len),
        cap: std::mem::offset_of!(InPlace, cap),
        inline: std::mem::offset_of!(InPlace, data),
        heap: std::mem::offset_of!(InPlace, data),
        inline_capacity: 4,
    };
    let vec = JitVecLayout::Vec {
        ptr: 0,
        len: 8,
        cap: 16,
    };
    for seq in [inline, vec] {
        assert!(vec_len_fits(seq, 0) && vec_data_fits(seq, 0));
        let (_len_code, len) = assemble(|a| vec_len(a, seq, AX, DI, 0));
        let (_data_code, data) = assemble(|a| vec_data(a, seq, AX, DI, 0));
        // The data address may replace its base.
        let (_onto_code, data_onto_base) = assemble(|a| {
            vec_data(a, seq, DI, DI, 0);
            a.nr_mov(AX, DI);
        });
        let buffer = [0u64; 9];
        let spilled = buffer.as_ptr() as usize;
        match seq {
            JitVecLayout::Inline { .. } => {
                // In place while the capacity word is 0, then spilled to `buffer`.
                let mut words = InPlace {
                    data: [0; 4],
                    len: 3,
                    cap: 0,
                };
                let base = &mut words as *mut InPlace as usize;
                assert_eq!(len(base, 0), 3);
                assert_eq!((data(base, 0), data_onto_base(base, 0)), (base, base));
                words.data[0] = spilled as u64;
                words.cap = 9;
                words.len = 6;
                let base = &mut words as *mut InPlace as usize;
                // The length word is 4 bytes wide: the capacity beside it is not read with it.
                assert_eq!(len(base, 0), 6);
                assert_eq!((data(base, 0), data_onto_base(base, 0)), (spilled, spilled));
            }
            JitVecLayout::Vec { .. } => {
                let mut words = [spilled as u64, 2, 9];
                let base = words.as_mut_ptr() as usize;
                assert_eq!(len(base, 0), 2);
                assert_eq!((data(base, 0), data_onto_base(base, 0)), (spilled, spilled));
            }
        }
    }
}

#[test]
fn named_field_emitters_read_live_objects() {
    let engine = crate::Engine::new();
    let layout = layout(&engine);
    assert!(fields_len_fits(&layout) && fields_data_fits(&layout));
    // Field `rsi` of the Object at rdi, as the property templates read it: the bounds check
    // against the live count, then the packed word at `data + slot * entry_size`.
    let (_code, value) = assemble(|a| {
        let miss = a.label();
        fields_len(a, &layout, CX, DI);
        a.nr_cmp(SI, CX);
        a.jcc(AE, miss);
        fields_data(a, &layout, AX, DI);
        a.nr_mul_imm(SI, layout.entry_size);
        a.nr_reg(0x01, AX, SI); // add rax,rsi
        a.nr_load(AX, AX, layout.entry_value);
        ret_none(a, miss);
    });
    let JitVecLayout::Inline {
        inline_capacity, ..
    } = layout.repr.props_fields
    else {
        panic!("gc-heap's named fields are an InlineVec")
    };
    // In place up to four fields; spilled beyond.
    for count in [1usize, 2, 4, 5, 9] {
        let object = Object::new(None);
        for slot in 0..count {
            object.borrow_mut().props.insert(
                format!("f{slot}").as_str(),
                Property::plain(Value::Num(slot as f64 + 0.5)),
            );
        }
        // The Object, as a template finds it from the stored handle word.
        let stored = unsafe { *(&object as *const crate::value::Gc as *const usize) };
        let base = stored + layout.obj_from_rc;
        assert_eq!(base, &*object.borrow() as *const Object as usize);
        let fields = layout.obj_props + layout.props_entries;
        assert_eq!(
            capacity_word(layout.repr.props_fields, base, fields) != 0,
            count > inline_capacity,
            "{count} fields are spilled exactly past the inline capacity"
        );
        for slot in 0..=count {
            // A packed Number is its f64 bits; past the count the sequence answers all ones.
            let read = value(base, slot);
            match object.borrow().props.property_at(slot) {
                Some(property) => assert_eq!(
                    Some(f64::from_bits(read as u64)),
                    property.number_value(),
                    "{count} fields, slot {slot}"
                ),
                None => assert_eq!(read, usize::MAX, "{count} fields, slot {slot}"),
            }
        }
    }
}

#[test]
fn binding_entry_emitters_read_live_scopes() {
    use crate::interpreter::{new_scope, Binding};
    let engine = crate::Engine::new();
    let layout = layout(&engine);
    assert!(scope_entries_fit(&layout));
    let entries = layout
        .repr
        .scope_small
        .expect("probed binding storage")
        .storage;
    let JitVecLayout::Inline {
        inline_capacity, ..
    } = entries
    else {
        panic!("gc-heap's binding entries are an InlineVec")
    };
    // Entry `rsi`'s `Binding` in the scope at rdi, as the name templates find a fresh holder's.
    let (_code, binding) = assemble(|a| {
        let miss = a.label();
        scope_entries_len(a, &layout, CX, DI);
        a.nr_cmp(SI, CX);
        a.jcc(AE, miss);
        scope_entries_data(a, &layout, AX, DI);
        a.nr_mul_imm(SI, layout.scope_binding_stride);
        a.nr_reg(0x01, AX, SI); // add rax,rsi
        a.nr_lea(AX, AX, layout.scope_binding_offset as i32);
        ret_none(a, miss);
    });
    // One binding stays in a gc-heap environment cell; more spill. Indexed maps share the
    // Small arm's entries.
    for (count, indexed) in [(1usize, false), (2, false), (5, false), (3, true)] {
        let env = new_scope(None);
        {
            let mut scope = env.borrow_mut();
            for slot in 0..count {
                scope.vars.insert(
                    format!("b{slot}"),
                    Binding::data(Value::Num(slot as f64), true, true),
                );
            }
            if indexed {
                scope.vars.share_layout(&std::cell::OnceCell::new());
            }
        }
        let base = crate::interpreter::Env::as_ptr(&env) as usize;
        assert_eq!(
            capacity_word(entries, base, layout.scope_small_vec) != 0,
            count > inline_capacity,
            "{count} bindings are spilled exactly past the inline capacity"
        );
        let scope = env.borrow();
        for slot in 0..=count {
            let expected = scope
                .vars
                .fixed_binding_at(slot)
                .map_or(usize::MAX, |binding| binding as *const Binding as usize);
            assert_eq!(
                binding(base, slot),
                expected,
                "{count} bindings, slot {slot}"
            );
        }
    }
}
