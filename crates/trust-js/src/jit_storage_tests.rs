//! P1 S5 step 5: the storage accessors read the engine's layouts from machine code. Small
//! ARM64 sequences built from the emitters run against live objects and scopes (in-cell
//! `InlineVec`s, in place and spilled, and 2-byte memos) and against synthetic sequences of
//! either kind (an `InlineVec` and a `Vec`).

use super::*;
use crate::value::{JitLayout, JitVecLayout, Object, Property, Value};

/// A function of two word arguments (x0, x1) returning x0, whose body `body` emits. The buffer
/// keeps the code alive.
fn assemble(
    body: impl FnOnce(&mut asm::Asm),
) -> (ExecutableBuffer, extern "C" fn(usize, usize) -> usize) {
    let mut a = asm::Asm::new();
    body(&mut a);
    a.ret();
    let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
    let code = ExecutableBuffer::from_bytes(&bytes).expect("executable memory");
    // SAFETY: the body only reads and writes through its arguments and returns in x0.
    let entry = unsafe {
        std::mem::transmute::<*const u8, extern "C" fn(usize, usize) -> usize>(code.as_ptr())
    };
    (code, entry)
}

/// Return x`reg`.
fn ret_reg(a: &mut asm::Asm, reg: u32) {
    a.mov(0, reg);
}

/// `none`: return all ones.
fn ret_none(a: &mut asm::Asm, none: usize) {
    let done = a.new_label();
    a.b(done);
    a.bind(none);
    a.mov_imm64(0, u64::MAX);
    a.bind(done);
}

fn layout(engine: &crate::Engine) -> JitLayout {
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(layout.valid);
    layout
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
        assert!(vec_len_fits(seq, 0) && vec_capacity_fits(seq, 0) && vec_data_fits(seq, 0));
        let (_len_code, len) = assemble(|a| {
            vec_len(a, seq, 1, 0, 0);
            ret_reg(a, 1);
        });
        let (_cap_code, capacity) = assemble(|a| {
            vec_capacity(a, seq, 1, 0, 0);
            ret_reg(a, 1);
        });
        let (_data_code, data) = assemble(|a| {
            vec_data(a, seq, 1, 0, 0);
            ret_reg(a, 1);
        });
        let (_set_code, set_len) = assemble(|a| vec_set_len(a, seq, 1, 0, 0));
        let buffer = [0u64; 9];
        match seq {
            JitVecLayout::Inline { .. } => {
                // In place while the capacity word is 0, then spilled to `buffer`.
                let mut words = InPlace {
                    data: [0; 4],
                    len: 3,
                    cap: 0,
                };
                let base = &mut words as *mut InPlace as usize;
                assert_eq!((len(base, 0), capacity(base, 0)), (3, 4));
                assert_eq!(data(base, 0), base);
                words.data[0] = buffer.as_ptr() as u64;
                words.cap = 9;
                words.len = 6;
                let base = &mut words as *mut InPlace as usize;
                assert_eq!((len(base, 0), capacity(base, 0)), (6, 9));
                assert_eq!(data(base, 0), buffer.as_ptr() as usize);
                // The length word is 4 bytes wide: the capacity beside it is untouched.
                set_len(base, 0x1_0000_0005);
                assert_eq!((words.len, words.cap), (5, 9));
            }
            JitVecLayout::Vec { .. } => {
                let mut words = [buffer.as_ptr() as u64, 2, 9];
                let base = words.as_mut_ptr() as usize;
                assert_eq!((len(base, 0), capacity(base, 0)), (2, 9));
                assert_eq!(data(base, 0), buffer.as_ptr() as usize);
                set_len(base, 0x1_0000_0005);
                assert_eq!(words, [buffer.as_ptr() as u64, 0x1_0000_0005, 9]);
            }
        }
    }
}

#[test]
fn named_slot_emitters_read_live_objects() {
    let engine = crate::Engine::new();
    let layout = layout(&engine);
    assert!(named_slots_fit(&layout));
    // The word of the slot whose location word (W2d step 3) is x1, of the Object at x0, as the
    // property templates read it: in the cell or the record.
    let (_value_code, value) = assemble(|a| {
        slot_addr_loc(a, &layout, 3, 0, 1);
        a.ldur(0, 3, 0);
        a.ret();
    });
    // The live count, which the templates bound a cached slot's index by.
    let (_len_code, len) = assemble(|a| {
        named_len(a, &layout, 0, 0);
        a.ret();
    });
    // The room check of a creation (x1 = the new slot's location word), and the owner kind of a
    // name's property.
    let (_room_code, room) = assemble(|a| {
        let miss = a.new_label();
        guard_loc_room(a, &layout, 1, 0, 2, 3, miss);
        a.movz(0, 1, 0);
        a.ret();
        a.bind(miss);
        a.movz(0, 0, 0);
        a.ret();
    });
    let (_kind_code, kind) = assemble(|a| {
        emit_property_kind(a, &layout, 0, 1, 2, 3);
        ret_reg(a, 7);
    });
    let (_set_code, set_len) = assemble(|a| named_set_len(a, &layout, 1, 0));
    // In the cell up to seven slots; the rest in the record.
    for count in [1usize, 2, 6, 7, 8, 9, 13, 14, 40] {
        let object = Object::new(None);
        for slot in 0..count {
            object.borrow_mut().props.insert(
                format!("f{slot}").as_str(),
                Property::plain(Value::Num(slot as f64 + 0.5)),
            );
        }
        // The Object, as a template finds it from the stored handle word.
        let stored = unsafe { *(&object as *const crate::value::Gc as *const usize) };
        let base = stored + layout.obj_from_cell;
        assert_eq!(base, &*object.borrow() as *const Object as usize);
        assert_eq!(len(base, 0), count);
        for slot in 0..count {
            // A packed Number is its f64 bits.
            let loc = object
                .borrow()
                .props
                .slot_location(slot)
                .expect("a location");
            let read = value(base, usize::from(loc));
            let expected = object
                .borrow()
                .props
                .property_at(slot)
                .expect("a property")
                .number_value();
            assert_eq!(
                Some(f64::from_bits(read as u64)),
                expected,
                "{count} fields, slot {slot}"
            );
            // Plain data: kind 1.
            assert_eq!(kind(base, slot), 1, "{count} fields, slot {slot}");
        }
        let capacity = object.borrow().props.named_capacity();
        let next = object
            .borrow()
            .props
            .slot_location(count)
            .expect("a location");
        assert_eq!(
            room(base, usize::from(next)),
            usize::from(count < capacity),
            "{count} fields: room for one more"
        );
        // A creation commit publishes the count last; drop the count and restore it.
        set_len(base, count - 1);
        assert_eq!(object.borrow().props.named_len(), count - 1);
        set_len(base, count);
        assert_eq!(object.borrow().props.named_len(), count);
    }
    // A slot that is not a writable data property: kind 2, from the layout's attribute byte.
    let object = Object::new(None);
    object
        .borrow_mut()
        .props
        .insert("data", Property::plain(Value::Num(1.0)));
    object
        .borrow_mut()
        .props
        .insert("fixed", Property::data(Value::Num(2.0), false, true, true));
    object
        .borrow_mut()
        .props
        .insert("get", Property::accessor_prop(None, None, true, true));
    let stored = unsafe { *(&object as *const crate::value::Gc as *const usize) };
    let base = stored + layout.obj_from_cell;
    assert_eq!([kind(base, 0), kind(base, 1), kind(base, 2)], [1, 2, 2]);
}

#[test]
fn binding_entry_emitters_read_live_scopes() {
    use crate::interpreter::{new_scope, Binding};
    let engine = crate::Engine::new();
    let layout = layout(&engine);
    assert!(scope_entries_fit(&layout));
    let (_code, binding) = assemble(|a| {
        let miss = a.new_label();
        scope_entries_len(a, &layout, 2, 0);
        a.cmp_reg_x(1, 2);
        a.b_cond(C_HS, miss);
        scope_entries_data(a, &layout, 3, 0);
        a.mov_imm64(4, layout.scope_binding_stride as u64);
        a.madd(3, 1, 4, 3);
        a.add_imm(0, 3, layout.scope_binding_offset as u32);
        ret_none(a, miss);
    });
    // One binding stays in the environment cell; more spill. Indexed maps share the
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
        let scope = env.borrow();
        for slot in 0..=count {
            let expected = scope
                .vars
                .fixed_binding_at(slot)
                .map_or(usize::MAX, |binding| binding.as_ptr() as usize);
            assert_eq!(
                binding(base, slot),
                expected,
                "{count} bindings, slot {slot}"
            );
        }
    }
}
