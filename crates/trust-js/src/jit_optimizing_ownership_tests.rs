//! Native owner operations must preserve value identity and destruction boundaries.
use super::tests::{check, check_with_gc_pressure};
use super::*;

#[test]
fn ordinary_returns_move_without_completion_helpers() {
    for (source, expected) in [
        ("function subject(){return 3*7+2;}", "23"),
        ("function subject(){return;}", "undefined"),
        ("function subject(){}", "undefined"),
        (
            "function subject(){var held={value:7};return held;}",
            "[object Object]",
        ),
    ] {
        bytecode::TEST_JIT_COMPLETE_HELPERS.with(|count| count.set(0));
        bytecode::TEST_JIT_RETURN_HELPERS.with(|count| count.set(0));
        check(source, expected);
        assert_eq!(
            bytecode::TEST_JIT_COMPLETE_HELPERS.with(|count| count.get()),
            0
        );
        assert_eq!(
            bytecode::TEST_JIT_RETURN_HELPERS.with(|count| count.get()),
            0
        );
    }
}

#[test]
fn direct_returns_preserve_occupied_results_aliases_and_caller_handlers() {
    use crate::bytecode::{Handler, HandlerTarget};
    use crate::value::{Object, Value as JsValue};
    use std::rc::Rc;

    // Exercise the native body ABI deliberately: a failed tail call can leave ret
    // occupied, and shared-context calls inherit handlers below handler_floor.
    // Neither condition is normally constructible by a standalone JS leaf call.
    for returns_value in [false, true] {
        for occupied in 0..3 {
            let source = if returns_value {
                "function subject(x){return x;}"
            } else {
                "function subject(x){return;}"
            };
            let parsed = crate::parser::parse_script(source, false).ok().unwrap();
            let crate::ast::Stmt::FuncDecl(function) = &parsed[0] else {
                unreachable!()
            };
            let chunk = bytecode::compile(function).unwrap();
            let mut engine = crate::Engine::new();
            let values = crate::value::jit_layout(&engine.interp.object_proto);
            let layout = crate::interpreter::interp_layout(&mut engine.interp);
            let code = compile_checked(&chunk, &values, &layout).unwrap();
            assert!(!code.needs_global && !chunk.prepared_entry);
            let returned = Object::new(None);
            let returned_weak = Rc::downgrade(&returned);
            let displaced = Object::new(None);
            let displaced_weak = Rc::downgrade(&displaced);
            let initial = match occupied {
                1 => JsValue::Obj(displaced.clone()),
                2 => JsValue::Obj(returned.clone()),
                _ => JsValue::Undefined,
            };
            drop(displaced);
            let mut slots = vec![PackedValue::pack(JsValue::Obj(returned))];
            slots.resize_with(chunk.jit_frame().1, || {
                PackedValue::pack(JsValue::Undefined)
            });
            let mut stack = Vec::<PackedValue>::with_capacity(code.max_stack.max(1));
            let base = stack.as_mut_ptr();
            let env = engine.interp.global_env.clone();
            engine.interp.strict = true;
            let mut ctx = JitCtx {
                helpers: engine.interp.jit_helpers.as_ptr(),
                stack_base: base,
                final_sp: base,
                slots: slots.as_mut_ptr(),
                inline_ic_safe: &engine.interp.inline_ic_safe as *const _ as *const u8,
                env_raw: Rc::as_ptr(&env) as *const u8,
                this_raw: std::ptr::null(),
                global_body: std::ptr::null(),
                genv: Rc::as_ptr(&env) as usize,
                interp: &mut *engine.interp,
                chunk: Rc::as_ptr(&chunk),
                this_val: JsValue::Undefined,
                n_slots: slots.len(),
                handlers: vec![Handler {
                    target: HandlerTarget::Catch { throw_pc: 123 },
                    stack_depth: 0,
                }],
                handler_floor: 1,
                code_base: code.mem,
                pc_offsets: code.pc_offsets.as_ptr(),
                error: None,
                ret: PackedValue::pack(initial),
                env_parent_raw: std::ptr::null(),
                opstat_enabled: false,
                callstat_enabled: false,
                inline_recompile_at: 0,
                live_objects: crate::value::live_objects_ptr(&engine.interp.gc_heap),
                activation: None,
                resume_activation: std::ptr::null_mut(),
                resume_pc: 0,
                resume_step: None,
                references_raw: std::ptr::null_mut(),
            };
            ctx.this_raw = &ctx.this_val;
            bytecode::TEST_JIT_COMPLETE_HELPERS.with(|count| count.set(0));
            bytecode::TEST_JIT_RETURN_HELPERS.with(|count| count.set(0));
            let entry: extern "C" fn(*mut JitCtx) -> u64 = unsafe { std::mem::transmute(code.mem) };
            assert_eq!(entry(&mut ctx), 1);
            assert_eq!(ctx.final_sp, base);
            assert_eq!(ctx.handlers.len(), 1);
            assert_eq!(ctx.handler_floor, 1);
            assert!(engine.interp.strict);
            assert_eq!(code.residency.active.get(), 0);
            assert_eq!(displaced_weak.strong_count(), 0);
            assert_eq!(returned_weak.strong_count(), 1 + usize::from(returns_value));
            assert_eq!(ctx.ret.is_undefined(), !returns_value);
            assert_eq!(
                bytecode::TEST_JIT_COMPLETE_HELPERS.with(|count| count.get())
                    + bytecode::TEST_JIT_RETURN_HELPERS.with(|count| count.get()),
                usize::from(occupied != 0),
            );
            drop(ctx);
            drop(slots);
            assert_eq!(returned_weak.strong_count(), 0);
        }
    }
}

#[test]
fn shared_heap_local_copies_moves_and_discards_stay_native() {
    for root in [
        "({value:7})",
        "'shared string owner'",
        "Symbol('shared symbol owner')",
        "1234567890123456789012345678901234567890n",
    ] {
        ownership::TEST_OWNERSHIP_HELPERS.with(|count| count.set([0; 4]));
        ownership::TEST_OWNERSHIP_SITES.with(|count| count.set([0; 4]));
        check(
            &format!(
                r#"
                function subject() {{
                    var a=root,b=a,copy,sum=0;
                    for(var k=0;k<64;k++) {{
                        b=(copy=a);a=b;a=a;copy;
                        if(a===b)sum++;
                    }}
                    return sum;
                }}
                var root={root};
                "#
            ),
            "64",
        );
        assert_eq!(
            ownership::TEST_OWNERSHIP_HELPERS.with(|count| count.get()),
            [0; 4],
            "no LoadLocal/Dup/Pop/StoreLocal helper for shared {root}"
        );
        assert!(
            ownership::TEST_OWNERSHIP_SITES
                .with(|count| count.get().iter().all(|&sites| sites > 0)),
            "the fixture must cover all four native ownership operations"
        );
    }
}

#[test]
fn last_owner_discard_and_overwrite_keep_checked_destruction() {
    ownership::TEST_OWNERSHIP_HELPERS.with(|count| count.set([0; 4]));
    check(
        r#"
        function subject() {
            var old,sum=0;
            for(var k=0;k<40;k++) {
                old={value:k};old=null;
                ({discard:k});
                sum++;
            }
            return sum;
        }
        "#,
        "40",
    );
    // Three independent explicitly compiled invocations. The last owner MUST reach the
    // ordinary destructor; the compiler must not just decrement its strong count to zero.
    let [loads, duplicates, pops, stores] =
        ownership::TEST_OWNERSHIP_HELPERS.with(|count| count.get());
    assert_eq!([loads, duplicates], [0; 2]);
    assert_eq!(pops, 120);
    assert_eq!(stores, 120);
}

#[test]
fn moved_heap_local_owners_survive_loop_collection_and_interrupt_polls() {
    check_with_gc_pressure(
        r#"
        function subject() {
            var first={value:7},held=first,copy,total=0;
            first=null;
            for(var k=0;k<17000;k++) {
                copy=held;held=copy;copy;
                total++;
            }
            return total+'|'+held.value;
        }
        "#,
        "17000|7",
        true,
    );
}

#[test]
fn packed_owner_layout_declines_unproven_header_offsets() {
    let engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(crate::value::jit_packed_owners_supported(&layout));
    let mut invalid = layout;
    invalid.valid = false;
    assert!(!crate::value::jit_packed_owners_supported(&invalid));
    let mut invalid = layout;
    invalid.rc_strong_off += 1;
    assert!(!crate::value::jit_packed_owners_supported(&invalid));
    let mut invalid = layout;
    invalid.gc_data_off += std::mem::size_of::<usize>();
    assert!(!crate::value::jit_packed_owners_supported(&invalid));
    assert!(crate::value::jit_packed_owners_supported(&layout));
}
