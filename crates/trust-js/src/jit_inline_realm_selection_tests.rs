//! Invocation selection for one snapshot AST shared by multiple Window Realms.
//! ECMA-262 e28783d5 [[Realm]], FunctionDeclarationInstantiation and mapped arguments.

use crate::bytecode::{self, Tier};
use crate::realm_inline_guard_tests::{eval_string, in_realm, set_global};
use crate::value::{Callable, Value};
use crate::Engine;
use std::rc::Rc;

fn subject(i: &mut crate::interpreter::Interp) -> crate::value::UserFn {
    let env = i.global_env.clone();
    let value = i.get_var("subject", &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("snapshot subject")
    };
    user.clone()
}

fn run(i: &mut crate::interpreter::Interp, snapshot: &'static [u8]) {
    let body = i
        .shared_snapshot_program(snapshot)
        .expect("snapshot decodes");
    assert!(i.run_classic_program(&body, None).is_ok());
}

#[test]
fn prepared_shared_body_uses_primary_in_other_realms_and_inline_in_its_own() {
    const SOURCE: &str = r#"(function(){
        globalThis.calls=0;
        globalThis.callee=function leaf(x){globalThis.calls++;return x+3;};
        globalThis.subject=function subject(value){
            var read=()=>value;
            arguments[0]=value+1;
            var result=callee(value);
            eval('value += 2');
            return result+':'+read()+':'+arguments[0];
        };
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] = SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).unwrap());
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let i = &mut *engine.interp;
        run(i, snapshot);
        let user = subject(i);
        let function = &user.func;
        let primary = bytecode::compile(function).expect("prepared primary");
        assert!(primary.prepared_entry);
        primary.inline_attempted.set(true);
        primary.inline_retry_at.set(0);
        assert!(function.runtime().code.set(Some(primary.clone())).is_ok());
        for _ in 0..8 {
            assert_eq!(eval_string(i, "subject(5)"), "9:8:8");
        }
        let genv = crate::interpreter::Env::as_ptr(&i.global_env) as usize;
        let plan = bytecode::plan_inlines(&primary, function, user.realm, &i.global_env);
        assert!(!plan.is_empty());
        let inlined = bytecode::compile_with_inlines(function, &plan, &primary)
            .expect("prepared inline body");
        assert!(inlined.prepared_entry);
        assert!(inlined.inline_realm_compatible(genv));
        assert!(function.runtime().code2.set(Some(inlined.clone())).is_ok());
        assert_eq!(eval_string(i, "calls=0;subject(5)+'|'+calls"), "9:8:8|1");
        let primary_code = primary
            .jit
            .get()
            .flatten()
            .expect("warm primary native code");
        let inline_code = inlined
            .jit
            .get()
            .flatten()
            .expect("warm inline native code");
        let main = Value::Obj(i.global.clone());
        let child = i.create_realm();
        in_realm(i, &child, |i| {
            run(i, snapshot);
            set_global(i, "mainRealm", main.clone());
            let child_user = subject(i);
            assert!(
                Rc::ptr_eq(function, &child_user.func),
                "shared AST and second-stage slot"
            );
            let child_genv = crate::interpreter::Env::as_ptr(&i.global_env) as usize;
            assert!(!inlined.inline_realm_compatible(child_genv));
            assert!(Rc::ptr_eq(
                function
                    .runtime()
                    .execution_code_for(child_genv)
                    .and_then(Option::as_ref)
                    .unwrap(),
                &primary
            ));
        });
        i.tier = tier;
        primary_code.residency.referenced.set(0);
        inline_code.residency.referenced.set(0);
        assert_eq!(
            in_realm(i, &child, |i| eval_string(
                i,
                "calls=0;subject(5)+'|'+calls"
            )),
            "9:8:8|1"
        );
        assert_eq!(
            inline_code.residency.referenced.get(),
            0,
            "no guaranteed-missing inline entry"
        );
        if tier == Tier::Jit {
            assert_eq!(primary_code.residency.referenced.get(), 1);
        }
        primary_code.residency.referenced.set(0);
        inline_code.residency.referenced.set(0);
        assert_eq!(
            in_realm(i, &child, |i| eval_string(
                i,
                "mainRealm.calls=0;mainRealm.subject(5)+'|'+mainRealm.calls"
            )),
            "9:8:8|1"
        );
        assert_eq!(
            primary_code.residency.referenced.get(),
            0,
            "callee's Realm governs cross-Realm invocation selection"
        );
        if tier == Tier::Jit {
            assert_eq!(inline_code.residency.referenced.get(), 1);
        }
        assert!(Rc::ptr_eq(
            function
                .runtime()
                .execution_code_for(genv)
                .and_then(Option::as_ref)
                .unwrap(),
            &inlined
        ));
        assert_eq!(primary_code.residency.active.get(), 0);
        assert_eq!(inline_code.residency.active.get(), 0);
        assert_eq!(i.depth, 0);
        assert!(i.fn_frames.is_empty());
        i.gc_collect();
    }
}

#[test]
fn second_stage_without_realm_dependent_inlines_remains_selected_in_other_realms() {
    const SOURCE: &str = "globalThis.subject=function subject(n){return n+1;};";
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] = SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).unwrap());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let i = &mut *engine.interp;
    run(i, snapshot);
    let user = subject(i);
    let primary = bytecode::compile(&user.func).unwrap();
    assert!(user.func.runtime().code.set(Some(primary.clone())).is_ok());
    let secondary = bytecode::compile(&user.func).unwrap();
    assert!(user
        .func
        .runtime()
        .code2
        .set(Some(secondary.clone()))
        .is_ok());
    assert_eq!(eval_string(i, "subject(5)"), "6");
    let code = secondary.jit.get().flatten().expect("native second stage");
    let child = i.create_realm();
    code.residency.referenced.set(0);
    in_realm(i, &child, |i| {
        run(i, snapshot);
        let genv = crate::interpreter::Env::as_ptr(&i.global_env) as usize;
        assert!(secondary.inline_realm_compatible(genv));
        assert!(Rc::ptr_eq(
            user.func
                .runtime()
                .execution_code_for(genv)
                .and_then(Option::as_ref)
                .unwrap(),
            &secondary
        ));
        assert_eq!(eval_string(i, "subject(5)"), "6");
    });
    assert_eq!(
        code.residency.referenced.get(),
        1,
        "unrestricted native body remains available"
    );
    assert_eq!(code.residency.active.get(), 0);
    i.gc_collect();
}

/// The inline metadata names the planning environment only by address and by weak pin.
/// The environment dies at the first collection after its last strong handle, its address
/// stays reserved while a compiled guard pins it, and dropping the last such chunk lets the
/// next collection free it.
#[test]
fn realm_inline_metadata_pins_the_planning_environment_allocation_without_roots() {
    use crate::gc_test::HiddenAddress;
    use crate::interpreter::Env;
    /// Whether the cell at `address` is still allocated (live, or dead and weakly pinned).
    #[inline(never)]
    fn reserved(i: &crate::interpreter::Interp, address: HiddenAddress) -> bool {
        i.gc_heap
            .core
            .borrow()
            .is_allocated_word(address.get() as u64)
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let i = &mut *engine.interp;
    eval_string(
        i,
        "function leaf(n){return n+3;}function subject(n){return leaf(n)*2;}",
    );
    let user = subject(i);
    let primary = bytecode::compile(&user.func).unwrap();
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(user.func.runtime().code.set(Some(primary.clone())).is_ok());
    for _ in 0..8 {
        assert_eq!(eval_string(i, "subject(5)"), "16");
    }
    // The planning environment's handle and address stay in this helper frame (the collector
    // scans the stack).
    let (chunk, weak, address) = crate::gc_test::in_frame(|| {
        let planning = crate::interpreter::new_scope(None);
        let address = Env::as_ptr(&planning) as usize;
        let plan = bytecode::plan_inlines(&primary, &user.func, user.realm, &planning);
        assert!(!plan.is_empty());
        let chunk = bytecode::compile_with_inlines(&user.func, &plan, &primary).unwrap();
        assert!(chunk.inline_realm_compatible(address));
        let target = chunk
            .jit_ops()
            .iter()
            .find_map(|op| match *op {
                bytecode::Op::InlineGuard(target, _) => Some(chunk.jit_inline_target(target)),
                _ => None,
            })
            .unwrap();
        assert_eq!(target.genv_pin.as_ptr() as usize, address);
        (
            chunk,
            Env::downgrade(&planning),
            HiddenAddress::from_fn(|| address),
        )
    });
    i.gc_collect();
    assert!(
        !crate::gc_test::alive(&weak),
        "metadata must not keep bindings/Realm values alive"
    );
    // Only the chunk's pin names the dead environment now.
    drop(weak);
    i.gc_collect();
    assert!(
        reserved(i, address),
        "a retained inline pin keeps the dead environment's address reserved"
    );
    let reused = crate::gc_test::in_frame(|| {
        (0..64)
            .map(|_| crate::interpreter::new_scope(None))
            .any(|env| Env::as_ptr(&env) as usize == address.get())
    });
    assert!(
        !reused,
        "a retained inline pin prevents another environment reusing the guard's address"
    );
    drop(chunk);
    i.gc_collect();
    assert!(
        !reserved(i, address),
        "last inline owner releases its allocation lease"
    );
}
