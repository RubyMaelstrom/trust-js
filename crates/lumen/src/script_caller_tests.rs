//! ECMA-262 ScriptEvaluation / PrepareForOrdinaryCall / BuiltinCallOrConstruct.
use crate::{Engine, Value};

#[test]
fn host_entry_survives_foreign_calls_and_eval_but_changes_for_callbacks() {
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("entry", 0, |ctx, _, _| Ok(ctx.script_entry_global()));
        engine.define_global("callback", 1, |ctx, _, args| {
            ctx.invoke_callback_entry(args[0].clone(), Value::Undefined, &[])
        });
        let root = engine.global_this();
        let child = engine.ctx().create_embed_realm();
        assert!(engine
            .ctx()
            .member_set(&root, "child", child.clone())
            .is_ok());
        assert!(engine.ctx().member_set(&child, "parent", root).is_ok());
        let result=engine.eval(r#"
            let step=0; function check(v) { step++; if (!v) throw Error('entry realm '+step); }
            child.eval('globalThis.f=()=>parent.entry(); globalThis.fail=()=>{throw 42}; parent.check(parent.entry()===parent)');
            check(child.f()===globalThis);
            check(child.f.bind(null)()===globalThis);
            check(callback(child.f)===child);
            check(callback(child.f.bind(null))===child);
            check(callback(new Proxy(child.f,{}))===child);
            try { callback(child.fail); } catch(e) { check(e===42); }
            check(entry()===globalThis);
            const revoked=Proxy.revocable(child.f,{}); revoked.revoke();
            try { callback(revoked.proxy); throw Error('accepted revoked callback'); }
            catch(e) { check(e instanceof TypeError); }
            check(entry()===globalThis);
            'entry-ok'
        "#,false).unwrap();
        match result {
            crate::Completion::Value(value) => assert_eq!(value, "entry-ok", "{tier:?}"),
            crate::Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn native_operation_observes_script_caller_across_realms_and_eval() {
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("caller", 0, |ctx, _, _| Ok(ctx.script_caller_global()));
        let root = engine.global_this();
        let child = engine.ctx().create_embed_realm();
        assert!(engine
            .ctx()
            .member_set(&root, "child", child.clone())
            .is_ok());
        assert!(engine
            .ctx()
            .member_set(&child, "parent", root.clone())
            .is_ok());
        assert!(engine
            .ctx()
            .with_embed_realm(&child, |ctx| {
                ctx.define_embed_global("caller", 0, |ctx, _, _| Ok(ctx.script_caller_global()));
            })
            .is_ok());
        let outcome = engine.eval(r#"
            function assert(x) { if (!x) throw Error('script caller'); }
            assert(child.caller() === globalThis);
            assert(child.caller.call(null) === globalThis);
            assert(Reflect.apply(child.caller, null, []) === globalThis);
            assert(child.caller.bind(null)() === globalThis);
            child.eval('parent.assert(parent.caller() === globalThis); globalThis.f = () => parent.caller()');
            assert(child.f() === child);
            (function () { child.eval('parent.assert(parent.caller() === globalThis)'); })();
            assert(child.caller() === globalThis);
            'ok'
        "#, false).unwrap();
        assert!(
            matches!(outcome, crate::Completion::Value(ref value) if value == "ok"),
            "{tier:?}"
        );
        let module = engine
            .eval(
                "if (child.caller() !== globalThis) throw Error('module caller');",
                true,
            )
            .unwrap();
        assert!(
            matches!(module, crate::Completion::Value(_)),
            "module {tier:?}"
        );
        assert!(matches!(engine.ctx().script_caller_global(), Value::Obj(_)));
    }
}

/// A loader recording each `(specifier, referrer)` it is asked for.
fn recording_loader(engine: &mut Engine) -> std::rc::Rc<std::cell::RefCell<Vec<String>>> {
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let log = seen.clone();
    engine.set_module_loader(move |specifier, referrer| {
        log.borrow_mut().push(format!("{specifier} <- {referrer}"));
        Some((
            format!("{referrer}#{specifier}"),
            "export default 1;".into(),
        ))
    });
    seen
}

#[test]
fn import_resolves_against_the_active_script_or_module() {
    // ECMA-262 EvaluateImportCall uses GetActiveScriptOrModule(): a function keeps the Script
    // it was created in (OrdinaryFunctionCreate), eval code and dynamic functions take the
    // active ScriptOrModule where they are created, and HTML HostMakeJobCallback saves the
    // active script for a built-in job callback, which HostCallJobCallback pushes again.
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.set_job_callback_script_caller_capture(true);
        let seen = recording_loader(&mut engine);
        engine.set_import_base("https://a.test/a.js");
        let script_a = engine.eval_value(
            "globalThis.fromA = () => import('./fn.js');
             globalThis.dynamicA = new Function(\"return import('./dynamic.js')\");
             (0, eval)(\"globalThis.evalA = () => import('./eval.js')\");
             Promise.resolve(\"import('./job.js')\").then(eval);
             Promise.resolve().then(() => import('./arrow-job.js'));
             Promise.resolve({then(resolve) { resolve(\"import('./thenable.js')\"); }}).then(eval);",
        );
        assert!(matches!(script_a, Ok(Ok(_))), "{tier:?}");
        // The host restores the document's base after the script, before the checkpoint.
        engine.set_import_base("https://doc.test/page.html");
        engine.run_microtasks();
        engine.set_import_base("https://b.test/b.js");
        let script_b = engine.eval_value(
            "fromA(); dynamicA(); evalA();
             (0, eval)(\"import('./indirect.js')\");
             new Function(\"return import('./function-b.js')\")();",
        );
        assert!(matches!(script_b, Ok(Ok(_))), "{tier:?}");
        engine.set_import_base("https://doc.test/page.html");
        engine.run_microtasks();
        // With no active script, the current Realm's base applies.
        engine
            .ctx()
            .define_embed_global("hostImport", 1, |ctx, _, args| {
                let import = ctx.member_get(&ctx.global_this(), "eval")?;
                ctx.invoke(import, Value::Undefined, &[args[0].clone()])
            });
        let host = engine.global_this();
        let host_import = engine
            .ctx()
            .member_get(&host, "hostImport")
            .unwrap_or_else(|_| panic!("hostImport"));
        let _ = engine.ctx().invoke(
            host_import,
            Value::Undefined,
            &[Value::lstr("import('./host.js')")],
        );
        engine.run_microtasks();
        assert_eq!(
            *seen.borrow(),
            [
                "./job.js <- https://a.test/a.js",
                "./arrow-job.js <- https://a.test/a.js",
                "./thenable.js <- https://a.test/a.js",
                "./fn.js <- https://a.test/a.js",
                "./dynamic.js <- https://a.test/a.js",
                "./eval.js <- https://a.test/a.js",
                "./indirect.js <- https://b.test/b.js",
                "./function-b.js <- https://b.test/b.js",
                "./host.js <- https://doc.test/page.html",
            ],
            "{tier:?}"
        );
    }
}

#[test]
fn job_callbacks_keep_the_module_that_registered_them() {
    let mut engine = Engine::new();
    engine.set_job_callback_script_caller_capture(true);
    engine.set_import_base("https://doc.test/page.html");
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let log = seen.clone();
    let result = engine.eval_module(
        "await null; Promise.resolve(\"import('./job.js')\").then(eval);",
        "https://m.test/dir/m.js",
        move |specifier, referrer| {
            log.borrow_mut().push(format!("{specifier} <- {referrer}"));
            Some((
                format!("{referrer}#{specifier}"),
                "export default 1;".into(),
            ))
        },
    );
    assert!(result.is_ok());
    engine.run_microtasks();
    assert_eq!(*seen.borrow(), ["./job.js <- https://m.test/dir/m.js"]);
}

#[test]
fn non_browser_hosts_keep_the_default_job_callbacks() {
    // ECMA-262 §9.5.2: without the HTML hooks a built-in job callback has no saved script, so
    // its import() refers to the current Realm.
    let mut engine = Engine::new();
    let seen = recording_loader(&mut engine);
    engine.set_import_base("https://a.test/a.js");
    assert!(matches!(
        engine.eval_value("Promise.resolve(\"import('./job.js')\").then(eval);"),
        Ok(Ok(_))
    ));
    engine.set_import_base("https://doc.test/page.html");
    engine.run_microtasks();
    assert_eq!(*seen.borrow(), ["./job.js <- https://doc.test/page.html"]);
}
