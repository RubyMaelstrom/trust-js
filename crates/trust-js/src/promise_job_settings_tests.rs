//! ECMA-262 #sec-newpromisereactionjob and HTML #hostenqueuepromisejob: a reaction's handler
//! Realm and settings survive its settlement in another Realm. GetFunctionRealm still runs when
//! the job is enqueued.

use crate::embed::{Ctx, Handle, Value};
use crate::realm_inline_guard_tests::TIERS;
use crate::Engine;

fn current_context<'s>(
    cx: &mut Ctx<'s>,
    _this: Value<'s>,
    _args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    Ok(Value::Num(cx.host_job_context() as f64))
}

/// Evaluate `source` in the active Realm and return its completion as text.
fn text(engine: &mut Engine, source: &str) -> String {
    engine.scope(|cx| {
        let value = cx
            .eval_value_interruptible(source)
            .expect("settings fixture parses")
            .unwrap_or_else(|_| panic!("settings fixture evaluates: {source}"));
        cx.coerce_string(&value)
            .unwrap_or_else(|_| panic!("settings result converts to text"))
            .to_string()
    })
}

/// A child Realm whose settings token is 41, published as the main Realm's `foreign` (the main
/// Realm's token stays 0), with the main Realm's global as its `main`.
fn foreign(engine: &mut Engine) -> Handle {
    engine.define_global("currentHostContext", 0, current_context);
    engine.scope(|cx| {
        let main = cx.global_this();
        let child = cx.create_embed_realm();
        cx.member_set(&main, "foreign", child)
            .unwrap_or_else(|_| panic!("publish child Realm"));
        cx.with_embed_realm(&child, |cx| {
            cx.set_host_job_context(41);
            cx.define_embed_global("currentHostContext", 0, current_context);
            let global = cx.global_this();
            cx.member_set(&global, "main", main)
                .unwrap_or_else(|_| panic!("publish main Realm"));
        })
        .unwrap_or_else(|_| panic!("configure child Realm"));
        Handle::new(cx, &child).expect("a handle to the child's global")
    })
}

#[test]
fn promise_job_settings_pending_chains_keep_the_handler_settings_in_both_directions() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let child = foreign(&mut engine);
        // The first handler runs in main and settles the child-created dependent promise there.
        // Its second handler was registered by the child's Promise.prototype.then, so the
        // registration token alone is not the main handler's settings. Reversing the Realms
        // must work too.
        text(
            &mut engine,
            r#"
            globalThis.observed = [];
            foreign.Promise.resolve('plain').then(value => value + '!')
                .then(value => observed.push(value + ':' + currentHostContext()));
            foreign.Promise.resolve('adopted').then(value => foreign.Promise.resolve(value + '!'))
                .then(value => observed.push(value + ':' + currentHostContext()));
            foreign.Promise.resolve().then(() => { throw 'rejected'; })
                .catch(reason => observed.push(reason + ':' + currentHostContext()));
        "#,
        );
        engine
            .with_embed_realm(&child, |engine| {
                text(
                    engine,
                    r#"
                globalThis.observed = [];
                main.Promise.resolve('plain').then(value => value + '!')
                    .then(value => observed.push(value + ':' + currentHostContext()));
                main.Promise.resolve('adopted').then(value => main.Promise.resolve(value + '!'))
                    .then(value => observed.push(value + ':' + currentHostContext()));
                main.Promise.resolve().then(() => { throw 'rejected'; })
                    .catch(reason => observed.push(reason + ':' + currentHostContext()));
            "#,
                )
            })
            .unwrap_or_else(|_| panic!("register child handlers"));
        engine.run_microtasks();
        let main = text(&mut engine, "observed.sort().join('|')");
        let child_result = engine
            .with_embed_realm(&child, |engine| text(engine, "observed.sort().join('|')"))
            .unwrap_or_else(|_| panic!("read child settings"));
        assert_eq!(
            (main.as_str(), child_result.as_str()),
            (
                "adopted!:0|plain!:0|rejected:0",
                "adopted!:41|plain!:41|rejected:41",
            ),
            "{tier:?}: a settlement Realm cannot reinterpret a registration token",
        );
        assert_eq!(engine.host_job_context(), 0, "{tier:?}");
    }
}

#[test]
fn promise_job_settings_same_realm_preserve_the_registration_context() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("currentHostContext", 0, current_context);
        engine.set_host_job_context(41);
        text(
            &mut engine,
            r#"
            globalThis.observed = [];
            globalThis.release = null;
            const pending = new Promise(resolve => globalThis.release = resolve);
            pending.then(value => observed.push(value + ':' + currentHostContext()));
        "#,
        );
        engine.set_host_job_context(42);
        text(&mut engine, "release('same')");
        engine.run_microtasks();
        assert_eq!(
            text(&mut engine, "observed.join('|')"),
            "same:41",
            "{tier:?}"
        );
        assert_eq!(engine.host_job_context(), 42, "{tier:?}");
    }
}

#[test]
fn promise_job_settings_resolve_a_revoked_handler_realm_at_enqueue_time() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let _child = foreign(&mut engine);
        text(
            &mut engine,
            r#"
            globalThis.observed = '';
            globalThis.release = null;
            const pending = new foreign.Promise(resolve => globalThis.release = resolve);
            const pair = Proxy.revocable(() => 'unused', {});
            pending.then(pair.proxy).catch(error => {
                observed = currentHostContext() + ':' + (error instanceof foreign.TypeError);
            });
            pair.revoke();
            release('ready');
        "#,
        );
        engine.run_microtasks();
        assert_eq!(text(&mut engine, "observed"), "0:true", "{tier:?}");
    }
}
