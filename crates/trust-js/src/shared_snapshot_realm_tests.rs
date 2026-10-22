//! Realms that evaluate one shared host snapshot (see `Interp::shared_snapshot_program`).
//!
//! Every Realm runs its own ScriptEvaluation of the same decoded Script body: GlobalDeclaration-
//! Instantiation, closures, environments and template objects belong to the Realm that ran the
//! code (ECMA-262 OrdinaryFunctionCreate's [[Realm]], PrepareForOrdinaryCall, and
//! GetTemplateObject's per-Realm [[TemplateMap]]; local snapshot e28783d).
//! Only the immutable AST and the code compiled for it are shared, so these tests check every
//! tier for Realm-correct literals, intrinsics, global references and errors, both in the Realm
//! that compiled a function and in another Realm reusing that code.

use crate::bytecode::Tier;
use crate::interpreter::{Abrupt, Interp};
use crate::realm_inline_guard_tests::{eval_string, in_realm, render, set_global, TIERS};
use crate::value::{Callable, Value};
use crate::Engine;
use std::rc::Rc;

/// A platform-style bootstrap: an IIFE whose closures read per-Realm configuration, a global
/// binding, intrinsics through literals and constructors, and a hot loop that tiers up.
const BOOTSTRAP: &str = r#"(function () {
    "use strict";
    const realmTag = globalThis.realmTag;
    class Base {
        constructor(x) { this.x = x; }
        describe() { return realmTag + ":" + this.x; }
    }
    class Derived extends Base {
        describe() { return "derived " + super.describe(); }
    }
    function makeArray() { return [1, 2, 3]; }
    function makeObject() { return { a: 1, b: 2 }; }
    function makeReader(value) { return { get value() { return value; } }; }
    function makeRegExp() { return /a+/g; }
    function tag(strings) { return strings; }
    function template() { return tag`x${1}y`; }
    function makeTypeError() { try { null.f; } catch (e) { return e; } }
    function makeRangeError() { try { new Array(-1); } catch (e) { return e; } }
    function readGlobal() { return globalThis.realmTag + "|" + realmTag + "|" + typeof realmOnly; }
    function makeClosure() { return () => this; }
    function hot(n) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
            const instance = new Derived(i);
            sum += makeArray().length + instance.x + (makeObject() instanceof Object ? 1 : 0);
        }
        return sum;
    }
    globalThis.probe = {
        Base, Derived, makeArray, makeObject, makeReader, makeRegExp, template, makeTypeError,
        makeRangeError, readGlobal, makeClosure, hot,
    };
})();
"#;

/// The Realm-local facts a shared bootstrap must report, evaluated in the Realm itself.
const CHECKS: &str = r#"(function () {
    const results = [];
    for (let round = 0; round < 3; round++) {
        const array = probe.makeArray();
        const object = probe.makeObject();
        const derived = new probe.Derived(4);
        results.push([
            probe.hot(300),
            Object.getPrototypeOf(array) === Array.prototype,
            Array.isArray(array),
            Object.getPrototypeOf(object) === Object.prototype,
            Object.getPrototypeOf(probe.makeRegExp()) === RegExp.prototype,
            Object.getPrototypeOf(probe.Base) === Function.prototype,
            Object.getPrototypeOf(probe.Base.prototype) === Object.prototype,
            Object.getPrototypeOf(probe.makeArray) === Function.prototype,
            derived instanceof probe.Base && derived instanceof Object,
            derived.describe(),
            probe.makeTypeError() instanceof TypeError,
            probe.makeTypeError().constructor === TypeError,
            probe.makeRangeError() instanceof RangeError,
            probe.template() === probe.template(),
            Object.getPrototypeOf(probe.template()) === Array.prototype,
            probe.readGlobal(),
        ].join(","));
    }
    return results.join(";");
})()"#;

fn expected(tag: &str, realm_only: &str) -> String {
    let round = format!(
        "{},true,true,true,true,true,true,true,true,derived {tag}:4,true,true,true,true,true,{tag}|{tag}|{realm_only}",
        // hot(300): Σ (3 + i + 1) for i in 0..300.
        300 * 4 + (299 * 300) / 2,
    );
    [round.as_str(), round.as_str(), round.as_str()].join(";")
}

fn bootstrap_snapshot() -> &'static [u8] {
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    SNAPSHOT.get_or_init(|| crate::compile_host_snapshot(BOOTSTRAP).expect("bootstrap parses"))
}

fn run_shared(it: &mut Interp) {
    let body = it
        .shared_snapshot_program(bootstrap_snapshot())
        .expect("snapshot decodes");
    match it.run_classic_program(&body, None) {
        Ok(_) => {}
        Err(Abrupt::Throw(error)) => panic!("bootstrap threw: {}", render(it, &error)),
        Err(_) => panic!("bootstrap completed abruptly"),
    }
}

fn shared_method(it: &mut Interp, name: &str) -> Rc<crate::ast::Function> {
    let global = Value::Obj(it.global.clone());
    let Ok(shared) = it.get_member_key(&global, "shared") else {
        panic!("shared is defined");
    };
    let Ok(Value::Obj(function)) = it.get_member_key(&shared, name) else {
        panic!("shared.{name} is a function");
    };
    let borrowed = function.borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("shared.{name} is an ordinary function");
    };
    user.func.clone()
}

/// The AST Function behind a probe method, to prove both Realms run the same code.
fn probe_function(it: &mut Interp, name: &str) -> Rc<crate::ast::Function> {
    let global = Value::Obj(it.global.clone());
    let Ok(probe) = it.get_member_key(&global, "probe") else {
        panic!("probe is defined");
    };
    let Ok(Value::Obj(function)) = it.get_member_key(&probe, name) else {
        panic!("probe.{name} is a function");
    };
    let borrowed = function.borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("probe.{name} is an ordinary function");
    };
    user.func.clone()
}

fn global_user_callable(it: &mut Interp, name: &str) -> crate::value::UserFn {
    let env = it.global_env.clone();
    let value = it
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing global {name}"));
    let Value::Obj(object) = value else {
        panic!("{name} is not an object")
    };
    let borrowed = object.borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("{name} is not an ordinary function")
    };
    user.clone()
}

fn has_binding_getter_cache(user: &crate::value::UserCallable) -> bool {
    let cache = user.take_binding_getter_cache();
    let cached = cache.is_some();
    user.put_binding_getter_cache(cache);
    cached
}

#[test]
fn realms_sharing_a_snapshot_keep_distinct_intrinsics_globals_and_objects() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        let child = it.create_realm();
        let main = Value::Obj(it.global.clone());

        set_global(it, "realmTag", Value::lstr("main"));
        run_shared(it);
        // The first Realm warms the shared functions: bytecode and native code now exist.
        assert_eq!(
            eval_string(it, CHECKS),
            expected("main", "undefined"),
            "{tier:?} main"
        );

        in_realm(it, &child, |it| {
            set_global(it, "realmTag", Value::lstr("child"));
            set_global(it, "realmOnly", Value::Num(1.0));
            run_shared(it);
            assert_eq!(
                eval_string(it, CHECKS),
                expected("child", "number"),
                "{tier:?} child"
            );
        });
        assert_eq!(
            it.shared_snapshots.len(),
            1,
            "the snapshot is decoded once per Agent"
        );

        // Both Realms' closures run the same immutable code ...
        let main_hot = probe_function(it, "hot");
        let child_hot = in_realm(it, &child, |it| probe_function(it, "hot"));
        assert!(Rc::ptr_eq(&main_hot, &child_hot), "{tier:?}: shared AST");
        if tier != Tier::Interp {
            assert!(
                main_hot.runtime().code.get().is_some_and(Option::is_some),
                "{tier:?}: the shared function was compiled"
            );
        }

        // ... but every object they create belongs to the Realm that created it.
        set_global(it, "childRealm", child.clone());
        in_realm(it, &child, |it| set_global(it, "mainRealm", main.clone()));
        let cross = eval_string(
            it,
            r#"[
                probe !== childRealm.probe,
                probe.makeArray !== childRealm.probe.makeArray,
                probe.Base !== childRealm.probe.Base,
                Object.getPrototypeOf(childRealm.probe.makeArray()) === childRealm.Array.prototype,
                Object.getPrototypeOf(childRealm.probe.makeArray()) !== Array.prototype,
                Object.getPrototypeOf(childRealm.probe.makeObject()) === childRealm.Object.prototype,
                childRealm.probe.makeTypeError() instanceof childRealm.TypeError,
                !(childRealm.probe.makeTypeError() instanceof TypeError),
                !(new childRealm.probe.Derived(1) instanceof probe.Base),
                childRealm.probe.template() !== probe.template(),
                childRealm.probe.makeRegExp() !== probe.makeRegExp(),
                Object.getPrototypeOf(childRealm.probe.Base) === childRealm.Function.prototype,
                childRealm.probe.readGlobal(),
                probe.readGlobal(),
                new childRealm.probe.Derived(2).describe(),
                childRealm.probe.makeClosure.call(7)() === 7,
            ].join(",")"#,
        );
        assert_eq!(
            cross,
            "true,true,true,true,true,true,true,true,true,true,true,true,\
             child|child|number,main|main|undefined,derived child:2,true",
            "{tier:?} cross-Realm"
        );
        // Calls into the other Realm's closures from a warmed caller.
        let calls = in_realm(it, &child, |it| {
            eval_string(
                it,
                r#"(function () {
                    let same = 0;
                    for (let i = 0; i < 400; i++) {
                        if (Object.getPrototypeOf(mainRealm.probe.makeArray()) === mainRealm.Array.prototype) same++;
                        if (Object.getPrototypeOf(probe.makeArray()) === Array.prototype) same++;
                    }
                    return same;
                })()"#,
            )
        });
        assert_eq!(calls, "800", "{tier:?} alternating Realm calls");
    }
}

#[test]
fn shared_snapshot_binding_getters_cache_each_realms_live_closure() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        let child = it.create_realm();
        run_shared(it);
        assert_eq!(
            eval_string(
                it,
                r#"
                var reader = probe.makeReader('main');
                var readerGetter = Object.getOwnPropertyDescriptor(reader, 'value').get;
                for (var n = 0; n < 4; n++) reader.value;
                reader.value
            "#,
            ),
            "main"
        );
        let main_getter = global_user_callable(it, "readerGetter");
        assert!(
            has_binding_getter_cache(&main_getter),
            "{tier:?}: main cache"
        );

        in_realm(it, &child, |it| {
            run_shared(it);
            assert_eq!(
                eval_string(
                    it,
                    r#"
                    var reader = probe.makeReader('child');
                    var readerGetter = Object.getOwnPropertyDescriptor(reader, 'value').get;
                    for (var n = 0; n < 4; n++) reader.value;
                    reader.value
                "#,
                ),
                "child"
            );
            let child_getter = global_user_callable(it, "readerGetter");
            assert!(
                has_binding_getter_cache(&child_getter),
                "{tier:?}: child cache"
            );
            assert!(
                Rc::ptr_eq(&main_getter.func, &child_getter.func),
                "{tier:?}: snapshot AST is shared"
            );
            assert!(
                !std::ptr::eq(
                    crate::value::user_fn_ptr(&main_getter),
                    crate::value::user_fn_ptr(&child_getter)
                ),
                "{tier:?}: cached environments are closure-local"
            );
        });
    }
}

/// Speculative inlining (`bytecode::plan_inlines`) splices a callee only into a caller of the
/// same Realm, and the splice evaluates the callee's literals in the running Realm. Code on a
/// shared AST runs in every Realm, so another Realm's closure of the same caller must not take
/// a splice planned for the first Realm even when it is handed that Realm's exact callee: the
/// guard falls back to an ordinary call, which enters the callee's own Realm (see
/// `realm_inline_guard_tests`).
#[test]
fn a_shared_inlined_caller_runs_cross_realm_callees_in_their_realm() {
    const SOURCE: &str = r#"(function () {
        "use strict";
        globalThis.shared = {
            // Not a tail call: a strict `return f()` is a proper tail call, never inlined.
            call(f) { const result = f(); return result; },
            callee() { return [7]; },
        };
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] =
        SNAPSHOT.get_or_init(|| crate::compile_host_snapshot(SOURCE).expect("source parses"));
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        let main = Value::Obj(it.global.clone());
        let run = |it: &mut Interp| {
            let body = it.shared_snapshot_program(snapshot).expect("decodes");
            assert!(it.run_classic_program(&body, None).is_ok());
        };
        run(it);
        // Warm the caller with this Realm's callee until a second-stage compile inlines it,
        // as a page does before it creates its first nested Window.
        assert_eq!(
            eval_string(
                it,
                "(function () { let n = 0; for (let i = 0; i < 2000; i++) n += shared.call(shared.callee)[0]; return n; })()"
            ),
            "14000"
        );
        if tier == Tier::Jit {
            let call = shared_method(it, "call");
            assert!(
                call.runtime().code2.get().is_some_and(Option::is_some),
                "the warmed caller has an inlined second-stage body"
            );
        }
        let child = it.create_realm();
        in_realm(it, &child, run);
        in_realm(it, &child, |it| set_global(it, "mainRealm", main.clone()));
        let result = in_realm(it, &child, |it| {
            eval_string(
                it,
                r#"(function () {
                    let main = 0, child = 0;
                    for (let i = 0; i < 300; i++) {
                        const fromMain = shared.call(mainRealm.shared.callee);
                        if (Object.getPrototypeOf(fromMain) === mainRealm.Array.prototype) main++;
                        const own = shared.call(shared.callee);
                        if (Object.getPrototypeOf(own) === Array.prototype) child++;
                    }
                    return main + "/" + child;
                })()"#,
            )
        });
        assert_eq!(result, "300/300", "{tier:?}");
    }
}

/// Unresolvable global references in shared code: a plain read (caught ReferenceError),
/// `typeof`, and call position, in Script code, a once-run loop body that compiles at entry,
/// and a function hot enough to tier up.
const UNRESOLVABLE: &str = r#"
// A Window-like global: failed lookups walk Window.prototype and a named-properties Proxy
// whose traps are closures of this shared code (Web IDL #named-properties-object).
(function () {
    "use strict";
    const g = globalThis;
    function EventTarget() {}
    function Window() {}
    let epoch = -1, names = new Map();
    function refresh() {
        if (epoch === 1) return;
        epoch = 1;
        names = new Map([["namedFrame", g]]);
    }
    const target = Object.create(EventTarget.prototype);
    const windowProperties = new Proxy(target, {
        has(target, property) {
            if (typeof property !== "string") return Reflect.has(target, property);
            refresh();
            return names.has(property) || Reflect.has(target, property);
        },
        get(target, property, receiver) {
            if (typeof property === "string") {
                refresh();
                if (names.has(property)) return names.get(property);
            }
            return Reflect.get(target, property, receiver);
        },
    });
    Object.setPrototypeOf(Window.prototype, windowProperties);
    Object.setPrototypeOf(g, Window.prototype);
})();
const scriptKind = typeof NoSuchGlobalForTrustJsProbe === "function";
let scriptRead = "unset";
try { scriptRead = NoSuchGlobalForTrustJsProbe; } catch (e) { scriptRead = e instanceof ReferenceError; }
(function () {
    "use strict";
    function probeRead() {
        try { return NoSuchGlobalForTrustJsProbe; } catch (e) { return e instanceof ReferenceError; }
    }
    function probeTypeof() { return typeof NoSuchGlobalForTrustJsProbe; }
    function probeCall() {
        try { return NoSuchGlobalForTrustJsProbe(); } catch (e) { return e instanceof ReferenceError; }
    }
    let results = [];
    for (let i = 0; i < 300; i++) {
        results = [probeRead(), probeTypeof(), probeCall(),
                   typeof NoSuchGlobalForTrustJsProbe === "function"];
    }
    globalThis.unresolvable = results.join() + "|" + scriptKind + "," + scriptRead;
})();
"#;

fn unresolvable_snapshot() -> &'static [u8] {
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    SNAPSHOT.get_or_init(|| crate::compile_host_snapshot(UNRESOLVABLE).expect("source parses"))
}

/// A Realm that evaluated shared code with unresolvable references is collected once nothing
/// references it: no cache filled by the failed lookups may own its global, environment or
/// objects (`AGENTS.md`: compiled code owns no JavaScript objects).
#[test]
fn a_dropped_realm_that_ran_shared_unresolvable_lookups_is_collected() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        let run = |it: &mut Interp| {
            let body = it
                .shared_snapshot_program(unresolvable_snapshot())
                .expect("decodes");
            if let Err(Abrupt::Throw(error)) = it.run_classic_program(&body, None) {
                panic!("snapshot threw: {}", render(it, &error));
            }
            eval_string(it, "unresolvable")
        };
        let expected = "true,undefined,true,false|false,true";
        assert_eq!(run(it), expected, "{tier:?} main");
        let mut addresses = Vec::new();
        for round in 0..3 {
            // The child Realm lives only in this helper frame and its address stays hidden
            // (the collector scans the stack).
            addresses.push(crate::gc_test::in_frame(|| {
                let child = it.create_realm();
                let address = crate::gc_test::HiddenAddress::of(&child);
                assert_eq!(
                    in_realm(it, &child, run),
                    expected,
                    "{tier:?} child {round}"
                );
                address
            }));
        }
        it.gc_collect();
        for (round, address) in addresses.iter().enumerate() {
            assert!(
                !address.in_map(&it.realms),
                "{tier:?}: dropped child Realm {round} is still live"
            );
        }
    }
}

/// The legacy `RegExp.$1` statics are recorded lazily against the matching Realm's %RegExp%.
/// That pending record must not keep the Realm alive: a dropped Realm whose code ran the last
/// regular expression is collected, and the statics of live Realms are unaffected.
#[test]
fn a_dropped_realm_that_ran_the_last_regexp_is_collected() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        assert_eq!(eval_string(it, "/(b)c/.exec('abcd'); RegExp.$1"), "b");
        // The child Realm lives only in this helper frame and its address stays hidden (the
        // collector scans the stack).
        let address = crate::gc_test::in_frame(|| {
            let child = it.create_realm();
            let address = crate::gc_test::HiddenAddress::of(&child);
            let child_statics = in_realm(it, &child, |it| {
                eval_string(
                    it,
                    "(function () { let n = 0; for (let i = 0; i < 300; i++) n += /(x+)y/.test('axxy'); \
                     return n + ':' + RegExp.$1 + ':' + RegExp.lastMatch; })()",
                )
            });
            assert_eq!(child_statics, "300:xx:xxy", "{tier:?}");
            // The child ran the most recent match, so its %RegExp% holds the pending statics.
            in_realm(it, &child, |it| {
                assert_eq!(
                    eval_string(it, "/(q)/.test('q') && typeof RegExp"),
                    "function"
                );
            });
            address
        });
        it.gc_collect();
        assert!(
            !address.in_map(&it.realms),
            "{tier:?}: the Realm of the last regexp match is still live"
        );
        assert_eq!(
            eval_string(it, "RegExp.$1 + RegExp.lastMatch"),
            "bbc",
            "{tier:?}"
        );
    }
}
