//! W3's item 7, W4's call feedback where W3 needs it: every native's builtin id and call-cache
//! class, declared as its realm's setup installs it and recorded by the call cache's fill (V8's
//! `Builtin` id, which the optimizing tiers key their reductions on).

use crate::bytecode::*;
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("feedback fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The native function `path` names in a fresh realm of `engine`, through a global it binds.
fn native(engine: &mut Engine, path: &str) -> crate::value::NativeFn {
    eval(engine, &format!("var probedNative = {path};"));
    let env = engine.interp.global_env;
    let value: Value = engine
        .interp
        .get_var("probedNative", &env)
        .ok()
        .expect("bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{path} is not an object")
    };
    let call = &function.borrow().kind;
    match call {
        ObjectKind::Native(f) => *f,
        _ => panic!("{path} is not a bare native"),
    }
}

/// The classes the realm's setup declares, by the natives that carry them: each call template's
/// native, the operand-only collection methods, Math's templated and numeric-only functions,
/// DataView's accessors, the light natives of item 2a, and natives with no class.
const CLASSES: &[(&str, u8)] = &[
    ("String.prototype.charCodeAt", INTRINSIC_CHAR_CODE_AT),
    ("String.prototype.codePointAt", INTRINSIC_CODE_POINT_AT),
    ("String.prototype.charAt", INTRINSIC_CHAR_AT),
    ("String.prototype.slice", INTRINSIC_STRING_SLICE),
    ("Object.hasOwn", INTRINSIC_OBJECT_HAS_OWN),
    ("Function.prototype.apply", INTRINSIC_FUNCTION_APPLY),
    ("Function.prototype.call", INTRINSIC_FUNCTION_CALL),
    ("Math.sqrt", INTRINSIC_MATH_SQRT),
    ("Array.prototype.push", INTRINSIC_ARRAY_PUSH),
    ("Array.prototype.pop", INTRINSIC_ARRAY_POP),
    ("RegExp.prototype.exec", INTRINSIC_REGEXP_EXEC_DISCARD),
    ("String.prototype.replace", INTRINSIC_STRING_REPLACE_DISCARD),
    ("String.prototype.split", INTRINSIC_STRING_SPLIT_DISCARD),
    ("isNaN", INTRINSIC_IS_NAN),
    ("Number.isNaN", INTRINSIC_IS_NAN),
    ("String.fromCharCode", INTRINSIC_FROM_CHAR_CODE),
    ("Reflect.apply", INTRINSIC_REFLECT_APPLY),
    ("Map.prototype.get", INTRINSIC_OPERAND_ONLY),
    ("Map.prototype.set", INTRINSIC_OPERAND_ONLY),
    ("Set.prototype.has", INTRINSIC_OPERAND_ONLY),
    ("WeakMap.prototype.get", INTRINSIC_OPERAND_ONLY),
    ("WeakSet.prototype.delete", INTRINSIC_OPERAND_ONLY),
    ("Math.floor", INTRINSIC_MATH_FLOOR),
    ("Math.abs", INTRINSIC_MATH_ABS),
    ("Math.imul", INTRINSIC_MATH_IMUL),
    ("Math.max", INTRINSIC_MATH_MAX),
    ("Math.min", INTRINSIC_MATH_MIN),
    ("Math.sin", INTRINSIC_NUMERIC_ONLY),
    ("Math.log", INTRINSIC_NUMERIC_ONLY),
    ("DataView.prototype.getInt8", INTRINSIC_DATAVIEW_GET),
    ("DataView.prototype.setFloat64", INTRINSIC_DATAVIEW_SET),
    ("parseInt", INTRINSIC_LIGHT_PRIMITIVE_ARGS),
    ("Number", INTRINSIC_LIGHT_PRIMITIVE_ARGS),
    ("Number.isInteger", INTRINSIC_LIGHT),
    ("Boolean", INTRINSIC_LIGHT),
    ("String.prototype.indexOf", INTRINSIC_LIGHT_PRIMITIVE),
    ("Array.isArray", INTRINSIC_ARRAY_IS_ARRAY),
    ("Object.is", INTRINSIC_OBJECT_IS),
    ("Object.keys", 0),
    ("JSON.stringify", 0),
    ("Math.atan2", 0),
    ("Map.prototype.forEach", 0),
];

#[test]
fn every_native_has_a_builtin_id_and_its_declared_class() {
    let mut first = Engine::new();
    let mut second = Engine::new();
    let mut ids = crate::fasthash::FastMap::default();
    for &(path, class) in CLASSES {
        let f = native(&mut first, path);
        let (id, declared) = crate::builtins::builtin(f as usize);
        assert_ne!(id, 0, "{path} has no builtin id");
        assert_eq!(declared, class, "{path}'s class");
        if let Some(other) = ids.insert(id, path) {
            panic!("{path} and {other} share builtin id {id}");
        }
        // The id names the native, whatever the realm that installed it.
        let again = native(&mut second, path);
        assert_eq!(again as usize, f as usize);
        assert_eq!(
            crate::builtins::builtin(again as usize),
            (id, class),
            "{path}"
        );
    }
}

/// The global function `name`'s compiled chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> std::rc::Rc<Chunk> {
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let call = &function.borrow().kind;
    let ObjectKind::User(user) = call else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} stayed interpreted"));
    std::rc::Rc::clone(chunk)
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_native_entry_records_its_builtin_id_and_class() {
    for tier in [Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let source = r#"
var wm = new WeakMap(), key = {}; wm.set(key, 2);
function run(n) { var s = 0; for (var k = 0; k < n; k++) s += Math.floor(k / 2) + wm.get(key) + [k].indexOf(k); return s; }
run(50) + run(50)"#;
        assert_eq!(eval(&mut engine, source), "1400", "{tier:?}");
        let chunk = chunk_of(&mut engine, "run");
        let mut seen = Vec::new();
        for site in &chunk.call_caches {
            for way in &site.entries {
                let entry = way.get();
                if entry.native != 0 {
                    let (builtin, class) = crate::builtins::builtin(entry.native);
                    assert_eq!(
                        (entry.builtin, entry.intrinsic),
                        (builtin, class),
                        "{tier:?}"
                    );
                    seen.push(class);
                } else if entry.callee != 0 {
                    assert_eq!(entry.builtin, 0, "a user function has no builtin id");
                }
            }
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            [0, INTRINSIC_OPERAND_ONLY, INTRINSIC_MATH_FLOOR],
            "{tier:?}: Array.prototype.indexOf, WeakMap.prototype.get and Math.floor"
        );
    }
}

/// Call sites by their callees: `churn` calls fresh closures of ten function literals in turn
/// (and the ten makers, from another site), `stable` six functions in turn, `wide` twelve.
const SITES: &str = r#"
var makers = [];
for (var q = 0; q < 10; q++) makers.push(new Function('return function (a) { return a + ' + q + '; }'));
function churn(n) { var s = 0; for (var k = 0; k < n; k++) { var f = makers[k % 10](); s += f(k); } return s; }
var six = []; for (var q = 0; q < 6; q++) six.push(new Function('a', 'return a * ' + q));
function stable(n) { var s = 0; for (var k = 0; k < n; k++) s += six[k % 6](k); return s; }
var twelve = []; for (var q = 0; q < 12; q++) twelve.push(new Function('a', 'return a - ' + q));
function wide(n) { var s = 0; for (var k = 0; k < n; k++) s += twelve[k % 12](k); return s; }
[churn(400) + churn(400), stable(400) + stable(400), wide(480) + wide(480)].join()"#;

/// The call sites of the global function `name`'s chunk that went megamorphic, by index.
fn megamorphic_sites(engine: &mut Engine, name: &str) -> Vec<usize> {
    let chunk = chunk_of(engine, name);
    (0..chunk.call_caches.len())
        .filter(|&site| chunk.call_caches[site].megamorphic())
        .collect()
}

#[test]
fn a_site_whose_callees_churn_goes_megamorphic_and_keeps_its_ways() {
    use std::sync::atomic::Ordering::Relaxed;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        // CALL_IC_EPOCH is process-wide: a test running beside this one that recompiles a
        // chunk or reclaims code stales every way, and the refills after it evict ways again.
        // A site counts its evictions within one epoch (see `CallSite::fill`), so which sites
        // go megamorphic does not depend on such bumps; the pins and the ways are checked on a
        // run that ended at the epoch it started at.
        for attempt in 0..200 {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let start = CALL_IC_EPOCH.load(Relaxed);
            assert_eq!(eval(&mut engine, SITES), "163200,398140,224640", "{tier:?}");
            #[cfg(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            if tier == Tier::Jit {
                // Both of `churn`'s sites see more than eight callees; `wide`'s twelve stay
                // stable, but its ways churned before the overflow table held them all.
                assert_eq!(megamorphic_sites(&mut engine, "churn").len(), 2);
                assert_eq!(megamorphic_sites(&mut engine, "wide").len(), 1);
                assert!(megamorphic_sites(&mut engine, "stable").is_empty());
                if CALL_IC_EPOCH.load(Relaxed) != start && attempt < 199 {
                    continue;
                }
                // A megamorphic site pins only the callees its ways held: the others went to
                // the overflow table, which pins its own.
                let pins = chunk_of(&mut engine, "churn").call_pins.borrow().len();
                assert!(pins <= 2 * (CALL_IC_WAYS + usize::from(CALL_SITE_MEGAMORPHIC_EVICTIONS)));
                // The ways of a megamorphic site are not refilled: more calls leave them as they
                // were (while the call-cache epoch stays).
                let chunk = chunk_of(&mut engine, "churn");
                let before: Vec<Vec<usize>> = chunk
                    .call_caches
                    .iter()
                    .map(|site| site.entries.iter().map(|e| e.get().callee).collect())
                    .collect();
                let epoch = CALL_IC_EPOCH.load(Relaxed);
                assert_eq!(eval(&mut engine, "churn(100)"), "5400");
                if epoch == CALL_IC_EPOCH.load(Relaxed) {
                    let after: Vec<Vec<usize>> = chunk
                        .call_caches
                        .iter()
                        .map(|site| site.entries.iter().map(|e| e.get().callee).collect())
                        .collect();
                    assert_eq!(before, after);
                }
            }
            break;
        }
    }
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_stable_site_counts_its_evictions_within_one_epoch() {
    // Six callees through four ways evict two ways once; each epoch's bump stales the ways and
    // the overflow table, and the refills evict two again. Counted across epochs, three bumps
    // would make the site megamorphic; counted within one, it never is.
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "var six = []; for (var q = 0; q < 6; q++) six.push(new Function('a', 'return a * ' + q));
         function stable(n) { var s = 0; for (var k = 0; k < n; k++) s += six[k % 6](k); return s; }",
    );
    for _ in 0..6 {
        assert_eq!(eval(&mut engine, "stable(60)"), "4600");
        crate::bytecode::invalidate_call_caches();
    }
    assert!(megamorphic_sites(&mut engine, "stable").is_empty());
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_megamorphic_site_puts_a_budget_of_callees_in_the_overflow_table() {
    use std::sync::atomic::Ordering::Relaxed;
    // As in the test above, on a run that ended at the epoch it started at.
    for attempt in 0..200 {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let start = CALL_IC_EPOCH.load(Relaxed);
        assert_eq!(eval(&mut engine, SITES), "163200,398140,224640");
        let epoch = CALL_IC_EPOCH.load(Relaxed);
        if epoch != start && attempt < 199 {
            continue;
        }
        // `churn`'s fresh closures never come back: each of its two sites put its evicted ways
        // and its budget in the table, not one entry for each of its 800 calls. `stable` and
        // `wide` add at most two and eight.
        let budget = usize::from(CALL_SITE_MEGAMORPHIC_EVICTIONS + CALL_SITE_OVERFLOW_ADMISSIONS);
        let occupied = engine.interp.call_overflow.occupied();
        assert!(
            occupied <= 2 * budget + 2 + 8,
            "{occupied} overflow entries"
        );
        // `wide`'s twelve callees repeat: its ways and the table hold them all.
        let chunk = chunk_of(&mut engine, "wide");
        let site = (0..chunk.call_caches.len())
            .find(|&site| chunk.call_caches[site].megamorphic())
            .expect("wide's megamorphic site");
        let genv = crate::interpreter::Env::as_ptr(&engine.interp.global_env) as usize;
        for q in 0..12 {
            eval(&mut engine, &format!("var probedCallee = twelve[{q}];"));
            let env = engine.interp.global_env;
            let value: Value = engine
                .interp
                .get_var("probedCallee", &env)
                .ok()
                .expect("bound name");
            let ValueView::Obj(function) = value.view() else {
                panic!("twelve[{q}] is not an object")
            };
            let key = crate::value::Gc::as_ptr(&function) as usize;
            let in_ways = chunk.call_caches[site]
                .entries
                .iter()
                .any(|way| way.get().callee == key && way.get().epoch == epoch);
            assert!(
                in_ways
                    || engine
                        .interp
                        .call_overflow
                        .lookup(key, genv, epoch)
                        .is_some(),
                "twelve[{q}] is not cached"
            );
        }
        break;
    }
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn the_inline_planner_skips_megamorphic_sites_and_recompiles_keep_them() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(eval(&mut engine, SITES), "163200,398140,224640");
    let chunk = chunk_of(&mut engine, "wide");
    let megamorphic = megamorphic_sites(&mut engine, "wide");
    let env = engine.interp.global_env;
    let value: Value = engine
        .interp
        .get_var("wide", &env)
        .ok()
        .expect("bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("wide is not an object")
    };
    let (func, realm) = match &function.borrow().kind {
        ObjectKind::User(user) => (std::rc::Rc::clone(&user.func), user.realm),
        _ => panic!("wide is not a user function"),
    };
    let plan = plan_inlines(&chunk, &func, realm, &env);
    for site in &megamorphic {
        assert!(!plan.contains_key(&(*site as u32)), "site {site} spliced");
    }
    // A recompilation's sites start from the old chunk's (see `call_cache_seeds`).
    let seeds = call_cache_seeds(&chunk);
    assert!(seeds
        .iter()
        .any(|seed| seed.evictions >= CALL_SITE_MEGAMORPHIC_EVICTIONS));
}

/// A `new` site that constructs a fresh class, or a fresh closure of one function, on each pass
/// pins one constructor, the one its entry names, not every one it has seen: a dead
/// constructor's pin keeps its cell reserved and its weak cell in every minor collection's scan.
/// ARES-6 Air's `Liveness.prototype.localCalc` (`class LocalCalc` and `new LocalCalc()`) left
/// 62,000 of them in one chunk's call pins a round. V8's construct feedback slot holds one weak
/// target too (src/builtins/ic-callable.tq:227-276).
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_construct_site_of_fresh_constructors_pins_only_its_entry() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        eval(
            &mut engine,
            "function makeClass(x) { class Local { constructor() { this.x = x; } } return new Local(); }
             function makeFunction(x) { function Local() { this.x = x; } return new Local(); }
             var sum = 0;
             for (var k = 0; k < 300; k++) sum += makeClass(k).x + makeFunction(k).x;
             sum"
        ),
        "89700"
    );
    for maker in ["makeClass", "makeFunction"] {
        // Every chunk the function runs (a second-stage compile included) holds this.
        let env = engine.interp.global_env;
        let value: Value = engine.interp.get_var(maker, &env).ok().expect("bound name");
        let ValueView::Obj(function) = value.view() else {
            panic!("{maker} is not an object")
        };
        let call = &function.borrow().kind;
        let ObjectKind::User(user) = call else {
            panic!("{maker} is not a user function")
        };
        let runtime = user.func.runtime();
        let chunks: Vec<std::rc::Rc<Chunk>> = [runtime.code.get(), runtime.code2.get()]
            .into_iter()
            .flatten()
            .flatten()
            .cloned()
            .collect();
        let mut filled = 0;
        for chunk in &chunks {
            let named: Vec<usize> = chunk
                .construct_caches
                .iter()
                .map(|site| site.get().call.callee)
                .filter(|&callee| callee != 0)
                .collect();
            let pinned: Vec<usize> = chunk
                .construct_pins
                .iter()
                .map(|pin| pin.borrow().as_ptr() as usize)
                .filter(|&ptr| ptr != 0)
                .collect();
            assert_eq!(
                pinned, named,
                "{maker} pins exactly its entry's constructor"
            );
            assert!(
                chunk.call_pins.borrow().is_empty(),
                "{maker}'s call pins hold {} constructors",
                chunk.call_pins.borrow().len()
            );
            filled += named.len();
        }
        assert!(filled >= 1, "{maker}'s site took the construct cache");
    }
}
