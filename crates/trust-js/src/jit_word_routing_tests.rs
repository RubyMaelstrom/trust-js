//! Execution words that own nothing (P1 S5 step 9a). Under traced ownership generated code
//! copies, overwrites and pops words without routing any tag to a checked path, so two kinds of
//! word are driven through the paths that used to route them:
//! - A function's deferred `prototype` is a property-only word and must never become an
//!   execution word. Every native path that can read a property keyed `prototype` sends it to
//!   its checked operation, which materializes it: the named read, the computed method read,
//!   `LoadRef` through a `with` object and the region's borrowed own and method reads. Each test
//!   reads `F.prototype` before and after materialization and checks the object's identity
//!   across reads, its `constructor` back edge and its own prototype.
//! - A BigInt used to leave the fast path wherever its owner needed the checked one. Each test
//!   drives a BigInt through such a site and checks the results and the BigInt's value after
//!   every store or compare.
//! - A deferred prototype's word names a cell and owns nothing either (step 9b): the native
//!   property stores that used to send a lazy old value to the checked store or to a drop helper
//!   overwrite it, materializing nothing, and the old cell dies at a collection.
//!
//! Programs run at every tier with the same expected results; at `Tier::Jit` the functions named
//! ran as native code and, where a test counter records it, the native path ran (or, for a
//! deferred prototype, declined to the checked operation). The stress child
//! (`jit::rooting_tests::rooting_holds_under_gc_stress`) also runs them at `TRUST_JS_GC_STRESS=1`.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

/// A fresh engine at `tier` that compiles every function at its first call.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Every named global function has native code.
fn assert_native(engine: &mut Engine, names: &[&str]) {
    for name in names {
        let env = engine.interp.global_env;
        let value = engine.interp.get_var(name, &env).ok().expect("bound name");
        let function_handle = value.as_obj().expect("a function object");
        let object = function_handle.borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("{name} is not a user function")
        };
        let runtime = user.func.runtime();
        let native = runtime
            .code
            .get()
            .and_then(Option::as_ref)
            .is_some_and(|chunk| chunk.jit.get().flatten().is_some());
        assert!(native, "{name} has no native code");
    }
}

/// How far `counter` advanced while `run` ran.
fn delta(counter: impl Fn() -> u64, run: impl FnOnce()) -> u64 {
    let before = counter();
    run();
    counter() - before
}

/// Native named-read hits and checked named reads (`regions::property_paths` 0 and 2).
fn named_reads(engine: &mut Engine, source: &str) -> (u64, u64) {
    let before = super::regions::property_paths();
    eval(engine, source);
    let after = super::regions::property_paths();
    (after[0] - before[0], after[2] - before[2])
}

/// Functions with a deferred `prototype` (ordinary function expressions, all of one shape) and
/// the check of a materialized one: the same object on every read, whose `constructor` is the
/// function and whose prototype is `Object.prototype`.
const FUNCTIONS: &str = r#"
    function make() { return function () {}; }
    function fresh(n) { var fns = []; for (var i = 0; i < n; i++) fns[i] = make(); return fns; }
    function materialized(F, p) {
        return p === F.prototype && p === F.prototype && p.constructor === F &&
            Object.getPrototypeOf(p) === Object.prototype;
    }
"#;

/// The named read (the GetProp family's template): a function's deferred `prototype` takes the
/// checked read, which materializes it; once materialized, every read is a native hit. Ten fresh
/// functions read twice each through a warmed `readPrototype`: ten checked reads and no native
/// hit first, then ten native hits, and both reads return the materialized object.
#[test]
fn deferred_prototype_named_reads_take_the_checked_read_once() {
    let source = format!(
        "{FUNCTIONS}
        function readPrototype(F) {{ return F.prototype; }}
        function readAll(fns, out, n) {{ for (var i = 0; i < n; i++) out[i] = readPrototype(fns[i]); }}
        function check(fns, firsts, seconds, n) {{
            for (var i = 0; i < n; i++)
                if (firsts[i] !== seconds[i] || !materialized(fns[i], firsts[i])) return false;
            return true;
        }}
        var warm = fresh(8), fns = fresh(10), firsts = [], seconds = [];
        readAll(warm, [], 8); readAll(warm, [], 8);
        0;"
    );
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(&mut engine, &source);
        let lazy = named_reads(&mut engine, "readAll(fns, firsts, 10); 0");
        let warm = named_reads(&mut engine, "readAll(fns, seconds, 10); 0");
        assert_eq!(
            eval(&mut engine, "check(fns, firsts, seconds, 10)"),
            "true",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert_native(&mut engine, &["readPrototype", "readAll"]);
            assert_eq!(lazy, (0, 10), "deferred prototypes take the checked read");
            assert_eq!(warm, (10, 0), "materialized prototypes are native hits");
        }
    }
}

/// `F[k]()` with `k` = "prototype" at run time, through the computed method read: a deferred
/// prototype declines to the checked read (one `GetMethodElem` helper call per fresh function),
/// a materialized one is a native hit (no helper call). Either way the method is the
/// materialized prototype object, which is not callable: the call throws a TypeError, and the
/// prototype is then the same object on every read.
#[test]
fn deferred_prototype_computed_method_reads_take_the_checked_read_once() {
    let source = format!(
        "{FUNCTIONS}
        function callKey(F, k) {{ return F[k](); }}
        function viaKey(F, k) {{
            try {{ callKey(F, k); return 'called'; }}
            catch (e) {{ return e instanceof TypeError ? 'TypeError' : 'other'; }}
        }}
        function callAll(fns, k, n) {{
            var out = '';
            for (var i = 0; i < n; i++) out += viaKey(fns[i], k) === 'TypeError' ? 't' : 'x';
            return out;
        }}
        function check(fns, n) {{
            for (var i = 0; i < n; i++) if (!materialized(fns[i], fns[i].prototype)) return false;
            return true;
        }}
        var key = 'prototype', warm = fresh(8), fns = fresh(10);
        callAll(warm, key, 8); callAll(warm, key, 8);
        0;"
    );
    let helpers = || crate::bytecode::TEST_JIT_GET_ELEM_HELPERS.with(|count| count.get()[2] as u64);
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(&mut engine, &source);
        let mut results = Vec::new();
        let lazy = delta(helpers, || {
            results.push(eval(&mut engine, "callAll(fns, key, 10)"))
        });
        let warm = delta(helpers, || {
            results.push(eval(&mut engine, "callAll(fns, key, 10)"))
        });
        assert_eq!(results, ["tttttttttt", "tttttttttt"], "{tier:?}");
        assert_eq!(eval(&mut engine, "check(fns, 10)"), "true", "{tier:?}");
        if tier == Tier::Jit {
            assert_native(&mut engine, &["callKey", "viaKey", "callAll"]);
            assert_eq!(
                lazy, 10,
                "deferred prototypes take the checked computed read"
            );
            assert_eq!(warm, 0, "materialized prototypes are native computed reads");
        }
    }
}

/// `prototype` read through a `with (F)` object Reference (`LoadRef`; a generator, whose
/// resumable body compiles `with`): per fresh function the first read declines to the checked
/// Reference operation (it materializes the prototype) and the second, through another Reference
/// of the same name, completes natively; for a materialized prototype both are native. Both
/// reads return the materialized object.
#[test]
fn deferred_prototype_with_object_references_take_the_checked_read_once() {
    let source = format!(
        "{FUNCTIONS}
        function* withPrototype(F) {{
            var first, second;
            with (F) {{ first = (prototype ||= 0); second = (prototype ||= 0); }}
            yield first === second && materialized(F, first);
        }}
        function viaWith(F) {{ return withPrototype(F).next().value; }}
        function viaAll(fns, n) {{
            var out = true;
            for (var i = 0; i < n; i++) out = viaWith(fns[i]) && out;
            return out;
        }}
        var warm = fresh(8), fns = fresh(10);
        viaAll(warm, 8); viaAll(warm, 8);
        0;"
    );
    let loads = || super::names::TEST_NATIVE_REFERENCES.with(|counts| counts[1].get() as u64);
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(&mut engine, &source);
        let mut results = Vec::new();
        let lazy = delta(loads, || results.push(eval(&mut engine, "viaAll(fns, 10)")));
        let warm = delta(loads, || results.push(eval(&mut engine, "viaAll(fns, 10)")));
        assert_eq!(results, ["true", "true"], "{tier:?}");
        if tier == Tier::Jit {
            assert_native(&mut engine, &["withPrototype", "viaWith", "viaAll"]);
            assert_eq!(lazy, 10, "only the read after materialization is native");
            assert_eq!(
                warm, 20,
                "materialized prototypes are native Reference reads"
            );
        }
    }
}

/// `F.prototype` through a loop region's borrowed own read (feedback warmed at
/// `Tier::Bytecode`): a deferred prototype leaves the region at that read, whose checked read
/// materializes it, and every later read in the region is borrowed natively. The loop sees one
/// object throughout, the materialized prototype.
#[test]
fn deferred_prototype_region_own_reads_leave_the_region_once() {
    let source = format!(
        "{FUNCTIONS}
        function regionRead(F, n) {{
            var p = null, changes = 0;
            for (var i = 0; i < n; i++) {{ var q = F.prototype; if (q !== p) changes++; p = q; }}
            return changes === 1 ? p : null;
        }}
        var warm = fresh(8);
        for (var r = 0; r < 4; r++) for (var w = 0; w < 8; w++) regionRead(warm[w], 5);
        var F = make(), first, second;
        0;"
    );
    let reads = || super::regions::borrowed_reads().0;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        // The region's borrowed read needs warm feedback when the function compiles.
        let mut engine = engine(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        eval(&mut engine, &source);
        engine.set_tier(tier);
        let lazy = delta(reads, || {
            assert_eq!(eval(&mut engine, "first = regionRead(F, 20); 0"), "0")
        });
        let warm = delta(reads, || {
            assert_eq!(eval(&mut engine, "second = regionRead(F, 20); 0"), "0")
        });
        assert_eq!(
            eval(
                &mut engine,
                "first !== null && first === second && materialized(F, first)"
            ),
            "true",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert_native(&mut engine, &["regionRead"]);
            assert_eq!(
                lazy, 19,
                "only the deferred prototype's read leaves the region"
            );
            assert_eq!(
                warm, 20,
                "materialized prototypes are borrowed in the region"
            );
        }
    }
}

/// `F.prototype(i)` through a loop region's borrowed method read (the call inlined once a
/// callable `prototype` warmed the site): a deferred prototype fails that read before it is
/// borrowed, and its checked read materializes it; a materialized one is borrowed and then fails
/// the inline guard. Either way the call throws a TypeError (the prototype object is not
/// callable), and the prototype is the same object on every later read.
#[test]
fn deferred_prototype_region_method_reads_are_not_borrowed() {
    let source = format!(
        "{FUNCTIONS}
        function callPrototype(F, n) {{
            var s = 0;
            for (var i = 0; i < n; i++) {{ s += F.prototype(i); if (i === 3) s++; }}
            return s;
        }}
        function viaCall(F) {{
            try {{ return callPrototype(F, 9); }}
            catch (e) {{ return e instanceof TypeError ? 'TypeError' : 'other'; }}
        }}
        var W = make();
        W.prototype = function (x) {{ return x + 1; }};
        var good = true;
        for (var r = 0; r < 160; r++) good = callPrototype(W, 9) === 46 && good;
        var F = make();
        good;"
    );
    let borrowed = || super::regions::guarded_regions()[3];
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, &source), "true", "{tier:?}");
        let mut results = Vec::new();
        let lazy = delta(borrowed, || results.push(eval(&mut engine, "viaCall(F)")));
        let warm = delta(borrowed, || results.push(eval(&mut engine, "viaCall(F)")));
        assert_eq!(results, ["TypeError", "TypeError"], "{tier:?}");
        assert_eq!(
            eval(&mut engine, "materialized(F, F.prototype)"),
            "true",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert_native(&mut engine, &["callPrototype", "viaCall"]);
            assert_eq!(
                delta(borrowed, || assert_eq!(
                    eval(&mut engine, "callPrototype(W, 9)"),
                    "46"
                )),
                9,
                "the callable prototype's method reads are borrowed"
            );
            assert_eq!(lazy, 0, "a deferred prototype is not borrowed");
            assert_eq!(
                warm, 1,
                "a materialized prototype is borrowed, then fails the guard"
            );
        }
    }
}

/// The deferred-prototype cells `engine`'s heap holds.
fn lazy_cells(engine: &Engine) -> usize {
    let heap = &engine.interp.gc_heap;
    heap.core.borrow().live_count(heap.lazy_proto_desc)
}

/// `F.prototype = p` over unobserved deferred prototypes through the generated SetProp
/// template (a warmed, compiled `setPrototype`): every store completes in the template, with no
/// set-property helper call (so no checked store), no old prototype is materialized (the probe),
/// the new value reads back, and the old cells are gone after the next collection.
#[test]
fn deferred_prototype_set_prop_overwrites_are_native_and_free_the_old_cells() {
    let source = format!(
        "{FUNCTIONS}
        function setPrototype(F, p) {{ F.prototype = p; }}
        function setAll(fns, p, n) {{ for (var i = 0; i < n; i++) setPrototype(fns[i], p); }}
        function check(fns, p, n) {{
            for (var i = 0; i < n; i++) if (fns[i].prototype !== p) return false;
            return true;
        }}
        var p = {{}}, fns = null;
        setAll(fresh(8), p, 8); setAll(fresh(8), p, 8);
        0;"
    );
    let set_helpers =
        || crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get) as u64;
    let mut engine = engine(Tier::Jit);
    eval(&mut engine, &source);
    engine.interp.gc_collect();
    let cells = lazy_cells(&engine);
    eval(&mut engine, "fns = fresh(10); 0");
    assert_eq!(
        lazy_cells(&engine),
        cells + 10,
        "one cell per fresh function"
    );
    let materialized = crate::value::prototypes_materialized();
    let helpers = delta(set_helpers, || {
        eval(&mut engine, "setAll(fns, p, 10); 0");
    });
    assert_native(&mut engine, &["setPrototype", "setAll"]);
    assert_eq!(helpers, 0, "every store over a lazy word is native");
    assert_eq!(
        crate::value::prototypes_materialized(),
        materialized,
        "no old prototype is materialized"
    );
    assert_eq!(eval(&mut engine, "check(fns, p, 10)"), "true");
    engine.interp.gc_collect();
    assert_eq!(
        lazy_cells(&engine),
        cells,
        "the old cells died by the next collection"
    );
}

/// `prototype = p` inside `with (F)` over unobserved deferred prototypes (`StoreRef`
/// through the with object's Reference; a generator, whose resumable body compiles `with`):
/// every store completes natively (the native Reference-store counter) and overwrites the lazy
/// word, no old prototype is materialized (the probe), the new value reads back, and the old
/// cells are gone after the next collection.
#[test]
fn deferred_prototype_with_object_stores_are_native_and_free_the_old_cells() {
    let source = format!(
        "{FUNCTIONS}
        function* withStore(F, p) {{ with (F) {{ prototype = p; }} yield 0; }}
        function viaWith(F, p) {{ withStore(F, p).next(); }}
        function storeAll(fns, p, n) {{ for (var i = 0; i < n; i++) viaWith(fns[i], p); }}
        function check(fns, p, n) {{
            for (var i = 0; i < n; i++) if (fns[i].prototype !== p) return false;
            return true;
        }}
        var p = {{}}, fns = null;
        storeAll(fresh(8), p, 8); storeAll(fresh(8), p, 8);
        0;"
    );
    let stores = || super::names::TEST_NATIVE_REFERENCES.with(|counts| counts[2].get() as u64);
    let mut engine = engine(Tier::Jit);
    eval(&mut engine, &source);
    engine.interp.gc_collect();
    let cells = lazy_cells(&engine);
    eval(&mut engine, "fns = fresh(10); 0");
    assert_eq!(
        lazy_cells(&engine),
        cells + 10,
        "one cell per fresh function"
    );
    let materialized = crate::value::prototypes_materialized();
    let native = delta(stores, || {
        eval(&mut engine, "storeAll(fns, p, 10); 0");
    });
    assert_native(&mut engine, &["withStore", "viaWith", "storeAll"]);
    assert_eq!(
        native, 10,
        "every Reference store over a lazy word is native"
    );
    assert_eq!(
        crate::value::prototypes_materialized(),
        materialized,
        "no old prototype is materialized"
    );
    assert_eq!(eval(&mut engine, "check(fns, p, 10)"), "true");
    engine.interp.gc_collect();
    assert_eq!(
        lazy_cells(&engine),
        cells,
        "the old cells died by the next collection"
    );
}

/// BigInts through the baseline templates that used to send a BigInt to their checked path
/// only for its owner: a retained local store (`b = (a = v)`), `instanceof` with a BigInt left
/// operand, a retained property store and a property store over a BigInt, loose `==` against
/// null in both orders, strict `===` across different tags, and stores over a BigInt in a
/// global property and in a closure binding. Results and the BigInt's value are checked after
/// every store and compare. The property stores run inline (no set-property helper call).
#[test]
fn bigints_take_the_baseline_fast_paths_with_their_values() {
    const SOURCE: &str = r#"
        var big = 12345678901234567890123n, small = 7n;
        function same(v) { return v === 12345678901234567890123n; }
        function keep(v) { var a, b; b = (a = v); return [a, b]; }
        function isInstance(v, F) { return v instanceof F; }
        function setKeep(o, v) { return o.x = v; }
        function setOver(o, v) { o.x = v; }
        function nullEq(v) { return null == v; }
        function eqNull(v) { return v == null; }
        function strict(a, b) { return a === b; }
        var globalValue = 0;
        function setGlobal(v) { globalValue = v; }
        function makeCell(v) {
            var x = v;
            return { set: function (w) { x = w; }, get: function () { return x; } };
        }
        var F = function () {}, o = { x: 0 }, cell = makeCell(0);
        for (var i = 0; i < 4; i++) {
            keep(i); isInstance({}, F); setKeep(o, i); setOver(o, i); nullEq(i); eqNull(i);
            strict(i, i); setGlobal(i); cell.set(i); cell.get();
        }
        0;
    "#;
    let checks = [
        "var k = keep(big); same(k[0]) && same(k[1]) && same(big)",
        "isInstance(big, F) === false && isInstance(small, F) === false && same(big)",
        "same(setKeep(o, big)) && same(o.x) && same(big)",
        "setOver(o, 1); o.x === 1 && same(big)",
        "nullEq(big) === false && eqNull(big) === false && nullEq(small) === false && same(big)",
        "strict(big, 1) === false && strict(1, big) === false && strict('1', big) === false && \
         strict(big, null) === false && strict(big, big) === true && same(big)",
        "setGlobal(big); same(globalValue) && (setGlobal(2), globalValue === 2) && same(big)",
        "cell.set(small); cell.set(big); same(cell.get()) && (cell.set(3), cell.get() === 3) && \
         same(big) && small === 7n",
    ];
    let set_helpers =
        || crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get) as u64;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(&mut engine, SOURCE);
        let mut helpers = Vec::new();
        for check in checks {
            helpers.push(delta(set_helpers, || {
                assert_eq!(eval(&mut engine, check), "true", "{tier:?}: {check}")
            }));
        }
        if tier == Tier::Jit {
            assert_native(
                &mut engine,
                &[
                    "keep",
                    "isInstance",
                    "setKeep",
                    "setOver",
                    "nullEq",
                    "eqNull",
                    "strict",
                    "setGlobal",
                ],
            );
            // The retained store of a BigInt and the store over one.
            assert_eq!((helpers[2], helpers[3]), (0, 0));
        }
    }
}

/// BigInts through the region sites that used to leave the region (or skip the fused path) for
/// a BigInt's owner: a borrowed own read of a BigInt property, a tagged-home read of a BigInt
/// local, the self-hosted `map` loop's fused HasProperty/Get pair over BigInt elements, and a
/// linked-list scan entered with a BigInt in its `peek` local. Results and the BigInts' values
/// are checked; the borrowed reads and the fused pair hit natively for BigInts.
#[test]
fn bigints_take_the_region_fast_paths_with_their_values() {
    const SOURCE: &str = r#"
        var big = 12345678901234567890123n;
        function same(v) { return v === 12345678901234567890123n; }
        function ownRead(o, n) { var v; for (var i = 0; i < n; i++) v = o.big; return v; }
        function homeRead(v, n, sink) {
            var r, last = 0;
            for (var i = 0; i < n; i++) { r = v; last = sink(r); }
            return [r, last];
        }
        function sink(v) { return v; }
        function mapped(values) { return values.map(function (v) { return v; }); }
        function lastNode(list, start) {
            var next = list, peek = start;
            while ((peek = next.link) != null) next = peek;
            return [next, peek];
        }
        var plain = { big: 1n };
        for (var r = 0; r < 4; r++) { ownRead({ big: 2n }, 5); homeRead(3n, 5, sink); }
        var list = { v: 1, link: { v: 2, link: { v: 3, link: null } } };
        for (var r = 0; r < 4; r++) lastNode(list, 0);
        0;
    "#;
    let reads = || super::regions::borrowed_reads().0;
    let pairs = || {
        let hits = super::test_tdz_array_pair_hits();
        hits[0] + hits[1]
    };
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        eval(&mut engine, SOURCE);
        engine.set_tier(tier);
        let borrowed = delta(reads, || {
            assert_eq!(
                eval(
                    &mut engine,
                    "var o = { big: big }; same(ownRead(o, 20)) && same(o.big)"
                ),
                "true",
                "{tier:?}"
            )
        });
        assert_eq!(
            eval(
                &mut engine,
                "var h = homeRead(big, 20, sink); same(h[0]) && same(h[1]) && same(big)"
            ),
            "true",
            "{tier:?}"
        );
        let fused = delta(pairs, || {
            assert_eq!(
                eval(
                    &mut engine,
                    "var m = mapped([big, 2n, big]); m.length === 3 && same(m[0]) && \
                         m[1] === 2n && same(m[2]) && same(big)"
                ),
                "true",
                "{tier:?}"
            )
        });
        assert_eq!(
            eval(
                &mut engine,
                "var s = lastNode(list, big); s[0].v === 3 && s[1] === null && same(big)"
            ),
            "true",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert_native(&mut engine, &["ownRead", "homeRead", "lastNode"]);
            assert_eq!(borrowed, 20, "a BigInt property is borrowed in the region");
            assert!(fused > 0, "the fused pair admits BigInt elements");
        }
    }
}
