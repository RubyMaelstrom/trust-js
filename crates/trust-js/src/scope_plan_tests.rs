//! Decision 21: a function's plan (`bytecode::scope_plan`) makes its activation the same record,
//! with the same leading bindings, in every tier; private activations are skipped by the
//! closures made in them; and a free-name read that reaches its binding through planned records
//! only reads it at a static (hops, slot) address. ECMA-262 e28783d5: ResolveBinding and
//! GetIdentifierReference (spec.html:12043, 10669), the declarative Environment Record's
//! GetBindingValue (10887), FunctionDeclarationInstantiation (14247) and PerformEval (18960).
//! Expected results are Node's (v24.22.0).
use crate::{bytecode::Tier, Completion, Engine};

fn run(source: &str, tier: Tier, threshold: u32) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(threshold);
    match engine
        .eval(source, false)
        .expect("scope plan fixture parses")
    {
        Completion::Value(actual) => actual,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 8] {
            assert_eq!(
                run(source, tier, threshold),
                expected,
                "{tier:?}/{threshold}: {source}"
            );
        }
    }
}

/// Static reads the bytecode tier made while `source` ran.
fn static_reads(source: &str) -> usize {
    super::TEST_STATIC_NAME_READS.with(|count| count.set(0));
    let _ = run(source, Tier::Bytecode, 0);
    super::TEST_STATIC_NAME_READS.with(std::cell::Cell::get)
}

#[test]
fn outer_names_read_through_public_and_private_activations() {
    // `mid` captures nothing: its activation is private, so `inner`'s closure skips it in the
    // tree-walker as compiled code makes none. `outer` runs once (tree-walked) and `mid` both
    // tree-walked and compiled, so the closures' records come from both tiers.
    check(
        r#"
        function outer(a) {
            var v = a + 1; let l = a + 2; const c = a + 3;
            function mid(b) {
                var unused = b;
                return function inner(k) { return v + l + c + a + k; };
            }
            var out = [];
            for (var k = 0; k < 30; k++) out.push(mid(k)(k));
            return out[0] + ',' + out[29];
        }
        outer(1) + '|' + outer(10);
        "#,
        "10,39|46,75",
    );
}

#[test]
fn outer_names_read_from_two_and_three_functions_out() {
    check(
        r#"
        function a(x) {
            var y = x * 2;
            return function b() {
                var z = y + 1;
                return function c() {
                    return function d() { return x + y + z; };
                };
            };
        }
        var total = 0;
        for (var k = 0; k < 40; k++) total += a(k)()()();
        total;
        "#,
        "3940",
    );
    assert!(static_reads("function a(x) { return function b() { return x; }; } var t = 0; for (var k = 0; k < 20; k++) t += a(k)(); t;") > 0);
}

#[test]
fn a_static_read_of_a_lexical_in_its_dead_zone_throws() {
    check(
        r#"
        function outer() {
            function read() { return early; }
            var caught = 'none';
            try { read(); } catch (e) { caught = e.name; }
            let early = 5;
            return caught + ':' + read();
        }
        var out = [];
        for (var k = 0; k < 30; k++) out.push(outer());
        out[0] + '|' + out[29];
        "#,
        "ReferenceError:5|ReferenceError:5",
    );
}

#[test]
fn eval_and_with_keep_the_names_they_can_shadow_dynamic() {
    // A sloppy direct eval in `b` adds a var `x` that shadows `a`'s: `b` is opaque, so `c`'s
    // read stays a name lookup. A `with` object's property shadows likewise.
    check(
        r#"
        function a() {
            var x = 1;
            function b(code) {
                eval(code);
                return function c() { return x; };
            }
            return b('var x = 2')() + ',' + b('')();
        }
        function w(o) {
            var x = 'outer';
            with (o) { return function () { return x; }; }
        }
        var out = [];
        for (var k = 0; k < 30; k++) out.push(a() + ',' + w({ x: 'with' })() + ',' + w({})());
        out[0] + '|' + out[29];
        "#,
        "2,1,with,outer|2,1,with,outer",
    );
    assert_eq!(
        static_reads("function a() { var x = 1; eval(''); return function c() { return x; }; } var t = 0; for (var k = 0; k < 20; k++) t += a()(); t;"),
        0
    );
}

#[test]
fn closures_in_records_one_tier_makes_alone_resolve_their_names() {
    // A `switch` block, `for` heads, `catch` clauses and blocks with lexicals are records the
    // tree-walker makes where compiled code may not: their closures stay behind a barrier.
    check(
        r#"
        function f(n) {
            var v = 'v' + n;
            var fs = [];
            switch (n % 2) { case 0: fs.push(() => v + ':even'); break; default: fs.push(() => v + ':odd'); }
            for (var i = 0; i < 2; i++) fs.push(() => v + ':' + i);
            for (let j = 0; j < 2; j++) fs.push(() => v + ':' + j);
            try { throw 'e'; } catch (e) { fs.push(() => v + ':' + e); }
            { let b = 'b'; fs.push(() => v + ':' + b); }
            return fs.map((g) => g()).join(',');
        }
        var out = [];
        for (var k = 0; k < 30; k++) out.push(f(k));
        out[0] + '|' + out[29];
        "#,
        "v0:even,v0:2,v0:2,v0:0,v0:1,v0:e,v0:b|v29:odd,v29:2,v29:2,v29:0,v29:1,v29:e,v29:b",
    );
}

#[test]
fn self_names_home_objects_and_arrow_this_resolve_statically() {
    check(
        r#"
        var count = function down(n) { return n ? down(n - 1) + 1 : 0; };
        function maker(x) {
            return {
                plain() { return () => x; },
                sup() { return () => super.toString === Object.prototype.toString; },
            };
        }
        function Box(v) { this.v = v; this.get = () => this.v; }
        var out = [];
        for (var k = 0; k < 30; k++) {
            var m = maker(k);
            out.push(count(k) + ',' + m.plain()() + ',' + m.sup()() + ',' + new Box(k).get());
        }
        out[0] + '|' + out[29];
        "#,
        "0,0,true,0|29,29,true,29",
    );
}

#[test]
fn generators_and_derived_constructors_keep_their_own_records() {
    check(
        r#"
        function* gen(a) { var b = a + 1; yield () => a + b; yield () => b; }
        class Base { constructor(v) { this.v = v; } }
        class Derived extends Base {
            constructor(v) { var read = () => this.v; super(v); this.read = read; }
        }
        var out = [];
        for (var k = 0; k < 30; k++) {
            var g = gen(k);
            out.push(g.next().value() + ',' + g.next().value() + ',' + new Derived(k).read());
        }
        out[0] + '|' + out[29];
        "#,
        "1,1,0|59,30,29",
    );
}
