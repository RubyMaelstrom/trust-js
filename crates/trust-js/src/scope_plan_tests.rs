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
fn closures_in_compiled_coroutines_read_their_own_and_outer_names() {
    // A compiled generator makes an activation of its own below the record its call's
    // FunctionDeclarationInstantiation made, with its captured bindings in declaration order
    // (`u` before `s`, the reverse of a planned prefix's): the method and the arrow made in it
    // must find `u`, `s` and the outer `x` by name, not at a planned record's slots.
    check(
        r#"
        function outer(o) {
            var x = o + 'x';
            function* gen(n) {
                var u = 'u' + n, s = 's' + n;
                var m = { k() { return x + s + u; } };
                yield m.k();
                yield (() => x + u + s)();
            }
            return gen;
        }
        var out = [];
        for (var i = 0; i < 20; i++) {
            var it = outer('o' + i)(i);
            out.push(it.next().value + ',' + it.next().value);
        }
        out[0] + '|' + out[19];
        "#,
        "o0xs0u0,o0xu0s0|o19xs19u19,o19xu19s19",
    );
}

#[test]
fn parameter_expressions_do_not_see_captured_body_bindings() {
    // FunctionDeclarationInstantiation (spec.html:14247) gives a list with expressions its own
    // record: the body's vars, functions (an Annex B block function's var among them) and
    // lexicals are not visible to it, even where a closure captures them, and a body function
    // that redeclares a parameter does not replace the value a later default reads.
    check(
        r#"
        var x = 'outer';
        globalThis.x2 = function () { return 'outerfn'; };
        var cases = [
            function (a = x) { var x = 'body'; var g = () => x; return a; },
            function (a = x) { function x() {} var g = () => x; return typeof a; },
            function (a = x) { let x = 'body'; var g = () => x; return a; },
            function (a = typeof x) { var x = 1; var g = () => x; return a; },
            function (a = x) { var x; var g = () => x; return a + ':' + g(); },
            function (a, b = a) { function a() {} var g = () => a; return typeof b + ':' + typeof g(); },
            function (a = x) { { function x() {} } var g = () => x; return typeof a; },
            function (a = x2()) { function x2() { return 'inner'; } var g = () => x2; return a; },
            function ({p = x} = {}) { var x = 'body'; var g = () => x; return p; },
            function (a = x) { { let x = 'block'; var g = () => x; } return a; },
            function (a, b = a) { function a() {} return typeof b; },
        ];
        var out = [];
        for (var i = 0; i < 20; i++) out.push(cases.map((f) => f()).join(' '));
        out[0] + '|' + (out[19] === out[0]);
        "#,
        "outer string outer string outer:undefined undefined:function string outerfn outer outer \
         undefined|true",
    );
}

#[test]
fn closures_in_a_using_head_loop_read_its_binding() {
    // ForLoopEvaluation (spec.html:22617, e28783d5) binds a `using` head in the loop's own
    // record, which the closures made in the loop capture; the head's resource is disposed once,
    // after the loop. (Node v24.22.0 disposes it 4 times here.)
    check(
        r#"
        var x = 'global';
        function f(n) {
            var y = 'y' + n, disposed = 0;
            var fns = [];
            for (using x = { v: 'head' + n, [Symbol.dispose]() { disposed++; } }; fns.length < 2; ) {
                fns.push(() => x.v + y);
            }
            return fns.map((g) => g()).join(',') + ':' + disposed;
        }
        var out = [];
        for (var i = 0; i < 20; i++) out.push(f(i));
        out[0] + '|' + out[19];
        "#,
        "head0y0,head0y0:1|head19y19,head19y19:1",
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

#[test]
fn parameter_lists_with_expressions_share_the_activation_when_unobservable() {
    // Defaults, destructuring with defaults, `this` and `arguments` in defaults, and a rest
    // parameter make no closure: the parameters move into the planned activation, which the
    // body's closures read at static addresses in every tier.
    check(
        r#"
        function f(a = 1, b = a + 1) { var c = a + b; return () => a + ':' + b + ':' + c; }
        function g(a, {p = a, q} = {q: 2}, ...rest) { return () => [a, p, q, rest.length].join(','); }
        function k(a = this && this.v) { return () => a; }
        function m(a = arguments.length) { return () => a; }
        function n(a = 1) { function a2() { return a; } var a = 5; return a2(); }
        function o(a = 2) { a = a * 3; return () => a; }
        var out = [];
        for (var i = 0; i < 30; i++) {
            out.push([f()(), f(i)(), g(i)(), g(i, {q: 7}, 1, 2)(), k.call({v: i})(),
                m(undefined, 2, 3)(), n(), o()(), o(i)()].join('|'));
        }
        out[0] + ' / ' + out[29];
        "#,
        "1:2:3|0:1:1|0,0,2,0|0,0,7,2|0|3|5|6|0 / 1:2:3|29:30:59|29,29,2,0|29,29,7,2|29|3|5|6|87",
    );
}

#[test]
fn class_members_read_outer_names_through_the_class_records() {
    // A class's constructor, methods and accessors close over its instance or static record,
    // its private-name record and its name's record, which every tier makes alike: their reads
    // of the enclosing function's names, and of the class's own name, are static.
    check(
        r#"
        function outer(k) {
            var helper = function (v) { return v + k; };
            var shared = 'S' + k;
            class Box {
                #p = 1;
                constructor(v) { this.v = helper(v); }
                get twice() { return this.v * 2 + shared.length + this.#p; }
                set twice(x) { this.v = x - k; }
                static make(v) { return new Box(v); }
                label() { return typeof Box + ':' + Box.name + ':' + shared; }
                arrow() { return [1, 2].map((x) => x + k + this.v).join('/'); }
            }
            const Anon = class { m() { return shared + k; } };
            const Named = class Inner { m() { return Inner.name + k; } };
            var b = Box.make(k);
            var before = b.twice;
            b.twice = 10;
            return [before, b.v, b.twice, b.label(), b.arrow(), new Anon().m(), new Named().m()]
                .join(',');
        }
        var out = [];
        for (var i = 0; i < 30; i++) out.push(outer(i));
        out[0] + '|' + out[29];
        "#,
        "3,10,23,function:Box:S0,11/12,S00,Inner0|120,-19,-34,function:Box:S29,11/12,S2929,Inner29",
    );
    assert!(
        static_reads(
            "function a(x) { return class { m() { return x; } }; } var t = 0; \
             for (var k = 0; k < 20; k++) t += new (a(k))().m(); t;"
        ) > 0
    );
}

#[test]
fn closures_a_class_encloses_otherwise_keep_their_names_dynamic() {
    // A class in a private function (whose activation only the tree-walker makes), a class in
    // a heritage, a computed key's closure, a static block, a class in a block with lexicals and
    // field initializers' closures resolve their names by name in every tier.
    check(
        r#"
        var top = 'T';
        function wrap(k) {
            var near = 'n' + k;
            function priv(x) { var unused = x; return class { m() { return top + near; } }; }
            function heritage() {
                return class extends (class { m() { return near + k; } }) {
                    n() { return this.m() + near; }
                };
            }
            var keyed = class { [(() => 'k' + near)()]() { return near; } static { this.s = near + k; } };
            var block;
            { let local = 'L' + k; block = class { m() { return local + near; } }; }
            var withField = class { f = near + k; g = () => near; };
            var wf = new withField();
            return [new (priv(k))().m(), new (heritage())().n(), new keyed()['kn' + k](), keyed.s,
                new block().m(), wf.f, wf.g()].join(',');
        }
        var out = [];
        for (var i = 0; i < 30; i++) out.push(wrap(i));
        out[0] + '|' + out[29];
        "#,
        "Tn0,n00n0,n0,n00,L0n0,n00,n0|Tn29,n2929n29,n29,n2929,L29n29,n2929,n29",
    );
}

#[test]
fn a_compiled_class_closes_over_its_frame_past_the_slot_projection() {
    // A compiled frame evaluates a class through a projection of its slots, which its heritage
    // and computed keys read (`parent`, `name`: no closure captures them); its records then close
    // over the frame's own record before a static field calls a static method, which reads the
    // frame's captured names at their static addresses.
    check(
        r#"
        function make(k) {
            var parent = k % 2 ? Array : Object;
            var name = 'm' + (k % 3);
            var base = parent;
            var shared = 's' + k;
            return class extends parent {
                static early = this.peek();
                static peek() { return shared + ':' + typeof base; }
                [name]() { return shared + k; }
                get s() { return typeof base + shared.length; }
            };
        }
        var out = [];
        for (var i = 0; i < 30; i++) {
            var C = make(i);
            var c = new C();
            out.push([C.early, c['m' + (i % 3)](), c.s, c instanceof Array].join(','));
        }
        out[0] + '|' + out[29];
        "#,
        "s0:function,s00,function2,false|s29:function,s2929,function3,true",
    );
}
