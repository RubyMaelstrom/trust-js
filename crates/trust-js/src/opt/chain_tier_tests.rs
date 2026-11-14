//! P3 stage 3 step L2 end to end: named loads through the prototype chain against the VM, with
//! every way JavaScript can change the chain between two loads (the brief's list): a method
//! redefined on its prototype, made an accessor, deleted and added again, shadowed on the
//! receiver and on a level between, a prototype swapped (Object.setPrototypeOf, __proto__) for
//! one of the same shape and for null, a Proxy in the chain, frozen and sealed prototypes, a
//! holder grown to many properties and shrunk, classes extending classes and `super` chains;
//! each before the loads, between two loads of one function (a call between, which ends the
//! facts) and inside a callee called between them (a lowered store at the holder's location, which
//! a graph-inlined callee makes without an effect).

use super::*;

/// The functions: `m1` a method load and its call, `p1` a prototype's data (GetPropLocal), `pt`
/// a `this` read, `m2` two loads around a call, `m3` two loads around a store to the holder's
/// slot by a small callee, `deep` loads at depths 2 and 3, `sup` a `super` chain, `sum` a loop
/// whose only load is the prototype's, `poly` receivers of two shapes sharing a walk and of
/// another walk. The driver mutates the chain in phases.
pub(super) const CHAIN: &str = "class A { m() { return 1; } } A.prototype.p = 20;\n\
     A.prototype.pt = function () { return this.p; };\n\
     class A2 { m() { return 30; } } A2.prototype.p = 21;\n\
     A2.prototype.pt = function () { return this.p + 1000; };\n\
     class B extends A { n() { return 2; } }\n\
     class C extends B { k() { return 3; } }\n\
     class D extends A { m() { return super.m() + 100; } }\n\
     function m1(o) { return o.m(); }\n\
     function p1(o) { return o.p; }\n\
     function pt1(o) { return o.pt(); }\n\
     function m2(o, f) { var a = o.m(); f(o); return a * 1000 + o.m(); }\n\
     function setm(q, g) { q.m = g; }\n\
     function m3(o, q, g) { var a = o.m(); setm(q, g); return a * 1000 + o.m(); }\n\
     function deep(o) { return o.m() * 10 + o.n(); }\n\
     function sup(o) { return o.m(); }\n\
     function sum(o, n) { var s = 0; for (var i = 0; i < n; i++) s = s + o.p; return s; }\n\
     function poly(o) { return o.m(); }\n\
     function noop(o) {}\n\
     function h(x) { return typeof x === 'number' ? x % 1000003 : x === undefined ? -1 : 7; }\n\
     const os = []; for (let i = 0; i < 8; i++) { const o = new A(); if (i & 1) o.extra = i; os.push(o); }\n\
     const bs = [new B(), new B()], cs = [new C(), new C()], ds = [new D()];\n\
     const f1 = function () { return 40; }, f2 = function () { return 41; };\n\
     const keep = A.prototype.m;\n\
     const events = [];\n\
     let t = 0;\n\
     function step(k) {\n\
       const o = os[k & 7], c = cs[k & 1], b = bs[k & 1];\n\
       let r = 0;\n\
       try { r = h(m1(o)); } catch (e) { r = 99; }\n\
       t = (t * 31 + r + h(p1(o)) + h(pt1(o))) % 1000003;\n\
       const between = k === 650 ? function (x) { A.prototype.m = f1; } \
         : k === 1250 ? function (x) { Object.setPrototypeOf(x, A2.prototype); } \
         : k === 1450 ? function (x) { x.m = f2; } \
         : k === 1650 ? function (x) { delete A.prototype.m; A.prototype.m = keep; } \
         : noop;\n\
       try { r = h(m2(o, between)); } catch (e) { r = 98; }\n\
       t = (t * 31 + r) % 1000003;\n\
       try { r = h(m3(o, A.prototype, k % 3 ? keep : f1)); } catch (e) { r = 97; }\n\
       t = (t * 31 + r + h(deep(c)) + h(sup(ds[0])) + h(sum(o, k % 5))) % 1000003;\n\
       try { r = h(poly(k % 3 ? o : (k % 6 === 0 ? c : b))); } catch (e) { r = 96; }\n\
       t = (t * 31 + r) % 1000003;\n\
     }\n\
     for (let k = 0; k < 3000; k++) {\n\
       if (k === 500) A.prototype.m = function () { return 5; };\n\
       if (k === 700) Object.defineProperty(A.prototype, 'm', \
         { get() { return function () { return 6; }; }, configurable: true });\n\
       if (k === 900) { delete A.prototype.m; A.prototype.m = function () { return 7; }; }\n\
       if (k === 1100) os[3].m = function () { return 8; };\n\
       if (k === 1300) B.prototype.m = function () { return 9; };\n\
       if (k === 1500) Object.setPrototypeOf(os[1], A2.prototype);\n\
       if (k === 1700) os[2].__proto__ = { m() { return 11; }, p: 12, pt() { return 13; } };\n\
       if (k === 1900) Object.setPrototypeOf(os[4], new Proxy(A.prototype, \
         { get(t, key, r) { return key === 'm' ? function () { return 12; } : Reflect.get(t, key, r); } }));\n\
       if (k === 2100) { Object.freeze(A2.prototype); Object.seal(B.prototype); }\n\
       if (k === 2300) { for (let i = 0; i < 200; i++) A.prototype['x' + i] = i; \
         for (let i = 0; i < 200; i += 3) delete A.prototype['x' + i]; }\n\
       if (k === 2500) Object.setPrototypeOf(B.prototype, A2.prototype);\n\
       if (k === 2700) Object.setPrototypeOf(os[5], null);\n\
       if (k === 2800) { Object.freeze(os[6]); Object.freeze(A.prototype); }\n\
       step(k);\n\
     }\n\
     t + ':' + m1(os[0]) + ':' + p1(os[7]) + ':' + deep(cs[0]) + ':' + sup(ds[0]) + ':' + sum(os[0], 3)";

/// The gates' configurations, at splice and at TRUST_JS_OPT_INLINE=graph.
fn configurations() -> Vec<(&'static str, TestSwitches)> {
    let mut all = Vec::new();
    for (inline, graph) in [("splice", Some(false)), ("graph", Some(true))] {
        let base = TestSwitches {
            inline_graph: graph,
            ..Default::default()
        };
        all.push((inline, TestSwitches { on: true, ..base }));
        all.push((
            inline,
            TestSwitches {
                on: true,
                eager: true,
                ..base
            },
        ));
        for exit_every in [7, 100] {
            all.push((
                inline,
                TestSwitches {
                    on: true,
                    eager: true,
                    exit_every,
                    ..base
                },
            ));
        }
        all.push((
            inline,
            TestSwitches {
                on: true,
                eager: true,
                exit_limit: 1,
                ..base
            },
        ));
    }
    all
}

#[test]
fn loads_through_the_chain_have_the_vms_results_under_every_mutation() {
    let expected = run(Tier::Bytecode, CHAIN);
    let mut compiles = 0;
    for (label, switches) in configurations() {
        compiled();
        exits();
        assert_eq!(
            with_switches(switches, CHAIN),
            expected,
            "{label} {switches:?}"
        );
        compiles += compiled().len();
    }
    assert!(compiles > 0, "the tier compiled the functions");
}

/// Whether the tier's code of `name` (its graph as a compile builds it now) walks a chain.
fn walks(engine: &mut Engine, name: &str) -> bool {
    let chunk = tier_chunk(engine, name).unwrap_or_else(|| panic!("{name}'s tier chunk"));
    graph_now(&chunk)
        .nodes
        .iter()
        .any(|node| matches!(node.op, super::super::ir::NodeOp::CheckProto { .. }))
}

#[test]
fn the_programs_loads_walk_the_chain_before_its_mutations() {
    // The suite above tests the walk: before the first mutation the tier's code of each function
    // walks the chain (poly's other walk keeps the effect).
    let warm = CHAIN.replace("k < 3000", "k < 400");
    let expected = run(Tier::Bytecode, &warm);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &warm), expected);
    for name in ["m1", "p1", "pt1", "m2", "m3", "deep", "sup", "sum"] {
        assert!(walks(&mut engine, name), "{name} walks the chain");
    }
}

#[test]
fn a_method_load_then_its_call_runs_in_the_tiers_code_without_an_exit() {
    // Warm, then run: the tier's code walks the chain (GetMethod then CallWithThis), and no exit
    // is taken while nothing changes.
    let source = "class A { m(x) { return x + 1; } } class B extends A {} class C extends B {}\n\
         function f(o, x) { return o.m(x); } function g(o) { return o.m(2); }\n\
         const a = new A(), c = new C();\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = t + f(a, k) + g(c); t";
    let more = "let u = 0; for (let k = 0; k < 500; k++) u = u + f(a, k) + g(c); u";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [eval(&mut vm, source), eval(&mut vm, more)]
    };
    eager().set();
    let mut engine = jit_engine();
    compiled();
    assert_eq!(eval(&mut engine, source), expected[0]);
    exits();
    assert_eq!(eval(&mut engine, more), expected[1]);
    assert!(exits().is_empty(), "no exit");
    assert!(walks(&mut engine, "f"), "f walks the chain");
    assert!(walks(&mut engine, "g"), "g walks it to depth 3");
}

#[test]
fn a_redefined_method_reads_its_new_value_with_no_exit() {
    // A value change keeps the holder's shape: the walk passes and the load reads the new
    // method; nothing exits.
    let source =
        "class A { m() { return 1; } } function f(o) { return o.m(); } const a = new A();\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = t + f(a); t";
    let expected = run(Tier::Bytecode, source);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, source), expected);
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "A.prototype.m = function () { return 5; }; let u = 0; \
             for (let k = 0; k < 100; k++) u = u + f(a); u"
        ),
        "500"
    );
    assert!(exits().is_empty(), "a value change exits nothing");
    assert!(walks(&mut engine, "f"));
}

#[test]
fn a_prototype_swapped_for_one_of_the_same_shape_reads_the_new_holder() {
    // The receiver's shape says nothing of its prototype: A2.prototype has A.prototype's shape,
    // so the walk passes on the swapped receiver and must read A2's method, as the VM does.
    let source = "class A { m() { return 1; } } class A2 { m() { return 2; } }\n\
         function f(o) { return o.m(); } const a = new A(), b = new A();\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = t + f(a); t";
    let expected = run(Tier::Bytecode, source);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, source), expected);
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "Object.setPrototypeOf(b, A2.prototype); let u = 0; \
             for (let k = 0; k < 100; k++) u = u + f(a) * 10 + f(b); u"
        ),
        "1200"
    );
    assert!(exits().is_empty(), "the same shape: no exit");
}

#[test]
fn a_shadowed_or_swapped_chain_exits_and_converges_on_the_generic_access() {
    // A shadowing level between changes that level's shape: the hop exits WrongShape until the
    // discard, then the recompile takes the effect (the way is flagged, its ways did not grow).
    let warm = "class A { m() { return 1; } } class B extends A {}\n\
         function f(o) { return o.m(); } const b = new B();\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + f(b); t";
    let chunk = converges(
        "f",
        warm,
        "B.prototype.m = function () { return 2; }; \
         let u = 0; for (let k = 0; k < 600; k++) u = u + f(b); u",
        "f(b) + f(new B())",
        ExitReason::WrongShape,
    );
    let _ = chunk;
}

/// The WrongShape sites keyed on `chunk` and on its first stage, with each site's byte (the most
/// receiver shapes a discarded check there expected).
fn shape_sites(chunk: &Chunk) -> Vec<(u32, u8)> {
    let mut found = Vec::new();
    let first = profile::first_stage(chunk);
    for of in [chunk, first] {
        if let Some(profile) = ChunkProfile::get(of) {
            for pc in 0..of.jit_ops().len() as u32 {
                profile.sites_at(pc, |reason, _, byte| {
                    if reason == ExitReason::WrongShape as u8 {
                        found.push((pc, byte));
                    }
                });
            }
        }
    }
    found
}

#[test]
fn a_hops_exit_keeps_its_accesss_receiver_count_in_the_exit_profile() {
    // Two receiver shapes share the walk; one receiver's prototype is swapped for one of another
    // shape, so the hop's check exits: the site keeps 2, the access's count, as the receiver's
    // check would (`Rules::stale_shapes` compares the site's ways with it).
    let warm = "function P() {} P.prototype.m = function () { return 1; };\n\
         function Q() {} Q.prototype.q = 0; Q.prototype.m = function () { return 2; };\n\
         function f(o) { return o.m(); }\n\
         var x = new P(), y = new P(); y.extra = 1;\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + f(x) + f(y); t";
    let swap = "Object.setPrototypeOf(x, Q.prototype); let u = 0; \
         for (let k = 0; k < 150; k++) u = u + f(x); u";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [eval(&mut vm, warm), eval(&mut vm, swap)]
    };
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, warm), expected[0]);
    assert!(walks(&mut engine, "f"), "f walks the chain");
    let chunk = tier_chunk(&mut engine, "f").expect("f's tier chunk");
    exits();
    assert_eq!(eval(&mut engine, swap), expected[1]);
    assert!(
        exits()
            .iter()
            .any(|&(_, why)| why == ExitReason::WrongShape),
        "the hop exited"
    );
    let sites = shape_sites(&chunk);
    assert!(
        !sites.is_empty() && sites.iter().all(|&(_, byte)| byte == 2),
        "{sites:?}"
    );
}

/// A string's methods (P3 stage 3 step L2's string form): `code` a String.prototype method and
/// its call, `has` one of Object.prototype's through String.prototype, `two` two loads around a
/// call, `any` a site that meets a String wrapper and an object besides strings; the driver
/// redefines a method, adds and deletes String.prototype's properties, shadows Object.prototype's
/// method on String.prototype, freezes String.prototype and swaps its prototype.
pub(super) const STRINGS: &str = "function code(s, i) { return s.charCodeAt(i); }\n\
     function has(s) { return s.hasOwnProperty('length') ? 1 : 0; }\n\
     function two(s, f) { var a = s.charCodeAt(0); f(); return a * 1000 + s.charCodeAt(1); }\n\
     function any(s) { return s.indexOf('b'); }\n\
     function noop() {}\n\
     const strs = ['abc', 'bcd', 'x\\u00e9y', 'longer string'];\n\
     const keep = String.prototype.charCodeAt, keepHas = Object.prototype.hasOwnProperty;\n\
     let t = 0;\n\
     for (let k = 0; k < 3000; k++) {\n\
       const s = strs[k & 3];\n\
       if (k === 600) String.prototype.charCodeAt = function (i) { return 7 + i; };\n\
       if (k === 800) String.prototype.charCodeAt = keep;\n\
       if (k === 1000) String.prototype.extra = 1;\n\
       if (k === 1200) delete String.prototype.extra;\n\
       if (k === 1400) String.prototype.hasOwnProperty = function () { return false; };\n\
       if (k === 1600) delete String.prototype.hasOwnProperty;\n\
       if (k === 1800) Object.prototype.hasOwnProperty = function () { return true; };\n\
       if (k === 2000) Object.prototype.hasOwnProperty = keepHas;\n\
       if (k === 2200) Object.setPrototypeOf(String.prototype, { hasOwnProperty() { return true; } });\n\
       if (k === 2400) Object.setPrototypeOf(String.prototype, Object.prototype);\n\
       if (k === 2600) Object.freeze(String.prototype);\n\
       const between = k === 700 ? function () { String.prototype.charCodeAt = function () { return 3; }; } \
         : k === 1500 ? function () { String.prototype.more = 2; } : noop;\n\
       let r;\n\
       try { r = two(s, between); } catch (e) { r = -9; }\n\
       const odd = k % 50 === 0 ? new String('abc') : k % 50 === 1 ? { indexOf() { return 99; } } : s;\n\
       t = (t * 31 + code(s, k & 1) + has(s) + r + any(odd)) % 1000003;\n\
     }\n\
     t + ':' + code('zz', 1) + ':' + has('q') + ':' + any('abc')";

#[test]
fn a_strings_loads_have_the_vms_results_under_every_mutation_of_string_prototype() {
    let expected = run(Tier::Bytecode, STRINGS);
    let mut compiles = 0;
    for (label, switches) in configurations() {
        compiled();
        exits();
        assert_eq!(
            with_switches(switches, STRINGS),
            expected,
            "{label} {switches:?}"
        );
        compiles += compiled().len();
    }
    assert!(compiles > 0, "the tier compiled the functions");
}

#[test]
fn the_string_programs_loads_take_string_prototype_before_its_mutations() {
    let warm = STRINGS.replace("k < 3000", "k < 400");
    let expected = run(Tier::Bytecode, &warm);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &warm), expected);
    for name in ["code", "has", "two"] {
        let chunk = tier_chunk(&mut engine, name).unwrap_or_else(|| panic!("{name}'s tier chunk"));
        assert!(
            graph_now(&chunk)
                .nodes
                .iter()
                .any(|node| matches!(node.op, super::super::ir::NodeOp::CheckStringProto { .. })),
            "{name} takes String.prototype"
        );
    }
}

#[test]
fn a_this_read_after_a_call_that_swaps_its_prototype_reads_the_new_holder() {
    // `this` is one node through the whole function, so its facts would cross a call if the
    // link facts did not end at effects: the call (a native's, always an effect) swaps the
    // receiver's prototype for one of the same shape and another value, and the second read
    // must walk again.
    let source = "function tm(q) { var a = this.p; Object.setPrototypeOf(this, q); \
         return a * 10 + this.p; }\n\
         function P() {} P.prototype.p = 1; P.prototype.t = tm;\n\
         function Q() {} Q.prototype.p = 2; Q.prototype.t = tm;\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = t + new P().t(P.prototype); t";
    let after = "let u = 0; for (let k = 0; k < 50; k++) u = u + new P().t(Q.prototype); u";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [eval(&mut vm, source), eval(&mut vm, after)]
    };
    assert_eq!(expected[1], "600", "the VM reads Q's p after the swap");
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, source), expected[0]);
    assert!(walks(&mut engine, "tm"), "tm walks the chain from `this`");
    exits();
    assert_eq!(eval(&mut engine, after), expected[1]);
    assert!(
        exits().is_empty(),
        "the same shape: the walk passes, no exit"
    );
}

#[test]
fn a_prototype_set_to_null_exits_at_the_walk_and_converges() {
    // A receiver of the way's shape whose [[Prototype]] is null: the hop's null test exits
    // (the VM reads undefined) until the discard; the recompile takes the effect.
    let warm = "function P() { this.v = 1; } P.prototype.p = 7;\n\
         function f(o) { return o.p; } const a = new P();\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + f(a); t";
    converges(
        "f",
        warm,
        "const n = new P(); Object.setPrototypeOf(n, null); \
         let u = 0; for (let k = 0; k < 600; k++) u = u + (f(n) === undefined ? 1 : 0); u",
        "f(a) + ':' + f(n)",
        ExitReason::WrongShape,
    );
}

#[test]
fn a_proxy_in_the_chain_exits_at_the_walk_and_converges() {
    // A Proxy as a receiver's prototype is no ordinary IC_PLAIN object: the hop exits until the
    // discard (the VM runs the trap), and the recompile takes the effect.
    let warm = "function P() { this.v = 1; } P.prototype.p = 7;\n\
         function f(o) { return o.p; } const a = new P();\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + f(a); t";
    converges(
        "f",
        warm,
        "const x = new P(); Object.setPrototypeOf(x, new Proxy(P.prototype, \
         { get(t, k, r) { return k === 'p' ? 9 : Reflect.get(t, k, r); } })); \
         let u = 0; for (let k = 0; k < 600; k++) u = u + f(x); u",
        "f(a) + ':' + f(x)",
        ExitReason::WrongShape,
    );
}

/// Polymorphic sites (P3 stage 3 step L2's dispatch on the receiver's shape): `call` over three
/// classes' methods (A's and B's prototypes share a shape, so two walks), `own` over three own slots, `mixed` over an own slot and a
/// walk, `two` two loads around a call; the driver redefines, deletes and shadows methods,
/// swaps a receiver's prototype for another arm's, freezes a receiver and a prototype, and adds a
/// fifth receiver shape late.
pub(super) const POLY: &str = "class A { constructor() { this.a = 1; } m() { return 1; } }\n\
     class B extends A { constructor() { super(); this.b = 2; } m() { return 2; } }\n\
     class C { constructor() { this.c = 3; this.d = 4; } n() { return 0; } m() { return 3; } }\n\
     function call(o) { return o.m(); }\n\
     function own(o) { return o.x; }\n\
     function mixed(o) { return o.m; }\n\
     function two(o, f) { var a = o.m(); f(o); return a * 10 + o.m(); }\n\
     function noop(o) {}\n\
     function h(x) { return typeof x === 'number' ? x : typeof x === 'function' ? 50 + x.length : -1; }\n\
     const rs = [new A(), new B(), new C(), new A(), new C()];\n\
     const ps = [{x: 1}, {y: 2, x: 3}, {z: 4, w: 5, x: 6}];\n\
     const ms = [new A(), {m: 7}];\n\
     let t = 0;\n\
     for (let k = 0; k < 3000; k++) {\n\
       if (k === 500) A.prototype.m = function () { return 11; };\n\
       if (k === 700) delete B.prototype.m;\n\
       if (k === 900) rs[3].m = function () { return 12; };\n\
       if (k === 1100) Object.setPrototypeOf(rs[0], C.prototype);\n\
       if (k === 1300) { Object.freeze(rs[4]); Object.freeze(C.prototype); }\n\
       if (k === 1500) ps[1].x = 30;\n\
       if (k === 1700) delete ps[2].z;\n\
       if (k === 1900) ms[1].m = function (q) { return q; };\n\
       if (k === 2100) rs.push({q: 1, m() { return 13; }}, {r: 2, m() { return 14; }});\n\
       const o = rs[k % rs.length];\n\
       const f = k === 600 ? function (x) { C.prototype.m = function () { return 15; }; } \
         : k === 1200 ? function (x) { Object.setPrototypeOf(x, A.prototype); } : noop;\n\
       let r;\n\
       try { r = call(o) + two(o, f); } catch (e) { r = -9; }\n\
       t = (t * 31 + r + h(own(ps[k % 3])) + h(mixed(ms[k & 1]))) % 1000003;\n\
     }\n\
     t + ':' + call(rs[1]) + ':' + own(ps[1]) + ':' + h(mixed(ms[0]))";

#[test]
fn polymorphic_loads_have_the_vms_results_under_every_mutation() {
    let expected = run(Tier::Bytecode, POLY);
    let mut compiles = 0;
    for (label, switches) in configurations() {
        compiled();
        exits();
        assert_eq!(
            with_switches(switches, POLY),
            expected,
            "{label} {switches:?}"
        );
        compiles += compiled().len();
    }
    assert!(compiles > 0, "the tier compiled the functions");
}

#[test]
fn the_polymorphic_programs_loads_dispatch_before_its_mutations() {
    let warm = POLY.replace("k < 3000", "k < 400");
    let expected = run(Tier::Bytecode, &warm);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &warm), expected);
    for name in ["call", "own", "mixed", "two"] {
        let chunk = tier_chunk(&mut engine, name).unwrap_or_else(|| panic!("{name}'s tier chunk"));
        assert!(
            graph_now(&chunk)
                .nodes
                .iter()
                .any(|node| matches!(node.op, super::super::ir::NodeOp::ShapeIs(_))),
            "{name} dispatches on the receiver's shape"
        );
    }
}

#[test]
fn a_polymorphic_site_settles_within_its_ways_and_then_on_the_generic_access() {
    // Three receiver shapes dispatch with no exit; a fourth fills a way, its first calls exit
    // until the discard and the recompile dispatches on four; a fifth makes the site generic
    // (the cache's mark), whose recompile takes the effect: a few discards, the VM's results.
    let source = "function S(n) { this['f' + n] = n; this.x = n; }\n\
         function get(o) { return o.x; }\n\
         const os = [new S(1), new S(2), new S(3)];\n\
         let t = 0; for (let k = 0; k < 2000; k++) t = t + get(os[k % 3]);\n\
         os.push(new S(4)); for (let k = 0; k < 3000; k++) t = t + get(os[k % 4]);\n\
         os.push(new S(5)); for (let k = 0; k < 3000; k++) t = t + get(os[k % 5]); t";
    let expected = run(Tier::Bytecode, source);
    eager().set();
    let mut engine = jit_engine();
    exits();
    assert_eq!(eval(&mut engine, source), expected);
    let chunk = tier_chunk(&mut engine, "get").expect("get's tier chunk");
    assert!(discards(&chunk) <= 3, "{} discards", discards(&chunk));
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "let u = 0; for (let k = 0; k < 500; k++) u = u + get(os[k % 5]); u"
        ),
        "1500"
    );
    assert!(exits().is_empty(), "settled");
}

#[test]
fn an_object_at_a_strings_site_exits_at_the_tag_check_and_converges() {
    // A string's method site meets an object of its own method: the String tag check exits (the
    // VM calls the object's method) until the discard; the recompile takes the effect.
    let warm = "function f(s) { return s.charCodeAt(0); }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + f('abc'); t";
    converges(
        "f",
        warm,
        "const odd = { charCodeAt() { return 99; } }; \
         let u = 0; for (let k = 0; k < 600; k++) u = u + f(odd); u",
        "f('b') + ':' + f(odd)",
        ExitReason::WrongShape,
    );
}
