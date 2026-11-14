//! P3 stage 3 step L3 end to end: stores that create their property against the VM, with every
//! way the brief names that a creation can go wrong: constructors of several fields, a creation
//! on an object that is another object's prototype between two loads through that object (the
//! walk's facts must end: the load after sees the new property's shadowing), two names for one
//! object across a creation, a setter and a read-only property appearing on the prototype (the
//! creation epoch moves), frozen, sealed and non-extensible receivers, a receiver in dictionary
//! mode, records that grow at each capacity step, strict code's throws.

use super::*;

/// `P` three fields; `Q` two, its receiver frozen, sealed or made non-extensible first by
/// its flag; `S` a strict constructor whose store into a read-only inherited property throws;
/// `walk` a load through `c`'s chain around a creation on `p`, `c`'s prototype, and `walk2`
/// around one on `q`, of `p`'s shape and sometimes `p` (the second load's way expects `p`'s old
/// shape); `alias` a load of
/// `a` around a creation on `b`, sometimes `a` itself; `fill` seven fields on an object literal
/// (a record grows past the cell's three slots), `wide` twelve (past the commit's 8); `dict`
/// a creation on an object in dictionary mode; `N` a chain ending at a user object whose null
/// [[Prototype]] gains a setter's holder late (the chain's end check); `addv` a young value into
/// old receivers (the card, under GC stress).
pub(super) const CREATE: &str = "function P(a, b, c) { this.a = a; this.b = b; this.c = c; }\n\
     function Q(x, f) { if (f === 1) Object.freeze(this); else if (f === 2) Object.seal(this); \
       else if (f === 3) Object.preventExtensions(this); this.x = x; this.y = x + 1; }\n\
     function S(v) { 'use strict'; this.r = v; this.w = v * 2; }\n\
     const base = { m() { return 1; } };\n\
     function walk(c, p, v) { const a = c.m(); p.m = v; return a * 1000 + c.m(); }\n\
     function walk2(c, p, q, v) { const a = c.m(); q.m = v; return a * 1000 + c.m(); }\n\
     function alias(a, b, v) { const x = a.k; b.z = v; return x * 100 + a.k + (a.z === undefined ? 0 : a.z); }\n\
     function fill(o, v) { o.a = v; o.b = v; o.c = v; o.d = v; o.e = v; o.f = v; o.g = v; return o; }\n\
     function wide(o, v) { o.a = v; o.b = v; o.c = v; o.d = v; o.e = v; o.f = v; o.g = v; o.h = v; \
       o.i = v; o.j = v; o.k = v; o.l = v; return o; }\n\
     function dict(o, v) { o.q = v; return o.q; }\n\
     const nb = Object.create(null); nb.q = 1;\n\
     function N(v) { this.s = v; } N.prototype = Object.create(nb);\n\
     function Old() { this.a = 1; } const olds = []; for (let i = 0; i < 64; i++) olds.push(new Old());\n\
     function addv(o, i) { o.v = { i: i, s: 'v' + i }; }\n\
     function h(x) { return typeof x === 'number' ? x % 1000003 : x === undefined ? -1 : 7; }\n\
     function sum(o) { let s = 0; for (const k in o) s = (s * 7 + h(o[k])) % 1000003; return s; }\n\
     let t = 0;\n\
     function step(k) {\n\
       const o = new P(k, k + 1, k + 2);\n\
       t = (t * 31 + h(o.a) + h(o.b) + h(o.c) + sum(o)) % 1000003;\n\
       let q;\n\
       try { q = new Q(k, k > 400 ? k % 4 : 0); t = (t * 31 + sum(q) + (Object.isFrozen(q) ? 5 : 0)) % 1000003; } \
         catch (e) { t = (t * 31 + 11) % 1000003; }\n\
       try { const s = new S(k); t = (t * 31 + sum(s)) % 1000003; } catch (e) { t = (t * 31 + 13) % 1000003; }\n\
       const p = Object.create(base); p.k = k;\n\
       const c = Object.create(p);\n\
       t = (t * 31 + h(walk(c, p, k % 2 ? function () { return 2; } : function () { return 3; }))) % 1000003;\n\
       const p2 = Object.create(base), c2 = Object.create(p2);\n\
       t = (t * 31 + h(walk2(c2, p2, k > 500 && k % 7 === 0 ? p2 : Object.create(base), function () { return 4; }))) % 1000003;\n\
       const a = { k: k }, b = k % 3 ? { k: 0 } : a;\n\
       t = (t * 31 + h(alias(a, b, k))) % 1000003;\n\
       t = (t * 31 + sum(fill({}, k)) + sum(wide({}, k))) % 1000003;\n\
       const d = { u: 1, v: 2, w: 3 }; if (k > 700) { delete d.v; }\n\
       t = (t * 31 + h(dict(d, k))) % 1000003;\n\
       const nn = new N(k); t = (t * 31 + h(nn.s) + h(nn.ss)) % 1000003;\n\
       if (k >= 1000 && k < 1064) addv(olds[k - 1000], k);\n\
     }\n\
     for (let k = 0; k < 1500; k++) {\n\
       if (k === 600) Object.defineProperty(P.prototype, 'b', \
         { set(v) { this.bb = v * 3; }, get() { return this.bb; }, configurable: true });\n\
       if (k === 800) Object.defineProperty(S.prototype, 'w', { value: 9, writable: false });\n\
       if (k === 1000) { delete P.prototype.b; }\n\
       if (k === 1200) Object.defineProperty(Q.prototype, 'y', { value: 4, writable: false });\n\
       if (k === 1300) Object.setPrototypeOf(nb, { set s(v) { this.ss = v * 5; } });\n\
       step(k);\n\
     }\n\
     for (let i = 0; i < 64; i++) t = (t * 31 + olds[i].v.i + olds[i].v.s.length) % 1000003;\n\
     t";

/// The gates' configurations, at splice and at TRUST_JS_OPT_INLINE=graph (chain_tier_tests'
/// set: on, eager, forced exits every 7 and 100, an exit limit of 1).
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
fn creations_have_the_vms_results_under_every_change_of_their_proof() {
    let expected = run(Tier::Bytecode, CREATE);
    let mut compiles = 0;
    for (label, switches) in configurations() {
        compiled();
        exits();
        assert_eq!(
            with_switches(switches, CREATE),
            expected,
            "{label} {switches:?}"
        );
        compiles += compiled().len();
    }
    assert!(compiles > 0, "the tier compiled the functions");
}

#[test]
fn a_creation_against_its_objects_predicted_layout_has_the_vms_results() {
    // `R`'s instances are predicted to take `a` then `b` (its first runs' order); one in eight
    // takes `b` first, against the prediction: the commit must take the landing shape's layout,
    // not the object's predicted one, or its keys (for-in, Object.keys) name `a` where `b` is.
    let source = "function R(f, v) { if (f) { this.a = v; this.b = v + 1; } \
           else { this.b = v + 2; this.a = v + 3; } }\n\
         let t = 0;\n\
         for (let k = 0; k < 1500; k++) {\n\
           const o = new R(k % 8 !== 5, k);\n\
           const keys = Object.keys(o).join();\n\
           t = (t * 31 + (keys === 'a,b' ? 1 : keys === 'b,a' ? 2 : 3) + o.a + o.b) % 1000003;\n\
           for (const key in o) t = (t * 31 + key.charCodeAt(0) + o[key]) % 1000003;\n\
         }\n\
         t";
    let expected = run(Tier::Bytecode, source);
    for (label, switches) in configurations() {
        assert_eq!(
            with_switches(switches, source),
            expected,
            "{label} {switches:?}"
        );
    }
}

#[test]
fn a_creations_slow_path_stores_a_value_from_its_spill_slot() {
    // `W`'s creations past its 8th key take the slow path's call, around which the caller-saved
    // values live there (a, b and c) are saved on the stack; the stored values, 24 of them live
    // across `g()`'s generic node, are in spill slots, whose offsets are the stack pointer's: the
    // value is read before the save moves it (TypeScript's getParameterList stored another
    // word when it was read after).
    let n = 24;
    let values: Vec<String> = (0..n).map(|i| format!("v{i} = new O(k + {i})")).collect();
    let stores: String = (0..n).map(|i| format!("this.f{i} = v{i}; ")).collect();
    let reads: Vec<String> = (0..n).map(|i| format!("w.f{i}.i * {}", i + 1)).collect();
    let source = format!(
        "function g() {{ return 1; }}\n\
         function O(i) {{ this.i = i; }}\n\
         function W(k, o) {{ var {}; g(); var a = o.x + k, b = o.y * k, c = o.x - k; \
           {stores}this.s = a + b + c; }}\n\
         let t = 0; const o = {{ x: 3, y: 5 }};\n\
         for (let k = 0; k < 3000; k++) {{ const w = new W(k, o); \
           t = (t * 31 + {} + w.s) % 1000003; }}\n\
         t",
        values.join(", "),
        reads.join(" + ")
    );
    let expected = run(Tier::Bytecode, &source);
    for (label, switches) in configurations() {
        assert_eq!(
            with_switches(switches, &source),
            expected,
            "{label} {switches:?}"
        );
    }
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &source), expected);
    assert_eq!(
        creations(&mut engine, "W"),
        n + 1,
        "W's creations, the 9th on by the slow path, are the tier's"
    );
}

#[test]
fn a_creations_commit_writes_the_slot_before_the_shape() {
    // The commit's order (design note 5): the slot's word, its card, the shape and the count
    // last, with no call or poll between, as the template orders them, so that the collector,
    // which runs only at safepoints, never sees the shape without the slot. No program can see
    // the order, so the code is read: each slot store, `str x, [x30]`, is followed by the shape
    // store, `str w, [x17, #shape]`, and preceded by none, and the card mark's byte store,
    // `strb w9, [x9]` (jit.rs emit_card_mark), lies between them (a young value stored into an
    // old receiver needs it; the GC-stress tests do not reach the commit in line for that case).
    let source = "function P(x) { this.a = x; this.b = x + 1; this.c = x + 2; }\n\
         let t = 0; for (let k = 0; k < 2000; k++) t = (t + new P(k).c) % 1000003; t";
    let expected = run(Tier::Bytecode, source);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, source), expected);
    assert_eq!(creations(&mut engine, "P"), 3);
    let chunk = tier_chunk(&mut engine, "P").expect("P's tier chunk");
    let code = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&code), "P runs the tier's code");
    // SAFETY: the installed code's mapped bytes, alive while `code` is.
    let bytes = unsafe { std::slice::from_raw_parts(code.mem_ptr(), code.code_bytes()) };
    let (chunks, _) = bytes.as_chunks::<4>();
    let words: Vec<u32> = chunks.iter().map(|&w| u32::from_le_bytes(w)).collect();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let shape_at = (layout.obj_props + layout.props_shape) as u32;
    let shape_store = |w: u32| w & 0xffff_ffe0 == 0xb900_0000 | ((shape_at / 4) << 10) | (17 << 5);
    let slot_store = |w: u32| w & 0xffff_ffe0 == 0xf900_03c0;
    let commits: Vec<usize> = (0..words.len())
        .filter(|&k| {
            slot_store(words[k])
                && words[k + 1..(k + 16).min(words.len())]
                    .iter()
                    .any(|&w| shape_store(w))
        })
        .collect();
    assert_eq!(commits.len(), 3, "P's three commits");
    for &k in &commits {
        assert!(
            !words[k.saturating_sub(8)..k]
                .iter()
                .any(|&w| shape_store(w)),
            "a shape store before the slot's at word {k}"
        );
        let shape = (k + 1..words.len())
            .find(|&j| shape_store(words[j]))
            .expect("the shape store");
        assert!(
            words[k + 1..shape].contains(&0x3900_0129),
            "the card mark between the slot's store at word {k} and the shape's"
        );
    }
}

/// The creations of the tier's code of `name` (its graph as a compile builds it now, its
/// snapshot taken in the engine's realm, as the driver's): of its tier chunk, else of its first
/// stage (the chunk a compile of a function with nothing to splice takes).
fn creations(engine: &mut Engine, name: &str) -> usize {
    let chunk = tier_chunk(engine, name).unwrap_or_else(|| first_stage(engine, name));
    let snapshot = super::super::snapshot::Snapshot::take_in(&chunk, Some(&engine.interp));
    super::super::build::build(&chunk, &snapshot)
        .expect("built")
        .graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, super::super::ir::NodeOp::StoreNew { .. }))
        .count()
}

/// Whether the tier's code of `name` creates a property.
fn creates(engine: &mut Engine, name: &str) -> bool {
    creations(engine, name) > 0
}

#[test]
fn the_programs_creations_are_the_tiers_nodes_before_its_changes() {
    // The suite above tests the creations: before the first change the tier's code of each
    // function creates its properties itself (wide's past the 8th and dict's stay the VM's).
    let warm = CREATE.replace("k < 1500", "k < 300");
    let expected = run(Tier::Bytecode, &warm);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &warm), expected);
    for name in ["P", "Q", "S", "walk", "alias", "fill", "wide"] {
        assert!(
            creates(&mut engine, name),
            "{name} creates in the tier's code"
        );
    }
}

#[test]
fn a_constructor_runs_its_creations_in_the_tiers_code_without_an_exit() {
    let source = "function P(x, y) { this.x = x; this.y = y; this.z = x * y; }\n\
         function W(v) { this.a = v; this.b = v; this.c = v; this.d = v; this.e = v; this.f = v; \
           this.g = v; this.h = v; this.i = v; this.j = v; this.k = v; this.l = v; }\n\
         var C = class { constructor(v) { this.v = v; this.w = [v]; } };\n\
         function f(k) { const p = new P(k, k + 1), c = new C(k), w = new W(k); \
           return p.x + p.y + p.z + c.v + c.w[0] + w.l; }\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = (t + f(k)) % 1000003; t";
    // `var`s: a script's new global lexical moves the global record's generation, which a
    // global lexical's read checks (WrongName).
    let more = "var u = 0; for (var k = 0; k < 500; k++) u = (u + f(k)) % 1000003; u";
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
    let taken = exits();
    assert!(taken.is_empty(), "no exit: {taken:?}");
    assert_eq!(
        creations(&mut engine, "P"),
        3,
        "P creates in the tier's code"
    );
    // Had the commit's failing conditions exited, the exits would have flagged the 9th to 12th
    // creations and the recompile taken them as effects, with no exit left in `more`.
    assert_eq!(
        creations(&mut engine, "W"),
        12,
        "W creates its twelve fields in the tier's code, the 9th to 12th by the slow path"
    );
    let chunk = tier_chunk(&mut engine, "P").expect("P's tier chunk");
    let optimized = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&optimized), "P runs the tier's code");
}

#[test]
fn a_setter_appearing_on_the_prototype_exits_and_converges_on_the_effect() {
    // The prototype's new accessor moves the creation epoch: the creation's check exits
    // WrongShape until the discard, after which the recompile takes the effect (flagged, its
    // ways did not grow), and the setter runs.
    let warm = "function P(x) { this.x = x; this.y = x + 1; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + new P(k).y; t";
    converges(
        "P",
        warm,
        "Object.defineProperty(P.prototype, 'y', { set(v) { this.yy = v * 2; }, \
           get() { return this.yy; }, configurable: true }); \
         let u = 0; for (let k = 0; k < 600; k++) u = u + new P(k).y; u",
        "new P(3).y + new P(4).x",
        ExitReason::WrongShape,
    );
}

#[test]
fn non_extensible_receivers_exit_and_converge_on_the_effect() {
    // A receiver made non-extensible fails the creation's checks (its shape demoted, its flag
    // clear), which the snapshot's chain cannot see: the exits flag the site, and the recompile
    // takes the effect (its WrongShape flag), whose store fails silently in sloppy code.
    let warm = "function P(x) { this.x = x; return this; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + new P(k).x; t";
    converges(
        "P",
        warm,
        "let u = 0; for (let k = 0; k < 600; k++) \
           u = u + (P.call(Object.preventExtensions({}), k).x === undefined ? 1 : 0); u",
        "(P.call(Object.preventExtensions({}), 3).x === undefined) + ':' + new P(4).x",
        ExitReason::WrongShape,
    );
}

#[test]
fn a_setter_appearing_before_the_first_compile_keeps_the_creation_the_effect() {
    // The creation's way was filled before the prototype took a setter of the name, and the tier
    // compiles after: the snapshot's walk of the chain finds the setter and records no chain, so
    // the creation stays the effect, which calls the setter (no exit ever teaches it). A graph
    // built between the setter's definition and the site's next run reads the way alone, so
    // only the walk refuses it.
    let warm = "function P(x) { this.x = x; return this; }\n\
         let t = 0; for (let k = 0; k < 300; k++) t = t + new P(k).x; t";
    let setter = "Object.defineProperty(P.prototype, 'x', { set(v) { this.y = v * 2; }, \
           configurable: true }); 0";
    let then = "let u = 0; for (let k = 0; k < 600; k++) u = u + new P(k).y; u";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [
            eval(&mut vm, warm),
            eval(&mut vm, setter),
            eval(&mut vm, then),
        ]
    };
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, warm), expected[0]);
    assert!(
        creates(&mut engine, "P"),
        "the way alone, no setter: a creation"
    );
    assert_eq!(eval(&mut engine, setter), expected[1]);
    assert!(
        !creates(&mut engine, "P"),
        "the walk finds the setter before the site runs"
    );
    eager().set();
    assert_eq!(eval(&mut engine, then), expected[2]);
    assert!(!creates(&mut engine, "P"), "the setter keeps the effect");
}
