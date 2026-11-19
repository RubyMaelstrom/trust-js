//! The union rule (F4 part 1's miscompile, which the coordinator's programs g01, g03 and g08 found
//! at TRUST_JS_OPT_INLINE=graph): what a loop that keeps heap facts may write is the union over
//! its own operations and every frame the build may inline into it, at every depth
//! (`build::Writes`, `Rules::writes_at`, `inline::Tree::writes`), and the verifier checks each
//! built loop against it (`verify::kept_heap`). The first form's tests all stored in the compiled
//! function itself, where the prepass saw the store; these store from inlined frames: through an
//! alias of the global object, a method's `this`, a `.call` receiver, updates, a global typed
//! array replaced on a rare path, one way of a polymorphic site, two levels down and from a
//! by-literal frame, each against the tier off.

use super::*;

/// `source`'s completion with the tier off, then at graph inlining: not eager, eager, and eager
/// with an exit forced every seventh check, each the same; the ways the first graph run inlined.
fn agrees_at_graph(source: &str) -> usize {
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, source);
    assert!(!expected.starts_with("throw"), "{expected}");
    let mut inlined = 0;
    for (eager, exit_every) in [(false, 0), (true, 0), (true, 7)] {
        inlines();
        TestSwitches {
            on: true,
            eager,
            exit_every,
            asynchronous: false,
            inline_graph: Some(true),
            ..Default::default()
        }
        .set();
        let mut engine = jit_engine();
        assert_eq!(
            eval(&mut engine, source),
            expected,
            "eager {eager}, an exit every {exit_every}:\n{source}"
        );
        let ways = inlines()
            .iter()
            .filter(|note| {
                matches!(
                    note.outcome,
                    InlineOutcome::Inlined | InlineOutcome::ByLiteral
                )
            })
            .count();
        if !eager {
            inlined = ways;
        }
    }
    inlined
}

#[test]
fn an_inlined_store_through_an_alias_of_the_global_object_is_read_in_the_next_iteration() {
    // g01: the quiet loop reads G and calls setG, inlined, whose `o.G = v` stores it through an
    // alias of the global object.
    let inlined = agrees_at_graph(
        "var G = 1;
         function setG(o, v) { o.G = v; }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; setG(g, i); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40);
         r",
    );
    assert!(inlined > 0, "setG inlined");
}

#[test]
fn an_inlined_methods_store_through_this_and_a_call_receiver_is_read_in_the_next_iteration() {
    // g03: a method's `this.G = v` through g.bump(i), and bumpThis.call(globalThis, v).
    let inlined = agrees_at_graph(
        "var G = 1, H = 5;
         var obj = { bump: function (v) { this.G = v; } };
         function bumpThis(v) { this.G = v; }
         function W(n) { var s = 0; var g = globalThis; g.bump = obj.bump; for (var i = 0; i < n; i++) { s += G + H; g.bump(i); } return s; }
         function W2(n) { var s = 0; for (var i = 0; i < n; i++) { s += G; bumpThis.call(globalThis, i + 1); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40) + W2(40);
         r",
    );
    assert!(inlined > 0, "the methods inlined");
}

#[test]
fn an_inlined_update_of_a_global_through_an_alias_is_read_in_the_next_iteration() {
    // An update (`o.G++`) and a compound store (`o.G += v`) of the global the loop reads.
    agrees_at_graph(
        "var G = 1, F = 0.5;
         function inc(o) { o.G++; }
         function addF(o, v) { o.F += v; }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G + F; inc(g); addF(g, 0.25); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r = (r + W(40)) % 1000003;
         r",
    );
}

#[test]
fn a_global_typed_array_replaced_by_an_inlined_callee_on_a_rare_path_is_read_anew() {
    // g08's third loop: the loop reads A[1], a global typed array whose checks its entry makes
    // once, and on one iteration an inlined callee stores another array into A through an alias.
    let inlined = agrees_at_graph(
        "var A = new Int32Array(8), B = new Int32Array(8); B[1] = 9; A[1] = 2;
         function setA(o) { o.A = B; }
         function W3(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += A[1]; if (i === 5) setA(g); } A = new Int32Array(8); A[1] = 2; return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W3(40);
         r",
    );
    assert!(inlined > 0, "setA inlined");
}

#[test]
fn a_store_by_one_way_of_a_polymorphic_site_is_read_in_the_next_iteration() {
    // The loop's call site has two ways, one storing G and the other not; the storing callee is
    // the one a quarter of W's runs take.
    agrees_at_graph(
        "var G = 1;
         function keep(o, v) { return v; }
         function store(o, v) { o.G = v; return v; }
         function W(n, f) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; f(g, i); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40, (k & 3) === 3 ? store : keep);
         r",
    );
}

#[test]
fn stores_two_levels_down_and_from_a_by_literal_frame_are_read_in_the_next_iteration() {
    // A nested way's store (outer calls inner, which stores G) and a by-literal frame's (helper
    // calls the closure W creates, which stores G): today each makes its site's frame generic
    // (its callee calls), so the loop keeps no heap facts; the results hold either way.
    agrees_at_graph(
        "var G = 1;
         function inner(o, v) { o.G = v; }
         function outer(o, v) { inner(o, v + 1); }
         function helper(f, a) { return f(a); }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; outer(g, i); } return s; }
         function W2(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; helper(function (x) { g.G = x; }, i); } return s; }
         function W3(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; [i].forEach(x => { g.G = x; }); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40) + W2(40) + W3(40);
         r",
    );
}

#[test]
fn a_script_level_lexical_and_a_global_stored_by_name_from_a_callee_are_read_anew() {
    // g08's first two loops: no planned callee stores a free name (the planner refuses the
    // operation, bytecode.rs inline_capability), so these calls stay calls; the results hold.
    agrees_at_graph(
        "let L = 3; var G = 2;
         function setL(v) { L = v; }
         function setG(v) { G = v; }
         function W(n) { var s = 0; for (var i = 0; i < n; i++) { s += L; setL(i); } return s; }
         function W2(n) { var s = 0; for (var i = 0; i < n; i++) { s += G; setG(i); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40) + W2(40);
         r",
    );
}

#[test]
fn stores_from_an_inlined_callees_loop_an_inner_loop_and_a_polymorphic_method_are_read_anew() {
    // The lowerings' own programs (f4-impl/union/adv2, each wrong at graph before the union): the
    // callee's own loop stores G; an inner loop's inlined call stores the G its outer loop reads;
    // a callee swaps the global typed array the loop reads with another, through an alias; one
    // way of three of a method site stores G, another H.
    for source in [
        "var G = 1;
         function fill(o, n) { for (var j = 0; j < n; j++) o.G = j; }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; fill(g, i & 3); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40);
         r",
        "var G = 1;
         function setG(o, v) { o.G = v; }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; for (var j = 0; j < 3; j++) { s += j; if (j === 2) setG(g, i + j); } } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r += W(30);
         r",
        "var A = new Float64Array(16), B = new Float64Array(16);
         for (var q = 0; q < 16; q++) { A[q] = q; B[q] = 100 + q; }
         function swo(o, i) { if ((i & 7) === 7) { var t = o.A; o.A = o.B; o.B = t; } }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += A[i & 15]; swo(g, i); } return s; }
         var r = 0; for (var k = 0; k < 3000; k++) r = (r + W(40)) % 1000003;
         r",
        "var G = 1, H = 0;
         var p1 = { m: function (o, v) { return v; } }, p2 = { m: function (o, v) { o.G = v; return v; } }, p3 = { m: function (o, v) { o.H = v; return v; } };
         function W(n, p) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; p.m(g, i); } return s; }
         var ps = [p1, p3, p1, p2];
         var r = 0; for (var k = 0; k < 3000; k++) r += W(40, ps[k & 3]);
         r",
    ] {
        assert!(agrees_at_graph(source) > 0, "inlined: {source}");
    }
}

#[test]
fn a_loop_bound_and_a_typed_array_stored_by_inlined_callees_are_read_anew() {
    // The coordinator's g10 (stepb-review/f4adv): a global typed array an inlined callee replaces
    // in every iteration, a global loop bound one stores on one iteration, and a loop that stores
    // the elements of the global typed array it reads.
    agrees_at_graph(
        "var A = new Int32Array(16), B = new Int32Array(16), n = 16; B[3] = 5; A[3] = 2;
         function pick(o, t) { o.A = t; }
         function len(o, v) { o.n = v; }
         function W(m) { var s = 0; var g = globalThis; for (var i = 0; i < m; i++) { s += A[3]; pick(g, (i & 1) ? A : B); } return s; }
         function W2(m) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += A[i]; if (i === 4) len(g, 8); } n = 16; return s; }
         function W3(m) { var s = 0; for (var i = 0; i < m; i++) { s += A[i & 15]; A[(i + 1) & 15] = i; } return s; }
         var r = 0; for (var k = 0; k < 4000; k++) r += W(40) + W2(40) + W3(40);
         r",
    );
}

/// The location of the global-object property `name` as the compile of `compile` reads it.
fn global_location(compile: &Compile, name: &str) -> u16 {
    let ops = compile.chunk.jit_ops();
    (0..ops.len())
        .find_map(|pc| match (ops[pc], compile.snapshot.name(pc)) {
            (
                Op::LoadName(n, _) | Op::LoadNameForCall(n, _),
                super::super::snapshot::NameView::Global { location, .. },
            ) if compile.chunk.jit_name(n) == name => Some(location),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a global read of {name}\n{}", ops_of(&compile.chunk)))
}

fn ops_of(chunk: &Chunk) -> String {
    ops(chunk)
}

/// The blocks holding a load of the global object's field at `location`, and the first loop
/// header's block.
fn global_field_loads(graph: &Graph, location: u16) -> (Vec<u32>, u32) {
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop") as u32;
    let blocks = graph
        .nodes
        .iter()
        .filter(|node| {
            matches!(node.op, NodeOp::LoadField { location: l, .. } if l == location)
                && matches!(graph.node(node.inputs[0]).op, NodeOp::GlobalObject)
        })
        .map(|node| node.block.0)
        .collect();
    (blocks, header)
}

const ALIASED: &str = "
    var G = 1, H = 2;
    function setG(o, v) { o.G = v; }
    function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G + H; setG(g, i); } return s; }
    for (var k = 0; k < 40; k++) W(20);
";

#[test]
fn a_loop_keeps_no_fact_of_a_global_its_inlined_callee_stores_and_keeps_the_others() {
    // The loop reads G and H; the inlined setG stores G through an alias: G is read in the loop
    // body and not loaded at its entry, H still once at the entry (the union drops only what a
    // frame writes).
    let mut engine = warmed(ALIASED);
    let compile = Compile::new(&mut engine, "W");
    let built = compile.build();
    let graph = &built.graph;
    assert!(
        outcomes(&built)
            .iter()
            .any(|(_, outcome)| *outcome == InlineOutcome::Inlined),
        "setG inlined: {:?}",
        outcomes(&built)
    );
    let (g, header) = global_field_loads(graph, global_location(&compile, "G"));
    let (h, _) = global_field_loads(graph, global_location(&compile, "H"));
    // G read in the loop, and not at its entry: no read of the loop would take that value.
    assert!(
        !g.is_empty() && g.iter().all(|&block| block > header),
        "G read in the loop alone (header {header}): {g:?}\n{}",
        dump(graph)
    );
    assert!(
        !h.is_empty() && h.iter().all(|&block| block < header),
        "H read once, before the header {header}: {h:?}\n{}",
        dump(graph)
    );
    let [kept] = &graph.kept[..] else {
        panic!("one loop kept heap facts: {:?}", graph.kept)
    };
    // H's field kept (and setG's, which the loop reads to call it), G's not.
    assert!(
        kept.fields.contains(&global_location(&compile, "H"))
            && !kept.fields.contains(&global_location(&compile, "G")),
        "{kept:?}"
    );
}

#[test]
fn the_union_takes_nested_ways_and_by_literal_frames_at_every_depth() {
    // outer's frame calls inner, which stores G; helper's frame calls the closure W creates,
    // which stores G: neither way's own chunk stores G, so G's location in each site's union comes
    // from the nested way and the by-literal way (`inline::Tree::writes`).
    let mut engine = warmed(
        "var G = 1;
         function inner(o, v) { o.G = v; }
         function outer(o, v) { inner(o, v + 1); }
         function helper(f, a) { return f(a); }
         function W(n) { var s = 0; var g = globalThis; for (var i = 0; i < n; i++) { s += G; outer(g, i); helper(function (x) { g.G = x; }, i); } return s; }
         for (var k = 0; k < 40; k++) W(20);",
    );
    let compile = Compile::new(&mut engine, "W");
    let location = global_location(&compile, "G");
    let job = compile.job.as_ref().expect("a plan");
    let callees: Vec<super::super::build::Callee<'_>> = job
        .callees
        .iter()
        .zip(&compile.callee_snapshots)
        .map(|((view, _), snapshot)| (view, snapshot))
        .collect();
    let (mut nested, mut literal) = (false, false);
    for site in &job.sites {
        let writes =
            super::super::build::site_writes(&compile.view, &job.sites, &callees, site.cache);
        let way = &site.ways[0];
        // The site's union with only its nested ways, or only its by-literal ways, under its way
        // (helper's frame may hold both for its call of the closure), or neither.
        let with = |nested: bool, literals: bool| {
            super::super::build::site_writes(
                &compile.view,
                &[super::super::build::InlineSite {
                    cache: site.cache,
                    ways: vec![super::super::build::InlineWay {
                        nested: if nested {
                            way.nested.clone()
                        } else {
                            Vec::new()
                        },
                        literals: if literals {
                            way.literals.clone()
                        } else {
                            Vec::new()
                        },
                        ..way.clone()
                    }],
                }],
                &callees,
                site.cache,
            )
        };
        assert!(writes.fields.contains(&location), "{writes:?}");
        if !way.nested.is_empty() && with(true, false).fields.contains(&location) {
            nested = true;
        }
        if !way.literals.is_empty() {
            literal = true;
            let only = with(false, true);
            assert!(
                only.fields.contains(&location),
                "helper's site takes the closure's store through its by-literal way: {only:?}"
            );
        }
        // A way's own chunk stores nothing: the location comes from below it.
        let own = with(false, false);
        assert!(own.fields.is_empty(), "{own:?}");
    }
    assert!(
        nested && literal,
        "a nested way {nested}, a by-literal way {literal}"
    );
}

#[test]
fn the_union_takes_an_inlined_callees_binding_stores() {
    // The planner splices no callee that stores a closure binding (bytecode.rs inline_capability),
    // so the plan is made by hand, as typed_loop_inline_tests' does: bump's store of `c` is in its
    // site's union, its slot, and a nested way's under it too.
    let source = "
        var mg = (function () {
          var c = 0;
          function bump() { c = (c + 1) | 0; }
          function twice() { bump(); }
          function lp(n) { var s = c | 0; for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + c) | 0; twice(); } return s | 0; }
          return { lp: lp, bump: bump, twice: twice };
        })();
        var lp = mg.lp, bmp = mg.bump, tw = mg.twice;
        for (var k = 0; k < 30; k++) lp(8);";
    let mut engine = warmed(source);
    let env = engine.interp.global_env;
    let way = |engine: &mut Engine, name: &str| {
        let (f, _) = function(engine, name);
        let object = engine
            .interp
            .get_var(name, &env)
            .ok()
            .unwrap()
            .as_obj()
            .unwrap();
        crate::bytecode::InlineWay {
            f,
            obj: object,
            check_this: false,
            uses_this: false,
            leaf: true,
            free_names: Vec::new(),
            expected_env: 0,
            expected_genv: crate::interpreter::Env::as_ptr(&env) as usize,
            genv_pin: crate::interpreter::Env::downgrade(&env),
            nested: Default::default(),
        }
    };
    let twice = chunk_of(&mut engine, "tw");
    let (Op::Call(_, inner) | Op::CallWithThis(_, inner)) =
        twice.jit_ops()[pc_of(&twice, is_call) as usize]
    else {
        unreachable!()
    };
    let mut outer = way(&mut engine, "tw");
    outer.leaf = false;
    outer.nested.insert(
        inner,
        crate::bytecode::InlinePlanEntry {
            ways: vec![way(&mut engine, "bmp")],
        },
    );
    let lp = chunk_of(&mut engine, "lp");
    let (Op::Call(_, cache) | Op::CallWithThis(_, cache)) =
        lp.jit_ops()[pc_of(&lp, is_call) as usize]
    else {
        unreachable!()
    };
    let mut plan = crate::fasthash::FastMap::default();
    plan.insert(
        cache,
        crate::bytecode::InlinePlanEntry { ways: vec![outer] },
    );
    let compile = Compile::with(&mut engine, "lp", Request::Plan(plan));
    let job = compile.job.as_ref().expect("the plan");
    let callees: Vec<super::super::build::Callee<'_>> = job
        .callees
        .iter()
        .zip(&compile.callee_snapshots)
        .map(|((view, _), snapshot)| (view, snapshot))
        .collect();
    let [site] = &job.sites[..] else {
        panic!("one site: {:?}", job.sites)
    };
    assert_eq!(site.ways[0].nested.len(), 1, "{site:?}");
    let writes = super::super::build::site_writes(&compile.view, &job.sites, &callees, cache);
    assert_eq!(
        writes.slots.len(),
        1,
        "c's slot, from bump two levels down: {writes:?}"
    );
}

/// A quiet loop that reads a global, G, and a closure binding, c, each read before it too, so
/// that its header keeps G's field and c's value.
const KEPT: &str = "
    var G = 3;
    var W = (function () { var c = 2; return function (n) { var s = (c + G) | 0; for (var i = 0; i < n; i++) s = (s + G + c) | 0; return s; }; })();
    for (var k = 0; k < 40; k++) W(20);
";

#[test]
fn the_verifier_refuses_a_kept_loop_that_writes_what_its_header_kept() {
    let mut engine = warmed(KEPT);
    let compile = Compile::new(&mut engine, "W");
    let built = compile.build();
    let graph = &built.graph;
    let location = global_location(&compile, "G");
    let [kept] = &graph.kept[..] else {
        panic!(
            "one loop kept heap facts: {:?}\n{}",
            graph.kept,
            dump(graph)
        )
    };
    assert_eq!(kept.fields, vec![location], "G's field kept");
    let [slot] = kept.slots[..] else {
        panic!("c's value kept: {kept:?}\n{}", dump(graph))
    };
    let header = super::super::verify::kept_header(graph, kept).expect("its loop header");
    // A block of the loop's body (after the header, branching back to it), and the operands a
    // store takes there: the global object and the entry's load of G.
    let body = graph.blocks[header.index()]
        .preds
        .iter()
        .copied()
        .find(|latch| latch.index() > header.index())
        .expect("the loop's latch");
    let after = nodes(graph, |node| node.op == NodeOp::Return)
        .first()
        .map(|&ret| graph.node(ret).block)
        .expect("the return after the loop");
    let global = nodes(graph, |node| node.op == NodeOp::GlobalObject)[0];
    let loaded = nodes(graph, |node| {
        matches!(node.op, NodeOp::LoadField { location: l, .. } if l == location)
            && node.block.index() < header.index()
    })[0];
    let env = nodes(graph, |node| {
        matches!(node.op, NodeOp::Env | NodeOp::ConstEnv(_))
    })[0];
    let inject = |block: BlockIx, op: NodeOp, inputs: &[NodeIx]| {
        let mut broken = graph.clone();
        let node = broken.new_node(
            block,
            op,
            inputs,
            Rep::None,
            graph.blocks[block.index()].first_pc,
        );
        let body = &mut broken.blocks[block.index()].body;
        body.insert(body.len() - 1, node);
        broken
    };
    let field = NodeOp::StoreField { location, slot: 0 };
    // Through the whole verifier: a store at G's location in the loop.
    let broken = inject(body, field.clone(), &[global, loaded]);
    let frames: Vec<super::super::verify::FrameChunk<'_>> = Vec::new();
    let error = super::super::verify::verify_frames(&broken, (&compile.view, &built.cfg), &frames)
        .expect_err("a store at a kept field's location");
    assert!(
        error.contains("a store at the location of a field it kept"),
        "{error}"
    );
    // The rule alone: a creation at G's location, a store of c's slot, an effect.
    let refused = |broken: Graph, expected: &str| {
        let error = super::super::verify::verify_kept_heap(&broken).expect_err(expected);
        assert!(error.contains(expected), "{expected}: {error}");
    };
    let creation = NodeOp::StoreNew {
        location,
        shape: 0,
        key: 0,
        ways: 0,
    };
    refused(
        inject(body, creation, &[global, loaded]),
        "a store at the location of a field it kept",
    );
    refused(
        inject(body, NodeOp::StoreBinding { slot }, &[env, loaded]),
        "a store at the slot of a binding it kept",
    );
    let pc = graph.blocks[body.index()].first_pc;
    refused(
        inject(
            body,
            NodeOp::Effect {
                pc,
                pops: 0,
                pushes: 0,
            },
            &[],
        ),
        "an effect",
    );
    // What the header did not keep, and the code after the loop, may be written.
    let other = NodeOp::StoreField {
        location: location ^ 2,
        slot: 0,
    };
    super::super::verify::verify_kept_heap(&inject(body, other, &[global, loaded]))
        .expect("another location");
    super::super::verify::verify_kept_heap(&inject(after, field, &[global, loaded]))
        .expect("after the loop");
    super::super::verify::verify_kept_heap(&inject(
        body,
        NodeOp::StoreBinding { slot: slot + 1 },
        &[env, loaded],
    ))
    .expect("another slot");
}

/// A typed read, then the global it reads stored with another array (in the function itself, or
/// by an inlined callee), then a typed store through the global: the read checks its data for a
/// load only (TypedData { write: false }), since the store's receiver is another array
/// (`Rules::store_checking_reads` ends a global's source at a lowered store at its location).
#[test]
fn a_typed_read_before_its_global_is_stored_checks_no_store() {
    for (name, source) in [
        (
            "f",
            "var A = new Int32Array(8), B = new Int32Array(8);
             function f(g) { var x = A[0]; g.A = B; A[1] = x; return x; }
             for (var k = 0; k < 40; k++) { f(globalThis); A = new Int32Array(8); }",
        ),
        (
            "h",
            "var A = new Int32Array(8), B = new Int32Array(8);
             function setA(g) { g.A = B; }
             function h(g) { var x = A[0]; setA(g); A[1] = x; return x; }
             for (var k = 0; k < 40; k++) { h(globalThis); A = new Int32Array(8); }",
        ),
    ] {
        // The tier on, so that template code records the sites' in-line typed hits (rule U).
        let mut engine = warmed_on(source);
        let built = Compile::new(&mut engine, name).build();
        let graph = &built.graph;
        let reads = nodes(graph, |node| {
            matches!(node.op, NodeOp::TypedData { write: false, .. })
        });
        assert!(
            !reads.is_empty(),
            "{name}: the read's data checked for a load\n{}",
            dump(graph)
        );
    }
}
