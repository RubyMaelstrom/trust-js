//! The quiet loop's facts (target/p3/stage3/select-impl/design-note-loopfacts.txt, commit 1, the
//! carry): a loop that keeps heap facts keeps through its header the shape, field, link and
//! creation facts of the objects its sites take, but where something the loop may run ends them:
//! a creation (a shape transition) any of its frames makes, a store at a field's location any of
//! its frames makes (the union rule, `build::Writes`), an effect. Against the tier off at graph,
//! the stores and creations made by inlined frames among them.

use super::*;

/// `source`'s completion with the tier off, then at graph inlining: not eager, eager, and eager
/// with an exit forced every seventh check, each the same.
fn agrees_at_graph(source: &str) {
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, source);
    assert!(!expected.starts_with("throw"), "{expected}");
    for (eager, exit_every) in [(false, 0), (true, 0), (true, 7)] {
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
    }
}

/// The nodes of `graph`'s first loop (its header's natural loop) that `is` matches.
fn in_loop(graph: &Graph, is: impl Fn(&Node) -> bool) -> Vec<NodeIx> {
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop");
    let mut member = vec![false; graph.blocks.len()];
    member[header] = true;
    let mut work: Vec<usize> = graph.blocks[header]
        .preds
        .iter()
        .map(|p| p.index())
        .filter(|&p| p > header)
        .collect();
    while let Some(b) = work.pop() {
        if !std::mem::replace(&mut member[b], true) {
            work.extend(graph.blocks[b].preds.iter().map(|p| p.index()));
        }
    }
    nodes(graph, |node| member[node.block.index()] && is(node))
}

fn shape_checks(graph: &Graph) -> usize {
    in_loop(graph, |node| matches!(node.op, NodeOp::CheckShape { .. })).len()
}

/// `name`'s graph after `source` ran in template code.
fn graph_after(source: &str, name: &str) -> Graph {
    let mut engine = warmed(source);
    Compile::new(&mut engine, name).build().graph
}

#[test]
fn a_quiet_loop_over_an_object_checked_before_it_checks_and_loads_it_no_more() {
    // `p.x` read before the loop: the loop checks p's shape and loads p.x at no iteration.
    let graph = graph_after(
        "var o = {x: 1, y: 2};
         function f(p, n) { var s = p.x; for (var i = 0; i < n; i++) s = (s + p.x + p.y) | 0; return s; }
         for (var k = 0; k < 40; k++) f(o, 20);",
        "f",
    );
    assert_eq!(shape_checks(&graph), 0, "{}", dump(&graph));
    let loads = in_loop(&graph, |node| matches!(node.op, NodeOp::LoadField { .. }));
    assert_eq!(
        loads.len(),
        1,
        "p.y alone, first read in the loop\n{}",
        dump(&graph)
    );
    let [kept] = &graph.kept[..] else {
        panic!("{:?}", graph.kept)
    };
    assert!(kept.shapes, "{kept:?}");
}

#[test]
fn a_loop_that_creates_keeps_no_shape_and_one_that_stores_keeps_no_field_at_its_location() {
    // A creation in the loop (each of an array's fresh objects' first property), in the function
    // or by an inlined callee: p's shape is checked in the loop. A store at p.x's location (another object's), in
    // the function or by an inlined callee: p.x is loaded in the loop, p.y not.
    for (name, creates) in [("c1", true), ("c2", true), ("s1", false), ("s2", false)] {
        let graph = graph_after(
            "var o = {x: 1, y: 2}, w = {x: 0, y: 0};
             function mk(q, v) { q.a = v; }
             function setx(q, v) { q.x = v; }
             function c1(p, n, qs) { var s = p.x; for (var i = 0; i < n; i++) { qs[i].a = i; s = (s + p.x) | 0; } return s; }
             function c2(p, n, qs) { var s = p.x; for (var i = 0; i < n; i++) { mk(qs[i], i); s = (s + p.x) | 0; } return s; }
             function s1(p, n, q) { var s = p.x + p.y; for (var i = 0; i < n; i++) { q.x = i; s = (s + p.x + p.y) | 0; } return s; }
             function s2(p, n, q) { var s = p.x + p.y; for (var i = 0; i < n; i++) { setx(q, i); s = (s + p.x + p.y) | 0; } return s; }
             function fresh(n) { var a = []; for (var j = 0; j < n; j++) a.push({}); return a; }
             for (var k = 0; k < 40; k++) { c1(o, 20, fresh(20)); c2(o, 20, fresh(20)); s1(o, 20, w); s2(o, 20, w); }",
            name,
        );
        let [kept] = &graph.kept[..] else {
            panic!("{name}: one quiet loop {:?}\n{}", graph.kept, dump(&graph))
        };
        if creates {
            assert!(!kept.shapes, "{name}: {kept:?}");
            assert!(shape_checks(&graph) > 0, "{name}\n{}", dump(&graph));
        } else {
            let loads = in_loop(&graph, |node| matches!(node.op, NodeOp::LoadField { .. }));
            assert_eq!(loads.len(), 1, "{name}: p.x alone\n{}", dump(&graph));
        }
    }
}

#[test]
fn carried_facts_read_as_the_vm_does() {
    // The shape changed between two runs of the loop (the entry's check fails, not the body's);
    // a field stored through an alias inside the loop, in the function and by an inlined callee;
    // a creation by an inlined callee on the object the loop reads; a prototype swapped between
    // runs; the same object as both arguments; a loop with a call that changes the object's field
    // and shape (it keeps no fact).
    agrees_at_graph(
        "var P1 = {m: 1}, P2 = {m: 5};
         function setx(q, v) { q.x = v; }
         function grow(q, v) { q.z = v; }
         function f(p, n) { var s = p.x; for (var i = 0; i < n; i++) s = (s + p.x + p.y) | 0; return s; }
         function g(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { q.x = i; s = (s + p.x) | 0; } return s; }
         function h(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { setx(q, i); s = (s + p.x) | 0; } return s; }
         function c(p, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; if (i === 3) grow(p, i); } return s; }
         function m(p, n) { var s = p.x; for (var i = 0; i < n; i++) s = (s + p.m) | 0; return s; }
         function chg(q) { q.x = q.x + 1; delete q.y; return 0; }
         function e2(p, n, k) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; if (i === 3) p[k] = 100; } return s; }
         function e(p, n) { var s = p.x + p.y; for (var i = 0; i < n; i++) { s = (s + p.x + (p.y | 0)) | 0; if (i === 3) s = (s + chg(p)) | 0; } return s; }
         var r = 0;
         for (var k = 0; k < 3000; k++) {
             var o = {x: k & 7, y: 1}, w = {x: 0, y: 0};
             if (k % 97 === 0) o.extra = 1;
             var a = Object.create((k & 1) ? P1 : P2); a.x = 1;
             r = (r + f(o, 10) + g(o, w, 10) + g(o, o, 10) + h(o, w, 10) + h(o, o, 10) + c({x: 1, y: 2}, 10) + m(a, 10) + e({x: 2, y: 3}, 10) + e2({x: 2, y: 3}, 10, 'x')) % 1000003;
         }
         r",
    );
}

#[test]
fn the_verifier_refuses_a_creation_in_a_loop_that_kept_shapes() {
    // A creation at a location the header kept no field of, in the loop of a header that kept
    // p's shape: refused (it may be p's transition); the same graph with the header keeping no
    // shape: admitted.
    let graph = graph_after(
        "var o = {x: 1, y: 2};
         function f(p, n) { var s = p.x; for (var i = 0; i < n; i++) s = (s + p.x + p.y) | 0; return s; }
         for (var k = 0; k < 40; k++) f(o, 20);",
        "f",
    );
    let [kept] = &graph.kept[..] else {
        panic!("{:?}", graph.kept)
    };
    assert!(kept.shapes, "{kept:?}");
    let header = super::super::verify::kept_header(&graph, kept).expect("its loop header");
    let latch = graph.blocks[header.index()]
        .preds
        .iter()
        .copied()
        .find(|latch| latch.index() > header.index())
        .expect("the latch");
    let location = (0..u16::MAX)
        .find(|location| !kept.fields.contains(location) && location % 2 == 0)
        .expect("a location");
    let creation = NodeOp::StoreNew {
        location,
        shape: 0,
        key: 0,
        ways: 0,
    };
    let mut broken = graph.clone();
    let pc = graph.blocks[latch.index()].first_pc;
    let node = broken.new_node(latch, creation, &[], Rep::None, pc);
    let body = &mut broken.blocks[latch.index()].body;
    body.insert(body.len() - 1, node);
    let error = super::super::verify::verify_kept_heap(&broken).expect_err("a creation");
    assert!(error.contains("a creation (a shape transition)"), "{error}");
    broken.kept[0].shapes = false;
    super::super::verify::verify_kept_heap(&broken).expect("no shape kept");
    // A store at the location of p.x, whose value the header kept (p's field, carried): refused,
    // as a creation landing there is.
    let carried = nodes(&graph, |node| {
        matches!(node.op, NodeOp::LoadField { .. })
            && in_loop(&graph, |n| std::ptr::eq(n, node)).is_empty()
    });
    let NodeOp::LoadField { location, .. } = graph.node(carried[0]).op else {
        unreachable!()
    };
    assert!(kept.fields.contains(&location), "{kept:?}");
    for store in [
        NodeOp::StoreField { location, slot: 0 },
        NodeOp::StoreNew {
            location,
            shape: 0,
            key: 0,
            ways: 0,
        },
    ] {
        let mut broken = graph.clone();
        broken.kept[0].shapes = false;
        let node = broken.new_node(latch, store, &[], Rep::None, pc);
        let body = &mut broken.blocks[latch.index()].body;
        body.insert(body.len() - 1, node);
        let error = super::super::verify::verify_kept_heap(&broken).expect_err("a kept field");
        assert!(
            error.contains("a store at the location of a field it kept"),
            "{error}"
        );
    }
}

#[test]
fn a_quiet_loop_keeps_only_the_facts_of_the_objects_it_takes() {
    // q is checked and read before and after a quiet loop that takes p alone: the header keeps
    // p's facts, not q's, so q.x is loaded again after the loop (the builder's joins and prunes in
    // the loop walk no fact its sites do not take).
    let graph = graph_after(
        "var o = {x: 1}, w = {x: 2};
         function f(p, q, n) { var s = (q.x + p.x) | 0; for (var i = 0; i < n; i++) s = (s + p.x) | 0; return (s + q.x) | 0; }
         for (var k = 0; k < 40; k++) f(o, w, 20);",
        "f",
    );
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop");
    let after: Vec<NodeIx> = nodes(&graph, |node| {
        matches!(node.op, NodeOp::LoadField { .. })
            && node.block.index() > header
            && in_loop(&graph, |n| std::ptr::eq(n, node)).is_empty()
    });
    assert_eq!(
        after.len(),
        1,
        "q.x loaded after the loop\n{}",
        dump(&graph)
    );
}

#[test]
fn the_union_takes_creations_from_nested_ways_and_by_literal_frames_at_every_depth() {
    // outerC's frame calls innerC, which creates `z` on a fresh object; helper's frame calls the
    // closure W makes, which creates `z`: neither way's own chunk creates, so each site's union
    // says a creation may run only through the nested way and the by-literal way
    // (`inline::Tree::writes`, `Writes::creates`).
    let mut engine = warmed(
        "function innerC(o, v) { o.z = v; }
         function outerC(o, v) { innerC(o, v); }
         function helper(f, a) { return f(a); }
         function W(n) { var s = 0; for (var i = 0; i < n; i++) { outerC({}, i); helper(function (o) { o.z = i; }, {}); s += i; } return s; }
         for (var k = 0; k < 40; k++) W(20);",
    );
    let compile = Compile::new(&mut engine, "W");
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
        assert!(writes.creates, "{writes:?}");
        if !way.nested.is_empty() && with(true, false).creates {
            nested = true;
        }
        if !way.literals.is_empty() {
            literal = true;
            assert!(
                with(false, true).creates,
                "helper's site takes the closure's creation through its by-literal way"
            );
        }
        // A way's own chunk creates nothing: the creation comes from below it.
        assert!(!with(false, false).creates, "{:?}", with(false, false));
    }
    assert!(
        nested && literal,
        "a nested way {nested}, a by-literal way {literal}"
    );
}

#[test]
fn a_store_on_a_path_the_loop_rarely_takes_ends_the_field_it_writes() {
    // The inlined setx stores q.x (p.x's location) on one iteration in 32: the header keeps p's
    // shape but not p.x, which the loop loads in every iteration; p.y is kept.
    let graph = graph_after(
        "var o = {x: 1, y: 2}, w = {x: 0, y: 0};
         function setx(q, v) { q.x = v; }
         function f(p, q, n) { var s = p.x + p.y; for (var i = 0; i < n; i++) { s = (s + p.x + p.y) | 0; if ((i & 31) === 31) setx(q, i); } return s; }
         for (var k = 0; k < 40; k++) f(o, w, 64);",
        "f",
    );
    let [kept] = &graph.kept[..] else {
        panic!("one quiet loop {:?}\n{}", graph.kept, dump(&graph))
    };
    assert!(kept.shapes, "{kept:?}");
    // q's shape is checked in the loop (by setx's store: nothing checked q before it), p's not.
    let p_checks = in_loop(&graph, |node| {
        matches!(node.op, NodeOp::CheckShape { .. })
            && graph.node(node.inputs[0]).op == NodeOp::EntrySlot(0)
    });
    assert!(p_checks.is_empty(), "{p_checks:?}\n{}", dump(&graph));
    let loads = in_loop(&graph, |node| matches!(node.op, NodeOp::LoadField { .. }));
    assert_eq!(loads.len(), 1, "p.x alone\n{}", dump(&graph));
}

#[test]
fn creations_and_stores_by_inlined_frames_at_every_depth_read_as_the_vm_does() {
    // The lowerings' adversarial programs (loopfacts-impl/adv l02, l11): a field the loop reads
    // stored, updated and compound-stored by an inlined callee on a rare path, through the same
    // object or another; a store and a creation two levels down (a nested way) and by a
    // by-literal frame, on a rare path and in every iteration.
    for source in [
        "function setx(q, v) { q.x = v; }
         function inc(q) { q.x++; }
         function addx(q, v) { q.x += v; }
         function f(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; if ((i & 31) === 31) setx(q, i); } return s; }
         function g(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; if ((i & 15) === 15) inc(q); } return s; }
         function h(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; if (i === n - 2) addx(q, 3); } return s; }
         var r = 0;
         for (var k = 0; k < 3000; k++) {
             var o = {x: k & 15, y: 0}, w = {x: 1, y: 0};
             var q = (k % 5 === 0) ? o : w;
             r = (r + f(o, q, 70) + g(o, q, 40) + h(o, q, 10)) % 1000003;
         }
         r",
        "function inner(o, v) { o.x = v; }
         function outer(o, v) { inner(o, v + 1); }
         function innerC(o, v) { o.z = v; }
         function outerC(o, v) { innerC(o, v); }
         function helper(f, a) { return f(a); }
         function W1(p, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; if ((i & 7) === 7) outer(p, i); } return s; }
         function W2(p, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x + (p.z | 0)) | 0; if (i === 5) outerC(p, i); } return s; }
         function W3(p, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x) | 0; helper(function (o) { o.x = i; }, p); } return s; }
         function W4(p, n) { var s = p.x; for (var i = 0; i < n; i++) { s = (s + p.x + (p.z | 0)) | 0; if (i === 6) helper(function (o) { o.z = i; }, p); } return s; }
         var r = 0;
         for (var k = 0; k < 3000; k++) { r = (r + W1({x: 1}, 20) + W2({x: 2}, 10) + W3({x: 3}, 10) + W4({x: 4}, 10)) % 1000003; }
         r",
    ] {
        agrees_at_graph(source);
    }
}

#[test]
fn a_methods_quiet_loop_keeps_the_facts_of_this_of_a_kept_fields_object_and_of_a_prototype() {
    // `this` checked and read before the loop; p.q's object, kept as p's field, checked and read
    // before it; a method found on p's prototype before it: the loop checks none of their shapes,
    // loads none of the fields read before it and checks no prototype link again.
    for (name, source) in [
        (
            "run",
            "function M() { this.x = 1; this.y = 2; }
             function run(n) { var s = this.x + this.y; for (var i = 0; i < n; i++) s = (s + this.x + this.y) | 0; return s; }
             M.prototype.run = run;
             var m = new M(); for (var k = 0; k < 40; k++) m.run(20);",
        ),
        (
            "f",
            "var o = {q: {x: 1, y: 2}};
             function f(p, n) { var s = p.q.x + p.q.y; for (var i = 0; i < n; i++) s = (s + p.q.x + p.q.y) | 0; return s; }
             for (var k = 0; k < 40; k++) f(o, 20);",
        ),
        (
            "g",
            "function A() { this.v = 1; }
             A.prototype.k = 3;
             var a = new A();
             function g(p, n) { var s = p.k; for (var i = 0; i < n; i++) s = (s + p.k) | 0; return s; }
             for (var k = 0; k < 40; k++) g(a, 20);",
        ),
    ] {
        let graph = graph_after(source, name);
        let [kept] = &graph.kept[..] else {
            panic!("{name}: one quiet loop {:?}\n{}", graph.kept, dump(&graph))
        };
        assert!(kept.shapes, "{name}: {kept:?}");
        let checks = in_loop(&graph, |node| {
            matches!(
                node.op,
                NodeOp::CheckShape { .. } | NodeOp::CheckProto { .. } | NodeOp::LoadField { .. }
            )
        });
        assert!(checks.is_empty(), "{name}: {checks:?}\n{}", dump(&graph));
    }
}

#[test]
fn a_loop_keeps_this_only_where_it_reads_it_and_an_inlined_frames_this_is_its_receiver() {
    // `run`'s loop reads p alone: the header keeps no fact of `this`, whose shape is checked again
    // after the loop (the facts a header keeps stay in the states after the loop: only those its
    // sites take). `outer` inlines `m.sum(n)`, whose loop reads its own `this` (m, the inlined
    // frame's receiver, not outer's `this`): that loop checks m's shape at no iteration.
    let graph = graph_after(
        "function M() { this.x = 1; this.y = 2; }
         function run(p, n) { var s = this.x; for (var i = 0; i < n; i++) s = (s + p.x) | 0; return s + this.y; }
         M.prototype.run = run;
         var m = new M(), o = {x: 3};
         for (var k = 0; k < 40; k++) m.run(o, 20);",
        "run",
    );
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop");
    let this_checks = nodes(&graph, |node| {
        matches!(node.op, NodeOp::CheckShape { .. })
            && graph.node(node.inputs[0]).op == NodeOp::This
            && node.block.index() > header
            && in_loop(&graph, |n| std::ptr::eq(n, node)).is_empty()
    });
    assert_eq!(
        this_checks.len(),
        1,
        "this checked after the loop\n{}",
        dump(&graph)
    );
    let graph = graph_after(
        "function M() { this.x = 1; this.y = 2; }
         M.prototype.sum = function (n) { var s = this.x; for (var i = 0; i < n; i++) s = (s + this.x + this.y) | 0; return s; };
         var m = new M();
         function outer(q, n) { return q.sum(n) + 1; }
         for (var k = 0; k < 40; k++) outer(m, 20);",
        "outer",
    );
    assert!(
        graph.kept.iter().any(|kept| kept.shapes),
        "the inlined loop kept m's shape: {:?}\n{}",
        graph.kept,
        dump(&graph)
    );
    assert_eq!(shape_checks(&graph), 0, "{}", dump(&graph));
}

#[test]
fn appends_dictionary_keys_prototype_replacements_and_aliased_receivers_read_as_the_vm_does() {
    // The coordinator's three shapes (loopfacts-impl/adv l14-l16): (i) an array's length read while
    // the loop appends at it, in the loop and by an inlined callee, and an object used as a
    // dictionary by computed keys one of which is a kept field's name; (ii) a data property and a
    // method found through a kept prototype link, replaced on the prototype through an alias of it, by
    // the loop and by an inlined callee (the link stays, the field must not); (iii) a kept `this` in
    // an inlined method's loop whose caller's loop stores its field through another name for the same
    // object, and the inlined loop storing it through an argument that is the receiver.
    for source in [
        "function app(a, v) { a[a.length] = v; }
         function f(a, n) { var s = a.length; for (var i = 0; i < n; i++) { var L = a.length; a[L] = i; s = (s + a.length + L) | 0; } return s; }
         function g(a, n) { var s = a.length; for (var i = 0; i < n; i++) { app(a, i); s = (s + a.length) | 0; } return s; }
         var keys = ['p', 'q', 'x', 'r'];
         function h(o, n) { var s = o.x; for (var i = 0; i < n; i++) { o[keys[i & 3]] = i; s = (s + o.x) | 0; } return s; }
         function h2(o, n, k) { var s = o.x; for (var i = 0; i < n; i++) { s = (s + o.x) | 0; if ((i & 7) === 7) o[k] = i; } return s; }
         var r = 0;
         for (var k = 0; k < 3000; k++) { r = (r + f([1, 2], 10) + g([3, 4, 5], 10) + h({ x: 1, p: 0, q: 0, r: 0 }, 12) + h2({ x: 2, y: 3 }, 16, (k & 1) ? 'x' : 'y')) % 1000003; }
         r",
        "function A() { this.v = 1; }
         A.prototype.k = 3;
         A.prototype.m = function () { return 10; };
         var m0 = A.prototype.m, m1 = function () { return 1000; };
         function setk(o, v) { o.k = v; }
         function setm(o, f) { o.m = f; }
         function f(p, P, n) { var s = p.k; for (var i = 0; i < n; i++) { s = (s + p.k) | 0; if (i === 4) P.k = 50; } P.k = 3; return s; }
         function g(p, P, n) { var s = p.k; for (var i = 0; i < n; i++) { s = (s + p.k) | 0; if (i === 5) setk(P, 70); } P.k = 3; return s; }
         function h(p, P, n) { var s = p.m(); for (var i = 0; i < n; i++) { s = (s + p.m()) | 0; if (i === 3) P.m = m1; } P.m = m0; return s; }
         function h2(p, P, n) { var s = p.m(); for (var i = 0; i < n; i++) { s = (s + p.m()) | 0; if (i === 6) setm(P, m1); } P.m = m0; return s; }
         var r = 0;
         for (var k = 0; k < 3000; k++) { var a = new A(); var Q = Object.getPrototypeOf(a); r = (r + f(a, Q, 10) + g(a, Q, 10) + h(a, Q, 10) + h2(a, Q, 10)) % 1000003; }
         r",
        "function M() { this.x = 1; this.y = 2; }
         M.prototype.sum = function (n) { var s = this.x; for (var i = 0; i < n; i++) s = (s + this.x + this.y) | 0; return s; };
         M.prototype.sum2 = function (q, n) { var s = this.x; for (var i = 0; i < n; i++) { s = (s + this.x) | 0; if (i === 2) q.x = s & 15; } return s; };
         function outer(m, q, n) { var s = 0; for (var j = 0; j < n; j++) { q.x = j; s = (s + m.sum(4)) | 0; } return s; }
         function outer2(m, q, n) { var s = 0; for (var j = 0; j < n; j++) { s = (s + m.sum2(q, 5)) | 0; } return s; }
         var r = 0;
         for (var k = 0; k < 3000; k++) { var m = new M(), w = new M(); var q = (k % 3) ? m : w; r = (r + outer(m, q, 6) + outer2(m, q, 4)) % 1000003; }
         r",
    ] {
        agrees_at_graph(source);
    }
}

// ---- the blocks the builder builds: a block that never ran is its NeverRan exit --------------

/// A loop with p's shape and p.x read before it, whose only effect, `op`, is in a block that
/// never ran (`k < i` with k 100 in the warm-up, past the loop's 10 iterations), in the function (`f`) or in a callee the graph
/// inlines (`outer`'s `cb`): the block's operation is a store to a global the loop reads, a
/// creation on p, a store to p's field the loop reads, or a call the graph does not inline.
const COLD_OPS: [(&str, &str); 4] = [
    ("global", "G = p.y + i;"),
    ("creation", "p.z = i;"),
    ("field", "p.x = i;"),
    ("call", "s = (s + Object.keys(p).length) | 0;"),
];

fn cold_source(op: &str, then_k: &str, warm: u32, rounds: u32) -> String {
    format!(
        "var G = 3;
         function f(p, n, k) {{ var s = p.x; for (var i = 0; i < n; i++) {{ s = (s + p.x + G) | 0; if (k < i) {{ {op} }} }} return s; }}
         function cb(p, i, k) {{ var s = 0; if (k < i) {{ {op} }} return s; }}
         function outer(p, n, k) {{ var s = p.x; for (var i = 0; i < n; i++) s = (s + p.x + G + cb(p, i, k)) | 0; return s; }}
         var r = 0;
         for (var j = 0; j < {rounds}; j++) {{
             var k = j < {warm} ? 100 : {then_k};
             r = (r + f({{x: j & 7, y: 1}}, 10, k) + outer({{x: j & 7, y: 1}}, 10, k)) % 1000003;
         }}
         r"
    )
}

#[test]
fn a_loop_whose_only_effect_is_in_a_block_that_never_ran_keeps_its_facts() {
    // The block is the builder's NeverRan exit (`Rules::exits_unbuilt`, which the prepass, the
    // inlined callee's summary and the inliner's E0 read with the frame's own feedback): the loop
    // keeps heap facts, its header keeps p's shape, and p's shape is checked nowhere in it.
    for (label, op) in COLD_OPS {
        let source = cold_source(op, "100", 300, 300);
        // The planner inlines no callee with a free-name store (`bytecode::inline_capability`):
        // `cb` with the global store is a call, an effect, in `outer`'s loop.
        let names: &[&str] = if label == "global" {
            &["f"]
        } else {
            &["f", "outer"]
        };
        for &name in names {
            let graph = graph_after(&source, name);
            assert!(
                graph.kept.iter().any(|kept| kept.shapes),
                "{label} {name}: {:?}\n{}",
                graph.kept,
                dump(&graph)
            );
            assert_eq!(shape_checks(&graph), 0, "{label} {name}\n{}", dump(&graph));
        }
    }
}

#[test]
fn a_cold_block_that_runs_exits_and_its_recompile_keeps_no_fact_its_operation_ends() {
    // After the warm-up the block runs in the loop's middle iteration: the compiled loop leaves
    // at the block (NeverRan), the VM runs it, and the recompile builds it, whose loop keeps no
    // fact the operation ends: no heap fact for the global store and the call (effects), no
    // shape for the creation, p.x reloaded in the loop for the field store. Results are the tier
    // off's.
    for (label, op) in COLD_OPS {
        let source = cold_source(op, "5", 2000, 4000);
        let expected = tier_off(&source);
        TestSwitches {
            on: true,
            inline_graph: Some(true),
            ..Default::default()
        }
        .set();
        super::super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        assert_eq!(eval(&mut engine, &source), expected, "{label}");
        let exits = super::super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
        assert!(
            exits
                .iter()
                .any(|&(_, reason)| reason == ExitReason::NeverRan),
            "{label}: a NeverRan exit: {:?}",
            exits.iter().take(8).collect::<Vec<_>>()
        );
        for name in ["f", "outer"] {
            let graph = Compile::new(&mut engine, name).build().graph;
            let loads = in_loop(&graph, |node| matches!(node.op, NodeOp::LoadField { .. }));
            match label {
                "global" | "call" => assert!(
                    graph.kept.is_empty(),
                    "{label} {name}: {:?}\n{}",
                    graph.kept,
                    dump(&graph)
                ),
                "creation" => assert!(
                    graph.kept.iter().all(|kept| !kept.shapes),
                    "{label} {name}: {:?}\n{}",
                    graph.kept,
                    dump(&graph)
                ),
                _ => assert!(!loads.is_empty(), "{label} {name}\n{}", dump(&graph)),
            }
        }
        super::super::release_retired(&mut engine.interp);
    }
}

#[test]
fn a_cold_block_flagged_never_ran_is_built_and_its_loop_keeps_no_fact() {
    // The block never ran and its start is flagged NeverRan (an exit there taught the build to make
    // it): the builder builds its call, an effect, and the prepass, reading the builder's own
    // predicate (`Rules::exits_unbuilt`), counts it, so that the loop keeps no heap fact. A prepass
    // that skipped the block would keep them, which the verifier refuses (`verify::kept_heap`).
    let source = cold_source(COLD_OPS[3].1, "100", 300, 300);
    let mut engine = warmed(&source);
    let chunk = chunk_of(&mut engine, "f");
    let ops = chunk.jit_ops();
    let lts: Vec<usize> = (0..ops.len()).filter(|&pc| ops[pc] == Op::Lt).collect();
    // `i < n`, then `k < i`, whose JumpIfFalse leads to the cold block.
    let start = lts[1] + 2;
    let graph = Compile::new(&mut engine, "f").build().graph;
    assert!(!graph.kept.is_empty(), "unflagged: quiet\n{}", dump(&graph));
    flag(&chunk, start as u32, ExitReason::NeverRan, false, 0);
    let graph = Compile::new(&mut engine, "f").build().graph;
    assert!(graph.kept.is_empty(), "{:?}\n{}", graph.kept, dump(&graph));
}

/// The tier off's result of `source`.
fn tier_off(source: &str) -> String {
    TestSwitches::default().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source)
}
