//! P3 stage 3 step L3 end to end: named updates and a String's `length` against the VM: `o.x++`,
//! `++o.x`, `o.x--`, `o.x += 1`, `o.x += 0.5`, `o.x += "s"`, an int32 overflow, -0 and NaN, a
//! getter-backed `x`; the lengths of an empty, an ASCII, a non-ASCII, a concatenated and a sliced
//! (a view's) String, and an Array's at a site without a way (RegExp's exec result).

use super::*;

/// `inc`, `dec`, `pre` and `add` update `o.x` (its value an int32, a double, -0, NaN, a String
/// late, then a getter's); `big` overflows the int32 range; `len` reads Strings' lengths, `alen`
/// an exec result's.
pub(super) const UPDATE: &str = "function inc(o) { return o.x++; }\n\
     function dec(o) { o.x--; return o.x; }\n\
     function pre(o) { return ++o.x; }\n\
     function add(o, v) { o.x += v; return o.x; }\n\
     function big(o) { o.n++; return o.n; }\n\
     function len(s) { let n = 0; for (let i = 0; i < 3; i++) n += s.length; return n; }\n\
     function alen(re, s) { const a = re.exec(s); return a ? a.length : -1; }\n\
     function h(x) { return typeof x === 'number' ? (x !== x ? -7 : Object.is(x, -0) ? -9 : x % 1000003) \
       : typeof x === 'string' ? x.length : -1; }\n\
     const long = 'x'.repeat(600) + '\\u00e9' + 'y'.repeat(300);\n\
     const strs = ['', 'abc', '\\u00e9t\\u00e9', 'a' + 'b'.repeat(3), long.slice(1, 500), long.slice(2, 700), long];\n\
     let t = 0;\n\
     const o = { x: 1 }, p = { x: 0.5 }, q = { x: -0 }, r = { x: NaN };\n\
     const g = { get x() { return this.v; }, set x(w) { this.v = w; }, v: 3 };\n\
     const n = { n: 2147483640 };\n\
     for (let k = 0; k < 1500; k++) {\n\
       if (k === 900) { o.x = 'q'; p.x = '7'; }\n\
       if (k === 1100) { q.x = -0; r.x = NaN; }\n\
       const os = [o, p, q, r];\n\
       const a = os[k & 3];\n\
       t = (t * 31 + h(inc(a)) + h(dec(a)) + h(pre(a)) + h(add(a, k % 3 ? 1 : 0.5))) % 1000003;\n\
       if (k > 1300) t = (t * 31 + h(add(a, 's'))) % 1000003;\n\
       if (k > 1200) t = (t * 31 + h(inc(g)) + h(add(g, 2))) % 1000003;\n\
       if (k % 100 === 0) n.n = 2147483640;\n\
       t = (t * 31 + h(big(n))) % 1000003;\n\
       t = (t * 31 + len(strs[k % strs.length]) + alen(/(a)(b)?/, k % 2 ? 'xab' : 'xa')) % 1000003;\n\
       if (k > 1400) t = (t * 31 + len([1, 2, 3])) % 1000003;\n\
     }\n\
     t + ':' + o.x + ':' + p.x + ':' + q.x + ':' + r.x + ':' + g.v + ':' + n.n";

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
fn updates_and_lengths_have_the_vms_results() {
    let expected = run(Tier::Bytecode, UPDATE);
    let mut compiles = 0;
    for (label, switches) in configurations() {
        compiled();
        exits();
        assert_eq!(
            with_switches(switches, UPDATE),
            expected,
            "{label} {switches:?}"
        );
        compiles += compiled().len();
    }
    assert!(compiles > 0, "the tier compiled the functions");
}

/// Whether the tier's code of `name` (its graph as a compile builds it now) holds a node `which`
/// accepts and no effect at an operation `op` accepts.
fn lowered(
    engine: &mut Engine,
    name: &str,
    which: impl Fn(&super::super::ir::NodeOp) -> bool,
    op: impl Fn(&crate::bytecode::Op) -> bool,
) -> bool {
    use super::super::ir::NodeOp;
    let chunk = tier_chunk(engine, name).unwrap_or_else(|| panic!("{name}'s tier chunk"));
    let graph = graph_now(&chunk);
    graph.nodes.iter().any(|node| which(&node.op))
        && !graph.nodes.iter().any(|node| {
            matches!(node.op, NodeOp::Effect { pc, .. } if op(&chunk.jit_ops()[pc as usize]))
        })
}

#[test]
fn the_programs_updates_and_lengths_are_the_tiers_nodes_before_its_changes() {
    use super::super::ir::NodeOp;
    use crate::bytecode::Op;
    let warm = UPDATE.replace("k < 1500", "k < 300");
    let expected = run(Tier::Bytecode, &warm);
    eager().set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &warm), expected);
    for name in ["inc", "dec", "pre", "big"] {
        assert!(
            lowered(
                &mut engine,
                name,
                |node| matches!(node, NodeOp::StoreField { .. }),
                |op| matches!(op, Op::UpdateProp(..))
            ),
            "{name} updates in the tier's code"
        );
    }
    assert!(
        lowered(
            &mut engine,
            "add",
            |node| *node == NodeOp::F64Add,
            |op| matches!(op, Op::AppendProp(..))
        ),
        "add appends in the tier's code"
    );
    assert!(
        lowered(
            &mut engine,
            "len",
            |node| *node == NodeOp::StringLength,
            |op| matches!(op, Op::GetPropLocal(..))
        ),
        "len reads lengths in the tier's code"
    );
}

#[test]
fn an_arrays_length_at_a_strings_site_exits_and_converges_on_the_arrays_length() {
    // A site without a way takes a String's length; an Array there exits WrongShape until the
    // discard, after which the recompile reads the Array's (and an exit there would leave the
    // generic access).
    let warm = "function f(a) { return a.length; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = t + f('ab' + k); t";
    let chunk = converges(
        "f",
        warm,
        "let u = 0; for (let k = 0; k < 600; k++) u = u + f([k, k, k]); u",
        "f([1, 2]) + f([3])",
        ExitReason::WrongShape,
    );
    assert!(
        graph_now(&chunk)
            .nodes
            .iter()
            .any(|node| node.op == super::super::ir::NodeOp::LoadArrayLength),
        "the recompile reads the Array's length"
    );
}
