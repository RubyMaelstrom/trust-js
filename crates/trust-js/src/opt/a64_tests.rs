//! The optimizing tier's back end on real chunks (P3 stage 1, milestone 1): until the builder
//! lands, the graphs are built by hand over the chunks of small functions the bytecode VM ran
//! (so their caches hold feedback); the code is generated with `opt::a64::compile_with` and
//! installed as the chunk's native code, and the functions then run through the normal entry
//! paths: Rust entries from scripts and the VM, and the first stage's direct calls.

use super::*;
use crate::bytecode::Tier;
use crate::opt::exit::TEST_EXITS;
use crate::value::Value;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// An engine whose functions compile to bytecode at their first call and run in the VM, which
/// fills their caches, until a test installs the tier's code and moves to the JIT tier.
fn fixture(source: &str) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    engine
}

fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let object = value.as_obj().expect("a function");
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is an author function")
    };
    let runtime = user.func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .expect("the function ran as bytecode")
}

/// Generate `graph`'s code for `name`'s chunk, install it as the chunk's native code and move the
/// engine to the JIT tier, whose entries then run it.
fn install(engine: &mut Engine, name: &str, graph: &Graph, options: Options) -> Rc<JitCode> {
    let chunk = chunk_of(engine, name);
    let alloc = crate::opt::regalloc::allocate(graph, &crate::opt::regalloc::ARM64).unwrap();
    crate::opt::regalloc::tests::assert_consistent(graph, &alloc);
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = crate::opt::a64::compile_with(&chunk, graph, &layout, &ilayout, options)
        .unwrap_or_else(|refusal| panic!("the tier refused {name}: {refusal:?}"));
    chunk
        .jit
        .begin_compile()
        .expect("an untried code slot")
        .commit(Some(code.clone()));
    engine.set_tier(Tier::Jit);
    code
}

fn exits() -> Vec<(u32, ExitReason)> {
    TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

/// Collect at the first safepoint after every allocation (`TRUST_JS_GC_STRESS=1` on this
/// engine's heap), marking in one pause.
fn stress(engine: &Engine) {
    let mut core = engine.interp.gc_heap.core.borrow_mut();
    core.set_stress_every(1);
    core.set_incremental(crate::gc_core::IncrementalMode::Off);
}

fn num(x: f64) -> u64 {
    Value::Num(x).bits()
}

/// A hand-built graph.
struct Build {
    g: Graph,
}

impl Build {
    fn new(slots: usize) -> Self {
        Build {
            g: Graph::new(slots),
        }
    }

    fn block(&mut self, pc: u32, loop_header: bool) -> BlockIx {
        let b = self.g.add_block(pc, loop_header);
        if self.g.entry.is_none() {
            self.g.entry = Some(b);
        }
        b
    }

    fn edge(&mut self, from: BlockIx, to: BlockIx) {
        self.g.add_edge(from, to);
    }

    fn node(&mut self, b: BlockIx, op: NodeOp, inputs: &[NodeIx], rep: Rep, pc: u32) -> NodeIx {
        self.g.add_node(b, op, inputs, rep, pc)
    }

    /// A frame state at `pc`: `slots` names the slots a value holds, the rest are the frame's.
    fn state(
        &mut self,
        pc: u32,
        stack: &[NodeIx],
        slots: &[(u16, NodeIx)],
        handlers: &[(u32, u32)],
    ) -> StateIx {
        let mut sources = vec![SlotSource::Frame; self.g.n_slots];
        for &(slot, v) in slots {
            sources[slot as usize] = SlotSource::Value(v);
        }
        self.g.add_state(FrameState {
            pc,
            slots: sources.into(),
            stack: stack.into(),
            handlers: handlers.into(),
        })
    }

    /// A node that can exit, with its state.
    #[allow(clippy::too_many_arguments)]
    fn exiting(
        &mut self,
        b: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
        state: StateIx,
        reason: ExitReason,
    ) -> NodeIx {
        let n = self.g.add_node(b, op, inputs, rep, pc);
        self.g.node_mut(n).state = Some(state);
        self.g.node_mut(n).reason = Some(reason);
        n
    }
}

/// `add(a, b)` (LoadLocal 0, LoadLocal 1, Add at pc 2, Return) speculated on Numbers: both words
/// checked at the Add's pc, an f64 add, the sum boxed.
fn add_f64() -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at_add = b.state(2, &[x, y], &[], &[]);
    let fx = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let fy = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[y],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let sum = b.node(entry, NodeOp::F64Add, &[fx, fy], Rep::F64, 2);
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

const ADD: &str = "function add(a, b) { return a + b; } add(1, 2); add('a', 'b');";

#[test]
fn numbers_add_in_registers_and_exit_to_the_vm_on_a_string() {
    let mut engine = fixture(ADD);
    let code = install(&mut engine, "add", &add_f64(), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "add(1, 2)"), "3");
    assert_eq!(eval(&mut engine, "add(1.5, 2.25)"), "3.75");
    assert_eq!(eval(&mut engine, "add(0.1, 0.2)"), "0.30000000000000004");
    assert_eq!(eval(&mut engine, "Object.is(add(-0, -0), -0)"), "true");
    assert_eq!(
        eval(&mut engine, "add(NaN, 1) !== add(NaN, 1)"),
        "true",
        "a NaN sum is the canonical NaN"
    );
    assert!(exits().is_empty(), "Numbers never exit");
    assert_eq!(code.referenced.get(), 1, "a Rust entry ran the code");
    assert_eq!(eval(&mut engine, "add('x', 1)"), "x1");
    assert_eq!(eval(&mut engine, "add(1, 'y')"), "1y");
    assert_eq!(
        eval(&mut engine, "add({ valueOf() { return 40; } }, 2)"),
        "42"
    );
    assert_eq!(
        exits(),
        vec![
            (2, ExitReason::NotNumber),
            (2, ExitReason::NotNumber),
            (2, ExitReason::NotNumber)
        ]
    );
}

#[test]
fn direct_calls_from_the_first_stage_enter_the_code() {
    let mut engine = fixture(ADD);
    let code = install(&mut engine, "add", &add_f64(), Options::default());
    eval(
        &mut engine,
        "function drive(n) { var t = 0; for (var i = 0; i < n; i++) t = add(t, i); return t; }",
    );
    code.referenced.set(0);
    exits();
    assert_eq!(eval(&mut engine, "drive(1000)"), "499500");
    assert!(
        code.referenced.get() > 1,
        "a call-cache way's direct call entered the code"
    );
    // A string sum exits in a direct callee: the VM finishes it and returns to the caller's
    // direct sequence.
    eval(
        &mut engine,
        "function words(n) { var t = ''; for (var i = 0; i < n; i++) t = add(t, 'w'); return t; }",
    );
    assert_eq!(eval(&mut engine, "words(4)"), "wwww");
    assert_eq!(exits().len(), 4);
}

/// `add(a, b)` on int32s: an overflow or a fraction exits at the Add.
fn add_i32() -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at_add = b.state(2, &[x, y], &[], &[]);
    let int = |b: &mut Build, v| {
        let f = b.exiting(
            entry,
            NodeOp::CheckNumber,
            &[v],
            Rep::F64,
            2,
            at_add,
            ExitReason::NotNumber,
        );
        b.exiting(
            entry,
            NodeOp::CheckInt32,
            &[f],
            Rep::I32,
            2,
            at_add,
            ExitReason::NotInt32,
        )
    };
    let ix = int(&mut b, x);
    let iy = int(&mut b, y);
    let sum = b.exiting(
        entry,
        NodeOp::I32Add,
        &[ix, iy],
        Rep::I32,
        2,
        at_add,
        ExitReason::Overflow,
    );
    let boxed = b.node(entry, NodeOp::I32ToTagged, &[sum], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

/// `mul(a, b)` on int32s: a product past int32, or -0, exits at the Mul.
fn mul_i32() -> Graph {
    let mut g = add_i32();
    for node in &mut g.nodes {
        if node.op == NodeOp::I32Add {
            node.op = NodeOp::I32Mul;
        }
    }
    g
}

#[test]
fn int32_arithmetic_exits_on_overflow_fractions_and_minus_zero() {
    let mut engine = fixture(ADD);
    install(&mut engine, "add", &add_i32(), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "add(20, 22)"), "42");
    assert_eq!(eval(&mut engine, "add(-5, 3)"), "-2");
    assert_eq!(eval(&mut engine, "add(2147483647, -2147483648)"), "-1");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "add(2147483647, 1)"), "2147483648");
    assert_eq!(eval(&mut engine, "add(-2147483648, -1)"), "-2147483649");
    assert_eq!(eval(&mut engine, "add(1.5, 1)"), "2.5");
    assert_eq!(eval(&mut engine, "Object.is(add(-0, -0), -0)"), "true");
    assert_eq!(
        exits(),
        vec![
            (2, ExitReason::Overflow),
            (2, ExitReason::Overflow),
            (2, ExitReason::NotInt32),
            (2, ExitReason::NotInt32)
        ]
    );
    let mut engine = fixture("function mul(a, b) { return a * b; } mul(2, 3);");
    install(&mut engine, "mul", &mul_i32(), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "mul(6, 7)"), "42");
    assert_eq!(eval(&mut engine, "mul(-6, 7)"), "-42");
    assert_eq!(eval(&mut engine, "mul(0, 7)"), "0");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "mul(65536, 65536)"), "4294967296");
    assert_eq!(eval(&mut engine, "Object.is(mul(-1, 0), -0)"), "true");
    assert_eq!(eval(&mut engine, "Object.is(mul(0, -5), -0)"), "true");
    assert_eq!(exits().len(), 3);
}

/// `getx(o)` (GetPropLocal 0 at pc 0, Return): the receiver's shape checked against its site's
/// way, the field loaded where the way's location word says.
fn getx(chunk: &Chunk) -> Graph {
    let Op::GetPropLocal(_, _, cache) = chunk.jit_ops()[0] else {
        panic!("getx reads a local's property")
    };
    let way = chunk.jit_cache_preferred(cache).expect("one way");
    assert_eq!(way.depth, 0, "an own data property");
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let o = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at_get = b.state(0, &[], &[], &[]);
    let cell = b.exiting(
        entry,
        NodeOp::CheckShape(vec![way.recv_shape]),
        &[o],
        Rep::Cell,
        0,
        at_get,
        ExitReason::WrongShape,
    );
    let x = b.node(
        entry,
        NodeOp::LoadField {
            location: way.loc,
            slot: way.slot,
        },
        &[cell],
        Rep::Tagged,
        0,
    );
    b.node(entry, NodeOp::Return, &[x], Rep::None, 1);
    b.g
}

#[test]
fn a_shape_check_reads_a_field_and_exits_on_another_receiver() {
    let record: String = (0..20).map(|k| format!("p{k}: 0, ")).collect();
    for prefix in ["", record.as_str()] {
        // The second fixture's x lies in the object's record, past its cell's fifteen slots.
        let source = format!(
            "function make(x) {{ return {{{prefix}x: x}}; }}\n\
             function getx(o) {{ return o.x; }} getx(make(1));"
        );
        let mut engine = fixture(&source);
        let chunk = chunk_of(&mut engine, "getx");
        install(&mut engine, "getx", &getx(&chunk), Options::default());
        exits();
        assert_eq!(eval(&mut engine, "getx(make(5))"), "5");
        assert_eq!(eval(&mut engine, "getx(make('s'))"), "s");
        assert_eq!(eval(&mut engine, "getx(make(make(7))).x"), "7");
        assert!(exits().is_empty(), "one shape");
        assert_eq!(eval(&mut engine, "getx({y: 1, x: 2})"), "2");
        assert_eq!(eval(&mut engine, "getx(5)"), "undefined");
        assert_eq!(eval(&mut engine, "getx([])"), "undefined");
        assert_eq!(
            eval(&mut engine, "getx(null)"),
            "throw TypeError: cannot read property 'x' of null"
        );
        assert_eq!(
            exits()
                .into_iter()
                .map(|(_, reason)| reason)
                .collect::<Vec<_>>(),
            vec![ExitReason::WrongShape; 4]
        );
    }
}

/// `setx(o, v)` (LoadLocal 1, SetPropLocalDrop at pc 1, GetPropLocal at pc 2, Return): the
/// store's way checked, the field stored with its card marked, and read back.
fn setx(chunk: &Chunk) -> Graph {
    let Op::SetPropLocalDrop(_, _, cache) = chunk.jit_ops()[1] else {
        panic!("setx stores a local's property")
    };
    let way = chunk.jit_cache_preferred(cache).expect("one way");
    assert_eq!(way.depth, 0, "an own data property");
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let o = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let v = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let at_set = b.state(1, &[v], &[], &[]);
    let cell = b.exiting(
        entry,
        NodeOp::CheckShape(vec![way.recv_shape]),
        &[o],
        Rep::Cell,
        1,
        at_set,
        ExitReason::WrongShape,
    );
    let field = (way.loc, way.slot);
    b.node(
        entry,
        NodeOp::StoreField {
            location: field.0,
            slot: field.1,
        },
        &[cell, v],
        Rep::None,
        1,
    );
    let read = b.node(
        entry,
        NodeOp::LoadField {
            location: field.0,
            slot: field.1,
        },
        &[cell],
        Rep::Tagged,
        2,
    );
    b.node(entry, NodeOp::Return, &[read], Rep::None, 3);
    b.g
}

#[test]
fn a_field_store_marks_its_card_and_its_value_survives_collections() {
    let mut engine = fixture(
        "function make(x) { return {x: x}; }\n\
         function setx(o, v) { o.x = v; return o.x; } setx(make(0), 1);\n\
         var held = [];",
    );
    let chunk = chunk_of(&mut engine, "setx");
    install(&mut engine, "setx", &setx(&chunk), Options::default());
    stress(&engine);
    exits();
    // Old holders take young values: only their cards keep the values through the collections
    // between the stores and the reads.
    assert_eq!(
        eval(
            &mut engine,
            "for (var k = 0; k < 64; k++) held.push(make(0));\n\
             for (var k = 0; k < 64; k++) setx(held[k], {tag: 'v' + k});\n\
             var junk = []; for (var k = 0; k < 2000; k++) junk.push([k]);\n\
             var ok = 0; for (var k = 0; k < 64; k++) if (held[k].x.tag === 'v' + k) ok++; ok"
        ),
        "64"
    );
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "setx({y: 1, x: 2}, 3)"), "3");
    assert_eq!(exits(), vec![(1, ExitReason::WrongShape)]);
}

/// `callf(f, x)` (LoadLocal 0, LoadLocal 1, Call(1) at pc 2, Const 1, Add at pc 4, Return): the
/// call an effect, its result checked a Number; `x`, checked before the call, an f64 that lives
/// across it in a callee-saved register, and added to the result.
fn callf() -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let f = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let x = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let at_call = b.state(2, &[f, x], &[], &[]);
    let fx = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        2,
        at_call,
        ExitReason::NotNumber,
    );
    let call = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 2,
            pops: 2,
            pushes: 1,
        },
        &[f, x],
        Rep::None,
        2,
        at_call,
        ExitReason::Forced,
    );
    let r = b.node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 2);
    let one = b.node(entry, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 3);
    let at_add = b.state(4, &[r, one], &[], &[]);
    let fr = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[r],
        Rep::F64,
        4,
        at_add,
        ExitReason::NotNumber,
    );
    let plus_one = b.node(entry, NodeOp::F64Add, &[fr, one], Rep::F64, 4);
    // Not the bytecode's sum (it adds only 1): x's register is what this checks.
    let plus_x = b.node(entry, NodeOp::F64Add, &[plus_one, fx], Rep::F64, 4);
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[plus_x], Rep::Tagged, 4);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 5);
    b.g
}

#[test]
fn an_effect_calls_back_into_javascript_and_values_live_across_it() {
    let mut engine =
        fixture("function callf(f, x) { return f(x) + 1; } callf(function (y) { return y; }, 1);");
    install(&mut engine, "callf", &callf(), Options::default());
    exits();
    // x + (f(x) + 1), x in d8 across a callee that uses every FP register it likes.
    assert_eq!(
        eval(&mut engine, "callf(function (y) { return y * 2; }, 5)"),
        "16"
    );
    assert_eq!(
        eval(
            &mut engine,
            "callf(function (y) { var s = 0.5; for (var k = 0; k < 100; k++) s = s * 1.01 + y; \
             return Math.floor(s) - Math.floor(s); }, 2.5)"
        ),
        "3.5"
    );
    assert!(exits().is_empty());
    // A String result exits at the Add, where the VM adds 1 as the bytecode does.
    assert_eq!(
        eval(&mut engine, "callf(function (y) { return 'r' + y; }, 5)"),
        "r51"
    );
    assert_eq!(exits(), vec![(4, ExitReason::NotNumber)]);
    // A throw from the callee, at a pc with no handler, leaves through the throw epilogue.
    assert_eq!(
        eval(
            &mut engine,
            "try { callf(function () { throw 7; }, 1) } catch (e) { 'caught ' + e }"
        ),
        "caught 7"
    );
    assert!(exits().is_empty());
    // The callee re-enters the code (a recursion through the effect).
    assert_eq!(
        eval(
            &mut engine,
            "callf(function (y) { return y > 0 ? callf(function (z) { return z; }, y - 1) : 0; }, 3)"
        ),
        // callf(g, 2) = 2 + (g(2) + 1) = 5, so callf(h, 3) = 3 + (5 + 1).
        "9"
    );
}

#[test]
fn collections_during_an_effect_keep_the_values_living_across_it() {
    let mut engine =
        fixture("function callf(f, x) { return f(x) + 1; } callf(function (y) { return y; }, 1);");
    install(&mut engine, "callf", &callf(), Options::default());
    stress(&engine);
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "var t = 0; for (var k = 0; k < 50; k++) t += callf(function (y) { \
             var a = []; for (var j = 0; j < 20; j++) a.push({j: j}); return a.length + y; }, k); t"
        ),
        // 50 * (20 + 1) + 2 * (0 + ... + 49)
        "3500"
    );
    assert!(exits().is_empty());
}

/// `keep(f)` (Const 42, MakeObject at pc 1, StoreLocal 1, LoadLocal 0, Call(0) at pc 4, Pop,
/// GetPropLocal(1) at pc 6, Return): the object's literal an effect, its cell checked and then
/// held only in a callee-saved register across the call (this graph never stores it to its
/// slot), its field read after the call.
fn keep(chunk: &Chunk) -> Graph {
    let Op::GetPropLocal(_, _, cache) = chunk.jit_ops()[6] else {
        panic!("keep reads its object's property")
    };
    let way = chunk.jit_cache_preferred(cache).expect("one way");
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let f = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let k42 = b.node(entry, NodeOp::ConstTagged(num(42.0)), &[], Rep::Tagged, 0);
    let at_make = b.state(1, &[k42], &[], &[]);
    let make = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 1,
            pops: 1,
            pushes: 1,
        },
        &[k42],
        Rep::None,
        1,
        at_make,
        ExitReason::Forced,
    );
    let o = b.node(entry, NodeOp::Projection(0), &[make], Rep::Tagged, 1);
    let at_load = b.state(3, &[], &[(1, o)], &[]);
    let cell = b.exiting(
        entry,
        NodeOp::CheckShape(vec![way.recv_shape]),
        &[o],
        Rep::Cell,
        3,
        at_load,
        ExitReason::WrongShape,
    );
    let at_call = b.state(4, &[f], &[(1, o)], &[]);
    let call = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 4,
            pops: 1,
            pushes: 1,
        },
        &[f],
        Rep::None,
        4,
        at_call,
        ExitReason::Forced,
    );
    b.node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 4);
    let x = b.node(
        entry,
        NodeOp::LoadField {
            location: way.loc,
            slot: way.slot,
        },
        &[cell],
        Rep::Tagged,
        6,
    );
    b.node(entry, NodeOp::Return, &[x], Rep::None, 7);
    b.g
}

#[test]
fn a_cell_held_only_in_a_register_survives_collections_during_an_effect() {
    let mut engine =
        fixture("function keep(f) { var o = {x: 42}; f(); return o.x; } keep(function () {});");
    let chunk = chunk_of(&mut engine, "keep");
    install(&mut engine, "keep", &keep(&chunk), Options::default());
    stress(&engine);
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "var churn = function () { var a = []; for (var j = 0; j < 200; j++) a.push({x: j, y: [j]}); };\n\
             var t = 0; for (var k = 0; k < 20; k++) t += keep(churn); t"
        ),
        "840"
    );
    assert!(exits().is_empty());
}

/// `guarded(f)` (PushHandler(6), LoadLocal 0, Call(0) at pc 2, Return, ..., the catch pad at 6):
/// the call inside the try is an effect whose state has the handler (pushed at pc 0, depth 0);
/// the handler operations emit nothing.
fn guarded() -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let f = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 1);
    let at_call = b.state(2, &[f], &[], &[(0, 0)]);
    let call = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 2,
            pops: 1,
            pushes: 1,
        },
        &[f],
        Rep::None,
        2,
        at_call,
        ExitReason::Forced,
    );
    let r = b.node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[r], Rep::None, 3);
    b.g
}

#[test]
fn a_throw_at_a_pc_with_a_handler_exits_to_the_vms_catch() {
    let mut engine = fixture(
        "function guarded(f) { try { return f(); } catch (e) { return 'caught ' + e; } }\n\
         guarded(function () { return 1; });",
    );
    install(&mut engine, "guarded", &guarded(), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "guarded(function () { return 5; })"), "5");
    assert!(exits().is_empty());
    assert_eq!(
        eval(&mut engine, "guarded(function () { throw 'e'; })"),
        "caught e"
    );
    assert_eq!(
        eval(&mut engine, "guarded(function () { null.x; })"),
        "caught TypeError: cannot read property 'x' of null"
    );
    assert_eq!(
        exits(),
        vec![(2, ExitReason::Throw), (2, ExitReason::Throw)]
    );
    // Stack overflow in the callee is a RangeError the catch also takes.
    assert_eq!(
        eval(
            &mut engine,
            "function deep(n) { return deep(n + 1) + 1; } guarded(function () { return deep(0); })"
        )
        .split(':')
        .next()
        .unwrap(),
        "caught RangeError"
    );
}

/// `count(n)` (`let s = 0; for (let i = 0; i < n; i++) s = s + i; return s;`: the header at pc
/// 6, the body's Add at 12 and increment at 16, the exit at 18) on int32s: `s` and `i` phis, the
/// header's poll, both additions checked for overflow with the loop's state.
fn count() -> Graph {
    let mut b = Build::new(3);
    let entry = b.block(0, false);
    let header = b.block(6, true);
    let body = b.block(10, false);
    let exit = b.block(18, false);
    b.edge(entry, header);
    b.edge(header, body);
    b.edge(header, exit);
    b.edge(body, header);
    let n = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at_entry = b.state(0, &[], &[], &[]);
    let fn_ = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[n],
        Rep::F64,
        0,
        at_entry,
        ExitReason::NotNumber,
    );
    let in_ = b.exiting(
        entry,
        NodeOp::CheckInt32,
        &[fn_],
        Rep::I32,
        0,
        at_entry,
        ExitReason::NotInt32,
    );
    let zero = b.node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
    b.node(entry, NodeOp::Jump(header), &[], Rep::None, 5);
    let i = b.node(header, NodeOp::Phi, &[zero], Rep::I32, 6);
    let s = b.node(header, NodeOp::Phi, &[zero], Rep::I32, 6);
    let at_header = b.state(6, &[], &[(1, s), (2, i)], &[]);
    b.exiting(
        header,
        NodeOp::Poll,
        &[],
        Rep::None,
        6,
        at_header,
        ExitReason::Throw,
    );
    let lt = b.node(header, NodeOp::I32Cmp(Cond::Lt), &[i, in_], Rep::Bool, 8);
    b.node(
        header,
        NodeOp::Branch {
            then_block: body,
            else_block: exit,
        },
        &[lt],
        Rep::None,
        9,
    );
    let at_add = b.state(12, &[s, i], &[(1, s), (2, i)], &[]);
    let s2 = b.exiting(
        body,
        NodeOp::I32Add,
        &[s, i],
        Rep::I32,
        12,
        at_add,
        ExitReason::Overflow,
    );
    let one = b.node(body, NodeOp::ConstI32(1), &[], Rep::I32, 16);
    let at_inc = b.state(16, &[], &[(1, s2), (2, i)], &[]);
    let i2 = b.exiting(
        body,
        NodeOp::I32Add,
        &[i, one],
        Rep::I32,
        16,
        at_inc,
        ExitReason::Overflow,
    );
    b.node(body, NodeOp::Jump(header), &[], Rep::None, 17);
    b.g.node_mut(i).inputs.push(i2);
    b.g.node_mut(s).inputs.push(s2);
    let boxed = b.node(exit, NodeOp::I32ToTagged, &[s], Rep::Tagged, 18);
    b.node(exit, NodeOp::Return, &[boxed], Rep::None, 19);
    b.g
}

const COUNT: &str =
    "function count(n) { let s = 0; for (let i = 0; i < n; i++) { s = s + i; } return s; } count(3);";

#[test]
fn a_loop_runs_on_phis_and_exits_with_its_state_inside_it() {
    let mut engine = fixture(COUNT);
    install(&mut engine, "count", &count(), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "count(10)"), "45");
    assert_eq!(eval(&mut engine, "count(0)"), "0");
    assert_eq!(eval(&mut engine, "count(65536)"), "2147450880");
    assert!(exits().is_empty());
    // The sum passes int32 inside the loop: the VM finishes the loop from the Add.
    assert_eq!(eval(&mut engine, "count(100000)"), "4999950000");
    assert_eq!(eval(&mut engine, "count(2.5)"), "3");
    assert_eq!(eval(&mut engine, "count('4')"), "6");
    assert_eq!(
        exits(),
        vec![
            (12, ExitReason::Overflow),
            (0, ExitReason::NotInt32),
            (0, ExitReason::NotNumber)
        ]
    );
}

#[test]
fn a_loop_polls_with_its_values_in_caller_saved_registers() {
    let mut engine = fixture(COUNT);
    install(&mut engine, "count", &count(), Options::default());
    stress(&engine);
    let before = engine.interp.gc_heap.core.borrow().collections();
    // Each run allocates its string before the call, so the header's first poll collects.
    assert_eq!(
        eval(
            &mut engine,
            "var t = 0, junk; for (var k = 0; k < 30; k++) { junk = 'x' + k; t += count(100 + k); } t"
        ),
        (100..130i64).map(|m| m * (m - 1) / 2).sum::<i64>().to_string()
    );
    assert!(engine.interp.gc_heap.core.borrow().collections() > before);
    // The poll word set by hand: every poll takes the slow path while it stays set.
    engine.interp.poll.set_heap_due(true);
    assert_eq!(eval(&mut engine, "count(1000)"), "499500");
}

/// The swap of two tagged and two f64 values each iteration (cycles of the back edge's moves),
/// over `count`'s chunk: the result is the first tagged value after `n` swaps.
fn swaps() -> Graph {
    let mut b = Build::new(3);
    let entry = b.block(0, false);
    let header = b.block(6, true);
    let body = b.block(10, false);
    let exit = b.block(18, false);
    b.edge(entry, header);
    b.edge(header, body);
    b.edge(header, exit);
    b.edge(body, header);
    let n = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at_entry = b.state(0, &[], &[], &[]);
    let fn_ = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[n],
        Rep::F64,
        0,
        at_entry,
        ExitReason::NotNumber,
    );
    let a0 = b.node(entry, NodeOp::ConstTagged(num(1.0)), &[], Rep::Tagged, 0);
    let b0 = b.node(entry, NodeOp::ConstTagged(num(2.0)), &[], Rep::Tagged, 0);
    let p0 = b.node(entry, NodeOp::ConstF64(10f64.to_bits()), &[], Rep::F64, 0);
    let q0 = b.node(entry, NodeOp::ConstF64(20f64.to_bits()), &[], Rep::F64, 0);
    let k0 = b.node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 0);
    b.node(entry, NodeOp::Jump(header), &[], Rep::None, 5);
    let x = b.node(header, NodeOp::Phi, &[a0], Rep::Tagged, 6);
    let y = b.node(header, NodeOp::Phi, &[b0], Rep::Tagged, 6);
    let p = b.node(header, NodeOp::Phi, &[p0], Rep::F64, 6);
    let q = b.node(header, NodeOp::Phi, &[q0], Rep::F64, 6);
    let k = b.node(header, NodeOp::Phi, &[k0], Rep::F64, 6);
    let at_header = b.state(6, &[], &[], &[]);
    b.exiting(
        header,
        NodeOp::Poll,
        &[],
        Rep::None,
        6,
        at_header,
        ExitReason::Throw,
    );
    let lt = b.node(header, NodeOp::F64Cmp(Cond::Lt), &[k, fn_], Rep::Bool, 8);
    b.node(
        header,
        NodeOp::Branch {
            then_block: body,
            else_block: exit,
        },
        &[lt],
        Rep::None,
        9,
    );
    let one = b.node(body, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 16);
    let k2 = b.node(body, NodeOp::F64Add, &[k, one], Rep::F64, 16);
    b.node(body, NodeOp::Jump(header), &[], Rep::None, 17);
    b.g.node_mut(x).inputs.push(y);
    b.g.node_mut(y).inputs.push(x);
    b.g.node_mut(p).inputs.push(q);
    b.g.node_mut(q).inputs.push(p);
    b.g.node_mut(k).inputs.push(k2);
    // x + p + q, boxed: 1 + 30 after an even number of swaps, 2 + 30 after an odd one.
    let fx = b.exiting(
        exit,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        18,
        at_entry,
        ExitReason::NotNumber,
    );
    let pq = b.node(exit, NodeOp::F64Add, &[p, q], Rep::F64, 18);
    let sum = b.node(exit, NodeOp::F64Add, &[fx, pq], Rep::F64, 18);
    let boxed = b.node(exit, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 18);
    b.node(exit, NodeOp::Return, &[boxed], Rep::None, 19);
    b.g
}

#[test]
fn edge_moves_resolve_cycles_in_both_register_classes() {
    let mut engine = fixture(COUNT);
    install(&mut engine, "count", &swaps(), Options::default());
    assert_eq!(eval(&mut engine, "count(4)"), "31");
    assert_eq!(eval(&mut engine, "count(5)"), "32");
    assert_eq!(eval(&mut engine, "count(0)"), "31");
}

/// More values live at once than either class has registers for: forty f64s and thirty int32s
/// made from `add`'s operands and summed in reverse, so that every one lives to the sum.
fn pressure() -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at_add = b.state(2, &[x, y], &[], &[]);
    let fx = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let fy = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[y],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let iy = b.exiting(
        entry,
        NodeOp::CheckInt32,
        &[fy],
        Rep::I32,
        2,
        at_add,
        ExitReason::NotInt32,
    );
    let floats: Vec<NodeIx> = (0..40)
        .map(|k| {
            let c = b.node(
                entry,
                NodeOp::ConstF64(f64::from(k).to_bits()),
                &[],
                Rep::F64,
                2,
            );
            b.node(entry, NodeOp::F64Mul, &[fx, c], Rep::F64, 2)
        })
        .collect();
    let ints: Vec<NodeIx> = (0..30)
        .map(|k| {
            let c = b.node(entry, NodeOp::ConstI32(k), &[], Rep::I32, 2);
            b.exiting(
                entry,
                NodeOp::I32Add,
                &[iy, c],
                Rep::I32,
                2,
                at_add,
                ExitReason::Overflow,
            )
        })
        .collect();
    let mut total = b.node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 2);
    for &v in floats.iter().rev() {
        total = b.node(entry, NodeOp::F64Add, &[total, v], Rep::F64, 2);
    }
    for &v in ints.iter().rev() {
        let f = b.node(entry, NodeOp::I32ToF64, &[v], Rep::F64, 2);
        total = b.node(entry, NodeOp::F64Add, &[total, f], Rep::F64, 2);
    }
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[total], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

#[test]
fn values_beyond_the_registers_spill_at_their_definitions() {
    let graph = pressure();
    let alloc = crate::opt::regalloc::allocate(&graph, &crate::opt::regalloc::ARM64).unwrap();
    crate::opt::regalloc::tests::assert_consistent(&graph, &alloc);
    assert!(
        alloc.spill_bytes >= 8 * 8,
        "{} spill bytes",
        alloc.spill_bytes
    );
    let mut engine = fixture(ADD);
    install(&mut engine, "add", &graph, Options::default());
    exits();
    // x * (0 + ... + 39) + 30 y + (0 + ... + 29)
    assert_eq!(
        eval(&mut engine, "add(2, 1)"),
        (2 * 780 + 30 + 435).to_string()
    );
    assert_eq!(
        eval(&mut engine, "add(0.5, -3)"),
        (390 - 90 + 435).to_string()
    );
    // An int32 add in the middle overflows: its recipe reads the operands from their homes, some
    // of them spilled, and the VM adds as the bytecode does.
    assert_eq!(eval(&mut engine, "add(1, 2147483640)"), "2147483641");
    assert_eq!(exits(), vec![(2, ExitReason::Overflow)]);
}

#[test]
fn forced_exits_resume_checks_and_effects_in_the_vm() {
    let mut engine = fixture(COUNT);
    install(
        &mut engine,
        "count",
        &count(),
        Options { forced_exits: true },
    );
    engine.interp.opt_exit_every = 7;
    engine.interp.opt_exit_countdown = 7;
    exits();
    for n in [0, 1, 2, 3, 10, 50] {
        let expected = (n * (n - 1) / 2).to_string();
        assert_eq!(
            eval(&mut engine, &format!("count({n})")),
            expected,
            "count({n})"
        );
    }
    let forced = exits();
    assert!(
        forced.len() >= 4
            && forced
                .iter()
                .all(|&(pc, reason)| reason == ExitReason::Forced && [0, 12, 16].contains(&pc)),
        "{forced:?}"
    );
    // An effect exits before its operation: the VM performs the call.
    let mut engine =
        fixture("function callf(f, x) { return f(x) + 1; } callf(function (y) { return y; }, 1);");
    install(
        &mut engine,
        "callf",
        &callf(),
        Options { forced_exits: true },
    );
    engine.interp.opt_exit_every = 2;
    engine.interp.opt_exit_countdown = 2;
    exits();
    let mut results = Vec::new();
    for k in 0..6 {
        results.push(eval(
            &mut engine,
            &format!("callf(function (y) {{ return y * 2; }}, {k})"),
        ));
    }
    // Code runs give x + 2x + 1; a run the gate exits gives the bytecode's 2x + 1.
    for (k, result) in results.iter().enumerate() {
        let k = k as i32;
        assert!(
            *result == (3 * k + 1).to_string() || *result == (2 * k + 1).to_string(),
            "callf({k}) = {result}"
        );
    }
    assert!(exits()
        .iter()
        .all(|&(_, reason)| reason == ExitReason::Forced));
}

#[test]
fn the_arguments_count_and_an_elided_rest_reach_the_code_and_the_vm() {
    // `arguments.length` alone: the prologue keeps the count in the slot, from both calling
    // conventions.
    let mut engine = fixture(
        "function al() { return arguments.length; } al(1, 2);\n\
         function rl(...r) { return r.length; } rl(1, 2);",
    );
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let count = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    b.node(entry, NodeOp::Return, &[count], Rep::None, 1);
    let code = install(&mut engine, "al", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "al(1, 2, 3)"), "3");
    code.referenced.set(0);
    assert_eq!(
        eval(
            &mut engine,
            "function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += al(i, i, i, i); return t; } drive(100)"
        ),
        "400"
    );
    assert!(code.referenced.get() > 1, "a direct call entered the code");
    // An elided rest Array: an exit before RestLength makes the Array the VM reads, from the
    // arguments the prologue kept; the effect's template reads them in place.
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let at_length = b.state(0, &[], &[], &[]);
    b.exiting(
        entry,
        NodeOp::Exit(ExitReason::Unsupported),
        &[],
        Rep::None,
        0,
        at_length,
        ExitReason::Unsupported,
    );
    install(&mut engine, "rl", &b.g, Options::default());
    exits();
    assert_eq!(eval(&mut engine, "rl(1, 2, 3)"), "3");
    assert_eq!(eval(&mut engine, "rl()"), "0");
    assert_eq!(exits(), vec![(0, ExitReason::Unsupported); 2]);
    let mut engine = fixture("function rl(...r) { return r.length; } rl(1, 2);");
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let at_length = b.state(0, &[], &[], &[]);
    let length = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 0,
            pops: 0,
            pushes: 1,
        },
        &[],
        Rep::None,
        0,
        at_length,
        ExitReason::Forced,
    );
    let r = b.node(entry, NodeOp::Projection(0), &[length], Rep::Tagged, 0);
    b.node(entry, NodeOp::Return, &[r], Rep::None, 1);
    install(&mut engine, "rl", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "rl(1, 2, 3, 4)"), "4");
}

/// `viaLeaf(x)` after its second stage spliced `leaf(y) { return y + 1; }` in: LoadNameForCall at
/// pc 0 (the receiver, then the callee), LoadLocal 0, InlineGuard at pc 2, the splice (StoreLocal
/// 1, StoreLocal 2, Pop, LoadLocal 1, Const 1, Add at pc 8, ...) and the generic call at pc 13. The free name is an effect, the
/// callee is checked as InlineGuard checks it, and the splice's Add is speculated on Numbers.
fn via_leaf() -> Graph {
    let mut b = Build::new(3);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at_name = b.state(0, &[], &[], &[]);
    let name = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 0,
            pops: 0,
            pushes: 2,
        },
        &[],
        Rep::None,
        0,
        at_name,
        ExitReason::Forced,
    );
    // The receiver, then the callee, which InlineGuard reads argc + 1 words deep.
    let this = b.node(entry, NodeOp::Projection(0), &[name], Rep::Tagged, 0);
    let callee = b.node(entry, NodeOp::Projection(1), &[name], Rep::Tagged, 0);
    let at_guard = b.state(2, &[this, callee, x], &[], &[]);
    b.exiting(
        entry,
        NodeOp::CheckCallee { target: 0 },
        &[callee],
        Rep::None,
        2,
        at_guard,
        ExitReason::WrongCallee,
    );
    let one = b.node(entry, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 7);
    // The splice stored the argument and the callee; it popped the receiver.
    let at_add = b.state(8, &[x, one], &[(1, x), (2, callee)], &[]);
    let fx = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        8,
        at_add,
        ExitReason::NotNumber,
    );
    let sum = b.node(entry, NodeOp::F64Add, &[fx, one], Rep::F64, 8);
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 8);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 14);
    b.g
}

#[test]
fn a_spliced_callees_guard_exits_when_the_callee_changes() {
    let mut engine = fixture(
        "function leaf(y) { return y + 1; }\n\
         function viaLeaf(x) { return leaf(x); }",
    );
    engine.set_tier(Tier::Jit);
    eval(&mut engine, "for (var k = 0; k < 20; k++) viaLeaf(k);");
    let primary = chunk_of(&mut engine, "viaLeaf");
    let env = engine.interp.global_env;
    let value = engine.interp.get_var("viaLeaf", &env).ok().unwrap();
    let object = value.as_obj().unwrap();
    let (func, realm) = {
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        (user.func.clone(), user.realm)
    };
    let plan = crate::bytecode::plan_inlines(&primary, &func, realm, &env);
    assert!(!plan.is_empty(), "leaf is spliced");
    let inlined = crate::bytecode::compile_with_inlines(&func, &plan, &primary).unwrap();
    assert!(matches!(inlined.jit_ops()[2], Op::InlineGuard(0, 13)));
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code =
        crate::opt::a64::compile_with(&inlined, &via_leaf(), &layout, &ilayout, Options::default())
            .unwrap();
    inlined.jit.begin_compile().unwrap().commit(Some(code));
    assert!(func.runtime().code2.set(Some(inlined)).is_ok());
    exits();
    assert_eq!(eval(&mut engine, "viaLeaf(41)"), "42");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "viaLeaf('a')"), "a1");
    assert_eq!(
        eval(
            &mut engine,
            "leaf = function (y) { return y * 10; }; viaLeaf(4)"
        ),
        "40"
    );
    assert_eq!(
        exits(),
        vec![(8, ExitReason::NotNumber), (2, ExitReason::WrongCallee)]
    );
}

#[test]
fn an_effect_closes_over_the_activation_the_prologue_made() {
    // `cap(a)` (MakeClosure at pc 0, Return): the prologue makes the activation with the
    // captured parameter copied in, and the closure the effect makes reads it.
    let mut engine = fixture("function cap(a) { return function () { return a; }; } cap(1);");
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let at = b.state(0, &[], &[], &[]);
    let make = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 0,
            pops: 0,
            pushes: 1,
        },
        &[],
        Rep::None,
        0,
        at,
        ExitReason::Forced,
    );
    let closure = b.node(entry, NodeOp::Projection(0), &[make], Rep::Tagged, 0);
    b.node(entry, NodeOp::Return, &[closure], Rep::None, 1);
    install(&mut engine, "cap", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "cap(5)()"), "5");
    assert_eq!(
        eval(&mut engine, "var f = cap('s'), g = cap(2); f() + g()"),
        "s2"
    );
    assert_eq!(
        eval(
            &mut engine,
            "function dc(n) { var t = 0; for (var i = 0; i < n; i++) t += cap(i)(); return t; } dc(100)"
        ),
        "4950"
    );
}

#[test]
fn an_effect_runs_with_the_bodys_strictness_and_the_callers_returns() {
    // `strictset()` ('use strict'; Const 1, StoreNameCached at pc 1, Const 2, Return).
    let mut engine = fixture(
        "function strictset() { 'use strict'; undeclared_x = 1; return 2; }\n\
         try { strictset(); } catch (e) {}",
    );
    let mut b = Build::new(0);
    let entry = b.block(0, false);
    let one = b.node(entry, NodeOp::ConstTagged(num(1.0)), &[], Rep::Tagged, 0);
    let at = b.state(1, &[one], &[], &[]);
    b.exiting(
        entry,
        NodeOp::Effect {
            pc: 1,
            pops: 1,
            pushes: 0,
        },
        &[one],
        Rep::None,
        1,
        at,
        ExitReason::Forced,
    );
    let two = b.node(entry, NodeOp::ConstTagged(num(2.0)), &[], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[two], Rep::None, 3);
    install(&mut engine, "strictset", &b.g, Options::default());
    assert_eq!(
        eval(
            &mut engine,
            "try { strictset(); 'stored' } catch (e) { e.name }"
        ),
        "ReferenceError"
    );
    assert_eq!(eval(&mut engine, "typeof undeclared_x"), "undefined");
    assert_eq!(eval(&mut engine, "sloppy_y = 3; sloppy_y"), "3");
}

#[test]
fn lazy_arguments_reach_the_vm_as_its_entry_makes_them() {
    // `la()` (Const 1, ArgElem at pc 1, Return) reads `arguments[1]`, whose object the body
    // elides: an exit before the read gives the VM the plain Array of the arguments its entry
    // would make, and the effect's template reads the arguments in place.
    let source = "function la() { return arguments[1]; } la(1, 2);";
    let mut engine = fixture(source);
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let at = b.state(0, &[], &[], &[]);
    b.exiting(
        entry,
        NodeOp::Exit(ExitReason::Unsupported),
        &[],
        Rep::None,
        0,
        at,
        ExitReason::Unsupported,
    );
    install(&mut engine, "la", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "la(7, 8, 9)"), "8");
    assert_eq!(eval(&mut engine, "la(7)"), "undefined");
    let mut engine = fixture(source);
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let one = b.node(entry, NodeOp::ConstTagged(num(1.0)), &[], Rep::Tagged, 0);
    let at = b.state(1, &[one], &[], &[]);
    let read = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 1,
            pops: 1,
            pushes: 1,
        },
        &[one],
        Rep::None,
        1,
        at,
        ExitReason::Forced,
    );
    let r = b.node(entry, NodeOp::Projection(0), &[read], Rep::Tagged, 1);
    b.node(entry, NodeOp::Return, &[r], Rep::None, 2);
    install(&mut engine, "la", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "la(7, 8, 9)"), "8");
    assert_eq!(
        eval(
            &mut engine,
            "function d(n) { var t = 0; for (var i = 0; i < n; i++) t += la(i, i + 1); return t; } d(10)"
        ),
        "55"
    );
}

#[test]
fn this_is_the_receiver_the_caller_bound() {
    // `me()` (LoadThis, Return).
    let mut engine = fixture("function me() { return this; } me();");
    let mut b = Build::new(0);
    let entry = b.block(0, false);
    let this = b.node(entry, NodeOp::This, &[], Rep::Tagged, 0);
    b.node(entry, NodeOp::Return, &[this], Rep::None, 1);
    install(&mut engine, "me", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "var o = {me: me}; o.me() === o"), "true");
    assert_eq!(eval(&mut engine, "typeof me.call(5)"), "object");
    assert_eq!(eval(&mut engine, "me() === globalThis"), "true");
}

/// `tl(n)`: `count`'s loop inside a try (PushHandler(23) at pc 0, the header at 7, the Add at 13,
/// the increment at 17, the exit at 19, the catch pad at 23): every state inside the loop holds
/// the handler pushed at pc 0 at depth 0. With `spin`, the loop only counts to `n` in an f64.
fn try_loop(spin: bool) -> Graph {
    let mut b = Build::new(4);
    let entry = b.block(0, false);
    let header = b.block(7, true);
    let body = b.block(11, false);
    let exit = b.block(19, false);
    b.edge(entry, header);
    b.edge(header, body);
    b.edge(header, exit);
    b.edge(body, header);
    let handler = [(0u32, 0u32)];
    let n = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at_entry = b.state(0, &[], &[], &[]);
    let fn_ = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[n],
        Rep::F64,
        0,
        at_entry,
        ExitReason::NotNumber,
    );
    if spin {
        let k0 = b.node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 0);
        b.node(entry, NodeOp::Jump(header), &[], Rep::None, 6);
        let k = b.node(header, NodeOp::Phi, &[k0], Rep::F64, 7);
        let at_header = b.state(7, &[], &[], &handler);
        b.exiting(
            header,
            NodeOp::Poll,
            &[],
            Rep::None,
            7,
            at_header,
            ExitReason::Throw,
        );
        let lt = b.node(header, NodeOp::F64Cmp(Cond::Lt), &[k, fn_], Rep::Bool, 9);
        b.node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            10,
        );
        let one = b.node(body, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 17);
        let k2 = b.node(body, NodeOp::F64Add, &[k, one], Rep::F64, 17);
        b.node(body, NodeOp::Jump(header), &[], Rep::None, 18);
        b.g.node_mut(k).inputs.push(k2);
        let boxed = b.node(exit, NodeOp::F64ToTagged, &[k], Rep::Tagged, 19);
        b.node(exit, NodeOp::Return, &[boxed], Rep::None, 20);
        return b.g;
    }
    let in_ = b.exiting(
        entry,
        NodeOp::CheckInt32,
        &[fn_],
        Rep::I32,
        0,
        at_entry,
        ExitReason::NotInt32,
    );
    let zero = b.node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
    b.node(entry, NodeOp::Jump(header), &[], Rep::None, 6);
    let i = b.node(header, NodeOp::Phi, &[zero], Rep::I32, 7);
    let s = b.node(header, NodeOp::Phi, &[zero], Rep::I32, 7);
    let at_header = b.state(7, &[], &[(1, s), (2, i)], &handler);
    b.exiting(
        header,
        NodeOp::Poll,
        &[],
        Rep::None,
        7,
        at_header,
        ExitReason::Throw,
    );
    let lt = b.node(header, NodeOp::I32Cmp(Cond::Lt), &[i, in_], Rep::Bool, 9);
    b.node(
        header,
        NodeOp::Branch {
            then_block: body,
            else_block: exit,
        },
        &[lt],
        Rep::None,
        10,
    );
    let at_add = b.state(13, &[s, i], &[(1, s), (2, i)], &handler);
    let s2 = b.exiting(
        body,
        NodeOp::I32Add,
        &[s, i],
        Rep::I32,
        13,
        at_add,
        ExitReason::Overflow,
    );
    let one = b.node(body, NodeOp::ConstI32(1), &[], Rep::I32, 17);
    let at_inc = b.state(17, &[], &[(1, s2), (2, i)], &handler);
    let i2 = b.exiting(
        body,
        NodeOp::I32Add,
        &[i, one],
        Rep::I32,
        17,
        at_inc,
        ExitReason::Overflow,
    );
    b.node(body, NodeOp::Jump(header), &[], Rep::None, 18);
    b.g.node_mut(i).inputs.push(i2);
    b.g.node_mut(s).inputs.push(s2);
    let boxed = b.node(exit, NodeOp::I32ToTagged, &[s], Rep::Tagged, 19);
    b.node(exit, NodeOp::Return, &[boxed], Rep::None, 20);
    b.g
}

const TRY_LOOP: &str =
    "function tl(n) { try { let s = 0; for (let i = 0; i < n; i++) { s = s + i; } \
                        return s; } catch (e) { return 'caught'; } } tl(3);";

#[test]
fn exits_inside_a_try_rebuild_its_handler_for_the_vm() {
    let mut engine = fixture(TRY_LOOP);
    install(&mut engine, "tl", &try_loop(false), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "tl(10)"), "45");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "tl(100000)"), "4999950000");
    assert_eq!(eval(&mut engine, "tl('5')"), "10");
    assert_eq!(
        exits(),
        vec![(13, ExitReason::Overflow), (0, ExitReason::NotNumber)]
    );
    // Forced exits inside the try, every third check.
    let mut engine = fixture(TRY_LOOP);
    install(
        &mut engine,
        "tl",
        &try_loop(false),
        Options { forced_exits: true },
    );
    engine.interp.opt_exit_every = 3;
    engine.interp.opt_exit_countdown = 3;
    for n in [4, 9, 30] {
        assert_eq!(
            eval(&mut engine, &format!("tl({n})")),
            (n * (n - 1) / 2).to_string()
        );
    }
}

#[test]
fn a_host_interruption_at_a_poll_leaves_through_the_handlers_region() {
    let mut engine = fixture(TRY_LOOP);
    install(&mut engine, "tl", &try_loop(true), Options::default());
    assert_eq!(eval(&mut engine, "tl(1000)"), "1000");
    exits();
    let interrupt = engine.interrupt_handle();
    interrupt.set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(100),
    ));
    let outcome = engine.eval_interruptible("tl(3e9)", false).unwrap();
    interrupt.set_deadline(None);
    assert!(
        matches!(outcome, crate::ExecutionOutcome::Interrupted { .. }),
        "the interruption is no throw the catch takes"
    );
    assert_eq!(exits(), vec![(7, ExitReason::Throw)]);
    assert_eq!(eval(&mut engine, "tl(10)"), "10");
}

#[derive(Clone, Copy, PartialEq)]
enum Operands {
    F64,
    I32,
}

/// `opK(a, b)` (`return a + b`, so that an exit computes a + b): `op` over the operands as
/// `operands` checks them, its result boxed.
fn op_graph(op: NodeOp, operands: Operands, unary: bool, out: Rep) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    let check = |b: &mut Build, v| match operands {
        Operands::F64 => b.exiting(
            entry,
            NodeOp::CheckNumber,
            &[v],
            Rep::F64,
            2,
            at,
            ExitReason::NotNumber,
        ),
        Operands::I32 => {
            let f = b.exiting(
                entry,
                NodeOp::CheckNumber,
                &[v],
                Rep::F64,
                2,
                at,
                ExitReason::NotNumber,
            );
            b.exiting(
                entry,
                NodeOp::CheckInt32,
                &[f],
                Rep::I32,
                2,
                at,
                ExitReason::NotInt32,
            )
        }
    };
    let a = check(&mut b, x);
    let c = check(&mut b, y);
    let inputs: Vec<NodeIx> = if unary { vec![a] } else { vec![a, c] };
    let (op, out) = match op {
        // `!(a < b)`: a comparison's Bool negated.
        NodeOp::BoolNot => {
            let lt = b.node(entry, NodeOp::F64Cmp(Cond::Lt), &inputs, Rep::Bool, 2);
            let r = b.node(entry, NodeOp::BoolNot, &[lt], Rep::Bool, 2);
            let boxed = b.node(entry, NodeOp::BoolToTagged, &[r], Rep::Tagged, 2);
            b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
            return b.g;
        }
        op => (op, out),
    };
    let r = if op.can_exit() {
        b.exiting(entry, op, &inputs, out, 2, at, ExitReason::Overflow)
    } else {
        b.node(entry, op, &inputs, out, 2)
    };
    let boxed = match out {
        Rep::I32 => b.node(entry, NodeOp::I32ToTagged, &[r], Rep::Tagged, 2),
        Rep::F64 => b.node(entry, NodeOp::F64ToTagged, &[r], Rep::Tagged, 2),
        Rep::Bool => b.node(entry, NodeOp::BoolToTagged, &[r], Rep::Tagged, 2),
        _ => r,
    };
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

#[test]
fn every_pure_operation_computes_as_the_language_does() {
    use Operands::{F64, I32};
    let cases: Vec<(NodeOp, Operands, bool, Rep, &str)> = vec![
        (NodeOp::I32And, I32, false, Rep::I32, "a & b"),
        (NodeOp::I32Or, I32, false, Rep::I32, "a | b"),
        (NodeOp::I32Xor, I32, false, Rep::I32, "a ^ b"),
        (NodeOp::I32Shl, I32, false, Rep::I32, "a << b"),
        (NodeOp::I32Sar, I32, false, Rep::I32, "a >> b"),
        (NodeOp::I32Not, I32, true, Rep::I32, "~a"),
        (NodeOp::I32Sub, I32, false, Rep::I32, "a - b"),
        (NodeOp::I32ToF64, I32, true, Rep::F64, "a"),
        (NodeOp::F64Sub, F64, false, Rep::F64, "a - b"),
        (NodeOp::F64Mul, F64, false, Rep::F64, "a * b"),
        (NodeOp::F64Div, F64, false, Rep::F64, "a / b"),
        (NodeOp::F64Mod, F64, false, Rep::F64, "a % b"),
        (NodeOp::F64Neg, F64, true, Rep::F64, "-a"),
        (NodeOp::I32Cmp(Cond::Lt), I32, false, Rep::Bool, "a < b"),
        (NodeOp::I32Cmp(Cond::Le), I32, false, Rep::Bool, "a <= b"),
        (NodeOp::I32Cmp(Cond::Gt), I32, false, Rep::Bool, "a > b"),
        (NodeOp::I32Cmp(Cond::Ge), I32, false, Rep::Bool, "a >= b"),
        (NodeOp::I32Cmp(Cond::Eq), I32, false, Rep::Bool, "a == b"),
        (NodeOp::I32Cmp(Cond::Ne), I32, false, Rep::Bool, "a != b"),
        (NodeOp::F64Cmp(Cond::Lt), F64, false, Rep::Bool, "a < b"),
        (NodeOp::F64Cmp(Cond::Le), F64, false, Rep::Bool, "a <= b"),
        (NodeOp::F64Cmp(Cond::Gt), F64, false, Rep::Bool, "a > b"),
        (NodeOp::F64Cmp(Cond::Ge), F64, false, Rep::Bool, "a >= b"),
        (NodeOp::F64Cmp(Cond::Eq), F64, false, Rep::Bool, "a == b"),
        (NodeOp::F64Cmp(Cond::Ne), F64, false, Rep::Bool, "a != b"),
        (NodeOp::BoolNot, F64, false, Rep::Bool, "!(a < b)"),
    ];
    let ints = [
        (0, 0),
        (7, 3),
        (-7, 3),
        (2147483647, 31),
        (-2147483648, 1),
        (-1, 33),
        (12345, -2),
        (5, 5),
    ];
    let floats = [
        "0, 0",
        "1.5, 0.25",
        "-0, 0",
        "NaN, 1",
        "1, NaN",
        "Infinity, -Infinity",
        "-7.5, 2",
        "7.5, -2",
        "1e308, 10",
        "3, 3",
    ];
    let source: String = (0..cases.len())
        .map(|k| format!("function op{k}(a, b) {{ return a + b; }} op{k}(1, 2);\n"))
        .collect();
    let mut engine = fixture(&source);
    for (k, &(ref op, operands, unary, out, expression)) in cases.iter().enumerate() {
        let graph = op_graph(op.clone(), operands, unary, out);
        install(&mut engine, &format!("op{k}"), &graph, Options::default());
        let inputs: Vec<String> = match operands {
            I32 => ints.iter().map(|(a, b)| format!("{a}, {b}")).collect(),
            _ => floats.iter().map(|s| s.to_string()).collect(),
        };
        for input in inputs {
            let (a, b) = input.split_once(", ").unwrap();
            let expected = eval(
                &mut engine,
                &format!("(function (a, b) {{ return {expression}; }})({a}, {b})"),
            );
            exits();
            let got = eval(&mut engine, &format!("op{k}({input})"));
            let exited = !exits().is_empty();
            if exited {
                // Only the int32 subtraction exits (on overflow): the VM adds as the bytecode does.
                assert_eq!(*op, NodeOp::I32Sub, "{op:?}({input}) exited");
                let sum = eval(&mut engine, &format!("({a}) + ({b})"));
                assert_eq!(got, sum, "{op:?}({input})");
            } else {
                assert_eq!(got, expected, "{op:?}({input}) as {expression}");
            }
        }
    }
    // >>> fits an int32 unless the result passes 2^31 - 1, where it exits.
    let mut engine = fixture("function ushr(a, b) { return a + b; } ushr(1, 2);");
    install(
        &mut engine,
        "ushr",
        &op_graph(NodeOp::I32UShr, I32, false, Rep::I32),
        Options::default(),
    );
    exits();
    assert_eq!(eval(&mut engine, "ushr(-8, 1)"), "2147483644");
    assert_eq!(eval(&mut engine, "ushr(-8, 33)"), "2147483644");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "ushr(-8, 0)"), "-8");
    assert_eq!(exits(), vec![(2, ExitReason::Overflow)]);
}

#[test]
fn to_boolean_answers_as_the_language_does_for_every_kind_of_word() {
    let mut engine = fixture(
        "function truthy(a, b) { return a + b; } truthy(1, 2);\n\
         function pick(a, b) { return a + b; } pick(1, 2);",
    );
    // `!!a`.
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let t = b.node(entry, NodeOp::ToBoolean, &[x], Rep::Bool, 2);
    let boxed = b.node(entry, NodeOp::BoolToTagged, &[t], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    install(&mut engine, "truthy", &b.g, Options::default());
    for word in [
        "0",
        "-0",
        "NaN",
        "1",
        "-1.5",
        "Infinity",
        "''",
        "'a'",
        "null",
        "undefined",
        "true",
        "false",
        "({})",
        "[]",
        "Symbol()",
        "0n",
        "1n",
        "-1n",
        "(function () {})",
        "new Boolean(false)",
        "Object(0n)",
    ] {
        assert_eq!(
            eval(&mut engine, &format!("truthy({word}, 0)")),
            eval(&mut engine, &format!("!!({word})")),
            "ToBoolean({word})"
        );
    }
    // `a ? b + 1 : b - 1`, with b held in a caller-saved register across ToBoolean's slow path
    // (a BigInt, or an object that is not IC-plain, calls the interpreter's ToBoolean).
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let then_block = b.block(2, false);
    let else_block = b.block(2, false);
    let merge = b.block(3, false);
    b.edge(entry, then_block);
    b.edge(entry, else_block);
    b.edge(then_block, merge);
    b.edge(else_block, merge);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    let fy = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[y],
        Rep::F64,
        2,
        at,
        ExitReason::NotNumber,
    );
    let one = b.node(entry, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 2);
    let t = b.node(entry, NodeOp::ToBoolean, &[x], Rep::Bool, 2);
    b.node(
        entry,
        NodeOp::Branch {
            then_block,
            else_block,
        },
        &[t],
        Rep::None,
        2,
    );
    let plus = b.node(then_block, NodeOp::F64Add, &[fy, one], Rep::F64, 2);
    let boxed_plus = b.node(then_block, NodeOp::F64ToTagged, &[plus], Rep::Tagged, 2);
    b.node(then_block, NodeOp::Jump(merge), &[], Rep::None, 2);
    let minus = b.node(else_block, NodeOp::F64Sub, &[fy, one], Rep::F64, 2);
    let boxed_minus = b.node(else_block, NodeOp::F64ToTagged, &[minus], Rep::Tagged, 2);
    b.node(else_block, NodeOp::Jump(merge), &[], Rep::None, 2);
    let r = b.node(
        merge,
        NodeOp::Phi,
        &[boxed_plus, boxed_minus],
        Rep::Tagged,
        3,
    );
    b.node(merge, NodeOp::Return, &[r], Rep::None, 3);
    install(&mut engine, "pick", &b.g, Options::default());
    for (word, expected) in [
        ("0n", "4"),
        ("2n", "6"),
        ("Object(0n)", "6"),
        ("''", "4"),
        ("'s'", "6"),
        ("NaN", "4"),
        ("null", "4"),
        ("({})", "6"),
    ] {
        assert_eq!(
            eval(&mut engine, &format!("pick({word}, 5)")),
            expected,
            "{word}"
        );
    }
}

#[test]
fn object_boolean_and_identity_checks_exit_on_other_words() {
    let mut engine = fixture(
        "function isobj(a, b) { return a + b; } isobj(1, 2);\n\
         function isbool(a, b) { return a + b; } isbool(1, 2);\n\
         function same(a, b) { return a + b; } same(1, 2);",
    );
    // `a` checked an object, returned.
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    b.exiting(
        entry,
        NodeOp::CheckObject,
        &[x],
        Rep::Cell,
        2,
        at,
        ExitReason::NotObject,
    );
    b.node(entry, NodeOp::Return, &[x], Rep::None, 3);
    install(&mut engine, "isobj", &b.g, Options::default());
    // `a` checked a Boolean, negated.
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    let bx = b.exiting(
        entry,
        NodeOp::CheckBoolean,
        &[x],
        Rep::Bool,
        2,
        at,
        ExitReason::NotBoolean,
    );
    let not = b.node(entry, NodeOp::BoolNot, &[bx], Rep::Bool, 2);
    let boxed = b.node(entry, NodeOp::BoolToTagged, &[not], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    install(&mut engine, "isbool", &b.g, Options::default());
    // Word identity, as `===` answers it on objects and Booleans.
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let same = b.node(entry, NodeOp::TaggedSame, &[x, y], Rep::Bool, 2);
    let boxed = b.node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    install(&mut engine, "same", &b.g, Options::default());
    exits();
    assert_eq!(eval(&mut engine, "var o = {}; isobj(o, 1) === o"), "true");
    assert_eq!(
        eval(&mut engine, "typeof isobj(function () {}, 1)"),
        "function"
    );
    assert_eq!(eval(&mut engine, "isbool(true, 0)"), "false");
    assert_eq!(eval(&mut engine, "isbool(false, 0)"), "true");
    assert_eq!(eval(&mut engine, "same(o, o)"), "true");
    assert_eq!(eval(&mut engine, "same(o, {})"), "false");
    assert_eq!(eval(&mut engine, "same(true, true)"), "true");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "isobj(1, 2)"), "3");
    assert_eq!(eval(&mut engine, "isobj('s', 2)"), "s2");
    assert_eq!(eval(&mut engine, "isbool(1, 2)"), "3");
    assert_eq!(eval(&mut engine, "isbool(null, 2)"), "2");
    assert_eq!(
        exits(),
        vec![
            (2, ExitReason::NotObject),
            (2, ExitReason::NotObject),
            (2, ExitReason::NotBoolean),
            (2, ExitReason::NotBoolean)
        ]
    );
}

#[test]
fn slots_flush_to_the_frame_and_reload_from_it() {
    // `add`: the doubled first operand stored to its slot, then an exit to pc 0, where the VM
    // reads the slot: 2a + b.
    let mut engine = fixture(ADD);
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at = b.state(0, &[], &[], &[]);
    let fx = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        0,
        at,
        ExitReason::NotNumber,
    );
    let twice = b.node(entry, NodeOp::F64Add, &[fx, fx], Rep::F64, 0);
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[twice], Rep::Tagged, 0);
    b.node(entry, NodeOp::StoreSlot(0), &[boxed], Rep::None, 0);
    b.exiting(
        entry,
        NodeOp::Exit(ExitReason::Unsupported),
        &[],
        Rep::None,
        0,
        at,
        ExitReason::Unsupported,
    );
    install(&mut engine, "add", &b.g, Options::default());
    assert_eq!(eval(&mut engine, "add(5, 1)"), "11");
    assert_eq!(eval(&mut engine, "add(0.25, 'x')"), "0.5x");
    // `callf`: a constant stored to x's slot before the call, which the call's effect leaves
    // alone, and reloaded after it: f(x) + 100.
    let mut engine =
        fixture("function callf(f, x) { return f(x) + 1; } callf(function (y) { return y; }, 1);");
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let f = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let x = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let hundred = b.node(entry, NodeOp::ConstTagged(num(100.0)), &[], Rep::Tagged, 0);
    b.node(entry, NodeOp::StoreSlot(1), &[hundred], Rep::None, 0);
    let at_call = b.state(2, &[f, x], &[], &[]);
    let call = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 2,
            pops: 2,
            pushes: 1,
        },
        &[f, x],
        Rep::None,
        2,
        at_call,
        ExitReason::Forced,
    );
    let r = b.node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 2);
    let reloaded = b.node(entry, NodeOp::FrameSlot(1), &[], Rep::Tagged, 3);
    let one = b.node(entry, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 3);
    let at_add = b.state(4, &[r, one], &[], &[]);
    let fr = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[r],
        Rep::F64,
        4,
        at_add,
        ExitReason::NotNumber,
    );
    let fh = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[reloaded],
        Rep::F64,
        4,
        at_add,
        ExitReason::NotNumber,
    );
    let sum = b.node(entry, NodeOp::F64Add, &[fr, fh], Rep::F64, 4);
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 4);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 5);
    install(&mut engine, "callf", &b.g, Options::default());
    assert_eq!(
        eval(&mut engine, "callf(function (y) { return y * 3; }, 7)"),
        "121"
    );
}

#[test]
fn the_perf_map_names_the_code_as_v8_names_maglevs() {
    let capture = crate::jit::profiler::PerfMapCapture::start("opt-a64");
    let mut engine = fixture(ADD);
    let code = install(&mut engine, "add", &add_f64(), Options::default());
    capture.stop();
    let lines = capture.lines();
    let (start, size, name) = lines
        .iter()
        .find(|(start, _, _)| *start == code.mem as usize)
        .expect("the code is listed");
    assert_eq!((*start, *size), (code.mem as usize, code.len));
    assert!(name.starts_with("JS:+add "), "{name}");
    assert_eq!(
        crate::jit::profiler::optimized_label("JS:^f a.js:1:2"),
        "JS:+f a.js:1:2"
    );
}

#[test]
fn malformed_graphs_are_refused() {
    let mut engine = fixture(COUNT);
    let chunk = chunk_of(&mut engine, "count");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let mut graph = count();
    let header = graph.blocks[1].body.remove(0);
    graph.nodes[header.index()].op = NodeOp::I32Not;
    assert_eq!(
        crate::opt::a64::compile_with(&chunk, &graph, &layout, &ilayout, Options::default())
            .err()
            .map(|refusal| refusal.reason),
        Some("a loop header without a poll")
    );
    let mut b = Build::new(3);
    let entry = b.block(0, false);
    let at = b.state(9, &[], &[], &[]);
    b.exiting(
        entry,
        NodeOp::Effect {
            pc: 9,
            pops: 1,
            pushes: 0,
        },
        &[],
        Rep::None,
        9,
        at,
        ExitReason::Forced,
    );
    assert!(
        crate::opt::a64::compile_with(&chunk, &b.g, &layout, &ilayout, Options::default()).is_err(),
        "an effect on a branch"
    );
}
