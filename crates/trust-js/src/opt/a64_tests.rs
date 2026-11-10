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
/// fills their caches, until a test installs the tier's code and moves to the JIT tier. The
/// switches are set whatever the environment's: the tier's driver stays off, so that no
/// checkpoint replaces the installed code with a second stage, and no exit is forced but the
/// tests' own (`Interp::opt_exit_every`), so that their exit counts hold.
fn fixture(source: &str) -> Engine {
    crate::opt::TestSwitches {
        on: false,
        eager: false,
        exit_every: 0,
        ..Default::default()
    }
    .set();
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

/// `compile_with` on a graph built by hand, with the chunk's own Cfg and initialized reads, as a
/// build hands them over.
fn compile_graph(
    chunk: &Chunk,
    graph: &Graph,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    options: Options,
) -> Result<Rc<JitCode>, Refusal> {
    let cfg = crate::jit_ir::Cfg::build(chunk).expect("the chunk's stack depths settle");
    let initialized = cfg.initialized_reads(chunk.jit_ops(), true);
    crate::opt::a64::compile_with(chunk, graph, &cfg, initialized, layout, ilayout, options)
}

/// Generate `graph`'s code for `name`'s chunk, install it as the chunk's native code and move the
/// engine to the JIT tier, whose entries then run it.
fn install(engine: &mut Engine, name: &str, graph: &Graph, options: Options) -> Rc<JitCode> {
    let chunk = chunk_of(engine, name);
    let alloc = crate::opt::regalloc::allocate(graph, &crate::opt::regalloc::ARM64).unwrap();
    crate::opt::regalloc::tests::assert_consistent(graph, &alloc);
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = compile_graph(&chunk, graph, &layout, &ilayout, options)
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

/// `f(a, b)` (LoadLocal 0, LoadLocal 1, a bitwise operation at pc 2, Return) on Numbers: both
/// words checked at the operation's pc and truncated to their ToInt32, then `op`: an int32 boxed
/// for `I32Or`, the uint32 Number for `I32UShrToF64`.
fn truncated(op: NodeOp) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at_op = b.state(2, &[x, y], &[], &[]);
    let int = |b: &mut Build, v| {
        let f = b.exiting(
            entry,
            NodeOp::CheckNumber,
            &[v],
            Rep::F64,
            2,
            at_op,
            ExitReason::NotNumber,
        );
        b.node(entry, NodeOp::TruncateToInt32, &[f], Rep::I32, 2)
    };
    let ix = int(&mut b, x);
    let iy = int(&mut b, y);
    let boxed = if op == NodeOp::I32UShrToF64 {
        let result = b.node(entry, op, &[ix, iy], Rep::F64, 2);
        b.node(entry, NodeOp::F64ToTagged, &[result], Rep::Tagged, 2)
    } else {
        let result = b.node(entry, op, &[ix, iy], Rep::I32, 2);
        b.node(entry, NodeOp::I32ToTagged, &[result], Rep::Tagged, 2)
    };
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

/// JavaScript source for `x`, which parses back to it.
fn js_number(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else if x == 0.0 && x.is_sign_negative() {
        "-0".into()
    } else {
        format!("({x:?})")
    }
}

#[test]
fn truncation_is_to_int32_with_and_without_fjcvtzs() {
    let two = |e: i32| 2f64.powi(e);
    // In the int32 range and past it; fractions, -0, NaN and the infinities; past 2^63, where
    // a conversion to 64 bits saturates and the low word comes from the mantissa (-2^63
    // converts exactly to INT64_MIN, which reads as saturated); 2^84 and above, whose low word
    // is zero.
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.9,
        -1.9,
        0.5,
        two(31) - 1.0,
        -two(31),
        two(31),
        -two(31) - 1.0,
        two(32) - 1.0,
        two(32),
        two(32) + 1.5,
        1e10,
        -1e10,
        two(53),
        two(53) + 2.0,
        1e18,
        two(63),
        -two(63),
        two(63) + two(11),
        -two(63) - two(11),
        two(64) + two(12),
        two(63) + two(62),
        two(83) + two(31),
        -two(83) - two(31),
        two(84),
        1e20,
        -1e20,
        1e300,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for no_jscvt in [false, true] {
        let mut engine = fixture(
            "function or(a, b) { return a | b; } or(1, 2);
             function ushr(a, b) { return a >>> b; } ushr(1, 2);",
        );
        for (name, op) in [("or", Op::BitOr), ("ushr", Op::UShr)] {
            let chunk = chunk_of(&mut engine, name);
            assert_eq!(chunk.jit_ops()[2], op, "{name}: {:?}", chunk.jit_ops());
        }
        let options = Options {
            no_jscvt,
            ..Options::default()
        };
        install(&mut engine, "or", &truncated(NodeOp::I32Or), options);
        install(
            &mut engine,
            "ushr",
            &truncated(NodeOp::I32UShrToF64),
            options,
        );
        exits();
        for x in values {
            let source = js_number(x);
            let int = crate::eval::to_int32(x);
            let context = format!("{source} without FJCVTZS: {no_jscvt}");
            assert_eq!(
                eval(&mut engine, &format!("or({source}, 0)")),
                int.to_string(),
                "{context}"
            );
            assert_eq!(
                eval(&mut engine, &format!("ushr({source}, 0)")),
                (int as u32).to_string(),
                "{context}"
            );
        }
        assert_eq!(eval(&mut engine, "or(-1.5, 2147483648)"), "-1");
        assert_eq!(eval(&mut engine, "ushr(-1, 28)"), "15");
        assert_eq!(eval(&mut engine, "ushr(-1.5, 32)"), "4294967295");
        assert!(exits().is_empty(), "Numbers never exit");
        assert_eq!(eval(&mut engine, "or('3', 4)"), "7");
        assert_eq!(exits(), vec![(2, ExitReason::NotNumber)]);
    }
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
        Options {
            forced_exits: true,
            ..Options::default()
        },
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
        Options {
            forced_exits: true,
            ..Options::default()
        },
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
    let code = compile_graph(&inlined, &via_leaf(), &layout, &ilayout, Options::default()).unwrap();
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
        Options {
            forced_exits: true,
            ..Options::default()
        },
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
        // ToBoolean of an untagged phi's f64 (stage 2 step 6): false for ±0 and NaN.
        (NodeOp::F64ToBoolean, F64, true, Rep::Bool, "!!a"),
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
        compile_graph(&chunk, &graph, &layout, &ilayout, Options::default())
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
        compile_graph(&chunk, &b.g, &layout, &ilayout, Options::default()).is_err(),
        "an effect on a branch"
    );
}

#[test]
fn a_code_past_its_exit_limit_is_discarded_once_for_the_template_jits() {
    // The tier off (the fixture's switches): the discard is stage 1's fallback, at 100 exits.
    let mut engine = fixture(ADD);
    let code = install(&mut engine, "add", &add_f64(), Options::default());
    let chunk = chunk_of(&mut engine, "add");
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "var s = ''; for (var k = 0; k < 150; k++) s = add('', k % 10); s"
        ),
        "9"
    );
    assert_eq!(exits().len(), crate::opt::EXIT_LIMIT as usize);
    let now = chunk.jit.get().flatten().expect("the template JIT's code");
    assert!(!Rc::ptr_eq(&now, &code));
    assert_eq!(crate::opt::exit::table_bytes(now.mem_ptr()), 0);
    assert!(
        engine
            .interp
            .opt_retired
            .iter()
            .any(|retired| Rc::ptr_eq(retired, &code)),
        "retired, not freed"
    );
}

#[test]
fn a_discard_whose_template_compile_waits_for_capacity_leaves_the_slot_cold() {
    // The executable budget is process-wide: hold all of it in a process of its own, as
    // `jit::executable_code_budget_tests` does, rather than steal it from concurrent tests.
    const CHILD: &str = "TRUST_JS_TEST_OPT_DEFERRED_DISCARD_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::opt_a64::tests::\
                 a_discard_whose_template_compile_waits_for_capacity_leaves_the_slot_cold",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use std::sync::atomic::Ordering::Relaxed;
    let mut engine = fixture(ADD);
    let code = install(&mut engine, "add", &add_f64(), Options::default());
    let chunk = chunk_of(&mut engine, "add");
    exits();
    let budget = crate::jit::executable_code_budget();
    let held =
        crate::jit::ExecutableCodeReservation::try_new(budget, budget.remaining.load(Relaxed))
            .expect("the whole budget");
    // The 100th exit discards the code, whose template compile finds no capacity: the slot goes
    // cold, not unavailable, and the VM runs the calls meanwhile.
    assert_eq!(
        eval(
            &mut engine,
            "var s = ''; for (var k = 0; k < 150; k++) s = add('', k % 10); s"
        ),
        "9"
    );
    assert_eq!(exits().len(), crate::opt::EXIT_LIMIT as usize);
    assert!(chunk.jit.get().is_none(), "a cold slot");
    assert!(chunk.jit_budget_wait_bytes.get() > 0);
    drop(held);
    // Capacity again: the next entry compiles the template JIT's code.
    assert_eq!(eval(&mut engine, "add(1, 2) + add('a', 'b')"), "3ab");
    let now = chunk
        .jit
        .get()
        .flatten()
        .expect("compiled at its next entry");
    assert!(!Rc::ptr_eq(&now, &code));
    assert_eq!(crate::opt::exit::table_bytes(now.mem_ptr()), 0);
    assert!(exits().is_empty());
}

#[test]
fn element_access_encodings_match_the_assembler() {
    // `llvm-mc --triple=aarch64 --show-encoding`: the w index register zero-extended and scaled.
    let cases: [(&str, u32, fn(&mut asm::Asm)); 1] =
        [("ldr x14, [x15, w9, uxtw #3]", 0xf86959ee, |a| {
            a.ldr_x_uxtw3(14, 15, 9)
        })];
    for (text, word, emit) in cases {
        let mut a = asm::Asm::new();
        emit(&mut a);
        assert_eq!(a.finish(), vec![word], "{text}");
    }
}

/// `get(a, i)` (LoadLocal 1, GetElemLocal 0 at pc 1, Return): `a` checked an Array, `i` a Number
/// and an index, the words and their used length loaded, the bounds checked and the element read
/// by `load` (LoadElement(_) or LoadElementOrUndefined), which the graph returns.
const GET: &str = "function get(a, i) { return a[i]; } get([1, 2], 0);";

fn get(load: NodeOp, kind: ElemKind) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let a = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let i = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let at_get = b.state(1, &[i], &[], &[]);
    let cell = b.exiting(
        entry,
        NodeOp::CheckObject,
        &[a],
        Rep::Cell,
        1,
        at_get,
        ExitReason::NotObject,
    );
    b.exiting(
        entry,
        NodeOp::CheckElements(kind),
        &[cell],
        Rep::None,
        1,
        at_get,
        ExitReason::WrongElements,
    );
    let number = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[i],
        Rep::F64,
        1,
        at_get,
        ExitReason::NotNumber,
    );
    let index = b.exiting(
        entry,
        NodeOp::CheckIndex,
        &[number],
        Rep::I32,
        1,
        at_get,
        ExitReason::NotInt32,
    );
    let words = b.node(entry, NodeOp::LoadElements, &[cell], Rep::Ptr, 1);
    let used = b.node(entry, NodeOp::LoadElementsUsed, &[words], Rep::I32, 1);
    let value = if load == NodeOp::LoadElementOrUndefined {
        b.exiting(
            entry,
            load,
            &[cell, words, index, used],
            Rep::Tagged,
            1,
            at_get,
            ExitReason::OutOfBounds,
        )
    } else {
        b.exiting(
            entry,
            NodeOp::CheckBounds,
            &[index, used],
            Rep::None,
            1,
            at_get,
            ExitReason::OutOfBounds,
        );
        b.exiting(
            entry,
            load,
            &[cell, words, index],
            Rep::Tagged,
            1,
            at_get,
            ExitReason::Hole,
        )
    };
    b.node(entry, NodeOp::Return, &[value], Rep::None, 2);
    b.g
}

fn reasons() -> Vec<ExitReason> {
    exits().into_iter().map(|(_, reason)| reason).collect()
}

#[test]
fn an_element_read_checks_its_receiver_key_and_bounds_and_exits_on_a_hole() {
    let mut engine = fixture(GET);
    install(
        &mut engine,
        "get",
        &get(NodeOp::LoadElement(Holes::Exit), ElemKind::Array),
        Options::default(),
    );
    exits();
    assert_eq!(eval(&mut engine, "get([1, 2, 3], 1)"), "2");
    assert_eq!(eval(&mut engine, "get([1.5, 'x', {}], 1)"), "x");
    assert_eq!(eval(&mut engine, "get([7, 8], -0)"), "7", "-0 is index 0");
    assert_eq!(
        eval(
            &mut engine,
            "var big = []; for (var k = 0; k < 300; k++) big.push(k); get(big, 299)"
        ),
        "299"
    );
    assert!(exits().is_empty());
    // Each failure resumes in the VM at the read, which reads as the language does.
    for (call, result, reason) in [
        ("get([1, , 3], 1)", "undefined", ExitReason::Hole),
        ("get([1, 2], 2)", "undefined", ExitReason::OutOfBounds),
        ("get([1, 2], -1)", "undefined", ExitReason::OutOfBounds),
        ("get([1, 2], 1.5)", "undefined", ExitReason::NotInt32),
        ("get([1, 2], 2147483648)", "undefined", ExitReason::NotInt32),
        ("get([1, 2], '1')", "2", ExitReason::NotNumber),
        ("get({0: 5}, 0)", "5", ExitReason::WrongElements),
        (
            "get(new Uint8Array([7, 8]), 1)",
            "8",
            ExitReason::WrongElements,
        ),
        ("get('abc', 1)", "b", ExitReason::NotObject),
    ] {
        assert_eq!(eval(&mut engine, call), result, "{call}");
        assert_eq!(reasons(), vec![reason], "{call}");
    }
    // Ordinary objects' words, under the Words check.
    let mut engine = fixture(GET);
    install(
        &mut engine,
        "get",
        &get(NodeOp::LoadElement(Holes::Exit), ElemKind::Words),
        Options::default(),
    );
    exits();
    assert_eq!(eval(&mut engine, "get({0: 'a', 1: 'b'}, 1)"), "b");
    assert_eq!(eval(&mut engine, "get([4, 5], 0)"), "4");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "get(new Proxy([1], {}), 0)"), "1");
    assert_eq!(reasons(), vec![ExitReason::WrongElements]);
}

#[test]
fn a_hole_reads_undefined_while_no_element_can_come_from_the_prototype_chain() {
    let mut engine = fixture(GET);
    install(
        &mut engine,
        "get",
        &get(NodeOp::LoadElement(Holes::Undefined), ElemKind::Array),
        Options::default(),
    );
    stress(&engine);
    exits();
    assert_eq!(eval(&mut engine, "get([1, , 3], 1)"), "undefined");
    assert_eq!(eval(&mut engine, "get([1, , 3], 2)"), "3");
    assert!(exits().is_empty());
    // A marked prototype's change lapses the protector's proof: the cold path renews it.
    assert_eq!(
        eval(
            &mut engine,
            "Array.prototype.extra = 1; delete Array.prototype.extra; get([1, , 3], 1)"
        ),
        "undefined"
    );
    assert!(exits().is_empty(), "the proof was renewed in place");
    // An index on Array.prototype, an own element among the named entries, or another
    // prototype: the VM reads it.
    assert_eq!(
        eval(
            &mut engine,
            "Object.defineProperty(Array.prototype, 1, \
             { get() { return 'inherited'; }, configurable: true }); get([1, , 3], 1)"
        ),
        "inherited"
    );
    assert_eq!(reasons(), vec![ExitReason::Hole]);
    assert_eq!(
        eval(&mut engine, "delete Array.prototype[1]; get([1, , 3], 1)"),
        "undefined"
    );
    assert!(exits().is_empty());
    assert_eq!(
        eval(
            &mut engine,
            "var far = [1, , 3]; Object.defineProperty(far, 1, { get() { return 'own'; } }); \
             get(far, 1)"
        ),
        "own"
    );
    assert_eq!(
        eval(
            &mut engine,
            "get(Object.setPrototypeOf([1, , 3], { 1: 'proto' }), 1)"
        ),
        "proto"
    );
    assert_eq!(reasons(), vec![ExitReason::Hole, ExitReason::Hole]);
}

#[test]
fn a_read_past_the_used_words_reads_undefined_and_a_negative_index_exits() {
    let mut engine = fixture(GET);
    install(
        &mut engine,
        "get",
        &get(NodeOp::LoadElementOrUndefined, ElemKind::Array),
        Options::default(),
    );
    exits();
    assert_eq!(eval(&mut engine, "get([1, 2], 1)"), "2");
    assert_eq!(eval(&mut engine, "get([1, 2], 2)"), "undefined");
    assert_eq!(eval(&mut engine, "get([1, 2], 2147483647)"), "undefined");
    assert_eq!(eval(&mut engine, "get([], 0)"), "undefined");
    assert_eq!(eval(&mut engine, "get(new Array(5), 3)"), "undefined");
    assert_eq!(eval(&mut engine, "get([1, , 3], 1)"), "undefined");
    assert!(exits().is_empty());
    assert_eq!(
        eval(&mut engine, "var neg = [1]; neg[-1] = 'n'; get(neg, -1)"),
        "n"
    );
    assert_eq!(reasons(), vec![ExitReason::OutOfBounds]);
    assert_eq!(
        eval(
            &mut engine,
            "Object.defineProperty(Array.prototype, 7, \
             { get() { return 'seven'; }, configurable: true }); \
             var r = get([1, 2], 7) + get([1, , 3], 1); delete Array.prototype[7]; r"
        ),
        "sevenundefined"
    );
    // The proof fails for the hole too, which exits with the hole's reason.
    assert_eq!(reasons(), vec![ExitReason::OutOfBounds, ExitReason::Hole]);
}

/// `set(a, i, v)` (LoadLocal 1, LoadLocal 2, SetElemLocalDrop 0 at pc 2, LoadLocal 0, Return):
/// `a` checked an Array, `i` an index, and `v` stored as `value` takes it (a word, an int32 or an
/// f64), under `mode`; with Present, a read of a[i] first proves the element.
const SET: &str = "function set(a, i, v) { a[i] = v; return a; } set([1, 2], 0, 1);";

#[derive(Clone, Copy, PartialEq)]
enum Value3 {
    Word,
    Int32,
    F64,
}

fn set(value: Value3, mode: Store) -> Graph {
    let mut b = Build::new(3);
    let entry = b.block(0, false);
    let a = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let i = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let v = b.node(entry, NodeOp::EntrySlot(2), &[], Rep::Tagged, 0);
    let at_set = b.state(2, &[i, v], &[], &[]);
    let check = |b: &mut Build, op: NodeOp, inputs: &[NodeIx], rep: Rep, reason: ExitReason| {
        b.exiting(entry, op, inputs, rep, 2, at_set, reason)
    };
    let cell = check(
        &mut b,
        NodeOp::CheckObject,
        &[a],
        Rep::Cell,
        ExitReason::NotObject,
    );
    check(
        &mut b,
        NodeOp::CheckElements(ElemKind::Array),
        &[cell],
        Rep::None,
        ExitReason::WrongElements,
    );
    let fi = check(
        &mut b,
        NodeOp::CheckNumber,
        &[i],
        Rep::F64,
        ExitReason::NotNumber,
    );
    let index = check(
        &mut b,
        NodeOp::CheckIndex,
        &[fi],
        Rep::I32,
        ExitReason::NotInt32,
    );
    let stored = match value {
        Value3::Word => v,
        Value3::F64 => check(
            &mut b,
            NodeOp::CheckNumber,
            &[v],
            Rep::F64,
            ExitReason::NotNumber,
        ),
        Value3::Int32 => {
            let fv = check(
                &mut b,
                NodeOp::CheckNumber,
                &[v],
                Rep::F64,
                ExitReason::NotNumber,
            );
            check(
                &mut b,
                NodeOp::CheckInt32,
                &[fv],
                Rep::I32,
                ExitReason::NotInt32,
            )
        }
    };
    let words = b.node(entry, NodeOp::LoadElements, &[cell], Rep::Ptr, 2);
    if mode == Store::Present {
        let used = b.node(entry, NodeOp::LoadElementsUsed, &[words], Rep::I32, 2);
        check(
            &mut b,
            NodeOp::CheckBounds,
            &[index, used],
            Rep::None,
            ExitReason::OutOfBounds,
        );
        check(
            &mut b,
            NodeOp::LoadElement(Holes::Exit),
            &[cell, words, index],
            Rep::Tagged,
            ExitReason::Hole,
        );
    }
    check(
        &mut b,
        NodeOp::StoreElement(mode),
        &[cell, words, index, stored],
        Rep::Ptr,
        ExitReason::WrongElements,
    );
    b.node(entry, NodeOp::Return, &[a], Rep::None, 4);
    b.g
}

/// The kind bits of the global `name`'s element words, those a scan computes, and whether a scan
/// made them exact.
fn kind_bits(engine: &mut Engine, name: &str) -> (u32, u32, bool) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let object = value.as_obj().expect("an object");
    let object = object.borrow();
    let words = object.props.words().expect("element words");
    (words.kind(), words.computed_kind(), words.scanned())
}

#[test]
fn element_stores_write_each_representation_and_keep_the_kind_bits_exact() {
    use crate::value::word_block::{KIND_INT32, KIND_NUMBERS};
    for (value, mode) in [
        (Value3::Word, Store::Existing),
        (Value3::Int32, Store::Existing),
        (Value3::F64, Store::Existing),
        (Value3::Word, Store::Present),
        (Value3::Int32, Store::Present),
    ] {
        let mut engine = fixture(SET);
        install(&mut engine, "set", &set(value, mode), Options::default());
        stress(&engine);
        exits();
        let label = format!("{:?} {mode:?}", value as u8);
        // Int32s into int32 words keep INT32.
        assert_eq!(
            eval(
                &mut engine,
                "var ints = [1, 2, 3]; set(ints, 1, 7); ints.join()"
            ),
            "1,7,3",
            "{label}"
        );
        let (kind, computed, _) = kind_bits(&mut engine, "ints");
        assert_eq!(kind & !computed, 0, "{label}");
        assert_ne!(kind & KIND_INT32, 0, "{label}");
        if value != Value3::Int32 {
            // A fraction and -0 clear INT32 but keep NUMBERS.
            assert_eq!(
                eval(
                    &mut engine,
                    "set(ints, 0, 2.5); set(ints, 2, -0); ints[0] + ':' + Object.is(ints[2], -0)"
                ),
                "2.5:true",
                "{label}"
            );
            let (kind, computed, _) = kind_bits(&mut engine, "ints");
            assert_eq!(kind & !computed, 0, "{label}");
            assert_eq!(kind & (KIND_INT32 | KIND_NUMBERS), KIND_NUMBERS, "{label}");
        }
        if value == Value3::Word {
            // A reference clears NUMBERS; the array keeps the object through collections.
            assert_eq!(
                eval(
                    &mut engine,
                    "var nums = [1.5, 2.5]; set(nums, 1, {tag: 'n'}); \
                     var junk = []; for (var k = 0; k < 500; k++) junk.push([k]); nums[1].tag"
                ),
                "n",
                "{label}"
            );
            let (kind, computed, _) = kind_bits(&mut engine, "nums");
            assert_eq!(kind & !computed, 0, "{label}");
            assert_eq!(kind & KIND_NUMBERS, 0, "{label}");
            assert_eq!(
                eval(
                    &mut engine,
                    "set([1, 2], 0, 'str')[0] + set([1], 0, null)[0]"
                ),
                "strnull",
                "{label}"
            );
        }
        if value == Value3::F64 {
            assert_eq!(
                eval(&mut engine, "var nan = set([1], 0, 0 / 0)[0]; nan !== nan"),
                "true",
                "{label}"
            );
        }
        assert!(exits().is_empty(), "{label}");
    }
}

#[test]
fn a_store_copies_a_shared_literal_and_exits_where_the_vm_writes() {
    let mut engine = fixture(SET);
    install(
        &mut engine,
        "set",
        &set(Value3::Int32, Store::Existing),
        Options::default(),
    );
    stress(&engine);
    exits();
    // A constant literal's copies share one block: the write copies it first.
    assert_eq!(
        eval(
            &mut engine,
            "function lit() { return [1, 2, 3]; } var p = lit(), q = lit(); \
             set(p, 0, 9); set(p, 2, 8); p.join() + ':' + q.join() + ':' + lit().join()"
        ),
        "9,2,8:1,2,3:1,2,3"
    );
    // Sealed elements take overwrites.
    assert_eq!(
        eval(&mut engine, "set(Object.seal([1, 2]), 0, 5).join()"),
        "5,2"
    );
    assert!(exits().is_empty());
    for (call, result, reason) in [
        // Frozen elements take no write (a sloppy function's store does nothing).
        (
            "set(Object.freeze([1, 2]), 0, 5).join()",
            "1,2",
            ExitReason::WrongElements,
        ),
        ("set([1, 2], 2, 3).join()", "1,2,3", ExitReason::OutOfBounds),
        ("set([1, , 3], 1, 2).join()", "1,2,3", ExitReason::Hole),
        ("set([1, 2], -1, 3)[-1]", "3", ExitReason::OutOfBounds),
        ("set([1, 2], 0, 1.5).join()", "1.5,2", ExitReason::NotInt32),
    ] {
        assert_eq!(eval(&mut engine, call), result, "{call}");
        assert_eq!(reasons(), vec![reason], "{call}");
    }
}

#[test]
fn an_arrays_store_fills_holes_and_appends_in_line() {
    use crate::value::word_block::KIND_INT32;
    for value in [Value3::Word, Value3::Int32, Value3::F64] {
        let mut engine = fixture(SET);
        install(
            &mut engine,
            "set",
            &set(value, Store::Create),
            Options::default(),
        );
        stress(&engine);
        exits();
        let label = value as u8;
        for (call, result) in [
            // Appends, from the static empty block and across capacities.
            (
                "var a = []; for (var k = 0; k < 100; k++) set(a, k, k); \
                 a.length + ':' + a[0] + ':' + a[99]",
                "100:0:99",
            ),
            ("set([1, 2], 2, 3).join()", "1,2,3"),
            ("set([], 0, 4).join()", "4"),
            // Holes filled below the used words, and below `length`.
            ("set([1, , 3], 1, 2).join()", "1,2,3"),
            (
                "var h = new Array(5); set(h, 2, 6); h.length + ':' + h[2]",
                "5:6",
            ),
            (
                "var l = new Array(50); set(l, 0, 7); l.length + ':' + l[0] + ':' + (1 in l)",
                "50:7:false",
            ),
            // A shared literal's exact block grows into a private one.
            (
                "function lit() { return [1, 2]; } var p = lit(); set(p, 2, 3); \
                 p.join() + ':' + lit().join()",
                "1,2,3:1,2",
            ),
        ] {
            assert_eq!(eval(&mut engine, call), result, "{label}: {call}");
        }
        assert!(exits().is_empty(), "{label}");
        let (kind, computed, _) = kind_bits(&mut engine, "a");
        assert_eq!(kind & !computed, 0, "{label}");
        assert_ne!(kind & KIND_INT32, 0, "{label}: int32s appended keep INT32");
        if value != Value3::Int32 {
            assert_eq!(
                eval(&mut engine, "set(a, 100, 0.5); a.length"),
                "101",
                "{label}"
            );
            let (kind, computed, _) = kind_bits(&mut engine, "a");
            assert_eq!(kind & !computed, 0, "{label}");
            assert_eq!(kind & KIND_INT32, 0, "{label}");
        }
        if value == Value3::Word {
            assert_eq!(
                eval(
                    &mut engine,
                    "var o = []; for (var k = 0; k < 40; k++) set(o, k, {k: k}); \
                     var junk = []; for (var k = 0; k < 300; k++) junk.push([k]); o[39].k"
                ),
                "39",
                "{label}"
            );
        }
        assert!(exits().is_empty(), "{label}");
        // What the in-line path does not take, the VM does.
        for (call, result) in [
            ("set([1, 2], 5, 1).length", "6"),
            (
                "Object.isExtensible(set(Object.preventExtensions([1]), 1, 2))",
                "false",
            ),
            ("set(Object.freeze([1, 2]), 2, 3).length", "2"),
            (
                "class Sub extends Array {} set(Sub.from([1]), 1, 2).join()",
                "1,2",
            ),
            (
                "var nl = [1, 2]; Object.defineProperty(nl, 'length', { writable: false }); \
                 set(nl, 2, 3).length",
                "2",
            ),
            ("set({0: 1}, 1, 2)[1]", "2"),
        ] {
            assert_eq!(eval(&mut engine, call), result, "{label}: {call}");
        }
        let taken = reasons();
        assert!(
            taken.len() == 6
                && taken[0] == ExitReason::OutOfBounds
                && taken[1..]
                    .iter()
                    .all(|&reason| reason == ExitReason::WrongElements),
            "{label}: {taken:?}"
        );
        // An index setter on Array.prototype intercepts an append (OrdinarySet).
        assert_eq!(
            eval(
                &mut engine,
                "var seen = 0; Object.defineProperty(Array.prototype, 2, \
                 { set(v) { seen = v; }, configurable: true }); \
                 var r = set([1, 2], 2, 9); delete Array.prototype[2]; seen + ':' + r.length"
            ),
            "9:2",
            "{label}"
        );
        assert_eq!(reasons(), vec![ExitReason::WrongElements], "{label}");
    }
}

/// `rdi(f, o)` (LoadLocal 0, LoadLocal 1, Call(1) at pc 2, Pop, Const 0, GetElemLocal 1 at pc 5,
/// Return): `o` checked element words before the call, a proof the read past the used words after
/// it keeps, as a value fact.
fn rdi(chunk: &Chunk) -> Graph {
    assert!(
        matches!(chunk.jit_ops()[5], Op::GetElemLocal(1)),
        "{:?}",
        chunk.jit_ops()
    );
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let f = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let o = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let at_call = b.state(2, &[f, o], &[], &[]);
    let cell = b.exiting(
        entry,
        NodeOp::CheckObject,
        &[o],
        Rep::Cell,
        2,
        at_call,
        ExitReason::NotObject,
    );
    b.exiting(
        entry,
        NodeOp::CheckElements(ElemKind::Words),
        &[cell],
        Rep::None,
        2,
        at_call,
        ExitReason::WrongElements,
    );
    let call = b.exiting(
        entry,
        NodeOp::Effect {
            pc: 2,
            pops: 2,
            pushes: 1,
        },
        &[f, o],
        Rep::None,
        2,
        at_call,
        ExitReason::Throw,
    );
    b.node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 2);
    let zero = b.node(entry, NodeOp::ConstTagged(num(0.0)), &[], Rep::Tagged, 4);
    let at_get = b.state(5, &[zero], &[], &[]);
    let words = b.node(entry, NodeOp::LoadElements, &[cell], Rep::Ptr, 5);
    let used = b.node(entry, NodeOp::LoadElementsUsed, &[words], Rep::I32, 5);
    let index = b.node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 5);
    let value = b.exiting(
        entry,
        NodeOp::LoadElementOrUndefined,
        &[cell, words, index, used],
        Rep::Tagged,
        5,
        at_get,
        ExitReason::OutOfBounds,
    );
    b.node(entry, NodeOp::Return, &[value], Rep::None, 6);
    b.g
}

#[test]
fn a_host_index_put_on_an_object_the_code_proved_is_read_through_its_getter() {
    // The element check's proof outlives effects under the contract that no object JavaScript
    // has seen loses IC_PLAIN (ir.rs, ElemKind). A host that installs indexed properties on such
    // an object breaks it; the read past the used words, the one read its empty words answer,
    // tests IC_PLAIN again and exits, so its getter runs.
    let mut engine = fixture("function rdi(f, o) { f(o); return o[0]; } rdi(function () {}, [1]);");
    let index_onto = engine
        .interp
        .make_native("indexOnto", 1, |interp, _, args| {
            let getter = interp.make_native("at", 1, |_, _, _| Ok(Value::Num(42.0)));
            interp.install_readonly_indexed_properties(&args[0], 3, Value::Obj(getter))?;
            Ok(Value::Undefined)
        });
    engine.interp.global.borrow_mut().props.insert(
        "indexOnto",
        crate::value::Property::data(Value::Obj(index_onto), true, true, true),
    );
    let chunk = chunk_of(&mut engine, "rdi");
    install(&mut engine, "rdi", &rdi(&chunk), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "rdi(function () {}, {})"), "undefined");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "rdi(indexOnto, {})"), "42");
    assert_eq!(reasons(), vec![ExitReason::OutOfBounds]);
}

/// `len(a)` (GetPropLocal 0 `length` at pc 0, Return): `a` checked an Array, its length read.
fn len() -> Graph {
    let mut b = Build::new(1);
    let entry = b.block(0, false);
    let a = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let at_len = b.state(0, &[], &[], &[]);
    let cell = b.exiting(
        entry,
        NodeOp::CheckObject,
        &[a],
        Rep::Cell,
        0,
        at_len,
        ExitReason::NotObject,
    );
    b.exiting(
        entry,
        NodeOp::CheckElements(ElemKind::Array),
        &[cell],
        Rep::None,
        0,
        at_len,
        ExitReason::WrongElements,
    );
    let length = b.node(entry, NodeOp::LoadArrayLength, &[cell], Rep::Tagged, 0);
    b.node(entry, NodeOp::Return, &[length], Rep::None, 1);
    b.g
}

#[test]
fn an_arrays_length_is_its_first_slot() {
    let mut engine = fixture("function len(a) { return a.length; } len([1]);");
    install(&mut engine, "len", &len(), Options::default());
    exits();
    assert_eq!(eval(&mut engine, "len([1, 2, 3])"), "3");
    assert_eq!(eval(&mut engine, "len(new Array(50))"), "50");
    assert_eq!(eval(&mut engine, "var g = [1]; g.length = 7; len(g)"), "7");
    assert_eq!(
        eval(&mut engine, "class Sub extends Array {} len(Sub.of(1, 2))"),
        "2"
    );
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, "len({length: 5})"), "5");
    assert_eq!(eval(&mut engine, "len('four')"), "4");
    assert_eq!(
        reasons(),
        vec![ExitReason::WrongElements, ExitReason::NotObject]
    );
}

#[test]
fn edges_stored_into_large_arrays_mark_their_runs() {
    // An old array of more than 64 words keeps a remembered record: a minor collection traces
    // its dirty runs and the words past its traced length only, so an edge stored below that
    // length must mark its run, or the young value it names dies. A collection every 37 cells
    // leaves most stored values young (a collection at every allocation would age each one at
    // the call's safepoint before its store).
    let mut engine = fixture(SET);
    install(
        &mut engine,
        "set",
        &set(Value3::Word, Store::Existing),
        Options::default(),
    );
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(37);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "var big = []; for (var k = 0; k < 100; k++) big.push(k % 2 ? {k: k} : k);\n\
             var junk = []; for (var k = 0; k < 200; k++) junk.push([k]);\n\
             for (var r = 0; r < 4; r++) for (var k = 0; k < 33; k++) \
             set(big, k * 3, {tag: 'v' + k + r});\n\
             junk = []; for (var k = 0; k < 400; k++) junk.push([k]);\n\
             var ok = 0; for (var k = 0; k < 33; k++) if (big[k * 3].tag === 'v' + k + 3) ok++; ok"
        ),
        "33"
    );
    assert!(exits().is_empty());
}

/// `get(a, i)` (GET) on a typed array: `a` checked a typed array of `kind` (of any kind for the
/// template's stub where `kind` is None), `i` an index, then the element by its kind, its bounds
/// checked, or undefined past them where `past`; the graph returns it.
fn typed_get(kind: Option<TypedKind>, past: bool) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let a = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let i = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let at_get = b.state(1, &[i], &[], &[]);
    let check = |b: &mut Build, op: NodeOp, inputs: &[NodeIx], rep: Rep, reason: ExitReason| {
        b.exiting(entry, op, inputs, rep, 1, at_get, reason)
    };
    let cell = check(
        &mut b,
        NodeOp::CheckObject,
        &[a],
        Rep::Cell,
        ExitReason::NotObject,
    );
    check(
        &mut b,
        NodeOp::CheckElements(ElemKind::Typed(kind)),
        &[cell],
        Rep::None,
        ExitReason::WrongElements,
    );
    let number = check(
        &mut b,
        NodeOp::CheckNumber,
        &[i],
        Rep::F64,
        ExitReason::NotNumber,
    );
    let index = check(
        &mut b,
        NodeOp::CheckIndex,
        &[number],
        Rep::I32,
        ExitReason::NotInt32,
    );
    let value = match kind {
        None => check(
            &mut b,
            NodeOp::TypedElement { write: false },
            &[cell, index],
            Rep::Tagged,
            ExitReason::OutOfBounds,
        ),
        Some(kind) => {
            let view = b.node(entry, NodeOp::TypedView, &[cell], Rep::Ptr, 1);
            let length = b.node(entry, NodeOp::TypedLength, &[view], Rep::I32, 1);
            if !past {
                check(
                    &mut b,
                    NodeOp::CheckBounds,
                    &[index, length],
                    Rep::None,
                    ExitReason::OutOfBounds,
                );
            }
            let data = check(
                &mut b,
                NodeOp::TypedData { write: false },
                &[cell, view],
                Rep::Ptr,
                ExitReason::Detached,
            );
            if past {
                let op = NodeOp::LoadTypedOrUndefined(kind);
                b.node(entry, op, &[cell, data, index, length], Rep::Tagged, 1)
            } else {
                let rep = kind.rep(false);
                let op = NodeOp::LoadTyped(kind);
                let element = b.node(entry, op, &[cell, data, index], rep, 1);
                let boxing = match rep {
                    Rep::I32 => NodeOp::I32ToTagged,
                    _ => NodeOp::F64ToTagged,
                };
                b.node(entry, boxing, &[element], Rep::Tagged, 1)
            }
        }
    };
    b.node(entry, NodeOp::Return, &[value], Rep::None, 2);
    b.g
}

/// Each element kind's constructor and four values that test its conversions.
const TYPED_KINDS: [(TypedKind, &str, &str); 9] = [
    (TypedKind::U8, "Uint8Array", "0, 255, 128, 1"),
    (TypedKind::I8, "Int8Array", "-128, 127, -1, 0"),
    (TypedKind::U16, "Uint16Array", "65535, 0, 32768, 1"),
    (TypedKind::I16, "Int16Array", "-32768, 32767, -1, 0"),
    (
        TypedKind::U32,
        "Uint32Array",
        "4294967295, 0, 2147483648, 1",
    ),
    (
        TypedKind::I32,
        "Int32Array",
        "-2147483648, 2147483647, -1, 0",
    ),
    (TypedKind::F32, "Float32Array", "1.1, -0, NaN, 3.4e38"),
    (TypedKind::F64, "Float64Array", "1.1, -0, NaN, -Infinity"),
    (TypedKind::U8Clamped, "Uint8ClampedArray", "0, 255, 128, 1"),
];

/// What the bytecode VM answers to `source` in a fixture of `fixture_source`.
fn vm_answer(fixture_source: &str, source: &str) -> String {
    let mut engine = fixture(fixture_source);
    eval(&mut engine, source)
}

/// `readAll(t)`: every element of `t` through `get`, -0 as such, joined.
const READ_ALL: &str = "function readAll(t) { var r = []; \
     for (var k = 0; k < t.length; k++) { var x = get(t, k); \
     r.push(Object.is(x, -0) ? '-0' : String(x)); } return r.join(); }";

#[test]
fn typed_elements_load_by_their_kind_as_the_vm_reads_them() {
    for (kind, ctor, values) in TYPED_KINDS {
        for past in [false, true] {
            let label = format!("{ctor} past {past}");
            let mut engine = fixture(GET);
            eval(&mut engine, READ_ALL);
            install(
                &mut engine,
                "get",
                &typed_get(Some(kind), past),
                Options::default(),
            );
            stress(&engine);
            exits();
            // The values, then a view one element into its buffer.
            for read in [
                format!("var t = new {ctor}([{values}]); readAll(t)"),
                format!(
                    "readAll(new {ctor}(new {ctor}([{values}, {values}]).buffer, \
                     {ctor}.BYTES_PER_ELEMENT, 5))"
                ),
            ] {
                let expected = vm_answer(GET, &format!("{READ_ALL} {read}"));
                assert_eq!(eval(&mut engine, &read), expected, "{label}: {read}");
            }
            assert!(exits().is_empty(), "{label}");
            // Past the length and a negative index: undefined, in line where `past`.
            assert_eq!(
                eval(
                    &mut engine,
                    "get(t, 4) + ':' + get(t, -1) + ':' + get(t, 2147483647)"
                ),
                "undefined:undefined:undefined",
                "{label}"
            );
            let expect = if past {
                vec![]
            } else {
                vec![ExitReason::OutOfBounds; 3]
            };
            assert_eq!(reasons(), expect, "{label}");
            // Another kind, a view the native code declines, an Array, an ordinary object.
            let other = if kind == TypedKind::F64 {
                "Float32Array"
            } else {
                "Float64Array"
            };
            for (call, result) in [
                (format!("get(new {other}([2.5]), 0)"), "2.5"),
                ("String(get(new BigInt64Array([5n]), 0))".to_string(), "5"),
                ("get([5], 0)".to_string(), "5"),
                ("get({0: 6}, 0)".to_string(), "6"),
            ] {
                assert_eq!(eval(&mut engine, &call), result, "{label}: {call}");
                assert_eq!(
                    reasons(),
                    vec![ExitReason::WrongElements],
                    "{label}: {call}"
                );
            }
        }
    }
}

#[test]
fn a_float_elements_nan_is_read_as_the_canonical_nan() {
    // A buffer's bytes may hold any NaN, and every NaN but one is a tag in the boxing (the first
    // here is the hole's).
    for past in [false, true] {
        for (kind, setup) in [
            (
                TypedKind::F64,
                "var t = new Float64Array(2), u = new Uint32Array(t.buffer); \
                 u[1] = 0x7ffa0000; u[3] = 0xfff40000; u[2] = 1;",
            ),
            (
                TypedKind::F32,
                "var t = new Float32Array(2), u = new Uint32Array(t.buffer); \
                 u[0] = 0x7fa00001; u[1] = 0xffc00000;",
            ),
        ] {
            let mut engine = fixture(GET);
            install(
                &mut engine,
                "get",
                &typed_get(Some(kind), past),
                Options::default(),
            );
            exits();
            let read = format!(
                "{setup} var x = get(t, 0), y = get(t, 1); \
                 [x !== x, y !== y, typeof x, typeof y, [x, y].indexOf(x)].join()"
            );
            assert_eq!(
                eval(&mut engine, &read),
                "true,true,number,number,-1",
                "{kind:?} {past}"
            );
            assert!(exits().is_empty());
        }
    }
}

#[test]
fn a_detached_or_shrunk_buffer_exits_where_the_vm_reads_undefined() {
    for past in [false, true] {
        let mut engine = fixture(GET);
        install(
            &mut engine,
            "get",
            &typed_get(Some(TypedKind::F64), past),
            Options::default(),
        );
        exits();
        assert_eq!(
            eval(
                &mut engine,
                "var d = new Float64Array([1.5, 2.5]); var before = get(d, 1); \
                 d.buffer.transfer(); before + ':' + get(d, 1)"
            ),
            "2.5:undefined"
        );
        assert_eq!(reasons(), vec![ExitReason::Detached], "{past}");
        // A fixed view past a shrunk buffer's end reads nothing until the buffer grows back.
        assert_eq!(
            eval(
                &mut engine,
                "var rab = new ArrayBuffer(32, {maxByteLength: 64}); \
                 var f = new Float64Array(rab, 8, 2); f[0] = 2; f[1] = 3; var a1 = get(f, 1); \
                 rab.resize(16); var a2 = get(f, 0); rab.resize(64); \
                 [a1, a2, get(f, 0), get(f, 1)].join()"
            ),
            "3,,2,0"
        );
        assert_eq!(reasons(), vec![ExitReason::Detached], "{past}");
        // A length-tracking view has no element count: a read exits at the bounds or, past
        // them, at the buffer's end, which none reaches.
        assert_eq!(
            eval(
                &mut engine,
                "var tracking = new Float64Array(new ArrayBuffer(16, {maxByteLength: 32})); \
                 tracking[1] = 4; get(tracking, 1)"
            ),
            "4"
        );
        let reason = if past {
            ExitReason::Detached
        } else {
            ExitReason::OutOfBounds
        };
        assert_eq!(reasons(), vec![reason], "{past}");
    }
}

#[test]
fn the_template_stub_reads_any_kind_and_exits_where_it_declines() {
    let mut engine = fixture(GET);
    eval(&mut engine, READ_ALL);
    install(
        &mut engine,
        "get",
        &typed_get(None, false),
        Options::default(),
    );
    stress(&engine);
    exits();
    for (_, ctor, values) in TYPED_KINDS {
        let read = format!("readAll(new {ctor}([{values}]))");
        let expected = vm_answer(GET, &format!("{READ_ALL} {read}"));
        assert_eq!(eval(&mut engine, &read), expected, "{ctor}");
    }
    assert!(exits().is_empty());
    for (call, result, reason) in [
        // One status for every decline: the bounds, a detached buffer, a declined view.
        (
            "get(new Uint8Array(2), 2)",
            "undefined",
            ExitReason::OutOfBounds,
        ),
        (
            "var gone = new Int16Array(2); gone.buffer.transfer(); get(gone, 0)",
            "undefined",
            ExitReason::OutOfBounds,
        ),
        (
            "String(get(new BigInt64Array([5n]), 0))",
            "5",
            ExitReason::OutOfBounds,
        ),
        ("get([5], 0)", "5", ExitReason::WrongElements),
    ] {
        assert_eq!(eval(&mut engine, call), result, "{call}");
        assert_eq!(reasons(), vec![reason], "{call}");
    }
}

/// `set(a, i, v)` (SET) on a typed array: `a` checked a typed array of `kind` (of any kind for the
/// stub where `kind` is None), `i` an index and `v` a Number, converted as `kind` stores it.
fn typed_set(kind: Option<TypedKind>) -> Graph {
    let mut b = Build::new(3);
    let entry = b.block(0, false);
    let a = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let i = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let v = b.node(entry, NodeOp::EntrySlot(2), &[], Rep::Tagged, 0);
    let at_set = b.state(2, &[i, v], &[], &[]);
    let check = |b: &mut Build, op: NodeOp, inputs: &[NodeIx], rep: Rep, reason: ExitReason| {
        b.exiting(entry, op, inputs, rep, 2, at_set, reason)
    };
    let cell = check(
        &mut b,
        NodeOp::CheckObject,
        &[a],
        Rep::Cell,
        ExitReason::NotObject,
    );
    check(
        &mut b,
        NodeOp::CheckElements(ElemKind::Typed(kind)),
        &[cell],
        Rep::None,
        ExitReason::WrongElements,
    );
    let fi = check(
        &mut b,
        NodeOp::CheckNumber,
        &[i],
        Rep::F64,
        ExitReason::NotNumber,
    );
    let index = check(
        &mut b,
        NodeOp::CheckIndex,
        &[fi],
        Rep::I32,
        ExitReason::NotInt32,
    );
    let fv = check(
        &mut b,
        NodeOp::CheckNumber,
        &[v],
        Rep::F64,
        ExitReason::NotNumber,
    );
    match kind {
        None => {
            check(
                &mut b,
                NodeOp::TypedElement { write: true },
                &[cell, index, fv],
                Rep::None,
                ExitReason::OutOfBounds,
            );
        }
        Some(kind) => {
            let stored = match kind {
                TypedKind::F32 | TypedKind::F64 => fv,
                TypedKind::U8Clamped => b.node(entry, NodeOp::ClampToUint8, &[fv], Rep::I32, 2),
                _ => b.node(entry, NodeOp::TruncateToInt32, &[fv], Rep::I32, 2),
            };
            let view = b.node(entry, NodeOp::TypedView, &[cell], Rep::Ptr, 2);
            let length = b.node(entry, NodeOp::TypedLength, &[view], Rep::I32, 2);
            check(
                &mut b,
                NodeOp::CheckBounds,
                &[index, length],
                Rep::None,
                ExitReason::OutOfBounds,
            );
            let data = check(
                &mut b,
                NodeOp::TypedData { write: true },
                &[cell, view],
                Rep::Ptr,
                ExitReason::Detached,
            );
            let op = NodeOp::StoreTyped(kind);
            b.node(entry, op, &[cell, view, data, index, stored], Rep::None, 2);
        }
    }
    b.node(entry, NodeOp::Return, &[a], Rep::None, 4);
    b.g
}

/// `writeAll(t)`: values every kind's store converts (NaN, -0, the int32 bounds and past them,
/// fractions both ways, the infinities, halves between bytes, one past Float32's range) stored
/// through `set`, then `t`'s elements, -0 as such, joined.
const WRITE_ALL: &str = "function writeAll(t) { var v = [NaN, -0, 2147483648, 4294967301, -1, \
     1.9, -1.9, Infinity, -Infinity, 254.5, 255.5, 0.5, 1.5, 1e40, -2147483649, 65535.5]; \
     for (var k = 0; k < v.length; k++) set(t, k, v[k]); \
     return Array.from(t, function (x) { return Object.is(x, -0) ? '-0' : String(x); }).join(); }";

#[test]
fn typed_elements_store_each_kinds_conversion_as_the_vm_does() {
    let all = TYPED_KINDS.map(|(kind, ctor, _)| (Some(kind), ctor));
    for (kind, ctor) in [(None, "Uint8Array")].into_iter().chain(all) {
        let label = format!("{kind:?} {ctor}");
        let mut engine = fixture(SET);
        eval(&mut engine, WRITE_ALL);
        install(&mut engine, "set", &typed_set(kind), Options::default());
        stress(&engine);
        exits();
        let ctors: Vec<&str> = match kind {
            Some(_) => vec![ctor],
            None => TYPED_KINDS.iter().map(|&(_, ctor, _)| ctor).collect(),
        };
        for ctor in ctors {
            // The values, then a view four elements into its buffer.
            for write in [
                format!("writeAll(new {ctor}(16))"),
                format!(
                    "var whole = new {ctor}(20); writeAll(new {ctor}(whole.buffer, \
                     4 * {ctor}.BYTES_PER_ELEMENT, 16)); Array.from(whole).join()"
                ),
            ] {
                let expected = vm_answer(SET, &format!("{WRITE_ALL} {write}"));
                assert_eq!(eval(&mut engine, &write), expected, "{label}: {write}");
            }
        }
        // The stub declines values FCVTZS cannot take without FJCVTZS: the VM stores them.
        let taken = reasons();
        if kind.is_some() || crate::jit::jscvt_available() {
            assert!(taken.is_empty(), "{label}: {taken:?}");
        } else {
            assert!(
                taken.iter().all(|&r| r == ExitReason::OutOfBounds),
                "{label}"
            );
        }
        // The stub's one status, or the buffer's reason for a kind's store.
        let buffer = if kind.is_none() {
            ExitReason::OutOfBounds
        } else {
            ExitReason::Detached
        };
        let other = if kind == Some(TypedKind::F64) {
            "Float32Array"
        } else {
            "Float64Array"
        };
        let mut cases = vec![
            (
                format!("var short = new {ctor}(2); set(short, 2, 1); Array.from(short).join()"),
                ExitReason::OutOfBounds,
            ),
            (
                format!("set(new {ctor}(2), 0, '3')[0] + ''"),
                ExitReason::NotNumber,
            ),
            (
                format!("var gone = new {ctor}(2); gone.buffer.transfer(); set(gone, 0, 1).length"),
                buffer,
            ),
            (
                format!(
                    "var im = new {ctor}(new ArrayBuffer(8).transferToImmutable()); \
                     try {{ set(im, 0, 1); 'stored:' + im[0] }} catch (e) {{ e.constructor.name }}"
                ),
                buffer,
            ),
            (
                "set([1], 0, 2)[0] + ''".to_string(),
                ExitReason::WrongElements,
            ),
        ];
        if kind.is_some() {
            cases.push((
                format!("set(new {other}(1), 0, 2.5)[0] + ''"),
                ExitReason::WrongElements,
            ));
        }
        for (call, reason) in cases {
            let expected = vm_answer(SET, &call);
            assert_eq!(eval(&mut engine, &call), expected, "{label}: {call}");
            assert_eq!(reasons(), vec![reason], "{label}: {call}");
        }
    }
}

#[test]
fn typed_stores_keep_the_buffers_dirty_ranges_and_generation() {
    // Embedders mirror a buffer by its generation and dirty ranges: the stores fold into them
    // as the template's do (`emit_dirty`).
    let mut engine = fixture(SET);
    install(
        &mut engine,
        "set",
        &typed_set(Some(TypedKind::I16)),
        Options::default(),
    );
    exits();
    eval(
        &mut engine,
        "var buf = new ArrayBuffer(32); var t = new Int16Array(buf, 4, 8);",
    );
    let env = engine.interp.global_env;
    let buf = engine.interp.get_var("buf", &env).ok().expect("the buffer");
    engine.interp.take_array_buffer_dirty_ranges(&buf);
    let version = engine.interp.array_buffer_version(&buf).expect("attached");
    eval(&mut engine, "set(t, 1, 7); set(t, 3, 8);");
    assert!(exits().is_empty());
    assert_eq!(
        engine.interp.take_array_buffer_dirty_ranges(&buf),
        Some(vec![6..12])
    );
    assert_eq!(engine.interp.array_buffer_version(&buf), Some(version + 2));
    assert_eq!(eval(&mut engine, "Array.from(t).join()"), "0,7,0,8,0,0,0,0");
}

// ---- immediate operands (P3 stage 2 step 7) ----

/// The words of installed code.
fn code_words(code: &JitCode) -> Vec<u32> {
    let bytes = unsafe { std::slice::from_raw_parts(code.mem_ptr(), code.code_bytes()) };
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|&w| u32::from_le_bytes(w)).collect()
}

/// Whether `code` holds `word` in the bits `mask` keeps (its registers masked out).
fn holds(code: &[u32], word: u32, mask: u32) -> bool {
    code.iter().any(|&w| w & mask == word & mask)
}

/// JavaScript source for an int32.
fn js_int(x: i32) -> String {
    format!("({x})")
}

/// `f(a, b)` (LoadLocal 0, LoadLocal 1, the operation at pc 2, Return) as the int32 operation
/// `op` with the constant `c` for its left operand (`left`) or its right one: the other operand
/// is its argument's int32, checked at pc 2, where an exiting `op` exits too; the VM's operation
/// then reads the arguments, which the calls make `c` where the graph has the constant.
fn const_operand(op: NodeOp, c: i32, left: bool, out: Rep) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    let word = if left { y } else { x };
    let f = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[word],
        Rep::F64,
        2,
        at,
        ExitReason::NotNumber,
    );
    let v = b.exiting(
        entry,
        NodeOp::CheckInt32,
        &[f],
        Rep::I32,
        2,
        at,
        ExitReason::NotInt32,
    );
    let k = b.node(entry, NodeOp::ConstI32(c), &[], Rep::I32, 2);
    let inputs = if left { [k, v] } else { [v, k] };
    let r = if op.can_exit() {
        b.exiting(entry, op, &inputs, out, 2, at, ExitReason::Overflow)
    } else {
        b.node(entry, op, &inputs, out, 2)
    };
    let boxing = match out {
        Rep::I32 => NodeOp::I32ToTagged,
        Rep::F64 => NodeOp::F64ToTagged,
        _ => NodeOp::BoolToTagged,
    };
    let boxed = b.node(entry, boxing, &[r], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

/// The instruction, its registers aside, that takes the constant `c` of `op` (on the left where
/// `left`) as its immediate, and the mask of the bits it fixes; None where `c` stays in a
/// register. An independent model of the encodings (Arm A64): ADDS and SUBS (CMN and CMP, which
/// write XZR) with a 12-bit immediate or one shifted by 12, a negative constant by the opposite
/// operation; AND, ORR and EOR with a 32-bit logical immediate; LSL, LSR and ASR as UBFM and SBFM
/// by the count's low five bits.
fn imm_form(op: &NodeOp, c: i32, left: bool) -> Option<(u32, u32)> {
    let add_sub = |sub: bool| {
        let m = c.unsigned_abs();
        let field = if m < 1 << 12 {
            m << 10
        } else if m < 1 << 24 && m & 0xFFF == 0 {
            1 << 22 | (m >> 12) << 10
        } else {
            return None;
        };
        Some(
            if sub != (c < 0) {
                0x7100_0000
            } else {
                0x3100_0000
            } | field,
        )
    };
    let shift = |op: &NodeOp| {
        let s = c as u32 & 31;
        match op {
            NodeOp::I32Shl => 0x5300_0000 | ((32 - s) & 31) << 16 | (31 - s) << 10,
            NodeOp::I32Sar => 0x1300_0000 | s << 16 | 31 << 10,
            _ => 0x5300_0000 | s << 16 | 31 << 10,
        }
    };
    const REGISTERS: u32 = 0xFFFF_FC00;
    match op {
        NodeOp::I32Add => Some((add_sub(false)?, REGISTERS)),
        NodeOp::I32Sub if !left => Some((add_sub(true)?, REGISTERS)),
        // CMP or CMN, either side (a left constant reverses the condition): XZR's Rd.
        NodeOp::I32Cmp(_) => Some((add_sub(true)? | 31, 0xFFFF_FC1F)),
        NodeOp::I32And | NodeOp::I32Or | NodeOp::I32Xor => {
            let base = match op {
                NodeOp::I32And => 0x1200_0000,
                NodeOp::I32Or => 0x3200_0000,
                _ => 0x5200_0000,
            };
            let field = asm::logical_imm_w(c as u32)?;
            Some((base | field << 10, REGISTERS))
        }
        NodeOp::I32Shl | NodeOp::I32Sar | NodeOp::I32UShr | NodeOp::I32UShrToF64 if !left => {
            Some((shift(op), REGISTERS))
        }
        _ => None,
    }
}

/// The int32 nodes with a constant operand (P3 stage 2 step 7), on its right and on its left, at
/// each encoding's bounds and past them: the code takes an immediate exactly where the encoding
/// allows (a commutative operation's left constant swapped to the right, a comparison's with
/// its condition reversed, a negative add or subtract constant by the opposite operation, a
/// shift count's low five bits), keeps the register elsewhere, and computes as the language does
/// for operands at the edges of int32, its overflow exits taken with an immediate operand.
#[test]
fn int32_constants_take_the_immediate_where_it_encodes() {
    let add_sub: &[i32] = &[
        0,
        1,
        4095,
        4096,
        4097,
        0xFF_F000,
        0xFF_F001,
        0x100_0000,
        -1,
        -4095,
        -4096,
        -4097,
        -0xFF_F000,
        -0x100_0000,
        i32::MAX,
        i32::MIN,
    ];
    let logical: &[i32] = &[
        0x3FFF,
        0xFFF_FFFF,
        0x12345,
        0,
        -1,
        -8,
        0x5555_5555,
        i32::MIN,
        i32::MAX,
        1,
        5,
    ];
    let counts: &[i32] = &[0, 1, 14, 31, 32, 33, 63, -1, -32, 100, i32::MIN];
    let ops: Vec<(NodeOp, &str, Rep, &[i32])> = vec![
        (NodeOp::I32Add, "+", Rep::I32, add_sub),
        (NodeOp::I32Sub, "-", Rep::I32, add_sub),
        (NodeOp::I32And, "&", Rep::I32, logical),
        (NodeOp::I32Or, "|", Rep::I32, logical),
        (NodeOp::I32Xor, "^", Rep::I32, logical),
        (NodeOp::I32Shl, "<<", Rep::I32, counts),
        (NodeOp::I32Sar, ">>", Rep::I32, counts),
        (NodeOp::I32UShr, ">>>", Rep::I32, counts),
        (NodeOp::I32UShrToF64, ">>>", Rep::F64, counts),
        (NodeOp::I32Cmp(Cond::Lt), "<", Rep::Bool, add_sub),
        (NodeOp::I32Cmp(Cond::Le), "<=", Rep::Bool, add_sub),
        (NodeOp::I32Cmp(Cond::Gt), ">", Rep::Bool, add_sub),
        (NodeOp::I32Cmp(Cond::Ge), ">=", Rep::Bool, add_sub),
        (NodeOp::I32Cmp(Cond::Eq), "==", Rep::Bool, add_sub),
        (NodeOp::I32Cmp(Cond::Ne), "!=", Rep::Bool, add_sub),
    ];
    // The argument beside the constant: int32s at the edges, and a fraction, which exits.
    let ints = [
        0,
        1,
        -1,
        2,
        7,
        4095,
        4096,
        -4096,
        0x3FFF,
        0x12345,
        0xFF_F000,
        0x100_0000,
        i32::MAX,
        i32::MIN,
        0x7FFF_F000,
        -0x7FFF_F001,
        123_456_789,
        -987_654_321,
    ];
    let mut cases = Vec::new();
    for (op, js, out, constants) in &ops {
        for &c in *constants {
            for left in [false, true] {
                cases.push((op.clone(), *js, *out, c, left));
            }
        }
    }
    let source: String = cases
        .iter()
        .enumerate()
        .map(|(k, (_, js, ..))| format!("function f{k}(a, b) {{ return a {js} b; }} f{k}(1, 2);\n"))
        .collect();
    let mut engine = fixture(&source);
    let (mut immediates, mut overflows) = (0, 0);
    for (k, (op, js, out, c, left)) in cases.into_iter().enumerate() {
        let name = format!("f{k}");
        let graph = const_operand(op.clone(), c, left, out);
        let code = install(&mut engine, &name, &graph, Options::default());
        let context = format!(
            "{op:?} with {c} on the {}",
            if left { "left" } else { "right" }
        );
        if let Some((word, mask)) = imm_form(&op, c, left) {
            assert!(
                holds(&code_words(&code), word, mask),
                "{context}: no immediate"
            );
            immediates += 1;
        }
        let pair = |v: &str| {
            if left {
                format!("{}, {v}", js_int(c))
            } else {
                format!("{v}, {}", js_int(c))
            }
        };
        let calls: Vec<String> = ints
            .iter()
            .map(|&v| format!("{name}({})", pair(&js_int(v))))
            .chain([format!("{name}({})", pair("1.5"))])
            .collect();
        let direct: Vec<String> = ints
            .iter()
            .map(|&v| pair(&js_int(v)).replace(", ", &format!(" {js} ")))
            .chain([pair("1.5").replace(", ", &format!(" {js} "))])
            .collect();
        exits();
        let got = eval(&mut engine, &format!("[{}].join()", calls.join(", ")));
        let exited = exits();
        let expected = eval(&mut engine, &format!("[{}].join()", direct.join(", ")));
        assert_eq!(got, expected, "{context}");
        // The fraction exits at its check; an int32 result past int32 at the operation.
        assert!(exited.contains(&(2, ExitReason::NotInt32)), "{context}");
        let past = ints
            .iter()
            .filter(|&&v| {
                let (a, b) = if left { (c, v) } else { (v, c) };
                match op {
                    NodeOp::I32Add => a.checked_add(b).is_none(),
                    NodeOp::I32Sub => a.checked_sub(b).is_none(),
                    NodeOp::I32UShr => (a as u32) >> (b as u32 & 31) > i32::MAX as u32,
                    _ => false,
                }
            })
            .count();
        let taken = exited
            .iter()
            .filter(|&&exit| exit == (2, ExitReason::Overflow))
            .count();
        assert_eq!(taken, past, "{context}: overflow exits");
        if imm_form(&op, c, left).is_some() {
            overflows += taken;
        }
    }
    assert!(immediates > 200, "{immediates} immediates");
    assert!(
        overflows > 20,
        "{overflows} overflow exits with an immediate operand"
    );
}

/// `cb(a, b)` (LoadLocal 0, LoadLocal 1, Add at pc 2, Return) as an element bounds check on
/// int32s, the constant `c` its index (`index_constant`) or its length, the argument's int32 the
/// other: the argument back where the check holds, an OutOfBounds exit where it fails (the VM
/// then adds, the call passing `c` where the graph has it).
fn bounds_check(c: i32, index_constant: bool) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    let f = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        2,
        at,
        ExitReason::NotNumber,
    );
    let v = b.exiting(
        entry,
        NodeOp::CheckInt32,
        &[f],
        Rep::I32,
        2,
        at,
        ExitReason::NotInt32,
    );
    let k = b.node(entry, NodeOp::ConstI32(c), &[], Rep::I32, 2);
    let inputs = if index_constant { [k, v] } else { [v, k] };
    b.exiting(
        entry,
        NodeOp::CheckBounds,
        &inputs,
        Rep::None,
        2,
        at,
        ExitReason::OutOfBounds,
    );
    let boxed = b.node(entry, NodeOp::I32ToTagged, &[v], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

/// The branching comparison, CheckBounds, with a constant index (`a[3]`, compared from the
/// length's side with the unsigned condition reversed) or a constant length: CMP's immediate,
/// or CMN's for a negative constant (the same flags, the carry among them), past every
/// unsigned bound as a negative index must be.
#[test]
fn bounds_checks_take_a_constant_as_the_immediate() {
    let constants = [
        0,
        1,
        4095,
        4096,
        4097,
        0xFF_F000,
        0x100_0000,
        -1,
        -4096,
        i32::MAX,
        i32::MIN,
    ];
    let values = [
        0,
        1,
        2,
        4095,
        4096,
        4097,
        0xFF_F000,
        0xFF_F001,
        0x100_0000,
        -1,
        -4096,
        i32::MAX,
        i32::MIN,
    ];
    let mut cases = Vec::new();
    for c in constants {
        for index_constant in [true, false] {
            cases.push((c, index_constant));
        }
    }
    let source: String = (0..cases.len())
        .map(|k| format!("function cb{k}(a, b) {{ return a + b; }} cb{k}(1, 2);\n"))
        .collect();
    let mut engine = fixture(&source);
    for (k, (c, index_constant)) in cases.into_iter().enumerate() {
        let code = install(
            &mut engine,
            &format!("cb{k}"),
            &bounds_check(c, index_constant),
            Options::default(),
        );
        let context = format!(
            "{c} as the {}",
            if index_constant { "index" } else { "length" }
        );
        if let Some((word, mask)) = imm_form(&NodeOp::I32Cmp(Cond::Lt), c, false) {
            assert!(holds(&code_words(&code), word, mask), "{context}");
        }
        for v in values {
            let (index, length) = if index_constant { (c, v) } else { (v, c) };
            let inside = (index as u32) < (length as u32);
            exits();
            let got = eval(&mut engine, &format!("cb{k}({}, {})", js_int(v), js_int(c)));
            let expected = if inside {
                v.to_string()
            } else {
                (i64::from(v) + i64::from(c)).to_string()
            };
            assert_eq!(got, expected, "{context}, {v}");
            let out = if inside {
                vec![]
            } else {
                vec![(2, ExitReason::OutOfBounds)]
            };
            assert_eq!(exits(), out, "{context}, {v}");
        }
    }
}

/// `f(a, b)` (LoadLocal 0, LoadLocal 1, Add or Sub at pc 2, Return) on Numbers as `op` with the
/// f64 constant whose bits are `bits` on its right, or on its left (`left`), the argument's f64
/// the other operand.
fn f64_constant(op: NodeOp, bits: u64, left: bool) -> Graph {
    let mut b = Build::new(2);
    let entry = b.block(0, false);
    let x = b.node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let y = b.node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let at = b.state(2, &[x, y], &[], &[]);
    let word = if left { y } else { x };
    let f = b.exiting(
        entry,
        NodeOp::CheckNumber,
        &[word],
        Rep::F64,
        2,
        at,
        ExitReason::NotNumber,
    );
    let k = b.node(entry, NodeOp::ConstF64(bits), &[], Rep::F64, 2);
    let inputs = if left { [k, f] } else { [f, k] };
    let r = b.node(entry, op, &inputs, Rep::F64, 2);
    let boxed = b.node(entry, NodeOp::F64ToTagged, &[r], Rep::Tagged, 2);
    b.node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
    b.g
}

/// F64 constants FMOV's 8-bit immediate encodes take it; the rest keep x30's (zero XZR's), with
/// -0, NaN and the infinities exact.
#[test]
fn f64_constants_take_fmovs_immediate_where_it_encodes() {
    let constants = [
        1.0,
        0.5,
        -2.5,
        31.0,
        0.125,
        -0.1875,
        2.0,
        0.0,
        -0.0,
        0.1,
        32.0,
        0.0625,
        4095.0,
        1e300,
        f64::INFINITY,
        f64::NAN,
    ];
    let values = ["0", "-0", "1.5", "-7.25", "1e10", "NaN", "-Infinity", "0.1"];
    let mut cases = Vec::new();
    for x in constants {
        for (op, js) in [(NodeOp::F64Add, "+"), (NodeOp::F64Sub, "-")] {
            for left in [false, true] {
                cases.push((op.clone(), js, x, left));
            }
        }
    }
    let source: String = cases
        .iter()
        .enumerate()
        .map(|(k, (_, js, ..))| format!("function g{k}(a, b) {{ return a {js} b; }} g{k}(1, 2);\n"))
        .collect();
    let mut engine = fixture(&source);
    // A result's sign shows: -0 as "-0".
    let shown = |e: &str| format!("(r => Object.is(r, -0) ? '-0' : String(r))({e})");
    for (k, (op, js, x, left)) in cases.into_iter().enumerate() {
        let name = format!("g{k}");
        let code = install(
            &mut engine,
            &name,
            &f64_constant(op.clone(), x.to_bits(), left),
            Options::default(),
        );
        let context = format!(
            "{op:?} with {x:?} on the {}",
            if left { "left" } else { "right" }
        );
        if let Some(imm8) = asm::fmov_imm8_d(x.to_bits()) {
            assert!(
                holds(&code_words(&code), 0x1E60_1000 | imm8 << 13, 0xFFFF_FFE0),
                "{context}"
            );
        }
        for v in values {
            let (a, b) = if left {
                (js_number(x), v.to_string())
            } else {
                (v.to_string(), js_number(x))
            };
            exits();
            let got = eval(&mut engine, &shown(&format!("{name}({a}, {b})")));
            assert!(exits().is_empty(), "{context}, {v}");
            let expected = eval(&mut engine, &shown(&format!("{a} {js} {b}")));
            assert_eq!(got, expected, "{context}, {v}");
        }
    }
}
