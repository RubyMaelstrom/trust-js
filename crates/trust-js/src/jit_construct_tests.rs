//! `new` through a site's construct cache, and on ARM64 through the direct construct sequence.
//! ECMA-262 (e28783d5): EvaluateNew, Construct, [[Construct]] for ECMAScript function objects,
//! OrdinaryCreateFromConstructor, GetPrototypeFromConstructor and InitializeInstanceElements.
//! Expected values are V8's.

#![cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]

use crate::bytecode::{Chunk, Tier};
use crate::value::{Gc, ObjectKind, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env;
    let ValueView::Obj(object) = (engine
        .interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing {name}")))
    .view() else {
        panic!("{name} is not an object")
    };
    object
}

fn chunk(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let object = object(engine, name);
    let object = object.borrow();
    let ObjectKind::User(user) = &object.kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime
        .execution_code()
        .and_then(Option::as_ref)
        .expect("compiled caller")
        .clone();
    assert!(
        chunk.jit.get().is_some_and(|code| code.is_some()),
        "{name} was not native"
    );
    chunk
}

/// Whether this process runs `test` alone: the parent runs it in a child process, checks that it
/// passed, and returns false. The call-cache epoch is process-global, so other tests' engines
/// must not make a site miss while a test takes counts.
fn run_alone(test: &str) -> bool {
    const CHILD: &str = "TRUST_JS_TEST_CONSTRUCT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([test, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

/// A base class constructor whose class has no instance elements constructs through the site's
/// cache like a function constructor; fields, private methods and a derived class keep the
/// generic [[Construct]]. Either way the instance, its prototype, new.target, a returned object
/// and a returned primitive are the specification's.
#[test]
fn base_classes_without_instance_elements_take_the_construct_cache() {
    let source = r#"
        class Plain { constructor(x) { this.x = x; this.target = new.target; }
            get double() { return this.x * 2; } }
        class Fields { y = 2; constructor(x) { this.x = x; } }
        class Private { #m() { return 3; } constructor(x) { this.x = x; } read() { return this.#m(); } }
        class Base { constructor(x) { this.x = x; } }
        class Derived extends Base { constructor(x) { super(x); this.d = new.target === Derived; } }
        class Returns { constructor(o) { if (o) return o; this.x = 1; } }
        class Empty {}
        class Supered { constructor(x) { this.x = x; this.inherited = super.hasOwnProperty('x'); } }
        function makePlain(k) { return new Plain(k); }
        function makeFields(k) { return new Fields(k); }
        function makePrivate(k) { return new Private(k); }
        function makeDerived(k) { return new Derived(k); }
        function makeReturns(o) { return new Returns(o); }
        function makeEmpty() { return new Empty(); }
        function makeSupered(k) { return new Supered(k); }
        var okay = true, returned = { marker: 1 };
        for (var k = 0; k < 200; k++) {
            var p = makePlain(k), f = makeFields(k), v = makePrivate(k), d = makeDerived(k);
            okay = okay && p.double === 2 * k && p.target === Plain
                && Object.getPrototypeOf(p) === Plain.prototype && Object.keys(p).join() === 'x,target'
                && f.x === k && f.y === 2 && Object.keys(f).join() === 'y,x'
                && v.x === k && v.read() === 3
                && d.x === k && d.d && d instanceof Base && Object.getPrototypeOf(d) === Derived.prototype
                && makeReturns(returned) === returned && makeReturns(0).x === 1
                && makeReturns(5) instanceof Returns && makeReturns(5).x === undefined
                && makeEmpty() instanceof Empty && makeSupered(k).inherited;
        }
        var called;
        try { Plain(1); called = 'no error'; } catch (e) { called = e instanceof TypeError; }
        okay + ':' + called;
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source), "true:true", "{tier:?}");
        assert!(engine.interp.fn_frames.is_empty());
        if tier != Tier::Jit {
            continue;
        }
        for (maker, class, cached) in [
            ("makePlain", "Plain", true),
            ("makeEmpty", "Empty", true),
            ("makeSupered", "Supered", true),
            ("makeFields", "Fields", false),
            ("makePrivate", "Private", false),
            ("makeDerived", "Derived", false),
        ] {
            let target = object(&mut engine, class);
            assert_eq!(
                chunk(&mut engine, maker).cached_construct_for(Gc::as_ptr(&target) as usize),
                cached,
                "{class}"
            );
        }
    }
}

/// A construction counts toward its constructor's inline checkpoints as a call does: a class
/// constructor and a function constructor run only under `new`, through their sites' caches (on
/// ARM64 the direct construct sequence), reach their second-stage compiles with their hot
/// callees spliced in. The splice keeps each callee's strictness.
#[test]
fn constructors_run_only_under_new_reach_their_second_stage_compile() {
    // Its subject is a splice's second stage, which TRUST_JS_OPT_INLINE=graph replaces for a
    // plan of leaf callees.
    crate::opt::pin_splice_for_test();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let before = direct_constructs();
    assert_eq!(
        evaluate(
            &mut engine,
            r#"
            function strictStep(x) { 'use strict'; return x + 1; }
            function sloppyStep(x) { return x + 2; }
            class Box { constructor(x) { this.v = strictStep(x); } }
            function Plain(x) { this.v = sloppyStep(x); }
            function make(k) { return new Box(k).v + new Plain(k).v; }
            var s = 0;
            for (var k = 0; k < 400; k++) s += make(k);
            s;
        "#
        ),
        "160800"
    );
    if cfg!(target_arch = "aarch64") {
        assert!(direct_constructs() > before, "no direct construction");
    }
    let second_stage = |engine: &mut Engine, name: &str| {
        let constructor = object(engine, name);
        let constructor = constructor.borrow();
        let ObjectKind::User(user) = &constructor.kind else {
            panic!("{name} is not a user function")
        };
        user.func
            .runtime()
            .code2
            .get()
            .is_some_and(|code| code.is_some())
    };
    // With the optimizing tier on, a second stage waits for settled feedback (nothing forces
    // it): the constructions go on until both are compiled.
    for _ in 0..40 {
        if !crate::opt::enabled()
            || ["Box", "Plain"]
                .iter()
                .all(|name| second_stage(&mut engine, name))
        {
            break;
        }
        assert_eq!(
            evaluate(
                &mut engine,
                "var s = 0; for (var k = 0; k < 400; k++) s += make(k); s"
            ),
            "160800"
        );
    }
    for name in ["Box", "Plain"] {
        assert!(
            second_stage(&mut engine, name),
            "{name} did not reach its second-stage compile"
        );
    }
}

/// The graph inliner's counterpart (P3 stage 3 step 3g): at TRUST_JS_OPT_INLINE=graph a class
/// constructor and a function constructor run only under `new` reach their compiles in place,
/// their callees built into their graphs, and each callee keeps its strictness (a strict one's
/// `this` undefined, a sloppy one's the global object).
#[test]
fn constructors_run_only_under_new_reach_their_in_place_compile_at_graph() {
    crate::opt::pin_graph_for_test();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let before = direct_constructs();
    const RUN: &str = "var s = 0; for (var k = 0; k < 400; k++) s += make(k); s";
    assert_eq!(
        evaluate(
            &mut engine,
            r#"
            function strictStep(x) { 'use strict'; return x + (this === undefined ? 1 : 1000); }
            function sloppyStep(x) { return x + (this === globalThis ? 2 : 2000); }
            class Box { constructor(x) { this.v = strictStep(x); } }
            function Plain(x) { this.v = sloppyStep(x); }
            function make(k) { return new Box(k).v + new Plain(k).v; }
            var s = 0;
            for (var k = 0; k < 400; k++) s += make(k);
            s;
        "#
        ),
        "160800"
    );
    if cfg!(target_arch = "aarch64") {
        assert!(direct_constructs() > before, "no direct construction");
    }
    let in_place = |engine: &mut Engine, name: &str| {
        let constructor = object(engine, name);
        let constructor = constructor.borrow();
        let ObjectKind::User(user) = &constructor.kind else {
            panic!("{name} is not a user function")
        };
        let chunk = user.func.runtime().code.get().and_then(Option::clone);
        chunk.is_some_and(|chunk| crate::opt::in_place(&chunk))
    };
    // The tier compiles at a checkpoint of settled feedback: the constructions go on until both
    // are compiled.
    for _ in 0..40 {
        if !crate::opt::enabled()
            || ["Box", "Plain"]
                .iter()
                .all(|name| in_place(&mut engine, name))
        {
            break;
        }
        assert_eq!(evaluate(&mut engine, RUN), "160800");
    }
    for name in ["Box", "Plain"] {
        assert_eq!(
            in_place(&mut engine, name),
            crate::opt::enabled(),
            "{name} did not reach its compile in place"
        );
    }
    assert_eq!(evaluate(&mut engine, RUN), "160800");
}

/// Each construction counts once toward its constructor's checkpoints (an entry's weight: one,
/// or K with the optimizing tier on, `opt::entry_weight`), whichever path makes it: one whose
/// allocation declines (a `prototype` that is not an Object) leaves the direct sequence's count
/// to the generic [[Construct]] that makes it.
#[test]
fn every_construction_counts_once() {
    if !run_alone("jit::construct_tests::every_construction_counts_once") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let runs = |engine: &mut Engine| {
        let constructor = object(engine, "F");
        let constructor = constructor.borrow();
        let ObjectKind::User(user) = &constructor.kind else {
            panic!("F is not a user function")
        };
        let runtime = user.func.runtime();
        let chunk = runtime
            .execution_code()
            .and_then(Option::as_ref)
            .expect("compiled F");
        assert_ne!(chunk.inline_retry_at.get(), 0, "a checkpoint is pending");
        chunk.jit_runs.get()
    };
    evaluate(
        &mut engine,
        r#"
        function F(x) { this.x = x; }
        function make(n) { var s = 0; for (var k = 0; k < n; k++) s += new F(k).x; return s; }
        function deep(n) { return n > 0 ? deep(n - 1) + 1 : 0; }
        make(30);
    "#,
    );
    let before = runs(&mut engine);
    #[cfg(target_arch = "aarch64")]
    let declines = crate::bytecode::TEST_CONSTRUCT_DECLINES.with(std::cell::Cell::get);
    // `deep` leaves records pooled, so the sequence reaches its allocation and declines there.
    assert_eq!(
        evaluate(
            &mut engine,
            "deep(8); F.prototype = null; var o = new F(1); \
             make(20) + ':' + (Object.getPrototypeOf(o) === Object.prototype)"
        ),
        "190:true"
    );
    #[cfg(target_arch = "aarch64")]
    assert!(
        crate::bytecode::TEST_CONSTRUCT_DECLINES.with(std::cell::Cell::get) > declines,
        "no allocation declined"
    );
    let declined = runs(&mut engine);
    let entry = crate::opt::entry_weight();
    assert_eq!(
        declined - before,
        21 * entry,
        "constructions with a null prototype"
    );
    assert_eq!(evaluate(&mut engine, "F.prototype = {}; make(20)"), "190");
    assert_eq!(
        runs(&mut engine) - declined,
        20 * entry,
        "constructions with an Object prototype"
    );
}

/// Constructions the ARM64 direct construct sequence has completed on this thread.
fn direct_constructs() -> usize {
    #[cfg(target_arch = "aarch64")]
    return super::TEST_DIRECT_CONSTRUCTS.with(std::cell::Cell::get);
    #[cfg(not(target_arch = "aarch64"))]
    0
}

/// Runs `source` at every tier, each giving `expected`, and checks that the frame stack and the
/// interpreter's construct state are back to the program's. On ARM64 the machine-code tier
/// constructs through the direct sequence.
fn all_tiers(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let before = direct_constructs();
        assert_eq!(evaluate(&mut engine, source), expected, "{tier:?}");
        if cfg!(target_arch = "aarch64") && tier == Tier::Jit {
            assert!(direct_constructs() > before, "no direct construction");
        }
        assert!(engine.interp.fn_frames.is_empty(), "{tier:?}");
        assert!(!engine.interp.constructing, "{tier:?}");
        assert_eq!(
            engine.interp.new_target.bits(),
            crate::value::PACK_UNDEFINED,
            "{tier:?}"
        );
    }
}

/// Hot `new` sites (the direct construct sequence on ARM64) against the generic [[Construct]]:
/// argument binding with missing and surplus arguments, sloppy and strict bodies, new.target in
/// the body and in its own calls and constructions (the sequence's full form), an Object result
/// replacing `this` and other results not, a throw, a base class, bodies whose operations
/// materialize a native activation (the finish stub's helper fallback), a body that needs the
/// global object, the constructor's live `prototype` (replaced, then not an Object), sites in a
/// loop (the sequence in line) and elsewhere (a shared stub's), and constructions nested past
/// the sequence's depth limit.
#[test]
fn constructions_match_the_specification_at_every_tier() {
    all_tiers(
        r#"
        function Point(x, y) { this.x = x; this.y = y; }
        function Nine(a, b, c, d, e, f, g, h, i) { this.sum = a + b + c + d + e + f + g + h + i; }
        function Strict(x) { 'use strict'; this.target = new.target === Strict; this.x = x; }
        var replacement = { replaced: true };
        function Returns(mode) {
            this.own = 1;
            if (mode === 1) return replacement;
            if (mode === 2) return 7;
            if (mode === 3) return null;
        }
        function Throws(x) { this.x = x; if (x % 3 === 0) throw new RangeError('t' + x); }
        function plain() { return new.target === undefined; }
        function Inner() { this.target = new.target === Inner; }
        function Outer() {
            this.before = new.target === Outer;
            this.inner = new Inner();
            this.called = plain();
            this.after = new.target === Outer;
        }
        class C { constructor(x) { this.x = x; } get double() { return this.x * 2; } }
        class Supered { constructor(x) { this.x = x; this.own = super.hasOwnProperty('x'); } }
        function Spread(a) { this.list = [...a, 1]; }
        var scale = 3;
        function Scaled(x) { this.v = x * scale; }
        function Live() { this.tag = 'live'; }
        var protoA = { kind: 'a' }, protoB = { kind: 'b' };
        function makeLive() { return new Live(); }
        function Deep(n) { this.next = n > 0 ? new Deep(n - 1) : null; this.n = n; }
        function loopSum(n) {
            var s = 0;
            for (var i = 0; i < n; i++) { var p = new Point(i, 2); s += p.x + p.y; }
            return s;
        }
        function depth(d) { var n = 0; while (d.next) { d = d.next; n++; } return n + ':' + d.n; }
        function drive(k) {
            var a = new Point(k), b = new Point(k, 1), c = new Point(k, 2, 3);
            var threw = '';
            try { new Throws(k); } catch (e) { threw = e instanceof RangeError ? e.message : 'bad'; }
            var o = new Outer(), s = new Strict(k), d = new C(k);
            Live.prototype = k % 2 ? protoA : protoB;
            return [typeof a.y, b.y + c.y, Object.keys(c).join(), new Nine(k, 1, 2, 3, 4, 5, 6, 7, 8).sum,
                s.target && s.x === k, new Returns(1) === replacement, new Returns(2).own,
                new Returns(3).own, new Returns(0).own, threw,
                o.before && o.inner.target && o.called && o.after, d.double, d instanceof C,
                new Scaled(k).v, makeLive().kind, loopSum(k % 7), new Supered(k).own,
                new Spread([k]).list.join('-')].join(';');
        }
        var out = [];
        for (var k = 0; k < 300; k++) out[k % 3] = drive(k);
        Live.prototype = 5;
        var plainLive = makeLive();
        out.push(Object.getPrototypeOf(plainLive) === Object.prototype, plainLive.tag);
        out.push(depth(new Deep(300)));
        out.join('|');
    "#,
        "undefined;3;x,y;333;true;true;1;1;1;t297;true;594;true;891;a;9;true;297-1|\
         undefined;3;x,y;334;true;true;1;1;1;;true;596;true;894;b;14;true;298-1|\
         undefined;3;x,y;335;true;true;1;1;1;;true;598;true;897;a;20;true;299-1|true|live|300:0",
    );
}

/// The stack a constructor's body captures names its frame `new F` at its caller's position,
/// at every tier.
#[test]
fn constructor_frames_appear_in_stack_traces() {
    let source = r#"
        function Traced(x) { this.x = x; this.stack = new Error('s').stack; }
        function makeTraced(k) { return new Traced(k); }
        var stack;
        for (var k = 0; k < 200; k++) stack = makeTraced(k).stack;
        stack;
    "#;
    let mut reference = None;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let stack = evaluate(&mut engine, source);
        assert!(stack.contains("at new Traced ("), "{tier:?}: {stack}");
        match &reference {
            None => reference = Some(stack),
            Some(reference) => assert_eq!(&stack, reference, "{tier:?}"),
        }
    }
}

/// The instance lives while its constructor runs, across collections in its body and in the
/// bodies it constructs, whatever else refers to it.
#[test]
fn constructions_keep_their_instance_across_collections() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine
            .interp
            .def_method(&engine.interp.global, "collect", 0, |interp, _, _| {
                interp.gc_collect();
                Ok(crate::value::Value::Undefined)
            });
        let before = direct_constructs();
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            function Holder(x) { this.first = { v: x }; collect(); this.second = 'n' + x; }
            function Wrapped(x) { var inner = new Holder(x); collect(); this.inner = inner; }
            function Dropped(x) { this.lost = { v: x }; collect(); return { kept: x }; }
            function make(k) { return new Wrapped(k); }
            function drop(k) { return new Dropped(k); }
            var okay = true;
            for (var k = 0; k < 120; k++) {
                var w = make(k), d = drop(k);
                okay = okay && w.inner.first.v === k && w.inner.second === 'n' + k
                    && w instanceof Wrapped && d.kept === k && !(d instanceof Dropped);
            }
            okay;
        "#
            ),
            "true",
            "{tier:?}"
        );
        if cfg!(target_arch = "aarch64") && tier == Tier::Jit {
            assert!(direct_constructs() > before, "no direct construction");
        }
        assert!(engine.interp.fn_frames.is_empty());
    }
}

/// Hot `new` sites of a function constructor and of a base class complete through the ARM64
/// direct construct sequence, and a constructor's throw unwinds through it.
#[cfg(target_arch = "aarch64")]
#[test]
fn hot_new_sites_construct_through_the_direct_sequence() {
    // Counts the constructions the direct sequence completes, in the tier's effects too; a
    // forced exit hands its construction to the VM, and a compile off the main thread that
    // installs during a count changes the code under it: compiled on the main thread.
    crate::opt::pin_forced_exits_off_for_test();
    crate::opt::pin_synchronous_for_test();
    if !run_alone("jit::construct_tests::hot_new_sites_construct_through_the_direct_sequence") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let program = r#"
        function Point(x, y) { this.x = x; this.y = y; }
        class C { constructor(x) { this.x = x; } }
        function Throws(x) { this.x = x; throw x; }
        function make(k) { return new Point(k, 1).y + new C(k).x; }
        function fail(k) { try { new Throws(k); } catch (e) { return e; } }
        function drive(n) {
            var s = 0;
            for (var k = 0; k < n; k++) s += make(k) + fail(k) + new Point(k, 2).y;
            return s;
        }
    "#;
    // Each caller's own recompiles start its sites' caches afresh: warm them all.
    evaluate(
        &mut engine,
        &format!("{program} for (var round = 0; round < 8; round++) drive(100); 0"),
    );
    let constructs = || super::TEST_DIRECT_CONSTRUCTS.with(std::cell::Cell::get);
    let throws = || super::TEST_DIRECT_PACKED_THROWS.with(std::cell::Cell::get);
    let before = (constructs(), throws());
    assert_eq!(evaluate(&mut engine, "drive(100)"), "10200");
    assert_eq!(
        constructs() - before.0,
        300,
        "three hot sites (two through a shared stub, one in line), 100 times each"
    );
    assert_eq!(throws() - before.1, 100, "the throwing site, 100 times");
    assert!(engine.interp.fn_frames.is_empty());

    // A simple constructor, whose fields the helper copies without a record, at a site whose
    // caller holds the only record: no call has filled the pool.
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    evaluate(
        &mut engine,
        r#"
        function Pair(x, y) { this.x = x; this.y = y; }
        function run(n) { var s = 0; for (var k = 0; k < n; k++) s = (s + new Pair(k, 1).y) | 0; return s; }
        for (var round = 0; round < 4; round++) run(1000);
    "#,
    );
    let before = constructs();
    assert_eq!(evaluate(&mut engine, "run(1000)"), "1000");
    assert_eq!(constructs() - before, 1000);
}

// ---- P3 stage 3 steps 6-7, A1: a chunk's learned instance layout ----

/// `source` at the three tiers: the same completion at each, which is returned.
fn same_at_every_tier(source: &str) -> String {
    let mut results = Vec::new();
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        results.push(evaluate(&mut engine, source));
        assert!(engine.interp.fn_frames.is_empty(), "{tier:?}");
    }
    assert_eq!(results[0], results[1], "the interpreter against the VM");
    assert_eq!(
        results[0], results[2],
        "the interpreter against the template JIT"
    );
    results.remove(0)
}

/// Constructors the static layout pass predicts nothing for (a store behind a jump, fields set
/// by a called super constructor, no fields at all) go through `Op::New`'s helper for their
/// first [`crate::bytecode`] learning constructions and through the direct construct sequence
/// from the second construction after it ends (the first refreshes the site's entry): the
/// chunk takes the layout of an instance it measured, or settles without keys. Their instances
/// are the VM's.
#[test]
fn a_constructor_without_a_static_layout_takes_the_direct_sequence_once_its_learning_ends() {
    crate::opt::pin_forced_exits_off_for_test();
    crate::opt::pin_synchronous_for_test();
    if !run_alone(
        "jit::construct_tests::a_constructor_without_a_static_layout_takes_the_direct_sequence_once_its_learning_ends",
    ) {
        return;
    }
    let program = r#"
        function Variable(name, initial) { this.value = initial || 0; this.name = name; }
        function Base(a) { this.a = a; this.b = 1; }
        function Sub(a) { Base.call(this, a); }
        function Empty() {}
        function Bare() {}
        function variable(k) { return new Variable("v" + k, k); }
        function sub(k) { return new Sub(k); }
        function empty(k) { var e = new Empty(); e.late = k; return e; }
        function bare(k) { return new Bare(); }
        function shape(o) { return Object.keys(o).join() + ":" + JSON.stringify(o); }
        function drive(from, to) {
            var out = [];
            for (var k = from; k < to; k++) out.push(shape(variable(k)), shape(sub(k)), shape(empty(k)), shape(bare(k)));
            return out.join(";");
        }
    "#;
    let expected = same_at_every_tier(&format!("{program} drive(0, 40)"));
    assert!(
        expected.contains(
            "value,name:{\"value\":39,\"name\":\"v39\"};a,b:{\"a\":39,\"b\":1};late:{\"late\":39};:{}"
        ),
        "{expected}"
    );
    for (name, layout) in [
        ("Variable", Some(2)),
        ("Sub", Some(2)),
        ("Empty", Some(1)),
        ("Bare", None),
    ] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        evaluate(&mut engine, &format!("{program} 0"));
        let maker = name.to_lowercase();
        // The learning's constructions (its first eight through the construct cache: those
        // before the constructor has machine code are not counted): none direct.
        let before = direct_constructs();
        evaluate(&mut engine, &format!("{maker}(0); {maker}(1); 0"));
        let constructor = chunk(&mut engine, name);
        let mut made = 2;
        while !constructor.instance_layout_settled() {
            assert!(made < 24, "{name}: its learning never ended");
            evaluate(&mut engine, &format!("{maker}({made}); 0"));
            made += 1;
        }
        assert!(made >= 8, "{name}: settled after {made} constructions");
        assert_eq!(
            direct_constructs(),
            before,
            "{name}: no direct construction while learning"
        );
        assert_eq!(
            constructor.instance_layout().map(|layout| layout.len()),
            layout,
            "{name}'s learned layout"
        );
        if cfg!(target_arch = "aarch64") {
            // The next construction refreshes the site's entry through the helper; the rest are
            // direct (but one where the caller's recompile starts its site afresh).
            evaluate(&mut engine, &format!("{maker}({made}); 0"));
            assert_eq!(
                direct_constructs(),
                before,
                "{name}: the construction after the learning refreshes the entry"
            );
            evaluate(
                &mut engine,
                &format!("for (var k = 0; k < 100; k++) {maker}(k); 0"),
            );
            let direct = direct_constructs() - before;
            assert!(
                (98..=100).contains(&direct),
                "{name}: {direct} of 100 direct"
            );
        }
    }
}

/// What a chunk's learning may keep, and what it gives later instances: only the layout of an
/// instance of one to sixteen plain String-keyed data properties; an instance with an accessor,
/// a non-default attribute (frozen), a Symbol or an index key, a deleted property (its layout
/// then its own), or forty keys settles the chunk without keys, or with the keys of the plain
/// instance measured last. A layout is a prediction: whatever was kept, every later instance,
/// of the learned shape or another, equals the VM's, as do a constructor that throws or returns
/// an object while it learns, one that constructs itself, closures of one literal with
/// different fields, and a base class reached by `new` and by two subclasses' `super`.
#[test]
fn a_learned_layout_is_a_prediction_whatever_the_measured_instances_were() {
    let prelude = r#"
        function show(o) {
            var own = Object.getOwnPropertyNames(o).map(function(k) {
                var d = Object.getOwnPropertyDescriptor(o, k);
                return k + (d.get ? "@get" : "=" + d.value) + (d.writable === false ? "!w" : "") + (d.enumerable ? "" : "!e");
            });
            return own.join(",") + "/" + Object.getOwnPropertySymbols(o).length + "/" + Object.isFrozen(o) + "/" + (o instanceof Object);
        }
        function run(make, n) { var out = []; for (var k = 0; k < n; k++) { try { out.push(show(make(k))); } catch (e) { out.push("threw " + e); } } return out.join(";"); }
    "#;
    let cases = [
        // The shape changes after the learning (the previous agent's L2).
        "function Q(x, y) { if (x < 12) { this.a = x; this.b = y; } else { this.c = x; this.a = y; } }
         run(function(k) { return new Q(k, 1); }, 40)",
        // A return before the stores, a throw, a returned object, while it learns and after.
        "function R(k) { if (k % 3 == 0) return; this.a = k; if (k % 5 == 1) throw 'no'; if (k % 7 == 2) return { other: k }; this.b = 2; }
         run(function(k) { return new R(k); }, 40)",
        // A constructor that constructs itself: the measured instance is unfinished.
        "function N(d) { this.d = d; if (d > 0) this.child = new N(d - 1).d; this.e = 5; }
         run(function(k) { return new N(k % 3); }, 30)",
        // An outlier among the first eight: a bag of forty keys.
        "function Bag(k) { if (k == 3) { for (var j = 0; j < 40; j++) this['f' + j] = j; } else { this.x = k || 1; this.y = 2; } }
         run(function(k) { var b = new Bag(k); return k == 3 ? { keys: Object.keys(b).length } : b; }, 40)",
        // Measured instances that are not plain: an accessor, a non-enumerable property, a
        // Symbol key, an index key, a frozen instance, a deleted property, another prototype.
        "function A(k) { this.p = k || 0; Object.defineProperty(this, 'g', { get: function() { return 1; }, enumerable: true, configurable: true }); this.q = 2; }
         run(function(k) { return new A(k); }, 30)",
        "function H(k) { this.p = k || 0; Object.defineProperty(this, 'h', { value: 3, enumerable: false, writable: true, configurable: true }); }
         run(function(k) { return new H(k); }, 30)",
        "var sym = Symbol('s'); function S(k) { this.p = k || 0; this[sym] = 1; this[7] = 2; this.q = 3; }
         run(function(k) { return new S(k); }, 30)",
        "function F(k) { this.p = k || 0; this.q = 1; if (k < 8) Object.freeze(this); }
         run(function(k) { return new F(k); }, 30)",
        "function D(k) { this.p = k || 0; this.q = 1; this.r = 2; if (k < 9) delete this.q; }
         run(function(k) { return new D(k); }, 30)",
        "var other = { marker: 1 }; function P(k) { this.p = k || 0; if (k % 2) Object.setPrototypeOf(this, other); this.q = 1; }
         run(function(k) { var p = new P(k); return { keys: Object.keys(p).join(), marker: p.marker }; }, 30)",
        // Fields the caller adds after the constructor returned are measured too.
        "function L(k) { this.p = k || 0; }
         run(function(k) { var l = new L(k); l.after = k; l.more = 1; return l; }, 30)",
        // Closures of one literal with different fields share the chunk's layout.
        "function maker(names) { return function(k) { for (var j = 0; j < names.length; j++) this[names[j]] = k + j; if (k < 0) this.neg = 1; }; }
         var One = maker(['a', 'b']), Two = maker(['c', 'a', 'd']);
         run(function(k) { return new One(k); }, 20) + '|' + run(function(k) { return new Two(k); }, 20)",
        // A base class reached by `new` and by two subclasses' `super`, with different fields.
        "class Base { constructor(k) { this.base = k || 0; } }
         class Left extends Base { constructor(k) { super(k); this.left = 1; } }
         class Right extends Base { constructor(k) { super(k); this.right = 2; this.more = 3; } }
         run(function(k) { return k % 3 == 0 ? new Base(k) : k % 3 == 1 ? new Left(k) : new Right(k); }, 60)",
        "function FBase(k) { this.base = k || 0; }
         function FLeft(k) { FBase.call(this, k); this.left = 1; }
         function FRight(k) { FBase.call(this, k); this.right = 2; this.more = 3; }
         run(function(k) { return k % 3 == 0 ? new FBase(k) : k % 3 == 1 ? new FLeft(k) : new FRight(k); }, 60)",
    ];
    for case in cases {
        let result = same_at_every_tier(&format!("{prelude} {case}"));
        assert!(result.len() > 100, "{case}\n{result}");
    }
}

/// The end of a constructor's learning changes no shape: an instance starts at the empty shape of
/// its function's capacity class whether or not its chunk has a layout, so the creation caches at
/// the constructor's stores, and code the optimizing tier compiled while the constructor was
/// still learning, see the same shapes before and after. With the tier on (eagerly too) the
/// functions that read the early instances take no exit for the instances the direct sequence
/// makes after the switch; with it off the record of exits stays empty. The optimizing tier
/// and its record of exits are ARM64's.
#[test]
#[cfg(target_arch = "aarch64")]
fn the_end_of_a_constructors_learning_exits_no_compiled_function() {
    crate::opt::pin_forced_exits_off_for_test();
    crate::opt::pin_synchronous_for_test();
    if !run_alone(
        "jit::construct_tests::the_end_of_a_constructors_learning_exits_no_compiled_function",
    ) {
        return;
    }
    let early = r#"
        function Variable(name, initial) { this.value = initial || 0; this.name = name; }
        function read(o) { return o.value + o.name.length; }
        function sum(list) { var s = 0; for (var j = 0; j < list.length; j++) s += read(list[j]); return s; }
        var first = [];
        for (var k = 0; k < 4; k++) first.push(new Variable("v" + k, k));
        var total = 0;
        for (var round = 0; round < 3000; round++) total += sum(first);
        total
    "#;
    let late = r#"
        var later = [];
        for (var k = 0; k < 40; k++) later.push(new Variable("w" + k, k));
        for (var round = 0; round < 300; round++) total += sum(later) + sum(first);
        total
    "#;
    let mut vm = Engine::new();
    vm.set_tier(Tier::Bytecode);
    let expected = (evaluate(&mut vm, early), evaluate(&mut vm, late));
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(evaluate(&mut engine, early), expected.0);
    let constructor = chunk(&mut engine, "Variable");
    assert!(!constructor.instance_layout_settled(), "four constructions");
    let exits = || crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow().len());
    let before = exits();
    let direct = direct_constructs();
    assert_eq!(evaluate(&mut engine, late), expected.1);
    assert!(constructor.instance_layout_settled());
    if cfg!(target_arch = "aarch64") {
        assert!(
            direct_constructs() > direct,
            "the later instances are the direct sequence's"
        );
    }
    assert_eq!(
        exits(),
        before,
        "no exit for the instances made after the learning: {:?}",
        crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow().clone())
    );
}

/// The learned layout cell (`Chunk::learned_layout`) has three setters, each owning a kind of
/// chunk (P3 stage 3 steps 6-7, A1): `Both` is a simple constructor and a forwarder's
/// initializer, so the simple constructor's path and the Rust forwarder's plan may each set its
/// cell, from one straight-line body and so with the same keys; `Branchy` has neither plan nor a
/// static layout and is also a forwarder's initializer, so A1 alone sets its cell, from its own
/// constructions, and the forwarders' constructions take it as a prediction; a recognized
/// forwarder's own cell stays empty. Every tier makes the same instances.
#[test]
fn a_learned_layout_has_one_setter_for_each_kind_of_chunk() {
    let program = r#"
        var Class = { create: function() { return function() { this.initialize.apply(this, arguments); } } };
        function Both(x, y) { this.x = x; this.y = y; }
        function Branchy(x) { if (x >= 0) this.a = x; this.b = 1; }
        var F = Class.create(); F.prototype = { initialize: Both };
        var G = Class.create(); G.prototype = { initialize: Branchy };
        function shape(o) { return Object.keys(o).join() + ":" + JSON.stringify(o); }
        // The instances are kept: the learning measures each construction's predecessor
        // through a weak handle, which a collection between the two (GC stress) would clear.
        var kept = [];
        function made(o) { kept.push(o); return shape(o); }
        function drive(from, to) {
            var out = [];
            for (var k = from; k < to; k++) {
                out.push(made(new Both(k, -k)), made(new F(k, -k)), made(new Branchy(k)), made(new G(k)));
                if (k % 7 == 0) out.push(made(new Branchy(-k - 1)), made(new G(-k - 1)));
            }
            return out.join(";");
        }
    "#;
    let expected = same_at_every_tier(&format!("{program} drive(0, 60)"));
    assert!(
        expected.ends_with(
            "x,y:{\"x\":59,\"y\":-59};x,y:{\"x\":59,\"y\":-59};a,b:{\"a\":59,\"b\":1};a,b:{\"a\":59,\"b\":1}"
        ),
        "{expected}"
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    evaluate(&mut engine, &format!("{program} drive(0, 60); 0"));
    let both = chunk(&mut engine, "Both");
    assert!(both.jit_simple_constructor().is_some() && both.jit_initializer_plan().is_some());
    assert_eq!(both.learned_layout().map(|layout| layout.len()), Some(2));
    let branchy = chunk(&mut engine, "Branchy");
    assert!(branchy.jit_simple_constructor().is_none() && branchy.jit_initializer_plan().is_none());
    assert!(branchy.instance_layout_settled());
    assert_eq!(branchy.learned_layout().map(|layout| layout.len()), Some(2));
    // The path: part A2's recognizer, which TRUST_JS_JIT_NO_FORWARDER turns off (the forwarders
    // are then ordinary constructors, whose own cells may learn a layout).
    for forwarder in ["F", "G"] {
        let chunk = chunk(&mut engine, forwarder);
        if crate::bytecode::forwarder_disabled() {
            assert!(
                chunk.jit_arguments_apply_forwarder().is_none(),
                "{forwarder}"
            );
            continue;
        }
        assert!(
            chunk.jit_arguments_apply_forwarder().is_some(),
            "{forwarder}"
        );
        assert!(chunk.learned_layout().is_none(), "{forwarder}'s own cell");
    }
}

/// The end of a constructor's learning takes the measured instance's layout only where that
/// instance is one of the largest measured and its layout is one a chunk may take (P3 stage 3
/// steps 6-7, A1; `value::Props::learnable_layout`): `Small`'s last measured instance holds one
/// key where earlier ones held two (it settles without keys), `Attr`'s holds a read-only
/// property, and `Many`'s seventeen keys, one past the bound (an instance past sixteen keys
/// carries a key index of its own, `value::Props::learnable_layout`): each settles and takes no
/// layout; `Bound`'s sixteen, at the bound, are learnt.
#[test]
fn a_learned_layout_is_only_a_largest_plain_small_instances() {
    let program = r#"
        function Small(k) { if (k >= 0) this.a = k; if (k > 100) this.b = k; }
        function Attr(k) { if (k >= 0) Object.defineProperty(this, "a", { value: k, writable: false, enumerable: true, configurable: true }); this.b = 1; }
        function Many(k) { if (k >= 0) { this.f0 = k; this.f1 = k; this.f2 = k; this.f3 = k; this.f4 = k; this.f5 = k; this.f6 = k; this.f7 = k; this.f8 = k; this.f9 = k; this.f10 = k; this.f11 = k; this.f12 = k; this.f13 = k; this.f14 = k; this.f15 = k; this.f16 = k; } }
        function Bound(k) { if (k >= 0) { this.f0 = k; this.f1 = k; this.f2 = k; this.f3 = k; this.f4 = k; this.f5 = k; this.f6 = k; this.f7 = k; this.f8 = k; this.f9 = k; this.f10 = k; this.f11 = k; this.f12 = k; this.f13 = k; this.f14 = k; this.f15 = k; } }
        function bound(k) { return new Bound(k); }
        function small(k) { return new Small(k); }
        function attr(k) { return new Attr(k); }
        function many(k) { return new Many(k); }
        var kept = [];
    "#;
    for (name, learnt) in [
        ("Small", None),
        ("Attr", None),
        ("Many", None),
        ("Bound", Some(16)),
    ] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        evaluate(&mut engine, &format!("{program} 0"));
        let maker = name.to_lowercase();
        evaluate(&mut engine, &format!("{maker}(1000); 0"));
        let constructor = chunk(&mut engine, name);
        let mut made = 0;
        while !constructor.instance_layout_settled() {
            assert!(made < 40, "{name}: its learning never ended");
            // Small: two keys until the seventh counted construction, whose instance the
            // eighth measures, then one.
            let k = if constructor.construct_learning_count() < 6 {
                1000
            } else {
                5
            };
            // Kept: the learning measures each instance through a weak handle.
            evaluate(&mut engine, &format!("kept.push({maker}({k})); 0"));
            made += 1;
        }
        assert_eq!(
            constructor.learned_layout().map(|layout| layout.len()),
            learnt,
            "{name}'s learnt layout"
        );
    }
}
