//! `new` through a site's construct cache, and on ARM64 through the direct construct sequence.
//! ECMA-262 (e28783d5): EvaluateNew, Construct, [[Construct]] for ECMAScript function objects,
//! OrdinaryCreateFromConstructor, GetPrototypeFromConstructor and InitializeInstanceElements.
//! Expected values are V8's.

#![cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]

use crate::bytecode::{Chunk, Tier};
use crate::value::{Callable, Gc, ValueView};
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
    let Callable::User(user) = &object.call else {
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
    for name in ["Box", "Plain"] {
        let constructor = object(&mut engine, name);
        let constructor = constructor.borrow();
        let Callable::User(user) = &constructor.call else {
            panic!("{name} is not a user function")
        };
        assert!(
            user.func
                .runtime()
                .code2
                .get()
                .is_some_and(|code| code.is_some()),
            "{name} did not reach its second-stage compile"
        );
    }
}

/// Each construction counts once toward its constructor's checkpoints, whichever path makes
/// it: one whose allocation declines (a `prototype` that is not an Object) leaves the direct
/// sequence's count to the generic [[Construct]] that makes it.
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
        let Callable::User(user) = &constructor.call else {
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
    assert_eq!(declined - before, 21, "constructions with a null prototype");
    assert_eq!(evaluate(&mut engine, "F.prototype = {}; make(20)"), "190");
    assert_eq!(
        runs(&mut engine) - declined,
        20,
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
