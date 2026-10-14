//! Function bodies that stay encoded in a static host snapshot until they run
//! (`ast::FunctionBody`, `snapshot::decode_static`), and the capture summaries that compile the
//! functions around them without keeping them (`bytecode::CaptureSummary`).
//!
//! ECMA-262 OrdinaryFunctionCreate stores a function's body Parse Node as its
//! [[ECMAScriptCode]], which FunctionDeclarationInstantiation and evaluation read when the
//! function is called (local snapshot e28783d5). When an implementation materializes those Parse
//! Nodes is unobservable, so a program must behave the same whether it is parsed or decoded
//! lazily, in every tier and every Realm. Test builds additionally check each capture summary a
//! compilation applies against walking the same function (`capture_summary_tests::check`).

use crate::bytecode::Tier;
use crate::interpreter::{Abrupt, Interp};
use crate::realm_inline_guard_tests::{eval_string, in_realm, render, TIERS};
use crate::value::{Callable, Value};
use crate::Engine;
use std::rc::Rc;

/// Closures of every kind, nested at several depths, with the scoping constructs a capture
/// analysis must get right: per-iteration and catch bindings, parameter scopes, shadowing, arrow
/// `this`/`arguments`/`new.target`/`super`, class elements, sloppy direct eval and `with`, Annex B
/// block functions, generators and tagged templates.
const CORPUS: &str = r#"
var log = [];
function note(label, value) { log.push(label + "=" + value); }

function counterFactory(start) {
    let count = start;
    const step = 2;
    function increment() { count += step; return count; }
    return { increment, read: () => count, nested() { return () => () => count * step; } };
}
var counter = counterFactory(1);
counter.increment();
note("counter", counter.read() + ":" + counter.nested()()());

function loops() {
    const fns = [];
    for (let i = 0; i < 3; i++) fns.push(() => i);
    for (const [k, v] of [["a", 1], ["b", 2]]) fns.push(() => k + v);
    try { throw "boom"; } catch (error) { fns.push(() => error); }
    switch (fns.length) { case 6: { let inSwitch = "sw"; fns.push(() => inSwitch); } }
    return fns.map(f => f()).join(",");
}
note("loops", loops());

class Base {
    constructor(name) { this.name = name; }
    greet() { return "hi " + this.name; }
    static make(name) { return new this(name); }
}
class Derived extends Base {
    #secret = 7;
    field = () => this.name + "!";
    static count = 0;
    static { this.count = 10; }
    greet() { const viaArrow = () => super.greet() + "/" + this.#secret; return viaArrow(); }
    get upper() { return this.name.toUpperCase(); }
    has(o) { return #secret in o; }
}
function Plain() { const target = () => new.target === Plain; this.isNew = target(); }
var proto = { hello() { return "proto"; } };
var literal = {
    __proto__: proto,
    hello() { return super.hello() + "+literal"; },
    args() { return (() => arguments.length)(); },
};
var d = Derived.make("dee");
note("classes", [d.greet(), d.upper, d.field(), Derived.count, d.has(d), new Plain().isNew,
    literal.hello(), literal.args(1, 2, 3)].join(","));

function defaults(a, b = () => a, c = a + 1) { var a = "body"; return [b(), c, a].join(","); }
note("defaults", defaults(5));

var fact = function self(n) { return n <= 1 ? 1 : n * self(n - 1); };
var shadow = "global";
function shadows() {
    var shadow = "outer";
    function inner() { let shadow = "inner"; return [shadow, (() => shadow)()].join(); }
    return [shadow, inner(), (function () { return shadow; })()].join("|");
}
note("names", fact(5) + ";" + shadows());

function evals() {
    var x = 1;
    function viaEval(code) { return eval(code); }
    function inWith(o) { with (o) { return y + x; } }
    return [viaEval("x + 41"), inWith({ y: 2 }), (function () { var z = 3; return eval("z * x"); })()].join();
}
note("eval", evals());

function annexB() {
    var seen = typeof hoisted;
    { function hoisted() { return "block"; } }
    outer: for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) { if (j === 1) continue outer; if (i === 2) break outer; }
    }
    return seen + "," + hoisted() + "," + i + j;
}
note("annexB", annexB());

function* fib() { let [a, b] = [0, 1]; for (;;) { yield a; [a, b] = [b, a + b]; } }
function firstFib(n) {
    const out = [];
    for (const v of fib()) { if (out.length === n) break; out.push(v); }
    return out.join("");
}
function tagged() {
    const tag = (s) => s;
    const sites = [];
    for (let i = 0; i < 2; i++) sites.push(tag`x${i}`);
    return sites[0] === sites[1];
}
note("gen", firstFib(7) + "," + tagged());

function hot() {
    let sum = 0;
    const add = (v) => { sum += v; };
    for (let i = 0; i < 2000; i++) add(i % 7);
    return sum;
}
note("hot", hot());

function neverCalled() { return function deeper() { return neverCalled; }; }

globalThis.result = log.join(";");
"#;

const EXPECTED: &str = "counter=3:6;loops=0,1,2,a1,b2,boom,sw;\
    classes=hi dee/7,DEE,dee!,10,true,true,proto+literal,3;defaults=5,6,body;\
    names=120;outer|inner,inner|outer;eval=42,3,3;annexB=undefined,block,20;\
    gen=0112358,true;hot=5995";

fn corpus_snapshot() -> &'static [u8] {
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    SNAPSHOT.get_or_init(|| crate::compile_snapshot(CORPUS).expect("corpus parses"))
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    // Compile every function on its first call: each compilation analyzes the still-encoded
    // functions nested in it.
    engine.set_tier_threshold(0);
    engine
}

fn run_shared(it: &mut Interp) -> String {
    let body = it
        .shared_snapshot_program(corpus_snapshot())
        .expect("snapshot decodes");
    if let Err(Abrupt::Throw(error)) = it.run_classic_program(&body, None) {
        panic!("corpus threw: {}", render(it, &error));
    }
    eval_string(it, "result")
}

fn global_function(it: &mut Interp, name: &str) -> Rc<crate::ast::Function> {
    let env = it.global_env.clone();
    let Ok(Value::Obj(object)) = it.get_var(name, &env) else {
        panic!("{name} is a global function");
    };
    let borrowed = object.borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("{name} is an ordinary function");
    };
    user.func.clone()
}

#[test]
fn lazily_decoded_bodies_behave_like_parsed_source_in_every_tier() {
    for tier in TIERS {
        let mut parsed = engine(tier);
        let it = &mut *parsed.interp;
        eval_string(it, CORPUS);
        let expected = eval_string(it, "result");
        assert_eq!(
            expected,
            EXPECTED.replace("    ", ""),
            "{tier:?}: parsed source"
        );

        for summaries in [false, true] {
            let mut shared = engine(tier);
            let it = &mut *shared.interp;
            // Static bootstraps keep summaries for their whole evaluation; without the scope,
            // every compilation summarizes its inner functions afresh.
            let scope = summaries.then(crate::bytecode::capture_summaries);
            assert_eq!(
                run_shared(it),
                expected,
                "{tier:?} summaries kept: {summaries}"
            );
            drop(scope);

            // The functions that ran were decoded once; the one that never ran was not.
            let ran = global_function(it, "counterFactory");
            assert!(ran.body.materialized().is_some(), "{tier:?}");
            let never = global_function(it, "neverCalled");
            assert!(never.body.encoding().is_some(), "{tier:?}: never called");
            if tier != Tier::Interp {
                assert!(
                    global_function(it, "hot")
                        .code
                        .get()
                        .is_some_and(Option::is_some),
                    "{tier:?}: compiled around encoded inner functions"
                );
            }

            // Another Realm runs the same decoded program and compiled code.
            let child = it.create_realm();
            let child_result = in_realm(it, &child, run_shared);
            assert_eq!(child_result, expected, "{tier:?} child Realm");
            let child_never = in_realm(it, &child, |it| global_function(it, "neverCalled"));
            assert!(Rc::ptr_eq(&never, &child_never), "{tier:?}: one shared AST");
            assert!(
                never.body.encoding().is_some(),
                "{tier:?}: still never called"
            );
        }
    }
}

/// Memory accounting reports what an Agent owns: an encoded body is the process's static
/// bytes, and measuring a function never decodes it.
#[test]
fn memory_accounting_counts_only_materialized_bodies() {
    let mut engine = engine(Tier::Bytecode);
    let it = &mut *engine.interp;
    assert_eq!(run_shared(it), EXPECTED.replace("    ", ""));
    let never = global_function(it, "neverCalled");
    let measure = |function: &crate::ast::Function| {
        let mut visitor = crate::memory::Visitor::default();
        crate::ast::scan_function_retained_memory(function, &mut visitor)
    };
    let encoded = measure(&never);
    assert!(never.body.encoding().is_some(), "measuring does not decode");
    assert!(!never.body.is_empty());
    let materialized = measure(&never);
    assert!(
        materialized > encoded,
        "{materialized} bytes once decoded, {encoded} before"
    );
}

/// Capture summaries reproduce every outcome of walking an inner function, including the ones
/// that leave the enclosing function on the tree-walker: a direct eval or `with` statement in
/// an inner function of a function compiled without them.
#[test]
fn capture_summaries_preserve_unanalyzable_inner_functions() {
    const SOURCE: &str = r#"
        function withEval(a) { function inner(s) { return eval(s); } return inner("a + 1"); }
        function withWith(o) { function inner() { with (o) { return p; } } return inner(); }
        function strictArrows() {
            "use strict";
            const self = this;
            return [() => this === self, function () { return this; }];
        }
        function annexInner() { function inner() { { function f() {} } return f; } return inner(); }
        globalThis.out = [withEval(1), withWith({ p: 2 }), strictArrows.call(3)[0](),
            typeof annexInner()].join();
    "#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] =
        SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).expect("source parses"));
    for tier in TIERS {
        let mut engine = engine(tier);
        let it = &mut *engine.interp;
        let _summaries = crate::bytecode::capture_summaries();
        let body = it.shared_snapshot_program(snapshot).expect("decodes");
        if let Err(Abrupt::Throw(error)) = it.run_classic_program(&body, None) {
            panic!("{tier:?} threw: {}", render(it, &error));
        }
        assert_eq!(eval_string(it, "out"), "2,2,true,function", "{tier:?}");
    }
}

/// The immediately invoked function of `body[0]` (`(function () { … })();`).
fn bootstrap_function(body: &[crate::ast::Stmt]) -> Rc<crate::ast::Function> {
    use crate::ast::{Expr, Stmt};
    let Stmt::Expr(Expr::Call { callee, .. }) = &body[0] else {
        panic!("an immediately invoked bootstrap function")
    };
    let mut callee = &**callee;
    while let Expr::Paren(inner) = callee {
        callee = inner;
    }
    let Expr::Func(function) = callee else {
        panic!("a function expression")
    };
    function.clone()
}

fn has_native_code(function: &crate::ast::Function) -> bool {
    function
        .code
        .get()
        .and_then(Option::as_ref)
        .is_some_and(|chunk| chunk.jit.get().is_some_and(|code| code.is_some()))
}

/// A static snapshot's run-once bootstrap function (`Function::run_once`) runs once per Realm,
/// so its bytecode stays on the VM: it never gets native code. The functions it defines keep
/// their tiering, and so does the same source when it is parsed.
#[test]
fn run_once_bootstrap_functions_never_get_native_code() {
    const SOURCE: &str = r#"(function () {
        let total = 0;
        for (let i = 0; i < 50; i++) total += i;
        function hot(n) { let sum = 0; for (let i = 0; i < n; i++) sum += i; return sum; }
        globalThis.bootstrapTotal = total + hot(10);
        globalThis.hot = hot;
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] =
        SNAPSHOT.get_or_init(|| crate::compile_host_snapshot(SOURCE).expect("source parses"));
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let it = &mut *engine.interp;
    let program = it.shared_snapshot_program(snapshot).expect("decodes");
    if let Err(Abrupt::Throw(error)) = it.run_classic_program(&program, None) {
        panic!("snapshot threw: {}", render(it, &error));
    }
    assert_eq!(eval_string(it, "bootstrapTotal"), "1270");
    let bootstrap = bootstrap_function(&program);
    assert!(bootstrap.run_once);
    assert!(
        bootstrap.code.get().is_some_and(Option::is_some),
        "explicitly forced bytecode compiles the bootstrap on its first call"
    );
    assert!(!has_native_code(&bootstrap), "but it runs on the VM");
    let hot = global_function(it, "hot");
    assert!(!hot.run_once);
    assert!(
        has_native_code(&hot),
        "the functions it defines keep their tiering"
    );

    let mut parsed_engine = Engine::new();
    parsed_engine.set_tier(Tier::Jit);
    parsed_engine.set_tier_threshold(0);
    let it = &mut *parsed_engine.interp;
    let Ok(parsed) = crate::parser::parse_script(SOURCE, false) else {
        panic!("source parses")
    };
    if let Err(Abrupt::Throw(error)) = it.run_classic_program(&parsed, None) {
        panic!("parsed source threw: {}", render(it, &error));
    }
    assert_eq!(eval_string(it, "bootstrapTotal"), "1270");
    let parsed_bootstrap = bootstrap_function(&parsed);
    assert!(!parsed_bootstrap.run_once);
    assert!(
        has_native_code(&parsed_bootstrap),
        "parsed source keeps today's tiering"
    );
}

#[test]
fn run_once_outer_cleanup_fragments_remain_on_the_vm() {
    const SOURCE: &str = r#"(function () {
        let total = 0;
        try { for (let index = 0; index < 3000; index++) total += index; }
        finally { globalThis.bootstrapCleanup = 1; }
        globalThis.bootstrapTotal = total;
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] =
        SNAPSHOT.get_or_init(|| crate::compile_host_snapshot(SOURCE).expect("source parses"));
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        let it = &mut *engine.interp;
        let program = it.shared_snapshot_program(snapshot).expect("decodes");
        let bootstrap = bootstrap_function(&program);
        assert!(bootstrap.run_once);
        crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.set(0));
        crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
        if let Err(Abrupt::Throw(error)) = it.run_classic_program(&program, None) {
            panic!("snapshot threw: {}", render(it, &error));
        }
        assert_eq!(
            eval_string(it, "bootstrapTotal+':'+bootstrapCleanup"),
            "4498500:1"
        );
        assert!(
            crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get()) > 0
        );
        assert_eq!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()), 0);
        assert!(!has_native_code(&bootstrap));
    }
}

/// A cold run-once bootstrap body is not compiled on its first call for containing a short
/// loop: each Realm interprets it until the
/// call-count threshold compiles it (to bytecode that stays on the VM). Results are the same
/// in every Realm and in parsed source.
#[test]
fn huge_run_once_bootstrap_bodies_are_interpreted_until_the_call_threshold() {
    const STATEMENTS: usize = 1000;
    fn source() -> String {
        format!(
            "(function () {{ let total = 0; for (let i = 0; i < 3; i++) total += i; {} \
             globalThis.bootstrapTotal = total; }})();",
            "total += 1; ".repeat(STATEMENTS)
        )
    }
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] =
        SNAPSHOT.get_or_init(|| crate::compile_host_snapshot(&source()).expect("source parses"));
    let expected = (3 + STATEMENTS).to_string();
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        let threshold = it.tier_threshold;
        let program = it.shared_snapshot_program(snapshot).expect("decodes");
        let bootstrap = bootstrap_function(&program);
        assert!(bootstrap.run_once);
        let run = |it: &mut Interp| {
            let program = it.shared_snapshot_program(snapshot).expect("decodes");
            if let Err(Abrupt::Throw(error)) = it.run_classic_program(&program, None) {
                panic!("snapshot threw: {}", render(it, &error));
            }
            eval_string(it, "bootstrapTotal")
        };
        assert_eq!(run(it), expected, "{tier:?}");
        assert!(bootstrap.code.get().is_none(), "{tier:?}: interpreted");
        // Each later Realm of the Agent calls it once more.
        for realm in 1..=threshold {
            let child = it.create_realm();
            assert_eq!(
                in_realm(it, &child, run),
                expected,
                "{tier:?} Realm {realm}"
            );
        }
        assert!(
            bootstrap.code.get().is_some_and(Option::is_some),
            "{tier:?}: the call-count threshold compiles it"
        );
        assert!(!has_native_code(&bootstrap), "{tier:?}: on the VM");
    }

    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    let it = &mut *engine.interp;
    let Ok(parsed) = crate::parser::parse_script(&source(), false) else {
        panic!("source parses")
    };
    if let Err(Abrupt::Throw(error)) = it.run_classic_program(&parsed, None) {
        panic!("parsed source threw: {}", render(it, &error));
    }
    assert_eq!(eval_string(it, "bootstrapTotal"), expected);
    assert!(
        bootstrap_function(&parsed).code.get().is_none(),
        "parsed source also defers cold-loop compilation"
    );
}
