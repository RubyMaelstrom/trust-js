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
use crate::value::Callable;
use crate::value::ValueView;
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

/// While alive, capture scans summarize still-encoded functions by decoding and scanning them
/// instead of reading the summaries the snapshot stores (`computed`), or read them (`!computed`).
struct ComputeSummaries(bool);

impl ComputeSummaries {
    fn new(computed: bool) -> Self {
        let flag = &crate::bytecode::capture_summary_tests::COMPUTE_INSTEAD;
        Self(flag.with(|flag| flag.replace(computed)))
    }
}

impl Drop for ComputeSummaries {
    fn drop(&mut self) {
        let flag = &crate::bytecode::capture_summary_tests::COMPUTE_INSTEAD;
        flag.with(|flag| flag.set(self.0));
    }
}

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
    let env = it.global_env;
    let Ok(ValueView::Obj(object)) = (it.get_var(name, &env)).map(|__value| (__value).view())
    else {
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

        for computed in [false, true] {
            let mut shared = engine(tier);
            let it = &mut *shared.interp;
            // Scans read the summaries the snapshot stores, or, as before it stored them,
            // decode and scan each still-encoded inner function: the results are the same.
            let compute = ComputeSummaries::new(computed);
            assert_eq!(
                run_shared(it),
                expected,
                "{tier:?} summaries computed: {computed}"
            );
            drop(compute);

            // The functions that ran were decoded once; the one that never ran was not.
            let ran = global_function(it, "counterFactory");
            assert!(ran.body.materialized().is_some(), "{tier:?}");
            let never = global_function(it, "neverCalled");
            assert!(
                never.body.materialized().is_none(),
                "{tier:?}: never called"
            );
            if tier != Tier::Interp {
                assert!(
                    global_function(it, "hot")
                        .runtime()
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
                never.body.materialized().is_none(),
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
    assert!(
        never.body.materialized().is_none(),
        "measuring does not decode"
    );
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
        .runtime()
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
        bootstrap.runtime().code.get().is_some_and(Option::is_some),
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
        assert!(
            bootstrap.runtime().code.get().is_none(),
            "{tier:?}: interpreted"
        );
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
            bootstrap.runtime().code.get().is_some_and(Option::is_some),
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
        bootstrap_function(&parsed).runtime().code.get().is_none(),
        "parsed source also defers cold-loop compilation"
    );
}

/// Every function of a Script in preorder, with whether it is a derived class's constructor.
/// Reading a function's statements materializes them if they are still encoded.
fn functions_of(body: &[crate::ast::Stmt]) -> Vec<(Rc<crate::ast::Function>, bool)> {
    let mut functions = Vec::new();
    visit_functions(body, true, &mut |function, derived| {
        functions.push((function.clone(), derived));
    });
    functions
}

/// The functions that `body` contains directly, not inside other functions: what its scan
/// meets as inner functions. Nothing is materialized.
fn inner_functions_of(body: &[crate::ast::Stmt]) -> Vec<(Rc<crate::ast::Function>, bool)> {
    let mut functions = Vec::new();
    visit_functions(body, false, &mut |function, derived| {
        functions.push((function.clone(), derived));
    });
    functions
}

/// Call `visit` on each function of `body` in preorder, with whether it is a derived class's
/// constructor, before reading the function's parameters and statements (when `deep`).
fn visit_functions(
    body: &[crate::ast::Stmt],
    deep: bool,
    visit: &mut dyn FnMut(&Rc<crate::ast::Function>, bool),
) {
    use crate::ast::{
        ArrayElem, ArrayPatElem, Class, Expr, ForInit, Function, MemberKind, Pattern, PropDef,
        PropKey, Stmt,
    };
    struct Collect<'v>(&'v mut dyn FnMut(&Rc<Function>, bool), bool);
    impl Collect<'_> {
        fn function(&mut self, function: &Rc<Function>, derived: bool) {
            (self.0)(function, derived);
            if !self.1 {
                return;
            }
            for parameter in &function.params {
                self.pattern(&parameter.pattern);
                if let Some(default) = &parameter.default {
                    self.expr(default);
                }
            }
            self.stmts(&function.body);
        }
        fn class(&mut self, class: &Class) {
            for decorator in &class.decorators {
                self.expr(decorator);
            }
            if let Some(heritage) = &class.superclass {
                self.expr(heritage);
            }
            for member in &class.members {
                for decorator in &member.decorators {
                    self.expr(decorator);
                }
                self.key(&member.key);
                if let Some(function) = &member.func {
                    let derived =
                        class.superclass.is_some() && member.kind == MemberKind::Constructor;
                    self.function(function, derived);
                }
                if let Some(value) = &member.value {
                    self.expr(value);
                }
            }
        }
        fn key(&mut self, key: &PropKey) {
            if let PropKey::Computed(key) = key {
                self.expr(key);
            }
        }
        fn pattern(&mut self, pattern: &Pattern) {
            match pattern {
                Pattern::Ident(_) => {}
                Pattern::Array(elements) => {
                    for element in elements {
                        match element {
                            ArrayPatElem::Hole => {}
                            ArrayPatElem::Elem { pattern, default } => {
                                self.pattern(pattern);
                                if let Some(default) = default {
                                    self.expr(default);
                                }
                            }
                            ArrayPatElem::Rest(pattern) => self.pattern(pattern),
                        }
                    }
                }
                Pattern::Object(object) => {
                    for property in &object.props {
                        self.key(&property.key);
                        self.pattern(&property.value);
                        if let Some(default) = &property.default {
                            self.expr(default);
                        }
                    }
                }
                Pattern::Member(target) => self.expr(target),
            }
        }
        fn elements(&mut self, elements: &[ArrayElem]) {
            for element in elements {
                if let ArrayElem::Item(e) | ArrayElem::Spread(e) = element {
                    self.expr(e);
                }
            }
        }
        fn decls(&mut self, decls: &[(Pattern, Option<Expr>)]) {
            for (pattern, init) in decls {
                self.pattern(pattern);
                if let Some(init) = init {
                    self.expr(init);
                }
            }
        }
        fn expr(&mut self, e: &Expr) {
            match e {
                Expr::Paren(e)
                | Expr::ToStr(e)
                | Expr::Await(e)
                | Expr::OptionalChain(e)
                | Expr::Unary { arg: e, .. }
                | Expr::Update { arg: e, .. }
                | Expr::Member { obj: e, .. }
                | Expr::PrivateIn { obj: e, .. } => self.expr(e),
                Expr::Yield { arg, .. } => {
                    if let Some(arg) = arg {
                        self.expr(arg);
                    }
                }
                Expr::Array(elements) => self.elements(elements),
                Expr::Object(properties) => {
                    for property in properties {
                        match property {
                            PropDef::KeyValue { key, value } | PropDef::Cover { key, value } => {
                                self.key(key);
                                self.expr(value);
                            }
                            PropDef::Method { key, func }
                            | PropDef::Getter { key, func }
                            | PropDef::Setter { key, func } => {
                                self.key(key);
                                self.function(func, false);
                            }
                            PropDef::Spread(e) | PropDef::Proto(e) => self.expr(e),
                        }
                    }
                }
                Expr::Func(function) => self.function(function, false),
                Expr::Class(class) => self.class(class),
                Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
                    self.expr(left);
                    self.expr(right);
                }
                Expr::Assign { target, value, .. } => {
                    self.expr(target);
                    self.expr(value);
                }
                Expr::Cond { test, cons, alt } => {
                    self.expr(test);
                    self.expr(cons);
                    self.expr(alt);
                }
                Expr::Call { callee, args, .. } | Expr::New { callee, args, .. } => {
                    self.expr(callee);
                    self.elements(args);
                }
                Expr::Index { obj, index, .. } => {
                    self.expr(obj);
                    self.expr(index);
                }
                Expr::Seq(items) => {
                    for item in items {
                        self.expr(item);
                    }
                }
                Expr::TaggedTemplate { tag, subs, .. } => {
                    self.expr(tag);
                    for sub in subs.iter() {
                        self.expr(sub);
                    }
                }
                Expr::ImportCall { spec, options, .. } => {
                    self.expr(spec);
                    if let Some(options) = options {
                        self.expr(options);
                    }
                }
                Expr::Num(_)
                | Expr::BigInt(_)
                | Expr::Str(_)
                | Expr::Bool(_)
                | Expr::Null
                | Expr::Undefined
                | Expr::Ident(..)
                | Expr::This
                | Expr::Regex { .. }
                | Expr::Super
                | Expr::ImportMeta
                | Expr::NewTarget => {}
            }
        }
        fn stmts(&mut self, stmts: &[Stmt]) {
            for statement in stmts {
                self.stmt(statement);
            }
        }
        fn stmt(&mut self, statement: &Stmt) {
            match statement {
                Stmt::Expr(e) | Stmt::Throw(e) => self.expr(e),
                Stmt::Return(value) => {
                    if let Some(value) = value {
                        self.expr(value);
                    }
                }
                Stmt::VarDecl { decls, .. } => self.decls(decls),
                Stmt::FuncDecl(function) => self.function(function, false),
                Stmt::If { test, cons, alt } => {
                    self.expr(test);
                    self.stmt(cons);
                    if let Some(alt) = alt {
                        self.stmt(alt);
                    }
                }
                Stmt::Block(body) => self.stmts(body),
                Stmt::While { test, body } | Stmt::DoWhile { body, test } => {
                    self.expr(test);
                    self.stmt(&body.statement);
                }
                Stmt::For {
                    init,
                    test,
                    update,
                    body,
                } => {
                    match init.as_deref() {
                        Some(ForInit::VarDecl { decls, .. }) => self.decls(decls),
                        Some(ForInit::Expr(e)) => self.expr(e),
                        None => {}
                    }
                    for e in [test, update].into_iter().flatten() {
                        self.expr(e);
                    }
                    self.stmt(&body.statement);
                }
                Stmt::ForInOf {
                    left, right, body, ..
                } => {
                    self.pattern(left);
                    self.expr(right);
                    self.stmt(&body.statement);
                }
                Stmt::Try {
                    block,
                    handler,
                    finalizer,
                } => {
                    self.stmts(block);
                    if let Some((parameter, body)) = handler {
                        if let Some(parameter) = parameter {
                            self.pattern(parameter);
                        }
                        self.stmts(body);
                    }
                    if let Some(finalizer) = finalizer {
                        self.stmts(finalizer);
                    }
                }
                Stmt::Switch { disc, cases } => {
                    self.expr(disc);
                    for case in cases {
                        if let Some(test) = &case.test {
                            self.expr(test);
                        }
                        self.stmts(&case.body);
                    }
                }
                Stmt::Labeled { body, .. } | Stmt::ExportDecl(body) | Stmt::ExportDefault(body) => {
                    self.stmt(body)
                }
                Stmt::With { obj, body } => {
                    self.expr(obj);
                    self.stmt(body);
                }
                Stmt::ClassDecl(class) => self.class(class),
                Stmt::Break(_)
                | Stmt::Continue(_)
                | Stmt::Empty
                | Stmt::Debugger
                | Stmt::Import(_)
                | Stmt::ExportNamed { .. }
                | Stmt::ExportAll { .. } => {}
            }
        }
    }
    let mut collect = Collect(visit, deep);
    collect.stmts(body);
}

/// The engine's own static glue as its build scripts assemble it for their snapshots (web's in
/// trust-js-web/build.rs's order; node's files, each in a block as its build script wraps most
/// of them, in name order), and the corpus above.
fn real_scripts() -> Vec<(String, String)> {
    const WEB: &[&str] = &[
        "preamble.js",
        "events.js",
        "messaging.js",
        "encoding.js",
        "serialize.js",
        "url.js",
        "urlpattern.js",
        "streams.js",
        "streams-webidl.js",
        "writable.js",
        "compression.js",
        "blob.js",
        "fetch.js",
        "websocket.js",
        "eventsource.js",
        "server.js",
        "crypto.js",
        "platform.js",
        "wasm.js",
    ];
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let read = |path: String| {
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
    };
    let mut web = String::from("(() => {\n");
    for file in WEB {
        web.push_str(&read(format!("{root}/trust-js-web/src/js/{file}")));
    }
    web.push_str("\n})();");
    let mut node_files: Vec<_> = std::fs::read_dir(format!("{root}/trust-js-node/src/js"))
        .expect("node glue directory")
        .map(|entry| entry.expect("node glue file").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "js"))
        .collect();
    node_files.sort();
    let mut node = String::from("(() => {\n");
    for path in node_files {
        node.push_str("{\n");
        node.push_str(&read(path.display().to_string()));
        node.push_str("\n}\n");
    }
    node.push_str("\n})();");
    vec![
        ("web glue".into(), web),
        ("node glue".into(), node),
        ("corpus".into(), CORPUS.into()),
    ]
}

/// What a function's plan decided, in a canonical form: its kind, why, capacity and prefix, and
/// the capture analysis it keeps for compiling the function.
fn plan_facts(function: &crate::ast::Function, derived: bool) -> String {
    let plan = function.scope_plan(derived);
    let prefix: Vec<String> = plan
        .prefix
        .iter()
        .map(|binding| format!("{}:{}:{}", binding.name, binding.lexical, binding.mutable))
        .collect();
    let scan = plan.scan.as_ref().map(|scan| {
        let mut captured: Vec<&String> = scan.captured.iter().collect();
        captured.sort();
        let mut runtime: Vec<&String> = scan.runtime.iter().collect();
        runtime.sort();
        format!(
            "captured {captured:?} env_this {} homed {:?} runtime {runtime:?} eval {} with {}",
            scan.env_this, scan.homed, scan.direct_eval, scan.with
        )
    });
    format!(
        "{:?} {:?} capacity {} prefix {prefix:?} scan {scan:?}",
        plan.kind, plan.why, plan.capacity
    )
}

/// Plan every function of `body`, the Script's own inner functions first (as ScriptEvaluation
/// does), then each function before the functions nested in it are read, as its first call
/// plans it before its body makes their closures; return each one's plan and its outer context,
/// in preorder.
fn plans_of(body: &[crate::ast::Stmt], strict: bool) -> Vec<(String, String)> {
    crate::bytecode::scope_plan::plan_script(body, strict);
    let mut plans = Vec::new();
    visit_functions(body, true, &mut |function, derived| {
        let facts = plan_facts(function, derived);
        plans.push((facts, format!("{:?}", function.outer_context_rc())));
    });
    plans
}

fn decode_leaked(snapshot: Vec<u8>) -> Vec<crate::ast::Stmt> {
    crate::snapshot::decode_static(Box::leak(snapshot.into_boxed_slice())).expect("decodes")
}

/// Each function's capture summary as the snapshot stores it, against walking the parsed
/// function whole: the encoder's summaries, each made from the summaries of the functions
/// nested in it, are the walks'. Returns the number of functions, the snapshot's size and the
/// bytes its summaries take (with their lengths).
fn check_stored_summaries(name: &str, source: &str) -> (usize, usize, usize) {
    let parsed = crate::parser::parse_script(source, false)
        .ok()
        .unwrap_or_else(|| panic!("{name} parses"));
    let snapshot = crate::compile_host_snapshot(source).expect("compiles");
    let size = snapshot.len();
    let lazy = decode_leaked(snapshot);
    let parsed_functions = functions_of(&parsed);
    let lazy_functions = functions_of(&lazy);
    assert_eq!(parsed_functions.len(), lazy_functions.len(), "{name}");
    let mut summary_bytes = 0;
    for (index, ((parsed, _), (lazy, _))) in
        parsed_functions.iter().zip(&lazy_functions).enumerate()
    {
        let stored = lazy.body.capture_summary().expect("a stored summary");
        assert_eq!(
            stored.bytes(),
            crate::bytecode::capture_summary_tests::walked(parsed),
            "{name}: function {index} ({:?})",
            parsed.name
        );
        let len = stored.bytes().len();
        summary_bytes += len + (usize::BITS - len.leading_zeros()).div_ceil(7).max(1) as usize;
    }
    (parsed_functions.len(), size, summary_bytes)
}

#[test]
fn stored_summaries_equal_walking_each_function_of_real_glue() {
    for (name, source) in real_scripts() {
        let (functions, ..) = check_stored_summaries(&name, &source);
        assert!(functions >= 40, "{name}: {functions} functions");
    }
}

/// Plans made from the summaries a snapshot stores equal the plans made from summaries computed
/// by decoding and scanning each inner function, and both equal the plans of the parsed source.
/// A function whose body was still encoded when the function around it was planned gets no
/// outer context from that plan, whichever summary it was given (the parsed source's functions
/// all get one), so outer contexts are compared between the two snapshot trees only.
fn check_plans(name: &str, source: &str) -> usize {
    let parsed = crate::parser::parse_script(source, false)
        .ok()
        .unwrap_or_else(|| panic!("{name} parses"));
    let snapshot = crate::compile_host_snapshot(source).expect("compiles");
    let stored = plans_of(&decode_leaked(snapshot.clone()), false);
    let computed = {
        let _compute = ComputeSummaries::new(true);
        plans_of(&decode_leaked(snapshot), false)
    };
    let scanned = plans_of(&parsed, false);
    assert_eq!(stored.len(), scanned.len(), "{name}");
    for (index, ((stored, computed), scanned)) in
        stored.iter().zip(&computed).zip(&scanned).enumerate()
    {
        assert_eq!(
            stored, computed,
            "{name}: function {index}, stored and computed"
        );
        assert_eq!(
            stored.0, scanned.0,
            "{name}: function {index}, stored and parsed"
        );
    }
    stored.len()
}

#[test]
fn plans_from_stored_summaries_equal_plans_from_scanning_real_glue() {
    for (name, source) in real_scripts() {
        check_plans(&name, &source);
    }
}

/// What planning a snapshot's bootstrap function found: the stored summaries it applied, the
/// bodies that summarizing by scanning instead decodes, and the bodies the plan decodes
/// temporarily for other syntactic questions (the `this`, `arguments` and `super` flags of the
/// arrows in it, which other plans and closure creation ask too).
struct BootstrapPlan {
    applied: usize,
    computed_decodes: usize,
    other_decodes: usize,
}

/// A static snapshot's bootstrap function, planned at its first call as every Agent plans it:
/// its plan reads each inner function's stored summary and decodes no body to summarize one.
/// Summarizing instead by scanning, as before snapshots stored summaries, decodes them all.
fn check_bootstrap_plan(name: &str, source: &str) -> BootstrapPlan {
    use crate::bytecode::capture_summary_tests::{SCAN_DECODES, STORED_SUMMARIES_APPLIED};
    use crate::snapshot::DECODED_BODIES;
    let snapshot = crate::compile_host_snapshot(source).expect("compiles");
    let plan = |computed: bool| {
        let program = decode_leaked(snapshot.clone());
        crate::bytecode::scope_plan::plan_script(&program, false);
        let bootstrap = bootstrap_function(&program);
        bootstrap.body.prepare_to_run();
        let _compute = ComputeSummaries::new(computed);
        SCAN_DECODES.with(|count| count.set(0));
        STORED_SUMMARIES_APPLIED.with(|count| count.set(0));
        DECODED_BODIES.with(|count| count.set(0));
        let facts = plan_facts(&bootstrap, false);
        let decodes = SCAN_DECODES.with(std::cell::Cell::get);
        let applied = STORED_SUMMARIES_APPLIED.with(std::cell::Cell::get);
        let all_decodes = DECODED_BODIES.with(std::cell::Cell::get);
        let inner = inner_functions_of(&bootstrap.body);
        let encoded = inner
            .iter()
            .filter(|(function, _)| function.body.materialized().is_none())
            .count();
        (facts, decodes, applied, all_decodes, inner.len(), encoded)
    };
    let (stored_facts, stored_decodes, applied, other_decodes, inner, encoded) = plan(false);
    let (computed_facts, computed_decodes, ..) = plan(true);
    assert_eq!(stored_facts, computed_facts, "{name}");
    assert_eq!(
        stored_decodes, 0,
        "{name}: the plan decoded bodies to summarize them"
    );
    assert_eq!(
        applied, inner,
        "{name}: one stored summary for each inner function"
    );
    assert_eq!(encoded, inner, "{name}: the inner functions stay encoded");
    assert!(computed_decodes > 0, "{name}");
    BootstrapPlan {
        applied,
        computed_decodes,
        other_decodes,
    }
}

#[test]
fn bootstrap_plans_read_stored_summaries_and_decode_no_inner_body() {
    for (name, source) in real_scripts() {
        if name != "corpus" {
            check_bootstrap_plan(&name, &source);
        }
    }
}

/// The same checks over another Script, such as a browser's assembled platform prelude:
/// `TRUST_JS_SUMMARY_SOURCE=path cargo test -p trust-js --lib --features embed -- --ignored
/// summary_source` (DIAGNOSTICS.md).
#[test]
#[ignore]
fn summary_source_checks() {
    let path = std::env::var("TRUST_JS_SUMMARY_SOURCE").expect("TRUST_JS_SUMMARY_SOURCE");
    let source = std::fs::read_to_string(&path).expect("readable source");
    let (functions, size, summary_bytes) = check_stored_summaries(&path, &source);
    check_plans(&path, &source);
    let plan = check_bootstrap_plan(&path, &source);
    eprintln!(
        "{path}: {functions} functions, a {size}-byte snapshot of which summaries take \
         {summary_bytes}; the bootstrap's plan applied {} stored summaries, decoded {} bodies \
         for other questions, and decodes {} to compute the summaries",
        plan.applied, plan.other_decodes, plan.computed_decodes
    );
}
