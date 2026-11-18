//! The apply form (P3 stage 3 steps 6-7): `<f>.apply(<t>, arguments)` in a body whose other uses
//! of `arguments` are only `arguments.length` and `arguments[k]` calls `f` with the actual
//! arguments, a mapped index reading its formal, and makes no arguments object
//! (`bytecode::lazy_arguments_rewrite`, `Op::CallApplyFrameArgs`). Every program runs at every
//! tier and must give the tree-walker's result; the optimizing tier's configuration is the
//! environment's (the suites run these with it off, on, eager, with forced exits and under GC
//! stress).

use crate::bytecode::{Op, Tier};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// What one run did: the apply form's hits and misses, and the arguments objects compiled code
/// made.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    hits: usize,
    misses: usize,
    objects: usize,
}

fn reset() {
    crate::bytecode::TEST_APPLY_FORM_HITS.with(|count| count.set(0));
    crate::bytecode::TEST_APPLY_FORM_MISSES.with(|count| count.set(0));
    crate::interpreter::TEST_COMPILED_ARGUMENTS_OBJECTS.with(|count| count.set(0));
}

fn counts() -> Counts {
    Counts {
        hits: crate::bytecode::TEST_APPLY_FORM_HITS.with(|count| count.get()),
        misses: crate::bytecode::TEST_APPLY_FORM_MISSES.with(|count| count.get()),
        objects: crate::interpreter::TEST_COMPILED_ARGUMENTS_OBJECTS.with(|count| count.get()),
    }
}

fn run(tier: Tier, threshold: Option<u32>, source: &str) -> (String, Counts) {
    reset();
    let mut engine = Engine::new();
    engine.set_tier(tier);
    if let Some(threshold) = threshold {
        engine.set_tier_threshold(threshold);
    }
    let result = eval(&mut engine, source);
    (result, counts())
}

/// Run `source` at the tree-walker, the VM and the JIT (each from the first call), and the JIT
/// at the default thresholds: every run gives the tree-walker's result, which is returned with
/// the VM's and the eager JIT's counts.
fn same_at_every_tier(source: &str) -> (String, Counts, Counts) {
    let (expected, _) = run(Tier::Interp, None, source);
    let (vm, vm_counts) = run(Tier::Bytecode, Some(0), source);
    assert_eq!(vm, expected, "the VM");
    let (jit, jit_counts) = run(Tier::Jit, Some(0), source);
    assert_eq!(jit, expected, "the JIT from the first call");
    let (tiered, _) = run(Tier::Jit, None, source);
    assert_eq!(tiered, expected, "the JIT at the default thresholds");
    (expected, vm_counts, jit_counts)
}

/// The operations of global function `name`'s compiled chunk.
fn ops_of(engine: &mut Engine, name: &str) -> Vec<Op> {
    chunk_of(engine, name).jit_ops().to_vec()
}

/// Global function `name`'s compiled (first-stage) chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> std::rc::Rc<crate::bytecode::Chunk> {
    let env = engine.interp.global_env;
    let value: Value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("a bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .unwrap_or_else(|| panic!("{name} stayed interpreted"))
}

/// Whether the form is on (TRUST_JS_JIT_NO_APPLY_FORM turns it off; then every program must
/// still give the tree-walker's results, and no body holds the operation).
fn form() -> bool {
    !crate::bytecode::apply_form_disabled()
}

/// Whether global function `name` of `source` (after it ran at the JIT from the first call) holds
/// the apply form's operation; under TRUST_JS_JIT_NO_APPLY_FORM, that none does.
fn rewritten(source: &str, name: &str) -> bool {
    form() && rewritten_now(source, name)
}

fn rewritten_now(source: &str, name: &str) -> bool {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    ops_of(&mut engine, name)
        .iter()
        .any(|op| matches!(op, Op::CallApplyFrameArgs(..)))
}

const TARGET: &str = r#"
var out = [];
function show(x) {
    if (x === undefined) return 'u';
    if (x === null) return 'n';
    if (typeof x === 'function') return 'fn';
    if (typeof x === 'object') return x.tag !== undefined ? String(x.tag) : 'obj';
    return String(x);
}
function target(a, b, c) {
    var parts = [show(this)];
    for (var k = 0; k < arguments.length; k++) parts.push(show(arguments[k]));
    return parts.join(',');
}
var o = { tag: 'o', inner: { tag: 'in' } };
"#;

#[test]
fn the_apply_form_passes_the_actual_arguments_at_every_tier() {
    let source = format!(
        "{TARGET}{}",
        r#"
        function w0() { return target.apply(this, arguments); }
        function wStrict(a, b) { 'use strict'; a = 'x'; var r = target.apply(this, arguments); return r; }
        // A strict body's call in tail position is a tail call, outside the form's reach.
        function wTail(a) { 'use strict'; return target.apply(this, arguments); }
        function wInner() { return target.apply(this.inner, arguments); }
        function wLen() { return arguments.length + ':' + arguments[1] + ':' + target.apply(this, arguments); }
        function wChain() { return w0.apply(this, arguments); }
        function wTwice() { return target.apply(this, arguments) + '/' + target.apply(o, arguments); }
        function wCond(x) { return target.apply(x ? this : o, arguments); }
        function wAnd(x) { return target.apply(x && this, arguments); }
        for (var k = 0; k < 40; k++) {
            out.push(w0.call(o, 1, 2, 3), w0.call(o), w0.call(o, 1, 2, 3, 4, 5), w0(k));
            out.push(wStrict.call(o, 7, 8), wStrict.call(o, 7), wStrict.call(o, 7, 8, 9), wTail.call(o, k));
            out.push(wInner.call(o, k), wLen.call(o, 'a', 'b'), wChain.call(o, k, 'c'));
            out.push(wTwice.call(o, k), wCond.call(o, true, k), wCond.call(o, false), wAnd.call(o, true, k), wAnd.call(o, 0));
        }
        out.join(';')
        "#
    );
    let (_, vm, jit) = same_at_every_tier(&source);
    if !form() {
        return;
    }
    // The objects: wTail's, wCond's and wAnd's, made at entry (40, 80 and 80).
    for (tier, counts) in [("vm", vm), ("jit", jit)] {
        assert!(counts.hits > 0, "{tier}: no apply form ran: {counts:?}");
        assert_eq!(counts.misses, 0, "{tier}: {counts:?}");
        assert_eq!(counts.objects, 200, "{tier}: {counts:?}");
    }
    for name in ["w0", "wStrict", "wInner", "wLen", "wChain", "wTwice"] {
        assert!(rewritten(&source, name), "{name} keeps its object");
    }
    // A branch in `<t>` (`?:`, `&&`): a jump lands between the method and the call, where the
    // walk back over stack effects is trusted no more; the object is made.
    assert!(
        !rewritten(&source, "wCond") && !rewritten(&source, "wAnd") && !rewritten(&source, "wTail")
    );
}

#[test]
fn every_kind_of_callee_is_called_as_apply_calls_it() {
    let source = format!(
        "{TARGET}{}",
        r#"
        function wf(f) { return f.apply(this, arguments); }
        var bound = target.bind({ tag: 'b' }, 'pre');
        var proxy = new Proxy(target, { apply: function (t, self, list) {
            return 'trap:' + Array.isArray(list) + ':' + t.apply(self, list); } });
        class K {}
        function thrower() { throw new Error('thrown ' + arguments.length); }
        function attempt(f, x) { try { return String(wf.call(o, f, x)); } catch (e) { return e.name + ': ' + e.message; } }
        for (var k = 0; k < 40; k++) {
            out.push(attempt(target, k), attempt(bound, k), attempt(proxy, k), attempt(Math.max, k));
            out.push(attempt(K, k), attempt(thrower, k), attempt(5, k), attempt(undefined, k), attempt({}, k));
        }
        out.join(';')
        "#
    );
    let (_, vm, jit) = same_at_every_tier(&source);
    if !form() {
        return;
    }
    for counts in [vm, jit] {
        assert!(counts.hits > 0 && counts.misses > 0, "{counts:?}");
    }
}

#[test]
fn a_replaced_or_shadowed_apply_receives_the_exact_object() {
    let source = format!(
        "{TARGET}{}",
        r#"
        var marks = 0;
        var builtin = Function.prototype.apply;
        function w() { var r = target.apply(this, arguments); return r + '/' + target.apply(this, arguments) + '/' + arguments[0]; }
        // Each object it receives gets a mark at first sight: an activation's two sites pass one.
        function spy(self, list) {
            if (!list.mark) list.mark = ++marks;
            list[0] = 'w';
            return Object.prototype.toString.call(list) + ':' + list.length + ':' + (list.callee === w) + ':' + list.mark;
        }
        var own = { apply: function (self, list) { return 'own:' + list.length; }, tag: 'own' };
        function wOwn() { return own.apply(this, arguments); }
        var getter = {};
        Object.defineProperty(getter, 'apply', { get: function () { return spy; } });
        function wGet() { return getter.apply(this, arguments); }
        for (var k = 0; k < 40; k++) {
            if (k % 10 === 5) Function.prototype.apply = spy;
            out.push(w.call(o, k, 2), wOwn.call(o, k), wGet.call(o, k));
            if (k % 10 === 5) Function.prototype.apply = builtin;
        }
        out.join(';')
        "#
    );
    let (_, vm, jit) = same_at_every_tier(&source);
    if !form() {
        return;
    }
    for counts in [vm, jit] {
        assert!(counts.hits > 0 && counts.misses > 0, "{counts:?}");
    }
}

/// Mapped arguments (a sloppy function with simple formals): each index below the count that
/// its formal maps reads the formal's binding at the call; an index past the count, or one a
/// later duplicate formal names (CreateMappedArgumentsObject maps a name's last occurrence,
/// `interpreter::mapped_parameter_slots`), reads the argument.
#[test]
fn a_mapped_body_passes_its_formals_current_values() {
    let source = format!(
        "{TARGET}{}",
        r#"
        function written(a, b) { a = 'A'; return target.apply(this, arguments); }
        function captured(a) {
            var set = function () { a = 'C'; return this; };
            return target.apply(set.call(this), arguments);
        }
        function capturedCallee(a) {
            var pick = function () { a = 'P'; return target; };
            return pick().apply(this, arguments);
        }
        function returned(a) { var r = target.apply(this, arguments); return r + '|' + show(a); }
        function dup(a, a) { return target.apply(this, arguments); }
        function dupWritten(a, a) { a = 'D'; return target.apply(this, arguments); }
        function fewer(a, b, c) { c = 'late'; b = 'B'; return target.apply(this, arguments); }
        function shadowVar(a) { var a; return target.apply(this, arguments); }
        function shadowFn(a) { function a() {} return target.apply(this, arguments); }
        function mixed(a, b) { b = arguments[0]; return arguments.length + ':' + target.apply(this, arguments); }
        var builtin = Function.prototype.apply;
        function writer(self, list) { list[0] = 'W'; return 'spied'; }
        for (var k = 0; k < 40; k++) {
            out.push(written.call(o, 1, 2, 3), written.call(o), captured.call(o, 1, 2), capturedCallee.call(o, 1));
            out.push(dup.call(o, 1, 2), dup.call(o, 1), dupWritten.call(o, 1, 2), dupWritten.call(o, 1));
            out.push(fewer.call(o, 1), fewer.call(o, 1, 2, 3, 4), shadowVar.call(o, 1), shadowFn.call(o, 1));
            out.push(mixed.call(o, 1, 2));
            if (k % 8 === 3) {
                Function.prototype.apply = writer;
                out.push(returned.call(o, 'r'));
                Function.prototype.apply = builtin;
            } else {
                out.push(returned.call(o, 'r'));
            }
        }
        out.join(';')
        "#
    );
    let (_, vm, jit) = same_at_every_tier(&source);
    if !form() {
        return;
    }
    for counts in [vm, jit] {
        assert!(counts.hits > 0 && counts.misses > 0, "{counts:?}");
    }
    // The spec's answers themselves (the comparison above holds the tree-walker to them as well):
    // a duplicate's earlier index is never mapped, even when the later one is not passed.
    let exact = format!(
        "{TARGET}{}",
        r#"
        function dup(a, a) { return target.apply(this, arguments); }
        function dupWritten(a, a) { a = 'D'; return target.apply(this, arguments); }
        function fewer(a, b, c) { c = 'late'; b = 'B'; return target.apply(this, arguments); }
        var r;
        for (var k = 0; k < 10; k++) r = [dup.call(o, 1, 2), dupWritten.call(o, 1, 2), dupWritten.call(o, 1), fewer.call(o, 1), fewer.call(o, 1, 2, 3, 4)].join(';');
        r
        "#
    );
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (result, _) = run(tier, Some(0), &exact);
        assert_eq!(result, "o,1,2;o,1,D;o,1;o,1;o,1,B,late,4", "{tier:?}");
    }
    for name in ["written", "dup", "fewer", "returned", "mixed"] {
        assert!(rewritten(&source, name), "{name} keeps its object");
    }
}

/// An object a read materialized (a key `arguments[k]` cannot read from the frame) is the list
/// every later site passes, with whatever an accessor wrote into it; `arguments` as `apply`'s
/// receiver, not its list, is no apply site.
#[test]
fn a_materialized_object_is_the_list_from_then_on() {
    let source = format!(
        "{TARGET}{}",
        r#"
        Object.defineProperty(Object.prototype, '-1', { configurable: true,
            get: function () { if (Object.prototype.toString.call(this) === '[object Arguments]') this[0] = 'G'; return 'g'; } });
        function w() { var before = target.apply(this, arguments); var g = arguments[-1]; return before + '/' + g + '/' + target.apply(this, arguments); }
        function wm(a) { var g = arguments[-1]; return g + '/' + target.apply(this, arguments) + '/' + a; }
        function notLast() { return target.apply(arguments, [7]); }
        for (var k = 0; k < 40; k++) out.push(w.call(o, k, 2), wm.call(o, k), notLast(k, 3));
        out.join(';')
        "#
    );
    let (_, vm, jit) = same_at_every_tier(&source);
    if !form() {
        return;
    }
    for counts in [vm, jit] {
        assert!(counts.hits > 0 && counts.objects > 0, "{counts:?}");
    }
    assert!(rewritten(&source, "w") && rewritten(&source, "wm"));
    assert!(!rewritten(&source, "notLast"));
}

/// Every other way to `arguments` or a formal: the rule declines where the body or a nested
/// function can reach the object, and the results stay the tree-walker's either way.
#[test]
fn other_ways_to_arguments_decline_or_stay_exact() {
    let source = format!(
        "{TARGET}{}",
        r#"
        function arrow() { var g = () => arguments; var r = target.apply(this, arguments); return r + (g() === g()); }
        function nestedEval() { function h() { return eval('arguments.length'); } return target.apply(this, arguments) + h(9, 9); }
        function arrowEval() { var h = () => eval('arguments.length'); return target.apply(this, arguments) + h(); }
        function directEval() { eval(''); return target.apply(this, arguments); }
        function withed(scope) { with (scope) { return target.apply(this, arguments); } }
        function param(arguments) { return target.apply(this, arguments); }
        function varArgs() { var arguments; return target.apply(this, arguments); }
        function fnArgs() { function arguments() {} return target.apply(this, arguments); }
        function assigned() { arguments = ['x', 'y']; return target.apply(this, arguments); }
        function stored() { var a = arguments; return target.apply(this, arguments) + (a === arguments); }
        function defaults(a = 1) { return target.apply(this, arguments); }
        function rest(...r) { return target.apply(this, arguments) + r.length; }
        function init() { this.n = arguments.length; this.first = arguments[0]; }
        class Base { constructor() { this.base = true; } m() { return 'm' + arguments.length; } }
        class Derived extends Base { constructor() { super(); init.apply(this, arguments); } }
        class Plain { constructor() { init.apply(this, arguments); } }
        class Sup extends Base { m() { return super.m.apply(this, arguments); } }
        function ctor() { init.apply(this, arguments); }
        for (var k = 0; k < 40; k++) {
            out.push(arrow.call(o, k), nestedEval.call(o, k), arrowEval.call(o, k), directEval.call(o, k));
            out.push(withed.call(o, { arguments: ['in-with'] }, k), withed.call(o, {}, k));
            out.push(param.call(o, [1, 2]), varArgs.call(o, k), fnArgs.call(o, k), assigned.call(o, k));
            out.push(stored.call(o, k), defaults.call(o, k), defaults.call(o), rest.call(o, k, 1));
            var d = new Derived(k, 2), p = new Plain(k), c = new ctor(k, 3, 4);
            out.push([d.base, d.n, d.first, p.n, p.first, c.n, c.first].join(','), new Sup().m(1, 2));
        }
        out.join(';')
        "#
    );
    same_at_every_tier(&source);
    if !form() {
        return;
    }
    for name in ["arrow", "stored", "assigned", "param"] {
        assert!(!rewritten(&source, name), "{name} lost its object");
    }
    assert!(rewritten(&source, "ctor"), "ctor keeps its object");
}

/// Box2D's two constructor forms, and RayTrace's (part A2's recognized forwarder), construct the
/// same instances, and with the form no arguments object is made: a recognized forwarder's
/// first constructions included, which run its body before its site's entry is filled.
#[test]
fn the_constructor_forms_make_no_arguments_object() {
    let source = r#"
        function Init(x, y) { this.x = x; this.y = y; }
        function F1() { F1.init.apply(this, arguments); }
        F1.init = Init;
        function F2() { F2.init.apply(this, arguments); this.constructor === F2 && this.more.apply(this, arguments); }
        F2.init = Init;
        F2.prototype.more = function (x, y) { this.z = x + y; };
        var Class = { create: function () { return function () { this.initialize.apply(this, arguments); }; } };
        var F3 = Class.create();
        F3.prototype.initialize = function (x, y) { this.x = x; this.y = y; };
        var out = [];
        for (var k = 0; k < 40; k++) {
            var a = new F1(k, 1), b = new F2(k, 2), c = new F3(k, 3);
            out.push([a.x, a.y, b.x, b.y, b.z, c.x, c.y].join(','));
        }
        out.join(';')
    "#;
    let (_, vm, jit) = same_at_every_tier(source);
    if !form() {
        return;
    }
    for (tier, counts) in [("vm", vm), ("jit", jit)] {
        assert_eq!(counts.objects, 0, "{tier}: {counts:?}");
    }
    assert!(rewritten(source, "F1") && rewritten(source, "F2"));
    // The recognizer reads the rewritten body.
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    let chunk = chunk_of(&mut engine, "F3");
    assert!(
        matches!(chunk.jit_ops().get(4), Some(Op::CallApplyFrameArgs(..))),
        "{:?}",
        chunk.jit_ops()
    );
    assert!(
        chunk.jit_arguments_apply_forwarder().is_some() || crate::bytecode::forwarder_disabled(),
        "part A2 recognizes the rewritten forwarder"
    );
}

/// What a compiled frame's legacy `w.arguments` answers is unchanged: null, as for every compiled
/// frame today (the tree-walker's answers the frame's object), and `f.caller` is the forwarding
/// function, or null for a strict one.
#[test]
fn legacy_reflection_answers_as_before() {
    let source = format!(
        "{TARGET}{}",
        r#"
        function peek() { var a = w.arguments; return (a === null ? 'null' : typeof a) + ':' + (peek.caller === w) + ':' + String(strictPeek.caller); }
        function strictPeek() { return peek.caller === null; }
        function w() { return peek.apply(this, arguments); }
        function ws() { 'use strict'; return strictPeek.apply(this, arguments); }
        for (var k = 0; k < 40; k++) out.push(w(1, 2), ws(1));
        out[out.length - 2] + ';' + out[out.length - 1]
        "#
    );
    let (interp, _) = run(Tier::Interp, None, &source);
    assert_eq!(interp, "object:true:null;true");
    for tier in [Tier::Bytecode, Tier::Jit] {
        let (result, counts) = run(tier, Some(0), &source);
        assert_eq!(result, "null:true:null;true", "{tier:?}");
        assert!(counts.hits > 0 || !form(), "{tier:?}");
    }
}

/// The argument words a forwarding frame reads late, after code that collects: objects only its
/// caller's words name (the arguments past the formals, a strict formal overwritten), for a
/// caller in machine code, in the VM, and in Rust (a native's callback, a Rust-held list).
#[test]
fn late_argument_words_survive_collections() {
    // Under GC stress every allocation collects: a few allocations between entry and the apply
    // are enough, where the ordinary run allocates enough to start collections of its own.
    let churn = if crate::gc_test::stress_active() {
        8
    } else {
        3000
    };
    let source = format!(
        "{}{churn}{}",
        r#"
        function total() { var s = 0; for (var k = 0; k < arguments.length; k++) s += arguments[k].v; return s; }
        function churn() { var a = []; for (var k = 0; k < "#,
        r#"; k++) a.push({ v: k, w: [k] }); return a.length; }
        function w0() { churn(); return total.apply(this, arguments); }
        function ws(a) { 'use strict'; a = null; churn(); return total.apply(this, arguments); }
        function wm(a) { churn(); return total.apply(this, arguments); }
        function list(n) { var l = []; for (var k = 0; k < n; k++) l.push({ v: k + 1 }); return l; }
        var out = [];
        for (var k = 0; k < 12; k++) {
            out.push(w0({ v: 1 }, { v: 2 }, { v: 3 }), ws({ v: 4 }, { v: 5 }), wm({ v: 6 }, { v: 7 }, { v: 8 }));
            out.push(Reflect.apply(w0, null, list(4)), Reflect.apply(ws, null, list(3)));
            out.push([{ v: 9 }].map(function (x) { return w0(x, { v: 10 }); })[0]);
            out.push(w0.call(null, { v: 11 }, { v: 12 }));
        }
        out.join(',')
    "#
    );
    let (_, vm, jit) = same_at_every_tier(&source);
    assert!(vm.hits > 0 && jit.hits > 0 || !form());
}

/// Deep recursion through the form ends in the same RangeError, and an error's stack names the
/// forwarding function's frame.
#[test]
fn recursion_and_stacks_through_the_form() {
    // Under GC stress a collection at every allocation scans every frame of the recursion, which
    // makes exhausting the stack quadratic: the stack text alone is checked then (as
    // constructor_transfer_tests' depth test does).
    let exhaust = !crate::gc_test::stress_active();
    let source = format!(
        "var exhaust = {exhaust};{}",
        r#"
        function deep() { return deep.apply(this, arguments); }
        var message = 'RangeError: Maximum call stack size exceeded';
        if (exhaust) try { deep(1); } catch (e) { message = e.name + ': ' + e.message; }
        function inner() { return new Error('x').stack; }
        function forward() { return inner.apply(this, arguments); }
        var stack;
        for (var k = 0; k < 30; k++) stack = forward(1);
        message + '|' + /forward/.test(stack) + '|' + /inner/.test(stack)
    "#
    );
    let (result, _, _) = same_at_every_tier(&source);
    assert!(result.ends_with("|true|true"), "{result}");
}

/// The optimizing tier compiles a function that holds the form, the operation running as its
/// template's canonical-frame effect (the builder lowers no such operation, and liveness gives
/// it every slot, `opt::liveness::slot_use`'s default): the VM's results eagerly and with forced
/// exits at both inline modes, with the object's path taken inside tier code (`apply` replaced
/// after the compile; the slot it writes is the one the next site and `arguments[1]` read), and
/// the callee's throw leaving through the operation.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn the_tier_compiles_the_form_with_the_vms_results() {
    use crate::opt::TestSwitches;
    const PROGRAM: &str = r#"
        var builtin = Function.prototype.apply, marks = 0;
        function target(a, b) { return (this && this.tag) + ':' + a + ':' + b + ':' + arguments.length; }
        function spy(self, list) { if (!list.mark) list.mark = ++marks; list[1] = 'S'; return 'spy:' + list.length + ':' + list.mark; }
        function thrower() { throw new Error('thrown ' + arguments.length); }
        var fwd = target;
        function w() {
            var r1;
            try { r1 = fwd.apply(this, arguments); } catch (e) { r1 = 'caught ' + e.message; }
            var r2 = fwd.apply(this, arguments);
            return r1 + '/' + r2 + '/' + arguments[1];
        }
        function wm(x, y) { y = 'Y'; var r = fwd.apply(this, arguments); return r + '/' + arguments[1] + '/' + x; }
        var o = { tag: 'o' }, out = [];
        for (var k = 0; k < 400; k++) {
            if (k === 300) Function.prototype.apply = spy;
            if (k === 320) Function.prototype.apply = builtin;
            fwd = k % 60 === 7 ? thrower : target;
            try { out.push(w.call(o, k, k + 1, k + 2), wm.call(o, k, 1)); } catch (e) { out.push(e.message); }
        }
        out.join(';')
    "#;
    fn tier_chunk(engine: &mut Engine, name: &str) -> Option<std::rc::Rc<crate::bytecode::Chunk>> {
        let env = engine.interp.global_env;
        let value = engine.interp.get_var(name, &env).ok()?;
        let object = value.as_obj()?;
        let ObjectKind::User(user) = &object.borrow().kind else {
            return None;
        };
        let runtime = user.func.runtime();
        runtime.code2.get().and_then(Option::clone).or_else(|| {
            runtime
                .code
                .get()
                .and_then(Option::clone)
                .filter(|first| crate::opt::in_place(first))
        })
    }
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let (expected, _) = run(Tier::Bytecode, Some(0), PROGRAM);
    assert!(
        expected.contains("spy:") && expected.contains("thrown"),
        "{expected}"
    );
    for graph in [false, true] {
        // Forced exits at several phases, so that some exit resumes at the operation itself,
        // after its LoadLocal pushed the slot's word (the slot then holds the VM's Array).
        for exit_every in [0, 2, 3, 5, 7, 11, 13] {
            TestSwitches {
                on: true,
                eager: true,
                exit_every,
                inline_graph: Some(graph),
                ..Default::default()
            }
            .set();
            crate::opt::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
            crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
            crate::opt::TEST_INLINES.with(|inlines| inlines.borrow_mut().clear());
            reset();
            let mut engine = Engine::new();
            engine.set_tier(Tier::Jit);
            engine.set_tier_threshold(0);
            let result = eval(&mut engine, PROGRAM);
            let counts = counts();
            assert_eq!(result, expected, "graph {graph}, exits every {exit_every}");
            if !form() {
                continue;
            }
            assert!(counts.hits > 0 && counts.misses > 0, "{counts:?}");
            let compiled = crate::opt::TEST_COMPILED.with(|compiled| compiled.borrow().clone());
            let exits = crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
            let inlined: Vec<u32> = crate::opt::TEST_INLINES
                .with(|inlines| inlines.borrow().iter().map(|note| note.pc).collect());
            let mut at_the_operation = false;
            for name in ["w", "wm"] {
                let chunk = tier_chunk(&mut engine, name)
                    .unwrap_or_else(|| panic!("the tier did not compile {name}"));
                assert!(
                    compiled.contains(&(std::rc::Rc::as_ptr(&chunk) as usize)),
                    "{name}"
                );
                let sites: Vec<u32> = chunk
                    .jit_ops()
                    .iter()
                    .enumerate()
                    .filter(|(_, op)| matches!(op, Op::CallApplyFrameArgs(..)))
                    .map(|(pc, _)| pc as u32)
                    .collect();
                assert!(!sites.is_empty(), "{name}");
                // No plan at the site, at either inline mode: no inlined frame (graph), no splice
                // (the splice planner skips the site, so the chunk holds no guard).
                assert!(
                    !inlined.iter().any(|pc| sites.contains(pc)),
                    "{name}: {inlined:?}"
                );
                assert!(
                    !chunk
                        .jit_ops()
                        .iter()
                        .any(|op| matches!(op, Op::InlineGuard(..))),
                    "{name}"
                );
                at_the_operation |= exits.iter().any(|&(pc, reason)| {
                    sites.contains(&pc) && reason == crate::opt::ir::ExitReason::Throw
                });
            }
            // The callee's throw caught in w leaves tier code through the operation's exit.
            assert!(at_the_operation || exit_every != 0, "{exits:?}");
        }
    }
    TestSwitches::default().set();
}

/// No splice takes an apply form's call: its site's way is the forwarded callee, not the call's
/// callee operand (`apply`), so a recompile that spliced it would guard `apply` against `f` and
/// lose the form (`bytecode::spread_call_sites`). A hot forwarding body's recompiles keep the
/// operation and splice nothing at it.
#[test]
fn a_recompiled_body_keeps_the_form() {
    let source = format!(
        "{TARGET}{}",
        r#"
        function leaf(a, b) { return a + b; }
        function w() { return leaf.apply(this, arguments); }
        function loop(n) { var s = 0; for (var k = 0; k < n; k++) s += w.call(o, k, 2); return s; }
        for (var r = 0; r < 40; r++) loop(500);
        loop(2)
        "#
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    assert_eq!(eval(&mut engine, &source), "5");
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var("w", &env).ok().expect("w");
    let ValueView::Obj(function) = value.view() else {
        panic!("w")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("w")
    };
    let runtime = user.func.runtime();
    let chunks: Vec<_> = [runtime.code.get(), runtime.code2.get()]
        .into_iter()
        .flatten()
        .flatten()
        .cloned()
        .collect();
    assert!(!chunks.is_empty());
    for chunk in chunks {
        let ops = chunk.jit_ops();
        assert!(
            !ops.iter().any(|op| matches!(op, Op::InlineGuard(..))),
            "{ops:?}"
        );
        if form() {
            assert!(
                ops.iter()
                    .any(|op| matches!(op, Op::CallApplyFrameArgs(..))),
                "{ops:?}"
            );
        }
    }
}
