//! W3 item 3's step 6 (Decision 20f): a sloppy body with simple formals that reads its
//! `arguments` beyond its length keeps its formals in its activation, and its arguments object
//! keeps the ParameterMap (the environment and a slot per mapped index), so `arguments[i]` and
//! the formal share one word; an elided object's element reads take the formal's binding, and
//! a materialized one aliases the activation. ECMA-262 e28783d5: CreateMappedArgumentsObject
//! (spec.html:15083-15148), the arguments exotic object's [[DefineOwnProperty]], [[Get]],
//! [[Set]] and [[Delete]] (14960-15058). Expected results are Node's (v24.22.0).
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 8] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            match engine
                .eval(source, false)
                .expect("mapped arguments fixture parses")
            {
                Completion::Value(actual) => {
                    assert_eq!(actual, expected, "{tier:?}/{threshold}: {source}");
                }
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{threshold}: {name}: {message}\n{source}");
                }
            }
        }
    }
}

#[test]
fn mapped_indices_alias_their_formals_in_every_body_shape() {
    check(
        r#"
        function f(a, b) { a = 10; return arguments[0] + ':' + arguments[1] + ':' + arguments.length; }
        function g(a) { arguments[0] = 5; return a; }
        function h(a) { a = 7; var o = arguments; return o[0]; }
        function d(a, a) { a = 3; return arguments[0] + ':' + arguments[1]; }
        function p(a) { return arguments[1]; }
        function q(a, b) { b = 4; return arguments[1] + ':' + arguments.length; }
        function m(a) { a = 2; return arguments['0'] + ':' + arguments[0]; }
        function u(a) {
            Object.defineProperty(arguments, '0', { value: 3, writable: false });
            a = 4;
            return arguments[0] + ':' + a;
        }
        function dl(a) { delete arguments[0]; a = 5; return arguments[0] + ':' + a; }
        function fz(a) { Object.freeze(arguments); a = 6; return arguments[0] + ':' + a; }
        function ar(a) { var g = () => arguments[0]; a = 8; return g(); }
        function ap(a, b) { a = 'x'; return Array.prototype.join.call(arguments, '-'); }
        function k(a, b) {
            var s = 0;
            for (var i = 0; i < arguments.length; i++) { s += arguments[i]; a = 100; }
            return s;
        }
        function run() {
            var out = [];
            for (var n = 0; n < 40; n++) {
                out.push([f(1, 2), f(1), f(1, 2, 3), g(1), h(1), d(1, 2), p(1, 9), q(1), m(1),
                    u(1), dl(1), fz(1), ar(1), ap(1, 2), k(1, 2, 3)].join(' '));
            }
            return out[0] + '|' + (out[0] === out[39]);
        }
        run();
        "#,
        "10:2:2 10:undefined:1 10:2:3 5 7 1:3 9 undefined:1 2:2 3:4 undefined:5 1:6 8 x-2 6|true",
    );
}

#[test]
fn a_materialized_object_aliases_the_activation() {
    // A prototype getter receives the object an elided read materializes, and its later write
    // reaches the formal.
    check(
        r#"
        var keep;
        Object.defineProperty(Object.prototype, 'leak', {
            get() { keep = this; return 1; }, configurable: true });
        function e(a) { var r = arguments.leak; keep[0] = 9; var v = a; return r + ':' + v; }
        var first = e(1);
        Object.defineProperty(Object.prototype, 5, {
            get() { keep = this; return 'p'; }, configurable: true });
        function e2(a) { var r = arguments[5]; keep[0] = 9; return r + ':' + a; }
        var last = e2(1), same = true;
        for (var n = 0; n < 30; n++) same = same && e2(1) === last;
        first + ' ' + last + ' ' + same;
        "#,
        "1:9 p:9 true",
    );
}

/// A body that reads its mapped `arguments` by element compiles with its formals in its
/// activation and no prepared entry.
#[test]
fn element_reads_leave_the_prepared_entry() {
    use crate::value::{Callable, ValueView};
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let Completion::Value(result) = engine
        .eval(
            "function argsMapped(a, b) { return arguments.length + arguments[0]; }\
             argsMapped(1, 2, 3) + argsMapped(4);",
            false,
        )
        .expect("fixture parses")
    else {
        panic!("fixture throws")
    };
    assert_eq!(result, "9");
    let env = engine.interp.global_env;
    let Ok(function) = engine.interp.get_var("argsMapped", &env) else {
        panic!("argsMapped is defined")
    };
    let ValueView::Obj(function) = function.view() else {
        panic!("argsMapped is a function")
    };
    let function = function.borrow();
    let Callable::User(user) = &function.call else {
        panic!("argsMapped is a user function")
    };
    let chunk = user
        .func
        .runtime()
        .execution_code()
        .and_then(Option::as_ref)
        .expect("compiled argsMapped")
        .clone();
    assert!(!chunk.prepared_entry);
    assert!(chunk.lazy_arguments);
    assert_eq!(chunk.mapped_formals.len(), 2);
}
