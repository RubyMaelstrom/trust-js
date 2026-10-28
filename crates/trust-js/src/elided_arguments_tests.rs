//! Decision 20e's elided `arguments` objects and rest Arrays (W3 item 3): a body that reads its
//! `arguments` or rest parameter only by `length` and element makes neither in a fresh native
//! frame, reading its actual arguments instead. ECMA-262 e28783d5: CreateUnmappedArgumentsObject
//! and CreateMappedArgumentsObject (spec.html:15059-15148), the arguments exotic object's own
//! properties, OrdinaryGet through %Object.prototype% and %Array.prototype%. Expected results are
//! Node's (v24.22.0).
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 8] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            engine
                .interp
                .def_method(&engine.interp.global, "collect", 0, |interp, _, _| {
                    interp.gc_collect();
                    Ok(crate::value::Value::Undefined)
                });
            match engine
                .eval(source, false)
                .expect("elided arguments fixture parses")
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
fn elided_arguments_read_the_actual_arguments() {
    check(
        r#"
        function args() { return arguments.length + ':' + arguments[0] + ':' + arguments[2]; }
        function run() {
            var out = [];
            for (var k = 0; k < 40; k++) out.push(args(k, 'b', k * 2));
            return out[0] + '|' + out[39];
        }
        run();
        "#,
        "3:0:0|3:39:78",
    );
}

#[test]
fn elided_arguments_past_the_count_read_the_prototype() {
    check(
        r#"
        function args() { return String(arguments[3]) + ',' + arguments.length; }
        function run(n) { var s = ''; for (var k = 0; k < n; k++) s = args(1, 2); return s; }
        var a = run(40);
        Object.prototype[3] = 'proto';
        var b = run(40);
        delete Object.prototype[3];
        var c = run(40);
        a + '|' + b + '|' + c;
        "#,
        "undefined,2|proto,2|undefined,2",
    );
}

#[test]
fn elided_arguments_materialize_the_exact_object_for_other_keys() {
    check(
        r#"
        function args() {
            return [arguments['callee'] === args, arguments[Symbol.iterator] === Array.prototype.values,
                arguments[0], arguments.length].join(',');
        }
        function strictArgs(a) {
            'use strict';
            var threw = false;
            try { arguments['callee']; } catch (e) { threw = e instanceof TypeError; }
            return [threw, arguments[0], arguments[1], arguments.length].join(',');
        }
        function run() {
            var s, t;
            for (var k = 0; k < 40; k++) { s = args(k, 7); t = strictArgs(k, 'x'); }
            return s + '|' + t;
        }
        run();
        "#,
        "true,true,39,2|true,39,x,2",
    );
}

#[test]
fn a_prototype_getter_sees_and_changes_the_materialized_arguments() {
    check(
        r#"
        var seen;
        Object.defineProperty(Object.prototype, '5', {
            get() { seen = this; this[0] = 'changed'; return 'g'; }, configurable: true });
        function args() {
            var first = arguments[0];
            var g = arguments[5];
            return [first, g, arguments[0],
                seen === undefined ? 'none' : Object.prototype.toString.call(seen),
                arguments.length].join(',');
        }
        function run() { var s; for (var k = 0; k < 40; k++) { seen = undefined; s = args('a', 'b'); } return s; }
        var r = run();
        delete Object.prototype[5];
        r;
        "#,
        "a,g,changed,[object Arguments],2",
    );
}

#[test]
fn elided_strict_arguments_keep_an_assigned_formals_argument() {
    check(
        r#"
        function f(a) { 'use strict'; a = 9; return arguments[0] + ':' + a + ':' + arguments.length + ':' + arguments[1]; }
        function run() { var s; for (var k = 0; k < 40; k++) s = f(k, 1); return s; }
        run();
        "#,
        "39:9:2:1",
    );
}

#[test]
fn elided_rest_parameters_read_the_arguments_past_the_formals() {
    check(
        r#"
        function r(a, ...xs) {
            return [xs.length, xs[0], xs[1], String(xs[5]), xs['length'], typeof xs['map']].join(',');
        }
        function none(a, ...xs) { return xs.length + ':' + String(xs[0]); }
        function run() {
            var s;
            for (var k = 0; k < 40; k++) s = r(0, k, 'q') + '/' + none(1) + '/' + none();
            return s;
        }
        var a = run();
        Array.prototype[5] = 'ap';
        var b = run();
        delete Array.prototype[5];
        a + '|' + b + '|' + run();
        "#,
        "2,39,q,undefined,2,function/0:undefined/0:undefined|2,39,q,ap,2,function/0:undefined/0:undefined|2,39,q,undefined,2,function/0:undefined/0:undefined",
    );
}

#[test]
fn elided_reads_keep_their_arguments_across_collections() {
    check(
        r#"
        function args() {
            var first = arguments[0];
            collect();
            return first.n + arguments[1].n + arguments.length;
        }
        function rest(a, ...xs) { collect(); return a.n + xs[0].n + xs[1].n + xs.length; }
        function run() {
            var s = 0;
            for (var k = 0; k < 40; k++) {
                s += args({ n: k }, { n: 1 });
                s += rest({ n: 1 }, { n: 2 }, { n: k });
            }
            return s;
        }
        run();
        "#,
        // args: k + 1 + 2 over k = 0..39 = 780 + 120; rest: 1 + 2 + k + 2 over k = 0..39 = 780 + 200.
        "1880",
    );
}

/// An entry from Rust reads the caller's argument list, which the call roots for its whole
/// duration (`Interp::call`, I1): the source Array no longer holds the values at the collection.
#[test]
fn elided_reads_keep_an_apply_list_its_source_no_longer_holds() {
    check(
        r#"
        var source;
        function args() {
            source.length = 0;
            collect();
            return arguments[0].n + arguments[2].n + arguments.length;
        }
        function rest(a, ...xs) {
            source.length = 0;
            collect();
            return xs[0].n + xs[1].n + xs.length;
        }
        function run() {
            var s = 0;
            for (var k = 0; k < 40; k++) {
                source = [{ n: k }, { n: 1 }, { n: 2 }];
                s += args.apply(null, source);
                source = [{ n: 0 }, { n: k }, { n: 3 }];
                s += Reflect.apply(rest, null, source);
            }
            return s;
        }
        run();
        "#,
        // Both: k + 5 over k = 0..39 = 780 + 200.
        "1960",
    );
}
