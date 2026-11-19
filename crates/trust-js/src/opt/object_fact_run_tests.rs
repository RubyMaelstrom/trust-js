//! P3 stage 3 (c-tag): the tier's shape checks of words proven objects (`CheckShape::object`, no
//! tag test) answer as the VM does, at the splice and at graph, with forced exits too: reads after
//! a call, after a join, in loops, polymorphic reads, a method's reads of `this`; and a word that
//! no check proved, a phi of an object and a number, still exits at its check when it is the
//! number.

use super::TestSwitches;
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn vm(program: &str) -> String {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, program)
}

fn agrees_at_every_switch(source: &str) {
    let expected = vm(source);
    for (exit_every, graph) in [(0, false), (0, true), (7, false), (7, true)] {
        TestSwitches {
            on: true,
            eager: exit_every != 0,
            exit_every,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        assert_eq!(
            eval(&mut engine, source),
            expected,
            "exits every {exit_every}, graph {graph}"
        );
    }
}

#[test]
fn checks_of_proven_objects_answer_as_the_vm_does() {
    agrees_at_every_switch(
        "
        function P(k) { this.x = k; this.y = k + 1; this.z = k + 2; }
        function Q(k) { this.w = 0; this.x = k; this.y = 2 * k; this.z = 3; }
        P.prototype.sum = function (g) { var a = this.x; g(); return a + this.y + this.z; };
        var calls = 0;
        function g() { calls++; }
        function around(o, g) { var a = o.x; g(); return a + o.y; }
        function el(o, g) { var s = o[0]; g(); return s + o.x; }
        function upd(o, g) { o[0] += 1; g(); return o.x; }
        function join(o, c, g) { var a; if (c) a = o.x; else a = o.y; g(); return a + o.z; }
        function oneArm(o, c, g) { var a = 0; if (c) a = o.x; g(); return a + o.z; }
        function before(o, n, g) {
          var s = o.x; for (var i = 0; i < n; i++) { g(); s += o.y; } return s;
        }
        function twoInside(o, n, g) {
          var s = 0; for (var i = 0; i < n; i++) { s += o.x; g(); s += o.y; } return s;
        }
        function phi(o, k, c, g) { var a = o.x + k.x; var w = c ? o : k; g(); return a + w.y; }
        function poly(o, g) { var a = o.x; g(); return a + o.y + o.z; }
        var t = 0, e = {x: 1}; e[0] = 5;
        for (var r = 0; r < 3000; r++) {
          var p = new P(r & 63), q = new Q(r & 31), o = r % 3 ? p : q;
          t = (t * 31 + around(p, g) + el(e, g) + upd(e, g) + join(p, r & 1, g)) % 1000003;
          t = (t * 31 + oneArm(p, r & 2, g) + before(p, r & 3, g) + twoInside(p, r & 3, g)) % 1000003;
          t = (t * 31 + phi(p, o, r & 1, g) + poly(o, g) + p.sum(g)) % 1000003;
        }
        t + ':' + calls",
    );
}

#[test]
fn a_non_object_after_a_phi_still_exits_at_its_check() {
    // `w` is a phi of `o`, which a check proved, and `k`, which nothing proved: its check tests
    // the tag, and every 500th call `w` is `k` and `k` a Number, which must exit there (a test
    // omitted there would take the Number's payload, here null, as a cell).
    agrees_at_every_switch(
        "
        function P(k) { this.x = k; this.y = k + 1; }
        function g() {}
        function ph(o, k, c) { var a = o.x; var w = c ? o : k; g(); return a + w.y; }
        var t = 0, p = new P(3), q = new P(5);
        for (var r = 0; r < 3000; r++) {
          var odd = r % 500 == 499;
          var v = ph(p, odd ? 1.5 : q, odd ? 0 : r & 1);
          t = (t * 31 + (v === v ? v : 7)) % 1000003;
        }
        t",
    );
}

#[test]
fn a_guard_of_a_proven_receiver_answers_as_the_vm_does() {
    // Sloppy methods that read `this`, spliced (or inlined at graph) behind guards whose
    // receivers the method loads' shape checks proved; and a String receiver, whose method loads
    // keep its word, that passes the guard of a callee that reads no `this`, which tests no
    // receiver and so proves nothing, then reaches the guard of one that does, which must test
    // it: `this` is the String's wrapper there, typeof "object". (A strict callee's guard tests
    // none either, but no strict callee is spliced into a sloppy function: the callee set keeps
    // the compiled function's strictness.)
    agrees_at_every_switch(
        "
        function P(k) { this.x = k; this.y = k + 1; }
        P.prototype.sum = function () { return this.x + this.y; };
        P.prototype.twice = function () { return this.sum() * 2; };
        String.prototype.n = function () { return 'n'; };
        String.prototype.l = function () { return typeof this; };
        function m(o) { return o.sum() + o.twice(); }
        function both(v) { return v.n() + v.l(); }
        var t = 0, last = '';
        for (var r = 0; r < 3000; r++) {
          t = (t * 31 + m(new P(r & 63))) % 1000003;
          last = both('s' + (r & 3));
        }
        t + ':' + last",
    );
}
