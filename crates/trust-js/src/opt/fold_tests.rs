//! The tier reads an object's fields and header bytes from its cell with the Object's offset in
//! the cell folded into each load's and store's immediate (a64.rs `field_address`, the checks'
//! loads, `object_or`): no add of the offset first. Every form whose address folds gives the VM's
//! result: fields in the cell and in the record past it, read and written, prototype hops, the
//! String prototype's, property creation, an Array's length, and the object tests of ToBoolean,
//! `== null` and typeof.

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
fn fields_in_the_cell_and_in_the_record_read_and_write_as_the_vm_does() {
    // `small` keeps its fields in its cell; `big` has more than the largest cell holds (fifteen
    // slots, `class_slots`), its fields from p15 on in the record past it; each is read, written
    // and read again, monomorphic and through a prototype hop; a property is created on fresh
    // objects, some of them not extensible (the creation check's flag byte).
    agrees_at_every_switch(
        "
        function Small(k) { this.a = k; this.b = k * 2; }
        function Big(k) {
          this.p0 = k; this.p1 = k + 1; this.p2 = k + 2; this.p3 = k + 3; this.p4 = k + 4;
          this.p5 = k + 5; this.p6 = k + 6; this.p7 = k + 7; this.p8 = k + 8; this.p9 = k + 9;
          this.p10 = k + 10; this.p11 = k + 11; this.p12 = k + 12; this.p13 = k + 13;
          this.p14 = k + 14; this.p15 = k + 15; this.p16 = k + 16; this.p17 = k + 17;
          this.p18 = k + 18; this.p19 = k + 19; this.p20 = k + 20; this.p21 = k + 21;
        }
        Big.prototype.base = 1000;
        function readSmall(o) { return o.a + o.b; }
        function readBig(o) { return o.p0 + o.p6 + o.p13 + o.p14 + o.p15 + o.p21 + o.base; }
        function writeSmall(o, v) { o.b = v; return o.b; }
        function writeBig(o, v) {
          o.p13 = v; o.p1 = v + 1; o.p15 = v + 2; o.p21 = v + 3;
          return o.p13 + o.p1 + o.p15 + o.p21;
        }
        function addY(o, v) { o.y = v; return o.y === undefined ? -1 : o.y; }
        var t = 0;
        for (var r = 0; r < 3000; r++) {
          var s = new Small(r), b = new Big(r);
          t = (t * 31 + readSmall(s) + readBig(b)) % 1000003;
          t = (t * 31 + writeSmall(s, r & 15) + writeBig(b, r & 7)) % 1000003;
          t = (t * 31 + readSmall(s) + readBig(b)) % 1000003;
          // A creation on a fresh object, and on one of its shape that is not extensible.
          var n = new Small(r);
          if (r % 5 == 4) Object.preventExtensions(n);
          t = (t * 31 + addY(n, r & 31)) % 1000003;
        }
        t",
    );
}

#[test]
fn prototype_hops_creations_lengths_and_object_tests_answer_as_the_vm_does() {
    // A method two prototypes up (CheckProto's hops), a String method (the String prototype's
    // check), a property added to an object (the creation check), an Array's length, and an
    // object in ToBoolean, `== null`, typeof (against "object", "function" and "undefined") and
    // an equality of objects and nullish values (their object paths read the flag and kind bytes).
    agrees_at_every_switch(
        "
        function A() {} A.prototype.m = function () { return 3; };
        function B() {} B.prototype = Object.create(A.prototype);
        function C() {} C.prototype = Object.create(B.prototype);
        function hop(o) { return o.m(); }
        function str(s) { return s.charCodeAt(0) + s.length; }
        function make(k) { var o = {}; o.x = k; o.y = k + 1; return o.x + o.y; }
        function len(a) { return a.length; }
        function truthy(o) { return o ? 1 : 2; }
        function nullish(o) { return o == null ? 3 : 4; }
        function kind(o) { return typeof o === 'object' ? 5 : typeof o === 'function' ? 6 : 7; }
        function undef(o) { return typeof o === 'undefined' ? 8 : 9; }
        function equal(p, q) { return p == q ? 10 : 11; }
        var c = new C(), arr = [1, 2, 3], f = function () {};
        var vals = [c, arr, f, null, undefined, 0, 'x'];
        var both = [c, null, arr, undefined, c];
        var t = 0;
        for (var r = 0; r < 3000; r++) {
          var v = vals[r % 7];
          t = (t * 31 + hop(c) + str('ab' + (r & 3)) + make(r) + len(arr)) % 1000003;
          t = (t * 31 + truthy(v) + nullish(v) + kind(v) + undef(v)) % 1000003;
          t = (t * 31 + equal(both[r % 5], both[(r >> 2) % 5])) % 1000003;
        }
        t",
    );
}
