//! W2d step 3e: a constructor's instances take one capacity class, decided at its first
//! construction from the `this` stores in its code and its prototype's defaults
//! ([`crate::value::constructed_class`], JSC's rule).
use super::*;

/// Run `source`, then read its global `probe`, an object.
fn object(engine: &mut crate::Engine, source: &str) -> Gc {
    let completion = engine.eval(source, false).expect("fixture parses");
    if let crate::Completion::Throw { name, message } = completion {
        panic!("{source}: {name}: {message}");
    }
    let env = engine.interp.global_env;
    let value = engine
        .interp
        .get_var("probe", &env)
        .ok()
        .expect("bound probe");
    let ValueView::Obj(object) = value.view() else {
        panic!("{source}: the probe is an object")
    };
    object
}

fn class(engine: &mut crate::Engine, source: &str) -> u32 {
    object(engine, source).borrow().props.class()
}

#[test]
fn constructors_take_the_class_their_stores_and_prototype_defaults_count() {
    let mut engine = crate::Engine::new();
    let cases = [
        // Two stores: the three-slot class.
        (
            "function Pair(a, b) { this.car = a; this.cdr = b; } var probe = new Pair(1, 2);",
            0,
        ),
        // Two stores and two prototype defaults (Splay's nodes): the seven-slot class.
        (
            "function Node(k, v) { this.key = k; this.value = v; }
             Node.prototype.left = null; Node.prototype.right = null; var probe = new Node(1, 2);",
            1,
        ),
        // No store and no default: the default class.
        (
            "function Init() { this.init(); }
             Init.prototype.init = function () { this.a = 1; }; var probe = new Init();",
            1,
        ),
        // A forwarder's prototype defaults (RayTrace's Class.create): the three-slot class.
        (
            "function Fwd() { this.initialize.apply(this, arguments); }
             Fwd.prototype = { x: 0, y: 0, z: 0,
                 initialize: function (x, y, z) { this.x = x; this.y = y; this.z = z; } };
             var probe = new Fwd(1, 2, 3);",
            0,
        ),
        // Stores in both arms of a branch count, each name once.
        (
            "function Branchy(c) { if (c) { this.a = 1; } else { this.b = 2; } this.c = 3;
                 this.a = 4; } var probe = new Branchy(true);",
            0,
        ),
        // A comma sequence's and a chained assignment's stores.
        (
            "function Seq(a) { this.a = a, this.b = 0, this.c = this.d = 0; } var probe = new Seq(1);",
            1,
        ),
        // Eight stores: the fifteen-slot class.
        (
            "function Big() { this.a = 1; this.b = 2; this.c = 3; this.d = 4; this.e = 5;
                 this.f = 6; this.g = 7; this.h = 8; } var probe = new Big();",
            2,
        ),
        // Defaults never take a constructor past seven slots, and a nested function's stores
        // are its own.
        (
            "function Few() { this.a = 1; var f = function () { this.b = 2; this.c = 3; }; }
             Few.prototype = { p: 1, q: 2, r: 3, s: 4, t: 5, u: 6, v: 7, w: 8 }; var probe = new Few();",
            1,
        ),
        // Accessors and functions on the prototype are no defaults.
        (
            "function Acc() { this.a = 1; }
             Object.defineProperty(Acc.prototype, 'b', { get() { return 1; } });
             Acc.prototype.c = function () {}; Acc.prototype.d = function () {};
             Acc.prototype.e = function () {}; var probe = new Acc();",
            0,
        ),
        // A derived class's instance is the base constructor's.
        (
            "class Base { constructor() { this.a = 1; } }
             class Derived extends Base { constructor() { super(); this.b = 2; } }
             var probe = new Derived();",
            0,
        ),
    ];
    for (source, expected) in cases {
        assert_eq!(class(&mut engine, source), expected, "{source}");
    }
}

#[test]
fn a_constructors_class_is_decided_once_whatever_path_constructs() {
    // The first construction runs on the tree-walker's generic dispatch, the rest on compiled
    // code once the loop and the constructor tier up; defaults added after the first
    // construction change nothing, so every instance shares the first one's shape.
    let mut engine = crate::Engine::new();
    let first = object(
        &mut engine,
        "function Pt(x, y) { this.x = x; this.y = y; }
         var first = new Pt(0, 0);
         Pt.prototype.z = 0; Pt.prototype.w = 0;
         var all = [];
         for (var i = 0; i < 3000; i++) all.push(new Pt(i, i));
         var probe = first;",
    );
    let shape = first.borrow().props.shape();
    assert_eq!(first.borrow().props.class(), 0);
    for index in [0, 7, 100, 2999] {
        let instance = object(&mut engine, &format!("var probe = all[{index}];"));
        assert_eq!(instance.borrow().props.class(), 0, "instance {index}");
        assert_eq!(instance.borrow().props.shape(), shape, "instance {index}");
    }
}

#[test]
fn empty_objects_nothing_sizes_take_the_default_class() {
    // Object.create and `new Object()` are filled by their author, as JSC's default six slots
    // assume; a literal's own keys size it.
    let mut engine = crate::Engine::new();
    for source in [
        "var probe = Object.create(null);",
        "var probe = Object.create({ inherited: 1 });",
        "var probe = new Object();",
        "var probe = Object();",
        "class Sub extends Object { constructor() { super(); this.a = 1; } } var probe = new Sub();",
    ] {
        assert_eq!(class(&mut engine, source), DEFAULT_CLASS, "{source}");
    }
    assert_eq!(class(&mut engine, "var probe = { a: 1 };"), 0);
}
