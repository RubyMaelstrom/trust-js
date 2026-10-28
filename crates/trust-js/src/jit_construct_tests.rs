//! `new` through a site's construct cache. ECMA-262 (e28783d5): EvaluateNew, Construct,
//! [[Construct]] for ECMAScript function objects, OrdinaryCreateFromConstructor,
//! GetPrototypeFromConstructor and InitializeInstanceElements. Expected values are V8's.

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
