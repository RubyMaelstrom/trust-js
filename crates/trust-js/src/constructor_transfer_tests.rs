//! Packed constructor-to-property ownership transfers. Local ECMA-262 e28783d5fc9d:
//! [[Construct]] (§10.2.2), OrdinarySetWithOwnDescriptor (§10.1.9.2), ToBoolean (§7.1.2).
//! Guards still perform the normative operations; only the intermediate representation changes.
use super::*;
use crate::bytecode::Tier;
use crate::{Completion, Engine};
use std::cell::Cell;

thread_local! {
    static SIMPLE: Cell<usize> = const { Cell::new(0) };
    static FORWARDED: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_simple() {
    SIMPLE.with(|count| count.set(count.get() + 1));
}

pub(super) fn record_forwarded() {
    FORWARDED.with(|count| count.set(count.get() + 1));
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("constructor fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn setup(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function Simple(x,y){this.value=x;this.other=y;}
        function Wrapper(){this.initialize.apply(this,arguments);}
        Wrapper.prototype.initialize=function(x,y){
            if(!x)x=19;if(!y)y=23;this.value=x;this.other=y;
        };
        function simple(x,y){return new Simple(x,y);}
        function forwarded(x,y){return new Wrapper(x,y);}
        for(var warm=0;warm<600;warm++){simple(warm,1);forwarded(warm,2);}
    "#,
    );
    engine
}

fn assert_native_hit(tier: Tier, forwarded: bool, before: usize) {
    if tier == Tier::Jit
        && cfg!(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(
                target_os = "macos",
                target_os = "linux",
                target_os = "windows"
            )
        ))
    {
        let after = if forwarded {
            FORWARDED.with(Cell::get)
        } else {
            SIMPLE.with(Cell::get)
        };
        assert!(
            after > before,
            "must exercise committed packed transfer, not just its fallback"
        );
    }
}

#[test]
fn constructor_transfer_every_value_kind_and_descriptor_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let before = SIMPLE.with(Cell::get);
        let mut engine = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,
                3.25,-4.5,0n,-98765432109876543210n,'','λ string',Symbol('s'),{key:7}];
            var okay=true;
            for(var pass=0;pass<40;pass++)for(var k=0;k<values.length;k++){
                var result=simple(values[k],values[(k+1)%values.length]);
                var desc=Object.getOwnPropertyDescriptor(result,'value');
                okay=okay&&Object.is(result.value,values[k])&&
                    Object.is(result.other,values[(k+1)%values.length])&&
                    desc.writable&&desc.enumerable&&desc.configurable&&
                    Object.keys(result).join(',')==='value,other';
            }
            okay;
        "#
            ),
            "true",
            "{tier:?}"
        );
        assert_native_hit(tier, false, before);
    }
}

#[test]
fn constructor_transfer_forwarded_defaults_keep_truthiness_and_identity() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let before = FORWARDED.with(Cell::get);
        let mut engine = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var falseValues=[undefined,null,false,0,-0,NaN,0n,'',$262.IsHTMLDDA];
            var calls=0, coercion={valueOf:function(){calls++;throw 42;},toString:function(){calls++;throw 43;}};
            var trueValues=[true,3.25,-4.5,Infinity,-Infinity,1n,-1n,'0','λ',Symbol('s'),coercion];
            var okay=true;
            for(var pass=0;pass<40;pass++){
                for(var k=0;k<falseValues.length;k++){
                    var result=forwarded(falseValues[k],falseValues[k]);
                    okay=okay&&result.value===19&&result.other===23;
                }
                for(var k=0;k<trueValues.length;k++){
                    var result=forwarded(trueValues[k],trueValues[k]);
                    okay=okay&&Object.is(result.value,trueValues[k])&&Object.is(result.other,trueValues[k]);
                }
            }
            var missing=new Wrapper();
            okay&&calls===0&&missing.value===19&&missing.other===23;
        "#
            ),
            "true",
            "{tier:?}"
        );
        assert_native_hit(tier, true, before);
    }
}

#[test]
fn constructor_transfer_sole_object_owner_survives_gc_then_releases() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for maker in ["simple", "forwarded"] {
            let mut engine = setup(tier);
            eval(
                &mut engine,
                &format!("var retained={maker}({{marker:71}}, Symbol('retained'));"),
            );
            let env = engine.interp.global_env;
            // Strong handles live only in this helper frame (the collector scans the stack).
            let weak = crate::gc_test::in_frame(|| {
                let retained = engine.interp.get_var("retained", &env).ok().unwrap();
                let object = retained.as_obj().unwrap();
                let payload = object.borrow().props.get("value").unwrap().value();
                Gc::downgrade(&payload.as_obj().unwrap())
            });
            engine.interp.gc_collect();
            assert_eq!(
                eval(
                    &mut engine,
                    "retained.value.marker+'|'+String(retained.other)"
                ),
                "71|Symbol(retained)"
            );
            eval(&mut engine, "retained=null;");
            engine.interp.gc_collect();
            assert!(
                !crate::gc_test::alive(&weak),
                "{tier:?}, {maker}: no stale argument owner"
            );
        }
    }
}

#[test]
fn constructor_transfer_surplus_evaluation_duplicates_and_fallback_order() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var log=[], token={id:1};
            function argument(n){log.push(n);return token;}
            var result=new Simple(argument(1),argument(2),argument(3));
            function Duplicate(x){this.a=x;this.b=x;}
            var dup;
            for(var k=0;k<200;k++)dup=new Duplicate(token);
            Object.defineProperty(Simple.prototype,'value',{
                configurable:true,set:function(v){log.push(v===token?'setter':'bad');this.saved=v;}
            });
            var fallback=new Simple(argument(4),argument(5),argument(6));
            var beforeThrow={};
            Object.defineProperty(Simple.prototype,'value',{
                configurable:true,set:function(v){log.push('throw');throw beforeThrow;}
            });
            var thrown=false;try{new Simple(argument(7),argument(8),argument(9));}catch(e){thrown=e===beforeThrow;}
            [result.value===token,result.other===token,dup.a===token,dup.b===token,
                fallback.saved===token,!Object.hasOwn(fallback,'value'),fallback.other===token,
                thrown,log.join(',')].join('|');
        "#
            ),
            "true|true|true|true|true|true|true|true|1,2,3,4,5,6,setter,7,8,9,throw",
            "{tier:?}"
        );
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn simple_and_forwarded_initializers_copy_an_immutable_argument_window() {
    // Both plans cache the global prototype/call epochs. Isolate strict per-call native
    // counters so unrelated Engine startup/retirement cannot make a correct guard miss.
    const CHILD: &str = "TRUST_JS_TEST_CONSTRUCTOR_ARGUMENT_LOAN_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "interpreter::constructor_transfer_tests::simple_and_forwarded_initializers_copy_an_immutable_argument_window", "--test-threads=1"])
            .env(CHILD, "1").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let mut engine = setup(Tier::Jit);
    let values = crate::rooting::RootedVec::from_vec(vec![
        Value::Undefined,
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Num(-0.0),
        Value::Num(f64::NAN),
        Value::Num(3.5),
        Value::bigint_from_i128((1i128 << 100) + 7),
        Value::from_string("constructor argument storage".repeat(32)),
        Value::Sym(crate::value::new_symbol_ref(crate::value::SymbolData {
            id: u64::MAX,
            description: None,
            weak_observers: Default::default(),
            agent_held: Default::default(),
        })),
        Value::Obj(engine.interp.new_object()),
    ]);
    for (name, forwarded) in [("Simple", false), ("Wrapper", true)] {
        let env = engine.interp.global_env;
        let callee = engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("constructor"));
        for value in &values {
            // This is the original immutable canonical Value array, not a mutable transfer
            // adapter. The two initializer branches previously replaced these words.
            let arguments = [*value, *value];
            let before = if forwarded {
                FORWARDED.with(Cell::get)
            } else {
                SIMPLE.with(Cell::get)
            };
            let result = unsafe {
                engine.interp.construct_jit_fast(
                    &callee,
                    arguments.as_ptr().cast::<Value>(),
                    arguments.len(),
                    None,
                    None,
                )
            }
            .expect("warm constructor plan")
            .unwrap_or_else(|_| panic!("constructor call"));
            let after = if forwarded {
                FORWARDED.with(Cell::get)
            } else {
                SIMPLE.with(Cell::get)
            };
            assert_eq!(
                after,
                before + 1,
                "the exact {name} initializer plan must commit"
            );
            assert!(
                arguments
                    .iter()
                    .all(|argument| argument.bits() == value.bits()),
                "{name} rewrote its caller's immutable window"
            );
            let result = result.as_obj().expect("constructed instance");
            let expected = if forwarded && !engine.interp.to_boolean(value) {
                Value::Num(19.0)
            } else {
                *value
            };
            assert_eq!(
                result.borrow().props.get("value").unwrap().value().bits(),
                expected.bits()
            );
        }
    }
}

/// A forwarding constructor whose initializer is not a field plan calls it, and the initializer's
/// frame is called from the body's forwarding call (`this.initialize.apply(this, arguments)`): the
/// constructor's frame prints that call's position, at every tier, the VM's as the template JIT's
/// Rust forwarder's (which recorded none, so that the frame printed the function's start). The
/// rest of what the initializer can observe, too: its caller, its `arguments`, `new.target`, a
/// throw's stack, a returned object ignored, missing arguments.
#[test]
fn a_forwarded_initializer_is_called_from_the_forwarding_call_at_every_tier() {
    let source = r#"
        var Class = { create: function() { return function() { this.initialize.apply(this, arguments); } } };
        var V = Class.create();
        var seen = [];
        V.prototype = {
            initialize: function(x, y) {
                this.x = x; this.y = y;
                if (x === 7) {
                    seen.push(new Error("probe").stack.split("\n").slice(0, 4).join(" | "));
                    seen.push((V.prototype.initialize.caller === V) + " " + arguments.length + " ["
                        + Array.prototype.slice.call(arguments) + "] " + (typeof new.target));
                }
                if (x === 8) throw new Error("thrown");
                if (x === 9) return { other: 1 };
            }
        };
        function make(i) { return new V(i, i + 1, "extra"); }
        var s = 0;
        for (var i = 0; i < 600; i++) s += make(i % 6).x;
        make(7);
        try { make(8); } catch (e) { seen.push(e.stack.split("\n").slice(0, 3).join(" | ")); }
        var r9 = make(9);
        seen.push((r9.other === undefined) + " " + (r9 instanceof V));
        var m = new V(1);
        seen.push(m.x + " " + m.y);
        s + "\n" + seen.join("\n");
    "#;
    let mut results = Vec::new();
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        results.push((tier, eval(&mut engine, source)));
    }
    let (_, first) = &results[0];
    assert!(
        first.contains("at new <anonymous> (") && first.contains("true 3 [7,8,extra] undefined"),
        "{first}"
    );
    for (tier, result) in &results {
        assert_eq!(result, first, "{tier:?} against {:?}", results[0].0);
    }
}

// ---- P3 stage 3 steps 6-7, part A2: the direct construct sequence's forward form ----

/// The forward form's completions on this thread (ARM64 only; 0 where it is not emitted).
fn forwards() -> usize {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        crate::jit::TEST_DIRECT_FORWARDS.with(Cell::get)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )))]
    {
        0
    }
}

fn forward_form_emitted() -> bool {
    cfg!(all(
        target_arch = "aarch64",
        any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )
    ))
}

/// A forwarding constructor's observables (A0's), from a cold site (the construct stub's forward
/// form) and a site in a loop (an in-line site's call of that stub), with the initializer's own
/// `apply` the builtin where `own_apply` (the forward form's fill refuses an own `apply`, so the
/// template JIT takes the Rust forwarder: the miss forced).
fn forwarder_observables(own_apply: bool) -> String {
    let own = if own_apply {
        "Object.defineProperty(V.prototype.initialize, 'apply', { value: Function.prototype.apply, writable: true, configurable: true });"
    } else {
        ""
    };
    format!(
        r#"
        var Class = {{ create: function() {{ return function() {{ this.initialize.apply(this, arguments); }} }} }};
        var V = Class.create();
        var seen = [];
        V.prototype = {{
            initialize: function(x, y) {{
                this.x = x; this.y = y;
                if (x === 7) {{
                    seen.push(new Error("probe").stack.split("\n").slice(0, 4).join(" | "));
                    seen.push((V.prototype.initialize.caller === V) + " " + arguments.length + " ["
                        + Array.prototype.slice.call(arguments) + "] " + (typeof new.target) + " "
                        + (this instanceof V));
                }}
                if (x === 8) throw new Error("thrown");
                if (x === 9) return {{ other: 1 }};
            }}
        }};
        {own}
        function make(i) {{ return new V(i, i + 1, "extra"); }}
        function loop(n) {{ var s = 0; for (var i = 0; i < n; i++) {{ var o = new V(i % 6, i); s += o.x + o.y; }} return s; }}
        var s = 0;
        for (var i = 0; i < 600; i++) s += make(i % 6).x;
        s += loop(3000);
        make(7);
        try {{ make(8); }} catch (e) {{ seen.push(e.stack.split("\n").slice(0, 3).join(" | ")); }}
        var r9 = make(9);
        seen.push((r9.other === undefined) + " " + (r9 instanceof V) + " " + r9.x);
        var m = new V(1);
        seen.push(m.x + " " + m.y + " " + Object.keys(m).join(","));
        s + "\n" + seen.join("\n");
    "#
    )
}

/// The forward form (part A2) gives what the VM and the Rust forwarder give: Error.stack through
/// the constructor and the initializer (both frames, the constructor's at the forwarding call, as
/// commit 0 made the Rust path's), the initializer's caller, its `arguments` (the site's: an extra
/// argument visible, a missing one undefined), new.target undefined, a throw's stack, a returned
/// object ignored; from a cold site and a site in a loop; and the template JIT, with the Rust
/// forwarder forced by an own `apply`, gives the same.
#[test]
fn a_forwarded_construction_enters_the_initializer_as_the_rust_forwarder_does() {
    let mut outputs = Vec::new();
    for own_apply in [false, true] {
        let source = forwarder_observables(own_apply);
        let mut results = Vec::new();
        for tier in [Tier::Bytecode, Tier::Jit] {
            let before = forwards();
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            results.push(eval(&mut engine, &source));
            let taken = forwards() - before;
            if tier == Tier::Jit && forward_form_emitted() && !own_apply {
                assert!(taken > 3000, "the forward form must run (it ran {taken})");
            } else {
                assert_eq!(taken, 0, "{tier:?}, own apply {own_apply}: no forward form");
            }
        }
        assert_eq!(
            results[0], results[1],
            "the VM against the template JIT, own apply {own_apply}"
        );
        assert!(
            results[0].contains("true 3 [7,8,extra] undefined true"),
            "{}",
            results[0]
        );
        outputs.push(results.remove(0));
    }
    assert_eq!(
        outputs[0], outputs[1],
        "the forward form against the Rust forwarder"
    );
}

/// Each of the forward form's proofs (`bytecode::forward_this`) fails where the program
/// changes what it proves, and the construction then does what the VM does: the initializer
/// replaced on the prototype (another body, other fields; twice, the second write keeping the
/// prototype's shape), an accessor `initialize`, the
/// initializer moved to another key (the prototype's shape), an own `apply` on the initializer,
/// another [[Prototype]] for the initializer, Function.prototype.apply replaced, a Proxy as the
/// prototype; the forward form stands down at each and takes the constructions again once the
/// Rust forwarder proves a new state (an initializer, the apply restored). A learned layout that
/// goes stale (the same initializer storing other fields) costs only the instance's transitions.
/// Each phase runs after the forward form has warmed; which form ran each phase is checked
/// apart from the program's results, which the VM's equal.
#[test]
fn the_forward_forms_proofs_fail_where_the_program_changes_what_they_prove() {
    let source = r#"
        var Class = { create: function() { return function() { this.initialize.apply(this, arguments); } } };
        var V = Class.create();
        V.prototype = { initialize: function(x, y) { this.x = x; this.y = y; } };
        var out = [], modes = [];
        // Every construction's own keys, counted, and the last instance: a construction that ran
        // another initializer than the VM shows in the counts.
        function build(n) {
            var s = "", keys = {};
            for (var i = 0; i < n; i++) { var o = new V(i, 2); var k = Object.keys(o).join(); keys[k] = (keys[k] || 0) + 1; s = JSON.stringify(o) + (o instanceof V); }
            var counts = Object.keys(keys).map(function(k) { return k + "=" + keys[k]; }).join(";");
            return counts === Object.keys(JSON.parse(s.slice(0, -4))).join() + "=" + n ? s : s + " MIXED " + counts;
        }
        function phase(name, f) {
            var before = forwardsNow(), r;
            try { r = f(); } catch (e) { r = e.constructor.name; }
            out.push(name + " " + r);
            modes.push(name + (forwardsNow() - before > 400 ? " forward" : " rust"));
        }
        phase("warm", function() { return build(2000); });
        V.prototype.initialize = function(a, b) { this.a = a * 10; this.b = b; this.c = 3; };
        phase("replaced", function() { return build(500); });
        V.prototype.initialize = function(a, b) { this.d = a + b; };
        phase("replaced again", function() { return build(500); });
        var got = 0;
        var keep = V.prototype.initialize;
        Object.defineProperty(V.prototype, "initialize", { get: function() { got++; return keep; }, configurable: true });
        phase("getter", function() { return build(500) + " " + got; });
        Object.defineProperty(V.prototype, "initialize", { value: function(x) { this.z = x; }, writable: true, configurable: true });
        phase("data again", function() { return build(500); });
        var moved = V.prototype.initialize;
        delete V.prototype.initialize;
        V.prototype.other = moved;
        phase("moved", function() { return build(500); });
        delete V.prototype.other;
        V.prototype.initialize = moved;
        phase("back", function() { return build(500); });
        var applied = 0;
        V.prototype.initialize.apply = function(t, args) { applied++; t.own = args.length; };
        phase("own apply", function() { return build(500) + " " + applied; });
        delete V.prototype.initialize.apply;
        phase("own apply gone", function() { return build(500) + " " + applied; });
        // Another holder (setPrototypeOf keeps the initializer's shape), whose slots hold the
        // builtin apply under other keys, its own `apply` another function.
        var via = 0, fa = Function.prototype.apply;
        Object.setPrototypeOf(V.prototype.initialize, { k0: fa, k1: fa, k2: fa, k3: fa, k4: fa, k5: fa, k6: fa, k7: fa, apply: function(t, args) { via++; t.via = args[0]; } });
        phase("other holder", function() { return build(500) + " " + via; });
        Object.setPrototypeOf(V.prototype.initialize, Function.prototype);
        phase("holder back", function() { return build(500); });
        var builtin = Function.prototype.apply, wrapped = 0;
        Function.prototype.apply = function(t, args) { wrapped++; return builtin.call(this, t, args); };
        phase("apply replaced", function() { return build(500) + " " + wrapped; });
        Function.prototype.apply = builtin;
        phase("apply restored", function() { return build(500) + " " + wrapped; });
        var L = Class.create();
        L.prototype = { initialize: function(k) { if (k < 1000) { this.p = k; this.q = 1; } else { this.r = k; this.p = 2; } } };
        function layout(from, n) { var s = "", bad = 0; for (var i = from; i < from + n; i++) { var o = new L(i); if (Object.keys(o).join() !== (i < 1000 ? "p,q" : "r,p")) bad++; s = JSON.stringify(o) + Object.keys(o); } return s + (bad ? " BAD " + bad : ""); }
        phase("layout learned", function() { return layout(0, 1000); });
        phase("layout stale", function() { return layout(1000, 1000); });
        // Function.prototype's shape changes while the builtin stays in it under another key,
        // added before `apply` is deleted, then `apply` replaced.
        var b2 = Function.prototype.apply, movedApply = 0;
        Function.prototype.zzz = b2;
        delete Function.prototype.apply;
        Function.prototype.apply = function(t, args) { movedApply++; return b2.call(this, t, args); };
        phase("apply moved", function() { return build(500) + " " + movedApply; });
        var W = Class.create(), traps = 0;
        W.prototype = new Proxy({ initialize: function(x) { this.w = x; } }, { get: function(t, k, r) { traps++; return t[k]; } });
        phase("proxy", function() { var w = ""; for (var i = 0; i < 500; i++) w = new W(i).w; return w + " " + (traps >= 500); });
        out.join("\n") + "\n#" + modes.join(",");
    "#;
    let mut results = Vec::new();
    let mut modes = Vec::new();
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("forwardsNow", 0, |_, _, _| {
            Ok(crate::embed::Value::Num(forwards() as f64))
        });
        let result = eval(&mut engine, source);
        let (out, mode) = result.split_once("\n#").expect("the modes line");
        results.push(out.to_string());
        modes.push(mode.to_string());
    }
    assert_eq!(results[0], results[1], "the VM against the template JIT");
    assert_eq!(
        results[0],
        [
            r#"warm {"x":1999,"y":2}true"#,
            r#"replaced {"a":4990,"b":2,"c":3}true"#,
            r#"replaced again {"d":501}true"#,
            r#"getter {"d":501}true 500"#,
            r#"data again {"z":499}true"#,
            "moved TypeError",
            r#"back {"z":499}true"#,
            r#"own apply {"own":2}true 500"#,
            r#"own apply gone {"z":499}true 500"#,
            r#"other holder {"via":499}true 500"#,
            r#"holder back {"z":499}true"#,
            r#"apply replaced {"z":499}true 500"#,
            r#"apply restored {"z":499}true 500"#,
            r#"layout learned {"p":999,"q":1}p,q"#,
            r#"layout stale {"r":1999,"p":2}r,p"#,
            r#"apply moved {"z":499}true 500"#,
            "proxy 499 true",
        ]
        .join("\n")
    );
    let expected_jit = if forward_form_emitted() {
        "warm forward,replaced forward,replaced again forward,getter rust,data again forward,moved rust,back forward,\
         own apply rust,own apply gone forward,other holder rust,holder back forward,\
         apply replaced rust,apply restored forward,layout learned forward,layout stale forward,\
         apply moved rust,proxy rust"
    } else {
        "warm rust,replaced rust,replaced again rust,getter rust,data again rust,moved rust,back rust,own apply rust,\
         own apply gone rust,other holder rust,holder back rust,apply replaced rust,\
         apply restored rust,layout learned rust,layout stale rust,apply moved rust,proxy rust"
    };
    assert!(
        modes[0].split(',').all(|m| m.ends_with(" rust")),
        "the VM: {}",
        modes[0]
    );
    assert_eq!(modes[1], expected_jit);
}

/// The forward form counts a forwarding construction's depth as the Rust forwarder does (the
/// constructor, the elided apply and the initializer: the depth inside the initializer is the
/// caller's + 3 on both, and on the VM), and a recursion through forwarding constructions that
/// exhausts the stack throws the RangeError and unwinds to the caller's depth. (The exact level
/// it throws at is the native stack's: the forward form's frames are smaller than the Rust
/// path's, so it reaches deeper.)
#[test]
fn the_forward_form_counts_a_forwarding_constructions_depth_as_the_rust_path() {
    // Under GC stress a collection at every allocation scans every frame of the recursion, which
    // makes exhausting the stack quadratic: the depth's accounting alone is checked then.
    let exhaust = !crate::gc_test::stress_active();
    let run = |tier: Tier, own_apply: bool| -> String {
        let own = if own_apply {
            "Object.defineProperty(D.prototype.initialize, 'apply', { value: Function.prototype.apply });"
        } else {
            ""
        };
        let source = format!(
            r#"
            var Class = {{ create: function() {{ return function() {{ this.initialize.apply(this, arguments); }} }} }};
            var D = Class.create();
            D.prototype = {{ initialize: function(n) {{ this.d = depthNow(); this.v = n > 0 ? deep(n - 1) + 1 : 0; }} }};
            {own}
            function deep(n) {{ return (new D(n)).v; }}
            function probe() {{ var base = depthNow(); return (new D(0)).d - base; }}
            for (var w = 0; w < 300; w++) {{ probe(); deep(3); }}
            var before = probe(), top = depthNow(), thrown = "none";
            if ({exhaust}) {{ try {{ deep(1e7); }} catch (e) {{ thrown = e.constructor.name; }} }}
            else thrown = "RangeError";
            [before, probe(), depthNow() - top, thrown].join(" ");
        "#
        );
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("depthNow", 0, |ctx, _, _| {
            Ok(crate::embed::Value::Num(f64::from(ctx.interp().depth)))
        });
        eval(&mut engine, &source)
    };
    let before = forwards();
    let forward = run(Tier::Jit, false);
    if forward_form_emitted() {
        assert!(forwards() > before, "the forward form must run");
    }
    let rust = run(Tier::Jit, true);
    let vm = run(Tier::Bytecode, false);
    assert_eq!(forward, rust, "the forward form against the Rust forwarder");
    assert_eq!(forward, vm, "the forward form against the VM");
    assert!(forward.ends_with(" 0 RangeError"), "{forward}");
}

/// Collections at every allocation while forwarding constructions run, the initializer allocating
/// (so its entry poll and its own allocations collect while the instance is held only by the
/// initializer's record and the sequence's save area), and an interrupt raised inside the
/// initializer: the instances keep their fields, the interrupt leaves no frame behind (a later
/// construction's stack is a fresh one) and the forward form still runs after it.
#[test]
fn forwarding_constructions_survive_collections_and_interrupts() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine.define_global("interruptNow", 0, |ctx, _, _| {
        ctx.interp()
            .runtime_interrupt
            .set_deadline(Some(std::time::Instant::now()));
        Ok(crate::embed::Value::Undefined)
    });
    eval(
        &mut engine,
        r#"
        var Class = { create: function() { return function() { this.initialize.apply(this, arguments); } } };
        var V = Class.create();
        var stop = false;
        V.prototype = { initialize: function(x, y) { this.x = x; this.list = [x, y, { y: y }]; if (stop) { interruptNow(); while (true) {} } } };
        function build(n) { var s = 0; for (var i = 0; i < n; i++) { var o = new V(i, i + 1); s += o.x + o.list[2].y; } return s; }
        build(600);
    "#,
    );
    let fresh_stack = "(function f() { return new Error('x').stack.split('\\n').length; })()";
    let baseline = eval(&mut engine, fresh_stack);
    let before = forwards();
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(1);
    assert_eq!(eval(&mut engine, "build(400)"), "160000");
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    if forward_form_emitted() {
        assert!(
            forwards() > before,
            "the forward form must run under the collections"
        );
    }
    let result = engine
        .eval_interruptible("stop = true; build(10)", false)
        .expect("parses");
    assert!(
        matches!(
            result,
            crate::ExecutionOutcome::Interrupted {
                reason: crate::InterruptReason::DeadlineExceeded
            }
        ),
        "the interrupt inside the initializer"
    );
    engine.interrupt_handle().set_deadline(None);
    let before = forwards();
    assert_eq!(eval(&mut engine, "stop = false; build(300)"), "90000");
    assert_eq!(
        eval(&mut engine, fresh_stack),
        baseline,
        "no frame left behind"
    );
    if forward_form_emitted() {
        assert!(
            forwards() > before,
            "the forward form runs after the interrupt"
        );
    }
}

/// At the engine's own thresholds the initializer's code passes its inline checkpoints though the
/// Rust forwarder, which the forward form leaves each checkpoint's count to, runs a field plan in
/// its place: the forward form takes nearly every construction once warm, rather than declining
/// at the first checkpoint for ever.
#[test]
fn the_forward_form_passes_the_initializers_inline_checkpoints() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    eval(
        &mut engine,
        r#"
        var Class = { create: function() { return function() { this.initialize.apply(this, arguments); } } };
        var V = Class.create();
        V.prototype = { x: 0, y: 0, initialize: function(x, y) { this.x = x; this.y = y; } };
        function loop(n) { var s = 0; for (var i = 0; i < n; i++) { var o = new V(i, i); s += o.x; } return s; }
        for (var k = 0; k < 20; k++) loop(1000);
    "#,
    );
    let before = forwards();
    assert_eq!(eval(&mut engine, "loop(10000)"), "49995000");
    if forward_form_emitted() {
        let taken = forwards() - before;
        assert!(
            taken > 9900,
            "the forward form took {taken} of 10000 warm constructions"
        );
    }
}
