//! Inlining must retain the callee's execution-context semantics. ECMA-262
//! GetThisEnvironment, GetSuperBase, MakeSuperPropertyReference and ResolveBinding;
//! official local snapshot e28783d5fc9dc12b3de905961e2c71410b38a202 (2026-09-06).

use crate::{ast::Function, bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("inline context fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn function(engine: &mut Engine, name: &str) -> Rc<Function> {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing fixture function {name}"));
    let object_handle = value.as_obj().expect("fixture function object");
    let object = object_handle.borrow();
    let Callable::User(user) = &object.call else {
        panic!("fixture user function")
    };
    user.func.clone()
}

fn require_hot_native(engine: &mut Engine, name: &str, inlined: bool) {
    let function = function(engine, name);
    let runtime = function.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .expect("hot fixture must compile");
    if cfg!(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )
    )) {
        assert!(
            chunk.jit.get().flatten().is_some(),
            "{name} must retain ordinary native execution"
        );
        if crate::bytecode::inline_recompile_at() == 100 {
            assert!(
                chunk.inline_attempted.get(),
                "{name} must cross the hot inline checkpoint"
            );
            assert_eq!(
                function
                    .runtime()
                    .code2
                    .get()
                    .and_then(Option::as_ref)
                    .is_some(),
                inlined,
                "{name} must respect the splice context capability"
            );
        }
    }
}

#[test]
fn hot_super_method_keeps_home_object_in_an_ordinary_caller() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            'use strict';
            class Base {value(x){return x+1;}}
            class Derived extends Base {value(x){return super.value(x)+1;}}
            var receiver=new Derived(), nativeSuper=Derived.prototype.value;
            function invokeSuper(x){return receiver.value(x)+1;}
            function driveSuper(){var sum=0;for(var k=0;k<1000;k++)sum+=invokeSuper(k);return sum;}
            driveSuper();
        "#
            ),
            "502500",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            require_hot_native(&mut engine, "invokeSuper", false);
            let method = function(&mut engine, "nativeSuper");
            let runtime = method.runtime();
            let chunk = runtime
                .code
                .get()
                .and_then(Option::as_ref)
                .expect("native super body");
            if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
                assert!(chunk.jit.get().flatten().is_some());
            }
            assert!(chunk
                .jit_ops()
                .iter()
                .any(|op| matches!(op, crate::bytecode::Op::SuperBase)));
        }
    }
}

#[test]
fn hot_super_methods_keep_distinct_homes_receivers_and_live_prototypes() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            'use strict';
            class Base {read(){return this.mark+10;}}
            class Derived extends Base {read(){return super.read()+1;}}
            class OtherBase {read(){return this.mark+100;}}
            class Other extends OtherBase {invoke(obj){return obj.read()+super.read();}}
            var target=new Derived(), caller=new Other();target.mark=7;caller.mark=2;
            function drive(){var sum=0;for(var k=0;k<600;k++)sum+=caller.invoke(target);return sum;}
            var before=drive();
            Object.setPrototypeOf(Derived.prototype,{read(){return this.mark+1000;}});
            before+'|'+caller.invoke(target)+'|'+target.mark+'|'+caller.mark;
        "#
            ),
            "72000|1110|7|2",
            "{tier:?}"
        );
    }
}

#[test]
fn hot_private_names_and_typeof_closure_reads_keep_definition_environments() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            'use strict';
            class Secret {#value=7;read(x){return this.#value+x;}}
            var secret=new Secret();
            function hotPrivate(x){return secret.read(x)+1;}
            var closed='wrong';
            function make(){var closed=12;return function(){return typeof closed;};}
            var closureRead=make();
            function hotTypeof(){return closureRead()+'!';}
            function drive(){var sum=0,out;for(var k=0;k<600;k++){
                sum+=hotPrivate(1);out=hotTypeof();}return sum+'|'+out;}
            drive();
        "#
            ),
            "5400|number!",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            require_hot_native(&mut engine, "hotPrivate", false);
            require_hot_native(&mut engine, "hotTypeof", false);
        }
    }
}

#[test]
fn frame_independent_effects_still_inline_with_local_and_receiver_remapping() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        evaluate(
            &mut engine,
            r#"
        'use strict';
        var receiver={offset:3,compute(x){var y=x+this.offset;return {value:y};}};
        function hotIndependent(x){var y=receiver.compute(x);return y.value+1;}
        function drive(){var sum=0;for(var k=0;k<600;k++)sum+=hotIndependent(2);return sum;}
        drive();
    "#
        ),
        "3600"
    );
    require_hot_native(&mut engine, "hotIndependent", true);
}

/// A spliced self-hosted built-in (`crate::self_hosted`) compiles its intrinsics to the same
/// operations as its own body: they resolve in its private environment, never as free names of
/// the caller. The built-ins are strict, so only strict callers splice them (and only at calls
/// that are not proper tail calls).
#[test]
fn hot_self_hosted_methods_keep_their_intrinsics_when_spliced() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            'use strict';
            var values=[1,2,3,4];
            function anyLarge(list){var r=list.some(function(x){return x>3;});return r;}
            function firstAbove(list){var r=list.find(function(x){return x>2;});return r;}
            function drive(){var n=0;for(var k=0;k<1000;k++)n+=anyLarge(values)+firstAbove(values);return n;}
            drive()+'|'+anyLarge([])+'|'+(function(){try{anyLarge(null)}catch(e){return e.name}})();
        "#
            ),
            "4000|false|TypeError",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            require_hot_native(&mut engine, "anyLarge", true);
            // Where the JIT spliced the call (asserted above), the body kept its operations.
            let caller = function(&mut engine, "anyLarge");
            if let Some(spliced) = caller.runtime().code2.get().and_then(Option::as_ref) {
                assert!(spliced
                    .jit_ops()
                    .iter()
                    .any(|op| matches!(op, crate::bytecode::Op::Abstract(_))));
            }
        }
    }
}

/// A spliced callee resolves its free names from its own [[Environment]] (ECMA-262
/// PrepareForOrdinaryCall/ResolveBinding), never through the caller's scope chain: here the
/// caller's enclosing activation captures its own `shadowed`, which a global-closure callee
/// must not observe.
#[test]
fn spliced_global_closure_reads_ignore_caller_environment_bindings() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var shadowed = 1;
            function globalRead() { return shadowed; }
            var shadowCaller;
            function outer() {
                var shadowed = 2;
                function peek() { return shadowed; }
                shadowCaller = function () {
                    var s = 0;
                    for (var k = 0; k < 10; k++) s += globalRead();
                    return s;
                };
                var r = 0;
                for (var j = 0; j < 500; j++) r = shadowCaller();
                return r + '|' + peek();
            }
            var first = outer();
            shadowed = 5;
            first + '|' + shadowCaller();
        "#
            ),
            "10|2|50",
            "{tier:?}"
        );
    }
}

/// The obfuscated string-table accessor: a callee that reads a binding of its defining
/// function's scope, whose own callee was reassigned to a closure over another activation.
/// Spliced reads stay live, and replacing either function falls back to the generic call.
#[test]
fn spliced_closures_over_non_global_environments_read_live_bindings() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var table, setTable, decode, replaceTable;
            (function () {
                function I(Ns) {
                    Ns = ['a', 'bb', 'ccc', 'dddd'];
                    I = function () { return Ns; };
                    setTable = function (next) { Ns = next; };
                    return I();
                }
                function Q(q, n, B, Y) { return q = q - 218, B = I(), Y = B[q], Y; }
                decode = function (k) { return Q(218 + (k & 3)).length; };
                replaceTable = function () { I = function () { return ['zzzzz', 'z', 'z', 'z']; }; };
            })();
            function hotDecode() {
                var s = 0;
                for (var k = 0; k < 8; k++) s += decode(k);
                return s;
            }
            var warm = 0;
            for (var j = 0; j < 400; j++) warm += hotDecode();
            setTable(['x', 'x', 'x', 'x']);
            var live = hotDecode();
            replaceTable();
            var replaced = hotDecode();
            warm + '|' + live + '|' + replaced;
        "#
            ),
            "8000|8|16",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            require_hot_native(&mut engine, "hotDecode", true);
        }
    }
}

/// Each closure instance has its own [[Environment]]: a splice guarded on one instance must
/// not answer for another instance of the same function.
#[test]
fn spliced_closure_reads_follow_the_called_instance() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            function makeReader(v) { return function () { return v; }; }
            var readers = [makeReader(1), makeReader(10)];
            function readAt(i) { var f = readers[i & 1]; return f() + f(); }
            var s = 0;
            for (var k = 0; k < 1000; k++) s += readAt(k < 900 ? 0 : k);
            s;
        "#
            ),
            "2900",
            "{tier:?}"
        );
    }
}

/// Free names a callee resolves through a `with` object keep their object Environment Record
/// semantics when spliced, including the with-object receiver of a call.
#[test]
fn spliced_reads_through_with_environments_keep_object_bindings() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var scopeObject = { base: 3, twice: function () { return this.base * 2; } };
            var readBase, callTwice;
            with (scopeObject) {
                readBase = function () { return base; };
                callTwice = function () { return twice(); };
            }
            function hotWith() { return readBase() + callTwice(); }
            var s = 0;
            for (var k = 0; k < 500; k++) s += hotWith();
            scopeObject.base = 4;
            s + '|' + hotWith();
        "#
            ),
            "4500|12",
            "{tier:?}"
        );
    }
}

/// Large callers exhaust the shared splice budget by their own size; small callees still
/// splice from their separate budget.
#[test]
fn small_callees_splice_into_callers_beyond_the_shared_budget() {
    let mut body = String::new();
    for k in 0..48 {
        body.push_str(&format!("acc = acc + get({k}) + (x & {k});\n"));
    }
    let source = format!(
        r#"
        var store = [];
        for (var i = 0; i < 64; i++) store.push(i * 2);
        function get(k) {{ return store[k]; }}
        function bigCaller(x) {{ var acc = 0;
        {body}
        return acc; }}
        var t = 0;
        for (var j = 0; j < 400; j++) t += bigCaller(j);
        t;
    "#
    );
    let mut expected = 0f64;
    for j in 0..400u32 {
        for k in 0..48u32 {
            expected += f64::from(k * 2) + f64::from(j & k);
        }
    }
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(&mut engine, &source),
            expected.to_string(),
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let caller = function(&mut engine, "bigCaller");
            let runtime = caller.runtime();
            let chunk = runtime
                .code
                .get()
                .and_then(Option::as_ref)
                .expect("compiled");
            assert!(
                chunk.jit_ops().len() > 320,
                "the fixture must exceed the shared budget"
            );
            require_hot_native(&mut engine, "bigCaller", true);
        }
    }
}

/// A splice compiles the callee's declarations in fresh slots without touching the caller's
/// capture analysis: a callee `let n` spliced before the caller's own captured block `let n`
/// must not consume that binding's homed declaration (even when the splice is abandoned).
#[test]
fn spliced_lexicals_leave_the_callers_homed_block_bindings_alone() {
    // A global callee, and the closure form Glimmer uses (inlinable since splices resolve
    // free names from the callee's environment).
    let definitions = [
        "function Df(t){if(!Sf(t))return!1;let n=t[0];return n===30||n===35}",
        "function Cf(e){return t=>{if(!Sf(t))return!1;let n=t[0];return n===30||n===e}}\n\
         const Df=Cf(35);",
    ];
    for definition in definitions {
        let source = r#"
            'use strict';
            function Sf(e){return Array.isArray(e)&&e.length===2}
            DEFINITION
            var out = [];
            function mp(e,a){a()}
            var homedCaller=(e,t)=>{if(Df(t))e(1);else if(t[0]===6){let n=t[1];
                Df(n)?e(2):mp(e,()=>{e(n)})}else e(0)};
            function rec(a) { out.push(String(a)); }
            for (var i = 0; i < 2000; i++) {
                homedCaller(rec, [35, 'x']);
                homedCaller(rec, [6, [35, 'y']]);
                homedCaller(rec, [6, [7, 'z']]);
                homedCaller(rec, [9]);
            }
            out.length + '|' + out.slice(-6).join(',');
        "#
        .replace("DEFINITION", definition);
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(&mut engine, &source),
                "8000|7,z,0,1,2,7,z,0",
                "{tier:?} {definition}"
            );
        }
    }
}

#[test]
fn spliced_callee_outlives_dropping_its_last_reference() {
    // The splice reads the callee's captured bindings from its [[Environment]], which lives as
    // long as the function: the body dropping every other reference to the function must leave
    // it alive until the splice completes (warm machine-code reads and the cold helper alike).
    let source = r#"
        function makeWarm(){ var secret = {v: 42}; var other = [1,2,3];
            return function target(holder){ if (holder.drop) holder.fn = null;
                return secret.v + other.length; }; }
        function makeCold(){ var rare = 'cold'; var hot = 1;
            return function target(holder){ if (holder.drop) { holder.fn = null; return rare; }
                return hot; }; }
        function invoke(holder){ return holder.fn(holder); }
        function drive(make){
            var holder = { fn: make(), drop: false }, last;
            for (var i = 0; i < 3000; i++) last = invoke(holder);
            holder.drop = true;
            return last + '->' + invoke(holder);
        }
        drive(makeWarm) + '|' + drive(makeCold);
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source), "45->45|1->cold", "{tier:?}");
        if tier == Tier::Jit {
            require_hot_native(&mut engine, "invoke", true);
        }
    }
}

#[test]
fn legacy_caller_finds_spliced_activations() {
    // `f.caller` of a frame called from a spliced callee names that callee, and a function that
    // reads its own `caller` keeps its frame (nested splices included).
    let source = r#"
        function inspect() { return inspect.caller; }
        function leaf(x) { return inspect() === leaf ? x : -1; }
        function mid(x) { return leaf(x); }
        function invoke(x) { return mid(x); }
        function self(x) { return self.caller === viaSelf ? x : -1; }
        function viaSelf(x) { return self(x); }
        function strictLeaf(x) { 'use strict'; return probe(x); }
        function probe(x) { return probe.caller === null ? x : -1; }
        function viaStrict(x) { return strictLeaf(x); }
        var bad = 0;
        for (var n = 0; n < 2000; n++) {
            if (invoke(n) !== n) bad++;
            if (viaSelf(n) !== n) bad++;
            if (viaStrict(n) !== n) bad++;
        }
        String(bad);
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source), "0", "{tier:?}");
        if tier == Tier::Jit {
            require_hot_native(&mut engine, "invoke", true);
        }
    }
}

#[test]
fn legacy_caller_keeps_splices_across_implicit_author_calls() {
    // Expected results verified with Node. Legacy caller is an implementation extension;
    // ECMA-262 Forbidden Extensions still forbids exposing a strict function as its value.
    let source = r#"
        var seen, fail = false;
        function observe() {
            if (fail) throw new Error('getter');
            return observe.caller;
        }
        function store(value) { seen = store.caller; }
        function number() { seen = number.caller; return 7; }
        var receiver = {
            named() { return this.value; },
            computed(key) { return this[key]; },
            write() { this.value = 1; return 1; },
            writeComputed(key) { this[key] = 1; return 1; },
            numeric() { return +this.numericValue; },
            strict() { 'use strict'; return this.value; }
        };
        Object.defineProperty(receiver, 'value', { get: observe, set: store });
        receiver.numericValue = { valueOf: number };
        function invokeNamed() { return receiver.named(); }
        function invokeComputed(key) { return receiver.computed(key); }
        function invokeWrite() { return receiver.write(); }
        function invokeWriteComputed(key) { return receiver.writeComputed(key); }
        function invokeNumeric() { return receiver.numeric(); }
        function invokeStrict() { 'use strict'; return receiver.strict(); }
        function nested() { return invokeNamed(); }
        function direct() { return observe(); }
        var bad = 0;
        for (var n = 0; n < 2000; n++) {
            if (invokeNamed() !== receiver.named) bad++;
            if (invokeComputed('value') !== receiver.computed) bad++;
            if (nested() !== receiver.named) bad++;
            invokeWrite();
            if (seen !== receiver.write) bad++;
            invokeWriteComputed('value');
            if (seen !== receiver.writeComputed) bad++;
            if (invokeNumeric() !== 7 || seen !== receiver.numeric) bad++;
            if (invokeStrict() !== null) bad++;
        }
        fail = true;
        try { nested(); } catch (error) { if (error.message !== 'getter') bad++; }
        fail = false;
        if (direct() !== direct || invokeNamed() !== receiver.named) bad++;
        String(bad);
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source), "0", "{tier:?}");
        if tier == Tier::Jit {
            for name in [
                "invokeNamed",
                "invokeComputed",
                "invokeWrite",
                "invokeWriteComputed",
                "invokeNumeric",
                "invokeStrict",
                "nested",
            ] {
                require_hot_native(&mut engine, name, true);
            }
        }
    }
}

#[test]
fn spliced_implicit_callbacks_preserve_captured_and_thrown_stack_positions() {
    let source = r#"var throwing = false;
function observeStack() {
    var error = new Error('getter');
    if (throwing) throw error;
    return error;
}
function storeStack(value) { throw new Error('setter'); }
var receiver = {
    read() { return this.value; },
    write() { this.value = 1; },
    directError() { return null.value; }
};
Object.defineProperty(receiver, 'value', {get: observeStack, set: storeStack});
function invokeRead() { return receiver.read(); }
function invokeWrite() { return receiver.write(); }
function invokeDirect() { return receiver.directError(); }
function firstFrames(error) { return error.stack.split('\n').slice(0, 4).join('\n'); }
for (var n = 0; n < 1500; n++) {
    invokeRead();
    try { invokeWrite(); } catch (error) {}
    try { invokeDirect(); } catch (error) {}
}
var result = firstFrames(invokeRead());
throwing = true;
try { invokeRead(); } catch (error) { result += '|' + firstFrames(error); }
try { invokeWrite(); } catch (error) { result += '|' + firstFrames(error); }
try { invokeDirect(); } catch (error) {
    result += '|' + firstFrames(error).replace(/^[^\n]*/, 'TypeError');
}
result;"#;
    let mut expected = None;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = evaluate(&mut engine, source);
        match &expected {
            None => expected = Some(result),
            Some(expected) => assert_eq!(&result, expected, "{tier:?}"),
        }
        if tier == Tier::Jit {
            for name in ["invokeRead", "invokeWrite", "invokeDirect"] {
                require_hot_native(&mut engine, name, true);
            }
        }
    }
}
