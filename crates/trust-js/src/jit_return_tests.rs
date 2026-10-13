//! Packed native return ownership. ECMA-262 snapshot e28783d5: Completion Records,
//! Ordinary [[Call]]/[[Construct]], ReturnStatement, TryStatement and IteratorClose.
//! Representation changes must not merge return/throw/bare-return completion kinds.

#![cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]

use super::*;
use crate::{bytecode::Tier, Completion, Engine};

fn compile_return(engine: &mut Engine, source: &str) -> (Rc<Chunk>, JitCode) {
    let statements = crate::parser::parse_script(source, false).ok().unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("return fixture function")
    };
    let chunk = crate::bytecode::compile(function).expect("return fixture bytecode");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = super::compile(&chunk, &layout, &ilayout).expect("return fixture native code");
    (chunk, code)
}

/// Deliberately seed the private result slot to exercise BOTH native vacancy branches.
/// These fixtures have no environment operations or calls; the real native prologue,
/// local clone, return template, epilogue and checked fallback all execute unchanged.
fn enter(
    engine: &mut Engine,
    chunk: &Chunk,
    code: &JitCode,
    value: Value,
    previous: Value,
) -> (Value, usize) {
    let i: &mut Interp = &mut engine.interp;
    let env = i.global_env.clone();
    let (_, count) = chunk.jit_frame();
    let mut slots: Vec<PackedValue> = (0..count)
        .map(|_| PackedValue::pack(Value::Undefined))
        .collect();
    if count != 0 {
        slots[0] = PackedValue::pack(value);
    }
    let mut stack: Vec<std::mem::MaybeUninit<PackedValue>> =
        Vec::with_capacity(code.max_stack.max(1));
    let base = stack.as_mut_ptr().cast::<PackedValue>();
    let mut ctx = JitCtx {
        helpers: i.jit_helpers.as_ptr(),
        stack_base: base,
        final_sp: base,
        slots: slots.as_mut_ptr(),
        inline_ic_safe: &i.inline_ic_safe as *const std::cell::Cell<bool> as *const u8,
        env_raw: Rc::as_ptr(&env) as *const u8,
        this_raw: std::ptr::null(),
        global_body: super::jit_global_body(i, code),
        genv: Rc::as_ptr(&env) as usize,
        interp: i as *mut Interp,
        chunk,
        this_val: Value::Undefined,
        n_slots: count,
        handlers: Vec::new(),
        handler_floor: 0,
        code_base: code.mem,
        pc_offsets: code.pc_offsets.as_ptr(),
        error: None,
        ret: PackedValue::pack(previous),
        env_parent_raw: super::jit_env_parent_raw(&env),
        opstat_enabled: false,
        callstat_enabled: false,
        inline_recompile_at: crate::bytecode::inline_recompile_at(),
        live_objects: crate::value::live_objects_ptr(&i.gc_heap),
        activation: None,
        resume_activation: std::ptr::null_mut(),
        references_raw: std::ptr::null_mut(),
        resume_pc: 0,
        resume_step: None,
    };
    ctx.this_raw = &ctx.this_val;
    crate::bytecode::TEST_JIT_RETURN_HELPERS.with(|calls| calls.set(0));
    let entry: extern "C" fn(*mut JitCtx) -> u64 = unsafe { std::mem::transmute(code.mem) };
    let status = entry(&mut ctx);
    let calls = crate::bytecode::TEST_JIT_RETURN_HELPERS.with(|calls| calls.get());
    // Native code owns only [base, final_sp); everything above it has been moved.
    unsafe {
        let mut operand = base;
        while operand < ctx.final_sp {
            std::ptr::drop_in_place(operand);
            operand = operand.add(1);
        }
    }
    assert_eq!(status, 1, "native fixture threw");
    assert_eq!(ctx.final_sp, base, "returned operand remains live");
    drop(slots);
    // The result remains an owning GC root after the callee's local owners are
    // destroyed, before widening at the host boundary. This also covers cycles.
    i.gc_collect();
    (ctx.take_ret().into_value(), calls)
}

#[test]
fn packed_native_return_all_tags_use_empty_slot_without_helper() {
    let mut engine = Engine::new();
    engine
        .eval("var returnSymbol=Symbol('return');", false)
        .unwrap();
    let env = engine.interp.global_env.clone();
    let symbol = engine.interp.get_var("returnSymbol", &env).ok().unwrap();
    let values = [
        Value::Undefined,
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Num(0.0),
        Value::Num(-0.0),
        Value::Num(1.25),
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Num(f64::NEG_INFINITY),
        Value::Str("owned return".into()),
        symbol,
        Value::Obj(crate::value::Object::new(None)),
        Value::BigInt(crate::bigint::JsBigInt::from_i128(i128::MAX)),
    ];
    let (chunk, code) = compile_return(&mut engine, "function pass(value){return value;}");
    for value in values {
        let (result, helpers) = enter(&mut engine, &chunk, &code, value.clone(), Value::Undefined);
        assert_eq!(helpers, 0, "empty native return called the checked helper");
        match (&value, &result) {
            (Value::Num(expected), Value::Num(actual)) => {
                assert!(
                    expected.is_nan() && actual.is_nan() || expected.to_bits() == actual.to_bits()
                );
            }
            _ => assert!(engine.interp.strict_equals(&value, &result)),
        }
    }
}

#[test]
fn packed_native_return_occupied_alias_and_bare_cleanup_are_exact() {
    let mut engine = Engine::new();
    for source in [
        "function pass(value){return value;}",
        "function pass(value){return;}",
        "function pass(value){}",
    ] {
        let (chunk, code) = compile_return(&mut engine, source);
        for occupied in [false, true] {
            let object = crate::value::Object::new(None);
            let weak = Rc::downgrade(&object);
            let previous = if occupied {
                Value::Obj(object.clone())
            } else {
                Value::Undefined
            };
            let (result, helpers) = enter(&mut engine, &chunk, &code, Value::Obj(object), previous);
            assert_eq!(helpers, usize::from(occupied), "{source}");
            if source.contains("return value") {
                assert!(matches!(result, Value::Obj(_)));
                assert_eq!(weak.strong_count(), 1);
            } else {
                assert!(matches!(result, Value::Undefined));
                assert_eq!(weak.strong_count(), 0);
            }
            drop(result);
            assert_eq!(weak.strong_count(), 0);
        }
    }
}

#[test]
fn packed_native_return_sole_cyclic_result_survives_teardown_collection() {
    let mut engine = Engine::new();
    let (chunk, code) = compile_return(&mut engine, "function pass(value){return value;}");
    let object = crate::value::Object::new(Some(engine.interp.object_proto.clone()));
    let weak = Rc::downgrade(&object);
    object.borrow_mut().props.insert(
        "self",
        crate::value::Property::plain(Value::Obj(object.clone())),
    );
    let (result, helpers) = enter(
        &mut engine,
        &chunk,
        &code,
        Value::Obj(object),
        Value::Undefined,
    );
    assert_eq!(helpers, 0);
    assert_eq!(
        weak.strong_count(),
        2,
        "one return owner plus its self edge"
    );
    let returned = result.as_obj().unwrap();
    let self_value = returned.borrow().props.get("self").unwrap().value();
    assert!(matches!(&self_value, Value::Obj(value) if Rc::ptr_eq(value, returned)));
    drop(self_value);
    drop(result);
    engine.interp.gc_collect();
    assert_eq!(
        weak.strong_count(),
        0,
        "a dropped result must not leave a hidden native owner"
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn packed_native_return_direct_call_moves_and_clears_the_entire_word() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
        .eval(
            r#"
        function returned(value){return value;}
        function invoked(value){var result=returned(value);return result;}
        returned(0);invoked(0);
    "#,
            false,
        )
        .unwrap();
    // Isolate the ordinary direct-call ABI from AST splicing. Both chunks keep
    // their real native entry and live inline caches; only the test's retry
    // scheduler is settled so this exact boundary cannot disappear by inlining.
    let env = engine.interp.global_env.clone();
    for name in ["returned", "invoked"] {
        let function = engine.interp.get_var(name, &env).ok().unwrap();
        let object = function.as_obj().unwrap().borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("user function")
        };
        let chunk = user.func.code.get().and_then(Option::as_ref).unwrap();
        assert!(chunk.jit.get().flatten().is_some());
        chunk.inline_attempted.set(true);
        chunk.inline_retry_at.set(0);
    }
    super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    let result = engine
        .eval(
            r#"
        var resultSymbol=Symbol('packed'), resultObject={}, okay=true;
        var results=[resultObject,resultSymbol,'text',12345678901234567890n,
                     undefined,null,false,true,0,-0,NaN,Infinity,-Infinity];
        for(var round=0;round<5;round++)for(var k=0;k<results.length;k++)
            okay=okay&&Object.is(invoked(results[k]),results[k]);
        okay;
    "#,
            false,
        )
        .unwrap();
    assert!(matches!(result, Completion::Value(value) if value == "true"));
    assert!(
        super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get()) >= 65,
        "the result must traverse the actual shared-context native return tail"
    );
}

fn all_tiers(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("return fixture parses") {
            Completion::Value(actual) => assert_eq!(actual, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn packed_native_return_finally_iterator_close_and_replacement_order() {
    // TryStatement replaces a pending completion only with an abrupt finalizer;
    // IteratorClose preserves an incoming throw over a failing return() method.
    all_tiers(
        r#"
        var trace=[], original={}, replacement={};
        function nested(mode){try{try{return original;}finally{trace.push('inner');}}
            finally{trace.push('outer');if(mode===1)return replacement;if(mode===2)throw replacement;}}
        function iteration(mode){var iterable={[Symbol.iterator](){return this;},
            next(){return {value:original,done:false};},return(){trace.push('close');throw replacement;}};
            for(var value of iterable){if(mode)throw original;return value;}}
        var good=true;
        for(var i=0;i<130;i++){
            good=good&&nested(0)===original&&nested(1)===replacement;
            try{nested(2);good=false;}catch(e){good=good&&e===replacement;}
            try{iteration(0);good=false;}catch(e){good=good&&e===replacement;}
            try{iteration(1);good=false;}catch(e){good=good&&e===original;}
        }
        good+':'+trace.length+':'+trace.slice(0,8).join(',');
    "#,
        "true:1040:inner,outer,inner,outer,inner,outer,close,close",
    );
}

#[test]
fn packed_native_return_constructors_and_tail_throw_keep_result_owners() {
    all_tiers(
        r#"
        var object={id:1}, symbol=Symbol('result'), good=true;
        function base(value){this.seen=true;return value;}
        class derived extends base{constructor(value){super(0);return value;}}
        function identity(value){'use strict';return value;}
        function throwing(value){'use strict';throw value;}
        function tail(value,fail){'use strict';if(fail)return throwing(value);return identity(value);}
        function recover(value){try{return tail(value,true);}catch(e){return e;}}
        function chain(value){var a=identity(value),b=identity(a);return b;}
        var values=[undefined,null,false,-0,NaN,'owned',symbol,object,12345678901234567890n];
        for(var i=0;i<140;i++)for(var k=0;k<values.length;k++){
            var value=values[k];
            good=good&&Object.is(chain(value),value)&&Object.is(tail(value,false),value)&&Object.is(recover(value),value);
            var result=new base(value);good=good&&(value===object?result===object:result.seen);
            if(value===undefined||value===object){result=new derived(value);good=good&&(value===object?result===object:result.seen);}
            else{try{new derived(value);good=false;}catch(e){good=good&&e instanceof TypeError;}}
        }
        good;
    "#,
        "true",
    );
}

/// A callee whose body needs the checked one-operation bridge (block scopes, array and object
/// destructuring, object-literal methods, template literals) materializes a native activation
/// on its own frame record. Such callees take the direct JIT→JIT sequence like any other: the
/// generated finish stub routes a materialized activation to `jit_direct_finish`, whose record
/// release drops it, for normal and abrupt completions alike. Expected output from Node.
#[cfg(target_arch = "aarch64")]
#[test]
fn direct_calls_enter_callees_that_materialize_native_activations() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
        .eval(
            r#""use strict";
        function shaped(n, tag) {
          const pair = [n, n + 1];
          const [a, b] = pair;
          let total = 0;
          for (let k = 0; k < 3; k++) { const f = () => k * 2; total += f() + a; }
          const o = { tag, b, m() { return this.tag + ":" + this.b; } };
          const { tag: t, ...rest } = { tag: upper(tag), extra: n };
          if (n === 13) throw new RangeError("boom" + rest.extra);
          return total + "|" + o.m() + "|" + t + "|" + rest.extra + "|" + `${a}-${b}`;
        }
        function upper(x) { return x.toUpperCase(); }
        shaped(0, "warm"); upper("warm");
    "#,
            false,
        )
        .unwrap();
    let env = engine.interp.global_env.clone();
    for name in ["shaped", "upper"] {
        let function = engine.interp.get_var(name, &env).ok().unwrap();
        let object = function.as_obj().unwrap().borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("user function")
        };
        let chunk = user.func.code.get().and_then(Option::as_ref).unwrap();
        assert!(chunk.jit.get().flatten().is_some());
        assert_eq!(chunk.jit_needs_activation_state(), name == "shaped");
        chunk.inline_attempted.set(true);
        chunk.inline_retry_at.set(0);
    }
    super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    let result = engine
        .eval(
            r#"
        var out = [], errors = 0;
        for (var i = 0; i < 200; i++) {
          try { out.push(shaped(i, "x" + (i % 3))); }
          catch (e) { errors++; out.push(e.name + ":" + e.message); }
        }
        [out.length, errors, out[0], out[13], out[199]].join(";")
    "#,
            false,
        )
        .unwrap();
    match result {
        Completion::Value(value) => assert_eq!(
            value,
            "200;1;6|x0:1|X0|0|0-1;RangeError:boom13;603|x1:200|X1|199|199-200"
        ),
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    // 199 normal returns from `shaped` plus the 200 `upper` returns it makes directly, less
    // the first call at each site, which fills its cache through the helper. Without direct
    // entry into `shaped`, only the `upper` returns would count.
    let direct = super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get());
    assert!(
        direct >= 390,
        "bridge-using callees must take the direct sequence ({direct} direct returns)"
    );
    engine.interp.gc_collect();
    assert_eq!(
        engine
            .interp
            .frame_pool
            .iter()
            .filter(|record| unsafe { record.as_ref().ctx.activation.is_some() })
            .count(),
        0,
        "a pooled record must not retain a materialized activation"
    );
}

/// A direct method call whose receiver is the only reference (`new Target(k).to(k)`) releases
/// `this` through the finish stub's full-drop path, which must still vacate the record's
/// binding. A pooled record left naming the destroyed receiver was released again by its next
/// user, corrupting the heap (route-recognizer's `match(path).to(handler)` on Discourse).
#[cfg(target_arch = "aarch64")]
#[test]
fn direct_finish_vacates_a_last_reference_receiver() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let source = r#"
        function Target(p) { this.p = p; }
        Target.prototype.to = function (x) { return this.p + x; };
        function routes(n) {
          var s = 0;
          for (var k = 0; k < n; k++) s += new Target(k).to(k);
          return s;
        }
        routes(3);
    "#;
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    match engine.eval("routes(500)", false).expect("call parses") {
        Completion::Value(value) => assert_eq!(value, "249500"),
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    let direct = super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get());
    assert!(
        direct >= 350,
        "method calls must take the direct sequence ({direct})"
    );
    assert!(!engine.interp.frame_pool.is_empty());
    assert_eq!(
        engine
            .interp
            .frame_pool
            .iter()
            .filter(|record| unsafe { !matches!(record.as_ref().ctx.this_val, Value::Undefined) })
            .count(),
        0,
        "a pooled record must not retain a `this` binding"
    );
}

/// ECMA-262 OrdinaryFunctionCreate gives each evaluation of a function expression a new closure
/// with its own [[Environment]], while all of them share the function's code. A call site that
/// sees a fresh closure of an already-cached function takes the direct JIT→JIT sequence (the
/// code-keyed probe) and runs the LIVE closure: each call observes its own captured `k`, arrows
/// keep their lexical `this`, method calls bind their receiver, and a closure created under a
/// `with` statement still resolves through the object environment. Expected output from Node.
#[cfg(target_arch = "aarch64")]
#[test]
fn recreated_closures_take_direct_calls_with_their_own_environment() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let source = r#"
        function make(k) { return function (x) { return x * 3 + k; }; }
        function makeArrow(k) { return (x) => x - k; }
        function makeThis(k) { return function (x) { return this.base + x + k; }; }
        var withObj = { y: 1000 };
        var underWith;
        with (withObj) { underWith = function (x) { return x + y; }; }
        function run(n) {
          var s = 0, t = 0, u = 0, w = 0;
          var holder = { base: 7 };
          for (var r = 0; r < n; r++) {
            var f = make(r), g = makeArrow(r), h = makeThis(r);
            holder.m = h;
            for (var i = 0; i < 10; i++) {
              s = (s + f(i)) | 0;
              t = (t + g(i)) | 0;
              u = (u + holder.m(i)) | 0;
            }
          }
          for (var i = 0; i < 50; i++) { withObj.y = i; w = (w + underWith(i)) | 0; }
          return [s, t, u, w].join(",");
        }
        run(3);
    "#;
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    match engine.eval("run(500)", false).expect("call parses") {
        Completion::Value(value) => assert_eq!(value, "1315000,-1225000,1305000,2450"),
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    // 15000 closure calls, each closure fresh every outer iteration. Only each callee's first
    // ~100 runs (before its one-shot recompile settles) and the cache-filling calls take the
    // helper. The factories capture `k` in an activation and are not direct entries. Without
    // the code-keyed probe every fresh closure misses the identity ways (no direct returns).
    let direct = super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get());
    assert!(
        direct >= 14_000,
        "fresh closures of a cached function must take the direct sequence ({direct})"
    );
}
