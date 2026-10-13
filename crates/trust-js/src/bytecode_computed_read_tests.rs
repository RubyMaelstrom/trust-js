//! Compact computed GetValue boundary. Local ECMA-262 e28783d5: sec-getvalue,
//! sec-evaluate-property-access-with-expression-key and sec-evaluatecall.
//! The helper must commit a compact owner, not leave a wide completion in native scratch.

use super::*;
use crate::{jit::JitCtx, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("computed-read fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn binding(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env.clone();
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("fixture binding")
}

/// Enter the real checked helper with an owning prefix below its operands. No
/// artificial initialized values exist above the returned SP; only that exact
/// canonical prefix is decoded/dropped, just as the native unwinder does.
fn invoke(
    engine: &mut Engine,
    op: Op,
    object: Value,
    key: Value,
    prefix: Value,
) -> (Vec<Value>, Option<Abrupt>) {
    let parsed =
        crate::parser::parse_script("function read(object,key){return object[key];}", false)
            .ok()
            .unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &parsed[0] else {
        unreachable!()
    };
    let mut chunk = compile(function).unwrap();
    Rc::get_mut(&mut chunk).unwrap().ops[0] = op;
    let local = matches!(op, Op::GetElemLocal(_));
    let mut slots = [
        PackedValue::pack(if local {
            object.clone()
        } else {
            Value::Undefined
        }),
        PackedValue::pack(Value::Undefined),
    ];
    let mut storage: [std::mem::MaybeUninit<PackedValue>; 4] =
        std::array::from_fn(|_| std::mem::MaybeUninit::uninit());
    let base = storage.as_mut_ptr().cast::<PackedValue>();
    let count = if local { 2 } else { 3 };
    unsafe {
        base.write(PackedValue::pack(prefix));
        if !local {
            base.add(1).write(PackedValue::pack(object));
        }
        base.add(count - 1).write(PackedValue::pack(key));
    }
    let i = &mut *engine.interp;
    let env = i.global_env.clone();
    let mut ctx = JitCtx {
        helpers: i.jit_helpers.as_ptr(),
        stack_base: base,
        final_sp: base,
        slots: slots.as_mut_ptr(),
        inline_ic_safe: &i.inline_ic_safe as *const _ as *const u8,
        env_raw: Rc::as_ptr(&env).cast(),
        this_raw: std::ptr::null(),
        global_body: std::ptr::null(),
        genv: Rc::as_ptr(&env) as usize,
        interp: i,
        chunk: &*chunk,
        this_val: Value::Undefined,
        n_slots: slots.len(),
        handlers: Vec::new(),
        handler_floor: 0,
        code_base: std::ptr::null(),
        pc_offsets: std::ptr::null(),
        error: None,
        ret: PackedValue::pack(Value::Undefined),
        env_parent_raw: std::ptr::null(),
        opstat_enabled: false,
        callstat_enabled: false,
        inline_recompile_at: inline_recompile_at(),
        live_objects: crate::value::live_objects_ptr(&i.gc_heap),
        activation: None,
        resume_activation: std::ptr::null_mut(),
        resume_pc: 0,
        resume_step: None,
        references_raw: std::ptr::null_mut(),
    };
    ctx.this_raw = &ctx.this_val;
    let result = unsafe { jit_get_element(&mut ctx, 0, base.add(count)) };
    assert_eq!(result.flag != 0, ctx.error.is_some());
    let live = unsafe { result.sp.offset_from(base) as usize };
    assert_eq!(
        live,
        if result.flag != 0 {
            1
        } else if matches!(op, Op::GetMethodElem) {
            3
        } else {
            2
        },
        "wrong canonical stack extent for {op:?}"
    );
    let values = (0..live)
        .map(|index| unsafe { base.add(index).read().into_value() })
        .collect();
    (values, ctx.error.take())
}

#[test]
fn computed_read_helper_canonical_stack_all_tags_and_last_owners() {
    let mut engine = Engine::new();
    evaluate(&mut engine, "var token=Symbol('computed');");
    let values = [
        Value::Undefined,
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Num(-0.0),
        Value::Num(f64::from_bits(0xffff_ffff_ffff_ffff)),
        Value::Num(f64::INFINITY),
        Value::Num(-1.25),
        Value::Str("owned".into()),
        binding(&mut engine, "token"),
        Value::Obj(crate::value::Object::new(None)),
        Value::BigInt(crate::bigint::JsBigInt::from_i128(i128::MAX)),
    ];
    for op in [Op::GetElem, Op::GetElemLocal(0), Op::GetMethodElem] {
        for value in &values {
            let object = crate::value::Object::new(None);
            object
                .borrow_mut()
                .props
                .insert("value", crate::value::Property::plain(value.clone()));
            let weak = Rc::downgrade(&object);
            let (result, error) = invoke(
                &mut engine,
                op,
                Value::Obj(object),
                Value::Str("value".into()),
                Value::Str("prefix".into()),
            );
            assert!(error.is_none());
            assert!(matches!(&result[0], Value::Str(s) if s.as_str()=="prefix"));
            let actual = result.last().unwrap();
            engine.interp.gc_collect();
            match (value, actual) {
                (Value::Num(a), Value::Num(b)) => {
                    assert!(a.is_nan() && b.is_nan() || a.to_bits() == b.to_bits())
                }
                _ => assert!(engine.interp.strict_equals(value, actual)),
            }
            assert_eq!(
                weak.upgrade().is_some(),
                matches!(op, Op::GetMethodElem),
                "only a method Reference retains its consumed receiver"
            );
            drop(result);
            assert!(weak.upgrade().is_none());
        }
    }
}

#[test]
fn computed_read_helper_errors_consume_operands_not_prefix_or_local() {
    let mut engine = Engine::new();
    evaluate(
        &mut engine,
        r#"
        var calls=0, thrown={identity:1}, key={ [Symbol.toPrimitive](){calls++;throw thrown;} };
        var getter={get value(){calls++;throw thrown;}}, proxy=new Proxy({}, {
            get(target,key,receiver){calls++;throw thrown;}
        });
    "#,
    );
    for op in [Op::GetElem, Op::GetElemLocal(0), Op::GetMethodElem] {
        for (object, key, expected_calls, expected_error) in [
            (Value::Null, binding(&mut engine, "key"), 0, "TypeError"),
            (
                Value::Obj(crate::value::Object::new(None)),
                binding(&mut engine, "key"),
                1,
                "sentinel",
            ),
            (
                binding(&mut engine, "getter"),
                Value::Str("value".into()),
                1,
                "sentinel",
            ),
            (
                binding(&mut engine, "proxy"),
                Value::Str("value".into()),
                1,
                "sentinel",
            ),
        ] {
            evaluate(&mut engine, "calls=0;");
            let prefix = crate::value::Object::new(None);
            let (result, error) = invoke(&mut engine, op, object, key, Value::Obj(prefix.clone()));
            assert!(matches!(&result[0],Value::Obj(actual) if Rc::ptr_eq(actual,&prefix)));
            assert_eq!(evaluate(&mut engine, "calls"), expected_calls.to_string());
            let Some(Abrupt::Throw(error)) = error else {
                panic!("expected exact throw");
            };
            if expected_error == "sentinel" {
                let sentinel = binding(&mut engine, "thrown");
                assert!(engine.interp.strict_equals(&error, &sentinel));
            } else {
                let name = engine.interp.get_member(&error, "name").ok().unwrap();
                assert!(matches!(name,Value::Str(s) if s.as_str()==expected_error));
            }
        }
    }
    // The compiler fuses only never-TDZ locals; the helper still fails closed if
    // an empty canonical local reaches it, without coercing the consumed key.
    evaluate(&mut engine, "calls=0;");
    let key = binding(&mut engine, "key");
    let (_, error) = invoke(
        &mut engine,
        Op::GetElemLocal(0),
        Value::Empty,
        key,
        Value::Undefined,
    );
    assert!(matches!(error, Some(Abrupt::Throw(_))));
    assert_eq!(evaluate(&mut engine, "calls"), "0");
}

fn all_tiers(source: &str, expected: &str, required_shapes: &[usize], continuation: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.interp.def_method(
            &engine.interp.global,
            "collectComputedTest",
            0,
            |i, _, _| {
                i.gc_collect();
                Ok(Value::Undefined)
            },
        );
        TEST_JIT_GET_ELEM_HELPERS.with(|count| count.set([0; 3]));
        TEST_JIT_EXEC_ELEMENT_HELPERS.with(|count| count.set(0));
        crate::jit::TEST_NATIVE_SLICES.with(|count| count.set(0));
        assert_eq!(evaluate(&mut engine, source), expected, "{tier:?}");
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit {
            let counts = TEST_JIT_GET_ELEM_HELPERS.with(std::cell::Cell::get);
            for &shape in required_shapes {
                assert!(
                    counts[shape] > 0,
                    "missing actual helper shape {shape}: {counts:?}"
                );
            }
            assert_eq!(
                TEST_JIT_EXEC_ELEMENT_HELPERS.with(std::cell::Cell::get),
                0,
                "computed read unexpectedly entered generic opcode dispatch"
            );
            if continuation {
                assert!(crate::jit::TEST_NATIVE_SLICES.with(std::cell::Cell::get) > 0);
            }
        }
    }
}

#[test]
fn computed_read_native_shapes_keep_symbol_receiver_coercion_and_gc() {
    all_tiers(
        r#"
        var symbol=Symbol('key'),calls=0,coercions=0;
        var object={value:7,[symbol](){return this.value;}};
        var key={[Symbol.toPrimitive](){coercions++;collectComputedTest();return symbol;}};
        function stackRead(o,k){return Object(o)[k];}
        function localRead(o,k){return o[k];}
        function method(o,k){return o[k]();}
        var proxy=new Proxy(object,{get(target,key,receiver){calls++;collectComputedTest();return Reflect.get(target,key,receiver);}});
        var good=true;for(var i=0;i<12;i++){
            good=stackRead(proxy,key)===localRead(proxy,key)&&method(proxy,key)===7&&good;
        }
        good+'|'+coercions+'|'+calls;
    "#,
        "true|36|48",
        &[0, 1, 2],
        false,
    );
}

#[test]
fn computed_read_native_fused_numeric_misses_and_post_effect_types_do_not_replay() {
    all_tiers(
        r#"
        var reads=0,coercions=0,token={valueOf(){coercions++;collectComputedTest();return 4;}};
        var array={get 0(){reads++;collectComputedTest();return token;},get 1(){reads++;return 5;}};
        function keyed(a){var index=-1,sum=1;sum+=a[++index];sum+=a[++index];return sum;}
        function loop(a){var sum=0;for(var i=0;i<2;i++)sum+=a[i];return sum;}
        var good=true;for(var n=0;n<12;n++)good=keyed(array)===10&&loop(array)===9&&good;
        good+'|'+reads+'|'+coercions;
    "#,
        "true|48|24",
        &[1],
        false,
    );
}

#[test]
fn computed_read_native_abrupt_order_keeps_original_base_and_finally() {
    all_tiers(
        r#"
        var trace='',token={},key={ [Symbol.toPrimitive](){trace+='K';throw token;} };
        function read(o,k){try{return 3+o[k];}finally{trace+='F';}}
        function caught(o,k){try{read(o,k);return false;}catch(e){return e===token? 'token':e.name;}}
        var first=caught(null,key),second=caught({},key);
        var getter={get value(){trace+='G';collectComputedTest();throw token;}};
        var third=caught(getter,'value');
        var proxy=new Proxy({}, {get(){trace+='P';collectComputedTest();throw token;}});
        var fourth=caught(proxy,'value');
        function replace(){var base={value:7};var k={toString(){base={value:9};collectComputedTest();return 'value';}};
            return base[k]+base.value;}
        function tdz(){return before['value'];let before={value:1};}
        var fifth;try{tdz();}catch(e){fifth=e.name;}
        [first,second,third,fourth,trace,replace(),fifth].join('|');
    "#,
        "TypeError|token|token|token|FKFGFPF|16|ReferenceError",
        &[1],
        false,
    );
}

#[test]
fn computed_read_native_continuations_preserve_prefix_throw_finally_and_receiver() {
    all_tiers(
        r#"
        var trace='',token={},key={ [Symbol.toPrimitive](){trace+='K';collectComputedTest();return 'value';} };
        var object={get value(){trace+='G';collectComputedTest();return 7;}};
        function* read(o){try{return 3+(o[yield 'key']);}finally{trace+='F';}}
        var it=read(object),first=it.next().value,result=it.next(key).value;
        var bad={get value(){trace+='T';throw token;}};
        var next=read(bad);next.next();var same=false;try{next.next(key);}catch(e){same=e===token;}
        function* method(o){return o[yield 'method']();}
        var receiver={value:11,method(){return this.value;}};
        var call=method(receiver);call.next();var returned=call.next('method').value;
        first+'|'+result+'|'+same+'|'+returned+'|'+trace;
    "#,
        "key|10|true|11|KGFKTF",
        &[0, 2],
        true,
    );
}
