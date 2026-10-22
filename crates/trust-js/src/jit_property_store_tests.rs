//! ECMA-262 e28783d5: assignment returns the RHS after PutValue; live
//! OrdinarySet descriptors and owner destruction remain observable on misses.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("property store fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn assignment_results_retain_scalar_and_reference_owners_across_creation_and_gc() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function put(o,v){return o.field=v;}
            function create(v){var o={};return [o,put(o,v)];}
            var o={field:0},text='owned'.repeat(30),symbol=Symbol('field'),object={id:81};
            var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,text,symbol,object];
            var ok=true;
            for(var warm=0;warm<160;warm++)put(o,warm);
            for(var warm=0;warm<160;warm++)create(warm);
            for(var v of values){
                var returned=put(o,v),made=create(v);
                ok=ok&&Object.is(returned,v)&&Object.is(o.field,v);
                ok=ok&&Object.is(made[0].field,v)&&Object.is(made[1],v);
            }
            var big=1234567890123456789012345678901234567890n;
            ok=ok&&put(o,big)===big&&o.field===big;
            var alias={field:object};
            ok=ok&&put(alias,object)===object&&alias.field===object;
            ok=ok&&put(alias,alias)===alias&&alias.field===alias;
            ok=ok&&put(alias,alias)===alias;
            var rooted=put(o,{kept:91});o.field=null;
            ok
        "#
            ),
            "true",
            "{tier:?}"
        );
        engine.interp.gc_collect();
        assert_eq!(
            eval(
                &mut engine,
                "rooted.kept+':'+object.id+':'+(alias.field===alias)"
            ),
            "91:81:true",
            "{tier:?}"
        );
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn assignment_store_misses_preserve_descriptor_side_effects_and_expression_values() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function put(o,v){return o.field=v;}
            function strictPut(o,v){'use strict';return o.field=v;}
            var o={field:0},trace='',value={id:7},ok=true;
            for(var warm=0;warm<160;warm++){put(o,warm);strictPut(o,warm);}
            Object.defineProperty(o,'field',{set(v){trace+='s'+v.id;},configurable:true});
            ok=ok&&put(o,value)===value&&trace==='s7';
            Object.defineProperty(o,'field',{value:11,writable:false});
            ok=ok&&put(o,value)===value&&o.field===11;
            try{strictPut(o,value);ok=false;}catch(e){ok=ok&&e instanceof TypeError;}
            var proxy=new Proxy({field:0},{set(t,k,v){trace+='p'+v.id;return false;}});
            ok=ok&&put(proxy,value)===value;
            try{strictPut(proxy,value);ok=false;}catch(e){ok=ok&&e instanceof TypeError;}
            var inherited=Object.create({set field(v){trace+='i'+v.id;}});
            ok=ok&&put(inherited,value)===value&&!Object.hasOwn(inherited,'field');
            var sealed=Object.preventExtensions({});ok=ok&&put(sealed,value)===value;
            try{strictPut(sealed,value);ok=false;}catch(e){ok=ok&&e instanceof TypeError;}
            var throwing={set field(v){trace+='t';throw 81;}};
            try{put(throwing,value);ok=false;}catch(e){ok=ok&&e===81;}
            var a=[1,2,3];a.field=0;
            ok=ok&&put(a,value)===value&&a.length===3&&a[1]===2;
            function receiver(){trace+='r';return a;}
            function rhs(){trace+='v';return value;}
            ok=ok&&(receiver().field=rhs())===value;
            ok&&trace==='s7p7p7i7trv'
        "#
            ),
            "true",
            "{tier:?}"
        );
        engine.interp.gc_collect();
        assert_eq!(
            eval(&mut engine, "a.field===value&&o.field===11"),
            "true",
            "{tier:?}"
        );
    }
}

#[test]
fn rejecting_proxy_assignments_check_strictness_without_changing_reflect_set() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var calls=0,proxy=new Proxy({},{set(){calls++;return false;}}),key=Symbol('key');
            function named(){'use strict';return proxy.field=7;}
            function computed(k){'use strict';return proxy[k]=8;}
            class Base{}
            Object.setPrototypeOf(Base.prototype,proxy);
            class Derived extends Base{
                named(){return super.field=9;}
                computed(k){return super[k]=10;}
            }
            var derived=new Derived(),inherited=Object.create(proxy);
            function inheritedWrite(){'use strict';return inherited.field=11;}
            var rejected=0;
            for(var run of [named,()=>computed('field'),()=>computed(key),
                            ()=>derived.named(),()=>derived.computed(key),inheritedWrite]){
                try{run();}catch(e){if(e instanceof TypeError)rejected++;}
            }
            var reflected=Reflect.set(proxy,'field',11),sloppy=(proxy.field=12);
            [rejected,calls,reflected,sloppy].join(':')
        "#
            ),
            "6:8:false:12",
            "{tier:?}"
        );
    }
}

#[test]
fn assignment_store_retains_new_host_owners_and_releases_final_old_owners() {
    use crate::value::Value;
    use std::{cell::Cell, rc::Rc};
    struct Witness(Rc<Cell<usize>>);
    impl Drop for Witness {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let dropped = Rc::new(Cell::new(0usize));
        engine.interp.op_state().put(dropped.clone());
        engine
            .interp
            .def_method(&engine.interp.global, "makeStoreWitness", 0, |i, _, _| {
                let witness = Witness(i.host::<Rc<Cell<usize>>>().unwrap().clone());
                Ok(Value::Obj(i.make_native_closure(
                    "witness",
                    0,
                    Rc::new(move |_, _, _| {
                        let _keep = &witness;
                        Ok(Value::Undefined)
                    }),
                    crate::value::NativeCaptures::None,
                )))
            });
        // The destructor runs when the next collection finds the witness dead.
        let eval = |engine: &mut Engine, source: &str| {
            eval(engine, source);
            engine.interp.gc_collect();
        };
        eval(&mut engine, "function put(o,v){return o.field=v;}var o={field:0};for(var warm=0;warm<160;warm++)put(o,warm);");
        eval(
            &mut engine,
            "var result=put(o,makeStoreWitness());o.field=null;true;",
        );
        assert_eq!(dropped.get(), 0, "the expression result remains an owner");
        eval(&mut engine, "result=null;");
        assert_eq!(dropped.get(), 1, "the moved result releases exactly once");
        eval(&mut engine, "put(o,makeStoreWitness());true;");
        assert_eq!(dropped.get(), 1, "the property owns the discarded result");
        eval(&mut engine, "put(o,5);true;");
        assert_eq!(dropped.get(), 2, "a final old owner uses its destructor");
        eval(
            &mut engine,
            "var fresh={},kept=put(fresh,makeStoreWitness());fresh.field=null;true;",
        );
        assert_eq!(dropped.get(), 2, "creation keeps the RHS owner");
        eval(&mut engine, "kept=null;");
        assert_eq!(dropped.get(), 3, "creation releases exactly once");
    }
}

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn warm_assignment_expressions_use_native_stores_without_losing_rhs_owners() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function put(o,v){return o.field=v;}
        var o={field:0},text='native'.repeat(20),symbol=Symbol('native'),object={id:23};
        for(var warm=0;warm<160;warm++)put(o,warm);
    "#,
    );
    let before = crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var ok=true,values=[1,-0,NaN,undefined,null,true,text,symbol,object];
        for(var n=0;n<30;n++)for(var v of values)ok=ok&&Object.is(put(o,v),v)&&Object.is(o.field,v);
        ok
    "#
        ),
        "true"
    );
    assert_eq!(
        crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get),
        before,
        "warm own assignment expressions retain the native caller and store"
    );
    engine.interp.gc_collect();
    assert_eq!(eval(&mut engine, "o.field.id"), "23");
}

/// Every native entry captures its own heap's shape directory. Alternating Engines must keep
/// filling and committing creation ICs through the directory of the current activation.
#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn creation_ics_use_each_active_heaps_shape_directory() {
    const CHILD: &str = "TRUST_JS_TEST_CREATION_DIRECTORY_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::property_store_tests::creation_ics_use_each_active_heaps_shape_directory",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_INLINE_AT", "4294967295")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    fn create_chunk(engine: &mut Engine) -> std::rc::Rc<crate::bytecode::Chunk> {
        engine.interp.activate_gc_heap();
        let env = engine.interp.global_env;
        let value = engine
            .interp
            .get_var("create", &env)
            .ok()
            .expect("create binding");
        let object = value.as_obj().expect("create function").borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("create is a user function")
        };
        let runtime = user.func.runtime();
        runtime
            .execution_code()
            .and_then(Option::as_ref)
            .expect("create compiled")
            .clone()
    }
    let mut first = Engine::new();
    let mut second = Engine::new();
    for (engine, key) in [(&mut first, "alpha"), (&mut second, "beta")] {
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            engine,
            &format!(
                "function create(v){{var o={{}};o.{key}=v;return o;}}\
             for(var warm=0;warm<400;warm++)create(warm);"
            ),
        );
    }
    for _ in 0..4 {
        for (engine, key) in [(&mut first, "alpha"), (&mut second, "beta")] {
            let chunk = create_chunk(engine);
            assert!(
                chunk.jit.get().flatten().is_some(),
                "create has native code"
            );
            let runs = chunk.jit_runs.get();
            let mut stores = chunk.jit_ops().iter().filter_map(|op| match *op {
                crate::bytecode::Op::SetProp(_, cache)
                | crate::bytecode::Op::SetPropDrop(_, cache)
                | crate::bytecode::Op::SetPropThisDrop(_, cache)
                | crate::bytecode::Op::SetPropLocalDrop(_, _, cache) => Some(cache),
                _ => None,
            });
            let cache = stores.next().expect("create has one property store");
            assert!(stores.next().is_none());
            assert_eq!(
                chunk
                    .jit_cache_preferred(cache)
                    .expect("monomorphic creation cache")
                    .depth,
                crate::bytecode::IC_CREATE,
                "creation depth0xFD cannot enter numeric ChainOp::StoreProp, which requires depth0"
            );
            let epoch = crate::value::proto_epoch();
            let before = (
                crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get),
                crate::bytecode::TEST_JIT_EXEC_PROPERTY_HELPERS.with(std::cell::Cell::get),
            );
            assert_eq!(
                eval(
                    engine,
                    &format!(
                        "for(var n=0;n<40;n++){{if(create(n).{key}!==n)throw Error('field');}}\
                 create(123).{key}"
                    )
                ),
                "123"
            );
            assert!(
                chunk.jit_runs.get() > runs,
                "create entered its native body"
            );
            assert_eq!(
                crate::value::proto_epoch(),
                epoch,
                "isolated creation proof must have a stable prototype epoch"
            );
            assert_eq!(
                (
                    crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get),
                    crate::bytecode::TEST_JIT_EXEC_PROPERTY_HELPERS.with(std::cell::Cell::get),
                ),
                before,
                "a warm creation commits through the active directory without a helper"
            );
        }
    }
}
