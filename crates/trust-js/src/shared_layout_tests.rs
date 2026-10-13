//! Production shared-key storage. Normative ECMA-262 snapshot e28783d5, spec.html:
//! OrdinaryGetOwnProperty 13212; ValidateAndApplyPropertyDescriptor 13286;
//! OrdinarySetWithOwnDescriptor 13449; OrdinaryDelete 13501; OrdinaryOwnPropertyKeys 13531.

use crate::bytecode::Tier;
use crate::value::{Callable, Gc, Property, Props, Value};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("layout fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env.clone();
    let Value::Obj(object) = engine
        .interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing {name}"))
    else {
        panic!("{name} is not an object")
    };
    object
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "1+2"), "3");
    }
}

/// Creation ICs intentionally share a process-global prototype epoch. Other Rust tests may
/// invalidate it between warming and the measured store, which correctly takes a helper and
/// says nothing about this test's native path. Accept only a stable-epoch trial; retain the
/// exact helper-count assertion and fail explicitly if contention never permits one.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn measured_creation_write(engine: &mut Engine, target: &Gc, fresh: impl Fn() -> Props) -> usize {
    for _ in 0..256 {
        let epoch = crate::value::proto_epoch();
        target.borrow_mut().props = fresh();
        eval(engine, "write()");
        target.borrow_mut().props = fresh();
        crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(|count| count.set(0));
        eval(engine, "write()");
        let helpers = crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(|count| count.get());
        if crate::value::proto_epoch() == epoch {
            return helpers;
        }
        std::thread::yield_now();
    }
    panic!("prototype epoch changed during all 256 native creation trials");
}

#[test]
fn templates_share_keys_but_not_values_or_descriptors() {
    let mut template = Props::new();
    template.insert("x", Property::plain(Value::Undefined));
    template.insert("y", Property::plain(Value::Undefined));
    let layout = template.shared_layout().unwrap().clone();
    let key_owners = Rc::strong_count(&layout[0]);
    let mut a = template.instantiate_plain([Value::Num(1.0), Value::Num(2.0)].into_iter());
    let b = template.instantiate_plain([Value::Num(3.0), Value::Num(4.0)].into_iter());
    assert!(Rc::ptr_eq(
        a.shared_layout().unwrap(),
        b.shared_layout().unwrap()
    ));
    assert_eq!(
        Rc::strong_count(&layout[0]),
        key_owners,
        "instances must not clone individual keys"
    );
    assert_eq!(
        a.retained_requested_storage_bytes(),
        (2 * std::mem::size_of::<Property>(), true)
    );
    a.get_mut("x").unwrap().set_writable(false);
    assert!(b.get("x").unwrap().writable());
    assert!(matches!(b.get("x").unwrap().value(), Value::Num(3.0)));
    a.remove("x");
    a.insert("z", Property::plain(Value::Num(5.0)));
    assert_eq!(
        a.ordered_keys().iter().map(|k| &**k).collect::<Vec<_>>(),
        ["y", "z"]
    );
    assert_eq!(
        b.ordered_keys().iter().map(|k| &**k).collect::<Vec<_>>(),
        ["x", "y"]
    );
    assert!(!Rc::ptr_eq(a.shared_layout().unwrap(), &layout));
}

#[test]
fn ordinary_shared_index_respects_live_prefix_and_mutation() {
    let mut template = Props::new();
    for i in 0..12 {
        template.insert(format!("field{i}"), Property::plain(Value::Num(i as f64)));
    }
    let layout = template.shared_layout().unwrap().clone();
    assert!(layout.has_index());

    let mut short = Props::with_layout(layout.len(), Some(layout.clone()));
    for (slot, key) in layout.iter().take(9).enumerate() {
        short.append_initialized_field(key, Property::plain(Value::Num(slot as f64)));
    }
    assert!(short.contains("field8"));
    assert!(
        !short.contains("field9"),
        "predictions are not own properties"
    );
    short.append_initialized_field(&layout[9], Property::plain(Value::Num(9.0)));
    assert_eq!(short.get("field9").unwrap().value().as_num_opt(), Some(9.0));

    let mut other = Props::with_layout(layout.len(), Some(layout.clone()));
    for (slot, key) in layout.iter().enumerate() {
        other.append_initialized_field(key, Property::plain(Value::Num((100 + slot) as f64)));
    }
    assert!(Rc::ptr_eq(
        short.shared_layout().unwrap(),
        other.shared_layout().unwrap()
    ));
    assert_eq!(
        short.retained_requested_storage_bytes(),
        (12 * std::mem::size_of::<Property>(), true)
    );

    short.remove("field2");
    short.insert("field2", Property::plain(Value::Num(222.0)));
    assert_eq!(
        short.get("field2").unwrap().value().as_num_opt(),
        Some(222.0)
    );
    assert_eq!(
        other.get("field2").unwrap().value().as_num_opt(),
        Some(102.0)
    );
    assert!(!Rc::ptr_eq(
        short.shared_layout().unwrap(),
        other.shared_layout().unwrap()
    ));
    assert!(short.shared_layout().unwrap().has_index());
    assert_eq!(
        short.retained_requested_storage_bytes(),
        (12 * std::mem::size_of::<Property>(), true)
    );

    let private = crate::value::new_property_layout(
        (0..20).map(|i| Rc::from(format!("private{i}"))).collect(),
    );
    assert!(
        !private.has_index(),
        "long/private layouts use the instance table"
    );
    let mut long = Props::with_layout(private.len(), Some(private.clone()));
    for (slot, key) in private.iter().enumerate() {
        long.append_initialized_field(key, Property::plain(Value::Num(slot as f64)));
    }
    assert_eq!(
        long.get("private8").unwrap().value().as_num_opt(),
        Some(8.0)
    );
    assert_eq!(
        long.get("private19").unwrap().value().as_num_opt(),
        Some(19.0)
    );
}

#[test]
fn javascript_property_index_sharing_preserves_order_and_descriptors_in_all_tiers() {
    check(
        r#"
        function make(seed) {
            var object = {};
            for (var i = 0; i < 12; i++) object["field" + i] = seed + i;
            return object;
        }
        var a = make(1), b = make(100);
        delete a.field2;
        a.field2 = 222;
        Object.defineProperty(a, "field8", { writable: false });
        var rejected = false;
        try { a.field8 = 999; } catch (error) { rejected = true; }
        [a.field0, a.field2, a.field8, b.field2, rejected,
         Object.keys(a).join(","), Object.keys(b).join(",")].join("|");
        "#,
        "1|222|9|102|false|field0,field1,field3,field4,field5,field6,field7,field8,field9,field10,field11,field2|field0,field1,field2,field3,field4,field5,field6,field7,field8,field9,field10,field11",
    );
}

#[test]
fn ordinary_dynamic_and_json_objects_share_keys_across_creation_sites() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            function make(x){var o={};o['v'+(x&1)]=x;o.next=null;return o}
            var a=make(2), b=make(4);
            var c=JSON.parse('{"v0":6,"next":null}');
            var d=JSON.parse('{"v0":8,"next":null}');
        "#,
        );
        let a = object(&mut engine, "a");
        for name in ["b", "c", "d"] {
            let other = object(&mut engine, name);
            assert!(
                Rc::ptr_eq(
                    a.borrow().props.shared_layout().unwrap(),
                    other.borrow().props.shared_layout().unwrap()
                ),
                "{tier:?}: {name}"
            );
        }
        assert_eq!(
            eval(
                &mut engine,
                "delete c.v0; c.extra=7; [a.v0,b.v0,d.v0,Object.keys(c)].join('|')"
            ),
            "2|4|8|next,extra"
        );
    }
}

#[test]
fn reserved_layout_prefix_is_not_an_own_property() {
    let layout = crate::value::new_property_layout(vec![Rc::from("x"), Rc::from("y")]);
    let mut props = Props::with_layout(2, Some(layout.clone()));
    assert!(props.keys().is_empty());
    assert!(!props.contains("x"));
    props.insert("x", Property::plain(Value::Num(1.0)));
    assert!(!props.contains("y"));
    assert!(props.entry_at(1).is_none());
    assert!(props.property_at(1).is_none());
    assert_eq!(props.values().count(), 1);
    props.insert("different", Property::plain(Value::Num(2.0)));
    assert!(!Rc::ptr_eq(props.shared_layout().unwrap(), &layout));
    assert!(!props.contains("y"));
    assert_eq!(&*layout[1], "y");
}

#[test]
fn learned_insertion_chains_reserve_once_without_creating_predicted_fields() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function make(){var o={};o.chainFirst=1;o.chainSecond=2;o.chainThird=3;return o} var warm;for(var n=0;n<100;n++)warm=make();var partial={};partial.chainFirst=4;");
    let partial = object(&mut engine, "partial");
    let warm = object(&mut engine, "warm");
    assert!(Rc::ptr_eq(
        partial.borrow().props.shared_layout().unwrap(),
        warm.borrow().props.shared_layout().unwrap()
    ));
    assert_eq!(
        partial.borrow().props.retained_requested_storage_bytes(),
        (3 * std::mem::size_of::<Property>(), true),
        "the whole small learned sequence is reserved by its first insertion"
    );
    assert_eq!(eval(&mut engine, "[Object.keys(partial),'chainSecond' in partial,Object.getOwnPropertyDescriptor(partial,'chainThird')].join('|')"), "chainFirst|false|");
    assert_eq!(eval(&mut engine, "partial.alternative=8;[Object.keys(partial),Object.keys(warm),warm.chainThird,'chainSecond' in partial].join('|')"), "chainFirst,alternative|chainFirst,chainSecond,chainThird|3|false");
}

#[test]
fn constructed_instances_use_shared_layouts_in_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, "function Box(x){this.x=x;this.y=x+1} function make(x){return new Box(x)} var warm; for(var n=0;n<80;n++)warm=make(n); var a=make(1), b=make(2);");
        let a = object(&mut engine, "a");
        let b = object(&mut engine, "b");
        assert!(
            Rc::ptr_eq(
                a.borrow().props.shared_layout().unwrap(),
                b.borrow().props.shared_layout().unwrap()
            ),
            "{tier:?}"
        );
        if tier == Tier::Jit && cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            let maker = object(&mut engine, "make");
            let maker = maker.borrow();
            let Callable::User(user) = &maker.call else {
                panic!("user maker")
            };
            assert!(user
                .func
                .code
                .get()
                .and_then(Option::as_ref)
                .unwrap()
                .jit
                .get()
                .is_some_and(|code| code.is_some()));
        }
        assert_eq!(eval(&mut engine, "Object.defineProperty(a,'x',{writable:false}); b.x=9; delete a.y; a.z=7; [a.x,b.x,b.y,Object.keys(a)].join('|')"), "1|9|3|x,z");
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn shared_forwarder_chunks_use_the_selected_initializer_layout() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function wrapper(){return function(){this.initialize.apply(this,arguments)}}var Wrapper=wrapper(),Other=wrapper();var init=Wrapper.prototype.initialize=function(x,y){if(!x)x=0;if(!y)y=0;this.x=x;this.y=y};var otherInit=Other.prototype.initialize=function(x,y){if(!x)x=0;if(!y)y=0;this.alpha=x;this.beta=y};function make(){return new Wrapper(3,4)}function makeOther(){return new Other(5,6)}var a,b,c;for(var n=0;n<100;n++){a=make();c=makeOther()}b=make();");
    let a = object(&mut engine, "a");
    let wrapper = object(&mut engine, "init");
    {
        let wrapper = wrapper.borrow();
        let Callable::User(user) = &wrapper.call else {
            panic!("user wrapper")
        };
        let chunk = user
            .func
            .code2
            .get()
            .or_else(|| user.func.code.get())
            .and_then(Option::as_ref)
            .expect("compiled initializer");
        assert!(Rc::ptr_eq(
            chunk.instance_layout().expect("learned chunk layout"),
            a.borrow().props.shared_layout().unwrap()
        ));
    }
    engine.interp.construct_capacity_hints.clear();
    assert_eq!(
        eval(
            &mut engine,
            "b=make();c=makeOther();[b.x+b.y,c.alpha+c.beta,Object.keys(c)].join('|')"
        ),
        "7|11|alpha,beta"
    );
    let c = object(&mut engine, "c");
    assert!(!Rc::ptr_eq(
        a.borrow().props.shared_layout().unwrap(),
        c.borrow().props.shared_layout().unwrap()
    ));
    assert!(
        engine.interp.construct_capacity_hints.is_empty(),
        "guarded plans must not re-enter the identity hint table per instance"
    );
    assert_eq!(eval(&mut engine, "Wrapper.prototype.initialize=function(x,y){this.different=x+y};b=make();[Object.keys(b),b.different,'x' in b].join('|')"), "different|7|false");
}

#[test]
fn warmed_literals_detach_on_delete_and_preserve_key_order() {
    check(
        r#"
        function make(x){return {a:x,b:x+1,c:x+2}}
        function read(o){return o.b}
        var a,b; for(var n=0;n<100;n++){a=make(n);b=make(n+1);read(a)}
        var s=Symbol('s'); a[s]=1; a[10]=10; a[2]=2;
        delete a.b; a.b=500; Object.defineProperty(a,'c',{enumerable:false});
        [read(a),read(b),Object.keys(a).join(','),Reflect.ownKeys(a).map(String).join(',')].join('|')
    "#,
        "500|101|2,10,a,b|2,10,a,c,b,Symbol(s)",
    );
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn native_creation_accepts_equal_keys_from_distinct_allocations() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function Init(x){this.alpha=x;this.beta=x+1} var target={}; function write(){Init.call(target,7)}");
    let target = object(&mut engine, "target");
    // These strings are deliberately not compiler/AST atoms. This models a layout learned
    // before tier-up or shared from an independent creation site.
    let layout = crate::value::new_property_layout(vec![Rc::from("alpha"), Rc::from("beta")]);
    for _ in 0..100 {
        target.borrow_mut().props = Props::with_layout(2, Some(layout.clone()));
        eval(&mut engine, "write()");
    }
    assert_eq!(
        measured_creation_write(&mut engine, &target, || {
            Props::with_layout(2, Some(layout.clone()))
        }),
        0,
        "warmed stores must create fields in native code, not just produce correct fallback values"
    );
    assert_eq!(
        eval(
            &mut engine,
            "[target.alpha,target.beta,Object.keys(target)].join('|')"
        ),
        "7|8|alpha,beta"
    );
    assert!(Rc::ptr_eq(
        target.borrow().props.shared_layout().unwrap(),
        &layout
    ));
}

#[test]
fn constructor_predictions_do_not_bypass_observable_mutations() {
    check(
        r#"
        var log=[], skip=false;
        function observe(o){log.push(Object.keys(o).join(',')); if(skip)Object.preventExtensions(o)}
        function Box(x){this.a=x; observe(this); this.b=x+1}
        function make(x){return new Box(x)}
        for(var n=0;n<100;n++)make(n);
        log=[]; skip=true; var a=make(4); skip=false; var b=make(5);
        var hits=0; Object.defineProperty(Box.prototype,'b',{set(v){hits+=v},configurable:true});
        var c=make(6); delete Box.prototype.b;
        [Object.keys(a),a.b,Object.keys(b),b.b,Object.keys(c),hits,log.join(';')].join('|')
    "#,
        "a||a,b|6|a|7|a;a;a",
    );
}

#[test]
fn warmed_descriptors_accessors_and_proxies_remain_instance_local() {
    check(
        r#"
        function make(x){return {x:x,y:2}}
        function read(o){return o.x}
        function write(o,x){'use strict';o.x=x;return o.x}
        var a,b; for(var n=0;n<100;n++){a=make(n);b=make(n);write(a,n);read(a)}
        var gets=0,sets=0; Object.defineProperty(a,'x',{get(){gets++;return this.y},set(x){sets++;this.y=x}});
        var r=[write(a,8),read(b),gets,sets];
        Object.freeze(b); try{write(b,3)}catch(e){r.push(e.name)}
        var traps=0,p=new Proxy(a,{get(t,k,r){traps++;return Reflect.get(t,k,r)}});
        r.push(read(p),traps,Object.getOwnPropertyDescriptor(a,'x').get!==undefined,Object.isFrozen(b));
        r.join('|')
    "#,
        "8|99|1|1|TypeError|8|2|true|true",
    );
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn native_creation_adopts_cached_layouts_and_defers_last_owner_drops() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "var target={};function write(){target.alpha=7;target.beta=8}",
    );
    let target = object(&mut engine, "target");
    for _ in 0..100 {
        target.borrow_mut().props = Props::with_capacity(3);
        eval(&mut engine, "write()");
    }
    for shared in [false, true] {
        let previous =
            crate::value::new_property_layout(vec![Rc::from("wrong"), Rc::from("unused")]);
        assert_eq!(
            measured_creation_write(&mut engine, &target, || {
                if shared {
                    Props::with_layout(3, Some(previous.clone()))
                } else {
                    Props::with_capacity(3)
                }
            }),
            0
        );
        assert_eq!(
            Rc::strong_count(&previous),
            1,
            "native replacement releases exactly one old layout owner"
        );
        assert_eq!(
            eval(
                &mut engine,
                "[target.alpha,target.beta,Object.keys(target)].join('|')"
            ),
            "7|8|alpha,beta"
        );
    }
    assert!(
        measured_creation_write(&mut engine, &target, || {
            Props::with_layout(
                3,
                Some(crate::value::new_property_layout(vec![Rc::from("private")])),
            )
        }) > 0,
        "Rust must release a last-owned old layout"
    );
    assert_eq!(
        eval(&mut engine, "Object.keys(target).join(',')"),
        "alpha,beta"
    );
}

#[test]
fn array_named_slots_remain_key_checked_after_truncation() {
    check(
        r#"
        function read(a){return a.extra}
        var a=[];for(var n=0;n<40;n++)a[n]=n; a.extra=90;
        for(var n=0;n<100;n++)read(a);
        a.length=3; var r=[read(a),a.length,Object.keys(a).join(',')];
        a[7]=7; delete a[1]; a.extra=91;
        r.push(read(a),Object.keys(a).join(','));
        Object.defineProperty(a,'5',{value:5,configurable:false});
        try{Object.defineProperty(a,'length',{value:2})}catch(e){r.push(e.name)}
        r.push(a.length,read(a),a[5]);r.join('|')
    "#,
        "90|3|0,1,2,extra|91|0,2,7,extra|TypeError|6|91|5",
    );
}

#[test]
fn layout_values_accessors_symbols_and_cycles_survive_collection() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            function make(v){return {child:v,self:null}}
            var a=make({n:42}), b=make({n:17}); a.self=a; b.self=b;
            var key=Symbol('held');a[key]={n:9};
            Object.defineProperty(a,'getter',{get:function(){return a.child.n}});
        "#,
        );
        engine.interp.gc_collect();
        assert_eq!(
            eval(
                &mut engine,
                "[a.getter,b.child.n,a.self===a,a[Reflect.ownKeys(a)[3]].n].join('|')"
            ),
            "42|17|true|9"
        );
        eval(&mut engine, "a=null;b=null;key=null");
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "make({n:3}).child.n"), "3");
    }
}
