//! ECMA-262 e28783d5 EnumerableOwnProperties / CreateArrayFromList.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("enumeration fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn ordinary_record_entries_have_independent_arrays_and_live_values() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let before = crate::builtins::TEST_RECORD_ENUMERATIONS.with(std::cell::Cell::get);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function entries(o){return Object.entries(o)}
            function values(o){return Object.values(o)}
            var child={n:1},symbol=Symbol(),source={first:child,second:symbol,third:3n,zero:-0,missing:undefined,'é\uD800':7};
            Object.freeze(source);
            var a=entries(source),b=entries(source),v=values(source),ok=true;
            for(var n=0;n<a.length;n++){
                var d=Object.getOwnPropertyDescriptor(a[n],'0');
                ok=ok&&a[n]!==b[n]&&Array.isArray(a[n])&&Object.getPrototypeOf(a[n])===Array.prototype&&a[n].length===2;
                ok=ok&&d.writable&&d.enumerable&&d.configurable&&Object.is(a[n][1],v[n]);
            }
            a[0][0]='changed';a[0][1]=99;child.n=8;
            ok&&a!==b&&b[0][0]==='first'&&b[0][1]===child&&source.first.n===8&&b[1][1]===symbol&&b[2][1]===3n&&Object.is(b[3][1],-0)&&b[4][1]===undefined&&b[5][0]==='é\uD800'
        "#
            ),
            "true",
            "{tier:?}"
        );
        assert!(crate::builtins::TEST_RECORD_ENUMERATIONS.with(std::cell::Cell::get) - before >= 3);
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "Object.entries(source)[0][1].n"), "8");
    }
}

#[test]
fn enumeration_fallbacks_keep_order_traps_and_live_descriptor_effects() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, r#"
            var source={first:1,second:2},one=Object.entries(source);
            Object.defineProperty(source,'first',{get(){delete source.second;source.third=3;return 11}});
            var two=Object.entries(source),hidden={a:1,b:2};
            Object.defineProperty(hidden,'b',{enumerable:false});
            var symbol=Symbol();hidden[symbol]=3;
            var ordered={'10':'ten',z:'z','2':'two',a:'a'},trace='';
            var proxy=new Proxy({x:4,y:5},{
                ownKeys(t){trace+='k';return ['x','y']},
                getOwnPropertyDescriptor(t,k){trace+='d'+k;return Reflect.getOwnPropertyDescriptor(t,k)},
                get(t,k,r){trace+='g'+k;return Reflect.get(t,k,r)}
            });
            var pairs=Object.entries(proxy),errors=0;
            for(var bad of [null,undefined])try{Object.entries(bad)}catch(e){if(e instanceof TypeError)errors++}
            [JSON.stringify(one),JSON.stringify(two),JSON.stringify(Object.entries(hidden)),Object.entries(ordered).map(p=>p[0]).join(','),trace,JSON.stringify(pairs),errors,JSON.stringify(Object.entries('é'))].join('|')
        "#), "[[\"first\",1],[\"second\",2]]|[[\"first\",11]]|[[\"a\",1]]|2,10,z,a|kdxgxdygy|[[\"x\",4],[\"y\",5]]|2|[[\"0\",\"é\"]]", "{tier:?}");
    }
}

#[test]
fn immutable_key_encoding_cache_is_bounded_and_owns_no_objects() {
    use std::rc::Rc;
    let mut engine = Engine::new();
    let key: Rc<str> = "é\u{10f800}".into();
    let first = engine.interp.property_key_string(&key);
    let second = engine.interp.property_key_string(&key);
    assert!(crate::lstr::LStr::ptr_eq(&first, &second));
    for n in 0..600 {
        let key: Rc<str> = format!("named-key-{n}").into();
        engine.interp.property_key_string(&key);
    }
    let (entries, bytes) = engine.interp.property_key_strings.stats();
    assert!(entries <= 512 && bytes <= 256 << 10);
    let large: Rc<str> = "x".repeat(256 << 10).into();
    engine.interp.property_key_string(&large);
    let (entries, bytes) = engine.interp.property_key_strings.stats();
    assert!(entries <= 512 && bytes <= 256 << 10);
    engine.interp.gc_collect();
    assert_eq!(&*engine.interp.property_key_string(&key), &*key);
}
