//! ECMA-262 e28783d5, Array Exotic Objects / ArraySetLength / OrdinaryOwnPropertyKeys.
//! Representation checks accompany observable regressions: the change must remove
//! decimal-key storage, not merely happen to produce the same array contents.
use super::*;
use crate::{bytecode::Tier, Completion, Engine};

#[test]
fn dense_elements_large_lists_and_growth_keep_only_named_keys() {
    let mut props =
        Props::packed_array_from_values((0..8192).map(|n| Value::Num(n as f64)).collect());
    assert_eq!(props.entries.len(), 1);
    assert!(props.elems.elems.is_empty());
    assert_eq!(props.elems.packed_ref().unwrap().len(), 8192);
    assert_eq!(props.mirror_get(8191), Some(8191.0));
    assert!(props
        .try_append_element(8192, Property::plain(Value::Num(-0.0)))
        .is_ok());
    assert_eq!(
        props.mirror_get(8192).unwrap().to_bits(),
        (-0.0f64).to_bits()
    );
    assert!(props.set_index_value(8191, Value::Num(3.5)).is_ok());
    assert_eq!(props.mirror_get(8191), Some(3.5));

    let mut grown = Props::packed_array_from_values(Vec::new());
    for n in 0..8192 {
        assert!(grown
            .try_append_element(n, Property::plain(Value::Num(n as f64)))
            .is_ok());
    }
    assert_eq!(grown.entries.len(), 1);
    assert!(grown.elems.elems.is_empty());
    assert_eq!(grown.mirror_get(8191), Some(8191.0));
    assert!(grown.set_index_value(12, Value::lstr("mixed")).is_ok());
    assert_eq!(grown.mirror_get(12), None);
    assert!(matches!(grown.get_index(12).unwrap().value(), Value::Str(s) if s.as_str() == "mixed"));
}

#[test]
fn dense_elements_sparse_length_does_not_reserve_storage() {
    let mut props = Props::packed_array_from_values(Vec::new());
    props
        .get_mut("length")
        .unwrap()
        .set_value(Value::Num(u32::MAX as f64));
    assert!(props.elems.0.is_none());
    assert!(props
        .try_define_dense_element(200, Property::plain(Value::Num(2.0)))
        .is_ok());
    assert!(props.get_index(199).is_none());
    assert_eq!(props.elems.packed_ref().unwrap().len(), 201);
    props.insert("4000000000", Property::plain(Value::Num(4.0)));
    assert_eq!(props.elems.packed_ref().unwrap().len(), 201);
    assert_eq!(props.get("4000000000").unwrap().number_value(), Some(4.0));
    assert_eq!(
        props
            .ordered_keys()
            .iter()
            .map(|s| &**s)
            .collect::<Vec<_>>(),
        ["200", "4000000000", "length"]
    );
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("dense element fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

#[test]
fn dense_elements_observable_descriptors_holes_keys_and_partial_shrink() {
    let source = r#"
    (function () {
        var a = []; for (var n = 0; n < 160; ++n) a.push(n);
        var s = Symbol('s'); a.extra = 1; a[s] = 2;
        delete a[80];
        var proto = Object.create(Array.prototype);
        Object.defineProperty(proto, '80', {get:function(){return 800}, configurable:true});
        Object.setPrototypeOf(a, proto);
        if (a[80] !== 800 || Object.hasOwn(a, 80)) throw 'hole';
        Object.defineProperty(a, '81', {get:function(){return 810}, enumerable:true, configurable:true});
        if (a[81] !== 810) throw 'accessor';
        Object.defineProperty(a, '100', {value:100, writable:false, configurable:false});
        var shrunk = Reflect.defineProperty(a, 'length', {value:90, writable:false});
        if (shrunk || a.length !== 101 || 102 in a || a[100] !== 100) throw 'partial shrink';
        if (Object.getOwnPropertyDescriptor(a,'length').writable) throw 'length writable';
        if (Reflect.set(a,'101',1)) throw 'length gate';
        var keys=Reflect.ownKeys(a);
        if (keys[keys.length-3] !== 'length' || keys[keys.length-2] !== 'extra' || keys[keys.length-1] !== s) throw 'keys';
        var b = new Array(4294967295); b[0]=5; b[4000000000]=7; b['4294967295']=9;
        if (b.length !== 4294967295 || b[4000000000] !== 7 || Object.keys(b).join() !== '0,4000000000,4294967295') throw 'sparse';
        b.length=1;
        if (b[4000000000] !== undefined || b['4294967295'] !== 9) throw 'sparse shrink';
        var c=[]; for(var j=0;j<150;j++) c[j]={v:j}; Object.freeze(c);
        if (!Object.isFrozen(c) || Reflect.set(c,'2',4) || Reflect.deleteProperty(c,'2')) throw 'freeze';
        var d=[]; for(var j=0;j<40;j++)d.push(j);
        var seen=0;
        Object.defineProperty(d,'0',{set:function(v){seen+=v},configurable:true});
        d.fill(7);
        if(seen!==7 || !Object.getOwnPropertyDescriptor(d,'0').set) throw 'fill setter';
        Object.defineProperty(d,'2',{value:22,writable:false});
        var error='';try{d.fill(9)}catch(e){error=e.name}
        if(error!=='TypeError' || seen!==16 || d[1]!==9 || d[2]!==22 || d[3]!==7) throw 'fill partial effects';
        return 'ok';
    })()
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), "ok", "{tier:?}");
        engine.interp.gc_collect();
    }
}

#[test]
fn dense_elements_numeric_loops_and_cycles_keep_owners() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var alive=[];
            for(var j=0;j<600;j++) alive.push({n:j, owner:alive});
            function work(n) {
                var a=[]; for(var i=0;i<n;i++) a.push(i);
                for(var i=0;i<n;i++) a[i]=a[i]*2+1;
                var total=0; for(var i=0;i<n;i++) total+=a[i];
                return total;
            }
            work(8192)
        "#
            ),
            "67108864",
            "{tier:?}"
        );
        engine.interp.gc_collect();
        assert_eq!(
            eval(&mut engine, "alive[599].n+':'+(alive[599].owner===alive)"),
            "599:true"
        );
        eval(&mut engine, "alive=null");
        engine.interp.gc_collect();
    }
}
