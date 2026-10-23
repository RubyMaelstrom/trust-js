//! ECMA-262 e28783d5, Array Exotic Objects / ArraySetLength / OrdinaryOwnPropertyKeys.
//! Representation checks accompany observable regressions: the change must remove
//! decimal-key storage, not merely happen to produce the same array contents.
use super::*;
use crate::value::ValueView;
use crate::{bytecode::Tier, Completion, Engine};

#[test]
fn dense_elements_large_lists_and_growth_keep_only_named_keys() {
    let mut props = Props::packed_array((0..8192).map(|n| Value::Num(n as f64)));
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

    let mut grown = Props::packed_array(std::iter::empty());
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
    assert!(
        matches!((grown.get_index(12).unwrap().value()).view(), ValueView::Str(s) if s.as_str() == "mixed")
    );
}

#[test]
fn dense_elements_sparse_length_does_not_reserve_storage() {
    let mut props = Props::packed_array(std::iter::empty());
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

#[test]
fn dense_elements_far_deletion_and_truncation_restore_packed_coverage() {
    for length in [0, 3, 40, 4096] {
        for truncate in [false, true] {
            let mut props = Props::packed_array((0..length).map(|n| Value::Num(n as f64)));
            props.insert("1000000", Property::plain(Value::Num(1.0)));
            props.insert("2000000", Property::plain(Value::Num(2.0)));
            assert!(props.has_far.get());
            assert!(!props.prepare_packed_numeric_mirror());
            assert!(props.remove("2000000"));
            assert!(props.has_far.get(), "another far key survives");
            if truncate {
                props.remove_indices_from(length);
            } else {
                assert!(props.remove("1000000"));
            }
            assert!(!props.has_far.get(), "{length}, truncate={truncate}");
            assert_eq!(props.entries.len(), 1);
            assert!(props.get("1000000").is_none());
            if length != 0 {
                assert!(props.prepare_packed_numeric_mirror());
                assert_eq!(
                    props.mirror_get(length as u32 - 1),
                    Some((length - 1) as f64)
                );
            }
            assert!(props
                .try_append_element(length as u32, Property::plain(Value::Num(7.0)))
                .is_ok());
            assert_eq!(
                props.pop_last_element(length as u32).unwrap().as_num_opt(),
                Some(7.0)
            );
        }
    }
}

#[test]
fn dense_elements_recovery_keeps_named_indices_outside_packed_storage_guarded() {
    let mut props = Props::packed_array([Value::Num(0.0)].into_iter());
    props.insert("1000000", Property::plain(Value::Num(1.0)));
    // While packed growth is blocked, even a nearby canonical key is a named entry.
    props.insert("1", Property::data(Value::Num(11.0), false, true, false));
    assert!(props.elems.get(1).is_some());
    assert!(props.remove("1000000"));
    assert!(props.has_far.get(), "a dense-map slot is not a packed slot");
    assert!(props
        .try_append_element(1, Property::plain(Value::Num(2.0)))
        .is_err());
    props.remove_indices_from(2);
    assert!(props.has_far.get());
    assert_eq!(props.get("1").unwrap().number_value(), Some(11.0));
    assert!(!props.get("1").unwrap().configurable());
    // The caller's ArraySetLength validation owns configurability checks. Here only
    // the storage proof is tested, after a validated boundary removes that index.
    props.remove_indices_from(1);
    assert!(!props.has_far.get());
    assert!(props
        .try_append_element(1, Property::plain(Value::Num(2.0)))
        .is_ok());
    assert_eq!(props.get_index(1).unwrap().number_value(), Some(2.0));
    assert_eq!(props.keys().iter().filter(|key| &***key == "1").count(), 1);
}

#[test]
fn dense_elements_classic_recovery_bounds_delete_work_and_rebuilds_on_truncation() {
    for length in [12, 300] {
        let mut props = Props::packed_array(std::iter::empty());
        props.insert("1000000", Property::plain(Value::Num(1.0)));
        for n in 0..length {
            props.insert(index_key(n), Property::plain(Value::Num(n as f64)));
        }
        assert!(!props.elems.packed_is_some());
        assert!(props.remove("1000000"));
        assert_eq!(props.has_far.get(), length > 256);
        props.remove_indices_from(length / 2);
        assert!(!props.has_far.get());
        for n in 0..length / 2 {
            assert_eq!(
                props.get_index(n as u32).unwrap().number_value(),
                Some(n as f64)
            );
        }
        assert!(props
            .try_append_element((length / 2) as u32, Property::plain(Value::Num(9.0)))
            .is_ok());
        props.insert("1000000", Property::plain(Value::Num(1.0)));
        props.clear();
        assert!(!props.has_far.get());
    }
}

#[test]
fn dense_elements_recovered_numeric_mirror_rechecks_descriptors_and_holes() {
    for blocked in ["hole", "accessor", "non-writable", "non-number"] {
        let mut props = Props::packed_array([Value::Num(-0.0), Value::Num(2.0)].into_iter());
        props.insert("1000000", Property::plain(Value::Num(1.0)));
        match blocked {
            "hole" => assert!(props.remove("1")),
            "accessor" => props.insert("1", Property::accessor_prop(None, None, true, true)),
            "non-writable" => props.get_mut("1").unwrap().set_writable(false),
            "non-number" => props.get_mut("1").unwrap().set_value(Value::lstr("s")),
            _ => unreachable!(),
        }
        assert!(!props.prepare_packed_numeric_mirror());
        assert!(props.remove("1000000"));
        assert!(!props.has_far.get());
        assert!(!props.prepare_packed_numeric_mirror(), "{blocked}");
        props.insert("1", Property::plain(Value::Num(2.0)));
        assert!(props.prepare_packed_numeric_mirror(), "{blocked}");
        assert_eq!(props.mirror_get(0).unwrap().to_bits(), (-0.0f64).to_bits());
    }
}

#[test]
fn dense_elements_sparse_recovery_preserves_observers_and_partial_length_failure() {
    let source = r#"
    (function () {
        function assert(ok, message) { if (!ok) throw new Error(message); }
        function sum(a) { var total=0; for(var k=0;k<a.length;k++)total+=a[k]; return total; }
        function recover(a) { a[1000000]=7; a.length=3; return a; }
        var a=[1,2,3];
        for(var k=0;k<4;k++)assert(sum(a)===6,'numeric warm');
        recover(a);
        for(var k=0;k<40;k++){a.push(k);assert(a.pop()===k,'tail');assert(sum(a)===6,'numeric');}
        a[2000000]=8; a[1000000]=7; delete a[2000000];
        assert(a[1000000]===7,'remaining far key'); a.length=3;
        var log=[];
        Object.defineProperty(a,'2',{get:function(){log.push('get');return 30;},configurable:true});
        assert(a.pop()===30 && log.join()==='get','own accessor');
        a.push(3); delete a[1];
        var proto=Object.create(Array.prototype);
        Object.defineProperty(proto,'1',{get:function(){return 20;},configurable:true});
        Object.defineProperty(proto,'3',{set:function(v){log.push('set'+v);},
            get:function(){log.push('proto');return 99;},configurable:true});
        Object.setPrototypeOf(a,proto);
        assert(sum(a)===24 && !Object.hasOwn(a,1),'hole');
        assert(a.push(4)===4 && !Object.hasOwn(a,3),'inherited setter');
        assert(a.pop()===99 && log.join()==='get,set4,proto','inherited getter');
        var b=recover([1,2,3]);Object.defineProperty(b,'length',{writable:false});
        var error='';try{b.push(4);}catch(e){error=e.name;}
        assert(error==='TypeError' && b.length===3,'length writable');
        b=recover([1,2,3]);Object.preventExtensions(b);
        assert(!Reflect.set(b,3,4) && b.length===3,'extensible');
        var c=[1,2,3];
        Object.defineProperty(c,'4096',{value:41,writable:true,enumerable:true,configurable:false});
        c[1000000]=7;
        assert(!Reflect.defineProperty(c,'length',{value:3}) && c.length===4097,'shrink barrier');
        assert(!Object.hasOwn(c,1000000) && c[4096]===41,'partial shrink');
        assert(c.push(42)===4098 && c.pop()===42,'surviving far key');
        error='';try{c.pop();}catch(e){error=e.name;}
        assert(error==='TypeError' && c.length===4097 && c[4096]===41,'barrier descriptor');
        // A nearby named index must never be overwritten by restored packed growth.
        var d=[0]; d[1000000]=7;
        Object.defineProperty(d,'1',{value:11,writable:false,enumerable:true,configurable:true});
        d.length=2;
        assert(d[1]===11 && !Reflect.set(d,1,2),'named index descriptor');
        d.length=1;assert(d.push(2)===2 && d[1]===2,'named index removal');
        var n=recover([-0,NaN,Infinity]);
        assert(Object.is(n[0],-0) && Number.isNaN(n[1]) && n[2]===Infinity,'number bits');
        assert(Reflect.ownKeys(n).join()==='0,1,2,length','own key order');
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
fn dense_elements_sparse_recovery_preserves_cyclic_owners_across_gc() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            var alive=[], gone;
            for(var k=0;k<40;k++)alive.push({n:k,owner:alive});
            (function(){var far={owner:alive};far.self=far;
                gone=new WeakRef(far);alive[1000000]=far;})();
            alive.length=40;
            alive.push(alive[0]);var tail=alive.pop();
            if(tail!==alive[0])throw 'owner move';tail=null;
            'ready'
        "#,
        );
        engine.interp.gc_collect();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            if(gone.deref()!==undefined)throw 'far owner retained';
            if(alive[39].n!==39 || alive[39].owner!==alive)throw 'packed owner lost';
            alive.unshift(alive[39]);alive.pop();
            if(alive[0].n!==39 || alive[0].owner!==alive)throw 'recovered edit owner';
            'ok'
        "#
            ),
            "ok",
            "{tier:?}"
        );
        eval(&mut engine, "alive=null");
        engine.interp.gc_collect();
    }
}
