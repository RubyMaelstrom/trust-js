//! ECMA-262 e28783d5, Array Exotic Objects / ArraySetLength / OrdinaryOwnPropertyKeys.
//! Representation checks accompany observable regressions: the change must remove
//! decimal-key storage, not merely happen to produce the same array contents.
use super::*;
use crate::value::ValueView;
use crate::{bytecode::Tier, Completion, Engine};

#[test]
fn dense_elements_large_lists_and_growth_keep_only_named_keys() {
    let number = |props: &Props, n: u32| props.get_index(n).and_then(|e| e.number_value());
    let mut props = Props::packed_array((0..8192).map(|n| Value::Num(n as f64)));
    assert_eq!(props.entries.len(), 1);
    assert!(props.elems.0.is_none());
    assert_eq!(props.elems.words().unwrap().len(), 8192);
    assert_eq!(number(&props, 8191), Some(8191.0));
    assert!(props
        .try_append_element(8192, Property::plain(Value::Num(-0.0)))
        .is_ok());
    assert_eq!(number(&props, 8192).unwrap().to_bits(), (-0.0f64).to_bits());
    assert!(props.set_index_value(8191, Value::Num(3.5)).is_ok());
    assert_eq!(number(&props, 8191), Some(3.5));

    let mut grown = Props::packed_array(std::iter::empty());
    for n in 0..8192 {
        assert!(grown
            .try_append_element(n, Property::plain(Value::Num(n as f64)))
            .is_ok());
    }
    assert_eq!(grown.entries.len(), 1);
    assert!(grown.elems.0.is_none());
    assert_eq!(number(&grown, 8191), Some(8191.0));
    assert!(grown.set_index_value(12, Value::lstr("mixed")).is_ok());
    assert_eq!(number(&grown, 12), None);
    assert_eq!(
        grown.elems.words().unwrap().kind(),
        word_block::KIND_NO_HOLES
    );
    assert!(
        matches!((grown.get_index(12).unwrap().value()).view(), ValueView::Str(s) if s.text() == "mixed")
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
    assert_eq!(props.elems.words().unwrap().len(), 201);
    props.insert("4000000000", Property::plain(Value::Num(4.0)));
    assert_eq!(props.elems.words().unwrap().len(), 201);
    assert_eq!(props.get("4000000000").unwrap().number_value(), Some(4.0));
    assert_eq!(
        props
            .ordered_keys()
            .iter()
            .map(|s| s.text().to_string())
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
            assert!(
                props.element_words().is_none(),
                "far keys block the in-place views"
            );
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
                assert_eq!(props.element_words().map(<[u64]>::len), Some(length));
                assert_eq!(
                    props
                        .get_index(length as u32 - 1)
                        .and_then(|element| element.number_value()),
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

/// A far index lives in the named entries alone (`has_far`), so once the map holds more than
/// INDEX_THRESHOLD entries its key index must cover index keys: without it every later
/// absent-index insert, and every read past the packed elements, scanned all the entries
/// (filling and reading 200,000 elements after one write at index 1000 took 106.8 s).
#[test]
fn dense_elements_far_keys_index_the_named_entries() {
    // A packed array with one far write, and an empty one whose first write is far: past the
    // sparse limit and too sparse to densify (W2c step 6), so every later index joins it.
    let far = 1_000_000u32;
    for start in [3u32, 0] {
        let mut props = Props::packed_array((0..start).map(|n| Value::Num(n as f64)));
        props.insert(far.to_string().as_str(), Property::plain(Value::Num(-1.0)));
        assert!(props.has_far.get(), "{start}");
        for n in start..start + 2000 {
            props.insert(
                n.to_string().as_str(),
                Property::plain(Value::Num(n as f64)),
            );
        }
        assert!(props.has_far.get(), "{start}");
        assert!(props.elems.index().is_some(), "{start}: no key index");
        for n in 0..start + 2000 {
            assert_eq!(
                props.get(&n.to_string()).and_then(|p| p.number_value()),
                Some(n as f64),
                "{start}: {n}"
            );
        }
        assert_eq!(
            props.get(&far.to_string()).and_then(|p| p.number_value()),
            Some(-1.0)
        );
        assert!(props.get(&(start + 2000).to_string()).is_none());
        let keys = props.ordered_keys();
        assert_eq!(
            keys.len(),
            (start + 2000) as usize + 2,
            "the indices, the far index and length"
        );
        assert!(keys
            .iter()
            .take((start + 2000) as usize)
            .enumerate()
            .all(|(i, key)| key.as_index() == Some(i as u32)));
        assert_eq!(keys[(start + 2000) as usize].as_index(), Some(far));
    }
}

#[test]
fn dense_elements_recovery_keeps_named_indices_outside_packed_storage_guarded() {
    let mut props = Props::packed_array([Value::Num(0.0)].into_iter());
    props.insert("1000000", Property::plain(Value::Num(1.0)));
    // While packed growth is blocked, even a nearby canonical key is a named entry.
    props.insert("1", Property::data(Value::Num(11.0), false, true, false));
    assert!(props.find(PropKey::index(1)).is_some());
    assert!(props.elems.words().unwrap().get(1).is_none());
    assert!(props.remove("1000000"));
    assert!(props.has_far.get(), "a named element remains");
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
    assert_eq!(props.keys().iter().filter(|key| key.is("1")).count(), 1);
}

#[test]
fn dense_elements_named_recovery_keeps_named_indices_until_truncation() {
    for length in [12, 300] {
        let mut props = Props::packed_array(std::iter::empty());
        props.insert("1000000", Property::plain(Value::Num(1.0)));
        // With an index among the named entries, every later one joins it there.
        for n in 0..length {
            props.insert(index_key(n), Property::plain(Value::Num(n as f64)));
        }
        assert!(!props.elems.has_words());
        assert!(props.remove("1000000"));
        assert!(props.has_far.get(), "the named elements remain");
        props.remove_indices_from(length / 2);
        assert!(props.has_far.get());
        for n in 0..length / 2 {
            assert_eq!(
                props.get_index(n as u32).unwrap().number_value(),
                Some(n as f64)
            );
        }
        assert!(props
            .try_append_element((length / 2) as u32, Property::plain(Value::Num(9.0)))
            .is_err());
        // Truncation past every named index restores the words.
        props.remove_indices_from(0);
        assert!(!props.has_far.get());
        assert!(props
            .try_append_element(0, Property::plain(Value::Num(9.0)))
            .is_ok());
        assert!(props.elems.has_words());
        props.insert("1000000", Property::plain(Value::Num(1.0)));
        props.clear();
        assert!(!props.has_far.get());
    }
}

#[test]
fn dense_elements_recovery_keeps_named_elements_far_and_word_kinds_exact() {
    for blocked in ["hole", "accessor", "non-writable", "non-number"] {
        let mut props = Props::packed_array([Value::Num(-0.0), Value::Num(2.0)].into_iter());
        props.insert("1000000", Property::plain(Value::Num(1.0)));
        match blocked {
            "hole" => assert!(props.remove("1")),
            "accessor" => props.insert("1", Property::accessor_prop(None, None, true, true)),
            "non-writable" => props.get_mut("1").unwrap().set_writable(false),
            "non-number" => assert!(props.set_index_value(1, Value::lstr("s")).is_ok()),
            _ => unreachable!(),
        }
        assert!(props.remove("1000000"));
        // A non-plain element stays in the named entries, so the map keeps an index there.
        let named = matches!(blocked, "accessor" | "non-writable");
        assert_eq!(props.has_far.get(), named, "{blocked}");
        props.insert("1", Property::plain(Value::Num(2.0)));
        assert_eq!(props.has_far.get(), named, "{blocked}");
        let words = props.elems.words().unwrap();
        assert_eq!(words.kind() & !words.computed_kind(), 0, "{blocked}");
        assert_eq!(words.get(1).is_some(), !named, "{blocked}");
        assert_eq!(
            props
                .get_index(0)
                .unwrap()
                .number_value()
                .unwrap()
                .to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(props.get("1").unwrap().number_value(), Some(2.0));
        assert!(props.get("1").unwrap().writable());
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

/// W2c step 6: an index joins the words while V8's ShouldConvertToSlowElements would keep fast
/// elements: below the capacity, under 1024 past it, and past a grown capacity of 5,000 only
/// while the named entries would not save a factor of 3.
#[test]
fn the_sparse_limit_follows_v8() {
    let plain = |n: f64| Property::plain(Value::Num(n));
    // An empty array's first index: 1023 past its capacity of none joins the words, 1024 not.
    for (index, words) in [(1023u32, true), (1024, false)] {
        let mut props = Props::packed_array(std::iter::empty());
        props.insert(index.to_string().as_str(), plain(1.0));
        assert_eq!(!props.has_far.get(), words, "{index}");
        assert_eq!(
            props.elems.words().map(word_block::Words::len),
            words.then_some(index as usize + 1),
            "{index}"
        );
    }
    // Past a packed literal's exact capacity of 3.
    for (index, words) in [(1026u32, true), (1027, false)] {
        let mut props = Props::packed_array((0..3).map(|n| Value::Num(n as f64)));
        props.insert(index.to_string().as_str(), plain(1.0));
        assert_eq!(!props.has_far.get(), words, "{index}");
    }
    // Index 5000 past a capacity of 4000 grows it to 7,517 words: words while at least 342 of
    // the 4000 are present (3 * ComputeCapacity(342) * 3 = 9,216), not with 341 (4,608). (The
    // insert itself would then densify again at once, as V8 converts back on its next add: the
    // way back compares the dictionary with the length, 5,001, not the grown capacity.)
    for (present, words) in [(342usize, true), (341, false)] {
        let mut props = Props::packed_array((0..4000).map(|n| Value::Num(n as f64)));
        for n in present..4000 {
            assert!(props.remove(&n.to_string()));
        }
        assert_eq!(props.elems.words().unwrap().present(), present);
        assert_eq!(props.words_admit(5000), words, "{present}");
    }
}

/// W2c step 6's way back: when an array's named entries reach a power of two, its index entries
/// move into the words once each is a plain element and the named form saves less than half
/// the space (V8's ShouldConvertToFastElements), keeping every value, the order and `length`.
#[test]
fn sparse_arrays_densify_when_dense_enough() {
    let plain = |n: f64| Property::plain(Value::Num(n));
    for blocked in [false, true] {
        let mut props = Props::packed_array(std::iter::empty());
        props.insert("100000", plain(-1.0));
        assert!(props.has_far.get());
        if blocked {
            props.insert("7", Property::data(Value::Num(7.0), false, true, true));
        }
        // With the far index and `length`, 16,382 more entries reach 16,384: 16,383 index
        // entries need 2 * 32,768 * 3 >= 100,001 words; at 8,192 entries they did not.
        for n in 0..16381u32 {
            if !(blocked && n == 7) {
                props.insert(n.to_string().as_str(), plain(n as f64));
            }
            assert!(props.has_far.get(), "{n}: {} entries", props.entries.len());
        }
        props.insert("16381", plain(16_381.0));
        if blocked {
            assert!(props.has_far.get(), "a non-plain element blocks the move");
            assert!(props.get_index(7).is_some());
            continue;
        }
        assert!(!props.has_far.get());
        assert_eq!(props.entries.len(), 1, "only `length` remains named");
        let words = props.elems.words().unwrap();
        assert_eq!(words.len(), 100_001);
        assert_eq!(words.present(), 16_383);
        assert_eq!(
            props.get_index(100_000).and_then(|e| e.number_value()),
            Some(-1.0)
        );
        assert_eq!(
            props.get_index(16_381).and_then(|e| e.number_value()),
            Some(16_381.0)
        );
        assert!(props.get_index(16_382).is_none());
        let keys = props.ordered_keys();
        assert_eq!(keys.len(), 16_383 + 1);
        assert!(keys
            .iter()
            .take(16_382)
            .enumerate()
            .all(|(i, key)| key.as_index() == Some(i as u32)));
        assert_eq!(keys[16_382].as_index(), Some(100_000));
    }
    // An ordinary object's index keys stay named: they are part of its shape.
    let mut object = Props::new();
    object.insert("100000", plain(-1.0));
    for n in 0..20000u32 {
        object.insert(n.to_string().as_str(), plain(n as f64));
    }
    assert!(object.has_far.get());
}

/// The script-visible results of both transitions are the specification's: values, order,
/// descriptors and `length`, before and after an array goes to the named entries and back.
#[test]
fn sparse_transitions_keep_properties_and_order() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = match engine
            .eval(
                r#"
            function check() {
                var out = [];
                var a = [1, 2, 3]; a[1000] = 'near';
                out.push(a.length, a[1000], Object.keys(a).length, 999 in a);
                var b = []; b[100000] = 'far';
                for (var i = 0; i < 20000; i++) b[i] = i;
                var d = Object.getOwnPropertyDescriptor(b, 100000);
                out.push(b.length, b[19999], b[100000], d.writable && d.enumerable && d.configurable,
                         Object.keys(b).length, Object.keys(b)[20000], 20000 in b);
                var c = []; c[50000] = 1; Object.defineProperty(c, 3, { value: 'ro', writable: false });
                for (var j = 0; j < 9000; j++) if (j !== 3) c[j] = j;
                c[3] = 'changed';
                out.push(c[3], Object.keys(c).length, c.length);
                return out.join();
            }
            check()
        "#,
                false,
            )
            .unwrap()
        {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        };
        assert_eq!(
            result, "1001,near,4,false,100001,19999,far,true,20001,100000,false,ro,9000,50001",
            "{tier:?}"
        );
    }
}

/// W2c step 7b: generated code reads an Array's `length` at slot 0 (V8's descriptor 0). Every way
/// of making an Array defines it first, and nothing moves it.
#[test]
fn every_array_keeps_length_at_slot_zero() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    match engine
        .eval(
            r#"
        class Sub extends Array {}
        var made = [
            [], [1, 2, 3], new Array(10), new Array(1, 2), Array(3), Array.of(4, 5), Array.from('ab'),
            [1, 2, 3].map(function (x) { return x; }), [1, 2, 3].filter(function () { return true; }),
            [1].concat([2]), [1, 2, 3].slice(1), [3, 1, 2].sort(), 'a,b'.split(','), Object.keys({ a: 1 }),
            JSON.parse('[1,[2]]'), JSON.parse('[1,[2]]')[1], [...'xy'], Array.prototype, new Sub(3), Sub.of(1),
            (function () { return Array.prototype.slice.call(arguments); })(1, 2), /(a)(b)/.exec('ab'),
            Object.entries({ k: 1 })[0], [0, 1].flat(), [[0], [1]].flatMap(function (x) { return x; }),
        ];
        var named = [1, 2]; named.extra = 1; named.length = 1; delete named.extra; named.more = 2;
        made.push(named);
        var frozen = Object.freeze([1, 2]); made.push(frozen);
        var sealed = Object.seal([1, 2]); made.push(sealed);
        var ro = [1, 2]; Object.defineProperty(ro, 'length', { writable: false }); made.push(ro);
        made.length
        "#,
            false,
        )
        .unwrap()
    {
        Completion::Value(count) => assert_eq!(count, "29"),
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    let global = Value::Obj(engine.interp.global);
    let made = engine
        .interp
        .get_member_key(&global, "made")
        .unwrap_or_else(|_| panic!("made"))
        .as_obj()
        .unwrap();
    let count = made.borrow().props.words().unwrap().len();
    let length = crate::atom::names::LENGTH.key();
    for n in 0..count {
        let array = made
            .borrow()
            .props
            .get_index(n as u32)
            .unwrap()
            .value()
            .as_obj()
            .unwrap();
        let body = array.borrow();
        assert!(matches!(body.exotic, Exotic::Array), "{n}");
        assert!(body.props.elem_mode.get(), "{n}");
        assert_eq!(
            body.props.entries.keys()[0],
            length,
            "{n}: length is slot 0"
        );
    }
}

/// W2c step 7a: a frozen or sealed array keeps its element words, with the integrity level in
/// the block's flags; every operation keeps the specification's results in every tier.
#[test]
fn frozen_and_sealed_arrays_keep_their_words_and_their_semantics() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let run = |engine: &mut Engine, source: &str| match engine.eval(source, false).unwrap() {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}\n{source}"),
        };
        let result = run(
            &mut engine,
            r#"
            function attempt(f) { try { f(); return 'ok'; } catch (e) { return e.name; } }
            function strictStore(a, i, v) { 'use strict'; a[i] = v; }
            var out = [];
            var f = Object.freeze([1, 2, 3]);
            f[0] = 9; f[5] = 9;
            var d = Object.getOwnPropertyDescriptor(f, 1);
            out.push(Object.isFrozen(f), Object.isSealed(f), f.join(), f.length,
                     d.writable, d.enumerable, d.configurable, delete f[1], Object.keys(f).join());
            out.push(attempt(function () { strictStore(f, 0, 9); }),
                     attempt(function () { f.push(4); }), attempt(function () { f.pop(); }),
                     attempt(function () { f.shift(); }), attempt(function () { f.unshift(0); }),
                     attempt(function () { f.reverse(); }), attempt(function () { f.sort(); }),
                     attempt(function () { f.fill(0); }), attempt(function () { f.copyWithin(0, 1); }),
                     attempt(function () { f.splice(0, 1); }), attempt(function () { f.length = 0; }),
                     attempt(function () { Object.defineProperty(f, 0, { value: 9 }); }),
                     attempt(function () { Object.defineProperty(f, 0, { value: 1 }); }), f.join());
            var s = Object.seal([3, 1, 2]);
            s[0] = 5;
            out.push(Object.isSealed(s), Object.isFrozen(s), s.join(), delete s[0],
                     attempt(function () { s.push(4); }), attempt(function () { s.pop(); }),
                     attempt(function () { s.sort(); }), s.join(), attempt(function () { s.reverse(); }),
                     s.join(), attempt(function () { s.fill(7, 2); }), s.join(),
                     attempt(function () { s.length = 0; }), s.length, s.join());
            Object.defineProperty(s, 1, { writable: false });
            var e = Object.getOwnPropertyDescriptor(s, 1);
            s[1] = 0; s[0] = 8;
            out.push(e.writable, e.configurable, Object.isSealed(s), Object.isFrozen(s), delete s[1],
                     s.join(), Object.keys(s).join());
            var h = [1, , 3]; Object.freeze(h);
            out.push(1 in h, h.length, Object.isFrozen(h), JSON.stringify(f));
            out.join('|')
        "#,
        );
        assert_eq!(
            result,
            // Node 24 prints the same (a sloppy `length` store that fails reports no error).
            "true|true|1,2,3|3|false|true|false|false|0,1,2|\
             TypeError|TypeError|TypeError|TypeError|TypeError|TypeError|TypeError|TypeError|\
             TypeError|TypeError|ok|TypeError|ok|1,2,3|\
             true|false|5,1,2|false|TypeError|TypeError|ok|1,2,5|ok|5,2,1|ok|5,2,7|\
             ok|3|5,2,7|\
             false|false|true|false|false|8,2,7|0,1,2|\
             false|3|true|[1,2,3]",
            "{tier:?}"
        );
        // The representation: the words stay, with the integrity bits.
        let global = Value::Obj(engine.interp.global);
        for (name, bits) in [
            ("f", word_block::KIND_INTEGRITY),
            ("h", word_block::KIND_INTEGRITY),
        ] {
            let array = engine
                .interp
                .get_member_key(&global, name)
                .unwrap_or_else(|_| panic!("{name}"))
                .as_obj()
                .unwrap();
            let body = array.borrow();
            assert!(
                !body.props.has_far.get(),
                "{tier:?} {name}: no named elements"
            );
            assert_eq!(body.props.word_integrity(), bits, "{tier:?} {name}");
        }
    }
}

/// Generated stores, increments and loop chains warmed on writable arrays leave a frozen one
/// as it is (W2c step 7a's guard at every store into words).
#[test]
fn generated_code_never_writes_a_frozen_array() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let result = match engine
        .eval(
            r#"
            function store(a, i, v) { a[i] = v; }
            function storeKeep(a, i, v) { return a[i] = v; }
            function bump(a, i) { a[i]++; }
            function fill(a, n) { for (var i = 0; i < n; i++) a[i] = i * 2; return a; }
            function scale(a, n) { for (var i = 0; i < n; i++) a[i] = a[i] * 3; return a; }
            for (var w = 0; w < 50; w++) {
                var t = [1, 2, 3, 4];
                store(t, 1, 'x'); store(t, 2, 0.5); storeKeep(t, 3, {}); bump(t, 2);
                fill([0, 0, 0, 0], 4); scale([1, 2, 3, 4], 4); scale([1.5, 2.5], 2);
            }
            var f = Object.freeze([1, 2, 3, 4]);
            store(f, 1, 'x'); store(f, 2, 0.5); var k = storeKeep(f, 3, 7); bump(f, 0);
            fill(f, 4); scale(f, 4);
            var s = Object.seal([1, 2, 3, 4]);
            store(s, 1, 'x'); bump(s, 0); scale(s, 4);
            // Doubles: the kind update's second test, where FROZEN shares SCANNED's mask.
            var fd = Object.freeze([1.5, 2.5, 3.5, 4.5]);
            store(fd, 2, 0.25); bump(fd, 0); fill(fd, 4); scale(fd, 4);
            var sd = Object.seal([1.5, 2.5, 3.5, 4.5]);
            store(sd, 2, 0.25); bump(sd, 0); scale(sd, 4);
            [f.join(), k, s.join(), fd.join(), sd.join()].join('|')
        "#,
            false,
        )
        .unwrap()
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    };
    // Node 24 prints the same.
    assert_eq!(
        result,
        "1,2,3,4|7|6,NaN,9,12|1.5,2.5,3.5,4.5|7.5,7.5,0.75,13.5"
    );
}
