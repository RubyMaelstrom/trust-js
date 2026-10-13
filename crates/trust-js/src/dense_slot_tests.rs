//! ECMA-262 e28783d5, spec.html: OrdinaryGetOwnProperty 13212,
//! ValidateAndApplyPropertyDescriptor 13286, OrdinaryDelete 13501,
//! OrdinaryOwnPropertyKeys 13531, Array [[DefineOwnProperty]] 14659.

use crate::bytecode::Tier;
use crate::value::{Property, Props, Value};
use crate::{Completion, Engine};

fn number(prop: Option<&Property>) -> f64 {
    match prop.expect("existing property").value() {
        Value::Num(n) => n,
        _ => panic!("not numeric"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("index fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1+2", false), Ok(Completion::Value(v)) if v == "3"));
    }
}

#[test]
fn dense_numeric_typedarray_copy_keeps_holes_accessors_and_conversion_order() {
    // SetTypedArrayFromArrayLike reads/coerces each element in order. Only already-numeric
    // ordinary own data properties may batch; a failed proof must not partially write.
    check(
        r#"
        var trace=[],out=[],target=new Float64Array(4);
        var source=[1,-0,Infinity,NaN];target.set(source);
        out.push(target[0]===1,Object.is(target[1],-0),target[2]===Infinity,Number.isNaN(target[3]));
        var inherited=Object.create(Array.prototype);
        Object.defineProperty(inherited,'1',{get(){trace.push('inherited');return 7}});
        source=[2,3,4];delete source[1];Object.setPrototypeOf(source,inherited);
        target.set(source);out.push(target[0],target[1],target[2],trace.join(','));
        trace=[];target.fill(0);source=[5,6,7];
        Object.defineProperty(source,'1',{get(){trace.push(target[0]);throw 'sentinel'}});
        try{target.set(source)}catch(e){out.push(e)}
        out.push(target.join(','),trace.join(','));
        trace=[];target.fill(0);source=[8,{valueOf(){trace.push(target[0]);return 9}},10];
        target.set(source);out.push(target.join(','),trace.join(','));
        trace=[];source=new Proxy([11,12],{get(t,k){trace.push(k);return t[k]}});
        target.set(source);out.push(trace.join(','));
        out.join('|');
        "#,
        "true|true|true|true|2|7|4|inherited|sentinel|5,0,0,0|5|8,9,10,0|8|length,0,1",
    );
}

#[test]
fn dense_slot_mutation_removal_and_reinsertion_keep_sidecar_consistent() {
    for force_classic in [false, true] {
        let mut props = Props::new();
        props.mark_array();
        props.insert(
            "length",
            Property::data(Value::Num(4096.0), true, false, false),
        );
        // A first far index selects classic keyed storage; ascending first writes
        // exercise packed storage when enabled. Both must preserve identical properties.
        if force_classic {
            props.insert("4095", Property::plain(Value::Num(4095.0)));
        }
        let packed = !force_classic && crate::value::dense_elements_enabled();
        for n in 0..4096 {
            props.insert(n.to_string(), Property::plain(Value::Num(n as f64)));
        }
        props.insert("named", Property::plain(Value::Num(-1.0)));
        for n in (0..4096).step_by(3) {
            props
                .get_mut(&n.to_string())
                .unwrap()
                .set_value(Value::Num(-(n as f64)));
        }
        for n in (0..4096).step_by(5) {
            assert!(props.remove(&n.to_string()));
            assert!(props.get_mut(&n.to_string()).is_none());
            assert!(props.slot_of(&n.to_string()).is_none());
            assert!(!props.remove(&n.to_string()));
        }
        for n in 0..4096 {
            if n % 5 == 0 {
                props.insert(n.to_string(), Property::plain(Value::Num(999.0)));
            }
            let key = n.to_string();
            let expected = if n % 5 == 0 {
                999.0
            } else if n % 3 == 0 {
                -(n as f64)
            } else {
                n as f64
            };
            assert_eq!(number(props.get(&key)), expected);
            assert_eq!(number(props.get_index(n)), expected);
            if packed {
                // Packed indices have no named entry slot; the general and indexed
                // property reads above must still agree after every structural edit.
                assert!(props.slot_of(&key).is_none());
            } else {
                let (entry_key, entry) = props.entry_at(props.slot_of(&key).unwrap()).unwrap();
                assert_eq!(&**entry_key, key);
                assert_eq!(number(Some(entry)), expected);
            }
        }
        assert_eq!(number(props.length_property()), 4096.0);
        assert_eq!(number(props.get("named")), -1.0);
        props.remove_indices_from(100);
        assert!(props.get_mut("100").is_none());
        props.insert("101", Property::plain(Value::Num(101.0)));
        assert_eq!(number(props.get_index(101)), 101.0);
    }
}

#[test]
fn dense_slot_far_indices_and_noncanonical_keys_remain_distinct() {
    check(
        r#"
        function run() {
            var a = new Array(100), out = [], sym = Symbol('index');
            for (var i = 0; i < 100; i++) a[i] = i;
            a[1000000] = 10; a[4294967294] = 20;
            a['01'] = 30; a['-0'] = 40; a['4294967295'] = 50;
            a['1e0'] = 60; a[sym] = 70;
            a[1000000]++; a[4294967294]++;
            delete a[42]; delete a[1000000];
            out.push(a.length, Object.hasOwn(a, 42), Object.hasOwn(a, 1000000));
            a[42] = 142; a[1000000] = 11;
            out.push(a[42], a[1000000], a[4294967294], a['01'], a['-0'],
                     a['4294967295'], a['1e0'], a[sym]);
            var keys = Reflect.ownKeys(a);
            out.push(keys[100], keys[101], keys[102], keys[103], keys[keys.length-1] === sym);
            a.length = 50;
            out.push(a.length, Object.hasOwn(a, 1000000), a['4294967295']);
            var o = {0: 1, 1: 2, '01': 3}; o[1] = 4; delete o[0]; o[0] = 5;
            out.push(JSON.stringify(o));
            return out.join('|');
        }
        run()
        "#,
        "4294967295|false|false|142|11|21|30|40|50|60|70|1000000|4294967294|length|01|true|50|false|50|{\"0\":5,\"1\":4,\"01\":3}",
    );
}

#[test]
fn dense_slot_descriptors_prototypes_and_failed_truncation() {
    check(
        r#"
        function run() {
            var a = new Array(100), out = [], calls = [];
            for (var i = 0; i < 100; i++) a[i] = i;
            Object.defineProperty(a, '42', {get() { calls.push('get'); return 7; },
                set(v) { calls.push('set:' + v); }, configurable: true});
            a[42] += 1; out.push(calls.join(','));
            Object.defineProperty(a, '42', {value: 42, writable: false, configurable: false});
            a[42] = 99; out.push(a[42], delete a[42]);
            Object.defineProperty(a, '70', {value: 70, configurable: false});
            out.push(Reflect.defineProperty(a, 'length', {value: 40, writable: false}));
            out.push(a.length, Object.hasOwn(a, 71), Object.hasOwn(a, 70),
                     Object.getOwnPropertyDescriptor(a, 'length').writable);
            var b = new Array(100), proto = Object.create(Array.prototype);
            Object.defineProperty(proto, '42', {set(v) { calls.push('proto:' + v); }, configurable: true});
            Object.setPrototypeOf(b, proto); b[42] = 88;
            out.push(Object.hasOwn(b, 42), calls[calls.length-1]);
            var c = [1, 2, 3]; delete c[1]; c[1000000] = 8; c[1] = 9;
            out.push(c[0], c[1], c[2], c[1000000]);
            return out.join('|');
        }
        run()
        "#,
        "get,set:8|42|false|false|71|false|true|false|false|proto:88|1|9|3|8",
    );
}

#[test]
fn dense_named_lookup_preserves_own_keys_prototypes_and_index_namespaces() {
    // ECMA-262 OrdinaryGetOwnProperty/OrdinaryGet: fast absence is not a
    // prototype result, and any named insertion must invalidate its proof.
    check(
        r#"
        function run() {
            var a = Array.from({length: 1024}, (_, i) => i), out = [];
            var key = ['push'][0];
            a[key](1024);
            out.push(a.length, a[1024], Object.hasOwn(a, key));
            var count = 0, receiver = false, proto = Object.create(Array.prototype);
            Object.defineProperty(proto, key, {configurable:true, get() {
                count++; receiver = this === a; return Array.prototype.push;
            }});
            Object.setPrototypeOf(a, proto);
            a[key](1025);
            out.push(count, receiver, a.length);
            Object.defineProperty(a, key, {value: 73, writable: true, configurable: true});
            out.push(a[key], Object.hasOwn(a, key));
            delete a[key];
            out.push(a[key] === Array.prototype.push, count);
            var sym = Symbol('named'); a[sym] = 81;
            a['01'] = 82; a['4294967295'] = 83; a[1000000] = 84;
            out.push(a[sym], a['01'], a['4294967295'], a[1000000]);
            a.length = 2;
            out.push(a[1000000], a[sym], a['01'], a['4294967295']);
            var b = Array.from({length: 1024}, (_, i) => i), traps = 0;
            Object.setPrototypeOf(b, new Proxy(Array.prototype, {get(t,k,r) {
                if (k === key) traps++; return Reflect.get(t,k,r);
            }}));
            b[key](1024); out.push(traps, b.length);
            return out.join('|');
        }
        run()
        "#,
        "1025|1024|false|1|true|1026|73|true|true|2|81|82|83|84||81|82|83|1|1025",
    );
}
