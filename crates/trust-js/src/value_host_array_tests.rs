//! CreateArrayFromList / ArrayCreate / Array exotic [[DefineOwnProperty]],
//! ECMA-262 e28783d5. Host-produced lists retain the ordinary observable contract.
use super::*;
use crate::{bytecode::Tier, Completion, Engine};

#[test]
fn host_arrays_move_owners_and_preserve_descriptors_at_storage_boundaries() {
    let engine = Engine::new();
    for len in [0, 1, 10, 11, 32, 33] {
        let owner = engine.interp.new_object();
        let weak = Rc::downgrade(&owner);
        let array = engine
            .interp
            .make_array((0..len).map(|_| Value::Obj(owner.clone())).collect());
        assert_eq!(Rc::strong_count(&owner), len + 1);
        {
            let array = array.as_obj().unwrap().borrow();
            assert!(Rc::ptr_eq(
                array.proto.as_ref().unwrap(),
                &engine.interp.array_proto
            ));
            assert!(matches!(array.exotic, Exotic::Array));
            let length = array.props.length_property().unwrap();
            assert_eq!(length.number_value(), Some(len as f64));
            assert!(length.writable() && !length.enumerable() && !length.configurable());
            for index in 0..len {
                let property = array.props.get_index(index as u32).unwrap();
                assert!(property.writable() && property.enumerable() && property.configurable());
                assert!(Rc::ptr_eq(property.value().as_obj().unwrap(), &owner));
            }
            if (1..=32).contains(&len) {
                assert!(
                    array.props.elems.packed_ref().is_some(),
                    "host path actually uses compact storage"
                );
            }
        }
        drop(owner);
        assert_eq!(weak.strong_count(), len);
        drop(array);
        assert!(
            weak.upgrade().is_none(),
            "the array relinquishes every moved owner exactly once"
        );
    }
}

#[test]
fn host_arrays_keep_holes_getters_length_and_numeric_values_across_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let held = Value::Obj(engine.interp.new_object());
        let array = engine.interp.make_array(vec![
            held.clone(),
            Value::Num(-0.0),
            Value::Num(f64::NAN),
            Value::lstr("text"),
            Value::Undefined,
        ]);
        engine
            .interp
            .global
            .borrow_mut()
            .props
            .insert("hostArray", Property::plain(array));
        engine
            .interp
            .global
            .borrow_mut()
            .props
            .insert("held", Property::plain(held));
        let result = engine.eval(r#"
            function check() {
                var a=hostArray,trace=[],d=Object.getOwnPropertyDescriptor(a,'length');
                var valid=Object.getPrototypeOf(a)===Array.prototype && a[0]===held &&
                    Object.is(a[1],-0) && Number.isNaN(a[2]) && a[3]==='text' && a[4]===undefined &&
                    Object.keys(a).join(',')==='0,1,2,3,4' && d.writable && !d.enumerable && !d.configurable;
                Object.defineProperty(a,'1',{get(){trace.push('get');return 7;},configurable:true});
                valid=valid && a[1]===7;
                delete a[0]; Array.prototype[0]=45;
                valid=valid && a[0]===45 && !Object.hasOwn(a,'0');
                a.length=1; a.push(held);
                valid=valid && a.shift()===45 && a[0]===held && a.length===1;
                Object.defineProperty(a,'length',{writable:false});
                var error='';try{a.push('rejected');}catch(e){error=e.name;}
                return [valid,trace.join(','),error,Object.keys(a).join(','),a[0]===held].join('|');
            }
            check();
        "#,false).unwrap();
        match result {
            Completion::Value(value) => assert_eq!(value, "true|get|TypeError|0|true"),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
}
