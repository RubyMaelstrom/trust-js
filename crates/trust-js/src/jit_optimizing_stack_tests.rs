//! Exceptional and observable lifetime boundaries for operand SSA.
use super::tests::{check, check_with_installer};
use crate::value::Value;
use crate::{interpreter::Interp, Engine};
use std::rc::Rc;

#[test]
fn expression_suffixes_merge_through_branches_calls_and_coercions() {
    check(
        r#"
        function subject() {
            var trace=[], a={value:7}, b={value:11};
            function record(x) { trace.push(x.value); return x; }
            var sum=0;
            for(var k=0;k<20;k++) {
                var pair=[a,k%2?record(a):record(b),a===b?b:a];
                sum=sum+pair[0].value+pair[1].value+pair[2].value;
            }
            var left={valueOf(){trace.push('L');return 3;}};
            var right={valueOf(){trace.push('R');return 5;}};
            return sum+'|'+(left+right)+'|'+trace.length+'|'+trace.slice(-2).join('');
        }
    "#,
        "460|8|22|LR",
    );
}

#[test]
fn throwing_destructuring_consumes_inputs_without_reviving_saved_prefix() {
    check(
        r#"
        function subject() {
            var trace=[], keep={value:13};
            var source={ [Symbol.iterator](){return {
                next(){return {value:undefined,done:false};},
                return(){trace.push('close');throw 'close error';}
            };}};
            function fail(){trace.push('default');throw 'original';}
            try { let [a=fail()]=source;trace.push('unreachable'); }
            catch(e) {trace.push(e);}
            finally {trace.push(keep.value);}
            try {let [a]=null;trace.push('unreachable');}
            catch(e) {trace.push(e.name);}
            return trace.join('|');
        }
    "#,
        "default|close|original|13|TypeError",
    );
}

#[test]
fn abrupt_jump_finalizers_join_normal_expression_suffixes() {
    check(
        r#"
        function subject() {
            var trace=[], keep={value:2}, sum=0;
            outer:for(var k=0;k<6;k++) {
                try {
                    try {
                        sum=sum+(k%2?keep.value:3);
                        if(k===1)continue outer;
                        if(k===4)break outer;
                        trace.push(k);
                    } finally {sum=sum+10;trace.push('i'+k);}
                } finally {sum=sum+100;trace.push('o'+k);}
            }
            return sum+'|'+trace.join(',');
        }
    "#,
        "563|0,i0,o0,i1,o1,2,i2,o2,3,i3,o3,i4,o4",
    );
}

thread_local! {
    static LIVE_CYCLE: std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>> = const { std::cell::RefCell::new(None) };
    static COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn make_tracked(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    let object = i.new_object();
    object
        .borrow_mut()
        .props
        .insert("value", crate::value::Property::plain(Value::Num(7.0)));
    object.borrow_mut().props.insert(
        "self",
        crate::value::Property::plain(Value::Obj(object.clone())),
    );
    LIVE_CYCLE.with(|root| *root.borrow_mut() = Some(Rc::downgrade(&object)));
    Ok(Value::Obj(object))
}

fn force_collection(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    i.gc_collect();
    LIVE_CYCLE.with(|root| {
        assert!(
            root.borrow().as_ref().unwrap().upgrade().is_some(),
            "live operand/local cycle must remain rooted across the native callback"
        )
    });
    COLLECTIONS.with(|count| count.set(count.get() + 1));
    Ok(Value::Undefined)
}

#[test]
fn unpublished_operand_owner_is_materialized_before_reentrant_collection() {
    COLLECTIONS.with(|count| count.set(0));
    check_with_installer(
        r#"
        function subject() {
            var pair=[makeTracked(),(forceCollection(),9)];
            var owned=pair[0];pair=null;
            return owned.value+(forceCollection(),1);
        }
    "#,
        "8",
        |engine: &mut Engine| {
            let global = engine.interp.global.clone();
            engine
                .interp
                .def_method(&global, "makeTracked", 0, make_tracked);
            engine
                .interp
                .def_method(&global, "forceCollection", 0, force_collection);
        },
    );
    // Three native invocations and their three independent interpreter oracles, twice each.
    assert_eq!(COLLECTIONS.with(|count| count.get()), 12);
}

#[test]
fn dead_register_copies_leave_canonical_local_owners_rooted() {
    COLLECTIONS.with(|count| count.set(0));
    check_with_installer(
        r#"
        function subject() {
            var owned=makeTracked();
            var result=owned.value;
            forceCollection();
            forceCollection();
            return result;
        }
        "#,
        "7",
        |engine: &mut Engine| {
            let global = engine.interp.global.clone();
            engine
                .interp
                .def_method(&global, "makeTracked", 0, make_tracked);
            engine
                .interp
                .def_method(&global, "forceCollection", 0, force_collection);
        },
    );
    // `owned` has no later SSA read, but this optimization removes only its
    // redundant register copy. A host callback still sees a canonical root.
    assert_eq!(COLLECTIONS.with(|count| count.get()), 12);
    LIVE_CYCLE.with(|root| assert!(root.borrow().as_ref().unwrap().upgrade().is_none()));
}

#[test]
fn sparse_local_reloads_preserve_normal_finally_and_loop_join_owners() {
    check(
        r#"
        function subject() {
            var owner={v:7}, n=1, trace=[];
            outer:for(var i=0;i<5;i++) {
                n=n+2;
                try {
                    if(i===1)continue outer;
                    if(i===3)break outer;
                    var snapshot=owner;
                    trace.push(snapshot.v+n);
                } finally {
                    n=n+10;
                    owner={v:n};
                    trace.push('f'+n);
                }
            }
            return trace.join(',')+'|'+n+'|'+owner.v+'|'+i;
        }
        "#,
        "10,f13,f25,52,f37,f49|49|49|3",
    );
}

#[test]
fn private_locals_survive_calls_that_mutate_heap_shapes_and_captured_bindings() {
    check(
        r#"
        function subject() {
            var captured=1, stable=17, receiver={value:3}, trace=[];
            function change(object) {
                captured+=2;
                delete object.value;
                Object.setPrototypeOf(object,{get value(){captured++;return 11;}});
                return 5;
            }
            for(var n=0;n<4;n++) {
                stable+=change(receiver);
                trace.push(stable,receiver.value,captured);
            }
            return trace.join('|');
        }
        "#,
        "22|11|4|27|11|7|32|11|10|37|11|13",
    );
}

#[test]
fn private_word_effects_keep_eval_arguments_and_coercion_bindings_observable() {
    super::tests::check_real_call(
        r#"
        function subject() {
            var local=3, trace=[];
            function mapped(a) {
                function write(args) { args[0]=9; }
                write(arguments);
                return a;
            }
            function strict(a) {
                'use strict';
                function write(args) { args[0]=9; }
                write(arguments);
                return a;
            }
            function mutate() { eval('local=21'); }
            mutate();
            trace.push(local,mapped(4),strict(4));
            var object={valueOf(){eval('local+=2');return 7;}};
            trace.push(object+1,local);
            eval('local=31');
            trace.push(local);
            return trace.join('|');
        }
        "#,
        "21|9|4|8|23|31",
    );
}

#[test]
fn private_word_effects_preserve_destructuring_progress_on_abrupt_completion() {
    check(
        r#"
        function subject() {
            var local=13, first=1, second=2, trace=[];
            var iterator={ [Symbol.iterator](){ return {
                next(){return {value:undefined,done:false};},
                return(){trace.push('closed');return {};}
            };}};
            function fail(){trace.push('default');throw 'stop';}
            try { [first=7,second=fail()]=iterator; }
            catch(e) {trace.push(e,first,second,local);}
            var source={a:1,b:2,c:3};
            for(var key in source) {trace.push(key);delete source.b;}
            return trace.join('|');
        }
        "#,
        "default|closed|stop|7|2|13|a|c",
    );
}

#[test]
fn private_word_effects_keep_heap_identity_rooted_across_reentrant_collection() {
    COLLECTIONS.with(|count| count.set(0));
    check_with_installer(
        r#"
        function subject() {
            var owner=makeTracked(), total=0;
            for(var n=0;n<4;n++) {
                forceCollection();
                total+=owner.value;
                owner.value++;
            }
            return total+'|'+owner.value;
        }
        "#,
        "34|11",
        |engine: &mut Engine| {
            let global = engine.interp.global.clone();
            engine
                .interp
                .def_method(&global, "makeTracked", 0, make_tracked);
            engine
                .interp
                .def_method(&global, "forceCollection", 0, force_collection);
        },
    );
    assert_eq!(COLLECTIONS.with(|count| count.get()), 24);
    LIVE_CYCLE.with(|root| assert!(root.borrow().as_ref().unwrap().upgrade().is_none()));
}
