//! OrdinaryGet regression and actual-native-path coverage (ECMA-262 e28783d5).
use super::tests::{check, check_warmed, check_warmed_then};
use super::*;

#[test]
fn warm_monomorphic_reads_compile_compact_guards_without_refilling() {
    property::TEST_READ_HELPERS.with(|count| count.set(0));
    property::TEST_COMPACT_READS.with(|count| count.set(0));
    check_warmed(
        r#"
        function subject() {
            var row={value:7},sum=0;
            for(var k=0;k<50;k++)sum+=row.value;
            return sum;
        }
        "#,
        "350",
    );
    assert_eq!(property::TEST_READ_HELPERS.with(|count| count.get()), 0);
    assert_eq!(property::TEST_COMPACT_READS.with(|count| count.get()), 3);
}

#[test]
fn compact_guards_still_observe_live_descriptors_and_values() {
    property::TEST_COMPACT_READS.with(|count| count.set(0));
    check_warmed_then(
        r#"
        function subject() {
            var row={value:7},sum=0,gets=0;
            for(var k=0;k<48;k++) {
                if(mutate && k===32)Object.defineProperty(row,'value',{
                    get:function(){gets++;return 9;},configurable:true});
                if(mutate && k===40)Object.defineProperty(row,'value',{
                    value:11,writable:true,configurable:true});
                sum+=row.value;
            }
            return sum+'|'+gets;
        }
        var mutate=false;
        "#,
        "336|0",
        "mutate=true;",
        "384|8",
    );
    // Only row.value ran during warmup: neither defineProperty method site may count as
    // coverage. Each independently compiled engine must specialize the changing data field.
    assert_eq!(property::TEST_COMPACT_READS.with(|count| count.get()), 3);
}

#[test]
fn compact_prototype_guards_follow_replacement_shadowing_and_deletion() {
    property::TEST_COMPACT_READS.with(|count| count.set(0));
    check_warmed_then(
        r#"
        function subject() {
            var row=root,total=0,gets=0;
            for(var k=0;k<32;k++) {
                if(mutate && k===8)Object.defineProperty(proto,'value',{
                    get:function(){gets++;return this.marker+8;},configurable:true});
                if(mutate && k===12)Object.defineProperty(proto,'value',{
                    value:5,writable:true,configurable:true});
                if(mutate && k===16)middle.value=11;
                if(mutate && k===20)delete middle.value;
                if(mutate && k===24)Object.setPrototypeOf(middle,{value:13});
                if(mutate && k===28)delete Object.getPrototypeOf(middle).value;
                total+=row.value||0;
            }
            return total+'|'+gets;
        }
        var mutate=false,proto={value:7},middle=Object.create(proto),root=Object.create(middle);
        root.marker=1;
        "#,
        "224|0",
        "mutate=true;",
        "228|4",
    );
    assert_eq!(property::TEST_COMPACT_READS.with(|count| count.get()), 3);
}

#[test]
fn compact_receiver_proof_rejects_proxy_replacement() {
    property::TEST_COMPACT_READS.with(|count| count.set(0));
    check_warmed_then(
        r#"
        function subject() {
            var row={value:1},total=0,trace='';
            for(var k=0;k<9;k++) {
                if(mutate && k===3)row=new Proxy(row,{get:function(t,key,r){
                    trace+=key+',';return Reflect.get(t,key,r)*2;}});
                if(mutate && k===6)row.value=5;
                total+=row.value;
            }
            return total+'|'+trace;
        }
        var mutate=false;
        "#,
        "9|",
        "mutate=true;",
        "39|value,value,value,value,value,value,",
    );
    assert_eq!(property::TEST_COMPACT_READS.with(|count| count.get()), 3);
}

#[test]
fn compact_named_shapes_do_not_confuse_array_owned_length_with_inheritance() {
    property::TEST_COMPACT_READS.with(|count| count.set(0));
    check_warmed_then(
        r#"
        function subject() {
            var row=ordinary,total=0;
            for(var k=0;k<32;k++) {
                if(mutate && k===8)row=array;
                if(mutate && k===16)array.value=11;
                if(mutate && k===24)delete array.value;
                total+=row.value*100+row.length;
            }
            return total;
        }
        var mutate=false,proto={value:7},ordinary=Object.create(proto);
        ordinary.length=3;
        var array=[1,2];Object.setPrototypeOf(array,proto);
        "#,
        "22496",
        "mutate=true;",
        "25672",
    );
    // Both ordinary sites specialize during warmup. ArrayCreate's own non-configurable
    // length and the later named shadow must still win over any inherited data property.
    assert_eq!(property::TEST_COMPACT_READS.with(|count| count.get()), 6);
}

#[test]
fn ordinary_field_loop_uses_one_cache_fill_not_one_helper_per_read() {
    property::TEST_READ_HELPERS.with(|count| count.set(0));
    check(
        r#"
        function subject() {
            var row={value:7}, total=0;
            for(var k=0;k<50;k++)total+=row.value;
            return total;
        }
        "#,
        "350",
    );
    // check() explicitly compiles the subject three times in independent engines. Each
    // function starts with an empty property cache: exactly its first read must fill it.
    assert_eq!(property::TEST_READ_HELPERS.with(|count| count.get()), 3);
}

#[test]
fn polymorphic_fields_keep_all_four_live_shape_ways_native() {
    property::TEST_READ_HELPERS.with(|count| count.set(0));
    check(
        r#"
        function subject() {
            var rows=[{value:1},{tag:0,value:2},{extra:0,tag:0,value:3},
                {before:0,extra:0,tag:0,value:4}], total=0;
            for(var k=0;k<120;k++){var row=rows[k%4];total+=row.value;}
            return total;
        }
        "#,
        "300",
    );
    assert_eq!(property::TEST_READ_HELPERS.with(|count| count.get()), 12);
}

#[test]
fn live_prototype_descriptors_shadowing_and_same_shape_replacement_are_observed() {
    check(
        r#"
        function subject() {
            var proto={value:7}, middle=Object.create(proto), row=Object.create(middle);
            row.marker=1;
            var total=0,trace=[];
            for(var k=0;k<32;k++) {
                if(k===8)Object.defineProperty(proto,'value',{
                    get:function(){trace.push('g');return this.marker+8;},configurable:true});
                if(k===12)Object.defineProperty(proto,'value',{
                    value:5,writable:true,configurable:true});
                if(k===16)middle.value=11;
                if(k===20)delete middle.value;
                if(k===24)Object.setPrototypeOf(middle,{value:13});
                if(k===28)delete Object.getPrototypeOf(middle).value;
                total+=row.value||0;
            }
            return total+'|'+trace.join(',');
        }
        "#,
        "228|g,g,g,g",
    );
}

#[test]
fn proxy_replacement_cannot_reuse_an_ordinary_receiver_proof() {
    check(
        r#"
        function subject() {
            var trace=[],row={value:1},total=0;
            for(var k=0;k<9;k++) {
                if(k===3)row=new Proxy(row,{get:function(t,key,r){
                    trace.push(key);return Reflect.get(t,key,r)*2;}});
                if(k===6)row.value=5;
                total+=row.value;
            }
            return total+'|'+trace.join(',');
        }
        "#,
        "39|value,value,value,value,value,value",
    );
}

#[test]
fn field_values_keep_every_owner_and_number_identity() {
    check(
        r#"
        function subject() {
            var held={},symbol=Symbol('field'),values=[undefined,null,false,true,0,-0,NaN,
                Infinity,-Infinity,'owned string',symbol,held,123456789012345678901234567890n];
            var row={value:undefined},matched=0;
            for(var k=0;k<values.length;k++) {
                row.value=values[k];
                if(Object.is(row.value,values[k]))matched++;
            }
            function create(){return {value:held};}
            var last=create().value;
            var cycle={};cycle.self=cycle;
            for(var k=0;k<40;k++)cycle=cycle.self;
            function constructor(){}
            var prototype=constructor.prototype;
            return [matched,last===held,cycle===cycle.self,
                prototype===constructor.prototype,prototype.constructor===constructor].join('|');
        }
        "#,
        "13|true|true|true|true",
    );
}

#[test]
fn deep_methods_retain_receiver_and_array_index_fallbacks() {
    check(
        r#"
        function subject() {
            var proto={read:function(){return this.marker;}},row=proto;
            for(var k=0;k<5;k++)row=Object.create(row);
            row.marker=7;
            var sum=0;for(var k=0;k<30;k++)sum+=row.read();
            var arrays=[[1],[1,2],[1,2,3]],lengths=0;
            for(var k=0;k<30;k++){var a=arrays[k%3];lengths+=a.length+a[0];}
            return sum+'|'+lengths;
        }
        "#,
        "210|90",
    );
}
