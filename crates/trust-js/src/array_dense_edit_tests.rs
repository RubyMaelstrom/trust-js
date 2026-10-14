//! ECMA-262 e28783d5: Array.prototype.unshift, splice, reverse, slice and concat over plain
//! dense elements (`edit_plain_elements`, `own_data_elements`) and every fallback to the
//! specified HasProperty/Get/Set/DeletePropertyOrThrow steps. Expected values from Node v24.
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("array fixture parses") {
            Completion::Value(actual) => assert_eq!(actual, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("21+21", false), Ok(Completion::Value(v)) if v == "42"));
    }
}

#[test]
fn plain_element_edits_match_the_specified_steps_and_fall_back_on_every_observer() {
    check(
        r#"(function(){
    var log=[],r=[];
    function show(f){try{return JSON.stringify(f())}catch(e){return e.name}}
    function keys(a){return Object.keys(a).join(',')+'/'+a.length}
    // plain
    var a=[1,2,3,4,5];r.push(a.unshift(0,0.5),a.join(),a.reverse().join(),a.splice(1,2,'x','y','z').join(),a.join(),a.slice(2,5).join(),a.slice(-2).join(),a.splice(0).join(),keys(a));
    a=[1,2,3];r.push(a.splice(1,0,'i').length,a.join(),a.splice(-1,5).join(),a.join(),a.splice(1,1).join(),a.join(),a.splice().length,a.join());
    // holes
    a=[1,,3,,5];r.push(a.unshift(0),keys(a),a.reverse().length,keys(a),keys(a.splice(1,3,'h')),keys(a),keys(a.slice(0,3)));
    Array.prototype[1]='P';a=[0,,2];r.push(a.slice().join(),a.reverse().join(),keys(a),a.splice(0,2).join());delete Array.prototype[1];
    // getters
    a=[1,2,3];Object.defineProperty(a,1,{get(){log.push('g');return 'G'},set(v){log.push('s'+v)},configurable:true});
    r.push(a.slice().join(),a.reverse().join(),log.splice(0).join());
    a=[1,2,3];Object.defineProperty(a,0,{get(){log.push('g0');return 'G'},configurable:true});r.push(show(()=>a.unshift('u')),a.join(),log.splice(0).join());
    // non-writable / non-configurable / frozen / sealed / non-extensible
    a=[1,2,3];Object.defineProperty(a,1,{writable:false});r.push(show(()=>a.reverse()),show(()=>a.unshift(0)),a.join());
    a=[1,2,3];Object.defineProperty(a,2,{configurable:false});r.push(show(()=>a.splice(0,1)),a.join(),show(()=>a.splice(1,0,'z')),a.join());
    a=Object.freeze([1,2]);r.push(show(()=>a.reverse()),show(()=>a.unshift(1)),show(()=>a.splice(0,1)),show(()=>a.slice(0,1)));
    a=Object.seal([1,2]);r.push(show(()=>a.reverse()),show(()=>a.unshift(1)),show(()=>a.splice(0,1)),a.join());
    a=Object.preventExtensions([1,2,3]);r.push(show(()=>a.splice(0,1)),a.join(),show(()=>a.unshift(9)),a.join(),show(()=>a.splice(0,0,8)),a.join());
    // non-writable length
    a=[1,2,3];Object.defineProperty(a,'length',{writable:false});r.push(show(()=>a.splice(0,1)),a.join(),show(()=>a.unshift(0)),show(()=>a.reverse()));
    // indexed setter on the prototype sees growth
    Object.defineProperty(Array.prototype,3,{configurable:true,set(v){log.push('ps'+v)},get(){return 'pg'}});
    a=[1,2,3];r.push(a.unshift(0),keys(a),log.splice(0).join());a=[1,2,3];r.push(a.splice(1,0,'q').length,keys(a),log.splice(0).join());
    delete Array.prototype[3];
    Object.defineProperty(Object.prototype,4,{configurable:true,set(v){log.push('os'+v)}});
    a=[1,2,3,4];r.push(a.unshift(0),keys(a),log.splice(0).join());delete Object.prototype[4];
    // species
    class A extends Array{}var s=A.from([1,2,3]);r.push(s.splice(0,1) instanceof A,s.slice() instanceof A,s.reverse() instanceof A);
    a=[1,2,3];a.constructor={[Symbol.species]:function(n){log.push('species'+n);return {}}};r.push(show(()=>a.splice(0,1)),show(()=>a.slice(1)),log.splice(0).join());
    var Sp=Object.getOwnPropertyDescriptor(Array,Symbol.species);Object.defineProperty(Array,Symbol.species,{configurable:true,get(){log.push('sp');return undefined}});
    r.push([5,6].slice().join(),[5,6].splice(0,1).join(),log.splice(0).join());Object.defineProperty(Array,Symbol.species,Sp);
    // argument coercions that mutate the receiver
    a=[1,2,3,4];r.push(a.slice({valueOf(){a.length=2;return 0}}).length,keys(a.slice(0,{valueOf(){a.push(9);return 3}})));
    a=[1,2,3,4];r.push(a.splice({valueOf(){a.length=1;return 0}},2).join(),keys(a));
    // proxies, array-likes, typed arrays, strings
    var p=new Proxy([1,2,3],{set(t,k,v,rc){log.push('set'+String(k));return Reflect.set(t,k,v,rc)},deleteProperty(t,k){log.push('del'+String(k));return Reflect.deleteProperty(t,k)}});
    r.push(p.unshift(0),p.reverse().length,p.splice(1,1).join(),log.splice(0).join());
    var al={length:3,0:'a',1:'b',2:'c'};r.push(Array.prototype.reverse.call(al)[0],Array.prototype.unshift.call(al,'z'),Array.prototype.splice.call(al,1,2).join(),Array.prototype.slice.call(al).join(),keys(al));
    var ta=new Uint8Array([1,2,3]);r.push(Array.prototype.reverse.call(ta).join(),Array.prototype.slice.call(ta,1).join());
    r.push(show(()=>Array.prototype.reverse.call('ab')),Array.prototype.slice.call('abc',1).join());
    // object and mixed values keep identity
    var o1={},o2={};a=[o1,'s',o2,1.5,-0,NaN];a.reverse();r.push(a[5]===o1,a[3]===o2,Object.is(a[1],-0),Number.isNaN(a[0]));
    a.splice(1,1);a.unshift(o2);r.push(a[0]===o2,a[1]===NaN,a.length);
    // numbers: mirror rebuilt
    a=[1,2,3,4,5,6,7,8,9,10,11,12];var t=0;for(var k=0;k<a.length;k++)t+=a[k];a.reverse();a.splice(3,2);a.unshift(100);for(var k=0;k<a.length;k++)t+=a[k];r.push(t,a.join());
    return r.join('|');
})()"#,
        "7|0,0.5,1,2,3,4,5|5,4,3,2,1,0.5,0|4,3|5,x,y,z,2,1,0.5,0|y,z,2|0.5,0|5,x,y,z,2,1,0.5,0|/0\
         |0|1,i,2,3|3|1,i,2|i|1,2|0|1,2|6|0,1,3,5/6|6|0,2,4,5/6|1/3|0,1,2,3/4|0,1,2/3|0,P,2|2,P,0\
         |0,2/3|2,P|1,G,3|3,G,1|g,g|TypeError|G,G,2,3|g0,g0|[3,2,1]|TypeError|3,2,2,1|TypeError|2\
         ,3,3|[]|2,z,3,3|TypeError|TypeError|TypeError|[1]|[2,1]|TypeError|TypeError|1,1|[1]|2,3|\
         TypeError|2,3|TypeError|2,3|TypeError|2,3,|TypeError|[3,2,0]|4|0,1,2/4|ps3|0|0,1,2/4|ps3\
         |5|0,1,2,3/5|os4|true|true|true|{\"0\":1,\"length\":1}|{\"0\":3,\"length\":1}|species1,s\
         pecies1|5,6|5|sp,sp|4|0,1/2|1,|/2|4|4|2|set3,set2,set1,set0,setlength,set0,set3,set1,set\
         2,set1,set2,del3,setlength|c|4|c,b|z,a|0,1,length/2|3,2,1|2,1|TypeError|b,c|true|true|tr\
         ue|true|true|false|6|239|100,12,11,10,7,6,5,4,3,2,1",
    );
}

#[test]
fn concat_copies_own_data_elements_and_keeps_spreadability_and_hole_steps() {
    check(
        r#"(function(){
    var log=[],r=[];
    function keys(a){return Object.keys(a).join(',')+'/'+a.length}
    var a=[1,2],b=[3,,5],c=[6];
    r.push(keys(a.concat(b,c,7,[[8]])),a.concat().join());
    Object.defineProperty(b,0,{get(){log.push('g');b.length=1;return 'G'},configurable:true});r.push(keys(a.concat(b,c)),log.splice(0).join());
    c[Symbol.isConcatSpreadable]=false;r.push(a.concat(c).length);
    var al={length:2,0:'x',1:'y',[Symbol.isConcatSpreadable]:true};r.push(a.concat(al).join());
    Object.defineProperty(Array.prototype,Symbol.isConcatSpreadable,{configurable:true,get(){log.push('spread:'+this.length);return undefined}});
    r.push(a.concat([9]).join(),log.splice(0).join());delete Array.prototype[Symbol.isConcatSpreadable];
    var d=[1,2,3];Object.defineProperty(Array.prototype,Symbol.isConcatSpreadable,{configurable:true,get(){if(this===d)d.length=1;return undefined}});
    r.push(keys([0].concat(d)));delete Array.prototype[Symbol.isConcatSpreadable];
    Array.prototype[1]='P';r.push(keys([0].concat([1,,3])));delete Array.prototype[1];
    class A extends Array{};r.push(A.from([1]).concat([2]) instanceof A);
    var p=new Proxy([1,2],{get(t,k){log.push(typeof k==='symbol'?'sym':k);return t[k]}});r.push([0].concat(p).join(),log.splice(0).join());
    var o={};r.push([o].concat([o])[1]===o);
    return r.join('|');
})()"#,
        "0,1,2,4,5,6,7/8|1,2|0,1,2,5/6|g|3|1,2,x,y|1,2,9|spread:2,spread:1|0,1/2|0,1,2,3/4|true|0\
         ,1,2|sym,length,0,1|true",
    );
}

#[test]
fn concat_species_writes_recheck_source_properties_and_aliases() {
    check(
        r#"(function(){
        var a=[1,2,3],log=[];
        a.constructor={[Symbol.species]:function(){
            return new Proxy([],{
                defineProperty(target,key,descriptor){
                    log.push(key+':'+descriptor.value);
                    if(key==='0')delete a[1];
                    if(key==='2')a[3]=4;
                    return Reflect.defineProperty(target,key,descriptor);
                }
            });
        }};
        var result=a.concat();
        if(!(0 in result)||(1 in result)||!(2 in result)||result.length!==3)
            throw 'species must recheck HasProperty';
        if(log.join('|')!=='0:1|2:3|length:3')throw 'define order '+log.join('|');
        var b=[4,5,6];b.constructor={[Symbol.species]:function(){return b;}};
        // A destination that is also a later spread source overwrites source
        // indices ahead of that source's subsequent Get operations.
        var c=[1,2];c.constructor={[Symbol.species]:function(){return b;}};
        var aliased=c.concat(b);
        return result.join(',')+'|'+a.join(',')+'|'+aliased.join(',');
        })()"#,
        "1,,3|1,,3,4|1,2,1,2,1",
    );
}

#[test]
fn plain_element_edits_keep_values_alive_and_numeric_reads_coherent() {
    check(
        r#"(function(){
        var a=[],objects=[];
        for(var k=0;k<40;k++){objects.push({k:k});a.push(objects[k])}
        objects=null;
        a.reverse();a.splice(10,5,{k:'x'});a.unshift({k:'u'},{k:'v'});
        var b=a.slice(3,20).concat(a.slice(0,3));
        a=null;
        var n=[1.5,2,3,4,5,6,7,8,9,10,11,12],t=0;
        for(var r=0;r<3;r++){n.reverse();n.splice(2,1);n.unshift(r);for(var k=0;k<n.length;k++)t+=n[k]}
        return b.map(function(o){return o.k}).join()+'|'+t+'|'+n.join();
    })()"#,
        "38,37,36,35,34,33,32,31,30,x,24,23,22,21,20,19,18,u,v,39|192.5|2,0,12,9,8,7,6,5,4,2,1.5,1",
    );
}
