//! Element stores of any value, hole fills and appends in ARM64 generated code (W2c step 4).
//! ECMA-262 e28783d5: OrdinarySet (§10.1.9.2), OrdinarySetWithOwnDescriptor, Array
//! [[DefineOwnProperty]] (§10.4.2.1) and CreateDataProperty (§7.3.5). Each store either stays
//! in native code, with the element, the kind bits, the used length and `length` exactly as
//! `Interp::fast_set_elem` leaves them, or reaches the helper with nothing written.
use crate::bytecode::Tier;
use crate::value::word_block::{KIND_INT32, KIND_NO_HOLES, KIND_NUMBERS};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine() -> Engine {
    // The subject is the template JIT's element stores and its helper: with the tier on (P3
    // stage 2 step 2), a second stage's stores are the tier's own and call no helper.
    crate::opt::pin_tier_off_for_test();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

fn object(engine: &mut Engine, name: &str) -> crate::value::Gc {
    let global = crate::value::Value::Obj(engine.interp.global);
    engine
        .interp
        .get_member_key(&global, name)
        .unwrap_or_else(|_| panic!("{name}"))
        .as_obj()
        .unwrap()
}

/// The element words' used length, capacity, kind bits and whether a scan proved them.
fn words(engine: &mut Engine, name: &str) -> (usize, usize, u32, bool) {
    let handle = object(engine, name);
    let body = handle.borrow();
    let words = body.props.words().expect("element words");
    (words.len(), words.capacity(), words.kind(), words.scanned())
}

fn reset_helpers() {
    crate::bytecode::TEST_JIT_SET_ELEM_HELPERS.with(|count| count.set(0));
}

fn helpers() -> usize {
    crate::bytecode::TEST_JIT_SET_ELEM_HELPERS.with(|count| count.get())
}

/// Run `setup` (fresh state, ending in a creation store that renews the protector's proof),
/// then `measure`, until the process-wide creation epoch holds across both, and return the
/// measured result and its helper calls. Other tests may bump the epoch at any moment, which
/// correctly sends a creation to the helper and says nothing about the native path (the
/// convention of `shared_layout_tests::measured_creation_write`).
fn stable(engine: &mut Engine, setup: &str, measure: &str) -> (String, usize) {
    stable_with(engine, setup, &|_| {}, measure)
}

/// [`stable`], with `between` run after `setup` in each trial.
fn stable_with(
    engine: &mut Engine,
    setup: &str,
    between: &dyn Fn(&mut Engine),
    measure: &str,
) -> (String, usize) {
    for _ in 0..256 {
        let epoch = crate::value::proto_epoch();
        eval(engine, setup);
        between(engine);
        reset_helpers();
        let result = eval(engine, measure);
        let count = helpers();
        if crate::value::proto_epoch() == epoch {
            return (result, count);
        }
        std::thread::yield_now();
    }
    panic!("the prototype epoch changed during all 256 trials");
}

/// A creation store that renews the protector's proof (in the helper, when it is stale).
const RENEW: &str = "var renew=[0,1];renew[4]=4;localDrop(renew,2,0);";

/// The four store forms (local receiver or stack receiver, statement or expression).
const FORMS: &str = r#"
    function localDrop(a,k,v){a[k]=v;}
    function localKeep(a,k,v){return a[k]=v;}
    function stackDrop(a,k,v){(k<1e9?a:[])[k]=v;}
    function stackKeep(a,k,v){return (k<1e9?a:[])[k]=v;}
    var forms=[localDrop,localKeep,stackDrop,stackKeep];
"#;

#[test]
fn any_value_overwrites_stay_native_and_clear_the_number_facts_exactly() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "var a=[0,1,2,3,4,5,6,7];\
         for(var w=0;w<50;w++)for(var f=0;f<4;f++)forms[f](a,f,w);",
    );
    {
        let handle = object(&mut engine, "a");
        handle.borrow_mut().props.reprove_words();
    }
    let (_, _, kind, scanned) = words(&mut engine, "a");
    assert_eq!(kind, KIND_NO_HOLES | KIND_NUMBERS | KIND_INT32);
    assert!(scanned);
    reset_helpers();
    assert_eq!(
        eval(
            &mut engine,
            "var o={x:1};localDrop(a,0,o);var r=localKeep(a,1,'s');stackDrop(a,2,true);\
             var q=stackKeep(a,3,null);localDrop(a,4,undefined);\
             [a[0]===o,r,a[1],a[2],q,a[3],a[4],a.length].join()",
        ),
        "true,s,s,true,,,,8"
    );
    assert_eq!(helpers(), 0, "every overwrite stays in native code");
    let (len, _, kind, scanned) = words(&mut engine, "a");
    assert_eq!(len, 8);
    // NUMBERS and INT32 are false and exact; NO_HOLES is untouched; the scan's proof stands.
    assert_eq!(kind, KIND_NO_HOLES);
    assert!(scanned);
}

#[test]
fn edges_into_blocks_with_a_remembered_record_take_the_helper() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "var big=[];for(var i=0;i<100;i++)big[i]=i;\
         for(var w=0;w<50;w++)for(var f=0;f<4;f++)forms[f](big,f,w);",
    );
    reset_helpers();
    // A Number is no edge: native at any size.
    assert_eq!(eval(&mut engine, "localDrop(big,50,0.5);big[50]"), "0.5");
    assert_eq!(helpers(), 0);
    // A reference below the traced length needs the dirty block the helper marks.
    assert_eq!(
        eval(&mut engine, "var o={};localDrop(big,60,o);big[60]===o"),
        "true"
    );
    assert_eq!(helpers(), 1);
}

#[test]
fn appends_with_room_stay_native_and_grow_length() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "function room(n){var a=[];for(var i=0;i<n;i++)a[i]=i;a.length=0;return a;}\
         for(var w=0;w<20;w++){var t=room(40);for(var f=0;f<4;f++)forms[f](t,f,w);}",
    );
    let (result, count) = stable(
        &mut engine,
        &format!("var nums=room(40),cells=room(40),o={{}};{RENEW}"),
        "for(var i=0;i<32;i++){forms[i&3](nums,i,i);forms[i&3](cells,i,i&1?o:'s');}\
         localDrop(nums,32,0.5);\
         [nums.length,nums[31],nums[32],cells.length,cells[30],cells[31]===o].join()",
    );
    assert_eq!(result, "33,31,0.5,32,s,true");
    assert_eq!(count, 0, "appends with room stay in native code");
    // A fraction keeps NUMBERS and clears INT32; a string or an object clears both.
    let (len, _, kind, _) = words(&mut engine, "nums");
    assert_eq!((len, kind), (33, KIND_NO_HOLES | KIND_NUMBERS));
    let (len, _, kind, _) = words(&mut engine, "cells");
    assert_eq!((len, kind), (32, KIND_NO_HOLES));
}

#[test]
fn appends_below_length_leave_it_and_a_gap_takes_the_helper() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "for(var w=0;w<20;w++){var t=new Array(10);for(var f=0;f<4;f++)forms[f](t,f,w);}",
    );
    let (result, count) = stable(
        &mut engine,
        &format!("var a=new Array(10);localDrop(a,0,'first');{RENEW}"),
        "localDrop(a,1,'b');stackDrop(a,2,'c');[a.length,a[1],a[2]].join()",
    );
    assert_eq!(result, "10,b,c");
    assert_eq!(count, 0, "appends below length stay native");
    reset_helpers();
    assert_eq!(
        eval(
            &mut engine,
            "localDrop(a,5,'gap');[a.length,a[4],a[5]].join()"
        ),
        "10,,gap"
    );
    assert_eq!(helpers(), 1, "a store past the used words takes the helper");
}

#[test]
fn hole_fills_stay_native_and_withdraw_the_scan() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "for(var w=0;w<20;w++){var t=[];t[9]=0;for(var f=0;f<4;f++)forms[f](t,f,w);}",
    );
    // The top-down fill of chai's arrays: every store below the first fills a hole. Before
    // it, a scan proves the words' facts.
    let reprove = |name: &'static str| {
        move |engine: &mut Engine| {
            let handle = object(engine, name);
            handle.borrow_mut().props.reprove_words();
            let (_, _, kind, scanned) = words(engine, name);
            assert!(scanned && kind & KIND_NO_HOLES == 0, "{name}: {kind:#x}");
        }
    };
    let (result, count) = stable_with(
        &mut engine,
        &format!("var h=[];h[9]=9;{RENEW}"),
        &reprove("h"),
        "for(var i=8;i>=0;i--)forms[i&3](h,i,i);[h.length,h[0],h[8],1 in h].join()",
    );
    assert_eq!(result, "10,0,8,true");
    assert_eq!(count, 0, "hole fills stay in native code");
    let (_, _, kind, scanned) = words(&mut engine, "h");
    // The last hole is gone, but only a scan may say so: NO_HOLES stays clear, the scan's
    // proof is withdrawn, and a new scan proves it.
    assert_eq!((kind, scanned), (KIND_NUMBERS | KIND_INT32, false));
    {
        let handle = object(&mut engine, "h");
        handle.borrow_mut().props.reprove_words();
    }
    let (_, _, kind, scanned) = words(&mut engine, "h");
    assert_eq!(
        (kind, scanned),
        (KIND_NO_HOLES | KIND_NUMBERS | KIND_INT32, true)
    );
    // A hole fill that leaves holes withdraws the scan too.
    let (result, count) = stable_with(
        &mut engine,
        &format!("var g=[];g[3]=3;g[1]=1;{RENEW}"),
        &reprove("g"),
        "localDrop(g,0,0);g.length",
    );
    assert_eq!((result.as_str(), count), ("4", 0));
    let (_, _, kind, scanned) = words(&mut engine, "g");
    assert_eq!((kind, scanned), (KIND_NUMBERS | KIND_INT32, false));
}

#[test]
fn hole_fills_of_edges_into_blocks_with_a_record_take_the_helper() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "for(var w=0;w<20;w++){var t=[];t[9]=0;for(var f=0;f<4;f++)forms[f](t,f,w);}",
    );
    let (result, count) = stable(
        &mut engine,
        &format!("var big=[];for(var i=0;i<100;i++)big[i]=i;delete big[40];delete big[41];{RENEW}"),
        "localDrop(big,40,1);big[40]",
    );
    assert_eq!(result, "1");
    assert_eq!(count, 0, "a Number fills a hole at any size");
    reset_helpers();
    assert_eq!(
        eval(&mut engine, "var o={};localDrop(big,41,o);big[41]===o"),
        "true"
    );
    assert_eq!(helpers(), 1);
}

#[test]
fn creation_consults_the_prototype_chain_through_the_protector() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "for(var w=0;w<20;w++){var t=[0,1];t[4]=4;for(var f=0;f<4;f++)forms[f](t,f+2,w);}",
    );
    // An inherited setter on Array.prototype intercepts both a hole fill and an append.
    assert_eq!(
        eval(
            &mut engine,
            "var hits=[];\
             Object.defineProperty(Array.prototype,'2',{set:function(v){hits.push(v)},configurable:true});\
             Object.defineProperty(Array.prototype,'5',{set:function(v){hits.push(v)},configurable:true});\
             var x=[0,1];x[4]=4;localDrop(x,2,'hole');var y=[0,1,2,3,4];stackDrop(y,5,'append');\
             [hits.join(),x.hasOwnProperty(2),y.length,y.hasOwnProperty(5)].join('|')",
        ),
        "hole,append|false|5|false"
    );
    // On Object.prototype, as well; then without either, the stores create own elements
    // again (the helper renews the proof, and native code takes them once more).
    assert_eq!(
        eval(
            &mut engine,
            "delete Array.prototype[2];delete Array.prototype[5];\
             Object.defineProperty(Object.prototype,'3',{set:function(v){hits.push('o'+v)},configurable:true});\
             var z=[0,1,2];stackKeep(z,3,'via');delete Object.prototype[3];\
             [hits.join(),z.length,z.hasOwnProperty(3)].join('|')",
        ),
        "hole,append,ovia|3|false"
    );
    let (result, count) = stable(
        &mut engine,
        &format!("var v=[0,1];v[4]=4;{RENEW}"),
        "localDrop(v,2,'own');localDrop(v,5,'tail');[v[2],v.length,v.hasOwnProperty(2)].join()",
    );
    assert_eq!(result, "own,6,true");
    assert_eq!(count, 0, "a renewed proof readmits native creation");
}

#[test]
fn inherited_read_only_elements_refuse_creation() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "function strictDrop(a,k,v){'use strict';a[k]=v;}\
         for(var w=0;w<20;w++){var t=[0,1];t[4]=4;forms[w&3](t,2,w);strictDrop(t,3,w);}",
    );
    assert_eq!(
        eval(
            &mut engine,
            "Object.defineProperty(Array.prototype,'2',{value:'proto',writable:false,configurable:true});\
             var x=[0,1];x[4]=4;localDrop(x,2,'own');var r=[x[2],x.hasOwnProperty(2)].join();\
             try{strictDrop(x,2,'own');r+='|no throw'}catch(e){r+='|'+e.name}\
             delete Array.prototype[2];r",
        ),
        "proto,false|TypeError"
    );
}

#[test]
fn non_extensible_arrays_and_read_only_length_refuse_creation() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "function strictDrop(a,k,v){'use strict';a[k]=v;}\
         for(var w=0;w<20;w++){var t=[0,1,2];t[5]=5;forms[w&3](t,3,w);strictDrop(t,6,w);}",
    );
    assert_eq!(
        eval(
            &mut engine,
            "var p=[0,1,2];p[5]=5;Object.preventExtensions(p);localDrop(p,3,'hole');\
             localDrop(p,6,'tail');var r=[p.hasOwnProperty(3),p.length].join();\
             try{strictDrop(p,6,'tail');r+='|no throw'}catch(e){r+='|'+e.name}r",
        ),
        "false,6|TypeError"
    );
    assert_eq!(
        eval(
            &mut engine,
            "var l=[0,1,2];Object.defineProperty(l,'length',{writable:false});\
             localDrop(l,3,'tail');var r=[l.length,l.hasOwnProperty(3)].join();\
             try{strictDrop(l,3,'tail');r+='|no throw'}catch(e){r+='|'+e.name}\
             var m=new Array(4);m[0]=0;Object.defineProperty(m,'length',{writable:false});\
             localDrop(m,1,'below');r+='|'+[m.length,m[1]].join();r",
        ),
        "3,false|TypeError|4,below"
    );
}

#[test]
fn creation_on_other_receivers_and_other_prototypes_takes_the_helper() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "for(var w=0;w<20;w++){var t=[0,1];t[4]=4;for(var f=0;f<4;f++)forms[f](t,f+2,w);}",
    );
    reset_helpers();
    // An ordinary object's chain is not the protector's; neither is a subclass prototype.
    assert_eq!(
        eval(
            &mut engine,
            "var log=[];var proto={set 1(v){log.push(v)}};var o=Object.create(proto);o[0]='own';\
             localDrop(o,1,'setter');var s=[0];var sp=Object.create(Array.prototype,\
             {1:{set:function(v){log.push('s'+v)},configurable:true}});\
             Object.setPrototypeOf(s,sp);localDrop(s,1,'sub');\
             [log.join(),o.hasOwnProperty(1),s.length,s.hasOwnProperty(1)].join('|')",
        ),
        "setter,ssub|false|1|false"
    );
    assert!(helpers() >= 2);
}

#[test]
fn appends_to_a_marked_prototype_invalidate_creation_proofs() {
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "for(var w=0;w<20;w++){var t=[0,1];t[4]=4;for(var f=0;f<4;f++)forms[f](t,f+2,w);}",
    );
    // Array.prototype is itself an Array. Its own append (a marked prototype) must bump the
    // creation epoch: then a hole of every Array consults the new inherited element, so a
    // read-only one refuses the store.
    assert_eq!(
        eval(
            &mut engine,
            "var len=Array.prototype.length;localDrop(Array.prototype,len,'p');\
             Object.defineProperty(Array.prototype,len,{writable:false});\
             var x=[];x[len+1]=1;localDrop(x,len,'own');\
             var r=[x[len],x.hasOwnProperty(len)].join();Array.prototype.length=len;r",
        ),
        "p,false"
    );
}

#[test]
fn stored_cells_survive_minor_collections_of_old_arrays() {
    // Old arrays (promoted by a major collection) gain young cells through every native path:
    // overwrites, hole fills and appends, by a stub and in a loop. Only the owner's card tells
    // the minor collection that follows about them.
    let mut engine = engine();
    eval(&mut engine, FORMS);
    eval(
        &mut engine,
        "function fill(a,from,to,r){for(var i=from;i<to;i++)a[i]={v:r*100+i};}\
         var keep=[];for(var r=0;r<8;r++){var a=[0,1,2,3];a[9]=9;a[10]=10;keep.push(a);}\
         for(var w=0;w<20;w++){var t=[0,1];t[4]=4;for(var f=0;f<4;f++)forms[f](t,f,w);fill(t,0,6,w);}",
    );
    engine.interp.gc_collect();
    eval(
        &mut engine,
        "for(var r=0;r<8;r++){var a=keep[r];\
           for(var f=0;f<4;f++)forms[f](a,f,{v:r*100+f});\
           for(var i=4;i<9;i++)forms[i&3](a,i,{v:r*100+i});\
           forms[r&3](a,11,{v:r*100+11});fill(a,12,16,r);}",
    );
    engine.interp.gc_collect_minor();
    // Fresh allocation after the collection would reuse any cell it wrongly freed.
    eval(
        &mut engine,
        "var junk=[];for(var g=0;g<2000;g++)junk.push({v:-1,w:g});",
    );
    let expected: i64 = (0..8)
        .map(|r| {
            (0..16)
                .filter(|i| *i != 9 && *i != 10)
                .map(|i| r * 100 + i)
                .sum::<i64>()
        })
        .sum();
    assert_eq!(
        eval(
            &mut engine,
            "var sum=0;for(var r=0;r<8;r++){var a=keep[r];\
             for(var i=0;i<16;i++)if(i!=9&&i!=10)sum+=a[i].v;}sum",
        ),
        expected.to_string()
    );
}

#[test]
fn creation_inside_hot_loops_stays_native() {
    // Stores in loops, whose regions and chains carry their own code around the sites, create
    // elements as the stand-alone forms above do.
    let mut engine = engine();
    eval(
        &mut engine,
        "function up(a,n,v){for(var i=0;i<n;i++)a[i]=v;return a;}\
         function down(a,n,v){for(var i=n-1;i>=0;i--)a[i]=v;return a;}\
         for(var w=0;w<20;w++){var t=[];t.length=0;up(t,20,w);var u=[];u[19]=0;down(u,19,w);}\
         var room=[];for(var i=0;i<40;i++)room[i]=i;",
    );
    let (result, count) = stable(
        &mut engine,
        "room.length=0;var top=[];top[29]=0;var o={};\
         var r0=[0,1];r0[4]=4;up(r0,0,0);r0[2]=0;",
        "up(room,40,o);down(top,29,'s');\
         [room.length,room[39]===o,top.length,top[0],top[28],top[29]].join()",
    );
    assert_eq!(result, "40,true,30,s,s,0");
    assert_eq!(
        count, 0,
        "appends and hole fills in loops stay in native code"
    );
    let (len, _, kind, _) = words(&mut engine, "room");
    assert_eq!((len, kind), (40, KIND_NO_HOLES));
    let (len, _, kind, scanned) = words(&mut engine, "top");
    assert_eq!(
        (len, kind & (KIND_NUMBERS | KIND_INT32), scanned),
        (30, 0, false)
    );
}

#[test]
fn loop_chains_scan_at_most_4096_words_to_prove_a_receiver() {
    // A chained loop's preamble proves a receiver's words again by a scan when its kind bits
    // fall short, and every hole the loop fills withdraws the scan; a failed proof runs the
    // iteration in plain code, whose back edge enters the preamble again. Over a large holey
    // receiver (W2c step 6 admits holes up to 1023 past the capacity) the scan is skipped.
    let mut engine = engine();
    eval(
        &mut engine,
        "function fill(a,n){for(var i=0;i<n;i++)a[i]=i;return a.length;}\
         for(var w=0;w<20;w++)fill([],8);",
    );
    let calls = || super::TEST_REPROVE_CALLS.with(|count| count.get());
    super::TEST_REPROVE_CALLS.with(|count| count.set(0));
    assert_eq!(
        eval(
            &mut engine,
            "var big=[];big[100000]=0;fill(big,20000)+':'+big[19999]"
        ),
        "100001:19999"
    );
    assert_eq!(calls(), 0, "a receiver past 4,096 words is never scanned");
    assert_eq!(
        eval(
            &mut engine,
            "var small=[];small[100]=0;fill(small,100)+':'+small[99]"
        ),
        "101:99"
    );
    assert!(calls() > 0, "a small one still is");
}

#[test]
fn loop_chains_read_frozen_receivers_and_refuse_to_store_to_them() {
    // W2c step 7a: a frozen receiver's words take no store, so a chain storing to one takes the
    // plain loop; a chain that only reads it runs (reads ignore the integrity level).
    let mut engine = engine();
    eval(
        &mut engine,
        "function sum(a,n){var s=0;for(var i=0;i<n;i++)s+=a[i];return s;}\
         function twice(a,n){for(var i=0;i<n;i++)a[i]=a[i]*2;return a;}\
         for(var w=0;w<20;w++){sum([1,2,3,4],4);twice([1,2,3,4],4);}",
    );
    let entries = || super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.get());
    super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.set(0));
    assert_eq!(
        eval(&mut engine, "var f=Object.freeze([1,2,3,4]);sum(f,4)"),
        "10"
    );
    assert!(entries() > 0, "a chain reading a frozen receiver runs");
    super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.set(0));
    assert_eq!(eval(&mut engine, "twice(f,4).join()"), "1,2,3,4");
    assert_eq!(entries(), 0, "a chain storing to it takes the plain loop");
}
