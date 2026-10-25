//! A reusable native-to-JavaScript call entry, local to one builtin invocation.
//!
//! ECMA-262 Call / OrdinaryCallBindThis (snapshot e28783d5fc9d): proxies, bound functions,
//! cross-realm calls, prepared parameter lists and cold bodies retain the generic path. A
//! warmed ordinary function uses precisely the same guarded moved-frame entry as JS-to-JS
//! calls. Array iteration still performs each HasProperty/Get/Call in its specified order.

use crate::bytecode::{CallSite, Tier};
use crate::fasthash::FastMap;
use crate::interpreter::{Abrupt, Interp, PreparedCall};
use crate::value::Value;
use std::cell::{Cell, RefCell};

pub(crate) struct Callback {
    // This strong root owns the closure and its environment across every GC/callback reentry.
    callee: Value,
    cache: CallbackCache,
    /// The cache entry resolved once for this fixed callee (see [`Interp::prepare_call`]).
    /// A `Callback` is local to one builtin invocation, so this is never reached reentrantly.
    prepared: RefCell<Option<PreparedCall>>,
    /// Remaining attempts to prepare after a generic call; a callee that never reaches a
    /// cacheable compiled or native entry stops paying for the attempt.
    prepare_budget: Cell<u8>,
}

/// Generic calls after which a `Callback` stops trying to prepare its callee. Ordinary
/// functions reach a compiled entry after the tier-up threshold.
const PREPARE_ATTEMPTS: u8 = 32;

/// Reusable identity/code feedback without a strong JS owner. The currently selected callback
/// remains rooted by the caller, and no cache borrow survives JavaScript reentry. A shared entry
/// never caches a property lookup: proxy traps and accessor functions are fetched afresh first.
pub(crate) struct CallbackCache {
    site: CallSite,
    pins: RefCell<FastMap<usize, crate::value::WeakGc>>,
}

impl Callback {
    pub(crate) fn new(callee: Value) -> Self {
        Self {
            callee,
            cache: CallbackCache::new(),
            prepared: RefCell::new(None),
            prepare_budget: Cell::new(PREPARE_ATTEMPTS),
        }
    }

    pub(crate) fn call<const N: usize>(
        &self,
        interp: &mut Interp,
        this: Value,
        args: [Value; N],
    ) -> Result<Value, Abrupt> {
        if interp.tier == Tier::Jit {
            let prepared = self.prepared.borrow();
            if let Some(prepared) = prepared
                .as_ref()
                .filter(|prepared| interp.prepared_call_current(prepared))
            {
                // The evaluated array remains immutable and in scope through all nested calls,
                // post-call polls, and tail drains. Its canonical words need no adapter array.
                return unsafe { interp.call_prepared(prepared, &this, args.as_ptr(), N) };
            }
        }
        let result = self.cache.call(interp, &self.callee, this, args);
        if interp.tier == Tier::Jit && self.prepare_budget.get() != 0 {
            self.prepare_budget.set(self.prepare_budget.get() - 1);
            let prepared = interp.prepare_call(&self.cache.site, &self.callee);
            if prepared.is_some() {
                self.prepare_budget.set(PREPARE_ATTEMPTS);
            }
            *self.prepared.borrow_mut() = prepared;
        }
        result
    }

    /// IteratorStepValue can consume a closed native iterator without materializing
    /// its result object. Custom next methods retain their ordinary callback cache.
    pub(crate) fn iterator_step(
        &self,
        interp: &mut Interp,
        iterator: &Value,
    ) -> Result<Option<Value>, Abrupt> {
        if let Some(result) = interp.try_intrinsic_iterator_step(iterator, &self.callee) {
            return result;
        }
        let result = self.call(interp, *iterator, [])?;
        interp.iterator_result_value(result)
    }
}

impl CallbackCache {
    pub(crate) fn new() -> Self {
        Self {
            site: CallSite::empty(),
            pins: RefCell::new(FastMap::default()),
        }
    }

    pub(crate) fn call<const N: usize>(
        &self,
        interp: &mut Interp,
        callee: &Value,
        this: Value,
        args: [Value; N],
    ) -> Result<Value, Abrupt> {
        self.call_values(interp, callee, this, &args)
    }

    pub(crate) fn call_values(
        &self,
        interp: &mut Interp,
        callee: &Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        if interp.tier != Tier::Jit {
            return interp.call(*callee, this, args);
        }
        // Safe callers may lend a Rust-heap argument window. The compiled entry polls before
        // seeding its own frame; keep these exact words rooted through both entry paths, any
        // generic fallback, post-call polls and pending tails, without copying the window.
        let _arguments = crate::rooting::root_slice(args);
        let this_word = this;
        let words = args;
        // SAFETY: the original evaluated argument window stays borrowed until the committed
        // call (including polls and tails) returns. Nothing mutates or pools it here.
        let result = unsafe {
            interp
                .call_jit_cached(&self.site, callee, &this_word, words.as_ptr(), words.len())
                .or_else(|| {
                    interp.call_jit_fast(
                        callee,
                        &this_word,
                        words.as_ptr(),
                        words.len(),
                        Some((&self.site, &self.pins)),
                    )
                })
        };
        result.unwrap_or_else(|| interp.call(*callee, this, args))
    }

    pub(crate) fn prune_dead(&self) {
        let mut pins = self.pins.borrow_mut();
        for entry in &self.site.entries {
            let ic = entry.get();
            if ic.callee != 0
                && pins
                    .get(&ic.callee)
                    .is_some_and(|pin| pin.strong_count() == 0)
            {
                // Invalidate before releasing the address pin: a recycled allocation must not
                // match a raw identity left behind in the site. The overflow owns its own pins.
                entry.set(crate::bytecode::CallIc::EMPTY);
            }
        }
        pins.retain(|_, pin| pin.strong_count() != 0);
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(
                self.pins
                    .borrow()
                    .len()
                    .saturating_mul(std::mem::size_of::<(usize, crate::value::WeakGc)>()),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Callable;
    use crate::value::ValueView;
    use crate::{Completion, Engine};

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("callback fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
        }
    }

    fn check(source: &str, expected: &str) {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
            assert!(engine.interp.fn_frames.is_empty());
            engine.interp.gc_collect();
            assert_eq!(eval(&mut engine, "1+2"), "3");
        }
    }

    /// ECMA-262 §23.1.3 (forEach, map, filter, some, every, find*, reduce*): the packed element
    /// and result paths still perform HasProperty/Get per index against the live array (holes
    /// consult the prototype, accessors run, mutations during iteration are observed), define
    /// results with CreateDataPropertyOrThrow, and apply ToBoolean to every value kind.
    #[test]
    fn packed_iteration_paths_keep_live_array_semantics() {
        check(
            r#"
            var log=[];
            Array.prototype[3]='proto';
            var a=[1,,{v:2},,'s'];
            Object.defineProperty(a, 5, {get(){log.push('get5');return 5}, configurable:true, enumerable:true});
            var m=a.map(function(x,k,o){ if(k===0){o[1]='late'; o.length=8; o[6]=6;} return typeof x==='object'?x.v:x; });
            var seen=[];a.forEach(function(x,k){seen.push(k+':'+(typeof x==='object'?'obj':x));});
            var f=a.filter(function(x){return x!=='s'&&x!=='proto';});
            var truth=[10n,0n,'','x',0,NaN,-0,{},null,undefined,true,false,Symbol()].map(function(v){return [v].some(function(x){return x;})?1:0}).join('');
            var every=[1,'a',{}].every(function(x){return x;})+'/'+[1,0].every(function(x){return x;});
            var red=[{n:1},{n:2},{n:3}].reduce(function(acc,o){return {n:acc.n+o.n};}).n;
            var redr=['a','b','c'].reduceRight(function(acc,x){return acc+x;});
            var obj={k:2};var fnd=[{k:1},obj].find(function(o){return o.k===2;})===obj;
            var fi=[5,6,7].findIndex(function(x){return x===7;})+'/'+[5,6,7].findLast(function(x){return x<7;})+'/'+[5,6].findLastIndex(function(x){return x>9;});
            delete Array.prototype[3];
            var big=[];for(var n=0;n<600;n++)big.push({n:n});
            var total=0;for(var round=0;round<20;round++){total+=big.map(function(o){return o.n;}).filter(function(n){return n%3===0;}).reduce(function(s,n){return s+n;},0);}
            [m.length, m.join(','), seen.join(','), f.length+':'+f.map(function(x){return typeof x}).join(','), truth, every, red, redr, fnd, fi, log.join(','), total].join('|')
        "#,
            "6|1,late,2,proto,s,5|0:1,1:late,2:obj,3:proto,4:s,5:5,6:6|5:number,string,object,number,number|1001000100101|true/false|6|cba|true|2/6/-1|get5,get5,get5|1194000",
        );
    }

    #[test]
    fn callbacks_preserve_live_iteration_holes_arguments_and_receiver() {
        check(
            r#"
            var trace = [];
            var source = [1,,3,4];
            var receiver = {value:10};
            function callback(value, index, array) {
                'use strict';
                trace.push(index+':'+value+':'+(array===source)+':'+arguments.length+':'+this.value);
                if(index===0){array[1]=2;delete array[2];array.push(5)}
                return value+this.value;
            }
            var mapped = source.map(callback, receiver);
            [mapped.length, mapped.join(','), trace.join('|')].join(';');
        "#,
            "4;11,12,,14;0:1:true:3:10|1:2:true:3:10|3:4:true:3:10",
        );
    }

    #[test]
    fn callbacks_preserve_proxy_species_order_and_abrupt_completion() {
        check(
            r#"
            var trace=[];
            var cb=new Proxy(function(v,k){trace.push('call'+k);if(k===2)throw 17;return v+1}, {
                apply(t,s,a){trace.push('apply'+a[1]);return Reflect.apply(t,s,a)}
            });
            var a=[1,,3];
            a.constructor = {[Symbol.species]:function(n){trace.push('species'+n);return new Proxy({}, {
                defineProperty(t,k,d){trace.push('define'+k);return Reflect.defineProperty(t,k,d)}
            })}};
            try {a.map(cb)} catch(e){trace.push('throw'+e)}
            trace.join(',');
        "#,
            "species3,apply0,call0,define0,apply2,call2,throw17",
        );
    }

    #[test]
    fn callbacks_cover_predicates_reducers_flatmap_and_prepared_parameters() {
        check(
            r#"
            var a=[1,,3], seen=[];
            var f=(v,k)=>{seen.push(k);return v===undefined};
            var r=[a.findIndex(f), a.findLastIndex(f), a.some(v=>v===3), a.every(v=>v>0),
                a.filter(v=>v>1).join(''), a.reduce((x,v)=>x+v,10),
                a.reduceRight((x,v)=>x+'-'+v,'s'), a.flatMap(v=>[v,v+1]).join(',')];
            var sum=0;a.forEach(v=>{sum+=v});
            r.push(sum,seen.join(','),[{}, {x:7}].map(({x=5},...rest)=>x+rest.length).join(','));
            r.join('|');
        "#,
            "1|1|true|true|3|14|s-3-1|1,2,3,4|4|0,1,2,1|7,9",
        );
    }

    #[test]
    fn typed_callbacks_cover_live_receivers_predicates_reducers_and_bigint() {
        check(
            r#"
            var a=new Uint8Array([1,2,3]), receiver={n:10}, trace=[];
            function cb(v,k,o){'use strict';trace.push(k+':'+v+':'+(o===a)+':'+arguments.length+':'+this.n);return v+this.n}
            var mapped=a.map(cb,receiver);
            var out=[mapped.join(','),trace.join('|'),a.every(v=>v>0),a.some(v=>v===2),
                a.find(v=>v>1),a.findIndex(v=>v>1),a.findLast(v=>v>1),a.findLastIndex(v=>v>1),
                a.filter(v=>v!==2).join(','),a.reduce((s,v)=>s+v,10),a.reduceRight((s,v)=>s+'-'+v,'x')];
            var total=0;a.forEach(v=>{total+=v});out.push(total);
            out.push(new BigInt64Array([1n,2n,3n]).map(v=>v+4n).reduce((s,v)=>s+v,0n));
            out.join(';');
        "#,
            "11,12,13;0:1:true:3:10|1:2:true:3:10|2:3:true:3:10;true;true;2;1;3;2;1,3;16;x-3-2-1;6;18",
        );
    }

    #[test]
    fn typed_callbacks_capture_length_but_read_resized_and_detached_buffers_live() {
        check(
            r#"
            var b=new ArrayBuffer(4,{maxByteLength:8}), a=new Uint8Array(b), trace=[];
            a.set([1,2,3,4]);
            a.forEach(function(v,k,o){trace.push(k+':'+v+':'+(o===a));if(k===0)b.resize(1)});
            var c=new Uint8Array([7,8,9]), seen=[];
            c.find(function(v,k){seen.push(k+':'+v);if(k===0)c.buffer.transfer();return false});
            trace.join('|')+';'+seen.join('|');
        "#,
            "0:1:true|1:undefined:true|2:undefined:true|3:undefined:true;0:7|1:undefined|2:undefined",
        );
    }

    #[test]
    fn typed_callbacks_preserve_species_order_throw_and_reentrant_prepared_calls() {
        check(
            r#"
            var trace=[], a=new Uint8Array([1,2,3]);
            a.constructor={[Symbol.species]:function(n){trace.push('species'+n);return new Uint8Array(n)}};
            function mapped(v,k){trace.push('m'+k);return v+1}
            function filtered(v,k){trace.push('f'+k);return v!==2}
            var m=a.map(mapped),f=a.filter(filtered);
            var proxy=new Proxy(function(v,k){if(k===1)throw 17;return v},{apply(t,s,args){trace.push('p'+args[1]);return Reflect.apply(t,s,args)}});
            try{a.forEach(proxy)}catch(e){trace.push('throw'+e)}
            var prepared=({value=3}={},...rest)=>new Uint8Array([value]).reduce((s,v)=>s+v,rest.length);
            var p=Array.from({length:2},prepared);
            [m.join(','),f.join(','),trace.join(','),p.join(',')].join(';');
        "#,
            "2,3,4;1,3;species3,m0,m1,m2,f0,f1,f2,species2,p0,p1,throw17;4,4",
        );
    }

    #[test]
    fn from_callbacks_preserve_iterator_collection_arraylike_reads_and_close_precedence() {
        check(
            r#"
            var trace=[], receiver={n:10};
            var iterable={[Symbol.iterator](){var k=0;trace.push('iter');return {
                next(){trace.push('next'+k);return k<2?{value:++k,done:false}:{done:true}},
                return(){trace.push('close');throw 99}
            }}};
            function mapper(v,k){'use strict';trace.push('map'+k+':'+this.n+':'+arguments.length);return v+this.n}
            var typed=Uint8Array.from(iterable,mapper,receiver);
            var source={length:2,get 0(){trace.push('get0');return 2},get 1(){trace.push('get1');return 3}};
            var array=Array.from(source,mapper,receiver);
            try{Array.from(iterable,function(v){throw 17})}catch(e){trace.push('throw'+e)}
            [typed.join(','),array.join(','),trace.join(',')].join(';');
        "#,
            "11,12;12,13;iter,next0,next1,next2,map0:10:2,map1:10:2,get0,map0:10:2,get1,map1:10:2,iter,next0,close,throw17",
        );
    }

    #[test]
    fn sort_callbacks_preserve_stability_numeric_coercion_and_abrupt_results() {
        check(
            r#"
            var items=[{k:1,id:'a'},{k:0,id:'b'},{k:1,id:'c'},{k:0,id:'d'}];
            var receiver=true;
            function cmp(a,b){'use strict';receiver=receiver&&this===undefined&&arguments.length===2;return {valueOf(){return a.k-b.k}}}
            var stable=items.sort(cmp).map(v=>v.id).join('');
            var n=new Float64Array([3,1,2]).sort(function(a,b){'use strict';if(this!==undefined)throw 'this';return {valueOf(){return a-b}}});
            var unchanged=new Uint8Array([3,1,2]).toSorted(()=>NaN);
            var big=new BigInt64Array([3n,1n,2n]).sort((a,b)=>a<b?-1:a>b?1:0);
            var a=[3,2,1], typed=new Uint8Array(a), errors=[];
            try{a.sort(()=>{throw 17})}catch(e){errors.push(e)}
            try{typed.sort(()=>1n)}catch(e){errors.push(e instanceof TypeError)}
            [stable,receiver,n.join(','),unchanged.join(','),big.join(','),a.join(','),typed.join(','),errors.join(',')].join(';');
        "#,
            "bdac;true;1,2,3;3,1,2;1,2,3;3,2,1;3,2,1;17,true",
        );
    }

    #[test]
    fn eager_iterator_callbacks_keep_next_snapshot_and_close_order() {
        check(
            r#"
            var trace=[];
            function source(){var k=0;return Iterator.from({next(){trace.push('n'+k);return k<3?{value:++k,done:false}:{done:true}},return(){trace.push('close');return {done:true}}})}
            var out=[source().some((v,k)=>k===1),source().every(v=>v<2),source().find(v=>v===2),source().reduce((s,v,k)=>s+v+k,0)];
            var s=0;source().forEach((v,k)=>{s+=v+k});out.push(s);
            try{source().forEach(()=>{throw 17})}catch(e){out.push(e)}
            out.push(trace.join(','));out.join(';');
        "#,
            "true;false;2;9;9;17;n0,n1,close,n0,n1,close,n0,n1,close,n0,n1,n2,n3,n0,n1,n2,n3,n0,close",
        );
    }

    #[test]
    fn shared_callback_entries_revalidate_accessor_and_proxy_targets_after_reentry() {
        check(
            r#"
            var leaf={value:2}, seen=0, receiver={value:5};
            Object.defineProperty(leaf,'x',{configurable:true,get:function(){return this.value}});
            var object={get x(){return leaf.x+this.value},set x(v){this.value=v}}, sum=0;
            var handler={get get(){seen++;return function(t,k,r){return Reflect.get(t,k,r)}}};
            var rev=Proxy.revocable(object,handler), proxy=rev.proxy;
            for(var k=0;k<240;k++)sum+=Reflect.get(proxy,'x',receiver);
            Object.defineProperty(leaf,'x',{get:()=>10});
            var later=Reflect.get(proxy,'x',receiver);
            handler.get=function(t,k,r){return 99};
            Object.defineProperty(handler,'get',{value:function(){return 99}});
            var replaced=proxy.x;
            rev.revoke();var revoked=false;try{proxy.x}catch(e){revoked=e instanceof TypeError}
            var p=new Proxy(function(v){return v+1},{apply(t,s,a){return Reflect.apply(t,s,a)+1}});
            var result;for(var k=0;k<200;k++)result=p(3);
            [sum,later,replaced,revoked,result,seen].join(';');
        "#,
            "1680;15;99;true;5;241",
        );
    }

    /// Rooting harness (`jit::rooting_tests`): a callback with more than four
    /// arguments goes through `CallbackCache::call_values`'s list arm into the compiled callee,
    /// which receives the receiver as its record's `this` and the arguments in its record's
    /// slots. A collection from the callee finds the receiver and an object argument, both fresh
    /// and held by nothing else but the calling native's frame, marked by those roots, and the
    /// callee reads them back. Afterwards nothing keeps them, and the shared cache, which cached
    /// the callee, does not root it: once unnamed it dies, and pruning empties the site.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    #[test]
    fn long_callback_calls_root_operands_and_cache_weakly() {
        use crate::jit::rooting_tests::{
            assert_tracked_collectable, install, jit_engine, last_census, run, track,
        };
        use crate::value::Property;
        /// `callSix(callee, collect)`: `callee` called back with a fresh receiver `{n: 10,
        /// collect}` and the arguments 1 to 5 and a fresh `{v: 6}`; with `collect` set, the
        /// two objects are tracked.
        fn call_six(i: &mut Interp, _: Value, args: &[Value]) -> Result<Value, Value> {
            let callee = args.first().cloned().unwrap_or(Value::Undefined);
            let collect = matches!(
                (args.get(1)).map(|__value| (__value).view()),
                Some(ValueView::Bool(true))
            );
            let receiver = i.new_object();
            receiver
                .borrow_mut()
                .props
                .insert("n", Property::plain(Value::Num(10.0)));
            receiver
                .borrow_mut()
                .props
                .insert("collect", Property::plain(Value::Bool(collect)));
            let object = i.new_object();
            object
                .borrow_mut()
                .props
                .insert("v", Property::plain(Value::Num(6.0)));
            if collect {
                track(i, Value::Undefined, &[Value::Obj(receiver)])?;
                track(i, Value::Undefined, &[Value::Obj(object)])?;
            }
            let mut arguments: Vec<Value> = (1..=5).map(|n| Value::Num(f64::from(n))).collect();
            arguments.push(Value::Obj(object));
            i.call_callback(callee, Value::Obj(receiver), &arguments)
                .map_err(crate::interpreter::abrupt_value)
        }
        let mut engine = jit_engine(0);
        install(&mut engine, "callSix", 2, call_six);
        let source = r#"
            function retained(a, b, c, d, e, f) {
                'use strict';
                return (this.collect ? collect() : '-') + ':' + (this.n + a + b + c + d + e + f.v);
            }
            for (var k = 0; k < 20; k++) callSix(retained, false);
            callSix(retained, true);
        "#;
        assert_eq!(run(&mut engine, source), "RR:31");
        assert!(last_census().records >= 1, "the callee's record");
        // The handles, and the cache entries' raw callee addresses, stay in helper frames.
        let (pin, cache) = crate::gc_test::in_frame(|| {
            let env = engine.interp.global_env;
            let callee = engine
                .interp
                .get_var("retained", &env)
                .ok()
                .expect("function");
            let cache = engine
                .interp
                .native_callback_cache
                .clone()
                .expect("lazy cache");
            assert!(
                cache
                    .site
                    .entries
                    .iter()
                    .any(|entry| entry.get().callee != 0),
                "the site cached its compiled callee"
            );
            (
                crate::value::Gc::downgrade(&callee.as_obj().unwrap()),
                cache,
            )
        });
        assert_tracked_collectable(&mut engine);
        run(&mut engine, "retained = null; 0;");
        engine.interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&pin),
            "shared feedback must not root its callback"
        );
        cache.prune_dead();
        assert!(crate::gc_test::in_frame(|| cache
            .site
            .entries
            .iter()
            .all(|entry| entry.get().callee == 0)));
        assert!(cache.pins.borrow().is_empty());
    }

    #[test]
    fn arrow_callbacks_keep_lexical_this_arguments_newtarget_and_super() {
        check(
            r#"
            class Base {method(){return this.value}}
            class Derived extends Base {
                constructor(value) {
                    var early = flag => flag ? this : 7;
                    for(var k=0;k<150;k++)if(early(false)!==7)throw 'early';
                    var tdz=false;try{early(true)}catch(e){tdz=e instanceof ReferenceError}
                    if(!tdz)throw 'missing TDZ';
                    super();this.value=value;
                    this.read = () => [this.value, arguments[0], new.target===Derived, super.method()].join(':');
                }
            }
            var d=new Derived(23), result;
            for(var k=0;k<180;k++)result=[0,1,2].map(d.read,{value:99}).join('|');
            result;
        "#,
            "23:23:true:23|23:23:true:23|23:23:true:23",
        );
    }

    #[test]
    fn callback_cache_keeps_live_fresh_arrow_environments_and_unwinds() {
        check(
            r#"
            function make(value){return x=>value+x}
            function invoke(f,x){return f(x)}
            var sum=0;
            for(var k=0;k<250;k++)sum+=invoke(make(k),1);
            var f=x=>{if(x===2)throw 91;return x};
            for(var k=0;k<150;k++){try{[0,1,2].map(f)}catch(e){if(e!==91)throw e}}
            sum+':'+[1,2].map(make(5)).join(',');
        "#,
            "31375:6,7",
        );
    }

    #[test]
    fn callback_entry_caches_arrows_and_consumes_owned_arguments() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            "var callback = x => x.value; callback({value:1});",
        );
        let env = engine.interp.global_env;
        let callee = engine
            .interp
            .get_var("callback", &env)
            .unwrap_or_else(|_| panic!("callback"));
        let entry = Callback::new(callee);
        for n in 0..160 {
            let arg = engine.interp.make_array(vec![Value::Num(n as f64)]);
            let result = entry.call(&mut engine.interp, Value::Undefined, [arg]);
            assert!(matches!(
                (result).map(|__value| (__value).view()),
                Ok(ValueView::Undefined)
            ));
        }
        let object_handle = entry.callee.as_obj().unwrap();
        let object = object_handle.borrow();
        let Callable::User(user) = &object.call else {
            panic!("user")
        };
        let runtime = user.func.runtime();
        let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
        if chunk.jit.get().flatten().is_some() {
            assert!(entry
                .cache
                .site
                .entries
                .iter()
                .any(|ic| ic.get().direct & crate::bytecode::CALL_IC_LEXICAL_THIS != 0));
        }
        assert!(engine.interp.fn_frames.is_empty());
    }

    #[test]
    fn inline_retry_observes_call_feedback_after_early_empty_plan() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function target(x){return x+2}
            function laterHot(x){if(x===0)return x;var y=target(x);return y+1}
            function drive(){var sum=0;for(var k=0;k<300;k++)sum+=laterHot(0);
                for(var k=0;k<600;k++)sum+=laterHot(1);return sum}
            drive();
        "#
            ),
            "2400"
        );
        let env = engine.interp.global_env;
        let value = engine
            .interp
            .get_var("laterHot", &env)
            .unwrap_or_else(|_| panic!("laterHot"));
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let Callable::User(user) = &object.call else {
            panic!("user")
        };
        let runtime = user.func.runtime();
        let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
        if crate::bytecode::inline_recompile_at() == 100 && chunk.jit.get().flatten().is_some() {
            assert!(
                user.func
                    .runtime()
                    .code2
                    .get()
                    .and_then(Option::as_ref)
                    .is_some(),
                "later feedback must be reconsidered after the empty first checkpoint"
            );
        }
    }

    #[test]
    fn plain_arrow_helpers_can_inline_without_changing_lexical_arrows() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var arrowTarget=x=>x+2;
            function arrowHot(x){var y=arrowTarget(x);return y+1}
            function driveArrows(){var result=0;for(var k=0;k<250;k++)result+=arrowHot(1);return result}
            driveArrows();
        "#
            ),
            "1000"
        );
        let env = engine.interp.global_env;
        let value = engine
            .interp
            .get_var("arrowHot", &env)
            .unwrap_or_else(|_| panic!("arrowHot"));
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let Callable::User(user) = &object.call else {
            panic!("user")
        };
        let runtime = user.func.runtime();
        let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
        if crate::bytecode::inline_recompile_at() == 100 && chunk.jit.get().flatten().is_some() {
            assert!(user
                .func
                .runtime()
                .code2
                .get()
                .and_then(Option::as_ref)
                .is_some());
        }
    }
}

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod entry_root_tests {
    use super::*;
    use crate::gc_core::MarkSource;
    use crate::value::{Callable, Gc};
    use crate::{Completion, Engine};

    fn check(warm_cache: bool) {
        let child = if warm_cache {
            "TRUST_JS_TEST_CACHED_CALLBACK_ENTRY_ROOT_CHILD"
        } else {
            "TRUST_JS_TEST_FAST_CALLBACK_ENTRY_ROOT_CHILD"
        };
        let name = if warm_cache {
            "callback::entry_root_tests::cached_callback_roots_heap_arguments_before_the_entry_poll"
        } else {
            "callback::entry_root_tests::fast_callback_roots_heap_arguments_before_the_entry_poll"
        };
        if std::env::var_os(child).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--test-threads=1"])
                .env(child, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        match engine
            .eval(
                "function entryRoot(a,b,c,d,e,f){return 17};entryRoot(0,0,0,0,0,0)",
                false,
            )
            .unwrap()
        {
            Completion::Value(value) => assert_eq!(value, "17"),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        let env = engine.interp.global_env;
        let callee = engine
            .interp
            .get_var("entryRoot", &env)
            .unwrap_or_else(|_| panic!("compiled callback"));
        let object = callee.as_obj().unwrap();
        let body = object.borrow();
        let Callable::User(user) = &body.call else {
            panic!("user callback");
        };
        let chunk = user
            .func
            .runtime()
            .execution_code()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        assert!(chunk.jit.get().flatten().is_some());
        drop(body);
        engine.interp.native_callback_cache = None;
        if warm_cache {
            engine
                .interp
                .invoke(callee, Value::Undefined, &[Value::Num(0.0); 6])
                .unwrap_or_else(|_| panic!("warm callback cache"));
            let key = Gc::as_ptr(&object) as usize;
            let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                engine
                    .interp
                    .native_callback_cache
                    .as_ref()
                    .unwrap()
                    .site
                    .entries
                    .iter()
                    .any(|entry| entry.get().callee == key && entry.get().epoch == epoch),
                "actual current compiled cache hit"
            );
        }
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_retention_tracing(true);
        // Only the Rust-heap Vec carries the argument word after this helper returns. A weak
        // pin protects diagnostic identity but supplies no strong edge or KeepDuringJob root.
        let (arguments, weak) = crate::gc_test::in_frame(|| {
            let object = engine.interp.new_object();
            (
                vec![
                    Value::Obj(object),
                    Value::Num(1.0),
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::Num(4.0),
                    Value::Num(5.0),
                ],
                Gc::downgrade(&object),
            )
        });
        chunk.jit_runs.set(0);
        chunk.inline_retry_at.set(u32::MAX);
        let before = engine.interp.gc_heap.core.borrow().collections();
        // The first compiled-entry poll must collect before argument slots are seeded. The
        // scalar-returning body neither reads this argument nor executes any later safepoint.
        engine.interp.gc_tick = crate::interpreter::GC_CALL_POLL_MASK;
        engine.interp.set_agent_next(0);
        let result = engine
            .interp
            .invoke(callee, Value::Undefined, &arguments)
            .unwrap_or_else(|_| panic!("native callback"));
        assert_eq!(result.as_num_opt(), Some(17.0));
        assert_eq!(
            engine.interp.gc_heap.core.borrow().collections(),
            before + 1,
            "one pre-entry collection, none in the body"
        );
        assert!(
            chunk.jit_runs.get() >= 1,
            "the exact compiled callee must enter native code"
        );
        let root = weak
            .upgrade()
            .expect("pre-entry collection lost the heap argument");
        let path = engine
            .interp
            .gc_heap
            .core
            .borrow()
            .retention_path(Gc::as_ptr(&root) as usize);
        assert!(
            matches!(path.first(), Some((_, _, MarkSource::Root))),
            "pre-entry heap arguments must be explicit roots, got {path:?}"
        );
        assert_eq!(
            arguments[0].cell_word().unwrap() as usize,
            Gc::as_ptr(&root) as usize
        );
        drop(arguments);
        engine
            .interp
            .gc_collect_precise_for_test(crate::gc_core::CollectionKind::Major);
        assert!(
            weak.upgrade().is_none(),
            "callback arguments remained rooted after the loan ended"
        );
    }

    #[test]
    fn fast_callback_roots_heap_arguments_before_the_entry_poll() {
        check(false);
    }
    #[test]
    fn cached_callback_roots_heap_arguments_before_the_entry_poll() {
        check(true);
    }
}
