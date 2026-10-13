//! A reusable native-to-JavaScript call entry, local to one builtin invocation.
//!
//! ECMA-262 Call / OrdinaryCallBindThis (snapshot e28783d5fc9d): proxies, bound functions,
//! cross-realm calls, prepared parameter lists and cold bodies retain the generic path. A
//! warmed ordinary function uses precisely the same guarded moved-frame entry as JS-to-JS
//! calls. Array iteration still performs each HasProperty/Get/Call in its specified order.

use crate::bytecode::{CallSite, Tier};
use crate::fasthash::FastMap;
use crate::interpreter::{Abrupt, Interp, PreparedCall};
use crate::value::{Object, PackedValue, Value};
use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::rc::Weak;

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
    pins: RefCell<FastMap<usize, Weak<RefCell<Object>>>>,
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
                let mut args = ManuallyDrop::new(args.map(PackedValue::pack));
                let this = ManuallyDrop::new(PackedValue::pack(this));
                // The prepared entry consumes every operand, exactly like a cache hit.
                return unsafe { interp.call_prepared(prepared, &*this, args.as_mut_ptr(), N) }
                    .map(PackedValue::into_value);
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
        let result = self.call(interp, iterator.clone(), [])?;
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
        if interp.tier != Tier::Jit {
            return interp.call(callee.clone(), this, &args);
        }
        let mut args = ManuallyDrop::new(args.map(PackedValue::pack));
        let mut this = ManuallyDrop::new(PackedValue::pack(this));
        // Both fast entries leave all arguments untouched on None and consume all of them on
        // Some, including early stack/interrupt errors. ManuallyDrop transfers ownership once;
        // the generic path below recovers normal RAII ownership on a miss.
        unsafe {
            let result = interp
                .call_jit_cached(&self.site, callee, &*this, args.as_mut_ptr(), N)
                .or_else(|| {
                    interp.call_jit_fast(
                        callee,
                        &*this,
                        args.as_mut_ptr(),
                        N,
                        Some((&self.site, &self.pins)),
                    )
                });
            if let Some(result) = result {
                return result;
            }
            let args = ManuallyDrop::take(&mut args).map(PackedValue::into_value);
            let this = ManuallyDrop::take(&mut this).into_value();
            interp.call(callee.clone(), this, &args)
        }
    }

    pub(crate) fn call_values(
        &self,
        interp: &mut Interp,
        callee: &Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        // Native algorithms and host callbacks overwhelmingly use at most four arguments.
        // Larger argument lists retain the same guarded ABI, allocating only actual arguments.
        if interp.tier != Tier::Jit {
            return interp.call(callee.clone(), this, args);
        }
        match args {
            [] => self.call(interp, callee, this, []),
            [a] => self.call(interp, callee, this, [a.clone()]),
            [a, b] => self.call(interp, callee, this, [a.clone(), b.clone()]),
            [a, b, c] => self.call(interp, callee, this, [a.clone(), b.clone(), c.clone()]),
            [a, b, c, d] => self.call(
                interp,
                callee,
                this,
                [a.clone(), b.clone(), c.clone(), d.clone()],
            ),
            _ => {
                let mut args: Vec<_> = args.iter().cloned().map(PackedValue::pack).collect();
                let mut this = ManuallyDrop::new(PackedValue::pack(this));
                unsafe {
                    let result = interp
                        .call_jit_cached(&self.site, callee, &*this, args.as_mut_ptr(), args.len())
                        .or_else(|| {
                            interp.call_jit_fast(
                                callee,
                                &*this,
                                args.as_mut_ptr(),
                                args.len(),
                                Some((&self.site, &self.pins)),
                            )
                        });
                    if let Some(result) = result {
                        // The committed entry consumed every element; keep only the allocation.
                        args.set_len(0);
                        return result;
                    }
                    let args: Vec<_> = args.into_iter().map(PackedValue::into_value).collect();
                    interp.call(
                        callee.clone(),
                        ManuallyDrop::take(&mut this).into_value(),
                        &args,
                    )
                }
            }
        }
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
                    .saturating_mul(std::mem::size_of::<(usize, Weak<RefCell<Object>>)>()),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Callable;
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

    #[test]
    fn shared_callback_cache_is_weak_and_large_arguments_transfer_exactly_once() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            "function retained(a,b,c,d,e,f){'use strict';return this.n+a+b+c+d+e+f}",
        );
        let env = engine.interp.global_env.clone();
        let callee = engine
            .interp
            .get_var("retained", &env)
            .ok()
            .expect("function");
        let pin = std::rc::Rc::downgrade(callee.as_obj().unwrap());
        let receiver = engine.interp.new_object();
        receiver
            .borrow_mut()
            .props
            .insert("n", crate::value::Property::plain(Value::Num(10.0)));
        for _ in 0..220 {
            let result = engine
                .interp
                .call_callback(
                    callee.clone(),
                    Value::Obj(receiver.clone()),
                    &[
                        Value::Num(1.0),
                        Value::Num(2.0),
                        Value::Num(3.0),
                        Value::Num(4.0),
                        Value::Num(5.0),
                        Value::Num(6.0),
                    ],
                )
                .ok()
                .expect("callback");
            assert!(matches!(result, Value::Num(31.0)));
        }
        assert_eq!(
            std::rc::Rc::strong_count(&receiver),
            1,
            "no argument owner remains in a frame"
        );
        let cache = engine
            .interp
            .native_callback_cache
            .clone()
            .expect("lazy cache");
        if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            assert!(cache
                .site
                .entries
                .iter()
                .any(|entry| entry.get().callee != 0));
        }
        eval(&mut engine, "retained=null");
        drop(callee);
        engine.interp.gc_collect();
        assert!(
            pin.upgrade().is_none(),
            "shared feedback must not root its callback"
        );
        cache.prune_dead();
        assert!(cache
            .site
            .entries
            .iter()
            .all(|entry| entry.get().callee == 0));
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
        let env = engine.interp.global_env.clone();
        let callee = engine
            .interp
            .get_var("callback", &env)
            .unwrap_or_else(|_| panic!("callback"));
        let entry = Callback::new(callee);
        for n in 0..160 {
            let arg = engine.interp.make_array(vec![Value::Num(n as f64)]);
            let result = entry.call(&mut engine.interp, Value::Undefined, [arg]);
            assert!(matches!(result, Ok(Value::Undefined)));
        }
        let object = entry.callee.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user")
        };
        let chunk = user.func.code.get().and_then(Option::as_ref).unwrap();
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
        let env = engine.interp.global_env.clone();
        let value = engine
            .interp
            .get_var("laterHot", &env)
            .unwrap_or_else(|_| panic!("laterHot"));
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user")
        };
        let chunk = user.func.code.get().and_then(Option::as_ref).unwrap();
        if crate::bytecode::inline_recompile_at() == 100 && chunk.jit.get().flatten().is_some() {
            assert!(
                user.func.code2.get().and_then(Option::as_ref).is_some(),
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
        let env = engine.interp.global_env.clone();
        let value = engine
            .interp
            .get_var("arrowHot", &env)
            .unwrap_or_else(|_| panic!("arrowHot"));
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user")
        };
        let chunk = user.func.code.get().and_then(Option::as_ref).unwrap();
        if crate::bytecode::inline_recompile_at() == 100 && chunk.jit.get().flatten().is_some() {
            assert!(user.func.code2.get().and_then(Option::as_ref).is_some());
        }
    }
}
