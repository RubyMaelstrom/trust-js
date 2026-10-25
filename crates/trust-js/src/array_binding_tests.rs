use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("array binding fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

#[test]
fn dense_array_bindings_retain_captured_values_and_use_the_closed_prefix() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let before = super::array_binding::TEST_ARRAY_BINDING_OPENS.with(std::cell::Cell::get);
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        let native_before =
            super::array_binding::TEST_JIT_ARRAY_BINDING_OPENS.with(std::cell::Cell::get);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function bind(records){
                let saved=[];
                for(const [key,value] of records) saved.push(()=>[key,value]);
                return saved;
            }
            var values=[['a',1],['b',{id:7}],['c',undefined]],saved=[];
            for(var n=0;n<120;n++) saved=bind(values);
            var correct=saved[0]()[0]==='a'&&saved[1]()[1]===values[1][1]&&saved[2]()[1]===undefined;
            values=null;
            correct
        "#
            ),
            "true",
            "{tier:?}"
        );
        let opens =
            super::array_binding::TEST_ARRAY_BINDING_OPENS.with(std::cell::Cell::get) - before;
        if tier != Tier::Interp {
            assert!(
                opens >= 300,
                "compiled fresh bindings must enter the path: {tier:?}, {opens}"
            );
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit {
            let native = super::array_binding::TEST_JIT_ARRAY_BINDING_OPENS
                .with(std::cell::Cell::get)
                - native_before;
            assert!(
                native >= 300,
                "native entries must use the closed prefix: {native}"
            );
        }
        engine.interp.gc_collect();
        assert_eq!(
            eval(&mut engine, "saved[1]()[1].id+':'+saved[0]()[0]"),
            "7:a"
        );
    }
}

#[test]
fn array_bindings_keep_custom_next_closing_and_index_getter_order() {
    let source = r#"
        function bind(records){let out=[];for(const [key,value] of records)out.push(()=>key+':'+value);return out.map(f=>f()).join(',')}
        var a=[1,2,3],proto=Object.getPrototypeOf(a.values()),original=proto.next,trace='';
        Object.defineProperty(proto,'return',{configurable:true,get(){trace+='r';return function(){trace+='c'+this.next().value;return {}}}});
        var close=bind([a]);delete proto.return;
        Object.defineProperty(a,'0',{configurable:true,get(){trace+='g';a[1]=9;return 5}});
        var getter=bind([a]);
        proto.next=function(){var result=original.call(this);trace+='n';return result};
        var next=bind([[7,8]]);proto.next=original;
        var custom=bind([{[Symbol.iterator](){let n=0;return {next(){return {value:++n,done:false}},return(){trace+='x';return {}}}}}]);
        [close,getter,next,custom,trace].join('|')
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "1:2|5:9|7:8|1:2|rc3gnnnnx",
            "{tier:?}"
        );
    }
}

#[test]
fn array_bindings_preserve_holes_exhaustion_defaults_and_observable_initializations() {
    let source = r#"
        function bind(records){let out=[];for(const [key,value] of records)out.push(()=>[key,value]);return out}
        var a=[,2],proto=Object.getPrototypeOf(a),reads=0;
        Object.defineProperty(proto,'0',{configurable:true,get(){reads++;return 11},set(v){Object.defineProperty(this,'0',{value:v,writable:true,configurable:true,enumerable:true})}});
        var hole=bind([a]);delete proto[0];
        var short=bind([[4]]),empty=bind([[]]),nil=bind([[undefined,null]]);
        var defaults=0;function defaultBind(record){let [a=(defaults++,9),b=10]=record;return ()=>a+b}
        var d=defaultBind([undefined]);
        var observe=[];
        function observeBind(){
            function read(){return a}
            var custom={[Symbol.iterator](){let index=0;return {next(){if(index++)observe.push(read());return {value:index,done:false}},return(){return {}}}}};
            const [a,b]=custom;return ()=>b
        }
        var observed=observeBind()();
        [hole[0]()[0],reads,short[0]()[0],short[0]()[1]===undefined,empty[0]()[0]===undefined,nil[0]()[1]===null,d(),defaults,observed,observe.join(',')].join('|')
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "11|1|4|true|true|true|19|1|2|1",
            "{tier:?}"
        );
    }
}

#[test]
fn flat_array_patterns_keep_every_observable_step_of_the_iterator_protocol() {
    // Expected values from Node v24. `flat` compiles to the batched Op::DestructureArr walk and
    // `captured` to the closed TryArrayBinding prefix; the tree-walker binds both directly.
    let source = r#"(function(){
        var log=[];
        function flat(s){const [a,b,c]=s;return [a,b,c].join(',')}
        function captured(s){const [a,b]=s;return (()=>a+','+b)()}
        function hole(s){const [,b]=s;return b}
        function show(f){try{return f()}catch(e){return e.name}}
        var AIP=Object.getPrototypeOf([].values()),IP=Object.getPrototypeOf(AIP);
        var r=[];
        r.push(flat([1,2,3,4]),flat([1]),flat([1,,3]),captured([5,6]),captured([7]));
        Array.prototype[1]='P';r.push(flat([1,,3]),hole([0,,2]),captured([8,,]));delete Array.prototype[1];
        Object.prototype[1]='O';r.push(flat([1,,3]));delete Object.prototype[1];
        var a=[1,2,3];Object.defineProperty(a,1,{get(){log.push('get1');a.length=1;return 'g'}});
        r.push(flat(a),log.splice(0).join());
        a=[1,2,3,4];Object.defineProperty(a,0,{get(){AIP.return=function(){log.push('ret:'+this.next().value);return {}};return 'g'}});
        r.push(captured(a),flat(a),log.splice(0).join());delete AIP.return;
        AIP.return=function(){log.push('r'+(Object.getPrototypeOf(this)===AIP));return {}};
        r.push(flat([1,2,3,4]),flat([1]),captured([1,2,3]),log.splice(0).join());delete AIP.return;
        Object.prototype.return=function(){log.push('o');return {}};r.push(flat([1,2,3,4]),log.splice(0).join());delete Object.prototype.return;
        Object.defineProperty(IP,'return',{configurable:true,get(){log.push('getter');return undefined}});
        r.push(flat([1,2,3,4]),flat([1,2,3]),log.splice(0).join());delete IP.return;
        AIP.return=5;r.push(show(()=>flat([1,2,3,4])));AIP.return=function(){return 1};r.push(show(()=>flat([1,2,3,4])));delete AIP.return;
        a=[1,2,3];a[Symbol.iterator]=function*(){yield 'x';yield 'y'};r.push(flat(a),captured(a));
        var values=Array.prototype[Symbol.iterator];Array.prototype[Symbol.iterator]=function*(){yield 'r'};
        r.push(flat([1,2]),captured([1,2]));Array.prototype[Symbol.iterator]=values;
        var next=AIP.next;AIP.next=function(){var x=next.call(this);if(!x.done)x.value*=10;return x};
        r.push(flat([1,2]),captured([3,4]));AIP.next=next;
        class A extends Array{}r.push(flat(A.from([5,6])));
        var p=new Proxy([7,8],{get(t,k,v){log.push(typeof k==='symbol'?'sym':k);return Reflect.get(t,k,v)}});
        r.push(flat(p),log.splice(0).join());
        r.push(flat(Object.freeze([1,2])),flat(new Uint8Array([3,4])),flat('ab'),(function(){return flat(arguments)})(9,8));
        r.push(show(()=>flat(null)),show(()=>flat(5)));
        var s=[1,2];Object.setPrototypeOf(s,{0:'sp',[Symbol.iterator]:values});r.push(hole([0,,2]),flat(s));
        return r.join('|');
    })()"#;
    let expected = "1,2,3|1,,|1,,3|5,6|7,undefined|1,P,3|P|8,P|1,O,3|1,g,|get1|g,2|g,2,3|\
        ret:3,ret:4|1,2,3|1,,|1,2|rtrue,rtrue|1,2,3|o|1,2,3|1,2,3|getter,getter|TypeError|\
        TypeError|x,y,|x,y|r,,|r,undefined|10,20,|30,40|5,6,|7,8,|sym,length,0,length,1,length|\
        1,2,|3,4,|a,b,|9,8,|TypeError|TypeError||1,2,";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
    }
}

#[test]
fn flat_array_patterns_allocate_no_iterator_and_respect_foreign_realms() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            "function sum(s,n){var t=0;for(var i=0;i<n;i++){const [a,b]=s;t+=a+b}return t}\
             var pair=[1,2];sum(pair,4);",
        );
        let before = crate::value::heap_allocated_objects(&engine.interp.gc_heap);
        assert_eq!(eval(&mut engine, "sum(pair,100)"), "300", "{tier:?}");
        assert_eq!(
            crate::value::heap_allocated_objects(&engine.interp.gc_heap),
            before,
            "no Array Iterator per destructuring: {tier:?}"
        );
        // GetIterator calls the @@iterator function it finds, which creates its own Realm's
        // iterator; an Array of another Realm iterates with that Realm's protocol.
        assert_eq!(
            eval(
                &mut engine,
                r#"(function(){
                var other=$262.createRealm().global;
                var otherNext=Object.getPrototypeOf(other.Array.prototype.values.call([]));
                otherNext.next=function(){return {value:'other',done:false}};
                function two(s){const [a,b]=s;return a+b}
                var values=Array.prototype[Symbol.iterator];
                Array.prototype[Symbol.iterator]=other.Array.prototype.values;
                var foreign=two([1,2]);
                Array.prototype[Symbol.iterator]=values;
                return [foreign,two([1,2]),two(new other.Array(5,6))].join('|');
            })()"#
            ),
            "otherother|3|otherother",
            "{tier:?}"
        );
    }
}
