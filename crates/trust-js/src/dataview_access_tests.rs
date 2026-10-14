//! GetViewValue/SetViewValue, ECMA-262 e28783d5, spec.html:46615/46645. A fresh buffer
//! witness follows index/value coercion; a primitive argument may omit native activation.
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("DataView fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn dataview_accesses_preserve_noncoercing_operands_and_method_identity() {
    // Expected result verified with Node v24.21.0. Warm the same call sites before changing
    // argument categories, receivers and methods, so the JIT consumes live call-cache entries.
    let source = r#"(function(){
        var b=new ArrayBuffer(32),d=new DataView(b,1,24),out=[];
        function read(d,p,e){return d.getUint32(p,e);}
        function write(d,p,v,e){return d.setUint32(p,v,e);}
        function bigread(d,p,e){return d.getBigInt64(p,e);}
        function bigwrite(d,p,v,e){return d.setBigInt64(p,v,e);}
        function err(f){try{return String(f());}catch(e){return e.name;}}
        for(var k=0;k<100;k++){write(d,1,k,true);read(d,1,true);bigwrite(d,8,9n,true);bigread(d,8,true);}
        var endian={valueOf(){throw 'ToBoolean must not coerce';}};
        write(d,1,0x12345678,endian);out.push(read(d,1,endian),read(d,'1',false));
        out.push(write(d,-0.75,1,true),read(d,-0.75,true),read(d,NaN,true));
        out.push(err(()=>read(d,-1.5)),err(()=>read(d,Infinity)),err(()=>read(d,1n)),err(()=>read(d,Symbol())));
        out.push(err(()=>write(d,0,1n)),err(()=>write(d,0,Symbol())),err(()=>read(d,21)));
        write(d,0,'65535',true);out.push(read(d,undefined,true));
        write(d,null,true,true);out.push(read(d,false,true));
        bigwrite(d,8,'18446744073709551615',endian);out.push(bigread(d,8,endian));
        out.push(err(()=>bigwrite(d,8,2,true)),err(()=>bigread(d,18,true)));
        out.push(err(()=>new DataView(new ArrayBuffer(4),1,Number.MAX_SAFE_INTEGER)));
        var original=DataView.prototype.getUint32;
        DataView.prototype.getUint32=function(){return 77;};out.push(read(d,0,true));
        DataView.prototype.getUint32=original;out.push(read(d,0,true));
        var hit=0,coerce={valueOf(){hit++;return 0;}};
        out.push(err(()=>read({getUint32:original},coerce,true)),hit);
        out.push(err(()=>original.call(new Proxy(d,{}),0,true)));
        out.push(original.call(d,0,true),Reflect.apply(original,d,[0,true]));
        var bytes=new Uint8Array(b);out.push(bytes[0],bytes[1],bytes[2],bytes[3],bytes[4]);
        return out.join('|');
    })()"#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source),
            "305419896|2018915346||1|1|RangeError|RangeError|TypeError|TypeError|TypeError|TypeError|RangeError|65535|1|-1|TypeError|RangeError|RangeError|77|1|TypeError|0|TypeError|1|1|0|1|0|0|0",
            "{tier:?}");
    }
}

#[test]
fn dataview_witnesses_follow_all_index_and_value_effects() {
    // Node-verified conversion order, including a temporarily out-of-bounds fixed view.
    let source = r#"(function(){
        var b=new ArrayBuffer(8,{maxByteLength:32}),fixed=new DataView(b,0,8),d=new DataView(b),out=[],trace='';
        function read(d,p){return d.getUint16(p,true);}
        function write(d,p,v){return d.setUint16(p,v,true);}
        function bigwrite(d,p,v){return d.setBigInt64(p,v,true);}
        function err(f){try{return String(f());}catch(e){return e.name;}}
        for(var k=0;k<100;k++){write(d,0,k);read(d,0);bigwrite(d,0,1n);}
        write(fixed,{valueOf(){trace+='o';b.resize(4);return 1;}},{valueOf(){trace+='v';b.resize(16);return 258;}});
        out.push(trace,read(fixed,1),read(d,1),d.byteLength);trace='';
        out.push(err(()=>write(fixed,{valueOf(){trace+='o';return 1;}},{valueOf(){trace+='v';b.resize(4);return 259;}})),trace);trace='';
        b.resize(16);out.push(err(()=>read(d,{valueOf(){trace+='o';b.resize(1);return 0;}})),trace);trace='';
        b.resize(16);out.push(err(()=>read(fixed,{valueOf(){trace+='o';b.resize(4);return 0;}})),trace);trace='';
        b.resize(16);bigwrite(d,{valueOf(){trace+='o';b.resize(1);return 0;}},{valueOf(){trace+='v';b.resize(16);return -5n;}});
        out.push(trace,d.getBigInt64(0,true));trace='';
        var marker={},same=false;try{write(d,{valueOf(){trace+='o';throw marker;}},{valueOf(){trace+='v';return 1;}});}catch(e){same=e===marker;}out.push(same,trace);trace='';
        var detached=new ArrayBuffer(8),dd=new DataView(detached);
        out.push(err(()=>write(dd,{valueOf(){trace+='o';detached.transfer();return 0;}},{valueOf(){trace+='v';return 1;}})),trace);trace='';
        var detached2=new ArrayBuffer(8),dd2=new DataView(detached2);
        out.push(err(()=>read(dd2,{valueOf(){trace+='o';detached2.transfer();return 0;}})),trace);
        var sab=new SharedArrayBuffer(8,{maxByteLength:16}),sd=new DataView(sab),sf=new DataView(sab,0,8);
        write(sd,6,513);sab.grow(16);write(sd,14,1027);
        out.push(sd.byteLength,sf.byteLength,read(sd,6),read(sd,14),err(()=>read(sf,14)));
        return out.join('|');
    })()"#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source),
            "ov|258|258|16|TypeError|ov|RangeError|o|TypeError|o|ov|-5|true|o|TypeError|ov|TypeError|o|16|8|513|1027|RangeError",
            "{tier:?}");
        engine.interp.gc_collect();
    }
}

#[test]
fn dataview_foreign_natives_preserve_error_realm() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var foreign=$262.createRealm().global;
            var d=new DataView(new ArrayBuffer(8));
            d.getUint32=foreign.DataView.prototype.getUint32;
            function read(d,p){return d.getUint32(p,true);}
            for(var k=0;k<100;k++)read(d,0);
            var foreignError=false;
            try{read(d,9);}catch(e){foreignError=e instanceof foreign.RangeError && !(e instanceof RangeError);}
            foreignError
        "#
            ),
            "true",
            "{tier:?}"
        );
    }
}
