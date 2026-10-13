//! ECMA-262 e28783d5: TypedArray construction from and `set` of another element type
//! (RawBytesToNumeric, NumericToRawBytes), TypedArray sort, DataView GetViewValue/SetViewValue,
//! BigInt element stores and Atomics read-modify-write encodings. Expected values from Node v24;
//! every bulk conversion is also checked against single-element stores.
use crate::bytecode::Tier;
use crate::{Completion, Engine};

#[test]
fn element_conversions_and_view_accesses_keep_their_encodings() {
    let source = r#"(function(){
    var r=[],bad=[];
    var kinds=[Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array];
    var values=[0,-0,1,-1,127,128,255,256,-129,32767,32768,65535,65536,2147483647,2147483648,4294967295,4294967296,-2147483649,1.5,2.5,-1.5,0.49999,1e10,-1e10,1e300,NaN,Infinity,-Infinity,3.4e38,1e-45,65504,65520,0.1];
    function same(x,y){return Object.is(x,y)}
    var f64=new Float64Array(values);
    for (var K of kinds) {
        var a=new K(f64);
        for (var K2 of kinds) {
            var made=new K2(a),set=new K2(a.length+2);set.set(a,1);
            for (var k=0;k<a.length;k++){var one=new K2(1);one[0]=a[k];if(!same(made[k],one[0])||!same(set[k+1],one[0]))bad.push(K.name+'>'+K2.name+'@'+k);}
            if(set[0]!==0||set[a.length+1]!==0)bad.push(K.name+'>'+K2.name+' bounds');
        }
    }
    r.push(bad.length?bad.slice(0,5).join():'ok');
    r.push(Array.from(new Uint8Array(new Float32Array([1.5,-1,300,NaN]))).join(','),Array.from(new Int16Array(new Float64Array([1e10,-32769,0.9]))).join(','),Array.from(new Uint8ClampedArray(new Float64Array([2.5,3.5,-1,256]))).join(','),Array.from(new Float32Array(new Int32Array([16777217,-1]))).join(','));
    var big=new BigInt64Array([1n,-1n,2n**63n-1n,-(2n**63n)]);r.push(Array.from(new BigUint64Array(big)).join(','),Array.from(new BigInt64Array(new BigUint64Array(big))).join(','));
    var buf=new ArrayBuffer(16);var u8=new Uint8Array(buf);for(var k=0;k<16;k++)u8[k]=k*17;var u16=new Uint16Array(buf,2,4);
    u8.set(u16,1);r.push(Array.from(u8).join(','));
    var f32=new Float32Array(buf,0,2);var i8=new Int8Array(buf,4,8);f32.set(i8.subarray(0,2));r.push(Array.from(new Uint8Array(buf)).join(','));
    var i16=new Int16Array(buf,0,8);u8.set(i16.subarray(1,5),3);r.push(Array.from(u8).join(','));
    r.push(Array.from(new Float32Array([3,-0,0,NaN,-Infinity,1.5,-2,NaN,Infinity]).sort(),function(x){return Object.is(x,-0)?'-0':String(x)}).join(','),Array.from(new Int16Array([5,-3,32767,-32768,0]).sort()).join(','));
    var dv=new DataView(new ArrayBuffer(24));
    dv.setInt16(0,-2);dv.setUint16(2,65535,true);dv.setInt32(4,-123456789);dv.setFloat32(8,1.5,true);dv.setFloat64(12,-0.1);dv.setInt8(20,-1);dv.setUint8(21,300);
    r.push(Array.from(new Uint8Array(dv.buffer)).join(','),dv.getInt16(0),dv.getUint16(2,true),dv.getUint16(2),dv.getInt32(4),dv.getFloat32(8,true),dv.getFloat64(12),dv.getInt8(20),dv.getUint8(21),dv.getUint32(0,true));
    dv.setBigInt64(0,-5n);dv.setBigUint64(8,2n**64n-2n,true);r.push(dv.getBigInt64(0),dv.getBigUint64(0),dv.getBigUint64(8,true),dv.getBigInt64(8,true));
    function show(f){try{return String(f())}catch(e){return e.name}}
    r.push(show(()=>dv.getInt32(22)),show(()=>dv.setFloat64(17,1)),show(()=>dv.getBigInt64(20)));
    var sdv=new DataView(new SharedArrayBuffer(8));sdv.setUint32(0,0xdeadbeef);sdv.setInt16(4,-300,true);r.push(sdv.getUint32(0),sdv.getUint8(0),sdv.getInt16(4,true),sdv.getUint16(4));
    function store(a,k,v){a[k]=v;return a[k]}
    var b=new BigInt64Array(4),u=new BigUint64Array(4);
    for(var n=0;n<3;n++)r.push(store(b,0,-3n),store(b,1,2n**70n+9n),store(u,2,-1n),store(b,9,5n),show(()=>store(b,0,1)),show(()=>store(new Float64Array(1),0,1n)));
    var sab=new BigInt64Array(new SharedArrayBuffer(16));for(var n=0;n<2;n++)r.push(store(sab,1,-(2n**63n)));
    for (var A of [Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array]) {var x=new A(2);x[0]=A===Int8Array?127:A===Int16Array?32767:A===Int32Array?2147483647:0;r.push(A.name+':'+Atomics.add(x,0,1)+','+x[0]+','+Atomics.sub(x,1,1)+','+x[1]+','+Atomics.exchange(x,0,-7)+','+x[0]);}
    var s32=new Int32Array(new SharedArrayBuffer(16));r.push(Atomics.add(s32,0,2147483647),Atomics.add(s32,0,1),s32[0]);
    var sf=new Float64Array(new SharedArrayBuffer(16));sf[0]=1.25;sf[1]=-0;r.push(sf[0],Object.is(sf[1],-0));
    return r.join('|');
})()"#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("typed array fixture parses")
        {
            Completion::Value(actual) => assert_eq!(
                actual,
                "ok|1,255,44,0|-7168,32767,0|2,4,0,255|16777216,-1|1,18446744073709551615,9223372\
                 036854775807,9223372036854775808|1,-1,9223372036854775807,-9223372036854775808|0\
                 ,34,68,102,136,85,102,119,136,153,170,187,204,221,238,255|0,0,240,194,0,0,170,66\
                 ,136,153,170,187,204,221,238,255|0,0,240,240,0,170,136,66,136,153,170,187,204,22\
                 1,238,255|-Infinity,-2,-0,0,1.5,3,Infinity,NaN,NaN|-32768,-3,0,5,32767|255,254,2\
                 55,255,248,164,50,235,0,0,192,63,191,185,153,153,153,153,153,154,255,44,0,0|-2|6\
                 5535|65535|-123456789|1.5|-0.1|-1|44|4294967039|-5|18446744073709551611|18446744\
                 073709551614|-2|RangeError|RangeError|RangeError|3735928559|222|-300|54526|-3|9|\
                 18446744073709551615||TypeError|TypeError|-3|9|18446744073709551615||TypeError|T\
                 ypeError|-3|9|18446744073709551615||TypeError|TypeError|-9223372036854775808|-92\
                 23372036854775808|Int8Array:127,-128,0,-1,-128,-7|Uint8Array:0,1,0,255,1,249|Int\
                 16Array:32767,-32768,0,-1,-32768,-7|Uint16Array:0,1,0,65535,1,65529|Int32Array:2\
                 147483647,-2147483648,0,-1,-2147483648,-7|Uint32Array:0,1,0,4294967295,1,4294967\
                 289|0|2147483647|-2147483648|1.25|true",
                "{tier:?}"
            ),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}
