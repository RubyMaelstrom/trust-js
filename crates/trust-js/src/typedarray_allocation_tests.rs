//! ECMA-262 e28783d5: TypedArray constructors, AllocateTypedArrayBuffer,
//! InitializeTypedArrayFromTypedArray/ArrayLike and ArrayBufferCopyAndDetach.
//! Local spec.html:43469,43531,43635,43657,45553.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = engine
            .eval(source, false)
            .expect("allocation fixture parses");
        assert!(
            matches!(&result, Completion::Value(value) if value == "ok"),
            "{tier:?}"
        );
    }
}

#[test]
fn typedarray_image_buffers_use_byte_ceiling_not_array_list_ceiling() {
    check(
        r#"
        function assert(x) { if (!x) throw Error('buffer allocation invariant'); }
        const count = 2 * 1024 * 1024 + 4;
        let pixels = new Uint8ClampedArray(count);
        assert(pixels.length === count && pixels[0] === 0 && pixels[count-1] === 0);
        pixels[count-1] = 255;
        const copied = new Uint8ClampedArray(pixels.subarray(4));
        assert(copied.length === count-4 && copied[copied.length-1] === 255 && copied.buffer !== pixels.buffer);
        const moved = pixels.buffer.transfer(count+4);
        assert(pixels.length === 0 && moved.byteLength === count+4);
        assert(new Uint8Array(moved)[count-1] === 255 && new Uint8Array(moved)[count+3] === 0);
        for (const TA of [Uint8Array,Uint8ClampedArray,Uint16Array,Int32Array,Float16Array,Float32Array,Float64Array,BigInt64Array]) {
            const a = new TA(1048577);
            assert(a.length === 1048577 && a.byteLength === 1048577*TA.BYTES_PER_ELEMENT);
            let limited = false;
            try { new TA(268435456/TA.BYTES_PER_ELEMENT+1); } catch(e) { limited = e instanceof RangeError; }
            assert(limited);
        }
        'ok'
    "#,
    );
}

#[test]
fn typedarray_constructor_copies_slots_and_preserves_float_bits() {
    check(
        r#"
        function assert(x) { if (!x) throw Error('typed source invariant'); }
        const bytes = new Uint32Array([0xdeadbeef,0x7fa12345,0x80000000,0x12345678]);
        const source = new Float32Array(bytes.buffer,4,2);
        for (const key of [Symbol.iterator,'length','constructor','buffer'])
            Object.defineProperty(source,key,{get(){throw Error('Must not read '+String(key));}});
        const copy = new Float32Array(source);
        const bits = new Uint32Array(copy.buffer);
        assert(bits.length === 2 && bits[0] === 0x7fa12345 && bits[1] === 0x80000000);
        const numbers = new Uint8ClampedArray(new Int16Array([-2,128,300]));
        assert(numbers.join(',') === '0,128,255');
        assert(new BigUint64Array(new BigInt64Array([-1n]))[0] === 18446744073709551615n);
        for (const [TA, value] of [[Uint8Array,new BigInt64Array(0)],[BigInt64Array,new Uint8Array(0)]]) {
            let mixed = false;
            try { new TA(value); } catch(e) { mixed = e instanceof TypeError; }
            assert(mixed);
        }
        const sab = new SharedArrayBuffer(4); new Uint8Array(sab)[2] = 71;
        const sharedCopy = new Uint8Array(new Uint8Array(sab));
        assert(sharedCopy[2] === 71 && sharedCopy.buffer instanceof ArrayBuffer);
        const rab = new ArrayBuffer(8,{maxByteLength:16}), tracking = new Uint16Array(rab,2);
        rab.resize(12); tracking[4] = 1234;
        const fixed = new Uint16Array(tracking);
        assert(fixed.length === 5 && fixed[4] === 1234 && fixed.buffer.resizable === false);
        rab.resize(0);
        let invalid = false;
        try { new Uint16Array(tracking); } catch(e) { invalid = e instanceof TypeError; }
        assert(invalid);
        'ok'
    "#,
    );
}

#[test]
fn typedarray_constructor_observes_prototype_once_and_interleaves_arraylike_reads() {
    check(
        r#"
        function assert(x) { if (!x) throw Error('constructor ordering invariant'); }
        let order = [], reads = 0;
        const proto = {};
        const target = new Proxy(function(){}, {get(t,k) {
            if (k === 'prototype') { reads++; order.push('prototype'); return proto; }
            return Reflect.get(t,k);
        }});
        const a = Reflect.construct(Uint8Array,[{ get length(){order.push('length');return 2;},
            get 0(){ order.push('get0'); return {valueOf(){order.push('convert0');return 7;}}; },
            get 1(){order.push('get1');return 8;} }],target);
        assert(reads === 1 && Object.getPrototypeOf(a) === proto && a[0] === 7 && a[1] === 8);
        assert(order.join(',') === 'prototype,length,get0,convert0,get1');
        const buffer = new ArrayBuffer(4), view = new Uint8Array(buffer);
        let invalid = false;
        const detachTarget = new Proxy(function(){},{get(t,k){if(k==='prototype'){buffer.transfer();return proto;}return Reflect.get(t,k);}});
        try { Reflect.construct(Uint8Array,[view],detachTarget); } catch(e) { invalid = e instanceof TypeError; }
        assert(invalid);
        reads = 0;
        let limited = false;
        try { Reflect.construct(Float64Array,[Number.MAX_SAFE_INTEGER],target); } catch(e) { limited = e instanceof RangeError; }
        assert(limited && reads === 1);
        let indexRead = false;
        try { new Float64Array({length:268435456/8+1,get 0(){indexRead=true;return 0;}}); } catch(e) { assert(e instanceof RangeError); }
        assert(!indexRead);
        'ok'
    "#,
    );
}
