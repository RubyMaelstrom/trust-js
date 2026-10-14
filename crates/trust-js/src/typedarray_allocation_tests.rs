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
            try { new TA(Number.MAX_SAFE_INTEGER+1); } catch(e) { limited = e instanceof RangeError; }
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
        try { new Float64Array({length:Number.MAX_SAFE_INTEGER,get 0(){indexRead=true;return 0;}}); } catch(e) { assert(e instanceof RangeError); }
        assert(!indexRead);
        'ok'
    "#,
    );
}

#[test]
fn array_buffer_ceiling_matches_node_and_invalid_indices_precede_object_creation() {
    #[cfg(target_pointer_width = "64")]
    assert_eq!(crate::interpreter::MAX_BUFFER_BYTES, (1usize << 53) - 1);
    #[cfg(target_pointer_width = "32")]
    assert_eq!(crate::interpreter::MAX_BUFFER_BYTES, i32::MAX as usize);
    check(
        r#"
        function assert(x) { if (!x) throw Error('buffer ceiling invariant'); }
        for (const C of [ArrayBuffer, SharedArrayBuffer]) {
            let reads = 0;
            const target = new Proxy(function(){}, {get(t,k) {
                if (k === 'prototype') reads++;
                return Reflect.get(t,k);
            }});
            let invalid = false;
            try { Reflect.construct(C,[Number.MAX_SAFE_INTEGER+1, {get maxByteLength(){reads++;return 0;}}],target); }
            catch(e) { invalid = e instanceof RangeError; }
            assert(invalid && reads === 0);
            invalid = false;
            try { Reflect.construct(C,[0, {maxByteLength:Number.MAX_SAFE_INTEGER+1}],target); }
            catch(e) { invalid = e instanceof RangeError; }
            assert(invalid && reads === 0);
            const growable = new C(0, {maxByteLength:268435464});
            assert(growable.byteLength === 0 && growable.maxByteLength === 268435464);
        }
        'ok'
    "#,
    );
}

#[test]
fn resizable_buffer_maximum_must_be_allocatable_even_when_initial_length_is_zero() {
    check(
        r#"
        function assert(x) { if (!x) throw Error('maximum reservation invariant'); }
        for (const C of [ArrayBuffer, SharedArrayBuffer]) {
            for (const maxByteLength of [7*1125899906842624, Number.MAX_SAFE_INTEGER]) {
                let order = [], limited = false;
                const target = new Proxy(function(){}, {get(t,k) {
                    if (k === 'prototype') order.push('prototype');
                    return Reflect.get(t,k);
                }});
                const options = {get maxByteLength(){order.push('maximum');return maxByteLength;}};
                try { Reflect.construct(C,[0,options],target); }
                catch(e) { limited = e instanceof RangeError; }
                assert(limited && order.join(',') === 'maximum,prototype');
            }
        }
        const buffer = new ArrayBuffer(4,{maxByteLength:16});
        let limited = false;
        try { buffer.transfer(17); } catch(e) { limited = e instanceof RangeError; }
        assert(limited && !buffer.detached && buffer.byteLength === 4);
        assert(buffer.transferToFixedLength(17).byteLength === 17);
        'ok'
    "#,
    );
}

#[test]
fn resizable_backing_retains_its_maximum_capacity_through_growth_and_transfer() {
    use crate::interpreter::BufferData;

    fn backing(engine: &Engine) -> (usize, usize, usize) {
        assert_eq!(engine.interp.array_buffers.len(), 1);
        match engine.interp.array_buffers.values().next().unwrap() {
            BufferData::Local(bytes) => {
                let bytes = bytes.borrow();
                (bytes.as_ptr() as usize, bytes.len(), bytes.capacity())
            }
            BufferData::Shared(block) => {
                let bytes = block.lock().unwrap();
                (bytes.as_ptr() as usize, bytes.len(), bytes.capacity())
            }
        }
    }

    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for constructor in ["ArrayBuffer", "SharedArrayBuffer"] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let result = engine.eval(&format!(
                "var buffer=new {constructor}(4,{{maxByteLength:1024}});var bytes=new Uint8Array(buffer);bytes[0]=31;"
            ), false).unwrap();
            assert!(matches!(result, Completion::Value(_)));
            let (address, len, capacity) = backing(&engine);
            assert_eq!((len, capacity), (4, 1024));
            let operation = if constructor == "ArrayBuffer" {
                "resize"
            } else {
                "grow"
            };
            let result = engine.eval(&format!(
                "buffer.{operation}(128);if(bytes[0]!==31||bytes[127]!==0)throw Error('growth bytes');bytes[127]=83;"
            ), false).unwrap();
            assert!(matches!(result, Completion::Value(_)));
            assert_eq!(backing(&engine), (address, 128, capacity));
            if constructor == "ArrayBuffer" {
                let result = engine.eval(
                    "buffer.resize(4);buffer.resize(128);if(bytes[127]!==0)throw Error('shrink regrowth');var moved=buffer.transfer(64);if(!buffer.detached||moved.maxByteLength!==1024||new Uint8Array(moved)[0]!==31)throw Error('transfer reservation');",
                    false,
                ).unwrap();
                assert!(matches!(result, Completion::Value(_)));
                assert_eq!(backing(&engine), (address, 64, capacity));
            }
        }
    }
}

/// Node v24.21.0 accepts both constructors at 4 GiB on this 64-bit host. Fresh pages remain
/// uncommitted except for the endpoint writes, so this covers the old cap and 32-bit boundary
/// without consuming four GiB of resident memory.
#[cfg(target_pointer_width = "64")]
#[test]
fn array_buffers_support_four_gibibytes_and_zeroed_endpoint_access_across_tiers() {
    check(
        r#"
        function assert(x) { if (!x) throw Error('large buffer invariant'); }
        for (const C of [ArrayBuffer, SharedArrayBuffer]) {
            const length = 4294967296;
            const buffer = new C(length), bytes = new Uint8Array(buffer), view = new DataView(buffer);
            assert(buffer.byteLength === length && bytes.length === length);
            assert(bytes[0] === 0 && bytes[length-1] === 0);
            bytes[0] = 31; bytes[length-1] = 71;
            assert(view.getUint8(0) === 31 && view.getUint8(length-1) === 71);
            view.setUint8(length-1, 83);
            assert(bytes[length-1] === 83);
        }
        const moved = new ArrayBuffer(4).transfer(268435464);
        assert(moved.byteLength === 268435464 && new Uint8Array(moved)[268435463] === 0);
        'ok'
    "#,
    );
}
