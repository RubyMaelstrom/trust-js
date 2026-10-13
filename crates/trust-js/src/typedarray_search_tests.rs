//! ECMA-262 e28783d5, spec.html:42775/42831: TypedArray indexOf/lastIndexOf;
//! 5841: ToClampedIndex; 5345: ToIntegerOrInfinity.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert!(
            matches!(engine.eval(source, false).expect("typed search fixture parses"),
            Completion::Value(ref value) if value == "ok"),
            "{tier:?}"
        );
    }
}

#[test]
fn typedarray_search_clamps_extreme_start_indices() {
    check(
        r#"
        function test(TA, big) {
            const values = big ? [42n,43n,43n,41n] : [42,43,43,41];
            const a = new TA(values), search = values[1];
            const starts = [-Infinity,-9,-4,-3,-1,-0,0,0.9,1.1,3,4,5,1e100,Infinity,NaN,undefined];
            const forward = [1,1,1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1];
            const backward = [-1,-1,-1,1,2,-1,-1,-1,1,2,2,2,2,2,-1,-1];
            for (let k=0; k<starts.length; k++) {
                if (a.indexOf(search,starts[k]) !== forward[k]) throw Error('forward '+k);
                if (a.lastIndexOf(search,starts[k]) !== backward[k]) throw Error('backward '+k);
            }
            if (a.indexOf(search) !== 1 || a.lastIndexOf(search) !== 2) throw Error('absent');
            if (a.indexOf(big ? 99n : 99) !== -1) throw Error('missing');
        }
        for (const TA of [Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,
                         Int32Array,Uint32Array,Float16Array,Float32Array,Float64Array]) test(TA,false);
        test(BigInt64Array,true); test(BigUint64Array,true);
        'ok'
    "#,
    );
}

#[test]
fn typedarray_search_preserves_coercion_order_and_live_buffer_bounds() {
    check(
        r#"
        function assert(value) { if (!value) throw Error('search invariant'); }
        let calls = 0;
        const from = {valueOf() { calls++; return Infinity; }};
        assert(new Uint8Array([]).indexOf(1,from) === -1 && calls === 0);
        assert(new Uint8Array([1]).indexOf(1,from) === -1 && calls === 1);
        const buffer = new ArrayBuffer(4, {maxByteLength:8});
        const a = new Uint8Array(buffer); a[1] = 7;
        assert(a.indexOf(7,{valueOf(){ buffer.resize(8); a[6]=7; return 5; }}) === -1);
        assert(a.indexOf(7,5) === 6);
        assert(a.indexOf(7,{valueOf(){ buffer.resize(1); return 0; }}) === -1);
        const detached = new Uint8Array([1,2]);
        assert(detached.indexOf(undefined,{valueOf(){ $262.detachArrayBuffer(detached.buffer); return 0; }}) === -1);
        let caught = false;
        try { new Uint8Array([1]).indexOf(1,Symbol()); } catch(e) { caught = e instanceof TypeError; }
        assert(caught);
        'ok'
    "#,
    );
}
