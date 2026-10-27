//! Array.prototype.indexOf, lastIndexOf and includes (ECMA-262 §23.1.3.17, §23.1.3.20,
//! §23.1.3.16) over an ordinary Array's packed elements, read in place: IsStrictlyEqual and
//! SameValueZero, holes and indices past the storage under the element protector, and every
//! case that must keep the specified HasProperty and Get steps.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("array search fixture parses")
        {
            Completion::Value(ref value) if value == "ok" => {}
            Completion::Value(value) => panic!("{tier:?}: {value}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

const ASSERT: &str = "function assert(value, message) { if (!value) throw Error(message); }\n";

#[test]
fn packed_searches_compare_as_specified() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        var obj = {}, sym = Symbol();
        var a = [1, 'two', 3.5, -0, NaN, null, undefined, true, 2n, obj, sym, 'tw' + 'o', 0];
        for (var k = 0; k < 20; k++) {
            assert(a.indexOf(1) === 0 && a.indexOf('two') === 1 && a.indexOf(3.5) === 2, 'scalars');
            assert(a.indexOf(0) === 3 && a.indexOf(-0) === 3 && a.lastIndexOf(-0) === 12, 'zeros');
            assert(a.includes(0) && a.includes(-0), 'zeros included');
            assert(a.indexOf(NaN) === -1 && a.lastIndexOf(NaN) === -1 && a.includes(NaN), 'NaN');
            assert(a.indexOf(null) === 5 && a.indexOf(undefined) === 6 && a.indexOf(true) === 7, 'words');
            assert(a.indexOf(false) === -1 && !a.includes(false) && a.indexOf('1') === -1, 'misses');
            assert(a.indexOf(2n) === 8 && a.includes(BigInt(2)) && a.indexOf(2) === -1, 'BigInt');
            assert(a.indexOf(obj) === 9 && a.indexOf({}) === -1 && a.indexOf(sym) === 10, 'identities');
            assert(a.indexOf(Symbol()) === -1 && !a.includes(Symbol()), 'another Symbol');
            assert(a.lastIndexOf('two') === 11 && a.indexOf('two', 2) === 11, 'equal strings');
            assert(a.indexOf('tw') === -1 && !a.includes('twoo'), 'other strings');
        }
        // fromIndex, as the generic algorithm reads it on an array-like copy.
        var b = [5, 6, 5, 6];
        var like = { length: 4, 0: 5, 1: 6, 2: 5, 3: 6 };
        var froms = [-Infinity, -5, -4, -2, -1, -0, 0, 0.5, 1, 2.9, 3, 4, 5, Infinity, NaN, undefined, '2'];
        for (var f of froms) {
            for (var v of [5, 6, 7]) {
                var ap = Array.prototype;
                assert(b.indexOf(v, f) === ap.indexOf.call(like, v, f), 'indexOf ' + v + ' ' + f);
                assert(b.lastIndexOf(v, f) === ap.lastIndexOf.call(like, v, f), 'lastIndexOf ' + v + ' ' + f);
                assert(b.includes(v, f) === ap.includes.call(like, v, f), 'includes ' + v + ' ' + f);
            }
        }
        assert(b.lastIndexOf(5) === 2 && b.lastIndexOf(6) === 3 && [].indexOf(undefined) === -1, 'defaults');
        // A long array, beyond the inline element slots.
        var big = [];
        for (var k = 0; k < 1000; k++) big.push(k % 7);
        assert(big.indexOf(6) === 6 && big.lastIndexOf(0) === 994 && !big.includes(7), 'long');
        assert(big.indexOf(3, 990) === 990 && big.lastIndexOf(3, 9) === 3, 'long, from');
        'ok'
    "#
    ));
}

#[test]
fn holes_and_absent_indices_read_through_the_prototypes() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        var h = [1, , 3];
        assert(h.indexOf(undefined) === -1 && h.lastIndexOf(undefined) === -1, 'holes skipped');
        assert(h.includes(undefined) && !h.includes(2), 'a hole reads undefined');
        var past = [1];
        past.length = 4;
        assert(past.indexOf(undefined) === -1 && past.includes(undefined), 'past the storage');
        assert(past.lastIndexOf(undefined) === -1 && !past.includes(undefined, 4), 'past the length');
        // An index on a prototype answers HasProperty and Get for a hole or an absent index.
        Array.prototype[1] = 'array';
        Array.prototype[2] = 'array';
        assert(h.indexOf('array') === 1 && h.lastIndexOf('array') === 1 && h.includes('array'), 'Array.prototype');
        assert(past.indexOf('array') === 1 && past.lastIndexOf('array') === 2, 'past, inherited');
        assert(!h.includes(undefined) && past.includes(undefined), 'inherited, not undefined');
        delete Array.prototype[1];
        delete Array.prototype[2];
        Object.prototype[1] = 'object';
        assert(h.indexOf('object') === 1 && h.includes('object') && past.lastIndexOf('object') === 1, 'Object.prototype');
        delete Object.prototype[1];
        assert(h.indexOf('object') === -1 && h.includes(undefined), 'restored');
        // (A counter, not a log: pushing onto an Array would meet the getter-only index.)
        var gets = 0;
        Object.defineProperty(Array.prototype, 1, { get() { gets++; return 'got'; }, configurable: true });
        assert(h.indexOf('got') === 1 && gets === 1, 'an inherited getter runs');
        assert(!h.includes('none') && gets === 2, 'includes reads it');
        assert(h.indexOf(1) === 0 && gets === 2, 'not before a match');
        delete Array.prototype[1];
        // An Array of another prototype keeps the specified steps.
        class Sub extends Array {}
        var sub = Sub.from([1, 2]);
        sub.length = 3;
        Sub.prototype[2] = 'sub';
        assert(sub.indexOf('sub') === 2 && sub.includes('sub') && sub.lastIndexOf('sub') === 2, 'subclass');
        var nullProto = [1, , 3];
        Object.setPrototypeOf(nullProto, null);
        assert(nullProto.indexOf === undefined, 'no methods');
        assert(Array.prototype.indexOf.call(nullProto, undefined) === -1, 'null prototype');
        assert(Array.prototype.includes.call(nullProto, undefined), 'null prototype includes');
        'ok'
    "#
    ));
}

#[test]
fn accessors_conversions_and_other_receivers_keep_the_specified_steps() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        var log = [];
        var g = [1, 2, 3];
        Object.defineProperty(g, 1, { get() { log.push('get'); return 2; } });
        assert(g.indexOf(2) === 1 && log.length === 1, 'an own getter runs');
        assert(g.indexOf(1) === 0 && log.length === 1, 'not past a match');
        assert(g.lastIndexOf(3) === 2 && log.length === 1, 'nor before one, descending');
        assert(g.includes(3) && log.length === 2, 'includes reads it');
        // fromIndex's conversion runs after the length is read and may change the Array.
        var d = [1, 2, 3, 4];
        assert(d.indexOf(4, { valueOf() { d.length = 2; return 0; } }) === -1, 'shrunk');
        var e = [1, 2, 3];
        assert(e.indexOf(9, { valueOf() { e.push(9); return 0; } }) === -1, 'grown past the length read');
        var f = [1, 2, 3];
        assert(f.includes(undefined, { valueOf() { f.length = 1; return 0; } }), 'shrunk, includes');
        var c = [1, 2, 3], calls = 0;
        var from = { valueOf() { Object.defineProperty(c, 1, { get() { calls++; return 2; } }); return 0; } };
        assert(c.indexOf(2, from) === 1 && calls === 1, 'an accessor added by the conversion');
        var p = [1, , 3];
        var proto = { valueOf() { Object.setPrototypeOf(p, { 1: 'set' }); return 0; } };
        assert(p.indexOf('set', proto) === 1, 'a prototype set by the conversion');
        var lengthLog = [];
        var lengthFirst = { valueOf() { lengthLog.push('from'); return 0; } };
        var counted = { get length() { lengthLog.push('length'); return 1; }, 0: 'x' };
        assert(Array.prototype.indexOf.call(counted, 'x', lengthFirst) === 0, 'array-like');
        assert(lengthLog.join() === 'length,from', lengthLog.join());
        assert(Array.prototype.includes.call('abc', 'b') && Array.prototype.indexOf.call('abc', 'c') === 2, 'String');
        var proxy = new Proxy([1, 2], {});
        assert(proxy.indexOf(2) === 1 && proxy.includes(1) && proxy.lastIndexOf(1) === 0, 'Proxy');
        var typed = new Int8Array([1, 2]);
        assert(Array.prototype.indexOf.call(typed, 2) === 1, 'typed array');
        function args() { return Array.prototype.indexOf.call(arguments, 'b'); }
        assert(args('a', 'b') === 1, 'arguments');
        var frozen = Object.freeze([1, 2, NaN]);
        assert(frozen.indexOf(2) === 1 && frozen.includes(NaN), 'frozen');
        'ok'
    "#
    ));
}
