//! A computed read of an array index absent from an ordinary receiver's own elements walks the
//! chain by the index itself (`Interp::ordinary_chain_get_index`), not by its decimal text: the
//! same answers as the text-keyed walk and the general [[Get]] it falls back to, at every tier
//! (ECMA-262 §7.3.2 GetV, §10.1.8.1 OrdinaryGet, §7.1.19 ToPropertyKey, §10.4.5 integer-indexed
//! exotic objects, §10.4.3 String exotic objects, §10.4.4 arguments exotic objects, §10.5 Proxy).
//! The expected string is Node's (v8) for the same program.

use crate::{bytecode::Tier, Completion, Engine};

const CASES: &str = r#"
function get(o, k) { return o[k]; }
var out = [];
function note(label, value) { out.push(label + '=' + String(value)); }
// 1. Far keys on the receiver: an absent index, a present one.
var far = {}; far[1000000] = 1; far[2000000] = 2;
note('far-absent', get(far, 1500000)); note('far-present', get(far, 2000000));
// 2. A prototype's data index.
var pd = {}; pd[1500000] = 'pd'; var o2 = Object.create(pd); o2[1000000] = 1;
note('proto-data', get(o2, 1500000));
// 3. A prototype's getter, the receiver passed.
var pg = {}; var o3;
Object.defineProperty(pg, 1500000, { get: function () { return this === o3 ? 'recv' : 'other'; } });
o3 = Object.create(pg); o3[1000000] = 1;
note('proto-getter', get(o3, 1500000));
// 4. A prototype's packed words.
var o4 = Object.create([10, 20, 30]); o4[1000000] = 1;
note('proto-words', get(o4, 1)); note('proto-words-absent', get(o4, 5));
// 5. A Proxy in the chain.
var o5;
var px = new Proxy({}, { get: function (t, k, r) { return 'proxy:' + String(k) + ':' + (r === o5); } });
o5 = Object.create(px); o5[1000000] = 1;
note('proxy', get(o5, 7));
// 6. A typed array in the chain: its index answers, an index past its length stops the walk.
var o6 = Object.create(new Int8Array([5, 6])); o6[1000000] = 1;
Object.prototype[5] = 'op';
note('typed', get(o6, 1)); note('typed-past', get(o6, 5));
delete Object.prototype[5];
// 7. A String wrapper in the chain.
var o7 = Object.create(new String('abc')); o7[1000000] = 1;
note('string', get(o7, 1)); note('string-past', get(o7, 3));
// 8. An arguments object in the chain.
var args = (function () { return arguments; })('x', 'y');
var o8 = Object.create(args); o8[1000000] = 1;
note('arguments', get(o8, 1));
// 9. Keys that are not array indices.
var p9 = {}; p9[4294967295] = 'max'; p9[4294967296] = 'big'; var o9 = Object.create(p9); o9[1000000] = 1;
note('2^32-1', get(o9, 4294967295)); note('2^32', get(o9, 4294967296));
// 10. -0, 1.5, "01" and the canonical string of an index.
var p10 = {}; p10[0] = 'zero'; p10['1.5'] = 'frac'; p10['01'] = 'lead'; p10[5] = 'five';
var o10 = Object.create(p10); o10[1000000] = 1;
note('-0', get(o10, -0)); note('1.5', get(o10, 1.5)); note('"01"', get(o10, '01'));
note('1', get(o10, 1)); note('"5"', get(o10, '5')); note('5', get(o10, 5));
// 11. A chain that changes between two reads.
var p11 = {}; var o11 = Object.create(p11); o11[1000000] = 1;
note('before', get(o11, 1500000)); p11[1500000] = 'late'; note('added', get(o11, 1500000));
Object.setPrototypeOf(o11, { 1500000: 'swapped' }); note('swapped', get(o11, 1500000));
Object.setPrototypeOf(o11, null); note('null-proto', get(o11, 1500000));
// 12. The receiver's own accessor at a far index.
var o12 = {}; o12[1000000] = 1;
Object.defineProperty(o12, 1500000, { get: function () { return 'own-get'; } });
note('own-getter', get(o12, 1500000));
// 13. Object.prototype's index.
Object.prototype[1700000] = 'objproto'; note('object-proto', get(far, 1700000)); delete Object.prototype[1700000];
out.join(' ');"#;

const EXPECTED: &str = "far-absent=undefined far-present=2 proto-data=pd proto-getter=recv proto-words=20 proto-words-absent=undefined proxy=proxy:7:true typed=6 typed-past=undefined string=b string-past=undefined arguments=y 2^32-1=max 2^32=big -0=zero 1.5=frac \"01\"=lead 1=undefined \"5\"=five 5=five before=undefined added=late swapped=swapped null-proto=undefined own-getter=own-get object-proto=objproto";

fn run(tier: Tier, rounds: usize) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    let mut last = String::new();
    for _ in 0..rounds {
        last = match engine.eval(CASES, false).expect("the fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        };
    }
    last
}

#[test]
fn an_absent_index_reads_as_the_general_get_at_every_tier() {
    // Once, and three times in one engine: the second and third rounds read through sites the
    // first warmed (the template's caches, the records).
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        assert_eq!(run(tier, 1), EXPECTED, "{tier:?}");
        assert_eq!(run(tier, 3), EXPECTED, "{tier:?}, warmed");
    }
}

#[test]
fn an_absent_index_reads_undefined_in_a_loop_over_far_keys() {
    // The read the walk is for: a plain object's far index keys (its named entries), misses and
    // hits in one loop, at the tiers that run the helper.
    let program = r#"
        var m = {};
        for (var i = 0; i < 300; i++) m[((i * 7 + 1) << 16) | i] = i;
        function rd(o, k) { return o[k]; }
        var hits = 0, misses = 0;
        for (var r = 0; r < 3000; r++) {
            var j = r % 300;
            var x = rd(m, ((j * 7 + 1) << 16) | (r & 1 ? j : j + 1));
            if (x === undefined) misses++; else if (x === j) hits++;
        }
        hits + ' ' + misses
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(program, false).expect("parses") {
            Completion::Value(value) => assert_eq!(value, "1500 1500", "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}
