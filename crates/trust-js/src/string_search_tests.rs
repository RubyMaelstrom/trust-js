//! String.prototype.indexOf, lastIndexOf and includes, and replace and replaceAll with a String
//! pattern (ECMA-262 §22.1.3.9, §22.1.3.11, §22.1.3.8, §22.1.3.19-20): StringIndexOf and
//! StringLastIndexOf (§6.1.4.1-2) over code units at either width, short texts scanned directly
//! and long ones by the Two-Way searcher, with the arguments converted in their order.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("string search fixture parses")
        {
            Completion::Value(ref value) if value == "ok" => {}
            Completion::Value(value) => panic!("{tier:?}: {value}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

const ASSERT: &str = "function assert(value, message) { if (!value) throw Error(message); }\n";

#[test]
fn string_searches_find_code_units_at_either_width() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        var hay = 'the quick brown fox jumps over the lazy dog';
        for (var k = 0; k < 20; k++) {
            assert(hay.indexOf('lazy') === 35 && hay.includes('lazy') && hay.lastIndexOf('the') === 31, 'basic');
            assert(hay.indexOf('the', 1) === 31 && hay.indexOf('t') === 0 && hay.indexOf('g') === 42, 'single');
            assert(hay.indexOf('dog!') === -1 && !hay.includes('cat') && hay.lastIndexOf('xyz') === -1, 'absent');
            assert(hay.indexOf('') === 0 && hay.indexOf('', 100) === 43 && hay.lastIndexOf('') === 43, 'empty');
            assert(hay.indexOf('the', -5) === 0 && hay.indexOf('dog', 41) === -1 && hay.includes('dog', 40), 'from');
            assert(hay.lastIndexOf('the', 30) === 0 && hay.lastIndexOf('o', -1) === -1, 'lastIndexOf from');
            assert(hay.lastIndexOf('the', NaN) === 31 && hay.lastIndexOf('t', 0) === 0, 'NaN position');
        }
        // A two-byte haystack, needle, or both; a two-byte view of ASCII units.
        var wide = 'ß' + hay;
        assert(wide.indexOf('lazy') === 36 && wide.indexOf('ß') === 0 && wide.lastIndexOf('the') === 32, 'wide');
        assert(wide.includes('fox') && !wide.includes('é') && hay.indexOf('ß') === -1, 'widths');
        var asciiUnits = ('é' + 'ab'.repeat(200)).slice(1);
        assert(('x' + 'ab'.repeat(200)).indexOf(asciiUnits) === 1, 'two-byte needle of ASCII units');
        assert(asciiUnits.indexOf('ba') === 1 && asciiUnits.lastIndexOf('ab') === 398, 'two-byte ASCII haystack');
        // Surrogates are units: half a pair matches.
        var pair = 'a😀b';
        assert(pair.indexOf('\ude00') === 2 && pair.lastIndexOf('\ud83d') === 1 && pair.includes('😀'), 'pairs');
        // Long texts and needles take the Two-Way searcher.
        var long = 'ab'.repeat(500) + 'needle' + 'ab'.repeat(10);
        assert(long.indexOf('needle') === 1000 && long.includes('eedl') && long.lastIndexOf('ab') === 1024, 'long');
        assert(long.indexOf('abab', 999) === 1006 && long.lastIndexOf('abab', 1005) === 996, 'long, from');
        var longNeedle = 'ab'.repeat(30) + 'needle';
        assert(long.indexOf(longNeedle) === 940 && long.lastIndexOf(longNeedle) === 940, 'long needle');
        assert(long.indexOf(longNeedle + 'x') === -1 && long.indexOf('b' + longNeedle) === 939, 'long misses');
        // this and the arguments convert in order; a non-String needle converts.
        var log = [];
        var needle = { toString() { log.push('needle'); return 'b'; } };
        var position = { valueOf() { log.push('position'); return 0; } };
        assert('abc'.indexOf(needle, position) === 1 && log.join() === 'needle,position', log.join());
        log = [];
        assert('abc'.lastIndexOf('abcd', position) === -1 && log.join() === 'position', 'ToNumber first');
        assert(String.prototype.indexOf.call(12345, 34) === 2 && 'a1b'.includes(1), 'conversions');
        try { 'abc'.includes(/b/); assert(false, 'no TypeError'); } catch (e) { assert(e instanceof TypeError, 'RegExp'); }
        'ok'
    "#
    ));
}

#[test]
fn string_replacements_search_patterns_directly() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        assert('a-b-c'.replace('-', '+') === 'a+b-c' && 'a-b-c'.replaceAll('-', '+') === 'a+b+c', 'single');
        assert('aaa'.replaceAll('aa', 'b') === 'ba' && 'x-y'.replace('', '_') === '_x-y', 'overlap, empty');
        assert('abcabc'.replace('bc', '[$&]') === 'a[bc]abc' && 'abcabc'.replaceAll('bc', '$$') === 'a$a$', 'templates');
        assert('abc'.replace('d', 'x') === 'abc' && 'abc'.replaceAll('abcd', 'x') === 'abc', 'absent');
        var long = 'x'.repeat(300) + 'yz' + 'x'.repeat(300) + 'yz';
        assert(long.replace('yz', 'Q').indexOf('Q') === 300 && long.replaceAll('yz', 'Q').length === 602, 'long');
        assert(long.replaceAll('xyz', '-').split('-').length === 3, 'long pattern');
        'ok'
    "#
    ));
}
