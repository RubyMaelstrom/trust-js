//! RegExp matching and the RegExp-based String methods on every tier. Expected values are
//! node's (V8) results for the same scripts.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("regexp fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

/// ECMA-262 bounds neither the time nor the depth of a match (§22.2.2): long subjects that
/// keep a choice point per iteration match like any other.
#[test]
fn long_subjects_match_without_a_depth_limit() {
    check(
        r#"
        function run() {
            var quoted = '"' + 'ab\\"c'.repeat(300) + '"';
            var string = /"(?:[^"\\]|\\.)*"/.exec(quoted)[0].length;
            var star = /(a|b)*/.exec('ab'.repeat(500));
            var lazy = /(a|ab)*?c/.exec('ab'.repeat(5000) + 'c');
            return [string, star[0].length, star[1], lazy[0].length, lazy[1]].join();
        }
        run()
        "#,
        "1502,1000,b,10001,ab",
    );
}

/// Backtracking out of an alternative undoes the captures it made, including those of a
/// lookaround inside it (§22.2.2.4) and those cleared by a later quantifier iteration
/// (RepeatMatcher step 4).
#[test]
fn backtracking_restores_lookaround_and_iteration_captures() {
    check(
        r#"
        JSON.stringify([
            /(?:(?=(a))b|a)/.exec("a"),
            /(?:(?<=(a))c|a$)/.exec("aa"),
            /(?:(?=(\w+))\1x|\w)(.)/.exec("ab"),
            /^(?:(a)|b)*$/.exec("ab"),
        ])
        "#,
        r#"[["a",null],["a",null],["ab",null,"b"],["ab",null]]"#,
    );
}

/// An inline modifier group's flags hold inside it only (§22.2.2.3.1, ModifiersAtom), also
/// when backtracking re-enters or leaves the group and when a lookaround fails inside one.
#[test]
fn modifier_flags_hold_inside_their_group_across_backtracking() {
    check(
        r#"
        JSON.stringify([
            /(?i:a+)A/.exec("aaA"),
            /(?i:a+)A/.exec("aaa"),
            /a(?i:b(?-i:c))d/.exec("aBcd"),
            /a(?i:b(?-i:c))d/.exec("aBCd"),
            /(?:x(?i:y))*xY/.exec("xYxyxy"),
            /(?!(?i:Ax))A/.exec("a"),
            /(?=(?i:A))a/.exec("a"),
            /(?i:(?<=A)b)/.exec("ab"),
        ])
        "#,
        r#"[["aaA"],null,["aBcd"],null,["xY"],null,["a"],["b"]]"#,
    );
}

/// Each match attempt starts with every capture unset (§22.2.2.1: the Matcher runs from a fresh
/// State at each index RegExpBuiltinExec tries): what a failed attempt captured, also through a
/// lookaround or where no choice point remained, is gone by the next one, in programs with more
/// than 64 capture slots too.
#[test]
fn attempts_start_with_captures_unset() {
    check(
        r#"
        var re = /(a)b|(c)/y;
        re.lastIndex = 1;
        var wide = new RegExp("(a)".repeat(33) + "x|(b)").exec("ab");
        var wideLook = new RegExp("(?:" + "(a)".repeat(33) + ")?(?=(a))ab|b").exec("acb");
        JSON.stringify([
            /(a)b|(c)/.exec("acx"),
            /(a)?b|c/.exec("ac"),
            /(?=(a))ab|b/.exec("acb"),
            /(?:(a)b)*c/.exec("ab".repeat(50000) + "c").slice(1),
            /(?:(a)|b)*?c/.exec("bac"),
            /(?:(a)|(b))+?$/.exec("abab"),
            re.exec("acx"),
            re.lastIndex,
            /(a)|b/g.exec("b"),
            "xaxb".replace(/(a)?b|x/g, "[$1]"),
            [wide.index, wide[1], wide[34]],
            [wideLook.index, wideLook[1], wideLook[34]],
        ])
        "#,
        r#"[["c",null,"c"],["c",null],["b",null],["a"],["bac","a"],["abab",null,"b"],["c",null,"c"],2,["b",null],"[]a[][]",[1,null,"b"],[2,null,null]]"#,
    );
}

/// A sticky RegExp tries `lastIndex` alone (§22.2.7.2 step 13.a): where no match can begin,
/// by its first element, a line start or a lookbehind, it fails there and resets `lastIndex`,
/// and where one can, including an empty one or one through `^`, it matches.
#[test]
fn sticky_regexps_try_only_lastindex() {
    check(
        r#"
        function at(re, s, i) { re.lastIndex = i; var m = re.exec(s); return [m, re.lastIndex]; }
        JSON.stringify([
            at(/(?<=a)b/y, "ab", 1),
            at(/(?<=a)b/y, "bb", 1),
            at(/^b|c/my, "a\nb", 2),
            at(/^b|c/my, "a\nb", 1),
            at(/x*/y, "ab", 1),
            at(/[ \t]/y, "a b", 1),
            at(/[ \t]/y, "a b", 0),
            at(/(?:)/y, "ab", 2),
            at(/\u{1F600}/uy, "a\u{1F600}", 2),
            at(/./uy, "a\u{1F600}", 2),
            at(/[^a]/uy, "a\u{1F600}", 2),
            at(/b/y, "abc", 3),
            at(/^/my, "a\n", 2),
        ])
        "#,
        r#"[[["b"],2],[null,0],[["b"],3],[null,0],[[""],1],[[" "],2],[null,0],[[""],2],[["😀"],3],[["😀"],3],[["😀"],3],[null,0],[[""],2]]"#,
    );
}

/// A pattern that matches one element (a character, `.` or a class) matches as the matcher's
/// one instruction would, at every start the scan proposes: case folding with and without `u`,
/// line terminators and `s`, surrogate pairs as one element or two, sticky and global starts, and
/// through `test`, `replace`, `split` and `search`, on one- and two-byte subjects.
#[test]
fn single_element_patterns_match_as_the_matcher_does() {
    check(
        r#"
        function all(re, s) {
            var out = [];
            re.lastIndex = 0;
            for (var m; (m = re.exec(s)); ) { out.push(m.index + ":" + m[0]); if (m[0] === "") re.lastIndex++; }
            return out.join(",");
        }
        function at(re, s, i) {
            re.lastIndex = i;
            var m = re.exec(s);
            return m ? [m.index, m[0], re.lastIndex] : [null, re.lastIndex];
        }
        var two = "a’<b>&c\n\r d\u{1F600}e";
        JSON.stringify([
            all(/[&<>]/g, two),
            all(/[&<>]/g, "a<b>&c"),
            /[&<>]/.test(two), /[&<>]/.test("abc’"),
            all(/./g, "\n\r a b"),
            all(/./gs, "\n\r"),
            all(/./gu, "\u{1F600}x\uD800y"),
            all(/./g, "\u{1F600}"),
            all(/[^a]/gu, "a\u{1F600}a"),
            all(/x/gi, "aXbx’X"),
            all(/ſ/giu, "sSſ"),
            all(/ſ/gi, "sSſ"),
            all(/k/giu, "KKk"),
            all(/\S/g, " \t ﻿a"),
            at(/[ \t]/y, "a b", 1), at(/[ \t]/y, "a b", 0), at(/[ \t]/dy, "a b", 1),
            at(/\u{1F600}/uy, "a\u{1F600}", 2), at(/./uy, "a\u{1F600}", 2),
            at(/[\r\n"'()*+.:<=[\\\]^_`${}~-]/g, "abc ’ def*ghi", 2),
            /[\r\n"'()*+.:<=[\\\]^_`${}~-]/d.exec("ab*").indices[0],
            "a<b>c".replace(/[<>]/g, "-"),
            "a<b>c".split(/[<>]/),
            "a’<b".search(/[<>]/),
            "x".repeat(70000).search(/y/),
            ("x".repeat(70000) + "’y").search(/y/),
        ])
        "#,
        r#"["2:<,4:>,5:&","1:<,3:>,4:&",true,false,"3:a,5:b","0:\n,1:\r","0:😀,2:x,3:\ud800,4:y","0:\ud83d,1:\ude00","1:😀","1:X,3:x,5:X","0:s,1:S,2:ſ","2:ſ","0:K,1:K,2:k","4:a",[1," ",2],[null,0],[1," ",2],[1,"😀",3],[1,"😀",3],[9,"*",10],[2,3],"a-b-c",["a","b","c"],2,-1,70001]"#,
    );
}

/// RegExp.prototype[@@match] with `g` (§22.2.6.8) and [@@search] (§22.2.6.12): the matches,
/// empty ones advancing by code point in Unicode mode, `lastIndex` as each step leaves it (also
/// a -0 that @@search restores), the legacy statics of the last match, and the TypeError of a
/// non-writable `lastIndex`.
#[test]
fn global_match_and_search_keep_their_observable_steps() {
    check(
        r#"
        function li(re, f) { var r = f(); return [r, re.lastIndex, Object.is(re.lastIndex, -0)]; }
        var g = /a(b)?/g; g.lastIndex = 3;
        var e = /x*/g;
        var u = /(?:)/gu;
        var nu = /(?:)/g;
        var gy = /a/gy;
        var s1 = /b/g; s1.lastIndex = 7;
        var s2 = /b/y; s2.lastIndex = -0;
        var s3 = /b/; s3.lastIndex = 2;
        var s4 = /c/gy; s4.lastIndex = 1;
        var frozen = /a/g; Object.defineProperty(frozen, "lastIndex", { writable: false, value: 0 });
        var r1 = "xabyab".match(/a(b)/g);
        var statics1 = [RegExp.lastMatch, RegExp.$1, RegExp.leftContext, RegExp.rightContext];
        var r2 = "q1w2".search(/(\d)/);
        var statics2 = [RegExp.lastMatch, RegExp.$1, RegExp.leftContext, RegExp.rightContext];
        var frozenErr;
        try { "aaa".match(frozen); } catch (err) { frozenErr = err.name; }
        JSON.stringify([
            li(g, () => "xababa".match(g)),
            li(e, () => "ab".match(e)),
            li(u, () => "\u{1F600}x".match(u)),
            li(nu, () => "\u{1F600}x".match(nu)),
            li(gy, () => "aab".match(gy)),
            "abc".match(/z/g),
            li(s1, () => "abcb".search(s1)),
            li(s2, () => "abcb".search(s2)),
            li(s2, () => "bbcb".search(s2)),
            li(s3, () => "abcb".search(s3)),
            li(s4, () => "abc".search(s4)),
            "abc".search(/z/),
            r1, statics1, r2, statics2, frozenErr,
            "a’b’".match(/’/g),
            "AbAB".match(/b/gi),
        ])
        "#,
        r#"[[["ab","ab","a"],0,false],[["","",""],0,false],[["","",""],0,false],[["","","",""],0,false],[["a","a"],0,false],null,[1,7,false],[-1,0,true],[0,0,true],[1,2,false],[-1,1,false],-1,["ab","ab"],["ab","b","xaby",""],1,["1","1","q","w2"],"TypeError",["’","’"],["b","B"]]"#,
    );
}

/// Counted repetitions too large to unroll run as loops over one copy of their body, keeping
/// RepeatMatcher's semantics: per-iteration capture resets, the empty-iteration rule beyond
/// `min`, and greedy versus lazy order (§22.2.2.3.1).
#[test]
fn large_counted_repetitions_match_like_unrolled_ones() {
    check(
        r#"
        function show(match) {
            return match ? match[0].length + ':' + match.index + ':' + match.slice(1).join('/') : 'null';
        }
        [
            show(/(?:(?:ab){1000}){1000}/.exec('ab'.repeat(1000))),
            show(/(?:(?:ab){10}){100}/.exec('x' + 'ab'.repeat(1000))),
            show(/(?:(a)|(b)){1500}/.exec('ab'.repeat(800))),
            show(/^(?:(a)|(b)){2,1500}?$/.exec('ab'.repeat(700))),
            show(/(?:a?){1001}b/.exec('aab')),
            show(/(?:(a)|b?){1001,1002}c/.exec('abc')),
            show(/(?:x(?:ab){1001}y){3}/.exec(('x' + 'ab'.repeat(1001) + 'y').repeat(3))),
            show(new RegExp('(?:'.repeat(10) + 'a|b' + '){2}'.repeat(10)).exec('ab'.repeat(600))),
        ].join()
        "#,
        "null,2000:1:,1500:0:/b,1400:0:/b,3:0:,3:0:,6012:0:,1024:0:",
    );
}

/// `split`'s limit is ToUint32 (§7.1.7), which wraps modulo 2^32 for every Number rather than
/// saturating, for string and RegExp separators alike (§22.1.3.23, §22.2.6.14 step 13).
/// RegExp.prototype[@@split] and [@@matchAll] default SpeciesConstructor to the realm's %RegExp%
/// intrinsic and never read the global `RegExp` binding (§22.2.6.14 step 4, §22.2.6.9 step 4).
/// String.prototype.split consults a separator Object's @@split only: a RegExp whose @@split
/// was removed is converted with ToString, which reads its `source` and `flags`, and split on
/// as a String (ECMA-262 #sec-string.prototype.split steps 3 and 5).
#[test]
fn split_converts_a_regexp_without_its_split_method_to_a_string() {
    check(
        r#"
        delete RegExp.prototype[Symbol.split];
        var plain = 'a/b/c'.split(/\//);
        var re = /b/;
        var reads = [];
        Object.defineProperty(re, 'source', { get() { reads.push('source'); return 'x'; } });
        Object.defineProperty(re, 'flags', { get() { reads.push('flags'); return 'g'; } });
        var converted = 'a/x/gb/x/gc'.split(re);
        JSON.stringify([plain, converted, reads])
        "#,
        r#"[["a/b/c"],["a","b","c"],["source","flags"]]"#,
    );
}

#[test]
fn split_limits_wrap_and_species_defaults_to_the_intrinsic() {
    check(
        r#"
        function run() {
            var lengths = [
                "a,b,c".split(",", -1e20).length,
                "a,b,c".split(",", 2 ** 64).length,
                "a,b,c".split(",", 2 ** 32 + 1).length,
                "a,b,c".split(",", -1).length,
                "a,b,c".split(",", -4294967294).length,
                "a,b,c".split(/,/, -1e20).length,
                "a,b,c".split(/,/, 2 ** 64).length,
                "a,b,c".split(/,/, 2 ** 32 + 2).length,
                "a,b,c".split(/,/, -4294967294).length,
                "a,b,c".split(/,/, 1e300).length,
            ];
            var intrinsic = RegExp, constructed = [];
            globalThis.RegExp = function () {
                constructed.push(arguments.length);
                return /x/y;
            };
            var splitter = /\d/;
            splitter.constructor = undefined;
            var parts = "a1b2c".split(splitter).join("|");
            var matcher = /\d/g;
            matcher.constructor = undefined;
            var all = Array.from("a1b2c".matchAll(matcher), function (m) { return m[0]; }).join("|");
            var reads = 0;
            Object.defineProperty(globalThis, "RegExp", {
                get: function () { reads++; return intrinsic; },
                configurable: true,
            });
            "x,y".split(/,/);
            Array.from("xy".matchAll(/./g));
            Object.defineProperty(globalThis, "RegExp", { value: intrinsic, writable: true, configurable: true });
            return [lengths.join(" "), parts, all, constructed.length, reads].join(",");
        }
        run()
        "#,
        "3 0 1 3 2 3 0 2 2 0,a|b|c,1|2,0,0",
    );
}

/// RegExp.prototype.test, [@@split] and [@@replace] take direct paths for ordinary RegExps (no
/// match arrays, no splitter object); their results, `lastIndex` writes, legacy statics,
/// captures, limits, templates and replacer calls must equal the generic algorithms'
/// (§22.2.6.16, §22.2.6.14, §22.2.6.11).
#[test]
fn live_split_replace_and_test_match_the_generic_algorithms() {
    check(
        r###"
        (function () {
        function show(v) {
            return JSON.stringify(v, function (k, x) { return x === undefined ? '<u>' : x; });
        }
        function run() {
            var r = [];
            // test
            var g = /a/g;
            r.push([g.test('aa'), g.lastIndex, g.test('aa'), g.lastIndex, g.test('aa'), g.lastIndex]);
            var y = /a/y;
            y.lastIndex = 1;
            r.push([y.test('ba'), y.lastIndex, y.test('ba'), y.lastIndex]);
            var far = /a/g;
            far.lastIndex = 9;
            r.push([far.test('aa'), far.lastIndex]);
            var plain = /b/;
            plain.lastIndex = 7;
            r.push([plain.test('ab'), plain.lastIndex]);
            /(\d)(\d)?/.test('x1y');
            r.push([RegExp.$1, RegExp.$2, RegExp.lastMatch, RegExp['$`'], RegExp["$'"]]);
            var own = /a/;
            own.exec = function () { return null; };
            r.push(own.test('a'));
            var reads = 0, li = /a/g;
            li.lastIndex = { valueOf: function () { reads++; return 0; } };
            r.push([li.test('a'), reads, li.lastIndex]);
            var frozen = /a/g;
            Object.defineProperty(frozen, 'lastIndex', { writable: false, value: 0 });
            try { frozen.test('a'); r.push('no throw'); } catch (e) { r.push(e.name); }
            // split
            r.push('btn btn-primary active large'.split(/\s+/));
            r.push('2024-01-02'.split(/(-)/));
            r.push('a,b,c'.split(/,/, 2), 'a1b2c'.split(/(\d)/, 2), 'a1b2c'.split(/(\d)/, 3));
            r.push('abc'.split(/(?:)/), 'abc'.split(/x*/), ',a,'.split(/,/), ''.split(/x/), ''.split(/(?:)/));
            r.push('😀a😀'.split(/(?:)/u).length, '😀a😀'.split(/(?:)/).length);
            r.push('a,b'.split(/,/y), 'ab'.split(/(x)?b/), 'a1b2'.split(/(?<=\d)/));
            var sx = /,/g;
            sx.lastIndex = 5;
            'a,b,c'.split(sx);
            r.push([sx.lastIndex, RegExp.lastMatch, RegExp['$`'], RegExp["$'"]]);
            r.push('aXbxC'.split(/x/i), 'a\nb'.split(/^/m), 'one  two'.split(/(\s)+/));
            // replace
            function escapeHtml(s) {
                return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            }
            r.push(escapeHtml('<a href="x">Tom & Jerry\'s</a>'));
            r.push('john smith'.replace(/(\w+)\s(\w+)/, '$2, $1 [$&] {$`|$\'} $$ $3 $10'));
            r.push('2024-01-02'.replace(/(?<y>\d+)-(?<m>\d+)/, '$<m>/$<y> $<z>.'));
            r.push('a-b'.replace(/(-)/, '$<x>'));
            r.push('abc'.replace(/x*/g, '-'), '😀😀'.replace(/(?:)/gu, '-').length,
                '😀😀'.replace(/(?:)/g, '-').length);
            var ys = /a/y;
            ys.lastIndex = 1;
            r.push(['aab'.replace(ys, 'X'), ys.lastIndex, 'aab'.replace(ys, 'X'), ys.lastIndex]);
            var ng = /a/;
            ng.lastIndex = 3;
            r.push(['aa'.replace(ng, 'b'), ng.lastIndex]);
            var gl = /a/g;
            gl.lastIndex = 1;
            r.push(['aXa'.replace(gl, 'b'), gl.lastIndex, RegExp.lastMatch, RegExp['$`']]);
            var calls = [];
            var fr = /(\w)(\d)?/g;
            r.push('a1b'.replace(fr, function (m, c1, c2, pos, str) {
                calls.push([m, c1, c2, pos, str, arguments.length, fr.lastIndex]);
                fr.lastIndex = 0;
                return '<' + m + '>';
            }), calls, fr.lastIndex);
            var named = [];
            r.push('x=1,y=2'.replace(/(?<k>\w)=(?<v>\d)/g, function () {
                var groups = arguments[arguments.length - 1];
                named.push([Object.getPrototypeOf(groups), groups.k, groups.v, arguments.length]);
                return groups.v + groups.k;
            }), named);
            try { 'aa'.replace(/a/g, function () { throw new Error('stop'); }); } catch (e) { r.push(e.message); }
            var cnt = /a/g;
            r.push('aaa'.replace(cnt, function (m, p) { return p; }), 'aaa'.replace(cnt, '$&$&'));
            var ownExec = /a/g;
            ownExec.exec = function () { return null; };
            r.push('aa'.replace(ownExec, 'b'));
            r.push('aaa'.replace(/a/gy, 'b'), 'baaa'.replace(/a/gy, 'b'));
            return r.map(show).join('\n');
        }
        return run();
        })()
        "###,
        r###"[true,1,true,2,false,0]
[true,2,false,0]
[false,0]
[true,7]
["1","","1","x","y"]
false
[true,1,1]
"TypeError"
["btn","btn-primary","active","large"]
["2024","-","01","-","02"]
["a","b"]
["a","1"]
["a","1","b"]
["a","b","c"]
["a","b","c"]
["","a",""]
[""]
[]
3
5
["a","b"]
["a","<u>",""]
["a1","b2"]
[5,",","a,b","c"]
["a","b","C"]
["a\n","b"]
["one"," ","two"]
"&lt;a href=&quot;x&quot;&gt;Tom &amp; Jerry's&lt;/a&gt;"
"smith, john [john smith] {|} $ $3 john0"
"01/2024 .-02"
"a$<x>b"
"-a-b-c-"
7
9
["aXb",2,"aab",0]
["ba",3]
["bXb",0,"a","aX"]
"<a1><b>"
[["a1","a","1",0,"a1b",5,0],["b","b","<u>",2,"a1b",5,0]]
0
"1x,2y"
[[null,"x","1",6],[null,"y","2",6]]
"stop"
"012"
"aaaaaa"
"aa"
"bbb"
"baaa""###,
    );
}

/// Any observable step of those algorithms (an accessor @@match read by the splitter's
/// IsRegExp, a species constructor or getter, a replaced or own `exec`) keeps the generic path
/// and runs in the specified order.
#[test]
fn regexp_fast_paths_yield_to_observable_lookups() {
    check(
        r###"
        (function () {
        function run() {
            var log = [];
            var match = Object.getOwnPropertyDescriptor(RegExp.prototype, Symbol.match);
            Object.defineProperty(RegExp.prototype, Symbol.match, {
                get: function () { log.push('match'); return match.value; },
                configurable: true,
            });
            var a = 'a,b'.split(/,/).join('|');
            Object.defineProperty(RegExp.prototype, Symbol.match, match);
            class R extends RegExp {
                constructor(pattern, flags) {
                    log.push('R:' + flags);
                    super(pattern, flags);
                }
            }
            var b = 'a,b'.split(new R(',')).join('|');
            var species = Object.getOwnPropertyDescriptor(RegExp, Symbol.species);
            Object.defineProperty(RegExp, Symbol.species, {
                get: function () { log.push('species'); return RegExp; },
                configurable: true,
            });
            var c = 'a,b'.split(/,/).join('|');
            Object.defineProperty(RegExp, Symbol.species, species);
            var exec = RegExp.prototype.exec;
            RegExp.prototype.exec = function (s) {
                log.push('exec:' + this.lastIndex);
                return exec.call(this, s);
            };
            var d = 'a,b'.split(/,/).join('|');
            var e = 'xa'.replace(/a/g, 'b');
            var f = /a/.test('a');
            RegExp.prototype.exec = exec;
            var own = /a/g;
            own.exec = function (s) { log.push('own'); return exec.call(this, s); };
            var g = 'aa'.replace(own, 'b') + own.test('a');
            return [a, b, c, d, e, f, g, log.join(' ')].join(';');
        }
        return run();
        })()
        "###,
        r###"a|b;a|b;a|b;a|b;xb;true;bbtrue;match R:undefined R:y species exec:0 exec:1 exec:2 exec:0 exec:2 exec:0 own own own own"###,
    );
}

/// Unicode-mode matching reads two-byte subjects in place, decoding surrogate pairs as it steps
/// in either direction, as Node does: a start inside a pair begins at the pair, a quantifier's
/// run and a lookbehind count code points, a backreference never ends inside a pair (under `i`
/// it compares code points), lone surrogates are their own elements, string sets and empty
/// matches step over pairs, and the prefilter's scans propose only element boundaries.
#[test]
fn unicode_matching_steps_over_surrogate_pairs_in_place() {
    check(
        r###"
        (function () {
        function show(v) {
            if (Array.isArray(v)) return '[' + v.map(show).join(',') + ']';
            var out = '';
            for (var k = 0; k < v.length; k++) {
                var c = v.charCodeAt(k);
                out += c < 0x80 && c >= 0x20 ? v[k] : '\\u' + c.toString(16);
            }
            return out;
        }
        function ex(m) { return m ? m.index + ':' + show(Array.prototype.slice.call(m)) : 'null'; }
        var r = [];
        var s = 'a\ud83d\ude00b\ud800c\udc00d';
        r.push(show(s.match(/./gu)), show(s.match(/./g)));
        for (var li = 0; li <= 3; li++) {
            var y = /./uy; y.lastIndex = li;
            r.push(li + ex(y.exec('a\ud83d\ude00b')) + '/' + y.lastIndex);
        }
        r.push(ex(/(\ud83d)\1/u.exec('\ud83d\ud83d\ude00')), ex(/(\ud83d)\1/.exec('\ud83d\ud83d\ude00')));
        r.push(ex(/(\u{10400})\1/ui.exec('x\u{10400}\u{10428}')), ex(/(?<=\1(.))x/ui.exec('\u{10428}\u{10400}x')));
        r.push(ex(/(?<=\ude00)b/u.exec('a\ud83d\ude00b')), ex(/(?<=\ude00)b/.exec('a\ud83d\ude00b')));
        r.push(ex(/(?<=^.{2})x/u.exec('\ud83d\ude00\ud83d\ude00x')), ex(/(?<=^.{2})x/u.exec('\ud83d\ude00x')));
        r.push(ex(/^(.+)\ud83d\ude00(.*)$/u.exec('x\ud83d\ude00y\ud83d\ude00z')), ex(/a.{1,2}?b/u.exec('a\ud83d\ude00\ud83d\ude00b')));
        r.push(ex(/\ud83d+/u.exec('\ud83d\ud83d\ud83d\ude00')), ex(/[^a]+/u.exec('a\ud800\ud83d\ude00\udc00a')));
        r.push(show('\ud83d\udc4d\ud83c\udffd\ud83c\uddeb\ud83c\uddf7x'.match(/\p{RGI_Emoji}/gv)));
        r.push(show('a\ud83d\ude00b'.replace(/(?:)/gu, '-')), show('a\ud83d\ude00b'.split(/(?:)/u)));
        var long = 'b'.repeat(40) + '\ud83d\ude00'.repeat(20) + 'x\ud83d\ude00\ud83d\ude00y' + '\ud83d\ude00'.repeat(20);
        r.push(ex(/(.)(?:\ud83d\ude00){2}y/u.exec(long)), ex(/\ude00y/u.exec(long)), ex(/\ude00y/.exec(long)));
        return r.join(';');
        })()
        "###,
        r###"[a,\ud83d\ude00,b,\ud800,c,\udc00,d];[a,\ud83d,\ude00,b,\ud800,c,\udc00,d];00:[a]/1;11:[\ud83d\ude00]/3;21:[\ud83d\ude00]/3;33:[b]/4;null;0:[\ud83d\ud83d,\ud83d];1:[\ud801\udc00\ud801\udc28,\ud801\udc00];4:[x,\ud801\udc00];null;3:[b];4:[x];null;0:[x\ud83d\ude00y\ud83d\ude00z,x\ud83d\ude00y,z];0:[a\ud83d\ude00\ud83d\ude00b];0:[\ud83d\ud83d];1:[\ud800\ud83d\ude00\udc00];[\ud83d\udc4d\ud83c\udffd,\ud83c\uddeb\ud83c\uddf7];-a-\ud83d\ude00-b-;[a,\ud83d\ude00,b];80:[x\ud83d\ude00\ud83d\ude00y,x];null;84:[\ude00y]"###,
    );
}

/// GetSubstitution and a match's strings read code units: templates and subjects of either
/// width, astral characters and lone surrogates expand as Node expands them, the generic path
/// expands every result (a skipped one's named-group getter included), and captures, contexts
/// and split pieces come straight from two-byte subjects.
#[test]
fn substitutions_and_captures_read_code_units() {
    check(
        r###"
        (function () {
        var r = [];
        function codes(s) { return Array.prototype.map.call(s, function (c) { return c.charCodeAt(0).toString(16); }).join('.'); }
        r.push('x\u00e9y\u00e9z'.replace(/\u00e9/g, '[$`|$&|$\']'));
        r.push(codes('a\ud83d\ude00b'.replace(/(\ud83d\ude00)/u, '\u00e9$1\ud800$$')));
        r.push('ab-cd'.replace(/(?<x>\w)(\w)/g, '$<x>$2$01$10$<y>$<'));
        r.push('ab'.replace(/(a)/, '$0$3$9$'));
        var custom = /./g, calls = 0;
        custom.exec = function () {
            if (++calls > 3) return null;
            var m = ['\u00e9' + calls];
            m.index = [4, 1, 6][calls - 1];
            m.groups = { get g() { r.push('get' + calls); return 'G'; } };
            return m;
        };
        r.push('abcdefghij'.replace(custom, '[$&|$`|$\'|$<g>]'));
        var m = /(\u00e9+)(X)(y*)/.exec('a' + '\u00e9'.repeat(3) + 'Xyy');
        r.push(m.index + ':' + m.join(',') + ':' + RegExp.$1 + RegExp['$`'] + RegExp["$'"] + RegExp.lastParen);
        r.push('\u00e91\u00e922'.split(/(\d)/).join('|'));
        return r.join(';');
        })()
        "###,
        "x[x|é|yéz]y[xéy|é|z]z;61.e9.d83d.de00.d800.24.62;abaa0$<-cdcc0$<;$0$3$9$b;get4;get4;get4;\
         abcd[é1|abcd|ghij|G][é3|abcdef|ij|G]ij;1:éééXyy,ééé,X,yy:éééayy;é|1|é|2||2|",
    );
}

/// A RegExp of the shape RegExpAlloc gave it reads and writes `lastIndex` at its slot when it
/// holds a plain Number; any other value, a RegExp with another own property and a
/// non-writable `lastIndex` keep ToLength and [[Set]]. An object of that shape that is no RegExp
/// still reaches the intrinsic symbol methods, which find no matcher, and the legacy statics
/// follow every path's matches.
#[test]
fn last_index_at_its_slot_keeps_the_observable_steps() {
    check(
        r###"
        (function () {
        function run() {
            var r = [];
            function attempt(f) { try { return f(); } catch (e) { return e.name; } }
            var fake = Object.create(RegExp.prototype);
            fake.lastIndex = 0;
            r.push(attempt(function () { return 'x'.replace(fake, 'y'); }),
                attempt(function () { return 'x'.match(fake); }),
                attempt(function () { return 'x'.split(fake); }),
                attempt(function () { return 'x'.search(fake); }),
                attempt(function () { return RegExp.prototype.exec.call(fake, 'x'); }));
            var extra = /a/g;
            extra.note = 1;
            r.push([extra.exec('aaa').index, extra.lastIndex, extra.test('aaa'), extra.lastIndex,
                'aaa'.replace(extra, 'b'), extra.lastIndex].join());
            var y = /a/y;
            var values = [1.5, -1, '1', 2 ** 53 + 2, { valueOf: function () { return 1; } }, NaN, Infinity];
            r.push(values.map(function (v) {
                y.lastIndex = v;
                var m = y.exec('aa');
                return (m ? m.index : 'null') + ':' + y.lastIndex;
            }).join(' '));
            var g = /a/g;
            r.push(values.map(function (v) {
                g.lastIndex = v;
                var hit = g.test('aa');
                return hit + ':' + g.lastIndex;
            }).join(' '));
            var ys = /b/y;
            ys.lastIndex = 1;
            r.push(['abab'.replace(ys, 'X'), ys.lastIndex, 'abab'.replace(ys, 'X'), ys.lastIndex].join());
            /(\d)(\d)/.exec('a12b');
            r.push([RegExp.$1, RegExp.$2, RegExp.lastMatch, RegExp['$`'], RegExp["$'"]].join());
            'x34y'.replace(/(\d)(\d)/, '');
            r.push([RegExp.$1, RegExp.$2, RegExp.lastMatch].join());
            'p5q'.match(/(\d)/);
            r.push([RegExp.$1, RegExp.lastMatch, RegExp.input].join());
            /(z)/.test('z');
            r.push(RegExp.$1);
            var frozen = /a/g;
            Object.defineProperty(frozen, 'lastIndex', { value: 0, writable: false });
            r.push(attempt(function () { return frozen.exec('a'); }), attempt(function () { return 'a'.replace(frozen, 'b'); }));
            return r.join(';');
        }
        return run();
        })()
        "###,
        r###"TypeError;TypeError;TypeError;TypeError;TypeError;0,1,true,2,bbb,0;1:2 0:1 1:2 null:0 1:2 0:1 null:0;true:2 true:1 true:2 false:0 true:2 true:1 false:0;aXab,2,abab,0;1,2,12,a,b;3,4,34;5,5,p5q;z;TypeError;TypeError"###,
    );
}

/// The RegExp protector proves %RegExp.prototype%'s methods and `constructor` by identity at
/// each use: assigning one changes no structure and bumps no epoch, yet String.prototype's
/// methods and RegExpExec observe it at once, and the intrinsic paths return once it is
/// restored. A non-global replace or match still reads `lastIndex` with ToLength when its
/// result is unused.
#[test]
fn regexp_protector_observes_assigned_members() {
    check(
        r###"
        (function () {
        function run() {
            var log = [];
            var r = [];
            r.push('a-b'.replace(/-/, '+'), 'a-b'.split(/-/).join('|'), 'xa'.search(/a/), 'aa'.match(/a/g).length);
            var proto = RegExp.prototype;
            var saved = {};
            var keys = ['exec', Symbol.match, Symbol.matchAll, Symbol.replace, Symbol.search, Symbol.split, 'constructor'];
            keys.forEach(function (k) { saved[typeof k === 'symbol' ? k.description.slice(7) : k] = proto[k]; });
            proto[Symbol.replace] = function (s, v) { log.push('replace:' + s + ':' + v); return 'R'; };
            r.push('a-b'.replace(/-/, '+'), 'a-b'.replaceAll(/-/g, '+'));
            proto[Symbol.replace] = saved.replace;
            proto[Symbol.match] = function (s) { log.push('match:' + s); return ['M']; };
            r.push('a-b'.match(/-/)[0]);
            proto[Symbol.match] = saved.match;
            proto[Symbol.split] = function (s, lim) { log.push('split:' + s + ':' + lim); return ['S']; };
            r.push('a-b'.split(/-/, 3)[0]);
            proto[Symbol.split] = saved.split;
            proto[Symbol.search] = function (s) { log.push('search:' + s); return 42; };
            r.push('a-b'.search(/-/));
            proto[Symbol.search] = saved.search;
            proto[Symbol.matchAll] = function (s) { log.push('matchAll:' + s); return [['A']].values(); };
            r.push([...'a-b'.matchAll(/-/g)][0][0]);
            proto[Symbol.matchAll] = saved.matchAll;
            proto.exec = function (s) { log.push('exec:' + this.source + ':' + s); return saved.exec.call(this, s); };
            r.push('a-b'.replace(/-/, '+'), /b/.test('ab'), 'a-b'.split(/-/).join('|'), 'xa'.search(/a/));
            proto.exec = saved.exec;
            proto.constructor = { [Symbol.species]: function (p, f) { log.push('species:' + f); return new RegExp(p, f); } };
            r.push('a-b'.split(/-/).join('|'));
            proto.constructor = saved.constructor;
            r.push('a-b'.replace(/-/, '+'), 'a-b'.split(/-/).join('|'), 'xa'.search(/a/), 'aa'.match(/a/g).length);
            var reads = 0, plain = /a/;
            plain.lastIndex = { valueOf: function () { reads++; return 0; } };
            'a'.replace(plain, 'b');
            'a'.match(plain);
            r.push(reads);
            try { 'a'.matchAll(/a/); r.push('no throw'); } catch (e) { r.push(e.name); }
            try { 'a'.replaceAll(/a/, 'b'); r.push('no throw'); } catch (e) { r.push(e.name); }
            r.push(log.join(' '));
            return r.join(';');
        }
        return run();
        })()
        "###,
        r###"a+b;a|b;1;2;R;R;M;S;42;A;a+b;true;a|b;1;a|b;a+b;a|b;1;2;2;TypeError;TypeError;replace:a-b:+ replace:a-b:+ match:a-b split:a-b:3 search:a-b matchAll:a-b exec:-:a-b exec:b:ab exec:-:a-b exec:-:a-b exec:-:a-b exec:a:xa species:y"###,
    );
}

/// A pattern whose alternatives begin with `^` or with an element (`(^|\s)name`) is only
/// attempted where one of them can begin, including the empty end of the input.
#[test]
fn caret_alternatives_only_start_where_caret_holds() {
    check(
        r###"
        (function () {
        function all(re, s) {
            var found = [], m;
            re.lastIndex = 0;
            while ((m = re.exec(s))) {
                found.push(m.index + ':' + m[0].length);
                if (m[0].length === 0) re.lastIndex++;
                if (found.length > 50) break;
            }
            return found.join(' ');
        }
        var cases = [
            [/(^|\s)active(\s|$)/g, 'active btn active-x  active'],
            [/(^|x)/g, ''],
            [/(^|x)/g, 'axbx'],
            [/(?:^|,)(\w*)/g, 'a,,b,'],
            [/(?:^|,)(\w*)/gm, 'a\n,b\nc'],
            [/^|a/gm, 'xa\nya\n'],
            [/(?:^|b)/gm, 'b\n\nb'],
            [/(?:^ +|\s)x/g, '  x x'],
            [/(^\d+|[a-z]+)/g, '12ab 34'],
            [/(^|\W)#(\w+)/g, '#tag and #more #'],
            [/(?:^|[?&])id=(\d+)/g, 'id=1&x=2&id=3'],
            [/(^|\s)a/g, 'a a a'],
            [/(^|\s)a/gu, '😀 a'],
            [/(?:^|\n)b/g, 'b\nb\r\nb'],
            [/(?:^|\n)b/gm, 'b\nb b'],
        ];
        var lines = cases.map(function (c) { return c[0] + ' ' + all(c[0], c[1]); });
        lines.push([/(^|\s)active(\s|$)/.test('btn active'), /(^|\s)active(\s|$)/.test('btnactive'), /(^|x)$/.test(''), /(?:^|b)c/.test('ac')].join());
        return lines.join('\n');
        })()
        "###,
        r###"/(^|\s)active(\s|$)/g 0:7 20:7
/(^|x)/g 0:0
/(^|x)/g 0:0 1:1 3:1
/(?:^|,)(\w*)/g 0:1 1:1 2:2 4:1
/(?:^|,)(\w*)/gm 0:1 2:0 5:1
/^|a/gm 0:0 1:1 3:0 4:1 6:0
/(?:^|b)/gm 0:0 2:0 3:0
/(?:^ +|\s)x/g 0:3 3:2
/(^\d+|[a-z]+)/g 0:2 2:2
/(^|\W)#(\w+)/g 0:4 8:6
/(?:^|[?&])id=(\d+)/g 0:4 8:5
/(^|\s)a/g 0:1 1:2 3:2
/(^|\s)a/gu 2:2
/(?:^|\n)b/g 0:1 1:2 4:2
/(?:^|\n)b/gm 0:1 1:2 4:1
true,false,true,false"###,
    );
}

/// Deeply nested patterns are an implementation limit: past it, a SyntaxError rather than a
/// native stack overflow. Patterns at the limit compile and match (node gives the same results
/// for them).
#[test]
fn deeply_nested_patterns_are_syntax_errors_not_crashes() {
    check(
        r#"
        function attempt(source, flags, subject) {
            try {
                var match = new RegExp(source, flags).exec(subject);
                return match ? match.length + ':' + match.index : 'null';
            } catch (error) {
                return error.name;
            }
        }
        function nest(open, body, close, depth) {
            return open.repeat(depth) + body + close.repeat(depth);
        }
        [
            attempt(nest('(', 'a', ')', 1000), '', 'a'),
            attempt(nest('(?:', 'a', ')*', 1000), '', 'aa'),
            attempt(nest('[', 'a', ']', 1000), 'v', 'a'),
            attempt(nest('(?<=', 'a', ')', 128) + 'b', '', 'ab'),
            attempt(nest('(', 'a', ')', 100000), '', 'a'),
            attempt(nest('(?:', 'a', ')*', 100000), '', 'a'),
            attempt(nest('[', 'a', ']', 100000), 'v', 'a'),
            attempt(nest('(?=', 'a', ')', 100000), '', 'a'),
        ].join()
        "#,
        "1001:0,1:0,1:0,1:1,SyntaxError,SyntaxError,SyntaxError,SyntaxError",
    );
}

/// A match array has its elements, then ArrayCreate's `length` (writable only), then `index`,
/// `input` and `groups` and, with the `d` flag, `indices`, all CreateDataPropertyOrThrow data
/// properties in that creation order; the indices array has `groups` after its `length`, and
/// both `groups` objects have a null prototype (ECMA-262 §22.2.7.2, §22.2.7.8).
#[test]
fn match_arrays_have_the_spec_keys_attributes_and_order() {
    check(
        r#"
        function describe(object) {
            return Object.getOwnPropertyNames(object).map(function (key) {
                var d = Object.getOwnPropertyDescriptor(object, key);
                return key + ':' + (d.writable ? 'w' : '-') + (d.enumerable ? 'e' : '-') +
                    (d.configurable ? 'c' : '-');
            }).join(' ');
        }
        var plain = /(a)(b)?/.exec('xa');
        var named = /(?<first>a)(?<second>z)?/.exec('xa');
        var indexed = /(?<first>a)(b)?/d.exec('xa');
        var unnamed = /(a)/d.exec('ba');
        [
            describe(plain),
            describe(named),
            Object.getPrototypeOf(named.groups) === null,
            describe(named.groups),
            describe(indexed),
            describe(indexed.indices),
            Object.getPrototypeOf(indexed.indices.groups) === null,
            describe(indexed.indices.groups),
            JSON.stringify(indexed.indices),
            String(unnamed.indices.groups),
            Object.keys(plain).join(),
            Object.getPrototypeOf(plain) === Array.prototype,
            Array.isArray(indexed.indices),
            plain.length + ':' + plain.index + ':' + plain.input + ':' + plain.groups,
            'abcabc'.match(/b/).index + ':' + 'abcabc'.search(/c/),
            Array.from('a1b22'.matchAll(/\d+/g), function (m) { return describe(m); }).join(' | '),
        ].join('\n')
        "#,
        "0:wec 1:wec 2:wec length:w-- index:wec input:wec groups:wec\n\
         0:wec 1:wec 2:wec length:w-- index:wec input:wec groups:wec\n\
         true\n\
         first:wec second:wec\n\
         0:wec 1:wec 2:wec length:w-- index:wec input:wec groups:wec indices:wec\n\
         0:wec 1:wec 2:wec length:w-- groups:wec\n\
         true\n\
         first:wec\n\
         [[1,2],[1,2],null]\n\
         undefined\n\
         0,1,2,index,input,groups\n\
         true\n\
         true\n\
         3:1:xa:undefined\n\
         1:2\n\
         0:wec length:w-- index:wec input:wec groups:wec | \
         0:wec length:w-- index:wec input:wec groups:wec",
    );
}

/// Match arrays share their named keys' layout, but each is an ordinary array: one that gains,
/// loses or redefines a property, is truncated or grows changes alone, and later matches are
/// built as before.
#[test]
fn match_arrays_change_independently() {
    check(
        r#"
        function keys(object) { return Object.getOwnPropertyNames(object).join(); }
        var re = /(\d+)/;
        var a = re.exec('x1'), b = re.exec('y22'), c = re.exec('z333');
        a.extra = 'a';
        delete b.index;
        Object.defineProperty(c, 'input', { writable: false, enumerable: false });
        c.input = 'ignored';
        b.groups = 'b';
        var d = re.exec('w4444');
        var e = /(a)/d.exec('a');
        e.indices.groups = 'e';
        delete e.indices[0];
        var f = /(a)/d.exec('a');
        var forIn = [];
        for (var key in d) forIn.push(key);
        [
            keys(a), keys(b), keys(c), keys(d),
            a.extra + ':' + ('extra' in b) + ':' + ('extra' in d),
            b.index + ':' + d.index + ':' + b.groups + ':' + d.groups,
            c.input + ':' + Object.getOwnPropertyDescriptor(c, 'input').writable + ':' +
                Object.getOwnPropertyDescriptor(d, 'input').writable,
            Object.keys(c).join() + '|' + Object.keys(d).join(),
            (b.index = 9, keys(b)),
            (a.length = 0, keys(a)),
            (a.push('p', 'q'), keys(a) + ':' + a.extra),
            forIn.join(),
            e.indices.groups + ':' + f.indices.groups + ':' + keys(e.indices) + ':' +
                keys(f.indices),
            JSON.stringify([d, f.indices]),
            Object.isFrozen(Object.freeze(d)) + ':' + keys(re.exec('v55555')) + ':' +
                Object.isFrozen(re.exec('u6')),
        ].join('\n')
        "#,
        "0,1,length,index,input,groups,extra\n\
         0,1,length,input,groups\n\
         0,1,length,index,input,groups\n\
         0,1,length,index,input,groups\n\
         a:false:false\n\
         undefined:1:b:undefined\n\
         z333:false:true\n\
         0,1,index,groups|0,1,index,input,groups\n\
         0,1,length,input,groups,index\n\
         length,index,input,groups,extra\n\
         0,1,length,index,input,groups,extra:a\n\
         0,1,index,input,groups\n\
         e:undefined:1,length,groups:0,1,length,groups\n\
         [[\"4444\",\"4444\"],[[0,1],[0,1]]]\n\
         true:0,1,length,index,input,groups:false",
    );
}

/// Every match array of a kind shares one key layout, so a match allocates none, and has the
/// shape that inserting its named keys into an array gives (V8 shares one map the same way).
/// The layout outlives collections, and an array that gains a key leaves it to the others.
#[test]
fn match_arrays_share_one_layout_and_the_ordinary_shape() {
    use crate::value::{PropertyLayout, ValueView};
    fn eval(engine: &mut Engine, source: &str) {
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(_) => {}
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
    /// The key layout and shape of the global `name`'s object.
    fn layout(engine: &mut Engine, name: &str) -> (PropertyLayout, u32) {
        let env = engine.interp.global_env;
        let value = engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("missing {name}"));
        let ValueView::Obj(object) = value.view() else {
            panic!("{name} is not an object")
        };
        let object = object.borrow();
        let layout = object.props.shared_layout().expect("named keys").clone();
        (layout, object.props.shape())
    }
    let mut engine = Engine::new();
    eval(
        &mut engine,
        r#"
        var a = /(a)/.exec('xa'), b = /(b)(c)(d)/.exec('bcd'), s = 'xyz'.match(/y/);
        var ordinary = ['a', 'a'];
        ordinary.index = 1; ordinary.input = 'xa'; ordinary.groups = undefined;
        var da = /(a)/d.exec('a'), db = /(?<n>b)/d.exec('b');
        var ordinaryD = ['a'];
        ordinaryD.index = 0; ordinaryD.input = 'a'; ordinaryD.groups = undefined;
        ordinaryD.indices = undefined;
        var ordinaryIndices = [[0, 1]];
        ordinaryIndices.groups = undefined;
        "#,
    );
    for (first, other, ordinary) in [
        ("a", "b", "ordinary"),
        ("a", "s", "ordinary"),
        ("da", "db", "ordinaryD"),
        ("da.indices", "db.indices", "ordinaryIndices"),
    ] {
        eval(
            &mut engine,
            &format!("var first = {first}, other = {other};"),
        );
        let (first_layout, first_shape) = layout(&mut engine, "first");
        let (other_layout, other_shape) = layout(&mut engine, "other");
        let (ordinary_layout, ordinary_shape) = layout(&mut engine, ordinary);
        assert!(
            PropertyLayout::ptr_eq(&first_layout, &other_layout),
            "{other}: one layout"
        );
        assert!(
            !PropertyLayout::ptr_eq(&first_layout, &ordinary_layout),
            "{ordinary}: insertion copies the layout"
        );
        assert_eq!(&first_layout[..], &ordinary_layout[..], "{other}: keys");
        assert_eq!(first_shape, other_shape, "{other}: one shape");
        assert_eq!(
            first_shape, ordinary_shape,
            "{other}: the shape of insertion"
        );
    }
    let (a, a_shape) = layout(&mut engine, "a");
    engine.interp.gc_collect();
    eval(&mut engine, "var later = /(q)/.exec('q'); a.extra = 1;");
    let (later, later_shape) = layout(&mut engine, "later");
    assert!(PropertyLayout::ptr_eq(&a, &later));
    assert_eq!(a_shape, later_shape);
    let (changed, changed_shape) = layout(&mut engine, "a");
    assert!(!PropertyLayout::ptr_eq(&changed, &later));
    assert_ne!(changed_shape, later_shape);
    let (b, _) = layout(&mut engine, "b");
    assert!(PropertyLayout::ptr_eq(&b, &later));
    assert_eq!(
        changed
            .iter()
            .map(|key| key.text().to_string())
            .collect::<Vec<_>>(),
        ["length", "index", "input", "groups", "extra"]
    );
    assert_eq!(
        later
            .iter()
            .map(|key| key.text().to_string())
            .collect::<Vec<_>>(),
        ["length", "index", "input", "groups"]
    );
}

/// A view of a known-ASCII string, a long slice or a string appended to in place, is known
/// ASCII too: RegExp methods match over its bytes, as on a flat copy, and leave the prepared
/// subject cache, whose entries keep their strings alive, to other subjects.
#[test]
fn regexp_subjects_that_view_ascii_strings_match_as_ascii() {
    use crate::value::ValueView;
    const FIXTURE: &str = r#"
        var base = 'x'.repeat(300) + ' needle 42 haystack';
        var sliced = base.slice(10);
        var appended = 'start:' + 'go'.repeat(4);
        for (var i = 0; i < 40; i++) appended += ' ab' + i;
        [
            /needle (\d+)/.exec(sliced)[1],
            sliced.replace(/x+/, '-').slice(0, 9),
            sliced.split(/\s/).length,
            /ab(39)$/.exec(appended)[1] + ':' + /ab(39)$/.exec(appended).index,
            appended.match(/ab\d+/g).length,
            appended.search(/ab7\b/),
            /(?<w>go)+/.exec(appended).groups.w,
            appended.replace(/ab(\d)\b/g, '<$1>').slice(0, 40),
        ].join()
    "#;
    const EXPECTED: &str =
        "42,- needle ,4,39:200,40,43,go,start:gogogogo <0> <1> <2> <3> <4> <5> <";
    check(FIXTURE, EXPECTED);
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(FIXTURE, false).expect("fixture parses") {
            Completion::Value(value) => assert_eq!(value, EXPECTED, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        for name in ["sliced", "appended"] {
            let env = engine.interp.global_env;
            let value = engine
                .interp
                .get_var(name, &env)
                .unwrap_or_else(|_| panic!("missing {name}"));
            let ValueView::Str(string) = value.view() else {
                panic!("{name} is not a string")
            };
            assert!(string.view_range().is_some(), "{tier:?}: {name} is a view");
            assert!(
                !string.ascii_hint(),
                "{tier:?}: a view has no hint of its own"
            );
            assert!(string.known_ascii(), "{tier:?}: {name} is known ASCII");
        }
    }
}

/// An assertion holds at the position it is reached, wherever it sits in the pattern: the
/// tiered one-byte matcher, which takes over after 64 executions, tested `^` and `$` at the
/// candidate's start, so `/a^/` began matching "a".
#[test]
fn anchors_inside_a_pattern_keep_holding_after_the_pattern_tiers_up() {
    check(
        r#"
        function run() {
            var cases = [[/a^/, 'a'], [/a^b/, 'ab'], [/x$y/, 'xy'], [/a.^/, 'ab'], [/^ab/, 'ab'],
                         [/ab$/, 'zab'], [/^b/, 'ab'], [/a$/, 'ab'], [/^a.c$/, 'abc'],
                         [/^a.c$/, 'abcd'], [/^b./, 'abc'], [/x.$/, 'zxy']];
            return cases.map(function (c) {
                var seen = [];
                for (var i = 0; i < 200; i++) {
                    var r = c[0].test(c[1]);
                    if (seen.indexOf(r) < 0) seen.push(r);
                }
                return seen.join('/');
            }).join();
        }
        run()
        "#,
        "false,false,false,false,true,true,false,false,true,false,false,true",
    );
}

/// ECMA-262 §22.2.2.9 (CompileToCharSet): under `v` a lone surrogate is a code point of the
/// subject that classes, properties, ranges and `\q{}` strings can hold, and under `i` every
/// operand is folded (MaybeSimpleCaseFolding) before complements, differences and
/// intersections, as is a property or a `\w`, `\d` or `\s` before its complement. An inline
/// modifier group decides `i` for the classes inside it. V8 (node 24) leaks a group's `i`
/// into the class after it: it finds no match for the last `scoped` pattern.
#[test]
fn unicode_sets_classes_keep_lone_surrogates_and_fold_their_operands() {
    check(
        r#"
        function run() {
            var lone = [/\p{Any}/v.test('\ud83d'), /[\s\S]/v.test('\udc00'), /[^a]/v.test('\ud800'),
                /\P{L}/v.test('\ud800'), /\p{Cs}/v.test('\ud800'), /[\ud800-\udbff]/v.test('\udb00'),
                /[\q{\ud800}]/v.test('\ud800'), /[\q{\ud800}]/v.test('�'),
                /[\q{\ud800x|ab}]/v.test('\ud800x'), /^[\p{Any}--\p{Cs}]$/v.test('\udc00'),
                /^[\p{Any}--\p{Cs}]$/v.test('😀')];
            var fold = [/[^a-z]/iv.test('b'), /[^a-z]/iv.test('B'), /[^a-z]/iv.test('1'),
                /[\p{L}--[a-z]]/iv.test('A'), /[\p{L}--[a-z]]/iv.test('é'),
                /[[a-z]&&[A-Z]]/iv.test('q'), /[\w--\d]/iv.test('ſ'), /[^\P{Ll}]/iv.test('A'),
                /[ẞ]/vi.test('ß'), /[^ß]/vi.test('ẞ'),
                /[\p{Lu}--ẞ]/vi.test('ß'), /[\q{AB}--\q{ab}]/vi.test('AB'),
                /[\q{AB}]/vi.test('ab'), /[\W]/vi.test('ſ'), /[^\w]/vi.test('K'),
                /\W/vi.test('ſ'), /\P{Ll}/iv.test('a'), /[\P{Ll}]/iv.test('a'),
                /[^\p{Lu}]/iv.test('a')];
            var scoped = [/(?i:[^a-z])/v.test('B'), /(?i:[\p{L}--[a-z]])/v.test('A'),
                /[^a-z]/v.test('B'), /(?-i:[^a-z])/vi.test('B'), /[\p{L}--[a-z]]/v.test('A'),
                /(?i:[^a-z])[^a-z]/v.test('BB'), /(?i:x)[^a-z]/v.test('XB')];
            var space = [/[\s]/v.test('\x85'), /[\S]/v.test('\x85'), /[\s]/v.test('﻿'),
                /[\s--\n]/v.test('\n')];
            return [lone, fold, scoped, space].map(function (group) {
                return group.map(Number).join('');
            }).join('|');
        }
        run()
        "#,
        "11111110101|0010111110001000000|0011101|0110",
    );
}
