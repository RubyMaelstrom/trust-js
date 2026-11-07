//! Fused String method calls (`jit::string_calls`, Djot's gap, candidate 4, step 2): a call the
//! ARM64 JIT finishes in one step answers as the call does, for every index form it reads and
//! every kind of string, and every write to the method, its removal, another Realm's method and
//! other receivers send the call back to the unfused templates. Each answer is checked in every
//! tier; the counts of fused calls, on ARM64 only, show which calls the step finished.
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

const TIERS: [Tier; 3] = [Tier::Interp, Tier::Bytecode, Tier::Jit];

/// Whether this build fuses calls (the ARM64 JIT).
fn fuses(tier: Tier) -> bool {
    tier == Tier::Jit
        && cfg!(all(
            target_arch = "aarch64",
            any(
                target_os = "macos",
                target_os = "linux",
                target_os = "windows"
            )
        ))
}

/// The fused calls `source` made, and its answer.
fn fused_in(engine: &mut Engine, source: &str) -> (usize, String) {
    let before = crate::jit::test_fused_string_calls();
    let answer = eval(engine, source);
    (crate::jit::test_fused_string_calls() - before, answer)
}

/// A local, a constant and a local plus or minus a constant, outside a loop (the shared stub) and
/// in one (in line: a loop the region compiler keeps whole, here for its `try`); one- and two-byte
/// strings, a view, surrogate pairs and lone surrogates; every kind of index, those the step
/// leaves to the call included (fractions, negative values, NaN, -0, past the length and past
/// 2^32), and indices that are not Numbers.
#[test]
fn fused_code_point_calls_answer_every_index_form_as_the_call_does() {
    let source = r#"
        function local(s, i) { return s.codePointAt(i); }
        function constant(s) { return s.codePointAt(1); }
        function plus(s, i) { return s.codePointAt(i + 1); }
        function minus(s, i) { return s.codePointAt(i - 1); }
        function scan(s) {
            var t = 0;
            for (var i = -1; i < s.length + 2; i++) {
                try {
                    var c = s.codePointAt(i), d = s.codePointAt(i + 1);
                    t = (t * 31 + (c === undefined ? 7 : c) + (d === undefined ? 5 : d)) | 0;
                } catch (e) {
                    t = -1;
                }
            }
            return t;
        }
        var one = 'abc', two = 'A😀\uD800z\uDC00é';
        var view = ('x'.repeat(40) + two + 'tail').slice(40, 40 + two.length);
        var indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 1.5, -0, NaN, Infinity, 2 ** 32, 2 ** 32 + 1,
            -0.5, '1', true, null];
        var out;
        for (var round = 0; round < 3; round++) {
            out = [];
            for (var s of [one, two, view]) {
                for (var i of indices) out.push(local(s, i), plus(s, i), minus(s, i));
                out.push(constant(s), scan(s));
            }
        }
        var trace = '', index = { valueOf() { trace += 'v'; return 2; } };
        out.push(local(one, index), trace, plus(one, index), trace);
        out.join()
    "#;
    let mut answers = Vec::new();
    for tier in TIERS {
        let mut engine = engine(tier);
        let (fused, answer) = fused_in(&mut engine, source);
        if fuses(tier) {
            // 142 calls a round have an exact unsigned index, three rounds; a site's first call
            // fills it, and the other indices take the call.
            assert!(fused >= 400, "{fused} fused calls");
        }
        answers.push(answer);
    }
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "{answers:#?}"
    );
    assert!(answers[0].starts_with("97,98,"), "{}", answers[0]);
}

/// charCodeAt's fused calls: its unit, NaN past the end, and no pair combined, for every index
/// form; a write to it sends its calls back to the call and leaves codePointAt's fused.
#[test]
fn fused_char_code_at_calls_answer_as_the_call_does() {
    let source = r#"
        function local(s, i) { return s.charCodeAt(i); }
        function constant(s) { return s.charCodeAt(1); }
        function plus(s, i) { return s.charCodeAt(i + 1); }
        function minus(s, i) { return s.charCodeAt(i - 1); }
        function scan(s) {
            var t = 0;
            for (var i = -1; i < s.length + 2; i++) {
                try {
                    var c = s.charCodeAt(i);
                    t = (t * 31 + (c !== c ? 7 : c)) | 0;
                } catch (e) {
                    t = -1;
                }
            }
            return t;
        }
        var one = 'abc', two = 'A\uD83D\uDE00\uD800z\uDC00\u00e9';
        var view = ('x'.repeat(40) + two + 'tail').slice(40, 40 + two.length);
        var indices = [0, 1, 2, 3, 6, 7, 8, -1, 1.5, -0, NaN, 2 ** 32, '1', null];
        var out;
        for (var round = 0; round < 3; round++) {
            out = [];
            for (var s of [one, two, view]) {
                for (var i of indices) out.push(local(s, i), plus(s, i), minus(s, i));
                out.push(constant(s), scan(s));
            }
        }
        out.join()
    "#;
    let mut answers = Vec::new();
    for tier in TIERS {
        let mut engine = engine(tier);
        let (fused, answer) = fused_in(&mut engine, source);
        if fuses(tier) {
            assert!(fused >= 200, "{fused} fused calls");
        }
        answers.push(answer);
    }
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "{answers:#?}"
    );
    assert!(answers[0].starts_with("97,98,NaN,"), "{}", answers[0]);
    for tier in TIERS {
        let mut engine = engine(tier);
        let warm = "function unit(s, i) { return s.charCodeAt(i); } \
                    function point(s, i) { return s.codePointAt(i); } \
                    for (var w = 0; w < 60; w++) { unit('ab', 1); point('ab', 1); } \
                    String.prototype.charCodeAt = function (i) { return 'u' + i; };";
        eval(&mut engine, warm);
        let (fused, answer) = fused_in(&mut engine, "unit('ab', 1) + ',' + point('ab', 1)");
        assert_eq!(answer, "u1,98", "{tier:?}");
        if fuses(tier) {
            assert_eq!(fused, 1, "{tier:?}: codePointAt's call stays fused");
        }
    }
}

/// A call outside a loop (`cp`, the shared stub's) and two in one (`scan`, in line: the region
/// compiler keeps a loop with a `try` whole), warmed; `both` answers for the three.
const WARM: &str = r#"
    function cp(s, i) { return s.codePointAt(i); }
    function scan(s) {
        var r = [];
        for (var i = 0; i < 2; i++) {
            try { r.push(s.codePointAt(i)); } catch (e) { r.push(e.constructor.name); }
        }
        return r.join('/');
    }
    function both() {
        var c;
        try { c = cp('ab', 0); } catch (e) { c = e.constructor.name; }
        return c + ',' + scan('ab');
    }
    for (var w = 0; w < 60; w++) both();
    both()
"#;

/// Every kind of write to String.prototype.codePointAt reaches the fused calls: assignment of
/// another value and of the same one, Reflect.set, defineProperty with a getter and with
/// attributes alone, a removal and a non-callable value. A polyfill replacing another method,
/// freezing String.prototype and giving it another prototype leave them fused.
#[test]
fn every_write_to_the_method_reaches_its_fused_calls() {
    // (the write, the answer after it, whether the calls stay fused)
    let cases: &[(&str, &str, bool)] = &[
        (
            "String.prototype.codePointAt = function (i) { return 'w' + i; };",
            "w0,w0/w1",
            false,
        ),
        (
            "String.prototype.codePointAt = String.prototype.codePointAt;",
            "97,97/98",
            false,
        ),
        (
            "Reflect.set(String.prototype, 'codePointAt', function (i) { return 'r' + i; });",
            "r0,r0/r1",
            false,
        ),
        (
            "Object.defineProperty(String.prototype, 'codePointAt', \
             { get() { return function (i) { return 'g' + i; }; } });",
            "g0,g0/g1",
            false,
        ),
        (
            "Object.defineProperty(String.prototype, 'codePointAt', { writable: false });",
            "97,97/98",
            false,
        ),
        (
            "delete String.prototype.codePointAt;",
            "TypeError,TypeError/TypeError",
            false,
        ),
        (
            "String.prototype.codePointAt = 5;",
            "TypeError,TypeError/TypeError",
            false,
        ),
        (
            "String.prototype.charAt = function () { return 'p'; };",
            "97,97/98",
            true,
        ),
        ("Object.freeze(String.prototype);", "97,97/98", true),
        (
            "Object.setPrototypeOf(String.prototype, { codePointAt() { return 'proto'; } });",
            "97,97/98",
            true,
        ),
    ];
    for &(write, expected, stays) in cases {
        for tier in TIERS {
            let mut engine = engine(tier);
            assert_eq!(eval(&mut engine, WARM), "97,97/98", "{tier:?}");
            eval(&mut engine, write);
            let (fused, answer) = fused_in(&mut engine, "both() + '|' + both()");
            assert_eq!(
                answer,
                format!("{expected}|{expected}"),
                "{tier:?}: {write}"
            );
            if fuses(tier) {
                // Two rounds of three calls: a call that refills its site runs unfused once.
                if stays {
                    assert!(fused >= 3, "{write}: {fused} fused calls");
                } else {
                    assert_eq!(fused, 0, "{write}");
                }
            }
        }
    }
}

/// A removed property that was tracked bumps the fuse's generation: the sites refill, run unfused
/// once, and are fused again.
#[test]
fn a_new_generation_refills_the_sites() {
    for tier in TIERS {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, WARM), "97,97/98", "{tier:?}");
        // `extra` is Constant after its second write, so its removal bumps the generation.
        eval(
            &mut engine,
            "String.prototype.extra = 1; String.prototype.extra = 2; delete String.prototype.extra;",
        );
        assert_eq!(eval(&mut engine, "both()"), "97,97/98", "{tier:?}");
        let (fused, answer) = fused_in(&mut engine, "both() + '|' + both()");
        assert_eq!(answer, "97,97/98|97,97/98", "{tier:?}");
        if fuses(tier) {
            assert_eq!(fused, 6, "{tier:?}");
        }
    }
}

/// A site runs in whichever Realm is active, and the active Realm's String.prototype is the one
/// the call reads: another Realm's own method answers there, and another Realm's intact one is
/// fused against that Realm's fuse.
#[test]
fn another_realms_string_prototype_is_its_own() {
    let mut answers = Vec::new();
    for tier in TIERS {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, WARM), "97,97/98", "{tier:?}");
        let replaced = engine.interp.create_realm();
        let intact = engine.interp.create_realm();
        let global = crate::value::Value::Obj(engine.interp.global);
        for (name, realm) in [("replaced", replaced), ("intact", intact)] {
            engine
                .interp
                .member_set(&global, name, realm)
                .unwrap_or_else(|_| panic!("{name}"));
        }
        eval(
            &mut engine,
            "replaced.both = both; intact.both = both; \
             replaced.eval(\"String.prototype.codePointAt = function (i) { return 'other' + i; }\");",
        );
        let (fused, in_intact) = fused_in(
            &mut engine,
            "intact.eval('both()') + '|' + intact.eval('both()')",
        );
        let answer = eval(
            &mut engine,
            "[replaced.eval('both()'), both(), replaced.eval('both()'), both()].join('|')",
        );
        assert_eq!(in_intact, "97,97/98|97,97/98", "{tier:?}");
        if fuses(tier) {
            // The sites refill for the intact Realm's fuse at its first round.
            assert!(fused >= 3, "{tier:?}: {fused} fused calls");
        }
        answers.push(answer);
    }
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "{answers:#?}"
    );
}

/// A String object, a Number with a method on Number.prototype and an object with its own method
/// take the call, and the primitive String after them is fused again.
#[test]
fn receivers_other_than_primitive_strings_take_the_call() {
    for tier in TIERS {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, WARM), "97,97/98", "{tier:?}");
        eval(
            &mut engine,
            "Number.prototype.codePointAt = function (i) { return 'n' + i; }; \
             var own = { codePointAt(i) { return 'o' + i; } };",
        );
        let (fused, answer) = fused_in(
            &mut engine,
            "[cp(new String('ab'), 1), cp(5, 2), cp(own, 3), cp('ab', 1)].join()",
        );
        assert_eq!(answer, "98,n2,o3,98", "{tier:?}");
        if fuses(tier) {
            assert_eq!(fused, 1, "{tier:?}");
        }
    }
}

/// Step 3: in a register region (Djot's getEol and skipSpace loops, whose region the compiler
/// keeps), the fused call reads its index from the loop's registers and leaves a Number for the
/// comparisons after it. Every miss runs the region's unfused operations: a code point past the
/// end, a String object, an index that is not a Number, a replaced method returning a Number
/// (joining the fast path's Number) and one returning something else (resuming after the call).
#[test]
fn fused_calls_in_register_regions_answer_as_the_call_does() {
    let source = r#"
        function eol(t, e) {
            for (; 10 !== (s = t.codePointAt(e)) && 13 !== s;) e++;
            var s;
            return e;
        }
        function skip(t, e) {
            for (; 32 === (s = t.codePointAt(e)) || 9 === s;) e++;
            var s;
            return e;
        }
        function units(t, e) {
            var n = 0;
            for (; t.charCodeAt(e + 1) !== 10; e++) n = (n * 31 + t.charCodeAt(e)) | 0;
            return n;
        }
        var line = 'abc def\tghié😀z\n', text = line.repeat(20);
        var out = [];
        for (var round = 0; round < 30; round++) {
            out = [];
            for (var at = 0; at < text.length; at += line.length) {
                out.push(eol(text, at), skip(text, at + 3), units(text, at));
            }
        }
        // Misses inside the warm loops: past the end (skip's), a String object, a String index.
        out.push(eol('ab\n', 0), skip('  ', 0), eol(new String('a\nb'), 0), eol('ab\ncd', '1'));
        out.join()
    "#;
    let mut answers = Vec::new();
    for tier in TIERS {
        let mut engine = engine(tier);
        let (fused, answer) = fused_in(&mut engine, source);
        if fuses(tier) {
            assert!(fused >= 10_000, "{fused} fused calls");
        }
        answers.push(answer);
    }
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "{answers:#?}"
    );
    // After the regions compiled: a replaced method returning a Number joins the loop's Number,
    // one returning a String resumes after the call, and the original fuses again only in a new
    // Realm's sites (the method stays NotConstant here).
    let after = r#"
        var original = String.prototype.codePointAt;
        String.prototype.codePointAt = function (i) { return original.call(this, i) === 100 ? 10 : 0; };
        var replaced = [eol('abcdef', 0), skip('abc', 0)];
        String.prototype.codePointAt = function (i) { return i < 2 ? 'x' : 10; };
        var strings = [eol('abcdef', 0)];
        String.prototype.codePointAt = original;
        [replaced, strings, eol('abc\n', 0), skip('  x', 0)].join('|')
    "#;
    let mut answers = Vec::new();
    for tier in TIERS {
        let mut engine = engine(tier);
        eval(&mut engine, source);
        let (fused, answer) = fused_in(&mut engine, after);
        if fuses(tier) {
            assert_eq!(fused, 0, "the method stays NotConstant");
        }
        answers.push(answer);
    }
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "{answers:#?}"
    );
    assert_eq!(answers[0], "3,0|2|3|2");
}
