//! W3 item 2c's call templates for proven natives (ECMA-262 at e28783d5, spec.html): Array.isArray
//! (IsArray, :5878) and Object.is (SameValue, :6008). Each program runs in every tier, its call
//! sites both outside loops (the JIT's shared stubs) and inside them (in line); the expected
//! values are Node v24.22.0's.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("template fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Array.isArray, as a method and through a captured alias (a plain call, Vue's
/// `const j = Array.isArray`), on every kind of value: a Proxy's target decides, a revoked
/// Proxy throws, Array.prototype and Array subclasses' instances are Arrays, typed arrays and
/// `arguments` are not. Object.is on Numbers (NaN, signed zeros), Strings and BigInts of equal
/// content in distinct cells, identities and mixed kinds. Replaced builtins fail the call
/// cache's proof.
const IS_ARRAY_AND_OBJECT_IS: &str = r#"
var log = [];
function rec(x) { log[log.length] = (typeof x === 'number' && x !== x ? 'NaN' : String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
// One call site per form, outside any loop (the shared stubs); the callers loop.
function isA(x) { return Array.isArray(x); }
var isAlias = (function () { const j = Array.isArray; return function (x) { return j(x); }; })();
function same(a, b) { return Object.is(a, b); }
// The same calls inside loops keep their sequences in line.
function hotIsA(list) { var s = ''; for (var q = 0; q < list.length; q++) s += Array.isArray(list[q]) ? '1' : '0'; return s; }
function hotAlias(list) { const j = Array.isArray; var s = ''; for (var q = 0; q < list.length; q++) s += j(list[q]) ? '1' : '0'; return s; }
function hotSame(list) { var s = ''; for (var q = 0; q < list.length; q++) s += Object.is(list[q][0], list[q][1]) ? '1' : '0'; return s; }
var revocable = Proxy.revocable([], {}); revocable.revoke();
var values = [[], [1, 2], {}, { length: 0 }, null, undefined, 0, -0, 1, NaN, '', 'a', true, false,
  Symbol('s'), 10n, function () {}, new Proxy([], {}), new Proxy({}, {}), new Proxy(new Proxy([], {}), {}),
  Array.prototype, Object.create(Array.prototype), new Uint8Array(2), (function () { return arguments; })(),
  new (class extends Array {})(), new Map(), /x/, new Date(0), new String('s'), Object(1n)];
var cold = '';
for (var r = 0; r < 20; r++) {
  cold = '';
  for (var i = 0; i < values.length; i++) cold += (isA(values[i]) ? '1' : '0') + (isAlias(values[i]) ? '1' : '0');
}
rec(cold);
var hot = '';
for (var r2 = 0; r2 < 20; r2++) hot = hotIsA(values) + ':' + hotAlias(values);
rec(hot);
rec(err(function () { return isA(revocable.proxy); })); rec(err(function () { return isAlias(revocable.proxy); }));
rec(err(function () { return hotIsA([[], revocable.proxy]); })); rec(err(function () { return hotAlias([revocable.proxy]); }));
var nested = Proxy.revocable([], {}); var outer = new Proxy(nested.proxy, {}); nested.revoke();
rec(err(function () { return isA(outer); }));
// SameValue (spec.html:6008): NaN is NaN, +0 is not -0, Strings and BigInts by content.
var s1 = 'ab' + 'cd', s2 = 'abc' + 'd', w1 = 'éx'.repeat(3), w2 = 'éx'.repeat(3);
var big1 = 2n ** 64n, big2 = 2n ** 64n, o = {}, sym = Symbol('q');
var pairs = [[NaN, NaN], [0, -0], [-0, 0], [-0, -0], [0, 0], [1, 1], [1, 2], [NaN, 0], [s1, s2], [s1, 'abce'],
  [s1, 'abc'], ['', ''], [w1, w2], [w1, w1.slice(1)], [w1, s1 + 'xy'], [o, o], [o, {}], [sym, sym], [sym, Symbol('q')],
  [big1, big2], [1n, 2n], [1n, 1], [true, true], [true, false], [null, null], [undefined, undefined], [null, undefined],
  [0, '0'], [NaN, 'NaN'], [Infinity, Infinity], [-Infinity, Infinity], [1.5, 1.5], [0.1 + 0.2, 0.3], [isA, isA],
  [values[17], values[17]], [values[17], []], [-0, false], [0, null], ['1', 1]];
var coldSame = '';
for (var r3 = 0; r3 < 20; r3++) {
  coldSame = '';
  for (var p = 0; p < pairs.length; p++) coldSame += same(pairs[p][0], pairs[p][1]) ? '1' : '0';
}
rec(coldSame);
var hotSameResult = '';
for (var r4 = 0; r4 < 20; r4++) hotSameResult = hotSame(pairs);
rec(hotSameResult);
rec(same(1) + ':' + same() + ':' + same(undefined, void 0));
// Replacing a builtin changes the call cache's identity proof.
var savedIsArray = Array.isArray, savedIs = Object.is;
Array.isArray = function (x) { return 'patched' + typeof x; }; Object.is = function () { return 'patched'; };
rec(isA([])); rec(hotIsA([[]])); rec(same(1, 1)); rec(hotSame([[1, 1]]));
Array.isArray = savedIsArray; Object.is = savedIs;
rec(isA([])); rec(hotIsA([[]])); rec(same(1, 1)); rec(hotSame([[1, 1]])); rec(isAlias([]));
log.join(',');
"#;

#[test]
fn is_array_and_object_is_templates_match_the_full_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, IS_ARRAY_AND_OBJECT_IS),
            "111100000000000000000000000000000011001111000000110000000000,110000000000000001011000100000:110000000000000001011000100000,TypeError,TypeError,TypeError,TypeError,TypeError,100111001001100101010010110001010110000,100111001001100101010010110001010110000,false:true:true,patchedobject,1,patched,1,true,1,true,1,true",
            "{tier:?}"
        );
        engine.interp.gc_collect();
    }
}

/// In compiled code the templates finish warm calls without the call-hit helper: Array.isArray as a method and through a plain call, and Object.is, count their commits,
/// both at sites outside loops (the shared stubs) and inside one (in line).
#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn call_templates_finish_compiled_calls() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function isA(x) { return Array.isArray(x); }
        var isAlias = (function () { const j = Array.isArray; return function (x) { return j(x); }; })();
        function same(a, b) { return Object.is(a, b); }
        function hot(n) {
            var t = 0;
            for (var q = 0; q < n; q++) {
                t += (Array.isArray(t) ? 1 : 0) + (Object.is(q, 3) ? 1 : 0) + q;
            }
            return t;
        }
        var arr = [1];
        for (var warm = 0; warm < 200; warm++) {
            isA(arr); isAlias(arr); same(1, 2);
        }
        hot(50);
    "#,
    );
    let counts = || crate::jit::TEST_CALL_TEMPLATES.with(std::cell::Cell::get);
    // CALL_IC_EPOCH is process-wide: a concurrently running test that recompiles a chunk or
    // reclaims native code invalidates every cached call site, and the next calls here
    // legitimately refill through the generic path. Measure a run that ended at the epoch the
    // previous run started at (see code_point_intrinsic_tests).
    let (mut served, mut epoch_before, mut epoch_after) = (0, 0, 1);
    let mut previous_start = None;
    for _attempt in 0..200 {
        let before = counts();
        epoch_before = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var ok = true;
            for (var n = 0; n < 40; n++) {
                ok = ok && isA(arr) && isAlias(arr) && !same(1, 2);
            }
            ok && hot(30) === 436
        "#
            ),
            "true"
        );
        let after = counts();
        served = after - before;
        epoch_after = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if previous_start == Some(epoch_after) {
            break;
        }
        previous_start = Some(epoch_before);
    }
    // Per run: the in-line templates, three 40 times and two 30 times.
    let templates = served;
    assert!(
        templates >= 180,
        "templates {templates}, epoch {epoch_before}->{epoch_after}"
    );
    assert!(engine.interp.fn_frames.is_empty());
}
