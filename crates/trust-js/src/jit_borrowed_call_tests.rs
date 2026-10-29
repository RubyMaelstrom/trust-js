//! Calls from borrowed frames (W3's item 4): a loop the tree-walker or the bytecode VM moved into
//! native code (a loop fragment, an OSR'd Script loop) and a coroutine's resumed slices make
//! their calls, constructions, getter reads and setter stores through the direct sequence, whose
//! callee runs on a record of its own. The frame's continuation state comes through every call
//! as it was: an abrupt completion from the callee reaches the handlers the frame borrowed from
//! the VM (ECMA-262 §14.15.3 TryStatement evaluation; §27.5.3.3 GeneratorResume), a return's
//! value passes the finalizers that make calls (§14.15.3 step 3, UpdateEmpty), and a generator
//! the callee re-enters stays running (§27.5.3.2 GeneratorValidate step 4). Every tier agrees;
//! the expected values are Node v24's.
//!
//! On ARM64 the named callees complete through the sequence (their chunks' direct return and
//! throw counters) while the frame is borrowed (its native entries or slices are counted).

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;
#[cfg(target_arch = "aarch64")]
use super::traced_tests::with_chunk;

/// Where a case's calls run in native code: a non-resumable borrowed frame (a loop fragment or
/// an OSR'd loop) or a coroutine's slices.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Borrowed {
    Loop,
    Slices,
}

struct Case {
    source: &'static str,
    expected: &'static str,
    borrowed: Borrowed,
    /// Global functions called only from borrowed frames, which must complete through the direct
    /// sequence on ARM64.
    direct: &'static [&'static str],
}

/// A fresh engine at `tier` with `collect()`. Functions tier up after 32 calls, so a function
/// called once stays with the tree-walker, which moves its hot loop into a native fragment.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(32);
    let collect = engine.interp.make_native("collect", 0, |interp, _, _| {
        interp.gc_collect();
        Ok(crate::value::Value::Undefined)
    });
    engine.interp.global.borrow_mut().props.insert(
        "collect",
        crate::value::Property::data(crate::value::Value::Obj(collect), true, true, true),
    );
    engine
}

fn check(case: &Case) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        super::TEST_OSR_ENTRIES.with(|n| n.set(0));
        super::TEST_NATIVE_SLICES.with(|n| n.set(0));
        assert_eq!(
            eval(&mut engine, case.source),
            case.expected,
            "{tier:?}: {}",
            case.source
        );
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit {
            let entries = match case.borrowed {
                Borrowed::Loop => super::TEST_OSR_ENTRIES.with(|n| n.get()),
                Borrowed::Slices => super::TEST_NATIVE_SLICES.with(|n| n.get()),
            };
            assert!(entries > 0, "no borrowed native frame ran: {}", case.source);
            for name in case.direct {
                let completions = with_chunk(&mut engine, name, |chunk| {
                    chunk.test_direct_returns.get() + chunk.test_direct_throws.get()
                });
                assert!(
                    completions > 0,
                    "{name} never completed through the direct sequence: {}",
                    case.source
                );
            }
        }
    }
}

#[test]
fn a_loop_fragment_enters_every_callee_form_directly() {
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
var obj = { v: 1, get(k) { return this.v + k; } };
function mk(x) { return function (y) { return x + y; }; } var add3 = mk(3);
function envCallee(a) { var t = a + 1; if (a < 0) return () => t; return t; }
function rest(...xs) { return xs.length + xs[0]; }
function args() { return arguments.length + arguments[0]; }
function mapped(a, b) { return arguments.length + arguments[1] + a; }
function cls(k) { 'use strict'; class A { constructor() { this.v = k; } } return new A().v; }
function inc(k) { return k + 1; }
function tail(k) { 'use strict'; return inc(k); }
function forms(n) {
  var s = 0;
  for (var k = 0; k < n; k++) {
    s = (s + add(s, k) + obj.get(k) + add3(k) + envCallee(k) + rest(k, 2) + args(k, 1, 2) + mapped(k, 5) + tail(k)) | 0;
    if ((k & 3) === 0) s = (s + cls(k)) | 0;
  }
  return s;
}
forms(3000)"#,
        expected: "324483014",
        borrowed: Borrowed::Loop,
        direct: &[
            "add",
            "add3",
            "envCallee",
            "rest",
            "args",
            "mapped",
            "cls",
            "tail",
        ],
    });
}

#[test]
fn a_script_loop_enters_its_callees_directly() {
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
var obj = { v: 1, get(k) { return this.v + k; } };
function mk(x) { return function (y) { return x + y; }; } var add3 = mk(3);
var s = 0;
for (var k = 0; k < 3000; k++) { s = (s + add(s, k) + obj.get(k) + add3(k)) | 0; }
s"#,
        expected: "-9007",
        borrowed: Borrowed::Loop,
        direct: &["add", "add3"],
    });
}

#[test]
fn a_generators_slices_enter_their_callees_directly() {
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
var obj = { v: 1, get(k) { return this.v + k; } };
function* g(n) { var s = 0; for (var k = 0; k < n; k++) { s = (s + add(s, k) + obj.get(k)) | 0; if ((k & 63) === 0) yield s; } return s; }
var it = g(3000), r, ys = 0; do { r = it.next(); if (!r.done) ys++; } while (!r.done);
[r.value, ys].join()"#,
        expected: "-6003,47",
        borrowed: Borrowed::Slices,
        direct: &["add"],
    });
}

#[test]
fn a_callees_throw_reaches_the_handlers_the_frame_borrowed() {
    // Caught in the fragment, with a finalizer that calls; escaping the fragment, with the
    // thrower's frame in the stack; caught in a generator's slices.
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function mayThrow(k) { if (k % 97 === 0) throw new Error('t' + k); return k & 7; }
function handlers(n) {
  var s = 0, caught = 0, fin = 0;
  for (var k = 0; k < n; k++) {
    try { s = (s + mayThrow(k)) | 0; } catch (e) { caught++; s = (s + e.message.length) | 0; } finally { fin = add(fin, 1); }
  }
  return [s, caught, fin].join();
}
handlers(3000)"#,
        expected: "10536,31,3000",
        borrowed: Borrowed::Loop,
        direct: &["add", "mayThrow"],
    });
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function boom(k, n) { if (k === n - 1) throw new Error('boom' + k); return 1; }
function escape(n) { var s = 0; for (var k = 0; k < n; k++) { s = (s + add(s, k) + boom(k, n)) | 0; } return s; }
var m; try { escape(3000); } catch (e) { m = e.message + ':' + (e.stack.indexOf('boom') >= 0); }
m"#,
        expected: "boom2999:true",
        borrowed: Borrowed::Loop,
        direct: &["add", "boom"],
    });
    check(&Case {
        source: r#"
function mayThrow(k) { if (k % 97 === 0) throw new Error('t' + k); return k & 7; }
function* gt(n) { var c = 0, y = 0; for (var k = 0; k < n; k++) { try { if (k % 64 === 0) { y++; yield mayThrow(k); } else mayThrow(k); } catch (e) { c++; } } return [c, y].join(); }
var it = gt(3000), r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "31,47",
        borrowed: Borrowed::Slices,
        direct: &["mayThrow"],
    });
}

#[test]
fn a_return_passes_finalizers_that_call() {
    // `cls` builds a class, which materializes its frame's native activation, so its finish
    // takes the helper, which moves the completion through the caller's `ret`.
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function cls(k) { 'use strict'; class A { constructor() { this.v = k; } } return new A().v; }
function retAfter(n) { var s = 0, t = 0; for (var k = 0; k < n; k++) { try { if (k === n - 7) return [add(s, k), t, cls(k)].join(); s = (s + add(s, k)) | 0; } finally { t = add(t, cls(1)); } } return 'no'; }
retAfter(3000)"#,
        expected: "-1,2993,2993",
        borrowed: Borrowed::Loop,
        direct: &["add", "cls"],
    });
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function cls(k) { 'use strict'; class A { constructor() { this.v = k; } } return new A().v; }
function* retGen(n) { var s = 0; for (var k = 0; k < n; k++) { s = (s + add(s, cls(k))) | 0; if (k === n - 3) return [cls(k), add(s, 1)].join(); if ((k & 127) === 0) yield k; } }
var it = retGen(3000), r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "2997,-2998",
        borrowed: Borrowed::Slices,
        direct: &["add", "cls"],
    });
}

#[test]
fn constructions_getters_and_setters_from_borrowed_frames() {
    check(&Case {
        source: r#"
function P(x, y) { this.x = x; this.y = y; }
class G { constructor(x) { this.x = x; } get d() { return this.x * 2; } set d(v) { this.x = v; } }
var getD = Object.getOwnPropertyDescriptor(G.prototype, 'd').get;
var setD = Object.getOwnPropertyDescriptor(G.prototype, 'd').set;
function objects(n) { var s = 0, g = new G(1); for (var k = 0; k < n; k++) { var p = new P(k, 1); g.d = k; s = (s + p.y + g.d + new G(k).x) | 0; } return s; }
objects(3000)"#,
        expected: "13498500",
        borrowed: Borrowed::Loop,
        direct: &["P", "G", "getD", "setD"],
    });
    check(&Case {
        source: r#"
function P(x, y) { this.x = x; this.y = y; }
class G { constructor(x) { this.x = x; } get d() { return this.x * 2; } set d(v) { this.x = v; } }
var getD = Object.getOwnPropertyDescriptor(G.prototype, 'd').get;
var setD = Object.getOwnPropertyDescriptor(G.prototype, 'd').set;
function* objGen(n) { var s = 0, g = new G(1); for (var k = 0; k < n; k++) { var p = new P(k, 2); g.d = k; s = (s + p.y + g.d) | 0; if ((k & 127) === 0) yield; } return s; }
var it = objGen(3000), r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "9003000",
        borrowed: Borrowed::Slices,
        direct: &["P", "getD", "setD"],
    });
}

#[test]
fn borrowed_frames_recurse_past_the_stack_limit() {
    check(&Case {
        source: r#"
function deep(n) { return n === 0 ? 0 : 1 + deep(n - 1); }
function recursion(n) { var s = 0, e = false; for (var k = 0; k < n; k++) { s = (s + deep(k & 15)) | 0; } try { deep(1e6); } catch (x) { e = x instanceof RangeError; } return [s, e].join(); }
recursion(3000)"#,
        expected: "22468,true",
        borrowed: Borrowed::Loop,
        direct: &["deep"],
    });
}

#[test]
fn a_borrowed_frames_operands_survive_its_callees_collections() {
    // `keep` waits on the fragment's operand stack while `alloc` runs and collects.
    check(&Case {
        source: r#"
function alloc(k) { var a = []; for (var i = 0; i < 20; i++) a.push({ i: i, k: k }); if ((k & 255) === 0) collect(); return a.length; }
function pair(o, x) { return o.tag.length + x; }
function collecting(n) { var keep = { tag: 'kept' }, s = 0; for (var k = 0; k < n; k++) { s = (s + pair(keep, alloc(k))) | 0; } return [s, keep.tag].join(); }
collecting(3000)"#,
        expected: "72000,kept",
        borrowed: Borrowed::Loop,
        direct: &["alloc", "pair"],
    });
}

#[test]
fn a_callee_that_resumes_the_running_generator_throws() {
    check(&Case {
        source: r#"
var it; function reenter() { try { it.next(); return 'no'; } catch (e) { return e.name; } }
function* gr(n) { var o = ''; for (var k = 0; k < n; k++) { var v = reenter(); if (k === n - 1) o = v; if ((k & 63) === 0) yield k; } return o; }
it = gr(500); var r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "TypeError",
        borrowed: Borrowed::Slices,
        direct: &["reenter"],
    });
}

#[test]
fn a_constructors_fragment_keeps_its_new_target() {
    // The tree-walker runs `Ctor` as a [[Construct]]: the fragment's calls save and restore its
    // construct state, and the arrow keeps the lexical new.target.
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function Ctor(n) { var f = () => new.target === Ctor, bad = 0, s = 0; for (var k = 0; k < n; k++) { if (!f()) bad++; s = (s + add(s, k)) | 0; } this.r = [s, bad].join(); }
new Ctor(3000).r"#,
        expected: "-3001,0",
        borrowed: Borrowed::Loop,
        direct: &["add"],
    });
}
