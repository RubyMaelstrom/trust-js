//! Generated code under gc-heap's traced ownership (P1 S5 step 6): the engine runs at
//! `Tier::Jit` (admitted under gc-heap on ARM64) and collects as usual, its JIT activations
//! being roots (whose rooting tests are `jit::rooting_tests`).
//!
//! Each program runs at `Tier::Jit` and at `Tier::Bytecode` (tier threshold 0) in the same build
//! and must produce the same, pinned, result; the functions it names must have run as native
//! code, and where a test counter records it, the inline template must have run rather than its
//! helper. Together they exercise every ownership-emitter family's `Traced` arm (clone, shared
//! drop and word release; increments, non-final guards, commits and decrements; the direct-call
//! teardown) and the guards that admit a Map's or Set's `Exotic::Collection` as ordinary. The
//! emitter tests at the end check the `Traced` arms' contracts on assembled code.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

/// A fresh engine at `tier` (threshold 0).
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    assert_eq!(
        engine.interp.tier, tier,
        "gc-heap admits every tier on ARM64"
    );
    engine
}

pub(super) fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Run `f` on the bytecode chunk of the global function `name`.
pub(super) fn with_chunk<R>(
    engine: &mut Engine,
    name: &str,
    f: impl FnOnce(&crate::bytecode::Chunk) -> R,
) -> R {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let object = value.as_obj().expect("a function object").borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    let chunk = user
        .func
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} stayed interpreted"));
    f(chunk)
}

/// Every named global function ran as native code.
pub(super) fn assert_native(engine: &mut Engine, names: &[&str]) {
    for name in names {
        assert!(
            with_chunk(engine, name, |chunk| chunk.jit.get().flatten().is_some()),
            "{name} has no native code"
        );
    }
}

/// Let callers' direct sequences admit `name` (its one-shot inline recompile is settled; see
/// `CallIc::direct`).
pub(super) fn settle(engine: &mut Engine, name: &str) {
    with_chunk(engine, name, |chunk| {
        chunk.inline_attempted.set(true);
        chunk.inline_retry_at.set(0);
    });
}

/// Evaluate `source` at both tiers: both must produce `expected`, and every function in
/// `natives` must have been compiled at `Tier::Jit`. Returns the JIT engine.
fn agree(source: &str, expected: &str, natives: &[&str]) -> Engine {
    let mut bytecode = engine(Tier::Bytecode);
    assert_eq!(eval(&mut bytecode, source), expected, "bytecode");
    let mut jit = engine(Tier::Jit);
    assert_eq!(eval(&mut jit, source), expected, "jit");
    assert_native(&mut jit, natives);
    jit
}

/// The property-load paths (native data hits, native absence hits, checked fallbacks; see
/// `regions::property_paths`) `run` takes on this thread.
fn property_paths(run: impl FnOnce()) -> [u64; 3] {
    let before = super::regions::property_paths();
    run();
    let after = super::regions::property_paths();
    std::array::from_fn(|k| after[k] - before[k])
}

/// An engine running generated code collects: a due safepoint collection runs inside JIT
/// code, explicit requests run, and garbage dies.
#[test]
fn admission_collects() {
    let mut jit = engine(Tier::Jit);
    // The victim's handle lives only in this helper frame (gc-heap scans the stack).
    let weak = crate::gc_test::in_frame(|| {
        eval(&mut jit, "var victim = {v: 1}; 0;");
        let env = jit.interp.global_env.clone();
        let victim = jit.interp.get_var("victim", &env).ok().expect("victim");
        crate::value::Gc::downgrade(victim.as_obj().expect("an object"))
    });
    let stats = |engine: &Engine| engine.interp.gc_heap.core.borrow().stats();
    let before = stats(&jit);
    // Every safepoint now finds a collection due by the live-object threshold.
    jit.interp.gc_next = 0;
    assert_eq!(
        eval(
            &mut jit,
            "function fill(n) { var junk = []; for (var i = 0; i < n; i++) junk.push({i: i}); \
             return junk.length; } victim = null; fill(1000);",
        ),
        "1000"
    );
    assert_native(&mut jit, &["fill"]);
    let during = stats(&jit);
    assert!(
        during.collections > before.collections,
        "a due safepoint collection ran"
    );
    jit.interp.gc_collect();
    assert!(
        stats(&jit).collections > during.collections,
        "an explicit request ran"
    );
    assert!(!crate::gc_test::alive(&weak), "garbage was collected");
}

/// Operand-stack and slot words of objects and strings: LoadLocal and Dup clone, StoreLocal
/// releases the slot's old word, Pop, Void and block-entry Tdz release words.
#[test]
fn execution_words_clone_drop_and_release() {
    agree(
        r#"
        function words(n) {
            var o = {v: 1}, s = 'abc' + n, t, u, k, last;
            for (var i = 0; i < n; i++) {
                t = o;
                u = s;
                t = u;
                o = {v: i};
                s = s + i;
                k = (t = o);
                last = (u = s);
                void o;
                ({w: i});
                'drop' + i;
                for (let j = 0; j < 2; j++) { let inner = {j: j}; t = inner; }
            }
            return typeof t + ':' + t.j + ':' + o.v + ':' + s.length + ':' + k.v + ':' + last.length;
        }
        words(40);
        "#,
        "object:1:39:75:39:75",
        &["words"],
    );
}

/// Named property loads, stores, updates and creations with operand-stack, `this` and slot
/// receivers, including a store whose old value is the receiver itself and a receiver whose
/// last owner is the operand.
#[test]
fn property_receivers_load_store_update_and_create() {
    let source = r#"
        function Point(x, y) { this.x = x; this.y = y; this.tag = 'p' + x; }
        Point.prototype.sum = function () { return this.x + this.y; };
        Point.prototype.label = function () { return this.tag; };
        function holder(v) { return {inner: {v: v, s: 'str'}}; }
        function props(n) {
            var acc = 0, text = '', q = {b: 'b', c: null, n: 0};
            for (var i = 0; i < n; i++) {
                var p = new Point(i, 2);
                acc += p.x + p.sum();
                p.x = i * 2;
                p.y++;
                q.b = 'b' + i;
                q.c = p;
                q.n += 1;
                acc += q.c.x + holder(i).inner.v;
                text = p.label() + q.b;
                acc += (q.c.z = q.c).y;
                q.c.z = 'z' + i;
            }
            return acc + ':' + text + ':' + q.n + ':' + q.c.y + ':' + q.c.z;
        }
        props(40);
    "#;
    let mut jit = agree(
        source,
        "4100:p39b39:40:3:z39",
        &["props", "Point", "holder"],
    );
    let paths = property_paths(|| {
        assert_eq!(eval(&mut jit, "props(40);"), "4100:p39b39:40:3:z39");
    });
    assert!(paths[0] >= 40 * 8, "native property hits: {paths:?}");
    assert_eq!(paths[2], 0, "checked property loads once warm: {paths:?}");
}

/// Computed method reads with a string key (`o[name](x)`) and dense element methods with a
/// Number key (`table[i](x)`).
#[test]
fn computed_and_dense_element_methods() {
    agree(
        r#"
        function computed(n) {
            var o = {f: function (x) { return x + 1; }, g: function (x) { return x * 2; }};
            var names = ['f', 'g'];
            var table = [function (x) { return x - 1; }, function (x) { return x + 10; }];
            var acc = 0;
            for (var i = 0; i < n; i++) {
                acc += o[names[i & 1]](i);
                acc += table[i & 1](i);
            }
            return acc;
        }
        computed(40);
        "#,
        "2160",
        &["computed"],
    );
}

/// Inline Math intrinsics (with `Math` and with Undefined as `this`), String#charCodeAt and its
/// helpers, isNaN and Number.isNaN, Math.sqrt.
#[test]
fn math_and_string_intrinsics() {
    agree(
        r#"
        function intrinsics(n) {
            var s = 'hello world', acc = 0, floor = Math.floor, text = '';
            for (var i = 0; i < n; i++) {
                acc += Math.floor(i / 3) + Math.abs(-i) + Math.min(i, 5) + Math.max(i, 7);
                acc += Math.trunc(i / 4) + Math.imul(i, 3) + Math.clz32(i) + Math.fround(i);
                acc += floor(i / 2);
                acc += s.charCodeAt(i % s.length) + s.codePointAt(i % 3);
                text = s.charAt(i % s.length) + String.fromCharCode(65 + (i % 26));
                acc += isNaN(i) ? 1 : 0;
                acc += Number.isNaN(i / 0 - i / 0) ? 2 : 0;
                acc += Math.sqrt(i * i);
            }
            return acc + ':' + text;
        }
        intrinsics(40);
        "#,
        "15867:wN",
        &["intrinsics"],
    );
}

/// `==`/`===` on strings and on aliased and distinct objects, nullish and mixed-tag compares,
/// `!`, instanceof (object and primitive left operands), string `length`, `typeof`, and string
/// constants.
#[test]
fn equality_instanceof_typeof_and_string_length() {
    agree(
        r#"
        function compare(n) {
            var a = {k: 1}, b = a, c = {k: 1};
            var box = {x: a, y: a, z: c, s: 'xy', t: 'x', u: 'xyz'};
            var acc = 0, tag = '';
            for (var i = 0; i < n; i++) {
                var t2 = box.t + 'y';
                if (box.x === box.y) acc += 1;
                if (box.x == box.y) acc += 1;
                if (box.x !== box.z) acc += 2;
                if (box.s === t2) acc += 4;
                if (box.s == box.u) acc -= 100;
                if (box.s === box.s) acc += 8;
                if (box.x === null) acc -= 100;
                if (box.s == null) acc -= 100;
                if (box.x == undefined) acc -= 100;
                if (a === b) acc += 16;
                if (a instanceof Object) acc += 32;
                if (!(a instanceof Array)) acc += 64;
                if (box.s instanceof Object) acc -= 100;
                acc += box.s.length + t2.length;
                if (typeof box.x === 'object') acc += 128;
                if (typeof box.s === 'string') acc += 256;
                if (!box.s) acc -= 100;
                if (!box.x) acc -= 100;
                tag = 'k' + (i & 3);
            }
            return acc + ':' + tag;
        }
        compare(40);
        "#,
        "20640:k3",
        &["compare"],
    );
}

/// The direct JIT-to-JIT call sequence and its teardown: arities 0 to 3, missing and surplus
/// arguments (objects and strings among them), a `this` receiver, a callee whose last owner is
/// its operand (a closure made by the call before), and recursion deeper than the frame pool's
/// 256 records. Each kind of call must return through the direct sequence on every iteration.
#[test]
fn direct_calls_and_frame_teardown() {
    let source = r#"
        function zero() { return {z: 1}; }
        function one(a) { return a; }
        function two(a, b) { return b; }
        function three(a, b, c) { return c === undefined ? 'missing' : c; }
        function method(x) { return this.k + x; }
        function make() { return function (x) { return x * 2; }; }
        function depth(n) { return n === 0 ? 0 : 1 + depth(n - 1); }
        var obj = {k: 5, m: method}, made = make();
        function arity0(n) { var r = 0; for (var i = 0; i < n; i++) r += zero().z; return r; }
        function arity1(n) { var r = 0; for (var i = 0; i < n; i++) r += one({v: i}).v; return r; }
        function arity2(n) {
            var r = 0;
            for (var i = 0; i < n; i++) r += two('a', 'b' + i).length;
            return r;
        }
        function missing(n) {
            var r = '';
            for (var i = 0; i < n; i++) r = three(1, 2) + three(1, 2, 'c' + i);
            return r;
        }
        function surplus(n) {
            var r = 0;
            for (var i = 0; i < n; i++) r += one(1, {x: 1}, 'surplus' + i);
            return r;
        }
        function withThis(n) { var r = 0; for (var i = 0; i < n; i++) r += obj.m(i); return r; }
        function lastOwner(n) { var r = 0; for (var i = 0; i < n; i++) r += make()(i); return r; }
        [arity0(3), arity1(3), arity2(3), missing(3), surplus(3), withThis(3), lastOwner(3),
            depth(300)].join();
    "#;
    let mut jit = agree(
        source,
        "3,3,6,missingc2,3,18,6,300",
        &[
            "zero",
            "one",
            "two",
            "three",
            "method",
            "make",
            "made",
            "depth",
            "arity0",
            "arity1",
            "arity2",
            "missing",
            "surplus",
            "withThis",
            "lastOwner",
        ],
    );
    for name in [
        "zero", "one", "two", "three", "method", "make", "made", "depth",
    ] {
        settle(&mut jit, name);
    }
    let direct_returns = || super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get());
    for (call, expected, direct) in [
        ("arity0(40)", "40", 40),
        ("arity1(40)", "780", 40),
        ("arity2(40)", "110", 40),
        ("missing(40)", "missingc39", 80),
        ("surplus(40)", "40", 40),
        ("withThis(40)", "980", 40),
        ("lastOwner(40)", "1560", 80),
    ] {
        // A concurrent test can exhaust the process-wide call-cache epoch, which sends calls
        // through the checked helper until their sites refill: retry until a round is direct.
        let mut returns = 0;
        for _ in 0..100 {
            super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
            assert_eq!(eval(&mut jit, call), expected, "{call}");
            returns = direct_returns();
            if returns >= direct {
                break;
            }
        }
        assert!(returns >= direct, "{call}: {returns} direct returns");
    }
    // Deeper than the frame pool: direct returns until the pool runs out, then the checked call.
    super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    assert_eq!(eval(&mut jit, "depth(300);"), "300");
    assert!(direct_returns() > 0, "depth recursion returns directly");
}

/// Maps and Sets (`Exotic::Collection` under gc-heap) through the ordinary-object guards: named
/// property loads (slot and operand-stack receivers, a Collection as a prototype hop, method
/// loads through Map.prototype and Set.prototype), stores, updates and creations (a Map
/// subclass's constructor), and element reads, writes, updates and `in`.
#[test]
fn maps_and_sets_take_the_ordinary_object_paths() {
    let source = r#"
        function readLabel(c) { return c.label; }
        function readInner(box) { return box.c.label; }
        function viaProto(o) { return o.label; }
        function methods(m, s) { return m.get(1) + (s.has(2) ? 1 : 0); }
        function bump(c) { c.count = c.count + 1; c.hits++; return c.count; }
        function elems(c, i) { c[0] = c[0] + i; c[1]++; return c[0] + c[1] + (0 in c ? 1 : 0); }
        class Tally extends Map {
            constructor() { super(); this.total = 0; this.name = 't'; }
            add(k) { this.set(k, 1); this.total++; return this.total; }
        }
        var m = new Map([[1, 10]]), s = new Set([2]), om = Object.create(m);
        m.label = 'map'; s.label = 'set'; m.count = 0; s.count = 0; m.hits = 0; s.hits = 0;
        m[0] = 1; m[1] = 2; s[0] = 3; s[1] = 4;
        var boxes = [{c: m}, {c: s}];
        function drive(n) {
            var acc = 0, last;
            for (var i = 0; i < n; i++) {
                acc += readLabel(i & 1 ? m : s).length + readInner(boxes[i & 1]).length;
                acc += viaProto(om).length + methods(m, s);
                acc += bump(m) + bump(s);
                acc += elems(m, i) + elems(s, i);
                last = new Tally();
                acc += last.add(i) + last.total + last.size;
            }
            return acc + ':' + m.count + ':' + s.hits + ':' + m[0] + ':' + s[1] + ':' + last.name;
        }
        function labels(n) {
            var r = 0;
            for (var k = 0; k < n; k++) {
                readLabel(k & 1 ? m : s);
                readInner(boxes[k & 1]);
                viaProto(om);
                methods(m, s);
            }
            return r;
        }
        drive(40);
    "#;
    let mut jit = agree(
        source,
        "26000:40:40:781:44:t",
        &[
            "readLabel",
            "readInner",
            "viaProto",
            "methods",
            "bump",
            "elems",
            "drive",
        ],
    );
    eval(&mut jit, "labels(2);");
    let paths = property_paths(|| {
        assert_eq!(eval(&mut jit, "labels(50);"), "0");
    });
    // readLabel, readInner (two loads), viaProto and two method loads per round.
    assert_eq!(paths, [50 * 6, 0, 0], "Collection receivers hit natively");
}

/// The `Traced` arm of every ownership emitter that takes a [`super::JitOwnership`] emits
/// nothing: the count emitters, the execution-word clone, shared drop and releases (whose
/// owner guard is counted-only, so no tag reaches a checked path through them), and the frame
/// teardown.
#[test]
fn traced_ownership_emitters_emit_nothing() {
    use super::JitOwnership::Traced;
    use super::*;
    let mut a = asm::Asm::new();
    let label = a.new_label();
    rc_inc(&mut a, Traced, 1, 2, 1);
    rc_load_count(&mut a, Traced, 1, 2);
    rc_check_nonfinal(&mut a, Traced, 2, 1, label);
    rc_guard_nonfinal(&mut a, Traced, 1, 2, 2, label);
    rc_commit_dec(&mut a, Traced, 1, 2, 1);
    rc_dec_nonfinal(&mut a, Traced, 1, 2);
    rc_dec_nonfinal_scaled(&mut a, Traced, 1, 2);
    rc_release_shared(&mut a, Traced, 1, 2, label);
    rc_release_unless_last(&mut a, Traced, 1, 2, label);
    exec_clone(&mut a, Traced, 0, 2, 3, label);
    exec_drop_shared(&mut a, Traced, 0, 2, 3, label);
    exec_release_word(&mut a, Traced, 20, -8);
    exec_release_word(&mut a, Traced, 22, 16);
    exec_release_loaded(&mut a, Traced, 22, 16);
    frame_surplus_release(&mut a, Traced);
    frame_callee_release(&mut a, Traced, 3);
    frame_this_release(&mut a, Traced, 64);
    frame_slots_release(&mut a, Traced, 64, 72);
    assert!(a.buf.is_empty(), "{} words emitted", a.buf.len());
    assert!(!a.label_referenced(label), "a Traced arm branches");
}

/// The property-word guard is three instructions and sends exactly the words whose tag is above
/// Object (PACK_LAZY_PROTO, a function's deferred `prototype`, and every other property-only or
/// unknown tag) to the slow path, over all 65,536 tags, each with a null and a wild payload:
/// no payload is read. The word is preserved on both paths.
#[test]
fn property_word_guard_routes_exactly_the_tags_above_object() {
    use super::*;
    use crate::value::{PACK_BIGINT, PACK_LAZY_PROTO, PACK_OBJ, PACK_STR, PACK_SYM};
    let mut a = asm::Asm::new();
    let slow = a.new_label();
    property_word_guard(&mut a, 0, 2, slow);
    assert_eq!(a.buf.len(), 3, "property-word guard size");
    a.str_imm(0, 1, 0);
    a.movz(0, 1, 0);
    a.ret();
    a.bind(slow);
    a.str_imm(0, 1, 0);
    a.movz(0, 0, 0);
    a.ret();
    let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
    let code = ExecutableBuffer::from_bytes(&bytes).unwrap();
    let run: unsafe extern "C" fn(u64, *mut u64) -> u64 =
        unsafe { std::mem::transmute(code.as_ptr()) };
    for high in 0..=u16::MAX as u64 {
        let tag = high << 48;
        // A null or wild payload: no word's payload may be dereferenced.
        for payload in [0u64, 0x0000_dead_beef_0008] {
            let raw = tag | payload;
            let mut output = !raw;
            let passed = unsafe { run(raw, &mut output) } == 1;
            assert_eq!(passed, tag <= PACK_OBJ, "{raw:016x}");
            assert_eq!(output, raw, "{raw:016x} preserved");
        }
    }
    // Every execution kind passes, BigInt included; the deferred prototype does not.
    for tag in [PACK_BIGINT, PACK_STR, PACK_SYM, PACK_OBJ] {
        let mut output = 0;
        assert_eq!(unsafe { run(tag | 8, &mut output) }, 1);
    }
    let mut output = 0;
    assert_eq!(unsafe { run(PACK_LAZY_PROTO | 8, &mut output) }, 0);
}
