//! `this` through generated code (P1 S5 step 9d). A JIT context holds the activation's `this`
//! at [48] as the packed word itself. Each test drives every kind of `this` through one family
//! of sites: an object; a primitive into a strict callee (as it is) and into a sloppy one
//! (boxed); undefined and null into a sloppy callee (the global object) and into a strict one
//! (as they are). The sites: LoadThis, GetPropThis, SetPropThis, a numeric chain, the region's
//! own read, the direct call's move of the receiver and a constructor's. `this` arrives through
//! direct calls, the checked call paths (which bind a sloppy callee's `this` on the word),
//! `Function.prototype.call` (a Rust entry) and `new`. Another reads `this` from an environment
//! (an arrow, a derived constructor after `super()`), and the direct-call test also checks a
//! pooled record's `this` (the finish stub's reset), which a receiver-less direct call reads.
//!
//! Programs run at every tier with the same expected results (Node's, run as a classic script);
//! at `Tier::Jit` the functions named ran as native code and, where a test counter records it,
//! the native path ran. The stress child (`jit::rooting_tests::
//! rooting_holds_under_gc_stress`) also runs them at `TRUST_JS_GC_STRESS=1`.
//!
//! The counters of ARM64's direct calls, regions, numeric chains and property templates are
//! checked on aarch64 only. On x86-64, which has none of those, x64's own-data template counts
//! the reads of `this`'s properties it ran (`TEST_OWN_PROPERTY_READS`), the `this` form reading
//! the packed word at [48] as a slot form reads its slot.

use crate::bytecode::Tier;
use crate::value::Value;
use crate::{Completion, Engine};

/// A fresh engine at `tier` that compiles every function at its first call.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Run `f` on the bytecode chunk of the global function `name`.
fn with_chunk<R>(
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
    let runtime = user.func.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} stayed interpreted"));
    f(chunk)
}

/// Every named global function has native code.
fn assert_native(engine: &mut Engine, names: &[&str]) {
    for name in names {
        assert!(
            with_chunk(engine, name, |chunk| chunk.jit.get().flatten().is_some()),
            "{name} has no native code"
        );
    }
}

/// Let callers' direct sequences admit each named function, and keep it from splicing its
/// callees inline (its one-shot inline recompile counts as settled; see `CallIc::direct`), so
/// every site under test stays in its own activation.
fn settle(engine: &mut Engine, names: &[&str]) {
    for name in names {
        with_chunk(engine, name, |chunk| {
            chunk.inline_attempted.set(true);
            chunk.inline_retry_at.set(0);
        });
    }
}

/// How far `counter` advanced while `run` ran.
fn delta<const N: usize>(counter: impl Fn() -> [u64; N], run: impl FnOnce()) -> [u64; N] {
    let before = counter();
    run();
    let after = counter();
    std::array::from_fn(|k| after[k] - before[k])
}

/// Returns through the direct call sequence on this thread.
#[cfg(target_arch = "aarch64")]
fn direct_returns() -> [u64; 1] {
    [super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get()) as u64]
}

/// Native data hits, native absence hits and checked fallbacks of the property template.
#[cfg(target_arch = "aarch64")]
fn property_paths() -> [u64; 3] {
    let paths = super::regions::property_paths();
    [paths[0], paths[1], paths[2]]
}

/// Calls of the property-store helper.
#[cfg(target_arch = "aarch64")]
fn set_prop_helpers() -> [u64; 1] {
    [crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get) as u64]
}

/// Borrowed own reads a loop region ran.
#[cfg(target_arch = "aarch64")]
fn borrowed_reads() -> [u64; 1] {
    [super::regions::borrowed_reads().0]
}

/// Own-data reads x64's property template ran on this thread: a local slot's receiver, and
/// `this`.
#[cfg(target_arch = "x86_64")]
fn own_reads() -> [u64; 2] {
    super::x64::TEST_OWN_PROPERTY_READS.with(|counts| counts.each_ref().map(|c| c.get() as u64))
}

/// Evaluate `source` until it takes at least `direct` direct returns: a site takes the direct
/// sequence once its call cache is filled for compiled code, and a concurrent test can exhaust
/// the process-wide call-cache epoch, which sends calls through the checked helper until their
/// sites refill. Returns the last result.
#[cfg(target_arch = "aarch64")]
fn direct_eval(engine: &mut Engine, source: &str, direct: u64) -> String {
    let mut result = String::new();
    for _ in 0..100 {
        let taken = delta(direct_returns, || result = eval(engine, source));
        if taken[0] >= direct {
            return result;
        }
    }
    panic!("{source}: fewer than {direct} direct returns (last result {result})");
}

/// `kind(v)`: one letter for what `this` was: `o` an object, `g` the global object, `N`/`S`/`B`
/// a boxed primitive, `n`/`s`/`b` a primitive, `u` undefined, `l` null.
const KIND: &str = r#"
    var G = this;
    function kind(v) {
        switch (typeof v) {
            case 'undefined': return 'u';
            case 'number': return 'n';
            case 'string': return 's';
            case 'boolean': return 'b';
            case 'object':
                if (v === null) return 'l';
                if (v === G) return 'g';
                if (v instanceof Number) return 'N';
                if (v instanceof String) return 'S';
                if (v instanceof Boolean) return 'B';
                return 'o';
        }
        return '?';
    }
    function attempt(f, t, v) {
        try { return String(f.call(t, v)); }
        catch (e) { return e instanceof TypeError ? 'T' : 'E'; }
    }
"#;

/// LoadThis pushes the callee's `this` as it was bound: an object; a primitive as it is into a
/// strict callee and boxed into a sloppy one; undefined and null as they are into a strict callee
/// and as the global object into a sloppy one. Method calls, and calls of a free name (which pass
/// an Undefined receiver), reach strict callees and sloppy ones given an object through the direct
/// sequence, which moves the receiver's word; a sloppy callee given anything else takes the
/// checked call, which binds it; `call` enters from Rust.
#[test]
fn load_this_pushes_every_kind_of_this() {
    let source = format!(
        "{KIND}
        function sloppyThis() {{ return this; }}
        function strictThis() {{ 'use strict'; return this; }}
        Number.prototype.sloppyThis = sloppyThis;
        Number.prototype.strictThis = strictThis;
        String.prototype.sloppyThis = sloppyThis;
        String.prototype.strictThis = strictThis;
        var o = {{ sloppyThis: sloppyThis, strictThis: strictThis }};
        function viaMethods(n, s) {{
            return kind(o.sloppyThis()) + kind(o.strictThis()) + kind(n.sloppyThis()) +
                kind(n.strictThis()) + kind(s.sloppyThis()) + kind(s.strictThis());
        }}
        function viaPlain() {{ return kind(sloppyThis()) + kind(strictThis()); }}
        function viaCall(v) {{ return kind(sloppyThis.call(v)) + kind(strictThis.call(v)); }}
        function round() {{
            return [viaMethods(5, 'str'), viaPlain(), viaCall(undefined), viaCall(null),
                viaCall(7), viaCall(o), viaCall(true)].join();
        }}
        round();"
    );
    const EXPECTED: &str = "ooNnSs,gu,gu,gl,Nn,oo,Bb";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, &source), EXPECTED, "{tier:?}");
        if tier == Tier::Jit {
            let names = [
                "kind",
                "sloppyThis",
                "strictThis",
                "viaMethods",
                "viaPlain",
                "viaCall",
                "round",
            ];
            assert_native(&mut engine, &names);
            settle(&mut engine, &names);
            // A round's 30 direct returns: its 7 calls, viaMethods' 4 methods (all but a
            // sloppy callee's primitive receivers) and 6 `kind`s, viaPlain's strict call and 2
            // `kind`s, and viaCall's 10 `kind`s (`call` is not a direct call).
            #[cfg(target_arch = "aarch64")]
            assert_eq!(direct_eval(&mut engine, "round()", 30), EXPECTED);
            // x64 has no direct call sequence: its calls take the checked helper.
            #[cfg(target_arch = "x86_64")]
            assert_eq!(eval(&mut engine, "round()"), EXPECTED);
        }
    }
}

/// GetPropThis and SetPropThis receive every kind of `this`. An object's property, a primitive's
/// through its wrapper prototype (unboxed into a strict callee, boxed into a sloppy one) and the
/// global object's are read; writes land on the object, on a sloppy callee's wrapper or the
/// global object, and throw a TypeError on a strict callee's primitive or undefined `this`.
/// `length` is read too, which the template probes for an object receiver only. Natively:
/// object and primitive receivers of the strict read of `x` are data hits of the template (the
/// primitive through its realm's wrapper prototype); for `length` an object is a data hit and a
/// primitive takes the checked read; the object store calls no helper, while a primitive store
/// reaches the helper, which throws.
#[test]
fn property_reads_and_writes_receive_every_kind_of_this() {
    let source = format!(
        "{KIND}
        var x = 'gx', y = 0;
        Number.prototype.x = 'Nx';
        String.prototype.x = 'Sx';
        Boolean.prototype.x = 'Bx';
        function sloppyRead() {{ return this.x; }}
        function strictRead() {{ 'use strict'; return this.x; }}
        function sloppyWrite(v) {{ this.y = v; return this.y; }}
        function strictWrite(v) {{ 'use strict'; this.y = v; return this.y; }}
        function sloppyLength() {{ return this.length; }}
        function strictLength() {{ 'use strict'; return this.length; }}
        var methods = {{ sloppyRead: sloppyRead, strictRead: strictRead,
            sloppyWrite: sloppyWrite, strictWrite: strictWrite, sloppyLength: sloppyLength,
            strictLength: strictLength }};
        for (var k in methods) {{
            Number.prototype[k] = methods[k];
            String.prototype[k] = methods[k];
            Boolean.prototype[k] = methods[k];
        }}
        var o = {{ x: 'ox', y: 0, length: 'ol', sloppyRead: sloppyRead, strictRead: strictRead,
            sloppyWrite: sloppyWrite, strictWrite: strictWrite, sloppyLength: sloppyLength,
            strictLength: strictLength }};
        function reads(n, s, b) {{
            return [o.sloppyRead(), o.strictRead(), n.sloppyRead(), n.strictRead(), s.sloppyRead(),
                s.strictRead(), b.sloppyRead(), b.strictRead(), sloppyRead()].join('|');
        }}
        function writes(k, n) {{
            var r = [o.sloppyWrite(k), o.strictWrite(k + 1), n.sloppyWrite(k)];
            try {{ r.push(n.strictWrite(k)); }}
            catch (e) {{ r.push(e instanceof TypeError ? 'T' : 'E'); }}
            try {{ r.push(strictWrite(k)); }}
            catch (e) {{ r.push(e instanceof TypeError ? 'T' : 'E'); }}
            r.push(sloppyWrite(k + 2), y);
            return r.join('|');
        }}
        function lengths(n, s) {{
            return [o.sloppyLength(), o.strictLength(), String(n.sloppyLength()),
                String(n.strictLength()), s.sloppyLength(), s.strictLength(),
                attempt(sloppyLength, undefined), attempt(strictLength, undefined)].join('|');
        }}
        function viaCall(t) {{
            return [attempt(sloppyRead, t), attempt(strictRead, t), attempt(sloppyWrite, t, 6),
                attempt(strictWrite, t, 7)].join('|');
        }}
        function round() {{
            return [reads(5, 'str', true), writes(3, 5), lengths(5, 'str'), viaCall(undefined),
                viaCall(null), viaCall(7), viaCall('s'), viaCall(o)].join(',');
        }}
        function readObject(n) {{ var s = ''; for (var i = 0; i < n; i++) s = o.strictRead(); return s; }}
        function readNumber(n) {{ var s = ''; for (var i = 0; i < n; i++) s = (5).strictRead(); return s; }}
        function lengthObject(n) {{ var s = ''; for (var i = 0; i < n; i++) s = o.strictLength(); return s; }}
        function lengthNumber(n) {{ var s = ''; for (var i = 0; i < n; i++) s = (5).strictLength(); return s; }}
        function writeObject(n) {{ for (var i = 0; i < n; i++) o.strictWrite(i); return o.y; }}
        function writeNumber(n) {{
            var thrown = 0;
            for (var i = 0; i < n; i++) try {{ (5).strictWrite(i); }} catch (e) {{ thrown++; }}
            return thrown;
        }}
        round();"
    );
    const EXPECTED: &str = "ox|ox|Nx|Nx|Sx|Sx|Bx|Bx|gx,3|4|3|T|T|5|5,\
                            ol|ol|undefined|undefined|3|3|undefined|T,gx|T|6|T,gx|T|6|T,\
                            Nx|Nx|6|T,Sx|Sx|6|T,ox|ox|6|7";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, &source), EXPECTED, "{tier:?}");
        for _ in 0..3 {
            assert_eq!(eval(&mut engine, "round()"), EXPECTED, "{tier:?}");
        }
        for (call, result) in [
            ("readObject(10)", "ox"),
            ("readNumber(10)", "Nx"),
            ("lengthObject(10)", "ol"),
            ("lengthNumber(10)", "undefined"),
            ("writeObject(10)", "9"),
            ("writeNumber(10)", "10"),
        ] {
            assert_eq!(eval(&mut engine, call), result, "{tier:?} {call}");
        }
        if tier == Tier::Jit {
            let names = [
                "sloppyRead",
                "strictRead",
                "sloppyWrite",
                "strictWrite",
                "reads",
                "writes",
                "viaCall",
                "round",
                "sloppyLength",
                "strictLength",
                "lengths",
                "readObject",
                "readNumber",
                "lengthObject",
                "lengthNumber",
                "writeObject",
                "writeNumber",
            ];
            assert_native(&mut engine, &names);
            settle(&mut engine, &names);
            #[cfg(target_arch = "aarch64")]
            {
                // A round's 41 direct returns: its 8 calls; reads' 5, writes' 2 and lengths' 4
                // methods that bind `this` without boxing and return normally, and lengths' 2
                // and viaCall's 20 `attempt`s (whose `call` is not a direct call).
                assert_eq!(direct_eval(&mut engine, "round()", 41), EXPECTED);
                // Each iteration's method load is a data hit, and so is its callee's read of
                // `this.x`, or of an object's `this.length`; a primitive's `length` is checked.
                for (call, expected) in [
                    ("readObject(20)", [40, 0, 0]),
                    ("readNumber(20)", [40, 0, 0]),
                    ("lengthObject(20)", [40, 0, 0]),
                    ("lengthNumber(20)", [20, 0, 20]),
                ] {
                    let paths = delta(property_paths, || {
                        eval(&mut engine, call);
                    });
                    assert_eq!(paths, expected, "{call}: property paths");
                }
                let helpers = delta(set_prop_helpers, || {
                    assert_eq!(eval(&mut engine, "writeObject(20)"), "19");
                });
                assert_eq!(helpers, [0], "an object `this` is stored natively");
                let helpers = delta(set_prop_helpers, || {
                    assert_eq!(eval(&mut engine, "writeNumber(20)"), "20");
                });
                assert_eq!(
                    helpers,
                    [20],
                    "a primitive `this` takes the throwing helper"
                );
            }
            // x64 reads an object `this`'s own `x` or `length` (its spilled named fields) in its
            // own-data template; a primitive `this`, whose `x` is its wrapper prototype's, and
            // the method loads take the checked helper.
            #[cfg(target_arch = "x86_64")]
            for (call, expected) in [
                ("readObject(20)", [0, 20]),
                ("readNumber(20)", [0, 0]),
                ("lengthObject(20)", [0, 20]),
                ("lengthNumber(20)", [0, 0]),
            ] {
                let reads = delta(own_reads, || {
                    eval(&mut engine, call);
                });
                assert_eq!(
                    reads, expected,
                    "{call}: own-data reads of a slot, of `this`"
                );
            }
        }
    }
}

/// A numeric chain whose operands are `this`'s properties (feedback warmed at `Tier::Bytecode`,
/// so the chain compiles): `this.state = this.state | FLAG` and `this.a < this.b`. An object
/// `this` runs the chain natively, with no property template or checked read; every other kind
/// leaves the chain at its receiver guard and replays the operations through the checked
/// helpers: a primitive reads its wrapper prototype, a strict callee's primitive or undefined
/// `this` throws on the store or the read.
#[test]
fn numeric_chains_receive_every_kind_of_this() {
    let source = format!(
        "{KIND}
        var FLAG = 2;
        function sloppyMark() {{ this.state = this.state | FLAG; return this.state; }}
        function strictMark() {{ 'use strict'; this.state = this.state | FLAG; return this.state; }}
        function sloppyOrder() {{ if (this.a < this.b) return 'lt'; return 'ge'; }}
        function strictOrder() {{ 'use strict'; if (this.a < this.b) return 'lt'; return 'ge'; }}
        Number.prototype.a = 3;
        Number.prototype.b = 1;
        Number.prototype.sloppyMark = sloppyMark;
        Number.prototype.strictMark = strictMark;
        var o = {{ state: 1, a: 1, b: 2, sloppyMark: sloppyMark, strictMark: strictMark,
            sloppyOrder: sloppyOrder, strictOrder: strictOrder }};
        function viaMethods(n) {{
            o.state = 1;
            var r = [o.sloppyMark()];
            o.state = 4;
            r.push(o.strictMark(), o.sloppyOrder(), o.strictOrder(), n.sloppyMark());
            try {{ r.push(n.strictMark()); }}
            catch (e) {{ r.push(e instanceof TypeError ? 'T' : 'E'); }}
            return r.join('|');
        }}
        function viaCall(t) {{
            return [attempt(sloppyMark, t), attempt(strictMark, t), attempt(sloppyOrder, t),
                attempt(strictOrder, t)].join('|');
        }}
        function round() {{
            return [viaMethods(5), viaCall(undefined), viaCall(null), viaCall(7),
                viaCall(o)].join(',');
        }}
        function lt() {{ return 'lt'; }}
        o.lt = lt;
        function orders(n) {{
            var r = '';
            for (var i = 0; i < n; i++) r = o.sloppyOrder() + o.strictOrder();
            return r;
        }}
        function controls(n) {{
            var r = '';
            for (var i = 0; i < n; i++) r = o.lt() + o.lt();
            return r;
        }}
        for (var w = 0; w < 40; w++) {{
            o.state = 1; o.sloppyMark(); o.strictMark(); o.sloppyOrder(); o.strictOrder();
        }}
        0;"
    );
    const EXPECTED: &str = "3|6|lt|lt|2|T,2|T|ge|T,2|T|ge|T,2|T|ge|ge,6|6|lt|lt";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        // The chains need warm property feedback when their functions compile.
        let mut engine = engine(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        eval(&mut engine, &source);
        engine.set_tier(tier);
        #[cfg(target_arch = "aarch64")]
        let chains = super::TEST_THIS_CHAIN_OPS.with(std::cell::Cell::get);
        for _ in 0..3 {
            assert_eq!(eval(&mut engine, "round()"), EXPECTED, "{tier:?}");
        }
        assert_eq!(
            eval(&mut engine, "orders(5) + controls(5)"),
            "ltltltlt",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let names = [
                "sloppyMark",
                "strictMark",
                "sloppyOrder",
                "strictOrder",
                "lt",
                "orders",
                "controls",
            ];
            assert_native(&mut engine, &names);
            #[cfg(target_arch = "aarch64")]
            {
                // The four chains' operations on `this`: two reads each, and the marks' stores.
                let compiled = super::TEST_THIS_CHAIN_OPS.with(std::cell::Cell::get) - chains;
                assert!(
                    compiled >= 10,
                    "{compiled} chain operations on `this` compiled"
                );
            }
            settle(&mut engine, &names);
            // `orders` and `controls` load their methods alike; the chains' four reads of `this`
            // per iteration add no property template path (no data hit, absence or fallback).
            #[cfg(target_arch = "aarch64")]
            {
                let paths = delta(property_paths, || {
                    assert_eq!(eval(&mut engine, "orders(20)"), "ltlt");
                });
                let control = delta(property_paths, || {
                    assert_eq!(eval(&mut engine, "controls(20)"), "ltlt");
                });
                assert_eq!(
                    paths, control,
                    "the chains read `this` without the template"
                );
            }
            // x64 has no numeric chains: its own-data template reads an object `this`'s `a` and
            // `b`, in each of an iteration's two callees; `controls`' callees read none.
            #[cfg(target_arch = "x86_64")]
            for (call, expected) in [("orders(20)", [0, 80]), ("controls(20)", [0, 0])] {
                let reads = delta(own_reads, || {
                    assert_eq!(eval(&mut engine, call), "ltlt");
                });
                assert_eq!(
                    reads, expected,
                    "{call}: own-data reads of a slot, of `this`"
                );
            }
        }
    }
}

/// A loop region's borrowed own read of `this.tag`, compiled with an object's feedback (warmed
/// at `Tier::Bytecode`): that object `this` is read in the region on every iteration, while a
/// primitive `this` fails the read's receiver guard and leaves the region there, its checked read
/// finding the wrapper prototype's `tag`. Then every kind of `this` (which leaves the site's cache
/// polymorphic) gets its own result: a primitive's or the global object's `tag`, or a TypeError
/// for a strict callee's undefined or null `this`. Each scan counts how often the value it reads
/// changes: once, from its first read.
#[test]
fn region_own_reads_receive_every_kind_of_this() {
    let source = format!(
        "{KIND}
        var tag = 'g';
        Number.prototype.tag = 'N';
        function sloppyScan(n) {{
            var p = null, c = 0;
            for (var i = 0; i < n; i++) {{ var q = this.tag; if (q !== p) c++; p = q; }}
            return c + p;
        }}
        function strictScan(n) {{
            'use strict';
            var p = null, c = 0;
            for (var i = 0; i < n; i++) {{ var q = this.tag; if (q !== p) c++; p = q; }}
            return c + p;
        }}
        Number.prototype.sloppyScan = sloppyScan;
        Number.prototype.strictScan = strictScan;
        var o = {{ tag: 'o', sloppyScan: sloppyScan, strictScan: strictScan }};
        function round(n) {{
            var r = [o.sloppyScan(n), o.strictScan(n), (5).sloppyScan(n), (5).strictScan(n),
                sloppyScan(n)];
            try {{ r.push(strictScan(n)); }}
            catch (e) {{ r.push(e instanceof TypeError ? 'T' : 'E'); }}
            for (var t of [undefined, null, 7, o]) {{
                r.push(attempt(sloppyScan, t, n), attempt(strictScan, t, n));
            }}
            return r.join('|');
        }}
        for (var w = 0; w < 8; w++) {{ o.sloppyScan(5); o.strictScan(5); }}
        0;"
    );
    const EXPECTED: &str = "1o|1o|1N|1N|1g|T|1g|T|1g|T|1N|1N|1o|1o";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        // The region's borrowed read needs warm feedback when the function compiles.
        let mut engine = engine(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        eval(&mut engine, &source);
        engine.set_tier(tier);
        // ARM64: the loop region's borrowed reads. x64, which has no regions: its own-data
        // template's reads of `this`.
        #[cfg(target_arch = "aarch64")]
        let native_reads = borrowed_reads;
        #[cfg(target_arch = "x86_64")]
        let native_reads = || [own_reads()[1]];
        let mut reads = Vec::new();
        for (call, result) in [
            ("o.sloppyScan(20)", "1o"),
            ("o.strictScan(20)", "1o"),
            ("(5).strictScan(20)", "1N"),
        ] {
            reads.push(
                delta(native_reads, || {
                    assert_eq!(eval(&mut engine, call), result, "{tier:?} {call}");
                })[0],
            );
        }
        for _ in 0..3 {
            assert_eq!(eval(&mut engine, "round(10)"), EXPECTED, "{tier:?}");
        }
        if tier == Tier::Jit {
            assert_native(&mut engine, &["sloppyScan", "strictScan"]);
            assert_eq!(
                reads,
                [20, 20, 0],
                "an object's reads are native, a primitive's are not"
            );
        }
    }
}

/// The direct call moves the receiver's word into the callee's record whatever its kind, and a
/// receiver-less direct call (a `Call` through a local binding; a call of a free name passes an
/// Undefined receiver) leaves the pooled record's `this` as it is: so the finish stub must reset
/// a finished callee's `this` to Undefined on every path. A strict callee sees each receiver as
/// it is and a sloppy one an object (a primitive receiver takes the checked call, which boxes
/// it); a callee that ignores `this` is called directly with every kind of receiver; and a
/// receiver-less strict callee called right after any of these, on the record they returned to
/// the pool, sees undefined. Afterwards every pooled record holds Undefined.
#[test]
fn direct_calls_move_every_kind_of_receiver_and_reset_this() {
    let source = format!(
        "{KIND}
        function ignore(k) {{ return k + 1; }}
        function strictThis() {{ 'use strict'; return this; }}
        function sloppyThis() {{ return this; }}
        Number.prototype.ignore = ignore;
        Number.prototype.strictThis = strictThis;
        Number.prototype.sloppyThis = sloppyThis;
        String.prototype.ignore = ignore;
        var o = {{ ignore: ignore, strictThis: strictThis, sloppyThis: sloppyThis }};
        function moved(n) {{
            return kind(o.strictThis()) + kind(n.strictThis()) + kind(o.sloppyThis()) +
                kind(n.sloppyThis()) + o.ignore(1) + n.ignore(2);
        }}
        function afterReceiver(n, s) {{
            var st = strictThis;
            var a = o.ignore(3), b = kind(st()), c = n.ignore(4), d = kind(st());
            var e = kind(o.strictThis()), f = kind(st()), g = s.ignore(5);
            return a + b + c + d + e + f + g + kind(st()) + kind(strictThis());
        }}
        function round() {{ return moved(5) + ',' + afterReceiver(6, 'str'); }}
        round();"
    );
    const EXPECTED: &str = "onoN23,4u5uou6uu";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, &source), EXPECTED, "{tier:?}");
        if tier == Tier::Jit {
            let names = [
                "kind",
                "ignore",
                "strictThis",
                "sloppyThis",
                "moved",
                "afterReceiver",
                "round",
            ];
            assert_native(&mut engine, &names);
            settle(&mut engine, &names);
            // A round's 26 direct returns: its 2 calls; moved's 5 methods (all but the sloppy
            // callee's primitive receiver) and 4 `kind`s; afterReceiver's 4 methods, its 4
            // receiver-less `st()` and 1 `strictThis()`, and their 5 `kind`s.
            #[cfg(target_arch = "aarch64")]
            assert_eq!(direct_eval(&mut engine, "round()", 26), EXPECTED);
            // x64 has no direct call sequence: its calls take the checked helper, whose native
            // callees run in pooled records too.
            #[cfg(target_arch = "x86_64")]
            assert_eq!(eval(&mut engine, "round()"), EXPECTED);
            assert!(!engine.interp.frame_pool.is_empty());
            for record in engine.interp.frame_pool.iter() {
                // SAFETY: a pooled record is allocated and runs nothing.
                let this = unsafe { record.as_ref().ctx.this_value() };
                assert!(
                    matches!(this, Value::Undefined),
                    "a pooled record holds Undefined"
                );
            }
        }
    }
}

/// A constructor's `this` is its new object, which the construct paths pass to the moved entries
/// (an activation-free body, one with an environment for a closure or an arrow reading `this`);
/// `Reflect.construct` and `call` on a constructor function enter from Rust, and a sloppy
/// constructor called without `new` sees the global object, or its boxed receiver.
#[test]
fn constructors_receive_their_new_object() {
    let source = format!(
        "{KIND}
        function Point(x, y) {{ this.x = x; this.y = y; this.sum = this.x + this.y; }}
        function Tagged(k) {{ 'use strict'; this.k = k; this.self = this; }}
        function Captured(k) {{ this.k = k; var self = this; this.get = function () {{ return self.k; }}; }}
        function Arrowed(k) {{ this.k = k; this.get = () => this.k; }}
        function Replaced(k) {{ this.k = k; return {{ k: -k }}; }}
        function Plain(k) {{ return kind(this); }}
        function make(k) {{
            var p = new Point(k, 1), t = new Tagged(k), c = new Captured(k), a = new Arrowed(k);
            var r = new Replaced(k), q = Reflect.construct(Point, [k, 2]), u = {{}};
            Tagged.call(u, k);
            return [p.sum, t.self === t, t.k, c.get(), a.get(), ({{ get: a.get }}).get(), r.k,
                q.sum, Plain(k), Plain.call(5, k), u.k].join('|');
        }}
        make(3);"
    );
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            eval(&mut engine, &source),
            "4|true|3|3|3|3|-3|5|g|N|3",
            "{tier:?}"
        );
        for k in 0..10 {
            assert_eq!(
                eval(&mut engine, &format!("make({k})")),
                format!("{}|true|{k}|{k}|{k}|{k}|{}|{}|g|N|{k}", k + 1, -k, k + 2),
                "{tier:?}"
            );
        }
        if tier == Tier::Jit {
            assert_native(
                &mut engine,
                &[
                    "Point", "Tagged", "Captured", "Arrowed", "Replaced", "Plain", "make",
                ],
            );
        }
    }
}

/// An arrow reads its lexical `this` from its environment, and a derived constructor reads the
/// `this` that `super()` bound there: neither sees the `this` its own call passed (an arrow
/// called as a method or through `call`, a derived constructor whose parent returned another
/// object). A derived constructor's `this` before `super()` throws a ReferenceError.
#[test]
fn environment_this_is_not_the_passed_word() {
    let source = format!(
        "{KIND}
        function arrows() {{
            var f = () => this, g = () => this.x;
            var holder = {{ f: f, g: g, x: 'holder' }};
            return [f() === this, holder.f() === this, f.call(holder) === this, g(), holder.g(),
                g.call(holder)].join('|');
        }}
        function attemptK(f) {{ try {{ return f(); }} catch (e) {{ return e.constructor.name; }} }}
        class Base {{ constructor(k) {{ this.k = k; }} }}
        class Derived extends Base {{ constructor(k) {{ super(k); this.twice = this.k * 2; }} }}
        var other = {{ tag: 'other' }};
        class Swap {{ constructor() {{ return other; }} }}
        class SwapDerived extends Swap {{
            constructor() {{ super(); this.seen = (this === other); this.read = this.tag; }}
        }}
        class Early extends Base {{ constructor() {{ var r = attemptK(() => this.k); super(1); this.r = r; }} }}
        function derived(k) {{
            var d = new Derived(k), s = new SwapDerived(), e = new Early();
            return [d.twice, s === other, other.seen, other.read, e.r, e.k].join('|');
        }}
        var o = {{ x: 'o', arrows: arrows }};
        function round(k) {{ return [o.arrows(), arrows.call({{ x: 'call' }}), derived(k)].join(','); }}
        round(3);"
    );
    const EXPECTED: &str = "true|true|true|o|o|o,true|true|true|call|call|call,\
                            6|true|true|other|ReferenceError|1";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, &source), EXPECTED, "{tier:?}");
        for _ in 0..10 {
            assert_eq!(eval(&mut engine, "round(3)"), EXPECTED, "{tier:?}");
        }
        if tier == Tier::Jit {
            assert_native(&mut engine, &["arrows", "derived", "round"]);
        }
    }
}
