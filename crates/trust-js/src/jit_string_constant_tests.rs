//! String constants through generated code (P1 S5 step 9e). The string-constant templates, the
//! baseline's and the region copy's, push the constant's word (`PACK_STR | cell`) as an
//! immediate that owns nothing: the chunk's constant pool roots the cell for as long as code
//! compiled from the chunk can run.
//!
//! Programs run at every tier with the same expected results; at `Tier::Jit` the functions
//! named ran as native code. From Rust each test checks the strings that literals evaluate to
//! against the constants of their chunks: at `Tier::Bytecode` and `Tier::Jit` a literal
//! evaluates to its constant itself, the same string each time (the tree-walker makes a string
//! at each evaluation), and a constant keeps its text across collections. A string a test checks
//! is held only in a frame of its own, so a constant stays alive only if its chunk roots it.
//! The stress child (`jit::rooting_tests::rooting_holds_under_gc_stress`) also runs
//! them at `TRUST_JS_GC_STRESS=1`, which collects at every safepoint; debug builds poison a freed
//! cell, so a constant freed while code embeds it no longer reads as its text.
//!
//! x64 has no regions (that check is ARM64's); its baseline template moves the word into rax as
//! an immediate, which its tests find in the native code of the constant's chunk.

use crate::bytecode::{Chunk, Op, Tier};
use crate::lstr::LStr;
use crate::value::Value;
use crate::value::ValueView;
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

fn global(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env.clone();
    engine.interp.get_var(name, &env).ok().expect("bound name")
}

/// Run `f` on the global function `name`'s first-stage chunk, or with `spliced` on its second
/// stage, the body recompiled with hot callees inlined.
fn with_chunk<R>(engine: &mut Engine, name: &str, spliced: bool, f: impl FnOnce(&Chunk) -> R) -> R {
    let value = global(engine, name);
    let object_handle = value.as_obj().expect("a function object");
    let object = object_handle.borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    let stage = if spliced {
        &user.func.runtime().code2
    } else {
        &user.func.runtime().code
    };
    let chunk = stage
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} has no chunk (second stage: {spliced})"));
    f(chunk)
}

/// Whether the global function `name` has its second-stage chunk.
fn second_stage(engine: &mut Engine, name: &str) -> bool {
    let value = global(engine, name);
    let object_handle = value.as_obj().expect("a function object");
    let object = object_handle.borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    user.func
        .runtime()
        .code2
        .get()
        .and_then(Option::as_ref)
        .is_some()
}

/// Every named global function has native code.
fn assert_native(engine: &mut Engine, names: &[&str]) {
    for name in names {
        assert!(
            with_chunk(engine, name, false, |chunk| chunk
                .jit
                .get()
                .flatten()
                .is_some()),
            "{name} has no native code"
        );
    }
}

/// Keep each named function on its first-stage chunk: its one-shot inline recompile counts as
/// settled, so no second stage with constants of its own replaces the chunk under test.
fn settle(engine: &mut Engine, names: &[&str]) {
    for name in names {
        with_chunk(engine, name, false, |chunk| {
            chunk.inline_attempted.set(true);
            chunk.inline_retry_at.set(0);
        });
    }
}

/// The string constants a chunk's literals push, in operation order: the pool does not
/// deduplicate, so each literal is a constant of its own.
fn constants(chunk: &Chunk) -> Vec<LStr> {
    chunk
        .jit_ops()
        .iter()
        .filter_map(|op| match *op {
            Op::Const(k) if chunk.jit_const_is_str(k) => {
                // SAFETY: the slot lives as long as the chunk, which the caller lends.
                match &(unsafe { &*chunk.jit_const_ptr(k) }).view() {
                    ValueView::Str(string) => Some(string.clone()),
                    _ => unreachable!("a string constant"),
                }
            }
            _ => None,
        })
        .collect()
}

/// Whether the native code of `name`'s first-stage chunk holds the word of its `index`th string
/// constant as an immediate: x64's string-constant template moves it into rax with `movabs`
/// (`48 b8` and the word's eight bytes).
#[cfg(target_arch = "x86_64")]
fn holds_immediate(engine: &mut Engine, name: &str, index: usize) -> bool {
    with_chunk(engine, name, false, |chunk| {
        let k = chunk
            .jit_ops()
            .iter()
            .filter_map(|op| match *op {
                Op::Const(k) if chunk.jit_const_is_str(k) => Some(k),
                _ => None,
            })
            .nth(index)
            .expect("the constant's index");
        let word = chunk
            .jit_const_str_word(k)
            .expect("a string constant's word");
        let code = chunk.jit.get().flatten().expect("native code");
        // SAFETY: the chunk keeps its code's executable buffer alive while it is lent.
        let bytes = unsafe { std::slice::from_raw_parts(code.mem, code.len) };
        let mut movabs = vec![0x48, 0xb8];
        movabs.extend_from_slice(&word.to_le_bytes());
        bytes.windows(movabs.len()).any(|window| window == movabs)
    })
}

/// Evaluate `call` and check from Rust that it returned a string reading `text`; at the tiers
/// that run chunks, also that the string is the `index`th string constant of `name`'s chunk
/// (`spliced`: its second stage) and that the constant's immediate is the word
/// the bytecode VM pushes for it. Returns the string's address complemented, in a frame of its
/// own: the caller holds no reference to the string (the conservative scan would take its plain
/// address for one).
#[inline(never)]
fn returned(
    engine: &mut Engine,
    tier: Tier,
    call: &str,
    (name, spliced, index): (&str, bool, usize),
    text: &str,
) -> usize {
    eval(engine, &format!("var returned = {call};"));
    let value = global(engine, "returned");
    eval(engine, "returned = undefined;");
    let ValueView::Str(string) = &(value).view() else {
        panic!("{call} returned no string")
    };
    assert_eq!(string.as_str(), text, "{tier:?} {call}");
    if tier != Tier::Interp {
        with_chunk(engine, name, spliced, |chunk| {
            let constant = &constants(chunk)[index];
            assert_eq!(constant.as_str(), text, "{tier:?} {name}'s constant");
            assert!(
                LStr::ptr_eq(string, constant),
                "{tier:?} {call} returns {name}'s constant itself"
            );
            let k = chunk
                .jit_ops()
                .iter()
                .filter_map(|op| match *op {
                    Op::Const(k) if chunk.jit_const_is_str(k) => Some(k),
                    _ => None,
                })
                .nth(index)
                .expect("the constant's index");
            assert_eq!(
                chunk.jit_const_str_word(k),
                Some(crate::value::PackedValue::pack(value.clone()).bits()),
                "the immediate is the word the VM pushes"
            );
        });
    }
    !(string.as_ptr() as usize)
}

/// A string literal evaluates to its chunk's constant, the same string at every evaluation, and
/// it equals a string with the same text built at run time (`===`, `==`), across collections:
/// `literal` returns its constant, `same` compares it at once, and `inLoop` pushes another
/// constant with the same text in a loop that runs as a register region. At `Tier::Jit` the
/// constants reach the operand stack through the string-constant templates, which embed the
/// constant's word.
#[test]
fn string_constants_keep_identity_and_equality_across_collections() {
    const SOURCE: &str = r#"
        var computed = ['constant', 'text'].join('-');
        function literal() { return 'constant-text'; }
        function same() { return literal() === computed; }
        function inLoop(n) {
            var s, hits = 0;
            for (var i = 0; i < n; i++) {
                s = 'constant-text';
                if (s === computed) hits++;
            }
            return hits === n ? s : 'missed';
        }
        function round(n) {
            var r = [same()];
            $262.gc();
            r.push(same(), computed === literal(), literal() == computed,
                literal() !== 'constant-texT', literal().length, literal().charCodeAt(9),
                typeof literal(), inLoop(n) === computed);
            return r.join();
        }
        round(40);
    "#;
    const EXPECTED: &str = "true,true,true,true,true,13,116,string,true";
    const TEXT: &str = "constant-text";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, SOURCE), EXPECTED, "{tier:?}");
        let names = ["literal", "same", "inLoop", "round"];
        if tier == Tier::Jit {
            assert_native(&mut engine, &names);
        }
        if tier != Tier::Interp {
            settle(&mut engine, &names);
        }
        let first = returned(&mut engine, tier, "literal()", ("literal", false, 0), TEXT);
        let looped = returned(&mut engine, tier, "inLoop(40)", ("inLoop", false, 0), TEXT);
        #[cfg(target_arch = "aarch64")]
        let regions = super::regions::executed_regions();
        for _ in 0..3 {
            engine.interp.gc_collect();
            assert_eq!(eval(&mut engine, "round(40)"), EXPECTED, "{tier:?}");
        }
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit {
            assert!(
                super::regions::executed_regions() > regions,
                "inLoop's loop runs as a register region"
            );
        }
        #[cfg(target_arch = "x86_64")]
        if tier == Tier::Jit {
            for name in ["literal", "inLoop"] {
                assert!(
                    holds_immediate(&mut engine, name, 0),
                    "{name}'s string constant is an immediate"
                );
            }
        }
        engine.interp.gc_collect();
        let again = returned(&mut engine, tier, "literal()", ("literal", false, 0), TEXT);
        let looped_again = returned(&mut engine, tier, "inLoop(40)", ("inLoop", false, 0), TEXT);
        if tier != Tier::Interp {
            assert_eq!(again, first, "{tier:?}: literal() returns the same string");
            assert_eq!(
                looped_again, looped,
                "{tier:?}: inLoop() returns the same string"
            );
        }
    }
}

/// `s += x` with `s` initialized from a literal appends to a new string and leaves the literal
/// as it was, for a local and for a binding a closure captures. The next evaluation of the
/// literal reads the same text, and the grown string's buffer is not the constant's. A constant
/// is allocated at exactly its length and appends write at a buffer's frontier, so any append
/// to one copies.
#[test]
fn appends_leave_string_constants_intact() {
    const SOURCE: &str = r#"
        function literal() { return 'abc'; }
        function grow(n) { var s = 'abc'; for (var i = 0; i < n; i++) s += 'd'; return s; }
        function growCaptured(n) {
            var s = 'abc';
            function read() { return s; }
            for (var i = 0; i < n; i++) s += 'd';
            return read();
        }
        function round(n) {
            return [grow(n), growCaptured(n), literal(), grow(n), growCaptured(n), literal(),
                grow(0), growCaptured(0)].join();
        }
        round(5);
    "#;
    const EXPECTED: &str = "abcddddd,abcddddd,abc,abcddddd,abcddddd,abc,abc,abc";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, SOURCE), EXPECTED, "{tier:?}");
        let names = ["literal", "grow", "growCaptured", "round"];
        if tier == Tier::Jit {
            assert_native(&mut engine, &names);
            #[cfg(target_arch = "x86_64")]
            for name in ["literal", "grow", "growCaptured"] {
                assert!(
                    holds_immediate(&mut engine, name, 0),
                    "{name}'s literal is an immediate"
                );
            }
        }
        if tier != Tier::Interp {
            settle(&mut engine, &names);
        }
        for n in [1, 5, 40] {
            engine.interp.gc_collect();
            let grown = "abc".to_owned() + &"d".repeat(n);
            assert_eq!(
                eval(&mut engine, &format!("round({n})")),
                format!("{grown},{grown},abc,{grown},{grown},abc,abc,abc"),
                "{tier:?}"
            );
            for name in ["grow", "growCaptured"] {
                grown_apart(&mut engine, tier, name, n);
                let zero = format!("{name}(0)");
                returned(&mut engine, tier, &zero, (name, false, 0), "abc");
            }
            returned(&mut engine, tier, "literal()", ("literal", false, 0), "abc");
        }
    }
}

/// `name(n)` returns its literal grown by `n` appends, in a buffer other than the literal's
/// constant, which still reads "abc" and has no room to grow in place.
#[inline(never)]
fn grown_apart(engine: &mut Engine, tier: Tier, name: &str, n: usize) {
    eval(engine, &format!("var returned = {name}({n});"));
    let value = global(engine, "returned");
    eval(engine, "returned = undefined;");
    let ValueView::Str(grown) = &(value).view() else {
        panic!("{name}({n}) returned no string")
    };
    assert_eq!(grown.as_str(), "abc".to_owned() + &"d".repeat(n));
    if tier != Tier::Interp {
        with_chunk(engine, name, false, |chunk| {
            let constant = &constants(chunk)[0];
            assert_eq!(constant.as_str(), "abc", "{tier:?} {name}'s literal");
            assert_eq!(
                constant.allocation_requested_bytes(),
                LStr::from("abc").allocation_requested_bytes(),
                "a constant has no spare capacity"
            );
            assert_ne!(
                grown.buffer_ptr(),
                constant.as_ptr(),
                "{tier:?} {name}({n}) grew into its literal's buffer"
            );
        });
    }
}

/// A literal of an inlined callee is a constant of the caller's second-stage chunk: the
/// compiler splices the callee's statements into the caller and pushes the literal into the
/// caller's own pool. At `Tier::Jit` the hot caller's spliced site returns the caller's
/// constant, not the callee's. Then the callee becomes unreachable and collections run, which
/// free it and drop its function and chunk: the caller's spliced constant still reads its text,
/// and the caller's code, whose inline guard now declines (a guard can never pass again once
/// its callee is dead), returns the replacement's string.
#[test]
fn inlined_string_constants_belong_to_the_caller() {
    const SOURCE: &str = r#"
        var callee = function () { return 'inlined-text'; };
        var expected = ['inlined', 'text'].join('-');
        function caller() { return callee(); }
        function drive(n) {
            var hits = 0;
            for (var i = 0; i < n; i++) if (caller() === expected) hits++;
            return hits;
        }
        drive(150);
    "#;
    const TEXT: &str = "inlined-text";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, SOURCE), "150", "{tier:?}");
        let spliced = tier == Tier::Jit && crate::bytecode::inline_recompile_at() == 100;
        if spliced {
            assert_native(&mut engine, &["caller", "drive"]);
            settle(&mut engine, &["drive"]);
            // Planning reads the caller's call-cache feedback, which a concurrent test's
            // process-wide cache invalidation can empty for a while: the checkpoint retries.
            for _ in 0..20 {
                if second_stage(&mut engine, "caller") {
                    break;
                }
                assert_eq!(eval(&mut engine, "drive(150)"), "150");
            }
            with_chunk(&mut engine, "caller", true, |chunk| {
                assert!(
                    chunk
                        .jit_ops()
                        .iter()
                        .any(|op| matches!(op, Op::InlineGuard(..))),
                    "caller's second stage splices callee"
                );
            });
            // The spliced site's string is the caller's constant, a string of its own.
            returned(&mut engine, tier, "caller()", ("caller", true, 0), TEXT);
            spliced_apart(&mut engine);
        } else if tier != Tier::Interp {
            returned(&mut engine, tier, "callee()", ("callee", false, 0), TEXT);
        }
        let (callee, function) = crate::gc_test::in_frame(|| {
            let callee = global(&mut engine, "callee");
            let object = callee.as_obj().expect("a function object");
            let function = {
                let borrowed = object.borrow();
                let crate::value::Callable::User(user) = &borrowed.call else {
                    panic!("callee is not a user function")
                };
                std::rc::Rc::downgrade(&user.func)
            };
            (crate::value::Gc::downgrade(&object), function)
        });
        eval(
            &mut engine,
            "callee = function () { return 'replacement'; };",
        );
        for _ in 0..3 {
            engine.interp.gc_collect();
        }
        assert!(
            !crate::gc_test::alive(&callee),
            "{tier:?}: the first callee is unreachable and collected"
        );
        assert_eq!(
            function.strong_count(),
            0,
            "{tier:?}: the first callee's function, which owns its chunk, is dropped"
        );
        assert_eq!(eval(&mut engine, "drive(50)"), "0", "{tier:?}");
        assert_eq!(eval(&mut engine, "caller()"), "replacement", "{tier:?}");
        if spliced {
            for _ in 0..3 {
                engine.interp.gc_collect();
            }
            spliced_text(&mut engine, TEXT);
        }
    }
}

/// The caller's spliced literal and the callee's own are separate constants (checked in a frame
/// of its own).
#[inline(never)]
fn spliced_apart(engine: &mut Engine) {
    let callee = with_chunk(engine, "callee", false, |chunk| {
        constants(chunk)[0].as_ptr()
    });
    let spliced = with_chunk(engine, "caller", true, |chunk| constants(chunk)[0].as_ptr());
    assert_ne!(
        callee, spliced,
        "the spliced literal is the caller's own constant"
    );
}

/// The caller's spliced constant reads `text` (checked in a frame of its own).
#[inline(never)]
fn spliced_text(engine: &mut Engine, text: &str) {
    with_chunk(engine, "caller", true, |chunk| {
        assert_eq!(constants(chunk)[0].as_str(), text);
    });
}
