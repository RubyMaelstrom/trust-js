//! Template code records its in-line element hits with the optimizing tier on (P3 stage 3, rule
//! U): each element form's typed arm stores the view's native kind in its site's last-seen typed
//! byte (`typed_array::emit`, `opt::element_records`), and its words path the Array's or ordinary
//! object's kind tag in its family byte (`emit_family_record`), which the tier's readers merge
//! into the element record (`opt::profile::merge_element_hits`), as JSC's baseline stores an
//! access's structure in its ArrayProfile (jit/JITInlines.h:328-348) and computeUpdatedPrediction
//! merges it (bytecode/ArrayProfile.cpp:126-136). The tests run template code only (no VM run of a site,
//! no helper miss: its record stays empty until the merge), through the shared stub and, in a
//! child process under TRUST_JS_JIT_TYPED_STUBS=0, in line; a second stage's hits reach the
//! baseline site's byte; views the native access declines record no kind; with the tier off no
//! recording word is emitted and the record stays empty.

use crate::bytecode::{Chunk, Op};
use crate::feedback::{elem_typed_kind, ELEM_KEY_INDEX, ELEM_TYPED_ARRAY, ELEM_TYPED_KINDS};
use crate::opt::profile::{self, ChunkProfile, UNSEEN};
use crate::opt::TestSwitches;
use crate::Engine;

use super::traced_tests::{eval, with_chunk};

/// The typed-array constructors and their native kinds (native_typed_array.rs).
const KINDS: [(&str, u8); 9] = [
    ("Uint8Array", 0),
    ("Int8Array", 1),
    ("Uint16Array", 2),
    ("Int16Array", 3),
    ("Uint32Array", 4),
    ("Int32Array", 5),
    ("Float32Array", 6),
    ("Float64Array", 7),
    ("Uint8ClampedArray", 8),
];

/// The element forms with a receiver on the stack or in a slot, one function each: GetElem,
/// SetElem, SetElemDrop (the receiver a property's value), GetElemLocal, SetElemLocal,
/// SetElemLocalDrop, and an element read in a numeric chain (`build_chain`).
const FORMS: &str = "function gs(o, i) { return o.t[i]; }
     function sk(o, i, v) { return (o.t[i] = v); }
     function sd(o, i, v) { o.t[i] = v; }
     function gl(t, i) { return t[i]; }
     function sl(t, i, v) { return (t[i] = v); }
     function sld(t, i, v) { t[i] = v; }
     function ch(t, k) { return t[k] + 7; }";

/// Each FORMS function, its call, and the element operation it is about.
const CALLS: [(&str, &str, fn(&Op) -> bool); 7] = [
    ("gs", "gs(holder, 1)", |op| matches!(op, Op::GetElem)),
    ("sk", "sk(holder, 1, 3)", |op| matches!(op, Op::SetElem)),
    ("sd", "sd(holder, 1, 3)", |op| matches!(op, Op::SetElemDrop)),
    ("gl", "gl(view, 1)", |op| matches!(op, Op::GetElemLocal(_))),
    ("sl", "sl(view, 1, 3)", |op| {
        matches!(op, Op::SetElemLocal(_))
    }),
    ("sld", "sld(view, 1, 3)", |op| {
        matches!(op, Op::SetElemLocalDrop(_))
    }),
    ("ch", "ch(view, 1)", |op| matches!(op, Op::GetElemLocal(_))),
];

/// Loops whose element sites the template lowers in its own loop forms: the fused name-element
/// read and write of a global view (`regions::emit_name_element`), and a register region's
/// element read and store on a local view.
const LOOPS: &str = "function rn(n) { n = n | 0; var s = 0; \
     for (var k = 0; (k | 0) < (n | 0); k = (k + 1) | 0) s += named[k & 7]; return s; }
     function wn(n) { n = n | 0; \
     for (var k = 0; (k | 0) < (n | 0); k = (k + 1) | 0) named[k & 7] = k & 3; return 0; }
     function rr(t, n) { n = n | 0; var s = 0; \
     for (var k = 0; (k | 0) < (n | 0); k = (k + 1) | 0) s = s + t[k & 7]; return s; }
     function wr(t, n) { n = n | 0; \
     for (var k = 0; (k | 0) < (n | 0); k = (k + 1) | 0) t[k & 7] = k & 3; return 0; }";

const LOOP_CALLS: [(&str, &str, fn(&Op) -> bool); 4] = [
    ("rn", "rn(64)", |op| matches!(op, Op::GetElem)),
    ("wn", "wn(64)", |op| matches!(op, Op::SetElemDrop)),
    ("rr", "rr(view, 64)", |op| matches!(op, Op::GetElemLocal(_))),
    ("wr", "wr(view, 64)", |op| {
        matches!(op, Op::SetElemLocalDrop(_))
    }),
];

/// The tier on, not eager, nothing forced: template code records, and no checkpoint of a chunk
/// the tests leave pending is due ([`template_only`]).
fn tier_on() -> TestSwitches {
    TestSwitches {
        on: true,
        ..Default::default()
    }
}

/// An engine at the JIT tier under `switches`, compiling each function at its first call.
fn engine(switches: TestSwitches) -> Engine {
    switches.set();
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

/// The pc of `chunk`'s one operation `which` accepts.
fn pc_of(chunk: &Chunk, which: fn(&Op) -> bool) -> usize {
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| which(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(pcs.len(), 1, "one such operation: {:?}", chunk.jit_ops());
    pcs[0]
}

/// Run `call` (the global function `name`'s) twice as template code only: its first call
/// compiles it, and its checkpoint is then never due, so that the tier compiles nothing.
fn template_only(engine: &mut Engine, name: &str, call: &str) {
    eval(engine, call);
    with_chunk(engine, name, |chunk| {
        assert!(chunk.jit.get().flatten().is_some(), "{name}: native");
        chunk.set_inline_retry_at(u32::MAX);
    });
    eval(engine, call);
}

/// The site's last-seen byte and its record (before any merge).
fn seen(chunk: &Chunk, pc: usize) -> (u8, u32) {
    (
        ChunkProfile::get(chunk).map_or(UNSEEN, |profile| profile.element_seen(pc)),
        chunk.feedback_vector().record_word(pc),
    )
}

/// The record a merge of a typed hit of native kind `kind` makes of an empty one.
fn merged(kind: u8) -> u32 {
    ELEM_TYPED_ARRAY | ELEM_KEY_INDEX | elem_typed_kind(kind)
}

/// Each form's site, run as template code on one view of every kind, holds the kind in its
/// last-seen byte and nothing in its record; the merge makes the record a typed array of that
/// kind with an index key and empties the byte.
fn every_form_records_every_kind(source: &str, calls: &[(&str, &str, fn(&Op) -> bool)]) {
    for (constructor, kind) in KINDS {
        let mut engine = engine(tier_on());
        eval(
            &mut engine,
            &format!(
                "var view = new {constructor}(8); var named = new {constructor}(8); \
                 var holder = {{t: view}}; {source}"
            ),
        );
        for &(name, call, which) in calls {
            template_only(&mut engine, name, call);
            with_chunk(&mut engine, name, |chunk| {
                let pc = pc_of(chunk, which);
                assert_eq!(
                    seen(chunk, pc),
                    (kind, 0),
                    "{name} on {constructor}: the byte holds the kind, the record nothing"
                );
                profile::merge_element_hits(chunk);
                assert_eq!(
                    seen(chunk, pc),
                    (UNSEEN, merged(kind)),
                    "{name} on {constructor}: merged and emptied"
                );
            });
        }
    }
}

#[test]
fn in_line_typed_hits_record_their_kind_in_every_element_form() {
    crate::opt::pin_forced_exits_off_for_test();
    every_form_records_every_kind(FORMS, &CALLS);
}

#[test]
fn the_template_loops_typed_hits_record_their_kind() {
    crate::opt::pin_forced_exits_off_for_test();
    every_form_records_every_kind(LOOPS, &LOOP_CALLS);
}

#[test]
fn hits_carried_in_line_without_the_stub_record_their_kind() {
    // TRUST_JS_JIT_TYPED_STUBS=0 is read once a process: the forms again in a child.
    const CHILD: &str = "TRUST_JS_TEST_ELEMENT_RECORD_INLINE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::element_record_tests::hits_carried_in_line_without_the_stub_record_their_kind",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_JIT_TYPED_STUBS", "0")
            .env("TRUST_JS_JIT_NO_NAME_ELEM_STUB", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    assert!(
        !super::typed_array::stubs_for_test(),
        "the child carries the access in line"
    );
    crate::opt::pin_forced_exits_off_for_test();
    every_form_records_every_kind(FORMS, &CALLS);
    every_form_records_every_kind(LOOPS, &LOOP_CALLS);
}

#[test]
fn a_record_that_holds_more_keeps_it_and_a_merge_ors_each_kind_seen() {
    // The VM's run records a Float64Array and a String key; the template's hits then store an
    // Int32Array's kind: the merge adds it, the rest stays.
    crate::opt::pin_forced_exits_off_for_test();
    tier_on().set();
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function gl(t, i) { return t[i]; } var f = new Float64Array(4), w = new Int32Array(4);
         gl(f, 1); gl({a: 1}, 'a');",
    );
    let wider = with_chunk(&mut engine, "gl", |chunk| {
        chunk
            .feedback_vector()
            .record_word(pc_of(chunk, |op| matches!(op, Op::GetElemLocal(_))))
    });
    assert_ne!(wider & elem_typed_kind(7), 0, "the VM's Float64Array");
    engine.set_tier(crate::bytecode::Tier::Jit);
    template_only(&mut engine, "gl", "gl(w, 1)");
    with_chunk(&mut engine, "gl", |chunk| {
        let pc = pc_of(chunk, |op| matches!(op, Op::GetElemLocal(_)));
        assert_eq!(seen(chunk, pc), (5, wider));
        profile::merge_element_hits(chunk);
        assert_eq!(seen(chunk, pc), (UNSEEN, wider | merged(5)));
    });
    // A Uint8Array's hit after the merge: its kind joins the others.
    eval(&mut engine, "gl(new Uint8Array(4), 1)");
    with_chunk(&mut engine, "gl", |chunk| {
        let pc = pc_of(chunk, |op| matches!(op, Op::GetElemLocal(_)));
        profile::merge_element_hits(chunk);
        let word = chunk.feedback_vector().record_word(pc);
        assert_eq!(word, wider | merged(5) | merged(0));
        assert_eq!(
            word & ELEM_TYPED_KINDS,
            elem_typed_kind(0) | elem_typed_kind(5) | elem_typed_kind(7)
        );
    });
}

#[test]
fn views_the_native_access_declines_record_no_kind() {
    // BigInt elements and a shared buffer's view leave the in-line access before its kind load
    // (the helper then records the access, with no kind for a declined view): the byte stays
    // empty.
    crate::opt::pin_forced_exits_off_for_test();
    let mut engine = engine(tier_on());
    eval(
        &mut engine,
        "function gl(t, i) { return t[i]; } function sld(t, i, v) { t[i] = v; }
         var big = new BigInt64Array(4), shared = new Int32Array(new SharedArrayBuffer(16));",
    );
    for (view, value) in [("big", "2n"), ("shared", "3")] {
        for (name, call, which) in [
            (
                "gl",
                format!("gl({view}, 1)"),
                (|op| matches!(op, Op::GetElemLocal(_))) as fn(&Op) -> bool,
            ),
            ("sld", format!("sld({view}, 1, {value})"), |op| {
                matches!(op, Op::SetElemLocalDrop(_))
            }),
        ] {
            template_only(&mut engine, name, &call);
            with_chunk(&mut engine, name, |chunk| {
                let pc = pc_of(chunk, which);
                let (byte, word) = seen(chunk, pc);
                assert_eq!(byte, UNSEEN, "{call}: no in-line hit");
                assert_eq!(word & ELEM_TYPED_KINDS, 0, "{call}: no kind");
                profile::merge_element_hits(chunk);
                assert_eq!(
                    chunk.feedback_vector().record_word(pc) & ELEM_TYPED_KINDS,
                    0,
                    "{call}: still no kind"
                );
            });
        }
    }
}

/// The global function `name`'s second stage, if it has one.
fn second_stage_of(engine: &mut Engine, name: &str) -> Option<std::rc::Rc<Chunk>> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok()?;
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    user.func.runtime().code2.get().and_then(Option::clone)
}

/// The recording stores in `code` of register `reg`'s byte: `strb w<reg>, [x16]` (and for w12
/// `[x13]`, a loop chain's), which only `typed_array::emit_record` (w17, the kind) and
/// `emit_family_record` (w12, the tag) emit.
fn stores_of(code: &super::JitCode, reg: u32) -> usize {
    let store = |base: u32| {
        let mut a = super::asm::Asm::new();
        a.strb_imm(reg, base, 0);
        a.buf[0]
    };
    let stores = [store(16), if reg == 12 { store(13) } else { store(16) }];
    // SAFETY: the code's `len` bytes are mapped and readable while `code` lives.
    let words = unsafe { std::slice::from_raw_parts(code.mem as *const u32, code.len / 4) };
    words.iter().filter(|&&word| stores.contains(&word)).count()
}

/// The typed hits' recording stores in `code`.
fn recording_stores(code: &super::JitCode) -> usize {
    stores_of(code, 17)
}

/// The words paths' recording stores in `code`.
fn family_stores(code: &super::JitCode) -> usize {
    stores_of(code, 12)
}

#[test]
fn with_the_tier_off_template_code_records_nothing() {
    // The tier off: no table, no recording word in any form's code, no byte, an empty record.
    // The tier on: a recording word in each.
    for on in [false, true] {
        let mut engine = engine(TestSwitches {
            on,
            ..Default::default()
        });
        eval(
            &mut engine,
            &format!(
                "var view = new Int32Array(8); var named = new Int32Array(8); \
                 var holder = {{t: view}}; {FORMS} {LOOPS}"
            ),
        );
        for &(name, call, which) in CALLS.iter().chain(&LOOP_CALLS) {
            template_only(&mut engine, name, call);
            with_chunk(&mut engine, name, |chunk| {
                let pc = pc_of(chunk, which);
                let code = chunk.jit.get().flatten().expect("native");
                let stores = recording_stores(&code);
                let families = family_stores(&code);
                assert_eq!(crate::opt::element_records(chunk).is_some(), on, "{name}");
                if on {
                    assert!(stores >= 1, "{name}: the tier on records");
                    assert!(
                        families >= 1 || name == "rn" || name == "wn",
                        "{name}: a words path records"
                    );
                    assert_eq!(seen(chunk, pc).0, 5, "{name}");
                } else {
                    assert_eq!(stores + families, 0, "{name}: no recording word");
                    assert!(
                        ChunkProfile::get(chunk).is_none_or(|p| p.element_seen(pc) == UNSEEN),
                        "{name}: no byte"
                    );
                    profile::merge_element_hits(chunk);
                    assert_eq!(seen(chunk, pc), (UNSEEN, 0), "{name}: an empty record");
                }
            });
        }
    }
}

#[test]
fn a_chunk_over_the_tiers_size_limit_records_nothing() {
    // JSC's baseline emits no profiling where the DFG cannot compile (jit/JIT.cpp:722-733): a
    // chunk the tier refuses before its snapshot records nothing either.
    crate::opt::pin_forced_exits_off_for_test();
    let mut engine = engine(tier_on());
    // Past the tier's 15,000 operations, within the template's 65,535.
    let mut big = String::from("var view = new Int32Array(8); function big(t, i) { var s = t[i];");
    for k in 0..crate::opt::MAX_OPTIMIZED_OPS / 2 {
        big.push_str(&format!(" s = s * {k};"));
    }
    big.push_str(" return s; }");
    eval(&mut engine, &big);
    template_only(&mut engine, "big", "big(view, 1)");
    with_chunk(&mut engine, "big", |chunk| {
        assert!(chunk.jit_ops().len() > crate::opt::MAX_OPTIMIZED_OPS);
        assert!(crate::opt::element_records(chunk).is_none());
        let pc = pc_of(chunk, |op| matches!(op, Op::GetElemLocal(_)));
        assert_eq!(seen(chunk, pc), (UNSEEN, 0));
        let code = chunk.jit.get().flatten().expect("native");
        assert_eq!(recording_stores(&code), 0);
    });
}

#[test]
fn a_second_stages_hits_reach_the_baseline_sites_bytes() {
    // `outer`'s second stage splices `inner`; the tier refuses it (its finally), so its template
    // code runs. Its own element site's hits and the splice's go to the bytes of the sites its
    // record map names: `outer`'s first stage's and `inner`'s, whose records the tier reads.
    crate::opt::pin_forced_exits_off_for_test();
    let mut engine = engine(TestSwitches {
        on: true,
        eager: true,
        ..Default::default()
    });
    eval(
        &mut engine,
        "function inner(t, i) { return t[i]; }
         function outer(t, u, n) { var s = 0;
           try { for (var i = 0; i < n; i++) s += inner(t, i & 3) + u[i & 3]; } finally { s += 1; }
           return s; }
         var w = new Int32Array(4), d = new Float64Array(4), b = new Uint8Array(4);
         for (var k = 0; k < 400; k++) outer(w, w, 4);",
    );
    let second = second_stage_of(&mut engine, "outer").expect("outer's second stage");
    let map = second
        .feedback_vector()
        .record_map()
        .expect("a second stage has a record map");
    // The second stage's element sites: the splice's GetElemLocal and outer's own.
    let sites: Vec<usize> = (0..second.jit_ops().len())
        .filter(|&pc| matches!(second.jit_ops()[pc], Op::GetElemLocal(_)))
        .collect();
    assert_eq!(sites.len(), 2, "{:?}", second.jit_ops());
    let table: std::collections::HashMap<u32, u64> = crate::opt::element_records(&second)
        .expect("the second stage records")
        .into_iter()
        .collect();
    let mut targets = Vec::new();
    for &pc in &sites {
        let (baseline, baseline_pc) = map.target(pc).expect("a mapped element site");
        assert!(!std::ptr::eq(baseline, &*second), "a baseline chunk");
        let byte = ChunkProfile::get(baseline)
            .expect("the baseline's record")
            .element_byte(baseline.jit_ops().len(), baseline_pc);
        assert_eq!(
            table.get(&(pc as u32)).copied(),
            byte,
            "the baseline site's byte"
        );
        // Empty them: what follows is the second stage's alone.
        profile::merge_element_hits(baseline);
        targets.push((baseline as *const Chunk, baseline_pc));
    }
    // Other kinds, which only the second stage's code now sees (inner's own code is not entered).
    eval(&mut engine, "outer(d, b, 4)");
    let kinds: Vec<u8> = targets
        .iter()
        .map(|&(baseline, pc)| {
            // SAFETY: the map holds the baseline chunks while `second` lives.
            ChunkProfile::get(unsafe { &*baseline }).map_or(UNSEEN, |p| p.element_seen(pc))
        })
        .collect();
    let mut sorted = kinds.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![0, 7],
        "the splice's Float64Array and outer's Uint8Array: {kinds:?}"
    );
    assert!(
        ChunkProfile::get(&second)
            .is_none_or(|p| (0..second.jit_ops().len()).all(|pc| p.element_seen(pc) == UNSEEN)),
        "nothing in the second stage's own bytes"
    );
    // Arrays, which the second stage's words paths take (its loop is a register region): both
    // sites' family bytes hold the Array's tag.
    eval(
        &mut engine,
        "var wa = [1, 2, 3, 4], wb = [5, 6, 7, 8]; outer(wa, wb, 4)",
    );
    for &(baseline, pc) in &targets {
        // SAFETY: the map holds the baseline chunks while `second` lives.
        let tag = ChunkProfile::get(unsafe { &*baseline }).map_or(UNSEEN, |p| p.family_seen(pc));
        assert_eq!(tag, crate::value::KIND_ARRAY_TAG, "pc {pc}: an Array's tag");
    }
}

#[test]
fn the_fused_name_element_site_records_at_its_own_hit() {
    // The fused `LoadName K; key; GetElem/SetElemDrop` site (`regions::emit_name_element`) runs
    // its access through a process-wide stub or its in-line body, neither of which stores: the
    // site stores W17 at its hit. Each call runs one iteration in the loop's plain template, which
    // records too, so the fused site's own store shows in its code: three recording stores, the
    // fused site's beside the loop's plain template's and the fused fallback's.
    crate::opt::pin_forced_exits_off_for_test();
    let mut engine = engine(tier_on());
    eval(
        &mut engine,
        &format!("var view = new Uint16Array(8); var named = new Uint16Array(8); {LOOPS}"),
    );
    for (name, call, which) in &LOOP_CALLS[..2] {
        let before = super::regions::name_element_commits();
        template_only(&mut engine, name, call);
        let after = super::regions::name_element_commits();
        assert!(
            after[0] + after[1] > before[0] + before[1] + 64,
            "{name}: the fused site ran: {before:?} -> {after:?}"
        );
        with_chunk(&mut engine, name, |chunk| {
            let code = chunk.jit.get().flatten().expect("native");
            assert_eq!(recording_stores(&code), 3, "{name}: the fused site's store");
            assert_eq!(seen(chunk, pc_of(chunk, *which)), (2, 0), "{name}");
        });
    }
}

/// The receivers whose element words the words paths take: an Array, and an ordinary object with
/// elements.
const FAMILIES: [(&str, &str, bool); 2] = [
    ("an Array", "[0, 1, 2, 3, 4, 5, 6, 7]", true),
    (
        "an object",
        "{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}",
        false,
    ),
];

#[test]
fn words_path_hits_record_the_receivers_family_in_every_element_form() {
    // An Array's and an ordinary object's in-line element hits store their kind tag in the
    // site's family byte, none in its typed byte; the merge makes the record an Array or an
    // ordinary receiver with an index key. The fused name-element site has no words path: its
    // binding holding an Array, the access runs in its fallback's plain templates, which record.
    use crate::feedback::{ELEM_ARRAY, ELEM_ORDINARY};
    crate::opt::pin_forced_exits_off_for_test();
    for (what, literal, array) in FAMILIES {
        let mut engine = engine(tier_on());
        eval(
            &mut engine,
            &format!(
                "var view = {literal}; var named = {literal}; var holder = {{t: view}};                  {FORMS} {LOOPS}"
            ),
        );
        for &(name, call, which) in CALLS.iter().chain(&LOOP_CALLS) {
            template_only(&mut engine, name, call);
            with_chunk(&mut engine, name, |chunk| {
                let pc = pc_of(chunk, which);
                let profile = ChunkProfile::get(chunk).expect("the bytes");
                let tag = profile.family_seen(pc);
                assert_ne!(tag, UNSEEN, "{name} on {what}: the family byte");
                assert_eq!(
                    tag == crate::value::KIND_ARRAY_TAG,
                    array,
                    "{name} on {what}"
                );
                assert_eq!(seen(chunk, pc), (UNSEEN, 0), "{name} on {what}");
                profile::merge_element_hits(chunk);
                let family = if array { ELEM_ARRAY } else { ELEM_ORDINARY };
                assert_eq!(
                    chunk.feedback_vector().record_word(pc),
                    family | ELEM_KEY_INDEX,
                    "{name} on {what}: merged"
                );
                assert_eq!(profile.family_seen(pc), UNSEEN, "{name} on {what}: emptied");
            });
        }
    }
}

#[test]
fn a_chains_later_accesses_of_a_receiver_record_its_family() {
    // A numeric chain validates a receiver at its first access and keeps it in a register for
    // the next ones, whose family store reloads the tag.
    crate::opt::pin_forced_exits_off_for_test();
    for (what, literal, array) in FAMILIES {
        let mut engine = engine(tier_on());
        eval(
            &mut engine,
            &format!("var view = {literal}; function two(t, k) {{ return t[k] + t[k + 1]; }}"),
        );
        template_only(&mut engine, "two", "two(view, 1)");
        with_chunk(&mut engine, "two", |chunk| {
            let code = chunk.jit.get().flatten().expect("native");
            assert!(family_stores(&code) >= 2, "{what}: both sites record");
            let profile = ChunkProfile::get(chunk).expect("the bytes");
            for pc in (0..chunk.jit_ops().len())
                .filter(|&pc| matches!(chunk.jit_ops()[pc], Op::GetElemLocal(_)))
            {
                let tag = profile.family_seen(pc);
                assert_ne!(tag, UNSEEN, "{what}: pc {pc}");
                assert_eq!(
                    tag == crate::value::KIND_ARRAY_TAG,
                    array,
                    "{what}: pc {pc}"
                );
            }
        });
    }
}
