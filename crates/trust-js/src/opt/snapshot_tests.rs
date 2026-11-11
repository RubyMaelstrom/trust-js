//! The snapshot in the exemplars' shape (P3 stage 2, step 16): a chunk carries its names' keys (a
//! first stage's table made at its first freeze where the VM has not made it, a second stage's
//! carried at its recompile from the tables of the chunks it was made from, a name no table
//! holds made there), and the snapshot's raw copy, taken on the main thread, decodes on any
//! thread to the views the live chunk decodes to, each snapshot reading the caches as they stood
//! at its own copy.

use super::snapshot::{check_decode, Raw, Snapshot};
use super::TestSwitches;
use crate::atom::PropKey;
use crate::bytecode::{Chunk, Op, Tier};
use crate::feedback::SiteView;
use crate::jit::JitChunk;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// An engine at `tier`, compiling at a function's first call, with the optimizing tier on
/// (synchronous, not eager) or off: the programs below call each function fewer times than its
/// first checkpoint, so that no tier compile and no second stage comes on its own, and the tests
/// make them. The template JIT's calls fill the call caches the inline plans read.
fn engine(on: bool, tier: Tier) -> Engine {
    TestSwitches {
        on,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

/// Global function `name`'s function and realm.
fn function(engine: &mut Engine, name: &str) -> (Rc<crate::ast::Function>, usize) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let object = value.as_obj().expect("a function");
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is an author function")
    };
    (user.func.clone(), user.realm)
}

/// Global function `name`'s first stage.
fn first_stage(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let (func, _) = function(engine, name);
    let runtime = func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .expect("the function ran as bytecode")
}

/// `name`'s recompile with its inline plan (a second stage), made now.
fn second_stage(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let primary = first_stage(engine, name);
    let (func, realm) = function(engine, name);
    let env = engine.interp.global_env;
    let plan = crate::bytecode::plan_inlines(&primary, &func, realm, &env);
    assert!(!plan.is_empty(), "{name} splices a callee");
    crate::bytecode::compile_with_inlines(&func, &plan, &primary).expect("its recompile")
}

/// The index of `chunk`'s name `text`.
fn name_index(chunk: &Chunk, text: &str) -> u32 {
    (0..chunk.name_count() as u32)
        .find(|&n| chunk.jit_name(n) == text)
        .unwrap_or_else(|| panic!("the chunk names {text}"))
}

/// Every key of `chunk`'s table is its name's atom, as a lookup by text finds it.
fn keys_equal_lookups(chunk: &Chunk) {
    for n in 0..chunk.name_count() as u32 {
        let name = chunk.jit_name(n);
        assert_eq!(
            Some(chunk.name_key(n)),
            PropKey::lookup_text(name),
            "name {n} ({name:?})"
        );
    }
}

const SPLICED: &str = "function leaf(o) { return o.alpha + o.beta; }\n\
     function caller(o, x) { var r = leaf(o); if (x > 1e9) return zq_made_at_the_recompile; \
     return r; }\n\
     for (var k = 0; k < 20; k++) caller({ alpha: 1, beta: 2 }, k);";

#[test]
fn a_second_stages_keys_come_with_its_recompile_and_equal_lookups_by_text() {
    let mut engine = engine(true, Tier::Jit);
    eval(&mut engine, SPLICED);
    // caller ran no named access of its own, so its first stage has no table, and the global it
    // never read has no atom (no template makes a free name's key, where a named read's template
    // makes its name's): the recompile makes it; leaf's names come from leaf's table.
    let primary = first_stage(&mut engine, "caller");
    assert!(!primary.has_name_keys());
    assert!(first_stage(&mut engine, "leaf").has_name_keys());
    assert!(PropKey::lookup_text("zq_made_at_the_recompile").is_none());
    let second = second_stage(&mut engine, "caller");
    assert!(second.has_name_keys(), "carried at the recompile");
    assert!(
        !primary.has_name_keys(),
        "the first stage's own table is not made for it"
    );
    let fresh = PropKey::lookup_text("zq_made_at_the_recompile").expect("made at the recompile");
    assert!(
        second.name_key(name_index(&second, "zq_made_at_the_recompile")) == fresh,
        "the new name's key is its atom"
    );
    keys_equal_lookups(&second);
    // The frozen view takes them from the table.
    let view = second.freeze();
    for n in 0..second.name_count() as u32 {
        assert_eq!(view.jit_name_key(n), second.name_key(n).word());
    }
}

#[test]
fn a_second_stages_keys_are_its_names_where_a_splice_moves_them() {
    // The caller's first stage has its table (it ran named accesses), and the splice puts the
    // callee's names between the caller's own: each key is its own name's, from the table that
    // holds that name, whatever its index there.
    let mut engine = engine(true, Tier::Jit);
    eval(
        &mut engine,
        "function inner(o) { return o.alpha + o.beta; }\n\
         function outer(o) { var a = o.first; var r = inner(o); return a + r + o.last; }\n\
         for (var k = 0; k < 20; k++) outer({ first: 1, alpha: 2, beta: 3, last: 4 });",
    );
    let primary = first_stage(&mut engine, "outer");
    assert!(primary.has_name_keys());
    let second = second_stage(&mut engine, "outer");
    assert!(second.has_name_keys());
    let moved = (0..second.name_count() as u32).any(|n| {
        let name = second.jit_name(n);
        (n as usize) < primary.name_count() && primary.jit_name(n) != name
    });
    assert!(moved, "the splice moves names to other indices");
    keys_equal_lookups(&second);
    for name in ["first", "alpha", "beta", "last"] {
        let n = name_index(&second, name);
        assert!(second.name_key(n).is(name), "{name}'s key is its own");
    }
}

#[test]
fn with_the_tier_off_a_recompile_makes_no_keys() {
    let mut engine = engine(false, Tier::Jit);
    eval(&mut engine, SPLICED);
    let second = second_stage(&mut engine, "caller");
    assert!(!second.has_name_keys(), "the tier off pays nothing new");
    assert!(PropKey::lookup_text("zq_made_at_the_recompile").is_none());
}

#[test]
fn a_name_with_no_atom_yet_is_made_at_the_first_freeze() {
    let mut engine = engine(true, Tier::Bytecode);
    eval(
        &mut engine,
        "function f(o, x) { if (x > 1e9) return o.zq_made_at_the_freeze; return x + 1; }\n\
         for (var k = 0; k < 3; k++) f({}, k);",
    );
    let chunk = first_stage(&mut engine, "f");
    assert!(!chunk.has_name_keys(), "no named access ran");
    assert!(PropKey::lookup_text("zq_made_at_the_freeze").is_none());
    let view = chunk.freeze();
    let key = PropKey::lookup_text("zq_made_at_the_freeze").expect("made at the freeze");
    assert_eq!(
        view.jit_name_key(name_index(&chunk, "zq_made_at_the_freeze")),
        key.word()
    );
    assert!(chunk.has_name_keys(), "once a chunk");
    keys_equal_lookups(&chunk);
    assert_eq!(
        chunk.freeze().keys,
        view.keys,
        "a later freeze reads the same table"
    );
}

/// Sites of every kind the decode reads: named loads, stores and updates (one way, several, a
/// generic site), value and element records, calls (one callee, several), a construct site,
/// literals (object, array, constant array, RegExp, closure) and free names.
const KINDS: &str = "var G = 3;\n\
     function C(v) { this.v = v; }\n\
     function one(x) { return x + 1; }\n\
     function two(x) { return x * 2; }\n\
     function kinds(o, a, t, k, f) {\n\
       var s = o.x + G;\n\
       o.y = s;\n\
       o.n++;\n\
       s += a[k] + t[k & 3];\n\
       a[k & 7] = s;\n\
       s = s < 1e9 ? -s : s;\n\
       var c = new C(s);\n\
       s += f(k) + one(k);\n\
       var lit = { p: s, q: c.v };\n\
       var arr = [s, k];\n\
       var cst = [1, 2, 3];\n\
       var re = /ab+c/;\n\
       var cl = function () { return lit.p; };\n\
       return s + arr.length + cst[1] + (re.test('abbc') ? 1 : 0) + cl() + lit.q;\n\
     }\n\
     function caller(o, a, t, k) { return kinds(o, a, t, k, two) + one(k) + ('' + k).length; }\n\
     var shapes = [{ x: 1, y: 0, n: 0 }, { y: 0, x: 1, n: 0 }, { n: 0, x: 1, y: 0 }];\n\
     var ta = new Int32Array(4);\n\
     for (var k = 0; k < 40; k++) {\n\
       kinds(shapes[k % 3], [1, 2, 3, 4, 5, 6, 7, 8], ta, k, (k & 1) ? one : two);\n\
       caller(shapes[0], [1, 2, 3, 4, 5, 6, 7, 8], ta, k);\n\
     }\n\
     function generic(o) { return o.z; }\n\
     for (var k = 0; k < 12; k++) { var g = { z: k }; g['w' + k] = k; generic(g); }";

/// The views of `chunk` decoded from its raw copy and from its live words agree (`check_decode`
/// asserts it pc by pc), and the copy is no more than the sites' words: `chunk`'s snapshot.
fn decodes_both_ways(chunk: &Chunk) -> Snapshot {
    let raw = Raw::take_in(chunk, None);
    let view = chunk.freeze();
    check_decode(chunk, &raw, &view);
    Snapshot::decode(&raw, &view)
}

#[test]
fn the_raw_copy_decodes_to_the_live_chunks_views() {
    let mut engine = engine(true, Tier::Jit);
    eval(&mut engine, KINDS);
    for name in ["kinds", "caller", "generic", "C", "one"] {
        let chunk = first_stage(&mut engine, name);
        let snapshot = decodes_both_ways(&chunk);
        assert_eq!(snapshot.len(), chunk.jit_ops().len());
    }
    // The site kinds are there to compare.
    let snapshot = decodes_both_ways(&first_stage(&mut engine, "kinds"));
    let kinds: Vec<&str> = (0..snapshot.len())
        .map(|pc| match snapshot.site(pc) {
            SiteView::Values(_) => "values",
            SiteView::Element(_) => "element",
            SiteView::Named(_) => "named",
            SiteView::NamedUpdate(..) => "update",
            SiteView::Call(_) => "call",
            SiteView::Construct(_) => "construct",
            SiteView::Allocation(_) => "allocation",
            SiteView::None => "none",
        })
        .collect();
    for kind in [
        "values",
        "element",
        "named",
        "update",
        "call",
        "construct",
        "allocation",
    ] {
        assert!(kinds.contains(&kind), "kinds has a {kind} site: {kinds:?}");
    }
    let generic = decodes_both_ways(&first_stage(&mut engine, "generic"));
    assert!(
        (0..generic.len()).any(|pc| generic.named(pc).is_some_and(|named| named.generic)),
        "a generic site"
    );
    // A transformed chunk: its value and element records are its baseline sites', which the copy
    // takes through the record map.
    let second = second_stage(&mut engine, "caller");
    assert!(second.feedback_vector().record_map().is_some());
    let snapshot = decodes_both_ways(&second);
    assert!(
        (0..snapshot.len()).any(|pc| snapshot.values(pc).is_some_and(|v| v.has_record())),
        "a value site with its baseline's record (the string concatenation's)"
    );
}

#[test]
fn each_snapshot_reads_the_ways_of_its_own_copy() {
    let mut engine = engine(true, Tier::Bytecode);
    eval(
        &mut engine,
        "function g(o) { return o.x; }\n\
         for (var k = 0; k < 5; k++) g({ x: k });",
    );
    let chunk = first_stage(&mut engine, "g");
    let (pc, cache) = chunk
        .jit_ops()
        .iter()
        .enumerate()
        .find_map(|(pc, op)| match *op {
            Op::GetProp(_, cache) | Op::GetPropLocal(_, _, cache) => Some((pc, cache)),
            _ => None,
        })
        .expect("g's named read");
    let filled = |ways: [crate::bytecode::IcState; crate::bytecode::PROP_IC_WAYS]| {
        ways.iter()
            .filter(|way| way.depth != crate::bytecode::IC_EMPTY)
            .count()
    };
    let first_raw = Raw::take_in(&chunk, None);
    let first_view = chunk.freeze();
    let first = Snapshot::decode(&first_raw, &first_view);
    // A second receiver shape fills a second way.
    eval(
        &mut engine,
        "for (var k = 0; k < 5; k++) g({ y: 0, x: k });",
    );
    let second_raw = Raw::take_in(&chunk, None);
    let second_view = chunk.freeze();
    let second = Snapshot::decode(&second_raw, &second_view);
    // The first copy: one way, which the templates bake as the compact form, and the decode's.
    assert_eq!(first.named(pc).map(|named| named.filled().count()), Some(1));
    assert_eq!(filled(first_view.jit_cache_ways(cache)), 1);
    assert!(first_view.jit_cache_preferred(cache).is_some());
    // The second: both ways, in the templates and the decode alike, read at its own copy.
    assert_eq!(
        second.named(pc).map(|named| named.filled().count()),
        Some(2)
    );
    assert_eq!(filled(second_view.jit_cache_ways(cache)), 2);
    assert!(second_view.jit_cache_preferred(cache).is_none());
    assert_eq!(
        second_view.jit_cache_ways(cache).map(|way| way.recv_shape),
        chunk.jit_cache_ways(cache).map(|way| way.recv_shape),
        "the live ways at the second copy"
    );
    // A copy is a copy: the first is as it was.
    assert_eq!(filled(first_view.jit_cache_ways(cache)), 1);
    assert_eq!(Snapshot::decode(&first_raw, &first_view), first);
    check_decode(&chunk, &second_raw, &second_view);
}

#[test]
fn a_finished_compiles_memory_is_filled_again_by_the_next_preparation() {
    // The tier on, eager and synchronous: each compile ends at once and gives its frozen view and
    // raw copy back to this thread, whose next preparation fills the same vectors (point 5).
    TestSwitches {
        on: true,
        eager: true,
        asynchronous: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let compiled = || super::TEST_COMPILED.with(|compiled| compiled.borrow().len());
    eval(
        &mut engine,
        "function big(o, k) { var s = 0; s += o.a + o.b * k; s += o.c - k; s += o.a * o.c;\n\
           return s + (k & 7) + (o.b | 0); }\n\
         function small(x) { return x + 1; }",
    );
    let before = compiled();
    eval(
        &mut engine,
        "for (var k = 0; k < 2000; k++) big({ a: 1, b: 2, c: 3 }, k);",
    );
    assert!(compiled() > before, "big compiled");
    let (kept, first) = super::job_memories_for_test();
    assert_eq!(kept, 1, "its memory kept");
    let before = compiled();
    eval(&mut engine, "for (var k = 0; k < 2000; k++) small(k);");
    assert!(compiled() > before, "small compiled");
    assert_eq!(
        super::job_memories_for_test(),
        (1, first),
        "the next compile filled the same memory and gave it back"
    );
}
