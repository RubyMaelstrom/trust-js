//! The snapshot in the exemplars' shape (P3 stage 2, step 16): a chunk carries its names' keys (a
//! first stage's table made at its first freeze where the VM has not made it, a second stage's
//! carried at its recompile from the tables of the chunks it was made from, a name no table
//! holds made there).

use super::TestSwitches;
use crate::atom::PropKey;
use crate::bytecode::{Chunk, Tier};
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
