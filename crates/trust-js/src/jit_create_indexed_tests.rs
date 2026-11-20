//! A generated creation on a map of `INDEX_THRESHOLD` (8) entries or more (the megamorphic line's
//! creations past eight entries; `emit_prop_create_probe`'s `indexed`): the template's creation
//! commit takes the store where the key layout the map holds after the append carries its shared
//! index, at a site's own way and through the store table, and leaves it to the helper where the
//! layout carries none or the map owns an index. After an in-line append every key of the map is
//! found at its slot through the layout's index, and a map that shared the layout is untouched.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier, warm: bool) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(if warm { 1 } else { 0 });
    if warm {
        engine.interp.jit_admit_runs = 20;
    }
    engine
}

fn helpers() -> usize {
    crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get)
}

/// See `mega_store_tests::alone`: a test that counts store helper calls runs in a child.
fn alone(name: &str) -> bool {
    super::construct_tests::run_alone(&format!("jit::create_indexed_tests::{name}"))
}

/// The index state of the global object `name`'s map (`Props::test_index_state`): the layout
/// carries an index, the map owns one, every key is found through the layout's.
fn index_state(engine: &mut Engine, name: &str) -> (bool, bool, bool) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let object = value.as_obj().expect("an object");
    let state = object.borrow().props.test_index_state();
    state
}

const LOOK: &str = r#"
function look(o) {
  var ks = Object.keys(o), fi = [], s = 0, has = 0;
  for (var k in o) fi.push(k);
  for (var i = 0; i < ks.length; i++) { s += o[ks[i]]; has += (ks[i] in o) + o.hasOwnProperty(ks[i]); }
  return ks.join(',') + '|' + fi.join(',') + '|' + s + '|' + has + '|' + JSON.stringify(o) + '|' + ('zz' in o);
}
"#;

/// A constructor of `fields` fields (each store a way of its own site), and what every lookup of
/// an instance answers.
fn mono_source(fields: usize) -> String {
    let body: String = (0..fields)
        .map(|k| format!("this.m{k} = k + {k};"))
        .collect();
    format!(
        "function Mono(k) {{ {body} }}\n{LOOK}\n\
         function mono(n) {{ var o; for (var i = 0; i < n; i++) o = new Mono(i); return look(o); }}\n\
         mono(64);"
    )
}

/// One megamorphic creation site (`add`'s `o.g = v`) over receivers of one shape of `entries`
/// entries under eight prototypes, so that the store table serves it; the receivers are made
/// beforehand, so a pass over a pool stores at that site alone.
fn mega_source(entries: usize) -> String {
    let body: String = (0..entries).map(|k| format!("o.m{k} = k + {k};")).collect();
    format!(
        "var protos = []; for (var k = 0; k < 8; k++) protos.push({{ ['p' + k]: k }});\n\
         function make(k) {{ var o = Object.create(protos[k]); {body} return o; }}\n\
         function pool(n) {{ var a = []; for (var i = 0; i < n; i++) a.push(make(i & 7)); return a; }}\n\
         function add(o, v) {{ o.g = v; }}\n{LOOK}\n\
         function fill(a) {{ for (var i = 0; i < a.length; i++) add(a[i], i); return look(a[a.length - 1]); }}\n\
         fill(pool(64)); var p1 = pool(80), p2 = pool(80);"
    )
}

/// `steps` on every tier; in generated code, cold and warm, the last step's store helper calls.
fn agree(source: &str, steps: [&str; 2], what: &str) -> [usize; 2] {
    let run = |engine: &mut Engine| {
        eval(engine, source);
        let first = eval(engine, steps[0]);
        let before = helpers();
        let second = eval(engine, steps[1]);
        ([first, second], helpers() - before)
    };
    let expected = run(&mut engine(Tier::Interp, false)).0;
    assert_eq!(
        run(&mut engine(Tier::Bytecode, false)).0,
        expected,
        "the VM, {what}"
    );
    [false, true].map(|warm| {
        let (values, calls) = run(&mut engine(Tier::Jit, warm));
        assert_eq!(values, expected, "generated code (warm {warm}), {what}");
        calls
    })
}

/// A creation landing on 8 to 16 entries (`SHARED_LAYOUT_MAX_FIELDS`) is made in generated code,
/// at a site's own way (constructors of 7, 8, 9, 15 and 16 fields: a second pass makes no store
/// helper call) and through the store table (a megamorphic site over maps of 7, 8, 9 and 15
/// entries); one landing past 16, where no layout carries an index, takes the helper every time
/// (constructors of 17 and 33 fields; a megamorphic site over maps of 16 entries). Every tier
/// agrees on every lookup of the instance.
#[test]
fn creations_past_eight_entries_are_made_in_generated_code() {
    crate::opt::pin_tier_off_for_test();
    if !alone("creations_past_eight_entries_are_made_in_generated_code") {
        return;
    }
    for fields in [7usize, 8, 9, 15, 16, 17, 33] {
        let calls = agree(
            &mono_source(fields),
            ["mono(80)", "mono(80)"],
            "a constructor",
        );
        let past = 80 * fields.saturating_sub(16);
        assert_eq!(calls, [past, past], "a constructor of {fields} fields");
    }
    for entries in [7usize, 8, 9, 15, 16] {
        let calls = agree(
            &mega_source(entries),
            ["fill(p1)", "fill(p2)"],
            "a megamorphic site",
        );
        let past = if entries < 16 { 0 } else { 80 };
        assert_eq!(
            calls,
            [past, past],
            "a megamorphic site over {entries} entries"
        );
    }
}

const PAIR: &str = r#"
function Mono(k) { this.m0 = k; this.m1 = 1; this.m2 = 2; this.m3 = 3; this.m4 = 4; this.m5 = 5; this.m6 = 6; this.m7 = 7; }
function ninth(o, v) { o.g8 = v; }
function tenth(o, v) { o.g9 = v; }
function look(o) { return Object.keys(o).join(',') + '|' + ('g8' in o) + ('g9' in o) + '|' + o.g8 + ',' + o.g9 + ',' + o.m7; }
var early = new Mono(9);
for (var i = 0; i < 40; i++) { var w = new Mono(i); ninth(w, i); tenth(w, i); }
var a = new Mono(1), b = new Mono(2), c = new Mono(3), d = new Mono(4);
"#;

/// THE INVARIANT the generated commit relies on (see `emit_prop_create_probe`'s `indexed`): after
/// the in-line append every key of the map is found at its slot through the layout's shared
/// index, and a map of the shape it left, which shared its layout, is untouched: it lacks the
/// key, and gains it later the same way.
#[test]
fn after_an_in_line_append_every_key_is_found_through_the_layouts_index() {
    crate::opt::pin_tier_off_for_test();
    if !alone("after_an_in_line_append_every_key_is_found_through_the_layouts_index") {
        return;
    }
    for warm in [false, true] {
        let mut jit = engine(Tier::Jit, warm);
        eval(&mut jit, PAIR);
        let before = helpers();
        eval(&mut jit, "ninth(a, 10)");
        assert_eq!(helpers() - before, 0, "the ninth key, in line");
        assert_eq!(index_state(&mut jit, "a"), (true, false, true));
        assert_eq!(
            eval(&mut jit, "look(a) + ' ' + look(b)"),
            "m0,m1,m2,m3,m4,m5,m6,m7,g8|truefalse|10,undefined,7 \
             m0,m1,m2,m3,m4,m5,m6,m7|falsefalse|undefined,undefined,7"
        );
        eval(&mut jit, "tenth(a, 11); ninth(b, 12)");
        assert_eq!(
            helpers() - before,
            0,
            "the tenth key, and the other map's ninth, in line"
        );
        assert_eq!(index_state(&mut jit, "a"), (true, false, true));
        assert_eq!(index_state(&mut jit, "b"), (true, false, true));
        assert_eq!(
            eval(&mut jit, "look(a) + ' ' + look(b)"),
            "m0,m1,m2,m3,m4,m5,m6,m7,g8,g9|truetrue|10,11,7 \
             m0,m1,m2,m3,m4,m5,m6,m7,g8|truefalse|12,undefined,7"
        );
        // A map made before any ninth key was seen holds the eight-key layout, which predicts
        // nothing and carries no index: the landing shape's layout lands, and its index is the
        // one tested.
        assert_eq!(index_state(&mut jit, "early"), (false, false, false));
        eval(&mut jit, "ninth(early, 13)");
        assert_eq!(helpers() - before, 0, "the landing layout's index");
        assert_eq!(index_state(&mut jit, "early"), (true, false, true));
        assert_eq!(
            eval(&mut jit, "look(early)"),
            "m0,m1,m2,m3,m4,m5,m6,m7,g8|truefalse|13,undefined,7"
        );
    }
}

/// A map whose layout predicts the ninth key and carries no index (as a layout past
/// `SHARED_LAYOUT_MAX_FIELDS` keys carries none) is refused: the helper appends and builds the
/// map an index of its own. That map, which now owns an index, is refused at the next creation
/// too: the helper's insert lands the shape's shared layout and drops the map's own index.
#[test]
fn a_map_without_a_shared_index_or_with_its_own_takes_the_helper() {
    crate::opt::pin_tier_off_for_test();
    if !alone("a_map_without_a_shared_index_or_with_its_own_takes_the_helper") {
        return;
    }
    for warm in [false, true] {
        let mut jit = engine(Tier::Jit, warm);
        eval(&mut jit, PAIR);
        eval(&mut jit, "ninth(d, 0)");
        {
            let env = jit.interp.global_env;
            let c = jit.interp.get_var("c", &env).ok().expect("c");
            let d = jit.interp.get_var("d", &env).ok().expect("d");
            let (c, d) = (c.as_obj().expect("c"), d.as_obj().expect("d"));
            let longer = d.borrow();
            c.borrow_mut()
                .props
                .test_take_layout_without_index(&longer.props);
        }
        assert_eq!(index_state(&mut jit, "c"), (false, false, false));
        let before = helpers();
        eval(&mut jit, "ninth(c, 20)");
        assert_eq!(helpers() - before, 1, "a layout with no index: the helper");
        assert_eq!(index_state(&mut jit, "c"), (false, true, false));
        eval(&mut jit, "tenth(c, 21)");
        assert_eq!(
            helpers() - before,
            2,
            "an index of the map's own: the helper"
        );
        assert_eq!(index_state(&mut jit, "c"), (true, false, true));
        assert_eq!(
            eval(&mut jit, "look(c)"),
            "m0,m1,m2,m3,m4,m5,m6,m7,g8,g9|truetrue|20,21,7"
        );
    }
}
