//! Built-in accessors read in an inlined callee at graph (P3 stage 3, the getter line): a typed
//! array's `length` (its way, a1; the landing's in-line answer, b) and a Map's `size` (the
//! direct native call, a2) in a callee the caller inlines, where the read's generic node runs its
//! template and switches its frame's context lazily around the helper call (graph-next piece 1).
//! A receiver that is no typed array or Map but shares their ways' shapes reaches the landing,
//! which declines it, and the native getter's TypeError, thrown from the helper in the inlined
//! frame, carries both frames in its stack. The values are Node's.

use super::TestSwitches;
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

const PROGRAM: &str = r#"
function inner(o) { return o.length; }
function outer(o) { return inner(o) + 1; }
function innerSize(m) { return m.size; }
function outerSize(m) { return innerSize(m) + 1; }
var ta = new Uint8Array(5), m = new Map([[1, 1]]), t = 0;
for (var r = 0; r < 20000; r++) t += outer(ta) + outerSize(m);
var fake = Object.setPrototypeOf({}, Uint8Array.prototype);
var fakeMap = Object.setPrototypeOf({}, Map.prototype);
var s1 = '', s2 = '', n = 0;
for (var r = 0; r < 200; r++) {
  try { outer(fake); } catch (e) { if (e instanceof TypeError) { n++; s1 = String(e.stack); } }
  try { outerSize(fakeMap); } catch (e) { if (e instanceof TypeError) { n++; s2 = String(e.stack); } }
  t += outer(ta) + outerSize(m);
}
t + '|' + n + '|' + /inner[\s\S]*outer/.test(s1) + '|' + /innerSize[\s\S]*outerSize/.test(s2)
"#;

#[test]
fn builtin_getters_in_an_inlined_callee_at_graph() {
    TestSwitches {
        on: true,
        inline_graph: Some(true),
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
    assert_eq!(eval(&mut engine, PROGRAM), "161600|400|true|true");
    assert!(
        super::TEST_COMPILED.with(|compiled| !compiled.borrow().is_empty()),
        "the tier compiled"
    );
}
