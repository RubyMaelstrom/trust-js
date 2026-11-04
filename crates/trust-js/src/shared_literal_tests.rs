//! W2d step 4: a constant array literal's copies share one element block
//! ([`crate::value::word_block::KIND_COW`]), and every write copies it first. Each case writes
//! one copy in one way and checks the copy's result, a sibling copy and the site's next copy.
use super::*;
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn global(engine: &mut Engine, name: &str) -> Gc {
    let global = Value::Obj(engine.interp.global);
    engine
        .interp
        .get_member_key(&global, name)
        .unwrap_or_else(|_| panic!("{name}"))
        .as_obj()
        .unwrap_or_else(|| panic!("{name} is an object"))
}

/// The first element word's address and whether the block is shared.
fn block(engine: &mut Engine, name: &str) -> Option<(usize, bool)> {
    let handle = global(engine, name);
    let body = handle.borrow();
    body.props
        .words()
        .map(|words| (words.as_slice().as_ptr() as usize, words.is_shared()))
}

const WRITES: &[(&str, &str)] = &[
    ("a[0] = 9", "9,2,3"),
    ("a[1] = 'x'", "1,x,3"),
    ("a[2] = {}", "1,2,[object Object]"),
    ("a[2] = 2.5", "1,2,2.5"),
    ("a[3] = 4", "1,2,3,4"),
    ("a[5] = 6", "1,2,3,,,6"),
    ("a[0]++", "2,2,3"),
    ("a[1] += 10", "1,12,3"),
    ("a.push(4)", "1,2,3,4"),
    ("a.pop()", "1,2"),
    ("a.shift()", "2,3"),
    ("a.unshift(0)", "0,1,2,3"),
    ("a.splice(1, 1)", "1,3"),
    ("a.splice(1, 0, 7)", "1,7,2,3"),
    ("a.reverse()", "3,2,1"),
    ("a.sort(function (x, y) { return y - x; })", "3,2,1"),
    ("a.fill(0)", "0,0,0"),
    ("a.copyWithin(0, 1)", "2,3,3"),
    ("a.length = 1", "1"),
    ("a.length = 5", "1,2,3,,"),
    ("a.length = 0", ""),
    ("Object.defineProperty(a, 0, { value: 9 })", "9,2,3"),
    (
        "Object.defineProperty(a, 1, { get: function () { return 'g'; } })",
        "1,g,3",
    ),
    ("delete a[0]", ",2,3"),
    ("Object.assign(a, [7])", "7,2,3"),
    ("Reflect.set(a, 2, 5)", "1,2,5"),
    ("a.x = 1", "1,2,3"),
    ("for (var i = 0; i < 3; i++) { a[i] = i * 2; }", "0,2,4"),
    (
        "for (var i = 0; i < 3; i++) { a[i] = a[i] + 0.5; }",
        "1.5,2.5,3.5",
    ),
    ("Object.freeze(a); a[0] = 9", "1,2,3"),
    ("Object.seal(a); a[0] = 9", "9,2,3"),
    ("Object.preventExtensions(a); a[0] = 9", "9,2,3"),
];

/// Every write of one copy leaves the others as the literal made them, in every tier.
#[test]
fn every_write_of_a_shared_literal_copies_it_first() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for (write, expected) in WRITES {
            let mut engine = engine(tier);
            let source = format!(
                "function make() {{ return [1, 2, 3]; }}
                 function write(a) {{ {write}; return a; }}
                 var a = make(), b = make();
                 var written = String(write(a));
                 var c = make();
                 [written, String(b), String(c), Object.isFrozen(b), Object.isSealed(c),
                  Object.isExtensible(b), b.length, c.length].join('|')"
            );
            let result = eval(&mut engine, &source);
            let expected = format!("{expected}|1,2,3|1,2,3|false|false|true|3|3");
            assert_eq!(result, expected, "{tier:?}: {write}");
            // The sibling and the next copy still share the site's block.
            if tier != Tier::Interp {
                let (b, b_shared) = block(&mut engine, "b").expect("b has words");
                let (c, c_shared) = block(&mut engine, "c").expect("c has words");
                assert!(b_shared && c_shared && b == c, "{tier:?}: {write}");
                // The written copy shares the site's block only where its words did not change
                // (a longer `length`, a named property, preventExtensions): no other block is
                // ever shared.
                if let Some((a, a_shared)) = block(&mut engine, "a") {
                    assert!(
                        !a_shared || a == b,
                        "{tier:?}: {write}: another shared block"
                    );
                }
            }
        }
    }
}

/// A write repeated in a loop after the copy, and the copy's later writes, stay in its own
/// block: the second write of a copy finds it private.
#[test]
fn a_copied_literal_keeps_its_own_block() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        let result = eval(
            &mut engine,
            "function make() { return [0, 0, 0, 0]; }
             var rows = [];
             for (var r = 0; r < 50; r++) {
                 var row = make();
                 for (var i = 0; i < 4; i++) { row[i] = r + i; }
                 rows.push(row);
             }
             var fresh = make();
             String(rows[0]) + '|' + String(rows[49]) + '|' + String(fresh)",
        );
        assert_eq!(result, "0,1,2,3|49,50,51,52|0,0,0,0", "{tier:?}");
    }
}

/// Literal kinds: doubles, strings, mixed primitives, signed numbers and BigInts are shared;
/// holes, spreads, nested literals and computed elements are not.
#[test]
fn which_literals_share_a_block() {
    let mut engine = engine(Tier::Jit);
    let cases = [
        ("[1, 2, 3]", true),
        ("[0.5, -0, 1e300]", true),
        ("['a', 'b']", true),
        ("[1, 'a', true, null, undefined, 2n]", true),
        ("[-1, +2, -(3), (4)]", true),
        ("[1, , 3]", false),
        ("[...[1, 2]]", false),
        ("[[1], [2]]", false),
        ("[1, 1 + 1]", false),
        ("[]", false),
    ];
    for (literal, shared) in cases {
        eval(
            &mut engine,
            &format!("function make() {{ return {literal}; }} var a = make(), b = make();"),
        );
        let a = block(&mut engine, "a");
        let b = block(&mut engine, "b");
        match (a, b) {
            (Some((a, a_shared)), Some((b, b_shared))) => {
                assert_eq!(a_shared && b_shared && a == b, shared, "{literal}");
            }
            _ => assert!(!shared, "{literal}: no words"),
        }
    }
    assert_eq!(
        eval(
            &mut engine,
            "function f() { return [-0, 0.5, 'a', 2n, undefined, null]; }
             var x = f();
             [Object.is(x[0], -0), x[1], x[2], typeof x[3], x[4] === undefined, x[5] === null]
                 .join()"
        ),
        "true,0.5,a,bigint,true,true"
    );
}

/// The site's block outlives every copy and survives collections while the chunk lives; copies
/// keep it after their site's function is gone.
#[test]
fn the_shared_block_lives_while_its_site_or_a_copy_does() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function konst() { return ['alpha', 'beta', 2.5]; }
         var k1 = konst();",
    );
    engine.interp.gc_collect_minor();
    engine.interp.gc_collect();
    eval(&mut engine, "var k2 = konst(); k1 = null;");
    engine.interp.gc_collect();
    engine.interp.gc_collect_minor();
    assert_eq!(
        eval(&mut engine, "String(k2) + '|' + String(konst())"),
        "alpha,beta,2.5|alpha,beta,2.5"
    );
    // The function goes; its copy keeps the block and its strings.
    eval(&mut engine, "konst = null;");
    engine.interp.gc_collect();
    engine.interp.gc_collect();
    assert_eq!(eval(&mut engine, "String(k2)"), "alpha,beta,2.5");
}
