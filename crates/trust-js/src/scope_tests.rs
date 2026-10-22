//! S6 step 11b: the embed API's scoped values (`scope`). Their layout is the engine's, so a
//! native function is called as the engine calls its own; values live in scopes, and the lists
//! and handles that keep them past a call that collects; each scope allocates in its own
//! Engine's heap whatever another Engine did in between; the Realms an embedder enters are
//! kept, as roots, until it leaves them. What a scoped value cannot do is checked at compile
//! time: see the compile_fail examples of the `embed` module's documentation.

use std::cell::Cell;
use std::rc::Rc;

use crate::embed::{Ctx, Handle, NativeCaptures, Value, ValueList};
use crate::{Completion, Engine};

fn run(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("scope fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// The value of `source` in the scope `cx`, with no microtask checkpoint.
fn value_in<'s>(cx: &mut Ctx<'s>, source: &str) -> Value<'s> {
    cx.eval_value(source)
        .expect("scope fixture parses")
        .unwrap_or_else(|_| panic!("{source} threw"))
}

/// Every kind of engine value is the scoped value of its kind, payload for payload, and EMPTY
/// (a completion marker the engine never hands out) reads as undefined.
#[test]
fn scoped_values_are_the_engine_values() {
    use crate::value::Value as Internal;
    let mut engine = Engine::new();
    engine.scope(|cx| {
        let internal = [
            Internal::Undefined,
            Internal::Null,
            Internal::Bool(true),
            Internal::Num(-2.5),
            value_in(cx, "10n ** 20n").into_internal(),
            value_in(cx, "'text'").into_internal(),
            value_in(cx, "Symbol('described')").into_internal(),
            value_in(cx, "({})").into_internal(),
        ];
        for value in &internal {
            let scoped = Value::from_internal(value.clone());
            // The same tag word, and the same payload where there is one (a Bool's byte, every
            // other payload's word): the layout of `repr(u64)` with equal payloads.
            let (a, b) = (
                std::ptr::from_ref(value).cast::<u64>(),
                std::ptr::from_ref(&scoped).cast::<u64>(),
            );
            // SAFETY: the tag word is initialized in every value, and each payload read is
            // that of the value's variant.
            unsafe {
                assert_eq!(*a, *b, "the tag word");
                match value {
                    Internal::Undefined | Internal::Null => {}
                    Internal::Bool(_) => assert_eq!(
                        *a.add(1).cast::<u8>(),
                        *b.add(1).cast::<u8>(),
                        "the payload byte"
                    ),
                    _ => assert_eq!(*a.add(1), *b.add(1), "the payload word"),
                }
            }
            assert_eq!(scoped.type_of(), value.type_of());
            assert!(
                matches!(
                    (value, scoped.into_internal()),
                    (Internal::Num(a), Internal::Num(b)) if a.to_bits() == b.to_bits()
                ) || !matches!(value, Internal::Num(_))
            );
        }
        assert!(matches!(
            Value::from_internal(Internal::Empty),
            Value::Undefined
        ));
        let text = value_in(cx, "'text'");
        assert_eq!(text.as_text().as_deref(), Some("text"));
        let bigint = value_in(cx, "-(2n ** 70n) + 5n");
        assert_eq!(
            bigint.bigint_as_i64(),
            Some(5),
            "wrapping past 2^64, as the engine does"
        );
        let symbol = value_in(cx, "Symbol('described')");
        let Value::Sym(symbol) = symbol else {
            panic!("a symbol")
        };
        assert_eq!(symbol.description().as_deref(), Some("described"));
    });
}

/// `echo(x)`: its first argument, or (with a second argument) thrown.
fn echo<'s>(_: &mut Ctx<'s>, _: Value<'s>, args: &[Value<'s>]) -> Result<Value<'s>, Value<'s>> {
    let value = args.first().copied().unwrap_or_default();
    if args.len() > 1 {
        Err(value)
    } else {
        Ok(value)
    }
}

/// `kind()`: the receiver's kind, as the native function sees it.
fn kind<'s>(cx: &mut Ctx<'s>, this: Value<'s>, _: &[Value<'s>]) -> Result<Value<'s>, Value<'s>> {
    let kind = match this {
        Value::Undefined => "undefined",
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Num(_) => "number",
        Value::BigInt(_) => "bigint",
        Value::Str(_) => "string",
        Value::Sym(_) => "symbol",
        Value::Obj(_) => "object",
    };
    Ok(cx.str(kind))
}

/// A native function receives and returns (or throws) every kind of value as it is: the engine
/// calls the scoped function as one of its own, at every tier.
#[test]
fn natives_receive_return_and_throw_every_kind_of_value() {
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("echo", 2, echo);
        engine.define_global("kind", 0, kind);
        let source = r#"
            function check(ok, message) { if (!ok) throw new Error(message); }
            const values = [undefined, null, true, -0, 2.5, 7n, 'text', Symbol('s'), {}, [], echo];
            for (let round = 0; round < 50; round++) {
                for (const value of values) {
                    check(Object.is(echo(value), value), 'echo ' + String(typeof value));
                    let thrown;
                    try { echo(value, true); } catch (error) { thrown = error; }
                    check(Object.is(thrown, value), 'throw ' + String(typeof value));
                    const receiver = kind.call(value);
                    const expected = value === null ? 'null'
                        : typeof value === 'function' ? 'object' : typeof value;
                    check(receiver === expected, 'receiver ' + receiver + ' ' + expected);
                }
            }
            'ok'
        "#;
        assert_eq!(run(&mut engine, source), "ok", "{tier:?}");
    }
}

/// A list of values held across calls that run JavaScript and collect keeps every one (a
/// `Vec` of scoped values would not: it is not a root); the list reads as a slice and becomes
/// an Array.
#[test]
fn a_value_list_keeps_its_values_across_major_and_minor_collections() {
    let mut engine = Engine::new();
    let joined = engine.scope(|cx| {
        let mut list = ValueList::new();
        for index in 0..64 {
            let object = value_in(cx, &format!("({{tag: 'kept {index}'}})"));
            list.push(object);
            // Garbage between the pushes, and a collection of either kind.
            value_in(
                cx,
                "var churn = []; for (let i = 0; i < 500; i++) churn.push({i}); churn = null; 0",
            );
            if index % 8 == 0 {
                cx.collect_garbage_for_host();
            } else if index % 8 == 4 {
                cx.interp_mut().gc_collect_minor();
            }
        }
        cx.collect_garbage_for_host();
        assert_eq!(list.len(), 64);
        for (index, object) in list.iter().enumerate() {
            let tag = cx
                .member_get(object, "tag")
                .unwrap_or_else(|_| panic!("a kept object"));
            assert_eq!(tag.as_text().as_deref(), Some(&*format!("kept {index}")));
        }
        let array = list.into_array(cx);
        let global = cx.global_this();
        cx.member_set(&global, "kept", array)
            .unwrap_or_else(|_| panic!("publish"));
        let joined = value_in(cx, "kept.map(object => object.tag).slice(0, 3).join()");
        joined.as_text().map(|text| text.into_owned())
    });
    assert_eq!(joined.as_deref(), Some("kept 0,kept 1,kept 2"));
}

/// A list built while a [[Set]] runs a setter that collects (the builders' case): every value
/// pushed before the setter survives it.
#[test]
fn a_value_list_keeps_its_values_while_a_setter_collects() {
    fn collect<'s>(
        cx: &mut Ctx<'s>,
        _: Value<'s>,
        _: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        cx.collect_garbage_for_host();
        Ok(Value::Undefined)
    }
    let mut engine = Engine::new();
    engine.define_global("collect", 0, collect);
    let tags = engine.scope(|cx| {
        let target = value_in(
            cx,
            "({set field(v) { collect(); var churn = []; for (let i = 0; i < 2000; i++) churn.push({i}); collect(); }})",
        );
        let mut list = ValueList::with_capacity(16);
        for index in 0..16 {
            list.push(value_in(cx, &format!("({{tag: {index}}})")));
            cx.member_set(&target, "field", Value::Num(f64::from(index)))
                .unwrap_or_else(|_| panic!("the setter ran"));
        }
        list.iter()
            .map(|object| {
                cx.member_get(object, "tag")
                    .ok()
                    .and_then(|tag| tag.as_num_opt())
            })
            .collect::<Vec<_>>()
    });
    assert_eq!(
        tags,
        (0..16)
            .map(|index| Some(f64::from(index)))
            .collect::<Vec<_>>()
    );
}

/// Each scope allocates in its own Engine's heap, whatever Engine ran in between: an Engine's
/// entry points leave their heap active, and a scope's strings, objects and BigInts must not
/// land there.
#[test]
fn a_scope_allocates_in_its_own_heap_after_another_engine_ran() {
    let mut first = Engine::new();
    let mut second = Engine::new();
    first.scope(|a| {
        run(&mut second, "globalThis.other = {}; 0");
        second.scope(|b| {
            let _ = b.new_object();
            let _ = b.str("the second engine's");
        });
        // The values made here are the first Engine's: its handles accept them.
        let values = [
            Value::Obj(a.new_object()),
            a.str("a string of the first engine"),
            a.string(String::from("another")),
            a.bigint_from_i128(i128::MAX),
            a.make_error("TypeError", "made in the first engine"),
            a.make_array(vec![Value::Num(1.0)]),
        ];
        for value in &values {
            assert!(
                Handle::new(a, value).is_ok(),
                "{value:?} belongs to the first engine"
            );
        }
        run(&mut second, "other.again = {}; 0");
        let after = value_in(a, "({made: 'after the second ran'})");
        assert!(Handle::new(a, &after).is_ok());
    });
}

/// A native closure carries Rust state and engine captures both: the captures live as long as
/// the function, through minor and major collections, and nothing else holds them.
#[test]
fn a_native_closure_keeps_rust_state_and_engine_captures() {
    let mut engine = Engine::new();
    let calls = Rc::new(Cell::new(0));
    crate::gc_test::in_frame(|| {
        engine.scope(|cx| {
            let captured = value_in(cx, "({tag: ['cap', 'tured'].join('')})");
            let counter = calls.clone();
            let function = cx.new_native_closure_with_captures(
                "readCapture",
                0,
                move |cx, _, _, captures| {
                    counter.set(counter.get() + 1);
                    cx.member_get(&captures[0], "tag")
                },
                &[captured],
                NativeCaptures::None,
            );
            let global = cx.global_this();
            cx.member_set(&global, "readCapture", function)
                .unwrap_or_else(|_| panic!("install"));
        });
    });
    for _ in 0..2 {
        run(
            &mut engine,
            "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
        );
        engine.interp.gc_collect_minor();
        engine.interp.gc_collect();
    }
    assert_eq!(
        run(&mut engine, "readCapture() + readCapture()"),
        "capturedcaptured"
    );
    assert_eq!(calls.get(), 2);
}

/// The Realm an embedder replaced by entering another stays a root of every collection until
/// the entry is left, even when nothing else names it: the entry is a token, and the Engine
/// keeps the replaced Realm's state.
#[test]
fn a_replaced_realm_is_kept_until_its_entry_is_left() {
    let mut engine = Engine::new();
    let (outer, inner) = crate::gc_test::in_frame(|| {
        engine.scope(|cx| {
            let replaced = cx.create_embed_realm();
            let entered = cx.create_embed_realm();
            cx.with_embed_realm(&replaced, |ctx| {
                assert!(matches!(
                    ctx.eval_classic_script_interruptible(
                        "globalThis.marker = {realm: ['re', 'placed'].join('')};"
                    ),
                    Ok(Ok(_))
                ));
            })
            .unwrap_or_else(|_| panic!("set up the replaced realm"));
            let outer = cx.enter_realm(&replaced).expect("enter the replaced realm");
            let inner = cx.enter_realm(&entered).expect("enter another");
            (outer, inner)
        })
    });
    for _ in 0..2 {
        run(
            &mut engine,
            "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
        );
        engine.interp.gc_collect_minor();
        engine.collect_garbage_at_idle();
    }
    engine.leave_realm(inner);
    assert_eq!(run(&mut engine, "globalThis.marker.realm"), "replaced");
    assert_eq!(run(&mut engine, "[1, 2].map(x => x * 2).join()"), "2,4");
    engine.leave_realm(outer);
    assert_eq!(run(&mut engine, "typeof globalThis.marker"), "undefined");
}

/// Entries are left innermost first, by their own Engine.
#[test]
#[should_panic(expected = "innermost first")]
fn leaving_an_outer_realm_entry_first_panics() {
    let mut engine = Engine::new();
    let (outer, _inner) = engine.scope(|cx| {
        let first = cx.create_embed_realm();
        let second = cx.create_embed_realm();
        (
            cx.enter_realm(&first).expect("enter"),
            cx.enter_realm(&second).expect("enter"),
        )
    });
    engine.leave_realm(outer);
}

/// An entry of another Engine is refused.
#[test]
#[should_panic(expected = "by its own Engine")]
fn leaving_another_engines_realm_entry_panics() {
    let mut first = Engine::new();
    let mut second = Engine::new();
    let entry = first.scope(|cx| {
        let realm = cx.create_embed_realm();
        cx.enter_realm(&realm).expect("enter")
    });
    second.leave_realm(entry);
}
