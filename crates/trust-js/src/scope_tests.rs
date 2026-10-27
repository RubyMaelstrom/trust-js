//! The embed API's scoped execution words (`scope`), called at the callback's declared Rust
//! signature with explicit Result mapping. Values live in scopes, and the lists
//! and handles that keep them past a call that collects; each scope allocates in its own
//! Engine's heap whatever another Engine did in between; the Realms an embedder enters are
//! kept, as roots, until it leaves them. What a scoped value cannot do is checked at compile
//! time: see the compile_fail examples of the `embed` module's documentation.

use crate::scope::ValueKind;
use crate::value::ValueView;
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

#[test]
fn scoped_text_borrows_survive_copied_frontier_append_growth_and_gc() {
    fn check<'s>(cx: &mut Ctx<'s>) {
        use crate::lstr::LStr;
        use crate::rooting::Rooted;
        use crate::value::Value as Internal;
        use std::borrow::Cow;

        let prefix = "scoped borrowed engine text: ".repeat(64);
        let initial = LStr::from(prefix.as_str()).concat_grown("");
        let owner = Rooted::new(Internal::Str(initial));
        let original = Value::<'s>::from_internal(*owner);
        let borrowed = original.as_engine_text().unwrap();
        let cow = original.as_text().unwrap();
        assert!(matches!(cow, Cow::Borrowed(_)));
        let original_address = borrowed.as_ptr();
        let original_len = borrowed.len();
        assert_eq!(cow.as_ptr(), original_address);

        let mut copied = initial;
        assert!(copied.append_in_place("frontier"));
        assert!(LStr::ptr_eq(copied.view_owner().unwrap(), &initial));
        assert_eq!(copied.text().as_ptr(), original_address);
        let mut current = Rooted::new(Internal::Str(copied));
        let mut expected = format!("{prefix}frontier");
        cx.collect_garbage_for_host();
        assert_eq!(borrowed, prefix);
        assert_eq!(borrowed.as_ptr(), original_address);
        assert_eq!(cow.as_ref(), prefix);
        assert_eq!(cow.as_ptr(), original_address);
        assert_eq!(current.as_engine_text().as_deref(), Some(expected.as_str()));

        let suffix = "replacement growth: ".repeat(prefix.len());
        assert!(!copied.append_in_place(&suffix));
        copied.append_or_replace(&LStr::from(suffix.as_str()));
        assert_ne!(copied.text().as_ptr(), original_address);
        current.replace(Internal::Str(copied));
        expected.push_str(&suffix);
        cx.collect_garbage_for_host();
        assert_eq!(borrowed, prefix);
        assert_eq!(borrowed.len(), original_len);
        assert_eq!(borrowed.as_ptr(), original_address);
        assert_eq!(cow.as_ref(), prefix);
        assert_eq!(cow.len(), original_len);
        assert_eq!(cow.as_ptr(), original_address);
        assert_eq!(current.as_engine_text().as_deref(), Some(expected.as_str()));
    }
    Engine::new().scope(check);
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
            let scoped = Value::from_internal(*value);
            assert_eq!(std::mem::size_of_val(value), 8);
            assert_eq!(std::mem::size_of_val(&scoped), 8);
            let (a, b) = (
                std::ptr::from_ref(value).cast::<u64>(),
                std::ptr::from_ref(&scoped).cast::<u64>(),
            );
            // SAFETY: each transparent execution word is fully initialized, with no padding.
            assert_eq!(unsafe { *a }, unsafe { *b }, "the complete canonical word");
            match (value.view(), scoped.kind()) {
                (ValueView::Undefined, ValueKind::Undefined)
                | (ValueView::Null, ValueKind::Null) => {}
                (ValueView::Bool(a), ValueKind::Bool(b)) => assert_eq!(a, b),
                (ValueView::Num(a), ValueKind::Num(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                (ValueView::BigInt(_), ValueKind::BigInt(v)) => assert_eq!(
                    Value::BigInt(v).into_internal().cell_word(),
                    value.cell_word()
                ),
                (ValueView::Str(_), ValueKind::Str(v)) => {
                    assert_eq!(Value::Str(v).into_internal().cell_word(), value.cell_word())
                }
                (ValueView::Sym(_), ValueKind::Sym(v)) => {
                    assert_eq!(Value::Sym(v).into_internal().cell_word(), value.cell_word())
                }
                (ValueView::Obj(_), ValueKind::Obj(v)) => {
                    assert_eq!(Value::Obj(v).into_internal().cell_word(), value.cell_word())
                }
                _ => panic!("scoped view changed the value's discriminant"),
            }
            assert_eq!(scoped.type_of(), value.type_of());
            assert!(
                matches!(
                    (value.view(), scoped.into_internal().view()),
                    (ValueView::Num(a), ValueView::Num(b)) if a.to_bits() == b.to_bits()
                ) || !matches!(&(value).view(), ValueView::Num(_))
            );
        }
        assert!(matches!(
            (Value::from_internal(Internal::Empty)).kind(),
            ValueKind::Undefined
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
        let ValueKind::Sym(symbol) = (symbol).kind() else {
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
    let kind = match (this).kind() {
        ValueKind::Undefined => "undefined",
        ValueKind::Null => "null",
        ValueKind::Bool(_) => "boolean",
        ValueKind::Num(_) => "number",
        ValueKind::BigInt(_) => "bigint",
        ValueKind::Str(_) => "string",
        ValueKind::Sym(_) => "symbol",
        ValueKind::Obj(_) => "object",
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

/// Obtain the real scoped pointer opaquely so optimizers cannot hide a wrong aggregate ABI
/// behind an inlined conversion. All calls retain this original declared Result signature.
#[inline(never)]
fn opaque_echo() -> crate::embed::NativeFn {
    std::hint::black_box(echo as crate::embed::NativeFn)
}

#[test]
fn bare_scoped_native_keeps_its_signature_in_native_and_prepared_caches() {
    const CHILD: &str = "TRUST_JS_TEST_SCOPED_NATIVE_SIGNATURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "scope_tests::bare_scoped_native_keeps_its_signature_in_native_and_prepared_caches",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use crate::bytecode::{CallIc, CallSite, CALL_IC_SCOPED_NATIVE};
    use crate::value::{Callable, Value as Internal};
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    let function = opaque_echo();
    engine.define_global("opaqueEcho", 2, function);
    let global = engine.interp.global;
    engine
        .interp
        .def_method(&global, "scopedNativeHits", 0, |_, _, _| {
            Ok(Internal::Num(if cfg!(target_arch = "aarch64") {
                crate::bytecode::TEST_SCOPED_NATIVE_IC_COMMITS.with(Cell::get)
            } else {
                crate::bytecode::TEST_SCOPED_NATIVE_CACHED_COMMITS.with(Cell::get)
            } as f64))
        });
    assert_eq!(
        run(
            &mut engine,
            r#"
        function through(value, fail) {
            try {
                if (fail) return opaqueEcho(value, true);
                return opaqueEcho(value);
            } catch (error) { return error; }
        }
        var words = [undefined, null, false, true, -0, 1.5, Infinity, -Infinity, NaN,
                     1n << 80n, 'opaque text', Symbol('opaque'), {}, opaqueEcho];
        for (var n = 0; n < 64; n++) {
            for (var k = 0; k < words.length; k++) {
                if (!Object.is(through(words[k], true), words[k])) throw 'typed throw changed';
                if (!Object.is(through(words[k], false), words[k])) throw 'typed return changed';
            }
        }
        'ok'
    "#
        ),
        "ok"
    );
    // ARM emits H_CALL_HIT probes; x64 enters H_CALL, whose warmed path calls call_jit_cached.
    // Count actual typed callback completions in each backend's own cached funnel, before the
    // manual cached/prepared probes below can contribute to either counter.
    {
        assert_eq!(
            run(
                &mut engine,
                r#"
        for (var k = 0; k < words.length; k++) {
            for (var warm = 0; warm < 4; warm++) {
                through(words[k], true); through(words[k], false);
            }
            var before = scopedNativeHits();
            for (var n = 0; n < 16; n++) {
                if (!Object.is(through(words[k], true), words[k])) throw 'committed throw changed';
                if (!Object.is(through(words[k], false), words[k])) throw 'committed return changed';
            }
            if (scopedNativeHits() - before !== 32) throw 'scoped native hit path did not commit for kind '+k+': '+(scopedNativeHits()-before);
        }
        'ok'
        "#
            ),
            "ok"
        );
    }
    let global = engine.interp.global.borrow();
    let callable = global.props.get("opaqueEcho").unwrap().value();
    let through = global.props.get("through").unwrap().value();
    drop(global);
    let callable_object = callable.as_obj().unwrap();
    assert!(matches!(&callable_object.borrow().call,
        Callable::ScopedNative(actual) if *actual as usize == function as usize));
    let user = match &through.as_obj().unwrap().borrow().call {
        Callable::User(user) => *user,
        _ => panic!("compiled caller"),
    };
    let runtime = user.func.runtime();
    let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
    assert!(chunk.jit.get().is_some_and(|code| code.is_some()));
    let entry = chunk
        .cached_call_for(crate::value::Gc::as_ptr(&callable_object) as usize)
        .expect("the actual scoped native call cache was filled");
    assert_eq!(entry.direct, CALL_IC_SCOPED_NATIVE);
    assert_eq!(entry.intrinsic, 0);
    assert!(entry.chunk_raw.is_null());
    assert!(entry.chunk.is_null());
    assert!(entry.code.is_null());
    let site = CallSite::empty();
    site.fill(entry, None);
    let prepared = engine
        .interp
        .prepare_call(&site, &callable)
        .expect("typed preparation");
    let this = Internal::Undefined;
    for argument in [
        Internal::Undefined,
        Internal::Null,
        Internal::Bool(true),
        Internal::Num(-0.0),
        Internal::Num(f64::NAN),
    ] {
        // Both the normal cache hit and a prepared call must preserve the actual signature.
        let arguments = [(argument)];
        let normal = unsafe {
            engine
                .interp
                .call_jit_cached(&site, &callable, &this, arguments.as_ptr(), 1)
        }
        .expect("cache hit")
        .unwrap_or_else(|_| panic!("normal return"));
        assert_eq!(normal.bits(), argument.bits());
        let prepared_result = unsafe {
            engine
                .interp
                .call_prepared(&prepared, &this, arguments.as_ptr(), 1)
        }
        .unwrap_or_else(|_| panic!("prepared return"));
        assert_eq!(prepared_result.bits(), argument.bits());
    }
    assert_ne!(entry.callee, CallIc::EMPTY.callee);
}

/// Deliberately bypass the normal constructor to exercise public EMPTY normalization in
/// release builds too. The transparent word is valid internally, but public reads are undefined.
#[inline(never)]
fn raw_scoped_empty<'s>() -> Value<'s> {
    unsafe { std::mem::transmute::<crate::value::Value, Value<'s>>(crate::value::Value::Empty) }
}

#[inline(never)]
fn empty_bare<'s>(
    cx: &mut Ctx<'s>,
    this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    assert!(this.is_undefined());
    assert!(matches!(this.kind(), ValueKind::Undefined));
    assert_eq!(this.into_internal().bits(), crate::value::PACK_UNDEFINED);
    for argument in args {
        assert!(argument.is_undefined());
        assert!(argument.is_nullish());
        assert!(matches!(argument.kind(), ValueKind::Undefined));
        assert_eq!(cx.type_of(argument), "undefined");
        assert!(cx.coerce_number(argument)?.is_nan());
        assert_eq!(&*cx.coerce_string(argument)?, "undefined");
        assert!(cx.values_strict_equal(argument, &Value::Undefined));
        assert_eq!(
            argument.into_internal().bits(),
            crate::value::PACK_UNDEFINED
        );
    }
    if args.len() == 2 {
        Err(raw_scoped_empty())
    } else {
        Ok(raw_scoped_empty())
    }
}

#[test]
fn bare_scoped_empty_is_undefined_in_inputs_outputs_forwarding_and_handles() {
    use crate::value::{Callable, NativeCallable, NativeCallableBody, Value as Internal};
    let mut engine = Engine::new();
    let function = std::hint::black_box(empty_bare as crate::embed::NativeFn);
    let this = Internal::Empty;
    for count in [1, 2] {
        let arguments = vec![Internal::Empty; count];
        let result = unsafe {
            engine.interp.call_native_address_committed(
                function as usize,
                crate::bytecode::CALL_IC_SCOPED_NATIVE,
                &this,
                arguments.as_ptr(),
                count,
            )
        };
        let result = match result {
            Ok(value) if count == 1 => value,
            Err(crate::interpreter::Abrupt::Throw(value)) if count == 2 => value,
            _ => panic!("the actual scoped Result branch changed"),
        };
        assert_eq!(result.bits(), crate::value::PACK_UNDEFINED);
        assert!(
            arguments.iter().all(|value| value.is_empty()),
            "normalization must not rewrite caller storage"
        );
    }
    #[inline(never)]
    fn captures<'s>(
        _: &mut Ctx<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
        captures: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        assert!(this.is_undefined());
        assert_eq!(args.len(), 1);
        assert_eq!(captures.len(), 2);
        assert!(args.iter().chain(captures).all(Value::is_undefined));
        assert!(args
            .iter()
            .chain(captures)
            .all(|value| value.into_internal().bits() == crate::value::PACK_UNDEFINED));
        Ok(raw_scoped_empty())
    }
    let call = Callable::native_data(Rc::new(NativeCallable {
        body: NativeCallableBody::ScopedCaptured {
            function: std::hint::black_box(captures as crate::embed::NativeCaptureFn),
            captures: vec![Internal::Empty, Internal::Empty].into_boxed_slice(),
        },
        retained: None,
        identity: Rc::from("empty captures"),
    }));
    assert_eq!(
        engine
            .interp
            .dispatch_native(&call, Internal::Empty, &[Internal::Empty])
            .unwrap_or_else(|_| panic!("captured call"))
            .bits(),
        crate::value::PACK_UNDEFINED
    );
    engine.scope(|cx| {
        let raw = raw_scoped_empty();
        assert!(raw.is_undefined());
        assert_eq!(raw.type_of(), "undefined");
        assert!(cx.values_strict_equal(&raw, &Value::Undefined));
        let handle = Handle::new(cx, &raw).unwrap();
        assert_eq!(handle.get(cx).unwrap().into_internal().bits(), crate::value::PACK_UNDEFINED);
        let mut list = ValueList::new(); list.push(raw); list.insert(0, raw); list.extend([raw]);
        assert!(list.iter().all(|value| value.into_internal().bits() == crate::value::PACK_UNDEFINED));
        let array = cx.make_array_of(&[raw]);
        let global = cx.global_this();
        cx.member_set(&global, "emptyForwarded", array).unwrap();
        cx.member_set(&global, "emptyProperty", raw).unwrap();
        let forwarded = value_in(cx, "0 in emptyForwarded && emptyForwarded.length === 1 && emptyForwarded[0] === undefined && emptyProperty === undefined");
        assert!(matches!(forwarded.kind(), ValueKind::Bool(true)));
        let check = value_in(cx, "function check(v) { return arguments.length === 1 && v === undefined; }; check");
        assert!(matches!(cx.invoke(check, raw, &[raw]).unwrap().kind(), ValueKind::Bool(true)));
        let captured = cx.new_native_fn_with_captures("forwardedCapture", 0,
            |_, _, _, captures| Ok(captures[0]), &[raw]);
        assert_eq!(cx.invoke(captured, raw, &[]).unwrap().into_internal().bits(), crate::value::PACK_UNDEFINED);
    });
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn committed_scoped_calls_loan_the_original_window_through_gc_and_pool_churn() {
    use crate::gc_core::CollectionKind;
    use crate::value::Value as Internal;
    #[derive(Clone)]
    struct Expected {
        address: usize,
        // Obscure words so neither diagnostic state nor conservative scanning can retain the
        // argument cells. Real precise collections below see only the boundary's slice root.
        hidden: Vec<u64>,
        weak: Vec<crate::gc_handle::WeakRef>,
        calls: usize,
    }
    #[inline(never)]
    fn check<'s>(
        cx: &mut Ctx<'s>,
        _: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let expected = cx.host::<Expected>().unwrap().clone();
        assert_eq!(
            args.as_ptr() as usize,
            expected.address,
            "the canonical window was copied"
        );
        assert_eq!(args.len(), expected.hidden.len());
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            cx.interp_mut().gc_collect_precise_for_test(kind);
            assert!(
                expected.weak.iter().all(|weak| weak.live_ptr().is_some()),
                "the live loan lost an argument cell"
            );
            for (value, hidden) in args.iter().zip(&expected.hidden) {
                assert_eq!(value.into_internal().bits(), !hidden);
            }
        }
        // Enter and retire over 256 distinct compiled frames, then invoke an oversized frame
        // and an abrupt call while this native still lends the original heap-backed arguments.
        fn chunk(cx: &mut Ctx<'_>, name: &str) -> Rc<crate::bytecode::Chunk> {
            let global = cx.interp_mut().global_env;
            let value = cx
                .interp_mut()
                .get_var(name, &global)
                .unwrap_or_else(|_| panic!("native churn function"));
            let object = value.as_obj().unwrap();
            let call = object.borrow();
            let crate::value::Callable::User(user) = &call.call else {
                panic!("user churn function");
            };
            let runtime = user.func.runtime();
            let chunk = runtime
                .execution_code()
                .and_then(Option::as_ref)
                .expect("compiled churn body")
                .clone();
            assert!(chunk.jit.get().flatten().is_some());
            chunk
        }
        value_in(
            cx,
            r#"
            function loanRecurse(n) { if (n) return loanRecurse(n-1)+1; return 0; }
            function loanThrow(n) { if (n) return loanThrow(n-1); throw 19; }
            loanRecurse(0)
        "#,
        );
        let recursive = chunk(cx, "loanRecurse");
        recursive.jit_runs.set(0);
        // Keep inline-retry accounting active without replacing this exact native image.
        recursive.inline_retry_at.set(u32::MAX);
        assert_eq!(value_in(cx, "loanRecurse(280)").as_num_opt(), Some(280.0));
        assert!(
            recursive.jit_runs.get() >= 281,
            "the exact non-tail recursive chunk must enter >256 native frames, got {}",
            recursive.jit_runs.get()
        );
        value_in(
            cx,
            "try {loanThrow(24)} catch(error) {if(error!==19)throw error}",
        );
        let mut oversized = String::from("function loanOversized(){let ");
        for n in 0..620 {
            if n != 0 {
                oversized.push(',');
            }
            oversized.push_str(&format!("local{n}={n}"));
        }
        oversized.push_str(";return local619};loanOversized()");
        assert_eq!(value_in(cx, &oversized).as_num_opt(), Some(619.0));
        let oversized = chunk(cx, "loanOversized");
        assert!(oversized.jit_frame().1 > crate::jit::FRAME_BUF);
        oversized.jit_runs.set(0);
        oversized.inline_retry_at.set(u32::MAX);
        assert_eq!(value_in(cx, "loanOversized()").as_num_opt(), Some(619.0));
        assert_eq!(
            oversized.jit_runs.get(),
            1,
            "the exact oversized chunk must enter native code"
        );

        cx.interp_mut()
            .gc_collect_precise_for_test(CollectionKind::Major);
        assert_eq!(args.as_ptr() as usize, expected.address);
        assert!(expected.weak.iter().all(|weak| weak.live_ptr().is_some()));
        for (value, hidden) in args.iter().zip(&expected.hidden) {
            assert_eq!(value.into_internal().bits(), !hidden);
        }
        cx.host_mut::<Expected>().unwrap().calls += 1;
        Ok(args.last().copied().unwrap_or(Value::Undefined))
    }
    #[inline(never)]
    fn queue_tail<'s>(
        cx: &mut Ctx<'s>,
        _: Value<'s>,
        _: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let global = cx.global_this();
        let collect = cx.member_get(&global, "loanTailCollect")?;
        cx.interp_mut().pending_tail = Some(Box::new((
            collect.into_internal(),
            Internal::Undefined,
            Vec::new(),
        )));
        Ok(Value::Undefined)
    }
    fn tail_collect(
        interp: &mut crate::interpreter::Interp,
        _: Internal,
        _: &[Internal],
    ) -> Result<Internal, Internal> {
        interp.gc_collect_precise_for_test(CollectionKind::Major);
        assert!(
            interp
                .host::<Expected>()
                .unwrap()
                .weak
                .iter()
                .all(|weak| weak.live_ptr().is_some()),
            "the original arguments lost their roots before the post-native tail drain"
        );
        Ok(Internal::Num(9.0))
    }
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    let global = engine.interp.global;
    engine
        .interp
        .def_method(&global, "loanTailCollect", 0, tail_collect);
    let queue = std::hint::black_box(queue_tail as crate::embed::NativeFn);
    let function = std::hint::black_box(check as crate::embed::NativeFn);
    let this = Internal::Undefined;
    for count in [0, 1, 4, 5, 64] {
        let seeds = crate::rooting::RootedVec::from_vec(vec![
            Internal::Obj(engine.interp.new_object()),
            Internal::from_string("a native loan's otherwise unrooted text".repeat(24)),
            Internal::bigint_from_i128(1i128 << 100),
            Internal::Sym(crate::value::new_symbol_ref(crate::value::SymbolData {
                id: u64::MAX - count as u64,
                description: None,
                weak_observers: Default::default(),
            })),
            Internal::Undefined,
            Internal::Null,
            Internal::Bool(true),
            Internal::Num(-0.0),
            Internal::Num(f64::NAN),
        ]);
        let arguments: Vec<_> = (0..count).map(|index| seeds[index % seeds.len()]).collect();
        drop(seeds);
        let weak: Vec<_> = arguments
            .iter()
            .filter_map(Internal::cell_word)
            .map(|word| crate::gc_handle::WeakRef::to(word as usize))
            .collect();
        engine.op_state().put(Expected {
            address: arguments.as_ptr() as usize,
            hidden: arguments.iter().map(|value| !value.bits()).collect(),
            weak: weak.clone(),
            calls: 0,
        });
        let words = &arguments;
        let result = unsafe {
            engine.interp.call_native_address_committed(
                function as usize,
                crate::bytecode::CALL_IC_SCOPED_NATIVE,
                &this,
                words.as_ptr(),
                words.len(),
            )
        }
        .unwrap_or_else(|_| panic!("reentrant native call"));
        assert_eq!(
            result.bits(),
            arguments
                .last()
                .copied()
                .unwrap_or(Internal::Undefined)
                .bits()
        );
        assert_eq!(engine.interp.host::<Expected>().unwrap().calls, 1);
        assert!(arguments
            .iter()
            .zip(&engine.interp.host::<Expected>().unwrap().hidden)
            .all(|(value, hidden)| value.bits() == !hidden));
        // The scoped native returns before this queued engine-native tail collects. No live
        // native callback frame still holds the inputs: only the committed boundary's ongoing
        // immutable loan/root scope can retain these otherwise unregistered heap words.
        let tail_result = unsafe {
            engine.interp.call_native_address_committed(
                queue as usize,
                crate::bytecode::CALL_IC_SCOPED_NATIVE,
                &this,
                arguments.as_ptr(),
                arguments.len(),
            )
        }
        .unwrap_or_else(|_| panic!("post-native tail collection"));
        assert_eq!(tail_result.as_num_opt(), Some(9.0));
        drop(arguments);
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Major);
        assert!(
            weak.iter().all(|weak| weak.live_ptr().is_none()),
            "argument roots outlived their loan"
        );
    }
}

#[test]
fn native_argument_loans_snapshot_mapped_bindings_and_apply_lists() {
    #[inline(never)]
    fn hold<'s>(
        cx: &mut Ctx<'s>,
        _: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let first = args.first().copied().unwrap_or_default();
        let callback = args.get(1).copied().unwrap_or_default();
        let address = args.as_ptr();
        cx.invoke(callback, Value::Undefined, &[])?;
        cx.collect_garbage_for_host();
        assert_eq!(args.as_ptr(), address);
        assert!(cx.values_strict_equal(&first, &args[0]));
        Ok(args[0])
    }
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global(
            "holdEvaluated",
            2,
            std::hint::black_box(hold as crate::embed::NativeFn),
        );
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_retention_tracing(true);
        struct Tracked(Vec<crate::value::WeakGc>);
        engine.op_state().put(Tracked(Vec::new()));
        engine.define_global("trackLoan", 1, |cx, _, args| {
            let value = args[0];
            cx.host_mut::<Tracked>()
                .unwrap()
                .0
                .push(crate::value::Gc::downgrade(&value.as_obj().unwrap().gc()));
            Ok(value)
        });
        engine.define_global("collectLoan", 0, |cx, _, _| {
            // General AST evaluation also needs its legitimate native-stack temporaries.
            // Keep Decision 0 collection enabled, then distinguish the prefix's explicit root
            // from conservative retention: roots are marked before native stack scanning.
            cx.collect_garbage_for_host();
            let watched = cx.host::<Tracked>().unwrap().0.clone();
            for weak in &watched {
                let object = weak
                    .upgrade()
                    .expect("the evaluated apply prefix stays live");
                let path = cx
                    .interp_mut()
                    .gc_heap
                    .core
                    .borrow()
                    .retention_path(crate::value::Gc::as_ptr(&object) as usize);
                assert!(
                    matches!(path.first(), Some((_, _, crate::gc_core::MarkSource::Root))),
                    "the evaluated apply prefix must be an explicit root, got {path:?}"
                );
            }
            Ok(Value::Undefined)
        });
        assert_eq!(
            run(
                &mut engine,
                r#"
            function mapped(value) {
                var original = value, source = arguments;
                var held = holdEvaluated(value, function(){source[0] = {changed:true}; collectLoan();});
                if (held !== original || value === original || value !== source[0]) throw 'mapped snapshot';
            }
            for (var warm=0; warm<32; warm++) mapped({tag:warm});
            var source = [{original:true}, null];
            var original = source[0];
            source[1] = function(){source[0] = {changed:true}; collectLoan();};
            var held = holdEvaluated.apply(undefined, source);
            if (held !== original || source[0] === original) throw 'apply snapshot';
            var trace = [];
            var arrayLike = {length:2,
                get 0(){ trace.push(0); return trackLoan({alive:true}); },
                get 1(){ trace.push(1); collectLoan(); return function(){collectLoan()}; }
            };
            var fresh = holdEvaluated.apply(undefined, arrayLike);
            if (!fresh.alive || trace.join() !== '0,1') throw 'evaluated prefix root';
            'ok'
        "#
            ),
            "ok",
            "{tier:?}"
        );
    }
}

#[test]
fn typed_float_and_dataview_ingress_canonicalize_all_nan_tags_without_changing_bytes() {
    #[inline(never)]
    fn canonical<'s>(
        _: &mut Ctx<'s>,
        _: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        assert_eq!(args[0].into_internal().bits(), crate::value::PACK_CANON_NAN);
        assert!(args[0].as_num_opt().unwrap().is_nan());
        Ok(Value::Bool(true))
    }
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global(
            "canonicalNumber",
            1,
            std::hint::black_box(canonical as crate::embed::NativeFn),
        );
        assert_eq!(
            run(
                &mut engine,
                r#"
            function check(v,m){if(!v)throw Error(m)}
            var b=new ArrayBuffer(8), d=new DataView(b), f64=new Float64Array(b), f32=new Float32Array(b), f16=new Float16Array(b);
            function read64(high,low){
                d.setUint32(0,low,true);d.setUint32(4,high,true);
                canonicalNumber(d.getFloat64(0,true));canonicalNumber(f64[0]);
                check(d.getUint32(0,true)===low && d.getUint32(4,true)===high,'raw f64 bytes changed');
                d.setUint32(0,high,false);d.setUint32(4,low,false);
                canonicalNumber(d.getFloat64(0,false));
                check(d.getUint32(0,false)===high && d.getUint32(4,false)===low,'raw big endian bytes changed');
            }
            for(var sign=0;sign<2;sign++)for(var prefix=0x7ff0;prefix<=0x7fff;prefix++) {
                var high=(prefix+sign*0x8000)*65536;
                read64(high,1);read64(high+0x8000,0x1234);
            }
            for(var bits of [0x7f800001,0x7fc00001,0x7fffffff,0xff800001,0xffc00001,0xffffffff]) {
                d.setUint32(0,bits,true);canonicalNumber(d.getFloat32(0,true));canonicalNumber(f32[0]);
                check(d.getUint32(0,true)===bits,'raw f32 bytes changed');
            }
            for(var bits of [0x7c01,0x7e01,0x7fff,0xfc01,0xfe01,0xffff]) {
                d.setUint16(0,bits,true);canonicalNumber(d.getFloat16(0,true));canonicalNumber(f16[0]);
                canonicalNumber(Math.f16round(d.getFloat16(0,true)));
                check(d.getUint16(0,true)===bits,'raw f16 bytes changed');
            }
            for(var number of [-0,0,Number.MIN_VALUE,-Number.MIN_VALUE,Infinity,-Infinity,1.5]) {
                d.setFloat64(0,number,true);
                check(Object.is(d.getFloat64(0,true),number) && Object.is(f64[0],number),'finite or signed-zero round trip');
            }
            d.setUint16(0,1,true);check(f16[0]===2**-24,'f16 subnormal');
            d.setUint32(0,1,true);check(f32[0]===2**-149,'f32 subnormal');
            'ok'
        "#
            ),
            "ok",
            "{tier:?}"
        );
    }
}

#[test]
fn a_native_argument_window_survives_resuming_a_distinct_suspended_continuation() {
    #[inline(never)]
    fn hold<'s>(
        cx: &mut Ctx<'s>,
        _: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let address = args.as_ptr();
        let words: Vec<_> = args
            .iter()
            .map(|value| value.into_internal().bits())
            .collect();
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        ))]
        let before = crate::jit::TEST_NATIVE_SLICES.with(Cell::get);
        cx.invoke(args[0], Value::Undefined, &[])?;
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        ))]
        if cx.interp_mut().tier == crate::bytecode::Tier::Jit {
            assert!(
                crate::jit::TEST_NATIVE_SLICES.with(Cell::get) > before,
                "the targeted distinct generator resume must enter a native slice"
            );
        }
        cx.collect_garbage_for_host();
        assert_eq!(args.as_ptr(), address);
        assert!(args
            .iter()
            .zip(&words)
            .all(|(value, word)| value.into_internal().bits() == *word));
        Ok(Value::Bool(true))
    }
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global(
            "holdContinuationArgs",
            6,
            std::hint::black_box(hold as crate::embed::NativeFn),
        );
        engine.define_global("collectContinuation", 0, |cx, _, _| {
            cx.collect_garbage_for_host();
            Ok(Value::Undefined)
        });
        assert_eq!(
            run(
                &mut engine,
                r#"
            var seed={tag:'paused'}, resumed=0;
            function* paused(input) {
                var saved=input,sum=0;
                yield input;
                for(var k=0;k<48;k++)sum+=k;
                collectContinuation();
                resumed++;return saved===input && sum===1128;
            }
            function outer(callback,object,string,bigint,symbol,number) {
                return holdContinuationArgs(callback,object,string,bigint,symbol,number);
            }
            for(var n=0;n<24;n++) {
                var iterator=paused(seed);if(iterator.next().value!==seed)throw 'pause';
                var symbol=Symbol('kept'), object={index:n};
                if(!outer(function(){var result=iterator.next();if(!result.done||!result.value)throw 'resume'},
                    object,'continuation text '+n,12345678901234567890n,symbol,-0))throw 'native hold';
            }
            if(resumed!==24)throw 'distinct continuation not resumed';
            'ok'
        "#
            ),
            "ok",
            "{tier:?}"
        );
    }
}

#[test]
fn opaque_scoped_captures_preserve_every_word_in_both_result_arms() {
    use crate::value::{Callable, NativeCallableBody};
    #[derive(Clone)]
    struct Expected {
        capture_address: usize,
        argument_address: usize,
        hidden: Vec<u64>,
        hidden_this: u64,
        calls: usize,
    }
    #[inline(never)]
    fn capture<'s>(
        cx: &mut Ctx<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
        captures: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let expected = cx.host::<Expected>().unwrap();
        assert_eq!(captures.as_ptr() as usize, expected.capture_address);
        assert_eq!(captures.len(), expected.hidden.len());
        assert_eq!(args.as_ptr() as usize, expected.argument_address);
        assert_eq!(args.len(), 2);
        assert_eq!(this.into_internal().bits(), !expected.hidden_this);
        for (value, hidden) in captures.iter().zip(&expected.hidden) {
            assert_eq!(value.into_internal().bits(), !hidden);
        }
        let value = captures[args[0].as_num_opt().unwrap() as usize];
        cx.host_mut::<Expected>().unwrap().calls += 1;
        if matches!(args[1].kind(), ValueKind::Bool(true)) {
            Err(value)
        } else {
            Ok(value)
        }
    }
    let mut engine = Engine::new();
    let function = std::hint::black_box(capture as crate::embed::NativeCaptureFn);
    engine.scope(|cx| {
        let captures: ValueList<'_> = [
            Value::Undefined,
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::Num(-0.0),
            Value::Num(1.5),
            Value::Num(f64::INFINITY),
            Value::Num(f64::NEG_INFINITY),
            Value::Num(f64::NAN),
            cx.bigint_from_i128(1i128 << 100),
            cx.string("opaque captured text".repeat(32)),
            value_in(cx, "Symbol('opaque captured symbol')"),
            Value::Obj(cx.new_object()),
            value_in(cx, "(function capturedIdentity(){})"),
        ]
        .into_iter()
        .collect();
        let callback = cx.new_native_fn_with_captures("opaqueCaptured", 2, function, &captures);
        let capture_address = {
            let object = callback.as_obj().unwrap().gc();
            let call = object.borrow();
            let Callable::NativeData(data) = &call.call else {
                panic!("captured callable");
            };
            let NativeCallableBody::ScopedCaptured {
                function: actual,
                captures: stored,
            } = &data.body
            else {
                panic!("declared scoped captured signature");
            };
            assert_eq!(*actual as usize, function as usize);
            assert_eq!(stored.len(), captures.len());
            stored.as_ptr() as usize
        };
        let receiver = captures[12];
        cx.op_state().put(Expected {
            capture_address,
            argument_address: 0,
            hidden: captures
                .iter()
                .map(|value| !value.into_internal().bits())
                .collect(),
            hidden_this: !receiver.into_internal().bits(),
            calls: 0,
        });
        for (index, value) in captures.iter().enumerate() {
            for throwing in [false, true] {
                let args = [Value::Num(index as f64), Value::Bool(throwing)];
                cx.host_mut::<Expected>().unwrap().argument_address = args.as_ptr() as usize;
                let result = cx.invoke(callback, receiver, &args);
                assert_eq!(result.is_err(), throwing);
                let actual = match result {
                    Ok(value) | Err(value) => value,
                };
                assert_eq!(actual.into_internal().bits(), value.into_internal().bits());
                assert_eq!(args[0].as_num_opt(), Some(index as f64));
                assert!(matches!(args[1].kind(), ValueKind::Bool(actual) if actual == throwing));
            }
        }
        assert_eq!(cx.host::<Expected>().unwrap().calls, captures.len() * 2);
    });
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn opaque_scoped_captures_trace_cycles_and_survive_their_functions_collection_during_call() {
    use crate::gc_core::CollectionKind;
    use crate::value::{Callable, NativeCallableBody};
    #[derive(Clone)]
    struct Watched {
        function: crate::value::WeakGc,
        captures: Vec<crate::gc_handle::WeakRef>,
    }
    #[inline(never)]
    fn capture<'s>(
        cx: &mut Ctx<'s>,
        _: Value<'s>,
        _: &[Value<'s>],
        captures: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        assert_eq!(captures.len(), 5);
        assert!(cx.values_strict_equal(&captures[0], &captures[1]));
        let watched = cx.host::<Watched>().unwrap().clone();
        let global = cx.global_this();
        cx.member_set(&global, "activeCapture", Value::Undefined)?;
        cx.member_set(&captures[0], "callback", Value::Undefined)?;
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            cx.interp_mut().gc_collect_precise_for_test(kind);
            assert!(
                watched
                    .captures
                    .iter()
                    .all(|weak| weak.live_ptr().is_some()),
                "active captures lost their sole roots"
            );
        }
        assert!(
            watched.function.upgrade().is_none(),
            "the active Rust callable lease must not root its function object"
        );
        assert_eq!(
            captures[2].as_text().unwrap(),
            "a captured string with independent storage".repeat(32)
        );
        assert_eq!(captures[3].bigint_as_i64(), Some(17));
        assert!(matches!(captures[4].kind(), ValueKind::Sym(_)));
        Ok(Value::Undefined)
    }
    let mut engine = Engine::new();
    let function = std::hint::black_box(capture as crate::embed::NativeCaptureFn);
    let watched = engine.scope(|cx| {
        let target = Value::Obj(cx.new_object());
        let values = [target, target, cx.string("a captured string with independent storage".repeat(32)),
            cx.bigint_from_i128((1i128 << 100) + 17), value_in(cx, "Symbol('capture identity')")];
        let captures = values.iter().map(|value| crate::gc_handle::WeakRef::to(value.into_internal().cell_word().unwrap() as usize)).collect();
        let callback = cx.new_native_fn_with_captures("activeCapture", 0, function, &values);
        let object = callback.as_obj().unwrap().gc();
        let call = object.borrow();
        assert!(matches!(&call.call, Callable::NativeData(data) if matches!(&data.body,
            NativeCallableBody::ScopedCaptured {function: actual, captures} if *actual as usize == function as usize && captures.len() == 5)));
        drop(call);
        cx.member_set(&target, "callback", callback).unwrap();
        let global = cx.global_this(); cx.member_set(&global, "activeCapture", callback).unwrap();
        Watched {function: crate::value::Gc::downgrade(&object), captures}
    });
    engine.op_state().put(watched.clone());
    for kind in [CollectionKind::Minor, CollectionKind::Major] {
        engine.interp.gc_collect_precise_for_test(kind);
        assert!(watched.function.upgrade().is_some());
        assert!(
            watched
                .captures
                .iter()
                .all(|weak| weak.live_ptr().is_some()),
            "the function's owned captures were not traced"
        );
    }
    engine.scope(|cx| {
        let global = cx.global_this();
        let callback = cx.member_get(&global, "activeCapture").unwrap();
        assert!(cx
            .invoke(callback, Value::Undefined, &[])
            .unwrap()
            .is_undefined());
    });
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert!(watched.function.upgrade().is_none());
    assert!(
        watched
            .captures
            .iter()
            .all(|weak| weak.live_ptr().is_none()),
        "completed invocation left captures rooted"
    );
}
