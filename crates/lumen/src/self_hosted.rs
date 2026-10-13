//! Self-hosted built-ins: ECMAScript functions the engine defines in JavaScript, so that they run
//! in the same execution model as author code (compiled by the same tiers, calling callbacks
//! through ordinary call sites) instead of crossing the native boundary for every callback.
//!
//! Each Realm evaluates the source once, in a private Declarative Environment Record whose only
//! bindings are the intrinsics below. That record has no outer environment: no global, author
//! prototype or `Function.prototype` member can intercept a step the specification does not make
//! observable. The resulting closures are ordinary strict method functions of that Realm — not
//! constructors, without a `prototype`, with the specified `name` and `length` — and their
//! functions are marked [`crate::ast::Function::self_hosted`]: they render as NativeFunction,
//! compile on their first call and compile their intrinsics to operations, also where the
//! inliner splices them into a caller (see `bytecode::Compiler::self_hosted`).
//!
//! Every intrinsic is bound to a native function, which is what the tree-walking interpreter
//! calls. The bytecode compiler instead compiles each intrinsic call to the operation it names
//! (`Call` to an ordinary method call, the rest to `bytecode::AbstractOp`); both forms run the
//! functions below, so the tiers agree step for step. The names are the specification's:
//!
//! | Intrinsic | ECMA-262 |
//! |---|---|
//! | `ToObject(value, "method")` | ToObject (§7.1.18); the TypeError names the method |
//! | `LengthOfArrayLike(obj)` | LengthOfArrayLike (§7.3.18) |
//! | `IsCallable(value)` | IsCallable (§7.2.3) |
//! | `ThrowTypeError("message")` | throw a *TypeError* of the current Realm |
//! | `ArraySpeciesCreate(obj, length)` | ArraySpeciesCreate (§10.4.2.3) |
//! | `Call(F, V, ...args)` | Call (§7.3.14) |
//! | `CreateDataPropertyOrThrow(O, P, V)` | CreateDataPropertyOrThrow (§7.3.7) |
//!
//! Message operands are string literals. Other operands must not depend on evaluation order
//! beyond left to right, which both forms preserve.

use crate::ast::{Expr, Stmt};
use crate::interpreter::{abrupt_value, Abrupt, Binding, Env, Interp};
use crate::value::{Exotic, Gc, PackedValue, Property, Value};
use std::rc::Rc;

/// The `Array.prototype` iteration methods (forEach, map, filter, some, every, find, findIndex,
/// findLast, findLastIndex, reduce, reduceRight).
const ARRAY_SOURCE: &str = include_str!("self_hosted/array.js");

/// Every method `ARRAY_SOURCE` defines, in installation order, with its specified `length`:
/// "the number of required parameters shown in the subclause heading … Optional parameters and
/// rest parameters are not included" (ECMA-262 §18, ECMAScript Standard Built-in Objects). The
/// source declares the optional parameters it reads, so its formal parameter count can differ.
const ARRAY_METHODS: [(&str, u32); 11] = [
    ("forEach", 1),
    ("map", 1),
    ("filter", 1),
    ("some", 1),
    ("every", 1),
    ("find", 1),
    ("findIndex", 1),
    ("findLast", 1),
    ("findLastIndex", 1),
    ("reduce", 1),
    ("reduceRight", 1),
];

/// The self-hosted source as an AST snapshot (see `crate::snapshot`), encoded by the first
/// interpreter of the process. A parsed AST cannot be shared between interpreters — its
/// functions cache compiled chunks, whose inline caches hold one heap's shapes and objects — but
/// decoding the snapshot costs a small fraction of lexing and parsing the text again.
static ARRAY_SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

/// The parsed self-hosted source, shared by every Realm of one interpreter: the object-literal
/// expression whose methods become the built-ins.
pub(crate) struct SelfHostedSource {
    array_methods: Expr,
}

impl SelfHostedSource {
    fn load() -> SelfHostedSource {
        let snapshot =
            ARRAY_SNAPSHOT.get_or_init(|| crate::snapshot::encode(&parse_array_source()));
        // A snapshot this build encoded always decodes; parsing again is the codec's fallback.
        let statements = crate::snapshot::decode(snapshot).unwrap_or_else(|_| parse_array_source());
        // The directive prologue, then exactly one expression statement.
        let array_methods = statements
            .into_iter()
            .rev()
            .find_map(|statement| match statement {
                Stmt::Expr(mut expr @ Expr::Paren(_)) => match &mut expr {
                    Expr::Paren(inner) => Some(std::mem::replace(&mut **inner, Expr::Null)),
                    _ => None,
                },
                Stmt::Expr(expr @ Expr::Object(_)) => Some(expr),
                _ => None,
            })
            .expect("self-hosted Array source ends with its method object literal");
        SelfHostedSource { array_methods }
    }

    /// Bytes retained by the parsed source (its AST; the source text itself is static).
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        std::mem::size_of::<Self>()
            + crate::ast::scan_expr_retained_memory(&self.array_methods, visitor)
    }
}

fn parse_array_source() -> Vec<Stmt> {
    crate::parser::parse_self_hosted(ARRAY_SOURCE)
        .unwrap_or_else(|error| panic!("self-hosted Array source: {}", error.message))
}

/// Install the self-hosted `Array.prototype` methods on the Realm's `array_proto` with the
/// built-in method attributes { [[Writable]]: true, [[Enumerable]]: false,
/// [[Configurable]]: true } (ECMA-262 §18: "Every other data property described in clauses 19
/// through 28 and in Annex B.2 has the attributes …") and their specified `length`s.
pub(crate) fn install_array_methods(it: &mut Interp, array_proto: &Gc) {
    let source = it
        .self_hosted
        .get_or_init(|| Rc::new(SelfHostedSource::load()))
        .clone();
    let env = intrinsic_environment(it);
    let methods = match it.eval(&source.array_methods, &env) {
        Ok(Value::Obj(methods)) => methods,
        Ok(_) => unreachable!("the self-hosted source evaluates to an object literal"),
        Err(_) => panic!("evaluating the self-hosted Array methods cannot throw"),
    };
    debug_assert_eq!(methods.borrow().props.iter().count(), ARRAY_METHODS.len());
    for (name, length) in ARRAY_METHODS {
        let method = methods
            .borrow()
            .props
            .get(name)
            .map(|property| property.value())
            .unwrap_or_else(|| panic!("the self-hosted source defines Array.prototype.{name}"));
        if let Value::Obj(function) = &method {
            // The function's own `length` keeps its attributes and position; only the value
            // differs from the formal parameter count.
            function.borrow_mut().props.insert(
                "length",
                Property::data(Value::Num(f64::from(length)), false, false, true),
            );
        }
        array_proto
            .borrow_mut()
            .props
            .insert(name, Property::data(method, true, false, true));
    }
}

/// A fresh Declarative Environment Record with no outer environment, binding each intrinsic
/// immutably to a native function of the running Realm.
fn intrinsic_environment(it: &Interp) -> Env {
    let env = crate::interpreter::new_scope(None);
    let intrinsics: [(&str, usize, crate::value::NativeFn); 7] = [
        ("ToObject", 2, intrinsic_to_object),
        ("LengthOfArrayLike", 1, intrinsic_length_of_array_like),
        ("IsCallable", 1, intrinsic_is_callable),
        ("ThrowTypeError", 1, intrinsic_throw_type_error),
        ("ArraySpeciesCreate", 2, intrinsic_array_species_create),
        ("Call", 2, intrinsic_call),
        (
            "CreateDataPropertyOrThrow",
            3,
            intrinsic_create_data_property_or_throw,
        ),
    ];
    {
        let mut record = env.borrow_mut();
        for (name, length, function) in intrinsics {
            record.vars.insert(
                name,
                Binding {
                    value: Value::Obj(it.make_native(name, length, function)),
                    mutable: false,
                    strict_immutable: true,
                    initialized: true,
                    import_ref: None,
                    imported: false,
                    deletable: false,
                },
            );
        }
    }
    env
}

/// ToObject(`value`) (ECMA-262 §7.1.18) for the receiver of the built-in `method`.
pub(crate) fn to_object(i: &mut Interp, value: Value, method: &str) -> Result<Value, Abrupt> {
    match value {
        object @ Value::Obj(_) => Ok(object),
        primitive => crate::builtins::to_object_for_method(i, primitive, method)
            .map(Value::Obj)
            .map_err(Abrupt::Throw),
    }
}

/// LengthOfArrayLike(`object`) (ECMA-262 §7.3.18): ℝ(? ToLength(? Get(obj, "length"))). An
/// ordinary Array's own `length` is always a data property holding an integral Number no
/// greater than 2^32 − 1, so its value is the answer without the generic Get.
pub(crate) fn length_of_array_like(i: &mut Interp, object: &Value) -> Result<f64, Abrupt> {
    let Value::Obj(target) = object else {
        unreachable!("LengthOfArrayLike is applied to ToObject's result");
    };
    {
        let body = target.borrow();
        if body.ic_plain.get() && matches!(body.exotic, Exotic::Array) {
            if let Some(Value::Num(length)) = body.props.length_property().map(|p| p.value()) {
                return Ok(length);
            }
        }
    }
    let length = i.get_member(object, "length")?;
    let length = i.to_number(&length)?;
    // ToLength: ToIntegerOrInfinity, then clamp to [+0, 2^53 − 1] (NaN and -0 become +0).
    Ok(if length.is_nan() || length <= 0.0 {
        0.0
    } else {
        length.trunc().min(9007199254740991.0)
    })
}

/// ArraySpeciesCreate(`original`, `length`) (ECMA-262 §10.4.2.3); `length` is a
/// LengthOfArrayLike result or a non-negative integer literal.
pub(crate) fn array_species_create(
    i: &mut Interp,
    original: &Value,
    length: &Value,
) -> Result<Value, Abrupt> {
    let Value::Num(length) = *length else {
        unreachable!("ArraySpeciesCreate receives a non-negative integer length");
    };
    crate::builtins::array_species_create_length(i, original, length).map_err(Abrupt::Throw)
}

/// CreateDataPropertyOrThrow(`object`, ToPropertyKey(`key`), `value`) (ECMA-262 §7.3.7).
pub(crate) fn create_data_property_or_throw(
    i: &mut Interp,
    object: &Value,
    key: &Value,
    value: PackedValue,
) -> Result<(), Abrupt> {
    crate::builtins::create_data_property_or_throw_key(i, object, key, value).map_err(Abrupt::Throw)
}

/// A *TypeError* of the current Realm with `message`.
pub(crate) fn type_error(i: &mut Interp, message: &str) -> Abrupt {
    i.throw("TypeError", message)
}

fn argument(args: &[Value], index: usize) -> Value {
    args.get(index).cloned().unwrap_or(Value::Undefined)
}

fn message(args: &[Value], index: usize) -> String {
    match argument(args, index) {
        Value::Str(text) => text.to_string(),
        _ => unreachable!("self-hosted intrinsic messages are string literals"),
    }
}

fn intrinsic_to_object(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    to_object(i, argument(args, 0), &message(args, 1)).map_err(abrupt_value)
}

fn intrinsic_length_of_array_like(
    i: &mut Interp,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    length_of_array_like(i, &argument(args, 0))
        .map(Value::Num)
        .map_err(abrupt_value)
}

fn intrinsic_is_callable(_: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    Ok(Value::Bool(argument(args, 0).is_callable()))
}

fn intrinsic_throw_type_error(
    i: &mut Interp,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    Err(abrupt_value(type_error(i, &message(args, 0))))
}

fn intrinsic_array_species_create(
    i: &mut Interp,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    array_species_create(i, &argument(args, 0), &argument(args, 1)).map_err(abrupt_value)
}

fn intrinsic_call(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let function = argument(args, 0);
    let this = argument(args, 1);
    let rest = args.get(2..).unwrap_or(&[]);
    i.call(function, this, rest).map_err(abrupt_value)
}

fn intrinsic_create_data_property_or_throw(
    i: &mut Interp,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    create_data_property_or_throw(
        i,
        &argument(args, 0),
        &argument(args, 1),
        PackedValue::pack(argument(args, 2)),
    )
    .map_err(abrupt_value)?;
    Ok(Value::Bool(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ast::{PropDef, PropKey};
    use crate::{Completion, Engine};

    /// Each `name() { … }` method of the source's object literal.
    fn each_method(expr: &Expr, visit: &mut impl FnMut(&str, &crate::ast::Function)) {
        let Expr::Object(properties) = expr else {
            panic!("the self-hosted source is an object literal");
        };
        for property in properties {
            match property {
                PropDef::Method {
                    key: PropKey::Ident(name),
                    func,
                } => visit(name, func),
                _ => panic!("the self-hosted object literal holds only named methods"),
            }
        }
    }

    /// The process-wide snapshot decodes to the same methods the parser produces, still marked
    /// self-hosted (so they compile their intrinsics and render as NativeFunction) and without
    /// source text, and a second interpreter decodes it instead of parsing.
    #[test]
    fn self_hosted_source_round_trips_through_the_process_snapshot() {
        let parsed = parse_array_source();
        let loaded = SelfHostedSource::load();
        assert!(ARRAY_SNAPSHOT.get().is_some());
        let mut names = Vec::new();
        each_method(&loaded.array_methods, &mut |name, func| {
            assert!(func.self_hosted && func.is_strict && func.source.is_none());
            names.push(name.to_string());
        });
        assert_eq!(
            names,
            [
                "forEach",
                "map",
                "filter",
                "some",
                "every",
                "find",
                "findIndex",
                "findLast",
                "findLastIndex",
                "reduce",
                "reduceRight"
            ]
        );
        assert_eq!(
            crate::snapshot::encode(&parsed),
            *ARRAY_SNAPSHOT.get().unwrap(),
            "the snapshot encodes exactly the parsed source"
        );
        for _ in 0..2 {
            let completion = Engine::new()
                .eval("[1, , 3].map((x, i) => x * i).join()", false)
                .expect("parse");
            assert!(matches!(completion, Completion::Value(ref v) if v == "0,,6"));
        }
    }
}
