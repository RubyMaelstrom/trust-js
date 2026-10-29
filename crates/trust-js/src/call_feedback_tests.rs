//! W3's item 7, W4's call feedback where W3 needs it: every native's builtin id and call-cache
//! class, declared as its realm's setup installs it and recorded by the call cache's fill (V8's
//! `Builtin` id, which the optimizing tiers key their reductions on).

use crate::bytecode::*;
use crate::value::{Callable, Value, ValueView};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("feedback fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The native function `path` names in a fresh realm of `engine`, through a global it binds.
fn native(engine: &mut Engine, path: &str) -> crate::value::NativeFn {
    eval(engine, &format!("var probedNative = {path};"));
    let env = engine.interp.global_env;
    let value: Value = engine
        .interp
        .get_var("probedNative", &env)
        .ok()
        .expect("bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{path} is not an object")
    };
    let call = &function.borrow().call;
    match call {
        Callable::Native(f) => *f,
        _ => panic!("{path} is not a bare native"),
    }
}

/// The classes the realm's setup declares, by the natives that carry them: each call template's
/// native, the operand-only collection methods, Math's templated and numeric-only functions,
/// DataView's accessors, the light natives of item 2a, and natives with no class.
const CLASSES: &[(&str, u8)] = &[
    ("String.prototype.charCodeAt", INTRINSIC_CHAR_CODE_AT),
    ("String.prototype.codePointAt", INTRINSIC_CODE_POINT_AT),
    ("String.prototype.charAt", INTRINSIC_CHAR_AT),
    ("String.prototype.slice", INTRINSIC_STRING_SLICE),
    ("Object.hasOwn", INTRINSIC_OBJECT_HAS_OWN),
    ("Function.prototype.apply", INTRINSIC_FUNCTION_APPLY),
    ("Function.prototype.call", INTRINSIC_FUNCTION_CALL),
    ("Math.sqrt", INTRINSIC_MATH_SQRT),
    ("Array.prototype.push", INTRINSIC_ARRAY_PUSH),
    ("Array.prototype.pop", INTRINSIC_ARRAY_POP),
    ("RegExp.prototype.exec", INTRINSIC_REGEXP_EXEC_DISCARD),
    ("String.prototype.replace", INTRINSIC_STRING_REPLACE_DISCARD),
    ("String.prototype.split", INTRINSIC_STRING_SPLIT_DISCARD),
    ("isNaN", INTRINSIC_IS_NAN),
    ("Number.isNaN", INTRINSIC_IS_NAN),
    ("String.fromCharCode", INTRINSIC_FROM_CHAR_CODE),
    ("Reflect.apply", INTRINSIC_REFLECT_APPLY),
    ("Map.prototype.get", INTRINSIC_OPERAND_ONLY),
    ("Map.prototype.set", INTRINSIC_OPERAND_ONLY),
    ("Set.prototype.has", INTRINSIC_OPERAND_ONLY),
    ("WeakMap.prototype.get", INTRINSIC_OPERAND_ONLY),
    ("WeakSet.prototype.delete", INTRINSIC_OPERAND_ONLY),
    ("Math.floor", INTRINSIC_MATH_FLOOR),
    ("Math.abs", INTRINSIC_MATH_ABS),
    ("Math.imul", INTRINSIC_MATH_IMUL),
    ("Math.max", INTRINSIC_MATH_MAX),
    ("Math.min", INTRINSIC_MATH_MIN),
    ("Math.sin", INTRINSIC_NUMERIC_ONLY),
    ("Math.log", INTRINSIC_NUMERIC_ONLY),
    ("DataView.prototype.getInt8", INTRINSIC_DATAVIEW_GET),
    ("DataView.prototype.setFloat64", INTRINSIC_DATAVIEW_SET),
    ("parseInt", INTRINSIC_LIGHT_PRIMITIVE_ARGS),
    ("Number", INTRINSIC_LIGHT_PRIMITIVE_ARGS),
    ("Number.isInteger", INTRINSIC_LIGHT),
    ("Boolean", INTRINSIC_LIGHT),
    ("String.prototype.indexOf", INTRINSIC_LIGHT_PRIMITIVE),
    ("Array.isArray", INTRINSIC_ARRAY_IS_ARRAY),
    ("Object.is", INTRINSIC_OBJECT_IS),
    ("Object.keys", 0),
    ("JSON.stringify", 0),
    ("Math.atan2", 0),
    ("Map.prototype.forEach", 0),
];

#[test]
fn every_native_has_a_builtin_id_and_its_declared_class() {
    let mut first = Engine::new();
    let mut second = Engine::new();
    let mut ids = crate::fasthash::FastMap::default();
    for &(path, class) in CLASSES {
        let f = native(&mut first, path);
        let (id, declared) = crate::builtins::builtin(f as usize);
        assert_ne!(id, 0, "{path} has no builtin id");
        assert_eq!(declared, class, "{path}'s class");
        if let Some(other) = ids.insert(id, path) {
            panic!("{path} and {other} share builtin id {id}");
        }
        // The id names the native, whatever the realm that installed it.
        let again = native(&mut second, path);
        assert_eq!(again as usize, f as usize);
        assert_eq!(
            crate::builtins::builtin(again as usize),
            (id, class),
            "{path}"
        );
    }
}

/// The global function `name`'s compiled chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> std::rc::Rc<Chunk> {
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let call = &function.borrow().call;
    let Callable::User(user) = call else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} stayed interpreted"));
    std::rc::Rc::clone(chunk)
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_native_entry_records_its_builtin_id_and_class() {
    for tier in [Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let source = r#"
var wm = new WeakMap(), key = {}; wm.set(key, 2);
function run(n) { var s = 0; for (var k = 0; k < n; k++) s += Math.floor(k / 2) + wm.get(key) + [k].indexOf(k); return s; }
run(50) + run(50)"#;
        assert_eq!(eval(&mut engine, source), "1400", "{tier:?}");
        let chunk = chunk_of(&mut engine, "run");
        let mut seen = Vec::new();
        for site in &chunk.call_caches {
            for way in &site.entries {
                let entry = way.get();
                if entry.native != 0 {
                    let (builtin, class) = crate::builtins::builtin(entry.native);
                    assert_eq!(
                        (entry.builtin, entry.intrinsic),
                        (builtin, class),
                        "{tier:?}"
                    );
                    seen.push(class);
                } else if entry.callee != 0 {
                    assert_eq!(entry.builtin, 0, "a user function has no builtin id");
                }
            }
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            [0, INTRINSIC_OPERAND_ONLY, INTRINSIC_MATH_FLOOR],
            "{tier:?}: Array.prototype.indexOf, WeakMap.prototype.get and Math.floor"
        );
    }
}
