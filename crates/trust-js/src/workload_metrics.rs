//! Opt-in, source-free inventories for choosing optimization targets.
//!
//! Counters are process aggregates with fixed cardinality, not roots or caches. No
//! author names, values, identities, or source are retained. Uncached-name sampling
//! inspects native environment metadata only; it stops at a `with` object or import
//! rather than executing HasBinding/GetValue a second time (ECMA-262
//! GetIdentifierReference and Object Environment Records.HasBinding).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::ast::Function;
use crate::interpreter::Env;

const FUNCTION_KINDS: [&str; 8] = [
    "ordinary",
    "method",
    "arrow",
    "async",
    "async_method",
    "async_arrow",
    "generator",
    "async_generator",
];
static FUNCTIONS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static PROTOTYPES: AtomicU64 = AtomicU64::new(0);
static SELF_SCOPES: AtomicU64 = AtomicU64::new(0);
static ELIDED_SELF_SCOPES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
enum NamePath {
    Declarative,
    Uninitialized,
    Import,
    With,
    BeyondScopes,
}
const NAME_PATHS: [&str; 5] = [
    "declarative",
    "uninitialized",
    "import",
    "with",
    "beyond_scopes",
];
// The final bucket combines depths >= 16; arbitrarily deep/user-varying scope
// chains cannot grow the inventory. Depth zero means the starting environment.
const DEPTH_BUCKETS: usize = 17;
static NAMES: [AtomicU64; NAME_PATHS.len() * DEPTH_BUCKETS] =
    [const { AtomicU64::new(0) }; NAME_PATHS.len() * DEPTH_BUCKETS];

fn function_kind(arrow: bool, method: bool, generator: bool, asynchronous: bool) -> usize {
    if generator {
        if asynchronous {
            7
        } else {
            6
        }
    } else if arrow {
        if asynchronous {
            5
        } else {
            2
        }
    } else if method {
        if asynchronous {
            4
        } else {
            1
        }
    } else if asynchronous {
        3
    } else {
        0
    }
}

/// Observe actual user-function creation, not compilation or invocation.
/// OrdinaryFunctionCreate/MakeConstructor still perform all their normal work.
#[inline]
pub(crate) fn function_created(function: &Function, has_prototype: bool) {
    if crate::jit::perf_metrics_enabled() {
        record_function(function, has_prototype);
    }
}

#[inline(never)]
fn record_function(function: &Function, has_prototype: bool) {
    let kind = function_kind(
        function.is_arrow,
        function.is_method,
        function.is_generator,
        function.is_async,
    );
    FUNCTIONS[kind].fetch_add(1, Relaxed);
    if has_prototype {
        PROTOTYPES.fetch_add(1, Relaxed);
    }
    if function.needs_self_environment() {
        SELF_SCOPES.fetch_add(1, Relaxed);
    } else if function.is_fn_expr && function.name.is_some() {
        ELIDED_SELF_SCOPES.fetch_add(1, Relaxed);
    }
}

/// Only calls that miss both the free-name IC hit and fill paths are sampled.
/// This is NOT an inventory of all name reads (inlined/cache hits are omitted).
#[inline]
pub(crate) fn uncached_name(env: &Env, name: &str) {
    if crate::jit::perf_metrics_enabled() {
        record_name(env, name);
    }
}

fn classify_name(env: &Env, name: &str) -> (NamePath, usize) {
    let mut current = env.clone();
    let mut depth = 0;
    loop {
        let scope = current.borrow();
        if let Some(binding) = scope.vars.get(name) {
            let kind = if !binding.initialized {
                NamePath::Uninitialized
            } else if binding.import_ref.is_some() {
                NamePath::Import
            } else {
                NamePath::Declarative
            };
            return (kind, depth);
        }
        if scope.with_obj.is_some() {
            return (NamePath::With, depth);
        }
        let Some(parent) = scope.parent.clone() else {
            // No global property/Proxy/getter inspection: the real lookup decides
            // whether this is a global binding, an inherited binding, or absence.
            return (NamePath::BeyondScopes, depth);
        };
        drop(scope);
        current = parent;
        depth = (depth + 1).min(DEPTH_BUCKETS - 1);
    }
}

#[inline(never)]
fn record_name(env: &Env, name: &str) {
    let (kind, depth) = classify_name(env, name);
    NAMES[kind as usize * DEPTH_BUCKETS + depth].fetch_add(1, Relaxed);
}

pub(crate) fn json_fields() -> String {
    let functions = FUNCTION_KINDS
        .iter()
        .zip(&FUNCTIONS)
        .map(|(name, count)| format!("\"{name}\":{}", count.load(Relaxed)))
        .collect::<Vec<_>>()
        .join(",");
    let mut names = Vec::new();
    for (kind, label) in NAME_PATHS.iter().enumerate() {
        for depth in 0..DEPTH_BUCKETS {
            let count = NAMES[kind * DEPTH_BUCKETS + depth].load(Relaxed);
            if count != 0 {
                names.push(format!(
                    "{{\"kind\":\"{label}\",\"depth\":{depth},\"calls\":{count}}}"
                ));
            }
        }
    }
    format!(
        "\"user_function_creations\":{{{functions}}},\"function_prototype_creations\":{},\"function_self_scope_creations\":{},\"function_self_scope_elisions\":{},\"uncached_name_depth_max\":{},\"uncached_name_paths\":[{}]",
        PROTOTYPES.load(Relaxed), SELF_SCOPES.load(Relaxed), ELIDED_SELF_SCOPES.load(Relaxed), DEPTH_BUCKETS - 1, names.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{new_scope, Binding};
    use crate::value::{Object, Value};

    #[test]
    fn workload_function_kinds_cover_prototype_and_nonprototype_families() {
        for (flags, expected) in [
            ((false, false, false, false), "ordinary"),
            ((false, true, false, false), "method"),
            ((true, false, false, false), "arrow"),
            ((false, false, false, true), "async"),
            ((false, true, false, true), "async_method"),
            ((true, false, false, true), "async_arrow"),
            ((false, false, true, false), "generator"),
            ((false, true, true, false), "generator"),
            ((false, false, true, true), "async_generator"),
            ((false, true, true, true), "async_generator"),
        ] {
            assert_eq!(
                FUNCTION_KINDS[function_kind(flags.0, flags.1, flags.2, flags.3)],
                expected
            );
        }
    }

    #[test]
    fn workload_name_classifier_tracks_depth_tdz_import_and_shadowing() {
        let outer = new_scope(None);
        outer
            .borrow_mut()
            .vars
            .insert("value", Binding::data(Value::Num(7.0), true, true));
        let middle = new_scope(Some(outer.clone()));
        let inner = new_scope(Some(middle.clone()));
        assert_eq!(classify_name(&inner, "value"), (NamePath::Declarative, 2));
        assert_eq!(classify_name(&inner, "absent"), (NamePath::BeyondScopes, 2));
        middle
            .borrow_mut()
            .vars
            .insert("value", Binding::data(Value::Undefined, false, false));
        assert_eq!(classify_name(&inner, "value"), (NamePath::Uninitialized, 1));
        {
            let mut scope = middle.borrow_mut();
            let binding = scope.vars.get_mut("value").unwrap();
            binding.initialized = true;
            binding.set_import_reference(Some((outer, "value".into())));
        }
        assert_eq!(classify_name(&inner, "value"), (NamePath::Import, 1));
        inner
            .borrow_mut()
            .vars
            .insert("value", Binding::data(Value::Num(9.0), true, true));
        let count = std::rc::Rc::strong_count(&inner);
        assert_eq!(classify_name(&inner, "value"), (NamePath::Declarative, 0));
        assert_eq!(
            std::rc::Rc::strong_count(&inner),
            count,
            "no retained scope handles"
        );
    }

    #[test]
    fn workload_name_classifier_stops_at_with_without_reading_object_or_import() {
        let outer = new_scope(None);
        outer
            .borrow_mut()
            .vars
            .insert("value", Binding::data(Value::Num(7.0), true, true));
        let object = Object::new(None);
        let with = new_scope(Some(outer));
        with.borrow_mut().with_obj = Some(Value::Obj(object.clone()));
        let inner = new_scope(Some(with.clone()));
        // Holding an exclusive object borrow proves classification never inspects
        // its properties, invokes traps/getters, or asks for @@unscopables.
        let _exclusive = object.borrow_mut();
        assert_eq!(classify_name(&inner, "value"), (NamePath::With, 1));
        // A declarative binding is checked first, exactly as the actual lookup.
        with.borrow_mut()
            .vars
            .insert("value", Binding::data(Value::Undefined, false, false));
        assert_eq!(classify_name(&inner, "value"), (NamePath::Uninitialized, 1));
    }

    #[test]
    fn workload_name_depth_inventory_is_bounded() {
        let outer = new_scope(None);
        outer
            .borrow_mut()
            .vars
            .insert("value", Binding::data(Value::Num(7.0), true, true));
        let mut inner = outer;
        for _ in 0..80 {
            inner = new_scope(Some(inner));
        }
        assert_eq!(classify_name(&inner, "value"), (NamePath::Declarative, 16));
        assert_eq!(
            classify_name(&inner, "absent"),
            (NamePath::BeyondScopes, 16)
        );
    }
}
