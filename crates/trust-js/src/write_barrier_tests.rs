//! S6 step 10b: the write barrier's sites, each with a test whose write debug builds' barrier
//! check (`gc_core::Heap::check_write_barrier`, as every collection starts) would flag were that
//! site's barrier removed. Each makes the holder old by collecting, writes a value allocated
//! since (young), and collects again: the check passes, and the value lives. The JIT's sites
//! are written by compiled code alone after their caches are warm, so no Rust write sets the
//! holder's card instead.

use crate::bytecode::Tier;
use crate::value::ValueView;
use crate::value::{Gc, Property, Value};
use crate::{Completion, Engine};

/// An engine at `tier` whose `collect()` collects (explicitly, from inside the running code).
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
        .interp
        .def_method(&engine.interp.global, "collect", 0, |interp, _, _| {
            interp.gc_collect();
            Ok(Value::Undefined)
        });
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("barrier fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// The global object named `name`.
fn global_object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env;
    match (engine.interp.get_var(name, &env)).map(|__value| (__value).view()) {
        Ok(ValueView::Obj(object)) => object,
        _ => panic!("{name}: not an object"),
    }
}

/// The handles' own mutable borrows (`Gc::borrow_mut`, `Gc::try_borrow_mut`, `Env::borrow_mut`,
/// `Env::try_borrow_mut`), each the only writer of its own old holder: two objects' properties
/// and two environments' bindings get young objects, round after round.
#[test]
fn handle_borrows_record_the_write_in_their_cells_cards() {
    use crate::interpreter::{new_scope, Binding};
    let mut engine = engine(Tier::Interp);
    let interp = &mut engine.interp;
    let objects = [interp.new_object(), interp.new_object()];
    let envs = [new_scope(None), new_scope(None)];
    interp.gc_collect();
    for round in 0..3 {
        let young: [Gc; 4] = std::array::from_fn(|_| interp.new_object());
        objects[0]
            .borrow_mut()
            .props
            .insert("a", Property::plain(Value::Obj(young[0])));
        objects[1]
            .try_borrow_mut()
            .expect("not borrowed")
            .props
            .insert("a", Property::plain(Value::Obj(young[1])));
        envs[0]
            .borrow_mut()
            .vars
            .insert("a", Binding::data(Value::Obj(young[2]), true, true));
        envs[1]
            .try_borrow_mut()
            .expect("not borrowed")
            .vars
            .insert("a", Binding::data(Value::Obj(young[3]), true, true));
        interp.gc_collect();
        for (object, target) in objects.iter().zip(&young) {
            let value = object.borrow().props.get("a").map(Property::value);
            assert!(
                matches!((value).map(|__value| (__value).view()), Some(ValueView::Obj(found)) if Gc::ptr_eq(&found, target)),
                "round {round}"
            );
        }
    }
}

/// `Chunk::store_cap_ic`'s cache hit writes a captured binding through a raw pointer, after
/// any borrow of its environment has ended (bytecode tier).
#[test]
fn captured_binding_store_records_its_environment() {
    let mut engine = engine(Tier::Bytecode);
    let source = r#"
        function outer() {
            let x = null;
            const get = () => x;
            for (let k = 0; k < 8; k++) {
                if (k === 4) collect();
                x = { k };
            }
            collect();
            return get().k;
        }
        outer();
    "#;
    assert_eq!(eval(&mut engine, source), "7");
}

/// `PropWord::materialize` promotes a function's deferred prototype word to the materialized
/// prototype's under a shared borrow of the function, and the deferred prototype's cell records
/// the prototype: the first read of an old function's `prototype`, at every tier.
#[test]
fn deferred_prototype_promotion_records_its_function() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        let source = r#"
            var F = function () {};
            collect();
            var p = F.prototype;
            collect();
            [p === F.prototype, p.constructor === F,
             Object.getPrototypeOf(p) === Object.prototype].join();
        "#;
        assert_eq!(eval(&mut engine, source), "true,true,true", "{tier:?}");
    }
}

/// `LazyFunctionPrototype::materialize` writes its own cell's state: moving an old function's
/// deferred prototype word out of its property (the function's own borrow records the move)
/// materializes the prototype, and only the old cell names it.
#[test]
fn deferred_prototype_materialization_records_its_cell() {
    let mut engine = engine(Tier::Interp);
    eval(&mut engine, "var F = function () {}; 0");
    engine.interp.gc_collect();
    let materialized = crate::value::prototypes_materialized();
    let is_object = crate::gc_test::in_frame(|| {
        let function = global_object(&mut engine, "F");
        let moved = function.borrow_mut().props.take_value("prototype");
        matches!(
            (moved).map(|__value| (__value).view()),
            Some(ValueView::Obj(_))
        )
    });
    assert!(is_object);
    assert_eq!(crate::value::prototypes_materialized() - materialized, 1);
    engine.interp.gc_collect();
    assert_eq!(eval(&mut engine, "F.prototype === undefined"), "true");
}

/// `Error.captureStackTrace`'s getter keeps its frames' names, engine strings, as engine
/// captures of the getter (S6 step 11b; before it, its host reporter reported them): they
/// survive minor and major collections, and the stack reads after them.
#[test]
fn captured_stack_getter_keeps_its_frame_names() {
    let mut engine = engine(Tier::Interp);
    let source = r#"
        function named() { var e = {}; Error.captureStackTrace(e); return e; }
        var e = named();
        0
    "#;
    eval(&mut engine, source);
    for _ in 0..2 {
        eval(
            &mut engine,
            "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
        );
        engine.interp.gc_collect_minor();
        engine.interp.gc_collect();
    }
    assert_eq!(eval(&mut engine, "/at named/.test(e.stack)"), "true");
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod arm64 {
    //! The ARM64 JIT's generated stores (S6 survey A6's sites A-E), each written by compiled
    //! code after its cache is warm. Each scenario runs in an engine of its own, where its
    //! holder is young until `k === 50` collects inside its loop; then the site stores values
    //! allocated since, and a collection from Rust checks them (a collection called from
    //! compiled code cannot unwind out of it, so a failed check there aborts the process).
    use super::*;

    /// Run `prelude`, then `stores` (see the module), collect, and return `result`'s value.
    fn jit_stores(prelude: &str, stores: &str, result: &str) -> String {
        let mut engine = engine(Tier::Jit);
        eval(&mut engine, prelude);
        eval(&mut engine, stores);
        engine.interp.gc_collect();
        eval(&mut engine, result)
    }

    /// Site A: `emit_set_prop_inline`'s update commit, an own data property of an old object.
    #[test]
    fn jit_property_update_records_its_receiver() {
        let stores = r#"
            var holder = { a: null };
            function store(o, v) { o.a = v; }
            function run() {
                for (let k = 0; k < 60; k++) {
                    if (k === 50) collect();
                    store(holder, { k });
                }
            }
            run();
        "#;
        assert_eq!(jit_stores("0", stores, "holder.a.k"), "59");
    }

    /// Site B: `emit_set_prop_inline`'s creation commit, a field added to a constructor's
    /// `this`, made old by a collection inside the constructor before the field is stored.
    #[test]
    fn jit_property_creation_records_its_receiver() {
        let stores = r#"
            function C(late) { if (late) collect(); this.a = { late }; }
            for (let k = 0; k < 50; k++) new C(false);
            var made = new C(true);
        "#;
        assert_eq!(jit_stores("0", stores, "made.a.late"), "true");
    }

    /// Site C from a free-name cache: a function's store to a global variable (the loop runs in
    /// a function, so the script itself stores no global).
    #[test]
    fn jit_global_property_store_from_a_cache_records_the_global_object() {
        let stores = r#"
            var g = null;
            function setg(v) { g = v; }
            function run() {
                for (let k = 0; k < 60; k++) {
                    if (k === 50) collect();
                    setg({ k });
                }
            }
            run();
        "#;
        assert_eq!(jit_stores("0", stores, "g.k"), "59");
    }

    /// Site C from a Reference: a script's own store to a global variable an earlier script
    /// declared.
    #[test]
    fn jit_global_property_store_from_a_reference_records_the_global_object() {
        let stores = r#"
            for (let k = 0; k < 60; k++) {
                if (k === 50) collect();
                r = { k };
            }
        "#;
        assert_eq!(jit_stores("var r = null; 0", stores, "r.k"), "59");
    }

    /// The loop of D's scenarios: `store` warms, then stores young values after `k === 50`.
    const DRIVE: &str = r#"
        function drive(store) {
            for (let k = 0; k < 60; k++) {
                if (k === 50) collect();
                store({ k });
            }
        }
    "#;

    /// Site D: `emit_store_name_value`'s wide binding, from a captured binding's own function
    /// (`StoreCap`): the running environment.
    #[test]
    fn jit_binding_store_records_a_captured_bindings_environment() {
        let stores = r#"
            function own() {
                let x = null;
                const get = () => x;
                for (let k = 0; k < 60; k++) {
                    if (k === 50) collect();
                    x = { k };
                }
                return get;
            }
            var get = own();
        "#;
        assert_eq!(jit_stores("0", stores, "get().k"), "59");
    }

    /// Site D from a free-name cache in direct mode: a closure that makes no environment of
    /// its own writes its defining function's binding, in the running environment.
    #[test]
    fn jit_binding_store_records_the_running_environment() {
        let stores = format!(
            "{DRIVE} function outer() {{ let x = null; return [v => {{ x = v; }}, () => x]; }}
             var fns = outer(); drive(fns[0]);"
        );
        assert_eq!(jit_stores("0", &stores, "fns[1]().k"), "59");
    }

    /// Site D from a cache in fixed-layout mode: the same closure code in two environments of
    /// one layout.
    #[test]
    fn jit_binding_store_records_a_fixed_layout_environment() {
        let stores = format!(
            "{DRIVE} function outer() {{ let x = null; return [v => {{ x = v; }}, () => x]; }}
             var a = outer(), b = outer();
             for (let k = 0; k < 20; k++) {{ a[0]({{ k }}); b[0]({{ k }}); }}
             drive(b[0]);"
        );
        assert_eq!(jit_stores("0", &stores, "b[1]().k"), "59");
    }

    /// Site D from a cache in depth-one mode: a function with an environment of its own writes
    /// a global lexical binding in its parent, the global environment, which publishes no
    /// layout.
    #[test]
    fn jit_binding_store_records_the_parent_environment() {
        let stores = format!(
            "{DRIVE} function set(v) {{ let s = v; const keep = () => s; gx = s; }} drive(set);"
        );
        assert_eq!(jit_stores("let gx = null; 0", &stores, "gx.k"), "59");
    }

    /// Holders whose layouts are published, one and two environments up (deep mode).
    const DEEP: &str = r#"
        function outer() {
            let x = null;
            function near(v) { let s = v; const keep = () => s; x = s; }
            function middle() {
                let m = 0;
                return function far(v) { let s = v; const keep = () => s + m; x = s; };
            }
            return [near, middle(), () => x];
        }
        var fns = outer();
    "#;

    /// Site D from a cache in deep mode, the holder one environment up.
    #[test]
    fn jit_binding_store_records_a_deep_holder_one_up() {
        let stores = format!("{DRIVE} {DEEP} drive(fns[0]);");
        assert_eq!(jit_stores("0", &stores, "fns[2]().k"), "59");
    }

    /// Site D from a cache in deep mode, the holder two environments up.
    #[test]
    fn jit_binding_store_records_a_deep_holder_two_up() {
        let stores = format!("{DRIVE} {DEEP} drive(fns[1]);");
        assert_eq!(jit_stores("0", &stores, "fns[2]().k"), "59");
    }

    /// Site D from a Reference: a script's store to a global lexical binding an earlier
    /// script declared, in the Reference's scope, from inside a block whose captured binding
    /// gives it an environment of its own: the scope is not the running environment.
    #[test]
    fn jit_binding_store_records_a_references_scope() {
        let stores = r#"
            {
                let y = 0;
                const keep = () => y;
                for (let k = 0; k < 60; k++) {
                    if (k === 50) collect();
                    h = { k };
                }
            }
        "#;
        assert_eq!(jit_stores("let h = null; 0", stores, "h.k"), "59");
    }

    /// Site E: `emit_define_elem_inline`'s element append (CreateDataPropertyOrThrow in the
    /// self-hosted `map`), into a result array made old by a collection in the callback.
    #[test]
    fn jit_element_append_records_its_array() {
        let stores = r#"
            var late = false;
            function f(x) { if (late && x === 2) collect(); return { x }; }
            for (let k = 0; k < 20; k++) [1, 2, 3, 4, 5, 6].map(f);
            late = true;
            var out = [1, 2, 3, 4, 5, 6].map(f);
        "#;
        assert_eq!(
            jit_stores("0", stores, "out.map(o => o.x).join()"),
            "1,2,3,4,5,6"
        );
    }
}
