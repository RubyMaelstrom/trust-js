//! S6 step 10d: a minor collection reports, of the engine's side-slot tables, its WeakMap and
//! WeakSet entries and its built-ins' [[Realm]] edges, only what was written since the last
//! collection (`gc_collect::ClientRecords`): an old entry not written since holds old values,
//! which the last collection traced. Each restriction is tested three ways: the client's report
//! (`GcState::client_report`) names the written and young entries and none of the old, clean
//! ones; a young value written into an old entry survives a minor collection; and debug builds'
//! check, as the next collection starts, panics when a write bypasses the record.
//!
//! These tests watch what each collection reports, so their engines' heaps run no stress
//! collections, whatever `TRUST_JS_GC_STRESS` says, and no task-boundary collections. They
//! collect from Rust, holding the young values they watch only in helper frames (see
//! `gc_test`).

use crate::bytecode::Tier;
use crate::gc_core::CollectionKind;
use crate::gc_test::{alive, HiddenAddress};
use crate::interpreter::gc_collect::ClientReport;
#[cfg(debug_assertions)]
use crate::interpreter::WeakKey;
use crate::interpreter::{new_scope, Env, Interp, WeakEnv};
use crate::value::ValueView;
use crate::value::{Gc, Property, Value, WeakGc};
use crate::{Completion, Engine};

/// An engine at `tier` (threshold 0) whose heap runs no stress or task-boundary collections.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine.interp.gc_task_deferred = true;
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("record fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// The global object named `name`.
fn global_object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env.clone();
    match (engine.interp.get_var(name, &env)).map(|__value| (__value).view()) {
        Ok(ValueView::Obj(object)) => object,
        _ => panic!("{name}: not an object"),
    }
}

/// The address of the global object named `name`, kept without retaining it.
#[inline(never)]
fn global_address(engine: &mut Engine, name: &str) -> HiddenAddress {
    HiddenAddress::of(&Value::Obj(global_object(engine, name)))
}

/// A weak handle to the global object named `name`, made in a frame of its own.
#[inline(never)]
fn weak_global(engine: &mut Engine, name: &str) -> WeakGc {
    Gc::downgrade(&global_object(engine, name))
}

/// What the client of the heap's last collection reported.
fn report(engine: &Engine) -> ClientReport {
    engine.interp.gc_heap.client_report.borrow().clone()
}

/// Whether `address` is among `list`, asked in a frame of its own.
#[inline(never)]
fn listed(address: HiddenAddress, list: &[usize]) -> bool {
    list.contains(&address.get())
}

/// How many of `entries` are entries of the collection at `owner`.
#[inline(never)]
fn entries_of(owner: HiddenAddress, entries: &[(usize, u64)]) -> usize {
    entries.iter().filter(|&&(of, _)| of == owner.get()).count()
}

// ---------------------------------------------------------------------------------------------
// Side-slot owners (`gc_side_slot_tables!` and the generators).

/// A minor collection reports the side-slot owners written since the last collection and the
/// young ones, and no old owner whose entries were not written: a reaction added to an old
/// promise records it, and the reaction's new promise is young; an old pending promise and an
/// old proxy beside it are not reported. A major collection reports every owner.
#[test]
fn a_minor_collection_reports_written_and_young_side_slot_owners_only() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            "var written = new Promise(function () {}), clean = new Promise(function () {});
             var proxy = new Proxy({}, {}); 0",
        );
        let written = global_address(&mut engine, "written");
        let clean = global_address(&mut engine, "clean");
        let proxy = global_address(&mut engine, "proxy");
        engine.interp.gc_collect();
        let major = report(&engine);
        assert_eq!(major.kind, Some(CollectionKind::Major));
        for owner in [written, clean, proxy] {
            assert!(
                listed(owner, &major.side_slot_owners),
                "{tier:?}: a major collection"
            );
        }
        // Nothing was written since: a minor collection reports no table owner.
        engine.interp.gc_collect_minor();
        let idle = report(&engine);
        assert_eq!(idle.kind, Some(CollectionKind::Minor));
        assert!(idle.side_slot_owners.is_empty(), "{tier:?}: {idle:?}");
        eval(
            &mut engine,
            "var derived = written.then(function (v) { return v; }); 0",
        );
        let derived = global_address(&mut engine, "derived");
        engine.interp.gc_collect_minor();
        let minor = report(&engine);
        assert!(
            listed(written, &minor.side_slot_owners),
            "{tier:?}: written"
        );
        assert!(listed(derived, &minor.side_slot_owners), "{tier:?}: young");
        assert!(!listed(clean, &minor.side_slot_owners), "{tier:?}: clean");
        assert!(
            !listed(proxy, &minor.side_slot_owners),
            "{tier:?}: clean proxy"
        );
    }
}

/// Young values written into old owners' entries survive a minor collection, reached only
/// through those entries: a reaction added to an old promise, a value sent into an old
/// suspended generator, a held value registered with an old FinalizationRegistry. Each is used
/// afterwards. At every tier.
#[test]
fn young_values_written_into_old_side_slot_entries_survive_a_minor_collection() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            r#"
            var settle, pending = new Promise(function (resolve) { settle = resolve; });
            var out = [];
            function* gen() { var held = yield 1; yield 2; yield held.tag; }
            var it = gen(); it.next();
            var registry = new FinalizationRegistry(function (held) { out.push(held.tag); });
            var target = {};
            0
            "#,
        );
        engine.interp.gc_collect();
        eval(
            &mut engine,
            r#"
            var reaction = { tag: 'reaction' }, sent = { tag: 'sent' }, held = { tag: 'held' };
            pending.then((function (kept) {
                return function () { out.push(kept.tag); };
            })(reaction));
            it.next(sent);
            registry.register(target, held);
            0
            "#,
        );
        let watched = ["reaction", "sent", "held"].map(|name| weak_global(&mut engine, name));
        eval(&mut engine, "reaction = sent = held = null; 0");
        engine.interp.gc_collect_minor();
        for (weak, name) in watched.iter().zip(["reaction", "sent", "held"]) {
            assert!(alive(weak), "{tier:?}: {name}");
        }
        eval(&mut engine, "settle(); target = null; 0");
        assert_eq!(
            eval(&mut engine, "out.join() + ':' + it.next().value"),
            "reaction:sent",
            "{tier:?}"
        );
        // The target dies old: a major collection finds it, and the cleanup gets the held value.
        engine.interp.gc_collect();
        eval(&mut engine, "0");
        assert_eq!(eval(&mut engine, "out.join()"), "reaction,held", "{tier:?}");
    }
}

/// One side-slot table's write from Rust: `owner`'s entry holding the object `young` and, where
/// the table's entries hold an environment, `env`; and its removal.
struct TableWrite {
    table: &'static str,
    /// Whether the entry holds the object, and whether it holds the environment.
    holds: (bool, bool),
    write: fn(&mut Interp, usize, Gc, Env),
    remove: fn(&mut Interp, usize),
}

/// A write to each of the 12 side-slot tables (`gc_side_slot_tables!` and the generators).
fn table_writes() -> [TableWrite; 12] {
    use crate::interpreter::{ClassInfo, FinalizationState, HostIndexedProperties, JobCallback};
    [
        TableWrite {
            table: "class_info",
            holds: (true, true),
            write: |i, owner, young, env| {
                i.class_info.insert(
                    &i.gc_heap,
                    owner,
                    ClassInfo {
                        fields: Vec::new().into(),
                        field_env: env,
                        derived: false,
                        instance_initializers: vec![Value::Obj(young)],
                        private_members: Vec::new(),
                    },
                );
            },
            remove: |i, owner| {
                i.class_info.remove(&owner);
            },
        },
        TableWrite {
            table: "module_ns",
            holds: (true, false),
            write: |i, owner, young, _| {
                let mut bindings = crate::fasthash::FastMap::default();
                bindings.insert(
                    "young".to_string(),
                    crate::modules::NsBinding::Static(Value::Obj(young)),
                );
                i.module_ns.insert(&i.gc_heap, owner, bindings);
            },
            remove: |i, owner| {
                i.module_ns.remove(&owner);
            },
        },
        TableWrite {
            table: "collection_iterators",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.collection_iterators.insert(
                    &i.gc_heap,
                    owner,
                    crate::ordered_collection::CollectionIterator {
                        target: Some(young),
                        cursor: None,
                        kind: 0,
                        brand: crate::ordered_collection::CollectionKind::Map,
                    },
                );
            },
            remove: |i, owner| {
                i.collection_iterators.remove(&owner);
            },
        },
        TableWrite {
            table: "ta_buffer",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.ta_buffer.insert(&i.gc_heap, owner, Value::Obj(young));
            },
            remove: |i, owner| {
                i.ta_buffer.remove(&owner);
            },
        },
        TableWrite {
            table: "proxies",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.proxies
                    .insert(&i.gc_heap, owner, (Value::Obj(young), Value::Null));
            },
            remove: |i, owner| {
                i.proxies.remove(&owner);
            },
        },
        TableWrite {
            table: "host_indexed",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.host_indexed.insert(
                    &i.gc_heap,
                    owner,
                    HostIndexedProperties {
                        length: 0,
                        getter: Value::Obj(young),
                        live: None,
                    },
                );
            },
            remove: |i, owner| {
                i.host_indexed.remove(&owner);
            },
        },
        TableWrite {
            table: "promises",
            holds: (true, false),
            write: |i, owner, young, _| {
                // An old entry's value replaced through the mutable accessor.
                let state = i.promises.entry(&i.gc_heap, owner).or_default();
                state.value = Value::Obj(young);
            },
            remove: |i, owner| {
                i.promises.remove(&owner);
            },
        },
        TableWrite {
            table: "promise_forward",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.promise_forward
                    .insert(&i.gc_heap, owner, Value::Obj(young));
            },
            remove: |i, owner| {
                i.promise_forward.remove(&owner);
            },
        },
        TableWrite {
            table: "async_gen_queue",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.async_gen_queue
                    .entry(&i.gc_heap, owner)
                    .or_default()
                    .push_back((
                        Value::Obj(young),
                        crate::coroutine::Resume::Next(Value::Undefined),
                    ));
            },
            remove: |i, owner| {
                i.async_gen_queue.remove(&owner);
            },
        },
        TableWrite {
            table: "finalization_registries",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.finalization_registries.insert(
                    &i.gc_heap,
                    owner,
                    FinalizationState {
                        cleanup_callback: JobCallback::plain(Value::Obj(young)),
                        cells: Default::default(),
                        cleanup_scheduled: false,
                    },
                );
            },
            remove: |i, owner| {
                i.finalization_registries.remove(&owner);
            },
        },
        TableWrite {
            table: "mapped_arguments",
            holds: (false, true),
            write: |i, owner, _, env| {
                i.mapped_arguments
                    .insert(&i.gc_heap, owner, (env, Vec::new()));
            },
            remove: |i, owner| {
                i.mapped_arguments.remove(&owner);
            },
        },
        TableWrite {
            table: "generators",
            holds: (true, false),
            write: |i, owner, young, _| {
                i.generators.insert(
                    &i.gc_heap,
                    owner,
                    crate::coroutine::unavailable(Value::Obj(young)),
                );
            },
            remove: |i, owner| {
                i.generators.remove(&owner);
            },
        },
    ]
}

/// Write a young object (and a young environment) into `owner`'s entry of `write`'s table, in a
/// frame of its own: the caller keeps only weak handles to them.
#[inline(never)]
fn write_young(interp: &mut Interp, write: &TableWrite, owner: HiddenAddress) -> (WeakGc, WeakEnv) {
    let young = interp.new_object();
    let env = new_scope(None);
    let watched = (Gc::downgrade(&young), Env::downgrade(&env));
    (write.write)(interp, owner.get(), young, env);
    watched
}

/// Each of the 12 side-slot tables, written from Rust: an old owner's entry gets a young object
/// (and a young environment), which then survives a minor collection through that entry alone;
/// the minor collection reports the written owner, and not an old owner beside it whose old
/// entry in the same table was not written.
#[test]
fn every_side_slot_table_records_its_writes_for_minor_collections() {
    let mut engine = engine(Tier::Interp);
    eval(&mut engine, "var owner = {}, clean = {}, old = {}; 0");
    let owner = global_address(&mut engine, "owner");
    let clean = global_address(&mut engine, "clean");
    for write in &table_writes() {
        // The clean owner's entry holds an old object and an old environment.
        let old = global_object(&mut engine, "old");
        let env = engine.interp.global_env.clone();
        (write.write)(&mut engine.interp, clean.get(), old, env);
        engine.interp.gc_collect();
        let (young, young_env) = write_young(&mut engine.interp, write, owner);
        engine.interp.gc_collect_minor();
        let minor = report(&engine);
        let (holds_object, holds_env) = write.holds;
        assert!(
            !holds_object || alive(&young),
            "{}: the object",
            write.table
        );
        assert!(
            !holds_env || alive(&young_env),
            "{}: the environment",
            write.table
        );
        assert!(
            listed(owner, &minor.side_slot_owners),
            "{}: written",
            write.table
        );
        assert!(
            !listed(clean, &minor.side_slot_owners),
            "{}: clean",
            write.table
        );
        (write.remove)(&mut engine.interp, owner.get());
        (write.remove)(&mut engine.interp, clean.get());
    }
}

/// The side-slot check panics when an old owner's entry got a young value that the record
/// missed: a write that bypasses the record.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "side-slot table promises: old owner")]
fn an_unrecorded_side_slot_write_fails_the_check() {
    let mut engine = engine(Tier::Interp);
    eval(&mut engine, "var p = new Promise(function () {}); 0");
    let promise = global_address(&mut engine, "p");
    engine.interp.gc_collect();
    let write = &table_writes()[6];
    assert_eq!(write.table, "promises");
    write_young(&mut engine.interp, write, promise);
    engine
        .interp
        .gc_heap
        .records
        .borrow_mut()
        .forget_side_slot_owner(promise.get());
    engine.interp.gc_collect_minor();
}

/// The side-slot check also panics on a young owner the record missed: a minor collection finds
/// young owners only through the record.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "side-slot table proxies: young owner")]
fn an_unrecorded_young_side_slot_owner_fails_the_check() {
    let mut engine = engine(Tier::Interp);
    engine.interp.gc_collect();
    eval(&mut engine, "var p = new Proxy({}, {}); 0");
    let proxy = global_address(&mut engine, "p");
    engine
        .interp
        .gc_heap
        .records
        .borrow_mut()
        .forget_side_slot_owner(proxy.get());
    engine.interp.gc_collect_minor();
}

// ---------------------------------------------------------------------------------------------
// WeakMap and WeakSet entries.

/// A minor collection registers as ephemerons the WeakMap entries written since the last
/// collection (a new value under an old key, a young key's entry) and none of the old, clean
/// ones; a major collection registers every entry. The values written survive, the young value
/// under the old key reached through its entry alone. At every tier.
#[test]
fn a_minor_collection_registers_written_and_young_keyed_weak_entries_only() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            "var map = new WeakMap(), set = new WeakSet(), keys = [];
             for (var i = 0; i < 8; i++) { keys.push({ i: i }); map.set(keys[i], { v: i }); set.add(keys[i]); }
             0",
        );
        let map = global_address(&mut engine, "map");
        engine.interp.gc_collect();
        assert_eq!(
            entries_of(map, &report(&engine).weak_entries),
            8,
            "{tier:?}: a major collection"
        );
        engine.interp.gc_collect_minor();
        assert!(
            report(&engine).weak_entries.is_empty(),
            "{tier:?}: nothing written"
        );
        eval(
            &mut engine,
            "var young = { v: 33 }, fresh = {}; map.set(keys[3], young); map.set(fresh, { v: 99 });
             set.add(fresh); 0",
        );
        let young = weak_global(&mut engine, "young");
        eval(&mut engine, "young = null; 0");
        engine.interp.gc_collect_minor();
        let minor = report(&engine);
        assert_eq!(
            entries_of(map, &minor.weak_entries),
            2,
            "{tier:?}: {minor:?}"
        );
        // A WeakSet entry has no value to register.
        assert_eq!(minor.weak_entries.len(), 2, "{tier:?}: {minor:?}");
        assert!(alive(&young), "{tier:?}: the young value under an old key");
        assert_eq!(
            eval(
                &mut engine,
                "map.get(keys[3]).v + ':' + map.get(fresh).v + ':' + set.has(fresh)"
            ),
            "33:99:true",
            "{tier:?}"
        );
    }
}

/// The weak-entry check panics when an old key's entry got a young value that the record missed.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "weak_collection_data: the entry of")]
fn an_unrecorded_weak_entry_write_fails_the_check() {
    let mut engine = engine(Tier::Interp);
    eval(
        &mut engine,
        "var map = new WeakMap(), key = {}; map.set(key, {}); 0",
    );
    let map = global_address(&mut engine, "map");
    let key = global_address(&mut engine, "key");
    engine.interp.gc_collect();
    eval(&mut engine, "map.set(key, { young: true }); 0");
    engine
        .interp
        .gc_heap
        .records
        .borrow_mut()
        .forget_weak_entry(map.get(), WeakKey::Object(key.get()));
    engine.interp.gc_collect_minor();
}

// ---------------------------------------------------------------------------------------------
// Built-ins' [[Realm]] edges.

/// A native function made from Rust, kept as the global `name`.
fn global_native(engine: &mut Engine, name: &str) {
    let function = engine
        .interp
        .make_native(name, 0, |_, _, _| Ok(Value::Undefined));
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert(name, Property::plain(Value::Obj(function)));
}

/// A minor collection reports the [[Realm]] edges of the built-ins registered since the last
/// collection only; a major collection reports every built-in's.
#[test]
fn a_minor_collection_reports_the_realm_edges_of_new_builtins_only() {
    let mut engine = engine(Tier::Interp);
    // The first collection frees what setting the engine up left behind, built-ins among it.
    engine.interp.gc_collect();
    engine.interp.gc_collect();
    let builtins = engine.interp.native_function_realms.borrow().len();
    assert!(builtins > 500, "{builtins} built-ins");
    assert_eq!(report(&engine).realm_edges.len(), builtins);
    engine.interp.gc_collect_minor();
    assert!(report(&engine).realm_edges.is_empty());
    global_native(&mut engine, "fresh");
    let fresh = global_address(&mut engine, "fresh");
    engine.interp.gc_collect_minor();
    let minor = report(&engine);
    assert_eq!(minor.realm_edges.len(), 1, "{minor:?}");
    assert!(listed(fresh, &minor.realm_edges));
    engine.interp.gc_collect_minor();
    assert!(report(&engine).realm_edges.is_empty());
    engine.interp.gc_collect();
    assert_eq!(report(&engine).realm_edges.len(), builtins + 1);
}

/// Register the [[Realm]] of the built-in the global `name` holds as a young object made here,
/// which only that edge reaches.
#[inline(never)]
fn young_realm_for(engine: &mut Engine, name: &str) -> WeakGc {
    let function = global_object(engine, name);
    let realm = engine.interp.new_object();
    engine.interp.native_function_realms.borrow_mut().register(
        &engine.interp.gc_heap,
        &function,
        Gc::as_ptr(&realm) as usize,
    );
    Gc::downgrade(&realm)
}

/// A young built-in's [[Realm]] edge keeps its target through a minor collection, and the edge
/// still holds once the built-in is old, through minor and major collections; the target dies
/// with the built-in, whose entry a major collection then evicts.
#[test]
fn a_young_builtins_realm_edge_keeps_its_realm_through_minor_collections() {
    let mut engine = engine(Tier::Interp);
    engine.interp.gc_collect();
    global_native(&mut engine, "fresh");
    let fresh = global_address(&mut engine, "fresh");
    let realm = young_realm_for(&mut engine, "fresh");
    engine.interp.gc_collect_minor();
    assert!(alive(&realm), "a young built-in's [[Realm]] edge");
    engine.interp.gc_collect_minor();
    assert!(alive(&realm), "an old built-in's");
    engine.interp.gc_collect();
    assert!(alive(&realm), "after a major collection");
    eval(&mut engine, "delete globalThis.fresh; 0");
    engine.interp.gc_collect();
    assert!(!alive(&realm), "the target dies with the built-in");
    assert!(!fresh.in_map(&engine.interp.native_function_realms.borrow()));
}

/// Make a native function no one keeps, in a frame of its own.
#[inline(never)]
fn dropped_native(engine: &mut Engine) -> HiddenAddress {
    let function = engine
        .interp
        .make_native("dropped", 0, |_, _, _| Ok(Value::Undefined));
    HiddenAddress::of(&Value::Obj(function))
}

/// A minor collection evicts the [[Realm]] entries of the young built-ins it finds dead (swept
/// from the functions registered since the last collection), and keeps an old one that died,
/// which only a major collection finds dead.
#[test]
fn a_minor_collection_evicts_young_dead_builtins_realm_entries() {
    let mut engine = engine(Tier::Interp);
    global_native(&mut engine, "kept");
    let old = global_address(&mut engine, "kept");
    engine.interp.gc_collect();
    eval(&mut engine, "delete globalThis.kept; 0");
    let young = dropped_native(&mut engine);
    assert!(young.in_map(&engine.interp.native_function_realms.borrow()));
    engine.interp.gc_collect_minor();
    assert!(
        !young.in_map(&engine.interp.native_function_realms.borrow()),
        "young and dead"
    );
    assert!(
        old.in_map(&engine.interp.native_function_realms.borrow()),
        "old and dead"
    );
    engine.interp.gc_collect();
    assert!(
        !old.in_map(&engine.interp.native_function_realms.borrow()),
        "after a major"
    );
}

/// The [[Realm]] check panics when an old built-in's entry names a young object that the record
/// missed: a write that bypasses the record.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "registered before the last collection, has an unmarked")]
fn an_unrecorded_realm_change_fails_the_check() {
    let mut engine = engine(Tier::Interp);
    global_native(&mut engine, "old");
    let old = global_address(&mut engine, "old");
    engine.interp.gc_collect();
    eval(&mut engine, "var young = {}; 0");
    let young = global_address(&mut engine, "young");
    engine
        .interp
        .native_function_realms
        .borrow_mut()
        .set_realm_unrecorded(old.get(), young.get());
    engine.interp.gc_collect_minor();
}

/// The [[Realm]] check also panics on a young built-in the record missed: a minor collection
/// reports young built-ins' edges, and sweeps their entries, only through the record.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "native_function_realms: young function")]
fn an_unrecorded_young_builtin_fails_the_check() {
    let mut engine = engine(Tier::Interp);
    engine.interp.gc_collect();
    global_native(&mut engine, "fresh");
    let fresh = global_address(&mut engine, "fresh");
    engine
        .interp
        .gc_heap
        .records
        .borrow_mut()
        .forget_native_function(fresh.get());
    engine.interp.gc_collect_minor();
}
