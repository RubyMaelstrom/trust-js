//! S6 step 11a: rooted handles (`handles::HandleTable`; `embed::Handle`, `WeakHandle` and
//! `CountedHandle`). What a handle keeps alive across minor and major collections, wherever it
//! is stored; what a weak handle lets die, and when it is cleared; how a handle of a dropped
//! Engine, or one used with another Engine, is refused; the records a minor collection reads,
//! and the debug check that catches a write that bypassed them; the memory record's count.
//!
//! The values a test watches are made in helper frames of their own, so that only handles keep
//! them (see `gc_test`). The tests of survival run their engines' stress collections under
//! `TRUST_JS_GC_STRESS`; those that watch what a particular collection does run on quiet engines
//! (no stress or task-boundary collections). Handles are made from a scope's values and read
//! back into a scope (S6 step 11b).

use std::cell::Cell;
use std::rc::Rc;

use crate::embed::{CountedHandle, Ctx, Handle, HandleError, Value, WeakHandle};
use crate::gc_core::{CellAge, CollectionKind};
use crate::gc_test::{alive, HiddenAddress};
use crate::host::{HostGc, HostGcVisitor, HostTrace, NativeGcId};
use crate::interpreter::gc_collect::ClientReport;
use crate::value::{Gc, WeakGc};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("handle fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// The value of `source` in the scope `cx`, evaluated without a microtask checkpoint.
fn value_of<'s>(cx: &mut Ctx<'s>, source: &str) -> Value<'s> {
    match cx.eval_value(source).expect("handle fixture parses") {
        Ok(value) => value,
        Err(_) => panic!("{source} threw"),
    }
}

/// A handle to the value of `source`, made in a frame of its own.
#[inline(never)]
fn handle_of(engine: &mut Engine, source: &str) -> Handle {
    engine.scope(|cx| {
        let value = value_of(cx, source);
        Handle::new(cx, &value).expect("a handle to a value of the engine")
    })
}

/// An engine whose heap runs no stress or task-boundary collections: for tests that watch what
/// one collection does.
fn quiet_engine() -> Engine {
    let mut engine = Engine::new();
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine.interp.gc_task_deferred = true;
    engine
}

/// The engine handle of the object `value` is, watched weakly.
fn observe(value: &Value<'_>) -> WeakGc {
    Gc::downgrade(&value.as_obj().expect("an object").gc())
}

/// A new object `{tag}`, held only by the returned handle, and a weak observer of it; made in a
/// frame of its own.
#[inline(never)]
fn held_object(engine: &mut Engine, tag: &str) -> (Handle, WeakGc) {
    engine.scope(|cx| {
        let value = value_of(cx, &format!("({{tag: '{tag}'}})"));
        let handle = Handle::new(cx, &value).expect("a handle to a value of the engine");
        (handle, observe(&value))
    })
}

/// A new object, named only by the returned weak handle, and its address; made in a frame of
/// its own.
#[inline(never)]
fn weak_object(engine: &mut Engine) -> (WeakHandle, HiddenAddress) {
    engine.scope(|cx| {
        let value = value_of(cx, "({tag: 'weak'})");
        let weak = WeakHandle::new(cx, &value).expect("a weak handle to a value of the engine");
        (weak, HiddenAddress::of(value.as_internal()))
    })
}

/// A new object held only by a counted handle of `count`, and a weak observer of it; made in a
/// frame of its own.
#[inline(never)]
fn counted_object(engine: &mut Engine, count: u32) -> (CountedHandle, WeakGc) {
    engine.scope(|cx| {
        let value = value_of(cx, "({tag: 'counted'})");
        let counted = CountedHandle::new(cx, &value, count).expect("a counted handle");
        (counted, observe(&value))
    })
}

/// The tag of the object `handle` holds, read in a frame of its own.
#[inline(never)]
fn tag(engine: &mut Engine, handle: &Handle) -> String {
    engine.scope(|cx| {
        let value = handle.get(cx).expect("the engine's handle");
        let tag = cx
            .member_get(&value, "tag")
            .unwrap_or_else(|_| panic!("reading the tag threw"));
        tag.as_text().expect("a string tag").into_owned()
    })
}

/// The age of the cell `handle` holds, asked in a frame of its own.
#[inline(never)]
fn age(engine: &mut Engine, handle: &Handle) -> Option<CellAge> {
    engine.scope(|cx| {
        let word = handle
            .get(cx)
            .expect("the engine's handle")
            .into_internal()
            .cell_word()
            .expect("a cell value");
        cx.interp().gc_heap.core.borrow().cell_age(word)
    })
}

/// Whether `weak` still names its value, asked in a frame of its own.
#[inline(never)]
fn named(engine: &mut Engine, weak: &WeakHandle) -> bool {
    engine.scope(|cx| weak.upgrade(cx).expect("the engine's handle").is_some())
}

/// Whether the counted handle still names its value, asked in a frame of its own.
#[inline(never)]
fn counted_named(engine: &mut Engine, counted: &CountedHandle) -> bool {
    engine.scope(|cx| counted.get(cx).expect("the engine's handle").is_some())
}

/// What the client of the heap's last collection reported.
fn report(engine: &Engine) -> ClientReport {
    engine.interp.gc_heap.client_report.borrow().clone()
}

/// The error of a refused handle operation.
fn err<T>(result: Result<T, HandleError>) -> HandleError {
    match result {
        Ok(_) => panic!("the handle operation was not refused"),
        Err(error) => error,
    }
}

fn stressed() -> bool {
    crate::interpreter::gc_collect::gc_stress() > 0
}

// ---------------------------------------------------------------------------------------------
// Survival.

/// A young object held only by a handle survives a minor collection, which promotes it, and
/// then a major one, and stays readable through the handle.
#[test]
fn a_handle_keeps_a_young_object_through_minor_and_major_collections() {
    let mut engine = Engine::new();
    engine.interp.gc_collect();
    let (handle, weak) = held_object(&mut engine, "young");
    if !stressed() {
        assert_eq!(age(&mut engine, &handle), Some(CellAge::Young));
    }
    engine.interp.gc_collect_minor();
    assert!(alive(&weak), "the minor collection kept it");
    assert_eq!(
        age(&mut engine, &handle),
        Some(CellAge::Old),
        "and promoted it"
    );
    eval(&mut engine, "for (let i = 0; i < 3000; i++) ({i}); 0");
    engine.interp.gc_collect();
    assert!(alive(&weak), "the major collection kept it");
    assert_eq!(tag(&mut engine, &handle), "young");
}

/// An old object held only by a handle survives minor collections that do not report its
/// entry (not written since the last collection: the cell stays marked) and then major ones,
/// which report every strong entry.
#[test]
fn a_handle_keeps_an_old_object_that_nothing_else_holds() {
    let mut engine = Engine::new();
    let (handle, weak) = held_object(&mut engine, "old");
    engine.interp.gc_collect();
    assert_eq!(age(&mut engine, &handle), Some(CellAge::Old));
    for _ in 0..3 {
        eval(
            &mut engine,
            "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
        );
        engine.interp.gc_collect_minor();
        assert!(alive(&weak), "a minor collection kept it");
    }
    for _ in 0..2 {
        engine.interp.gc_collect();
        assert!(alive(&weak), "a major collection kept it");
    }
    assert_eq!(tag(&mut engine, &handle), "old");
}

/// A value held by clones of a handle lives until the last clone drops, and dies at the next
/// collection after that.
#[test]
fn a_value_dies_once_the_last_handle_to_it_drops() {
    let mut engine = Engine::new();
    let (handle, weak) = held_object(&mut engine, "released");
    let clone = handle.clone();
    drop(handle);
    engine.interp.gc_collect();
    assert!(alive(&weak), "a clone still holds it");
    assert_eq!(tag(&mut engine, &clone), "released");
    drop(clone);
    engine.interp.gc_collect();
    assert!(!alive(&weak), "the last handle dropped");
}

#[derive(Default)]
struct Kept(Vec<Handle>);

/// `keep(value)`: hold `value` in the host state, in a handle made through the native call's
/// context.
fn keep<'s>(ctx: &mut Ctx<'s>, _: Value<'s>, args: &[Value<'s>]) -> Result<Value<'s>, Value<'s>> {
    let value = args.first().copied().unwrap_or(Value::Undefined);
    let handle =
        Handle::new(ctx, &value).map_err(|e| ctx.make_error("TypeError", e.to_string()))?;
    ctx.host_mut::<Kept>().expect("kept state").0.push(handle);
    Ok(Value::Undefined)
}

/// Host state that keeps values in handles needs no tracing: an object and a string made by a
/// script and kept only in an `OpState` slot, through handles made with the native call's
/// context, survive minor and major collections and the script's churn.
#[test]
fn host_state_keeps_values_in_handles_made_through_the_native_context() {
    let mut engine = Engine::new();
    engine.op_state().put(Kept::default());
    engine.define_global("keep", 1, keep);
    eval(
        &mut engine,
        "keep({tag: 'kept object'}); keep(['a string', 'made at run time'].join(' ')); 0",
    );
    for _ in 0..2 {
        eval(
            &mut engine,
            "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
        );
        engine.interp.gc_collect_minor();
        engine.interp.gc_collect();
    }
    let kept = std::mem::take(&mut engine.host_mut::<Kept>().expect("kept").0);
    assert_eq!(tag(&mut engine, &kept[0]), "kept object");
    let string = engine.scope(|cx| {
        let string = kept[1].get(cx).expect("the engine's handle");
        string.as_text().map(|text| text.into_owned())
    });
    assert_eq!(string.as_deref(), Some("a string made at run time"));
}

// ---------------------------------------------------------------------------------------------
// Inertness and rejection.

/// A handle of a dropped Engine is inert: reading it errs, and cloning, downgrading and
/// dropping it do not touch the freed heap.
#[test]
fn handles_of_a_dropped_engine_are_inert() {
    let mut other = Engine::new();
    let (strong, weak, mut counted) = {
        let mut engine = Engine::new();
        engine.scope(|cx| {
            let value = value_of(cx, "({tag: 'gone'})");
            (
                Handle::new(cx, &value).expect("a handle"),
                WeakHandle::new(cx, &value).expect("a weak handle"),
                CountedHandle::new(cx, &value, 1).expect("a counted handle"),
            )
        })
    };
    other.scope(|cx| {
        assert_eq!(err(strong.get(cx)), HandleError::Released);
        assert_eq!(err(weak.upgrade(cx)), HandleError::Released);
        assert_eq!(err(counted.get(cx)), HandleError::Released);
    });
    assert!(
        weak.is_cleared(),
        "a released heap's weak handle is cleared"
    );
    assert_eq!(err(strong.downgrade()), HandleError::Released);
    assert_eq!(err(counted.decrement()), HandleError::Released);
    let clone = strong.clone();
    other.scope(|cx| assert_eq!(err(clone.get(cx)), HandleError::Released));
    drop((clone, strong, weak.clone(), weak, counted));
}

/// A handle read through another Engine's scope is refused, and still reads through its own
/// Engine. A value of another Engine cannot reach a scope (its brand is another scope's: see
/// the embed module's compile_fail examples); the handles' own check refuses one made to cross
/// regardless, here forged from the engine's internal value.
#[test]
fn handles_and_values_of_another_engine_are_refused() {
    let mut first = Engine::new();
    let mut second = Engine::new();
    let handle = handle_of(&mut first, "({tag: 'first'})");
    let text = handle_of(&mut first, "['first', 'engine'].join(' ')");
    // The engine values, unbranded (the crate's own form): forged into the second's scopes.
    let (object, string) = first.scope(|cx| {
        (
            handle.get(cx).unwrap().into_internal(),
            text.get(cx).unwrap().into_internal(),
        )
    });
    second.scope(|cx| {
        assert_eq!(err(handle.get(cx)), HandleError::Foreign);
        let foreign = Value::from_internal(object.clone());
        assert_eq!(err(Handle::new(cx, &foreign)), HandleError::Foreign);
        assert_eq!(err(WeakHandle::new(cx, &foreign)), HandleError::Foreign);
        assert_eq!(
            err(CountedHandle::new(cx, &foreign, 1)),
            HandleError::Foreign
        );
        assert_eq!(err(handle.set(cx, &foreign)), HandleError::Foreign);
        let weak = handle.downgrade().expect("a weak handle");
        assert_eq!(err(weak.upgrade(cx)), HandleError::Foreign);
        // A string is a cell of its own Engine's heap too.
        let foreign = Value::from_internal(string.clone());
        assert_eq!(err(Handle::new(cx, &foreign)), HandleError::Foreign);
        // Primitives belong to no heap.
        let number = Handle::new(cx, &Value::Num(1.5)).expect("a primitive");
        assert_eq!(number.get(cx).expect("its own").as_num_opt(), Some(1.5));
    });
    drop((object, string));
    assert_eq!(tag(&mut first, &handle), "first");
}

// ---------------------------------------------------------------------------------------------
// Weak handles.

/// A weak handle to a young object nothing else holds is cleared by a minor collection.
#[test]
fn a_weak_handle_to_a_young_object_is_cleared_by_a_minor_collection() {
    let mut engine = quiet_engine();
    engine.interp.gc_collect();
    let (weak, _) = weak_object(&mut engine);
    assert!(named(&mut engine, &weak));
    assert!(!weak.is_cleared());
    engine.interp.gc_collect_minor();
    assert!(!named(&mut engine, &weak));
    assert!(weak.is_cleared());
}

/// A weak handle to an old object is cleared only by a major collection: a minor collection
/// leaves old dead cells allocated, and the handle names its object until a collection frees
/// it.
#[test]
fn a_weak_handle_to_an_old_object_is_cleared_by_a_major_collection() {
    let mut engine = quiet_engine();
    let (strong, observer) = held_object(&mut engine, "old");
    let weak = strong.downgrade().expect("a weak handle");
    engine.interp.gc_collect();
    assert_eq!(age(&mut engine, &strong), Some(CellAge::Old));
    drop(strong);
    engine.interp.gc_collect_minor();
    assert!(
        named(&mut engine, &weak),
        "a minor collection frees no old cell"
    );
    engine.interp.gc_collect();
    assert!(
        !named(&mut engine, &weak),
        "a major collection found it dead"
    );
    assert!(!alive(&observer));
}

/// A counted handle holds its value while its count is above 0 and lets it die at 0: strong to
/// weak and back to strong before a collection keeps it, and a value that died at 0 stays dead
/// when the count rises again.
#[test]
fn a_counted_handle_is_strong_above_zero_and_weak_at_zero() {
    let mut engine = quiet_engine();
    let (mut counted, weak) = counted_object(&mut engine, 1);
    engine.interp.gc_collect();
    assert!(alive(&weak), "a count of 1 keeps it");
    assert_eq!(counted.decrement(), Ok(0));
    assert_eq!(counted.increment(), Ok(1));
    assert_eq!(counted.increment(), Ok(2));
    engine.interp.gc_collect_minor();
    engine.interp.gc_collect();
    assert!(alive(&weak), "weak and strong again: kept");
    assert_eq!(counted.decrement(), Ok(1));
    assert_eq!(counted.decrement(), Ok(0));
    assert!(counted_named(&mut engine, &counted), "not collected yet");
    engine.interp.gc_collect();
    assert!(!alive(&weak), "a count of 0 let it die");
    assert!(!counted_named(&mut engine, &counted));
    assert_eq!(counted.increment(), Ok(1));
    assert!(
        !counted_named(&mut engine, &counted),
        "a dead value stays dead"
    );
    // A young value at a count of 0 dies at a minor collection.
    let (counted, weak) = counted_object(&mut engine, 0);
    engine.interp.gc_collect_minor();
    assert!(!alive(&weak));
    assert!(!counted_named(&mut engine, &counted));
}

/// Whether a cell is allocated at `address` again, asked in a frame of its own.
#[inline(never)]
fn reallocated(engine: &Engine, address: HiddenAddress) -> bool {
    let core = engine.interp.gc_heap.core.borrow();
    matches!(
        core.cell_age(address.get() as u64),
        Some(CellAge::Young | CellAge::Old)
    )
}

/// A weak handle cleared as its object died never yields the object a later allocation puts at
/// the same address.
#[test]
fn a_weak_handle_never_yields_a_new_object_at_its_dead_objects_address() {
    let mut engine = quiet_engine();
    engine.interp.gc_collect();
    let (weak, address) = weak_object(&mut engine);
    engine.interp.gc_collect_minor();
    assert!(!named(&mut engine, &weak), "cleared as its object died");
    assert!(!reallocated(&engine, address), "its cell was freed");
    eval(
        &mut engine,
        "var fill = []; for (let i = 0; i < 20000; i++) fill.push({i}); 0",
    );
    assert!(
        reallocated(&engine, address),
        "a new object took the dead object's address"
    );
    assert!(!named(&mut engine, &weak));
}

// ---------------------------------------------------------------------------------------------
// The records minor collections read.

/// A major collection reports every strong entry; a minor one only the strong entries written
/// since the last collection: made, given a new value or made strong again. Weak entries are
/// never reported as roots.
#[test]
fn a_minor_collection_reports_only_the_handles_written_since_the_last_collection() {
    let mut engine = quiet_engine();
    let (first, _) = held_object(&mut engine, "first");
    let (keeper, _) = held_object(&mut engine, "counted");
    let mut counted = engine.scope(|cx| {
        let kept = keeper.get(cx).expect("the engine's handle");
        CountedHandle::new(cx, &kept, 0).expect("a counted handle")
    });
    engine.interp.gc_collect();
    let major = report(&engine);
    assert_eq!(major.kind, Some(CollectionKind::Major));
    assert_eq!(major.handle_slots, vec![first.slot(), keeper.slot()]);
    engine.interp.gc_collect_minor();
    let idle = report(&engine);
    assert_eq!(idle.kind, Some(CollectionKind::Minor));
    assert!(
        idle.handle_slots.is_empty(),
        "nothing was written: {idle:?}"
    );
    let (second, _) = held_object(&mut engine, "second");
    let _weak = second.downgrade().expect("a weak handle");
    engine.interp.gc_collect_minor();
    assert_eq!(report(&engine).handle_slots, vec![second.slot()], "made");
    engine.scope(|cx| {
        let young = value_of(cx, "({tag: 'retargeted'})");
        first.set(cx, &young).expect("a value of the engine");
    });
    engine.interp.gc_collect_minor();
    assert_eq!(
        report(&engine).handle_slots,
        vec![first.slot()],
        "retargeted"
    );
    assert_eq!(tag(&mut engine, &first), "retargeted");
    assert_eq!(counted.increment(), Ok(1));
    engine.interp.gc_collect_minor();
    assert_eq!(
        report(&engine).handle_slots,
        vec![counted.slot()],
        "made strong"
    );
    engine.interp.gc_collect();
    let mut all = report(&engine).handle_slots;
    all.sort_unstable();
    let mut expected = vec![first.slot(), keeper.slot(), second.slot(), counted.slot()];
    expected.sort_unstable();
    assert_eq!(
        all, expected,
        "a major collection reports every strong entry"
    );
}

/// The handles' check panics, as the next collection starts, when a strong entry holding a
/// young value was not recorded: a write that bypassed the record, which a minor collection
/// would miss.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "strong entry")]
fn an_unrecorded_strong_handle_fails_the_check() {
    let mut engine = quiet_engine();
    engine.interp.gc_collect();
    let (handle, _) = held_object(&mut engine, "young");
    engine.interp.gc_heap.handles.forget_written(handle.slot());
    engine.interp.gc_collect_minor();
}

/// The check also panics on an unrecorded weak entry naming a young value: a minor collection
/// clears only the weak entries written since the last collection.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "weak entry")]
fn an_unrecorded_weak_handle_fails_the_check() {
    let mut engine = quiet_engine();
    engine.interp.gc_collect();
    let (weak, _) = weak_object(&mut engine);
    engine.interp.gc_heap.handles.forget_written(weak.slot());
    engine.interp.gc_collect_minor();
}

/// A host sweep that writes a handle, between a collection's phases.
struct WriteInSweep {
    handle: Handle,
    result: Rc<Cell<Option<Result<(), HandleError>>>>,
}

impl HostTrace for WriteInSweep {
    fn trace_gc(&self, _: &mut dyn HostGcVisitor) {}
}

impl HostGc for WriteInSweep {
    fn sweep_gc_with_native(&mut self, _: &dyn Fn(NativeGcId) -> bool) {
        self.result.set(Some(self.handle.downgrade().map(drop)));
    }
}

/// A sweep of `engine`'s host state writes a handle at its next collection.
fn write_in_sweep(engine: &mut Engine) -> Rc<Cell<Option<Result<(), HandleError>>>> {
    let result = Rc::new(Cell::new(None));
    let handle = handle_of(engine, "({})");
    engine.op_state().put(WriteInSweep {
        handle,
        result: result.clone(),
    });
    engine.op_state().register_gc::<WriteInSweep>();
    result
}

/// No handle can be written while a collection runs: debug builds panic.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "while a collection of its heap runs")]
fn writing_a_handle_during_a_collection_panics_in_debug_builds() {
    let mut engine = Engine::new();
    let _result = write_in_sweep(&mut engine);
    engine.interp.gc_collect();
}

/// No handle can be written while a collection runs: release builds refuse it.
#[test]
#[cfg(not(debug_assertions))]
fn writing_a_handle_during_a_collection_is_refused() {
    let mut engine = Engine::new();
    let result = write_in_sweep(&mut engine);
    engine.interp.gc_collect();
    assert_eq!(result.get(), Some(Err(HandleError::Collecting)));
}

// ---------------------------------------------------------------------------------------------
// The memory record.

/// The managed-memory record counts the heap's live handles, strong and weak: an entry per
/// handle and its clones, until the last drops.
#[test]
fn the_memory_record_counts_the_live_handles() {
    let mut engine = Engine::new();
    let strong = handle_of(&mut engine, "({})");
    let clone = strong.clone();
    let weak = strong.downgrade().expect("a weak handle");
    let counted = engine.scope(|cx| {
        let value = strong.get(cx).expect("the engine's handle");
        CountedHandle::new(cx, &value, 0).expect("a counted handle")
    });
    let json = crate::memory::json(&engine.interp);
    assert!(
        json.contains("\"handles\":{\"strong\":1,\"weak\":2}"),
        "{json}"
    );
    drop((strong, weak, counted));
    let json = crate::memory::json(&engine.interp);
    assert!(
        json.contains("\"handles\":{\"strong\":1,\"weak\":0}"),
        "{json}"
    );
    drop(clone);
    let json = crate::memory::json(&engine.interp);
    assert!(
        json.contains("\"handles\":{\"strong\":0,\"weak\":0}"),
        "{json}"
    );
}
