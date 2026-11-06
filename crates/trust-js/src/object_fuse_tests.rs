//! The ObjectFuse (`crate::object_fuse`): String.prototype's in every Realm, and how each kind of
//! write moves its properties' states, as SpiderMonkey's (js/src at fa1e90ea: vm/ObjectFuse.cpp,
//! vm/Watchtower.cpp, vm/NativeObject.cpp). Writes run in every tier, the loops warming the
//! store caches first.
use std::sync::Arc;

use crate::bytecode::Tier;
use crate::object_fuse::{ObjectFuse, PropertyState};
use crate::value::Value;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fuse fixture parses") {
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

fn fuse(engine: &Engine) -> Arc<ObjectFuse> {
    engine
        .interp
        .string_proto_fuse
        .clone()
        .expect("String.prototype has a fuse")
}

/// The slot of String.prototype's own `name`.
fn slot(engine: &Engine, name: &str) -> usize {
    engine
        .interp
        .string_proto
        .borrow()
        .props
        .slot_of(name)
        .unwrap_or_else(|| panic!("String.prototype.{name}"))
}

fn state(engine: &Engine, name: &str) -> PropertyState {
    fuse(engine).state(slot(engine, name))
}

const TIERS: [Tier; 3] = [Tier::Interp, Tier::Bytecode, Tier::Jit];

#[test]
fn string_prototype_has_a_fuse_whose_methods_start_untracked() {
    let engine = Engine::new();
    let fuse = fuse(&engine);
    assert!(fuse.intact());
    assert_eq!(fuse.generation(), 0);
    assert_eq!(state(&engine, "codePointAt"), PropertyState::Untracked);
    let data = fuse
        .try_optimize_constant(slot(&engine, "codePointAt"))
        .expect("an Untracked method can be optimized");
    assert_eq!(state(&engine, "codePointAt"), PropertyState::Constant);
    assert!(fuse.guard(data));
    // Tracking it again changes nothing.
    assert_eq!(
        fuse.try_optimize_constant(slot(&engine, "codePointAt")),
        Some(data)
    );
    // The active Realm's fuse is the one generated code reads.
    assert_eq!(
        engine.interp.string_proto_fuse_ptr,
        Arc::as_ptr(&fuse) as usize
    );
}

/// Each kind of value change makes a Constant method NotConstant and sets the fast word; the
/// site's guard then fails on its own bit. The same value counts, and so does a define that
/// changes attributes only (SpiderMonkey's define calls watchPropertyValueChange after changing
/// the flags).
#[test]
fn every_kind_of_write_makes_a_constant_method_not_constant() {
    let writes = [
        "String.prototype.codePointAt = function () { return 0; };",
        "String.prototype.codePointAt = String.prototype.codePointAt;",
        "var f = String.prototype.codePointAt; for (var k = 0; k < 40; k++) String.prototype.codePointAt = f;",
        "Reflect.set(String.prototype, 'codePointAt', function () {});",
        "Object.assign(String.prototype, { codePointAt: String.prototype.codePointAt });",
        "Object.defineProperty(String.prototype, 'codePointAt', { value: function () {} });",
        "Object.defineProperty(String.prototype, 'codePointAt', { get() { return 1; } });",
        "Object.defineProperty(String.prototype, 'codePointAt', { writable: false });",
        "Object.defineProperty(String.prototype, 'codePointAt', { enumerable: true });",
        "Object.defineProperties(String.prototype, { codePointAt: { value: String.prototype.codePointAt, writable: true, enumerable: false, configurable: false } });",
    ];
    for tier in TIERS {
        for write in writes {
            let mut engine = engine(tier);
            let fuse = fuse(&engine);
            let data = fuse
                .try_optimize_constant(slot(&engine, "codePointAt"))
                .unwrap();
            eval(&mut engine, write);
            assert_eq!(
                state(&engine, "codePointAt"),
                PropertyState::NotConstant,
                "{tier:?}: {write}"
            );
            assert!(!fuse.intact(), "{tier:?}: {write}");
            assert_eq!(fuse.generation(), 0, "{tier:?}: {write}");
            assert!(!fuse.guard(data), "{tier:?}: {write}");
            // A NotConstant property is not optimized again.
            assert_eq!(
                fuse.try_optimize_constant(slot(&engine, "codePointAt")),
                None,
                "{tier:?}: {write}"
            );
        }
    }
}

/// A redundant define changes nothing (SpiderMonkey's DefinePropertyIsRedundant), and freezing or
/// sealing changes flags only (its freezeOrSealProperties): the method stays Constant and the fast
/// word clear. String.prototype's [[SetPrototypeOf]] cannot change its own property.
#[test]
fn redundant_defines_freezing_sealing_and_a_new_prototype_leave_a_constant_method() {
    let writes = [
        "Object.defineProperty(String.prototype, 'codePointAt', {});",
        "Object.defineProperty(String.prototype, 'codePointAt', Object.getOwnPropertyDescriptor(String.prototype, 'codePointAt'));",
        "Object.defineProperty(String.prototype, 'codePointAt', { writable: true, configurable: true });",
        "Object.freeze(String.prototype);",
        "Object.seal(String.prototype);",
        "Object.preventExtensions(String.prototype);",
        "Object.setPrototypeOf(String.prototype, null);",
        "Object.setPrototypeOf(String.prototype, Array.prototype);",
        "String.prototype.addedLater = 1;",
        "Object.defineProperty(String.prototype, 'addedByDefine', { value: 2 });",
    ];
    for tier in TIERS {
        for write in writes {
            let mut engine = engine(tier);
            let fuse = fuse(&engine);
            let data = fuse
                .try_optimize_constant(slot(&engine, "codePointAt"))
                .unwrap();
            eval(&mut engine, write);
            assert_eq!(
                state(&engine, "codePointAt"),
                PropertyState::Constant,
                "{tier:?}: {write}"
            );
            assert!(fuse.intact(), "{tier:?}: {write}");
            assert!(fuse.guard(data), "{tier:?}: {write}");
            // Frozen, the method stays its own: a primitive's read still finds it.
            assert_eq!(
                eval(&mut engine, "'a\\u{1F600}'.codePointAt(1)"),
                "128512",
                "{tier:?}: {write}"
            );
        }
    }
}

/// Removing a tracked method bumps the generation and sets the word; the states after its slot
/// move down with their properties.
#[test]
fn removing_a_tracked_method_bumps_the_generation_and_moves_the_later_states() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let fuse = fuse(&engine);
        let removed = slot(&engine, "codePointAt");
        let later = (removed + 1..)
            .find(|&at| {
                engine
                    .interp
                    .string_proto
                    .borrow()
                    .props
                    .entry_at(at)
                    .is_some_and(|(key, _)| key.as_index().is_none())
            })
            .expect("a later named property");
        let later_name = engine
            .interp
            .string_proto
            .borrow()
            .props
            .entry_at(later)
            .unwrap()
            .0
            .text()
            .to_string();
        let data = fuse.try_optimize_constant(removed).unwrap();
        let later_data = fuse.try_optimize_constant(later).unwrap();
        eval(&mut engine, "delete String.prototype.codePointAt;");
        assert_eq!(fuse.generation(), 1, "{tier:?}");
        assert!(!fuse.intact(), "{tier:?}");
        assert!(!fuse.guard(data), "{tier:?}");
        assert!(!fuse.guard(later_data), "{tier:?}: the generation moved");
        // The later property moved down one slot, and its state with it.
        assert_eq!(slot(&engine, &later_name), later - 1, "{tier:?}");
        assert_eq!(
            state(&engine, &later_name),
            PropertyState::Constant,
            "{tier:?}"
        );
        assert_eq!(fuse.state(later), PropertyState::Untracked, "{tier:?}");
        // Removing an Untracked property after every tracked one moves no tracked slot: the
        // generation stays.
        let names = named_keys(&engine);
        let last = names
            .last()
            .expect("String.prototype's last property")
            .clone();
        assert!(slot(&engine, &last) > slot(&engine, &later_name));
        eval(&mut engine, &format!("delete String.prototype[{last:?}];"));
        assert_eq!(fuse.generation(), 1, "{tier:?}");
        // Removing one before a tracked property moves that property down a slot: the generation
        // moves, and the state moves with the property.
        let later_data = fuse
            .try_optimize_constant(slot(&engine, &later_name))
            .unwrap();
        let earlier = names
            .iter()
            .take(slot(&engine, &later_name))
            .find(|name| {
                let at = slot(&engine, name);
                fuse.state(at) == PropertyState::Untracked
                    && engine
                        .interp
                        .string_proto
                        .borrow()
                        .props
                        .property_at(at)
                        .is_some_and(|property| property.configurable())
            })
            .expect("an earlier configurable Untracked property")
            .clone();
        let moved = slot(&engine, &later_name);
        eval(
            &mut engine,
            &format!("delete String.prototype[{earlier:?}];"),
        );
        assert_eq!(fuse.generation(), 2, "{tier:?}");
        assert!(!fuse.guard(later_data), "{tier:?}");
        assert_eq!(slot(&engine, &later_name), moved - 1, "{tier:?}");
        assert_eq!(
            state(&engine, &later_name),
            PropertyState::Constant,
            "{tier:?}"
        );
    }
}

/// String.prototype's named keys in slot order.
fn named_keys(engine: &Engine) -> Vec<String> {
    let proto = engine.interp.string_proto.borrow();
    (0..)
        .map_while(|at| proto.props.entry_at(at).map(|(key, _)| key))
        .filter(|key| key.as_index().is_none() && !key.is_symbol())
        .map(|key| key.text().to_string())
        .collect()
}

/// An index key in String.prototype's named entries is no fused property, but removing it moves
/// a later tracked property down a slot, which bumps the generation all the same.
#[test]
fn removing_an_index_key_before_a_tracked_method_bumps_the_generation() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let fuse = fuse(&engine);
        // A method after an index key: put a non-plain element in (one the named entries keep),
        // then remove and re-add the method.
        eval(
            &mut engine,
            "var original = String.prototype.codePointAt; \
             Object.defineProperty(String.prototype, 7, {value: 'x', configurable: true}); \
             delete String.prototype.codePointAt; String.prototype.codePointAt = original;",
        );
        let index_slot = engine
            .interp
            .string_proto
            .borrow()
            .props
            .slot_of("7")
            .expect("the index key is a named entry");
        let method = slot(&engine, "codePointAt");
        assert!(index_slot < method, "{tier:?}");
        let generation = fuse.generation();
        let data = fuse.try_optimize_constant(method).unwrap();
        eval(&mut engine, "delete String.prototype[7];");
        assert_eq!(fuse.generation(), generation + 1, "{tier:?}");
        assert!(!fuse.guard(data), "{tier:?}");
        assert_eq!(slot(&engine, "codePointAt"), method - 1, "{tier:?}");
        assert_eq!(
            state(&engine, "codePointAt"),
            PropertyState::Constant,
            "{tier:?}"
        );
    }
}

/// A polyfill replacing one method leaves the others fused: the fast word is set, but a site on
/// another method passes on its own NotConstant bit, clear (SpiderMonkey's two levels).
#[test]
fn a_polyfill_replacing_one_method_keeps_the_others_fused() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let fuse = fuse(&engine);
        let code_point = fuse
            .try_optimize_constant(slot(&engine, "codePointAt"))
            .unwrap();
        let pad = fuse
            .try_optimize_constant(slot(&engine, "padStart"))
            .unwrap();
        eval(
            &mut engine,
            "String.prototype.padStart = function (n, s) { return String(s || ' ').repeat(n) + this; };",
        );
        assert!(!fuse.intact(), "{tier:?}");
        assert!(!fuse.guard(pad), "{tier:?}");
        assert!(fuse.guard(code_point), "{tier:?}");
        assert_eq!(
            state(&engine, "codePointAt"),
            PropertyState::Constant,
            "{tier:?}"
        );
        assert_eq!(
            state(&engine, "padStart"),
            PropertyState::NotConstant,
            "{tier:?}"
        );
        // A method never tracked can still be.
        assert!(fuse
            .try_optimize_constant(slot(&engine, "charCodeAt"))
            .is_some_and(|data| fuse.guard(data)));
    }
}

/// SpiderMonkey's handlePropertyValueChange marks an Untracked property Constant at its first
/// write (for a global's `var x = y`), and NotConstant at the next.
#[test]
fn an_untracked_property_becomes_constant_at_a_write_and_not_constant_at_the_next() {
    let mut engine = Engine::new();
    // Index keys are not tracked: writing one twice leaves the fuse as it was.
    eval(
        &mut engine,
        "String.prototype[7] = 1; String.prototype[7] = 2; String.prototype[7] = 3;",
    );
    assert!(fuse(&engine).intact());
    eval(&mut engine, "String.prototype.extra = 1;");
    assert_eq!(state(&engine, "extra"), PropertyState::Untracked);
    eval(&mut engine, "String.prototype.extra = 2;");
    assert_eq!(state(&engine, "extra"), PropertyState::Constant);
    assert!(fuse(&engine).intact());
    eval(&mut engine, "String.prototype.extra = 3;");
    assert_eq!(state(&engine, "extra"), PropertyState::NotConstant);
    assert!(!fuse(&engine).intact());
}

/// Each Realm's String.prototype has its own fuse: a write to one Realm's methods, from its own
/// code or another Realm's, leaves the other's as it was.
#[test]
fn another_realms_string_prototype_has_its_own_fuse() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let main = fuse(&engine);
        let other_global = engine.interp.create_realm();
        let other_ptr = other_global
            .as_obj()
            .map(|g| crate::value::Gc::as_ptr(&g) as usize);
        let other = engine
            .interp
            .realms
            .get(&other_ptr.unwrap())
            .and_then(|realm| realm.string_proto_fuse.clone())
            .expect("the new Realm's fuse");
        assert!(!Arc::ptr_eq(&main, &other));
        let main_data = main
            .try_optimize_constant(slot(&engine, "codePointAt"))
            .unwrap();
        let other_proto = engine.interp.realms[&other_ptr.unwrap()].string_proto;
        let other_slot = other_proto.borrow().props.slot_of("codePointAt").unwrap();
        let other_data = other.try_optimize_constant(other_slot).unwrap();
        let global = Value::Obj(engine.interp.global);
        engine
            .interp
            .member_set(&global, "otherGlobal", other_global)
            .unwrap_or_else(|_| panic!("otherGlobal"));
        // The main Realm writes the other's method; the other Realm's own code writes it too.
        eval(
            &mut engine,
            "otherGlobal.String.prototype.codePointAt = function () {}; otherGlobal.eval('String.prototype.codePointAt = 1');",
        );
        assert!(!other.guard(other_data), "{tier:?}");
        assert!(main.guard(main_data), "{tier:?}");
        assert!(main.intact(), "{tier:?}");
        assert_eq!(
            state(&engine, "codePointAt"),
            PropertyState::Constant,
            "{tier:?}"
        );
        assert!(engine
            .interp
            .string_proto_fuse
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &main)));
    }
}

/// Every switch of the active Realm carries its String.prototype's fuse along, the address
/// generated code reads (`Interp::string_proto_fuse_ptr`) included: a swap with a Realm's snapshot
/// (`Interp::swap_realm`, an embedder's entry and exit and `Interp::with_realm_snapshot`) and a
/// snapshot's restore (`Interp::restore_realm`).
#[test]
fn every_realm_switch_carries_the_string_prototypes_fuse() {
    let mut engine = Engine::new();
    let main = fuse(&engine);
    let other_global = engine.interp.create_realm();
    let key = other_global
        .as_obj()
        .map(|g| crate::value::Gc::as_ptr(&g) as usize)
        .unwrap();
    let other = engine.interp.realms[&key]
        .string_proto_fuse
        .clone()
        .expect("the new Realm's fuse");
    let other_proto = crate::value::Gc::as_ptr(&engine.interp.realms[&key].string_proto) as usize;
    let main_proto = crate::value::Gc::as_ptr(&engine.interp.string_proto) as usize;
    let active_is = |engine: &Engine, fuse: &Arc<ObjectFuse>, proto: usize| {
        engine
            .interp
            .string_proto_fuse
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, fuse))
            && engine.interp.string_proto_fuse_ptr == Arc::as_ptr(fuse) as usize
            && crate::value::Gc::as_ptr(&engine.interp.string_proto) as usize == proto
    };
    assert!(active_is(&engine, &main, main_proto));
    let mut saved = engine.interp.realms[&key].snapshot_clone();
    engine.interp.swap_realm(&mut saved);
    assert!(active_is(&engine, &other, other_proto), "entered");
    engine.interp.swap_realm(&mut saved);
    assert!(active_is(&engine, &main, main_proto), "left");
    assert!(saved
        .string_proto_fuse
        .as_ref()
        .is_some_and(|fuse| Arc::ptr_eq(fuse, &other)));
    let mine = engine.interp.snapshot_realm();
    let theirs = engine.interp.realms[&key].snapshot_clone();
    engine.interp.restore_realm(&theirs);
    assert!(
        active_is(&engine, &other, other_proto),
        "restored the other"
    );
    engine.interp.restore_realm(&mine);
    assert!(active_is(&engine, &main, main_proto), "restored mine");
    assert!(engine.interp.realms[&key]
        .string_proto_fuse
        .as_ref()
        .is_some_and(|fuse| Arc::ptr_eq(fuse, &other)));
}

/// The guard as generated code will read it (step 2's contract): the fast word at
/// `INVALIDATED_OFF`, then the generation at `GENERATION_OFF` and the state word through the
/// address published at `WORDS_OFF`, from the fuse's address in `Interp::string_proto_fuse_ptr`.
#[test]
fn the_fuses_words_read_through_its_published_offsets_agree_with_its_guard() {
    let mut engine = Engine::new();
    let fuse = fuse(&engine);
    let code_point = fuse
        .try_optimize_constant(slot(&engine, "codePointAt"))
        .unwrap();
    let pad = fuse
        .try_optimize_constant(slot(&engine, "padStart"))
        .unwrap();
    let raw_guard = |engine: &Engine, data: crate::object_fuse::GuardData| {
        let base = engine.interp.string_proto_fuse_ptr;
        // SAFETY: the active Realm's fuse lives while its Arc does (held by `fuse` above too);
        // the words' address is published by the fuse as its buffer grows.
        unsafe {
            let invalidated = *((base + ObjectFuse::INVALIDATED_OFF) as *const u32);
            if invalidated == 0 {
                return true;
            }
            let generation = *((base + ObjectFuse::GENERATION_OFF) as *const u32);
            if generation != data.generation {
                return false;
            }
            let words = *((base + ObjectFuse::WORDS_OFF) as *const *const u32);
            *words.add(data.index as usize) & data.mask == 0
        }
    };
    assert!(raw_guard(&engine, code_point) && raw_guard(&engine, pad));
    eval(&mut engine, "String.prototype.padStart = function () {};");
    assert!(raw_guard(&engine, code_point));
    assert!(!raw_guard(&engine, pad));
    assert_eq!(raw_guard(&engine, code_point), fuse.guard(code_point));
    eval(&mut engine, "delete String.prototype.padStart;");
    assert!(!raw_guard(&engine, code_point), "the generation moved");
    assert_eq!(raw_guard(&engine, code_point), fuse.guard(code_point));
}
