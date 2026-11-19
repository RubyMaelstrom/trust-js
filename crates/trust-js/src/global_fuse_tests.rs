//! The global object's fuse (F4 part 2, layer 1): each realm's global object keeps its
//! properties' states (`crate::object_fuse`, SpiderMonkey's ObjectFuse on its global, as
//! vm/ObjectFuse.h's SMDOC), every writer moves them, and a cache that writes a global property
//! with no report is filled only for a NotConstant property (R2, SpiderMonkey's
//! canOptimizeSetSlot). A watched global's shape is its own (R-1). No fold reads the states yet:
//! these tests read the state after each write.
use crate::bytecode::Tier;
use crate::object_fuse::PropertyState;
use crate::{Completion, Engine};

const TIERS: [Tier; 3] = [Tier::Interp, Tier::Bytecode, Tier::Jit];

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
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

fn state(engine: &Engine, name: &str) -> PropertyState {
    engine
        .interp
        .global_property_state(name)
        .unwrap_or_else(|| panic!("the global has {name}"))
}

/// Make the global property `name` Constant, as a compile that relies on it does.
fn make_constant(engine: &Engine, name: &str) {
    let fuse = engine
        .interp
        .global_fuse()
        .expect("the realm's global fuse");
    let slot = engine
        .interp
        .global
        .borrow()
        .props
        .slot_of(name)
        .expect("the property");
    assert!(fuse.try_optimize_constant(slot).is_some(), "{name}");
}

fn checked() -> u64 {
    crate::object_fuse::TEST_GLOBAL_CHECKED.with(std::cell::Cell::get)
}

fn native<'s>(
    _: &mut crate::embed::Ctx<'s>,
    this: crate::embed::Value<'s>,
    _: &[crate::embed::Value<'s>],
) -> Result<crate::embed::Value<'s>, crate::embed::Value<'s>> {
    Ok(this)
}

/// Every writer of a global property, each after the property was made Constant: the write
/// makes it NotConstant (SpiderMonkey's handlePropertyValueChange; the same value stored again
/// included, its Watchtower.cpp:754-757). The loops warm each kind of cache that writes first,
/// so that a cache filled while the property was Constant would store past the state.
#[test]
fn every_writer_makes_a_constant_global_not_constant() {
    let writes = [
        // W1/W2 by name: a script body's Reference, a function's name store and update, eval's.
        "G = 2;",
        "G = G;",
        "function f() { G = 2; } f();",
        "function f() { for (var k = 0; k < 40; k++) G = k; } f();",
        "function f() { for (var k = 0; k < 40; k++) G++; } f();",
        "function f() { for (var k = 0; k < 40; k++) G += 1; } f();",
        "for (var k = 0; k < 40; k++) G = k;",
        "for (var k = 0; k < 40; k++) G += 1;",
        "(function () { var a = 1; (function () { for (var k = 0; k < 40; k++) G = k + a; })(); })();",
        "eval('G = 2');",
        "(function () { eval('for (var k = 0; k < 40; k++) G = k;'); })();",
        "with (globalThis) G = 2;",
        "with ({}) { for (var k = 0; k < 40; k++) G = k; }",
        // W1/W2 through any alias: property stores, updates, appends.
        "globalThis.G = 2;",
        "this.G = 2;",
        "var o = globalThis; for (var k = 0; k < 40; k++) o.G = k;",
        "var o = globalThis; for (var k = 0; k < 40; k++) o.G++;",
        "var o = globalThis; for (var k = 0; k < 40; k++) o['G'] = k;",
        "function f(g) { for (var k = 0; k < 40; k++) g.G = k; } f(globalThis);",
        "Reflect.set(globalThis, 'G', 2);",
        "Object.assign(globalThis, { G: 2 });",
        // W4: defines.
        "Object.defineProperty(globalThis, 'G', { value: 2 });",
        "Object.defineProperty(globalThis, 'G', { get() { return 1; } });",
        "Object.defineProperty(globalThis, 'G', { writable: false });",
        "Object.defineProperty(globalThis, 'G', { enumerable: false });",
        "Object.defineProperties(globalThis, { G: { value: 3 } });",
        "eval('function G() {}');",
        "Reflect.defineProperty(globalThis, 'G', { value: 4 });",
        // Writers nobody listed: author code the write runs in, and other syntax.
        "var o = { get g() { G = 2; return 1; } }; for (var k = 0; k < 40; k++) o.g;",
        "var o = { set s(v) { G = v; } }; for (var k = 0; k < 40; k++) o.s = k;",
        "var o = { valueOf() { G = 2; return 1; } }; for (var k = 0; k < 40; k++) o + 1;",
        "var p = new Proxy({}, { get() { G = 2; return 1; } }); for (var k = 0; k < 40; k++) p.x;",
        "var p = new Proxy(globalThis, {}); for (var k = 0; k < 40; k++) p.G = k;",
        "[1, 2, 3].forEach(function (v) { G = v; });",
        "JSON.parse('{\"a\":1}', function (k, v) { G = 2; return v; });",
        "function w(v) { G = v; } function f() { for (var k = 0; k < 3000; k++) w(k); } f();",
        "function w(o, v) { o.G = v; } function f() { for (var k = 0; k < 3000; k++) w(globalThis, k); } f();",
        "Function('G = 2')();",
        "with (globalThis) G++;",
        "[G] = [2];",
        "({ G } = { G: 2 });",
        "for (G of [2, 3]);",
        "for (G in { a: 1 });",
        "G &&= 2;",
        "class C { static { G = 2; } }",
        "function* g() { G = 2; yield; } g().next();",
        "(async function () { G = 2; })();",
        "try { throw 1; } catch (e) { G = 2; }",
        "Object.setPrototypeOf(globalThis, { set G(v) {} }); G = 2;",
    ];
    for tier in TIERS {
        for write in writes {
            let mut engine = engine(tier);
            eval(&mut engine, "globalThis.G = 1;");
            make_constant(&engine, "G");
            eval(&mut engine, write);
            assert_eq!(
                state(&engine, "G"),
                PropertyState::NotConstant,
                "{tier:?}: {write}"
            );
        }
    }
}

/// Strings append in place (`o.k += s`: AppendProp, `Interp::append_prop_fast`), and an
/// accessor pair replaced in place is a value change (SpiderMonkey's Watchtower.cpp:745-747).
#[test]
fn appends_and_accessor_replacements_make_a_constant_global_not_constant() {
    for tier in TIERS {
        let mut engine = engine(tier);
        eval(&mut engine, "globalThis.S = 'a';");
        make_constant(&engine, "S");
        eval(
            &mut engine,
            "var o = globalThis; for (var k = 0; k < 40; k++) o.S += 'b';",
        );
        assert_eq!(state(&engine, "S"), PropertyState::NotConstant, "{tier:?}");
        eval(
            &mut engine,
            "Object.defineProperty(globalThis, 'A', { get() { return 1; }, configurable: true });",
        );
        make_constant(&engine, "A");
        eval(
            &mut engine,
            "globalThis.__defineGetter__('A', function () { return 2; });",
        );
        assert_eq!(state(&engine, "A"), PropertyState::NotConstant, "{tier:?}");
    }
}

/// A redundant define and freezing leave a Constant global Constant (SpiderMonkey's
/// DefinePropertyIsRedundant and freezeOrSealProperties); an addition is Untracked.
#[test]
fn redundant_defines_freezing_and_additions_leave_the_state() {
    for tier in TIERS {
        let mut engine = engine(tier);
        eval(&mut engine, "globalThis.G = 1;");
        assert_eq!(state(&engine, "G"), PropertyState::Untracked, "{tier:?}");
        make_constant(&engine, "G");
        eval(
            &mut engine,
            "Object.defineProperty(globalThis, 'G', { value: 1 });",
        );
        eval(&mut engine, "globalThis.H = 2; var I = 3;");
        assert_eq!(state(&engine, "G"), PropertyState::Constant, "{tier:?}");
        assert_eq!(state(&engine, "H"), PropertyState::Untracked, "{tier:?}");
        eval(&mut engine, "Object.freeze(globalThis);");
        assert_eq!(state(&engine, "G"), PropertyState::Constant, "{tier:?}");
    }
}

/// A removal bumps the generation, and the property added again is Untracked (SpiderMonkey's
/// handlePropertyRemove); a script's lexical declaration shadowing a tracked property bumps it
/// and keeps the state (its handleShadowedGlobalProperty).
#[test]
fn removals_and_shadowing_lexicals_move_the_generation() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let fuse = engine.interp.global_fuse().unwrap();
        eval(&mut engine, "globalThis.G = 1; globalThis.L = 1;");
        make_constant(&engine, "G");
        let generation = fuse.generation();
        eval(&mut engine, "delete globalThis.G; globalThis.G = 5;");
        assert_eq!(fuse.generation(), generation + 1, "{tier:?}");
        assert_eq!(state(&engine, "G"), PropertyState::Untracked, "{tier:?}");
        make_constant(&engine, "L");
        eval(&mut engine, "let L = 2;");
        assert_eq!(fuse.generation(), generation + 2, "{tier:?}");
        assert_eq!(state(&engine, "L"), PropertyState::Constant, "{tier:?}");
    }
}

/// A global stored in a loop reaches the in-line store at its third store at the latest: the
/// first makes it Constant, the second NotConstant, and only then does a writing cache fill
/// (R2). Every kind of cache that writes: a name store (and the VM's hit, which reports
/// nothing), an update, a deep chain, a Reference (a script body's), a property store, an
/// update and an append through an alias. A fill that skipped its check would leave the
/// property Constant under in-line stores.
#[test]
fn a_writing_cache_fills_only_for_a_not_constant_global() {
    let loops = [
        ("G", "function f() { for (var k = 0; k < 100; k++) G = k; } f();"),
        ("G", "function f() { for (var k = 0; k < 100; k++) G++; } f();"),
        (
            "G",
            "(function () { var a = 0; (function () { for (var k = 0; k < 100; k++) G = k + a; })(); })();",
        ),
        ("G", "for (var k = 0; k < 100; k++) G = k;"),
        ("G", "var o = globalThis; for (var k = 0; k < 100; k++) o.G = k;"),
        ("G", "var o = globalThis; for (var k = 0; k < 100; k++) o.G++;"),
        ("S", "var o = globalThis; for (var k = 0; k < 100; k++) o.S += 1;"),
    ];
    for tier in TIERS {
        for (name, program) in loops {
            let mut engine = engine(tier);
            eval(&mut engine, "globalThis.G = 1; globalThis.S = 1;");
            let before = checked();
            eval(&mut engine, program);
            assert_eq!(
                state(&engine, name),
                PropertyState::NotConstant,
                "{tier:?}: {program}"
            );
            // ARM64's template stores in line once its cache fills (a script body runs its first
            // iterations in the VM before it enters native code); the VM's property hits report,
            // and so do x86-64's script-body property stores, which take the helper.
            let reported = checked() - before;
            if tier == Tier::Jit && cfg!(target_arch = "aarch64") {
                assert!(
                    reported < 20,
                    "{tier:?}: {program}: {reported} checked stores"
                );
            }
        }
    }
}

/// The embedder's API defines over an existing global (W4) and the object losing its ordinary
/// state or its map (W9, W8) change every property.
#[test]
fn the_embedders_defines_and_whole_map_changes_move_the_state() {
    let mut engine = engine(Tier::Jit);
    eval(&mut engine, "globalThis.N = 1; globalThis.M = 1;");
    make_constant(&engine, "N");
    engine.define_global("N", 0, native);
    assert_eq!(state(&engine, "N"), PropertyState::NotConstant);
    // An accessor pair replaced in place (W7, Props::merge_accessor): the engine merges a
    // literal's or a class's getter and setter so, never the global's from author code, so the
    // map is driven directly.
    eval(
        &mut engine,
        "Object.defineProperty(globalThis, 'P', { get() { return 1; }, configurable: true }); var getter = function () { return 2; };",
    );
    make_constant(&engine, "P");
    let env = engine.interp.global_env;
    let getter = engine.interp.get_var("getter", &env).ok().unwrap();
    assert!(engine
        .interp
        .global
        .borrow_mut()
        .props
        .merge_accessor("P", Some(getter), None));
    assert_eq!(state(&engine, "P"), PropertyState::NotConstant);
    make_constant(&engine, "M");
    engine.interp.global.borrow().set_ic_plain(false);
    assert_eq!(state(&engine, "M"), PropertyState::NotConstant);
}

/// The state of `name` on the global object `global` (another realm's).
fn state_of(global: &crate::value::Gc, name: &str) -> PropertyState {
    let body = global.borrow();
    let slot = body.props.slot_of(name).expect("the property");
    crate::object_fuse::global_property_state(&body.props as *const _ as usize, slot)
}

/// R-1: a watched global's shape is its own. Two realms' globals built alike (each from
/// `create_realm`) take fresh shapes when marked (U1) and at every change (U2), so a store site
/// warmed through one realm's NotConstant property does not store in line to the other's
/// Constant one, from one chunk, and each realm's own name stores move its own state; a copy of
/// a global's map is not watched and takes a shape of its own keys (U3).
#[test]
fn two_realms_globals_never_share_a_shape() {
    use crate::value::{Property, Value, ValueView};
    for tier in TIERS {
        let mut engine = engine(tier);
        let realms: Vec<_> = (0..2)
            .map(|_| match engine.interp.create_realm().view() {
                ValueView::Obj(global) => global,
                _ => panic!("a realm's global"),
            })
            .collect();
        let (b, c) = (realms[0], realms[1]);
        // Built alike, marked apart (U1).
        assert_ne!(
            b.borrow().props.shape(),
            c.borrow().props.shape(),
            "{tier:?}"
        );
        for (global, name) in [(b, "realmB"), (c, "realmC")] {
            engine
                .interp
                .global
                .borrow_mut()
                .props
                .insert(name, Property::data(Value::Obj(global), true, true, true));
        }
        // A built-in's property, before any addition: B's JSON NotConstant through a site, then
        // C's Constant JSON through the same site.
        eval(
            &mut engine,
            "function stj(g) { g.JSON = g.JSON; } for (var k = 0; k < 40; k++) stj(realmB);",
        );
        assert_eq!(state_of(&b, "JSON"), PropertyState::NotConstant, "{tier:?}");
        let fuse = crate::object_fuse::fuse_for(&c.borrow().props as *const _ as usize).unwrap();
        assert!(fuse
            .try_optimize_constant(c.borrow().props.slot_of("JSON").unwrap())
            .is_some());
        eval(&mut engine, "for (var k = 0; k < 40; k++) stj(realmC);");
        assert_eq!(state_of(&c, "JSON"), PropertyState::NotConstant, "{tier:?}");
        for global in [b, c] {
            global
                .borrow_mut()
                .props
                .insert("x", Property::data(Value::Num(0.0), true, true, true));
        }
        assert_ne!(
            b.borrow().props.shape(),
            c.borrow().props.shape(),
            "{tier:?}"
        );
        // B's x NotConstant through the site, then C's Constant x through the same site.
        eval(
            &mut engine,
            "function st(g, v) { g.x = v; } for (var k = 0; k < 40; k++) st(realmB, k);",
        );
        assert_eq!(state_of(&b, "x"), PropertyState::NotConstant, "{tier:?}");
        let fuse = crate::object_fuse::fuse_for(&c.borrow().props as *const _ as usize).unwrap();
        let slot = c.borrow().props.slot_of("x").unwrap();
        assert!(fuse.try_optimize_constant(slot).is_some());
        eval(&mut engine, "for (var k = 0; k < 40; k++) st(realmC, k);");
        assert_eq!(state_of(&c, "x"), PropertyState::NotConstant, "{tier:?}");
        // Each realm's own chunk: its name stores move its own global's state alone.
        let run = |engine: &mut Engine, global: crate::value::Gc, source: &str| {
            assert!(engine
                .interp
                .eval_in_realm(&Value::Obj(global), source)
                .is_ok());
        };
        run(&mut engine, b, "globalThis.z = 1;");
        run(&mut engine, c, "globalThis.z = 1;");
        let fuse_b = crate::object_fuse::fuse_for(&b.borrow().props as *const _ as usize).unwrap();
        assert!(fuse_b
            .try_optimize_constant(b.borrow().props.slot_of("z").unwrap())
            .is_some());
        run(&mut engine, c, "for (var k = 0; k < 40; k++) z = k;");
        assert_eq!(state_of(&c, "z"), PropertyState::NotConstant, "{tier:?}");
        assert_eq!(state_of(&b, "z"), PropertyState::Constant, "{tier:?}");
        run(&mut engine, b, "for (var k = 0; k < 40; k++) z = k;");
        assert_eq!(state_of(&b, "z"), PropertyState::NotConstant, "{tier:?}");
        // A copy of a global's map is no watched map and holds none of its shapes.
        let copy = c.borrow().props.clone();
        assert_ne!(copy.shape(), c.borrow().props.shape(), "{tier:?}");
    }
}

/// The class field's fourth value marks the watched global and reads as no class: the global
/// is no class constructor to any reader of the class bits.
#[test]
fn the_watched_marker_reads_as_no_class() {
    let mut engine = engine(Tier::Jit);
    {
        let global = engine.interp.global.borrow();
        assert!(global.watched_global());
        assert_eq!(global.class_kind(), 0);
    }
    assert_eq!(
        eval(
            &mut engine,
            "class C {} var r = []; try { globalThis(); } catch (e) { r.push(e instanceof TypeError); } try { new globalThis(); } catch (e) { r.push(e instanceof TypeError); } r.push(typeof globalThis, Object.getPrototypeOf(globalThis) === Object.prototype); r.join()"
        ),
        "true,true,object,true"
    );
}

/// Every route by which a way reaches a store or update site (F4 part 2, C: the VM's way hit
/// writes with no report, as generated code's in-line store of the way does): its own fill, the
/// stub cache's promotion of a read's entry, the update's read, a writing site's seed and a read
/// site's seed (which seeding empties). Each brings a Constant global property to the site, whose
/// store must still report: the property is NotConstant after it.
#[test]
fn every_route_into_a_store_site_leaves_a_constant_global_reporting() {
    for tier in TIERS {
        // Its own fill: the first store makes x Constant and fills no way; the second reports.
        let mut engine = self::engine(tier);
        eval(
            &mut engine,
            "var x; function s(o, v) { o.x = v; } s(globalThis, 1);",
        );
        assert_eq!(state(&engine, "x"), PropertyState::Constant, "{tier:?}");
        eval(&mut engine, "s(globalThis, 2);");
        assert_eq!(
            state(&engine, "x"),
            PropertyState::NotConstant,
            "{tier:?}: own fill"
        );
        // The stub cache's entry a read put there: the store site's first store finds it and
        // promotes nothing while y is Constant.
        let mut engine = self::engine(tier);
        eval(
            &mut engine,
            "var y; function r(o) { return o.y; } function s(o, v) { o.y = v; }
             for (var k = 0; k < 3; k++) r(globalThis);
             s(globalThis, 1);",
        );
        assert_eq!(state(&engine, "y"), PropertyState::Constant, "{tier:?}");
        eval(&mut engine, "s(globalThis, 2);");
        assert_eq!(
            state(&engine, "y"),
            PropertyState::NotConstant,
            "{tier:?}: stub cache"
        );
        // The update's read, which fills the site its write takes.
        let mut engine = self::engine(tier);
        eval(&mut engine, "var u = 1; function up(o) { o.u++; }");
        assert_eq!(state(&engine, "u"), PropertyState::Constant, "{tier:?}");
        eval(&mut engine, "up(globalThis);");
        assert_eq!(
            state(&engine, "u"),
            PropertyState::NotConstant,
            "{tier:?}: update read"
        );
        // A read site's ways seeding a store site of a second stage (`rw`'s splices `id`), which
        // seeding empties: the store then reaches the Constant x of the global.
        let mut engine = self::engine(tier);
        eval(
            &mut engine,
            "var x = 7; function id(v) { return v; }
             function rw(p, q) { var t = id(p.x); q.x = t + 1; return t; }
             var w = { x: 0 };
             function drive(n) { var s = 0; for (var i = 0; i < n; i++) s += rw(globalThis, w); return s; }
             drive(150);",
        );
        for _ in 0..20 {
            assert_eq!(eval(&mut engine, "drive(150)"), "1050");
        }
        assert_eq!(state(&engine, "x"), PropertyState::Constant, "{tier:?}");
        assert_eq!(eval(&mut engine, "rw(globalThis, globalThis); x"), "8");
        assert_eq!(
            state(&engine, "x"),
            PropertyState::NotConstant,
            "{tier:?}: read seed"
        );
    }
}

/// A writing site's seed (a second stage of `st`, which splices `id`) holds the ways its first
/// stage filled under the rule, through realm B's NotConstant x; on realm C's global, whose x is
/// Constant, the seeded way does not hit (R-1: the globals' shapes are their own), and the store
/// reports.
#[test]
fn a_writing_seed_reaches_no_other_realms_constant_global() {
    use crate::value::{Property, Value, ValueView};
    for tier in TIERS {
        let mut engine = engine(tier);
        let realms: Vec<_> = (0..2)
            .map(|_| match engine.interp.create_realm().view() {
                ValueView::Obj(global) => global,
                _ => panic!("a realm's global"),
            })
            .collect();
        let (b, c) = (realms[0], realms[1]);
        for (global, name) in [(b, "realmB"), (c, "realmC")] {
            engine
                .interp
                .global
                .borrow_mut()
                .props
                .insert(name, Property::data(Value::Obj(global), true, true, true));
        }
        for global in [b, c] {
            global
                .borrow_mut()
                .props
                .insert("x", Property::data(Value::Num(0.0), true, true, true));
        }
        eval(
            &mut engine,
            "function id(v) { return v; } function st(g, v) { g.x = id(v); }
             function drive(g, n) { for (var i = 0; i < n; i++) st(g, i); }",
        );
        for _ in 0..20 {
            eval(&mut engine, "drive(realmB, 150);");
        }
        assert_eq!(state_of(&b, "x"), PropertyState::NotConstant, "{tier:?}");
        let fuse = crate::object_fuse::fuse_for(&c.borrow().props as *const _ as usize).unwrap();
        assert!(fuse
            .try_optimize_constant(c.borrow().props.slot_of("x").unwrap())
            .is_some());
        eval(&mut engine, "st(realmC, 1);");
        assert_eq!(state_of(&c, "x"), PropertyState::NotConstant, "{tier:?}");
    }
}

/// A name store site of a second stage seeded from a read site of the same name (`rw`'s splices
/// `id`): a writing site takes only a writing site's seed (`Compiler::new_name_cache`), so the
/// store of the Constant x, first run after the seeding, reports (B: a writing site's held
/// entry is a writing fill's, which its fill then trusts without asking again).
#[test]
fn a_name_store_site_takes_no_read_sites_seed() {
    for tier in TIERS {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            "var x = 7; function id(v) { return v; }
             function rw(f) { var t = id(x); if (f) x = t + 1; return t; }
             function drive(n) { var s = 0; for (var i = 0; i < n; i++) s += rw(false); return s; }
             drive(150);",
        );
        for _ in 0..20 {
            assert_eq!(eval(&mut engine, "drive(150)"), "1050");
        }
        assert_eq!(state(&engine, "x"), PropertyState::Constant, "{tier:?}");
        assert_eq!(eval(&mut engine, "rw(true); x"), "8");
        assert_eq!(state(&engine, "x"), PropertyState::NotConstant, "{tier:?}");
    }
}

/// A: the states read without the lock agree with the transitions made under it as the buffer
/// grows by doubling (its word 0 counting the words after it), and every buffer it published is
/// kept while the fuse lives, so that an address read before a growth stays readable.
#[test]
fn the_lock_free_states_follow_the_transitions_through_growth() {
    let fuse = crate::object_fuse::ObjectFuse::new_registered(0x10);
    let mut model = vec![PropertyState::Untracked; 400];
    for slot in (0..400).step_by(7) {
        assert!(fuse.try_optimize_constant(slot).is_some());
        model[slot] = PropertyState::Constant;
        if slot % 3 == 0 {
            assert_eq!(fuse.handle_value_change(slot), PropertyState::Constant);
            model[slot] = PropertyState::NotConstant;
            assert_eq!(fuse.handle_value_change(slot), PropertyState::NotConstant);
        }
        for (at, expected) in model.iter().enumerate() {
            assert_eq!(fuse.state(at), *expected, "slot {at} after {slot}");
        }
    }
    // 400 slots in 25 words: buffers of 4, 8, 16 and 32 words, every one kept.
    assert_eq!(fuse.published_buffers(), 4);
}

/// A: the active realm's fuse, reached from the realm with no registry, answers for its own
/// global's map alone; another realm's global is asked of its own fuse.
#[test]
fn the_realm_path_answers_for_its_own_global_alone() {
    use crate::value::{Property, Value, ValueView};
    let mut engine = engine(Tier::Bytecode);
    let ValueView::Obj(c) = engine.interp.create_realm().view() else {
        panic!("a realm's global")
    };
    c.borrow_mut()
        .props
        .insert("x", Property::data(Value::Num(0.0), true, true, true));
    let slot = c.borrow().props.slot_of("x").unwrap();
    let theirs = crate::object_fuse::fuse_for(&c.borrow().props as *const _ as usize).unwrap();
    assert!(theirs.try_optimize_constant(slot).is_some());
    // The active realm's property at the same slot is NotConstant.
    let ours = engine.interp.global_fuse().unwrap();
    let _ = ours.try_optimize_constant(slot);
    ours.handle_value_change(slot);
    assert_eq!(ours.state(slot), PropertyState::NotConstant);
    assert!(!engine.interp.may_store_in_line(&c.borrow().props, slot));
    theirs.handle_value_change(slot);
    assert!(engine.interp.may_store_in_line(&c.borrow().props, slot));
    assert!(engine
        .interp
        .may_store_in_line(&engine.interp.global.borrow().props, slot));
}

/// C's route by a read site's seed, at its rule: a recompilation's store or update site that a
/// read site's ways seeded starts empty (`clear_read_seeded_stores`; a read's way may hold a
/// Constant global property, which the store's hits would then write with no report), and a
/// writing name site takes only a writing site's seed (`name_seed_fits`, which B's held-entry
/// skip in `Chunk::name_ic_fill` relies on).
#[test]
fn a_read_sites_seed_never_stays_in_a_writing_site() {
    use crate::bytecode::{clear_read_seeded_stores, name_seed_fits, IcState, Op, PROP_IC_WAYS};
    let way = IcState {
        depth: 0,
        slot: 3,
        recv_shape: 77,
        holder_shape: 77,
        ..IcState::EMPTY
    };
    let ops = [
        Op::GetProp(0, 0),
        Op::SetProp(0, PROP_IC_WAYS as u32),
        Op::UpdateProp(
            0,
            2 * PROP_IC_WAYS as u32,
            crate::bytecode::UpdKind::IncDiscard,
        ),
        Op::SetProp(0, 3 * PROP_IC_WAYS as u32),
    ];
    let mut caches: Vec<_> = (0..4 * PROP_IC_WAYS)
        .map(|_| std::cell::Cell::new(way))
        .collect();
    // The first three sites took a read site's seed; the last a writing site's.
    let read_seeded = [0, PROP_IC_WAYS as u32, 2 * PROP_IC_WAYS as u32];
    clear_read_seeded_stores(&ops, &read_seeded, &mut caches);
    let held = |site: usize| {
        caches[site * PROP_IC_WAYS..(site + 1) * PROP_IC_WAYS]
            .iter()
            .filter(|cell| cell.get().recv_shape == 77)
            .count()
    };
    assert_eq!(
        (held(0), held(1), held(2), held(3)),
        (PROP_IC_WAYS, 0, 0, PROP_IC_WAYS)
    );
    assert!(name_seed_fits(true, true) && name_seed_fits(false, true));
    assert!(name_seed_fits(false, false) && !name_seed_fits(true, false));
}

/// B's held-entry skip compares the shape and the slot: a writing name site whose entry was
/// filled for a NotConstant x asks again once x was removed and added anew (a fresh shape), so
/// that its first store after (which makes x Constant) fills nothing and its second reports.
#[test]
fn a_writing_name_site_asks_again_for_a_property_added_anew() {
    for tier in TIERS {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            "globalThis.x = 0; function s(v) { x = v; } for (var k = 0; k < 40; k++) s(k);",
        );
        assert_eq!(state(&engine, "x"), PropertyState::NotConstant, "{tier:?}");
        eval(&mut engine, "delete globalThis.x; globalThis.x = 0; s(1);");
        assert_eq!(state(&engine, "x"), PropertyState::Constant, "{tier:?}");
        eval(&mut engine, "s(2);");
        assert_eq!(state(&engine, "x"), PropertyState::NotConstant, "{tier:?}");
    }
}

/// A: a watched global's reported write finds the active realm's fuse through a thread-local
/// with no registry. It names a live fuse only: two engines on one thread, the second made last
/// (its fuse the thread's) and then dropped (the thread's cleared); the first's writes move its
/// own states throughout, through the registry.
#[test]
fn two_engines_on_one_thread_move_their_own_states() {
    // Each engine's map is read under its own Agent (an embedder's entry activates it).
    let state = |engine: &Engine, name: &str| {
        let _agent = engine.interp.enter_agent();
        state(engine, name)
    };
    let make_constant = |engine: &Engine, name: &str| {
        let _agent = engine.interp.enter_agent();
        make_constant(engine, name)
    };
    let mut a = engine(Tier::Bytecode);
    eval(&mut a, "globalThis.G = 1; globalThis.H = 1;");
    make_constant(&a, "G");
    {
        let mut b = engine(Tier::Bytecode);
        eval(&mut b, "globalThis.G = 1;");
        make_constant(&b, "G");
        let b_props = {
            let global = b.interp.global.borrow();
            &global.props as *const _ as usize
        };
        assert_eq!(crate::object_fuse::active_global_props(), b_props);
        eval(&mut a, "G = 2;");
        assert_eq!(state(&a, "G"), PropertyState::NotConstant);
        assert_eq!(state(&b, "G"), PropertyState::Constant);
        eval(&mut b, "G = 2;");
        assert_eq!(state(&b, "G"), PropertyState::NotConstant);
    }
    assert_eq!(crate::object_fuse::active_global_props(), 0);
    make_constant(&a, "H");
    eval(&mut a, "H = 2;");
    assert_eq!(state(&a, "H"), PropertyState::NotConstant);
}
