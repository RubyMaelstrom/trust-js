//! Pilot C's store writability (W4 step 9; feedback-schema.txt section 5.6): a store's or an
//! update's site holds data ways for writable data properties only, which ARM64's store and
//! update templates then trust without testing the attribute byte, and every write to a
//! property that is not writable (or stops being so) still takes the checked path, in every tier.
use crate::bytecode::{Chunk, Tier};
use crate::feedback::{NamedWayKind, SiteView};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("writability fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn chunk_of(engine: &mut Engine, name: &str) -> Option<std::rc::Rc<Chunk>> {
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var(name, &env).ok()?;
    let ValueView::Obj(function) = value.view() else {
        return None;
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        return None;
    };
    let runtime = user.func.runtime();
    runtime.code.get().and_then(Option::as_ref).cloned()
}

/// The data ways of `name`'s one store or update site.
fn data_ways(engine: &mut Engine, name: &str) -> usize {
    let Some(chunk) = chunk_of(engine, name) else {
        return 0;
    };
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| crate::bytecode::writes_property(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(pcs.len(), 1, "one store in {name}: {:?}", chunk.jit_ops());
    match crate::feedback::read(&chunk, pcs[0]) {
        SiteView::Named(named) | SiteView::NamedUpdate(named, _) => {
            assert!(
                named.writable,
                "{name}: a store site's ways prove writability"
            );
            named
                .filled()
                .filter(|way| way.kind == NamedWayKind::Data)
                .count()
        }
        other => panic!("{name}: {other:?}"),
    }
}

const TIERS: [Tier; 3] = [Tier::Interp, Tier::Bytecode, Tier::Jit];

#[test]
fn a_warm_store_site_stops_at_a_property_made_read_only() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let out = eval(
            &mut engine,
            "var o = {x: 0}; function s(p, v) { p.x = v; }
             for (var i = 1; i <= 20; i++) s(o, i);
             Object.defineProperty(o, 'x', {writable: false});
             s(o, 99);
             var f = Object.freeze({x: 1}); s(f, 5);
             var g = {x: 0}; for (var i = 1; i <= 20; i++) s(g, i);
             Object.freeze(g); s(g, 99);
             var h = {x: 0}; for (var i = 1; i <= 20; i++) s(h, i);
             Object.seal(h); s(h, 99);
             o.x + ',' + f.x + ',' + g.x + ',' + h.x",
        );
        assert_eq!(
            out, "20,1,20,99",
            "{tier:?}: a sealed property stays writable"
        );
    }
}

#[test]
fn a_store_site_never_holds_a_read_only_property() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let out = eval(
            &mut engine,
            "var o = {}; Object.defineProperty(o, 'x', {value: 1, writable: false});
             function s(p, v) { p.x = v; }
             for (var i = 0; i < 20; i++) s(o, i);
             function t(p, v) { 'use strict'; p.x = v; }
             var thrown = 0; for (var i = 0; i < 20; i++) { try { t(o, i); } catch (e) { thrown++; } }
             o.x + ',' + thrown",
        );
        assert_eq!(out, "1,20", "{tier:?}");
        if tier != Tier::Interp {
            assert_eq!(data_ways(&mut engine, "s"), 0, "{tier:?}");
        }
    }
}

#[test]
fn an_update_site_reads_and_holds_writable_data_only() {
    for tier in TIERS {
        let mut engine = engine(tier);
        let out = eval(
            &mut engine,
            "var r = {}; Object.defineProperty(r, 'x', {value: 1, writable: false});
             var w = {x: 1};
             function u(p) { p.x++; }
             for (var i = 0; i < 20; i++) u(r);
             var before = r.x;
             for (var i = 0; i < 20; i++) u(w);
             Object.freeze(w); u(w);
             before + ',' + r.x + ',' + w.x",
        );
        assert_eq!(out, "1,1,21", "{tier:?}");
        if tier != Tier::Interp {
            let ways = data_ways(&mut engine, "u");
            assert_eq!(ways, 1, "{tier:?}: the writable receiver's way only");
        }
    }
}

#[test]
fn a_second_stage_store_site_holds_no_read_sites_way() {
    // A read and a store of one name, in a caller whose second stage splices a callee: the
    // recompilation's seeds follow the sites in order, and a read's ways never stay in a store's,
    // whose generated code tests no attribute byte.
    let mut engine = engine(Tier::Jit);
    let source = "var ro = {}; Object.defineProperty(ro, 'x', {value: 7, writable: false});
         function id(v) { return v; }
         function rw(p, q) { var t = id(p.x); q.x = t + 1; return t; }
         var w = {x: 0};
         function drive(n) { var s = 0; for (var i = 0; i < n; i++) s += rw(ro, w); return s; }
         drive(150);";
    assert_eq!(eval(&mut engine, source), "1050");
    for _ in 0..20 {
        assert_eq!(eval(&mut engine, "drive(150)"), "1050");
    }
    assert_eq!(eval(&mut engine, "rw(ro, ro); ro.x + ',' + w.x"), "7,8");
}
