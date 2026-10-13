//! Speculative inline splices and Realms.
//!
//! `bytecode::plan_inlines` splices a callee only into a caller of the same Realm: the splice
//! evaluates the callee's literals, intrinsics and global references in the running Realm.
//! Code cached on an AST shared by several Realms (self-hosted built-ins, shared host
//! snapshots) runs in every one of them, so an inline guard must also prove that the running
//! Realm is the one the splice was planned in. Otherwise the guard takes the ordinary call,
//! which enters the callee's own Realm (ECMA-262 PrepareForOrdinaryCall: the callee context's
//! Realm is _func_.[[Realm]]; local snapshot e28783d).

use crate::bytecode::Tier;
use crate::interpreter::{Abrupt, Interp};
use crate::value::Value;
use crate::Engine;
use std::rc::Rc;

pub(crate) const TIERS: [Tier; 3] = [Tier::Interp, Tier::Bytecode, Tier::Jit];

/// Run `f` with `realm` installed as the active Realm, as an embedder's host operation does.
pub(crate) fn in_realm<R>(it: &mut Interp, realm: &Value, f: impl FnOnce(&mut Interp) -> R) -> R {
    let Value::Obj(global) = realm else {
        panic!("a Realm is named by its global object");
    };
    let key = Rc::as_ptr(global) as usize;
    let target = it
        .realms
        .get(&key)
        .expect("registered Realm")
        .snapshot_clone();
    let saved = it.snapshot_realm();
    it.restore_realm(&target);
    let result = f(it);
    let updated = it.snapshot_realm();
    it.realms.insert(key, updated);
    it.restore_realm(&saved);
    result
}

pub(crate) fn render(it: &mut Interp, value: &Value) -> String {
    match it.coerce_string(value) {
        Ok(text) => text.to_string(),
        Err(_) => "<unprintable>".into(),
    }
}

pub(crate) fn eval_string(it: &mut Interp, source: &str) -> String {
    let env = it.global_env.clone();
    match it.perform_eval(source, &env, false) {
        Ok(value) => render(it, &value),
        Err(Abrupt::Throw(error)) => panic!("{source:.40} threw: {}", render(it, &error)),
        Err(_) => panic!("{source:.40} completed abruptly"),
    }
}

pub(crate) fn set_global(it: &mut Interp, name: &str, value: Value) {
    let global = Value::Obj(it.global.clone());
    assert!(
        it.set_member(&global, name, value).is_ok(),
        "set global {name}"
    );
}

/// The self-hosted Array methods have always shared one AST between the Realms of an Agent
/// (`crate::self_hosted`). A callback inlined into one Realm's `map` must still run in its own
/// Realm when another Realm's `map` receives it.
#[test]
fn self_hosted_methods_run_another_realms_inlined_callback_in_its_realm() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let it = &mut *engine.interp;
        let main = Value::Obj(it.global.clone());
        // `map` is strict; only a callee of the same strictness is spliced into it.
        assert_eq!(
            eval_string(
                it,
                r#"globalThis.literal = function () { "use strict"; return [1]; };
                (function () {
                    let n = 0;
                    for (let i = 0; i < 2000; i++) n += [i].map(literal)[0].length;
                    return n;
                })()"#,
            ),
            "2000"
        );
        let child = it.create_realm();
        in_realm(it, &child, |it| set_global(it, "mainRealm", main.clone()));
        let result = in_realm(it, &child, |it| {
            eval_string(
                it,
                r#"(function () {
                    let same = 0;
                    for (let i = 0; i < 300; i++) {
                        const made = [i].map(mainRealm.literal)[0];
                        if (Object.getPrototypeOf(made) === mainRealm.Array.prototype) same++;
                    }
                    return same;
                })()"#,
            )
        });
        assert_eq!(result, "300", "{tier:?}");
    }
}
