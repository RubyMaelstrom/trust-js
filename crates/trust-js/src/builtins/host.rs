//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

/// The test262 `$262` host object (test262 INTERPRETING.md): `global`, `gc`, `createRealm`,
/// `evalScript`, `detachArrayBuffer`, `IsHTMLDDA`, `agent` and `AbstractModuleSource`.
pub(super) fn install_host(it: &mut Interp) {
    let host = make_262(it, None);
    set_builtin(&it.global, "$262", host);
}

/// Build a `$262` host object. `realm_global` is the global of the realm this `$262` controls
/// (None for the main realm, in which case the live `it.global` is used and `evalScript` runs in the
/// current realm).
fn make_262(it: &mut Interp, realm_global: Option<Value>) -> Value {
    let host = Object::new(Some(it.object_proto.clone()));
    let global = realm_global
        .clone()
        .unwrap_or_else(|| Value::Obj(it.global.clone()));
    set_builtin(&host, "global", global.clone());
    it.def_method(&host, "gc", 0, |i, _t, _a| {
        i.gc_collect();
        Ok(Value::Undefined)
    });
    // $262.IsHTMLDDA: callable and recognized by the language-level Annex B.3.6 exceptions.
    set_builtin(&host, "IsHTMLDDA", it.make_html_dda());
    it.def_method(&host, "createRealm", 0, |i, _t, _a| {
        let g = i.create_realm();
        Ok(make_262(i, Some(g)))
    });
    // evalScript runs in this $262's realm (the main realm's $262 keeps using the current global).
    let rg = realm_global.clone();
    if let Some(rg) = rg {
        set_internal(&host, "#\u{0}realm_global", rg);
    }
    it.def_method(&host, "evalScript", 1, |i, this, args| {
        let code = {
            let __value_match_original = arg(args, 0);
            match (__value_match_original).view() {
                ValueView::Str(s) => s,
                _ => {
                    let other = __value_match_original;
                    return Ok(other);
                }
            }
        };
        let rg = ab(i.get_internal_slot(&this, "#\u{0}realm_global"))?;
        if let ValueView::Obj(_) = &(rg).view() {
            return match i.eval_script_in_realm(&rg, &code) {
                Ok(value) => Ok(value),
                Err(Abrupt::Throw(value)) => Err(value),
                Err(Abrupt::Interrupt(reason)) => {
                    Err(i.make_error("QuotaExceededError", reason.message()))
                }
                Err(_) => Ok(Value::Undefined),
            };
        }
        let (body, source) = crate::parser::parse_script_from(
            &code,
            false,
            crate::stack_trace::SourceKind::Script,
            None,
        )
        .map_err(|e| i.make_error("SyntaxError", e.message))?;
        // A script runs with full GlobalDeclarationInstantiation (clash checks, global-object
        // own properties for var/function declarations).
        match i.run_program_from(&body, Some(source)) {
            Ok(value) => Ok(value),
            Err(Abrupt::Throw(value)) => Err(value),
            Err(Abrupt::Interrupt(reason)) => {
                Err(i.make_error("QuotaExceededError", reason.message()))
            }
            Err(_) => Ok(Value::Undefined),
        }
    });
    it.def_method(&host, "detachArrayBuffer", 1, |i, _t, args| {
        if let ValueView::Obj(o) = (arg(args, 0)).view() {
            let p = Gc::as_ptr(&o) as usize;
            // DetachArrayBuffer asserts IsSharedArrayBuffer is false: a Shared Data Block cannot be
            // detached, so refuse instead of leaving the buffer half detached.
            if i.is_shared_buffer(p) {
                return Err(i.make_error("TypeError", "Cannot detach a SharedArrayBuffer"));
            }
            // Truly detach: drop the backing store (so views see it as detached) and zero the views.
            i.invalidate_native_buffer(p);
            i.array_buffers.remove(&p);
            i.array_buffer_versions.remove(&p);
            i.array_buffer_dirty_ranges.remove(&p);
            let views: Vec<usize> = i
                .typed_arrays
                .iter()
                .filter(|(_, info)| info.buffer == p)
                .map(|(k, _)| *k)
                .collect();
            for vp in views {
                if let Some(info) = i.typed_arrays.get_mut(&vp) {
                    info.len = 0;
                }
            }
        }
        Ok(Value::Undefined)
    });
    install_agent(it, &host);
    set_builtin(
        &host,
        "AbstractModuleSource",
        make_abstract_module_source(it),
    );
    Value::Obj(host)
}

/// The %AbstractModuleSource% intrinsic exposed as `$262.AbstractModuleSource`: an abstract
/// constructor (throws when called), whose `.prototype` carries the `@@toStringTag` getter used by
/// module-source objects.
fn make_abstract_module_source(it: &mut Interp) -> Value {
    let ctor = it.make_native("AbstractModuleSource", 0, |i, _t, _a| {
        Err(i.make_error(
            "TypeError",
            "Abstract class AbstractModuleSource not directly constructable",
        ))
    });
    let proto = Object::new(Some(it.object_proto.clone()));
    // `@@toStringTag` getter: returns the source's name for a real module-source object, else undefined.
    if let Some(key) = well_known_key(it, "toStringTag") {
        let getter = it.make_native("get [Symbol.toStringTag]", 0, |_i, _t, _a| {
            Ok(Value::Undefined)
        });
        proto.borrow_mut().props.insert(
            key,
            Property::accessor_prop(Some(Value::Obj(getter)), None, false, true),
        );
    }
    proto.borrow_mut().props.insert(
        "constructor",
        Property::data(Value::Obj(ctor.clone()), true, false, true),
    );
    // `.prototype` is non-writable, non-enumerable, non-configurable. A host-defined concrete
    // module-source prototype inherits from it; ModuleSource instances inherit from that concrete
    // prototype (Source Phase Imports §28.1).
    Rc::make_mut(&mut it.extra_protos).insert("%AbstractModuleSourceProto%", proto.clone());
    Rc::make_mut(&mut it.extra_protos)
        .insert("%HostModuleSourceProto%", Object::new(Some(proto.clone())));
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto), false, false, false),
    );
    Value::Obj(ctor)
}

/// Reconstruct a SharedArrayBuffer object in this agent from an owning broadcast block.
fn agent_make_shared(i: &mut Interp, block: crate::interpreter::SharedMem, len: usize) -> Value {
    let obj = Object::new(i.extra_protos.get("SharedArrayBuffer").cloned());
    let p = Gc::as_ptr(&obj) as usize;
    i.gc_pin(&obj);
    let id = block.id();
    i.array_buffers.insert(p, BufferData::Shared(block));
    set_internal(&obj, "#\u{0}abMaxByteLength", Value::Num(len as f64));
    set_internal(&obj, "#\u{0}abResizable", Value::Bool(false));
    set_internal(&obj, "#\u{0}sab_id", Value::Num(id as f64));
    Value::Obj(obj)
}

/// `$262.agent`: the multi-agent harness (real OS threads sharing SharedArrayBuffer memory).
pub(super) fn install_agent(it: &mut Interp, host: &Gc) {
    let agent = Object::new(Some(it.object_proto.clone()));
    it.def_method(&agent, "start", 1, |i, _t, a| {
        let src = ab(i.to_string(&arg(a, 0)))?.to_string();
        // Lazily create the main agent's report channel.
        if i.agent.is_none() {
            let (report_tx, report_rx) = std::sync::mpsc::channel();
            i.agent = Some(Box::new(crate::interpreter::AgentChannels {
                agent_broadcast_txs: Vec::new(),
                report_rx: Some(report_rx),
                report_tx,
                broadcast_rx: None,
            }));
        }
        let (bcast_tx, bcast_rx) = std::sync::mpsc::channel();
        let report_tx = i.agent.as_ref().unwrap().report_tx.clone();
        i.agent.as_mut().unwrap().agent_broadcast_txs.push(bcast_tx);
        std::thread::spawn(move || {
            let mut eng = crate::Engine::new();
            eng.run_as_agent(&src, bcast_rx, report_tx);
        });
        Ok(Value::Undefined)
    });
    it.def_method(&agent, "broadcast", 2, |i, _t, a| {
        let sab = arg(a, 0);
        // Accept a SharedArrayBuffer directly, or a TypedArray view over one.
        let p = match sab.as_obj() {
            Some(o) => {
                let p = Gc::as_ptr(&o) as usize;
                if i.is_shared_buffer(p) {
                    p
                } else if let Some(info) = i.typed_arrays.get(&p) {
                    info.buffer
                } else {
                    return Err(i.make_error("TypeError", "broadcast requires a SharedArrayBuffer"));
                }
            }
            None => return Err(i.make_error("TypeError", "broadcast requires a SharedArrayBuffer")),
        };
        let block = i
            .shared_block(p)
            .ok_or_else(|| i.make_error("TypeError", "broadcast requires a SharedArrayBuffer"))?;
        let len = i.array_buffer_byte_length(p).unwrap_or(0);
        if let Some(ag) = &i.agent {
            for tx in &ag.agent_broadcast_txs {
                let _ = tx.send((block.clone(), len));
            }
        }
        Ok(Value::Undefined)
    });
    it.def_method(&agent, "getReport", 0, |i, _t, _a| {
        // Block briefly for the next report (the producing agent typically reports very soon).
        if let Some(ag) = &i.agent {
            if let Some(rx) = &ag.report_rx {
                return Ok(match rx.recv_timeout(std::time::Duration::from_secs(4)) {
                    Ok(s) => Value::lstr(s),
                    Err(_) => Value::Null,
                });
            }
        }
        Ok(Value::Null)
    });
    it.def_method(&agent, "sleep", 1, |i, _t, a| {
        let ms = ab(i.to_number(&arg(a, 0)))?;
        if ms.is_finite() && ms > 0.0 {
            std::thread::sleep(std::time::Duration::from_millis(ms as u64));
        }
        Ok(Value::Undefined)
    });
    it.def_method(&agent, "setTimeout", 2, |i, _t, a| {
        let f = arg(a, 0);
        if !f.is_callable() {
            return Err(i.make_error("TypeError", "setTimeout requires a callable"));
        }
        let ms = ab(i.to_number(&arg(a, 1)))?.max(0.0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ms as u64);
        i.pending_timers.push((f, deadline));
        Ok(Value::Undefined)
    });
    it.def_method(&agent, "monotonicNow", 0, |_i, _t, _a| {
        Ok(Value::Num(crate::interpreter::monotonic_now_ms()))
    });
    it.def_method(&agent, "receiveBroadcast", 1, |i, _t, a| {
        let cb = arg(a, 0);
        let received = {
            match i.agent.as_ref().and_then(|ag| ag.broadcast_rx.as_ref()) {
                Some(rx) => rx.recv().ok(),
                None => None,
            }
        };
        if let Some((block, len)) = received {
            let sab = agent_make_shared(i, block, len);
            ab(i.call(cb, Value::Undefined, &[sab]))?;
        }
        Ok(Value::Undefined)
    });
    it.def_method(&agent, "report", 1, |i, _t, a| {
        let s = ab(i.to_string(&arg(a, 0)))?.to_string();
        if let Some(ag) = &i.agent {
            let _ = ag.report_tx.send(s);
        }
        Ok(Value::Undefined)
    });
    it.def_method(&agent, "leaving", 0, |_i, _t, _a| Ok(Value::Undefined));
    set_data(host, "agent", Value::Obj(agent));
}
