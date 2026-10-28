//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

/// `ShadowRealm`: each instance owns a fully isolated sub-interpreter. `evaluate` runs source in it and
/// only lets primitive completion values cross back (callables are wrapped; objects are a TypeError).
pub(super) fn install_shadow_realm(it: &mut Interp) {
    let proto = Object::new(Some(it.object_proto));
    Rc::make_mut(&mut it.extra_protos).insert("ShadowRealm", proto);
    set_to_string_tag(it, &proto, "ShadowRealm");
    it.def_method(&proto, "evaluate", 1, shadow_evaluate);
    it.def_method(&proto, "importValue", 2, shadow_import_value);

    let ctor = it.make_native("ShadowRealm", 0, |i, _t, _a| {
        if !i.constructing {
            return Err(i.make_error("TypeError", "ShadowRealm constructor requires 'new'"));
        }
        let obj = Object::new(i.extra_protos.get("ShadowRealm").cloned());
        let p = Gc::as_ptr(&obj) as usize;
        // The realm lives in a pointer-keyed internal-slot table. Pin the owner against allocator
        // address reuse; the collector evicts the Box before releasing the pin.
        i.gc_pin(&obj);
        let realm = Interp::new_shadow(i, &obj);
        i.shadow_realms.insert(p, realm);
        // Constructing the isolated implementation heap activates it while its intrinsics are
        // allocated. The constructor itself continues in the caller's surrounding Agent.
        i.activate_gc_heap();
        Ok(Value::Obj(obj))
    });
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor)));
    it.global
        .borrow_mut()
        .props
        .insert("ShadowRealm", Property::builtin(Value::Obj(ctor)));
}

/// `this` as a ShadowRealm object this interpreter owns the realm of.
fn this_shadow_realm(i: &Interp, this: &Value) -> Option<Gc> {
    this.as_obj()
        .filter(|object| i.shadow_realms.contains_key(&(Gc::as_ptr(object) as usize)))
}

fn shadow_evaluate(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    let realm = this_shadow_realm(i, &this).ok_or_else(|| {
        i.make_error(
            "TypeError",
            "ShadowRealm.prototype.evaluate called on a non-ShadowRealm",
        )
    })?;
    let src = match (arg(a, 0)).view() {
        ValueView::Str(s) => s.to_string(),
        _ => {
            return Err(i.make_error(
                "TypeError",
                "ShadowRealm.prototype.evaluate expects a string",
            ))
        }
    };
    // A parse failure is a SyntaxError of the *calling* realm (not wrapped).
    let (body, source) = match crate::parser::parse_script_from(
        &src,
        false,
        crate::stack_trace::SourceKind::Eval,
        None,
    ) {
        Ok(parsed) => parsed,
        Err(e) => return Err(i.make_error("SyntaxError", e.message)),
    };
    let result = i
        .with_shadow(&realm, |sub| {
            let origin = sub.default_runtime_origin(None);
            sub.with_runtime_origin(Some(origin), |sub| {
                // PerformShadowRealmEval: like eval code — `var`s instantiate in the realm's global,
                // lexicals live in a fresh declarative environment per evaluation.
                sub.strict = matches!(
                    body.first(),
                    Some(crate::ast::Stmt::Expr(crate::ast::Expr::Str(d))) if &**d == "use strict"
                );
                let genv = sub.global_env;
                let scope = crate::interpreter::new_scope(Some(genv));
                sub.hoist(&body, &genv, &[]);
                sub.declare_block_lexicals(&body, &scope, false);
                sub.in_code(Some(source), |sub| sub.run_stmt_list(&body, &scope))
                    .map(|v| match v.view() {
                        ValueView::Empty => Value::Undefined,
                        _ => v,
                    })
            })
        })
        .expect("checked above");
    match result {
        // Primitives cross the realm boundary directly; callables wrap; other objects throw.
        Ok(v) => ab(i.make_shadow_result(&realm, v)),
        // A host interruption is no completion for the boundary to copy (HTML "abort a running
        // script"): it passes, as V8's evaluate returns at once while execution terminates, and
        // the native boundary resumes it from this lowering.
        Err(crate::interpreter::Abrupt::Interrupt(_)) => Err(Value::Undefined),
        // An error thrown inside the shadow realm is re-thrown as a TypeError of the calling realm.
        Err(_) => Err(i.make_error(
            "TypeError",
            "ShadowRealm evaluate: the provided source threw an error",
        )),
    }
}

/// ShadowRealm.prototype.importValue: load `specifier` as a module inside the sub-realm and
/// marshal its `exportName` export out (primitive or wrapped callable).
fn shadow_import_value(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    // Brand check, specifier ToString, and exportName validation all throw synchronously.
    let realm = this_shadow_realm(i, &this).ok_or_else(|| {
        i.make_error(
            "TypeError",
            "ShadowRealm.prototype.importValue called on a non-ShadowRealm",
        )
    })?;
    // Capture the initiating authorization/base before specifier coercion can reenter host
    // code. The child execution Realm/settings remain independently selected below.
    let authority = i.load_authority();
    let base = i.import_base.clone();
    let spec = ab(i.to_string(&arg(a, 0)))?.to_string();
    let name = match (arg(a, 1)).view() {
        ValueView::Str(s) => s.to_string(),
        _ => return Err(i.make_error("TypeError", "importValue exportName must be a string")),
    };
    let promise = crate::rooting::Rooted::new(i.new_promise());
    let requester = i.realm_route();
    let requester_realm = i.global;
    let context = i.host_job_context;
    let loader = i.module_loader.clone();
    let async_loader = i.dynamic_module_loader.clone();
    // Promise tables are per interpreter. Reuse the ordinary asynchronous loader/evaluation
    // path inside the child, then publish a routed requester continuation; never pass an outer
    // capability to a child's local promise_then table or drain child jobs synchronously.
    let (request, child) = i
        .with_shadow(&realm, |sub| {
            sub.module_loader = loader;
            sub.dynamic_module_loader = async_loader;
            sub.import_base = base.clone();
            let request = sub.delegated_import(&spec, &base, authority);
            (request, sub.realm_route())
        })
        .expect("checked ShadowRealm owner");
    let request = crate::rooting::Rooted::new((request, child));
    let ptr = promise
        .as_obj()
        .map(|object| Gc::as_ptr(&object) as usize)
        .expect("promise object");
    i.routed_imports.insert(
        &i.gc_heap,
        ptr,
        crate::agent_control::RoutedImport {
            requester: requester.clone(),
            requester_realm,
            context,
            authority,
            source: request.1.clone(),
            request: request.0,
            mode: crate::agent_control::ImportMode::Shadow {
                shadow: realm,
                name,
            },
            origin: None,
        },
    );
    let realm_key = Gc::as_ptr(&requester_realm) as usize;
    let on_f = crate::interpreter::JobCallback {
        internal: crate::interpreter::InternalJob::RoutedImportFulfilled {
            requester: requester.target,
            realm: realm_key,
            context,
        },
        ..i.make_job_callback(*promise)
    };
    let on_r = crate::interpreter::JobCallback {
        internal: crate::interpreter::InternalJob::RoutedImportRejected {
            requester: requester.target,
            realm: realm_key,
            context,
        },
        ..i.make_job_callback(*promise)
    };
    let handlers = crate::rooting::Rooted::new((on_f, on_r));
    i.with_route(request.1.target, |sub| {
        sub.perform_promise_then(
            &request.0,
            handlers.0.clone(),
            handlers.1.clone(),
            Value::Undefined,
        );
    });
    Ok(*promise)
}

impl Interp {
    /// This built-in step runs in the requester Realm/settings, after the child's entire module
    /// evaluation (including TLA) settles. Its conditional slots stop being roots on completion.
    pub(crate) fn finish_routed_import(
        &mut self,
        outer: Value,
        namespace_or_error: Value,
        fulfilled: bool,
    ) -> Result<(), crate::InterruptReason> {
        let Some(ptr) = outer.as_obj().map(|object| Gc::as_ptr(&object) as usize) else {
            return Ok(());
        };
        let Some(state) = self.routed_imports.remove(&ptr) else {
            return Ok(());
        };
        let state = crate::rooting::Rooted::new((state, outer, namespace_or_error));
        assert_eq!(
            state.0.requester.target, self.realm_id,
            "importValue dispatch owner"
        );
        let crate::agent_control::ImportMode::Shadow { shadow, name } = &state.0.mode else {
            // Ordinary imports preserve original namespace/exports/rejection identity. Only
            // explicit ShadowRealm importValue performs wrapping or TypeError copying.
            if fulfilled {
                self.resolve_promise(&state.1, state.2);
            } else {
                self.reject_promise(&state.1, state.2);
            }
            return Ok(());
        };
        let outcome = if fulfilled {
            self.with_route(state.0.source.target, |child| {
                // HasOwnProperty distinguishes an existing `undefined` export from absence.
                match has_own_property_trapped(child, &state.2, name) {
                    Ok(true) => child.get_member_key(&state.2, name),
                    Ok(false) => Err(child.throw("TypeError", "importValue export does not exist")),
                    Err(error) => Err(crate::interpreter::Abrupt::Throw(error)),
                }
            })
            .and_then(|value| self.make_shadow_result(shadow, value))
        } else {
            // CreateTypeErrorCopy must execute no ECMAScript. The implementation-defined text
            // never reads properties/coerces the original error from the child (proposal9ff2).
            Err(self.throw(
                "TypeError",
                "ShadowRealm importValue module evaluation failed",
            ))
        };
        match outcome {
            Ok(value) => self.resolve_promise(&state.1, value),
            Err(crate::interpreter::Abrupt::Interrupt(reason)) => return Err(reason),
            Err(_) => {
                let error =
                    self.make_error("TypeError", "ShadowRealm importValue export is unavailable");
                self.reject_promise(&state.1, error);
            }
        }
        Ok(())
    }
}
