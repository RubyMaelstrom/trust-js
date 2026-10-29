//! Decision 21: one environment plan per function, which every tier follows.
//!
//! V8 (`Scope::Lookup`, `ContextChainLength`), SpiderMonkey (`EnvironmentCoordinate`,
//! `GetAliasedVar`) and JSC (`resolve_scope`/`get_from_scope`) each have one scope analysis
//! that decides which bindings live in an environment record, whether a scope makes a record at
//! all, and the record's slots; every tier builds records that way, so a (hops, slot) proven
//! once holds for every closure of a function. Here the compiled tiers make a function's record
//! from its `ActivationPlan` and the tree-walker makes its own, so the plan below is what both
//! follow:
//!
//! - [`PlanKind::Public`]: the function's activation is a record in every tier, and its first
//!   bindings, slot for slot, are [`FunctionPlan::prefix`]: the function-level bindings that an
//!   inner function names (and `this` when an inner arrow reads it). The compiled tiers order
//!   their `ActivationPlan` so; the tree-walker inserts the prefix first.
//! - [`PlanKind::Private`]: nothing in the activation is named by an inner function. Compiled
//!   code makes no record; the tree-walker's record is marked private, and a closure made in it
//!   takes the nearest record that is not private as its [[Environment]] (`closure_env`), as if
//!   the record did not exist.
//! - [`PlanKind::Opaque`]: anything else (a direct eval or `with` anywhere inside, parameter
//!   expressions, an arrow reading `new.target`, a derived constructor, ...): the records stay
//!   as each tier makes them, and no static address crosses the function's scope.
//!
//! Each function's [`OuterContext`] lists the planned records from its [[Environment]] outward,
//! as its enclosing function's plan found them; a name an inner function reads resolves through
//! it to a static (hops, slot) when only planned records lie on the path. Sloppy direct eval and
//! `with` make the scopes they could shadow opaque, so their names keep the name caches, which
//! are the guard (SpiderMonkey's rule: `EmitterScope.cpp`'s dynamic fallbacks).

use super::*;
use std::cell::OnceCell;

/// A function's plan and outer context, cached on its AST (shared by every heap: both are
/// syntactic).
#[derive(Default, Debug)]
pub(crate) struct FunctionScopes {
    plan: OnceCell<Rc<FunctionPlan>>,
    /// Set by the enclosing function's plan, when that plan was made.
    outer: OnceCell<Rc<OuterContext>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanKind {
    Opaque,
    Private,
    Public,
}

/// One planned binding of a record: its name and whether a read must test it for TDZ.
#[derive(Debug)]
pub(crate) struct PlanBinding {
    pub(crate) name: Rc<str>,
    /// A `let`/`const`/`class` (or `this`, uninitialized in a derived constructor) may be read
    /// before its initialization: a read tests the empty word.
    pub(crate) lexical: bool,
}

#[derive(Debug)]
pub(crate) struct FunctionPlan {
    pub(crate) kind: PlanKind,
    /// A public activation's leading bindings, slot for slot, in canonical (name) order.
    pub(crate) prefix: Rc<[PlanBinding]>,
    /// A public activation's in-place capacity for the tree-walker: at least every binding its
    /// FunctionDeclarationInstantiation can make, so that its record never spills.
    pub(crate) capacity: usize,
    /// Why an opaque plan is (for `TRUST_JS_SCOPE_LOG`).
    pub(crate) why: &'static str,
}

impl FunctionPlan {
    fn opaque(why: &'static str) -> Rc<Self> {
        Rc::new(Self {
            kind: PlanKind::Opaque,
            prefix: Rc::from(Vec::new()),
            capacity: 0,
            why,
        })
    }
}

/// The planned records around a function, innermost first: what lies between its
/// [[Environment]] and the first scope that is not planned (a barrier), as far as static
/// addressing may look.
#[derive(Debug)]
pub(crate) struct OuterContext {
    /// Each record's planned bindings, slot for slot; a record may hold others after them.
    records: Vec<Rc<[PlanBinding]>>,
    /// The enclosing function's context, which continues after `records`; `None` is a barrier.
    parent: Option<Rc<OuterContext>>,
    /// Why the context ends after `records` without a parent, when a plan found a barrier
    /// there (for `TRUST_JS_SCOPE_LOG`).
    stop: Option<&'static str>,
}

/// A name's static address from a function's [[Environment]]: `hops` parent links, then `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OuterAddress {
    pub(crate) hops: u16,
    pub(crate) slot: u16,
    pub(crate) lexical: bool,
}

impl OuterContext {
    /// The static address of `name`, if only planned records lie between the function's
    /// [[Environment]] and the record that binds it.
    pub(crate) fn resolve(&self, name: &str) -> Option<OuterAddress> {
        let mut hops = 0usize;
        let mut context = Some(self);
        while let Some(current) = context {
            for record in &current.records {
                if let Some(slot) = record.iter().position(|binding| &*binding.name == name) {
                    return Some(OuterAddress {
                        hops: u16::try_from(hops).ok()?,
                        slot: u16::try_from(slot).ok()?,
                        lexical: record[slot].lexical,
                    });
                }
                hops += 1;
            }
            context = current.parent.as_deref();
        }
        None
    }

    /// Where a resolution of a name it does not bind stops (for `TRUST_JS_SCOPE_LOG`).
    pub(crate) fn stop(&self) -> &'static str {
        let mut context = self;
        while let Some(parent) = context.parent.as_deref() {
            context = parent;
        }
        context.stop.unwrap_or("end of the planned records")
    }
}

impl crate::ast::Function {
    /// The function's plan, made on first use (see the module docs). `derived` says whether the
    /// function is a derived class's constructor, which keeps its records as each tier makes them.
    pub(crate) fn scope_plan(&self, derived: bool) -> &FunctionPlan {
        self.scopes.plan.get_or_init(|| {
            let plan = plan(self, derived);
            if super::scope_log_enabled() {
                eprintln!(
                    "[scope-plan] {}@{} {:?} {} prefix={}",
                    self.name.as_deref().unwrap_or("<anonymous>"),
                    self.start,
                    plan.kind,
                    plan.why,
                    plan.prefix.len()
                );
            }
            plan
        })
    }

    /// The planned records around the function (see [`OuterContext`]), if its enclosing
    /// function's plan recorded them.
    pub(crate) fn outer_context_rc(&self) -> Option<Rc<OuterContext>> {
        self.scopes.outer.get().cloned()
    }
}

/// The binding a named function expression's self-name record holds (see
/// `Interp::make_function_named`).
fn self_name_record(function: &crate::ast::Function) -> Rc<[PlanBinding]> {
    Rc::from(vec![PlanBinding {
        name: Rc::from(function.name.as_deref().unwrap_or_default()),
        lexical: false,
    }])
}

/// The binding an object literal method's home record holds (`%homeobject%`).
fn home_record() -> Rc<[PlanBinding]> {
    Rc::from(vec![PlanBinding {
        name: Rc::from("%homeobject%"),
        lexical: false,
    }])
}

fn plan(function: &crate::ast::Function, derived: bool) -> Rc<FunctionPlan> {
    if !scope_plans_enabled() {
        return FunctionPlan::opaque("plans off");
    }
    if function.self_hosted || function.default_ctor {
        return FunctionPlan::opaque("built-in");
    }
    let mut scan = CaptureScan::new(function, true);
    scan.children = Some(Vec::new());
    let Some((captured, env_this, homed, _runtime, direct_eval, scan)) =
        CaptureScan::run_scan(scan, function)
    else {
        return FunctionPlan::opaque("capture analysis");
    };
    let opaque = if derived {
        Some("derived constructor")
    } else if crate::interpreter::params_have_expr(&function.params) {
        Some("parameter expressions")
    } else if function.scan_flags() & (crate::ast::SCAN_NEW_TARGET | crate::ast::SCAN_HOME) != 0 {
        // `new.target` and `super` (a derived constructor's `super()` among them) are found by
        // walking the records, from the arrows that read them too; keep those records whole.
        Some("new.target or super")
    } else if direct_eval || scan.saw_with {
        Some("direct eval or with")
    } else {
        None
    };
    if let Some(why) = opaque {
        // Inner functions keep their own self-name and home records.
        record_children(function, &scan, None, Some(why));
        return FunctionPlan::opaque(why);
    }
    // The function-level bindings an inner function can name: its own name (a function
    // expression's) lives in its self-name record instead, outside the activation.
    let mut declared = crate::fasthash::FastSet::default();
    for name in crate::interpreter::param_bound_names(&function.params) {
        declared.insert(name);
    }
    if !function.is_arrow {
        declared.insert("arguments".to_string());
    }
    if !hoisted_vars(&function.body, true, function.is_strict, &mut declared) {
        return FunctionPlan::opaque("hoisting");
    }
    let blocked = crate::interpreter::param_bound_names(&function.params);
    let hoist = crate::interpreter::collect_hoist_ops(&function.body, function.is_strict, &blocked);
    for op in &hoist {
        if let HoistOp::AnnexB(name, _) = op {
            declared.insert(name.clone());
        }
    }
    let mut lexicals = crate::fasthash::FastSet::default();
    scan.declare_lexicals(&function.body, &mut lexicals);
    let mut prefix: Vec<PlanBinding> = captured
        .iter()
        .filter(|name| declared.contains(*name) || lexicals.contains(*name))
        .map(|name| PlanBinding {
            name: Rc::from(name.as_str()),
            lexical: lexicals.contains(name) && !declared.contains(name),
        })
        .collect();
    if env_this && !function.is_arrow {
        prefix.push(PlanBinding {
            name: Rc::from("this"),
            lexical: true,
        });
    }
    prefix.sort_by(|left, right| left.name.cmp(&right.name));
    let public = !prefix.is_empty() || !homed.is_empty();
    let coroutine = function.is_generator || function.is_async;
    let scan_flags = function.scan_flags();
    // A sloppy body that reads `arguments` with simple formals: whether its compiled form keeps
    // the formals in an activation depends on how it uses the object (`compile_inner`).
    let mapped = !function.is_strict
        && !function.is_arrow
        && scan_flags & crate::ast::SCAN_ARGUMENTS != 0
        && !function.params.is_empty()
        && function.params.iter().all(|parameter| {
            !parameter.rest
                && parameter.default.is_none()
                && matches!(parameter.pattern, Pattern::Ident(_))
        });
    let kind = if public {
        PlanKind::Public
    } else if coroutine {
        record_children(function, &scan, None, Some("coroutine"));
        return FunctionPlan::opaque("coroutine");
    } else if mapped {
        record_children(function, &scan, None, Some("mapped arguments"));
        return FunctionPlan::opaque("mapped arguments");
    } else {
        PlanKind::Private
    };
    // Every binding FunctionDeclarationInstantiation can make: formals, hoisted names, body
    // lexicals, and the activation's own (`this`, `arguments`, `new.target`, the super markers).
    let capacity = blocked.len() + hoist.len() + lexicals.len() + 8;
    if kind == PlanKind::Public && capacity > crate::interpreter::MAX_INLINE_SLOTS {
        record_children(function, &scan, None, Some("activation too large"));
        return FunctionPlan::opaque("activation too large");
    }
    let prefix: Rc<[PlanBinding]> = Rc::from(prefix);
    record_children(
        function,
        &scan,
        (kind == PlanKind::Public).then(|| prefix.clone()),
        None,
    );
    Rc::new(FunctionPlan {
        kind,
        prefix,
        capacity,
        why: "",
    })
}

/// Give each inner function the scan met directly in `function`'s body its outer context: its
/// own self-name or home record, then (unless a barrier or `stop` intervenes) `activation`, a
/// public activation's planned bindings, and `function`'s own outer context.
fn record_children(
    function: &crate::ast::Function,
    scan: &CaptureScan,
    activation: Option<Rc<[PlanBinding]>>,
    stop: Option<&'static str>,
) {
    let outer = function.scopes.outer.get().cloned();
    for &(child, barrier, home) in scan.children.as_deref().unwrap_or_default() {
        // SAFETY: the scan borrowed `function`'s AST, which owns every child it met.
        let child = unsafe { &*child };
        let mut records = Vec::new();
        if child.needs_self_environment() {
            records.push(self_name_record(child));
        } else if home && child.observes_home_object() {
            records.push(home_record());
        }
        let stop = barrier.or(stop);
        let parent = if stop.is_none() {
            records.extend(activation.clone());
            outer.clone()
        } else {
            None
        };
        let _ = child.scopes.outer.set(Rc::new(OuterContext {
            records,
            parent,
            stop,
        }));
    }
}

/// `TRUST_JS_SCOPE_PLANS=0` turns Decision 21's plans off: every function is opaque.
pub(crate) fn scope_plans_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TRUST_JS_SCOPE_PLANS").as_deref() != Ok("0"))
}
