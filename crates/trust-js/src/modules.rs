//! ES module loading, linking, and evaluation.
//!
//! TRust JS links modules in two passes, mirroring the spec's `Link` (Instantiate) and `Evaluate`
//! phases:
//!
//!   * **Instantiate** parses every reachable module, builds its export tables, creates its scope
//!     and (frozen) namespace object, hoists its top-level bindings, and wires each import to a
//!     *live* binding cell in the exporting module's scope. No user code runs.
//!   * **Evaluate** runs module bodies depth-first (dependencies before dependents, each once),
//!     so a live import observes the exporter's latest value even across circular dependencies.
//!
//! Specifier resolution + source fetching is delegated to a host loader (`Interp::module_loader`)
//! so the engine stays filesystem-agnostic.

use crate::ast::*;
use crate::builtins::make_bound_len;
use crate::interpreter::{new_scope, Abrupt, Binding, Env, Interp};
use crate::value::{Object, Property, Value};
use std::collections::HashMap;
use std::rc::Rc;

// One Engine can host more than one HTML environment settings object. The
// embedder's opaque host-job context is therefore part of the module-map key,
// while the canonical URL remains the only value exposed to loaders and
// `import.meta.url`. Context zero keeps the historical spelling so standalone
// and command-line embedders pay no allocation or compatibility cost.
const MODULE_CONTEXT_SEPARATOR: char = '\0';

pub(crate) fn module_map_key(context: u64, url: &str) -> String {
    if context == 0 || url.starts_with(MODULE_CONTEXT_SEPARATOR) {
        return url.to_string();
    }
    format!("{MODULE_CONTEXT_SEPARATOR}{context}{MODULE_CONTEXT_SEPARATOR}{url}")
}

pub(crate) fn module_map_context(key: &str) -> u64 {
    let Some(encoded) = key.strip_prefix(MODULE_CONTEXT_SEPARATOR) else {
        return 0;
    };
    encoded
        .split_once(MODULE_CONTEXT_SEPARATOR)
        .and_then(|(context, _)| context.parse().ok())
        .unwrap_or(0)
}

/// The URL a module's `import.meta.url` reports and its `import()` calls resolve against: its
/// public module-map URL, with a filesystem path as a `file:` URL.
fn module_meta_url(key: &str) -> String {
    let key = module_map_url(key);
    if key.starts_with('/') {
        format!("file://{key}")
    } else {
        key.to_string()
    }
}

pub(crate) fn module_map_url(key: &str) -> &str {
    let Some(encoded) = key.strip_prefix(MODULE_CONTEXT_SEPARATOR) else {
        return key;
    };
    encoded
        .split_once(MODULE_CONTEXT_SEPARATOR)
        .map_or(key, |(_, url)| url)
}

/// How a module-namespace property reads its current value.
#[derive(Clone)]
pub enum NsBinding {
    /// A live binding: read `local` from the exporting module's scope each time.
    Live(Env, String),
    /// A stable value (a star-as namespace re-export).
    Static(Value),
}

/// A parsed + linked module: its body, scope, namespace object, resolved export tables, and
/// evaluation status. Keyed by canonical specifier in `Interp::module_recs`.
pub(crate) struct ModuleRec {
    body: Rc<Vec<Stmt>>,
    /// The module's source record, for its top-level frame in stack traces.
    source: Rc<crate::stack_trace::ScriptSource>,
    env: Env,
    pub ns: Value,
    meta: Value,
    /// Dependency keys in source order (drives depth-first evaluation).
    dep_keys: Vec<String>,
    /// Specifier → resolved canonical key, for this module's own import/export-from clauses.
    resolved: HashMap<String, String>,
    /// `export name → local name` for names declared/defined in this module.
    local_exports: HashMap<String, String>,
    /// Local names that are themselves imports, so a re-export resolves through to the origin
    /// binding (spec: `import * as x; export {x}` resolves to the imported module's namespace).
    imports: HashMap<String, ImportOrigin>,
    /// `export name → (dependency key, imported name)` for `export { x } from 'dep'`.
    indirect: HashMap<String, (String, String)>,
    /// `export name → dependency key` for `export * as name from 'dep'`.
    star_as: HashMap<String, String>,
    /// Dependency keys of `export * from 'dep'` clauses.
    stars: Vec<String>,
    linked: bool,
    evaluated: bool,
    evaluating: bool,
    eval_error: Option<Value>,
    /// Dependency keys imported *only* via `import defer` — skipped during this module's
    /// evaluation phase (they evaluate on first namespace access instead).
    deferred_deps: Vec<String>,
    /// For a dynamically-imported module evaluating in a coroutine (top-level await): the promise
    /// that settles when the body finishes.
    top_promise: Option<Value>,
    /// This batch has visited the module (spec [[Status]] >= evaluating).
    started: bool,
    /// Count of not-yet-finished async-evaluating dependencies ([[PendingAsyncDependencies]]).
    pending_async: usize,
    /// Position in the async execution queue ([[AsyncEvaluationOrder]]).
    async_order: Option<u64>,
    /// Importers waiting on this module ([[AsyncParentModules]]).
    async_parents: Vec<String>,
    /// Tarjan bookkeeping for the evaluation DFS ([[DFSIndex]] / [[DFSAncestorIndex]]).
    dfs_index: Option<usize>,
    dfs_anc: usize,
    on_stack: bool,
    /// The root of this module's strongly-connected component ([[CycleRoot]]).
    cycle_root: Option<String>,
}

/// Heap-owned execution context for the AsyncBlock started by Source Text Module
/// ExecuteModule. The bytecode continuation owns every suspended stack/handler/resource frame;
/// this thin wrapper performs the module-record completion algorithms when that context ends.
pub(crate) struct ModuleCoro {
    body: crate::bytecode::VmCoro,
    key: String,
}

impl ModuleCoro {
    fn new(body: crate::bytecode::VmCoro, key: String) -> ModuleCoro {
        ModuleCoro { body, key }
    }

    pub(crate) fn resume(
        &mut self,
        i: &mut Interp,
        signal: crate::coroutine::Resume,
    ) -> crate::coroutine::Suspend {
        let result = self.body.resume(i, signal);
        match &result {
            crate::coroutine::Suspend::Done(_) => i.finish_dynamic_module(&self.key, None),
            crate::coroutine::Suspend::Throw(error) => {
                i.finish_dynamic_module(&self.key, Some(error.clone()))
            }
            crate::coroutine::Suspend::Await(_) | crate::coroutine::Suspend::Yield(_) => {}
        }
        result
    }

    pub(crate) fn done(&self) -> bool {
        self.body.done
    }

    pub(crate) fn started(&self) -> bool {
        self.body.started
    }

    pub(crate) fn terminate(&mut self, i: &mut Interp) {
        let _ = self.body.resume(i, crate::coroutine::Resume::Terminate);
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        std::mem::size_of::<ModuleCoro>()
            .saturating_add(self.key.capacity())
            .saturating_add(self.body.scan_retained_memory(visitor))
    }
}

/// The origin of a local name that is an import binding (so re-exports resolve to the source).
enum ImportOrigin {
    /// `import * as x from 'dep'` — resolves to `dep`'s namespace object.
    Namespace(String),
    /// `import defer * as x from 'dep'` — resolves to `dep`'s DEFERRED namespace object.
    DeferNamespace(String),
    /// `import { y as x } from 'dep'` / `import x from 'dep'` — resolves to `dep`'s export `y`.
    Named(String, String),
}

/// Promise capability and request metadata retained while an asynchronous host module fetch is in
/// flight. This is the engine-side payload passed conceptually from HostLoadImportedModule to
/// FinishLoadingImportedModule (ECMA-262 §16.2.1.10–11).
pub(crate) struct PendingDynamicImport {
    promise: Value,
    specifier: String,
    attr_type: Option<String>,
    defer: bool,
    pub(crate) host_context: u64,
}

impl ModuleRec {
    /// Scan allocations owned below one module-record table entry. The surrounding HashMap entry
    /// is accounted by Interp; shared AST, Values, and scope nodes retain canonical attribution.
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        fn strings(values: &Vec<String>) -> usize {
            values
                .capacity()
                .saturating_mul(std::mem::size_of::<String>())
                .saturating_add(
                    values
                        .iter()
                        .map(String::capacity)
                        .fold(0usize, usize::saturating_add),
                )
        }
        fn string_map<V>(values: &HashMap<String, V>) -> usize {
            values
                .len()
                .saturating_mul(std::mem::size_of::<(String, V)>())
                .saturating_add(
                    values
                        .keys()
                        .map(String::capacity)
                        .fold(0usize, usize::saturating_add),
                )
        }

        visitor.stmt_body(&self.body);
        visitor.value(&self.ns);
        visitor.value(&self.meta);
        if let Some(error) = &self.eval_error {
            visitor.value(error);
        }
        if let Some(promise) = &self.top_promise {
            visitor.value(promise);
        }
        let mut bytes = strings(&self.dep_keys)
            .saturating_add(strings(&self.stars))
            .saturating_add(strings(&self.deferred_deps))
            .saturating_add(strings(&self.async_parents))
            .saturating_add(string_map(&self.resolved))
            .saturating_add(string_map(&self.local_exports))
            .saturating_add(string_map(&self.imports))
            .saturating_add(string_map(&self.indirect))
            .saturating_add(string_map(&self.star_as));
        for value in self.resolved.values().chain(self.local_exports.values()) {
            bytes = bytes.saturating_add(value.capacity());
        }
        for origin in self.imports.values() {
            bytes = bytes.saturating_add(match origin {
                ImportOrigin::Namespace(module) | ImportOrigin::DeferNamespace(module) => {
                    module.capacity()
                }
                ImportOrigin::Named(module, name) => {
                    module.capacity().saturating_add(name.capacity())
                }
            });
        }
        for (module, name) in self.indirect.values() {
            bytes = bytes
                .saturating_add(module.capacity())
                .saturating_add(name.capacity());
        }
        for module in self.star_as.values() {
            bytes = bytes.saturating_add(module.capacity());
        }
        if let Some(root) = &self.cycle_root {
            bytes = bytes.saturating_add(root.capacity());
        }
        bytes
    }
}

impl NsBinding {
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        match self {
            NsBinding::Live(_, local) => local.capacity(),
            NsBinding::Static(value) => {
                visitor.value(value);
                0
            }
        }
    }
}

impl PendingDynamicImport {
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        visitor.value(&self.promise);
        self.specifier
            .capacity()
            .saturating_add(self.attr_type.as_ref().map_or(0, String::capacity))
    }
}

/// The result of resolving an export name (spec ResolveExport).
enum Resolution {
    /// A concrete binding: `local` in the given module scope.
    Local(Env, String),
    /// A re-exported `import defer * as ns` binding: the dep's DEFERRED namespace.
    DeferNs(String),
    /// A namespace object (a star-as re-export target).
    Ns(Value),
    /// Two star re-exports provide the name with different bindings.
    Ambiguous,
    /// The name is not exported.
    NotFound,
}

impl Interp {
    /// A new module evaluation promise: a [[TopLevelCapability]], or the promise this engine keeps
    /// for every module of an evaluating graph. Its rejection reaches authors only through the
    /// import() promise chained to it or the host's module script error report (HTML "run a
    /// module script" reacts to its rejection), so it is marked handled: HostPromiseRejectionTracker
    /// must not report the internal promise as an unhandled rejection.
    fn new_evaluation_promise(&mut self) -> Value {
        let promise = self.new_promise();
        if let Value::Obj(object) = &promise {
            if let Some(state) = self
                .promises
                .get_mut(&self.gc_heap, &(crate::value::Gc::as_ptr(object) as usize))
            {
                state.handled = true;
            }
        }
        promise
    }

    /// The evaluation promise for `key`, redirected to its cycle root once evaluation has begun.
    /// Browser embedders use this to delay a module script's completion steps through top-level
    /// await instead of treating the initial suspension as successful evaluation.
    #[cfg(feature = "embed")]
    pub(crate) fn module_evaluation_promise(&self, key: &str) -> Option<Value> {
        let key = module_map_key(self.host_job_context, key);
        let record = self.module_recs.get(&key)?;
        let evaluation_key = if (record.started || record.evaluated || record.evaluating)
            && record.cycle_root.is_some()
        {
            record.cycle_root.as_deref().unwrap_or(&key)
        } else {
            &key
        };
        self.module_recs.get(evaluation_key)?.top_promise.clone()
    }

    /// Load, link, and evaluate the module identified by canonical `key` (with initial `src`),
    /// returning its namespace object.
    pub(crate) fn load_module(&mut self, key: &str, src: &str) -> Result<Value, Abrupt> {
        let key = module_map_key(self.host_job_context, key);
        // Phase 1: parse the whole graph (so every module's export tables exist before any linking).
        self.load_requested_modules(&key, Some(ModuleSource::Source(src.to_string())))?;
        // Phase 2: link (hoist bindings, wire live imports, build namespaces) depth-first.
        self.link_module(&key)?;
        // Phase 3: evaluate module bodies depth-first. A graph containing top-level await
        // evaluates through the async machinery (awaits interleave with the job queue, an async
        // module doesn't block siblings, ancestors run in [[AsyncEvaluationOrder]]); a fully
        // synchronous graph runs directly.
        // Pre-create every graph member's evaluation promise, so a dynamic import of a
        // not-yet-executed batch member waits for the batch instead of executing it early.
        let mut graph = Vec::new();
        self.collect_eager_graph(&key, &mut graph);
        for m in &graph {
            if !self.module_recs[m].evaluated && self.module_recs[m].top_promise.is_none() {
                let p = self.new_evaluation_promise();
                self.module_recs.get_mut(m).unwrap().top_promise = Some(p);
            }
        }
        let top = self.module_recs[&key].top_promise.clone().unwrap();
        let mut stack = Vec::new();
        self.inner_module_evaluation_async(&key, &mut stack, &mut 0);
        self.drain_microtasks_interruptible()
            .map_err(Abrupt::Interrupt)?;
        if let Value::Obj(o) = &top {
            if let Some(ps) = self.promises.get(&(crate::value::Gc::as_ptr(o) as usize)) {
                if ps.status == 2 {
                    let reason = ps.value.clone();
                    return Err(Abrupt::Throw(reason));
                }
            }
        }
        Ok(self.module_recs[&key].ns.clone())
    }

    /// Every module in `key`'s dependency graph (including itself), depth-first.
    fn collect_graph(&self, key: &str, out: &mut Vec<String>) {
        if out.iter().any(|k| k == key) {
            return;
        }
        out.push(key.to_string());
        if let Some(rec) = self.module_recs.get(key) {
            for d in rec.dep_keys.clone() {
                self.collect_graph(&d, out);
            }
        }
    }

    /// Like collect_graph, but follows only eager (non-deferred) requests — exactly the modules a
    /// batch Evaluate() will execute, and therefore the ones whose evaluation promises may be
    /// pre-created. A deferred-only member must NOT get an orphan pending promise, or a later
    /// dynamic import of it would wait forever instead of evaluating it.
    fn collect_eager_graph(&self, key: &str, out: &mut Vec<String>) {
        if out.iter().any(|k| k == key) {
            return;
        }
        out.push(key.to_string());
        if let Some(rec) = self.module_recs.get(key) {
            let deferred = rec.deferred_deps.clone();
            let eager_named = self.eager_named_deps(key);
            for d in rec.dep_keys.clone() {
                // A dep imported both eagerly and with defer still evaluates with the batch.
                if deferred.contains(&d) && !eager_named.contains(&d) {
                    continue;
                }
                self.collect_eager_graph(&d, out);
            }
        }
    }

    /// Dependencies named by at least one non-defer import/export-from clause.
    fn eager_named_deps(&self, key: &str) -> Vec<String> {
        let rec = match self.module_recs.get(key) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let mut eager: Vec<String> = Vec::new();
        for stmt in rec.body.iter() {
            let (spec, defers) = match stmt {
                Stmt::Import(decl) => (
                    dep_map_key(&decl.source, decl.attr_type.as_deref()),
                    !decl.specs.is_empty()
                        && decl
                            .specs
                            .iter()
                            .all(|s| matches!(s, crate::ast::ImportSpec::DeferNamespace(_))),
                ),
                Stmt::ExportNamed {
                    source: Some(src), ..
                }
                | Stmt::ExportAll { source: src, .. } => (src.to_string(), false),
                _ => continue,
            };
            if defers {
                continue;
            }
            if let Some(k) = rec.resolved.get(&spec) {
                if !eager.iter().any(|x| x == k) {
                    eager.push(k.clone());
                }
            }
        }
        eager
    }

    // --- Parse phase --------------------------------------------------------------------------

    /// Parse and register a complete requested-module graph as one loading operation.
    ///
    /// ECMA-262 `LoadRequestedModules` only makes a graph linkable after every requested module
    /// has loaded successfully. `parse_and_register` necessarily publishes records early to break
    /// cycles, so an abrupt completion must discard records created by this attempt. Otherwise a
    /// later load of the same entry mistakes that partial graph for a complete one and linking can
    /// encounter a dependency for which no module record exists.
    fn load_requested_modules(
        &mut self,
        key: &str,
        src: Option<ModuleSource>,
    ) -> Result<(), Abrupt> {
        let existing: std::collections::HashSet<String> =
            self.module_recs.keys().cloned().collect();
        let result = self.parse_and_register(key, src);
        if result.is_err() {
            self.module_recs.retain(|key, _| existing.contains(key));
            self.modules.retain(|key, _| existing.contains(key));
        }
        result
    }

    /// Parse `key` and, transitively, every dependency — registering each module's environment,
    /// namespace object, and export tables. No user code runs and no linking happens yet, so a later
    /// ResolveExport can see the whole graph. Idempotent per key.
    fn parse_and_register(&mut self, key: &str, src: Option<ModuleSource>) -> Result<(), Abrupt> {
        if self.module_recs.contains_key(key) {
            return Ok(());
        }
        // A synthetic module (CreateDefaultExportSyntheticModule) has no code: it is an empty
        // module whose `default` binding holds a value made here, once, from the loaded contents.
        // An invalid JSON module fails the load (ParseJSONModule) before any module evaluates.
        let (src, default_export) = match src {
            Some(ModuleSource::Source(src)) => (src, None),
            Some(ModuleSource::Synthetic(contents)) => (
                String::new(),
                Some(self.synthetic_default_export(contents)?),
            ),
            None => return Err(self.throw("TypeError", format!("module not found: {key}"))),
        };
        // Module source text comes from the host (its loaders and entry points): Rust text, in
        // which a character of the engine's lone-surrogate range is a real character.
        let origin = crate::stack_trace::SourceOrigin::new(key);
        let (body, source) =
            crate::parser::parse_module_from(&crate::jstr::from_text(&src), Some(&origin))
                .map_err(|e| self.throw("SyntaxError", e.message))?;
        // The Module Record is the ScriptOrModule of its code; it refers by its URL.
        source.set_referrer(Rc::from(module_meta_url(key)));
        let body = Rc::new(body);

        // Resolve every dependency specifier to a canonical key up front (fetching its source), so
        // the export tables can name dependencies by key. Duplicate specifiers resolve once.
        let mut resolved: HashMap<String, String> = HashMap::new();
        let mut dep_keys: Vec<String> = Vec::new();
        let mut dep_srcs: Vec<(String, ModuleSource)> = Vec::new();
        // A dependency evaluates lazily only if *every* clause naming it is an `import defer`.
        let mut defer_specs: HashMap<String, bool> = HashMap::new();
        for stmt in body.iter() {
            let (spec, defers) = match stmt {
                Stmt::Import(decl) => (
                    decl.source.to_string(),
                    !decl.specs.is_empty()
                        && decl
                            .specs
                            .iter()
                            .all(|s| matches!(s, ImportSpec::DeferNamespace(_))),
                ),
                Stmt::ExportNamed {
                    source: Some(src), ..
                }
                | Stmt::ExportAll { source: src, .. } => (src.to_string(), false),
                _ => continue,
            };
            let e = defer_specs.entry(spec).or_insert(true);
            *e = *e && defers;
        }
        for (spec, attr_type) in module_dependency_specifiers(&body) {
            // The map key carries the attribute: the same specifier imported plainly AND with
            // `with { type: ... }` is two different dependencies.
            let map_key = dep_map_key(&spec, attr_type.as_deref());
            if resolved.contains_key(&map_key) {
                continue;
            }
            // The host-defined '<module source>' specifier resolves only in the source phase: it
            // gets a ModuleSource object but no module record.
            if spec == "<module source>" {
                resolved.insert(map_key, spec);
                continue;
            }
            let (canon, dsrc) = self.fetch_module(&spec, key, attr_type.as_deref())?;
            // A `with { type: ... }` dependency is a synthetic module (JSON/text/bytes) — keyed
            // separately from any ordinary module of the same file.
            let canon = match attr_type.as_deref() {
                Some(t @ ("json" | "text" | "bytes")) => format!("{canon}#{t}"),
                _ => canon,
            };
            let dsrc = typed_module_source(dsrc, attr_type.as_deref());
            resolved.insert(map_key, canon.clone());
            if !dep_keys.contains(&canon) {
                dep_keys.push(canon.clone());
                if !self.module_recs.contains_key(&canon) {
                    dep_srcs.push((canon, dsrc));
                }
            }
        }

        // Create the scope + namespace object, then build the export tables. This record must exist
        // before we recurse into dependencies so an import cycle resolves back to it.
        let env = new_scope(Some(self.global_env.clone()));
        // Top-level `this` in a module is `undefined` (not the global object).
        env.borrow_mut().vars.insert(
            "this".to_string(),
            Binding::data(Value::Undefined, false, true),
        );
        let ns_obj = Object::new(None);
        let ns = Value::Obj(ns_obj.clone());
        self.modules.insert(key.to_string(), ns.clone());
        let meta = self.build_import_meta(key);
        // `import.meta` resolves lexically: a function exported from this module keeps seeing
        // this module's object no matter who calls it.
        crate::eval::bind(&env, "%importmeta%", meta.clone());

        let mut tables = build_export_tables(&body, &resolved);
        if let Some(value) = default_export {
            env.borrow_mut()
                .vars
                .insert("*default*".to_string(), Binding::data(value, false, true));
            tables
                .local_exports
                .insert("default".to_string(), "*default*".to_string());
        }
        let deferred_deps: Vec<String> = resolved
            .iter()
            .filter(|(spec, _)| defer_specs.get(*spec).copied().unwrap_or(false))
            .map(|(_, canon)| canon.clone())
            .collect();
        self.module_recs.insert(
            key.to_string(),
            ModuleRec {
                body: body.clone(),
                source,
                env: env.clone(),
                ns,
                meta,
                dep_keys,
                resolved,
                local_exports: tables.local_exports,
                imports: tables.imports,
                indirect: tables.indirect,
                star_as: tables.star_as,
                stars: tables.stars,
                linked: false,
                evaluated: false,
                evaluating: false,
                eval_error: None,
                deferred_deps,
                top_promise: None,
                started: false,
                pending_async: 0,
                async_order: None,
                async_parents: Vec::new(),
                dfs_index: None,
                dfs_anc: 0,
                on_stack: false,
                cycle_root: None,
            },
        );

        // Recurse: parse each dependency (fetched during specifier resolution above).
        for (canon, dsrc) in dep_srcs {
            self.parse_and_register(&canon, Some(dsrc))?;
        }
        Ok(())
    }

    // --- Link phase ---------------------------------------------------------------------------

    /// Link `key` and its dependencies depth-first (once each): instantiate top-level bindings
    /// (functions initialized, lexicals in their temporal dead zone), wire imports to live cells,
    /// validate indirect exports, and build the namespace object.
    fn link_module(&mut self, key: &str) -> Result<(), Abrupt> {
        let mut stack = Vec::new();
        let result = self.inner_module_linking(key, &mut stack);
        if result.is_err() {
            // ECMA-262 Link step 4 resets every record which is still linking to unlinked. Records
            // that completed linking before a parent failed have already left this stack.
            for key in stack {
                if let Some(rec) = self.module_recs.get_mut(&key) {
                    rec.linked = false;
                }
            }
        }
        result
    }

    /// Recursive portion of `Link`. Entries remain in `stack` until their own initialization has
    /// succeeded, which lets the outer operation restore precisely the in-progress records after
    /// an abrupt completion.
    fn inner_module_linking(&mut self, key: &str, stack: &mut Vec<String>) -> Result<(), Abrupt> {
        let Some(rec) = self.module_recs.get(key) else {
            // A host loading failure must have rejected LoadRequestedModules before Link. Keep the
            // engine boundary total even if a corrupt/incomplete host graph reaches this point.
            return Err(self.throw("TypeError", format!("module graph is incomplete: {key}")));
        };
        if rec.linked {
            return Ok(());
        }
        self.module_recs.get_mut(key).unwrap().linked = true;
        stack.push(key.to_string());

        let (body, env, ns) = {
            let rec = &self.module_recs[key];
            (rec.body.clone(), rec.env.clone(), rec.ns.clone())
        };
        let dep_keys = self.module_recs[key].dep_keys.clone();
        for dep in &dep_keys {
            self.inner_module_linking(dep, stack)?;
        }

        self.hoist(&body, &env, &[]);
        self.declare_block_lexicals(&body, &env, false);
        self.declare_default_placeholder(&body, &env);
        self.validate_indirect_exports(key)?;
        self.link_imports(key, &body, &env)?;
        if let Value::Obj(ns_obj) = ns {
            self.build_namespace(key, &ns_obj)?;
        }
        let completed = stack.pop();
        debug_assert_eq!(completed.as_deref(), Some(key));
        Ok(())
    }

    /// Every `export { x } from 'dep'` entry must resolve to a single binding (spec: unresolvable or
    /// ambiguous indirect exports are a link-time SyntaxError).
    fn validate_indirect_exports(&mut self, key: &str) -> Result<(), Abrupt> {
        let names: Vec<String> = self.module_recs[key].indirect.keys().cloned().collect();
        for name in names {
            match self.resolve_export(key, &name, &mut Vec::new()) {
                Resolution::Local(..) | Resolution::Ns(..) | Resolution::DeferNs(..) => {}
                Resolution::Ambiguous => {
                    return Err(self.throw(
                        "SyntaxError",
                        format!("the requested module provides an ambiguous export named '{name}'"),
                    ));
                }
                Resolution::NotFound => {
                    return Err(self.throw(
                        "SyntaxError",
                        format!("the requested module does not provide an export named '{name}'"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Fetch a dependency's `(canonical_key, source)` via the host loader.
    fn fetch_module(
        &mut self,
        specifier: &str,
        referrer: &str,
        attr_type: Option<&str>,
    ) -> Result<(String, String), Abrupt> {
        let loader = match &self.module_loader {
            Some(l) => l.clone(),
            None => return Err(self.throw("TypeError", "no module loader configured")),
        };
        let context = module_map_context(referrer);
        let public_referrer = module_map_url(referrer);
        match loader(specifier, public_referrer, attr_type) {
            Some((canonical, source)) => Ok((module_map_key(context, &canonical), source)),
            None => Err(self.throw(
                "TypeError",
                format!("module not found: {specifier} (imported from {public_referrer})"),
            )),
        }
    }

    /// Build a module's `import.meta` object (`{ url, filename?, dirname? }`, extensible, no
    /// prototype). For a file-backed module the `url` is a `file://` URL (as Node's is, so
    /// `new URL(rel, import.meta.url)` resolves), and `filename`/`dirname` (Node 20.11+) are the
    /// plain paths.
    fn build_import_meta(&mut self, key: &str) -> Value {
        let meta = Object::new(None);
        let url = module_meta_url(key);
        let key = module_map_url(key);
        let is_path = key.starts_with('/');
        {
            let mut b = meta.borrow_mut();
            b.props
                .insert("url", Property::data(Value::lstr(url), true, true, true));
            if is_path {
                b.props.insert(
                    "filename",
                    Property::data(Value::lstr(key.to_string()), true, true, true),
                );
                let dir = std::path::Path::new(key)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                b.props.insert(
                    "dirname",
                    Property::data(Value::lstr(dir), true, true, true),
                );
            }
        }
        Value::Obj(meta)
    }

    /// Declare an uninitialized `*default*` binding for `export default <expression>` (and anonymous
    /// default function/class), so an importer can wire a live binding to it during linking.
    fn declare_default_placeholder(&mut self, body: &[Stmt], env: &Env) {
        for stmt in body {
            let Stmt::ExportDefault(inner) = stmt else {
                continue;
            };
            // `*default*` gets an uninitialized (TDZ) binding for an `export default <expression>` or
            // anonymous `class`. A named function/class binds its own name; an anonymous function is a
            // hoistable declaration already bound (initialized) by `hoist`.
            let needs_tdz = matches!(&**inner, Stmt::Expr(_))
                || matches!(&**inner, Stmt::ClassDecl(c) if c.name.is_none());
            if needs_tdz {
                env.borrow_mut().vars.insert(
                    "*default*".to_string(),
                    Binding {
                        value: Value::Undefined,
                        mutable: false,
                        strict_immutable: true,
                        initialized: false,
                        import_ref: None,
                        imported: false,
                        deletable: false,
                    },
                );
            }
        }
    }

    /// Wire every `import` binding in `body` to a live cell (or namespace object) of its dependency.
    fn link_imports(&mut self, key: &str, body: &[Stmt], env: &Env) -> Result<(), Abrupt> {
        for stmt in body {
            let Stmt::Import(decl) = stmt else { continue };
            let dep = self.resolved_key(key, &dep_map_key(&decl.source, decl.attr_type.as_deref()));
            for spec in &decl.specs {
                match spec {
                    ImportSpec::Namespace(local) => {
                        let ns = self.module_recs[&dep].ns.clone();
                        self.bind_local(env, local, ns);
                    }
                    ImportSpec::DeferNamespace(local) => {
                        let dns = self.make_deferred_ns(&dep);
                        self.bind_local(env, local, dns);
                    }
                    ImportSpec::Source(local) => {
                        // GetModuleSource of a Source Text Module Record throws a SyntaxError:
                        // only the host-defined '<module source>' module has a ModuleSource.
                        if dep != "<module source>" {
                            return Err(self.throw(
                                "SyntaxError",
                                "source-text modules have no module source",
                            ));
                        }
                        let src_obj = self.module_source_of(&dep);
                        self.bind_local(env, local, src_obj);
                    }
                    ImportSpec::Default(local) => {
                        self.link_named(env, local, &dep, "default")?;
                    }
                    ImportSpec::Named { imported, local } => {
                        self.link_named(env, local, &dep, imported)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Wire `local` to the binding that `dep` exports as `name` (a link-time SyntaxError if the
    /// export is missing or ambiguous).
    fn link_named(&mut self, env: &Env, local: &str, dep: &str, name: &str) -> Result<(), Abrupt> {
        match self.resolve_export(dep, name, &mut Vec::new()) {
            Resolution::Local(src_env, src_local) => {
                env.borrow_mut().vars.insert(
                    local.to_string(),
                    Binding {
                        value: Value::Undefined,
                        mutable: false,
                        strict_immutable: true,
                        initialized: true,
                        import_ref: Some(Box::new((src_env, Rc::from(src_local)))),
                        imported: true,
                        deletable: false,
                    },
                );
                Ok(())
            }
            Resolution::Ns(ns) => {
                self.bind_local(env, local, ns);
                Ok(())
            }
            Resolution::DeferNs(dep) => {
                let dns = self.make_deferred_ns(&dep);
                self.bind_local(env, local, dns);
                Ok(())
            }
            Resolution::Ambiguous => Err(self.throw(
                "SyntaxError",
                format!("the requested module provides an ambiguous export named '{name}'"),
            )),
            Resolution::NotFound => Err(self.throw(
                "SyntaxError",
                format!("the requested module '{dep}' does not provide an export named '{name}'"),
            )),
        }
    }

    /// The (cached, per canonical key) ModuleSource object a source-phase import binds. Source
    /// Phase Imports §28.1 requires its initial prototype to be a host-defined prototype whose own
    /// initial prototype is `%AbstractModuleSource%.prototype`.
    pub(crate) fn module_source_of(&mut self, dep: &str) -> Value {
        if let Some(v) = self.module_source_objs.get(dep) {
            return v.clone();
        }
        let proto = self.extra_protos.get("%HostModuleSourceProto%").cloned();
        let obj = Object::new(proto);
        let v = Value::Obj(obj);
        self.module_source_objs.insert(dep.to_string(), v.clone());
        v
    }

    fn bind_local(&self, env: &Env, local: &str, value: Value) {
        env.borrow_mut()
            .vars
            .insert(local.to_string(), Binding::data(value, false, true));
    }

    fn resolved_key(&self, referrer: &str, specifier: &str) -> String {
        self.module_recs[referrer]
            .resolved
            .get(specifier)
            .cloned()
            .unwrap_or_else(|| specifier.to_string())
    }

    // --- Export resolution (spec ResolveExport / GetExportedNames) -----------------------------

    /// Resolve `name` exported by module `key` to a concrete binding, following indirect and star
    /// re-exports. `seen` guards against cyclic re-export chains.
    fn resolve_export(
        &self,
        key: &str,
        name: &str,
        seen: &mut Vec<(String, String)>,
    ) -> Resolution {
        let pair = (key.to_string(), name.to_string());
        if seen.contains(&pair) {
            return Resolution::NotFound;
        }
        seen.push(pair);
        // A source-phase re-export resolves to the requested module's ModuleSource object.
        if name == "~source~" {
            return match self.module_source_objs.get(key) {
                Some(v) => Resolution::Ns(v.clone()),
                None => Resolution::NotFound,
            };
        }
        let rec = match self.module_recs.get(key) {
            Some(r) => r,
            None => return Resolution::NotFound,
        };
        if let Some(local) = rec.local_exports.get(name) {
            // A re-exported import resolves to its origin binding (so a namespace re-exported by two
            // paths compares equal, and a named re-export follows through to the real module).
            if let Some(origin) = rec.imports.get(local) {
                return match origin {
                    ImportOrigin::Namespace(dep) => match self.module_recs.get(dep) {
                        Some(d) => Resolution::Ns(d.ns.clone()),
                        None => Resolution::NotFound,
                    },
                    ImportOrigin::DeferNamespace(dep) => {
                        if self.module_recs.contains_key(dep) {
                            Resolution::DeferNs(dep.clone())
                        } else {
                            Resolution::NotFound
                        }
                    }
                    ImportOrigin::Named(dep, imported) => {
                        let (dep, imported) = (dep.clone(), imported.clone());
                        self.resolve_export(&dep, &imported, seen)
                    }
                };
            }
            return Resolution::Local(rec.env.clone(), local.clone());
        }
        if let Some((dep, imported)) = rec.indirect.get(name) {
            let (dep, imported) = (dep.clone(), imported.clone());
            return self.resolve_export(&dep, &imported, seen);
        }
        if let Some(dep) = rec.star_as.get(name) {
            if let Some(drec) = self.module_recs.get(dep) {
                return Resolution::Ns(drec.ns.clone());
            }
            return Resolution::NotFound;
        }
        if name == "default" {
            // `default` is never provided by a `export *` re-export.
            return Resolution::NotFound;
        }
        let stars = rec.stars.clone();
        let mut star_resolution = Resolution::NotFound;
        for dep in &stars {
            match self.resolve_export(dep, name, seen) {
                Resolution::Ambiguous => return Resolution::Ambiguous,
                Resolution::NotFound => {}
                r => match &star_resolution {
                    Resolution::NotFound => star_resolution = r,
                    existing => {
                        if !same_binding(existing, &r) {
                            return Resolution::Ambiguous;
                        }
                    }
                },
            }
        }
        star_resolution
    }

    /// All names module `key` exports (spec GetExportedNames), excluding `default` from stars.
    fn exported_names(&self, key: &str, seen: &mut Vec<String>) -> Vec<String> {
        if seen.contains(&key.to_string()) {
            return Vec::new();
        }
        seen.push(key.to_string());
        let rec = match self.module_recs.get(key) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let mut names: Vec<String> = Vec::new();
        let push = |n: &str, names: &mut Vec<String>| {
            if !names.iter().any(|x| x == n) {
                names.push(n.to_string());
            }
        };
        for n in rec.local_exports.keys() {
            push(n, &mut names);
        }
        for n in rec.indirect.keys() {
            push(n, &mut names);
        }
        for n in rec.star_as.keys() {
            push(n, &mut names);
        }
        let stars = rec.stars.clone();
        for dep in &stars {
            for n in self.exported_names(dep, seen) {
                if n != "default" {
                    push(&n, &mut names);
                }
            }
        }
        names
    }

    /// Populate `ns` with one entry per unambiguously-resolvable export, sorted by name, and record
    /// how each reads its live value. Namespace objects are frozen and prototype-less.
    fn build_namespace(&mut self, key: &str, ns: &crate::value::Gc) -> Result<(), Abrupt> {
        let mut names = self.exported_names(key, &mut Vec::new());
        names.sort();
        let mut live: crate::fasthash::FastMap<String, NsBinding> = Default::default();
        for name in &names {
            match self.resolve_export(key, name, &mut Vec::new()) {
                Resolution::Local(env, local) => {
                    live.insert(
                        crate::value::encode_owned_string_key(name.clone()),
                        NsBinding::Live(env, local),
                    );
                    // A placeholder own property makes the name enumerable / own-key-visible; reads
                    // go through the live map (see `get_member`).
                    ns.borrow_mut().props.insert(
                        crate::value::encode_string_key(name).as_ref(),
                        Property::data(Value::Undefined, true, true, false),
                    );
                }
                Resolution::Ns(v) => {
                    live.insert(
                        crate::value::encode_owned_string_key(name.clone()),
                        NsBinding::Static(v.clone()),
                    );
                    ns.borrow_mut().props.insert(
                        crate::value::encode_string_key(name).as_ref(),
                        Property::data(v, true, true, false),
                    );
                }
                Resolution::DeferNs(dep) => {
                    let v = self.make_deferred_ns(&dep);
                    live.insert(
                        crate::value::encode_owned_string_key(name.clone()),
                        NsBinding::Static(v.clone()),
                    );
                    ns.borrow_mut().props.insert(
                        crate::value::encode_string_key(name).as_ref(),
                        Property::data(v, true, true, false),
                    );
                }
                // Ambiguous / unresolvable star names are omitted from the namespace.
                _ => {}
            }
        }
        // @@toStringTag = "Module": non-writable, non-enumerable, non-configurable.
        if let Some(tag) = crate::builtins::to_string_tag_key(self) {
            ns.borrow_mut().props.insert(
                tag,
                Property::data(Value::lstr("Module".to_string()), false, false, false),
            );
        }
        {
            let mut b = ns.borrow_mut();
            b.extensible = false;
            b.props.demote_shape();
        }
        ns.borrow().ic_plain.set(false);
        // `module_ns` is keyed only by object address. Pin the namespace until
        // the collector can evict the side-table entry atomically; a
        // released per-Document module map may otherwise drop the last strong
        // namespace handle and let an unrelated allocation reuse its address.
        self.gc_pin(ns);
        self.module_ns
            .insert(&self.gc_heap, crate::value::Gc::as_ptr(ns) as usize, live);
        Ok(())
    }

    /// Build the distinct deferred-namespace object for `import defer * as ns`: same exports and
    /// live bindings as the module's ordinary namespace, but a separate identity, a
    /// "Deferred Module" @@toStringTag, and evaluation-on-first-string-keyed-access.
    fn make_deferred_ns(&mut self, dep: &str) -> Value {
        // One deferred namespace per module: every `import defer` of the same module (and every
        // re-export of such a binding) observes the same object.
        if let Some(v) = self.deferred_ns_objs.get(dep) {
            return v.clone();
        }
        let base = self.module_recs[dep].ns.clone();
        let Value::Obj(base_o) = &base else {
            return base;
        };
        let dns = Object::new(None);
        {
            let src = base_o.borrow();
            let mut dst = dns.borrow_mut();
            for (k, p) in src.props.iter() {
                let mut p = p.clone();
                if Interp::is_sym_key(k) {
                    if let Value::Str(tag) = p.value() {
                        if &*tag == "Module" {
                            p.set_value(Value::lstr("Deferred Module".to_string()));
                        }
                    }
                }
                dst.props.insert(k.clone(), p);
            }
            dst.extensible = false;
            dst.props.demote_shape();
        }
        if let Some(live) = self
            .module_ns
            .get(&(crate::value::Gc::as_ptr(base_o) as usize))
            .cloned()
        {
            dns.borrow().ic_plain.set(false);
            self.gc_pin(&dns);
            self.module_ns
                .insert(&self.gc_heap, crate::value::Gc::as_ptr(&dns) as usize, live);
        }
        dns.borrow().ic_plain.set(false);
        self.deferred_ns
            .insert(crate::value::Gc::as_ptr(&dns) as usize, dep.to_string());
        self.deferred_ns_objs
            .insert(dep.to_string(), Value::Obj(dns.clone()));
        Value::Obj(dns)
    }

    // --- Evaluate phase -----------------------------------------------------------------------

    /// Evaluate module `key` and its dependencies depth-first (each body runs at most once). A body
    /// that throws poisons the module so later imports observe the same error.
    /// Deferred-namespace trigger: evaluate a module on first access of its namespace.
    pub(crate) fn evaluate_deferred(&mut self, key: &str) -> Result<(), Abrupt> {
        if self.module_recs.contains_key(key) {
            // ReadyForSyncExecution: touching a deferred namespace while its module — or any
            // module in its dependency graph — is still evaluating is a TypeError.
            if !self.ready_for_sync(key, &mut Vec::new()) {
                return Err(self.throw(
                    "TypeError",
                    "cannot access a deferred namespace while its module graph is evaluating",
                ));
            }
            self.evaluate_module(key)?;
            // A stub minted mid-link for a cyclic dependency copied an empty export table —
            // hydrate it from the real namespace now that the module is linked and evaluated.
            if let Some(Value::Obj(stub)) = self.deferred_ns_objs.get(key).cloned() {
                let hydrated = self
                    .module_ns
                    .contains_key(&(crate::value::Gc::as_ptr(&stub) as usize));
                if !hydrated {
                    let base = self.module_recs[key].ns.clone();
                    if let Value::Obj(base_o) = &base {
                        {
                            let src = base_o.borrow();
                            let mut dst = stub.borrow_mut();
                            for (k, p) in src.props.iter() {
                                if dst.props.get(k).is_none() {
                                    let mut p = p.clone();
                                    if Interp::is_sym_key(k) {
                                        if let Value::Str(tag) = p.value() {
                                            if &*tag == "Module" {
                                                p.set_value(Value::lstr(
                                                    "Deferred Module".to_string(),
                                                ));
                                            }
                                        }
                                    }
                                    dst.props.insert(k.clone(), p);
                                }
                            }
                        }
                        if let Some(live) = self
                            .module_ns
                            .get(&(crate::value::Gc::as_ptr(base_o) as usize))
                            .cloned()
                        {
                            stub.borrow().ic_plain.set(false);
                            self.gc_pin(&stub);
                            self.module_ns.insert(
                                &self.gc_heap,
                                crate::value::Gc::as_ptr(&stub) as usize,
                                live,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether `key`'s whole (non-deferred) dependency graph is free of mid-evaluation modules.
    fn ready_for_sync(&self, key: &str, seen: &mut Vec<String>) -> bool {
        if seen.iter().any(|k| k == key) {
            return true;
        }
        seen.push(key.to_string());
        let rec = match self.module_recs.get(key) {
            Some(r) => r,
            None => return true,
        };
        if rec.evaluated && rec.eval_error.is_none() {
            return true;
        }
        // evaluating, evaluating-async (started and not finished), or parked at an await.
        if rec.evaluating || (rec.started && !rec.evaluated) {
            return false;
        }
        if !rec.evaluated && body_has_tla(&rec.body) {
            return false;
        }
        // Every requested module counts — deferred requests included (a deferred dep that is
        // itself mid-evaluation still blocks synchronous evaluation).
        let deps = rec.dep_keys.clone();
        deps.iter().all(|d| self.ready_for_sync(d, seen))
    }

    fn evaluate_module(&mut self, key: &str) -> Result<(), Abrupt> {
        {
            let rec = &self.module_recs[key];
            if let Some(err) = &rec.eval_error {
                return Err(Abrupt::Throw(err.clone()));
            }
            if rec.evaluated || rec.evaluating {
                return Ok(());
            }
        }
        self.module_recs.get_mut(key).unwrap().evaluating = true;

        let dep_keys = self.module_recs[key].dep_keys.clone();
        let deferred = self.module_recs[key].deferred_deps.clone();
        // Evaluate dependencies at the position of their first NON-defer clause (an
        // `import defer` earlier in the file must not pull the module's evaluation forward).
        let body_order = self.eager_dep_order(key);
        // Only deps in dep_keys evaluate here (source-phase pseudo-modules are excluded there).
        let order: Vec<String> = if body_order.is_empty() {
            dep_keys
                .iter()
                .filter(|d| !deferred.contains(*d))
                .cloned()
                .collect()
        } else {
            body_order
                .into_iter()
                .filter(|d| dep_keys.contains(d))
                .collect()
        };
        for dep in &order {
            if deferred.contains(dep) {
                // A dependency imported only via `import defer` evaluates lazily — but its
                // ASYNC (top-level-await) transitive dependencies still evaluate eagerly.
                self.evaluate_async_subgraph(dep, &mut Vec::new())?;
                continue;
            }
            self.evaluate_module(dep)?;
        }
        for dep in &dep_keys {
            if deferred.contains(dep) && !order.contains(dep) {
                self.evaluate_async_subgraph(dep, &mut Vec::new())?;
            }
        }

        let (body, env, meta, source) = {
            let rec = &self.module_recs[key];
            (
                rec.body.clone(),
                rec.env.clone(),
                rec.meta.clone(),
                rec.source.clone(),
            )
        };
        let saved_meta = self.import_meta.take();
        let saved_strict = self.strict;
        self.import_meta = Some(meta);
        self.strict = true;
        let result = self.in_code(Some(source), |this| {
            this.with_script_entry(|this| this.run_stmt_list(&body, &env))
        });
        self.import_meta = saved_meta;
        self.strict = saved_strict;

        let rec = self.module_recs.get_mut(key).unwrap();
        rec.evaluating = false;
        match result {
            Ok(_) => {
                rec.evaluated = true;
                Ok(())
            }
            Err(a) => {
                let v = crate::interpreter::abrupt_value(a);
                rec.eval_error = Some(v.clone());
                rec.evaluated = true;
                Err(Abrupt::Throw(v))
            }
        }
    }

    /// This module's dependencies in evaluation order: each dep at the position of its first
    /// non-defer import/export-from clause; defer-only deps at their first (defer) position.
    fn eager_dep_order(&self, key: &str) -> Vec<String> {
        let rec = match self.module_recs.get(key) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let body = rec.body.clone();
        let resolved = rec.resolved.clone();
        let mut eager: Vec<String> = Vec::new();
        let mut defer_seen: Vec<String> = Vec::new();
        let push = |list: &mut Vec<String>, k: &str| {
            if !list.iter().any(|x| x == k) {
                list.push(k.to_string());
            }
        };
        for stmt in body.iter() {
            let (spec, defers) = match stmt {
                Stmt::Import(decl) => (
                    dep_map_key(&decl.source, decl.attr_type.as_deref()),
                    !decl.specs.is_empty()
                        && decl
                            .specs
                            .iter()
                            .all(|s| matches!(s, crate::ast::ImportSpec::DeferNamespace(_))),
                ),
                Stmt::ExportNamed {
                    source: Some(src), ..
                }
                | Stmt::ExportAll { source: src, .. } => (src.to_string(), false),
                _ => continue,
            };
            let Some(k) = resolved.get(&spec) else {
                continue;
            };
            if defers {
                push(&mut defer_seen, k);
            } else {
                push(&mut eager, k);
            }
        }
        // A defer-only dep (never named eagerly) keeps its SOURCE position — its async
        // subgraph evaluates there. A dep imported both ways evaluates at its first EAGER
        // position (the defer clause never pulls it forward).
        let mut merged: Vec<String> = Vec::new();
        for stmt in body.iter() {
            let (spec, defers) = match stmt {
                Stmt::Import(decl) => (
                    dep_map_key(&decl.source, decl.attr_type.as_deref()),
                    !decl.specs.is_empty()
                        && decl
                            .specs
                            .iter()
                            .all(|s| matches!(s, crate::ast::ImportSpec::DeferNamespace(_))),
                ),
                Stmt::ExportNamed {
                    source: Some(src), ..
                }
                | Stmt::ExportAll { source: src, .. } => (src.to_string(), false),
                _ => continue,
            };
            if let Some(k) = resolved.get(&spec) {
                let is_defer_only = defer_seen.contains(k) && !eager.contains(k);
                let place = if is_defer_only { true } else { !defers };
                if place && !merged.iter().any(|x| x == k) {
                    merged.push(k.clone());
                }
            }
        }
        merged
    }

    /// InnerModuleEvaluation for a graph containing top-level await. Dependencies process in
    /// source order; a module executes once its [[PendingAsyncDependencies]] hits zero — in a
    /// coroutine for a TLA body (each await interleaves with the job queue), synchronously
    /// otherwise. Completion cascades through [[AsyncParentModules]] in [[AsyncEvaluationOrder]].
    fn inner_module_evaluation_async(
        &mut self,
        key: &str,
        stack: &mut Vec<String>,
        index: &mut usize,
    ) {
        if self.module_recs[key].started || self.module_recs[key].evaluated {
            return;
        }
        {
            let rec = self.module_recs.get_mut(key).unwrap();
            rec.started = true;
            rec.dfs_index = Some(*index);
            rec.dfs_anc = *index;
            rec.on_stack = true;
        }
        *index += 1;
        stack.push(key.to_string());

        let dep_keys = self.module_recs[key].dep_keys.clone();
        let deferred = self.module_recs[key].deferred_deps.clone();
        let body_order = self.eager_dep_order(key);
        let order: Vec<String> = if body_order.is_empty() {
            dep_keys
                .iter()
                .filter(|d| !deferred.contains(*d))
                .cloned()
                .collect()
        } else {
            body_order
                .into_iter()
                .filter(|d| dep_keys.contains(d))
                .collect()
        };
        for dep in &order {
            if deferred.contains(dep) {
                self.defer_async_deps(key, dep, stack, index);
                continue;
            }
            self.inner_module_evaluation_async(dep, stack, index);
            if self.module_recs[dep.as_str()].on_stack {
                // Still on the DFS stack: same strongly-connected component. An in-cycle
                // dependency that already parked at a top-level await still counts as pending.
                let danc = self.module_recs[dep.as_str()].dfs_anc;
                {
                    let rec = self.module_recs.get_mut(key).unwrap();
                    rec.dfs_anc = rec.dfs_anc.min(danc);
                }
                let dep_parked = {
                    let d = &self.module_recs[dep.as_str()];
                    d.async_order.is_some() && !d.evaluated
                };
                if dep_parked {
                    self.module_recs.get_mut(key).unwrap().pending_async += 1;
                    self.module_recs
                        .get_mut(dep.as_str())
                        .unwrap()
                        .async_parents
                        .push(key.to_string());
                }
                continue;
            }
            // Completed (or async-parked) dependency: waiting attaches to its CYCLE ROOT.
            let root = self.module_recs[dep.as_str()]
                .cycle_root
                .clone()
                .unwrap_or_else(|| dep.to_string());
            let (dep_err, dep_done, dep_async) = {
                let d = &self.module_recs[&root];
                (d.eval_error.clone(), d.evaluated, d.async_order.is_some())
            };
            if let Some(err) = dep_err {
                self.finish_scc(key, stack);
                self.async_module_rejected(key, err);
                return;
            }
            if !dep_done && dep_async {
                self.module_recs.get_mut(key).unwrap().pending_async += 1;
                self.module_recs
                    .get_mut(&root)
                    .unwrap()
                    .async_parents
                    .push(key.to_string());
            }
        }
        for dep in &dep_keys {
            if deferred.contains(dep) && !order.contains(dep) {
                self.defer_async_deps(key, dep, stack, index);
            }
        }
        let has_tla = body_has_tla(&self.module_recs[key].body);
        if self.module_recs[key].pending_async > 0 || has_tla {
            self.module_async_seq += 1;
            self.module_recs.get_mut(key).unwrap().async_order = Some(self.module_async_seq);
        }
        if self.module_recs[key].pending_async == 0 {
            self.module_execute_async(key);
        }
        self.finish_scc(key, stack);
    }

    /// `import defer`: the deferred module's own evaluation waits for a namespace access, but
    /// every module in its subgraph with top-level await evaluates NOW — and the deferring
    /// parent pends on each (so the later synchronous evaluation meets no pending async dep).
    fn defer_async_deps(
        &mut self,
        parent: &str,
        dep: &str,
        stack: &mut Vec<String>,
        index: &mut usize,
    ) {
        let mut graph = Vec::new();
        self.collect_graph(dep, &mut graph);
        for m in graph {
            if !body_has_tla(&self.module_recs[&m].body) {
                continue;
            }
            if !self.module_recs[&m].evaluated && self.module_recs[&m].top_promise.is_none() {
                let p = self.new_evaluation_promise();
                self.module_recs.get_mut(&m).unwrap().top_promise = Some(p);
            }
            self.inner_module_evaluation_async(&m, stack, index);
            if self.module_recs[&m].on_stack {
                continue;
            }
            let root = self.module_recs[&m]
                .cycle_root
                .clone()
                .unwrap_or_else(|| m.clone());
            let (dep_err, dep_done, dep_async) = {
                let d = &self.module_recs[&root];
                (d.eval_error.clone(), d.evaluated, d.async_order.is_some())
            };
            if dep_err.is_some() || dep_done || !dep_async {
                continue;
            }
            let already = self.module_recs[&root]
                .async_parents
                .iter()
                .any(|p| p == parent);
            if !already {
                self.module_recs.get_mut(parent).unwrap().pending_async += 1;
                self.module_recs
                    .get_mut(&root)
                    .unwrap()
                    .async_parents
                    .push(parent.to_string());
            }
        }
    }

    /// If `key` is its component's root, pop the SCC off the DFS stack and stamp each member's
    /// [[CycleRoot]].
    fn finish_scc(&mut self, key: &str, stack: &mut Vec<String>) {
        let (di, danc, on_stack) = {
            let r = &self.module_recs[key];
            (r.dfs_index, r.dfs_anc, r.on_stack)
        };
        if !on_stack {
            return;
        }
        let Some(di) = di else { return };
        if danc != di {
            return;
        }
        while let Some(m) = stack.pop() {
            let rec = self.module_recs.get_mut(&m).unwrap();
            rec.on_stack = false;
            rec.cycle_root = Some(key.to_string());
            if m == key {
                break;
            }
        }
    }

    /// Execute `key`'s own body (all dependencies done): a TLA body parks in a coroutine whose
    /// completion runs the ancestor cascade; a synchronous body runs now.
    fn module_execute_async(&mut self, key: &str) {
        let top = self.module_recs[key].top_promise.clone();
        let (body, env, meta, source) = {
            let rec = &self.module_recs[key];
            (
                rec.body.clone(),
                rec.env.clone(),
                rec.meta.clone(),
                rec.source.clone(),
            )
        };
        if body_has_tla(&body) {
            self.module_recs.get_mut(key).unwrap().evaluating = true;
            let module_key = key.to_string();
            let top = match top {
                Some(t) => t,
                None => self.new_evaluation_promise(),
            };
            // ExecuteModule receives the environment created during linking. Imports remain free
            // name operations so their live redirects are followed; all other own bindings map
            // directly to the pre-instantiated cells (ECMA-262 §16.2.1.7.3.2).
            let bindings: Vec<(String, bool)> = env
                .borrow()
                .vars
                .iter()
                .filter(|(name, binding)| {
                    binding.import_ref.is_none()
                        && name.as_ref() != "this"
                        && name.as_ref() != "%importmeta%"
                })
                .map(|(name, binding)| (name.to_string(), !binding.mutable))
                .collect();
            let coro = if let Some(chunk) = crate::bytecode::compile_module(&body, &bindings) {
                let mut vm = crate::bytecode::VmCoro::new(
                    self,
                    chunk,
                    env.clone(),
                    Value::Undefined,
                    &[],
                    &[],
                );
                vm.frame = Some(Interp::resumed_module_frame(source.clone()));
                crate::coroutine::Coroutine::Module(Box::new(ModuleCoro::new(vm, module_key)))
            } else {
                // Compiler coverage is a prerequisite for AsyncBlock execution. Contain a future
                // regression as a rejected module capability; never recreate a native-stack
                // execution path for source text.
                if std::env::var_os("TRUST_JS_TIER_LOG").is_some() {
                    eprintln!("[tier] unavailable VM module coroutine: {key}");
                }
                let error = self.make_error("Error", crate::coroutine::VM_UNAVAILABLE_MSG);
                self.finish_dynamic_module(key, Some(error.clone()));
                self.reject_promise(&top, error);
                return;
            };
            if let Value::Obj(o) = &top {
                self.generators
                    .insert(&self.gc_heap, crate::value::Gc::as_ptr(o) as usize, coro);
            }
            let top_key = match &top {
                Value::Obj(o) => crate::value::Gc::as_ptr(o) as usize,
                _ => return,
            };
            self.drive_async(
                top_key,
                top,
                crate::coroutine::Resume::Next(Value::Undefined),
            );
            return;
        }
        let saved_meta = self.import_meta.take();
        let saved_strict = self.strict;
        self.import_meta = Some(meta);
        self.strict = true;
        self.module_recs.get_mut(key).unwrap().evaluating = true;
        let result = self.in_code(Some(source), |this| {
            this.with_script_entry(|this| this.run_stmt_list(&body, &env))
        });
        self.import_meta = saved_meta;
        self.strict = saved_strict;
        self.module_recs.get_mut(key).unwrap().evaluating = false;
        match result {
            Ok(_) => {
                self.module_recs.get_mut(key).unwrap().evaluated = true;
                if let Some(t) = self.module_recs[key].top_promise.clone() {
                    self.resolve_promise(&t, Value::Undefined);
                }
            }
            Err(a) => {
                let v = crate::interpreter::abrupt_value(a);
                self.async_module_rejected(key, v);
            }
        }
    }

    /// AsyncModuleExecutionFulfilled: run every ancestor whose pending count reaches zero, in
    /// [[AsyncEvaluationOrder]] (ascending).
    pub(crate) fn async_module_fulfilled(&mut self, key: &str) {
        // Spec step 7: the fulfilled module's own capability resolves before any ancestor
        // executes — leaf-to-root fulfilment order is observable through dynamic import.
        if let Some(t) = self.module_recs[key].top_promise.clone() {
            self.resolve_promise(&t, Value::Undefined);
        }
        let mut exec: Vec<(u64, String)> = Vec::new();
        self.gather_available_ancestors(key, &mut exec);
        exec.sort_by_key(|(o, _)| *o);
        for (_, m) in exec {
            if self.module_recs[&m].evaluated || self.module_recs[&m].eval_error.is_some() {
                continue;
            }
            self.module_execute_async(&m);
        }
    }

    fn gather_available_ancestors(&mut self, key: &str, out: &mut Vec<(u64, String)>) {
        let parents = self.module_recs[key].async_parents.clone();
        for parent in parents {
            let (done, err, pending) = {
                let p = &self.module_recs[&parent];
                (p.evaluated, p.eval_error.is_some(), p.pending_async)
            };
            if done || err || pending == 0 {
                continue;
            }
            let p = self.module_recs.get_mut(&parent).unwrap();
            p.pending_async -= 1;
            if p.pending_async == 0 {
                let order = p.async_order.unwrap_or(u64::MAX);
                out.push((order, parent.clone()));
                // A synchronous ancestor completes as part of this cascade, so its own parents
                // become available too (GatherAvailableAncestors' recursion).
                if !body_has_tla(&self.module_recs[&parent].body) {
                    self.gather_available_ancestors(&parent, out);
                }
            }
        }
    }

    /// AsyncModuleExecutionRejected: the error propagates to every waiting ancestor.
    pub(crate) fn async_module_rejected(&mut self, key: &str, err: Value) {
        {
            let rec = self.module_recs.get_mut(key).unwrap();
            if rec.evaluated || rec.eval_error.is_some() {
                return;
            }
            rec.eval_error = Some(err.clone());
            rec.evaluated = true;
            rec.evaluating = false;
        }
        if let Some(t) = self.module_recs[key].top_promise.clone() {
            self.reject_promise(&t, err.clone());
        }
        for parent in self.module_recs[key].async_parents.clone() {
            self.async_module_rejected(&parent, err.clone());
        }
    }

    /// Evaluate every module with top-level await (plus its own dependencies) in `key`'s graph —
    /// the eager part of an `import defer`.
    fn evaluate_async_subgraph(&mut self, key: &str, seen: &mut Vec<String>) -> Result<(), Abrupt> {
        if seen.iter().any(|k| k == key) {
            return Ok(());
        }
        seen.push(key.to_string());
        let rec = match self.module_recs.get(key) {
            Some(r) => r,
            None => return Ok(()),
        };
        if rec.evaluated || rec.evaluating {
            return Ok(());
        }
        if body_has_tla(&rec.body) {
            // The async module (and its own graph) evaluates through the batch machinery — the
            // deferring parent does NOT wait on it.
            let mut graph = Vec::new();
            self.collect_eager_graph(key, &mut graph);
            for m in &graph {
                if !self.module_recs[m].evaluated && self.module_recs[m].top_promise.is_none() {
                    let p = self.new_evaluation_promise();
                    self.module_recs.get_mut(m).unwrap().top_promise = Some(p);
                }
            }
            self.inner_module_evaluation_async(key, &mut Vec::new(), &mut 0);
            return Ok(());
        }
        let deps = rec.dep_keys.clone();
        let deferred = rec.deferred_deps.clone();
        for dep in deps {
            if deferred.contains(&dep) {
                continue;
            }
            self.evaluate_async_subgraph(&dep, seen)?;
        }
        Ok(())
    }

    // --- Namespace exotic-object behaviour ----------------------------------------------------

    /// Whether `ptr` (an object pointer) is a module namespace exotic object. Asked on [[Get]]
    /// slow paths for every object, so an empty table answers without a probe.
    pub(crate) fn is_namespace(&self, ptr: usize) -> bool {
        !self.module_ns.is_empty() && self.module_ns.contains_key(&ptr)
    }

    /// A module namespace's `[[GetOwnProperty]]` for a string key: `Some(property)` with the export's
    /// *current* value (writable, enumerable, non-configurable) if `key` is an exported name, or
    /// `Some(Err(ReferenceError))` if the underlying binding is still uninitialized. `None` means the
    /// key is not an export (the caller falls back to ordinary lookup — e.g. `@@toStringTag`).
    pub(crate) fn namespace_own_property(
        &mut self,
        ptr: usize,
        key: &str,
    ) -> Option<Result<Property, Abrupt>> {
        let binding = self.module_ns.get(&ptr)?.get(key)?.clone();
        let value = match binding {
            NsBinding::Live(env, local) => match self.get_var(&local, &env) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            },
            NsBinding::Static(v) => v,
        };
        Some(Ok(Property::data(value, true, true, false)))
    }

    /// `import(specifier)`: ask an asynchronous embedder first, then retain the legacy synchronous
    /// loader as a fallback for filesystem/test hosts. In either case the language operation
    /// returns its promise immediately; a browser host finishes it from a later networking task.
    pub(crate) fn dynamic_import(
        &mut self,
        specifier: &str,
        attr_type: Option<&str>,
        defer: bool,
        referrer: Option<Value>,
    ) -> Value {
        let promise = self.new_promise();
        // EvaluateImportCall: the referrer is GetActiveScriptOrModule(), else the current Realm
        // (whose host-defined base is the host's import base). Module code finds its module
        // lexically: the `import.meta` captured at the call site survives `await`. Other code
        // finds the active Script or Module by its execution contexts, including the script
        // execution context a browser host pushes for a job callback (HTML HostCallJobCallback).
        let meta = match referrer {
            Some(meta) => Some(meta),
            None => match self.active_script_referrer() {
                Some(active) => {
                    return self.dynamic_import_from(promise, specifier, attr_type, defer, &active)
                }
                None => self.import_meta.clone(),
            },
        };
        let referrer = match meta {
            Some(m) => match self.get_member_key(&m, "url") {
                Ok(Value::Str(s)) => s.to_string(),
                _ => self.import_base.clone(),
            },
            None => self.import_base.clone(),
        };
        self.dynamic_import_from(promise, specifier, attr_type, defer, &referrer)
    }

    /// HostLoadImportedModule for `import()` from `referrer` (a module URL or a script's base),
    /// settling `promise`.
    fn dynamic_import_from(
        &mut self,
        promise: Value,
        specifier: &str,
        attr_type: Option<&str>,
        defer: bool,
        referrer: &str,
    ) -> Value {
        let host_context = self.host_job_context;
        if let Some(loader) = self.dynamic_module_loader.clone() {
            let request_id = self.next_dynamic_import_id;
            self.next_dynamic_import_id = self.next_dynamic_import_id.wrapping_add(1).max(1);
            if loader(request_id, host_context, specifier, referrer, attr_type) {
                self.pending_dynamic_imports.insert(
                    request_id,
                    PendingDynamicImport {
                        promise: promise.clone(),
                        specifier: specifier.to_string(),
                        attr_type: attr_type.map(str::to_string),
                        defer,
                        host_context,
                    },
                );
                return promise;
            }
        }
        let result = (|| {
            let scoped_referrer = module_map_key(host_context, referrer);
            let (canon, src) = self.fetch_module(specifier, &scoped_referrer, attr_type)?;
            let canon = match attr_type {
                Some(t @ ("json" | "text" | "bytes")) => format!("{canon}#{t}"),
                _ => canon,
            };
            let src = typed_module_source(src, attr_type);
            self.load_requested_modules(&canon, Some(src))?;
            self.link_module(&canon)?;
            Ok(canon)
        })();
        self.finish_dynamic_import(promise.clone(), attr_type, defer, result);
        promise
    }

    /// Finish one asynchronous HostLoadImportedModule operation. Returning false means the id was
    /// unknown or already completed; otherwise the associated import promise is resolved/rejected
    /// using the same link/evaluate path as the synchronous loader.
    pub(crate) fn finish_dynamic_module_load(
        &mut self,
        request_id: u64,
        result: Option<(String, String)>,
    ) -> bool {
        let Some(pending) = self.pending_dynamic_imports.remove(&request_id) else {
            return false;
        };
        let result = match result {
            Some((canon, source)) => {
                let canon = module_map_key(pending.host_context, &canon);
                let canon = match pending.attr_type.as_deref() {
                    Some(t @ ("json" | "text" | "bytes")) => format!("{canon}#{t}"),
                    _ => canon,
                };
                let source = typed_module_source(source, pending.attr_type.as_deref());
                self.load_requested_modules(&canon, Some(source))
                    .and_then(|()| self.link_module(&canon))
                    .map(|()| canon)
            }
            None => Err(self.throw(
                "TypeError",
                format!("module not found: {}", pending.specifier),
            )),
        };
        self.finish_dynamic_import(
            pending.promise,
            pending.attr_type.as_deref(),
            pending.defer,
            result,
        );
        true
    }

    fn finish_dynamic_import(
        &mut self,
        promise: Value,
        _attr_type: Option<&str>,
        defer: bool,
        result: Result<String, Abrupt>,
    ) {
        match result {
            Ok(canon) if defer => {
                // import.defer: link only; resolve with the (shared) deferred namespace. The
                // async subgraph still evaluates eagerly.
                let r = self.evaluate_async_subgraph(&canon, &mut Vec::new());
                match r {
                    Ok(()) => {
                        let dns = self.make_deferred_ns(&canon);
                        self.resolve_promise(&promise, dns);
                    }
                    Err(e) => {
                        let reason = crate::interpreter::abrupt_value(e);
                        self.reject_promise(&promise, reason);
                    }
                }
            }
            Ok(canon) => {
                // Evaluation may suspend at a top-level await: chain the import promise onto the
                // module's evaluation promise, resolving with the namespace.
                let top = self.evaluate_module_dynamic(&canon);
                let ns = self.module_recs[&canon].ns.clone();
                let on_f =
                    make_bound_len(self, dynamic_import_fulfil, vec![promise.clone(), ns], 1.0);
                let on_r = make_bound_len(self, dynamic_import_reject, vec![promise.clone()], 1.0);
                self.promise_then(&top, on_f, on_r);
            }
            Err(e) => {
                let reason = crate::interpreter::abrupt_value(e);
                self.reject_promise(&promise, reason);
            }
        }
    }

    /// Evaluate a dynamically-imported module, returning a promise that settles when its body
    /// (which may use top-level await) completes. Dependencies evaluate synchronously first; the
    /// module's own body runs in a coroutine so a top-level await parks it.
    fn evaluate_module_dynamic(&mut self, key: &str) -> Value {
        // ECMA-262 §16.2.1.6.1.3 Evaluate: once a cyclic module is evaluating-async or
        // evaluated, evaluation is redirected to its [[CycleRoot]] before consulting
        // [[TopLevelCapability]] or [[EvaluationError]]. A fulfilled member of a subsequently
        // errored async cycle must therefore observe the root's recorded error, not its own
        // already-fulfilled per-module promise.
        let evaluation_key = {
            let rec = &self.module_recs[key];
            if (rec.started || rec.evaluated || rec.evaluating) && rec.cycle_root.is_some() {
                rec.cycle_root.clone().unwrap()
            } else {
                key.to_string()
            }
        };
        let key = evaluation_key.as_str();
        let (top, err, settled) = {
            let rec = &self.module_recs[key];
            (
                rec.top_promise.clone(),
                rec.eval_error.clone(),
                rec.evaluated || rec.evaluating,
            )
        };
        // An existing top promise means the module belongs to an in-flight batch (or already
        // finished): the batch's DFS will execute it in order, so wait rather than preempt.
        if let Some(top) = top {
            return top;
        }
        if let Some(err) = err {
            let p = self.new_evaluation_promise();
            self.reject_promise(&p, err);
            return p;
        }
        if settled {
            let p = self.new_evaluation_promise();
            self.resolve_promise(&p, Value::Undefined);
            return p;
        }
        // Same batch machinery as a top-level Evaluate(): pre-create the graph's promises and
        // run InnerModuleEvaluation (awaits park; siblings continue; ancestors cascade).
        let mut graph = Vec::new();
        self.collect_eager_graph(key, &mut graph);
        for m in &graph {
            if !self.module_recs[m].evaluated && self.module_recs[m].top_promise.is_none() {
                let p = self.new_evaluation_promise();
                self.module_recs.get_mut(m).unwrap().top_promise = Some(p);
            }
        }
        self.inner_module_evaluation_async(key, &mut Vec::new(), &mut 0);
        self.module_recs[key]
            .top_promise
            .clone()
            .unwrap_or_else(|| {
                let p = self.new_evaluation_promise();
                match self.module_recs[key].eval_error.clone() {
                    Some(e) => self.reject_promise(&p, e),
                    None => self.resolve_promise(&p, Value::Undefined),
                }
                p
            })
    }

    /// Record a dynamically-evaluated module's completion (run from inside its coroutine).
    fn finish_dynamic_module(&mut self, key: &str, error: Option<Value>) {
        match error {
            None => {
                if let Some(rec) = self.module_recs.get_mut(key) {
                    rec.evaluated = true;
                    rec.evaluating = false;
                }
                // The module's own promise settles first (the coroutine driver resolves it when
                // this returns), then ancestors execute in [[AsyncEvaluationOrder]].
                self.async_module_fulfilled(key);
            }
            Some(err) => {
                if let Some(rec) = self.module_recs.get_mut(key) {
                    rec.evaluated = false; // async_module_rejected records the error
                }
                self.async_module_rejected(key, err);
            }
        }
    }
}

/// Whether two export resolutions denote the same binding (so a name re-exported by two star paths
/// is not ambiguous).
fn same_binding(a: &Resolution, b: &Resolution) -> bool {
    match (a, b) {
        (Resolution::Local(e1, l1), Resolution::Local(e2, l2)) => Env::ptr_eq(e1, e2) && l1 == l2,
        (Resolution::Ns(Value::Obj(o1)), Resolution::Ns(Value::Obj(o2))) => {
            crate::value::Gc::ptr_eq(o1, o2)
        }
        (Resolution::DeferNs(d1), Resolution::DeferNs(d2)) => d1 == d2,
        _ => false,
    }
}

/// The `resolved`-map key for a dependency: the plain specifier, or — for an attribute import
/// (`with { type: ... }`) — the specifier qualified by the type, so the same file imported both
/// ways in one module resolves to two distinct records. NUL never appears in real specifiers.
pub(crate) fn dep_map_key(spec: &str, attr: Option<&str>) -> String {
    match attr {
        Some(t @ ("json" | "text" | "bytes")) => format!("{spec}\u{0000}{t}"),
        _ => spec.to_string(),
    }
}

/// Every module specifier this body imports/re-exports from (with duplicates).
fn module_dependency_specifiers(body: &[Stmt]) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    for stmt in body {
        match stmt {
            Stmt::Import(decl) => out.push((decl.source.to_string(), decl.attr_type.clone())),
            Stmt::ExportNamed {
                source: Some(src), ..
            }
            | Stmt::ExportAll { source: src, .. } => out.push((src.to_string(), None)),
            _ => {}
        }
    }
    out
}

/// Reaction for a dynamic import: the module evaluated — resolve the import promise (`args[0]`)
/// with the namespace (`args[1]`).
fn dynamic_import_fulfil(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    let p = args.first().cloned().unwrap_or(Value::Undefined);
    let ns = args.get(1).cloned().unwrap_or(Value::Undefined);
    i.resolve_promise(&p, ns);
    Ok(Value::Undefined)
}

/// Reaction for a dynamic import: evaluation failed — reject the import promise (`args[0]`) with
/// the reason (`args[1]`).
fn dynamic_import_reject(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    let p = args.first().cloned().unwrap_or(Value::Undefined);
    let reason = args.get(1).cloned().unwrap_or(Value::Undefined);
    i.reject_promise(&p, reason);
    Ok(Value::Undefined)
}

/// What a host loaded for one module request.
pub(crate) enum ModuleSource {
    /// ECMAScript source text: a Source Text Module Record.
    Source(String),
    /// The contents of a `with { type: ... }` import, which becomes a Synthetic Module Record.
    Synthetic(SyntheticContents),
}

/// The loaded contents of a synthetic module, by its `type` attribute.
pub(crate) enum SyntheticContents {
    /// `json`: JSON text, parsed by ParseJSONModule.
    Json(String),
    /// `text`: exported as a String.
    Text(String),
    /// `bytes`: the file decoded latin-1 style (one char per byte) by the host.
    Bytes(String),
}

/// The module a `with { type: ... }` import loads from `text`: a synthetic JSON, text or bytes
/// module, or ordinary source text for an unknown type.
pub(crate) fn typed_module_source(text: String, attr_type: Option<&str>) -> ModuleSource {
    ModuleSource::Synthetic(match attr_type {
        Some("json") => SyntheticContents::Json(text),
        Some("text") => SyntheticContents::Text(text),
        Some("bytes") => SyntheticContents::Bytes(text),
        _ => return ModuleSource::Source(text),
    })
}

impl Interp {
    /// The default export of a synthetic module, made from the current Realm's intrinsics
    /// without running author code (ECMA-262 CreateDefaultExportSyntheticModule): a page that
    /// replaces `JSON.parse` or `Uint8Array` cannot change what a JSON or bytes import produces.
    fn synthetic_default_export(&mut self, contents: SyntheticContents) -> Result<Value, Abrupt> {
        match contents {
            // ParseJSONModule: ParseJSON of the source text, which is host text.
            SyntheticContents::Json(text) => {
                crate::builtins::parse_json(self, &crate::jstr::from_text(&text))
                    .map_err(Abrupt::Throw)
            }
            SyntheticContents::Text(text) => Ok(Value::from_string(text)),
            SyntheticContents::Bytes(text) => {
                // The host decodes a binary file latin-1 style (char == byte); text holding a
                // wider character came from a UTF-8 decoding, which re-encodes to its bytes.
                let bytes = if text.chars().all(|c| (c as u32) < 0x100) {
                    text.chars().map(|c| c as u8).collect()
                } else {
                    text.into_bytes()
                };
                Ok(crate::builtins::immutable_uint8_array(self, bytes))
            }
        }
    }
}

/// Whether a module body contains top-level `await` (a for-await head, an `await using`, or an
/// Await expression outside any function/class body) — the async-module test for `import defer`.
pub(crate) fn body_has_tla(body: &[Stmt]) -> bool {
    fn stmt(s: &Stmt) -> bool {
        if crate::interpreter::ast_stack_low() {
            return crate::interpreter::with_ast_stack(|| stmt(s));
        }
        match s {
            Stmt::Expr(e) | Stmt::Throw(e) => expr(e),
            Stmt::Return(v) => v.as_ref().map(expr).unwrap_or(false),
            Stmt::VarDecl { kind, decls } => {
                matches!(kind, crate::ast::DeclKind::AwaitUsing)
                    || decls
                        .iter()
                        .any(|(_, init)| init.as_ref().map(expr).unwrap_or(false))
            }
            Stmt::If { test, cons, alt } => {
                expr(test) || stmt(cons) || alt.as_ref().map(|a| stmt(a)).unwrap_or(false)
            }
            Stmt::Block(b) => b.iter().any(stmt),
            Stmt::While { test, body, .. } | Stmt::DoWhile { body, test, .. } => {
                expr(test) || stmt(body)
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => {
                init.as_ref()
                    .map(|i| match i.as_ref() {
                        crate::ast::ForInit::VarDecl { decls, .. } => decls
                            .iter()
                            .any(|(_, e)| e.as_ref().map(expr).unwrap_or(false)),
                        crate::ast::ForInit::Expr(e) => expr(e),
                    })
                    .unwrap_or(false)
                    || test.as_ref().map(expr).unwrap_or(false)
                    || update.as_ref().map(expr).unwrap_or(false)
                    || stmt(body)
            }
            Stmt::ForInOf {
                right,
                body,
                is_await,
                ..
            } => *is_await || expr(right) || stmt(body),
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                block.iter().any(stmt)
                    || handler
                        .as_ref()
                        .map(|(_, h)| h.iter().any(stmt))
                        .unwrap_or(false)
                    || finalizer
                        .as_ref()
                        .map(|f| f.iter().any(stmt))
                        .unwrap_or(false)
            }
            Stmt::Switch { disc, cases } => {
                expr(disc)
                    || cases.iter().any(|c| {
                        c.test.as_ref().map(expr).unwrap_or(false) || c.body.iter().any(stmt)
                    })
            }
            Stmt::Labeled { body, .. } => stmt(body),
            Stmt::With { obj, body } => expr(obj) || stmt(body),
            Stmt::ExportDefault(inner) | Stmt::ExportDecl(inner) => stmt(inner),
            _ => false,
        }
    }
    fn expr(e: &Expr) -> bool {
        if crate::interpreter::ast_stack_low() {
            return crate::interpreter::with_ast_stack(|| expr(e));
        }
        match e {
            Expr::Await(_) => true,
            Expr::ToStr(x) | Expr::OptionalChain(x) => expr(x),
            Expr::Unary { arg, .. } | Expr::Update { arg, .. } => expr(arg),
            Expr::Binary { left, right, .. }
            | Expr::Logical { left, right, .. }
            | Expr::Assign {
                target: left,
                value: right,
                ..
            } => expr(left) || expr(right),
            Expr::Cond { test, cons, alt } => expr(test) || expr(cons) || expr(alt),
            Expr::Call { callee, args, .. } => {
                expr(callee)
                    || args.iter().any(|a| match a {
                        crate::ast::ArrayElem::Item(e) | crate::ast::ArrayElem::Spread(e) => {
                            expr(e)
                        }
                        _ => false,
                    })
            }
            Expr::New { callee, args, .. } => {
                expr(callee)
                    || args.iter().any(|a| match a {
                        crate::ast::ArrayElem::Item(e) | crate::ast::ArrayElem::Spread(e) => {
                            expr(e)
                        }
                        _ => false,
                    })
            }
            Expr::Member { obj, .. } => expr(obj),
            Expr::Index { obj, index, .. } => expr(obj) || expr(index),
            Expr::Seq(v) => v.iter().any(expr),
            Expr::Array(elems) => elems.iter().any(|a| match a {
                crate::ast::ArrayElem::Item(e) | crate::ast::ArrayElem::Spread(e) => expr(e),
                _ => false,
            }),
            Expr::Object(props) => props.iter().any(|p| match p {
                crate::ast::PropDef::KeyValue { value, .. } => expr(value),
                crate::ast::PropDef::Spread(e) => expr(e),
                _ => false,
            }),
            Expr::Yield { arg, .. } => arg.as_ref().map(|a| expr(a)).unwrap_or(false),
            Expr::TaggedTemplate { tag, .. } => expr(tag),
            Expr::ImportCall { spec, options, .. } => {
                expr(spec) || options.as_ref().map(|o| expr(o)).unwrap_or(false)
            }
            _ => false,
        }
    }
    body.iter().any(stmt)
}

/// A module's parsed export/import tables.
struct ExportTables {
    local_exports: HashMap<String, String>,
    imports: HashMap<String, ImportOrigin>,
    indirect: HashMap<String, (String, String)>,
    star_as: HashMap<String, String>,
    stars: Vec<String>,
}

/// Build a module's export/import tables from its parsed body and resolved specifier→key map.
fn build_export_tables(body: &[Stmt], resolved: &HashMap<String, String>) -> ExportTables {
    let mut local_exports: HashMap<String, String> = HashMap::new();
    let mut imports: HashMap<String, ImportOrigin> = HashMap::new();
    let mut indirect: HashMap<String, (String, String)> = HashMap::new();
    let mut star_as: HashMap<String, String> = HashMap::new();
    let mut stars: Vec<String> = Vec::new();
    let key_of = |src: &str| {
        resolved
            .get(src)
            .cloned()
            .unwrap_or_else(|| src.to_string())
    };

    for stmt in body {
        match stmt {
            Stmt::Import(decl) => {
                let dep = key_of(&decl.source);
                for spec in &decl.specs {
                    match spec {
                        ImportSpec::Namespace(local) => {
                            imports.insert(local.clone(), ImportOrigin::Namespace(dep.clone()));
                        }
                        ImportSpec::DeferNamespace(local) => {
                            imports
                                .insert(local.clone(), ImportOrigin::DeferNamespace(dep.clone()));
                        }
                        ImportSpec::Source(local) => {
                            imports.insert(
                                local.clone(),
                                ImportOrigin::Named(dep.clone(), "~source~".to_string()),
                            );
                        }
                        ImportSpec::Default(local) => {
                            imports.insert(
                                local.clone(),
                                ImportOrigin::Named(dep.clone(), "default".to_string()),
                            );
                        }
                        ImportSpec::Named { imported, local } => {
                            imports.insert(
                                local.clone(),
                                ImportOrigin::Named(dep.clone(), imported.clone()),
                            );
                        }
                    }
                }
            }
            Stmt::ExportDecl(inner) => {
                for name in exported_decl_names(inner) {
                    local_exports.insert(name.clone(), name);
                }
            }
            Stmt::ExportDefault(inner) => {
                let local = match &**inner {
                    Stmt::FuncDecl(f) if f.name.is_some() => f.name.clone().unwrap(),
                    Stmt::ClassDecl(c) if c.name.is_some() => c.name.clone().unwrap(),
                    _ => "*default*".to_string(),
                };
                local_exports.insert("default".to_string(), local);
            }
            Stmt::ExportNamed { specs, source } => {
                for spec in specs {
                    match source {
                        Some(src) => {
                            indirect
                                .insert(spec.exported.clone(), (key_of(src), spec.local.clone()));
                        }
                        None => {
                            local_exports.insert(spec.exported.clone(), spec.local.clone());
                        }
                    }
                }
            }
            Stmt::ExportAll { source, exported } => match exported {
                Some(name) => {
                    star_as.insert(name.clone(), key_of(source));
                }
                None => stars.push(key_of(source)),
            },
            _ => {}
        }
    }
    ExportTables {
        local_exports,
        imports,
        indirect,
        star_as,
        stars,
    }
}

/// Names introduced by an `export <decl>` statement's inner declaration.
fn exported_decl_names(inner: &Stmt) -> Vec<String> {
    match inner {
        Stmt::VarDecl { decls, .. } => {
            let mut out = Vec::new();
            for (pat, _) in decls {
                crate::interpreter::pattern_idents(pat, &mut out);
            }
            out
        }
        Stmt::FuncDecl(f) => f.name.clone().into_iter().collect(),
        Stmt::ClassDecl(c) => c.name.clone().into_iter().collect(),
        _ => Vec::new(),
    }
}

impl crate::gc_trace::GcTrace for ModuleCoro {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self { body, key: _ } = self;
        crate::gc_trace::GcTrace::trace_gc(body, sink);
    }
}

impl crate::gc_trace::GcTrace for ModuleRec {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self {
            body: _,
            source: _,
            env,
            ns,
            meta,
            dep_keys: _,
            resolved: _,
            local_exports: _,
            imports: _,
            indirect: _,
            star_as: _,
            stars: _,
            linked: _,
            evaluated: _,
            evaluating: _,
            eval_error,
            deferred_deps: _,
            top_promise,
            started: _,
            pending_async: _,
            async_order: _,
            async_parents: _,
            dfs_index: _,
            dfs_anc: _,
            on_stack: _,
            cycle_root: _,
        } = self;
        env.trace_gc(sink);
        ns.trace_gc(sink);
        meta.trace_gc(sink);
        eval_error.trace_gc(sink);
        top_promise.trace_gc(sink);
    }
}

impl crate::gc_trace::GcTrace for PendingDynamicImport {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self {
            promise,
            specifier: _,
            attr_type: _,
            defer: _,
            host_context: _,
        } = self;
        crate::gc_trace::GcTrace::trace_gc(promise, sink);
    }
}
