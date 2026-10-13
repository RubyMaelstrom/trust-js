//! Engine-local reuse of source-owned loop plans. No live activation is cached.
//!
//! A ready entry owns one plan; an executing plan is represented only by a Weak handle.
//! Only its last activation can publish a newly charged quiescent plan. The byte quota is
//! known requested cache payload, including immutable retained source and key metadata, not
//! total heap size: independently growing shared callee code, allocator headers and opaque
//! HashMap bucket/control storage are reported separately by memory.rs.
use super::loop_fragment::{Context, Plan, RootKind};
use crate::ast::LoopSiteToken;
use crate::value::{Gc, Object};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

const ENTRIES: usize = 64;
const VARIANTS: usize = 4;
const LABEL_BYTES: usize = 16 * 1024;
const PLAN_BYTES: usize = 1024 * 1024;
const TOTAL_BYTES: usize = 4 * 1024 * 1024;

struct Key {
    site: Weak<LoopSiteToken>,
    realm: Weak<RefCell<Object>>,
    kind: RootKind,
    context: Context,
    labels: Box<[Box<str>]>,
}

impl Key {
    fn matches(
        &self,
        site: &Rc<LoopSiteToken>,
        realm: &Gc,
        kind: RootKind,
        context: Context,
        labels: &[&str],
    ) -> bool {
        self.site.as_ptr() == Rc::as_ptr(site)
            && self.realm.as_ptr() == Rc::as_ptr(realm)
            && self.kind == kind
            && self.context == context
            && self.labels.iter().map(|s| &**s).eq(labels.iter().copied())
    }
    fn alive(&self) -> bool {
        self.site.strong_count() != 0 && self.realm.strong_count() != 0
    }
    fn bytes(&self) -> usize {
        self.labels.len() * std::mem::size_of::<Box<str>>()
            + self.labels.iter().map(|label| label.len()).sum::<usize>()
    }
}

enum State {
    Ready(Rc<Plan>, usize),
    Running(Weak<Plan>),
    Unsupported,
}

struct Entry {
    key: Key,
    state: State,
}

struct Storage {
    entries: [Option<Entry>; ENTRIES],
    next: usize,
}

#[derive(Default)]
pub(crate) struct Cache {
    storage: Option<Box<Storage>>,
}

pub(super) enum Lookup {
    Miss,
    Unsupported,
    Plan(Rc<Plan>),
}

impl Cache {
    fn prune(&mut self) {
        if let Some(storage) = &mut self.storage {
            for entry in &mut storage.entries {
                if entry.as_ref().is_some_and(|entry| {
                    !entry.key.alive()
                        || matches!(&entry.state, State::Running(plan) if plan.strong_count() == 0)
                }) {
                    *entry = None;
                }
            }
        }
    }

    pub(super) fn lookup(
        &mut self,
        site: &Rc<LoopSiteToken>,
        realm: &Gc,
        kind: RootKind,
        context: Context,
        labels: &[&str],
    ) -> Lookup {
        self.prune();
        let Some(storage) = &mut self.storage else {
            return Lookup::Miss;
        };
        let Some(entry) = storage
            .entries
            .iter_mut()
            .flatten()
            .find(|entry| entry.key.matches(site, realm, kind, context, labels))
        else {
            return Lookup::Miss;
        };
        let plan = match &entry.state {
            State::Ready(plan, _) => plan.clone(),
            State::Running(plan) => {
                let Some(plan) = plan.upgrade() else {
                    return Lookup::Miss;
                };
                plan
            }
            State::Unsupported => return Lookup::Unsupported,
        };
        // Remove cache ownership BEFORE author code can grow the plan's runtime metadata.
        entry.state = State::Running(Rc::downgrade(&plan));
        Lookup::Plan(plan)
    }

    #[allow(clippy::too_many_arguments)]
    fn store(
        &mut self,
        site: &Rc<LoopSiteToken>,
        realm: &Gc,
        kind: RootKind,
        context: Context,
        labels: &[&str],
        state: State,
    ) {
        let label_bytes = labels.iter().try_fold(0usize, |sum, label| {
            sum.checked_add(label.len())?
                .checked_add(std::mem::size_of::<Box<str>>())
        });
        if !label_bytes.is_some_and(|bytes| bytes <= LABEL_BYTES) {
            return;
        }
        self.prune();
        let storage = self.storage.get_or_insert_with(|| {
            Box::new(Storage {
                entries: std::array::from_fn(|_| None),
                next: 0,
            })
        });
        if let Some(entry) = storage
            .entries
            .iter_mut()
            .flatten()
            .find(|entry| entry.key.matches(site, realm, kind, context, labels))
        {
            entry.state = state;
            Self::enforce_quota(storage);
            return;
        }
        // Bound variants (including negative and in-flight entries), not just distinct sites.
        let variants = storage
            .entries
            .iter()
            .flatten()
            .filter(|entry| entry.key.site.as_ptr() == Rc::as_ptr(site))
            .count();
        let index = if variants >= VARIANTS {
            (0..ENTRIES)
                .map(|n| (storage.next + n) % ENTRIES)
                .find(|&n| {
                    storage.entries[n]
                        .as_ref()
                        .is_some_and(|entry| entry.key.site.as_ptr() == Rc::as_ptr(site))
                })
                .unwrap()
        } else {
            storage
                .entries
                .iter()
                .position(Option::is_none)
                .unwrap_or(storage.next)
        };
        storage.next = (index + 1) % ENTRIES;
        storage.entries[index] = Some(Entry {
            key: Key {
                site: Rc::downgrade(site),
                realm: Rc::downgrade(realm),
                kind,
                context,
                labels: labels
                    .iter()
                    .map(|label| Box::<str>::from(*label))
                    .collect(),
            },
            state,
        });
        Self::enforce_quota(storage);
    }

    pub(super) fn unsupported(
        &mut self,
        site: &Rc<LoopSiteToken>,
        realm: &Gc,
        kind: RootKind,
        context: Context,
        labels: &[&str],
    ) {
        self.store(site, realm, kind, context, labels, State::Unsupported);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn running(
        &mut self,
        site: &Rc<LoopSiteToken>,
        realm: &Gc,
        kind: RootKind,
        context: Context,
        labels: &[&str],
        plan: &Rc<Plan>,
    ) {
        self.store(
            site,
            realm,
            kind,
            context,
            labels,
            State::Running(Rc::downgrade(plan)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish(
        &mut self,
        site: &Rc<LoopSiteToken>,
        realm: &Gc,
        kind: RootKind,
        context: Context,
        labels: &[&str],
        plan: &Rc<Plan>,
    ) {
        let active = plan.active.get();
        debug_assert!(active > 0);
        plan.active.set(active - 1);
        if active != 1 {
            return;
        }
        // A reentrant invocation can evict this weak slot and install a newer plan for the
        // same key. An older last return must not replace that newer (possibly active) plan.
        if let Some(entry) = self.storage.as_ref().and_then(|storage| {
            storage
                .entries
                .iter()
                .flatten()
                .find(|entry| entry.key.matches(site, realm, kind, context, labels))
        }) {
            let owns_slot = match &entry.state {
                State::Ready(current, _) => Rc::ptr_eq(current, plan),
                State::Running(current) => current.as_ptr() == Rc::as_ptr(plan),
                State::Unsupported => false,
            };
            if !owns_slot {
                return;
            }
        }
        // Bounded plan-only walk, not a Realm/object-graph census. Reconcile after growth, not
        // at each hit. The active Chunk/its descendant functions remain ordinary GC roots.
        let bytes = crate::memory::fragment_plan_charge(&plan.chunk)
            .saturating_add(plan.source_bytes)
            .saturating_add(std::mem::size_of::<Plan>())
            .saturating_add(plan.seed_slots.capacity() * std::mem::size_of::<u16>());
        if bytes > PLAN_BYTES {
            if let Some(storage) = &mut self.storage {
                for entry in &mut storage.entries {
                    if entry.as_ref().is_some_and(|entry| {
                        entry.key.matches(site, realm, kind, context, labels)
                            && matches!(&entry.state, State::Running(current)
                            if current.as_ptr() == Rc::as_ptr(plan))
                    }) {
                        *entry = None;
                    }
                }
            }
            return;
        }
        self.store(
            site,
            realm,
            kind,
            context,
            labels,
            State::Ready(plan.clone(), bytes),
        );
    }

    fn enforce_quota(storage: &mut Storage) {
        // Even if every slot is negative or in flight, bounded key metadata alone fits. Only
        // Ready plans own code here; evicting one never invalidates an active activation.
        debug_assert!(
            std::mem::size_of::<Storage>()
                + ENTRIES * (LABEL_BYTES + std::mem::size_of::<LoopSiteToken>())
                <= TOTAL_BYTES
        );
        while Self::charged(storage) > TOTAL_BYTES {
            let index = (0..ENTRIES)
                .map(|n| (storage.next + n) % ENTRIES)
                .find(|&n| {
                    storage.entries[n]
                        .as_ref()
                        .is_some_and(|entry| matches!(entry.state, State::Ready(..)))
                });
            let Some(index) = index else {
                break;
            };
            storage.entries[index] = None;
            storage.next = (index + 1) % ENTRIES;
        }
    }

    fn charged(storage: &Storage) -> usize {
        // Storage includes every inline Weak/policy/state slot, occupied or not. Label slices
        // and strings are exact requested payload. Tokens are conservatively charged per key
        // (shared sites can be overcounted); private allocator/Rc headers remain excluded.
        std::mem::size_of::<Storage>()
            + storage
                .entries
                .iter()
                .flatten()
                .map(|entry| {
                    entry.key.bytes()
                        + std::mem::size_of::<LoopSiteToken>()
                        + match entry.state {
                            State::Ready(_, bytes) => bytes,
                            _ => 0,
                        }
                })
                .sum::<usize>()
    }

    pub(super) fn abandon(plan: &Plan) {
        debug_assert!(plan.active.get() > 0);
        plan.active.set(plan.active.get() - 1);
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        let Some(storage) = &self.storage else {
            return 0;
        };
        let mut bytes = std::mem::size_of::<Storage>();
        for entry in storage.entries.iter().flatten() {
            bytes += entry.key.bytes();
            if let Some(site) = entry.key.site.upgrade() {
                visitor.loop_site_token(&site);
            }
            let plan = match &entry.state {
                State::Ready(plan, _) => Some(plan.clone()),
                State::Running(plan) => plan.upgrade(),
                State::Unsupported => None,
            };
            if let Some(plan) = plan {
                // Full memory reporting includes the shared source/callee graph, independently
                // of the deliberately plan-only eviction charge above.
                bytes += std::mem::size_of::<Plan>()
                    + plan.seed_slots.capacity() * std::mem::size_of::<u16>();
                visitor.chunk(&plan.chunk);
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::LoopSite,
        bytecode::{loop_fragment, Tier},
        Completion, Engine,
    };
    use std::cell::Cell;

    fn plan(source: &str) -> Rc<Plan> {
        let body = crate::parser::parse_script(source, false)
            .ok()
            .expect("source");
        let chunk = crate::bytecode::compile_script(&body, false).expect("chunk");
        Rc::new(Plan {
            source_bytes: crate::memory::fragment_plan_source_charge(&chunk),
            chunk,
            completion_slot: 0,
            seed_slots: Vec::new(),
            active: Cell::new(0),
        })
    }

    fn publish(cache: &mut Cache, site: &LoopSite, engine: &Engine, plan: &Rc<Plan>) {
        let context = Context::current(&engine.interp);
        plan.active.set(1);
        cache.running(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            plan,
        );
        cache.finish(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            plan,
        );
    }

    #[test]
    fn source_sites_are_lazy_deep_clone_fresh_and_snapshot_invisible() {
        let source = "while(false){};do{}while(false);for(;;)break;for(var x of []){}";
        let body = crate::parser::parse_script(source, false)
            .ok()
            .expect("source");
        let encoded = crate::snapshot::encode(&body);
        for stmt in &body {
            if let Some(body) = stmt.loop_body() {
                let site = &body.site;
                assert!(site.retained_token().is_none());
                let clone = site.clone();
                assert!(!Rc::ptr_eq(site.token(), clone.token()));
            }
        }
        assert_eq!(crate::snapshot::encode(&body), encoded);
        let decoded = crate::snapshot::decode(&encoded).expect("snapshot");
        for stmt in body.iter().cloned().chain(decoded) {
            if let Some(body) = stmt.loop_body() {
                assert!(body.site.retained_token().is_none());
            }
        }
        assert_eq!(
            std::mem::size_of::<LoopSite>(),
            std::mem::size_of::<usize>()
        );
        assert!(Cache::default().storage.is_none());
    }

    #[test]
    fn cache_reentrant_leases_remove_strong_ownership_and_survive_eviction() {
        let engine = Engine::new();
        let site = LoopSite::default();
        let context = Context::current(&engine.interp);
        let mut cache = Cache::default();
        let original = plan("42");
        let weak = Rc::downgrade(&original);
        publish(&mut cache, &site, &engine, &original);
        drop(original);
        let Lookup::Plan(outer) = cache.lookup(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
        ) else {
            panic!("cached")
        };
        outer.active.set(1);
        assert_eq!(Rc::strong_count(&outer), 1, "running cache is weak");
        let Lookup::Plan(inner) = cache.lookup(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
        ) else {
            panic!("recursive hit")
        };
        assert!(Rc::ptr_eq(&outer, &inner));
        inner.active.set(2);
        cache.finish(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            &inner,
        );
        drop(inner);
        assert_eq!(
            Rc::strong_count(&outer),
            1,
            "only last activation can publish"
        );
        let sites: Vec<_> = (0..ENTRIES + 1).map(|_| LoopSite::default()).collect();
        for other in &sites {
            cache.unsupported(
                other.token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &[],
            );
        }
        assert_eq!(
            Rc::strong_count(&outer),
            1,
            "eviction cannot drop executing code"
        );
        cache.finish(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            &outer,
        );
        drop(outer);
        assert!(weak.upgrade().is_some());
        drop(site);
        cache.prune();
        assert!(
            weak.upgrade().is_none(),
            "expired syntax cannot retain a plan"
        );
    }

    #[test]
    fn cache_keys_bound_variants_and_preserve_realm_policy_labels() {
        let mut engine = Engine::new();
        let site = LoopSite::default();
        let other_realm = engine.interp.new_object();
        let context = Context::current(&engine.interp);
        let mut cache = Cache::default();
        cache.unsupported(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &["outer"],
        );
        assert!(matches!(
            cache.lookup(
                site.token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &["outer"]
            ),
            Lookup::Unsupported
        ));
        assert!(matches!(
            cache.lookup(
                site.token(),
                &other_realm,
                RootKind::While,
                context,
                &["outer"]
            ),
            Lookup::Miss
        ));
        assert!(matches!(
            cache.lookup(
                site.token(),
                &engine.interp.global,
                RootKind::For,
                context,
                &["outer"]
            ),
            Lookup::Miss
        ));
        assert!(matches!(
            cache.lookup(
                site.token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &["other"]
            ),
            Lookup::Miss
        ));
        engine.interp.strict = !engine.interp.strict;
        assert!(matches!(
            cache.lookup(
                site.token(),
                &engine.interp.global,
                RootKind::While,
                Context::current(&engine.interp),
                &["outer"]
            ),
            Lookup::Miss
        ));
        for n in 0..100 {
            cache.unsupported(
                site.token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &[&n.to_string()],
            );
        }
        assert_eq!(
            cache
                .storage
                .as_ref()
                .unwrap()
                .entries
                .iter()
                .flatten()
                .count(),
            VARIANTS
        );
        assert_eq!(
            Cache::charged(cache.storage.as_ref().unwrap()),
            std::mem::size_of::<Storage>()
                + VARIANTS
                    * (std::mem::size_of::<Box<str>>() + 2 + std::mem::size_of::<LoopSiteToken>())
        );
        let mut oversized = Cache::default();
        oversized.unsupported(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[&"x".repeat(LABEL_BYTES)],
        );
        assert!(oversized.storage.is_none());
    }

    #[test]
    fn cache_charges_only_quiescent_plans_and_rejects_oversized_growth() {
        let engine = Engine::new();
        let site = LoopSite::default();
        let mut cache = Cache::default();
        let oversized = plan(&format!("'{}'", "x".repeat(PLAN_BYTES + 1)));
        publish(&mut cache, &site, &engine, &oversized);
        assert_eq!(
            Cache::charged(cache.storage.as_ref().unwrap()),
            std::mem::size_of::<Storage>()
        );
        drop(oversized);
        cache.prune();
        assert!(cache
            .storage
            .as_ref()
            .unwrap()
            .entries
            .iter()
            .all(Option::is_none));
        let sites: Vec<_> = (0..20).map(|_| LoopSite::default()).collect();
        for site in &sites {
            let plan = plan(&format!("'{}'", "y".repeat(300_000)));
            publish(&mut cache, site, &engine, &plan);
            assert!(Cache::charged(cache.storage.as_ref().unwrap()) <= TOTAL_BYTES);
        }
        assert!(
            cache
                .storage
                .as_ref()
                .unwrap()
                .entries
                .iter()
                .flatten()
                .count()
                < sites.len()
        );
    }

    #[test]
    fn every_cache_store_bounds_ready_plans_plus_negative_and_running_metadata() {
        let engine = Engine::new();
        let context = Context::current(&engine.interp);
        let sites: Vec<_> = (0..ENTRIES).map(|_| LoopSite::default()).collect();
        let retained = plan("42");
        let label = "x".repeat(LABEL_BYTES - std::mem::size_of::<Box<str>>());
        let plan_charge = (TOTAL_BYTES
            - std::mem::size_of::<Storage>()
            - 4 * std::mem::size_of::<LoopSiteToken>()
            - 128)
            / 4;
        assert!(plan_charge < PLAN_BYTES);
        // Model a quiescent cache within 132 bytes of its limit. Either kind of key-only
        // insertion must evict a Ready plan even though neither insertion calls finish().
        for running in [false, true] {
            let mut cache = Cache::default();
            for site in &sites[..4] {
                cache.store(
                    site.token(),
                    &engine.interp.global,
                    RootKind::While,
                    context,
                    &[],
                    State::Ready(retained.clone(), plan_charge),
                );
            }
            let before = Cache::charged(cache.storage.as_ref().unwrap());
            assert!(before <= TOTAL_BYTES && before > TOTAL_BYTES - 132);
            let state = if running {
                State::Running(Rc::downgrade(&retained))
            } else {
                State::Unsupported
            };
            cache.store(
                sites[4].token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &[&label],
                state,
            );
            let storage = cache.storage.as_ref().unwrap();
            assert!(Cache::charged(storage) <= TOTAL_BYTES);
            assert_eq!(
                storage
                    .entries
                    .iter()
                    .flatten()
                    .filter(|entry| matches!(entry.state, State::Ready(..)))
                    .count(),
                3
            );
            assert!(storage
                .entries
                .iter()
                .flatten()
                .any(|entry| entry.key.matches(
                    sites[4].token(),
                    &engine.interp.global,
                    RootKind::While,
                    context,
                    &[&label]
                )));

            // Replacement is subject to the same check, including Unsupported -> Ready.
            cache.store(
                sites[4].token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &[&label],
                State::Ready(retained.clone(), PLAN_BYTES),
            );
            assert!(Cache::charged(cache.storage.as_ref().unwrap()) <= TOTAL_BYTES);
        }
        // The all-key-only worst case cannot require eviction of executing weak entries.
        let mut cache = Cache::default();
        for site in &sites {
            cache.unsupported(
                site.token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &[&label],
            );
        }
        let storage = cache.storage.as_ref().unwrap();
        assert_eq!(storage.entries.iter().flatten().count(), ENTRIES);
        assert_eq!(
            Cache::charged(storage),
            std::mem::size_of::<Storage>()
                + ENTRIES * (LABEL_BYTES + std::mem::size_of::<LoopSiteToken>())
        );
        assert!(Cache::charged(storage) <= TOTAL_BYTES);
    }

    #[test]
    fn old_reentrant_return_cannot_replace_a_newer_same_key_plan() {
        let engine = Engine::new();
        let context = Context::current(&engine.interp);
        let site = LoopSite::default();
        let mut cache = Cache::default();
        let old = plan("1");
        old.active.set(1);
        cache.running(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            &old,
        );
        let sites: Vec<_> = (0..ENTRIES).map(|_| LoopSite::default()).collect();
        for other in &sites {
            cache.unsupported(
                other.token(),
                &engine.interp.global,
                RootKind::While,
                context,
                &[],
            );
        }
        let newer = plan("2");
        newer.active.set(1);
        cache.running(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            &newer,
        );
        cache.finish(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            &old,
        );
        let Lookup::Plan(found) = cache.lookup(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
        ) else {
            panic!("newer in-flight plan")
        };
        assert!(Rc::ptr_eq(&found, &newer));
        assert_eq!(old.active.get(), 0);
        assert_eq!(newer.active.get(), 1);
        cache.finish(
            site.token(),
            &engine.interp.global,
            RootKind::While,
            context,
            &[],
            &newer,
        );
    }

    #[test]
    fn cold_site_layout_has_no_allocation_and_reports_statement_size_delta() {
        use crate::ast::*;
        // Exact pre-site enum shape, to measure the production AST cost on every target ABI.
        #[allow(dead_code)]
        enum Before {
            Expr(Expr),
            VarDecl {
                kind: DeclKind,
                decls: Vec<(Pattern, Option<Expr>)>,
            },
            FuncDecl(Rc<Function>),
            Return(Option<Expr>),
            If {
                test: Expr,
                cons: P<Stmt>,
                alt: Option<P<Stmt>>,
            },
            Block(Vec<Stmt>),
            While {
                test: Expr,
                body: P<Stmt>,
            },
            DoWhile {
                body: P<Stmt>,
                test: Expr,
            },
            For {
                init: Option<P<ForInit>>,
                test: Option<Expr>,
                update: Option<Expr>,
                body: P<Stmt>,
            },
            ForInOf {
                decl: Option<DeclKind>,
                left: Pattern,
                right: Expr,
                of: bool,
                is_await: bool,
                body: P<Stmt>,
            },
            Break(Option<String>),
            Continue(Option<String>),
            Throw(Expr),
            Try {
                block: Vec<Stmt>,
                handler: Option<(Option<Pattern>, Vec<Stmt>)>,
                finalizer: Option<Vec<Stmt>>,
            },
            Switch {
                disc: Expr,
                cases: Vec<SwitchCase>,
            },
            Labeled {
                label: String,
                body: P<Stmt>,
            },
            With {
                obj: Expr,
                body: P<Stmt>,
            },
            ClassDecl(Rc<Class>),
            Empty,
            Debugger,
            Import(ImportDecl),
            ExportNamed {
                specs: Vec<ExportSpec>,
                source: Option<Rc<str>>,
            },
            ExportDecl(P<Stmt>),
            ExportDefault(P<Stmt>),
            ExportAll {
                source: Rc<str>,
                exported: Option<String>,
            },
        }
        let before = std::mem::size_of::<Before>();
        let after = std::mem::size_of::<Stmt>();
        eprintln!(
            "loop-site AST layout: Stmt before={before}, after={after}, site={} bytes",
            std::mem::size_of::<LoopSite>()
        );
        assert_eq!(
            after, before,
            "optimization metadata must not enlarge non-loop statements"
        );
        assert_eq!(
            std::mem::size_of::<LoopBody>(),
            after + std::mem::size_of::<LoopSite>()
        );
        let body = crate::parser::parse_script("for(var i=0;i<10;i++){}", false)
            .ok()
            .unwrap();
        let Stmt::For {
            body: loop_body, ..
        } = &body[0]
        else {
            panic!("loop")
        };
        let site = &loop_body.site;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(32);
            assert!(engine.interp.run_program(&body).is_ok());
            assert!(site.retained_token().is_none());
            assert!(engine.interp.fragment_cache.storage.is_none());
        }
    }

    fn check(source: &str, expected: &str, compilations: usize, hits: usize) {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(32);
            loop_fragment::TEST_FRAGMENT_COMPILES.with(|n| n.set(0));
            loop_fragment::TEST_FRAGMENT_CACHE_HITS.with(|n| n.set(0));
            match engine.eval(source, false).expect("source") {
                Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
                Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
            }
            assert_eq!(
                loop_fragment::TEST_FRAGMENT_COMPILES.with(Cell::get),
                if tier == Tier::Interp {
                    0
                } else {
                    compilations
                },
                "{tier:?}"
            );
            assert_eq!(
                loop_fragment::TEST_FRAGMENT_CACHE_HITS.with(Cell::get),
                if tier == Tier::Interp { 0 } else { hits },
                "{tier:?}"
            );
        }
    }

    #[test]
    fn repeated_inner_loops_reuse_one_plan_without_reusing_completion_or_bindings() {
        check("var sum=0,f=[];for(var outer=0;outer<5;outer++){for(let k=0;k<300;k++){sum+=k;if(k===299)f.push(()=>k)}}[sum,f.map(x=>x()).join(',')].join('|')",
            "224250|299,299,299,299,299", 1, 4);
        check("var n=0,closed=0;for(var outer=0;outer<3;outer++){for(const x of {[Symbol.iterator](){let k=0;return {next(){return {done:false,value:k++}},return(){closed++;return {}}}}}){n+=x;if(x===299)break}}[n,closed].join('|')",
            "134550|3", 1, 2);
    }

    #[test]
    fn shared_function_syntax_uses_fresh_captured_environments_on_cache_hits() {
        // under-with functions intentionally stay AST at the ordinary entry. No change to the
        // existing first-call function admission heuristic is needed to exercise fragment reuse.
        check("function make(v){with({})return function(){var s=0;for(var k=0;k<300;k++)s+=v+k;return s}}var a=make(1),b=make(7);[a(),b(),a()].join('|')",
            "45150|46950|45150", 1, 2);
    }

    #[test]
    fn shared_syntax_never_shares_fragment_ic_state_between_engines() {
        let body = crate::parser::parse_script(
            "var f;with({})f=function(){var s=0;for(var k=0;k<300;k++)s+=obj.x;return s};f()",
            false,
        )
        .ok()
        .expect("source");
        for tier in [Tier::Bytecode, Tier::Jit] {
            let mut a = Engine::new();
            let mut b = Engine::new();
            for engine in [&mut a, &mut b] {
                engine.set_tier(tier);
                engine.set_tier_threshold(32);
            }
            a.eval("var obj={x:2}", false).unwrap();
            b.eval("var obj={pad:0,x:7}", false).unwrap();
            for (first, expected) in [(true, 600.0), (false, 2100.0), (true, 600.0)] {
                let engine = if first { &mut a } else { &mut b };
                assert!(matches!(engine.interp.run_program(&body),
                    Ok(crate::value::Value::Num(value)) if value == expected));
            }
            let cached = |engine: &Engine| {
                engine
                    .interp
                    .fragment_cache
                    .storage
                    .as_ref()
                    .unwrap()
                    .entries
                    .iter()
                    .flatten()
                    .find_map(|entry| match &entry.state {
                        State::Ready(plan, _) => Some(plan.clone()),
                        _ => None,
                    })
                    .expect("retained plan")
            };
            assert!(!Rc::ptr_eq(&cached(&a).chunk, &cached(&b).chunk));
        }
    }

    #[test]
    fn large_nested_source_is_charged_once_and_not_retained_by_a_small_plan() {
        let engine = Engine::new();
        let site = LoopSite::default();
        let mut cache = Cache::default();
        let plan = plan(&format!(
            "var f=function(){{return '{}'}};f",
            "z".repeat(PLAN_BYTES)
        ));
        assert!(plan.source_bytes >= PLAN_BYTES);
        publish(&mut cache, &site, &engine, &plan);
        assert!(cache
            .storage
            .as_ref()
            .unwrap()
            .entries
            .iter()
            .all(Option::is_none));
        // Full reporting still includes that syntax while active; the quota only changes
        // cache retention, never whether the valid function/fragment can execute.
        let mut visitor = crate::memory::Visitor::default();
        visitor.chunk(&plan.chunk);
    }
}
