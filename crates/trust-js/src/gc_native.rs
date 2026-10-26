//! Transient mixed-heap graph of one collection.
//!
//! Native nodes cost no JavaScript object or permanent anchor. JavaScript owners, native roots,
//! and native edges participate in the same mark fixed point as ordinary objects and ephemerons.
//! The graph holds pointer identities only, never additional strong handles.

use crate::fasthash::FastMap;
use crate::host::NativeGcId;
use crate::value::Value;

#[derive(Clone, Copy)]
enum Target {
    Native(usize),
    JavaScript(usize),
}

#[derive(Clone)]
struct Edge {
    target: Target,
    next: usize,
}

const END: usize = usize::MAX;

#[derive(Clone, Default)]
pub(crate) struct NativeGcGraph {
    ids: FastMap<NativeGcId, usize>,
    /// Flat intrusive adjacency avoids an allocation per DOM node/wrapper during marking.
    edges: Vec<Edge>,
    heads: Vec<usize>,
    js_edges: FastMap<usize, usize>,
    roots: Vec<usize>,
    marked: Vec<bool>,
    pending: Vec<usize>,
    /// A minor collection's old prefixes, by identity domain and owner: the identities below the
    /// first young one are live (`HostGcVisitor::native_old_generation`).
    old_prefixes: FastMap<(&'static str, usize), usize>,
}

/// Whether `id` is below its domain and owner's first young identity.
fn is_old(prefixes: &FastMap<(&'static str, usize), usize>, id: NativeGcId) -> bool {
    prefixes
        .get(&(id.domain, id.owner))
        .is_some_and(|&first_young| id.id < first_young)
}

impl NativeGcGraph {
    pub(crate) fn declare_old_generation(
        &mut self,
        domain: &'static str,
        owner: usize,
        first_young_id: usize,
    ) {
        let first_young = self.old_prefixes.entry((domain, owner)).or_default();
        *first_young = (*first_young).max(first_young_id);
    }

    fn index(&mut self, id: NativeGcId) -> usize {
        if let Some(&index) = self.ids.get(&id) {
            return index;
        }
        let index = self.heads.len();
        self.ids.insert(id, index);
        self.heads.push(END);
        self.marked.push(false);
        index
    }

    pub(crate) fn native_edge(&mut self, from: NativeGcId, to: NativeGcId) {
        let from = self.index(from);
        let to = self.index(to);
        let next = self.heads[from];
        self.heads[from] = self.edges.len();
        self.edges.push(Edge {
            target: Target::Native(to),
            next,
        });
    }

    pub(crate) fn js_to_native(&mut self, owner: &Value, to: NativeGcId) {
        if let Some(owner) = owner.as_obj() {
            let to = self.index(to);
            let head = self
                .js_edges
                .entry(crate::value::Gc::as_ptr(&owner) as usize)
                .or_insert(END);
            let next = *head;
            *head = self.edges.len();
            self.edges.push(Edge {
                target: Target::Native(to),
                next,
            });
        }
    }

    pub(crate) fn native_to_js(&mut self, from: NativeGcId, value: &Value) {
        // Every cell a value holds is an edge, primitive payload cells included.
        let target = value.cell_word().map(|word| word as usize);
        if let Some(target) = target {
            let from = self.index(from);
            let next = self.heads[from];
            self.heads[from] = self.edges.len();
            self.edges.push(Edge {
                target: Target::JavaScript(target),
                next,
            });
        }
    }

    pub(crate) fn native_root(&mut self, id: NativeGcId) {
        let index = self.index(id);
        self.roots.push(index);
    }

    /// The JavaScript owners of native edges, traced once the owner is marked.
    pub(crate) fn js_owners(&self) -> impl Iterator<Item = usize> + '_ {
        self.js_edges.keys().copied()
    }

    pub(crate) fn trace_roots(&mut self, javascript: &mut Vec<usize>) {
        self.pending.extend_from_slice(&self.roots);
        // The old identities are live: the edges reported from them are traced, whichever host
        // reported them, once every host has reported.
        if !self.old_prefixes.is_empty() {
            for (&id, &index) in &self.ids {
                if is_old(&self.old_prefixes, id) {
                    self.pending.push(index);
                }
            }
        }
        self.drain(javascript);
    }

    pub(crate) fn trace_js(&mut self, owner: usize, javascript: &mut Vec<usize>) {
        if let Some(&head) = self.js_edges.get(&owner) {
            let mut edge = head;
            while edge != END {
                if let Target::Native(target) = self.edges[edge].target {
                    self.pending.push(target);
                }
                edge = self.edges[edge].next;
            }
            self.drain(javascript);
        }
    }

    fn drain(&mut self, javascript: &mut Vec<usize>) {
        while let Some(index) = self.pending.pop() {
            if std::mem::replace(&mut self.marked[index], true) {
                continue;
            }
            let mut edge = self.heads[index];
            while edge != END {
                match self.edges[edge].target {
                    Target::Native(target) => self.pending.push(target),
                    Target::JavaScript(target) => javascript.push(target),
                }
                edge = self.edges[edge].next;
            }
        }
    }

    pub(crate) fn is_live(&self, id: NativeGcId) -> bool {
        is_old(&self.old_prefixes, id) || self.ids.get(&id).is_some_and(|&index| self.marked[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(owner: usize, id: usize) -> NativeGcId {
        NativeGcId {
            domain: "test.dom",
            owner,
            id,
        }
    }

    /// S6 step 11c: a minor collection's old prefix. An old identity is live unreported, the
    /// edges reported from old identities are traced, and a young identity lives only if reached.
    #[test]
    fn native_graph_old_prefix_is_live_and_traces_its_reported_edges() {
        let mut graph = NativeGcGraph::default();
        graph.declare_old_generation("test.dom", 1, 10);
        graph.native_edge(node(1, 3), node(1, 12));
        graph.native_edge(node(1, 13), node(1, 14));
        graph.trace_roots(&mut Vec::new());
        assert!(graph.is_live(node(1, 5)), "an unreported old identity");
        assert!(graph.is_live(node(1, 3)) && graph.is_live(node(1, 12)));
        assert!(!graph.is_live(node(1, 13)) && !graph.is_live(node(1, 14)));
        assert!(!graph.is_live(node(2, 5)), "another owner's identities have no old prefix");
    }

    #[test]
    fn native_graph_iterative_deep_chain_and_root_identity_domains() {
        let mut graph = NativeGcGraph::default();
        for id in 0..50_000 {
            graph.native_edge(node(1, id), node(1, id + 1));
        }
        graph.native_root(node(1, 0));
        graph.trace_roots(&mut Vec::new());
        assert!(graph.is_live(node(1, 50_000)));
        assert!(!graph.is_live(node(2, 0)));
        assert!(!graph.is_live(NativeGcId {
            domain: "another",
            owner: 1,
            id: 0
        }));
    }

    /// A host whose JavaScript values live only through its native graph: weak handles that
    /// the graph's report keeps (S6 step 11b).
    #[cfg(feature = "embed")]
    struct MixedHost {
        values: Vec<crate::handles::WeakHandle>,
        root: bool,
        live_native: Vec<bool>,
    }

    #[cfg(feature = "embed")]
    impl crate::host::HostTrace for MixedHost {
        fn trace_gc(&self, visitor: &mut dyn crate::host::HostGcVisitor) {
            for (id, value) in self.values.iter().enumerate() {
                visitor.js_to_native(value, node(5, id));
                visitor.native_to_js(node(5, id), value);
            }
            // 0 -> native1 -> JS key. The WeakMap value then exposes native2 -> JS payload.
            visitor.native_edge(node(5, 0), node(5, 1));
            visitor.native_edge(node(5, 2), node(5, 3));
            visitor.native_edge(node(5, 3), node(5, 2));
            if self.root {
                visitor.native_root(node(5, 0));
            }
        }
    }

    #[cfg(feature = "embed")]
    impl crate::host::HostGc for MixedHost {
        fn sweep_gc_with_native(&mut self, native_live: &dyn Fn(NativeGcId) -> bool) {
            self.live_native = (0..self.values.len())
                .map(|id| native_live(node(5, id)))
                .collect();
        }
    }

    /// A host that reports its native graph generationally (S6 step 11c): identity 0 is a root
    /// holding `old`, 0 → 1 → 10, 10 holds `young` and 11 holds `stray`. In a minor collection it
    /// declares the identities below 10 old and reports only the young identities' edges and the
    /// edge of old identity 1, which changed since the last collection.
    #[cfg(feature = "embed")]
    #[derive(Default)]
    struct GenerationalHost {
        old: Option<crate::handles::WeakHandle>,
        young: Option<crate::handles::WeakHandle>,
        stray: Option<crate::handles::WeakHandle>,
        /// Whether each collection's report took the young part alone.
        reports: std::cell::RefCell<Vec<bool>>,
        live_native: Vec<bool>,
    }

    #[cfg(feature = "embed")]
    impl crate::host::HostTrace for GenerationalHost {
        fn trace_gc(&self, visitor: &mut dyn crate::host::HostGcVisitor) {
            let minor = visitor.is_minor() && visitor.native_old_generation("test.dom", 7, 10);
            self.reports.borrow_mut().push(minor);
            if !minor {
                visitor.native_root(node(7, 0));
                visitor.native_edge(node(7, 0), node(7, 1));
                if let Some(old) = &self.old {
                    visitor.native_to_js(node(7, 0), old);
                }
            }
            visitor.native_edge(node(7, 1), node(7, 10));
            if let Some(young) = &self.young {
                visitor.native_to_js(node(7, 10), young);
            }
            if let Some(stray) = &self.stray {
                visitor.native_to_js(node(7, 11), stray);
            }
        }
    }

    #[cfg(feature = "embed")]
    impl crate::host::HostGc for GenerationalHost {
        fn sweep_gc_with_native(&mut self, native_live: &dyn Fn(NativeGcId) -> bool) {
            self.live_native = [0, 1, 10, 11].map(|id| native_live(node(7, id))).to_vec();
        }
    }

    /// The weak handle of a new object, which lives only in this helper's frame (the collector
    /// scans the stack) and in the handle.
    #[cfg(feature = "embed")]
    fn weakly_held_object(engine: &mut crate::Engine) -> crate::handles::WeakHandle {
        crate::gc_test::in_frame(|| {
            engine.scope(|cx| {
                let object = crate::embed::Value::Obj(cx.new_object());
                crate::handles::WeakHandle::new(cx, &object).expect("a weak handle")
            })
        })
    }

    /// S6 step 11c: a minor collection keeps a host's old native identities, reported or not,
    /// traces the edges reported from them to young values, and frees the young values no
    /// reported edge reaches. A major collection takes the whole graph.
    #[cfg(feature = "embed")]
    #[test]
    fn minor_collections_keep_a_hosts_old_native_identities_and_trace_their_edges() {
        let mut engine = crate::Engine::new();
        engine.op_state().put(GenerationalHost::default());
        engine.op_state().register_gc::<GenerationalHost>();
        let old = weakly_held_object(&mut engine);
        engine.host_mut::<GenerationalHost>().unwrap().old = Some(old);
        engine.interp.gc_collect();
        let young = weakly_held_object(&mut engine);
        let stray = weakly_held_object(&mut engine);
        let host = engine.host_mut::<GenerationalHost>().unwrap();
        host.young = Some(young);
        host.stray = Some(stray);
        engine.interp.gc_collect_minor();
        let host = engine.host::<GenerationalHost>().unwrap();
        assert_eq!(*host.reports.borrow(), [false, true]);
        assert_eq!(host.live_native, [true, true, true, false]);
        assert!(!host.old.as_ref().unwrap().is_cleared());
        assert!(!host.young.as_ref().unwrap().is_cleared());
        assert!(host.stray.as_ref().unwrap().is_cleared());
        engine.interp.gc_collect();
        let host = engine.host::<GenerationalHost>().unwrap();
        assert_eq!(host.reports.borrow().last(), Some(&false));
        assert_eq!(host.live_native, [true, true, true, false]);
        assert!(!host.young.as_ref().unwrap().is_cleared());
    }

    #[cfg(feature = "embed")]
    #[test]
    fn mixed_native_heap_and_ephemerons_share_the_full_collector_fixed_point() {
        let mut engine = crate::Engine::new();
        engine.eval("var key={}, bridge={}, payload={n:42}, owner={}; var map = new WeakMap([[key,bridge]]);", false).unwrap();
        // The values live only in this helper frame (the collector scans the stack) and as weak
        // handles in host state.
        crate::gc_test::in_frame(|| {
            let values = engine.scope(|cx| {
                let global = cx.global_this();
                ["owner", "key", "bridge", "payload"]
                    .map(|name| {
                        let value = cx
                            .member_get(&global, name)
                            .unwrap_or_else(|_| panic!("global"));
                        crate::handles::WeakHandle::new(cx, &value).expect("a weak handle")
                    })
                    .to_vec()
            });
            engine.op_state().put(MixedHost {
                values,
                root: true,
                live_native: Vec::new(),
            });
        });
        engine.op_state().register_gc::<MixedHost>();
        engine.eval("key=bridge=payload=owner=null", false).unwrap();
        // Debug layout-owner verification must not consume the native graph's root worklist
        // before the real marker reaches this mixed native/WeakMap fixed point.
        for minor in [true, false, true, false] {
            if minor {
                engine.interp.gc_collect_minor();
            } else {
                engine.interp.gc_collect();
            }
            let host = engine.host::<MixedHost>().unwrap();
            assert!(host.values.iter().all(|value| !value.is_cleared()));
            assert_eq!(host.live_native, [true; 4]);
        }
        engine.host_mut::<MixedHost>().unwrap().root = false;
        engine.interp.gc_collect();
        let host = engine.host::<MixedHost>().unwrap();
        assert!(host.values.iter().all(|value| value.is_cleared()));
        assert_eq!(host.live_native, [false; 4]);
    }
}
