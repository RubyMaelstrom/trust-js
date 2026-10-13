//! Transient mixed-heap graph shared by the full and nursery collectors.
//!
//! Native nodes cost no JavaScript object or permanent anchor. JavaScript owners, native roots,
//! and native edges participate in the same mark fixed point as ordinary objects and ephemerons.
//! The graph holds pointer identities only, never additional strong handles.

use crate::fasthash::FastMap;
use crate::host::NativeGcId;
use crate::value::Value;
use std::rc::Rc;

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
    /// Compact native remembered-set declarations. Only represented old sources are visited;
    /// a large unchanged old DOM never needs a graph node solely to express its age.
    old_prefixes: FastMap<(&'static str, usize), usize>,
    /// Flat intrusive adjacency avoids an allocation per DOM node/wrapper during marking.
    edges: Vec<Edge>,
    heads: Vec<usize>,
    js_edges: FastMap<usize, usize>,
    roots: Vec<usize>,
    marked: Vec<bool>,
    pending: Vec<usize>,
}

impl NativeGcGraph {
    pub(crate) fn declare_old_generation(
        &mut self,
        domain: &'static str,
        owner: usize,
        first_young_id: usize,
    ) {
        let prefix = self.old_prefixes.entry((domain, owner)).or_default();
        *prefix = (*prefix).max(first_young_id);
    }

    fn is_old(&self, id: NativeGcId) -> bool {
        self.old_prefixes
            .get(&(id.domain, id.owner))
            .is_some_and(|&end| id.id < end)
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
                .entry(Rc::as_ptr(owner) as usize)
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
        if let Some(value) = value.as_obj() {
            let from = self.index(from);
            let next = self.heads[from];
            self.heads[from] = self.edges.len();
            self.edges.push(Edge {
                target: Target::JavaScript(Rc::as_ptr(value) as usize),
                next,
            });
        }
    }

    pub(crate) fn native_root(&mut self, id: NativeGcId) {
        let index = self.index(id);
        self.roots.push(index);
    }

    /// Nursery collections conservatively activate native edges from old JavaScript owners,
    /// including those outside the young-object snapshot; major collections do the exact mark.
    pub(crate) fn js_owners(&self) -> impl Iterator<Item = usize> + '_ {
        self.js_edges.keys().copied()
    }

    pub(crate) fn trace_roots(&mut self, javascript: &mut Vec<usize>) {
        self.pending.extend_from_slice(&self.roots);
        // Trace happens after ALL hosts have reported. A foreign host may add an edge
        // whose source belongs to an unchanged old domain declared by an earlier host.
        for (&id, &index) in &self.ids {
            if self.is_old(id) {
                self.pending.push(index);
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
        self.is_old(id) || self.ids.get(&id).is_some_and(|&index| self.marked[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Object;

    fn node(owner: usize, id: usize) -> NativeGcId {
        NativeGcId {
            domain: "test.dom",
            owner,
            id,
        }
    }

    #[test]
    fn native_graph_traces_mixed_cycles_without_strong_handles() {
        let left = Value::Obj(Object::new(None));
        let right = Value::Obj(Object::new(None));
        let left_id = Rc::as_ptr(left.as_obj().unwrap()) as usize;
        let right_id = Rc::as_ptr(right.as_obj().unwrap()) as usize;
        let before = Rc::strong_count(left.as_obj().unwrap());
        let mut graph = NativeGcGraph::default();
        graph.js_to_native(&left, node(1, 1));
        graph.native_edge(node(1, 1), node(1, 2));
        graph.native_edge(node(1, 2), node(1, 1));
        graph.native_to_js(node(1, 2), &right);
        graph.js_to_native(&right, node(1, 3));
        graph.native_to_js(node(1, 3), &left);
        graph.native_edge(node(2, 1), node(2, 2));
        assert_eq!(Rc::strong_count(left.as_obj().unwrap()), before);
        let mut reachable = Vec::new();
        graph.trace_roots(&mut reachable);
        assert!(reachable.is_empty());
        assert!(!graph.is_live(node(1, 1)));
        graph.trace_js(left_id, &mut reachable);
        assert_eq!(reachable, [right_id]);
        graph.trace_js(reachable.pop().unwrap(), &mut reachable);
        assert_eq!(reachable, [left_id]);
        graph.trace_js(reachable.pop().unwrap(), &mut reachable);
        assert!(reachable.is_empty());
        assert!(graph.is_live(node(1, 3)));
        assert!(!graph.is_live(node(2, 1)));
    }

    #[test]
    fn native_minor_old_prefix_activates_foreign_edges_without_enumerating_old_heap() {
        let live = Value::Obj(Object::new(None));
        let dead = Value::Obj(Object::new(None));
        let live_pointer = Rc::as_ptr(live.as_obj().unwrap()) as usize;
        let mut graph = NativeGcGraph::default();
        graph.declare_old_generation("test.dom", 1, 100_000);
        assert!(
            graph.ids.is_empty(),
            "old prefix declaration must be constant-space"
        );
        // Another host reports these edges AFTER the DOM's declaration. The old
        // source was not in the DOM's own mutation journal, but is still live.
        graph.native_edge(node(1, 23), node(2, 100_001));
        graph.native_to_js(node(2, 100_001), &live);
        graph.native_to_js(node(1, 100_000), &dead);
        graph.native_to_js(node(2, 23), &dead);
        let unrelated = NativeGcId {
            domain: "other.dom",
            owner: 1,
            id: 23,
        };
        graph.native_to_js(unrelated, &dead);
        let mut reached = Vec::new();
        graph.trace_roots(&mut reached);
        assert_eq!(reached, [live_pointer]);
        assert!(
            graph.is_live(node(1, 99_999)),
            "unrepresented old IDs stay live"
        );
        assert!(graph.is_live(node(2, 100_001)));
        assert!(!graph.is_live(node(1, 100_000)), "frontier itself is young");
        assert!(!graph.is_live(node(2, 23)), "owner identities do not alias");
        assert!(!graph.is_live(unrelated), "domains do not alias");
        assert_eq!(
            graph.ids.len(),
            5,
            "work scales with represented changed state"
        );

        let mut major = NativeGcGraph::default();
        major.native_edge(node(1, 23), node(2, 100_001));
        major.native_to_js(node(2, 100_001), &live);
        major.trace_roots(&mut reached);
        assert!(
            !major.is_live(node(1, 23)),
            "major remains exact without old prefixes"
        );
        assert!(!major.is_live(node(2, 100_001)));
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

    struct MixedHost {
        values: Vec<Value>,
        root: bool,
        live_native: Vec<bool>,
    }

    impl crate::host::HostGc for MixedHost {
        fn trace_gc(&self, visitor: &mut dyn crate::host::HostGcVisitor) {
            assert!(visitor.supports_native());
            for (id, value) in self.values.iter().enumerate() {
                visitor.internal(value);
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
        fn sweep_gc(&mut self, _: &dyn Fn(&Value) -> bool) {
            panic!("mixed collector must supply native liveness");
        }
        fn sweep_gc_with_native(
            &mut self,
            live: &dyn Fn(&Value) -> bool,
            native_live: &dyn Fn(NativeGcId) -> bool,
        ) {
            self.live_native = (0..self.values.len())
                .map(|id| native_live(node(5, id)))
                .collect();
            for value in &mut self.values {
                if !live(value) {
                    *value = Value::Undefined;
                }
            }
        }
    }

    #[test]
    fn mixed_native_heap_and_ephemerons_share_the_full_collector_fixed_point() {
        let mut engine = crate::Engine::new();
        engine.eval("var key={}, bridge={}, payload={n:42}, owner={}; var map = new WeakMap([[key,bridge]]);", false).unwrap();
        let global = Value::Obj(engine.interp.global.clone());
        let mut values = Vec::new();
        for name in ["owner", "key", "bridge", "payload"] {
            values.push(
                engine
                    .interp
                    .member_get(&global, name)
                    .unwrap_or_else(|_| panic!("global")),
            );
        }
        let weak: Vec<_> = values
            .iter()
            .map(|v| Rc::downgrade(v.as_obj().unwrap()))
            .collect();
        engine.interp.op_state().put(MixedHost {
            values,
            root: true,
            live_native: Vec::new(),
        });
        engine.interp.op_state().register_gc::<MixedHost>();
        engine.eval("key=bridge=payload=owner=null", false).unwrap();
        engine.interp.gc_collect();
        assert!(weak.iter().all(|v| v.upgrade().is_some()));
        assert_eq!(
            engine
                .interp
                .op_state()
                .get::<MixedHost>()
                .unwrap()
                .live_native,
            [true; 4]
        );
        engine
            .interp
            .op_state()
            .get_mut::<MixedHost>()
            .unwrap()
            .root = false;
        engine.interp.gc_collect();
        assert!(weak.iter().all(|v| v.upgrade().is_none()));
        assert_eq!(
            engine
                .interp
                .op_state()
                .get::<MixedHost>()
                .unwrap()
                .live_native,
            [false; 4]
        );
    }

    #[test]
    fn mixed_native_nursery_preserves_old_js_owner_edges_and_sweeps_unrooted_young_islands() {
        let mut engine = crate::Engine::new();
        let owner = Value::Obj(Object::new(None));
        engine.interp.gc_collect();
        let key = Value::Obj(Object::new(None));
        let dead1 = Value::Obj(Object::new(None));
        let dead2 = Value::Obj(Object::new(None));
        let weak: Vec<_> = [&key, &dead1, &dead2]
            .iter()
            .map(|v| Rc::downgrade(v.as_obj().unwrap()))
            .collect();
        engine.interp.op_state().put(MixedHost {
            values: vec![owner, key, dead1, dead2],
            root: false,
            live_native: Vec::new(),
        });
        engine.interp.op_state().register_gc::<MixedHost>();
        engine
            .interp
            .gc_collect_young(crate::value::GcCause::Explicit);
        assert!(
            weak[0].upgrade().is_some(),
            "old owner logically roots young target"
        );
        assert!(weak[1].upgrade().is_none());
        assert!(weak[2].upgrade().is_none());
        assert_eq!(
            engine
                .interp
                .op_state()
                .get::<MixedHost>()
                .unwrap()
                .live_native,
            [true, true, false, false]
        );
        engine.interp.gc_collect();
        assert!(
            weak[0].upgrade().is_none(),
            "major revisits the unrooted old owner"
        );
    }
}
