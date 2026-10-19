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
}

impl NativeGcGraph {
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
                .entry(crate::value::Gc::as_ptr(owner) as usize)
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
        self.ids.get(&id).is_some_and(|&index| self.marked[index])
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

    impl crate::host::HostTrace for MixedHost {
        fn trace_gc(&self, visitor: &mut dyn crate::host::HostGcVisitor) {
            assert!(visitor.supports_native());
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

    impl crate::host::HostGc for MixedHost {
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
        // Strong handles live only in this helper frame (gc-heap scans the stack).
        let weak: Vec<_> = crate::gc_test::in_frame(|| {
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
            let weak = values
                .iter()
                .map(|v| crate::value::Gc::downgrade(v.as_obj().unwrap()))
                .collect();
            engine.interp.op_state().put(MixedHost {
                values,
                root: true,
                live_native: Vec::new(),
            });
            weak
        });
        engine.interp.op_state().register_gc::<MixedHost>();
        engine.eval("key=bridge=payload=owner=null", false).unwrap();
        engine.interp.gc_collect();
        assert!(weak.iter().all(|v| crate::gc_test::alive(&v)));
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
        assert!(weak.iter().all(|v| !crate::gc_test::alive(&v)));
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
}
