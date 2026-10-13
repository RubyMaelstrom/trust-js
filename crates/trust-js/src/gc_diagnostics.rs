//! Opt-in retaining witnesses, computed independently after root classification.
//! No author hooks, writes to mark bits, or persistent JavaScript handles. This
//! diagnostic duplicates graph traversal and is unsuitable for timing runs.

use crate::fasthash::FastMap;
use crate::gc_native::NativeGcGraph;
use crate::interpreter::{Env, Interp};
use crate::value::{Callable, Gc, Value};
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Node {
    Object(usize),
    Scope(usize),
}

pub(crate) struct Snapshot<'a> {
    pub objects: &'a [Gc],
    pub scopes: &'a [Env],
    pub object_roots: &'a [Gc],
    pub scope_roots: &'a [Env],
    pub native: &'a NativeGcGraph,
    pub host_edges: &'a FastMap<usize, Vec<usize>>,
    pub realm_members: &'a FastMap<usize, usize>,
    pub realm_scope_members: &'a FastMap<usize, usize>,
    pub realm_groups: &'a FastMap<usize, (Vec<Gc>, Env)>,
    pub templates: &'a FastMap<usize, Vec<usize>>,
}

impl Snapshot<'_> {
    pub(crate) fn dump_realm_paths(self, interp: &Interp) {
        let objects: FastMap<_, _> = self
            .objects
            .iter()
            .map(|o| (Rc::as_ptr(o) as usize, o))
            .collect();
        let scopes: FastMap<_, _> = self
            .scopes
            .iter()
            .map(|s| (Rc::as_ptr(s) as usize, s))
            .collect();
        let mut parents = FastMap::<Node, Option<(Node, &'static str)>>::default();
        let mut pending = VecDeque::new();
        let mut add = |node| {
            if parents.insert(node, None).is_none() {
                pending.push_back(node);
            }
        };
        for object in self.object_roots {
            add(Node::Object(Rc::as_ptr(object) as usize));
        }
        for scope in self.scope_roots {
            add(Node::Scope(Rc::as_ptr(scope) as usize));
        }
        let mut native = self.native.clone();
        let mut object_edges = Vec::new();
        let mut scope_edges = Vec::new();
        let mut native_edges = Vec::new();
        let mut ephemerons = FastMap::<Node, Vec<(Node, Node)>>::default();
        for (&owner, entries) in &interp.weak_collection_data {
            for (key, value) in entries {
                let Value::Obj(value) = value else { continue };
                let owner = Node::Object(owner);
                let value = Node::Object(Rc::as_ptr(value) as usize);
                match key.upgrade() {
                    Some(Value::Obj(key)) => {
                        let key = Node::Object(Rc::as_ptr(&key) as usize);
                        ephemerons.entry(owner).or_default().push((key, value));
                        ephemerons.entry(key).or_default().push((owner, value));
                    }
                    Some(Value::Sym(..)) => {
                        ephemerons.entry(owner).or_default().push((owner, value))
                    }
                    _ => {}
                }
            }
        }
        while let Some(from) = pending.pop_front() {
            let mut destinations = Vec::new();
            let realm = match from {
                Node::Object(id) => {
                    let Some(object) = objects.get(&id) else {
                        continue;
                    };
                    interp.obj_refs_into(object, &mut object_edges);
                    interp.obj_scope_refs_into(object, &mut scope_edges);
                    if let Some(coroutine) = interp.generators.get(&id) {
                        coroutine.trace_gc(&mut crate::gc_edges::DirectGcEdges {
                            objects: &mut object_edges,
                            scopes: &mut scope_edges,
                        });
                    }
                    destinations.extend(object_edges.drain(..).map(|o| {
                        (
                            Node::Object(Rc::as_ptr(&o) as usize),
                            "object/internal-slot",
                        )
                    }));
                    destinations.extend(
                        scope_edges.drain(..).map(|s| {
                            (Node::Scope(Rc::as_ptr(&s) as usize), "captured-environment")
                        }),
                    );
                    native.trace_js(id, &mut native_edges);
                    destinations.extend(
                        native_edges
                            .drain(..)
                            .map(|id| (Node::Object(id), "native-graph")),
                    );
                    for edges in [self.host_edges.get(&id), self.templates.get(&id)]
                        .into_iter()
                        .flatten()
                    {
                        destinations
                            .extend(edges.iter().map(|&id| (Node::Object(id), "host/template")));
                    }
                    self.realm_members.get(&id)
                }
                Node::Scope(id) => {
                    let Some(scope) = scopes.get(&id) else {
                        continue;
                    };
                    let b = scope.borrow();
                    if let Some(parent) = &b.parent {
                        destinations.push((
                            Node::Scope(Rc::as_ptr(parent) as usize),
                            "parent-environment",
                        ));
                    }
                    if let Some(Value::Obj(object)) = &b.with_obj {
                        destinations
                            .push((Node::Object(Rc::as_ptr(object) as usize), "with-object"));
                    }
                    for binding in b.vars.values() {
                        if let Value::Obj(object) = &binding.value {
                            destinations
                                .push((Node::Object(Rc::as_ptr(object) as usize), "binding"));
                        }
                        if let Some((scope, _)) = binding.import_ref.as_deref() {
                            destinations
                                .push((Node::Scope(Rc::as_ptr(scope) as usize), "import-binding"));
                        }
                    }
                    self.realm_scope_members.get(&id)
                }
            };
            if let Some((members, scope)) = realm.and_then(|realm| self.realm_groups.get(realm)) {
                destinations.extend(
                    members
                        .iter()
                        .map(|o| (Node::Object(Rc::as_ptr(o) as usize), "realm-intrinsics")),
                );
                destinations.push((Node::Scope(Rc::as_ptr(scope) as usize), "realm-environment"));
            }
            if let Some(entries) = ephemerons.get(&from) {
                for &(other, value) in entries {
                    if parents.contains_key(&other) {
                        destinations.push((value, "ephemeron-both-inputs-live"));
                    }
                }
            }
            for (to, label) in destinations {
                if let std::collections::hash_map::Entry::Vacant(entry) = parents.entry(to) {
                    entry.insert(Some((from, label)));
                    pending.push_back(to);
                }
            }
        }
        for &realm in self.realm_groups.keys() {
            let mut node = Node::Object(realm);
            if !parents.contains_key(&node) {
                continue;
            }
            let mut path = Vec::new();
            for _ in 0..128 {
                let Some(parent) = parents.get(&node) else {
                    break;
                };
                let description = match node {
                    Node::Object(id) => objects
                        .get(&id)
                        .map(|object| {
                            let b = object.borrow();
                            let keys: Vec<_> =
                                b.props.iter().take(5).map(|(key, _)| &**key).collect();
                            let name = match &b.call {
                                Callable::User(user) => user.func.name.as_deref(),
                                _ => None,
                            };
                            format!("object:{id:x} name={name:?} props={keys:?}")
                        })
                        .unwrap_or_else(|| format!("foreign-object:{id:x}")),
                    Node::Scope(id) => scopes
                        .get(&id)
                        .map(|scope| {
                            let b = scope.borrow();
                            let names: Vec<_> = b.vars.keys().take(24).map(|key| &**key).collect();
                            format!("scope:{id:x} vars={names:?}")
                        })
                        .unwrap_or_else(|| format!("foreign-scope:{id:x}")),
                };
                path.push(format!(
                    "{} -> {description}",
                    parent.map_or("root", |(_, label)| label)
                ));
                let Some((previous, _)) = parent else { break };
                node = *previous;
            }
            path.reverse();
            eprintln!("[gc-realm-path] realm={realm:x} {}", path.join(" | "));
        }
    }
}
