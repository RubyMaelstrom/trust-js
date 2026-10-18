//! Deferred physical storage for MakeConstructor's already-present own `prototype` property.
//! ECMA-262 MakeConstructor, OrdinaryGetOwnProperty, OrdinaryOwnPropertyKeys (snapshot
//! e28783d5fc9d). The descriptor/key exists immediately. Every value observation materializes
//! one stable object with its creation realm's intrinsic parent and the correct constructor.
//! Collector and memory traversal must not count as value observations.

use super::{Exotic, Gc, Object, PropClass, Property, PropertyLayout, Props};
use std::cell::RefCell;

enum State {
    Parent(Gc),
    Materialized(Gc),
}

pub(super) struct LazyFunctionPrototype {
    /// The function whose `prototype` this is, for the materialized prototype's `constructor`.
    /// A lazy word lives only in that function's properties (cloning one materializes it), so
    /// the function is alive whenever the prototype materializes. gc-heap: a traced edge of
    /// this cell ([`trace_cell`]). The function and its cell form a cycle that dies in one
    /// collection, and a stale stack word that conservatively keeps the cell keeps the
    /// function too. A weak handle would allocate a weak cell per constructor and keep a dead
    /// constructor's cell reserved for a collection. The default build holds it weakly, since a
    /// strong `Rc` would be a cycle.
    #[cfg(feature = "gc-heap")]
    owner: Gc,
    #[cfg(not(feature = "gc-heap"))]
    owner: crate::value::WeakGc,
    state: RefCell<State>,
    // Reuse the creation Agent's exact key layout/shape, not whichever Agent first reads it.
    // Keys contain no ECMAScript object edges and do not keep the function/realm alive.
    layout: Option<PropertyLayout>,
    shape: u32,
    constructor: bool,
}

impl LazyFunctionPrototype {
    pub(super) fn new(owner: &Gc, parent: Gc, template: &Props) -> Self {
        // Materialization pushes a `Property::builtin` constructor field and reuses the
        // template shape: sound only while the template's field classes are the
        // builtin/plain data-writable ones that replay produces.
        debug_assert!(
            template
                .entries
                .fields
                .iter()
                .all(|property| property.class() == PropClass::DataWritable),
            "lazy prototype template with a non-writable or accessor field"
        );
        Self {
            #[cfg(feature = "gc-heap")]
            owner: *owner,
            #[cfg(not(feature = "gc-heap"))]
            owner: Gc::downgrade(owner),
            state: RefCell::new(State::Parent(parent)),
            layout: template.shared_layout().cloned(),
            shape: template.shape(),
            constructor: template.contains("constructor"),
        }
    }

    pub(super) fn materialize(&self) -> Gc {
        // Held across one allocation, the prototype's cell. Allocation never collects, so no
        // collection can trace this state while it is borrowed (gc-heap, I9; see `trace_cell`).
        let mut state = self.state.borrow_mut();
        if let State::Materialized(prototype) = &*state {
            return prototype.clone();
        }
        #[cfg(test)]
        MATERIALIZED.with(|count| count.set(count.get() + 1));
        let State::Parent(parent) = &*state else {
            unreachable!()
        };
        let mut props = Props::with_layout(self.constructor as usize, self.layout.clone());
        if self.constructor {
            // A descriptor clone materializes before it can outlive the function. All other
            // value reads retain their receiver, including reads while its RefCell is borrowed.
            #[cfg(feature = "gc-heap")]
            let owner = self.owner;
            #[cfg(not(feature = "gc-heap"))]
            let owner = self
                .owner
                .upgrade()
                .expect("live function owns deferred prototype");
            props
                .entries
                .fields
                .push(Property::builtin(super::Value::Obj(owner)));
        }
        props.shape = self.shape;
        let prototype = Object::new_with_parts(Some(parent.clone()), props, Exotic::None);
        *state = State::Materialized(prototype.clone());
        prototype
    }

    pub(super) fn with_gc_edge(&self, f: impl FnOnce(&Gc)) {
        match &*self.state.borrow() {
            State::Parent(parent) => f(parent),
            State::Materialized(prototype) => f(prototype),
        }
    }

    pub(super) fn visit_retained_memory(&self, visitor: &mut crate::memory::Visitor) {
        if let Some(layout) = &self.layout {
            visitor.property_layout(layout);
        }
        // Actual Object allocations are enumerated by their owning heap. The single stored Gc
        // edge is traced separately; never create a new Object during a retained-memory scan.
    }
}

/// The trace of a deferred-prototype cell (gc-heap): its function and the prototype's parent,
/// or once materialized the prototype. Its finalizer (`gc_box::finalize`) drops the key layout.
#[cfg(feature = "gc-heap")]
pub(super) fn trace_cell(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
    // SAFETY: deferred-prototype cells are initialized by `gc_box::new` before any collection.
    let lazy = unsafe { &*cell.cast::<LazyFunctionPrototype>() };
    visit(Gc::as_ptr(&lazy.owner).cast());
    // SAFETY: collections run at safepoints, where no state is mutably borrowed (I9):
    // `materialize` holds its borrow across one allocation, which never collects.
    // `try_borrow_unguarded` panics if one is.
    let state = unsafe { lazy.state.try_borrow_unguarded() }
        .expect("deferred prototype mutably borrowed during GC (I9)");
    let (State::Parent(edge) | State::Materialized(edge)) = state;
    visit(Gc::as_ptr(edge).cast());
}

/// Prototypes materialized on this thread so far (tests).
#[cfg(test)]
pub(crate) fn materialized() -> u64 {
    MATERIALIZED.with(std::cell::Cell::get)
}

#[cfg(test)]
thread_local! {
    static MATERIALIZED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Tier;
    use crate::value::{Value, PACK_LAZY_PROTO, PACK_OBJ};
    use crate::{Completion, Engine};

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine
            .eval(source, false)
            .expect("prototype fixture parses")
        {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
        }
    }

    fn object(engine: &mut Engine, name: &str) -> Gc {
        let env = engine.interp.global_env.clone();
        let Value::Obj(object) = engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("{name}"))
        else {
            panic!("object")
        };
        object
    }

    fn check(source: &str, expected: &str) {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
            engine.interp.gc_collect();
            assert_eq!(eval(&mut engine, "1+2"), "3");
        }
    }

    #[test]
    fn lazy_prototype_is_not_materialized_by_calls_gc_or_memory_scans() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            "var lazy = function(x){return x+1}; for(var i=0;i<160;i++)lazy(i);",
        );
        let function = object(&mut engine, "lazy");
        let tag = || {
            function
                .borrow()
                .props
                .get("prototype")
                .unwrap()
                .packed
                .tag()
        };
        assert_eq!(tag(), PACK_LAZY_PROTO);
        let before = crate::value::heap_live_objects(&engine.interp.gc_heap);
        let _ = crate::memory::json(&engine.interp);
        assert_eq!(tag(), PACK_LAZY_PROTO);
        assert_eq!(
            crate::value::heap_live_objects(&engine.interp.gc_heap),
            before
        );
        engine.interp.gc_collect();
        assert_eq!(tag(), PACK_LAZY_PROTO);
        // The physical state is promoted once; descriptor reads share the exact prototype.
        assert_eq!(
            eval(
                &mut engine,
                "lazy.prototype===Object.getOwnPropertyDescriptor(lazy,'prototype').value"
            ),
            "true"
        );
        assert_eq!(tag(), PACK_OBJ);
    }

    #[test]
    fn lazy_prototype_keys_descriptor_and_constructor_identity_are_exact() {
        check(
            r#"
            function F(){}
            var keys=Reflect.ownKeys(F).join(',');
            var d=Object.getOwnPropertyDescriptor(F,'prototype');
            var c=Object.getOwnPropertyDescriptor(d.value,'constructor');
            [keys, d.value===F.prototype, Object.getPrototypeOf(d.value)===Object.prototype,
             d.writable,d.enumerable,d.configurable,c.value===F,c.writable,c.enumerable,c.configurable,
             new F() instanceof F].join('|');
        "#,
            "length,name,prototype|true|true|true|false|false|true|true|false|true|true",
        );
    }

    #[test]
    fn lazy_prototype_freeze_seal_and_proxy_invariants() {
        check(
            r#"
            var F=function(){}, G=function(){};
            Object.freeze(F);Object.seal(G);
            var p=F.prototype;
            var denied=Reflect.set(F,'prototype',{}), allowed=Reflect.set(G,'prototype',p);
            var invariant=false, missing=false;
            try{new Proxy(F,{get(t,k){if(k==='prototype')return {};return t[k]}}).prototype}
            catch(e){invariant=e instanceof TypeError}
            try{Object.getOwnPropertyDescriptor(new Proxy(G,{getOwnPropertyDescriptor(){return undefined}}),'prototype')}
            catch(e){missing=e instanceof TypeError}
            [denied,allowed,F.prototype===p,G.prototype===p,invariant,missing,
             Object.isFrozen(F),Object.isSealed(G),p.constructor===F].join('|');
        "#,
            "false|true|true|true|true|true|true|true|true",
        );
    }

    #[test]
    fn lazy_prototype_jit_fresh_constructors_replacements_and_numeric_guards() {
        check(
            r#"
            function make(){return function(x){this.x=x}}
            function create(C,x){return new C(x)}
            function replace(F,p){F.prototype=p}
            function read(F){return F.prototype}
            function bump(F){return F.prototype++}
            var sum=0, last;
            for(var k=0;k<200;k++){
                var C=make(), x=create(C,k);if(!(x instanceof C))throw 'prototype';sum+=x.x;
                var D=make(), p={id:k};replace(D,p);if(read(D)!==p)throw 'replacement';
                last=create(D,k);if(Object.getPrototypeOf(last)!==p)throw 'new prototype';
                bump({prototype:k});
            }
            var old=Object.prototype.valueOf, calls=0;
            Object.prototype.valueOf=function(){calls++;return 7};
            var fresh=make(), result=bump(fresh);Object.prototype.valueOf=old;
            [sum,last.x,result,fresh.prototype,calls].join(':');
        "#,
            "19900:199:7:8:1",
        );
    }

    #[test]
    fn lazy_prototype_uses_creation_realm_and_generator_intrinsics() {
        check(
            r#"
            var other=$262.createRealm().global;
            var F=new other.Function('this.nt=new.target');
            var a=Reflect.construct(F,[],F);
            function* G(){yield 1}
            async function* AG(){yield 2}
            var gp=G.prototype, ap=AG.prototype;
            [Object.getPrototypeOf(F.prototype)===other.Object.prototype,
             F.prototype.constructor===F, Object.getPrototypeOf(a)===F.prototype,a.nt===F,
             Object.hasOwn(gp,'constructor'),Object.hasOwn(ap,'constructor'),
             Object.getPrototypeOf(G())===gp,Object.getPrototypeOf(AG())===ap,
             Object.hasOwn(()=>0,'prototype'),Object.hasOwn(async()=>0,'prototype')].join('|');
        "#,
            "true|true|true|true|false|false|true|true|false|false",
        );
    }

    /// P1 S4 (gc-heap): a dead constructor's cell is freed by the collection that finds it
    /// dead. Its deferred prototype holds no weak handle that would keep the cell reserved (a
    /// zombie) until the next collection.
    #[cfg(feature = "gc-heap")]
    #[test]
    fn dead_constructors_leave_no_reserved_cells() {
        let mut engine = Engine::new();
        engine.interp.gc_collect();
        let cells = |engine: &Engine| engine.interp.gc_heap.core.borrow().stats().live_cells;
        let before = cells(&engine);
        eval(
            &mut engine,
            "(() => { for (let i = 0; i < 1000; i++) (function () {}); })(); 0",
        );
        engine.interp.gc_collect();
        let after = cells(&engine);
        assert!(
            after < before + 100,
            "dead constructors' cells stay reserved: {before} -> {after}"
        );
    }

    /// The deferred-prototype cells `engine`'s heap holds (gc-heap).
    #[cfg(feature = "gc-heap")]
    fn lazy_cells(engine: &Engine) -> usize {
        let heap = &engine.interp.gc_heap;
        heap.core.borrow().live_count(heap.lazy_proto_desc)
    }

    /// The tag of the global function `name`'s `prototype` word, read in a helper frame: the
    /// conservative stack scan (gc-heap) would keep a handle left in the caller's frame.
    #[cfg(feature = "gc-heap")]
    fn prototype_tag(engine: &mut Engine, name: &str) -> u64 {
        crate::gc_test::in_frame(|| {
            object(engine, name)
                .borrow()
                .props
                .get("prototype")
                .unwrap()
                .packed
                .tag()
        })
    }

    /// P1 S5 (gc-heap): a function's deferred prototype is a cell that its function keeps
    /// alive. Across forced collections (in the stress child, also one at every safepoint) the
    /// heap keeps one such cell per live function, the words stay lazy and none is
    /// materialized, and a cell's first marker in a traced collection is its function's cell:
    /// it is marked through the function's property word, not by a stack word. Once the
    /// functions are unreachable, the collection that frees them frees their cells: both
    /// counts return to their baselines (give or take a stale stack or pooled-frame word).
    #[cfg(feature = "gc-heap")]
    #[test]
    fn deferred_prototype_cells_live_and_die_with_their_functions() {
        const N: usize = 256;
        let mut engine = Engine::new();
        eval(
            &mut engine,
            "function make() { return function () {}; } var fns = [], F = null; 0",
        );
        engine.interp.gc_collect();
        let objects = |engine: &Engine| crate::value::heap_live_objects(&engine.interp.gc_heap);
        let (cells_before, objects_before) = (lazy_cells(&engine), objects(&engine));
        let materialized = super::materialized();
        eval(
            &mut engine,
            &format!("for (var i = 0; i < {N}; i++) fns.push(make()); F = fns[7]; 0"),
        );
        for _ in 0..3 {
            engine.interp.gc_collect();
            assert_eq!(
                lazy_cells(&engine),
                cells_before + N,
                "one cell per function"
            );
            assert_eq!(prototype_tag(&mut engine, "F"), PACK_LAZY_PROTO);
        }
        assert_eq!(super::materialized(), materialized, "nothing materialized");
        let heap = engine.interp.gc_heap.clone();
        heap.core.borrow_mut().set_retention_tracing(true);
        engine.interp.gc_collect();
        let marked_through_function = crate::gc_test::in_frame(|| {
            let function = object(&mut engine, "F");
            let word = function
                .borrow()
                .props
                .get("prototype")
                .unwrap()
                .packed
                .bits();
            let cell = (word & 0xffff_ffff_ffff) as usize;
            let path = heap.core.borrow().retention_path(cell);
            matches!(
                path.first(),
                Some(&(start, "lazy prototype", crate::gc_core::MarkSource::Cell(owner)))
                    if start == cell && owner == Gc::as_ptr(&function) as usize
            )
        });
        heap.core.borrow_mut().set_retention_tracing(false);
        assert!(marked_through_function, "F's cell is marked through F");
        eval(&mut engine, "fns = null; F = null; 0");
        engine.interp.gc_collect();
        let (cells_after, objects_after) = (lazy_cells(&engine), objects(&engine));
        assert!(
            cells_after <= cells_before + 2 && objects_after <= objects_before + 2,
            "the functions and their cells die in one collection: cells {cells_before} -> \
             {cells_after}, objects {objects_before} -> {objects_after}"
        );
    }

    /// P1 S5 (gc-heap): the deferred-prototype descriptor's trace reports exactly the cell's
    /// two edges, in order: the function, then the prototype's parent while it is deferred;
    /// the function, then the prototype once it is materialized (the promoted word no longer
    /// names the cell, which stays allocated until a collection).
    #[cfg(feature = "gc-heap")]
    #[test]
    fn deferred_prototype_cell_traces_its_function_and_state_edge() {
        let mut engine = Engine::new();
        eval(&mut engine, "var F = function () {}; 0");
        let edges = |cell: *const u8| {
            let mut edges = Vec::new();
            super::trace_cell(cell, &mut |edge| edges.push(edge as usize));
            edges
        };
        let (deferred, materialized, expected_deferred, expected_materialized) =
            crate::gc_test::in_frame(|| {
                let function = object(&mut engine, "F");
                let word = function
                    .borrow()
                    .props
                    .get("prototype")
                    .unwrap()
                    .packed
                    .bits();
                assert_eq!(word & !0xffff_ffff_ffff, PACK_LAZY_PROTO);
                let cell = (word & 0xffff_ffff_ffff) as *const u8;
                let deferred = edges(cell);
                let prototype = function.borrow().props.get("prototype").unwrap().value();
                let materialized = edges(cell);
                let owner = Gc::as_ptr(&function) as usize;
                let parent = Gc::as_ptr(&engine.interp.object_proto) as usize;
                let prototype = Gc::as_ptr(prototype.as_obj().expect("an object")) as usize;
                (
                    deferred,
                    materialized,
                    vec![owner, parent],
                    vec![owner, prototype],
                )
            });
        assert_eq!(deferred, expected_deferred, "function, then parent");
        assert_eq!(
            materialized, expected_materialized,
            "function, then prototype"
        );
    }

    /// P1 S5 (gc-heap): deferred prototypes that collections traced as cells materialize as
    /// they did before the cell. For an ordinary function of this realm, a function made by a
    /// `$262.createRealm()` realm's `Function` and a generator of that realm, after forced
    /// collections each word is still lazy; the first reads materialize each prototype once,
    /// and after more collections every read returns that object, whose `constructor` is its
    /// function (a generator's prototype has none) and whose prototype is the creation realm's
    /// intrinsic.
    #[cfg(feature = "gc-heap")]
    #[test]
    fn deferred_prototypes_materialize_after_collections_in_their_creation_realm() {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            r#"
            var other = $262.createRealm().global;
            var F = function () {}, G = new other.Function(''), H = other.eval('(function* () {})');
            var otherGenerator = other.eval('Object.getPrototypeOf(function* () {}).prototype');
            var pf, pg, ph;
            0"#,
        );
        for _ in 0..3 {
            engine.interp.gc_collect();
        }
        for name in ["F", "G", "H"] {
            assert_eq!(prototype_tag(&mut engine, name), PACK_LAZY_PROTO, "{name}");
        }
        let materialized = super::materialized();
        eval(
            &mut engine,
            "pf = F.prototype; pg = G.prototype; ph = H.prototype; 0",
        );
        assert_eq!(super::materialized() - materialized, 3);
        for _ in 0..3 {
            engine.interp.gc_collect();
        }
        assert_eq!(
            eval(
                &mut engine,
                r#"[pf === F.prototype, pf.constructor === F,
                    Object.getPrototypeOf(pf) === Object.prototype,
                    pg === G.prototype, pg.constructor === G,
                    Object.getPrototypeOf(pg) === other.Object.prototype,
                    other.Object.prototype !== Object.prototype,
                    ph === H.prototype, !Object.hasOwn(ph, 'constructor'),
                    Object.getPrototypeOf(ph) === otherGenerator,
                    Object.getPrototypeOf(H()) === ph].join()"#
            ),
            "true,true,true,true,true,true,true,true,true,true,true"
        );
        assert_eq!(
            super::materialized() - materialized,
            3,
            "later reads share the materialized prototypes"
        );
    }

    /// P1 S5 (gc-heap): `F.prototype = p` over unobserved deferred prototypes through Rust's
    /// `Object.defineProperty` ([[DefineOwnProperty]]) and `Reflect.set` ([[Set]]): the new
    /// value reads back, and every old cell is gone after the next collection. Both operations
    /// first snapshot the current descriptor with `Property::clone`, which materializes the old
    /// prototype (both builds, as before the cell): one materialization per function, and the
    /// materialized object is garbage once the store completes.
    #[cfg(feature = "gc-heap")]
    #[test]
    fn rust_overwrites_of_deferred_prototypes_free_their_cells() {
        const N: usize = 16;
        for store in [
            "Object.defineProperty(f, 'prototype', { value: p })",
            "Reflect.set(f, 'prototype', p)",
        ] {
            let mut engine = Engine::new();
            eval(
                &mut engine,
                &format!(
                    "function make() {{ return function () {{}}; }} \
                     function store(f) {{ {store}; }} var p = {{}}, fns = []; 0"
                ),
            );
            engine.interp.gc_collect();
            let cells = lazy_cells(&engine);
            eval(
                &mut engine,
                &format!("for (var i = 0; i < {N}; i++) fns.push(make()); 0"),
            );
            assert_eq!(lazy_cells(&engine), cells + N, "{store}");
            let materialized = super::materialized();
            eval(&mut engine, "fns.forEach(store); 0");
            assert_eq!(
                super::materialized() - materialized,
                N as u64,
                "{store}: the descriptor snapshot materializes each old prototype"
            );
            assert_eq!(
                eval(&mut engine, "fns.every(f => f.prototype === p)"),
                "true",
                "{store}"
            );
            engine.interp.gc_collect();
            assert_eq!(
                lazy_cells(&engine),
                cells,
                "{store}: the old cells died by the next collection"
            );
        }
    }

    #[test]
    fn lazy_prototype_drops_without_a_cycle_and_materialized_escape_keeps_constructor() {
        let mut engine = Engine::new();
        eval(&mut engine, "var F=function(){}, G=function(){};");
        // Strong handles live in helper frames: the conservative stack scan (gc-heap) would
        // keep a handle in this frame alive after `drop`.
        let weak = crate::gc_test::in_frame(|| {
            let function = object(&mut engine, "F");
            assert_eq!(
                function
                    .borrow()
                    .props
                    .get("prototype")
                    .unwrap()
                    .packed
                    .tag(),
                PACK_LAZY_PROTO
            );
            Gc::downgrade(&function)
        });
        eval(&mut engine, "F=null;");
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(&weak));
        let weak_g = crate::gc_test::in_frame(|| Gc::downgrade(&object(&mut engine, "G")));
        eval(&mut engine, "var escaped=G.prototype;G=null;");
        engine.interp.gc_collect();
        assert!(crate::gc_test::alive(&weak_g));
        assert_eq!(eval(&mut engine, "typeof escaped.constructor"), "function");
        eval(&mut engine, "delete escaped.constructor;");
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(&weak_g));
    }
}
