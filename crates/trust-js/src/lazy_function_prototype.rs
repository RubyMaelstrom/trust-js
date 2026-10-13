//! Deferred physical storage for MakeConstructor's already-present own `prototype` property.
//! ECMA-262 MakeConstructor, OrdinaryGetOwnProperty, OrdinaryOwnPropertyKeys (snapshot
//! e28783d5fc9d). The descriptor/key exists immediately. Every value observation materializes
//! one stable object with its creation realm's intrinsic parent and the correct constructor.
//! Collector and memory traversal must not count as value observations.

use super::{Exotic, Gc, Object, Property, PropertyLayout, Props};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

enum State {
    Parent(Gc),
    Materialized(Gc),
}

pub(super) struct LazyFunctionPrototype {
    owner: Weak<RefCell<Object>>,
    state: RefCell<State>,
    // Reuse the creation Agent's exact key layout/shape, not whichever Agent first reads it.
    // Keys contain no ECMAScript object edges and do not keep the function/realm alive.
    layout: Option<PropertyLayout>,
    shape: u32,
    constructor: bool,
}

impl LazyFunctionPrototype {
    pub(super) fn new(owner: &Gc, parent: Gc, template: &Props) -> Self {
        Self {
            owner: Rc::downgrade(owner),
            state: RefCell::new(State::Parent(parent)),
            layout: template.shared_layout().cloned(),
            shape: template.shape(),
            constructor: template.contains("constructor"),
        }
    }

    pub(super) fn materialize(&self) -> Gc {
        let mut state = self.state.borrow_mut();
        if let State::Materialized(prototype) = &*state {
            return prototype.clone();
        }
        let State::Parent(parent) = &*state else {
            unreachable!()
        };
        let mut props = Props::with_layout(self.constructor as usize, self.layout.clone());
        if self.constructor {
            // A descriptor clone materializes before it can outlive the function. All other
            // value reads retain their receiver, including reads while its RefCell is borrowed.
            let owner = self
                .owner
                .upgrade()
                .expect("live function owns deferred prototype");
            props
                .entries
                .fields
                .push(Property::builtin(super::Value::Obj(super::Gc::from(owner))));
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

    #[test]
    fn lazy_prototype_drops_without_a_cycle_and_materialized_escape_keeps_constructor() {
        let mut engine = Engine::new();
        eval(&mut engine, "var F=function(){}, G=function(){};");
        let function = object(&mut engine, "F");
        let weak = Rc::downgrade(&function);
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
        drop(function);
        eval(&mut engine, "F=null;");
        engine.interp.gc_collect();
        assert!(weak.upgrade().is_none());
        let g = object(&mut engine, "G");
        let weak_g = Rc::downgrade(&g);
        drop(g);
        eval(&mut engine, "var escaped=G.prototype;G=null;");
        engine.interp.gc_collect();
        assert!(weak_g.upgrade().is_some());
        assert_eq!(eval(&mut engine, "typeof escaped.constructor"), "function");
        eval(&mut engine, "delete escaped.constructor;");
        engine.interp.gc_collect();
        assert!(weak_g.upgrade().is_none());
    }
}
