//! Live legacy platform objects (Web IDL 8f182624, internal methods and
//! #dfn-named-property-visibility). Native membership queries have no JS
//! effects; indexed/named value getters remain ordinary traced callables.

use crate::interpreter::{Abrupt, Interp};
use crate::value::{Gc, Value};
use std::rc::Rc;

impl Interp {
    pub(crate) fn host_named_supported(&self, object: &Gc, key: &str) -> bool {
        if object.borrow().ic_plain.get() || Self::is_sym_key(key) || Self::is_private_key(key) {
            return false;
        }
        self.host_indexed
            .get(&(Rc::as_ptr(object) as usize))
            .and_then(|properties| properties.live.as_ref())
            .is_some_and(|live| {
                live.names
                    .is_some_and(|names| names(self, &live.state).iter().any(|name| name == key))
            })
    }

    pub(crate) fn host_named_visible(&mut self, object: &Gc, key: &str) -> Result<bool, Abrupt> {
        if self.host_indexed.is_empty()
            || object.borrow().ic_plain.get()
            || Self::is_sym_key(key)
            || Self::is_private_key(key)
            || !self
                .host_indexed
                .get(&(Rc::as_ptr(object) as usize))
                .is_some_and(|properties| {
                    properties
                        .live
                        .as_ref()
                        .is_some_and(|live| live.names.is_some())
                })
        {
            return Ok(false);
        }
        // Native supported-name queries are pure. Ordinary own/prototype
        // properties can disprove visibility without constructing all names.
        // Stop this proof at an exotic object; its traps retain spec ordering.
        if object.borrow().props.contains(key)
            || crate::value::canonical_index(key)
                .is_some_and(|index| index < self.host_indexed_len(object).unwrap_or(0))
        {
            return Ok(false);
        }
        let mut prototype = object.borrow().proto.clone();
        while let Some(current) = prototype {
            let current = current.borrow();
            if !current.ic_plain.get() {
                break;
            }
            if current.props.contains(key) {
                return Ok(false);
            }
            prototype = current.proto.clone();
        }
        if !self.host_named_supported(object, key) {
            return Ok(false);
        }
        let mut prototype = crate::builtins::js_get_prototype_of(self, &Value::Obj(object.clone()))
            .map_err(Abrupt::Throw)?;
        while matches!(prototype, Value::Obj(_)) {
            if crate::builtins::has_own_property_trapped(self, &prototype, key)
                .map_err(Abrupt::Throw)?
            {
                return Ok(false);
            }
            prototype =
                crate::builtins::js_get_prototype_of(self, &prototype).map_err(Abrupt::Throw)?;
        }
        Ok(true)
    }

    pub(crate) fn host_named_own_value(
        &mut self,
        object: &Gc,
        key: &str,
    ) -> Result<Option<Value>, Abrupt> {
        if !self.host_named_visible(object, key)? {
            return Ok(None);
        }
        let getter = self
            .host_indexed
            .get(&(Rc::as_ptr(object) as usize))
            .and_then(|properties| properties.live.as_ref())
            .and_then(|live| live.named_getter.clone())
            .expect("named membership requires a getter");
        self.call_callback(
            getter,
            Value::Obj(object.clone()),
            &[Value::lstr(key.to_owned())],
        )
        .map(Some)
    }

    pub(crate) fn host_platform_enumerable(
        &mut self,
        object: &Gc,
        key: &str,
    ) -> Result<Option<bool>, Abrupt> {
        self.host_indexed_own_value(object, key)
            .map(|value| value.map(|_| crate::value::canonical_index(key).is_some()))
    }

    pub(crate) fn host_named_keys(&mut self, object: &Gc) -> Result<Vec<String>, Abrupt> {
        let Some(live) = self
            .host_indexed
            .get(&(Rc::as_ptr(object) as usize))
            .and_then(|properties| properties.live.clone())
        else {
            return Ok(Vec::new());
        };
        let Some(names) = live.names else {
            return Ok(Vec::new());
        };
        let mut result = Vec::new();
        for name in names(self, &live.state) {
            if self.host_named_visible(object, &name)? {
                result.push(name);
            }
        }
        Ok(result)
    }

    /// [[DefineOwnProperty]] tests supported names even when a prototype
    /// property makes them invisible; existing ordinary expandos can be redefined.
    pub(crate) fn host_named_reject_define(&self, object: &Gc, key: &str) -> bool {
        !object.borrow().props.contains(key) && self.host_named_supported(object, key)
    }
}

#[cfg(all(test, feature = "embed"))]
mod tests {
    use super::*;
    use crate::{bytecode::Tier, Completion, Engine};

    fn length(_: &Interp, state: &Value) -> u32 {
        state
            .as_obj()
            .unwrap()
            .borrow()
            .props
            .get("length")
            .unwrap()
            .value()
            .as_num_opt()
            .unwrap() as u32
    }
    fn names(ctx: &Interp, state: &Value) -> Vec<String> {
        if length(ctx, state) == 0 {
            Vec::new()
        } else {
            ["named", "hidden", "99"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        }
    }
    fn run(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
    fn value(engine: &mut Engine, source: &str) -> Value {
        engine
            .eval_value(source)
            .expect("fixture parses")
            .unwrap_or_else(|_| panic!("fixture evaluates"))
    }
    fn setup(tier: Tier) -> Engine {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let state = value(
            &mut engine,
            "globalThis.collectionState={length:3};collectionState",
        );
        let target = value(
            &mut engine,
            "globalThis.liveCollection=Object.create({hidden:42});liveCollection",
        );
        let getter = value(
            &mut engine,
            "(key)=>typeof key==='number' ? key+10 : key+' value'",
        );
        engine
            .ctx()
            .install_live_readonly_indexed_properties(
                &target,
                state,
                length,
                getter.clone(),
                Some((names, getter)),
            )
            .unwrap_or_else(|_| panic!("live collection installs"));
        engine
    }

    #[test]
    fn live_collections53_internal_methods_and_named_visibility() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = setup(tier);
            assert_eq!(
                run(
                    &mut engine,
                    r#"(() => {
                const c=liveCollection, check=(x,s)=>{if(!x)throw Error(s)};
                check(c[0]===10 && c.named==='named value' && c.hidden===42,'read');
                check(c[99]===undefined && c['01']===undefined,'array index boundary');
                const d=Object.getOwnPropertyDescriptor(c,'named');
                check(d.value==='named value' && !d.writable && !d.enumerable && d.configurable,'named descriptor');
                check(Object.keys(c).join()==='0,1,2','enumerability');
                check(Reflect.ownKeys(c).join()==='0,1,2,named,99','own key order');
                check(!Reflect.defineProperty(c,'hidden',{value:1}),'hidden supported name definition');
                check(!Reflect.set(c,'named',1) && !Reflect.deleteProperty(c,'named'),'named write/delete');
                check(!Reflect.set(c,'9',1) && !Reflect.defineProperty(c,'9',{value:1}),'unsupported index writes');
                check(Reflect.deleteProperty(c,'9') && !Reflect.deleteProperty(c,'1'),'indexed delete');
                c.extra=5;check(c.extra===5 && Object.assign({},c).extra===5,'expando');
                check(!Reflect.preventExtensions(c),'extensibility');
                const p=new Proxy(c,{});
                check(p.named==='named value' && !Object.getOwnPropertyDescriptor(p,'named').enumerable,'proxy forwarding');
                collectionState.length=1;
                check(c[1]===undefined && Reflect.ownKeys(c).join()==='0,named,99,extra','live shrink');
                collectionState.length=0;
                check(!('named' in c) && Reflect.deleteProperty(c,'named'),'empty names');
                check(Reflect.defineProperty(c,'named',{value:7,configurable:true}),'expando before supported name');
                collectionState.length=3;
                check(c.named===7 && Reflect.deleteProperty(c,'named') && c.named==='named value','expando precedence');
                return 'ok';
            })()"#
                ),
                "ok",
                "{tier:?}"
            );
        }
    }

    #[test]
    fn live_collections53_exotic_prototype_order_and_shrinking_enumeration() {
        let mut engine = setup(Tier::Jit);
        assert_eq!(
            run(
                &mut engine,
                r#"(() => {
            const c=liveCollection;let calls=0;
            Object.setPrototypeOf(c,new Proxy({},{
                getOwnPropertyDescriptor(t,p){calls++;if(p==='named')throw Error('visible query');},
                getPrototypeOf(){calls++;return null}
            }));
            if(c.absent!==undefined || calls!==0)throw Error('unsupported name inspected prototype');
            let error=false;try{Object.getOwnPropertyDescriptor(c,'named')}catch(e){error=e.message==='visible query'}
            if(!error)throw Error('visibility did not propagate abrupt completion');
            Object.setPrototypeOf(c,null);
            return 'ok';
        })()"#
            ),
            "ok"
        );
        let state = value(&mut engine, "collectionState");
        let target = value(
            &mut engine,
            "globalThis.shrinking=Object.create(null);shrinking",
        );
        let getter = value(
            &mut engine,
            "index=>{if(index===0)collectionState.length=1;return index+20}",
        );
        engine
            .ctx()
            .install_live_readonly_indexed_properties(&target, state, length, getter, None)
            .unwrap_or_else(|_| panic!("install"));
        assert_eq!(
            run(
                &mut engine,
                "collectionState.length=3;Object.keys({...shrinking}).join()"
            ),
            "0"
        );
    }

    #[test]
    fn live_collections53_state_and_getters_are_traced_without_pinning_cycles() {
        let mut engine = setup(Tier::Jit);
        let target = value(&mut engine, "liveCollection");
        let pointer = Rc::as_ptr(target.as_obj().unwrap()) as usize;
        run(
            &mut engine,
            "collectionState.owner=liveCollection;collectionState=null;liveCollection=null",
        );
        drop(target);
        engine.ctx().collect_garbage_for_host();
        assert!(!engine.ctx().host_indexed.contains_key(&pointer));
    }
}
