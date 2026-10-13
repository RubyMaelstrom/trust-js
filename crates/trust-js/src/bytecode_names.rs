//! Guarded lexical addresses, following ECMA-262 GetIdentifierReference and declarative
//! GetBindingValue/SetMutableBinding (local snapshot e28783d5fc9d). Every intermediate key set
//! is guarded. Object/with environments and live imports are never bypassed. Published layouts
//! describe fresh fixed-key activations, not captured values or parent links; every link is
//! followed live and every holder value, mutability and TDZ flag is read live.

use super::{Chunk, NameIc};
use crate::interpreter::{Abrupt, Binding, Env, Interp, Scope, VarMap};
use crate::value::{Exotic, Value};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

// No real aligned pointer has this value. Existing native compact probes safely miss it.
pub(crate) const DEEP_NAME_IC: usize = 4;
const MAX_SCOPES: usize = 16;

#[repr(C)]
struct Guard {
    identity: usize,
    generation: u32,
    layout: u32,
    pin: Weak<RefCell<Scope>>,
}

/// Auditable native descriptor prefix. It never owns a raw referent: the containing IC owns
/// the guard vector, whose Weak pins prohibit allocation ABA. Emitters load this prefix through
/// the live cache cell on every probe, never retain its address across a helper/cache refill.
#[repr(C)]
pub(crate) struct NativeNameDescriptor {
    pub(crate) guards: *const u8,
    pub(crate) len: usize,
    pub(crate) kind: u32,
    pub(crate) shape: u32,
    pub(crate) binding: *const Binding,
    pub(crate) slot: usize,
}

impl NativeNameDescriptor {
    fn new(guards: &[Guard], target: &Target) -> Self {
        let (kind, shape, binding, slot) = match *target {
            Target::Binding { pointer, slot } => (0, 0, pointer, slot.unwrap_or(usize::MAX)),
            Target::Global { shape, slot } => (1, shape, std::ptr::null(), slot),
        };
        Self {
            guards: guards.as_ptr().cast(),
            len: guards.len(),
            kind,
            shape,
            binding,
            slot,
        }
    }
}

pub(crate) const NAME_GUARD_SIZE: usize = std::mem::size_of::<Guard>();
pub(crate) const NAME_GUARD_IDENTITY: usize = std::mem::offset_of!(Guard, identity);
pub(crate) const NAME_GUARD_GENERATION: usize = std::mem::offset_of!(Guard, generation);
pub(crate) const NAME_GUARD_LAYOUT: usize = std::mem::offset_of!(Guard, layout);

impl Guard {
    fn matches(&self, env: &Env, scope: &Scope) -> bool {
        scope.with_obj.is_none()
            && ((self.pin.as_ptr() == Rc::as_ptr(env)
                && scope.vars.matches_generation(self.generation))
                || (self.layout != 0 && self.layout == scope.vars.layout_id()))
    }
}

enum Target {
    Binding {
        pointer: *const Binding,
        slot: Option<usize>,
    },
    Global {
        shape: u32,
        slot: usize,
    },
}

#[repr(C)]
pub(super) struct DeepNameIc {
    native: NativeNameDescriptor,
    guards: Vec<Guard>,
    name: u32,
    target: Target,
}

/// Separate from a value IC: resolving an identifier must not touch TDZ, import, mutable or
/// accessor state. Those checks belong to later GetValue / PutValue, possibly after user code.
#[repr(C)]
pub(super) struct ResolutionIc {
    native: NativeNameDescriptor,
    guards: Vec<Guard>,
    target: Target,
}

/// Stable per-name cell read by native capture templates. Refill publishes only after the
/// owning Box is installed, and templates retain no descriptor pointer across helper calls.
#[repr(C)]
pub(crate) struct ResolutionSite {
    pub(crate) native: Cell<*const NativeNameDescriptor>,
    cache: RefCell<Option<Box<ResolutionIc>>>,
}

impl Default for ResolutionSite {
    fn default() -> Self {
        Self {
            native: Cell::new(std::ptr::null()),
            cache: RefCell::new(None),
        }
    }
}

impl ResolutionIc {
    fn capture(
        &self,
        i: &Interp,
        start: &Env,
        name: &Rc<str>,
    ) -> Option<crate::eval::PreparedReference> {
        let mut env = start.clone();
        for (depth, guard) in self.guards.iter().enumerate() {
            let scope = env.borrow();
            if !guard.matches(&env, &scope) {
                return None;
            }
            if depth + 1 < self.guards.len() {
                let parent = scope.parent.clone()?;
                drop(scope);
                env = parent;
                continue;
            }
            match self.target {
                Target::Binding { pointer, slot } => {
                    let binding = if guard.pin.as_ptr() == Rc::as_ptr(&env)
                        && scope.vars.matches_generation(guard.generation)
                    {
                        unsafe { &*pointer }
                    } else {
                        slot.and_then(|slot| scope.vars.binding_at(slot, name))
                            .or_else(|| scope.vars.get(name))?
                    };
                    return Some(crate::eval::PreparedReference::scope_with_hint(
                        name.clone(),
                        env.clone(),
                        i.strict,
                        scope.vars.generation(),
                        binding as *const Binding as *mut Binding,
                    ));
                }
                Target::Global { shape, slot } => {
                    if !Rc::ptr_eq(&env, &i.global_env)
                        || !i.ordinary_get_ptr(Rc::as_ptr(&i.global) as usize)
                    {
                        return None;
                    }
                    let global = i.global.borrow();
                    if !matches!(global.exotic, Exotic::None)
                        || global.props.shape() != shape
                        || global.props.property_at(slot).is_none()
                    {
                        return None;
                    }
                    let mut reference = crate::eval::PreparedReference::object(
                        name.clone(),
                        i.global.clone(),
                        i.strict,
                    );
                    reference.shape = shape;
                    reference.slot = slot;
                    return Some(reference);
                }
            }
        }
        None
    }

    fn build(i: &Interp, start: &Env, name: &str) -> Option<Self> {
        let mut env = start.clone();
        let mut guards = Vec::new();
        for _ in 0..MAX_SCOPES {
            let scope = env.borrow();
            if scope.with_obj.is_some()
                || (scope.vars.generation() == u32::MAX && scope.vars.layout_id() == 0)
            {
                return None;
            }
            guards.push(Guard {
                identity: Rc::as_ptr(&env) as usize,
                pin: Rc::downgrade(&env),
                generation: scope.vars.generation(),
                layout: scope.vars.layout_id(),
            });
            let target = if let Some(binding) = scope.vars.get(name) {
                Some(Target::Binding {
                    pointer: binding,
                    slot: scope.vars.binding_slot(name),
                })
            } else if Rc::ptr_eq(&env, &i.global_env) {
                if !i.ordinary_get_ptr(Rc::as_ptr(&i.global) as usize) {
                    return None;
                }
                let global = i.global.borrow();
                if !matches!(global.exotic, Exotic::None)
                    || !crate::value::is_cacheable_shape(global.props.shape())
                {
                    return None;
                }
                // HasBinding is true for any own property, not merely readable data values.
                Some(Target::Global {
                    shape: global.props.shape(),
                    slot: global.props.slot_of(name)?,
                })
            } else {
                None
            };
            if let Some(target) = target {
                return Some(Self {
                    native: NativeNameDescriptor::new(&guards, &target),
                    guards,
                    target,
                });
            }
            let parent = scope.parent.clone()?;
            drop(scope);
            env = parent;
        }
        None
    }
}

impl Chunk {
    fn resolution_sites(&self) -> &[ResolutionSite] {
        self.resolution_caches.get_or_init(|| {
            (0..self.names.len())
                .map(|_| ResolutionSite::default())
                .collect()
        })
    }

    pub(crate) fn jit_resolution_cache_ptr(&self, name: u32) -> usize {
        &self.resolution_sites()[name as usize].native as *const _ as usize
    }

    pub(crate) fn jit_shared_name_ptr(&self, name: u32) -> usize {
        &self.names[name as usize] as *const Rc<str> as usize
    }

    pub(super) fn resolve_name_reference(
        &self,
        i: &mut Interp,
        env: &Env,
        name: u32,
    ) -> Result<crate::eval::PreparedReference, Abrupt> {
        let site = &self.resolution_sites()[name as usize];
        let mut cache = site.cache.borrow_mut();
        let key = &self.names[name as usize];
        if let Some(reference) = cache.as_ref().and_then(|cache| cache.capture(i, env, key)) {
            return Ok(reference);
        }
        if let Some(next) = ResolutionIc::build(i, env, key) {
            let reference = next.capture(i, env, key).expect("fresh resolution proof");
            let next = Box::new(next);
            let native = &next.native as *const NativeNameDescriptor;
            *cache = Some(next);
            site.native.set(native);
            return Ok(reference);
        }
        // Do not retain a RefCell borrow across proxy/@@unscopables author code: reentrancy may
        // reach this same Chunk and cache cell. Resolution remains exactly one observable walk.
        drop(cache);
        i.prepare_shared_name_reference(key.clone(), env)
    }

    pub(super) fn resolution_cache_bytes(&self) -> usize {
        self.resolution_caches.get().map_or(0, |caches| {
            caches.len() * std::mem::size_of::<ResolutionSite>()
                + caches
                    .iter()
                    .map(|cache| {
                        cache.cache.borrow().as_ref().map_or(0, |cache| {
                            std::mem::size_of::<ResolutionIc>()
                                + cache.guards.capacity() * std::mem::size_of::<Guard>()
                        })
                    })
                    .sum::<usize>()
        })
    }
}

impl DeepNameIc {
    fn binding<'a>(&self, env: &Env, vars: &'a VarMap, name: &str) -> Option<&'a Binding> {
        let Target::Binding { pointer, slot } = self.target else {
            return None;
        };
        let guard = self.guards.last()?;
        if guard.pin.as_ptr() == Rc::as_ptr(env) && vars.matches_generation(guard.generation) {
            // The identity is weak-pinned and structural generation is unchanged.
            Some(unsafe { &*pointer })
        } else {
            slot.and_then(|slot| vars.binding_at(slot, name))
        }
    }

    fn read(&self, chunk: &Chunk, i: &Interp, env: &Env, depth: usize) -> Option<Value> {
        let scope = env.borrow();
        if !self.guards.get(depth)?.matches(env, &scope) {
            return None;
        }
        if depth + 1 < self.guards.len() {
            return self.read(chunk, i, scope.parent.as_ref()?, depth + 1);
        }
        match self.target {
            Target::Binding { .. } => {
                let binding = self
                    .binding(env, &scope.vars, &chunk.names[self.name as usize])
                    .or_else(|| scope.vars.get(&chunk.names[self.name as usize]))?;
                (binding.initialized && binding.import_ref.is_none()).then(|| binding.value.clone())
            }
            Target::Global { shape, slot } => {
                if !Rc::ptr_eq(env, &i.global_env)
                    || !i.ordinary_get_ptr(Rc::as_ptr(&i.global) as usize)
                {
                    return None;
                }
                let global = i.global.borrow();
                if !matches!(global.exotic, Exotic::None) || global.props.shape() != shape {
                    return None;
                }
                let property = global.props.property_at(slot)?;
                (!property.accessor()).then(|| property.value())
            }
        }
    }

    fn store(
        &self,
        chunk: &Chunk,
        i: &Interp,
        env: &Env,
        depth: usize,
        value: &mut Option<Value>,
    ) -> bool {
        let scope = env.borrow();
        if !self
            .guards
            .get(depth)
            .is_some_and(|guard| guard.matches(env, &scope))
        {
            return false;
        }
        if depth + 1 < self.guards.len() {
            return scope
                .parent
                .as_ref()
                .is_some_and(|parent| self.store(chunk, i, parent, depth + 1, value));
        }
        match self.target {
            Target::Binding { slot, .. } => {
                drop(scope);
                let mut scope = env.borrow_mut();
                let binding = if let Some(slot) = slot {
                    scope
                        .vars
                        .binding_at_mut(slot, &chunk.names[self.name as usize])
                } else {
                    scope.vars.get_mut(&chunk.names[self.name as usize])
                };
                if let Some(binding) = binding {
                    if binding.initialized && binding.mutable && binding.import_ref.is_none() {
                        binding.value = value.take().unwrap();
                        return true;
                    }
                }
                false
            }
            Target::Global { shape, slot } => {
                if !Rc::ptr_eq(env, &i.global_env)
                    || !i.ordinary_get_ptr(Rc::as_ptr(&i.global) as usize)
                {
                    return false;
                }
                let mut global = i.global.borrow_mut();
                if !matches!(global.exotic, Exotic::None) || global.props.shape() != shape {
                    return false;
                }
                if let Some((_, property)) = global.props.entry_at_mut(slot) {
                    if !property.accessor() && property.writable() {
                        property.set_value(value.take().unwrap());
                        return true;
                    }
                }
                false
            }
        }
    }
}

impl Chunk {
    pub(super) fn deep_name_ic_hit(&self, i: &Interp, env: &Env, c: u32) -> Option<Value> {
        self.deep_name_caches
            .borrow()
            .get(c as usize)?
            .as_ref()?
            .read(self, i, env, 0)
    }

    pub(super) fn deep_name_ic_store(
        &self,
        i: &Interp,
        env: &Env,
        c: u32,
        value: &mut Option<Value>,
    ) -> bool {
        self.deep_name_caches
            .borrow()
            .get(c as usize)
            .and_then(Option::as_ref)
            .is_some_and(|cache| cache.store(self, i, env, 0, value))
    }

    pub(super) fn deep_name_ic_fill(&self, i: &Interp, env: &Env, n: u32, c: u32) -> Option<Value> {
        let mut current = env.clone();
        let mut guards = Vec::new();
        for _ in 0..MAX_SCOPES {
            let scope = current.borrow();
            if scope.with_obj.is_some()
                || (scope.vars.generation() == u32::MAX && scope.vars.layout_id() == 0)
            {
                return None;
            }
            guards.push(Guard {
                identity: Rc::as_ptr(&current) as usize,
                pin: Rc::downgrade(&current),
                generation: scope.vars.generation(),
                layout: scope.vars.layout_id(),
            });
            let resolved = if let Some(binding) = scope.vars.get(&self.names[n as usize]) {
                if !binding.initialized || binding.import_ref.is_some() {
                    return None;
                }
                Some((
                    Target::Binding {
                        pointer: binding,
                        slot: scope.vars.binding_slot(&self.names[n as usize]),
                    },
                    binding.value.clone(),
                ))
            } else if Rc::ptr_eq(&current, &i.global_env) {
                if !i.ordinary_get_ptr(Rc::as_ptr(&i.global) as usize) {
                    return None;
                }
                let global = i.global.borrow();
                if !matches!(global.exotic, Exotic::None)
                    || !crate::value::is_cacheable_shape(global.props.shape())
                {
                    return None;
                }
                let slot = global.props.slot_of(&self.names[n as usize])?;
                let property = global.props.property_at(slot)?;
                if property.accessor() {
                    return None;
                }
                Some((
                    Target::Global {
                        shape: global.props.shape(),
                        slot,
                    },
                    property.value(),
                ))
            } else {
                None
            };
            if let Some((target, value)) = resolved {
                let cache = Box::new(DeepNameIc {
                    native: NativeNameDescriptor::new(&guards, &target),
                    guards,
                    target,
                    name: n,
                });
                let mut caches = self.deep_name_caches.borrow_mut();
                if caches.is_empty() {
                    caches.resize_with(self.name_caches.len(), || None);
                }
                let descriptor = &cache.native as *const NativeNameDescriptor as usize as u64;
                caches[c as usize] = Some(cache);
                self.name_caches[c as usize].set(NameIc {
                    env: DEEP_NAME_IC,
                    binding: descriptor,
                    ..NameIc::EMPTY
                });
                self.name_pins.borrow_mut()[c as usize] = None;
                self.record_name_number(c as usize, &value);
                return Some(value);
            }
            let parent = scope.parent.clone()?;
            drop(scope);
            current = parent;
        }
        None
    }

    pub(super) fn deep_name_cache_bytes(&self) -> usize {
        let caches = self.deep_name_caches.borrow();
        caches.capacity() * std::mem::size_of::<Option<Box<DeepNameIc>>>()
            + caches
                .iter()
                .filter_map(Option::as_ref)
                .map(|cache| {
                    std::mem::size_of::<DeepNameIc>()
                        + cache.guards.capacity() * std::mem::size_of::<Guard>()
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Op;
    use crate::interpreter::{new_binding_layout_id, new_scope};
    use crate::value::Callable;
    use crate::Engine;

    fn reader() -> (Engine, Rc<Chunk>, u32, u32) {
        let mut engine = Engine::new();
        engine
            .eval("function readDeep(){return outer;}", false)
            .unwrap();
        let value = engine
            .interp
            .global
            .borrow()
            .props
            .get("readDeep")
            .unwrap()
            .value();
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("function")
        };
        let chunk = super::super::compile(&user.func).unwrap();
        let (name, cache) = chunk
            .ops
            .iter()
            .find_map(|op| match op {
                Op::LoadName(name, cache) => Some((*name, *cache)),
                _ => None,
            })
            .unwrap();
        (engine, chunk, name, cache)
    }

    fn scope(parent: Option<Env>, key: &str, value: f64, layout: u32) -> Env {
        let env = new_scope(parent);
        env.borrow_mut()
            .vars
            .insert(key, Binding::data(Value::Num(value), true, true));
        env.borrow_mut().vars.publish_layout(layout);
        env
    }

    #[test]
    fn deep_names_reuse_fresh_layouts_but_read_live_cells_and_links() {
        let (mut engine, chunk, name, cache) = reader();
        let layouts = [
            new_binding_layout_id(),
            new_binding_layout_id(),
            new_binding_layout_id(),
        ];
        let holder = scope(None, "outer", 11.0, layouts[0]);
        let middle = scope(Some(holder.clone()), "middle", 0.0, layouts[1]);
        let first = scope(Some(middle), "local", 0.0, layouts[2]);
        assert!(matches!(
            chunk.load_name_ic(&mut engine.interp, &first, name, cache),
            Ok(Value::Num(11.0))
        ));
        assert_eq!(chunk.name_caches[cache as usize].get().env, DEEP_NAME_IC);
        // Every scope is fresh, including the holder. Matching layout never reuses old values.
        let holder2 = scope(None, "outer", 29.0, layouts[0]);
        let middle2 = scope(Some(holder2.clone()), "middle", 0.0, layouts[1]);
        let second = scope(Some(middle2.clone()), "local", 0.0, layouts[2]);
        assert!(matches!(
            chunk.name_ic_hit(&engine.interp, &second, cache),
            Some(Value::Num(29.0))
        ));
        assert!(chunk
            .store_name_ic(&mut engine.interp, &second, name, cache, Value::Num(37.0))
            .is_ok());
        assert!(matches!(
            holder2.borrow().vars.get("outer").unwrap().value,
            Value::Num(37.0)
        ));
        assert!(matches!(
            holder.borrow().vars.get("outer").unwrap().value,
            Value::Num(11.0)
        ));
        holder2
            .borrow_mut()
            .vars
            .get_mut("outer")
            .unwrap()
            .initialized = false;
        assert!(chunk.name_ic_hit(&engine.interp, &second, cache).is_none());
        holder2
            .borrow_mut()
            .vars
            .get_mut("outer")
            .unwrap()
            .initialized = true;
        holder2.borrow_mut().vars.get_mut("outer").unwrap().mutable = false;
        holder2
            .borrow_mut()
            .vars
            .get_mut("outer")
            .unwrap()
            .strict_immutable = true;
        assert!(chunk
            .store_name_ic(&mut engine.interp, &second, name, cache, Value::Num(99.0))
            .is_err());
        middle2
            .borrow_mut()
            .vars
            .insert("outer", Binding::data(Value::Num(43.0), true, true));
        assert!(chunk.name_ic_hit(&engine.interp, &second, cache).is_none());
        assert!(matches!(
            chunk.load_name_ic(&mut engine.interp, &second, name, cache),
            Ok(Value::Num(43.0))
        ));
        let weak = Rc::downgrade(&first);
        drop(first);
        assert!(
            weak.upgrade().is_none(),
            "cache must not retain environments"
        );
    }

    #[test]
    fn deep_names_guard_unpublished_scope_identity_with_and_depth_bound() {
        let (mut engine, chunk, name, cache) = reader();
        let holder = scope(None, "outer", 5.0, 0);
        let middle = scope(Some(holder.clone()), "middle", 0.0, 0);
        let child = scope(Some(middle.clone()), "local", 0.0, 0);
        assert!(chunk
            .load_name_ic(&mut engine.interp, &child, name, cache)
            .is_ok());
        let shadow = scope(Some(holder.clone()), "outer", 99.0, 0);
        assert_eq!(
            shadow.borrow().vars.generation(),
            middle.borrow().vars.generation()
        );
        child.borrow_mut().parent = Some(shadow);
        assert!(chunk.name_ic_hit(&engine.interp, &child, cache).is_none());
        child.borrow_mut().parent = Some(middle.clone());
        middle.borrow_mut().with_obj = Some(Value::Obj(engine.interp.global.clone()));
        assert!(chunk.name_ic_hit(&engine.interp, &child, cache).is_none());
        assert!(chunk
            .deep_name_ic_fill(&engine.interp, &child, name, cache)
            .is_none());
        middle.borrow_mut().with_obj = None;
        let mut deep = holder;
        for _ in 0..MAX_SCOPES {
            deep = new_scope(Some(deep));
        }
        assert!(chunk
            .deep_name_ic_fill(&engine.interp, &deep, name, cache)
            .is_none());
    }

    #[test]
    fn name_pointer_caches_fail_closed_after_generation_exhaustion() {
        let (mut engine, chunk, name, cache) = reader();
        let holder = scope(None, "outer", 5.0, 0);
        holder
            .borrow_mut()
            .vars
            .set_generation_for_test(u32::MAX - 1);
        assert!(matches!(
            chunk.load_name_ic(&mut engine.interp, &holder, name, cache),
            Ok(Value::Num(5.0))
        ));
        assert_eq!(chunk.name_caches[cache as usize].get().gen, u32::MAX - 1);
        holder.borrow_mut().vars.remove("outer");
        holder
            .borrow_mut()
            .vars
            .insert("outer", Binding::data(Value::Num(17.0), true, true));
        assert_eq!(holder.borrow().vars.generation(), u32::MAX);
        assert!(chunk.name_ic_hit(&engine.interp, &holder, cache).is_none());
        assert!(chunk
            .name_ic_fill(&engine.interp, &holder, name, cache)
            .is_none());
        assert!(matches!(
            chunk.load_name_ic(&mut engine.interp, &holder, name, cache),
            Ok(Value::Num(17.0))
        ));
        assert!(chunk
            .store_name_ic(&mut engine.interp, &holder, name, cache, Value::Num(19.0))
            .is_ok());
        assert!(matches!(
            holder.borrow().vars.get("outer").unwrap().value,
            Value::Num(19.0)
        ));
        let middle = scope(Some(holder), "middle", 0.0, 0);
        let child = scope(Some(middle), "local", 0.0, 0);
        assert!(chunk
            .deep_name_ic_fill(&engine.interp, &child, name, cache)
            .is_none());
        // Captured-cell caches must also avoid publishing the exhausted token.
        let local = scope(None, "outer", 23.0, 0);
        local.borrow_mut().vars.set_generation_for_test(u32::MAX);
        assert!(matches!(
            chunk.load_cap_ic(&mut engine.interp, &local, name),
            Ok(Value::Num(23.0))
        ));
        assert_eq!(chunk.cap_caches[name as usize].get().env, 0);
    }

    #[test]
    fn resolution_cache_captures_tdz_const_and_fresh_layouts_without_reading() {
        let (mut engine, chunk, name, _) = reader();
        let layout = new_binding_layout_id();
        let holder = scope(None, "outer", 5.0, layout);
        holder
            .borrow_mut()
            .vars
            .get_mut("outer")
            .unwrap()
            .initialized = false;
        let mut reference = chunk
            .resolve_name_reference(&mut engine.interp, &holder, name)
            .ok()
            .unwrap();
        assert!(engine
            .interp
            .read_prepared_reference(&mut reference)
            .is_err());
        drop(reference); // Release the explicitly captured strong owner before testing cache pins.
        let fresh = scope(None, "outer", 31.0, layout);
        let mut reference = chunk
            .resolve_name_reference(&mut engine.interp, &fresh, name)
            .ok()
            .unwrap();
        assert!(matches!(
            engine.interp.read_prepared_reference(&mut reference),
            Ok(Value::Num(31.0))
        ));
        fresh.borrow_mut().vars.get_mut("outer").unwrap().mutable = false;
        fresh
            .borrow_mut()
            .vars
            .get_mut("outer")
            .unwrap()
            .strict_immutable = true;
        assert!(engine
            .interp
            .write_prepared_reference(&mut reference, Value::Num(37.0))
            .is_err());
        let weak = Rc::downgrade(&holder);
        drop(holder);
        assert!(
            weak.upgrade().is_none(),
            "resolution caches hold weak scope pins only"
        );
    }

    #[test]
    fn deep_names_global_cells_remain_live_and_accessors_miss() {
        let (mut engine, chunk, name, cache) = reader();
        engine.eval("globalThis.outer = 7;", false).unwrap();
        let middle = new_scope(Some(engine.interp.global_env.clone()));
        let child = new_scope(Some(middle));
        assert!(matches!(
            chunk.load_name_ic(&mut engine.interp, &child, name, cache),
            Ok(Value::Num(7.0))
        ));
        engine.eval("outer = 13;", false).unwrap();
        assert!(matches!(
            chunk.name_ic_hit(&engine.interp, &child, cache),
            Some(Value::Num(13.0))
        ));
        // Accessor conversion may preserve key layout, but must execute its getter.
        engine
            .eval(
                "Object.defineProperty(globalThis,'outer',{get(){return 19}, configurable:true});",
                false,
            )
            .unwrap();
        assert!(chunk.name_ic_hit(&engine.interp, &child, cache).is_none());
        assert!(matches!(
            chunk.load_name_ic(&mut engine.interp, &child, name, cache),
            Ok(Value::Num(19.0))
        ));
    }

    #[test]
    fn inline_feedback_retries_are_changed_only_exponential_and_bounded() {
        let (_engine, chunk, _, _) = reader();
        chunk.inline_attempted.set(false);
        chunk.jit_runs.set(100);
        assert!(chunk.advance_inline_feedback());
        let second = chunk.inline_retry_at.get();
        assert!(second > 100);
        assert!(!chunk.inline_retry_due(second - 1));
        assert!(chunk.inline_retry_due(second));
        chunk.jit_runs.set(second);
        assert!(
            !chunk.advance_inline_feedback(),
            "identical empty feedback must not replan"
        );
        let third = chunk.inline_retry_at.get();
        assert!(third - second > second - 100);
        chunk.jit_runs.set(third);
        assert!(!chunk.advance_inline_feedback());
        let fourth = chunk.inline_retry_at.get();
        assert!(fourth > third);
        chunk.jit_runs.set(fourth);
        assert!(!chunk.advance_inline_feedback());
        assert_eq!(chunk.inline_retry_at.get(), 0);
        assert!(!chunk.inline_retry_due(u32::MAX));
    }
}
