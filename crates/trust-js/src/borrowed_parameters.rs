//! Scoped homes for an AST activation whose uncaptured bindings move to VM slots.
//!
//! The slot owners are Cell-backed and shared throughout the borrowed VM/native invocation.
//! The registry contains no independent Value owner and never outlives that invocation.
//! Parked bindings retain their identity and attributes, but transfer their Value owner to
//! the shared slot until teardown; mapped parameters consult this canonical slot as well.

use crate::interpreter::Env;
use crate::value::{PackedValue, Value};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

struct Alias {
    environment: Env,
    name: Rc<str>,
    slot: *const Cell<PackedValue>,
}

#[derive(Default, Clone)]
pub(crate) struct Registry(Rc<RefCell<Vec<Alias>>>);

pub(crate) struct Scope<'a> {
    registry: Registry,
    floor: usize,
    end: usize,
    _slots: PhantomData<&'a [Cell<PackedValue>]>,
}

impl Registry {
    /// Transfer binding ownership to shared slot cells pinned by the caller.
    ///
    /// # Safety
    /// The returned scope must be dropped before any registered slot is moved, dropped, or
    /// borrowed exclusively. It must not be forgotten. VM/native writers must use the shared
    /// Cell storage contract, and native register copies must be published before observers.
    /// Slots must already contain the bindings' exact current values (Empty for TDZ), and
    /// every possible observer of a moved binding must use these cells. Environment-homed
    /// captured/eval bindings cannot be transferred. Exact environment bindings must remain
    /// present until teardown; nested scopes must drop in reverse entry order. All indices
    /// and mutable binding access are checked before any pointer or owner is transferred.
    pub(crate) unsafe fn enter<'a>(
        &self,
        homes: &[(Env, Rc<str>, u16)],
        slots: &'a [Cell<PackedValue>],
    ) -> Scope<'a> {
        for (environment, name, slot) in homes {
            assert!((*slot as usize) < slots.len());
            assert!(environment.borrow_mut().vars.contains_key(name));
        }
        let mut aliases = self.0.borrow_mut();
        let floor = aliases.len();
        aliases.extend(homes.iter().map(|(environment, name, slot)| Alias {
            environment: environment.clone(),
            name: name.clone(),
            slot: &slots[*slot as usize],
        }));
        let scope = Scope {
            registry: self.clone(),
            floor,
            end: aliases.len(),
            _slots: PhantomData,
        };
        drop(aliases);
        // Registration and the restoration guard precede clearing parked words. The binding's
        // initialization/mutability flags stay independent; the canonical slot retains its
        // current word while this AST record is parked.
        for (environment, name, _) in homes {
            environment
                .borrow_mut()
                .vars
                .get_mut(name)
                .expect("borrowed binding remains present")
                .set_value(Value::Undefined);
        }
        scope
    }

    pub(crate) fn read(&self, environment: &Env, name: &str) -> Option<Value> {
        let aliases = self.0.borrow();
        let alias = aliases.iter().rev().find(|alias| {
            Env::ptr_eq(&alias.environment, environment) && alias.name.as_ref() == name
        })?;
        // The scoped registration pins this shared Cell owner; it cannot resize or move.
        Some(unsafe { &*alias.slot }.get().unpack())
    }

    pub(crate) fn write(&self, environment: &Env, name: &str, value: &Value) -> bool {
        let aliases = self.0.borrow();
        let Some(alias) = aliases.iter().rev().find(|alias| {
            Env::ptr_eq(&alias.environment, environment) && alias.name.as_ref() == name
        }) else {
            return false;
        };
        // Match ordinary VM writes: transfer a new owner through the Cell, releasing the old.
        unsafe { &*alias.slot }.set(PackedValue::pack(*value));
        true
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        let aliases = self.0.borrow();
        for alias in aliases.iter() {
            visitor.rc_str(&alias.name);
        }
        std::mem::size_of::<RefCell<Vec<Alias>>>()
            + 2 * std::mem::size_of::<usize>()
            + aliases.capacity() * std::mem::size_of::<Alias>()
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        let mut aliases = self.registry.0.borrow_mut();
        assert_eq!(aliases.len(), self.end, "borrowed binding scopes drop LIFO");
        // Restore every parked record before removing its canonical lookup or allowing its
        // slot owner to be cleared/reused. Value destruction invokes no JavaScript here.
        for alias in &aliases[self.floor..] {
            let word = unsafe { &*alias.slot }.get();
            let mut scope = alias.environment.borrow_mut();
            let binding = scope
                .vars
                .get_mut(&alias.name)
                .expect("borrowed binding remains present");
            binding.initialized = !word.is_empty();
            binding.set_word(word);
        }
        aliases.truncate(self.floor);
    }
}

impl crate::gc_trace::GcTrace for Registry {
    /// Aliased homes' environments; the aliased slots belong to registered VM frames.
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        for alias in self
            .0
            .try_borrow()
            .expect("borrowed parameter registry mutably borrowed at a safepoint (I9)")
            .iter()
        {
            let Alias {
                environment,
                name: _,
                slot: _,
            } = alias;
            sink.env(environment);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{new_scope, Binding};
    use crate::value::ValueView;

    /// A Rust unwind out of a borrowed scope restores every parked binding from its slot, with
    /// the binding's attributes, and leaves no stale slot pointer. The object the slot held
    /// before its overwrite is then reachable from neither, so a collection frees it.
    #[test]
    fn transferred_owners_and_binding_attributes_restore_on_unwind() {
        let mut engine = crate::Engine::new();
        let environment = new_scope(None);
        let registry = Registry::default();
        // The object's handles and the slots live only in this helper frame (the collector
        // scans the stack).
        let weak = crate::gc_test::in_frame(|| {
            let old = crate::value::Object::new(None);
            let weak = crate::value::Gc::downgrade(&old);
            environment
                .borrow_mut()
                .vars
                .insert("old", Binding::data(Value::Obj(old.clone()), false, true));
            environment
                .borrow_mut()
                .vars
                .insert("later", Binding::data(Value::Undefined, true, false));
            let slots = [
                Cell::new(PackedValue::pack(Value::Obj(old))),
                Cell::new(PackedValue::pack(Value::Empty)),
            ];
            let homes = [
                (environment.clone(), Rc::from("old"), 0),
                (environment.clone(), Rc::from("later"), 1),
            ];
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _scope = unsafe { registry.enter(&homes, &slots) };
                {
                    let parked = environment.borrow();
                    let old = parked.vars.get("old").unwrap();
                    assert!(matches!((old.value()).view(), ValueView::Undefined));
                    assert!(old.initialized && !old.mutable && old.strict_immutable);
                    assert!(!parked.vars.get("later").unwrap().initialized);
                }
                slots[0].set(PackedValue::pack(Value::Num(7.0)));
                panic!("test restoration on Rust unwind");
            }));
            assert!(unwind.is_err());
            weak
        });
        assert!(
            registry.0.borrow().is_empty(),
            "no stale slot pointer survives"
        );
        {
            let restored = environment.borrow();
            let old = restored.vars.get("old").unwrap();
            assert!(matches!((old.value()).view(), ValueView::Num(7.0)));
            assert!(old.initialized && !old.mutable && old.strict_immutable);
            let later = restored.vars.get("later").unwrap();
            assert!(!later.initialized && later.mutable);
            assert!(matches!((later.value()).view(), ValueView::Empty));
        }
        engine.interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&weak),
            "the overwritten object is released"
        );
    }
}
