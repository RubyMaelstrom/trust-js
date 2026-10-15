//! Scoped homes for an AST activation whose uncaptured bindings move to VM slots.
//!
//! The slot owners are Cell-backed and shared throughout the borrowed VM/native invocation.
//! The registry contains no independent Value owner and never outlives that invocation.
//! Parked bindings retain their identity and attributes, but transfer their Value owner to
//! the shared slot until teardown; mapped parameters consult this canonical slot as well.

use crate::interpreter::Env;
use crate::value::{PackedValue, Value};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

struct Alias {
    environment: Env,
    name: Rc<str>,
    slot: *const PackedValue,
}

#[derive(Default, Clone)]
pub(crate) struct Registry(Rc<RefCell<Vec<Alias>>>);

pub(crate) struct Scope<'a> {
    registry: Registry,
    floor: usize,
    end: usize,
    _slots: PhantomData<&'a [PackedValue]>,
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
        slots: &'a [PackedValue],
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
        // Registration is complete and the restoration guard exists before releasing any
        // parked owner. The binding's initialization/mutability state remains unchanged;
        // only the canonical slot owns its current Value while this AST record is parked.
        for (environment, name, _) in homes {
            environment
                .borrow_mut()
                .vars
                .get_mut(name)
                .expect("borrowed binding remains present")
                .value = Value::Undefined;
        }
        scope
    }

    pub(crate) fn read(&self, environment: &Env, name: &str) -> Option<Value> {
        let aliases = self.0.borrow();
        let alias = aliases.iter().rev().find(|alias| {
            Env::ptr_eq(&alias.environment, environment) && alias.name.as_ref() == name
        })?;
        // The scoped registration pins this shared Cell owner; it cannot resize or move.
        Some(unsafe { &*alias.slot }.unpack())
    }

    pub(crate) fn write(&self, environment: &Env, name: &str, value: &Value) -> bool {
        let aliases = self.0.borrow();
        let Some(alias) = aliases.iter().rev().find(|alias| {
            Env::ptr_eq(&alias.environment, environment) && alias.name.as_ref() == name
        }) else {
            return false;
        };
        // Match ordinary VM writes: transfer a new owner through the Cell, releasing the old.
        unsafe { &*alias.slot }.replace_value(value.clone());
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
            let value = unsafe { &*alias.slot }.unpack();
            let mut scope = alias.environment.borrow_mut();
            let binding = scope
                .vars
                .get_mut(&alias.name)
                .expect("borrowed binding remains present");
            binding.initialized = !matches!(value, Value::Empty);
            binding.value = value;
        }
        aliases.truncate(self.floor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{new_scope, Binding};

    #[test]
    fn transferred_owners_and_binding_attributes_restore_on_unwind() {
        let environment = new_scope(None);
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
            PackedValue::pack(Value::Obj(old)),
            PackedValue::pack(Value::Empty),
        ];
        let homes = [
            (environment.clone(), Rc::from("old"), 0),
            (environment.clone(), Rc::from("later"), 1),
        ];
        let registry = Registry::default();
        assert_eq!(weak.strong_count(), 2);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = unsafe { registry.enter(&homes, &slots) };
            assert_eq!(weak.strong_count(), 1, "only the slot retains its owner");
            {
                let parked = environment.borrow();
                let old = parked.vars.get("old").unwrap();
                assert!(matches!(old.value, Value::Undefined));
                assert!(old.initialized && !old.mutable && old.strict_immutable);
                assert!(!parked.vars.get("later").unwrap().initialized);
            }
            slots[0].replace_value(Value::Num(7.0));
            assert_eq!(weak.strong_count(), 0, "overwrite releases the old owner");
            panic!("test restoration on Rust unwind");
        }));
        assert!(unwind.is_err());
        assert!(
            registry.0.borrow().is_empty(),
            "no stale slot pointer survives"
        );
        let restored = environment.borrow();
        let old = restored.vars.get("old").unwrap();
        assert!(matches!(old.value, Value::Num(7.0)));
        assert!(old.initialized && !old.mutable && old.strict_immutable);
        let later = restored.vars.get("later").unwrap();
        assert!(!later.initialized && later.mutable);
        assert!(matches!(later.value, Value::Empty));
    }
}
