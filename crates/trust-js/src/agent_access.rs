//! Scoped access to the distinct Rust allocations implementing one Agent.
//!
//! This module intentionally uses only std. The Miri harness imports these exact bytes.
//! Owning a raw allocation never creates a reference to its pointee. In particular, borrowing
//! a parent containing `Owned<T>` does not retag an executing child's allocation as `Box<T>` would.
//! References are made only by the caller holding the current access capability.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::{Rc, Weak};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RealmId(u64);

impl RealmId {
    pub(crate) const ROOT: Self = Self(1);
    pub(crate) const UNPUBLISHED: Self = Self(0);
    pub(crate) fn number(self) -> u64 {
        self.0
    }
}

struct Record<T> {
    parent: Option<RealmId>,
    allocation: Option<NonNull<T>>,
    leases: Vec<(u64, NonNull<T>)>,
}

/// A non-owning directory. The root has NO allocation pointer: it may move between entries.
/// An active member always resolves through its newest current-borrow lease, never through
/// its allocation pointer. The latter is only the capability of an inactive `Owned<T>`.
pub(crate) struct Access<T> {
    records: RefCell<BTreeMap<RealmId, Record<T>>>,
    next_id: Cell<u64>,
    next_lease: Cell<u64>,
}

impl<T> Default for Access<T> {
    fn default() -> Self {
        let mut records = BTreeMap::new();
        records.insert(
            RealmId::ROOT,
            Record {
                parent: None,
                allocation: None,
                leases: Vec::new(),
            },
        );
        Self {
            records: RefCell::new(records),
            next_id: Cell::new(2),
            next_lease: Cell::new(1),
        }
    }
}

impl<T> Access<T> {
    pub(crate) fn fresh_id(&self) -> RealmId {
        let id = self.next_id.get();
        self.next_id.set(
            id.checked_add(1)
                .expect("Agent interpreter identities exhausted"),
        );
        RealmId(id)
    }

    pub(crate) fn adopt(self: &Rc<Self>, id: RealmId, parent: RealmId, value: T) -> Owned<T> {
        let records = self.records.borrow();
        assert!(
            records.contains_key(&parent),
            "a realm's parent is registered"
        );
        assert!(
            !records.contains_key(&id),
            "a realm identity is registered once"
        );
        drop(records);
        let pointer =
            NonNull::new(Box::into_raw(Box::new(value))).expect("Box allocation is non-null");
        let mut records = self.records.borrow_mut();
        records.insert(
            id,
            Record {
                parent: Some(parent),
                allocation: Some(pointer),
                leases: Vec::new(),
            },
        );
        Owned {
            pointer,
            id,
            directory: Rc::downgrade(self),
            _affinity: PhantomData,
        }
    }

    /// Publish the capability derived from the CURRENT exclusive borrow before suspending it.
    ///
    /// # Safety
    /// `pointer` is the current exclusive reference to this id's live T. Its owner cannot be
    /// moved/freed until the scope ends. The exclusive reference is suspended while any access
    /// through the published pointer is in progress. Reentry uses a newer nested scope, and
    /// all such accesses end before the original reference resumes. No field/table borrow may
    /// survive a suspension. A raw pointer from an older owner is not an acceptable substitute.
    pub(crate) unsafe fn enter(self: &Rc<Self>, id: RealmId, pointer: NonNull<T>) -> Scope<T> {
        let token = self.next_lease.get();
        self.next_lease
            .set(token.checked_add(1).expect("Agent access leases exhausted"));
        let mut records = self.records.borrow_mut();
        let record = records
            .get_mut(&id)
            .expect("entered interpreter is registered");
        if let Some(allocation) = record.allocation {
            assert_eq!(
                allocation, pointer,
                "interpreter allocation identity changed"
            );
        }
        record.leases.push((token, pointer));
        Scope {
            directory: self.clone(),
            id,
            token,
        }
    }

    /// Suspend the caller's current borrow. Metadata must be prepared before this call; all
    /// callback access uses `pointer`, never the caller's older reference. A callback that
    /// creates a new &mut T must suspend THAT reference again before cross-reentry/traversal.
    pub(crate) fn suspend<R>(
        self: &Rc<Self>,
        id: RealmId,
        current: &mut T,
        operation: impl FnOnce(NonNull<T>) -> R,
    ) -> R {
        let pointer = NonNull::from(current); // LAST use of the incoming exclusive reference.
                                              // SAFETY: the exclusive input remains borrowed for this whole call. No pointee access
                                              // occurs here after publication; Scope::drop only restores directory metadata.
        let _scope = unsafe { self.enter(id, pointer) };
        operation(pointer)
    }

    pub(crate) fn resolve(&self, id: RealmId) -> Option<NonNull<T>> {
        let records = self.records.borrow();
        let record = records.get(&id)?;
        record
            .leases
            .last()
            .map(|&(_, pointer)| pointer)
            .or(record.allocation)
    }

    pub(crate) fn active(&self, id: RealmId) -> bool {
        self.records
            .borrow()
            .get(&id)
            .is_some_and(|record| !record.leases.is_empty())
    }

    /// Stable ids and resolved capabilities, with no directory borrow retained by the caller.
    /// Dereferencing them still requires the phase/suspension contract of enter().
    pub(crate) fn snapshot(&self) -> Vec<(RealmId, NonNull<T>)> {
        self.records
            .borrow()
            .iter()
            .map(|(&id, record)| {
                let pointer = record
                    .leases
                    .last()
                    .map(|&(_, pointer)| pointer)
                    .or(record.allocation)
                    .expect("Agent root has a current entry lease during traversal");
                (id, pointer)
            })
            .collect()
    }

    /// Shared diagnostics may supply their actual root reference without publishing a mutable
    /// capability. The returned pointer for that id is only usable for shared access.
    pub(crate) fn shared_snapshot(&self, id: RealmId, current: &T) -> Vec<(RealmId, *const T)> {
        self.records
            .borrow()
            .iter()
            .map(|(&other, record)| {
                let pointer = if other == id {
                    NonNull::from(current)
                } else {
                    record
                        .leases
                        .last()
                        .map(|&(_, pointer)| pointer)
                        .or(record.allocation)
                        .expect("diagnostic of a child has an enclosing root entry")
                };
                (other, pointer.as_ptr().cast_const())
            })
            .collect()
    }

    fn remove(&self, id: RealmId, pointer: NonNull<T>) {
        let mut records = self.records.borrow_mut();
        let record = records.get(&id).expect("owned realm is registered");
        assert!(
            record.leases.is_empty(),
            "an executing realm cannot be retired"
        );
        assert_eq!(record.allocation, Some(pointer));
        // Descendants are owned by T and are removed as T drops. Never resolve a detached
        // subtree again: collection quarantine ends before Owned is destroyed.
        records.remove(&id);
    }

    pub(crate) fn contains(&self, id: RealmId) -> bool {
        self.records.borrow().contains_key(&id)
    }

    pub(crate) fn parent(&self, id: RealmId) -> Option<RealmId> {
        self.records
            .borrow()
            .get(&id)
            .and_then(|record| record.parent)
    }

    /// Requested storage only. BTreeMap node/allocator overhead remains opaque to diagnostics.
    pub(crate) fn retained_metadata_bytes(&self) -> usize {
        let records = self.records.borrow();
        std::mem::size_of::<Self>()
            .saturating_add(
                records
                    .len()
                    .saturating_mul(std::mem::size_of::<(RealmId, Record<T>)>()),
            )
            .saturating_add(
                records
                    .values()
                    .map(|record| {
                        record
                            .leases
                            .capacity()
                            .saturating_mul(std::mem::size_of::<(u64, NonNull<T>)>())
                    })
                    .fold(0usize, usize::saturating_add),
            )
    }
}

/// Ownership only. Deliberately no Deref, DerefMut, as_ref, or temporary Box reconstruction.
pub(crate) struct Owned<T> {
    pointer: NonNull<T>,
    id: RealmId,
    directory: Weak<Access<T>>,
    _affinity: PhantomData<Rc<T>>,
}

impl<T> Owned<T> {
    pub(crate) fn id(&self) -> RealmId {
        self.id
    }
    /// Exclusive retirement only. Ordinary reads/writes must resolve the newest lease instead.
    pub(crate) fn retirement_pointer(&mut self) -> NonNull<T> {
        if let Some(directory) = self.directory.upgrade() {
            assert!(
                !directory.active(self.id),
                "retirement cannot bypass an active lease"
            );
        }
        self.pointer
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        if let Some(directory) = self.directory.upgrade() {
            directory.remove(self.id, self.pointer);
        }
        // SAFETY: unique ownership originated in adopt. No live access scope survives remove;
        // all collection visitors stopped using a detached allocation before retirement.
        unsafe {
            drop(Box::from_raw(self.pointer.as_ptr()));
        }
    }
}

/// Restores metadata only. It never dereferences T and cannot invoke author code or collect.
pub(crate) struct Scope<T> {
    directory: Rc<Access<T>>,
    id: RealmId,
    token: u64,
}

impl<T> Drop for Scope<T> {
    fn drop(&mut self) {
        let mut records = self.directory.records.borrow_mut();
        let record = records
            .get_mut(&self.id)
            .expect("active interpreter remains registered");
        assert_eq!(
            record.leases.last().map(|&(token, _)| token),
            Some(self.token),
            "Agent leases restore LIFO"
        );
        record.leases.pop();
    }
}

/// Suspend one current borrow without publishing it in an Agent directory. Host callbacks use
/// this until an actual cross-interpreter entry/collection refreshes the current capability.
/// `operation` must use only the supplied lineage; it cannot resume the older input reference.
pub(crate) fn suspend_local<T, R>(current: &mut T, operation: impl FnOnce(NonNull<T>) -> R) -> R {
    let pointer = NonNull::from(current); // LAST use of the incoming exclusive reference.
    operation(pointer)
}
