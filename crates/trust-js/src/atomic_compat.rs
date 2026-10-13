//! `try_update` for the atomic integers, available at the workspace MSRV.
//!
//! Rust 1.99 deprecates `AtomicU32::fetch_update` and its siblings in favor of `try_update`,
//! which is only stable since Rust 1.95; the workspace MSRV is 1.88. These functions perform
//! the same compare-exchange loop as the standard method, with the same orderings, arguments
//! and result. Once the MSRV reaches 1.95, each call can become `atomic.try_update(...)`.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

macro_rules! try_update {
    ($name:ident, $atomic:ty, $int:ty) => {
        #[doc = concat!("`", stringify!($atomic), "::try_update`: apply `f` to the current value")]
        /// and store its result unless it returns `None`. Returns `Ok(previous)` once the store
        /// succeeds, otherwise `Err(previous)`. `f` may run again when another thread changed the
        /// value in the meantime, but is applied only once to the stored value.
        pub(crate) fn $name(
            atomic: &$atomic,
            set_order: Ordering,
            fetch_order: Ordering,
            mut f: impl FnMut($int) -> Option<$int>,
        ) -> Result<$int, $int> {
            let mut prev = atomic.load(fetch_order);
            while let Some(next) = f(prev) {
                match atomic.compare_exchange_weak(prev, next, set_order, fetch_order) {
                    Ok(previous) => return Ok(previous),
                    Err(current) => prev = current,
                }
            }
            Err(prev)
        }
    };
}

try_update!(try_update_u32, AtomicU32, u32);
try_update!(try_update_u64, AtomicU64, u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_standard_try_update_contract() {
        let x = AtomicU32::new(7);
        let order = Ordering::SeqCst;
        assert_eq!(try_update_u32(&x, order, order, |_| None), Err(7));
        assert_eq!(try_update_u32(&x, order, order, |x| Some(x + 1)), Ok(7));
        assert_eq!(try_update_u32(&x, order, order, |x| Some(x + 1)), Ok(8));
        assert_eq!(x.load(order), 9);

        let exhausted = AtomicU64::new(u64::MAX);
        let relaxed = Ordering::Relaxed;
        assert_eq!(
            try_update_u64(&exhausted, relaxed, relaxed, |n| n.checked_add(1)),
            Err(u64::MAX)
        );
        assert_eq!(exhausted.load(relaxed), u64::MAX);
    }
}
