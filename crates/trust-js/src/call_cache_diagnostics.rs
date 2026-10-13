//! Count epoch churn and identity-matched stale probes without changing call routing.
//! Raw entries are observed as integer metadata only, never dereferenced by diagnostics.
use super::{CallIc, CallSite, CALL_IC_EPOCH};
use crate::value::Value;
use std::{cell::Cell, panic::Location, rc::Rc, sync::atomic::Ordering::Relaxed};

thread_local! { static COUNTS: Cell<[u64; 9]> = const { Cell::new([0; 9]) }; }
fn add(index: usize) {
    COUNTS.with(|counts| {
        let mut c = counts.get();
        c[index] += 1;
        counts.set(c);
    });
}

#[track_caller]
pub(super) fn invalidate() {
    let file = Location::caller().file();
    add(if file.ends_with("jit_optimizing.rs") {
        0
    } else if file.ends_with("jit_cache.rs") {
        1
    } else if file.ends_with("interpreter.rs") {
        2
    } else {
        3
    });
}
pub(super) fn fill(site: &CallSite, incoming: CallIc) {
    add(4);
    if site
        .entries
        .iter()
        .map(Cell::get)
        .any(|old| old.callee == incoming.callee && old.callee != 0 && old.epoch != incoming.epoch)
    {
        add(5);
    }
}
pub(crate) fn probe(site: &CallSite, callee: &Value) {
    add(6);
    let Value::Obj(callee) = callee else {
        return;
    };
    let key = Rc::as_ptr(callee) as usize;
    let epoch = CALL_IC_EPOCH.load(Relaxed);
    for old in site.entries.iter().map(Cell::get) {
        if old.callee == key {
            add(if old.epoch == epoch && epoch != u32::MAX {
                7
            } else {
                8
            });
            break;
        }
    }
}
pub(crate) fn report() {
    let c = COUNTS.with(Cell::get);
    eprintln!("[call-cache-diagnostic] {{\"scope\":\"cumulative thread\",\"optimizing_publications\":{},\"native_retirements\":{},\"inline_publications\":{},\"other_invalidations\":{},\"fills\":{},\"same_callee_stale_refills\":{},\"checked_probes\":{},\"same_callee_current_probes\":{},\"same_callee_stale_probes\":{},\"process_epoch\":{}}}",c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],CALL_IC_EPOCH.load(Relaxed));
}
