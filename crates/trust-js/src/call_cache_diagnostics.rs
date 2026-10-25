//! Count epoch churn and identity-matched stale probes without changing call routing.
//! Raw entries are observed as integer metadata only, never dereferenced by diagnostics.
use super::{CallIc, CallSite, CALL_IC_EPOCH};
use crate::value::Value;
use crate::value::ValueView;
use std::{cell::Cell, panic::Location, sync::atomic::Ordering::Relaxed};

thread_local! { static COUNTS: Cell<[u64; 8]> = const { Cell::new([0; 8]) }; }
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
    add(if file.ends_with("jit_cache.rs") {
        0
    } else if file.ends_with("interpreter.rs") {
        1
    } else {
        2
    });
}
pub(super) fn fill(site: &CallSite, incoming: CallIc) {
    add(3);
    if site
        .entries
        .iter()
        .map(Cell::get)
        .any(|old| old.callee == incoming.callee && old.callee != 0 && old.epoch != incoming.epoch)
    {
        add(4);
    }
}
pub(crate) fn probe(site: &CallSite, callee: &Value) {
    add(5);
    let ValueView::Obj(callee) = &(callee).view() else {
        return;
    };
    let key = crate::value::Gc::as_ptr(callee) as usize;
    let epoch = CALL_IC_EPOCH.load(Relaxed);
    for old in site.entries.iter().map(Cell::get) {
        if old.callee == key {
            add(if old.epoch == epoch && epoch != u32::MAX {
                6
            } else {
                7
            });
            break;
        }
    }
}
pub(crate) fn report() {
    let c = COUNTS.with(Cell::get);
    eprintln!("[call-cache-diagnostic] {{\"scope\":\"cumulative thread\",\"native_retirements\":{},\"inline_publications\":{},\"other_invalidations\":{},\"fills\":{},\"same_callee_stale_refills\":{},\"checked_probes\":{},\"same_callee_current_probes\":{},\"same_callee_stale_probes\":{},\"process_epoch\":{}}}",c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],CALL_IC_EPOCH.load(Relaxed));
}
