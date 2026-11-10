//! Count epoch churn and identity-matched stale probes without changing call routing.
//! Raw entries are observed as integer metadata only, never dereferenced by diagnostics.
use super::{CallIc, CallSite, CALL_IC_EPOCH};
use crate::value::Value;
use crate::value::ValueView;
use std::{cell::Cell, panic::Location, sync::atomic::Ordering::Relaxed};

thread_local! { static COUNTS: Cell<[u64; 14]> = const { Cell::new([0; 14]) }; }
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
    let same_callee = site
        .entries
        .iter()
        .map(Cell::get)
        .find(|old| old.callee == incoming.callee && old.callee != 0);
    match same_callee {
        Some(old) if old.epoch != incoming.epoch => add(4),
        // A refill in the same epoch with other code: a superseded chunk's way.
        Some(old) if old.chunk_raw != incoming.chunk_raw => add(9),
        _ => {}
    }
}
/// A chunk was superseded (see `Chunk::supersede`).
pub(super) fn supersede() {
    add(10);
}
/// Another Realm still selected a superseded chunk (see `bytecode::note_selected`).
pub(super) fn realm_fallback() {
    add(11);
}
/// A megamorphic site's overflow admissions renewed (see `CALL_SITE_OVERFLOW_RENEWAL`).
pub(super) fn overflow_renewal() {
    add(12);
}
/// A superseded chunk's way took its function's current code (see `refill_stale`).
pub(super) fn in_place_refill() {
    add(13);
}
/// A call site went megamorphic (see `CallSite::megamorphic`).
pub(super) fn megamorphic() {
    add(8);
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
    eprintln!("[call-cache-diagnostic] {{\"scope\":\"cumulative thread\",\"native_retirements\":{},\"inline_publications\":{},\"other_invalidations\":{},\"fills\":{},\"same_callee_stale_refills\":{},\"checked_probes\":{},\"same_callee_current_probes\":{},\"same_callee_stale_probes\":{},\"megamorphic_sites\":{},\"same_epoch_new_code_refills\":{},\"supersessions\":{},\"realm_fallbacks\":{},\"overflow_renewals\":{},\"in_place_refills\":{},\"process_epoch\":{}}}",c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],CALL_IC_EPOCH.load(Relaxed));
}
