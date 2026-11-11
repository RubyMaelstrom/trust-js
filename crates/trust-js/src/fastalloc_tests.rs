//! The shell allocator's own tests. They call `ClassAlloc`'s `GlobalAlloc` methods directly: the
//! library's test binary does not register it, so nothing else in the binary uses its spans.

use super::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::mpsc;
use std::sync::{Barrier, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

static SERIAL: Mutex<()> = Mutex::new(());

/// The segments returned to the OS (`unmap_segments`).
pub(super) static RELEASED: AtomicUsize = AtomicUsize::new(0);

/// Free segments stay mapped while set: most tests read a span's header after the span empties,
/// which a segment gone would fault. `Releasing` clears it for a test.
pub(super) static KEEP_SEGMENTS: AtomicBool = AtomicBool::new(true);

/// Free segments go back to the OS, as outside the tests, while it lives.
struct Releasing;

impl Releasing {
    fn new() -> Self {
        KEEP_SEGMENTS.store(false, Ordering::Relaxed);
        Releasing
    }
}

impl Drop for Releasing {
    fn drop(&mut self) {
        KEEP_SEGMENTS.store(true, Ordering::Relaxed);
    }
}

/// Spans' memory goes back to the OS alone, or only with their segments, while it lives (as where
/// the system's pages and huge pages decide it).
struct Alone(u32);

impl Alone {
    fn set(alone: bool) -> Self {
        let bits = span_bits();
        SPAN_BITS.store(
            if alone { bits | ALONE } else { bits & !ALONE },
            Ordering::Relaxed,
        );
        Alone(bits)
    }
}

impl Drop for Alone {
    fn drop(&mut self) {
        SPAN_BITS.store(self.0, Ordering::Relaxed);
    }
}

/// One test at a time: they share the allocator's spans and pool. Each test runs its allocations
/// on threads of its own, joined before it ends (their exit drains have run by then), and gives
/// every block back; so each test starts with no block out and every span carved in the pool,
/// every class's list of spans empty.
fn serial() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let audit = audit();
    assert_eq!(
        audit.blocks_out, 0,
        "blocks out as a test starts: {audit:?}"
    );
    guard
}

/// Run `f` on a thread of its own and wait for its exit (its exit drain included: a join waits
/// for the thread's destructors, where a scope's end waits only for its closure).
fn on_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|s| s.spawn(f).join().unwrap())
}

fn layout(size: usize) -> Layout {
    Layout::from_size_align(size, 8).unwrap()
}

fn alloc(size: usize) -> usize {
    // SAFETY: a nonzero size.
    let p = unsafe { ClassAlloc.alloc(layout(size)) };
    assert!(!p.is_null());
    p as usize
}

fn free(p: usize, size: usize) {
    // SAFETY: the tests free each block they allocated once, with its layout.
    unsafe { ClassAlloc.dealloc(p as *mut u8, layout(size)) }
}

fn per_span(class: usize) -> usize {
    span_size() / class_size(class)
}

fn bytes<'a>(p: usize, len: usize) -> &'a [u8] {
    // SAFETY: the tests read blocks they hold, or spans of the allocator's that are mapped.
    unsafe { std::slice::from_raw_parts(p as *const u8, len) }
}

fn fill(p: usize, len: usize, byte: u8) {
    // SAFETY: a block the test holds.
    unsafe { (p as *mut u8).write_bytes(byte, len) }
}

/// A span's header, read under its class's lock and the pool's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct View {
    base: usize,
    class: usize,
    place: Place,
    out: u32,
    returned: u32,
    first_returned: usize,
    cursor: u32,
    end: u32,
    tail_zero: bool,
}

fn view(at: usize) -> View {
    let span = span_of(at);
    // SAFETY: the header of a span carved from a mapped segment.
    let class = unsafe { (*span).class } as usize;
    let _class = CLASSES[class].lock.lock();
    let _pool = POOL.lock.lock();
    // SAFETY: as above, under the locks that guard it.
    unsafe {
        View {
            base: base_of(span),
            class,
            place: (*span).place,
            out: (*span).out,
            returned: (*span).returned_count,
            first_returned: (*span).returned as usize,
            cursor: (*span).cursor,
            end: (*span).end,
            tail_zero: (*span).tail_zero,
        }
    }
}

fn pooled(at: usize) -> bool {
    matches!(view(at).place, Place::Committed | Place::Decommitted)
}

/// The calling thread's list of `class`, first block first (its count checked).
fn thread_list(class: usize) -> Vec<usize> {
    CACHE.with(|c| {
        let mut list = Vec::new();
        let mut p = c.heads[class].get();
        while !p.is_null() {
            list.push(p as usize);
            // SAFETY: a block of the thread's list.
            p = unsafe { *(p as *mut *mut u8) };
        }
        assert_eq!(list.len(), c.counts[class].get());
        list
    })
}

/// The calling thread's region of `class`, and whether it reads as zero.
fn thread_region(class: usize) -> (usize, usize, bool) {
    CACHE.with(|c| {
        (
            c.bump[class].get(),
            c.bump_end[class].get(),
            c.bump_zero.get() & 1 << class != 0,
        )
    })
}

/// The spans on the class's list, first first.
fn class_spans(class: usize) -> Vec<usize> {
    let _guard = CLASSES[class].lock.lock();
    let mut spans = Vec::new();
    // SAFETY: the class's list, under its lock.
    unsafe {
        let mut span = *CLASSES[class].first.get();
        while !span.is_null() {
            spans.push(base_of(span));
            span = (*span).next;
        }
    }
    spans
}

/// The pool's committed and decommitted spans, first first.
fn pool_spans() -> (Vec<usize>, Vec<usize>) {
    let _guard = POOL.lock.lock();
    let mut lists = (Vec::new(), Vec::new());
    // SAFETY: the pool's state, under its lock.
    unsafe {
        let pool = &*POOL.state.get();
        let mut span = pool.committed;
        while !span.is_null() {
            lists.0.push(base_of(span));
            span = (*span).next;
        }
        let mut span = pool.decommitted;
        while !span.is_null() {
            lists.1.push(base_of(span));
            span = (*span).next;
        }
    }
    lists
}

#[derive(Debug, Default)]
struct Audit {
    segments: usize,
    free_segments: usize,
    carved: usize,
    committed: usize,
    decommitted: usize,
    partial: usize,
    out: usize,
    blocks_out: u64,
}

/// Check every span carved against the lists it is on and its blocks' accounting, under every
/// lock; count the spans by place and the blocks out.
fn audit() -> Audit {
    let _classes: Vec<_> = CLASSES.iter().map(|c| c.lock.lock()).collect();
    let _pool = POOL.lock.lock();
    let mut audit = Audit::default();
    // SAFETY: every header is read under the locks that guard it.
    unsafe {
        let pool = &*POOL.state.get();
        let mut listed = HashSet::new();
        for class in 0..NUM_CLASSES {
            let (mut span, mut prev) = (*CLASSES[class].first.get(), null_mut());
            while !span.is_null() {
                assert_eq!((*span).place, Place::Partial);
                assert_eq!((*span).class as usize, class);
                assert_eq!((*span).prev, prev);
                assert!(listed.insert(span as usize));
                (prev, span) = (span, (*span).next);
            }
        }
        let (mut span, mut prev, mut n) = (pool.committed, null_mut(), 0);
        while !span.is_null() {
            assert_eq!((*span).place, Place::Committed);
            assert_eq!((*span).prev, prev);
            assert!(listed.insert(span as usize));
            (prev, span, n) = (span, (*span).next, n + 1);
        }
        assert_eq!((n, prev), (pool.committed_len, pool.committed_last));
        let (mut span, mut prev, mut n) = (pool.decommitted, null_mut(), 0);
        while !span.is_null() {
            assert_eq!((*span).place, Place::Decommitted);
            assert_eq!((*span).prev, prev);
            assert!(listed.insert(span as usize));
            (prev, span, n) = (span, (*span).next, n + 1);
        }
        assert_eq!(n, pool.decommitted_len);
        let mut free = HashSet::new();
        let (mut segment, mut prev) = (pool.free, null_mut());
        while !segment.is_null() {
            assert!((*segment).free);
            assert_eq!((*segment).prev_free, prev);
            assert!(free.insert(segment as usize));
            (prev, segment) = (segment, (*segment).next_free);
        }
        assert_eq!(prev, pool.free_last);
        let mut seen = 0;
        let (mut segment, mut prev) = (pool.segments, null_mut());
        while !segment.is_null() {
            audit.segments += 1;
            assert_eq!((*segment).prev, prev);
            let start = segment as usize;
            assert_eq!(start % SEGMENT, 0);
            let (mut carved, mut pooled) = (0, 0);
            for k in 1..SEGMENT / span_size() {
                let base = start + k * span_size();
                if start + SEGMENT == pool.carve_end && base >= pool.carve {
                    break;
                }
                audit.carved += 1;
                carved += 1;
                let span = span_of(base);
                match (*span).place {
                    Place::Committed | Place::Decommitted => {
                        assert_eq!((*span).out, 0);
                        pooled += 1;
                        if (*span).place == Place::Committed {
                            audit.committed += 1;
                        } else {
                            audit.decommitted += 1;
                        }
                    }
                    place => {
                        let class = (*span).class as usize;
                        let size = class_size(class);
                        let (out, cursor, end) = (
                            (*span).out as usize,
                            (*span).cursor as usize,
                            (*span).end as usize,
                        );
                        assert_eq!(end, class_end(class) as usize);
                        assert!(cursor <= end && cursor.is_multiple_of(size));
                        let (mut p, mut last, mut n) = ((*span).returned, null_mut(), 0);
                        while !p.is_null() {
                            let at = p as usize;
                            assert!(
                                at >= base
                                    && at < base + cursor
                                    && (at - base).is_multiple_of(size)
                            );
                            (last, p, n) = (p, *(p as *mut *mut u8), n + 1);
                        }
                        assert_eq!(n, (*span).returned_count as usize);
                        assert_eq!(last, (*span).returned_last);
                        assert_eq!(out + n + (end - cursor) / size, end / size, "{base:#x}");
                        assert!(out > 0, "an empty span off the pool at {base:#x}");
                        if place == Place::Partial {
                            assert!(n > 0 || cursor < end);
                            audit.partial += 1;
                        } else {
                            assert!(n == 0 && cursor == end);
                            audit.out += 1;
                        }
                        audit.blocks_out += out as u64;
                    }
                }
                if (*span).place != Place::Out {
                    seen += 1;
                    assert!(listed.contains(&(span as usize)));
                }
            }
            assert_eq!(
                ((*segment).carved, (*segment).pooled),
                (carved, pooled),
                "{start:#x}"
            );
            assert_eq!((*segment).free, carved == pooled, "{start:#x}");
            assert_eq!((*segment).free, free.contains(&start), "{start:#x}");
            audit.free_segments += usize::from((*segment).free);
            (prev, segment) = (segment, (*segment).next);
        }
        assert_eq!(seen, listed.len(), "a listed span outside the segments");
        assert_eq!(audit.free_segments, free.len(), "a free segment unlisted");
    }
    audit
}

/// Run `f` with the pool's spans taken out, so that a refill carves a fresh span, and put them
/// back after (also when `f` panics, so that the tests after it find them pooled).
fn with_empty_pool<T>(f: impl FnOnce() -> T) -> T {
    /// The spans taken: the committed ones first.
    struct Taken(Vec<*mut Span>, usize);

    impl Drop for Taken {
        fn drop(&mut self) {
            let (mut again, mut gone) = (null_mut::<Span>(), null_mut::<Span>());
            for (k, &span) in self.0.iter().enumerate() {
                // SAFETY: spans taken from the pool, which nothing else reaches.
                unsafe {
                    if k < self.1 {
                        (*span).next = again;
                        again = span;
                    } else {
                        (*span).next = gone;
                        gone = span;
                    }
                }
            }
            // SAFETY: as above.
            unsafe {
                if !again.is_null() {
                    pool_put(again);
                }
                if !gone.is_null() {
                    decommit_spans(gone);
                }
            }
        }
    }

    let (committed, decommitted) = pool_spans();
    let taken: Vec<_> = (0..committed.len() + decommitted.len())
        .map(|_| take_span().0)
        .collect();
    assert!(taken.iter().all(|s| !s.is_null()));
    let _taken = Taken(taken, committed.len());
    f()
}

#[test]
fn the_spin_lock_excludes() {
    struct Shared(spin::SpinLock, UnsafeCell<u64>);
    // SAFETY: the counter is touched only under the lock.
    unsafe impl Sync for Shared {}
    impl Shared {
        fn bump(&self) {
            let _guard = self.0.lock();
            // SAFETY: under the lock.
            unsafe { *self.1.get() += 1 };
        }
    }
    let shared = Shared(spin::SpinLock::new(), UnsafeCell::new(0));
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| (0..20_000).for_each(|_| shared.bump()));
        }
    });
    assert_eq!(shared.1.into_inner(), 80_000);
}

#[test]
fn span_headers_sit_in_their_segments_first_span() {
    let segment = 0x7f12_3440_0000_usize;
    assert_eq!(segment % SEGMENT, 0);
    let span = span_size();
    for k in 1..SEGMENT / span {
        let base = segment + k * span;
        for at in [base, base + 4321, base + span - 1] {
            assert_eq!(span_of(at) as usize, segment + k * 64);
            assert_eq!(segment_of(span_of(at)) as usize, segment);
        }
        assert_eq!(base_of(span_of(base)), base);
    }
    assert_eq!(class_end(0), span as u32);
    assert_eq!(class_end(2), (span / 48 * 48) as u32);
    assert_eq!(class_end(63), span as u32);
}

/// A span is whole pages, so that its memory can go back alone: 16 KiB, or the page where that is
/// larger (a Linux kernel of 64 KiB pages: AArch64's RHEL, NVIDIA's Grace); past 64 KiB, spans of
/// 16 KiB whose memory goes back with their segments'. Where huge pages back segments, no span's
/// memory goes back alone.
#[test]
fn spans_are_whole_pages() {
    let _serial = serial();
    assert_eq!(span_for_page(4096), (16 << 10, true));
    assert_eq!(span_for_page(16 << 10), (16 << 10, true));
    assert_eq!(span_for_page(64 << 10), (64 << 10, true));
    assert_eq!(span_for_page(128 << 10), (16 << 10, false));
    assert_eq!(span_for_page(12 << 10), (16 << 10, false));
    let (span, alone_by_page) = span_for_page(segments::page_size());
    assert_eq!(span_size(), span);
    assert_eq!(alone(), alone_by_page && !segments::huge());
    assert!(span_size().is_multiple_of(segments::page_size()) || !alone());
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    {
        use crate::gc_core::os_pages::huge_page_setting;
        let mib2 = Some(2 << 20);
        assert_eq!(
            huge_page_setting(b"always [madvise] never\n", b"2097152\n"),
            mib2
        );
        assert_eq!(
            huge_page_setting(b"[always] madvise never\n", b"2097152\n"),
            mib2
        );
        assert_eq!(
            huge_page_setting(b"always madvise [never]\n", b"2097152\n"),
            None
        );
        assert_eq!(
            huge_page_setting(b"always [madvise] never\n", b"536870912\n"),
            Some(512 << 20)
        );
        assert_eq!(huge_page_setting(b"always [madvise] never\n", b"\n"), None);
        assert_eq!(huge_page_setting(b"", b"2097152\n"), None);
    }
    eprintln!(
        "page {} KiB, spans {} KiB, huge pages {}, a span's memory back alone: {}",
        segments::page_size() >> 10,
        span_size() >> 10,
        segments::huge(),
        alone()
    );
}

#[test]
fn class_blocks_come_from_spans_by_bump() {
    let _serial = serial();
    on_thread(|| {
        for class in [0, 2, 63] {
            let size = class_size(class);
            let first = alloc(size);
            let v = view(first);
            // An empty span from the pool, whole: the thread's region.
            assert_eq!((v.class, v.place, v.base), (class, Place::Out, first));
            assert_eq!(
                (v.out as usize, v.cursor, v.end),
                (per_span(class), v.end, class_end(class))
            );
            let (from, to, _) = thread_region(class);
            assert_eq!((from, to), (first + size, first + v.end as usize));
            // The region's blocks one a miss, in address order, then a new span's.
            let blocks: Vec<_> = std::iter::once(first)
                .chain((1..per_span(class) + 3).map(|_| alloc(size)))
                .collect();
            for (k, &b) in blocks.iter().take(per_span(class)).enumerate() {
                assert_eq!(b, first + k * size);
            }
            let next = blocks[per_span(class)];
            assert_ne!(view(next).base, first);
            assert_eq!(view(next).base, next);
            assert_eq!(thread_region(class).0, next + 3 * size);
            for b in blocks {
                free(b, size);
            }
        }
        // Large and over-aligned blocks pass through to the system.
        for l in [
            Layout::from_size_align(MAX_CLASS + 1, 8).unwrap(),
            Layout::from_size_align(64, 32).unwrap(),
        ] {
            // SAFETY: a nonzero size; freed below with its layout.
            unsafe {
                let p = ClassAlloc.alloc(l);
                assert!(!p.is_null() && (p as usize).is_multiple_of(l.align()));
                p.write_bytes(7, l.size());
                ClassAlloc.dealloc(p, l);
            }
        }
    });
    audit();
}

#[test]
fn a_refill_takes_the_blocks_returned_to_the_class_spans() {
    let _serial = serial();
    let class = 5;
    let (size, n) = (class_size(class), per_span(class));
    // A thread fills a span, then frees every other block: at its exit they go back to the span,
    // which joins its class's list; the others stay out.
    let kept: Vec<usize> = on_thread(|| {
        let blocks: Vec<_> = (0..n).map(|_| alloc(size)).collect();
        let (from, to, _) = thread_region(class);
        assert_eq!(from, to, "the region is spent");
        for &b in blocks.iter().step_by(2) {
            free(b, size);
        }
        blocks.into_iter().skip(1).step_by(2).collect()
    });
    let base = view(kept[0]).base;
    let v = view(base);
    assert_eq!(
        (v.place, v.out as usize, v.returned as usize, v.cursor),
        (Place::Partial, kept.len(), n - kept.len(), v.end)
    );
    assert_eq!(class_spans(class), [base]);
    // Another thread's miss takes them all as its list (no tail to take).
    on_thread(|| {
        let p = alloc(size);
        let list = thread_list(class);
        assert_eq!(list.len(), n - kept.len() - 1);
        assert!(list
            .iter()
            .chain([&p])
            .all(|&b| (b - base).is_multiple_of(2 * size) && b - base < span_size()));
        let v = view(base);
        assert_eq!((v.place, v.out as usize, v.returned), (Place::Out, n, 0));
        assert!(class_spans(class).is_empty());
        let (from, to, _) = thread_region(class);
        assert_eq!(from, to);
        free(p, size);
    });
    assert_eq!(view(base).out as usize, kept.len());
    on_thread(move || kept.into_iter().for_each(|b| free(b, size)));
    assert!(pooled(base));
    audit();
}

#[test]
fn a_refill_gathers_a_span_worth_from_a_few_spans_at_most() {
    let _serial = serial();
    let class = 9;
    let (size, n) = (class_size(class), per_span(class));
    // Six spans with 90 blocks returned to each, then 20 with 4 each (all allocated first: the
    // thread's frees join its list, which its allocations would take). The thread's exit drain
    // gives its list back last freed first, and each span joins its class's list first: the
    // 90-block spans come first.
    assert!(n >= 90 && 4 * 90 * size < REFILL_BYTES && 5 * 90 * size >= REFILL_BYTES);
    assert!((90 + 15 * 4) * size < REFILL_BYTES && REFILL_SPANS == 16);
    let kept: Vec<usize> = on_thread(|| {
        let blocks: Vec<_> = (0..26 * n).map(|_| alloc(size)).collect();
        let mut kept = Vec::new();
        for (s, returned) in [90; 6].into_iter().chain([4; 20]).enumerate() {
            assert_eq!(view(blocks[s * n]).base, blocks[s * n]);
            for (k, &b) in blocks[s * n..(s + 1) * n].iter().enumerate() {
                if k < returned {
                    free(b, size);
                } else {
                    kept.push(b);
                }
            }
        }
        kept
    });
    assert_eq!(class_spans(class).len(), 26);
    let refills = on_thread(|| {
        let mut taken = Vec::new();
        let mut refills = Vec::new();
        for _ in 0..2 {
            // A miss refills; the list's blocks are then taken one by one, to the next miss.
            taken.push(alloc(size));
            let list = thread_list(class).len() + 1;
            taken.extend((1..list).map(|_| alloc(size)));
            refills.push((list, class_spans(class).len()));
        }
        taken.into_iter().for_each(|b| free(b, size));
        refills
    });
    // Four 90-block spans fall short of REFILL_BYTES' worth, five do not; then a 90-block span
    // and 15 4-block spans: 16 spans is as far as a refill goes.
    assert_eq!(refills, [(450, 21), (150, 5)]);
    on_thread(move || kept.into_iter().for_each(|b| free(b, size)));
    audit();
}

#[test]
fn an_overflow_gives_the_colder_half_back() {
    let _serial = serial();
    let class = 63;
    let (size, cap, n) = (class_size(class), CLASS_CAPS[class], per_span(class));
    assert!(cap == 256 && (cap / 2).is_multiple_of(n));
    on_thread(|| {
        let blocks: Vec<_> = (0..300).map(|_| alloc(size)).collect();
        for &b in &blocks[..cap] {
            free(b, size);
        }
        assert_eq!(thread_list(class).len(), cap);
        free(blocks[cap], size);
        // The hotter half stays, last freed first, with the block freed; the colder half (the
        // first 128 freed: eight whole 16 KiB spans) went back, and those spans to the pool.
        let expected: Vec<_> = blocks[cap / 2..=cap].iter().rev().copied().collect();
        assert_eq!(thread_list(class), expected);
        assert!((0..cap / 2 / n).all(|k| pooled(blocks[k * n])));
        for k in cap / 2 / n..cap / n {
            let v = view(blocks[k * n]);
            assert_eq!((v.place, v.out as usize, v.returned), (Place::Out, n, 0));
        }
        for &b in &blocks[cap + 1..] {
            free(b, size);
        }
    });
    on_thread(|| {
        // A list of blocks of 16 spans in turn: its colder half is 128 runs of one block,
        // spliced 32 at a time.
        let turn = cap / n;
        let more: Vec<_> = (0..(turn + 2) * n).map(|_| alloc(size)).collect();
        let spans: Vec<_> = (0..turn + 2).map(|s| view(more[s * n]).base).collect();
        assert!((0..turn + 2).all(|s| spans[s] == more[s * n]));
        for k in 0..n {
            for s in 0..turn {
                free(more[s * n + k], size);
            }
        }
        free(more[turn * n], size);
        for &s in &spans[..turn] {
            let v = view(s);
            assert_eq!(
                (v.place, v.out as usize, v.returned as usize),
                (Place::Partial, n - n / 2, n / 2)
            );
        }
        for &b in &more[turn * n + 1..] {
            free(b, size);
        }
    });
    audit();
}

#[test]
fn an_emptied_span_goes_to_the_pool_and_serves_another_class() {
    let _serial = serial();
    let (a, b) = (10, 20);
    let base = on_thread(|| {
        let blocks: Vec<_> = (0..per_span(a)).map(|_| alloc(class_size(a))).collect();
        let base = view(blocks[0]).base;
        for p in blocks {
            fill(p, class_size(a), 0xAB);
            free(p, class_size(a));
        }
        base
    });
    assert_eq!(view(base).place, Place::Committed);
    assert_eq!(pool_spans().0[0], base);
    on_thread(move || {
        let p = alloc(class_size(b));
        assert_eq!(p, base);
        let v = view(base);
        assert_eq!(
            (v.class, v.place, v.out as usize),
            (b, Place::Out, per_span(b))
        );
        assert!(!thread_region(b).2, "a committed span is reused dirty");
        // SAFETY: a nonzero size.
        let z = unsafe { ClassAlloc.alloc_zeroed(layout(class_size(b))) } as usize;
        assert_eq!(z, base + class_size(b));
        assert!(bytes(z, class_size(b)).iter().all(|&x| x == 0));
        free(p, class_size(b));
        free(z, class_size(b));
    });
    audit();
}

#[test]
fn trim_returns_the_pools_memory_and_it_reads_zero_after() {
    let _serial = serial();
    let _alone = Alone::set(true);
    let class = 30;
    let size = class_size(class);
    let base = on_thread(|| {
        let blocks: Vec<_> = (0..per_span(class)).map(|_| alloc(size)).collect();
        let base = view(blocks[0]).base;
        for p in blocks {
            fill(p, size, 0x5A);
            free(p, size);
        }
        assert_eq!(view(base).out as usize, per_span(class));
        trim_now();
        assert!(thread_list(class).is_empty());
        assert_eq!(thread_region(class), (0, 0, false));
        base
    });
    let v = view(base);
    assert_eq!(v.place, Place::Decommitted);
    assert!(pool_spans().0.is_empty());
    assert_eq!(pool_spans().1[0], base);
    assert_eq!(v.tail_zero, segments::DECOMMIT_ZEROES);
    #[cfg(target_os = "linux")]
    assert!(bytes(base, span_size()).iter().all(|&x| x == 0));
    // The span serves the next class that needs one, recommitted.
    let other = 31;
    on_thread(move || {
        let p = alloc(class_size(other));
        assert_eq!(p, base);
        let (_, to, zero) = thread_region(other);
        assert_eq!(zero, segments::DECOMMIT_ZEROES);
        if zero {
            assert!(bytes(base, to - base).iter().all(|&x| x == 0));
        }
        free(p, class_size(other));
    });
    audit();
}

#[test]
fn blocks_freed_on_another_thread_go_back_to_their_spans() {
    let _serial = serial();
    let class = 7;
    let size = class_size(class);
    let blocks: Vec<usize> = on_thread(|| (0..100).map(|_| alloc(size)).collect());
    // The allocating thread's exit gave its region back: the span's tail.
    let base = view(blocks[0]).base;
    let v = view(base);
    assert_eq!(
        (v.place, v.out, v.cursor as usize),
        (Place::Partial, 100, 100 * size)
    );
    // Freed on another thread, they join its list, and go back at its exit.
    let again = blocks.clone();
    on_thread(move || {
        again.iter().for_each(|&b| free(b, size));
        assert_eq!(thread_list(class).len(), 100);
        assert_eq!(view(base).out, 100, "in the freeing thread's list");
    });
    assert!(pooled(base));
    // Past the freeing thread's cap, its overflow gives the colder half back while it runs.
    let class = 63;
    let size = class_size(class);
    let blocks: Vec<usize> = on_thread(|| (0..300).map(|_| alloc(size)).collect());
    on_thread(move || {
        blocks[..257].iter().for_each(|&b| free(b, size));
        assert_eq!(thread_list(class).len(), 129);
        assert!(pooled(blocks[0]) && pooled(blocks[64]));
        assert_eq!(view(blocks[128]).out as usize, per_span(class));
        blocks[257..].iter().for_each(|&b| free(b, size));
    });
    audit();
}

#[test]
fn a_threads_exit_gives_its_lists_and_regions_back() {
    let _serial = serial();
    let classes = [3, 11, 40];
    let (kept, regions) = on_thread(|| {
        let mut kept = Vec::new();
        let mut regions = Vec::new();
        for class in classes {
            let blocks: Vec<_> = (0..10).map(|_| alloc(class_size(class))).collect();
            for &b in &blocks[..4] {
                free(b, class_size(class));
            }
            kept.extend(blocks[4..].iter().map(|&b| (class, b)));
            regions.push(thread_region(class));
        }
        (kept, regions)
    });
    for (k, &class) in classes.iter().enumerate() {
        let (from, to, zero) = regions[k];
        let v = view(from);
        assert_eq!(v.base + v.end as usize, to);
        assert_eq!(
            (
                v.place,
                v.out,
                v.returned,
                v.base + v.cursor as usize,
                v.tail_zero
            ),
            (Place::Partial, 6, 4, from, zero),
            "class {class}"
        );
    }
    // Another thread's refill takes the returned blocks as its list and the tail as its region.
    on_thread(|| {
        for (k, &class) in classes.iter().enumerate() {
            let p = alloc(class_size(class));
            assert_eq!(thread_list(class).len(), 3);
            assert_eq!(thread_region(class), regions[k]);
            let v = view(p);
            assert_eq!((v.place, v.out as usize), (Place::Out, per_span(class)));
            free(p, class_size(class));
        }
    });
    on_thread(move || {
        kept.into_iter()
            .for_each(|(class, b)| free(b, class_size(class)))
    });
    for (from, _, _) in regions {
        assert!(pooled(from));
    }
    audit();
}

/// What a thread's last destructor saw: whether the exit drain had run before it, the span's out
/// count before and after a late free, the late allocation's block, and a new span's header.
type Late = (bool, u32, u32, usize, usize, View);

struct LateWork(Cell<usize>, Cell<Option<mpsc::Sender<Late>>>);

impl Drop for LateWork {
    fn drop(&mut self) {
        let class = 12;
        let size = class_size(class);
        let p = self.0.get();
        let drained = CACHE.with(|c| {
            let drained = c.exited.get();
            if !drained {
                // Where destructors run in another order, run the drain here.
                c.exited.set(true);
                drain(c);
            }
            drained
        });
        let before = view(p).out;
        free(p, size);
        let after = view(p);
        assert_eq!(after.first_returned, p, "back on its span at once");
        // An allocation takes one block from its class's spans, or sets a span up for it.
        let again = alloc(size);
        assert_eq!(again, p);
        free(again, size);
        let lone = alloc(class_size(13));
        let lone_view = view(lone);
        free(lone, class_size(13));
        let sender = self.1.take().unwrap();
        sender
            .send((drained, before, after.out, again, lone, lone_view))
            .unwrap();
    }
}

thread_local! {
    static LATE_WORK: LateWork = const { LateWork(Cell::new(0), Cell::new(None)) };
}

#[test]
fn frees_and_allocations_after_the_exit_drain_go_to_the_spans() {
    let _serial = serial();
    let (sender, receiver) = mpsc::channel();
    on_thread(move || {
        // Registered before the exit drain, the work's destructor runs after it.
        LATE_WORK.with(|w| w.1.set(Some(sender)));
        let p = alloc(class_size(12));
        free(alloc(class_size(12)), class_size(12));
        LATE_WORK.with(|w| w.0.set(p));
    });
    let (drained, before, after, again, lone, lone_view) = receiver.recv().unwrap();
    if cfg!(target_os = "linux") {
        assert!(
            drained,
            "glibc runs a thread's destructors last registered first"
        );
    }
    // The late block was the last out of its span (the drain gave back the list's block and the
    // region): the span went to the pool, served the late allocation as a span set up for one
    // block, went back, then served another class's the same way.
    assert_eq!((before, after), (1, 0));
    assert_eq!(lone, again);
    assert_eq!(
        (
            lone_view.base,
            lone_view.class,
            lone_view.out,
            lone_view.place
        ),
        (lone, 13, 1, Place::Partial)
    );
    assert_eq!(lone_view.cursor as usize, class_size(13));
    assert!(pooled(again));
    audit();
}

#[test]
fn alloc_zeroed_clears_only_blocks_that_may_be_dirty() {
    let _serial = serial();
    with_empty_pool(|| {
        on_thread(|| {
            let class = 15;
            let size = class_size(class);
            // A fresh span: its blocks read as zero, uncleared.
            let (p, zero) = alloc_miss(class);
            let p = p as usize;
            assert!(zero && view(p).base == p && thread_region(class).2);
            assert!(bytes(p, span_size()).iter().all(|&x| x == 0));
            // SAFETY: a nonzero size.
            let z = unsafe { ClassAlloc.alloc_zeroed(layout(250)) } as usize;
            assert_eq!(z, p + size);
            // Reused from the thread's list, a block is cleared.
            fill(z, size, 0xCD);
            free(z, 250);
            // SAFETY: as above.
            let again = unsafe { ClassAlloc.alloc_zeroed(layout(250)) } as usize;
            assert_eq!(again, z);
            assert!(bytes(again, 250).iter().all(|&x| x == 0));
            // Large blocks come zeroed from the system.
            let big = Layout::from_size_align(5000, 8).unwrap();
            // SAFETY: a nonzero size; freed below.
            unsafe {
                let b = ClassAlloc.alloc_zeroed(big);
                assert!(std::slice::from_raw_parts(b, 5000).iter().all(|&x| x == 0));
                ClassAlloc.dealloc(b, big);
            }
            free(p, size);
            free(again, 250);
        })
    });
    audit();
}

#[test]
fn realloc_keeps_blocks_within_a_class_and_moves_them_across() {
    let _serial = serial();
    on_thread(|| {
        let pattern = |p: usize, n: usize| {
            for k in 0..n {
                // SAFETY: a block the test holds, of `n` bytes at least.
                unsafe { *((p + k) as *mut u8) = (k * 7 + 3) as u8 };
            }
        };
        let intact = |p: usize, n: usize| (0..n).all(|k| bytes(p, n)[k] == (k * 7 + 3) as u8);
        let p = alloc(20);
        pattern(p, 20);
        // SAFETY: each block reallocated with the layout it has.
        unsafe {
            let q = ClassAlloc.realloc(p as *mut u8, layout(20), 30) as usize;
            assert_eq!(q, p, "within a class");
            let r = ClassAlloc.realloc(q as *mut u8, layout(30), 100) as usize;
            assert!(r != q && intact(r, 20));
            assert_eq!(thread_list(1)[0], q, "the old block back on the list");
            pattern(r, 100);
            let s = ClassAlloc.realloc(r as *mut u8, layout(100), 5000) as usize;
            assert!(intact(s, 100));
            pattern(s, 5000);
            let t = ClassAlloc.realloc(s as *mut u8, layout(5000), 64) as usize;
            assert!(intact(t, 64));
            let u = ClassAlloc.realloc(t as *mut u8, layout(64), 16) as usize;
            assert!(intact(u, 16));
            free(u, 16);
        }
    });
    audit();
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A block of the stress test: its address, layout and the tag its contents derive from.
#[derive(Clone, Copy)]
struct Held {
    at: usize,
    size: usize,
    align: usize,
    tag: u8,
}

impl Held {
    fn layout(&self) -> Layout {
        Layout::from_size_align(self.size, self.align).unwrap()
    }

    fn byte(&self, k: usize) -> u8 {
        self.tag ^ (k as u8).wrapping_mul(13)
    }

    fn write(&self) {
        for k in 0..self.size {
            // SAFETY: the block is the holder's.
            unsafe { *((self.at + k) as *mut u8) = self.byte(k) };
        }
    }

    fn check(&self, len: usize) {
        let contents = bytes(self.at, len);
        for (k, &b) in contents.iter().enumerate() {
            assert_eq!(
                b,
                self.byte(k),
                "block {:#x} size {} byte {k}",
                self.at,
                self.size
            );
        }
    }

    fn free(self) {
        self.check(self.size);
        // SAFETY: the holder's block, with its layout.
        unsafe { ClassAlloc.dealloc(self.at as *mut u8, self.layout()) }
    }
}

#[test]
fn a_randomized_multithreaded_stress_keeps_every_block_intact() {
    let _serial = serial();
    const THREADS: usize = 6;
    const STEPS: usize = 30_000;
    let (senders, receivers): (Vec<_>, Vec<_>) =
        (0..THREADS).map(|_| mpsc::channel::<Vec<Held>>()).unzip();
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|s| {
        let mut threads = Vec::new();
        for (t, inbox) in receivers.into_iter().enumerate() {
            let (senders, barrier) = (senders.clone(), &barrier);
            threads.push(s.spawn(move || {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((t as u64 + 1) * 0x1234_5678_9ABC_DEF1));
                let mut held: Vec<Held> = Vec::new();
                let adopt = |held: &mut Vec<Held>, gift: Vec<Held>, rng: &mut Rng| {
                    for h in gift {
                        h.check(h.size);
                        if rng.below(2) == 0 {
                            held.push(h);
                        } else {
                            h.free();
                        }
                    }
                };
                for step in 0..STEPS {
                    match rng.below(100) {
                        0..=44 => {
                            let (size, align) = match rng.below(100) {
                                0..=64 => (1 + rng.below(128), 8),
                                65..=89 => (129 + rng.below(896), 16),
                                90..=96 => (1025 + rng.below(7168), 8),
                                _ => (1 + rng.below(512), 32 << rng.below(2)),
                            };
                            let l = Layout::from_size_align(size, align).unwrap();
                            let zeroed = rng.below(8) == 0;
                            // SAFETY: a nonzero size.
                            let at = unsafe {
                                if zeroed {
                                    ClassAlloc.alloc_zeroed(l)
                                } else {
                                    ClassAlloc.alloc(l)
                                }
                            } as usize;
                            assert!(at != 0 && at.is_multiple_of(align));
                            if zeroed {
                                assert!(bytes(at, size).iter().all(|&x| x == 0));
                            }
                            let h = Held {
                                at,
                                size,
                                align,
                                tag: rng.next() as u8,
                            };
                            h.write();
                            held.push(h);
                        }
                        45..=79 if !held.is_empty() => {
                            let k = rng.below(held.len());
                            held.swap_remove(k).free();
                        }
                        80..=89 if !held.is_empty() => {
                            let k = rng.below(held.len());
                            let h = held[k];
                            h.check(h.size);
                            let size = if rng.below(2) == 0 {
                                1 + rng.below(1024)
                            } else {
                                1 + rng.below(4096)
                            };
                            // SAFETY: the block with its layout, and a nonzero size.
                            let at =
                                unsafe { ClassAlloc.realloc(h.at as *mut u8, h.layout(), size) }
                                    as usize;
                            assert!(at != 0);
                            let moved = Held { at, ..h };
                            moved.check(h.size.min(size));
                            let resized = Held {
                                at,
                                size,
                                align: h.align,
                                tag: rng.next() as u8,
                            };
                            resized.write();
                            held[k] = resized;
                        }
                        90..=94 if !held.is_empty() => {
                            let to = rng.below(THREADS);
                            let n = 1 + rng.below(16.min(held.len()));
                            let gift: Vec<_> = (0..n)
                                .map(|_| {
                                    let k = rng.below(held.len());
                                    held.swap_remove(k)
                                })
                                .collect();
                            senders[to].send(gift).unwrap();
                        }
                        95..=98 => {
                            if let Ok(gift) = inbox.try_recv() {
                                adopt(&mut held, gift, &mut rng);
                            }
                        }
                        99 if t == 0 && step.is_multiple_of(4) => trim_now(),
                        _ => {}
                    }
                }
                held.drain(..).for_each(Held::free);
                barrier.wait();
                drop(senders);
                while let Ok(gift) = inbox.try_recv() {
                    gift.into_iter().for_each(Held::free);
                }
            }));
        }
        // A scope's end waits for its threads' closures, not their exits (their destructors,
        // the exit drain among them): join waits for those.
        threads.into_iter().for_each(|t| t.join().unwrap());
    });
    let audit = audit();
    assert_eq!(
        (audit.blocks_out, audit.out, audit.partial),
        (0, 0, 0),
        "{audit:?}"
    );
    assert!(audit.carved > 0);
}

/// Blocks a thread-local destructor frees and allocates at the thread's exit: after the exit
/// drain where it was registered before the thread's first allocation (glibc runs them last
/// registered first), before it otherwise.
struct ChurnLate {
    held: RefCell<Vec<Held>>,
    to: RefCell<Option<mpsc::Sender<Vec<Held>>>>,
    seed: Cell<u64>,
}

impl Drop for ChurnLate {
    fn drop(&mut self) {
        let held = std::mem::take(&mut *self.held.borrow_mut());
        let mut rng = Rng(self.seed.get() | 1);
        let to = self.to.borrow_mut().take();
        let mut gift = Vec::new();
        for h in held {
            if rng.below(2) == 0 {
                h.free();
            } else {
                gift.push(h);
            }
        }
        for _ in 0..rng.below(40) {
            let size = 1 + rng.below(1024);
            let zeroed = rng.below(3) == 0;
            let l = Layout::from_size_align(size, 8).unwrap();
            // SAFETY: a nonzero size.
            let at = unsafe {
                if zeroed {
                    ClassAlloc.alloc_zeroed(l)
                } else {
                    ClassAlloc.alloc(l)
                }
            } as usize;
            assert!(at != 0);
            if zeroed {
                assert!(bytes(at, size).iter().all(|&x| x == 0), "late zeroed");
            }
            let h = Held {
                at,
                size,
                align: 8,
                tag: rng.next() as u8,
            };
            h.write();
            if rng.below(2) == 0 {
                h.free();
            } else {
                gift.push(h);
            }
        }
        if rng.below(4) == 0 {
            trim_now();
        }
        match to {
            Some(to) if !gift.is_empty() => {
                if let Err(e) = to.send(gift) {
                    e.0.into_iter().for_each(Held::free);
                }
            }
            _ => gift.into_iter().for_each(Held::free),
        }
    }
}

thread_local! {
    static CHURN_LATE: ChurnLate = const {
        ChurnLate {
            held: RefCell::new(Vec::new()),
            to: RefCell::new(None),
            seed: Cell::new(0),
        }
    };
}

fn churn_alloc(rng: &mut Rng, hot: bool) -> Held {
    let (size, align) = if hot {
        ([8, 16, 90, 1024][rng.below(4)], 8)
    } else {
        match rng.below(100) {
            0..=59 => (1 + rng.below(256), 8),
            60..=84 => (257 + rng.below(768), 16),
            85..=89 => ([15, 16, 17, 1023, 1024, 1025][rng.below(6)], 16),
            90..=95 => (1025 + rng.below(4096), 8),
            _ => (1 + rng.below(512), 32 << rng.below(2)),
        }
    };
    let l = Layout::from_size_align(size, align).unwrap();
    let zeroed = rng.below(6) == 0;
    // SAFETY: a nonzero size.
    let at = unsafe {
        if zeroed {
            ClassAlloc.alloc_zeroed(l)
        } else {
            ClassAlloc.alloc(l)
        }
    } as usize;
    assert!(at != 0 && at.is_multiple_of(align));
    if zeroed {
        assert!(
            bytes(at, size).iter().all(|&x| x == 0),
            "zeroed block of {size} at {at:#x} reads nonzero"
        );
    }
    let h = Held {
        at,
        size,
        align,
        tag: rng.next() as u8,
    };
    h.write();
    h
}

fn churn_realloc(rng: &mut Rng, h: Held) -> Held {
    h.check(h.size);
    let size = if rng.below(2) == 0 {
        [1, 15, 16, 17, 1023, 1024, 1025, 2000][rng.below(8)]
    } else {
        1 + rng.below(3000)
    };
    // SAFETY: the block with its layout, and a nonzero size.
    let at = unsafe { ClassAlloc.realloc(h.at as *mut u8, h.layout(), size) } as usize;
    assert!(at != 0 && at.is_multiple_of(h.align));
    Held { at, ..h }.check(h.size.min(size));
    let resized = Held {
        at,
        size,
        align: h.align,
        tag: rng.next() as u8,
    };
    resized.write();
    resized
}

fn churn_child(seed: u64, late_first: bool, hot: bool, senders: Vec<mpsc::Sender<Vec<Held>>>) {
    let mut rng = Rng(seed | 1);
    let arm = |rng: &mut Rng| {
        let (s, to) = (rng.next(), rng.below(senders.len()));
        CHURN_LATE.with(|l| {
            l.seed.set(s);
            *l.to.borrow_mut() = Some(senders[to].clone());
        });
    };
    let mut mine: Vec<Held> = Vec::new();
    if late_first {
        arm(&mut rng);
    } else {
        mine.push(churn_alloc(&mut rng, hot));
        arm(&mut rng);
    }
    for _ in 0..300 + rng.below(1500) {
        match rng.below(20) {
            0..=10 => mine.push(churn_alloc(&mut rng, hot)),
            11..=14 if !mine.is_empty() => {
                let k = rng.below(mine.len());
                mine.swap_remove(k).free();
            }
            15..=16 if !mine.is_empty() => {
                let k = rng.below(mine.len());
                mine[k] = churn_realloc(&mut rng, mine[k]);
            }
            17..=18 if !mine.is_empty() => {
                let n = 1 + rng.below(32.min(mine.len()));
                let gift: Vec<_> = (0..n)
                    .map(|_| {
                        let k = rng.below(mine.len());
                        mine.swap_remove(k)
                    })
                    .collect();
                senders[rng.below(senders.len())].send(gift).unwrap();
            }
            _ => {}
        }
    }
    for h in mine {
        match rng.below(4) {
            0 => CHURN_LATE.with(|l| l.held.borrow_mut().push(h)),
            1 => senders[rng.below(senders.len())].send(vec![h]).unwrap(),
            _ => h.free(),
        }
    }
}

fn churn_stress(hot: bool, spawners: usize, children: usize) -> Audit {
    const LIVE: usize = 4;
    let (senders, receivers): (Vec<_>, Vec<_>) =
        (0..LIVE).map(|_| mpsc::channel::<Vec<Held>>()).unzip();
    std::thread::scope(|s| {
        let mut threads = Vec::new();
        for (t, inbox) in receivers.into_iter().enumerate() {
            threads.push(s.spawn(move || {
                let mut rng = Rng(0xA5A5_0000 + t as u64 * 7919);
                let mut kept: Vec<Held> = Vec::new();
                loop {
                    match inbox.recv_timeout(Duration::from_micros(200)) {
                        Ok(gift) => {
                            for h in gift {
                                h.check(h.size);
                                if rng.below(3) == 0 {
                                    kept.push(h);
                                } else {
                                    h.free();
                                }
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                    if kept.len() > 3000 {
                        while kept.len() > 200 {
                            let k = rng.below(kept.len());
                            kept.swap_remove(k).free();
                        }
                    }
                    if t < 2 && rng.below(300) == 0 {
                        trim_now();
                    }
                    if rng.below(3) == 0 {
                        kept.push(churn_alloc(&mut rng, hot));
                    }
                    if !kept.is_empty() && rng.below(3) == 0 {
                        let k = rng.below(kept.len());
                        kept.swap_remove(k).free();
                    }
                }
                kept.drain(..).for_each(Held::free);
            }));
        }
        for sp in 0..spawners {
            let senders = senders.clone();
            threads.push(s.spawn(move || {
                let mut rng = Rng(0x5EED_0000 + sp as u64 * 104_729);
                for _ in 0..children {
                    let (seed, late_first) = (rng.next(), rng.below(2) == 0);
                    let senders = senders.clone();
                    std::thread::spawn(move || churn_child(seed, late_first, hot, senders))
                        .join()
                        .unwrap();
                }
            }));
        }
        if !hot {
            // Bursts past the pool's committed bound while the others run: its oldest spans are
            // decommitted, then served again as another class's zeroed blocks.
            threads.push(s.spawn(|| {
                for round in 0..3 {
                    let size = 1024;
                    let blocks: Vec<usize> = (0..6 * 1024)
                        .map(|_| {
                            let p = alloc(size);
                            // SAFETY: a block of `size` bytes the test holds.
                            unsafe { *(p as *mut u64) = p as u64 ^ 0x5555 };
                            p
                        })
                        .collect();
                    for &p in &blocks {
                        // SAFETY: as above.
                        assert_eq!(unsafe { *(p as *const u64) }, p as u64 ^ 0x5555);
                        free(p, size);
                    }
                    let other = 656 - round * 16;
                    let zs: Vec<usize> = (0..4096)
                        .map(|_| {
                            // SAFETY: a nonzero size.
                            let z = unsafe { ClassAlloc.alloc_zeroed(layout(other)) } as usize;
                            assert!(z != 0);
                            assert!(
                                bytes(z, other).iter().all(|&x| x == 0),
                                "burst zeroed at {z:#x}"
                            );
                            fill(z, other, 0xEE);
                            z
                        })
                        .collect();
                    for z in zs {
                        assert!(bytes(z, other).iter().all(|&x| x == 0xEE));
                        free(z, other);
                    }
                }
            }));
        }
        drop(senders);
        threads.into_iter().for_each(|t| t.join().unwrap());
    });
    let audit = audit();
    assert_eq!(
        (audit.blocks_out, audit.out, audit.partial),
        (0, 0, 0),
        "{audit:?}"
    );
    audit
}

/// Threads that churn blocks of every size, hand them to each other and exit, with thread-local
/// destructors that free and allocate after their exit drains, while others trim: past the
/// pool's bound and at the trims, empty spans' memory goes back alone (the review's stress, as on
/// Windows and macOS, or where no huge page backs a segment).
#[test]
fn threads_that_churn_exit_and_trim_keep_every_block_intact() {
    let _serial = serial();
    let _alone = Alone::set(true);
    let audit = churn_stress(false, 3, 40);
    eprintln!("churn, spans back alone: {audit:?}");
    assert!(audit.decommitted > 0, "{audit:?}");
}

/// The same, free segments going back to the OS whole as well (spans back alone, and not, as
/// where huge pages back segments); with every block back, a trim returns every segment.
#[test]
fn threads_that_churn_exit_and_trim_keep_every_block_intact_as_segments_go() {
    let _serial = serial();
    let _releasing = Releasing::new();
    for alone in [true, false] {
        let _alone = Alone::set(alone);
        let released = RELEASED.load(Ordering::Relaxed);
        let churn = churn_stress(false, 3, 40);
        let during = RELEASED.load(Ordering::Relaxed) - released;
        trim_now();
        let after = audit();
        eprintln!("churn, spans back alone {alone}: {during} segments released, {churn:?}");
        assert_eq!(after.segments, 0, "{after:?}");
        assert!(RELEASED.load(Ordering::Relaxed) - released >= churn.segments);
    }
}

/// Threads that churn a few hot sizes.
#[test]
fn threads_that_churn_hot_sizes_keep_every_block_intact() {
    let _serial = serial();
    let _releasing = Releasing::new();
    let audit = churn_stress(true, 4, 40);
    eprintln!("churn, hot sizes: {audit:?}");
}

/// A span's memory going back to the OS leaves the spans beside it alone: here a span at a 64 KiB
/// boundary is emptied and trimmed while the next span's blocks are live (on a kernel of 64 KiB
/// pages a 16 KiB span's discard took three neighbours' blocks with it).
#[test]
fn a_spans_discard_keeps_its_neighbours() {
    let _serial = serial();
    for alone in [true, false] {
        let _alone = Alone::set(alone);
        with_empty_pool(|| {
            on_thread(|| {
                let class = 63;
                let (size, n) = (class_size(class), per_span(class));
                let blocks: Vec<usize> = (0..12 * n).map(|_| alloc(size)).collect();
                for &b in &blocks {
                    fill(b, size, 0x77);
                }
                let s = (0..11)
                    .find(|&s| {
                        blocks[s * n].is_multiple_of(64 * 1024)
                            && blocks[(s + 1) * n] == blocks[s * n] + span_size()
                    })
                    .expect("a span at a 64 KiB boundary with its neighbour carved next");
                for &b in &blocks[s * n..(s + 1) * n] {
                    free(b, size);
                }
                trim_now();
                let v = view(blocks[s * n]);
                assert_eq!(
                    v.place,
                    if alone {
                        Place::Decommitted
                    } else {
                        Place::Committed
                    }
                );
                let lost: Vec<usize> = blocks[(s + 1) * n..(s + 2) * n]
                    .iter()
                    .copied()
                    .filter(|&b| !bytes(b, size).iter().all(|&x| x == 0x77))
                    .collect();
                for (k, &b) in blocks.iter().enumerate() {
                    if k / n != s {
                        free(b, size);
                    }
                }
                assert!(
                    lost.is_empty(),
                    "{} live blocks of the next span lost their contents: first {:#x}",
                    lost.len(),
                    lost[0]
                );
            })
        });
    }
    audit();
}

/// A segment whose spans are all in the pool is free: past the pool's bound the oldest free
/// segments go back to the OS whole, their spans off the pool's lists, as many as bring the
/// committed spans to the bound; at a trim all of them. A segment with a block out stays, its
/// empty spans committed where no span's memory goes back alone.
#[test]
fn a_free_segment_goes_back_to_the_os_whole() {
    let _serial = serial();
    let _releasing = Releasing::new();
    let _alone = Alone::set(false);
    trim_now();
    let start = audit();
    assert_eq!(start.free_segments, 0);
    let pooled = start.committed + start.decommitted;
    let per_segment = SEGMENT / span_size() - 1;
    let bound = POOL_COMMITTED_BYTES >> span_shift();
    let class = 63;
    let (size, n) = (class_size(class), per_span(class));
    // The pool's spans and three segments' worth more: two segments at least carved for these
    // blocks alone. The last block stays out, and its segment with it.
    let blocks: Vec<usize> = on_thread(|| {
        (0..(pooled + 3 * per_segment) * n)
            .map(|_| alloc(size))
            .collect()
    });
    let kept = *blocks.last().unwrap();
    let segments = audit().segments;
    let released = RELEASED.load(Ordering::Relaxed);
    on_thread(|| {
        blocks[..blocks.len() - 1]
            .iter()
            .for_each(|&b| free(b, size))
    });
    let past_bound = RELEASED.load(Ordering::Relaxed) - released;
    let a = audit();
    assert!(past_bound >= 1, "{a:?}");
    assert!(a.committed <= bound || a.free_segments == 0, "{a:?}");
    assert_eq!(a.segments, segments - past_bound);
    assert!(
        a.decommitted <= start.decommitted,
        "a span's memory went back alone: {a:?}"
    );
    trim_now();
    let all = RELEASED.load(Ordering::Relaxed) - released;
    let b = audit();
    assert!(all >= 2 && b.free_segments == 0, "{b:?}");
    assert_eq!(b.segments, segments - all);
    // The kept block's segment and span stay: the span on its class's list.
    let v = view(kept);
    assert_eq!((v.place, v.out), (Place::Partial, 1));
    on_thread(|| free(kept, size));
    audit();
}

/// Past the pool's bound, where no span's memory goes back alone (huge pages back segments), the
/// empty spans of segments with a block out stay committed: neither discarded alone nor returned,
/// at the bound or at a trim.
#[test]
fn past_the_bound_used_segments_keep_their_empty_spans_where_huge_pages_back_them() {
    let _serial = serial();
    let _releasing = Releasing::new();
    let _alone = Alone::set(false);
    let per_segment = SEGMENT / span_size() - 1;
    let bound = POOL_COMMITTED_BYTES >> span_shift();
    let class = 63;
    let (size, n) = (class_size(class), per_span(class));
    with_empty_pool(|| {
        // Three segments' worth of spans carved for these blocks, a block of each segment they
        // lie in kept out: once the others are back, past the bound, no segment of theirs is
        // free.
        let blocks: Vec<usize> =
            on_thread(|| (0..3 * per_segment * n).map(|_| alloc(size)).collect());
        let mut segments = HashSet::new();
        let kept: HashSet<usize> = blocks
            .iter()
            .copied()
            .filter(|&b| segments.insert(b & !(SEGMENT - 1)))
            .collect();
        let released = RELEASED.load(Ordering::Relaxed);
        on_thread(|| {
            for &b in blocks.iter().filter(|b| !kept.contains(b)) {
                free(b, size);
            }
        });
        let (committed, decommitted) = pool_spans();
        assert!(
            committed.len() > bound,
            "{} spans committed",
            committed.len()
        );
        assert!(decommitted.is_empty());
        trim_now();
        let (again, decommitted) = pool_spans();
        assert_eq!((again.len(), decommitted.len()), (committed.len(), 0));
        assert_eq!(RELEASED.load(Ordering::Relaxed), released);
        on_thread(|| kept.iter().for_each(|&b| free(b, size)));
        assert!(RELEASED.load(Ordering::Relaxed) > released);
    });
    audit();
}

/// The loader tells the process's end: ntdll exports `RtlDllShutdownInProgress`, FALSE while the
/// process runs.
#[cfg(windows)]
#[test]
fn the_loader_tells_whether_the_process_is_ending() {
    assert_ne!(
        ending::resolve(),
        ending::ABSENT,
        "ntdll exports no RtlDllShutdownInProgress"
    );
    assert!(!ending::process_is_shutting_down());
}

/// A thread-local whose destructor runs after the thread's exit drain's (registered first): at
/// Windows' process end it writes whether the loader says the process is ending (through
/// WriteFile: nothing allocated), and frees and allocates a block of `EXIT_CLASS` through the
/// thread's list, which the skipped drain left as it was.
#[cfg(windows)]
struct AfterTheDrain;

#[cfg(windows)]
const EXIT_CLASS: usize = 4;

#[cfg(windows)]
impl Drop for AfterTheDrain {
    fn drop(&mut self) {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetStdHandle(handle: u32) -> *mut std::ffi::c_void;
            fn WriteFile(
                file: *mut std::ffi::c_void,
                buffer: *const u8,
                len: u32,
                written: *mut u32,
                overlapped: *mut std::ffi::c_void,
            ) -> i32;
        }
        let ending = ending::process_is_shutting_down();
        let p = alloc(class_size(EXIT_CLASS));
        free(p, class_size(EXIT_CLASS));
        let line: &[u8] = if ending {
            b"[after the drain] shutting down: 1\n"
        } else {
            b"[after the drain] shutting down: 0\n"
        };
        let mut written = 0;
        const STD_ERROR_HANDLE: u32 = -12i32 as u32;
        // SAFETY: the process's standard error, a buffer of `len` bytes.
        unsafe {
            WriteFile(
                GetStdHandle(STD_ERROR_HANDLE),
                line.as_ptr(),
                line.len() as u32,
                &mut written,
                std::ptr::null_mut(),
            )
        };
    }
}

#[cfg(windows)]
thread_local! {
    static AFTER_THE_DRAIN: AfterTheDrain = const { AfterTheDrain };
}

/// Windows' ExitProcess ends the other threads wherever they are, a lock held held for good, and
/// then runs the exiting thread's thread-local destructors: its drain must not wait for the lock.
/// A child process (this test binary, running this test alone) lets a thread take the pool's lock
/// and the lock of the class the exiting thread's list holds, and exits; the loader says the
/// process is ending in the destructors that run after the drain's, and the thread's list still
/// serves it there.
#[cfg(windows)]
#[test]
fn a_process_exit_while_another_thread_holds_a_lock_ends() {
    use std::io::Read;
    use std::time::Instant;
    const CHILD: &str = "TRUST_JS_FASTALLOC_EXIT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        // Registered before the exit drain, its destructor runs after the drain's.
        AFTER_THE_DRAIN.with(|_| ());
        free(alloc(class_size(EXIT_CLASS)), class_size(EXIT_CLASS));
        let (locked_tx, locked) = mpsc::channel();
        std::thread::spawn(move || {
            let _class = CLASSES[EXIT_CLASS].lock.lock();
            let _pool = POOL.lock.lock();
            locked_tx.send(()).unwrap();
            loop {
                std::thread::park();
            }
        });
        locked.recv().unwrap();
        std::process::exit(7);
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "fastalloc::tests::a_process_exit_while_another_thread_holds_a_lock_ends",
            "--exact",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            panic!("the child process hung at its exit");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut err = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();
    assert_eq!(status.code(), Some(7), "{err}");
    assert!(err.contains("[after the drain] shutting down: 1"), "{err}");
}
