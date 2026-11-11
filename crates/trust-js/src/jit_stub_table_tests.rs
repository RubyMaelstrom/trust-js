//! The shared stubs' table across threads (P3 stage 2, step 17; the plan's "Step 17, the shared
//! stubs made on the compile thread"): an emission off the main thread that finds a stub missing
//! from its snapshot looks it up again under the table's lock and makes it there where still
//! missing, so that threads asking for one key at once make it once; and a stub that writes
//! d8-d15 sets the callee-saved flag before it enters the table. These tests drive the table's
//! two paths directly, as the template JIT's emission on a main thread (`global_stub_address`)
//! and a job's emission on another (`emit_shared_stubs_from` with the job's snapshot), each
//! "compile thread" carrying the main thread's counters as a job does.

use super::*;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Barrier};

/// The layouts a stub is assembled with, from a fresh engine.
fn layouts() -> (crate::value::JitLayout, crate::interpreter::InterpLayout) {
    let mut engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    (layout, ilayout)
}

/// Stubs that read no chunk (none enters the direct-call sequence), of several kinds.
fn keys() -> Vec<u32> {
    let keys = vec![
        SharedStub::NameValuePtr { property_ok: true }.key(),
        SharedStub::NameValuePtr { property_ok: false }.key(),
        SharedStub::NameValuePtrIn.key(),
        SharedStub::Helper {
            idx: H_INTERRUPT as u8,
        }
        .key(),
        SharedStub::NameLoad { for_call: true }.key(),
        SharedStub::ElemStore.key(),
    ];
    assert!(keys
        .iter()
        .all(|&key| !SharedStub::from_key(key).enters_direct_sequence()));
    keys
}

/// What a job's emission reads, as its main thread reads it now.
fn reads() -> StubReads {
    StubReads {
        entry_weight: crate::opt::entry_weight(),
        inline_construct: crate::value::object_inline_alloc_site(0).is_some(),
        memory_refused: false,
        writes_d8: false,
    }
}

/// A job's emission of the stub `key` on this thread, from `snapshot`: what it met missing, and
/// the address its veneer branches to (None: it carries the stub's body).
fn emit_from(
    key: u32,
    context: &StubContext<'_>,
    snapshot: &StubTableSnapshot,
    reads: &StubReads,
) -> (MissingStubs, Option<u64>) {
    let mut a = asm::Asm::new();
    let label = a.shared_stub(key);
    a.bl_label(label);
    let mut met = MissingStubs::default();
    emit_shared_stubs_from(&mut a, context, snapshot, reads, &mut met);
    let words = &a.buf;
    let veneer = (0..words.len().saturating_sub(3))
        .find(|&k| words[k] == VENEER_LDR && words[k + 1] == VENEER_BR)
        .map(|k| u64::from(words[k + 2]) | u64::from(words[k + 3]) << 32);
    (met, veneer)
}

/// Several threads and the main thread ask for one stub at once, many times over: whichever
/// takes the table's lock first makes the stub and every other finds it, so that each round makes
/// it once (the generation moves once), and every thread's code branches to that one copy. The
/// "compile threads" emit from a snapshot taken before the stub was made, so each of them meets
/// it missing and goes to the table: the lookup under the lock is what keeps a second copy from
/// being made (in release builds the table is the process's, so these are two engines' threads;
/// test builds keep a table a main thread, which its jobs carry).
#[test]
fn threads_asking_for_one_stub_at_once_make_it_once() {
    if env_flag!("TRUST_JS_JIT_CHUNK_STUBS") {
        return;
    }
    const COMPILES: usize = 3;
    let (layout, ilayout) = layouts();
    let context = StubContext {
        layout: &layout,
        ilayout: Some(&ilayout),
        direct: None,
    };
    let addresses = TestAddresses::capture();
    let reads = reads();
    let keys = keys();
    for round in 0..40 {
        let key = keys[round % keys.len()];
        // A snapshot that lacks every stub, as a job's taken before they were made.
        forget_stubs_for_test();
        let snapshot = stub_table_snapshot();
        let (generation, stubs) = stub_table_for_test();
        assert!(stubs.is_empty());
        let barrier = Barrier::new(COMPILES + 1);
        let (main, compiles) = std::thread::scope(|scope| {
            let compiles: Vec<_> = (0..COMPILES)
                .map(|_| {
                    scope.spawn(|| {
                        let _off = OffMainCompile::enter();
                        let _addresses = TestAddresses::install(addresses);
                        barrier.wait();
                        emit_from(key, &context, &snapshot, &reads)
                    })
                })
                .collect();
            barrier.wait();
            let main = StubTable::current()
                .stub(key, &context, None)
                .unwrap_or_else(|| panic!("round {round}: the main thread's stub {key:#x}"));
            let compiles: Vec<_> = compiles
                .into_iter()
                .map(|compile| compile.join().expect("a compile thread"))
                .collect();
            (main, compiles)
        });
        let made = usize::from(main.1)
            + compiles
                .iter()
                .map(|(met, _)| met.made.len())
                .sum::<usize>();
        assert_eq!(made, 1, "round {round}: key {key:#x} made once");
        for (met, veneer) in &compiles {
            assert_eq!(*veneer, Some(main.0), "every code branches to the one copy");
            assert_eq!(
                met.len(),
                1,
                "each compile met it missing from its snapshot"
            );
            assert!(met.copies.is_empty());
        }
        let (now, stubs) = stub_table_for_test();
        assert_eq!(now, generation + 1, "the generation moved once");
        assert_eq!(stubs, vec![(key, main.0)]);
    }
}

/// A job's emission that meets a stub missing from its snapshot finds it in the table, where the
/// main thread made it since (the lookup again under the lock), and makes nothing; one that the
/// table lacks too it makes, with the main thread's next snapshot then taken again and holding it.
#[test]
fn a_job_finds_a_stub_made_since_its_snapshot_and_makes_one_still_missing() {
    if env_flag!("TRUST_JS_JIT_CHUNK_STUBS") {
        return;
    }
    let (layout, ilayout) = layouts();
    let context = StubContext {
        layout: &layout,
        ilayout: Some(&ilayout),
        direct: None,
    };
    let addresses = TestAddresses::capture();
    let reads = reads();
    let [made_here, made_there] = [
        SharedStub::NameValuePtr { property_ok: true }.key(),
        SharedStub::ElemStore.key(),
    ];
    forget_stubs_for_test();
    let snapshot = stub_table_snapshot();
    // The main thread makes one of them after the snapshot.
    let here = global_stub_address(made_here, &context).expect("made on the main thread");
    let (generation, _) = stub_table_for_test();
    let (found, made) = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _off = OffMainCompile::enter();
                let _addresses = TestAddresses::install(addresses);
                (
                    emit_from(made_here, &context, &snapshot, &reads),
                    emit_from(made_there, &context, &snapshot, &reads),
                )
            })
            .join()
            .expect("a compile thread")
    });
    assert_eq!(found.0.found, vec![made_here], "found under the lock");
    assert!(found.0.made.is_empty());
    assert_eq!(found.1, Some(here), "the main thread's copy");
    assert_eq!(made.0.made, vec![made_there], "made under the lock");
    let (now, stubs) = stub_table_for_test();
    assert_eq!(now, generation + 1, "one stub made off the main thread");
    let there = stubs
        .iter()
        .find(|&&(key, _)| key == made_there)
        .map(|&(_, address)| address);
    assert_eq!(made.1, there, "in the main thread's table");
    // The main thread's next snapshot is taken again, and holds both.
    let next = stub_table_snapshot();
    assert!(
        !Arc::ptr_eq(&next, &snapshot),
        "a stale snapshot is not reused"
    );
    for (key, address) in [(made_here, Some(here)), (made_there, there)] {
        match next.lookup(key, &context, reads.entry_weight) {
            StubLookup::At(at) => assert_eq!(Some(at), address),
            _ => panic!("{key:#x} in the next snapshot"),
        }
    }
}

/// A stub that writes d8-d15 sets the callee-saved flag before it enters the table: a thread
/// that finds it there the moment the maker released the table's lock (the maker paused right
/// after, `TEST_STUB_MADE`) reads the flag set, as a compile that branches to the stub reads it
/// at its finish. In a process of its own, where no stub set the flag before.
#[test]
fn a_stub_that_writes_d8_sets_the_flag_before_it_enters_the_table() {
    const CHILD: &str = "TRUST_JS_STUB_FLAG_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::stub_table_tests::a_stub_that_writes_d8_sets_the_flag_before_it_enters_the_table",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    if env_flag!("TRUST_JS_JIT_CHUNK_STUBS") {
        return;
    }
    assert!(
        !STUB_WRITES_CALLEE_SAVED_FP.load(Relaxed),
        "no stub of this process wrote d8-d15 yet"
    );
    let (layout, ilayout) = layouts();
    let context = StubContext {
        layout: &layout,
        ilayout: Some(&ilayout),
        direct: None,
    };
    let addresses = TestAddresses::capture();
    let reads = StubReads {
        writes_d8: true,
        ..reads()
    };
    let key = SharedStub::Helper {
        idx: H_INTERRUPT as u8,
    }
    .key();
    forget_stubs_for_test();
    let snapshot = stub_table_snapshot();
    let patience = std::time::Duration::from_secs(60);
    let (made, resume) = (
        std::sync::mpsc::channel::<()>(),
        std::sync::mpsc::channel::<()>(),
    );
    let (to_main, from_maker) = made;
    let (to_maker, from_main) = resume;
    std::thread::scope(|scope| {
        let maker = scope.spawn(|| {
            let _off = OffMainCompile::enter();
            let _addresses = TestAddresses::install(addresses);
            TEST_STUB_MADE.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    to_main.send(()).unwrap();
                    from_main.recv_timeout(patience).unwrap();
                }))
            });
            emit_from(key, &context, &snapshot, &reads)
        });
        from_maker
            .recv_timeout(patience)
            .expect("the maker made the stub");
        // The stub is in the table, the maker paused: find it as a main thread's compile does,
        // and read the flag as its finish does.
        let (address, made) = StubTable::current()
            .stub(key, &context, None)
            .expect("the stub in the table");
        assert!(!made, "found, not made again");
        assert!(
            STUB_WRITES_CALLEE_SAVED_FP.load(Relaxed),
            "a compile that finds the stub reads the flag set"
        );
        to_maker.send(()).unwrap();
        let (met, veneer) = maker.join().expect("the maker");
        assert_eq!(met.made, vec![key]);
        assert_eq!(veneer, Some(address));
    });
}
