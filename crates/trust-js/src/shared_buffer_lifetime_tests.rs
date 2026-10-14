//! Shared Data Block ownership follows live wrappers, queued broadcasts and pending waits.
//! ECMA-262 e28783d5: Data Blocks, AllocateSharedArrayBuffer, DoWait.

use std::sync::{mpsc, Arc};

use crate::bytecode::Tier;
use crate::interpreter::{self, AgentChannels, BufferData};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) {
    assert!(matches!(
        engine.eval(source, false).expect("fixture parses"),
        Completion::Value(_)
    ));
}

#[test]
fn shared_buffer_backing_is_collected_after_its_last_view() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            "var view = new DataView(new SharedArrayBuffer(32, {maxByteLength:4096})); view.setUint32(0, 42);",
        );
        let (id, weak) = engine
            .interp
            .array_buffers
            .values()
            .find_map(|data| match data {
                BufferData::Shared(block) => Some((block.id(), Arc::downgrade(block))),
                _ => None,
            })
            .unwrap();
        engine.interp.gc_collect();
        assert!(weak.upgrade().is_some(), "a live view retains its block");
        eval(
            &mut engine,
            "if (view.getUint32(0) !== 42) throw Error('lost bytes'); view = undefined;",
        );
        engine.interp.gc_collect();
        assert!(
            weak.upgrade().is_none(),
            "{tier:?}: unreachable backing is freed"
        );
        assert!(interpreter::shared_mem_get(id).is_none());
    }
}

#[test]
fn queued_shared_buffer_broadcast_owns_bytes_until_receiving_agent_finishes() {
    let (broadcast_tx, broadcast_rx) = mpsc::channel();
    let (report_tx, report_rx) = mpsc::channel();
    let mut sender = Engine::new();
    sender.interp.agent = Some(Box::new(AgentChannels {
        agent_broadcast_txs: vec![broadcast_tx],
        report_rx: None,
        report_tx: report_tx.clone(),
        broadcast_rx: None,
    }));
    eval(
        &mut sender,
        "var shared = new SharedArrayBuffer(4); new Int32Array(shared)[0] = 42; $262.agent.broadcast(shared); shared = undefined;",
    );
    let (id, weak) = sender
        .interp
        .array_buffers
        .values()
        .find_map(|data| match data {
            BufferData::Shared(block) => Some((block.id(), Arc::downgrade(block))),
            _ => None,
        })
        .unwrap();
    sender.interp.gc_collect();
    drop(sender);
    assert!(
        weak.upgrade().is_some(),
        "the queued broadcast retains the block"
    );
    let receiver = std::thread::spawn(move || {
        let mut receiver = Engine::new();
        receiver.run_as_agent(
            "$262.agent.receiveBroadcast(function(shared) { $262.agent.report(Atomics.add(new Int32Array(shared), 0, 1)); });",
            broadcast_rx,
            report_tx,
        );
    });
    receiver.join().unwrap();
    assert_eq!(report_rx.recv().unwrap(), "42");
    assert!(
        weak.upgrade().is_none(),
        "agent teardown releases the last owner"
    );
    assert!(interpreter::shared_mem_get(id).is_none());
}

#[test]
fn shared_buffer_wait_owns_bytes_until_resolution_or_agent_teardown() {
    for resolve in [false, true] {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            "Atomics.waitAsync(new Int32Array(new SharedArrayBuffer(4)), 0, 0);",
        );
        let wait = &engine.interp.pending_async_waits[0];
        let id = wait.id;
        let weak = Arc::downgrade(&wait.block);
        engine.interp.gc_collect();
        assert!(engine.interp.array_buffers.is_empty());
        assert!(
            weak.upgrade().is_some(),
            "the pending wait retains the block"
        );
        if resolve {
            assert_eq!(interpreter::futex_notify(id, 0, 1, None).0, 1);
            assert!(engine.interp.settle_async_waits());
            assert!(weak.upgrade().is_none(), "settlement releases the owner");
        }
        drop(engine);
        assert!(
            weak.upgrade().is_none(),
            "agent teardown releases pending waits"
        );
        assert!(interpreter::shared_mem_get(id).is_none());
        assert_eq!(interpreter::futex_notify(id, 0, 1, None).0, 0);
    }
}
