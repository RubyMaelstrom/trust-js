use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use trust_js_runtime::{Completion, ConsoleOut, Runtime};

#[derive(Clone, Default)]
struct Captured(Rc<RefCell<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn lines(&self) -> Vec<String> {
        String::from_utf8(self.0.borrow().clone())
            .expect("utf8 console output")
            .lines()
            .map(str::to_string)
            .collect()
    }
}

#[test]
fn worker_threads_exchange_structured_messages() {
    let mut runtime = Runtime::new();
    let out = Captured::default();
    runtime.engine().op_state().put(ConsoleOut {
        out: Box::new(out.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
        const { Worker } = require("node:worker_threads");
        const worker = new Worker(`
            const { parentPort, workerData, isMainThread, threadId } = require("node:worker_threads");
            parentPort.on("message", message => {
                parentPort.postMessage({
                    answer: message.value + workerData.offset,
                    worker: !isMainThread && threadId > 0,
                });
                parentPort.close();
            });
        `, { eval: true, workerData: { offset: 2 } });
        worker.on("online", () => worker.postMessage({ value: 40 }));
        worker.on("message", message => console.log("message", JSON.stringify(message)));
        worker.on("exit", code => console.log("exit", code));
    "#;
    match runtime.eval(source).expect("source parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        out.lines(),
        ["message {\"answer\":42,\"worker\":true}", "exit 0"]
    );
}

#[test]
fn worker_threads_transfer_worker_data_and_reply_arraybuffers() {
    let mut runtime = Runtime::new();
    let out = Captured::default();
    runtime.engine().op_state().put(ConsoleOut {
        out: Box::new(out.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
        const { Worker } = require("node:worker_threads");
        const buffer = new ArrayBuffer(4);
        new Uint8Array(buffer).set([1, 2, 3, 4]);
        const worker = new Worker(`
            const { parentPort, workerData } = require("node:worker_threads");
            const buffer = workerData.buffer;
            const inputIdentity = workerData.view.buffer === buffer;
            new Uint8Array(buffer)[2] = 8;
            parentPort.postMessage({
                buffer,
                view: new Uint8Array(buffer, 1, 2),
                inputIdentity,
            }, [buffer]);
            parentPort.postMessage({ detached: buffer.detached });
            parentPort.close();
        `, {
            eval: true,
            workerData: { buffer, view: new Uint8Array(buffer, 1, 2) },
            transferList: [buffer],
        });
        console.log("main-detached", buffer.detached, buffer.byteLength);
        worker.on("message", message => {
            if (message.buffer) {
                console.log("reply", message.inputIdentity,
                    message.view.buffer === message.buffer, message.view.join(","));
            } else {
                console.log("worker-detached", message.detached);
            }
        });
    "#;
    match runtime.eval(source).expect("source parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        out.lines(),
        [
            "main-detached true 0",
            "reply true true 2,8",
            "worker-detached true"
        ]
    );
}
