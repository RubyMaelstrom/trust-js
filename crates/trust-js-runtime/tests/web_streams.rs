//! Focused WHATWG Streams coverage for controller, backpressure, BYOB, tee, and piping state.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use trust_js_runtime::{Completion, ConsoleOut, Runtime};

#[derive(Clone, Default)]
struct Captured(Rc<RefCell<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn lines(&self) -> Vec<String> {
        String::from_utf8(self.0.borrow().clone())
            .expect("UTF-8 console output")
            .lines()
            .map(str::to_string)
            .collect()
    }
}

#[test]
fn streams_follow_controller_backpressure_byob_tee_and_pipe_state_machines() {
    let mut runtime = Runtime::new();
    let output = Captured::default();
    runtime.engine().op_state().put(ConsoleOut {
        out: Box::new(output.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
      (async () => {
        const tick = () => Promise.resolve().then(() => Promise.resolve());

        let start;
        const startGate = new Promise(resolve => { start = resolve; });
        let pulls = 0;
        let activePulls = 0;
        let maximumPulls = 0;
        const gated = new ReadableStream({
          start() { return startGate; },
          async pull(controller) {
            activePulls++;
            maximumPulls = Math.max(maximumPulls, activePulls);
            const value = ++pulls;
            await Promise.resolve();
            controller.enqueue(value);
            activePulls--;
            if (value === 2) controller.close();
          },
        });
        const gatedReader = gated.getReader();
        const gatedReads = [gatedReader.read(), gatedReader.read()];
        await tick();
        console.log("pull-gated", pulls);
        start();
        const gatedValues = await Promise.all(gatedReads);
        console.log("pull-serial", gatedValues.map(result => result.value).join(","), pulls, maximumPulls);

        let desiredController;
        const sized = new ReadableStream({ start(controller) { desiredController = controller; } }, {
          highWaterMark: 5,
          size(chunk) { return chunk.cost; },
        });
        const initialDesired = desiredController.desiredSize;
        desiredController.enqueue({ cost: 2 });
        const firstDesired = desiredController.desiredSize;
        desiredController.enqueue({ cost: 4 });
        const secondDesired = desiredController.desiredSize;
        await sized.getReader().read();
        console.log("readable-size", initialDesired, firstDesired, secondDesired, desiredController.desiredSize);

        let startWrites;
        let finishWrite;
        const writeEvents = [];
        const writable = new WritableStream({
          start() {
            writeEvents.push("start");
            return new Promise(resolve => { startWrites = resolve; });
          },
          write(chunk) {
            writeEvents.push(`write:${chunk}`);
            return new Promise(resolve => { finishWrite = resolve; });
          },
          close() { writeEvents.push("close"); },
        }, {
          highWaterMark: 2,
          size(chunk) { return chunk.length; },
        });
        const writer = writable.getWriter();
        const write = writer.write("abc");
        let ready = false;
        writer.ready.then(() => { ready = true; });
        await tick();
        console.log("write-gated", writer.desiredSize, ready, writeEvents.join(","));
        startWrites();
        await tick();
        console.log("write-started", writeEvents.join(","));
        finishWrite();
        await write;
        await writer.ready;
        console.log("write-ready", writer.desiredSize, ready);
        const closing = writer.close();
        let postClose;
        try { await writer.write("late"); } catch (error) { postClose = error.name; }
        await closing;
        console.log("write-close", postClose, writeEvents.join(","));

        const shared = { marker: 1 };
        const [defaultBranch1, defaultBranch2] = new ReadableStream({
          start(controller) { controller.enqueue(shared); controller.close(); },
        }).tee();
        const defaultValues = await Promise.all([
          defaultBranch1.getReader().read(), defaultBranch2.getReader().read(),
        ]);
        console.log("tee-default", defaultValues[0].value === shared, defaultValues[1].value === shared);

        const originalBytes = new Uint8Array([7, 8]);
        const [byteBranch1, byteBranch2] = new ReadableStream({
          type: "bytes",
          start(controller) { controller.enqueue(originalBytes); controller.close(); },
        }).tee();
        const byteValues = await Promise.all([
          byteBranch1.getReader().read(), byteBranch2.getReader().read(),
        ]);
        byteValues[0].value[0] = 99;
        console.log("tee-bytes", originalBytes.buffer.detached, byteValues[1].value[0],
          byteValues[0].value.buffer !== byteValues[1].value.buffer);

        let compositeReason;
        const [cancelBranch1, cancelBranch2] = new ReadableStream({
          cancel(reason) { compositeReason = reason; },
        }).tee();
        await Promise.all([cancelBranch1.cancel("left"), cancelBranch2.cancel("right")]);
        console.log("tee-cancel", Array.isArray(compositeReason), compositeReason.join(","));

        let nextByte = 1;
        const byob = new ReadableStream({
          type: "bytes",
          pull(controller) {
            const request = controller.byobRequest;
            request.view[0] = nextByte++;
            request.view[1] = nextByte++;
            request.respond(2);
            if (nextByte === 5) controller.close();
          },
        });
        const byobReader = byob.getReader({ mode: "byob" });
        const supplied = new Uint8Array(4);
        const filled = await byobReader.read(supplied, { min: 4 });
        let badMin;
        try { await byobReader.read(new Uint8Array(1), { min: 0 }); }
        catch (error) { badMin = error.name; }
        console.log("byob", supplied.buffer.detached, filled.done,
          Array.from(filled.value).join(","), badMin);

        let cancelReason;
        const heldStream = new ReadableStream({ cancel(reason) { cancelReason = reason; } });
        const heldReader = heldStream.getReader();
        await heldReader.cancel("reader");
        console.log("reader-cancel", heldStream.locked, cancelReason);
        heldReader.releaseLock();
        console.log("reader-release", heldStream.locked);

        const writes = [];
        let sinkClosed = false;
        const pipeSource = new ReadableStream({
          start(controller) { controller.enqueue("a"); controller.enqueue("b"); controller.close(); },
        });
        const pipeSink = new WritableStream({
          write(chunk) { writes.push(chunk); },
          close() { sinkClosed = true; },
        });
        const piping = pipeSource.pipeTo(pipeSink);
        console.log("pipe-locks", pipeSource.locked, pipeSink.locked);
        await piping;
        console.log("pipe-forward", writes.join(""), sinkClosed, pipeSource.locked, pipeSink.locked);

        let preventedClose = false;
        const openSink = new WritableStream({ close() { preventedClose = true; } });
        await new ReadableStream({ start(controller) { controller.close(); } })
          .pipeTo(openSink, { preventClose: true });
        const openWriter = openSink.getWriter();
        await openWriter.close();
        console.log("pipe-prevent-close", preventedClose);

        let abortedWith;
        const sourceError = new Error("source failed");
        try {
          await new ReadableStream({ start(controller) { controller.error(sourceError); } })
            .pipeTo(new WritableStream({ abort(reason) { abortedWith = reason; } }));
        } catch (error) {
          console.log("pipe-source-error", error === sourceError, abortedWith === sourceError);
        }

        let canceledWith;
        const sinkError = new Error("sink failed");
        try {
          await new ReadableStream({
            start(controller) { controller.enqueue("x"); },
            cancel(reason) { canceledWith = reason; },
          }).pipeTo(new WritableStream({ write() { throw sinkError; } }));
        } catch (error) {
          console.log("pipe-sink-error", error === sinkError, canceledWith === sinkError);
        }

        let signalCancel;
        let signalAbort;
        const abortController = new AbortController();
        const abortedPipe = new ReadableStream({
          pull() { return new Promise(() => {}); },
          cancel(reason) { signalCancel = reason; },
        }).pipeTo(new WritableStream({ abort(reason) { signalAbort = reason; } }), {
          signal: abortController.signal,
        });
        abortController.abort("stopped");
        let pipeAbort;
        try { await abortedPipe; } catch (error) { pipeAbort = error; }
        console.log("pipe-abort", pipeAbort, signalCancel, signalAbort);

        const transform = new TransformStream();
        const transformWriter = transform.writable.getWriter();
        const transformWrite = transformWriter.write("held");
        let transformed = false;
        transformWrite.then(() => { transformed = true; });
        await tick();
        const beforeRead = transformed;
        const transformResult = await transform.readable.getReader().read();
        await transformWrite;
        console.log("transform-pressure", beforeRead, transformed, transformResult.value);
      })();
    "#;

    match runtime.eval(source).expect("Streams script parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        output.lines(),
        [
            "pull-gated 0",
            "pull-serial 1,2 2 1",
            "readable-size 5 3 -1 1",
            "write-gated -1 false start",
            "write-started start,write:abc",
            "write-ready 2 true",
            "write-close TypeError start,write:abc,close",
            "tee-default true true",
            "tee-bytes true 7 true",
            "tee-cancel true left,right",
            "byob true false 1,2,3,4 TypeError",
            "reader-cancel true reader",
            "reader-release false",
            "pipe-locks true true",
            "pipe-forward ab true false false",
            "pipe-prevent-close true",
            "pipe-source-error true true",
            "pipe-sink-error true true",
            "pipe-abort stopped stopped stopped",
            "transform-pressure false true held",
        ]
    );
}
