//! WHATWG Compression Standard coverage for format support, chunk consumption, and framing errors.

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
fn compression_stream_formats_copy_chunks_and_reject_bad_framing() {
    let mut runtime = Runtime::new();
    let out = Captured::default();
    runtime.engine().ctx().op_state().put(ConsoleOut {
        out: Box::new(out.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
      (async () => {
        async function collect(readable) {
          const chunks = [];
          let size = 0;
          const reader = readable.getReader();
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            chunks.push(value);
            size += value.byteLength;
          }
          const result = new Uint8Array(size);
          let offset = 0;
          for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.byteLength;
          }
          return result;
        }

        async function run(stream, chunks) {
          const output = collect(stream.readable);
          const writer = stream.writable.getWriter();
          for (const chunk of chunks) await writer.write(chunk);
          await writer.close();
          return output;
        }

        const expected = new TextEncoder().encode("abc123".repeat(1000));
        for (const format of ["brotli", "deflate", "deflate-raw", "gzip"]) {
          const first = expected.slice(0, 777);
          const stream = new CompressionStream(format);
          const compressedPromise = collect(stream.readable);
          const writer = stream.writable.getWriter();
          await writer.write(first);
          // Once the transform algorithm has returned, the native codec has consumed the bytes;
          // later author mutation must not retroactively change that input.
          first.fill(0);
          await writer.write(expected.slice(777));
          await writer.close();
          const compressed = await compressedPromise;
          const split = Math.max(1, Math.floor(compressed.length / 2));
          const decoded = await run(new DecompressionStream(format), [
            compressed.slice(0, split),
            compressed.slice(split),
          ]);
          console.log(format, Buffer.from(decoded).equals(Buffer.from(expected)));
        }

        // A large incompressible body produces output before close(): the decoder is a real
        // incremental context rather than a flush-time replay of all retained chunks.
        const large = new Uint8Array(20_000);
        let random = 0x12345678;
        for (let i = 0; i < large.length; i++) {
          random ^= random << 13;
          random ^= random >>> 17;
          random ^= random << 5;
          large[i] = random >>> 24;
        }
        const largeGzip = await run(new CompressionStream("gzip"), [large]);
        const incremental = new DecompressionStream("gzip");
        const incrementalReader = incremental.readable.getReader();
        const incrementalWriter = incremental.writable.getWriter();
        const firstRead = incrementalReader.read();
        const gzipSplit = Math.floor(largeGzip.length / 4);
        const firstWrite = incrementalWriter.write(largeGzip.slice(0, gzipSplit));
        const firstOutput = await firstRead;
        await firstWrite;
        const remainingOutput = (async () => {
          const chunks = [];
          for (;;) {
            const next = await incrementalReader.read();
            if (next.done) return chunks;
            chunks.push(next.value);
          }
        })();
        await incrementalWriter.write(largeGzip.slice(gzipSplit));
        await incrementalWriter.close();
        const remainingChunks = await remainingOutput;
        const rebuilt = Buffer.concat([firstOutput.value, ...remainingChunks]);
        console.log("incremental", !firstOutput.done, firstOutput.value.length > 0, rebuilt.equals(Buffer.from(large)));

        const gzip = await run(new CompressionStream("gzip"), [expected]);
        gzip[gzip.length - 8] ^= 1;
        try {
          await run(new DecompressionStream("gzip"), [gzip]);
          console.log("checksum", "accepted");
        } catch (error) {
          console.log("checksum", error.name);
        }

        try {
          new DecompressionStream("not-a-format");
        } catch (error) {
          console.log("format", error.name);
        }
      })();
    "#;

    match runtime
        .eval(source)
        .expect("CompressionStream script parses")
    {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        out.lines(),
        [
            "brotli true",
            "deflate true",
            "deflate-raw true",
            "gzip true",
            "incremental true true true",
            "checksum TypeError",
            "format TypeError",
        ]
    );
}
