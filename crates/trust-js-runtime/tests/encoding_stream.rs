//! Focused WHATWG Encoding API coverage for labels, decoder state, BOMs, errors, and streams.

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
fn encoding_labels_incremental_state_bom_errors_and_streams() {
    let mut runtime = Runtime::new();
    let output = Captured::default();
    runtime.engine().op_state().put(ConsoleOut {
        out: Box::new(output.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
      (async () => {
        let decoder = new TextDecoder();
        console.log("utf8-split", JSON.stringify(decoder.decode(new Uint8Array([0xf0, 0x9f]), { stream: true })),
          decoder.decode(new Uint8Array([0x92, 0xa9])));
        console.log("utf8-flush", new TextDecoder().decode(new Uint8Array([0xf0, 0x9f])));

        decoder = new TextDecoder("iso-8859-1");
        console.log("label", decoder.encoding, decoder.decode(new Uint8Array([0x80, 0x91, 0x41])));
        decoder = new TextDecoder("shift_jis");
        console.log("sjis", JSON.stringify(decoder.decode(new Uint8Array([0x82]), { stream: true })),
          decoder.decode(new Uint8Array([0xa0])));
        decoder = new TextDecoder("utf-16le");
        console.log("utf16", JSON.stringify(decoder.decode(new Uint8Array([0xff]), { stream: true })),
          decoder.decode(new Uint8Array([0xfe, 0x3d, 0xd8, 0xa9, 0xdc])));

        decoder = new TextDecoder();
        console.log("bom", JSON.stringify(decoder.decode(new Uint8Array([0xef]), { stream: true })),
          decoder.decode(new Uint8Array([0xbb, 0xbf, 0x41])));
        console.log("bom-reset", decoder.decode(new Uint8Array([0xef, 0xbb, 0xbf, 0x42])));
        console.log("bom-ignore", new TextDecoder("utf-8", { ignoreBOM: true })
          .decode(new Uint8Array([0xef, 0xbb, 0xbf, 0x43])).charCodeAt(0));

        const fatal = new TextDecoder("utf-8", { fatal: true });
        fatal.decode(new Uint8Array([0xf0]), { stream: true });
        try { fatal.decode(); } catch (error) { console.log("fatal", error.name); }
        try { new TextDecoder("replacement"); } catch (error) { console.log("label-error", error.name); }

        const encoder = new TextEncoder();
        console.log("encode", Buffer.from(encoder.encode("A\ud800B")).toString("hex"));
        const short = new Uint8Array(4);
        console.log("into-short", JSON.stringify(encoder.encodeInto("A💩B", short)),
          Buffer.from(short).toString("hex"));
        const enough = new Uint8Array(6);
        console.log("into-full", JSON.stringify(encoder.encodeInto("A💩B", enough)),
          Buffer.from(enough).toString("hex"));

        async function run(stream, chunks) {
          const reader = stream.readable.getReader();
          const writer = stream.writable.getWriter();
          const collected = (async () => {
            const values = [];
            for (;;) {
              const next = await reader.read();
              if (next.done) return values;
              values.push(next.value);
            }
          })();
          for (const chunk of chunks) await writer.write(chunk);
          await writer.close();
          return collected;
        }

        const decoded = await run(new TextDecoderStream("shift_jis"), [
          new Uint8Array([0x82]), new Uint8Array([0xa0, 0x41]),
        ]);
        console.log("decoder-stream", decoded.join("|"));
        const encoded = await run(new TextEncoderStream(), ["A\ud83d", "\udca9B"]);
        console.log("encoder-stream", Buffer.concat(encoded).toString("hex"));
        const stream = new TextDecoderStream("utf-8", { fatal: true, ignoreBOM: true });
        console.log("stream-props", stream.encoding, stream.fatal, stream.ignoreBOM);
        try { await run(stream, [new Uint8Array([0xff])]); }
        catch (error) { console.log("stream-fatal", error.name); }
      })();
    "#;

    match runtime
        .eval(source)
        .expect("Encoding conformance script parses")
    {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        output.lines(),
        [
            "utf8-split \"\" 💩",
            "utf8-flush �",
            "label windows-1252 €‘A",
            "sjis \"\" あ",
            "utf16 \"\" 💩",
            "bom \"\" A",
            "bom-reset B",
            "bom-ignore 65279",
            "fatal TypeError",
            "label-error RangeError",
            "encode 41efbfbd42",
            "into-short {\"read\":1,\"written\":1} 41000000",
            "into-full {\"read\":4,\"written\":6} 41f09f92a942",
            "decoder-stream あA",
            "encoder-stream 41f09f92a942",
            "stream-props utf-8 true true",
            "stream-fatal TypeError",
        ]
    );
}
