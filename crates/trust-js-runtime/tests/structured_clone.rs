//! Focused HTML structured serialization coverage: memory, properties, views, and transfers.

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
fn structured_serialization_preserves_graph_properties_views_and_transfer_order() {
    let mut runtime = Runtime::new();
    let output = Captured::default();
    runtime.engine().op_state().put(ConsoleOut {
        out: Box::new(output.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
      const date = new Date(1234);
      const regexp = /a.c/gi;
      const sparse = new Array(5);
      sparse[1] = undefined;
      sparse[3] = "present";
      sparse.extra = 7;
      Object.defineProperty(sparse, "__proto__", {
        value: "data", enumerable: true, writable: true, configurable: true,
      });
      const graph = structuredClone({ date, dateAgain: date, regexp, regexpAgain: regexp, sparse });
      console.log("identity", graph.date === graph.dateAgain, graph.regexp === graph.regexpAgain,
        graph.date.getTime(), String(graph.regexp));
      console.log("holes", graph.sparse.length, 0 in graph.sparse, 1 in graph.sparse,
        2 in graph.sparse, 3 in graph.sparse, 4 in graph.sparse, graph.sparse.extra,
        Object.getOwnPropertyDescriptor(graph.sparse, "__proto__").value,
        Object.getPrototypeOf(graph.sparse) === Array.prototype);

      const deletion = {
        get first() { delete this.second; return 1; },
        second: 2,
      };
      const deletionClone = structuredClone(deletion);
      console.log("keys", deletionClone.first, "second" in deletionClone,
        Object.keys(deletionClone).join(","));

      const backing = new ArrayBuffer(20);
      const bytes = new Uint8Array(backing);
      bytes.set([10, 11, 12, 13, 14, 15], 4);
      const typed = new Uint8Array(backing, 4, 6);
      const data = new DataView(backing, 6, 4);
      const views = structuredClone({ backing, typed, typedAgain: typed, data });
      console.log("views", views.typed === views.typedAgain,
        views.typed.buffer === views.backing, views.data.buffer === views.backing,
        views.typed.byteOffset, views.typed.length, views.data.byteOffset, views.data.byteLength,
        views.typed.join(","));

      const transferred = new ArrayBuffer(4, { maxByteLength: 12 });
      new Uint8Array(transferred).set([1, 2, 3, 4]);
      const transferredView = new Uint8Array(transferred, 1, 2);
      const moved = structuredClone({ transferred, transferredView }, {
        transfer: new Set([transferred]),
      });
      console.log("transfer", transferred.detached, transferred.byteLength,
        moved.transferred.resizable, moved.transferred.maxByteLength,
        moved.transferredView.buffer === moved.transferred, moved.transferredView.join(","));

      const detachedOnly = new ArrayBuffer(2);
      structuredClone({ ok: true }, { transfer: [detachedOnly] });
      console.log("unreachable-transfer", detachedOnly.detached);

      const duplicate = new ArrayBuffer(2);
      let duplicateGetter = 0;
      try {
        structuredClone({ get value() { duplicateGetter++; return 1; } },
          { transfer: [duplicate, duplicate] });
      } catch (error) {
        console.log("duplicate", error.name, duplicateGetter, duplicate.detached);
      }
      const failure = new ArrayBuffer(2);
      try { structuredClone({ bad: () => 1 }, { transfer: [failure] }); }
      catch (error) { console.log("failure", error.name, failure.detached); }
      try { structuredClone({}, { transfer: [{}] }); }
      catch (error) { console.log("nontransferable", error.name); }

      const lateBytes = new ArrayBuffer(2);
      new Uint8Array(lateBytes)[0] = 3;
      const late = {
        buffer: lateBytes,
        get mutate() { new Uint8Array(lateBytes)[0] = 9; return true; },
      };
      const lateClone = structuredClone(late, { transfer: [lateBytes] });
      console.log("late-transfer", new Uint8Array(lateClone.buffer)[0], lateBytes.detached);

      const roundTrip = value => __deserializeClone(__serializeForClone(value));
      const wireBacking = new ArrayBuffer(16);
      new Uint8Array(wireBacking).set([5, 6, 7, 8], 4);
      const wireView = new Uint8Array(wireBacking, 4, 4);
      const wireDate = new Date(88);
      const wireSparse = new Array(4);
      wireSparse[2] = undefined;
      wireSparse.extra = "x";
      const wire = roundTrip({
        backing: wireBacking,
        view: wireView,
        viewAgain: wireView,
        date: wireDate,
        dateAgain: wireDate,
        sparse: wireSparse,
      });
      console.log("wire", wire.view === wire.viewAgain, wire.view.buffer === wire.backing,
        wire.date === wire.dateAgain, wire.view.join(","), wire.sparse.length,
        0 in wire.sparse, 2 in wire.sparse, wire.sparse.extra);

      const wireTransfer = new ArrayBuffer(3);
      new Uint8Array(wireTransfer).set([7, 8, 9]);
      const wireBytes = __serializeForClone({
        buffer: wireTransfer,
        view: new Uint8Array(wireTransfer, 1, 2),
      }, [wireTransfer]);
      const wireMoved = __deserializeClone(wireBytes);
      console.log("wire-transfer", wireTransfer.detached,
        wireMoved.view.buffer === wireMoved.buffer, wireMoved.view.join(","));

      for (const malformed of [
        new Uint8Array(0),
        new Uint8Array([7, 0, 0, 0, 0]),
        new Uint8Array([...__serializeForClone(1), 0]),
      ]) {
        try { __deserializeClone(malformed); }
        catch (error) { console.log("malformed", error.name); }
      }

      const channel = new MessageChannel();
      channel.port1.onmessage = event => {
        console.log("port", event.data.view.buffer === event.data.buffer,
          event.data.view.join(","));
        channel.port1.close();
        channel.port2.close();
      };
      const portBuffer = new ArrayBuffer(3);
      new Uint8Array(portBuffer).set([4, 5, 6]);
      channel.port2.postMessage({
        buffer: portBuffer,
        view: new Uint8Array(portBuffer, 1, 2),
      }, [portBuffer]);
      console.log("port-detached", portBuffer.detached);
    "#;

    match runtime.eval(source).expect("source parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        output.lines(),
        [
            "identity true true 1234 /a.c/gi",
            "holes 5 false true false true false 7 data true",
            "keys 1 false first",
            "views true true true 4 6 6 4 10,11,12,13,14,15",
            "transfer true 0 true 12 true 2,3",
            "unreachable-transfer true",
            "duplicate DataCloneError 0 false",
            "failure DataCloneError false",
            "nontransferable DataCloneError",
            "late-transfer 9 true",
            "wire true true true 5,6,7,8 4 false true x",
            "wire-transfer true true 8,9",
            "malformed DataCloneError",
            "malformed DataCloneError",
            "malformed DataCloneError",
            "port-detached true",
            "port true 5,6",
        ]
    );
}
