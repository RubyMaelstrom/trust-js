//! Focused DOM Standard coverage for event flags, target phases, listeners, and redispatch.

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
fn dom_events_enforce_dispatch_flags_phases_listener_state_and_reset() {
    let mut runtime = Runtime::new();
    let output = Captured::default();
    runtime.engine().ctx().op_state().put(ConsoleOut {
        out: Box::new(output.clone()),
        err: Box::new(Captured::default()),
    });

    let source = r#"
      const target = new EventTarget();
      const event = new Event("phase", { bubbles: true, cancelable: true, composed: true });
      console.log("initial", event.eventPhase, event.target, event.currentTarget,
        event.composedPath().length, event.NONE, event.composed, event.isTrusted);

      const order = [];
      target.addEventListener("phase", e => {
        order.push(`bubble:${e.eventPhase}:${e.currentTarget === target}:${e.composedPath()[0] === target}`);
      });
      target.addEventListener("phase", e => {
        order.push(`capture:${e.eventPhase}`);
        try { target.dispatchEvent(e); }
        catch (error) { order.push(`redispatch:${error.name}:${error.code}`); }
        e.initEvent("changed", false, false);
      }, true);
      console.log("dispatch", target.dispatchEvent(event), order.join("|"));
      console.log("reset", event.type, event.eventPhase, event.currentTarget,
        event.composedPath().length, event.target === target, event.cancelBubble);

      const changes = [];
      const late = () => changes.push("late");
      const removed = () => changes.push("removed");
      target.addEventListener("change", removed);
      target.addEventListener("change", () => {
        changes.push("capture");
        target.removeEventListener("change", removed);
        target.addEventListener("change", late);
      }, true);
      target.dispatchEvent(new Event("change"));
      console.log("listener-changes", changes.join(","));

      let duplicate = 0;
      const duplicateCallback = () => duplicate++;
      target.addEventListener("duplicate", duplicateCallback);
      target.addEventListener("duplicate", duplicateCallback, { once: true, passive: true });
      target.dispatchEvent(new Event("duplicate"));
      target.dispatchEvent(new Event("duplicate"));
      console.log("duplicate", duplicate);

      const passiveEvent = new Event("cancel", { cancelable: true });
      target.addEventListener("cancel", e => e.preventDefault(), { passive: true });
      console.log("passive", target.dispatchEvent(passiveEvent), passiveEvent.defaultPrevented,
        passiveEvent.returnValue);
      target.addEventListener("cancel", e => { e.returnValue = false; }, { once: true });
      console.log("active", target.dispatchEvent(passiveEvent), passiveEvent.defaultPrevented,
        passiveEvent.returnValue);
      console.log("cancel-persists", target.dispatchEvent(passiveEvent));
      passiveEvent.initEvent("cancel", false, true);
      console.log("init-reset", passiveEvent.defaultPrevented, passiveEvent.target,
        target.dispatchEvent(passiveEvent));

      const stopped = new Event("stopped");
      const stoppedOrder = [];
      const stopFirst = e => { stoppedOrder.push("first"); e.stopImmediatePropagation(); };
      target.addEventListener("stopped", stopFirst);
      target.addEventListener("stopped", () => stoppedOrder.push("second"));
      target.dispatchEvent(stopped);
      target.removeEventListener("stopped", stopFirst);
      target.dispatchEvent(stopped);
      console.log("immediate-reset", stoppedOrder.join(","));

      const preStopped = new Event("pre-stopped");
      let preStoppedCount = 0;
      target.addEventListener("pre-stopped", () => preStoppedCount++);
      preStopped.cancelBubble = true;
      target.dispatchEvent(preStopped);
      target.dispatchEvent(preStopped);
      console.log("propagation-reset", preStoppedCount, preStopped.cancelBubble);

      const listenerObject = {
        calls: 0,
        handleEvent(e) { this.calls++; console.log("object", this === listenerObject, e.type); },
      };
      target.addEventListener("object", listenerObject, { once: true });
      target.dispatchEvent(new Event("object"));
      target.dispatchEvent(new Event("object"));
      console.log("object-once", listenerObject.calls);

      const controller = new AbortController();
      let signaled = 0;
      target.addEventListener("signal", () => signaled++, { signal: controller.signal });
      controller.abort();
      target.dispatchEvent(new Event("signal"));
      console.log("signal", signaled);

      const handlerController = new AbortController();
      const handlerOrder = [];
      handlerController.signal.addEventListener("abort", () => handlerOrder.push("before"));
      handlerController.signal.onabort = () => handlerOrder.push("old");
      handlerController.signal.addEventListener("abort", () => handlerOrder.push("after"));
      handlerController.signal.onabort = () => handlerOrder.push("handler");
      handlerController.abort();
      console.log("handler-order", handlerOrder.join(","));

      const custom = new CustomEvent("old", { detail: 1, composed: true });
      custom.initCustomEvent("new", true, true, 2);
      console.log("custom-init", custom.type, custom.bubbles, custom.cancelable,
        custom.composed, custom.detail);
    "#;

    match runtime.eval(source).expect("DOM events script parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }

    assert_eq!(
        output.lines(),
        [
            "initial 0 null null 0 0 true false",
            "dispatch true capture:2|redispatch:InvalidStateError:11|bubble:2:true:true",
            "reset phase 0 null 0 true false",
            "listener-changes capture,late",
            "duplicate 2",
            "passive true false true",
            "active false true false",
            "cancel-persists false",
            "init-reset false null true",
            "immediate-reset first,second",
            "propagation-reset 1 false",
            "object true object",
            "object-once 1",
            "signal 0",
            "handler-order before,handler,after",
            "custom-init new true true true 2",
        ]
    );
}
