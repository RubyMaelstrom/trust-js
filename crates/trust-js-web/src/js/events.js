// DOM Standard event state, listener registration, and dispatch algorithms. EventTarget itself
// has no parent; DOM/host objects defined in this glue can install their standard "get the parent"
// algorithm through setEventTargetParent().

const DOM_EXCEPTION_CONSTANTS = [
  ["INDEX_SIZE_ERR", "IndexSizeError", 1],
  ["DOMSTRING_SIZE_ERR", null, 2],
  ["HIERARCHY_REQUEST_ERR", "HierarchyRequestError", 3],
  ["WRONG_DOCUMENT_ERR", "WrongDocumentError", 4],
  ["INVALID_CHARACTER_ERR", "InvalidCharacterError", 5],
  ["NO_DATA_ALLOWED_ERR", null, 6],
  ["NO_MODIFICATION_ALLOWED_ERR", "NoModificationAllowedError", 7],
  ["NOT_FOUND_ERR", "NotFoundError", 8],
  ["NOT_SUPPORTED_ERR", "NotSupportedError", 9],
  ["INUSE_ATTRIBUTE_ERR", "InUseAttributeError", 10],
  ["INVALID_STATE_ERR", "InvalidStateError", 11],
  ["SYNTAX_ERR", "SyntaxError", 12],
  ["INVALID_MODIFICATION_ERR", "InvalidModificationError", 13],
  ["NAMESPACE_ERR", "NamespaceError", 14],
  ["INVALID_ACCESS_ERR", "InvalidAccessError", 15],
  ["VALIDATION_ERR", null, 16],
  ["TYPE_MISMATCH_ERR", "TypeMismatchError", 17],
  ["SECURITY_ERR", "SecurityError", 18],
  ["NETWORK_ERR", "NetworkError", 19],
  ["ABORT_ERR", "AbortError", 20],
  ["URL_MISMATCH_ERR", "URLMismatchError", 21],
  ["QUOTA_EXCEEDED_ERR", "QuotaExceededError", 22],
  ["TIMEOUT_ERR", "TimeoutError", 23],
  ["INVALID_NODE_TYPE_ERR", "InvalidNodeTypeError", 24],
  ["DATA_CLONE_ERR", "DataCloneError", 25],
];
const DOM_EXCEPTION_CODES = Object.fromEntries(
  DOM_EXCEPTION_CONSTANTS.filter(([, name]) => name !== null).map(([, name, code]) => [name, code])
);
const DOM_EXCEPTION_STATE = new WeakMap();

class DOMException extends Error {
  constructor(message = "", name = "Error") {
    super();
    DOM_EXCEPTION_STATE.set(this, { message: String(message), name: String(name) });
  }
  get message() {
    const state = DOM_EXCEPTION_STATE.get(this);
    if (!state) throw new TypeError("DOMException.message called on an incompatible receiver");
    return state.message;
  }
  get name() {
    const state = DOM_EXCEPTION_STATE.get(this);
    if (!state) throw new TypeError("DOMException.name called on an incompatible receiver");
    return state.name;
  }
  get code() {
    const state = DOM_EXCEPTION_STATE.get(this);
    if (!state) throw new TypeError("DOMException.code called on an incompatible receiver");
    return DOM_EXCEPTION_CODES[state.name] ?? 0;
  }
}

// Web IDL §3.14 exposes legacy constants on the interface object and prototype. Keep the
// platform accessors enumerable and branded: record conversion of DOMException.prototype must
// observe their incompatible-receiver TypeError instead of treating it as an ordinary record.
for (const name of ["message", "name", "code"]) {
  const descriptor = Object.getOwnPropertyDescriptor(DOMException.prototype, name);
  Object.defineProperty(DOMException.prototype, name, { ...descriptor, enumerable: true });
}
for (const [legacyName, , code] of DOM_EXCEPTION_CONSTANTS) {
  Object.defineProperty(DOMException, legacyName, { value: code, enumerable: true });
  Object.defineProperty(DOMException.prototype, legacyName, { value: code, enumerable: true });
}
Object.defineProperty(DOMException.prototype, Symbol.toStringTag, {
  value: "DOMException", configurable: true,
});

const QUOTA_EXCEEDED_STATE = new WeakMap();

// Web IDL §2.8.3: the derived exception carries nullable quota/requested details while retaining
// the legacy QuotaExceededError name/code for compatibility.
class QuotaExceededError extends DOMException {
  constructor(message = "", options = {}) {
    super(message, "QuotaExceededError");
    if (options === null || options === undefined) options = {};
    if ((typeof options !== "object" || options === null) && typeof options !== "function") {
      throw new TypeError("QuotaExceededError options must be a dictionary");
    }
    // Web IDL dictionary conversion reads members in lexicographic order. An undefined member is
    // absent even when a property with that value exists; nullable result attributes remain null.
    const quotaValue = options.quota;
    const requestedValue = options.requested;
    const quota = quotaValue === undefined ? null : Number(quotaValue);
    const requested = requestedValue === undefined ? null : Number(requestedValue);
    if ((quota !== null && !Number.isFinite(quota)) ||
        (requested !== null && !Number.isFinite(requested))) {
      throw new TypeError("QuotaExceededError quota and requested must be finite numbers");
    }
    if ((quota !== null && quota < 0) || (requested !== null && requested < 0) ||
        (quota !== null && requested !== null && requested < quota)) {
      throw new RangeError("QuotaExceededError requested must be at least quota");
    }
    QUOTA_EXCEEDED_STATE.set(this, { quota, requested });
  }
  get quota() {
    const state = QUOTA_EXCEEDED_STATE.get(this);
    if (!state) throw new TypeError("QuotaExceededError.quota called on an incompatible receiver");
    return state.quota;
  }
  get requested() {
    const state = QUOTA_EXCEEDED_STATE.get(this);
    if (!state) throw new TypeError("QuotaExceededError.requested called on an incompatible receiver");
    return state.requested;
  }
}
for (const name of ["quota", "requested"]) {
  const descriptor = Object.getOwnPropertyDescriptor(QuotaExceededError.prototype, name);
  Object.defineProperty(QuotaExceededError.prototype, name, { ...descriptor, enumerable: true });
}
Object.defineProperty(QuotaExceededError.prototype, Symbol.toStringTag, {
  value: "QuotaExceededError", configurable: true,
});

const EVENT_STATE = new WeakMap();
const EVENT_TARGET_STATE = new WeakMap();
const CUSTOM_EVENT_DETAIL = new WeakMap();

function eventState(event, operation) {
  const state = EVENT_STATE.get(event);
  if (!state) throw new TypeError(`${operation} called on an incompatible receiver`);
  return state;
}

function eventTargetState(target, operation) {
  const state = EVENT_TARGET_STATE.get(target);
  if (!state) throw new TypeError(`${operation} called on an incompatible receiver`);
  return state;
}

function initializeEvent(event, type, bubbles, cancelable) {
  const state = eventState(event, "Event initialization");
  state.initialized = true;
  state.stopPropagation = false;
  state.stopImmediatePropagation = false;
  state.canceled = false;
  state.isTrusted = false;
  state.target = null;
  state.type = String(type);
  state.bubbles = !!bubbles;
  state.cancelable = !!cancelable;
}

class Event {
  constructor(type, init = {}) {
    if (arguments.length === 0) throw new TypeError("Event constructor requires a type");
    init = init && typeof init === "object" ? init : {};
    EVENT_STATE.set(this, {
      type: String(type),
      target: null,
      currentTarget: null,
      path: [],
      phase: Event.NONE,
      bubbles: !!init.bubbles,
      cancelable: !!init.cancelable,
      composed: !!init.composed,
      isTrusted: false,
      timeStamp: performance.now(),
      stopPropagation: false,
      stopImmediatePropagation: false,
      canceled: false,
      inPassiveListener: false,
      initialized: true,
      dispatching: false,
    });
    // isTrusted is [LegacyUnforgeable]: it is an own, non-configurable accessor.
    Object.defineProperty(this, "isTrusted", {
      enumerable: true,
      configurable: false,
      get: eventIsTrusted,
    });
  }
  get type() { return eventState(this, "Event.type").type; }
  get target() { return eventState(this, "Event.target").target; }
  get srcElement() { return eventState(this, "Event.srcElement").target; }
  get currentTarget() { return eventState(this, "Event.currentTarget").currentTarget; }
  get eventPhase() { return eventState(this, "Event.eventPhase").phase; }
  get bubbles() { return eventState(this, "Event.bubbles").bubbles; }
  get cancelable() { return eventState(this, "Event.cancelable").cancelable; }
  get defaultPrevented() { return eventState(this, "Event.defaultPrevented").canceled; }
  get composed() { return eventState(this, "Event.composed").composed; }
  get timeStamp() { return eventState(this, "Event.timeStamp").timeStamp; }
  get cancelBubble() { return eventState(this, "Event.cancelBubble").stopPropagation; }
  set cancelBubble(value) {
    if (value) eventState(this, "Event.cancelBubble").stopPropagation = true;
  }
  get returnValue() { return !eventState(this, "Event.returnValue").canceled; }
  set returnValue(value) {
    if (!value) setEventCanceled(this);
  }
  composedPath() {
    const state = eventState(this, "Event.composedPath");
    // TRust JS currently has no ShadowRoot objects. With no closed-shadow filtering, the DOM
    // algorithm reduces to the invocation targets in target-to-root order while dispatching.
    return state.currentTarget === null ? [] : state.path.slice();
  }
  preventDefault() {
    setEventCanceled(this);
  }
  stopPropagation() {
    eventState(this, "Event.stopPropagation").stopPropagation = true;
  }
  stopImmediatePropagation() {
    const state = eventState(this, "Event.stopImmediatePropagation");
    state.stopPropagation = true;
    state.stopImmediatePropagation = true;
  }
  initEvent(type, bubbles = false, cancelable = false) {
    const state = eventState(this, "Event.initEvent");
    if (state.dispatching) return;
    initializeEvent(this, type, bubbles, cancelable);
  }
}

function eventIsTrusted() {
  return eventState(this, "Event.isTrusted").isTrusted;
}

function setEventCanceled(event) {
  const state = eventState(event, "Event.preventDefault");
  if (state.cancelable && !state.inPassiveListener) state.canceled = true;
}

for (const [name, value] of [
  ["NONE", 0], ["CAPTURING_PHASE", 1], ["AT_TARGET", 2], ["BUBBLING_PHASE", 3],
]) {
  Object.defineProperty(Event, name, { value, enumerable: true });
  Object.defineProperty(Event.prototype, name, { value, enumerable: true });
}

Object.defineProperty(Event.prototype, Symbol.toStringTag, {
  value: "Event", configurable: true,
});

class CustomEvent extends Event {
  constructor(type, init = {}) {
    super(type, init);
    init = init && typeof init === "object" ? init : {};
    CUSTOM_EVENT_DETAIL.set(this, "detail" in init ? init.detail : null);
  }
  get detail() {
    if (!CUSTOM_EVENT_DETAIL.has(this)) {
      throw new TypeError("CustomEvent.detail called on an incompatible receiver");
    }
    return CUSTOM_EVENT_DETAIL.get(this);
  }
  initCustomEvent(type, bubbles = false, cancelable = false, detail = null) {
    const state = eventState(this, "CustomEvent.initCustomEvent");
    if (!CUSTOM_EVENT_DETAIL.has(this)) {
      throw new TypeError("CustomEvent.initCustomEvent called on an incompatible receiver");
    }
    if (state.dispatching) return;
    initializeEvent(this, type, bubbles, cancelable);
    CUSTOM_EVENT_DETAIL.set(this, detail);
  }
}

Object.defineProperty(CustomEvent.prototype, Symbol.toStringTag, {
  value: "CustomEvent", configurable: true,
});

class EventTarget {
  constructor() {
    EVENT_TARGET_STATE.set(this, { listeners: new Map(), getParent: null });
  }
  addEventListener(type, callback, options = {}) {
    const target = eventTargetState(this, "EventTarget.addEventListener");
    // DOM §2.7 addEventListener first runs "flatten more options", including observable Gets and
    // Web IDL's non-null AbortSignal conversion, before the add-listener algorithm ignores a null
    // callback. Keep the specified capture → once → passive → signal order.
    let capture;
    let once = false;
    let passive = false;
    let signal = null;
    if (typeof options === "boolean") {
      capture = options;
    } else {
      options = options && typeof options === "object" ? options : {};
      capture = !!options.capture;
      once = !!options.once;
      passive = !!options.passive;
      if ("signal" in options) {
        signal = options.signal;
        if (!(signal instanceof AbortSignal)) {
          throw new TypeError("options.signal must be an AbortSignal");
        }
      }
    }

    if (callback === null || callback === undefined) return;
    if (typeof callback !== "function" && (typeof callback !== "object" || callback === null)) {
      throw new TypeError("Event listener must be a function or callback object");
    }

    const key = String(type);
    let list = target.listeners.get(key);
    if (!list) {
      list = [];
      target.listeners.set(key, list);
    }
    if (list.some(entry => !entry.removed && entry.callback === callback && entry.capture === !!capture)) {
      return;
    }
    if (signal !== null) {
      if (signal.aborted) return;
    }

    const entry = {
      callback,
      capture: !!capture,
      once,
      passive,
      signal,
      abortCallback: null,
      removed: false,
    };
    list.push(entry);
    if (signal !== null) {
      entry.abortCallback = () => removeEventListenerRecord(this, key, entry);
      signal.addEventListener("abort", entry.abortCallback, { once: true });
    }
  }
  removeEventListener(type, callback, options = {}) {
    const target = eventTargetState(this, "EventTarget.removeEventListener");
    const capture = typeof options === "boolean"
      ? options
      : !!(options && typeof options === "object" && options.capture);
    const key = String(type);
    const list = target.listeners.get(key);
    if (!list) return;
    const entry = list.find(item =>
      !item.removed && item.callback === callback && item.capture === capture
    );
    if (entry) removeEventListenerRecord(this, key, entry);
  }
  dispatchEvent(event) {
    eventTargetState(this, "EventTarget.dispatchEvent");
    const state = EVENT_STATE.get(event);
    if (!state) throw new TypeError("dispatchEvent expects an Event");
    if (state.dispatching || !state.initialized) {
      throw new DOMException("The event is already being dispatched", "InvalidStateError");
    }
    state.isTrusted = false;
    return dispatchEventToTarget(event, this);
  }
}

Object.defineProperty(EventTarget.prototype, Symbol.toStringTag, {
  value: "EventTarget", configurable: true,
});

function setEventTargetParent(target, getParent) {
  const state = eventTargetState(target, "setEventTargetParent");
  if (getParent !== null && typeof getParent !== "function") {
    throw new TypeError("EventTarget parent algorithm must be a function or null");
  }
  state.getParent = getParent;
}

function removeEventListenerRecord(target, type, entry) {
  if (entry.removed) return;
  entry.removed = true;
  const state = EVENT_TARGET_STATE.get(target);
  const list = state && state.listeners.get(type);
  if (list) {
    const index = list.indexOf(entry);
    if (index !== -1) list.splice(index, 1);
    if (list.length === 0) state.listeners.delete(type);
  }
  if (entry.signal !== null && entry.abortCallback !== null) {
    entry.signal.removeEventListener("abort", entry.abortCallback);
    entry.abortCallback = null;
  }
}

function dispatchEventToTarget(event, target) {
  const state = eventState(event, "Event dispatch");
  state.dispatching = true;
  state.target = target;
  state.path = [target];

  try {
    // DOM Standard §2.9: build the path by repeatedly invoking each target's internal get-parent
    // algorithm. Author-created EventTargets keep the default null algorithm.
    const seen = new Set(state.path);
    let current = target;
    for (;;) {
      const targetState = EVENT_TARGET_STATE.get(current);
      const parent = targetState && targetState.getParent
        ? targetState.getParent.call(current, event)
        : null;
      if (parent === null || parent === undefined) break;
      eventTargetState(parent, "EventTarget parent algorithm");
      if (seen.has(parent)) throw new DOMException("Cyclic event path", "InvalidStateError");
      state.path.push(parent);
      seen.add(parent);
      current = parent;
    }

    // Capture proceeds root-to-parent, then capture and non-capture listeners each get their
    // AT_TARGET invocation. Bubbling through ancestors only occurs for a bubbling event.
    for (let i = state.path.length - 1; i >= 1; i--) {
      state.phase = Event.CAPTURING_PHASE;
      invokeEventListeners(state.path[i], event, true);
    }
    state.phase = Event.AT_TARGET;
    invokeEventListeners(target, event, true);
    invokeEventListeners(target, event, false);
    if (state.bubbles) {
      for (let i = 1; i < state.path.length; i++) {
        state.phase = Event.BUBBLING_PHASE;
        invokeEventListeners(state.path[i], event, false);
      }
    }
  } finally {
    // The canceled flag and target persist. Dispatch/path/phase and both propagation flags are
    // reset so the same initialized Event can be dispatched again.
    state.phase = Event.NONE;
    state.currentTarget = null;
    state.path = [];
    state.dispatching = false;
    state.stopPropagation = false;
    state.stopImmediatePropagation = false;
    state.inPassiveListener = false;
  }
  return !state.canceled;
}

function invokeEventListeners(target, event, capture) {
  const state = eventState(event, "Event listener invocation");
  if (state.stopPropagation) return;
  state.currentTarget = target;
  const targetState = eventTargetState(target, "Event listener invocation");
  const listeners = (targetState.listeners.get(state.type) ?? []).slice();
  for (const entry of listeners) {
    if (entry.removed || entry.capture !== capture) continue;
    if (entry.once) removeEventListenerRecord(target, state.type, entry);
    state.inPassiveListener = entry.passive;
    try {
      if (typeof entry.callback === "function") {
        entry.callback.call(target, event);
      } else {
        const operation = entry.callback.handleEvent;
        if (typeof operation !== "function") throw new TypeError("EventListener.handleEvent is not callable");
        operation.call(entry.callback, event);
      }
    } catch (error) {
      reportEventListenerError(error);
    } finally {
      state.inPassiveListener = false;
    }
    if (state.stopImmediatePropagation) break;
  }
}

function reportEventListenerError(error) {
  try {
    if (typeof globalThis.reportError === "function") globalThis.reportError(error);
    else console.error("Uncaught (in event listener)", error instanceof Error
      ? `${error.name}: ${error.message}` : String(error));
  } catch {
    // Reporting an exception cannot escape dispatch.
  }
}

// HTML-style event-handler attributes use the same listener list, preserving their registration
// position when a function is replaced without first being set to null.
function defineEventHandler(proto, name, afterSet) {
  const listeners = new WeakMap();
  const handlers = new WeakMap();
  Object.defineProperty(proto, `on${name}`, {
    configurable: true,
    enumerable: true,
    get() {
      return handlers.get(this) ?? null;
    },
    set(value) {
      if (typeof value === "function") {
        handlers.set(this, value);
        if (!listeners.has(this)) {
          const wrapped = (event) => {
            const handler = handlers.get(this);
            if (handler && handler.call(this, event) === false) event.preventDefault();
          };
          listeners.set(this, wrapped);
          this.addEventListener(name, wrapped);
        }
        if (afterSet) afterSet(this);
      } else {
        handlers.delete(this);
        const wrapped = listeners.get(this);
        if (wrapped) this.removeEventListener(name, wrapped);
        listeners.delete(this);
      }
    },
  });
}

const kSignalCreate = Symbol("AbortSignal-internal-create");

class AbortSignal extends EventTarget {
  constructor(token) {
    if (token !== kSignalCreate) throw new TypeError("Illegal constructor");
    super();
    this.aborted = false;
    this.reason = undefined;
  }
  throwIfAborted() {
    if (this.aborted) throw this.reason;
  }
  _doAbort(reason) {
    if (this.aborted) return;
    this.aborted = true;
    this.reason = reason !== undefined
      ? reason
      : new DOMException("signal is aborted without reason", "AbortError");
    this.dispatchEvent(new Event("abort"));
  }
  static abort(reason) {
    const signal = new AbortSignal(kSignalCreate);
    signal._doAbort(reason);
    return signal;
  }
  static timeout(ms) {
    const controller = new AbortController();
    setTimeout(() => controller.abort(new DOMException("signal timed out", "TimeoutError")), ms);
    return controller.signal;
  }
}

defineEventHandler(AbortSignal.prototype, "abort");

class AbortController {
  constructor() {
    this.signal = new AbortSignal(kSignalCreate);
  }
  abort(reason) {
    this.signal._doAbort(reason);
  }
}

globalThis.DOMException = DOMException;
globalThis.QuotaExceededError = QuotaExceededError;
globalThis.Event = Event;
globalThis.CustomEvent = CustomEvent;
globalThis.EventTarget = EventTarget;
globalThis.AbortSignal = AbortSignal;
globalThis.AbortController = AbortController;
