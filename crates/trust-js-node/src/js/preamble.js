// Capture the raw op namespaces; everything below closes over these. Runs in one IIFE.
"use strict";
const __node = globalThis.__node;
const __os = globalThis.__os;
const __zlib = globalThis.__zlib;
const __bunhash = globalThis.__bunhash;
const __child = globalThis.__child;
const __ffi = globalThis.__ffi;
const __crypto = globalThis.__crypto;
const __password = globalThis.__password;
delete globalThis.__node;
delete globalThis.__os;
delete globalThis.__zlib;
delete globalThis.__bunhash;
delete globalThis.__child;
delete globalThis.__ffi;
delete globalThis.__crypto;
delete globalThis.__password;

// Node's `global` is an alias for the global object.
if (typeof globalThis.global === "undefined") {
  globalThis.global = globalThis;
}

// V8's `Error.prepareStackTrace` / CallSite API on top of the engine's `Error.captureStackTrace`
// and `Error.stackTraceLimit`. They are pervasive in the Node ecosystem: http-errors and depd build
// Error subclasses with captureStackTrace, and depd reads *structured* frames (a
// `prepareStackTrace` hook receiving CallSite objects it calls `.getFileName()`/`.getLineNumber()`
// on). The engine formats `.stack` itself but has no CallSite objects, so while such a hook is
// installed we hand back placeholder CallSites — enough that the deprecation machinery runs (with
// a `<trust-js>` location) instead of crashing. Without a hook the engine's own capture runs.
const nativeCaptureStackTrace = Error.captureStackTrace;
const makeCallSite = () => ({
  getThis: () => undefined,
  getTypeName: () => null,
  getFunction: () => undefined,
  getFunctionName: () => null,
  getMethodName: () => null,
  getFileName: () => "<trust-js>",
  getLineNumber: () => 0,
  getColumnNumber: () => 0,
  getEvalOrigin: () => undefined,
  isToplevel: () => true,
  isEval: () => false,
  isNative: () => false,
  isConstructor: () => false,
  isAsync: () => false,
  toString: () => "<trust-js>:0:0",
});
Error.captureStackTrace = function captureStackTrace(target, ctorOpt) {
  const prepare = Error.prepareStackTrace;
  if (typeof prepare !== "function") {
    // Leave this wrapper out of the captured frames.
    return nativeCaptureStackTrace(target, ctorOpt || captureStackTrace);
  }
  const sites = Array.from({ length: Error.stackTraceLimit || 10 }, makeCallSite);
  Object.defineProperty(target, "stack", {
    configurable: true,
    get() {
      const hook = Error.prepareStackTrace;
      if (typeof hook === "function") return hook(target, sites);
      return `${target.name || "Error"}: ${target.message || ""}\n    at <trust-js>:0:0`;
    },
    set(value) {
      Object.defineProperty(target, "stack", { value, writable: true, configurable: true });
    },
  });
  return target;
};

// A registry the module system fills in; each builtin registers itself as it is defined.
const __builtins = new Map();
