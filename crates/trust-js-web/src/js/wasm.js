// The WebAssembly JS API over the native __wasm ops (decoder + interpreter in Rust). Every wasm
// entity lives in one shared Rust-side Store; these JS handles carry integer *store addresses*, so
// a Memory/Table/Global can be created standalone and imported by any module (cross-module
// linking). Functions are called by their store address.
//
// A Memory.buffer ArrayBuffer and the Rust store identify the same Data Block. Successful growth
// detaches the old fixed buffer and installs a replacement synchronously, including for a
// memory.grow instruction immediately before an imported JS call. No SIMD/threads/GC.

class CompileError extends Error {
  constructor(m) { super(m); this.name = "CompileError"; }
}
class LinkError extends Error {
  constructor(m) { super(m); this.name = "LinkError"; }
}
class RuntimeError extends Error {
  constructor(m) { super(m); this.name = "RuntimeError"; }
}

function wrapWasmError(e) {
  const msg = (e && e.message) || String(e);
  if (msg.startsWith("CompileError:")) return new CompileError(msg.slice(13).trim());
  if (msg.startsWith("LinkError:")) return new LinkError(msg.slice(10).trim());
  if (msg.startsWith("RuntimeError:")) return new RuntimeError(msg.slice(13).trim());
  return e;
}

function toBytes(source) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  if (ArrayBuffer.isView(source)) return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  throw new TypeError("WebAssembly: expected a BufferSource");
}

// WebAssembly JS API § AddressValueToU64, for the i32 address type implemented by TRust JS. This is
// Web IDL [EnforceRange] unsigned long: ToNumber, truncate, then reject non-finite/out-of-range.
function addressValueToU32(value, what) {
  const number = +value;
  if (!Number.isFinite(number)) throw new TypeError(what + " must be finite");
  const integer = Math.trunc(number);
  if (integer < 0 || integer > 0xffffffff) {
    throw new TypeError(what + " is outside the unsigned 32-bit range");
  }
  return integer;
}

function descriptorObject(descriptor, name) {
  if ((typeof descriptor !== "object" && typeof descriptor !== "function") || descriptor === null) {
    throw new TypeError("WebAssembly." + name + " descriptor must be an object");
  }
  return descriptor;
}

const internalHandle = {};
const functionCache = new Map();
const memoryCache = new Map();
const tableCache = new Map();
const globalCache = new Map();

const ROOT_MODULE = 0;
const ROOT_INSTANCE = 1;
const ROOT_FUNCTION = 2;
const ROOT_MEMORY = 3;
const ROOT_TABLE = 4;
const ROOT_GLOBAL = 5;

// WebAssembly JS object caches preserve address identity without becoming owners. Native store
// roots follow the wrappers' actual reachability; releasing one root traces the remaining store
// graph, so an exported function can keep its defining instance alive while unrelated entities
// are reclaimed.
const handleFinalizer = new FinalizationRegistry(record => {
  try { __wasm.release(record.token); } catch {}
  if (record.cache) {
    const current = record.cache.get(record.key);
    if (!current || current.deref() === undefined) record.cache.delete(record.key);
  }
});

function cachedHandle(cache, key) {
  const reference = cache.get(key);
  const value = reference && reference.deref();
  if (value !== undefined) return value;
  if (reference) cache.delete(key);
  return undefined;
}

function retainWrapper(target, kind, address, cache) {
  const token = __wasm.retain(kind, address);
  target._root = token;
  if (cache) cache.set(address, new WeakRef(target));
  handleFinalizer.register(target, { token, cache, key: address });
  return target;
}

function retainImportKeepers(target, keepers) {
  if (!keepers || keepers.length === 0) return;
  if (!target._wasmKeepers) target._wasmKeepers = [];
  if (!target._wasmKeepers.includes(keepers)) target._wasmKeepers.push(keepers);
}

// Wrap a store function address as a callable.
function funcFromAddr(faddr, keepers) {
  const cached = cachedHandle(functionCache, faddr);
  if (cached !== undefined) {
    retainImportKeepers(cached, keepers);
    return cached;
  }
  const fn = (...args) => {
    let r;
    try { r = __wasm.call(faddr, args); } catch (err) { throw wrapWasmError(err); }
    return r.length === 0 ? undefined : r.length === 1 ? r[0] : r;
  };
  fn._funcAddr = faddr;
  retainImportKeepers(fn, keepers);
  return retainWrapper(fn, ROOT_FUNCTION, faddr, functionCache);
}

class Module {
  constructor(bytes) {
    try { this._id = __wasm.compile(toBytes(bytes)); }
    catch (e) { throw wrapWasmError(e); }
    retainWrapper(this, ROOT_MODULE, this._id, null);
  }
  static exports(module) { return __wasm.moduleExports(module._id); }
  static imports(module) { return __wasm.moduleImports(module._id); }
  static customSections() { return []; }
}

class Memory {
  constructor(descriptor) {
    if (descriptor && descriptor.__wasmHandle === internalHandle) {
      this._addr = descriptor.addr; // bound to an existing store memory (an export)
    } else {
      descriptor = descriptorObject(descriptor, "Memory");
      if (!("initial" in descriptor)) throw new TypeError("Memory descriptor requires initial");
      if (descriptor.address !== undefined && descriptor.address !== "i32") {
        throw new TypeError("TRust JS currently supports only i32 WebAssembly memories");
      }
      const initial = addressValueToU32(descriptor.initial, "Memory initial");
      const maximum = descriptor.maximum === undefined
        ? undefined
        : addressValueToU32(descriptor.maximum, "Memory maximum");
      if (initial > 65536 || maximum !== undefined && (maximum > 65536 || maximum < initial)) {
        throw new RangeError("invalid WebAssembly memory limits");
      }
      this._addr = __wasm.allocMemory(initial, maximum);
    }
    this._buf = __wasm.memBuffer(this._addr);
    retainWrapper(this, ROOT_MEMORY, this._addr, memoryCache);
  }
  get buffer() { return this._buf = __wasm.memBuffer(this._addr); }
  grow(delta) {
    const prev = __wasm.memGrow(this._addr, addressValueToU32(delta, "Memory grow delta"));
    if (prev < 0) throw new RangeError("WebAssembly.Memory.grow() failed");
    this._buf = __wasm.memBuffer(this._addr);
    return prev;
  }
}

function memoryFromAddr(addr) {
  const cached = cachedHandle(memoryCache, addr);
  return cached === undefined
    ? new Memory({ __wasmHandle: internalHandle, addr })
    : cached;
}

class Table {
  constructor(descriptor, value) {
    if (descriptor && descriptor.__wasmHandle === internalHandle) {
      this._addr = descriptor.addr;
      this._wasmElements = [];
      retainWrapper(this, ROOT_TABLE, this._addr, tableCache);
      return;
    }
    descriptor = descriptorObject(descriptor, "Table");
    if (!("element" in descriptor)) throw new TypeError("Table descriptor requires element");
    if (!("initial" in descriptor)) throw new TypeError("Table descriptor requires initial");
    if (descriptor.element !== "anyfunc") {
      throw new TypeError("TRust JS currently supports only anyfunc WebAssembly tables");
    }
    if (descriptor.address !== undefined && descriptor.address !== "i32") {
      throw new TypeError("TRust JS currently supports only i32 WebAssembly tables");
    }
    const initial = addressValueToU32(descriptor.initial, "Table initial");
    const maximum = descriptor.maximum === undefined
      ? undefined
      : addressValueToU32(descriptor.maximum, "Table maximum");
    if (maximum !== undefined && maximum < initial) {
      throw new RangeError("invalid WebAssembly table limits");
    }
    this._addr = __wasm.allocTable(initial, maximum);
    this._wasmElements = [];
    retainWrapper(this, ROOT_TABLE, this._addr, tableCache);
    if (value !== undefined) {
      for (let index = 0; index < initial; index++) this.set(index, value);
    }
  }
  get length() { return __wasm.tableSize(this._addr); }
  get(i) {
    const faddr = __wasm.tableGet(this._addr, addressValueToU32(i, "Table index"));
    return faddr < 0 ? null : funcFromAddr(faddr, this._wasmKeepers);
  }
  set(i, value) {
    i = addressValueToU32(i, "Table index");
    if (value == null) {
      this._wasmElements[i] = undefined;
      return __wasm.tableSet(this._addr, i, -1);
    }
    if (typeof value !== "function" || value._funcAddr === undefined) {
      throw new TypeError("Table.set expects an exported wasm function or null");
    }
    this._wasmElements[i] = value;
    __wasm.tableSet(this._addr, i, value._funcAddr);
  }
}

function tableFromAddr(addr) {
  const cached = cachedHandle(tableCache, addr);
  return cached === undefined
    ? new Table({ __wasmHandle: internalHandle, addr })
    : cached;
}

class Global {
  constructor(descriptor, value) {
    if (descriptor && descriptor.__wasmHandle === internalHandle) {
      this._addr = descriptor.addr;
      const info = __wasm.globalInfo(this._addr);
      this._mutable = info.mutable;
      this._type = info.value;
    } else {
      descriptor = descriptorObject(descriptor, "Global");
      if (!("value" in descriptor)) throw new TypeError("Global descriptor requires value");
      const type = String(descriptor.value);
      if (type !== "i32" && type !== "i64" && type !== "f32" && type !== "f64") {
        throw new TypeError("unsupported WebAssembly global value type");
      }
      this._mutable = !!descriptor.mutable;
      this._type = type;
      const defaultValue = type === "i64" ? 0n : 0;
      this._addr = __wasm.allocGlobal(value !== undefined ? value : defaultValue, this._mutable, type);
    }
    retainWrapper(this, ROOT_GLOBAL, this._addr, globalCache);
  }
  get value() { return __wasm.globalGet(this._addr); }
  set value(v) {
    if (!this._mutable) throw new TypeError("cannot set the value of an immutable global");
    __wasm.globalSet(this._addr, v);
  }
  valueOf() { return this.value; }
}

function globalFromAddr(addr) {
  const cached = cachedHandle(globalCache, addr);
  return cached === undefined
    ? new Global({ __wasmHandle: internalHandle, addr })
    : cached;
}

function buildExports(exportsMeta, importedMemory, keepers) {
  const exports = Object.create(null);
  let memory = importedMemory || null;
  for (const e of exportsMeta) {
    if (e.kind === "memory") {
      if (!memory) memory = memoryFromAddr(e.addr);
      exports[e.name] = memory;
    }
  }
  for (const e of exportsMeta) {
    if (e.kind === "function") {
      const faddr = e.addr;
      exports[e.name] = funcFromAddr(faddr, keepers);
    } else if (e.kind === "global") {
      exports[e.name] = globalFromAddr(e.addr);
    } else if (e.kind === "table") {
      const table = tableFromAddr(e.addr);
      retainImportKeepers(table, keepers);
      exports[e.name] = table;
    }
  }
  return exports;
}

// Resolve the JS import object into the flat, module-order array the native op consumes: {fn} for
// functions, and store addresses for memory/table/global (so imported entities are shared).
function resolveImports(module, importObject) {
  const descriptors = Module.imports(module);
  if (descriptors.length !== 0 && importObject === undefined) {
    throw new TypeError("WebAssembly imports require an import object");
  }
  const io = importObject || {};
  if ((typeof io !== "object" && typeof io !== "function") || io === null) {
    throw new TypeError("WebAssembly import object must be an object");
  }
  const types = __wasm.moduleImportTypes(module._id);
  let memory = null;
  const resolved = descriptors.map((imp, index) => {
    const namespace = io[imp.module];
    if ((typeof namespace !== "object" && typeof namespace !== "function") || namespace === null) {
      throw new TypeError("WebAssembly import namespace " + imp.module + " must be an object");
    }
    const v = namespace[imp.name];
    if (imp.kind === "function") {
      if (typeof v !== "function") throw new LinkError("function import is not callable");
      return { fn: v };
    }
    if (imp.kind === "memory") {
      if (!(v instanceof Memory)) throw new LinkError("memory import is not a WebAssembly.Memory");
      memory = v;
      return { memAddr: v._addr, memory: v };
    }
    if (imp.kind === "table") {
      if (!(v instanceof Table)) throw new LinkError("table import is not a WebAssembly.Table");
      return { tableAddr: v._addr, table: v };
    }
    if (imp.kind === "global") {
      if (v instanceof Global) return { globalAddr: v._addr, global: v };
      const type = types[index].value;
      if (type === "i64" ? typeof v !== "bigint" : typeof v !== "number") {
        throw new LinkError("global import has the wrong JavaScript value type");
      }
      const g = new Global({ value: type, mutable: false }, v);
      return { globalAddr: g._addr, global: g };
    }
    return {};
  });
  return { resolved, memory };
}

class Instance {
  constructor(module, importObject) {
    if (!(module instanceof Module)) throw new TypeError("WebAssembly.Instance expects a Module");
    const { resolved, memory } = resolveImports(module, importObject);
    let res;
    try { res = __wasm.instantiate(module._id, resolved); }
    catch (e) { throw wrapWasmError(e); }
    this._inst = res.inst;
    this._wasmKeepers = resolved;
    for (const entry of resolved) {
      if (entry.table) retainImportKeepers(entry.table, resolved);
    }
    this.exports = buildExports(res.exports, memory, resolved);
    retainWrapper(this, ROOT_INSTANCE, this._inst, null);
  }
}

function validate(bytes) {
  try { return __wasm.validate(toBytes(bytes)); } catch { return false; }
}

async function compile(bytes) {
  return new Module(bytes);
}

async function instantiate(source, importObject) {
  if (source instanceof Module) return new Instance(source, importObject);
  const module = new Module(source);
  const instance = new Instance(module, importObject);
  return { module, instance };
}

globalThis.WebAssembly = {
  validate,
  compile,
  instantiate,
  Module,
  Instance,
  Memory,
  Table,
  Global,
  CompileError,
  LinkError,
  RuntimeError,
};
