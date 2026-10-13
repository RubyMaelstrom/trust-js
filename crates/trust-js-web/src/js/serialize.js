// HTML Standard §2.7 structured serialization, plus a bounded binary representation for values
// sent across TRust JS Worker threads. A single record serializer feeds both `structuredClone()` and
// the wire encoder so cycles, duplicate identities, transfer ordering, sparse arrays, and views
// cannot diverge between the two paths.

const T_UNDEFINED = 0;
const T_NULL = 1;
const T_FALSE = 2;
const T_TRUE = 3;
const T_NUMBER = 4;
const T_STRING = 5;
const T_BIGINT = 6;
const T_REF = 7;
const T_ARRAY = 8;
const T_OBJECT = 9;
const T_DATE = 10;
const T_REGEXP = 11;
const T_MAP = 12;
const T_SET = 13;
const T_ARRAYBUFFER = 14;
const T_TYPEDARRAY = 15;
const T_DATAVIEW = 16;
const T_ERROR = 17;
const T_BOOLOBJ = 18;
const T_NUMOBJ = 19;
const T_STROBJ = 20;
const T_BIGINTOBJ = 21;

const CLONE_WIRE_LIMIT = 64 * 1024 * 1024;
const CLONE_RECORD_LIMIT = 1_000_000;
const CLONE_DEPTH_LIMIT = 512;

const TA_KINDS = [
  "Int8Array", "Uint8Array", "Uint8ClampedArray", "Int16Array", "Uint16Array",
  "Int32Array", "Uint32Array", "Float32Array", "Float64Array",
  "BigInt64Array", "BigUint64Array",
];
const ERROR_NAMES = [
  "Error", "TypeError", "RangeError", "ReferenceError", "SyntaxError", "EvalError", "URIError",
];
const ARRAY_BUFFER_TRANSFER = ArrayBuffer.prototype.transfer;

function cloneError(message = "value could not be cloned") {
  return new DOMException(message, "DataCloneError");
}

function normalizeTransferSequence(value) {
  if (value === undefined) return [];
  if (value === null || (typeof value !== "object" && typeof value !== "function")) {
    throw new TypeError("transfer must be an iterable sequence");
  }
  const iterator = value[Symbol.iterator];
  if (typeof iterator !== "function") throw new TypeError("transfer must be an iterable sequence");
  return [...value];
}

function structuredCloneTransferList(options) {
  if (options === undefined || options === null) return [];
  if (typeof options !== "object" && typeof options !== "function") {
    throw new TypeError("structuredClone options must be an object");
  }
  return normalizeTransferSequence(options.transfer);
}

// Worker/MessagePort postMessage accepts either the legacy sequence directly or an options
// dictionary. Keep that union conversion at the call boundary; StructuredSerialize itself only
// receives a concrete sequence.
function postMessageTransferList(transferOrOptions) {
  if (transferOrOptions === undefined) return [];
  if (transferOrOptions !== null &&
      (typeof transferOrOptions === "object" || typeof transferOrOptions === "function")) {
    if (!("transfer" in transferOrOptions) &&
        typeof transferOrOptions[Symbol.iterator] === "function") {
      return [...transferOrOptions];
    }
    return normalizeTransferSequence(transferOrOptions.transfer);
  }
  throw new TypeError("postMessage transfer argument must be a sequence or options object");
}

function isSharedArrayBuffer(value) {
  return typeof SharedArrayBuffer === "function" && value instanceof SharedArrayBuffer;
}

function isDetachedArrayBuffer(value) {
  try {
    return value.detached === true;
  } catch {
    return true;
  }
}

function addCloneRecord(records, memory, source, record) {
  if (records.length >= CLONE_RECORD_LIMIT) throw cloneError("structured clone is too complex");
  const index = records.length;
  records.push(record);
  memory.set(source, index);
  return { ref: index };
}

function cloneArrayBufferBytes(buffer) {
  if (isDetachedArrayBuffer(buffer)) throw cloneError("detached ArrayBuffer cannot be cloned");
  try {
    const length = buffer.byteLength;
    const clone = buffer.resizable
      ? new ArrayBuffer(length, { maxByteLength: buffer.maxByteLength })
      : new ArrayBuffer(length);
    new Uint8Array(clone).set(new Uint8Array(buffer));
    return clone;
  } catch (error) {
    if (error && error.name === "RangeError") throw error;
    throw cloneError("ArrayBuffer could not be cloned");
  }
}

function unsupportedInternalSlotObject(value) {
  return value instanceof Promise ||
    (typeof WeakMap === "function" && value instanceof WeakMap) ||
    (typeof WeakSet === "function" && value instanceof WeakSet) ||
    (typeof WeakRef === "function" && value instanceof WeakRef) ||
    (typeof FinalizationRegistry === "function" && value instanceof FinalizationRegistry);
}

// HTML StructuredSerializeWithTransfer. Transfer-list entries are installed in `memory` before
// walking the value. Actual detachment happens only after the whole graph serialized successfully,
// so a getter or unsupported descendant can still throw without consuming any buffers.
function structuredSerialize(value, transferList) {
  const records = [];
  const memory = new Map();
  const transfers = [];

  for (const transferable of transferList) {
    if (!(transferable instanceof ArrayBuffer) || isSharedArrayBuffer(transferable)) {
      throw cloneError("transfer list contains a non-transferable value");
    }
    if (memory.has(transferable)) throw cloneError("transfer list contains a duplicate value");
    const holder = { type: "transfer", source: transferable, buffer: undefined };
    addCloneRecord(records, memory, transferable, holder);
    transfers.push(holder);
  }

  const serialize = (input, depth = 0) => {
    if (depth > CLONE_DEPTH_LIMIT) throw cloneError("structured clone is too deeply nested");
    if (input === undefined || input === null) return input;
    const inputType = typeof input;
    if (inputType === "boolean" || inputType === "number" || inputType === "string" ||
        inputType === "bigint") return input;
    if (inputType === "symbol" || inputType === "function") throw cloneError();
    if (memory.has(input)) return { ref: memory.get(input) };

    let record;
    let reference;
    if (input instanceof Date) {
      record = { type: "date", value: input.getTime() };
      return addCloneRecord(records, memory, input, record);
    }
    if (input instanceof RegExp) {
      record = { type: "regexp", source: input.source, flags: input.flags };
      return addCloneRecord(records, memory, input, record);
    }
    if (isSharedArrayBuffer(input)) {
      // TRust JS does not expose an isolated agent cluster, so HTML requires DataCloneError here.
      throw cloneError("SharedArrayBuffer cannot be cloned outside an isolated agent cluster");
    }
    if (input instanceof ArrayBuffer) {
      record = { type: "arraybuffer", buffer: cloneArrayBufferBytes(input) };
      return addCloneRecord(records, memory, input, record);
    }
    if (input instanceof DataView) {
      if (isDetachedArrayBuffer(input.buffer)) throw cloneError("detached DataView cannot be cloned");
      record = {
        type: "dataview",
        byteOffset: input.byteOffset,
        byteLength: input.byteLength,
        buffer: undefined,
      };
      reference = addCloneRecord(records, memory, input, record);
      record.buffer = serialize(input.buffer, depth + 1);
      return reference;
    }
    if (ArrayBuffer.isView(input)) {
      if (isDetachedArrayBuffer(input.buffer)) throw cloneError("detached typed array cannot be cloned");
      const kind = TA_KINDS.indexOf(input.constructor.name);
      if (kind === -1) throw cloneError("unsupported typed array");
      record = {
        type: "typedarray",
        kind,
        byteOffset: input.byteOffset,
        length: input.length,
        buffer: undefined,
      };
      reference = addCloneRecord(records, memory, input, record);
      record.buffer = serialize(input.buffer, depth + 1);
      return reference;
    }
    if (input instanceof Map) {
      record = { type: "map", entries: [] };
      reference = addCloneRecord(records, memory, input, record);
      const entries = [...input];
      for (const [key, entryValue] of entries) {
        record.entries.push([
          serialize(key, depth + 1),
          serialize(entryValue, depth + 1),
        ]);
      }
      return reference;
    }
    if (input instanceof Set) {
      record = { type: "set", entries: [] };
      reference = addCloneRecord(records, memory, input, record);
      const entries = [...input];
      for (const entry of entries) record.entries.push(serialize(entry, depth + 1));
      return reference;
    }
    if (input instanceof Error) {
      let name = input.name;
      if (!ERROR_NAMES.includes(name)) name = "Error";
      const descriptor = Object.getOwnPropertyDescriptor(input, "message");
      const message = descriptor && "value" in descriptor ? String(descriptor.value) : undefined;
      record = {
        type: "error",
        name,
        message,
        stack: typeof input.stack === "string" ? input.stack : "",
      };
      return addCloneRecord(records, memory, input, record);
    }

    if (unsupportedInternalSlotObject(input)) throw cloneError();
    const tag = Object.prototype.toString.call(input);
    if (tag === "[object Boolean]") {
      return addCloneRecord(records, memory, input, { type: "boxed", kind: "boolean", value: input.valueOf() });
    }
    if (tag === "[object Number]") {
      return addCloneRecord(records, memory, input, { type: "boxed", kind: "number", value: input.valueOf() });
    }
    if (tag === "[object String]") {
      return addCloneRecord(records, memory, input, { type: "boxed", kind: "string", value: input.valueOf() });
    }
    if (tag === "[object BigInt]") {
      return addCloneRecord(records, memory, input, { type: "boxed", kind: "bigint", value: input.valueOf() });
    }
    if (tag === "[object Symbol]") throw cloneError();

    const isArray = Array.isArray(input);
    record = {
      type: isArray ? "array" : "object",
      length: isArray ? input.length : undefined,
      properties: [],
    };
    reference = addCloneRecord(records, memory, input, record);
    // EnumerableOwnProperties(value, key) snapshots the keys. HasOwnProperty is checked again
    // before each Get because an earlier getter is allowed to delete a later property.
    const keys = Object.keys(input);
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(input, key)) continue;
      record.properties.push([key, serialize(input[key], depth + 1)]);
    }
    return reference;
  };

  const root = serialize(value);

  // Only now perform transfer steps. ArrayBuffer.prototype.transfer gives the holder ownership of
  // the same backing store and detaches the source. A late failure can consume earlier entries,
  // matching the standard's ordered transfer loop.
  for (const holder of transfers) {
    if (isDetachedArrayBuffer(holder.source)) throw cloneError("detached ArrayBuffer cannot be transferred");
    try {
      holder.buffer = ARRAY_BUFFER_TRANSFER.call(holder.source);
      holder.type = "arraybuffer";
      delete holder.source;
    } catch {
      throw cloneError("ArrayBuffer could not be transferred");
    }
  }

  return { root, records };
}

function createDataProperty(target, key, value) {
  Object.defineProperty(target, key, {
    value,
    writable: true,
    enumerable: true,
    configurable: true,
  });
}

function structuredDeserialize(serialized) {
  const memory = new Array(serialized.records.length);
  const active = new Set();
  const deserialize = (value, depth = 0) => {
    if (depth > CLONE_DEPTH_LIMIT) throw cloneError("structured clone is too deeply nested");
    if (value === null || typeof value !== "object") return value;
    const index = value.ref;
    if (!Number.isInteger(index) || index < 0 || index >= serialized.records.length) {
      throw cloneError("malformed structured clone record");
    }
    if (memory[index] !== undefined) return memory[index];
    if (active.has(index)) throw cloneError("malformed structured clone cycle");
    active.add(index);
    const record = serialized.records[index];
    let output;
    switch (record.type) {
      case "date": output = new Date(record.value); break;
      case "regexp": output = new RegExp(record.source, record.flags); break;
      case "arraybuffer": output = record.buffer; break;
      case "dataview": {
        const buffer = deserialize(record.buffer, depth + 1);
        output = new DataView(buffer, record.byteOffset, record.byteLength);
        break;
      }
      case "typedarray": {
        const constructor = globalThis[TA_KINDS[record.kind]];
        output = new constructor(deserialize(record.buffer, depth + 1), record.byteOffset, record.length);
        break;
      }
      case "map": {
        output = new Map();
        memory[index] = output;
        active.delete(index);
        for (const [key, entryValue] of record.entries) {
          output.set(deserialize(key, depth + 1), deserialize(entryValue, depth + 1));
        }
        return output;
      }
      case "set": {
        output = new Set();
        memory[index] = output;
        active.delete(index);
        for (const entry of record.entries) output.add(deserialize(entry, depth + 1));
        return output;
      }
      case "error": {
        const constructor = globalThis[record.name] ?? Error;
        output = record.message === undefined ? new constructor() : new constructor(record.message);
        if (record.stack) output.stack = record.stack;
        break;
      }
      case "boxed": output = Object(record.value); break;
      case "array": output = new Array(record.length); break;
      case "object": output = {}; break;
      default: throw cloneError("malformed structured clone record");
    }
    memory[index] = output;
    active.delete(index);
    if (record.type === "array" || record.type === "object") {
      for (const [key, propertyValue] of record.properties) {
        createDataProperty(output, key, deserialize(propertyValue, depth + 1));
      }
    }
    return output;
  };
  return deserialize(serialized.root);
}

function structuredCloneValue(value, options) {
  return structuredDeserialize(structuredSerialize(value, structuredCloneTransferList(options)));
}

// Encode already-serialized records without touching author objects again. This preserves the
// standard's single getter pass and ensures transferred ArrayBuffers are detached before native
// worker delivery while their moved bytes are copied into the cross-thread message exactly once.
function encodeCloneWire(serialized) {
  let buffer = new Uint8Array(256);
  let length = 0;
  let dataView = new DataView(buffer.buffer);
  const emitted = new Map();
  const utf8 = new TextEncoder();

  const ensure = (extra) => {
    if (!Number.isSafeInteger(extra) || extra < 0 || length + extra > CLONE_WIRE_LIMIT) {
      throw cloneError("structured clone exceeds the wire-size limit");
    }
    if (length + extra <= buffer.length) return;
    let capacity = buffer.length;
    while (capacity < length + extra) capacity = Math.min(CLONE_WIRE_LIMIT, capacity * 2);
    if (capacity < length + extra) throw cloneError("structured clone exceeds the wire-size limit");
    const next = new Uint8Array(capacity);
    next.set(buffer.subarray(0, length));
    buffer = next;
    dataView = new DataView(buffer.buffer);
  };
  const u8 = (number) => { ensure(1); buffer[length++] = number & 0xff; };
  const u32 = (number) => {
    if (!Number.isInteger(number) || number < 0 || number > 0xffffffff) {
      throw cloneError("structured clone integer is out of range");
    }
    ensure(4);
    dataView.setUint32(length, number, true);
    length += 4;
  };
  const f64 = (number) => { ensure(8); dataView.setFloat64(length, number, true); length += 8; };
  const raw = (bytes) => { ensure(bytes.length); buffer.set(bytes, length); length += bytes.length; };
  const string = (value) => {
    const text = String(value);
    // UTF-8 needs at most three bytes per UTF-16 code unit. Reject before TextEncoder allocates a
    // temporary larger than the entire bounded wire message.
    if (text.length > Math.floor(CLONE_WIRE_LIMIT / 3)) {
      throw cloneError("structured clone string exceeds the wire-size limit");
    }
    const bytes = utf8.encode(text);
    u32(bytes.length);
    raw(bytes);
  };

  const write = (value, depth = 0) => {
    if (depth > CLONE_DEPTH_LIMIT) throw cloneError("structured clone is too deeply nested");
    if (value === undefined) return u8(T_UNDEFINED);
    if (value === null) return u8(T_NULL);
    const valueType = typeof value;
    if (valueType === "boolean") return u8(value ? T_TRUE : T_FALSE);
    if (valueType === "number") { u8(T_NUMBER); return f64(value); }
    if (valueType === "string") { u8(T_STRING); return string(value); }
    if (valueType === "bigint") { u8(T_BIGINT); return string(value.toString()); }

    const recordIndex = value.ref;
    if (!Number.isInteger(recordIndex) || recordIndex < 0 || recordIndex >= serialized.records.length) {
      throw cloneError("malformed structured clone record");
    }
    if (emitted.has(recordIndex)) { u8(T_REF); return u32(emitted.get(recordIndex)); }
    if (emitted.size >= CLONE_RECORD_LIMIT) throw cloneError("structured clone is too complex");
    emitted.set(recordIndex, emitted.size);
    const record = serialized.records[recordIndex];

    switch (record.type) {
      case "date": u8(T_DATE); return f64(record.value);
      case "regexp": u8(T_REGEXP); string(record.source); return string(record.flags);
      case "arraybuffer": {
        const bytes = new Uint8Array(record.buffer);
        u8(T_ARRAYBUFFER);
        u8(record.buffer.resizable ? 1 : 0);
        u32(bytes.length);
        u32(record.buffer.resizable ? record.buffer.maxByteLength : bytes.length);
        return raw(bytes);
      }
      case "dataview":
        u8(T_DATAVIEW); u32(record.byteOffset); u32(record.byteLength);
        return write(record.buffer, depth + 1);
      case "typedarray":
        u8(T_TYPEDARRAY); u8(record.kind); u32(record.byteOffset); u32(record.length);
        return write(record.buffer, depth + 1);
      case "map":
        u8(T_MAP); u32(record.entries.length);
        for (const [key, entryValue] of record.entries) {
          write(key, depth + 1);
          write(entryValue, depth + 1);
        }
        return;
      case "set":
        u8(T_SET); u32(record.entries.length);
        for (const entry of record.entries) write(entry, depth + 1);
        return;
      case "error":
        u8(T_ERROR); u8(ERROR_NAMES.indexOf(record.name));
        u8(record.message === undefined ? 0 : 1);
        if (record.message !== undefined) string(record.message);
        return string(record.stack);
      case "boxed":
        if (record.kind === "boolean") { u8(T_BOOLOBJ); return u8(record.value ? 1 : 0); }
        if (record.kind === "number") { u8(T_NUMOBJ); return f64(record.value); }
        if (record.kind === "string") { u8(T_STROBJ); return string(record.value); }
        u8(T_BIGINTOBJ); return string(record.value.toString());
      case "array":
        u8(T_ARRAY); u32(record.length); u32(record.properties.length);
        for (const [key, propertyValue] of record.properties) {
          string(key);
          write(propertyValue, depth + 1);
        }
        return;
      case "object":
        u8(T_OBJECT); u32(record.properties.length);
        for (const [key, propertyValue] of record.properties) {
          string(key);
          write(propertyValue, depth + 1);
        }
        return;
      default: throw cloneError("malformed structured clone record");
    }
  };

  write(serialized.root);
  return buffer.subarray(0, length);
}

function serializeForClone(value, transferOrOptions) {
  const transferList = postMessageTransferList(transferOrOptions);
  return encodeCloneWire(structuredSerialize(value, transferList));
}

function deserializeClone(input) {
  try {
    if (!(input instanceof Uint8Array) || input.byteLength > CLONE_WIRE_LIMIT) {
      throw cloneError("malformed clone data");
    }
    const buffer = input;
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const utf8 = new TextDecoder("utf-8", { fatal: true });
    const memory = [];
    const pending = Symbol("pending clone record");
    let position = 0;

    const requireBytes = (count) => {
      if (!Number.isSafeInteger(count) || count < 0 || position + count > buffer.length) {
        throw cloneError("truncated clone data");
      }
    };
    const u8 = () => { requireBytes(1); return buffer[position++]; };
    const u32 = () => {
      requireBytes(4);
      const value = view.getUint32(position, true);
      position += 4;
      return value;
    };
    const f64 = () => {
      requireBytes(8);
      const value = view.getFloat64(position, true);
      position += 8;
      return value;
    };
    const string = () => {
      const byteLength = u32();
      requireBytes(byteLength);
      const value = utf8.decode(buffer.subarray(position, position + byteLength));
      position += byteLength;
      return value;
    };
    const reserve = () => {
      if (memory.length >= CLONE_RECORD_LIMIT) throw cloneError("clone data is too complex");
      const index = memory.length;
      memory.push(pending);
      return index;
    };
    const readCount = () => {
      const count = u32();
      if (count > CLONE_RECORD_LIMIT) throw cloneError("clone data is too complex");
      return count;
    };

    const read = (depth = 0) => {
      if (depth > CLONE_DEPTH_LIMIT) throw cloneError("clone data is too deeply nested");
      const tag = u8();
      switch (tag) {
        case T_UNDEFINED: return undefined;
        case T_NULL: return null;
        case T_FALSE: return false;
        case T_TRUE: return true;
        case T_NUMBER: return f64();
        case T_STRING: return string();
        case T_BIGINT: return BigInt(string());
        case T_REF: {
          const index = u32();
          if (index >= memory.length || memory[index] === pending) throw cloneError("invalid clone reference");
          return memory[index];
        }
        case T_DATE: {
          const value = new Date(f64());
          memory.push(value);
          return value;
        }
        case T_REGEXP: {
          const value = new RegExp(string(), string());
          memory.push(value);
          return value;
        }
        case T_ARRAYBUFFER: {
          const resizable = u8();
          if (resizable > 1) throw cloneError("invalid ArrayBuffer flags");
          const byteLength = u32();
          const maxByteLength = u32();
          if (maxByteLength < byteLength || (!resizable && maxByteLength !== byteLength)) {
            throw cloneError("invalid ArrayBuffer length");
          }
          requireBytes(byteLength);
          const value = resizable
            ? new ArrayBuffer(byteLength, { maxByteLength })
            : new ArrayBuffer(byteLength);
          new Uint8Array(value).set(buffer.subarray(position, position + byteLength));
          position += byteLength;
          memory.push(value);
          return value;
        }
        case T_DATAVIEW: {
          const byteOffset = u32();
          const byteLength = u32();
          const index = reserve();
          const value = new DataView(read(depth + 1), byteOffset, byteLength);
          memory[index] = value;
          return value;
        }
        case T_TYPEDARRAY: {
          const kind = u8();
          if (kind >= TA_KINDS.length) throw cloneError("invalid typed array kind");
          const byteOffset = u32();
          const length = u32();
          const index = reserve();
          const value = new globalThis[TA_KINDS[kind]](read(depth + 1), byteOffset, length);
          memory[index] = value;
          return value;
        }
        case T_MAP: {
          const value = new Map();
          memory.push(value);
          const count = readCount();
          for (let index = 0; index < count; index++) {
            value.set(read(depth + 1), read(depth + 1));
          }
          return value;
        }
        case T_SET: {
          const value = new Set();
          memory.push(value);
          const count = readCount();
          for (let index = 0; index < count; index++) value.add(read(depth + 1));
          return value;
        }
        case T_ERROR: {
          const errorKind = u8();
          if (errorKind >= ERROR_NAMES.length) throw cloneError("invalid error kind");
          const hasMessage = u8();
          if (hasMessage > 1) throw cloneError("invalid error message flag");
          const message = hasMessage ? string() : undefined;
          const constructor = globalThis[ERROR_NAMES[errorKind]] ?? Error;
          const value = message === undefined ? new constructor() : new constructor(message);
          const stack = string();
          if (stack) value.stack = stack;
          memory.push(value);
          return value;
        }
        case T_BOOLOBJ: {
          const flag = u8();
          if (flag > 1) throw cloneError("invalid Boolean object");
          const value = new Boolean(flag === 1);
          memory.push(value);
          return value;
        }
        case T_NUMOBJ: {
          const value = new Number(f64());
          memory.push(value);
          return value;
        }
        case T_STROBJ: {
          const value = new String(string());
          memory.push(value);
          return value;
        }
        case T_BIGINTOBJ: {
          const value = Object(BigInt(string()));
          memory.push(value);
          return value;
        }
        case T_ARRAY: {
          const arrayLength = u32();
          const value = new Array(arrayLength);
          memory.push(value);
          const count = readCount();
          for (let index = 0; index < count; index++) {
            createDataProperty(value, string(), read(depth + 1));
          }
          return value;
        }
        case T_OBJECT: {
          const value = {};
          memory.push(value);
          const count = readCount();
          for (let index = 0; index < count; index++) {
            createDataProperty(value, string(), read(depth + 1));
          }
          return value;
        }
        default: throw cloneError("unknown clone tag");
      }
    };

    const value = read();
    if (position !== buffer.length) throw cloneError("trailing clone data");
    return value;
  } catch (error) {
    if (error && error.name === "DataCloneError") throw error;
    throw cloneError("malformed clone data");
  }
}

// Private bridge for the runtime's cross-thread Worker transport.
globalThis.__serializeForClone = serializeForClone;
globalThis.__deserializeClone = deserializeClone;
