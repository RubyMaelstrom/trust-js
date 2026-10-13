// Encoding Standard TextEncoder/TextDecoder over native, stateful WHATWG codecs, plus base64.

const EMPTY_ENCODING_INPUT = new Uint8Array(0);
const TEXT_DECODER_FINALIZER = new FinalizationRegistry((id) => __encoding.decoderDrop(id));
const TEXT_DECODER_STATE = new WeakMap();

function encodingUSVString(value) {
  const input = String(value);
  let output = null;
  let copiedThrough = 0;
  for (let i = 0; i < input.length; i++) {
    const first = input.charCodeAt(i);
    if (first >= 0xd800 && first <= 0xdbff) {
      const second = input.charCodeAt(i + 1);
      if (second >= 0xdc00 && second <= 0xdfff) {
        i++;
      } else {
        if (output === null) output = [];
        output.push(input.slice(copiedThrough, i), "\ufffd");
        copiedThrough = i + 1;
      }
    } else if (first >= 0xdc00 && first <= 0xdfff) {
      if (output === null) output = [];
      output.push(input.slice(copiedThrough, i), "\ufffd");
      copiedThrough = i + 1;
    }
  }
  if (output === null) return input;
  output.push(input.slice(copiedThrough));
  return output.join("");
}

function decoderInput(value) {
  if (value === undefined) return EMPTY_ENCODING_INPUT;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new TypeError("TextDecoder.decode expects a BufferSource");
}

class TextEncoder {
  get encoding() {
    return "utf-8";
  }
  encode(input = "") {
    return __encoding.encode(encodingUSVString(input));
  }
  encodeInto(source, destination) {
    if (arguments.length < 2) throw new TypeError("TextEncoder.encodeInto requires 2 arguments");
    if (!(destination instanceof Uint8Array)) {
      throw new TypeError("TextEncoder.encodeInto destination must be a Uint8Array");
    }
    source = String(source);
    let read = 0;
    let written = 0;
    while (read < source.length) {
      const first = source.charCodeAt(read);
      let codePoint;
      let codeUnits = 1;
      if (first >= 0xd800 && first <= 0xdbff) {
        const second = source.charCodeAt(read + 1);
        if (second >= 0xdc00 && second <= 0xdfff) {
          codePoint = 0x10000 + ((first - 0xd800) << 10) + second - 0xdc00;
          codeUnits = 2;
        } else {
          codePoint = 0xfffd;
        }
      } else if (first >= 0xdc00 && first <= 0xdfff) {
        codePoint = 0xfffd;
      } else {
        codePoint = first;
      }
      const needed = codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;
      if (written + needed > destination.length) break;
      if (needed === 1) {
        destination[written++] = codePoint;
      } else if (needed === 2) {
        destination[written++] = 0xc0 | (codePoint >> 6);
        destination[written++] = 0x80 | (codePoint & 0x3f);
      } else if (needed === 3) {
        destination[written++] = 0xe0 | (codePoint >> 12);
        destination[written++] = 0x80 | ((codePoint >> 6) & 0x3f);
        destination[written++] = 0x80 | (codePoint & 0x3f);
      } else {
        destination[written++] = 0xf0 | (codePoint >> 18);
        destination[written++] = 0x80 | ((codePoint >> 12) & 0x3f);
        destination[written++] = 0x80 | ((codePoint >> 6) & 0x3f);
        destination[written++] = 0x80 | (codePoint & 0x3f);
      }
      read += codeUnits;
    }
    return { read, written };
  }
}

class TextDecoder {
  constructor(label = "utf-8", options = {}) {
    options = options && typeof options === "object" ? options : {};
    const fatal = !!options.fatal;
    const ignoreBOM = !!options.ignoreBOM;
    const [id, encoding] = __encoding.decoderStart(String(label), fatal, ignoreBOM);
    TEXT_DECODER_STATE.set(this, { id, encoding, fatal, ignoreBOM });
    TEXT_DECODER_FINALIZER.register(this, id);
  }
  get encoding() {
    const state = TEXT_DECODER_STATE.get(this);
    if (!state) throw new TypeError("TextDecoder getter called on incompatible receiver");
    return state.encoding;
  }
  get fatal() {
    const state = TEXT_DECODER_STATE.get(this);
    if (!state) throw new TypeError("TextDecoder getter called on incompatible receiver");
    return state.fatal;
  }
  get ignoreBOM() {
    const state = TEXT_DECODER_STATE.get(this);
    if (!state) throw new TypeError("TextDecoder getter called on incompatible receiver");
    return state.ignoreBOM;
  }
  decode(input = undefined, options = {}) {
    const state = TEXT_DECODER_STATE.get(this);
    if (!state) throw new TypeError("TextDecoder.decode called on incompatible receiver");
    // Web IDL converts the BufferSource reference before the options dictionary. WHATWG Encoding
    // §7.2 copies its bytes only after those conversions, so a `stream` getter may detach it first;
    // a detached view then contributes the empty byte sequence.
    let bytes = decoderInput(input);
    options = options && typeof options === "object" ? options : {};
    const stream = !!options.stream;
    if (bytes.byteLength === 0) bytes = EMPTY_ENCODING_INPUT;
    return __encoding.decoderPush(state.id, bytes, stream);
  }
}

Object.defineProperty(TextEncoder.prototype, Symbol.toStringTag, {
  value: "TextEncoder", configurable: true,
});
Object.defineProperty(TextDecoder.prototype, Symbol.toStringTag, {
  value: "TextDecoder", configurable: true,
});

const B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function btoa(data) {
  const s = String(data);
  let out = "";
  for (let i = 0; i < s.length; i += 3) {
    const cs = [s.charCodeAt(i), s.charCodeAt(i + 1), s.charCodeAt(i + 2)];
    if (cs[0] > 255 || cs[1] > 255 || cs[2] > 255) {
      throw new DOMException("btoa: character beyond latin1 range", "InvalidCharacterError");
    }
    const n = (cs[0] << 16) | ((cs[1] || 0) << 8) | (cs[2] || 0);
    out += B64_ALPHABET[(n >> 18) & 63];
    out += B64_ALPHABET[(n >> 12) & 63];
    out += i + 1 < s.length ? B64_ALPHABET[(n >> 6) & 63] : "=";
    out += i + 2 < s.length ? B64_ALPHABET[n & 63] : "=";
  }
  return out;
}

function atob(data) {
  let s = String(data).replace(/[\t\n\f\r ]/g, "");
  if (s.length % 4 === 0) s = s.replace(/==?$/, "");
  if (s.length % 4 === 1 || /[^A-Za-z0-9+/]/.test(s)) {
    throw new DOMException("atob: invalid base64", "InvalidCharacterError");
  }
  let out = "";
  for (let i = 0; i < s.length; i += 4) {
    const bits = [0, 1, 2, 3].map((j) =>
      j + i < s.length ? B64_ALPHABET.indexOf(s[i + j]) : 0
    );
    const n = (bits[0] << 18) | (bits[1] << 12) | (bits[2] << 6) | bits[3];
    out += String.fromCharCode((n >> 16) & 255);
    if (i + 2 < s.length) out += String.fromCharCode((n >> 8) & 255);
    if (i + 3 < s.length) out += String.fromCharCode(n & 255);
  }
  return out;
}

function structuredClone(value, options = undefined) {
  // HTML StructuredSerializeWithTransfer/StructuredDeserializeWithTransfer. The implementation
  // lives beside the Worker wire codec in serialize.js so both paths share one serialization
  // memory, property algorithm, and transfer ordering.
  return structuredCloneValue(value, options);
}

globalThis.TextEncoder = TextEncoder;
globalThis.TextDecoder = TextDecoder;
globalThis.btoa = btoa;
globalThis.atob = atob;
globalThis.structuredClone = structuredClone;
