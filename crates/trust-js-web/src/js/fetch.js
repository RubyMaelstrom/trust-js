// Headers/Request/Response + fetch over the native __http.request op. Bodies are buffered; a body
// can be consumed once, either through `text()`/`json()`/… or by reading its `.body` ReadableStream
// (see streams.js). The two share one "consumed" flag.

function toByteString(value, context) {
  const string = String(value);
  for (let i = 0; i < string.length; ++i) {
    if (string.charCodeAt(i) > 0xff) {
      throw new TypeError(`${context} is not a ByteString`);
    }
  }
  return string;
}

function normalizeHeaderName(name) {
  name = toByteString(name, "header name");
  if (name === "" || /[^\x21-\x7e]/.test(name) || /[()<>@,;:\\"/[\]?={} \t]/.test(name)) {
    throw new TypeError(`invalid header name '${name}'`);
  }
  return name.toLowerCase();
}

// Fetch §2.2.2: normalization removes HTTP whitespace (SP, HTAB, CR, and LF)
// only at the ends; validation then rejects NUL and every remaining CR/LF.
function normalizeHeaderValue(value) {
  value = toByteString(value, "header value").replace(/^[\t\n\r ]+|[\t\n\r ]+$/g, "");
  if (/[\0\r\n]/.test(value)) throw new TypeError("invalid header value");
  return value;
}

const HEADER_ITERATOR_STATE = new WeakMap();
const HEADER_ITERATOR_PROTOTYPE = Object.create(
  Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]()))
);

function headerPairsToIterate(headers) {
  const pairs = [];
  for (const name of [...headers._map.keys()].sort()) {
    const values = headers._map.get(name).values;
    if (name === "set-cookie") {
      for (const value of values) pairs.push([name, value]);
    } else {
      pairs.push([name, values.join(", ")]);
    }
  }
  return pairs;
}

Object.defineProperty(HEADER_ITERATOR_PROTOTYPE, "next", {
  configurable: true,
  enumerable: true,
  writable: true,
  value: function next() {
    const state = HEADER_ITERATOR_STATE.get(this);
    if (!state) throw new TypeError("Headers iterator next called on an incompatible receiver");
    // Web IDL iterable objects expose the current value-pair list at each step. Recomputing the
    // sorted-and-combined view gives mutations their specified live indexed-iteration behavior.
    const pairs = headerPairsToIterate(state.headers);
    if (state.index >= pairs.length) return { value: undefined, done: true };
    const pair = pairs[state.index++];
    if (state.kind === "key") return { value: pair[0], done: false };
    if (state.kind === "value") return { value: pair[1], done: false };
    return { value: pair, done: false };
  },
});

function createHeaderIterator(headers, kind) {
  const iterator = Object.create(HEADER_ITERATOR_PROTOTYPE);
  HEADER_ITERATOR_STATE.set(iterator, { headers, kind, index: 0 });
  return iterator;
}

class Headers {
  constructor(init) {
    this._map = new Map(); // lower-name -> { name, values }
    this._guard = "none";
    if (init !== undefined) this._fill(init);
  }
  _fill(init) {
    if ((typeof init !== "object" || init === null) && typeof init !== "function") {
      throw new TypeError("Headers init must be a sequence or record");
    }
    // HeadersInit union conversion selects sequence before record. Do not special-case a Headers
    // instance: an author-provided @@iterator on it is observable and controls conversion.
    const iterator = init[Symbol.iterator];
    if (iterator !== undefined) {
      if (typeof iterator !== "function") throw new TypeError("Headers init iterator is not callable");
      const sequence = { [Symbol.iterator]() { return Reflect.apply(iterator, init, []); } };
      for (const member of sequence) {
        if (member == null || typeof member[Symbol.iterator] !== "function") {
          throw new TypeError("Headers: init member is not a sequence");
        }
        const pair = [...member];
        if (pair.length !== 2) throw new TypeError("Headers: init pair needs two items");
        this.append(pair[0], pair[1]);
      }
    } else {
      for (const k of Object.keys(init)) this.append(k, init[k]);
    }
  }
  _canModify(name, value) {
    if (this._guard === "immutable") throw new TypeError("Headers are immutable");
    if (this._guard === "request" && isForbiddenRequestHeader(name, value)) return false;
    if (this._guard === "response" && (name === "set-cookie" || name === "set-cookie2")) return false;
    return true;
  }
  append(name, value) {
    const key = normalizeHeaderName(name);
    value = normalizeHeaderValue(value);
    if (!this._canModify(key, value)) return;
    const existing = this._map.get(key);
    if (existing) existing.values.push(value);
    else this._map.set(key, { name: key, values: [value] });
  }
  delete(name) {
    const key = normalizeHeaderName(name);
    if (!this._canModify(key, "")) return;
    this._map.delete(key);
  }
  get(name) {
    const hit = this._map.get(normalizeHeaderName(name));
    return hit ? hit.values.join(", ") : null;
  }
  getSetCookie() {
    const hit = this._map.get("set-cookie");
    return hit ? hit.values.slice() : [];
  }
  has(name) {
    return this._map.has(normalizeHeaderName(name));
  }
  set(name, value) {
    const key = normalizeHeaderName(name);
    value = normalizeHeaderValue(value);
    if (!this._canModify(key, value)) return;
    this._map.set(key, { name: key, values: [value] });
  }
  forEach(fn, thisArg) {
    for (const [k, v] of this) fn.call(thisArg, v, k, this);
  }
  entries() {
    return createHeaderIterator(this, "entry");
  }
  keys() {
    return createHeaderIterator(this, "key");
  }
  values() {
    return createHeaderIterator(this, "value");
  }
  [Symbol.iterator]() {
    return this.entries();
  }
  _pairs() {
    return [...this].map(([k, v]) => [k, v]);
  }
}

function createHeaders(init, guard) {
  const headers = new Headers();
  headers._guard = guard === "immutable" ? "none" : guard;
  if (init !== undefined) headers._fill(init);
  headers._guard = guard;
  return headers;
}

const forbiddenRequestHeaderNames = new Set([
  "accept-charset", "accept-encoding", "access-control-request-headers",
  "access-control-request-method", "connection", "content-length", "cookie", "cookie2",
  "date", "dnt", "expect", "host", "keep-alive", "origin", "referer", "set-cookie",
  "te", "trailer", "transfer-encoding", "upgrade", "via",
]);

function isForbiddenRequestHeader(name, value) {
  if (forbiddenRequestHeaderNames.has(name) || name.startsWith("proxy-") || name.startsWith("sec-")) {
    return true;
  }
  if (name === "x-http-method" || name === "x-http-method-override" || name === "x-method-override") {
    return value.split(",").some((part) => /^(CONNECT|TRACE|TRACK)$/i.test(part.trim().replace(/^"|"$/g, "")));
  }
  return false;
}

function normalizeMethod(method) {
  method = toByteString(method, "request method");
  if (method === "" || /[^!#$%&'*+\-.^_`|~0-9A-Za-z]/.test(method)) {
    throw new TypeError("invalid request method");
  }
  if (/^(CONNECT|TRACE|TRACK)$/i.test(method)) throw new TypeError("forbidden request method");
  if (/^(DELETE|GET|HEAD|OPTIONS|POST|PUT)$/i.test(method)) return method.toUpperCase();
  return method;
}

function normalizeCredentials(credentials) {
  credentials = String(credentials);
  if (credentials !== "omit" && credentials !== "same-origin" && credentials !== "include") {
    throw new TypeError(`invalid Request credentials mode '${credentials}'`);
  }
  return credentials;
}

const kConsumed = Symbol("bodyConsumed");
const kBodyStream = Symbol("bodyStream");
// A user-supplied ReadableStream body, kept un-drained until the body is actually consumed or
// sent. Draining at construction would break feature-detection code (e.g. ky) that builds — but
// never sends — a `new Request(url, { body: new ReadableStream() })` just to probe support.
const kSourceStream = Symbol("bodySourceStream");

// Set `owner`'s body from a BodyInit, and (per spec) a default Content-Type when the body implies
// one and none is already set. A ReadableStream is stored, not drained (see kSourceStream).
function initBody(owner, body) {
  let contentType;
  if (body instanceof ReadableStream) {
    owner[kSourceStream] = body;
    owner._bodyBytes = undefined;
  } else if (body instanceof Blob) {
    owner._bodyBytes = body[kBlobBytes].slice();
    if (body.type) contentType = body.type;
  } else if (body instanceof FormData) {
    const encoded = encodeFormData(body);
    owner._bodyBytes = encoded.bytes;
    contentType = encoded.contentType;
  } else if (typeof body === "string") {
    owner._bodyBytes = new TextEncoder().encode(body);
    contentType = "text/plain;charset=UTF-8";
  } else if (body instanceof URLSearchParams) {
    owner._bodyBytes = new TextEncoder().encode(body.toString());
    contentType = "application/x-www-form-urlencoded;charset=UTF-8";
  } else {
    owner._bodyBytes = toBodyBytes(body);
  }
  if (contentType && owner.headers && !owner.headers.has("content-type")) {
    owner.headers.set("content-type", contentType);
  }
}

function bodyMixin(proto) {
  proto.text = async function () {
    return new TextDecoder().decode(await this._consume());
  };
  proto.json = async function () {
    return JSON.parse(await this.text());
  };
  proto.arrayBuffer = async function () {
    const bytes = await this._consume();
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  };
  proto.bytes = async function () {
    return this._consume();
  };
  proto.blob = async function () {
    const bytes = await this._consume();
    const type = (this.headers && this.headers.get("content-type")) || "";
    return new Blob([bytes], { type });
  };
  proto.formData = async function () {
    const ct = (this.headers && this.headers.get("content-type")) || "";
    if (ct.startsWith("application/x-www-form-urlencoded")) {
      // Fetch's Body.formData branch parses the consumed bytes with the URL-encoded parser. It
      // does not pass through Body.text's BOM-removing UTF-8 decoder.
      const input = new TextDecoder("utf-8", { ignoreBOM: true }).decode(await this._consume());
      const form = new FormData();
      for (const [k, v] of new URLSearchParams(input)) form.append(k, v);
      return form;
    }
    const m = /boundary=([^;]+)/i.exec(ct);
    if (ct.startsWith("multipart/form-data") && m) {
      return decodeMultipart(await this._consume(), m[1].trim().replace(/^"|"$/g, ""));
    }
    throw new TypeError(`formData(): unsupported content-type '${ct}'`);
  };
  proto._consume = async function () {
    const stream = this[kSourceStream] ?? this[kBodyStream];
    if (this.bodyUsed || (stream && stream.locked)) throw new TypeError("body already consumed");
    // Fetch §5.3 consumes a null body as an empty byte sequence without disturbing a stream;
    // bodyUsed therefore remains false and repeated null-body conversions are allowed.
    if (this._bodyBytes === undefined && this[kSourceStream] === undefined) {
      return new Uint8Array(0);
    }
    this[kConsumed] = true;
    this._materialize();
    // If the lazily-created stream was not the source just drained, discard its queued copy.
    if (this[kBodyStream] !== undefined) this[kBodyStream].cancel().catch(() => {});
    return this._bodyBytes || new Uint8Array(0);
  };
  // Drain a deferred ReadableStream body into bytes, on first consume/send.
  proto._materialize = function () {
    if (this[kSourceStream] !== undefined) {
      this._bodyBytes = drainStreamSync(this[kSourceStream]);
      this[kSourceStream] = undefined;
    }
  };
  Object.defineProperty(proto, "bodyUsed", {
    get() {
      const stream = this[kSourceStream] ?? this[kBodyStream];
      return !!this[kConsumed] || !!(stream && stream._disturbed);
    },
  });
  // `.body` is the user's ReadableStream if one was given (un-drained), else a stream over the
  // buffered bytes, or `null` when there is no body. The same stream instance is handed out on
  // repeated access (per spec); reading it consumes the body.
  Object.defineProperty(proto, "body", {
    configurable: true,
    get() {
      if (this[kSourceStream] !== undefined) return this[kSourceStream];
      if (this._bodyBytes === undefined) return null;
      if (this[kBodyStream] === undefined) this[kBodyStream] = makeBodyStream(this);
      return this[kBodyStream];
    },
  });
}

function toBodyBytes(body) {
  if (body === undefined || body === null) return undefined;
  if (typeof body === "string") return new TextEncoder().encode(body);
  if (body instanceof Uint8Array) return body;
  if (body instanceof ArrayBuffer) return new Uint8Array(body);
  if (ArrayBuffer.isView(body)) return new Uint8Array(body.buffer, body.byteOffset, body.byteLength);
  if (body instanceof ReadableStream) return drainStreamSync(body);
  if (body instanceof URLSearchParams) return new TextEncoder().encode(body.toString());
  return new TextEncoder().encode(String(body));
}

// Read a whole ReadableStream into one Uint8Array *synchronously*. Bodies are buffered, so a stream
// used as a request/response body must produce its data without awaiting (our own body streams do;
// so does any source whose `start`/`pull` enqueues synchronously).
function drainStreamSync(stream) {
  if (stream.locked) throw new TypeError("cannot construct a body from a locked ReadableStream");
  const reader = stream.getReader();
  const parts = [];
  let total = 0;
  for (;;) {
    const r = reader._readSync();
    if (r.pending) {
      throw new TypeError("a body ReadableStream must produce its data synchronously in this runtime");
    }
    if (r.done) break;
    let chunk = r.value;
    if (typeof chunk === "string") chunk = new TextEncoder().encode(chunk);
    else if (chunk instanceof ArrayBuffer) chunk = new Uint8Array(chunk);
    else if (ArrayBuffer.isView(chunk)) chunk = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
    else if (!(chunk instanceof Uint8Array)) throw new TypeError("ReadableStream chunk is not binary data");
    parts.push(chunk);
    total += chunk.length;
  }
  reader.releaseLock();
  if (parts.length === 1) return parts[0];
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// The `.body` ReadableStream for a Request/Response: it lazily hands out the buffered bytes as a
// single chunk (marking the body consumed, shared with `text()`/`json()`), or is empty when there
// is no body.
function makeBodyStream(owner) {
  return new ReadableStream({
    start(controller) {
      if (owner[kConsumed]) {
        controller.error(new TypeError("body already consumed"));
        return;
      }
      const bytes = owner._bodyBytes;
      if (bytes && bytes.length) controller.enqueue(bytes);
      controller.close();
    },
  });
}

class Request {
  constructor(input, init = {}) {
    init = init && typeof init === "object" ? init : {};
    if (input instanceof Request) {
      this.url = input.url;
      this.method = init.method !== undefined ? normalizeMethod(init.method) : input.method;
      this.credentials = init.credentials !== undefined
        ? normalizeCredentials(init.credentials)
        : input.credentials;
      this.headers = createHeaders("headers" in init ? init.headers : input.headers, "request");
      if ("body" in init) {
        initBody(this, init.body);
      } else {
        // Inherit the source request's body (its deferred stream, if any).
        this._bodyBytes = input._bodyBytes;
        this[kSourceStream] = input[kSourceStream];
      }
      this.signal = init.signal || input.signal || null;
    } else {
      const base = typeof globalThis.location === "object" &&
        globalThis.location !== null && typeof globalThis.location.href === "string"
        ? globalThis.location.href
        : undefined;
      const parsedURL = new URL(String(input), base);
      if (parsedURL.username !== "" || parsedURL.password !== "") {
        throw new TypeError("Request URL cannot include credentials");
      }
      this.url = parsedURL.href;
      this.method = init.method !== undefined ? normalizeMethod(init.method) : "GET";
      this.credentials = init.credentials !== undefined
        ? normalizeCredentials(init.credentials)
        : "same-origin";
      this.headers = createHeaders(init.headers, "request");
      initBody(this, init.body);
      this.signal = init.signal || null;
    }
    if (
      (this.method === "GET" || this.method === "HEAD") &&
      (this._bodyBytes !== undefined || this[kSourceStream] !== undefined)
    ) {
      throw new TypeError(`${this.method} request cannot have a body`);
    }
    this[kConsumed] = false;
  }
  clone() {
    this._materialize(); // can't tee a deferred stream; buffer it, then both share the bytes
    return new Request(this.url, {
      method: this.method,
      headers: this.headers,
      body: this._bodyBytes,
      signal: this.signal,
      credentials: this.credentials,
    });
  }
}
bodyMixin(Request.prototype);

// Server adapters create a Fetch Request from an already-validated HTTP message. This internal
// path is needed because the public constructor correctly forbids GET/HEAD bodies and the three
// forbidden author-controlled methods, while an HTTP server can receive either on the wire.
function createIncomingRequest(method, url, headers, body) {
  const request = new Request(url);
  request.method = method;
  request.headers = createHeaders(headers, "immutable");
  if (body !== undefined) request._bodyBytes = toBodyBytes(body);
  return request;
}

class Response {
  constructor(body = null, init = {}) {
    init = init && typeof init === "object" ? init : {};
    // A dictionary member set to `undefined` counts as absent (WebIDL), so `{ status: undefined }`
    // takes the default 200 rather than coercing to `Number(undefined)` → NaN.
    this.status = init.status !== undefined ? toUnsignedShort(init.status) : 200;
    if (this.status < 200 || this.status > 599) {
      throw new RangeError(`invalid response status ${this.status}`);
    }
    this.statusText = init.statusText !== undefined ? toByteString(init.statusText, "statusText") : "";
    if (this.statusText !== "" && /[^\t\x20-\x7e\x80-\xff]/.test(this.statusText)) {
      throw new TypeError("invalid response statusText");
    }
    this.headers = createHeaders(init.headers, "response");
    this.url = "";
    this.redirected = false;
    if (body !== null && [101, 103, 204, 205, 304].includes(this.status)) {
      throw new TypeError(`response status ${this.status} cannot have a body`);
    }
    initBody(this, body);
    this[kConsumed] = false;
  }
  get ok() {
    return this.status >= 200 && this.status < 300;
  }
  clone() {
    if (this[kConsumed]) throw new TypeError("cannot clone a used Response");
    this._materialize();
    const r = new Response(this._bodyBytes, {
      status: this.status,
      statusText: this.statusText,
      headers: this.headers,
    });
    r.url = this.url;
    r.redirected = this.redirected;
    r.headers._guard = this.headers._guard;
    return r;
  }
  static json(data, init) {
    const r = new Response(JSON.stringify(data), init);
    if (!r.headers.has("content-type")) r.headers.set("content-type", "application/json");
    return r;
  }
  static error() {
    const r = new Response(null, { status: 200 });
    r.status = 0;
    r.type = "error";
    return r;
  }
}
bodyMixin(Response.prototype);

function toUnsignedShort(value) {
  let number = Number(value);
  if (!Number.isFinite(number) || number === 0) return 0;
  number = Math.trunc(number);
  return ((number % 65536) + 65536) % 65536;
}

function percentDecodeBytes(input) {
  const bytes = new TextEncoder().encode(input);
  const output = new Uint8Array(bytes.length);
  let length = 0;
  for (let i = 0; i < bytes.length; ++i) {
    if (bytes[i] === 0x25 && i + 2 < bytes.length && isHex(bytes[i + 1]) && isHex(bytes[i + 2])) {
      output[length++] = parseInt(String.fromCharCode(bytes[i + 1], bytes[i + 2]), 16);
      i += 2;
    } else {
      output[length++] = bytes[i];
    }
  }
  return output.subarray(0, length);
}

// Fetch §6 data: URL processor. This is a scheme fetch, not an HTTP request: percent-decoding
// happens at the byte level and forgiving base64 decoding is applied only to the terminal marker.
function fetchDataURL(request) {
  const serialized = request.url.split("#", 1)[0].slice(5);
  const comma = serialized.indexOf(",");
  if (comma < 0) throw new TypeError("invalid data URL");
  let mimeType = serialized.slice(0, comma).replace(/^ +| +$/g, "");
  let body = percentDecodeBytes(serialized.slice(comma + 1));
  if (/; *base64$/i.test(mimeType)) {
    let encoded = "";
    for (const byte of body) encoded += String.fromCharCode(byte);
    let decoded;
    try {
      decoded = atob(encoded);
    } catch (_) {
      throw new TypeError("invalid base64 data URL");
    }
    body = new Uint8Array(decoded.length);
    for (let i = 0; i < decoded.length; ++i) body[i] = decoded.charCodeAt(i);
    mimeType = mimeType.replace(/; *base64$/i, "").replace(/ +$/g, "");
  }
  if (mimeType.startsWith(";")) mimeType = `text/plain${mimeType}`;
  if (!/^[^/;\s]+\/[^;\s]+(?:;.*)?$/.test(mimeType)) {
    mimeType = "text/plain;charset=US-ASCII";
  }
  const response = new Response(body, { headers: { "content-type": mimeType } });
  response.url = request.url;
  response.headers._guard = "immutable";
  return response;
}

function fetch(input, init = {}) {
  return new Promise((resolve, reject) => {
    let request;
    try {
      request = new Request(input, init);
    } catch (e) {
      reject(e);
      return;
    }
    const signal = request.signal;
    if (signal && signal.aborted) {
      reject(signal.reason || new DOMException("The operation was aborted", "AbortError"));
      return;
    }
    if (request.url.startsWith("data:")) {
      try {
        resolve(fetchDataURL(request));
      } catch (error) {
        reject(error);
      }
      return;
    }
    const headerPairs = request.headers._pairs();
    let bodyBytes;
    try {
      if (request.bodyUsed) throw new TypeError("body already consumed");
      if (request._bodyBytes !== undefined || request[kSourceStream] !== undefined) {
        request[kConsumed] = true;
      }
      request._materialize(); // drain a deferred stream body now that we're actually sending
      bodyBytes = request._bodyBytes;
    } catch (e) {
      reject(e);
      return;
    }
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      reject(signal.reason || new DOMException("The operation was aborted", "AbortError"));
    };
    if (signal) signal.addEventListener("abort", onAbort);

    __http.request(
      request.method,
      request.url,
      headerPairs,
      bodyBytes,
      request.credentials === "include",
      (raw) => {
        if (settled) return; // aborted first
        settled = true;
        if (signal) signal.removeEventListener("abort", onAbort);
        const nullBodyStatus = [101, 103, 204, 205, 304].includes(raw.status);
        const response = new Response(nullBodyStatus ? null : raw.body, {
          status: raw.status,
          statusText: raw.statusText,
          headers: raw.headers,
        });
        response.url = raw.url;
        response.redirected = raw.url !== request.url;
        response.headers._guard = "immutable";
        resolve(response);
      },
      (err) => {
        if (settled) return;
        settled = true;
        if (signal) signal.removeEventListener("abort", onAbort);
        reject(err instanceof Error ? err : new TypeError(String(err)));
      }
    );
  });
}

globalThis.Headers = Headers;
globalThis.Request = Request;
globalThis.Response = Response;
globalThis.fetch = fetch;
