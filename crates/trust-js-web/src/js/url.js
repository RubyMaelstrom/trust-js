// URL + URLSearchParams over the native WHATWG parser. Component setters use parser state
// overrides, and form encoding follows URL Standard §5 rather than encodeURIComponent's
// different escape set.

function toUSVString(value) {
  return encodingUSVString(value);
}

function requireArguments(actual, required, operation) {
  if (actual < required) {
    throw new TypeError(`${operation} requires at least ${required} argument${required === 1 ? "" : "s"}`);
  }
}

function isHex(byte) {
  return (byte >= 0x30 && byte <= 0x39) ||
    (byte >= 0x41 && byte <= 0x46) ||
    (byte >= 0x61 && byte <= 0x66);
}

function percentDecode(input) {
  const bytes = new TextEncoder().encode(input.replace(/\+/g, " "));
  const output = new Uint8Array(bytes.length);
  let length = 0;
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] === 0x25 && i + 2 < bytes.length && isHex(bytes[i + 1]) && isHex(bytes[i + 2])) {
      output[length++] = parseInt(String.fromCharCode(bytes[i + 1], bytes[i + 2]), 16);
      i += 2;
    } else {
      output[length++] = bytes[i];
    }
  }
  // URL Standard §5 uses Infra's UTF-8 decode without BOM handling: U+FEFF is data here,
  // unlike the default TextDecoder BOM-removal behavior.
  return new TextDecoder("utf-8", { ignoreBOM: true }).decode(output.subarray(0, length));
}

function formDecode(s) {
  const out = [];
  if (s.startsWith("?")) s = s.slice(1);
  if (s === "") return out;
  for (const part of s.split("&")) {
    if (part === "") continue;
    const eq = part.indexOf("=");
    const rawK = eq >= 0 ? part.slice(0, eq) : part;
    const rawV = eq >= 0 ? part.slice(eq + 1) : "";
    out.push([percentDecode(rawK), percentDecode(rawV)]);
  }
  return out;
}

function formEncode(list) {
  const enc = (value) => {
    const bytes = new TextEncoder().encode(toUSVString(value));
    let output = "";
    for (const byte of bytes) {
      if ((byte >= 0x41 && byte <= 0x5a) || (byte >= 0x61 && byte <= 0x7a) ||
          (byte >= 0x30 && byte <= 0x39) || byte === 0x2a || byte === 0x2d ||
          byte === 0x2e || byte === 0x5f) {
        output += String.fromCharCode(byte);
      } else if (byte === 0x20) {
        output += "+";
      } else {
        output += `%${byte.toString(16).toUpperCase().padStart(2, "0")}`;
      }
    }
    return output;
  };
  return list.map(([k, v]) => `${enc(k)}=${enc(v)}`).join("&");
}

class URLSearchParams {
  constructor(init = "") {
    this._list = [];
    this._onchange = null;
    if (typeof init === "string") {
      this._list = formDecode(toUSVString(init));
    } else if ((typeof init === "object" && init !== null) || typeof init === "function") {
      // Web IDL union conversion selects the sequence branch whenever @@iterator is present,
      // including for platform objects whose iterator has been overridden by script.
      const iterator = init[Symbol.iterator];
      if (iterator !== undefined) {
        if (typeof iterator !== "function") {
          throw new TypeError("URLSearchParams: init iterator is not callable");
        }
        const sequence = { [Symbol.iterator]() { return Reflect.apply(iterator, init, []); } };
        for (const pair of sequence) {
          if (pair == null || typeof pair[Symbol.iterator] !== "function") {
            throw new TypeError("URLSearchParams: each init pair needs exactly two items");
          }
          const values = [...pair];
          if (values.length !== 2) {
            throw new TypeError("URLSearchParams: each init pair needs exactly two items");
          }
          this._list.push([toUSVString(values[0]), toUSVString(values[1])]);
        }
      } else {
        // A record converts its keys to USVString before insertion. Distinct ECMAScript keys can
        // therefore collide (for example, different lone surrogates); the later value replaces
        // the earlier one without moving the record entry.
        const record = new Map();
        for (const key of Object.keys(init)) record.set(toUSVString(key), toUSVString(init[key]));
        this._list = [...record];
      }
    } else if (init !== undefined) {
      this._list = formDecode(toUSVString(init));
    }
  }
  _changed() {
    if (this._onchange) this._onchange();
  }
  _reset(search) {
    this._list = formDecode(search);
  }
  append(name, value) {
    requireArguments(arguments.length, 2, "URLSearchParams.append");
    this._list.push([toUSVString(name), toUSVString(value)]);
    this._changed();
  }
  delete(name, value = undefined) {
    requireArguments(arguments.length, 1, "URLSearchParams.delete");
    name = toUSVString(name);
    // In Web IDL an optional argument whose ECMAScript value is undefined is not given.
    if (value !== undefined) {
      value = toUSVString(value);
      this._list = this._list.filter(([k, v]) => k !== name || v !== value);
    } else {
      this._list = this._list.filter(([k]) => k !== name);
    }
    this._changed();
  }
  get(name) {
    requireArguments(arguments.length, 1, "URLSearchParams.get");
    name = toUSVString(name);
    const hit = this._list.find(([k]) => k === name);
    return hit ? hit[1] : null;
  }
  getAll(name) {
    requireArguments(arguments.length, 1, "URLSearchParams.getAll");
    name = toUSVString(name);
    return this._list.filter(([k]) => k === name).map(([, v]) => v);
  }
  has(name, value = undefined) {
    requireArguments(arguments.length, 1, "URLSearchParams.has");
    name = toUSVString(name);
    if (value !== undefined) {
      value = toUSVString(value);
      return this._list.some(([k, v]) => k === name && v === value);
    }
    return this._list.some(([k]) => k === name);
  }
  set(name, value) {
    requireArguments(arguments.length, 2, "URLSearchParams.set");
    name = toUSVString(name);
    value = toUSVString(value);
    const i = this._list.findIndex(([k]) => k === name);
    if (i >= 0) {
      this._list[i][1] = value;
      this._list = this._list.filter(([k], j) => k !== name || j <= i);
    } else {
      this._list.push([name, value]);
    }
    this._changed();
  }
  sort() {
    // Stable by key (Array.prototype.sort is stable in the engine).
    this._list.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    this._changed();
  }
  forEach(fn, thisArg) {
    requireArguments(arguments.length, 1, "URLSearchParams.forEach");
    for (let i = 0; i < this._list.length; i++) {
      const [k, v] = this._list[i];
      fn.call(thisArg, v, k, this);
    }
  }
  *entries() {
    for (let i = 0; i < this._list.length; i++) yield [...this._list[i]];
  }
  *keys() {
    for (const [k] of this._list) yield k;
  }
  *values() {
    for (const [, v] of this._list) yield v;
  }
  [Symbol.iterator]() {
    return this.entries();
  }
  get size() {
    return this._list.length;
  }
  toString() {
    return formEncode(this._list);
  }
}

class URL {
  constructor(input, base = undefined) {
    this._c = __url.parse(toUSVString(input), base === undefined ? undefined : toUSVString(base));
    this._searchParams = null;
  }
  _mutate(component, value) {
    this._c = __url.mutate(this._c.href, component, toUSVString(value));
    if (this._searchParams) this._searchParams._reset(this._c.query);
  }
  get href() {
    return this._c.href;
  }
  set href(v) {
    this._c = __url.parse(toUSVString(v), undefined);
    if (this._searchParams) this._searchParams._reset(this._c.query);
  }
  get origin() {
    return this._c.origin;
  }
  get protocol() {
    return `${this._c.scheme}:`;
  }
  set protocol(v) {
    this._mutate("protocol", v);
  }
  get username() {
    return this._c.username;
  }
  set username(v) {
    this._mutate("username", v);
  }
  get password() {
    return this._c.password;
  }
  set password(v) {
    this._mutate("password", v);
  }
  get host() {
    return this._c.port === "" ? this._c.host : `${this._c.host}:${this._c.port}`;
  }
  get hostname() {
    return this._c.host;
  }
  set host(v) {
    this._mutate("host", v);
  }
  set hostname(v) {
    this._mutate("hostname", v);
  }
  get port() {
    return this._c.port;
  }
  set port(v) {
    this._mutate("port", v);
  }
  get pathname() {
    return this._c.path;
  }
  set pathname(v) {
    this._mutate("pathname", v);
  }
  get search() {
    return this._c.query;
  }
  set search(v) {
    this._mutate("search", v);
  }
  get hash() {
    return this._c.fragment;
  }
  set hash(v) {
    this._mutate("hash", v);
  }
  get searchParams() {
    if (!this._searchParams) {
      const sp = new URLSearchParams(this._c.query);
      sp._onchange = () => {
        const q = sp.toString();
        // Keep this list live while the URL query is updated through its parser override.
        const saved = this._searchParams;
        this._searchParams = null;
        this._mutate("search", q === "" ? "" : `?${q}`);
        this._searchParams = saved;
      };
      this._searchParams = sp;
    }
    return this._searchParams;
  }
  toString() {
    return this.href;
  }
  toJSON() {
    return this.href;
  }
  static canParse(input, base = undefined) {
    requireArguments(arguments.length, 1, "URL.canParse");
    try {
      new URL(input, base);
      return true;
    } catch {
      return false;
    }
  }
  static parse(input, base = undefined) {
    requireArguments(arguments.length, 1, "URL.parse");
    try {
      return new URL(input, base);
    } catch {
      return null;
    }
  }
}

Object.defineProperty(URL.prototype, Symbol.toStringTag, { value: "URL", configurable: true });
Object.defineProperty(URLSearchParams.prototype, Symbol.toStringTag, {
  value: "URLSearchParams",
  configurable: true,
});

globalThis.URLSearchParams = URLSearchParams;
globalThis.URL = URL;
