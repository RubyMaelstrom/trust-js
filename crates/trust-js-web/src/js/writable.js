// Encoding Standard transform streams. The core ReadableStream, WritableStream, TransformStream,
// queuing strategy, controller, BYOB, tee, and piping algorithms are installed by streams.js.

class TextEncoderStream {
  constructor() {
    const enc = new TextEncoder();
    let pendingHighSurrogate = "";
    this._ts = new TransformStream({
      transform(chunk, controller) {
        let input = pendingHighSurrogate + String(chunk);
        pendingHighSurrogate = "";
        if (input.length) {
          const last = input.charCodeAt(input.length - 1);
          if (last >= 0xd800 && last <= 0xdbff) {
            pendingHighSurrogate = input[input.length - 1];
            input = input.slice(0, -1);
          }
        }
        const bytes = enc.encode(input);
        if (bytes.length) controller.enqueue(bytes);
      },
      flush(controller) {
        if (pendingHighSurrogate) controller.enqueue(enc.encode(pendingHighSurrogate));
      },
    });
  }
  get encoding() {
    return "utf-8";
  }
  get readable() {
    return this._ts.readable;
  }
  get writable() {
    return this._ts.writable;
  }
}

class TextDecoderStream {
  constructor(label = "utf-8", options = {}) {
    const dec = new TextDecoder(label, options);
    this._encoding = dec.encoding;
    this._fatal = dec.fatal;
    this._ignoreBOM = dec.ignoreBOM;
    this._ts = new TransformStream({
      transform(chunk, controller) {
        const text = dec.decode(chunk, { stream: true });
        if (text) controller.enqueue(text);
      },
      flush(controller) {
        const text = dec.decode();
        if (text) controller.enqueue(text);
      },
    });
  }
  get encoding() {
    return this._encoding;
  }
  get fatal() {
    return this._fatal;
  }
  get ignoreBOM() {
    return this._ignoreBOM;
  }
  get readable() {
    return this._ts.readable;
  }
  get writable() {
    return this._ts.writable;
  }
}

Object.defineProperty(TextEncoderStream.prototype, Symbol.toStringTag, {
  value: "TextEncoderStream", configurable: true,
});
Object.defineProperty(TextDecoderStream.prototype, Symbol.toStringTag, {
  value: "TextDecoderStream", configurable: true,
});

globalThis.TextEncoderStream = TextEncoderStream;
globalThis.TextDecoderStream = TextDecoderStream;
