// The Streams Standard's `async iterable<any>` parameter is a Web IDL async-sequence type, so
// primitive ECMAScript iterables such as strings are rejected before ReadableStreamFromIterable.
// web-streams-polyfill implements the latter algorithm directly and would otherwise accept them.
const readableStreamFrom = globalThis.ReadableStream.from;
Object.defineProperty(globalThis.ReadableStream, "from", {
  configurable: true,
  enumerable: true,
  writable: true,
  value: function from(asyncIterable) {
    if ((typeof asyncIterable !== "object" || asyncIterable === null) &&
        typeof asyncIterable !== "function") {
      throw new TypeError("ReadableStream.from requires an object implementing async iterable");
    }
    return Reflect.apply(readableStreamFrom, this, [asyncIterable]);
  },
});
