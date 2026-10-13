# Web Streams runtime source

`streams.js` is the minified ES2015 polyfill bundle built from
`web-streams-polyfill` 4.3.0 at commit
`12b4fedec6c01503a137d837204ca0b85992461e`. Its readable TypeScript source is
maintained at <https://github.com/MattiasBuelens/web-streams-polyfill>. The
upstream project ports the WHATWG Streams Standard reference implementation and
runs the Web Platform Tests in Node.js, Chromium, and Firefox.

TRust JS carries one private integration hook for the buffered Fetch adapter. Add
this method to `ReadableStreamDefaultReader` in
`src/lib/readable-stream/default-reader.ts` before building:

```ts
/** @internal */
_readSync(): ReadableStreamDefaultReadResult<R> | { pending: true } {
  if (!IsReadableStreamDefaultReader(this)) {
    throw defaultReaderBrandCheckException('_readSync');
  }
  if (this._ownerReadableStream === undefined) {
    throw readerLockException('read from');
  }
  if (!ReadableStreamDefaultReaderCanReadSync(this)) {
    return { pending: true };
  }

  let result: ReadableStreamDefaultReadResult<R> | undefined;
  ReadableStreamDefaultReaderRead(this, {
    _chunkSteps(chunk) {
      result = { value: chunk, done: false };
    },
    _closeSteps() {
      result = { value: undefined, done: true };
    },
    _errorSteps(e) {
      throw e;
    }
  });
  return result!;
}
```

Then run `npm ci` and `npm run build:bundle`, and copy `dist/polyfill.js` to
`streams.js`. The expected SHA-256 is
`4363c96ae4b777818a90d83e4118fbf441421d0eccbf34b2860b48fd641e62ce`.
The internal method only reads already-queued data; it never invokes a source,
waits for a promise, or changes any public Streams algorithm.
