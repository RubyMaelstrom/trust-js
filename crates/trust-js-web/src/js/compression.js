// CompressionStream / DecompressionStream over incremental, byte-bounded native contexts.
// Decoder output is split near 64 KiB so stream backpressure can be observed between chunks.
// WHATWG Compression Standard §§3-5.

const COMPRESS = {
  brotli: __compress.brotli,
  gzip: __compress.gzip,
  deflate: __compress.deflate,
  "deflate-raw": __compress.deflateRaw,
};
const DECOMPRESS = {
  brotli: __compress.unbrotli,
  gzip: __compress.gunzip,
  deflate: __compress.inflate,
  "deflate-raw": __compress.inflateRaw,
};
const EMPTY_CODEC_CHUNK = new Uint8Array(0);
const CODEC_FINALIZER = new FinalizationRegistry((id) => {
  __compress.decompressDrop(id);
});

function toU8(chunk) {
  if (chunk instanceof Uint8Array) return chunk;
  if (chunk instanceof ArrayBuffer) return new Uint8Array(chunk);
  if (ArrayBuffer.isView(chunk)) return new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  throw new TypeError("CompressionStream expects BufferSource chunks");
}

function makeDecompressionStream(format) {
  if (!DECOMPRESS[format]) {
    throw new TypeError(`Unsupported DecompressionStream format: '${format}'`);
  }
  const id = __compress.decompressStart(format);
  const lifetime = {};
  CODEC_FINALIZER.register(lifetime, id, lifetime);
  let closed = false;

  function cleanup(drop) {
    if (closed) return;
    closed = true;
    CODEC_FINALIZER.unregister(lifetime);
    if (drop) __compress.decompressDrop(id);
  }

  function process(chunk, finish, controller) {
    if (closed) {
      // A decoder can observe the end marker in transform(). The subsequent stream flush is
      // bookkeeping, not additional compressed input; any later non-empty write is trailing data.
      if (finish && chunk.byteLength === 0) return;
      throw new TypeError("compressed input supplied after the end of the stream");
    }
    let input = chunk;
    try {
      for (;;) {
        const [output, done, needsInput] = __compress.decompressPush(id, input, finish);
        if (output.byteLength) controller.enqueue(output);
        if (done) {
          cleanup(false);
          return;
        }
        if (needsInput) {
          if (finish) throw new TypeError("compressed input ended before the stream completed");
          return;
        }
        input = EMPTY_CODEC_CHUNK;
      }
    } catch (error) {
      cleanup(true);
      controller.error(error);
      throw error;
    }
  }

  return new TransformStream({
    transform(chunk, controller) {
      process(toU8(chunk), false, controller);
    },
    flush(controller) {
      process(EMPTY_CODEC_CHUNK, true, controller);
    },
  });
}

function makeCompressionStream(format) {
  if (!COMPRESS[format]) {
    throw new TypeError(`Unsupported CompressionStream format: '${format}'`);
  }
  const id = __compress.compressStart(format);
  const lifetime = {};
  CODEC_FINALIZER.register(lifetime, id, lifetime);
  let closed = false;

  function cleanup(drop) {
    if (closed) return;
    closed = true;
    CODEC_FINALIZER.unregister(lifetime);
    if (drop) __compress.decompressDrop(id);
  }

  function process(chunk, finish, controller) {
    try {
      const output = __compress.compressPush(id, chunk, finish);
      if (output.byteLength) controller.enqueue(output);
      if (finish) cleanup(false);
    } catch (error) {
      cleanup(true);
      controller.error(error);
      throw error;
    }
  }

  return new TransformStream({
    transform(chunk, controller) {
      process(toU8(chunk), false, controller);
    },
    flush(controller) {
      process(EMPTY_CODEC_CHUNK, true, controller);
    },
  });
}

class CompressionStream {
  constructor(format) {
    this._ts = makeCompressionStream(format);
  }
  get readable() {
    return this._ts.readable;
  }
  get writable() {
    return this._ts.writable;
  }
}

class DecompressionStream {
  constructor(format) {
    this._ts = makeDecompressionStream(format);
  }
  get readable() {
    return this._ts.readable;
  }
  get writable() {
    return this._ts.writable;
  }
}

globalThis.CompressionStream = CompressionStream;
globalThis.DecompressionStream = DecompressionStream;
