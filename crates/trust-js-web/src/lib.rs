//! trust-js-web — the WinterTC "Minimum Common Web Platform API", incrementally.
//!
//! Pure-JS pieces ship as `js_init` glue (see `src/js/`); Rust backs parsing, crypto, and the
//! network. Conformance checklist against the WinterTC minimum common API:
//!
//! - [x] `console`, timers, `queueMicrotask` (trust-js-runtime/trust-js-timers)
//! - [x] `DOMException`, `Event`, `CustomEvent`, `EventTarget`, `AbortController`,
//!   `AbortSignal` (including dispatch flags/phases and `abort()`/`timeout()` statics)
//! - [x] `TextEncoder` / `TextDecoder` and their stream forms (WHATWG labels, state, BOM/fatal)
//! - [x] `atob` / `btoa`
//! - [x] `structuredClone` (graph identity, sparse arrays, views, and ArrayBuffer transfer)
//! - [x] `URL` / `URLSearchParams` (WHATWG parsing/serialization, UTS #46 IDNA, form encoding)
//! - [x] `performance.now()` (+`timeOrigin`), `navigator.userAgent`
//! - [x] `crypto.getRandomValues` / `crypto.randomUUID` (`/dev/urandom` via std::fs — no
//!   syscalls, no crates), `crypto.subtle.digest` (SHA-256 only)
//! - [x] `fetch` / `Headers` / `Request` / `Response` — HTTP and certificate-verified HTTPS
//!   through the dynamically loaded system OpenSSL backend
//! - [~] `TRustJS.serve` — an HTTP/1.1 *server* (not a WinterTC API; follows the cross-runtime
//!   `serve((request) => Response)` convention of Deno/Bun/Workers). v1 is single-accept,
//!   `Connection: close`, buffered bodies, http only — see `server.rs` for what's deferred.
//! - [x] Streams controllers, BYOB byte streams, tee, piping, writable streams, transform streams,
//!   backpressure, and Request/Response body integration
//! - [ ] Remaining surface and depth are tracked in `AUDIT_REMEDIATION.md`.

use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use trust_js_host::{ops, Ctx, Extension, OpState, SpawnHandle, TaskRegistry, Value};

mod http;
mod http_syntax;
mod server;
mod sha1;
mod sha256;
mod sse;
mod url;
mod websocket;

/// WebSocket protocol internals — a from-scratch RFC 6455 codec. `websocket::testing` exposes a
/// minimal echo server other crates' tests and benchmarks drive the client against.
pub use websocket::testing as ws_testing;

/// SSE transport internals — `sse::testing` exposes a canned event-stream server for tests.
pub use sse::testing as sse_testing;
// The decoder parses the whole binary format; the MVP interpreter doesn't consume every field yet
// (reserved value-type data, mutability flags, etc.), and a few opcode matches read cleaner as
// explicit lists than ranges.
#[allow(dead_code, clippy::manual_range_patterns)]
#[doc(hidden)]
pub mod wasm;
mod wasm_ops;

pub fn extension() -> Extension {
    Extension {
        name: "web",
        globals: &[],
        namespaces: &[
            (
                "__perf",
                ops!["now" (0) => op_perf_now, "timeOrigin" (0) => op_time_origin],
            ),
            (
                "__encoding",
                ops![
                    "encode" (1) => op_encode,
                    "decoderStart" (3) => op_decoder_start,
                    "decoderPush" (3) => op_decoder_push,
                    "decoderDrop" (1) => op_decoder_drop,
                ],
            ),
            (
                "__url",
                ops!["parse" (2) => op_url_parse, "mutate" (3) => op_url_mutate],
            ),
            ("__http", ops!["request" (7) => op_http_request]),
            (
                "__http_server",
                ops![
                    "listen" (3) => server::op_server_listen,
                    "respond" (8) => server::op_server_respond,
                    "close" (1) => server::op_server_close,
                    "version" (0) => server::op_server_version,
                ],
            ),
            (
                "__crypto",
                ops![
                    "fill" (1) => op_random_fill,
                    "uuid" (0) => op_uuid,
                    "sha256" (1) => op_sha256,
                ],
            ),
            (
                "__ws",
                ops![
                    "connect" (3) => websocket::op_ws_connect,
                    "send" (2) => websocket::op_ws_send,
                    "close" (3) => websocket::op_ws_close,
                    "upgrade" (5) => websocket::op_ws_upgrade,
                ],
            ),
            (
                "__sse",
                ops![
                    "connect" (3) => sse::op_sse_connect,
                    "close" (1) => sse::op_sse_close,
                ],
            ),
            (
                "__compress",
                ops![
                    "deflate" (1) => op_deflate,
                    "inflate" (1) => op_inflate,
                    "deflateRaw" (1) => op_deflate_raw,
                    "inflateRaw" (1) => op_inflate_raw,
                    "gzip" (1) => op_gzip,
                    "gunzip" (1) => op_gunzip,
                    "brotli" (1) => op_brotli,
                    "unbrotli" (1) => op_unbrotli,
                    "decompressStart" (1) => op_decompress_start,
                    "decompressPush" (3) => op_decompress_push,
                    "decompressDrop" (1) => op_decompress_drop,
                    "compressStart" (1) => op_compress_start,
                    "compressPush" (3) => op_compress_push,
                ],
            ),
            (
                "__wasm",
                ops![
                    "validate" (1) => wasm_ops::op_validate,
                    "compile" (1) => wasm_ops::op_compile,
                    "retain" (2) => wasm_ops::op_retain,
                    "release" (1) => wasm_ops::op_release,
                    "moduleExports" (1) => wasm_ops::op_module_exports,
                    "moduleImports" (1) => wasm_ops::op_module_imports,
                    "moduleImportTypes" (1) => wasm_ops::op_module_import_types,
                    "allocMemory" (2) => wasm_ops::op_alloc_memory,
                    "allocTable" (2) => wasm_ops::op_alloc_table,
                    "allocGlobal" (3) => wasm_ops::op_alloc_global,
                    "instantiate" (2) => wasm_ops::op_instantiate,
                    "call" (2) => wasm_ops::op_call,
                    "memBuffer" (1) => wasm_ops::op_mem_buffer,
                    "memGrow" (2) => wasm_ops::op_mem_grow,
                    "tableGet" (2) => wasm_ops::op_table_get,
                    "tableSet" (3) => wasm_ops::op_table_set,
                    "tableSize" (1) => wasm_ops::op_table_size,
                    "globalGet" (1) => wasm_ops::op_global_get,
                    "globalInfo" (1) => wasm_ops::op_global_info,
                    "globalSet" (2) => wasm_ops::op_global_set,
                ],
            ),
        ],
        state_init: Some(|state: &mut OpState| {
            state.put(WebState::default());
            state.put(server::ServerRegistry::default());
            state.put(websocket::WsRegistry::default());
            state.put(sse::SseRegistry::default());
            state.put_external_memory(wasm_ops::WasmStore::default());
            state.put(CodecRegistry::default());
            state.put(TextDecoderRegistry::default());
        }),
        js_init: Some(JS_GLUE),
        js_init_snapshot: Some(JS_GLUE_SNAPSHOT),
    }
}

/// One IIFE (preamble captures and deletes the raw `__*` namespaces, the rest defines the
/// standard classes over them), assembled by `build.rs` from `src/js/*.js` — the single source
/// of truth. `JS_GLUE` is the fallback source; `JS_GLUE_SNAPSHOT` is its precompiled AST, decoded
/// at boot to skip re-parsing (see `trust_js_host::install` / `Engine::eval_snapshot`).
const JS_GLUE: &str = include_str!(concat!(env!("OUT_DIR"), "/web_glue.js"));
const JS_GLUE_SNAPSHOT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/web_glue.snap"));

#[derive(Default)]
struct WebState {
    /// `performance.now()`'s monotonic zero point and the wall-clock time (`timeOrigin`, Unix ms)
    /// captured at the same instant — set together on first access.
    start: Option<Instant>,
    time_origin_ms: f64,
    /// Cached `/dev/urandom` handle (macOS/Linux; the only randomness std can reach without
    /// syscalls or crates).
    urandom: Option<RefCell<File>>,
    /// Fetch's agent-local, origin/credentials-partitioned persistent connection pool.
    http: http::HttpClient,
}

// Clocks, entropy, HTTP connections and codec state: no JavaScript values.
trust_js_host::value_free!(WebState, CodecRegistry, TextDecoderRegistry);

#[derive(Default)]
struct CodecRegistry {
    next_id: u64,
    decoders: std::collections::HashMap<u64, CodecDecoder>,
    encoders: std::collections::HashMap<u64, CodecEncoder>,
}

enum CodecDecoder {
    Deflate(trust_js_host::deflate::DeflateDecoder),
    Brotli(trust_js_host::brotli::BrotliDecoder),
}

impl CodecDecoder {
    fn push(&mut self, input: &[u8], finish: bool) -> Result<(Vec<u8>, bool, bool), String> {
        match self {
            Self::Deflate(decoder) => decoder
                .push(input, finish)
                .map(|step| (step.output, step.done, step.needs_input)),
            Self::Brotli(decoder) => decoder
                .push(input, finish)
                .map(|step| (step.output, step.done, step.needs_input)),
        }
    }
}

enum CodecEncoder {
    Deflate(trust_js_host::deflate::DeflateEncoder),
    Brotli(trust_js_host::brotli::BrotliEncoder),
}

#[derive(Default)]
struct TextDecoderRegistry {
    next_id: u32,
    decoders: std::collections::HashMap<u32, TextDecoderState>,
}

struct TextDecoderState {
    encoding: &'static encoding_rs::Encoding,
    decoder: encoding_rs::Decoder,
    fatal: bool,
    ignore_bom: bool,
    do_not_flush: bool,
}

impl TextDecoderState {
    fn new(encoding: &'static encoding_rs::Encoding, fatal: bool, ignore_bom: bool) -> Self {
        let decoder = if ignore_bom {
            encoding.new_decoder_without_bom_handling()
        } else {
            encoding.new_decoder_with_bom_removal()
        };
        Self {
            encoding,
            decoder,
            fatal,
            ignore_bom,
            do_not_flush: false,
        }
    }

    fn reset(&mut self) {
        self.decoder = if self.ignore_bom {
            self.encoding.new_decoder_without_bom_handling()
        } else {
            self.encoding.new_decoder_with_bom_removal()
        };
    }

    fn decode(&mut self, input: &[u8], stream: bool) -> Result<String, String> {
        // Encoding Standard §7.2: a call after a non-streaming decode starts a fresh decoder;
        // streaming calls preserve the codec and BOM state across BufferSource boundaries.
        if !self.do_not_flush {
            self.reset();
        }
        self.do_not_flush = stream;
        let last = !stream;
        let capacity = if self.fatal {
            self.decoder
                .max_utf8_buffer_length_without_replacement(input.len())
        } else {
            self.decoder.max_utf8_buffer_length(input.len())
        }
        .ok_or_else(|| "TextDecoder output length overflow".to_string())?;
        if capacity > trust_js_host::MAX_DECOMPRESSED_BYTES {
            return Err("TextDecoder output exceeds byte limit".into());
        }
        let mut output = String::new();
        output
            .try_reserve(capacity.max(4))
            .map_err(|_| "TextDecoder output allocation failed".to_string())?;

        if self.fatal {
            let (result, read) =
                self.decoder
                    .decode_to_string_without_replacement(input, &mut output, last);
            debug_assert!(read <= input.len());
            match result {
                encoding_rs::DecoderResult::InputEmpty => Ok(output),
                encoding_rs::DecoderResult::Malformed(_, _) => {
                    Err("TextDecoder encountered malformed input in fatal mode".into())
                }
                encoding_rs::DecoderResult::OutputFull => {
                    Err("TextDecoder internal output bound was insufficient".into())
                }
            }
        } else {
            let (result, read, _) = self.decoder.decode_to_string(input, &mut output, last);
            debug_assert!(read <= input.len());
            match result {
                encoding_rs::CoderResult::InputEmpty => Ok(output),
                encoding_rs::CoderResult::OutputFull => {
                    Err("TextDecoder internal output bound was insufficient".into())
                }
            }
        }
    }
}

impl CodecEncoder {
    fn push(&mut self, input: &[u8], finish: bool) -> Result<Vec<u8>, String> {
        match self {
            Self::Deflate(encoder) => encoder.push(input, finish),
            Self::Brotli(encoder) => encoder.push(input, finish),
        }
    }
}

impl WebState {
    /// The monotonic clock's zero point, initializing it (and the paired `timeOrigin`) on first use.
    fn clock_start(&mut self) -> Instant {
        if self.start.is_none() {
            self.start = Some(Instant::now());
            self.time_origin_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
        }
        self.start.unwrap()
    }
}

fn op_perf_now(ctx: &mut Ctx, _this: Value, _args: &[Value]) -> Result<Value, Value> {
    let state = ctx.host_mut::<WebState>().expect("web state installed");
    let start = state.clock_start();
    Ok(Value::Num(start.elapsed().as_secs_f64() * 1000.0))
}

/// `performance.timeOrigin`: Unix-epoch milliseconds at the monotonic clock's zero point.
fn op_time_origin(ctx: &mut Ctx, _this: Value, _args: &[Value]) -> Result<Value, Value> {
    let state = ctx.host_mut::<WebState>().expect("web state installed");
    state.clock_start();
    Ok(Value::Num(state.time_origin_ms))
}

// ---- encoding ----

fn op_encode(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    // WHATWG Encoding §7.4 declares the input as USVString: valid surrogate pairs become their
    // scalar value and every unpaired surrogate becomes U+FFFD before UTF-8 encoding.
    let s = ctx.coerce_usv_string(args.first().unwrap_or(&Value::Undefined))?;
    let bytes = s.as_bytes().to_vec();
    ctx.make_uint8array(&bytes)
}

fn op_decoder_start(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let label = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let fatal = matches!(args.get(1), Some(Value::Bool(true)));
    let ignore_bom = matches!(args.get(2), Some(Value::Bool(true)));
    let encoding =
        encoding_rs::Encoding::for_label_no_replacement(label.as_bytes()).ok_or_else(|| {
            ctx.make_error(
                "RangeError",
                format!("TextDecoder: unsupported encoding '{label}'"),
            )
        })?;
    let registry = ctx
        .host_mut::<TextDecoderRegistry>()
        .expect("web installs text decoder registry");
    let id = loop {
        registry.next_id = registry.next_id.wrapping_add(1).max(1);
        if !registry.decoders.contains_key(&registry.next_id) {
            break registry.next_id;
        }
    };
    registry
        .decoders
        .insert(id, TextDecoderState::new(encoding, fatal, ignore_bom));
    Ok(ctx.make_array(vec![
        Value::Num(f64::from(id)),
        Value::from_string(encoding.name().to_ascii_lowercase()),
    ]))
}

fn text_decoder_id(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_num_opt)
        .filter(|id| {
            id.is_finite() && *id >= 1.0 && *id <= f64::from(u32::MAX) && id.fract() == 0.0
        })
        .map(|id| id as u32)
}

fn op_decoder_push(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let id = text_decoder_id(args.first())
        .ok_or_else(|| ctx.make_error("TypeError", "invalid TextDecoder id"))?;
    let bytes = ctx
        .typed_array_bytes(args.get(1).unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "TextDecoder.decode expects a BufferSource"))?;
    if bytes.len() > trust_js_host::MAX_DECOMPRESSED_BYTES {
        return Err(ctx.make_error("RangeError", "TextDecoder input exceeds byte limit"));
    }
    let stream = matches!(args.get(2), Some(Value::Bool(true)));
    let mut decoder = ctx
        .host_mut::<TextDecoderRegistry>()
        .expect("web installs text decoder registry")
        .decoders
        .remove(&id)
        .ok_or_else(|| ctx.make_error("TypeError", "TextDecoder is closed"))?;
    let result = decoder.decode(&bytes, stream);
    ctx.host_mut::<TextDecoderRegistry>()
        .expect("web installs text decoder registry")
        .decoders
        .insert(id, decoder);
    result
        .map(|value| ctx.string_from_utf8(value))
        .map_err(|error| ctx.make_error("TypeError", error))
}

fn op_decoder_drop(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    if let Some(id) = text_decoder_id(args.first()) {
        ctx.host_mut::<TextDecoderRegistry>()
            .expect("web installs text decoder registry")
            .decoders
            .remove(&id);
    }
    Ok(Value::Undefined)
}

// ---- url ----

/// `(input, base?)` -> component object. Throws TypeError, as the URL constructor must.
fn op_url_parse(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let input = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let base = match args.get(1) {
        None | Some(Value::Undefined) => None,
        Some(v) => Some(ctx.coerce_string(v)?.to_string()),
    };
    let u = url::parse(&input, base.as_deref())
        .map_err(|e| ctx.make_error("TypeError", format!("URL: {e}")))?;
    url_to_value(ctx, u)
}

fn op_url_mutate(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let href = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let component = ctx
        .coerce_string(args.get(1).unwrap_or(&Value::Undefined))?
        .to_string();
    let value = ctx
        .coerce_string(args.get(2).unwrap_or(&Value::Undefined))?
        .to_string();
    let url = url::mutate(&href, &component, &value)
        .map_err(|error| ctx.make_error("TypeError", format!("URL: {error}")))?;
    url_to_value(ctx, url)
}

fn url_to_value(ctx: &mut Ctx, u: url::Url) -> Result<Value, Value> {
    let obj = Value::Obj(ctx.new_object());
    let port = u.port.map(|p| p.to_string()).unwrap_or_default();
    let href = u.href();
    let origin = u.origin();
    // The URL API getters return the empty string for an explicitly empty query/fragment even
    // though their delimiters remain present in `href` and on the internal URL record.
    let query = if u.query == "?" {
        String::new()
    } else {
        u.query
    };
    let fragment = if u.fragment == "#" {
        String::new()
    } else {
        u.fragment
    };
    for (k, v) in [
        ("scheme", u.scheme),
        ("username", u.username),
        ("password", u.password),
        ("host", u.host),
        ("port", port),
        ("path", u.path),
        ("query", query),
        ("fragment", fragment),
        ("href", href),
        ("origin", origin),
    ] {
        let _ = ctx.set_member(&obj, k, Value::from_string(v));
    }
    Ok(obj)
}

// ---- crypto ----

/// `n` cryptographically-random bytes (the WebSocket handshake key needs these, same source as
/// `crypto.getRandomValues`).
pub(crate) fn web_random_bytes(ctx: &mut Ctx, n: usize) -> Result<Vec<u8>, Value> {
    random_bytes(ctx, n)
}

fn random_bytes(ctx: &mut Ctx, n: usize) -> Result<Vec<u8>, Value> {
    let state = ctx.host_mut::<WebState>().expect("web state installed");
    if state.urandom.is_none() {
        match File::open("/dev/urandom") {
            Ok(f) => state.urandom = Some(RefCell::new(f)),
            Err(e) => {
                return Err(ctx.make_error("Error", format!("no randomness source: {e}")));
            }
        }
    }
    let mut buf = vec![0u8; n];
    let ok = {
        let state = ctx.host_mut::<WebState>().expect("just set");
        let f = state.urandom.as_ref().expect("just set");
        f.borrow_mut().read_exact(&mut buf).is_ok()
    };
    if !ok {
        return Err(ctx.make_error("Error", "randomness source read failed"));
    }
    Ok(buf)
}

/// Fill the given typed array in place (the glue enforces the 65536-byte quota + returns it).
fn op_random_fill(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let v = args.first().cloned().unwrap_or(Value::Undefined);
    let Some(existing) = ctx.typed_array_bytes(&v) else {
        return Err(ctx.make_error("TypeError", "getRandomValues expects a typed array"));
    };
    let bytes = random_bytes(ctx, existing.len())?;
    ctx.typed_array_set_bytes(&v, &bytes);
    Ok(Value::Undefined)
}

fn op_uuid(ctx: &mut Ctx, _this: Value, _args: &[Value]) -> Result<Value, Value> {
    let mut b = random_bytes(ctx, 16)?;
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant 10
    let h: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
    let s = h.join("");
    Ok(Value::from_string(format!(
        "{}-{}-{}-{}-{}",
        &s[0..8],
        &s[8..12],
        &s[12..16],
        &s[16..20],
        &s[20..32]
    )))
}

fn op_sha256(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let v = args.first().unwrap_or(&Value::Undefined);
    let Some(bytes) = ctx.typed_array_bytes(v) else {
        return Err(ctx.make_error("TypeError", "digest expects a BufferSource"));
    };
    let digest = sha256::sha256(&bytes);
    ctx.make_uint8array(&digest)
}

// ---- compression (DEFLATE/zlib/gzip, backing CompressionStream/DecompressionStream) ----

fn compress_op(ctx: &mut Ctx, args: &[Value], codec: fn(&[u8]) -> Vec<u8>) -> Result<Value, Value> {
    let v = args.first().unwrap_or(&Value::Undefined);
    let Some(bytes) = ctx.typed_array_bytes(v) else {
        return Err(ctx.make_error("TypeError", "compression expects a BufferSource"));
    };
    ctx.make_uint8array(&codec(&bytes))
}

fn decompress_op(
    ctx: &mut Ctx,
    args: &[Value],
    codec: fn(&[u8]) -> Result<Vec<u8>, String>,
) -> Result<Value, Value> {
    let v = args.first().unwrap_or(&Value::Undefined);
    let Some(bytes) = ctx.typed_array_bytes(v) else {
        return Err(ctx.make_error("TypeError", "decompression expects a BufferSource"));
    };
    match codec(&bytes) {
        Ok(out) => ctx.make_uint8array(&out),
        Err(e) => Err(ctx.make_error("TypeError", e)),
    }
}

fn op_deflate(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    compress_op(ctx, a, trust_js_host::deflate::zlib_compress)
}
fn op_inflate(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    decompress_op(ctx, a, trust_js_host::deflate::zlib_decompress)
}
fn op_deflate_raw(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    compress_op(ctx, a, trust_js_host::deflate::deflate)
}
fn op_inflate_raw(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    decompress_op(ctx, a, trust_js_host::deflate::inflate)
}
fn op_gzip(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    compress_op(ctx, a, trust_js_host::deflate::gzip_compress)
}
fn op_gunzip(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    decompress_op(ctx, a, trust_js_host::deflate::gzip_decompress)
}
fn op_brotli(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    compress_op(ctx, a, trust_js_host::brotli::brotli_compress)
}
fn op_unbrotli(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    decompress_op(ctx, a, trust_js_host::brotli::brotli_decompress)
}

fn op_decompress_start(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let format = ctx
        .coerce_string(a.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let decoder = match format.as_str() {
        "brotli" => CodecDecoder::Brotli(trust_js_host::brotli::BrotliDecoder::new(
            trust_js_host::MAX_DECOMPRESSED_BYTES,
        )),
        "deflate" => CodecDecoder::Deflate(trust_js_host::deflate::DeflateDecoder::new(
            trust_js_host::deflate::DeflateFormat::Zlib,
            trust_js_host::MAX_DECOMPRESSED_BYTES,
        )),
        "deflate-raw" => CodecDecoder::Deflate(trust_js_host::deflate::DeflateDecoder::new(
            trust_js_host::deflate::DeflateFormat::Raw,
            trust_js_host::MAX_DECOMPRESSED_BYTES,
        )),
        "gzip" => CodecDecoder::Deflate(trust_js_host::deflate::DeflateDecoder::new(
            trust_js_host::deflate::DeflateFormat::Gzip,
            trust_js_host::MAX_DECOMPRESSED_BYTES,
        )),
        _ => return Err(ctx.make_error("TypeError", "unsupported incremental compression format")),
    };
    let registry = ctx
        .host_mut::<CodecRegistry>()
        .expect("web installs decompression registry");
    registry.next_id = registry.next_id.wrapping_add(1).max(1);
    let id = registry.next_id;
    registry.decoders.insert(id, decoder);
    Ok(Value::Num(id as f64))
}

fn op_compress_start(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let format = ctx
        .coerce_string(a.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let encoder = match format.as_str() {
        "brotli" => CodecEncoder::Brotli(trust_js_host::brotli::BrotliEncoder::new()),
        "deflate" => CodecEncoder::Deflate(trust_js_host::deflate::DeflateEncoder::new(
            trust_js_host::deflate::DeflateFormat::Zlib,
        )),
        "deflate-raw" => CodecEncoder::Deflate(trust_js_host::deflate::DeflateEncoder::new(
            trust_js_host::deflate::DeflateFormat::Raw,
        )),
        "gzip" => CodecEncoder::Deflate(trust_js_host::deflate::DeflateEncoder::new(
            trust_js_host::deflate::DeflateFormat::Gzip,
        )),
        _ => return Err(ctx.make_error("TypeError", "unsupported incremental compression format")),
    };
    let registry = ctx
        .host_mut::<CodecRegistry>()
        .expect("web installs compression registry");
    registry.next_id = registry.next_id.wrapping_add(1).max(1);
    let id = registry.next_id;
    registry.encoders.insert(id, encoder);
    Ok(Value::Num(id as f64))
}

fn codec_id(value: Option<&Value>) -> Option<u64> {
    value
        .and_then(Value::as_num_opt)
        .filter(|id| id.is_finite() && *id > 0.0 && id.fract() == 0.0)
        .map(|id| id as u64)
}

fn op_decompress_push(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let id =
        codec_id(a.first()).ok_or_else(|| ctx.make_error("TypeError", "invalid decoder id"))?;
    let bytes = ctx
        .typed_array_bytes(a.get(1).unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "decompression expects a BufferSource"))?;
    let finish = matches!(a.get(2), Some(Value::Bool(true)));
    let mut decoder = ctx
        .host_mut::<CodecRegistry>()
        .expect("web installs decompression registry")
        .decoders
        .remove(&id)
        .ok_or_else(|| ctx.make_error("TypeError", "decoder is closed"))?;
    let (output_bytes, done, needs_input) = decoder
        .push(&bytes, finish)
        .map_err(|error| ctx.make_error("TypeError", error))?;
    if !done {
        ctx.host_mut::<CodecRegistry>()
            .expect("web installs decompression registry")
            .decoders
            .insert(id, decoder);
    }
    let output = ctx.make_uint8array(&output_bytes)?;
    Ok(ctx.make_array(vec![output, Value::Bool(done), Value::Bool(needs_input)]))
}

fn op_decompress_drop(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    if let Some(id) = codec_id(a.first()) {
        let registry = ctx
            .host_mut::<CodecRegistry>()
            .expect("web installs compression registry");
        registry.decoders.remove(&id);
        registry.encoders.remove(&id);
    }
    Ok(Value::Undefined)
}

fn op_compress_push(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let id =
        codec_id(a.first()).ok_or_else(|| ctx.make_error("TypeError", "invalid encoder id"))?;
    let bytes = ctx
        .typed_array_bytes(a.get(1).unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "compression expects a BufferSource"))?;
    let finish = matches!(a.get(2), Some(Value::Bool(true)));
    let mut encoder = ctx
        .host_mut::<CodecRegistry>()
        .expect("web installs compression registry")
        .encoders
        .remove(&id)
        .ok_or_else(|| ctx.make_error("TypeError", "encoder is closed"))?;
    let output = encoder
        .push(&bytes, finish)
        .map_err(|error| ctx.make_error("TypeError", error))?;
    if !finish {
        ctx.host_mut::<CodecRegistry>()
            .expect("web installs compression registry")
            .encoders
            .insert(id, encoder);
    }
    ctx.make_uint8array(&output)
}

// ---- fetch ----

/// `(method, url, headerPairs, bodyOrUndefined, credentials, resolve, reject)`: one HTTP request
/// on the threadpool, settled through the TaskRegistry like every async op.
fn op_http_request(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let method = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let target = ctx
        .coerce_string(args.get(1).unwrap_or(&Value::Undefined))?
        .to_string();
    let headers = read_header_pairs(ctx, args.get(2).unwrap_or(&Value::Undefined))?;
    let body = match args.get(3) {
        None | Some(Value::Undefined) | Some(Value::Null) => None,
        Some(v) => match ctx.typed_array_bytes(v) {
            Some(bytes) => Some(bytes),
            None => Some(ctx.coerce_string(v)?.as_bytes().to_vec()),
        },
    };
    let credentials = matches!(args.get(4), Some(Value::Bool(true)));
    let (resolve, reject) = match (args.get(5), args.get(6)) {
        (Some(res), Some(rej)) if res.is_callable() && rej.is_callable() => {
            (res.clone(), rej.clone())
        }
        _ => return Err(ctx.make_error("TypeError", "__http.request expects (resolve, reject)")),
    };
    let id = ctx
        .host_mut::<TaskRegistry>()
        .expect("runtime installs the registry")
        .register(resolve, Some(reject), decode_http);
    let spawn = ctx
        .op_state()
        .get::<SpawnHandle>()
        .expect("runtime installs the spawn handle")
        .clone();
    let client = ctx
        .host_mut::<WebState>()
        .expect("web installs state")
        .http
        .clone();
    spawn.spawn_blocking(id, move || {
        Box::new(client.request(method, target, headers, body, credentials))
    });
    Ok(Value::Undefined)
}

/// A JS `[[k, v], ...]` array into Rust pairs, via the curated member API.
pub(crate) fn read_header_pairs(ctx: &mut Ctx, v: &Value) -> Result<Vec<(String, String)>, Value> {
    let mut out = Vec::new();
    if v.as_obj().is_none() {
        return Ok(out);
    }
    let len = ctx
        .get_member(v, "length")
        .map_err(|_| ctx.make_error("TypeError", "__http.request: headers must be an array"))?;
    let Value::Num(len) = len else {
        return Ok(out);
    };
    if !len.is_finite() || len.fract() != 0.0 || !(0.0..=1024.0).contains(&len) {
        return Err(ctx.make_error(
            "TypeError",
            "HTTP header list length is invalid or too large",
        ));
    }
    for i in 0..(len as usize) {
        let pair = ctx
            .get_member(v, &i.to_string())
            .unwrap_or(Value::Undefined);
        let k = ctx.get_member(&pair, "0").unwrap_or(Value::Undefined);
        let val = ctx.get_member(&pair, "1").unwrap_or(Value::Undefined);
        let name = ctx.coerce_string(&k)?.to_string();
        let value = ctx.coerce_string(&val)?.to_string();
        http_syntax::validate_header(&name, &value)
            .map_err(|error| ctx.make_error("TypeError", error))?;
        out.push((name, value));
    }
    http_syntax::validate_headers(&out).map_err(|error| ctx.make_error("TypeError", error))?;
    Ok(out)
}

/// Build the raw-response object the JS glue wraps into a `Response`.
fn decode_http(ctx: &mut Ctx, payload: Box<dyn std::any::Any + Send>) -> Result<Vec<Value>, Value> {
    let result = *payload
        .downcast::<Result<http::HttpResponse, String>>()
        .expect("http payload");
    let response = match result {
        Ok(r) => r,
        Err(message) => return Err(ctx.make_error("TypeError", message)),
    };
    let obj = Value::Obj(ctx.new_object());
    let _ = ctx.set_member(&obj, "status", Value::Num(response.status as f64));
    let _ = ctx.set_member(&obj, "statusText", Value::from_string(response.status_text));
    let _ = ctx.set_member(&obj, "url", Value::from_string(response.url));
    let pairs: Vec<Value> = response
        .headers
        .into_iter()
        .map(|(k, v)| ctx.make_array(vec![Value::from_string(k), Value::from_string(v)]))
        .collect();
    let headers = ctx.make_array(pairs);
    let _ = ctx.set_member(&obj, "headers", headers);
    let body = ctx.make_uint8array(&response.body)?;
    let _ = ctx.set_member(&obj, "body", body);
    Ok(vec![obj])
}

#[cfg(test)]
mod api_tests {
    use trust_js_host::{install, Completion, Engine};

    use super::extension;
    use crate::wasm_ops::WasmStore;

    // (module
    //   (import "env" "afterGrow" (func $afterGrow))
    //   (memory (export "memory") 1 3)
    //   (func (export "load") (param i32) (result i32)
    //     local.get 0 i32.load8_u)
    //   (func (export "store") (param i32 i32)
    //     local.get 0 local.get 1 i32.store8)
    //   (func (export "growThenCall") (param i32) (result i32)
    //     local.get 0 memory.grow call $afterGrow))
    const SHARED_MEMORY_MODULE: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // header
        0x01, 0x13, 0x04, 0x60, 0x00, 0x00, 0x60, 0x01, 0x7f, 0x01, 0x7f, 0x60, 0x02, 0x7f, 0x7f,
        0x00, 0x60, 0x01, 0x7f, 0x01, 0x7f, // types
        0x02, 0x11, 0x01, 0x03, b'e', b'n', b'v', 0x09, b'a', b'f', b't', b'e', b'r', b'G', b'r',
        b'o', b'w', 0x00, 0x00, // import
        0x03, 0x04, 0x03, 0x01, 0x02, 0x03, // functions
        0x05, 0x04, 0x01, 0x01, 0x01, 0x03, // memory
        0x07, 0x28, 0x04, 0x06, b'm', b'e', b'm', b'o', b'r', b'y', 0x02, 0x00, 0x04, b'l', b'o',
        b'a', b'd', 0x00, 0x01, 0x05, b's', b't', b'o', b'r', b'e', 0x00, 0x02, 0x0c, b'g', b'r',
        b'o', b'w', b'T', b'h', b'e', b'n', b'C', b'a', b'l', b'l', 0x00, 0x03, // exports
        0x0a, 0x1c, 0x03, 0x07, 0x00, 0x20, 0x00, 0x2d, 0x00, 0x00, 0x0b, 0x09, 0x00, 0x20, 0x00,
        0x20, 0x01, 0x3a, 0x00, 0x00, 0x0b, 0x08, 0x00, 0x20, 0x00, 0x40, 0x00, 0x10, 0x00,
        0x0b, // code
    ];

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine
            .eval(source, false)
            .expect("WebAssembly API test parses")
        {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
        }
    }

    #[test]
    fn request_credentials_mode_defaults_validates_inherits_and_clones() {
        let mut engine = Engine::new();
        install(&mut engine, &[extension()]);
        assert_eq!(
            eval(
                &mut engine,
                r#"
                const original = new Request("http://example.test/", { credentials: "include" });
                const inherited = new Request(original);
                const overridden = new Request(original, { credentials: "omit" });
                let invalid;
                try { new Request("http://example.test/", { credentials: "invalid" }); }
                catch (error) { invalid = error.name; }
                [
                  new Request("http://example.test/").credentials,
                  original.credentials,
                  inherited.credentials,
                  overridden.credentials,
                  original.clone().credentials,
                  invalid
                ].join(",")
                "#,
            ),
            "same-origin,include,include,omit,include,TypeError"
        );
    }

    #[test]
    fn wasm_descriptors_use_enforce_range_and_required_members() {
        let mut engine = Engine::new();
        install(&mut engine, &[extension()]);
        assert_eq!(
            eval(
                &mut engine,
                r#"
                const errorName = fn => { try { fn(); return "none" } catch (error) { return error.name } };
                [
                  errorName(() => new WebAssembly.Memory({})),
                  errorName(() => new WebAssembly.Memory({ initial: -1 })),
                  errorName(() => new WebAssembly.Memory({ initial: 2, maximum: 1 })),
                  new WebAssembly.Memory({ initial: 1.9 }).buffer.byteLength,
                  errorName(() => new WebAssembly.Table({ initial: 1 })),
                  new WebAssembly.Table({ element: "anyfunc", initial: 1.9 }).length,
                  errorName(() => new WebAssembly.Global({})),
                  new WebAssembly.Global({ value: "i64" }).value === 0n
                ].join(",")
                "#,
            ),
            "TypeError,TypeError,RangeError,65536,TypeError,1,TypeError,true"
        );
    }

    #[test]
    fn wasm_memory_and_array_buffer_identify_one_data_block_and_refresh_on_grow() {
        let mut engine = Engine::new();
        install(&mut engine, &[extension()]);
        let bytes = SHARED_MEMORY_MODULE
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            r#"
            const module = new WebAssembly.Module(new Uint8Array([{bytes}]));
            let memory, oldBuffer, callbackObservation;
            const instance = new WebAssembly.Instance(module, {{ env: {{ afterGrow() {{
              callbackObservation = [
                oldBuffer.byteLength,
                memory.buffer === oldBuffer,
                memory.buffer.byteLength
              ].join(":");
            }} }} }});
            memory = instance.exports.memory;
            oldBuffer = memory.buffer;
            const bytesView = new Uint8Array(oldBuffer);
            bytesView[7] = 171;
            const jsToWasm = instance.exports.load(7);
            instance.exports.store(8, 205);
            const wasmToJs = bytesView[8];
            const stableIdentity = memory.buffer === oldBuffer;
            const previousPages = instance.exports.growThenCall(1);
            [jsToWasm, wasmToJs, stableIdentity, previousPages,
             oldBuffer.byteLength, callbackObservation, memory.buffer.byteLength].join(",")
            "#
        );
        assert_eq!(
            eval(&mut engine, &source),
            "171,205,true,1,0,0:false:131072,131072"
        );
    }

    #[test]
    fn wasm_store_entities_follow_live_js_handle_graph() {
        let mut engine = Engine::new();
        install(&mut engine, &[extension()]);
        let bytes = SHARED_MEMORY_MODULE
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            r#"
            globalThis.__wasmTestModule = new WebAssembly.Module(new Uint8Array([{bytes}]));
            globalThis.__wasmTestInstance = new WebAssembly.Instance(
              __wasmTestModule, {{ env: {{ afterGrow() {{}} }} }}
            );
            globalThis.__wasmTestMemory = __wasmTestInstance.exports.memory;
            globalThis.__wasmTestLoad = __wasmTestInstance.exports.load;
            globalThis.__wasmTestStore = __wasmTestInstance.exports.store;
            globalThis.__wasmTestGrow = __wasmTestInstance.exports.growThenCall;
            0
            "#
        );
        assert_eq!(eval(&mut engine, &source), "0");
        let global = engine.global_this();
        for name in [
            "__wasmTestModule",
            "__wasmTestInstance",
            "__wasmTestMemory",
            "__wasmTestStore",
            "__wasmTestGrow",
        ] {
            let wrapper = engine
                .ctx()
                .get_member(&global, name)
                .unwrap_or_else(|_| panic!("missing wrapper {name}"));
            let token = engine
                .ctx()
                .get_member(&wrapper, "_root")
                .unwrap_or_else(|_| panic!("missing root token for {name}"));
            crate::wasm_ops::op_release(engine.ctx(), trust_js_host::Value::Undefined, &[token])
                .unwrap_or_else(|_| panic!("could not release root for {name}"));
        }
        assert_eq!(eval(&mut engine, "__wasmTestLoad(0)"), "0");
        assert_eq!(
            engine
                .ctx()
                .host_mut::<WasmStore>()
                .expect("wasm store")
                .test_stats(),
            [0, 1, 1, 4, 0, 1, 0, 1, 1],
            "the exported function root retains its defining instance graph"
        );

        let load = engine
            .ctx()
            .get_member(&global, "__wasmTestLoad")
            .unwrap_or_else(|_| panic!("missing load wrapper"));
        let token = engine
            .ctx()
            .get_member(&load, "_root")
            .unwrap_or_else(|_| panic!("missing load root"));
        crate::wasm_ops::op_release(engine.ctx(), trust_js_host::Value::Undefined, &[token])
            .unwrap_or_else(|_| panic!("could not release final root"));
        assert_eq!(
            engine
                .ctx()
                .host_mut::<WasmStore>()
                .expect("wasm store")
                .test_stats(),
            [0; 9],
            "the final root releases modules, callbacks, buffers, and every store entity"
        );
    }

    #[test]
    fn wasm_weak_caches_and_finalizers_release_unreachable_store_graphs() {
        let mut engine = Engine::new();
        install(&mut engine, &[extension()]);
        let bytes = SHARED_MEMORY_MODULE
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            r#"
            (() => {{
              const module = new WebAssembly.Module(new Uint8Array([{bytes}]));
              const instance = new WebAssembly.Instance(module, {{ env: {{ afterGrow() {{}} }} }});
              new Uint8Array(instance.exports.memory.buffer)[0] = 9;
              instance.exports.store(1, 10);
            }})();
            0
            "#
        );
        assert_eq!(eval(&mut engine, &source), "0");
        assert_eq!(
            engine
                .ctx()
                .host_mut::<WasmStore>()
                .expect("wasm store")
                .test_stats(),
            [1, 6, 1, 4, 0, 1, 0, 1, 1]
        );

        for _ in 0..3 {
            engine.collect_garbage_at_idle();
            engine.run_microtasks();
        }
        assert_eq!(
            engine
                .ctx()
                .host_mut::<WasmStore>()
                .expect("wasm store")
                .test_stats(),
            [0; 9]
        );
    }

    #[test]
    fn exported_function_keeps_weak_native_import_callback_callable() {
        let mut engine = Engine::new();
        install(&mut engine, &[extension()]);
        let bytes = SHARED_MEMORY_MODULE
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            r#"
            globalThis.__wasmCallbackCount = 0;
            globalThis.__wasmKeptGrow = (() => {{
              const module = new WebAssembly.Module(new Uint8Array([{bytes}]));
              const instance = new WebAssembly.Instance(module, {{ env: {{ afterGrow() {{
                __wasmCallbackCount++;
              }} }} }});
              return instance.exports.growThenCall;
            }})();
            0
            "#
        );
        assert_eq!(eval(&mut engine, &source), "0");
        for _ in 0..3 {
            engine.collect_garbage_at_idle();
            engine.run_microtasks();
        }
        assert_eq!(
            eval(&mut engine, "__wasmKeptGrow(1); __wasmCallbackCount"),
            "1"
        );
        assert_eq!(
            engine
                .ctx()
                .host_mut::<WasmStore>()
                .expect("wasm store")
                .test_stats(),
            [1, 6, 1, 4, 0, 1, 0, 1, 1]
        );

        assert_eq!(eval(&mut engine, "__wasmKeptGrow = undefined; 0"), "0");
        for _ in 0..3 {
            engine.collect_garbage_at_idle();
            engine.run_microtasks();
        }
        assert_eq!(
            engine
                .ctx()
                .host_mut::<WasmStore>()
                .expect("wasm store")
                .test_stats(),
            [0; 9]
        );
    }
}
