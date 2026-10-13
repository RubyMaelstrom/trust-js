//! A blocking HTTP/1.1 *server* on `std::net::TcpListener`, the mirror of the fetch client in
//! `http.rs`. `TRustJS.serve(handler)` (see `js/server.js`) drives a WinterCG-style fetch
//! handler: each connection is parsed into a `Request`, the handler returns a `Response`, and
//! the bytes are written back — the same `(Request) -> Response` contract Deno.serve/Bun.serve
//! and Cloudflare Workers converge on (WinterTC's Minimum Common API standardizes
//! fetch/Request/Response but not a server, so this follows that cross-runtime convention).
//!
//! ## How it runs on the loop
//! The engine is single-threaded and `!Send`, so the JS handler must run on the loop thread.
//! We use the runtime's dedicated blocking executor + a `TaskRegistry` completion with a
//! **re-arm** pattern: one accept task runs on an isolated thread, blocks in
//! `accept()`, reads+parses one request, and comes back to the loop as a completion. The
//! completion decoder ([`decode_accept`]) hands the request to JS *and* arms the next accept,
//! so the listener keeps running for the life of the process (an always-registered task is
//! also what keeps the event loop from going idle). Responses are written back on the pool
//! (blocking `write`), settled through the registry like any other async op.
//!
//! ## What's intentionally missing (v1 — cold-start / low-concurrency focus)
//! - **Concurrency**: exactly one `accept()` is in flight at a time, and it holds one of the
//!   bounded dedicated-executor slot while blocked. Fine for cold-start latency and light load; a real
//!   readiness reactor (epoll/kqueue) or a dedicated listener thread is future work and would
//!   need raw syscalls (out of scope under the zero-dep policy) or a new host primitive.
//! - **Keep-alive**: every response is `Connection: close`; one request per connection.
//! - **Streaming**: request and response bodies are fully buffered (no chunked *response*
//!   output, no backpressure) — same limitation the fetch client / body streams have today.
//! - **No HTTP/2, no `Expect: 100-continue`, no trailers on responses, no `Date` header**
//!   (formatting an HTTP-date without a date library is deferred), **no TLS/https** (same
//!   STOP-AND-FLAG as the client: TLS can't be built on std alone).

use std::collections::HashMap;
use std::io::{BufReader, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use trust_js_host::{CompletionSender, Ctx, SpawnHandle, TaskRegistry, Value};

use crate::http_syntax::{
    self, body_framing, field_value_bytes, header_values, read_body, read_fields, read_line,
    MAX_BODY, MAX_HEADER_BYTES,
};
use crate::read_header_pairs;

/// A slow or idle client must not pin a pool worker forever: bound the header/body read.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Live servers, keyed by the id handed to JS. Lives in `OpState`; the accept decoder reads it
/// to re-arm and the `close` op flips the `closed` flag. JS values (`dispatch`) are `!Send` and
/// so only ever touched on the loop thread — never moved into a pool closure.
#[derive(Default)]
pub(crate) struct ServerRegistry {
    next: u64,
    servers: HashMap<u64, ServerEntry>,
}

struct ServerEntry {
    listener: Arc<TcpListener>,
    local_addr: SocketAddr,
    /// Flipped by `close()`; the accept loop checks it and stops re-arming.
    closed: Arc<AtomicBool>,
    /// The fixed JS `__dispatch(serverId, connId, ...)` callback (same for every accept of
    /// this server); it looks the user handler up by id.
    dispatch: Value,
}

impl Drop for ServerRegistry {
    fn drop(&mut self) {
        for entry in self.servers.values() {
            entry.closed.store(true, Ordering::Release);
            wake_listener(entry.local_addr);
        }
    }
}

/// What one accept task sends back to the loop (all fields `Send`).
struct AcceptTaskResult {
    server_id: u64,
    outcome: AcceptOutcome,
}

enum AcceptOutcome {
    /// A parsed request plus the still-open socket to answer on.
    Request(Accepted),
    /// The listener was shut down (or `accept()` failed): stop serving this server.
    Closed,
}

#[derive(Debug)]
struct RequestError {
    status: u16,
    reason: &'static str,
    message: String,
}

impl RequestError {
    fn new(status: u16, reason: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            reason,
            message: message.into(),
        }
    }
}

impl From<String> for RequestError {
    fn from(message: String) -> Self {
        Self::new(400, "Bad Request", message)
    }
}

impl From<&str> for RequestError {
    fn from(message: &str) -> Self {
        Self::from(message.to_string())
    }
}

struct Accepted {
    stream: TcpStream,
    peer: SocketAddr,
    method: String,
    /// Absolute URL (`http://<host><target>`) so the JS `Request` constructor accepts it.
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

// ---- native ops -------------------------------------------------------------------------------

/// `__http_server.listen(hostname, port, dispatch)` -> `[serverId, boundPort]`. Binds, registers
/// the server, and arms the first accept. Throws if the bind fails.
pub(crate) fn op_server_listen(
    ctx: &mut Ctx,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let host = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let port = match args.get(1) {
        Some(Value::Num(n)) if n.is_finite() && n.fract() == 0.0 && (0.0..=65535.0).contains(n) => {
            *n as u16
        }
        _ => return Err(ctx.make_error("TypeError", "listen: port must be 0..=65535")),
    };
    let dispatch = match args.get(2) {
        Some(v) if v.is_callable() => v.clone(),
        _ => return Err(ctx.make_error("TypeError", "listen: dispatch must be a function")),
    };

    let listener = TcpListener::bind((host.as_str(), port))
        .map_err(|e| ctx.make_error("Error", format!("listen {host}:{port}: {e}")))?;
    let local_addr = listener
        .local_addr()
        .map_err(|e| ctx.make_error("Error", format!("local_addr: {e}")))?;
    let listener = Arc::new(listener);
    let closed = Arc::new(AtomicBool::new(false));

    let id = {
        let reg = ctx
            .host_mut::<ServerRegistry>()
            .expect("web installs ServerRegistry");
        let id = reg.next;
        reg.next += 1;
        reg.servers.insert(
            id,
            ServerEntry {
                listener: Arc::clone(&listener),
                local_addr,
                closed: Arc::clone(&closed),
                dispatch: dispatch.clone(),
            },
        );
        id
    };

    arm_accept(ctx, id, listener, closed, local_addr, dispatch);

    Ok(ctx.make_array(vec![
        Value::Num(id as f64),
        Value::Num(local_addr.port() as f64),
    ]))
}

/// `__http_server.respond(connId, status, statusText, headers, body, requestMethod, resolve,
/// reject)`.
/// Serializes the response and writes it on the pool; the promise settles when the write
/// finishes (or the client hung up). The socket is taken out of the resource table here, so a
/// second respond on the same connection rejects.
pub(crate) fn op_server_respond(
    ctx: &mut Ctx,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let conn_id = match args.first() {
        Some(Value::Num(n)) if *n >= 0.0 => *n as u32,
        _ => return Err(ctx.make_error("TypeError", "respond: bad connection id")),
    };
    let status = match args.get(1) {
        Some(Value::Num(n)) if n.is_finite() && n.fract() == 0.0 && (200.0..=599.0).contains(n) => {
            *n as u16
        }
        _ => return Err(ctx.make_error("TypeError", "respond: status must be 200..=599")),
    };
    let status_text = ctx
        .coerce_string(args.get(2).unwrap_or(&Value::Undefined))?
        .to_string();
    let headers = read_header_pairs(ctx, args.get(3).unwrap_or(&Value::Undefined))?;
    let body = match args.get(4) {
        None | Some(Value::Undefined) | Some(Value::Null) => Vec::new(),
        Some(v) => ctx.typed_array_bytes(v).unwrap_or_default(),
    };
    if body.len() as u64 > MAX_BODY {
        return Err(ctx.make_error("RangeError", "respond: body exceeds 32 MiB limit"));
    }
    let request_method = ctx
        .coerce_string(args.get(5).unwrap_or(&Value::Undefined))?
        .to_string();
    let (resolve, reject) = match (args.get(6), args.get(7)) {
        (Some(res), Some(rej)) if res.is_callable() && rej.is_callable() => {
            (res.clone(), rej.clone())
        }
        _ => return Err(ctx.make_error("TypeError", "respond expects (resolve, reject)")),
    };

    // Take ownership of the socket (removing it from the table); `Rc::try_unwrap` succeeds
    // because the table held the only reference.
    let stream = ctx
        .resource_table()
        .close(conn_id)
        .and_then(|rc| rc.downcast::<TcpStream>().ok())
        .and_then(|rc| std::rc::Rc::try_unwrap(rc).ok());
    let Some(stream) = stream else {
        return Err(ctx.make_error(
            "TypeError",
            "respond: unknown or already-answered connection",
        ));
    };

    let bytes = build_response(status, &status_text, &headers, &body, &request_method)
        .map_err(|error| ctx.make_error("TypeError", format!("respond: {error}")))?;

    let id = ctx
        .host_mut::<TaskRegistry>()
        .expect("runtime installs the registry")
        .register(resolve, Some(reject), decode_write);
    let spawn = ctx
        .op_state()
        .get::<SpawnHandle>()
        .expect("runtime installs the spawn handle")
        .clone();
    spawn.spawn_blocking(id, move || Box::new(write_all_and_close(stream, &bytes)));

    Ok(Value::Undefined)
}

/// `__http_server.close(serverId)`. Flags the listener closed and pokes it with a throwaway
/// connection so the blocked `accept()` wakes, sees the flag, and stops re-arming.
pub(crate) fn op_server_close(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let id = match args.first() {
        Some(Value::Num(n)) if *n >= 0.0 => *n as u64,
        _ => return Ok(Value::Undefined),
    };
    let target = ctx
        .host_mut::<ServerRegistry>()
        .and_then(|reg| reg.servers.get(&id))
        .map(|e| (Arc::clone(&e.closed), e.local_addr));
    if let Some((closed, local_addr)) = target {
        closed.store(true, Ordering::SeqCst);
        // Connect to a concrete loopback address when bound to the wildcard, so the wake
        // actually reaches our listener.
        wake_listener(local_addr);
    }
    Ok(Value::Undefined)
}

fn wake_listener(local_addr: SocketAddr) {
    let wake_addr = if local_addr.ip().is_unspecified() {
        let ip = if local_addr.is_ipv6() {
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        } else {
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        };
        SocketAddr::new(ip, local_addr.port())
    } else {
        local_addr
    };
    let _ = TcpStream::connect(wake_addr);
}

/// `__http_server.version()` -> the runtime version string (backs `TRustJS.version`).
pub(crate) fn op_server_version(
    _ctx: &mut Ctx,
    _this: Value,
    _args: &[Value],
) -> Result<Value, Value> {
    Ok(Value::from_string(env!("CARGO_PKG_VERSION").to_string()))
}

// ---- completion decoders (run on the loop thread with &mut Ctx) --------------------------------

/// Settle one accept: on a request, stash the socket, re-arm the next accept, and return the
/// args for `__dispatch(serverId, connId, method, url, headers, body, remoteHost, remotePort)`.
/// On close, return `[serverId, -1]` (JS resolves `finished`) and do not re-arm.
fn decode_accept(
    ctx: &mut Ctx,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value>, Value> {
    let AcceptTaskResult { server_id, outcome } = *payload
        .downcast::<AcceptTaskResult>()
        .expect("accept payload");

    let accepted = match outcome {
        AcceptOutcome::Closed => {
            if let Some(reg) = ctx.host_mut::<ServerRegistry>() {
                reg.servers.remove(&server_id);
            }
            return Ok(vec![Value::Num(server_id as f64), Value::Num(-1.0)]);
        }
        AcceptOutcome::Request(accepted) => accepted,
    };

    // Re-arm the next accept from the still-live server entry (a concurrent close removes the
    // entry, in which case this connection is the last one we serve).
    let rearm = ctx
        .host_mut::<ServerRegistry>()
        .and_then(|reg| reg.servers.get(&server_id))
        .map(|e| {
            (
                Arc::clone(&e.listener),
                Arc::clone(&e.closed),
                e.local_addr,
                e.dispatch.clone(),
            )
        });

    let peer = accepted.peer;
    let method = accepted.method;
    let url = accepted.url;
    let header_pairs = accepted.headers;
    let body = accepted.body;
    let conn_id = ctx.resource_table().add(accepted.stream);

    if let Some((listener, closed, local_addr, dispatch)) = rearm {
        arm_accept(ctx, server_id, listener, closed, local_addr, dispatch);
    }

    let headers_val = header_pairs_to_js(ctx, &header_pairs);
    let body_val = if body.is_empty() {
        Value::Undefined
    } else {
        ctx.make_uint8array(&body)?
    };
    Ok(vec![
        Value::Num(server_id as f64),
        Value::Num(conn_id as f64),
        Value::from_string(method),
        Value::from_string(url),
        headers_val,
        body_val,
        Value::from_string(peer.ip().to_string()),
        Value::Num(peer.port() as f64),
    ])
}

/// Settle a response write: `resolve()` on success, `reject(error)` if the socket write failed
/// (client hung up mid-response, etc.).
fn decode_write(
    ctx: &mut Ctx,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value>, Value> {
    match *payload
        .downcast::<Result<(), String>>()
        .expect("write payload")
    {
        Ok(()) => Ok(vec![]),
        Err(message) => Err(ctx.make_error("Error", message)),
    }
}

// ---- helpers ----------------------------------------------------------------------------------

/// Register a fresh accept task (its `on_ok` is `dispatch`) and spawn it on the pool.
fn arm_accept(
    ctx: &mut Ctx,
    server_id: u64,
    listener: Arc<TcpListener>,
    closed: Arc<AtomicBool>,
    local_addr: SocketAddr,
    dispatch: Value,
) {
    let id = ctx
        .host_mut::<TaskRegistry>()
        .expect("runtime installs the registry")
        .register(dispatch, None, decode_accept);
    let spawn = ctx
        .op_state()
        .get::<CompletionSender>()
        .expect("runtime installs the dedicated executor")
        .clone();
    let fallback_host = local_addr.to_string();
    spawn.run_blocking(id, move || {
        Box::new(AcceptTaskResult {
            server_id,
            outcome: accept_one(&listener, &closed, &fallback_host),
        })
    });
}

/// Accept and parse exactly one good request (answering malformed ones with 400 inline and
/// moving on), or report the listener as closed.
fn accept_one(listener: &TcpListener, closed: &AtomicBool, fallback_host: &str) -> AcceptOutcome {
    loop {
        let (stream, peer) = match listener.accept() {
            Ok(pair) => pair,
            Err(_) => return AcceptOutcome::Closed,
        };
        if closed.load(Ordering::SeqCst) {
            return AcceptOutcome::Closed; // woken by close()'s throwaway connection
        }
        stream.set_read_timeout(Some(READ_TIMEOUT)).ok();
        match read_request(&stream, fallback_host) {
            Ok((method, url, headers, body)) => {
                return AcceptOutcome::Request(Accepted {
                    stream,
                    peer,
                    method,
                    url,
                    headers,
                    body,
                })
            }
            Err(error) => {
                let _ = write_simple(
                    &stream,
                    error.status,
                    error.reason,
                    error.message.as_bytes(),
                );
                // Keep serving: fall through to accept the next connection.
            }
        }
    }
}

/// Parse a request off the socket: request line, headers, and a Content-Length/chunked body.
/// Returns `(METHOD, absolute-url, headers, body)`.
#[allow(clippy::type_complexity)]
fn read_request(
    stream: &TcpStream,
    fallback_host: &str,
) -> Result<(String, String, Vec<(String, String)>, Vec<u8>), RequestError> {
    let mut reader = BufReader::new(stream);
    let mut metadata = 0;
    // RFC 9112 §2.2 recommends ignoring at least one empty line before a request-line.
    let request_line = loop {
        let line = read_line(&mut reader, &mut metadata, MAX_HEADER_BYTES)?;
        if !line.is_empty() {
            break line;
        }
    };
    let (method, target, version) = parse_request_line(&request_line)?;
    let headers = read_fields(&mut reader, &mut metadata, false)?;

    let hosts: Vec<_> = header_values(&headers, "host").collect();
    if (version == "HTTP/1.1" && hosts.len() != 1) || hosts.len() > 1 {
        return Err("HTTP/1.1 requires exactly one Host field".into());
    }
    let host = hosts
        .first()
        .copied()
        .filter(|value| valid_host(value))
        .unwrap_or(fallback_host);
    if version == "HTTP/1.1" && hosts.first().is_none_or(|value| !valid_host(value)) {
        return Err("invalid Host field".into());
    }

    if target.contains('#') {
        return Err("request-target must not contain a fragment".into());
    }
    let url = if target.starts_with("http://") || target.starts_with("https://") {
        // RFC 9112 §3.2.2: absolute-form authority overrides Host.
        let absolute = crate::url::parse(&target, None)?;
        if !absolute.username.is_empty() || !absolute.password.is_empty() {
            return Err("absolute-form request-target contains userinfo".into());
        }
        absolute.href()
    } else if target.starts_with('/') {
        format!("http://{host}{target}")
    } else if target == "*" && method == "OPTIONS" {
        format!("http://{host}/")
    } else if method == "CONNECT" && valid_host(&target) {
        format!("http://{target}/")
    } else {
        return Err("invalid request-target form".into());
    };

    let framing = body_framing(&headers, true, false)?;
    if matches!(framing, http_syntax::BodyFraming::Length(length) if length > MAX_BODY) {
        return Err(RequestError::new(
            413,
            "Content Too Large",
            "HTTP body too large",
        ));
    }

    let expectations: Vec<_> = header_values(&headers, "expect")
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    if expectations
        .iter()
        .any(|expectation| !expectation.eq_ignore_ascii_case("100-continue"))
    {
        return Err(RequestError::new(
            417,
            "Expectation Failed",
            "unsupported HTTP expectation",
        ));
    }
    let content_follows = !matches!(
        framing,
        http_syntax::BodyFraming::None | http_syntax::BodyFraming::Length(0)
    );
    if version == "HTTP/1.1" && !expectations.is_empty() && content_follows {
        // RFC 9110 §10.1.1: an origin server must answer the expectation before waiting for the
        // content. All header-only errors and known size violations have already been handled.
        let mut writer = stream;
        writer
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .map_err(|error| format!("write 100 Continue: {error}"))?;
    }
    let body = read_body(&mut reader, framing).map_err(|message| {
        if message == "HTTP body too large" {
            RequestError::new(413, "Content Too Large", message)
        } else {
            RequestError::from(message)
        }
    })?;
    Ok((method, url, headers, body))
}

fn parse_request_line(line: &[u8]) -> Result<(String, String, String), String> {
    if !line.is_ascii() {
        return Err("non-ASCII request-line".into());
    }
    let line = std::str::from_utf8(line).unwrap();
    let mut parts = line.split(' ');
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    let version = parts.next().unwrap_or("");
    if parts.next().is_some()
        || !http_syntax::is_token(method)
        || target.is_empty()
        || target
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
        || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
    {
        return Err("malformed HTTP request-line".into());
    }
    Ok((method.to_string(), target.to_string(), version.to_string()))
}

fn valid_host(value: &str) -> bool {
    !value.is_empty()
        && !value.contains(['/', '\\', '@', ',', ' ', '\t'])
        && !value.chars().any(char::is_control)
        && crate::url::parse(&format!("http://{value}/"), None).is_ok()
}

/// Serialize a response. We own connection framing while retaining representation-length
/// metadata where RFC 9110 §8.6 permits it for HEAD and 304 responses.
fn build_response(
    status: u16,
    status_text: &str,
    headers: &[(String, String)],
    body: &[u8],
    request_method: &str,
) -> Result<Vec<u8>, String> {
    http_syntax::validate_headers(headers)?;
    let supplied_content_length = http_syntax::content_length(headers)?;
    if status_text.chars().any(|character| {
        character == '\r'
            || character == '\n'
            || character == '\0'
            || character > '\u{ff}'
            || character.is_control() && character != '\t'
    }) {
        return Err("invalid HTTP reason phrase".into());
    }
    let reason = if status_text.is_empty() {
        reason_phrase(status)
    } else {
        status_text
    };
    let mut out = Vec::with_capacity(body.len() + 256);
    out.extend_from_slice(format!("HTTP/1.1 {status} ").as_bytes());
    out.extend_from_slice(&field_value_bytes(reason)?);
    out.extend_from_slice(b"\r\n");

    let mut has_server = false;
    for (k, v) in headers {
        if k.eq_ignore_ascii_case("connection")
            || k.eq_ignore_ascii_case("transfer-encoding")
            || k.eq_ignore_ascii_case("content-length")
        {
            continue; // we own connection framing
        }
        has_server |= k.eq_ignore_ascii_case("server");
        out.extend_from_slice(k.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(&field_value_bytes(v)?);
        out.extend_from_slice(b"\r\n");
    }
    let head_response = request_method.eq_ignore_ascii_case("HEAD");
    let successful_connect =
        request_method.eq_ignore_ascii_case("CONNECT") && (200..300).contains(&status);
    let content_length = if status == 204 || successful_connect {
        // RFC 9110 §8.6: neither 204 nor a successful CONNECT response may carry this field.
        None
    } else if status == 205 {
        Some(0)
    } else if status == 304 {
        // An explicit value describes the selected 200 response representation; without one the
        // server is allowed to omit it.
        supplied_content_length
    } else if head_response && body.is_empty() {
        // A HEAD handler can avoid generating the GET body while still providing its known size.
        // If it did generate a body below, its exact byte length remains the framing authority.
        supplied_content_length.or(Some(0))
    } else {
        Some(body.len() as u64)
    };
    if let Some(content_length) = content_length {
        out.extend_from_slice(format!("Content-Length: {content_length}\r\n").as_bytes());
    }
    if !has_server {
        out.extend_from_slice(
            concat!("Server: trust-js/", env!("CARGO_PKG_VERSION"), "\r\n").as_bytes(),
        );
    }
    out.extend_from_slice(b"Connection: close\r\n\r\n");
    if out.len() > MAX_HEADER_BYTES {
        return Err("HTTP response metadata too large".into());
    }
    if !head_response && !successful_connect && !matches!(status, 204 | 205 | 304) {
        out.extend_from_slice(body);
    }
    Ok(out)
}

/// A bare status-only response used for the inline 400 path.
fn write_simple(stream: &TcpStream, status: u16, reason: &str, body: &[u8]) -> std::io::Result<()> {
    let mut s = stream;
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    s.flush()?;
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

fn write_all_and_close(stream: TcpStream, bytes: &[u8]) -> Result<(), String> {
    let mut s = &stream;
    s.write_all(bytes)
        .map_err(|e| format!("response write: {e}"))?;
    s.flush().ok();
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

fn header_pairs_to_js(ctx: &Ctx, headers: &[(String, String)]) -> Value {
    let pairs = headers
        .iter()
        .map(|(k, v)| {
            ctx.make_array(vec![
                Value::from_string(k.clone()),
                Value::from_string(v.clone()),
            ])
        })
        .collect();
    ctx.make_array(pairs)
}

/// The default reason phrase for the common statuses; anything else gets an empty phrase (valid
/// per HTTP — clients ignore it).
fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        410 => "Gone",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::thread;

    fn parse_raw_request(
        raw: &[u8],
    ) -> Result<(String, String, Vec<(String, String)>, Vec<u8>), String> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let raw = raw.to_vec();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream.write_all(&raw).unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
        });
        let (stream, _) = listener.accept().unwrap();
        let result = read_request(&stream, "fallback.invalid").map_err(|error| error.message);
        client.join().unwrap();
        result
    }

    #[test]
    fn server_reads_framed_bodies_independently_of_method() {
        let (method, url, _, body) = parse_raw_request(
            b"GET /resource HTTP/1.1\r\nHost: example.test\r\nContent-Length: 4\r\n\r\ndata",
        )
        .unwrap();
        assert_eq!(method, "GET");
        assert_eq!(url, "http://example.test/resource");
        assert_eq!(body, b"data");
    }

    #[test]
    fn server_answers_100_continue_before_waiting_for_content() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(
                    b"POST / HTTP/1.1\r\nHost: example.test\r\nContent-Length: 4\r\nExpect: 100-continue\r\n\r\n",
                )
                .unwrap();
            let mut interim = [0; 25];
            stream.read_exact(&mut interim).unwrap();
            assert_eq!(&interim, b"HTTP/1.1 100 Continue\r\n\r\n");
            stream.write_all(b"data").unwrap();
        });
        let (stream, _) = listener.accept().unwrap();
        let (_, _, _, body) = read_request(&stream, "fallback.invalid").unwrap();
        client.join().unwrap();
        assert_eq!(body, b"data");
    }

    #[test]
    fn server_rejects_ambiguous_framing_and_invalid_host_fields() {
        for raw in [
            &b"POST / HTTP/1.1\r\nHost: example.test\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nx"[..],
            &b"POST / HTTP/1.1\r\nHost: example.test\r\nTransfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n0\r\n\r\n"[..],
            &b"GET / HTTP/1.1\r\n\r\n"[..],
            &b"GET / HTTP/1.1\r\nHost: one.test\r\nHost: two.test\r\n\r\n"[..],
            &b"GET / HTTP/1.1\r\nHost: bad/path\r\n\r\n"[..],
        ] {
            assert!(parse_raw_request(raw).is_err());
        }
    }

    #[test]
    fn response_serialization_owns_framing_and_obeys_null_body_rules() {
        let caller_headers = vec![
            ("Content-Length".into(), "999".into()),
            ("Transfer-Encoding".into(), "chunked".into()),
            ("Connection".into(), "keep-alive".into()),
            ("X-Test".into(), "yes".into()),
        ];
        let head = build_response(200, "", &caller_headers, b"body", "HEAD").unwrap();
        let head = String::from_utf8(head).unwrap();
        assert!(head.contains("Content-Length: 4\r\n"));
        assert!(head.contains("Connection: close\r\n"));
        assert!(head.contains("X-Test: yes\r\n"));
        assert!(!head.ends_with("body"));
        assert!(!head.contains("Transfer-Encoding:"));
        assert!(!head.contains("Content-Length: 999"));

        let metadata_head = build_response(
            200,
            "",
            &[("Content-Length".into(), "5".into())],
            b"",
            "HEAD",
        )
        .unwrap();
        let metadata_head = String::from_utf8(metadata_head).unwrap();
        assert!(metadata_head.contains("Content-Length: 5\r\n"));

        let no_content = build_response(
            204,
            "",
            &[("Content-Length".into(), "7".into())],
            b"ignored",
            "GET",
        )
        .unwrap();
        let no_content = String::from_utf8(no_content).unwrap();
        assert!(!no_content.contains("Content-Length:"));
        assert!(!no_content.ends_with("ignored"));

        let reset = build_response(205, "", &[], b"ignored", "GET").unwrap();
        let reset = String::from_utf8(reset).unwrap();
        assert!(reset.contains("Content-Length: 0\r\n"));
        assert!(!reset.ends_with("ignored"));

        let not_modified = build_response(
            304,
            "",
            &[("Content-Length".into(), "11".into())],
            b"ignored",
            "GET",
        )
        .unwrap();
        let not_modified = String::from_utf8(not_modified).unwrap();
        assert!(not_modified.contains("Content-Length: 11\r\n"));
        assert!(!not_modified.ends_with("ignored"));

        let tunnel = build_response(
            200,
            "",
            &[("Content-Length".into(), "7".into())],
            b"ignored",
            "CONNECT",
        )
        .unwrap();
        let tunnel = String::from_utf8(tunnel).unwrap();
        assert!(!tunnel.contains("Content-Length:"));
        assert!(!tunnel.ends_with("ignored"));
    }

    #[test]
    fn response_serialization_rejects_injection_and_oversized_metadata() {
        assert!(build_response(200, "ok\r\ninjected", &[], b"", "GET").is_err());
        assert!(build_response(
            200,
            "",
            &[("X-Test".into(), "x".repeat(MAX_HEADER_BYTES))],
            b"",
            "GET"
        )
        .is_err());
    }
}
