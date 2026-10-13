//! WebSocket (RFC 6455) over plain TCP or verified TLS: a *client* (the upgrade sibling of the fetch
//! client in `http.rs`, driving the `WebSocket` global in `js/websocket.js`) plus a *server-side
//! adopt* (`op_ws_upgrade`), which takes a connection accepted by the HTTP server in `server.rs`,
//! answers the 101 handshake, and runs it through the same registry/read-loop with unmasked
//! outgoing frames — backing `TRustJS.upgradeWebSocket` and Bun.serve's `websocket` option.
//!
//! ## How it runs on the loop
//! Same re-arm pattern as the HTTP server: `connect` runs the TCP/TLS dial + HTTP upgrade handshake
//! on the bounded dedicated executor and comes back as a completion; the decoder stores the stream,
//! fires the socket's JS dispatch (`"open"`), and arms a reader task. Each reader task blocks
//! until ONE complete message (transparently answering pings and swallowing pongs), returns it as
//! a completion — which re-arms the next read — and carries the fragmentation-capable reader back
//! and forth by value. Sends/closes write on the loop thread under a per-socket write mutex (a
//! write timeout bounds a stalled peer; real backpressure handling is future work, matching the
//! server's v1 notes).
//!
//! ## What's intentionally missing (v1)
//! - **permessage-deflate** and other extensions (`extensions` is always `""`).
//! - **Backpressure**: `send` writes synchronously; `bufferedAmount` is 0 once `send` returns.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use trust_js_host::{CompletionSender, Ctx, TaskRegistry, Value};

use crate::sha1::sha1;
use crate::url;

const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
/// Bound the opening handshake and writes so a dead peer cannot pin a worker or the loop.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// Message size cap (mirrors the HTTP body cap); exceeding it fails the connection with 1009.
const MAX_MESSAGE: usize = 32 << 20;

// ---- base64 (the handshake key/accept values) -------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(crate) fn base64(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Strict RFC 4648 base64 decoding for the `Sec-WebSocket-Key` server-side check. WebSocket
/// keys must decode to exactly 16 bytes (RFC 6455 §4.2.1 item 5); keeping this decoder local
/// avoids making the JavaScript `atob()` implementation part of a wire-protocol trust boundary.
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    if input.is_empty() || !input.len().is_multiple_of(4) {
        return None;
    }
    let value = |byte: u8| match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    };
    let (chunks, _) = input.as_bytes().as_chunks::<4>();
    let chunk_count = chunks.len();
    let mut out = Vec::with_capacity(chunk_count * 3);
    for (index, chunk) in chunks.iter().enumerate() {
        let last = index + 1 == chunk_count;
        let a = value(chunk[0])?;
        let b = value(chunk[1])?;
        out.push(a << 2 | b >> 4);
        match (chunk[2], chunk[3]) {
            (b'=', b'=') if last && b & 0x0f == 0 => {}
            (b'=', _) => return None,
            (c, b'=') if last => {
                let c = value(c)?;
                if c & 0x03 != 0 {
                    return None;
                }
                out.push(b << 4 | c >> 2);
            }
            (c, d) => {
                let c = value(c)?;
                let d = value(d)?;
                out.push(b << 4 | c >> 2);
                out.push(c << 6 | d);
            }
        }
    }
    Some(out)
}

/// The `Sec-WebSocket-Accept` value for a handshake `Sec-WebSocket-Key` (RFC 6455 §4.2.2 step
/// 5.4). Public: a WebSocket-capable server upgrade (and this crate's own tests) need it too.
pub fn websocket_accept(key: &str) -> String {
    let mut input = key.trim().to_string();
    input.push_str(GUID);
    base64(&sha1(input.as_bytes()))
}

// ---- frame codec -------------------------------------------------------------------------------

/// One decoded event from the wire, message-level (fragments already assembled).
#[derive(Debug, PartialEq)]
pub(crate) enum WsEvent {
    Text(String),
    Binary(Vec<u8>),
    /// Peer close frame: `(code, reason)`; 1005 = no code present.
    Close(u16, String),
}

/// Why a read loop ended without a clean message.
#[derive(Debug)]
pub(crate) enum WsError {
    /// Protocol violation → fail the connection with this close code.
    Protocol(u16, &'static str),
    /// The socket died (EOF/reset/timeout).
    Io(String),
}

/// Encode one frame. Client frames are ALWAYS masked (RFC 6455 §5.3); `mask` comes from the
/// caller so the codec stays deterministic under test.
pub(crate) fn encode_frame(opcode: u8, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 14);
    out.push(0x80 | (opcode & 0x0f)); // FIN + opcode
    let len = payload.len();
    if len < 126 {
        out.push(0x80 | len as u8);
    } else if len <= 0xffff {
        out.push(0x80 | 126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0x80 | 127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(&mask);
    out.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
    out
}

/// Encode one UNMASKED frame — the server side of the wire (a server MUST NOT mask, RFC 6455
/// §5.1). Used by connections adopted via `op_ws_upgrade` (`TRustJS.serve` → WebSocket handoff).
pub(crate) fn encode_frame_unmasked(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 10);
    out.push(0x80 | (opcode & 0x0f)); // FIN + opcode
    let len = payload.len();
    if len < 126 {
        out.push(len as u8);
    } else if len <= 0xffff {
        out.push(126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

/// A close frame's payload (code + UTF-8 reason).
pub(crate) fn close_payload(code: u16, reason: &str) -> Vec<u8> {
    let mut p = code.to_be_bytes().to_vec();
    p.extend_from_slice(reason.as_bytes());
    p
}

struct RawFrame {
    fin: bool,
    opcode: u8,
    payload: Vec<u8>,
}

/// Incremental RFC 3629 UTF-8 validation for RFC 6455 §8.1. WebSocket text can split a scalar
/// value across both transport reads and continuation frames, so validating only the assembled
/// message needlessly buffers hostile input after its first provably-invalid byte.
#[derive(Default)]
struct Utf8Validator {
    continuation_bytes: u8,
    next_min: u8,
    next_max: u8,
}

impl Utf8Validator {
    fn push(&mut self, bytes: &[u8]) -> Result<(), WsError> {
        for &byte in bytes {
            if self.continuation_bytes != 0 {
                if !(self.next_min..=self.next_max).contains(&byte) {
                    return Err(WsError::Protocol(1007, "text message is not UTF-8"));
                }
                self.continuation_bytes -= 1;
                self.next_min = 0x80;
                self.next_max = 0xbf;
                continue;
            }
            match byte {
                0x00..=0x7f => {}
                0xc2..=0xdf => self.begin(1, 0x80, 0xbf),
                0xe0 => self.begin(2, 0xa0, 0xbf),
                0xe1..=0xec | 0xee..=0xef => self.begin(2, 0x80, 0xbf),
                0xed => self.begin(2, 0x80, 0x9f),
                0xf0 => self.begin(3, 0x90, 0xbf),
                0xf1..=0xf3 => self.begin(3, 0x80, 0xbf),
                0xf4 => self.begin(3, 0x80, 0x8f),
                _ => return Err(WsError::Protocol(1007, "text message is not UTF-8")),
            }
        }
        Ok(())
    }

    fn begin(&mut self, continuation_bytes: u8, next_min: u8, next_max: u8) {
        self.continuation_bytes = continuation_bytes;
        self.next_min = next_min;
        self.next_max = next_max;
    }

    fn finish(&self) -> Result<(), WsError> {
        if self.continuation_bytes == 0 {
            Ok(())
        } else {
            Err(WsError::Protocol(1007, "text message is not UTF-8"))
        }
    }
}

fn read_exact_buf(r: &mut impl Read, n: usize) -> Result<Vec<u8>, WsError> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)
        .map_err(|e| WsError::Io(e.to_string()))?;
    Ok(buf)
}

fn read_raw_frame(r: &mut impl Read, max: usize, expect_masked: bool) -> Result<RawFrame, WsError> {
    read_raw_frame_with_utf8(r, max, expect_masked, None, None)
}

fn read_raw_frame_with_utf8(
    r: &mut impl Read,
    max: usize,
    expect_masked: bool,
    fragmented_opcode: Option<u8>,
    mut utf8: Option<&mut Utf8Validator>,
) -> Result<RawFrame, WsError> {
    let head = read_exact_buf(r, 2)?;
    let fin = head[0] & 0x80 != 0;
    if head[0] & 0x70 != 0 {
        return Err(WsError::Protocol(
            1002,
            "reserved bits set (no extension negotiated)",
        ));
    }
    let opcode = head[0] & 0x0f;
    if !matches!(opcode, 0x0 | 0x1 | 0x2 | 0x8 | 0x9 | 0xA) {
        return Err(WsError::Protocol(1002, "reserved or unknown opcode"));
    }
    let masked = head[1] & 0x80 != 0;
    if masked != expect_masked {
        // RFC 6455 §5.1: client frames are always masked and server frames never are.
        return Err(WsError::Protocol(1002, "frame has incorrect masking"));
    }
    let short_len = head[1] & 0x7f;
    if opcode >= 0x8 && (!fin || short_len > 125) {
        // Control frames are never fragmented and their length marker itself must fit in seven
        // bits (RFC 6455 §5.5); do not wait for an invalid extended-length field to arrive.
        return Err(WsError::Protocol(1002, "malformed control frame"));
    }
    let mut len = short_len as usize;
    if short_len == 126 {
        let ext = read_exact_buf(r, 2)?;
        len = u16::from_be_bytes([ext[0], ext[1]]) as usize;
        if len < 126 {
            return Err(WsError::Protocol(1002, "non-minimal payload length"));
        }
    } else if short_len == 127 {
        let ext = read_exact_buf(r, 8)?;
        let n = u64::from_be_bytes(ext.try_into().unwrap());
        if n >> 63 != 0 || n <= u16::MAX as u64 {
            return Err(WsError::Protocol(1002, "invalid 64-bit payload length"));
        }
        if n > max as u64 {
            return Err(WsError::Protocol(1009, "message too big"));
        }
        len = n as usize;
    }
    if len > max {
        return Err(WsError::Protocol(1009, "message too big"));
    }
    let mask: Option<[u8; 4]> = if masked {
        Some(read_exact_buf(r, 4)?.try_into().unwrap())
    } else {
        None
    };
    let validates_text = utf8.is_some()
        && (opcode == 0x1 && fragmented_opcode.is_none()
            || opcode == 0x0 && fragmented_opcode == Some(0x1));
    let mut payload = vec![0; len];
    let mut offset = 0;
    while offset < len {
        // A direct read makes validation track available transport data. `read_exact` over the
        // whole frame would wait for attacker-controlled trailing bytes after an invalid prefix.
        let end = (offset + 8 * 1024).min(len);
        match r.read(&mut payload[offset..end]) {
            Ok(0) => return Err(WsError::Io("failed to fill whole buffer".into())),
            Ok(read) => {
                if let Some(mask) = mask {
                    for (index, byte) in payload[offset..offset + read].iter_mut().enumerate() {
                        *byte ^= mask[(offset + index) % 4];
                    }
                }
                if validates_text {
                    utf8.as_deref_mut()
                        .unwrap()
                        .push(&payload[offset..offset + read])?;
                }
                offset += read;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(WsError::Io(error.to_string())),
        }
    }
    if validates_text && fin {
        utf8.as_deref().unwrap().finish()?;
    }
    Ok(RawFrame {
        fin,
        opcode,
        payload,
    })
}

/// The reader half: owns the read stream plus cross-message fragmentation state, and a write
/// handle for transparent ping→pong replies. Moves into each pool read task and back out
/// through its completion.
trait WsStream: Read + Write + Send {}
impl<T: Read + Write + Send> WsStream for T {}

trait WsRead: Read + Send {}
impl<T: Read + Send> WsRead for T {}

#[repr(C)]
struct PollFd {
    fd: std::ffi::c_int,
    events: i16,
    revents: i16,
}

#[cfg(target_os = "linux")]
type PollCount = usize;
#[cfg(target_os = "macos")]
type PollCount = u32;

unsafe extern "C" {
    fn poll(fds: *mut PollFd, count: PollCount, timeout_ms: std::ffi::c_int) -> std::ffi::c_int;
}

const POLLIN: i16 = 0x001;
const POLLOUT: i16 = 0x004;

fn wait_for_socket(
    socket: &TcpStream,
    events: i16,
    timeout: Option<Duration>,
) -> std::io::Result<()> {
    let timeout_ms = timeout.map_or(-1, |duration| {
        duration.as_millis().min(std::ffi::c_int::MAX as u128) as std::ffi::c_int
    });
    loop {
        let mut descriptor = PollFd {
            fd: socket.as_raw_fd(),
            events,
            revents: 0,
        };
        let result = unsafe { poll(&mut descriptor, 1, timeout_ms) };
        if result > 0 {
            return Ok(());
        }
        if result == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "WebSocket TLS readiness wait timed out",
            ));
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[derive(Clone)]
struct SharedStream {
    stream: Arc<Mutex<Box<dyn WsStream>>>,
    /// Present for a nonblocking TLS transport. Readiness waits happen on this duplicated socket
    /// after releasing `stream`, so an idle SSL_read neither spins nor excludes SSL_write.
    readiness: Option<Arc<TcpStream>>,
}

impl SharedStream {
    fn new(stream: Box<dyn WsStream>, readiness: Option<Arc<TcpStream>>) -> Self {
        Self {
            stream: Arc::new(Mutex::new(stream)),
            readiness,
        }
    }

    fn retry_events(error: &std::io::Error) -> Option<i16> {
        match error.kind() {
            // `TlsStream` maps SSL_ERROR_WANT_READ and SSL_ERROR_WANT_WRITE to distinct standard
            // kinds so readiness waits do not wake continuously on an unrelated writable socket.
            std::io::ErrorKind::WouldBlock => Some(POLLIN),
            std::io::ErrorKind::Interrupted => Some(POLLOUT),
            _ => None,
        }
    }
}

impl Read for SharedStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let result = self
                .stream
                .lock()
                .map_err(|_| std::io::Error::other("WebSocket stream lock poisoned"))?
                .read(buffer);
            match result {
                Err(error) if Self::retry_events(&error).is_some() && self.readiness.is_some() => {
                    wait_for_socket(
                        self.readiness.as_deref().unwrap(),
                        Self::retry_events(&error).unwrap(),
                        None,
                    )?;
                }
                other => return other,
            }
        }
    }
}
impl Write for SharedStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        loop {
            let result = self
                .stream
                .lock()
                .map_err(|_| std::io::Error::other("WebSocket stream lock poisoned"))?
                .write(buffer);
            match result {
                Err(error) if Self::retry_events(&error).is_some() && self.readiness.is_some() => {
                    // OpenSSL requires retrying SSL_write with identical bytes after WANT_READ or
                    // WANT_WRITE. This loop retains `buffer` and bounds the event-loop stall.
                    wait_for_socket(
                        self.readiness.as_deref().unwrap(),
                        Self::retry_events(&error).unwrap(),
                        Some(WRITE_TIMEOUT),
                    )?;
                }
                other => return other,
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.stream
            .lock()
            .map_err(|_| std::io::Error::other("WebSocket stream lock poisoned"))?
            .flush()
    }
}

pub(crate) struct WsReader {
    stream: BufReader<Box<dyn WsRead>>,
    writer: SharedStream,
    mask_rng: Option<File>,
    /// true = we are the CLIENT end (mask outgoing pongs); false = server end (never mask).
    masked: bool,
}

impl WsReader {
    fn next_mask(&mut self) -> Result<[u8; 4], WsError> {
        let Some(rng) = &mut self.mask_rng else {
            return Err(WsError::Io("WebSocket masking entropy unavailable".into()));
        };
        let mut mask = [0; 4];
        rng.read_exact(&mut mask)
            .map_err(|error| WsError::Io(format!("WebSocket masking entropy: {error}")))?;
        Ok(mask)
    }

    /// Block until one complete MESSAGE (or close), answering pings and skipping pongs inline.
    pub(crate) fn read_message(&mut self) -> Result<WsEvent, WsError> {
        let mut partial: Option<(u8, Vec<u8>)> = None;
        let mut utf8 = Utf8Validator::default();
        loop {
            // RFC 6455 §5.1: a client expects unmasked server frames; a server expects masked
            // client frames. This is the inverse of whether this endpoint masks its own output.
            let frame = read_raw_frame_with_utf8(
                &mut self.stream,
                MAX_MESSAGE,
                !self.masked,
                partial.as_ref().map(|(opcode, _)| *opcode),
                Some(&mut utf8),
            )?;
            match frame.opcode {
                0x9 => {
                    // Ping → pong with the same payload (§5.5.3); masked only from the client end.
                    let pong = if self.masked {
                        let mask = self.next_mask()?;
                        encode_frame(0xA, &frame.payload, mask)
                    } else {
                        encode_frame_unmasked(0xA, &frame.payload)
                    };
                    self.writer
                        .write_all(&pong)
                        .map_err(|e| WsError::Io(e.to_string()))?;
                }
                0xA => {} // unsolicited pong: ignore (§5.5.3)
                0x8 => {
                    let (code, reason) = if frame.payload.len() >= 2 {
                        let code = u16::from_be_bytes([frame.payload[0], frame.payload[1]]);
                        if !valid_received_close_code(code) {
                            return Err(WsError::Protocol(1002, "invalid close status code"));
                        }
                        let reason = String::from_utf8(frame.payload[2..].to_vec())
                            .map_err(|_| WsError::Protocol(1007, "close reason is not UTF-8"))?;
                        (code, reason)
                    } else if frame.payload.len() == 1 {
                        return Err(WsError::Protocol(1002, "one-byte close payload"));
                    } else {
                        (1005, String::new())
                    };
                    return Ok(WsEvent::Close(code, reason));
                }
                0x1 | 0x2 => {
                    if partial.is_some() {
                        return Err(WsError::Protocol(
                            1002,
                            "new data frame during fragmented message",
                        ));
                    }
                    if frame.fin {
                        return finish_message(frame.opcode, frame.payload);
                    }
                    partial = Some((frame.opcode, frame.payload));
                }
                0x0 => {
                    let Some((op, mut buf)) = partial.take() else {
                        return Err(WsError::Protocol(1002, "continuation without a message"));
                    };
                    if buf
                        .len()
                        .checked_add(frame.payload.len())
                        .is_none_or(|length| length > MAX_MESSAGE)
                    {
                        return Err(WsError::Protocol(1009, "message too big"));
                    }
                    buf.extend_from_slice(&frame.payload);
                    if frame.fin {
                        return finish_message(op, buf);
                    }
                    partial = Some((op, buf));
                }
                _ => return Err(WsError::Protocol(1002, "unknown opcode")),
            }
        }
    }
}

/// Codes currently assigned by IANA for protocol use, plus the application/library ranges.
/// RFC 6455 §7.4 reserves 1004/1005/1006/1015 for APIs or future use and forbids them on the
/// wire; the IANA registry currently assigns 1012–1014 and leaves 1016–2999 unassigned.
fn valid_received_close_code(code: u16) -> bool {
    matches!(code, 1000..=1003 | 1007..=1014 | 3000..=4999)
}

fn finish_message(opcode: u8, payload: Vec<u8>) -> Result<WsEvent, WsError> {
    if opcode == 0x1 {
        String::from_utf8(payload)
            .map(WsEvent::Text)
            .map_err(|_| WsError::Protocol(1007, "text message is not UTF-8"))
    } else {
        Ok(WsEvent::Binary(payload))
    }
}

// ---- registry + ops ----------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct WsRegistry {
    next: u64,
    socks: HashMap<u64, WsEntry>,
}

struct WsEntry {
    /// Write half; `None` until the handshake completes.
    writer: Option<SharedStream>,
    /// One close frame ever goes out (ours or the echo of theirs).
    close_sent: Arc<AtomicBool>,
    /// Stops the read re-arm after close/error delivery.
    dead: bool,
    dispatch: Value,
    /// A duplicate OS socket used only to interrupt a blocked reader when the entry is dropped.
    /// It is deliberately outside the TLS/shared-stream mutex.
    cancel: Arc<Mutex<Option<Arc<TcpStream>>>>,
    cancelled: Arc<AtomicBool>,
    /// true = client end (outgoing frames masked); false = a connection adopted server-side.
    masked: bool,
}

impl Drop for WsEntry {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(socket) = self
            .cancel
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
}

/// What the connect task sends back to the loop.
struct ConnectResult {
    id: u64,
    outcome: Result<ConnectedSocket, String>,
}

enum WsTransport {
    Plain(TcpStream),
    Tls(Box<trust_js_tls::TlsStream>),
}

impl Read for WsTransport {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.read(buffer),
            Self::Tls(stream) => stream.read(buffer),
        }
    }
}

impl Write for WsTransport {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(stream) => stream.write(buffer),
            Self::Tls(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(stream) => stream.flush(),
            Self::Tls(stream) => stream.flush(),
        }
    }
}

struct ConnectedSocket {
    stream: WsTransport,
    cancel: Arc<TcpStream>,
    mask_rng: File,
    protocol: String,
}

/// What one read task sends back to the loop.
struct ReadResult {
    id: u64,
    outcome: Result<WsEvent, WsError>,
    /// Handed back for the next read task (None when the loop should stop).
    reader: Option<WsReader>,
}

fn ws_registry(ctx: &mut Ctx) -> &mut WsRegistry {
    ctx.host_mut::<WsRegistry>()
        .expect("web installs WsRegistry")
}

fn fresh_mask(ctx: &mut Ctx) -> Result<[u8; 4], Value> {
    crate::web_random_bytes(ctx, 4)?
        .try_into()
        .map_err(|_| ctx.make_error("Error", "WebSocket masking entropy returned wrong length"))
}

fn encode_endpoint_frame(
    ctx: &mut Ctx,
    masked: bool,
    opcode: u8,
    payload: &[u8],
) -> Result<Vec<u8>, Value> {
    if masked {
        Ok(encode_frame(opcode, payload, fresh_mask(ctx)?))
    } else {
        Ok(encode_frame_unmasked(opcode, payload))
    }
}

/// `__ws.connect(url, protocolsJoined, dispatch)` → id. The handshake runs on the pool; the
/// socket's lifecycle then flows entirely through `dispatch(kind, ...)`:
/// `("open", protocol)`, `("text", string)`, `("binary", u8array)`,
/// `("close", code, reason, wasClean)`, `("error", message)`.
pub(crate) fn op_ws_connect(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let target = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let protocols = ctx
        .coerce_string(args.get(1).unwrap_or(&Value::Undefined))?
        .to_string();
    if !valid_protocol_list(&protocols) {
        return Err(ctx.make_error(
            "SyntaxError",
            "WebSocket protocols must be comma-separated HTTP tokens",
        ));
    }
    let dispatch = match args.get(2) {
        Some(v) if v.is_callable() => v.clone(),
        _ => return Err(ctx.make_error("TypeError", "connect: dispatch must be a function")),
    };

    let u = url::parse(&target, None).map_err(|e| ctx.make_error("SyntaxError", e))?;
    match u.scheme.as_str() {
        "ws" | "wss" => {}
        other => {
            return Err(ctx.make_error(
                "SyntaxError",
                format!("WebSocket: unsupported scheme '{other}'"),
            ))
        }
    }

    // The WebSocket Standard opening handshake selects a fresh 16-byte nonce.
    let key_bytes = crate::web_random_bytes(ctx, 16)?;
    let key = base64(&key_bytes);

    let cancel = Arc::new(Mutex::new(None));
    let cancelled = Arc::new(AtomicBool::new(false));
    let id = {
        let reg = ws_registry(ctx);
        let id = reg.next;
        reg.next += 1;
        reg.socks.insert(
            id,
            WsEntry {
                writer: None,
                close_sent: Arc::new(AtomicBool::new(false)),
                dead: false,
                dispatch: dispatch.clone(),
                cancel: Arc::clone(&cancel),
                cancelled: Arc::clone(&cancelled),
                masked: true,
            },
        );
        id
    };

    let task = ctx
        .host_mut::<TaskRegistry>()
        .expect("runtime installs the registry")
        .register(dispatch, None, decode_connect);
    let spawn = ctx
        .op_state()
        .get::<CompletionSender>()
        .expect("runtime installs the dedicated executor")
        .clone();
    spawn.run_blocking(task, move || {
        Box::new(ConnectResult {
            id,
            outcome: handshake(&u, &key, &protocols, &cancel, &cancelled),
        })
    });
    Ok(Value::Num(id as f64))
}

/// Dial + HTTP/1.1 upgrade (RFC 6455 §4.1/§4.2). Returns the open stream and the negotiated
/// subprotocol ("" when none).
fn handshake(
    u: &url::Url,
    key: &str,
    protocols: &str,
    cancel_slot: &Mutex<Option<Arc<TcpStream>>>,
    cancelled: &AtomicBool,
) -> Result<ConnectedSocket, String> {
    let port = u.port.unwrap_or(if u.scheme == "wss" { 443 } else { 80 });
    let host = u.host.trim_matches(['[', ']']);
    let tcp = TcpStream::connect((host, port)).map_err(|e| format!("connect: {e}"))?;
    tcp.set_nodelay(true).ok();
    tcp.set_write_timeout(Some(WRITE_TIMEOUT)).ok();
    tcp.set_read_timeout(Some(HANDSHAKE_TIMEOUT)).ok();
    let cancel = Arc::new(
        tcp.try_clone()
            .map_err(|error| format!("clone WebSocket socket: {error}"))?,
    );
    *cancel_slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&cancel));
    if cancelled.load(Ordering::Acquire) {
        let _ = cancel.shutdown(Shutdown::Both);
        return Err("WebSocket connection was cancelled".into());
    }
    let mut stream = if u.scheme == "wss" {
        WsTransport::Tls(Box::new(trust_js_tls::TlsStream::connect(tcp, host)?))
    } else {
        WsTransport::Plain(tcp)
    };

    let default_port = if u.scheme == "wss" { 443 } else { 80 };
    let host_header = if u.port.is_some() && u.port != Some(default_port) {
        format!("{}:{}", u.host, port)
    } else {
        u.host.clone()
    };
    let mut path = if u.path.is_empty() {
        "/".to_string()
    } else {
        u.path.clone()
    };
    path.push_str(&u.query); // `query` already carries its leading '?' when present
    let mut req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\nUpgrade: websocket\r\n\
         Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
    );
    if !protocols.is_empty() {
        req.push_str(&format!("Sec-WebSocket-Protocol: {protocols}\r\n"));
    }
    req.push_str("\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("handshake write: {e}"))?;

    // Read the 101 response head. BufReader over a clone would over-read into the frame stream,
    // so read byte-wise until CRLFCRLF (the head is tiny).
    let mut head = Vec::with_capacity(256);
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if head.len() > 64 << 10 {
            return Err("handshake response too large".into());
        }
        stream
            .read_exact(&mut byte)
            .map_err(|e| format!("handshake read: {e}"))?;
        head.push(byte[0]);
    }
    let protocol = validate_handshake_response(&head, key, protocols)?;
    // Plain TCP gets independent read/write ownership and can block indefinitely. OpenSSL keeps
    // one serialized SSL state machine, but nonblocking calls wait for readiness outside that
    // serialization boundary, avoiding both idle wakeups and read/write exclusion.
    match &stream {
        WsTransport::Plain(socket) => socket.set_read_timeout(None),
        WsTransport::Tls(socket) => socket.set_nonblocking(true),
    }
    .map_err(|error| format!("configure WebSocket transport: {error}"))?;
    let mask_rng = File::open("/dev/urandom")
        .map_err(|error| format!("open WebSocket masking entropy: {error}"))?;
    Ok(ConnectedSocket {
        stream,
        cancel,
        mask_rng,
        protocol,
    })
}

fn validate_handshake_response(head: &[u8], key: &str, protocols: &str) -> Result<String, String> {
    let head = std::str::from_utf8(head)
        .map_err(|_| "handshake response headers are not UTF-8".to_string())?;
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap_or("");
    let mut status_parts = status.split_ascii_whitespace();
    if status_parts.next() != Some("HTTP/1.1")
        || status_parts.next() != Some("101")
        || status_parts.next().is_none()
    {
        return Err(format!("handshake refused: {status}"));
    }
    let mut accept = None;
    let mut upgrade_ok = false;
    let mut connection_ok = false;
    let mut protocol = None;
    let mut extension_seen = false;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            return Err("malformed handshake response header".into());
        };
        let (k, v) = (k.trim(), v.trim());
        if k.eq_ignore_ascii_case("sec-websocket-accept") {
            if accept.is_some() {
                return Err("duplicate Sec-WebSocket-Accept response header".into());
            }
            accept = Some(v.to_string());
        } else if k.eq_ignore_ascii_case("upgrade") {
            upgrade_ok |= header_has_token(v, "websocket");
        } else if k.eq_ignore_ascii_case("connection") {
            connection_ok |= header_has_token(v, "upgrade");
        } else if k.eq_ignore_ascii_case("sec-websocket-protocol") {
            if protocol.is_some() {
                return Err("duplicate Sec-WebSocket-Protocol response header".into());
            }
            if !crate::http_syntax::is_token(v) {
                return Err("invalid Sec-WebSocket-Protocol response header".into());
            }
            protocol = Some(v.to_string());
        } else if k.eq_ignore_ascii_case("sec-websocket-extensions") {
            extension_seen = true;
        }
    }
    if !upgrade_ok {
        return Err("handshake response missing 'Upgrade: websocket'".into());
    }
    if !connection_ok {
        return Err("handshake response missing 'Connection: Upgrade'".into());
    }
    if accept.as_deref() != Some(websocket_accept(key).as_str()) {
        return Err("handshake Sec-WebSocket-Accept mismatch".into());
    }
    if extension_seen {
        return Err("server selected an extension the client did not offer".into());
    }
    // Subprotocol values are case-sensitive. RFC 6455 §4.1 rejects an unoffered value; the
    // WebSocket Standard opening-handshake algorithm also rejects a missing selection when the
    // constructor supplied a non-empty protocol list.
    if !protocols.is_empty() && protocol.is_none() {
        return Err("server did not select a requested subprotocol".into());
    }
    let protocol = protocol.unwrap_or_default();
    if !protocol.is_empty() && !protocols.split(',').map(str::trim).any(|p| p == protocol) {
        return Err(format!(
            "server selected unrequested subprotocol '{protocol}'"
        ));
    }
    Ok(protocol)
}

fn header_has_token(value: &str, expected: &str) -> bool {
    value
        .split(',')
        .map(str::trim)
        .any(|token| token.eq_ignore_ascii_case(expected))
}

fn valid_protocol_list(protocols: &str) -> bool {
    protocols.is_empty()
        || protocols
            .split(',')
            .map(str::trim)
            .all(crate::http_syntax::is_token)
}

fn decode_connect(
    ctx: &mut Ctx,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value>, Value> {
    let ConnectResult { id, outcome } = *payload.downcast::<ConnectResult>().expect("ws payload");
    match outcome {
        Err(msg) => {
            if let Some(e) = ws_registry(ctx).socks.get_mut(&id) {
                e.dead = true;
            }
            ws_registry(ctx).socks.remove(&id);
            Ok(vec![
                Value::from_string("error".into()),
                Value::from_string(msg),
            ])
        }
        Ok(ConnectedSocket {
            stream,
            cancel,
            mask_rng,
            protocol,
        }) => {
            let (read_stream, writer): (Box<dyn WsRead>, SharedStream) = match stream {
                WsTransport::Plain(socket) => {
                    let reader = match socket.try_clone() {
                        Ok(reader) => reader,
                        Err(error) => {
                            ws_registry(ctx).socks.remove(&id);
                            return Ok(vec![
                                Value::from_string("error".into()),
                                Value::from_string(format!("clone WebSocket read half: {error}")),
                            ]);
                        }
                    };
                    let writer = SharedStream::new(Box::new(socket), None);
                    (Box::new(reader), writer)
                }
                WsTransport::Tls(socket) => {
                    let writer = SharedStream::new(socket, Some(Arc::clone(&cancel)));
                    (Box::new(writer.clone()), writer)
                }
            };
            {
                let reg = ws_registry(ctx);
                let Some(entry) = reg.socks.get_mut(&id) else {
                    return Ok(vec![Value::from_string("close".into()), Value::Num(1006.0)]);
                };
                if entry.dead {
                    reg.socks.remove(&id);
                    return Ok(vec![
                        Value::from_string("error".into()),
                        Value::from_string("WebSocket connection was cancelled".into()),
                    ]);
                }
                entry.writer = Some(writer.clone());
                *entry
                    .cancel
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cancel);
            }
            let reader = WsReader {
                stream: BufReader::new(read_stream),
                writer,
                mask_rng: Some(mask_rng),
                masked: true,
            };
            arm_read(ctx, id, reader);
            Ok(vec![
                Value::from_string("open".into()),
                Value::from_string(protocol),
            ])
        }
    }
}

fn arm_read(ctx: &mut Ctx, id: u64, mut reader: WsReader) {
    let dispatch = match ws_registry(ctx).socks.get(&id) {
        Some(e) if !e.dead => e.dispatch.clone(),
        _ => return,
    };
    let task = ctx
        .host_mut::<TaskRegistry>()
        .expect("runtime installs the registry")
        .register(dispatch, None, decode_read);
    let spawn = ctx
        .op_state()
        .get::<CompletionSender>()
        .expect("runtime installs the dedicated executor")
        .clone();
    spawn.run_blocking(task, move || {
        let outcome = reader.read_message();
        let keep = outcome.is_ok() && !matches!(outcome, Ok(WsEvent::Close(..)));
        Box::new(ReadResult {
            id,
            outcome,
            reader: keep.then_some(reader),
        })
    });
}

fn decode_read(ctx: &mut Ctx, payload: Box<dyn std::any::Any + Send>) -> Result<Vec<Value>, Value> {
    let ReadResult {
        id,
        outcome,
        reader,
    } = *payload.downcast::<ReadResult>().expect("ws payload");
    match outcome {
        Ok(WsEvent::Text(s)) => {
            if let Some(r) = reader {
                arm_read(ctx, id, r);
            }
            Ok(vec![
                Value::from_string("text".into()),
                Value::from_string(s),
            ])
        }
        Ok(WsEvent::Binary(b)) => {
            let arr = ctx.make_uint8array(&b)?;
            if let Some(r) = reader {
                arm_read(ctx, id, r);
            }
            Ok(vec![Value::from_string("binary".into()), arr])
        }
        Ok(WsEvent::Close(code, reason)) => {
            // Echo the close (once) so the TCP close handshake completes cleanly (§5.5.1),
            // then tear down.
            let entry = ws_registry(ctx).socks.remove(&id);
            let mut clean = false;
            if let Some(e) = entry {
                if let (Some(w), false) = (&e.writer, e.close_sent.swap(true, Ordering::SeqCst)) {
                    let payload = if code == 1005 {
                        Vec::new()
                    } else {
                        close_payload(code, &reason)
                    };
                    if let Ok(frame) = encode_endpoint_frame(ctx, e.masked, 0x8, &payload) {
                        let mut writer = w.clone();
                        clean = writer.write_all(&frame).is_ok();
                    }
                } else {
                    // We already sent our half of the closing handshake and just received theirs.
                    clean = true;
                }
            }
            Ok(vec![
                Value::from_string("close".into()),
                Value::Num(code as f64),
                Value::from_string(reason),
                Value::Bool(clean),
            ])
        }
        Err(WsError::Protocol(code, msg)) => {
            let entry = ws_registry(ctx).socks.remove(&id);
            if let Some(e) = entry {
                if let (Some(w), false) = (&e.writer, e.close_sent.swap(true, Ordering::SeqCst)) {
                    let payload = close_payload(code, msg);
                    if let Ok(frame) = encode_endpoint_frame(ctx, e.masked, 0x8, &payload) {
                        let mut writer = w.clone();
                        let _ = writer.write_all(&frame);
                    }
                }
            }
            Ok(vec![
                Value::from_string("fail".into()),
                Value::Num(code as f64),
                Value::from_string(msg.to_string()),
            ])
        }
        Err(WsError::Io(msg)) => {
            ws_registry(ctx).socks.remove(&id);
            Ok(vec![
                Value::from_string("io".into()),
                Value::from_string(msg),
            ])
        }
    }
}

/// `__ws.send(id, stringOrBytes)` — encodes and writes one masked data frame on the loop thread.
pub(crate) fn op_ws_send(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let id = match args.first() {
        Some(Value::Num(n)) => *n as u64,
        _ => return Err(ctx.make_error("TypeError", "send: bad socket id")),
    };
    let (opcode, bytes) = match args.get(1) {
        Some(v) => match ctx.typed_array_bytes(v) {
            Some(b) => (0x2u8, b),
            None => (0x1u8, ctx.coerce_string(v)?.to_string().into_bytes()),
        },
        None => return Err(ctx.make_error("TypeError", "send: missing data")),
    };
    let (writer, masked) = {
        let reg = ws_registry(ctx);
        let Some(e) = reg.socks.get(&id) else {
            return Ok(Value::Bool(false)); // already closed: spec drops silently
        };
        if e.close_sent.load(Ordering::SeqCst) {
            return Ok(Value::Bool(false));
        }
        match &e.writer {
            Some(w) => (w.clone(), e.masked),
            None => return Err(ctx.make_error("Error", "send before open")),
        }
    };
    let frame = encode_endpoint_frame(ctx, masked, opcode, &bytes)?;
    let mut writer = writer;
    writer
        .write_all(&frame)
        .map_err(|e| ctx.make_error("Error", format!("WebSocket send: {e}")))?;
    Ok(Value::Bool(true))
}

/// `__ws.close(id, code, reason)` — sends the close frame (once); the read loop then surfaces
/// the peer's echo as the close event.
pub(crate) fn op_ws_close(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let id = match args.first() {
        Some(Value::Num(n)) => *n as u64,
        _ => return Err(ctx.make_error("TypeError", "close: bad socket id")),
    };
    let code = match args.get(1) {
        Some(Value::Num(n))
            if n.is_finite()
                && n.fract() == 0.0
                && (*n == 1000.0 || (3000.0..=4999.0).contains(n)) =>
        {
            Some(*n as u16)
        }
        Some(Value::Num(_)) => {
            return Err(ctx.make_error("TypeError", "close: code must be 1000 or in 3000-4999"))
        }
        _ => None,
    };
    let reason = ctx
        .coerce_string(args.get(2).unwrap_or(&Value::Undefined))?
        .to_string();
    if reason.len() > 123 {
        return Err(ctx.make_error("TypeError", "close: UTF-8 reason exceeds 123 bytes"));
    }
    if code.is_none() && !reason.is_empty() {
        return Err(ctx.make_error("TypeError", "close: reason requires a status code"));
    }
    let (writer, masked) = {
        let reg = ws_registry(ctx);
        let Some(e) = reg.socks.get_mut(&id) else {
            return Ok(Value::Undefined);
        };
        if e.close_sent.swap(true, Ordering::SeqCst) {
            return Ok(Value::Undefined);
        }
        if e.writer.is_none() {
            // The WebSocket Standard's close() steps fail a connection that is still CONNECTING.
            // The handshake task observes this before publishing its transport.
            e.dead = true;
            e.cancelled.store(true, Ordering::Release);
            if let Some(socket) = e
                .cancel
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
            {
                let _ = socket.shutdown(Shutdown::Both);
            }
        }
        (e.writer.clone(), e.masked)
    };
    if let Some(mut w) = writer {
        let payload = code.map_or_else(Vec::new, |code| close_payload(code, &reason));
        let frame = encode_endpoint_frame(ctx, masked, 0x8, &payload)?;
        let _ = w.write_all(&frame);
    }
    Ok(Value::Undefined)
}

/// `__ws.upgrade(connId, secWebSocketKey, protocol, extraHeaderPairs, dispatch)` → id.
///
/// Adopts a connection accepted by `TRustJS.serve` (see server.rs: the parsed request's `TcpStream`
/// sits in the resource table under `connId`) as a SERVER-side WebSocket: writes the RFC 6455
/// 101 handshake response, then joins the same registry/read-loop machinery the client uses —
/// with `masked: false`, since a server must not mask (§5.1). `dispatch(kind, ...)` receives
/// `("text", string)`, `("binary", u8array)`, `("close", code, reason, wasClean)`,
/// `("fail", code, msg)` (protocol violation), and `("io", msg)` (socket died). There is no
/// "open" event: the connection is open the moment this op returns.
pub(crate) fn op_ws_upgrade(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let conn_id = match args.first() {
        Some(Value::Num(n)) if *n >= 0.0 => *n as u32,
        _ => return Err(ctx.make_error("TypeError", "upgrade: bad connection id")),
    };
    let key = ctx
        .coerce_string(args.get(1).unwrap_or(&Value::Undefined))?
        .to_string();
    let protocol = ctx
        .coerce_string(args.get(2).unwrap_or(&Value::Undefined))?
        .to_string();
    let extra_headers = crate::read_header_pairs(ctx, args.get(3).unwrap_or(&Value::Undefined))?;
    if !protocol.is_empty() && !crate::http_syntax::is_token(&protocol) {
        return Err(ctx.make_error("TypeError", "upgrade: protocol must be an HTTP token"));
    }
    let dispatch = match args.get(4) {
        Some(v) if v.is_callable() => v.clone(),
        _ => return Err(ctx.make_error("TypeError", "upgrade: dispatch must be a function")),
    };
    if decode_base64(&key).is_none_or(|decoded| decoded.len() != 16) {
        return Err(ctx.make_error(
            "TypeError",
            "upgrade: Sec-WebSocket-Key must encode exactly 16 bytes",
        ));
    }

    // Take the socket out of the resource table (same handoff as respond(); a later respond on
    // this connection now correctly fails as "already answered").
    let stream = ctx
        .resource_table()
        .close(conn_id)
        .and_then(|rc| rc.downcast::<TcpStream>().ok())
        .and_then(|rc| std::rc::Rc::try_unwrap(rc).ok());
    let Some(stream) = stream else {
        return Err(ctx.make_error(
            "TypeError",
            "upgrade: unknown or already-answered connection",
        ));
    };

    // The accept loop set a read timeout to bound header parsing; a WebSocket idles legitimately.
    stream.set_read_timeout(None).ok();
    stream.set_write_timeout(Some(WRITE_TIMEOUT)).ok();
    stream.set_nodelay(true).ok();

    // Write the 101 upgrade response. It is small; writing on the loop thread matches how sends
    // and closes are written.
    let mut resp = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n",
        websocket_accept(&key)
    );
    if !protocol.is_empty() {
        resp.push_str(&format!("Sec-WebSocket-Protocol: {protocol}\r\n"));
    }
    for (name, value) in &extra_headers {
        // The handshake-critical headers above must not be overridden by user extras.
        if name.eq_ignore_ascii_case("upgrade")
            || name.eq_ignore_ascii_case("connection")
            || name.eq_ignore_ascii_case("sec-websocket-accept")
            || name.eq_ignore_ascii_case("sec-websocket-protocol")
            || name.eq_ignore_ascii_case("sec-websocket-extensions")
        {
            continue;
        }
        resp.push_str(&format!("{name}: {value}\r\n"));
    }
    resp.push_str("\r\n");
    (&stream)
        .write_all(resp.as_bytes())
        .map_err(|e| ctx.make_error("Error", format!("WebSocket upgrade: handshake write: {e}")))?;

    // `TcpStream::try_clone` gives the reader and writer independent handles to one socket. A
    // blocking idle read therefore never owns the write mutex, and shutting down `cancel` wakes
    // that read immediately without a timeout/poll loop.
    let read_stream = stream
        .try_clone()
        .map_err(|error| ctx.make_error("Error", format!("WebSocket read half: {error}")))?;
    let cancel =
        Arc::new(stream.try_clone().map_err(|error| {
            ctx.make_error("Error", format!("WebSocket cancel handle: {error}"))
        })?);
    let writer = SharedStream::new(Box::new(stream), None);

    let id = {
        let reg = ws_registry(ctx);
        let id = reg.next;
        reg.next += 1;
        reg.socks.insert(
            id,
            WsEntry {
                writer: Some(writer.clone()),
                close_sent: Arc::new(AtomicBool::new(false)),
                dead: false,
                dispatch,
                cancel: Arc::new(Mutex::new(Some(cancel))),
                cancelled: Arc::new(AtomicBool::new(false)),
                masked: false,
            },
        );
        id
    };

    let reader = WsReader {
        stream: BufReader::new(Box::new(read_stream)),
        writer,
        mask_rng: None,
        masked: false,
    };
    arm_read(ctx, id, reader);
    Ok(Value::Num(id as f64))
}

// ---- test support -------------------------------------------------------------------------------

/// A minimal RFC 6455 echo server for this workspace's tests and benchmarks (NOT part of the
/// runtime): accepts one connection at a time, upgrades, then echoes text/binary messages,
/// answers nothing to pongs, replies to close. `behavior` tweaks let tests exercise edges.
#[doc(hidden)]
#[allow(dead_code)]
pub mod testing {
    use super::*;
    use std::net::TcpListener;

    /// What the echo server should do beyond plain echoing.
    #[derive(Clone, Copy, PartialEq)]
    pub enum Mode {
        /// Echo every message until the client closes.
        Echo,
        /// Send a ping (expecting the client's transparent pong), then echo.
        PingThenEcho,
        /// Send one fragmented text message ("frag" in 3 parts), then echo.
        FragmentedHello,
        /// Immediately close with (4001, "going away").
        CloseImmediately,
        /// Answer the upgrade with a WRONG Sec-WebSocket-Accept.
        BadAccept,
    }

    /// Spawn the server; returns its port. It serves `conns` connections then exits.
    pub fn spawn_echo(mode: Mode, conns: usize) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for _ in 0..conns {
                let Ok((stream, _)) = listener.accept() else {
                    return;
                };
                let _ = serve_one(stream, mode);
            }
        });
        port
    }

    fn serve_one(stream: TcpStream, mode: Mode) -> std::io::Result<()> {
        // Upgrade.
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            (&stream).read_exact(&mut byte)?;
            head.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&head);
        let key = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case("sec-websocket-key")
                    .then(|| v.trim().to_string())
            })
            .unwrap_or_default();
        let protocol = head.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case("sec-websocket-protocol")
                .then(|| v.split(',').next().unwrap_or("").trim().to_string())
        });
        let accept = if mode == Mode::BadAccept {
            "AAAAAAAAAAAAAAAAAAAAAAAAAAA=".to_string()
        } else {
            websocket_accept(&key)
        };
        let mut resp = format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Accept: {accept}\r\n"
        );
        if let Some(p) = protocol.filter(|p| !p.is_empty()) {
            resp.push_str(&format!("Sec-WebSocket-Protocol: {p}\r\n"));
        }
        resp.push_str("\r\n");
        (&stream).write_all(resp.as_bytes())?;

        let unmasked = |op: u8, fin: bool, payload: &[u8]| {
            let mut f = vec![if fin { 0x80 | op } else { op }];
            if payload.len() < 126 {
                f.push(payload.len() as u8);
            } else {
                f.push(126);
                f.extend_from_slice(&(payload.len() as u16).to_be_bytes());
            }
            f.extend_from_slice(payload);
            f
        };

        match mode {
            Mode::CloseImmediately => {
                let mut p = 4001u16.to_be_bytes().to_vec();
                p.extend_from_slice(b"going away");
                (&stream).write_all(&unmasked(0x8, true, &p))?;
            }
            Mode::PingThenEcho => {
                (&stream).write_all(&unmasked(0x9, true, b"marco"))?;
            }
            Mode::FragmentedHello => {
                (&stream).write_all(&unmasked(0x1, false, b"fr"))?;
                (&stream).write_all(&unmasked(0x0, false, b"agm"))?;
                (&stream).write_all(&unmasked(0x0, true, b"ent"))?;
            }
            _ => {}
        }

        // Echo loop over a buffered reader (frames from the client are masked).
        let mut r = BufReader::new(stream.try_clone()?);
        loop {
            let frame = match read_raw_frame(&mut r, MAX_MESSAGE, true) {
                Ok(f) => f,
                Err(_) => return Ok(()),
            };
            match frame.opcode {
                0x8 => {
                    (&stream).write_all(&unmasked(0x8, true, &frame.payload))?;
                    return Ok(());
                }
                0x9 => (&stream).write_all(&unmasked(0xA, true, &frame.payload))?,
                0xA => {
                    // A pong: the PingThenEcho handshake completed — tell the client via text.
                    (&stream).write_all(&unmasked(
                        0x1,
                        true,
                        format!("pong:{}", String::from_utf8_lossy(&frame.payload)).as_bytes(),
                    ))?;
                }
                0x1 | 0x2 if frame.fin => {
                    (&stream).write_all(&unmasked(frame.opcode, true, &frame.payload))?;
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    #[test]
    fn frame_roundtrip() {
        // Client-masked frames decode back to the payload (server view).
        for (op, payload) in [
            (0x1u8, b"hello".to_vec()),
            (0x2, vec![0u8, 1, 254, 255]),
            (0x1, vec![b'x'; 200]),    // 16-bit length form
            (0x1, vec![b'y'; 70_000]), // 64-bit length form
        ] {
            let frame = encode_frame(op, &payload, [1, 2, 3, 4]);
            let raw = read_raw_frame(&mut &frame[..], MAX_MESSAGE, true)
                .ok()
                .unwrap();
            assert!(raw.fin);
            assert_eq!(raw.opcode, op);
            assert_eq!(raw.payload, payload);
        }
    }

    #[test]
    fn accept_value_matches_rfc_example() {
        assert_eq!(
            websocket_accept("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn control_frame_rules() {
        // A fragmented (FIN=0) ping is a protocol error.
        let mut bad = encode_frame(0x9, b"p", [0, 0, 0, 0]);
        bad[0] &= 0x7f; // clear FIN
        assert!(matches!(
            read_raw_frame(&mut &bad[..], MAX_MESSAGE, true),
            Err(WsError::Protocol(1002, _))
        ));
        // Reserved bits fail (no extensions negotiated).
        let mut rsv = encode_frame(0x1, b"x", [0, 0, 0, 0]);
        rsv[0] |= 0x40;
        assert!(matches!(
            read_raw_frame(&mut &rsv[..], MAX_MESSAGE, true),
            Err(WsError::Protocol(1002, _))
        ));
        // Reserved opcodes and an extended control-frame length fail from the two-byte header;
        // the decoder must not wait for attacker-controlled payload bytes first.
        for header in [[0x83, 0x80], [0x8b, 0x80], [0x89, 0xfe]] {
            assert!(matches!(
                read_raw_frame(&mut &header[..], MAX_MESSAGE, true),
                Err(WsError::Protocol(1002, _))
            ));
        }
    }

    #[test]
    fn masking_direction_is_mandatory() {
        let masked = encode_frame(0x1, b"client", [1, 2, 3, 4]);
        assert!(matches!(
            read_raw_frame(&mut &masked[..], MAX_MESSAGE, false),
            Err(WsError::Protocol(1002, _))
        ));

        let unmasked = encode_frame_unmasked(0x1, b"server");
        assert!(matches!(
            read_raw_frame(&mut &unmasked[..], MAX_MESSAGE, true),
            Err(WsError::Protocol(1002, _))
        ));
    }

    #[test]
    fn payload_lengths_must_be_canonical_and_63_bit() {
        let nonminimal_16 = [0x81, 0xfe, 0, 125];
        assert!(matches!(
            read_raw_frame(&mut &nonminimal_16[..], MAX_MESSAGE, true),
            Err(WsError::Protocol(1002, _))
        ));

        let mut nonminimal_64 = vec![0x81, 0xff];
        nonminimal_64.extend_from_slice(&65_535u64.to_be_bytes());
        assert!(matches!(
            read_raw_frame(&mut &nonminimal_64[..], MAX_MESSAGE, true),
            Err(WsError::Protocol(1002, _))
        ));

        let mut high_bit = vec![0x81, 0xff];
        high_bit.extend_from_slice(&(1u64 << 63).to_be_bytes());
        assert!(matches!(
            read_raw_frame(&mut &high_bit[..], MAX_MESSAGE, true),
            Err(WsError::Protocol(1002, _))
        ));
    }

    #[test]
    fn streaming_utf8_validator_covers_scalar_boundaries() {
        let valid = "\0\u{80}\u{7ff}\u{800}\u{d7ff}\u{e000}\u{ffff}\u{10000}\u{10ffff}";
        let mut validator = Utf8Validator::default();
        for byte in valid.as_bytes() {
            validator.push(std::slice::from_ref(byte)).unwrap();
        }
        validator.finish().unwrap();

        for invalid in [
            &[0x80][..],
            &[0xc0, 0x80],
            &[0xe0, 0x9f, 0xbf],
            &[0xed, 0xa0, 0x80],
            &[0xf0, 0x8f, 0xbf, 0xbf],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0xf5, 0x80, 0x80, 0x80],
        ] {
            assert!(matches!(
                Utf8Validator::default().push(invalid),
                Err(WsError::Protocol(1007, _))
            ));
        }

        let mut incomplete = Utf8Validator::default();
        incomplete.push(&[0xe2, 0x82]).unwrap();
        assert!(matches!(
            incomplete.finish(),
            Err(WsError::Protocol(1007, _))
        ));
    }

    #[test]
    fn text_scalar_may_cross_fragment_boundaries() {
        let mut first = encode_frame(0x1, &[0xf0, 0x9f], [1, 2, 3, 4]);
        first[0] &= !0x80;
        let second = encode_frame(0x0, &[0x92, 0xa9], [5, 6, 7, 8]);
        first.extend_from_slice(&second);
        assert!(matches!(read_as_server(first), Ok(WsEvent::Text(text)) if text == "💩"));

        let mut incomplete = encode_frame(0x1, &[0xf0, 0x9f], [1, 2, 3, 4]);
        incomplete[0] &= !0x80;
        incomplete.extend_from_slice(&encode_frame(0x0, &[0x92], [5, 6, 7, 8]));
        assert!(matches!(
            read_as_server(incomplete),
            Err(WsError::Protocol(1007, _))
        ));
    }

    #[test]
    fn invalid_text_prefix_fails_before_the_rest_of_its_frame_arrives() {
        struct PrefixThenError {
            bytes: Cursor<Vec<u8>>,
        }

        impl Read for PrefixThenError {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if self.bytes.position() < self.bytes.get_ref().len() as u64 {
                    self.bytes.read(output)
                } else {
                    Err(std::io::Error::other(
                        "the rest of the frame has not arrived",
                    ))
                }
            }
        }

        // Declare a 126-byte masked text frame but provide only the first, already-invalid byte.
        // A whole-frame read would report I/O; streaming RFC 6455 §8.1 validation reports 1007.
        let mut wire = vec![0x81, 0xfe, 0, 126, 0, 0, 0, 0];
        wire.push(0xc0);
        let mut source = PrefixThenError {
            bytes: Cursor::new(wire),
        };
        let mut validator = Utf8Validator::default();
        assert!(matches!(
            read_raw_frame_with_utf8(&mut source, MAX_MESSAGE, true, None, Some(&mut validator)),
            Err(WsError::Protocol(1007, _))
        ));
    }

    fn read_as_server(frame: Vec<u8>) -> Result<WsEvent, WsError> {
        let writer = SharedStream::new(Box::new(Cursor::new(Vec::new())), None);
        WsReader {
            stream: BufReader::new(Box::new(Cursor::new(frame))),
            writer,
            mask_rng: None,
            masked: false,
        }
        .read_message()
    }

    #[test]
    fn close_payload_and_status_codes_are_validated() {
        let one_byte = encode_frame(0x8, &[3], [1, 2, 3, 4]);
        assert!(matches!(
            read_as_server(one_byte),
            Err(WsError::Protocol(1002, _))
        ));

        for code in [999u16, 1004, 1005, 1006, 1015, 1016, 2999, 5000] {
            let frame = encode_frame(0x8, &code.to_be_bytes(), [1, 2, 3, 4]);
            assert!(matches!(
                read_as_server(frame),
                Err(WsError::Protocol(1002, _))
            ));
        }
        for code in [1000u16, 1011, 1012, 1014, 3000, 4999] {
            let frame = encode_frame(0x8, &code.to_be_bytes(), [1, 2, 3, 4]);
            assert!(matches!(read_as_server(frame), Ok(WsEvent::Close(c, _)) if c == code));
        }
    }

    #[test]
    fn handshake_requires_every_normative_response_field() {
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let valid = format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: WebSocket\r\n\
             Connection: keep-alive, Upgrade\r\nSec-WebSocket-Accept: {}\r\n\
             Sec-WebSocket-Protocol: chat\r\n\r\n",
            websocket_accept(key)
        );
        assert_eq!(
            validate_handshake_response(valid.as_bytes(), key, "chat, superchat").unwrap(),
            "chat"
        );
        assert!(validate_handshake_response(
            valid
                .replace("Connection: keep-alive, Upgrade\r\n", "")
                .as_bytes(),
            key,
            "chat, superchat"
        )
        .is_err());
        assert!(validate_handshake_response(
            valid
                .replace("Sec-WebSocket-Protocol: chat\r\n", "")
                .as_bytes(),
            key,
            "chat"
        )
        .is_err());
        assert!(validate_handshake_response(
            valid.replace("chat\r\n\r\n", "Chat\r\n\r\n").as_bytes(),
            key,
            "chat"
        )
        .is_err());
        assert!(validate_handshake_response(
            valid
                .replace("\r\n\r\n", "\r\nSec-WebSocket-Extensions: unknown\r\n\r\n")
                .as_bytes(),
            key,
            "chat"
        )
        .is_err());
    }

    #[test]
    fn server_upgrade_key_is_exactly_sixteen_random_bytes() {
        assert_eq!(decode_base64("AQIDBAUGBwgJCgsMDQ4PEA==").unwrap().len(), 16);
        assert!(decode_base64("AQIDBAUGBwgJCgsMDQ4PEB==").is_none()); // non-zero pad bits
        assert_ne!(decode_base64("aGVsbG8=").unwrap().len(), 16);
        assert!(decode_base64("not base64").is_none());
    }

    #[test]
    fn idle_tcp_reader_does_not_own_the_writer() {
        struct SignalingRead {
            stream: TcpStream,
            started: Option<mpsc::Sender<()>>,
        }
        impl Read for SignalingRead {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if let Some(started) = self.started.take() {
                    let _ = started.send(());
                }
                self.stream.read(buffer)
            }
        }

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        let cancel = socket.try_clone().unwrap();
        let reader_stream = socket.try_clone().unwrap();
        let writer = SharedStream::new(Box::new(socket), None);
        let (started_tx, started_rx) = mpsc::channel();
        let reader_writer = writer.clone();
        let join = std::thread::spawn(move || {
            let mut reader = WsReader {
                stream: BufReader::new(Box::new(SignalingRead {
                    stream: reader_stream,
                    started: Some(started_tx),
                })),
                writer: reader_writer,
                mask_rng: None,
                masked: false,
            };
            reader.read_message()
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let mut outgoing = writer;
        outgoing
            .write_all(&encode_frame_unmasked(0x1, b"not blocked"))
            .unwrap();
        let frame = read_raw_frame(&mut peer, MAX_MESSAGE, false).unwrap();
        assert_eq!(frame.payload, b"not blocked");

        cancel.shutdown(Shutdown::Both).unwrap();
        assert!(matches!(join.join().unwrap(), Err(WsError::Io(_))));
    }

    #[test]
    fn tls_retry_waits_for_readiness_without_holding_the_ssl_lock() {
        struct RetryStream {
            reads: Arc<AtomicUsize>,
            writes: Arc<AtomicUsize>,
        }
        impl Read for RetryStream {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.reads.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::WouldBlock,
                        "TLS wants socket read",
                    ))
                } else {
                    buffer[0] = b'x';
                    Ok(1)
                }
            }
        }
        impl Write for RetryStream {
            fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
                self.writes.fetch_add(1, Ordering::SeqCst);
                Ok(buffer.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (readiness, _) = listener.accept().unwrap();
        let reads = Arc::new(AtomicUsize::new(0));
        let writes = Arc::new(AtomicUsize::new(0));
        let shared = SharedStream::new(
            Box::new(RetryStream {
                reads: Arc::clone(&reads),
                writes: Arc::clone(&writes),
            }),
            Some(Arc::new(readiness)),
        );
        let mut reader = shared.clone();
        let join = std::thread::spawn(move || {
            let mut byte = [0];
            reader.read_exact(&mut byte).map(|()| byte[0])
        });

        while reads.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(reads.load(Ordering::SeqCst), 1, "idle TLS read spun");
        let mut writer = shared;
        writer.write_all(b"outbound").unwrap();
        assert_eq!(writes.load(Ordering::SeqCst), 1, "reader retained SSL lock");

        peer.write_all(b"wake").unwrap();
        assert_eq!(join.join().unwrap().unwrap(), b'x');
        assert_eq!(reads.load(Ordering::SeqCst), 2);
    }
}
