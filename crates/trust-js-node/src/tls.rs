use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use trust_js_host::{ops, CallbackQueue, CompletionSender, Ctx, OpDecl, TaskRegistry, Value};

pub const TLS_OPS: &[OpDecl] = ops![
    "connect" (7) => op_connect,
    "upgrade" (6) => op_upgrade,
    "read" (3) => op_read,
    "write" (4) => op_write,
    "close" (1) => op_close,
    "listen" (5) => op_listen,
    "accept" (3) => op_accept,
    "closeServer" (1) => op_close_server,
];

fn op_upgrade<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let socket_id = args.first().and_then(Value::as_num_opt).unwrap_or(0.0) as u64;
    let servername = ctx
        .coerce_string(args.get(1).unwrap_or(&Value::Undefined))?
        .to_string();
    let alpn: Vec<String> = ctx
        .coerce_string(args.get(2).unwrap_or(&Value::Undefined))?
        .split(',')
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect();
    let verify_peer = !matches!(args.get(3), Some(Value::Bool(false)));
    let (resolve, reject) = callbacks(ctx, args.get(4), args.get(5))?;
    let tcp = crate::net::take_stream(ctx, socket_id)?;
    let local = tcp
        .local_addr()
        .map_err(|error| ctx.make_error("Error", error.to_string()))?;
    let peer = tcp
        .peer_addr()
        .map_err(|error| ctx.make_error("Error", error.to_string()))?;
    tcp.set_read_timeout(Some(Duration::from_millis(100))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(30))).ok();
    let task = TaskRegistry::register_callbacks(ctx, &resolve, Some(&reject), decode_connect)?;
    completions(ctx).run_blocking(task, move || {
        let result =
            trust_js_tls::TlsStream::connect_with_options(tcp, &servername, &alpn, verify_peer)
                .map(|stream| {
                    let protocol = stream.protocol();
                    let cipher = stream.cipher();
                    let negotiated_alpn = stream.alpn_protocol();
                    Connected {
                        stream,
                        local,
                        peer,
                        protocol,
                        cipher,
                        alpn: negotiated_alpn,
                    }
                });
        Box::new(result)
    });
    Ok(Value::Undefined)
}

struct Entry {
    stream: Arc<Mutex<trust_js_tls::TlsStream>>,
    closed: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct TlsRegistry {
    next: u64,
    sockets: std::collections::HashMap<u64, Entry>,
    next_listener: u64,
    listeners: std::collections::HashMap<u64, ListenerEntry>,
}

struct ListenerEntry {
    listener: Arc<TcpListener>,
    closed: Arc<AtomicBool>,
    local: SocketAddr,
    context: trust_js_tls::ServerContext,
}

impl Drop for TlsRegistry {
    fn drop(&mut self) {
        for entry in self.sockets.values() {
            entry.closed.store(true, Ordering::Release);
        }
        for entry in self.listeners.values() {
            entry.closed.store(true, Ordering::Release);
            wake_listener(entry.local);
        }
    }
}

fn wake_listener(local: SocketAddr) {
    let address = if local.ip().is_unspecified() {
        SocketAddr::new(
            if local.is_ipv6() {
                IpAddr::V6(Ipv6Addr::LOCALHOST)
            } else {
                IpAddr::V4(Ipv4Addr::LOCALHOST)
            },
            local.port(),
        )
    } else {
        local
    };
    let _ = TcpStream::connect_timeout(&address, Duration::from_millis(100));
}

struct Connected {
    stream: trust_js_tls::TlsStream,
    local: SocketAddr,
    peer: SocketAddr,
    protocol: String,
    cipher: String,
    alpn: String,
}

fn callbacks<'s>(
    ctx: &mut Ctx<'s>,
    resolve: Option<&Value<'s>>,
    reject: Option<&Value<'s>>,
) -> Result<(Value<'s>, Value<'s>), Value<'s>> {
    match (resolve, reject) {
        (Some(resolve), Some(reject)) if resolve.is_callable() && reject.is_callable() => {
            Ok((*resolve, *reject))
        }
        _ => Err(ctx.make_error("TypeError", "TLS op expects resolve and reject callbacks")),
    }
}

fn completions<'s>(ctx: &mut Ctx<'s>) -> CompletionSender {
    ctx.op_state()
        .get::<CompletionSender>()
        .expect("runtime installs completion sender")
        .clone()
}

fn op_connect<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let host = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let port = args.get(1).and_then(Value::as_num_opt).unwrap_or(443.0) as u16;
    let servername = ctx
        .coerce_string(args.get(2).unwrap_or(&Value::Undefined))?
        .to_string();
    let alpn: Vec<String> = ctx
        .coerce_string(args.get(3).unwrap_or(&Value::Undefined))?
        .split(',')
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect();
    let verify_peer = !matches!(args.get(4), Some(Value::Bool(false)));
    let (resolve, reject) = callbacks(ctx, args.get(5), args.get(6))?;
    let task = TaskRegistry::register_callbacks(ctx, &resolve, Some(&reject), decode_connect)?;
    completions(ctx).run_blocking(task, move || {
        let result = (|| -> Result<Connected, String> {
            let addresses: Vec<_> = (host.as_str(), port)
                .to_socket_addrs()
                .map_err(|error| format!("getaddrinfo {host}: {error}"))?
                .collect();
            let mut last_error = None;
            for address in addresses {
                match TcpStream::connect_timeout(&address, Duration::from_secs(30)) {
                    Ok(tcp) => {
                        tcp.set_read_timeout(Some(Duration::from_millis(100))).ok();
                        tcp.set_write_timeout(Some(Duration::from_secs(30))).ok();
                        let local = tcp.local_addr().map_err(|error| error.to_string())?;
                        let peer = tcp.peer_addr().map_err(|error| error.to_string())?;
                        let stream = trust_js_tls::TlsStream::connect_with_options(
                            tcp,
                            &servername,
                            &alpn,
                            verify_peer,
                        )?;
                        let protocol = stream.protocol();
                        let cipher = stream.cipher();
                        let alpn = stream.alpn_protocol();
                        return Ok(Connected {
                            stream,
                            local,
                            peer,
                            protocol,
                            cipher,
                            alpn,
                        });
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            Err(format!(
                "connect {host}:{port}: {}",
                last_error.map_or_else(|| "no address".into(), |error| error.to_string())
            ))
        })();
        Box::new(result)
    });
    Ok(Value::Undefined)
}

fn decode_connect<'s>(
    ctx: &mut Ctx<'s>,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value<'s>>, Value<'s>> {
    let connected = *payload
        .downcast::<Result<Connected, String>>()
        .expect("TLS connect payload");
    register_connected(
        ctx,
        connected.map_err(|message| ctx.make_error("Error", message))?,
    )
}

fn register_connected<'s>(
    ctx: &mut Ctx<'s>,
    connected: Connected,
) -> Result<Vec<Value<'s>>, Value<'s>> {
    let Connected {
        stream,
        local,
        peer,
        protocol,
        cipher,
        alpn,
    } = connected;
    let registry = ctx.host_mut::<TlsRegistry>().expect("TLS registry");
    let id = registry.next;
    registry.next += 1;
    registry.sockets.insert(
        id,
        Entry {
            stream: Arc::new(Mutex::new(stream)),
            closed: Arc::new(AtomicBool::new(false)),
        },
    );
    Ok(vec![
        Value::Num(id as f64),
        ctx.string(local.ip().to_string()),
        Value::Num(local.port() as f64),
        ctx.string(peer.ip().to_string()),
        Value::Num(peer.port() as f64),
        ctx.string(protocol),
        ctx.string(cipher),
        ctx.string(alpn),
    ])
}

fn op_listen<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let host = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();
    let port = args.get(1).and_then(Value::as_num_opt).unwrap_or(0.0) as u16;
    let certificate = ctx
        .typed_array_bytes(args.get(2).unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "TLS server certificate must be bytes"))?;
    let private_key = ctx
        .typed_array_bytes(args.get(3).unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "TLS server private key must be bytes"))?;
    let alpn: Vec<String> = ctx
        .coerce_string(args.get(4).unwrap_or(&Value::Undefined))?
        .split(',')
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect();
    // One SSL_CTX owns the server certificate, ALPN callback, and session cache for the listener;
    // accepted sockets only allocate their per-connection SSL object.
    let context = trust_js_tls::ServerContext::new(&certificate, &private_key, &alpn)
        .map_err(|error| ctx.make_error("Error", format!("TLS listen: {error}")))?;
    let listener = TcpListener::bind((if host.is_empty() { "0.0.0.0" } else { &host }, port))
        .map_err(|error| ctx.make_error("Error", format!("TLS listen: {error}")))?;
    let local = listener
        .local_addr()
        .map_err(|error| ctx.make_error("Error", error.to_string()))?;
    let registry = ctx.host_mut::<TlsRegistry>().expect("TLS registry");
    let id = registry.next_listener;
    registry.next_listener += 1;
    registry.listeners.insert(
        id,
        ListenerEntry {
            listener: Arc::new(listener),
            closed: Arc::new(AtomicBool::new(false)),
            local,
            context,
        },
    );
    let result = Value::Obj(ctx.new_object());
    let _ = ctx.set_member(&result, "serverId", Value::Num(id as f64));
    let _ = ctx.set_member(&result, "address", ctx.string(local.ip().to_string()));
    let _ = ctx.set_member(&result, "port", Value::Num(local.port() as f64));
    Ok(result)
}

enum AcceptResult {
    Connected(Box<Connected>),
    Closed,
    Error(String),
}

fn op_accept<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let id = args.first().and_then(Value::as_num_opt).unwrap_or(0.0) as u64;
    let (resolve, reject) = callbacks(ctx, args.get(1), args.get(2))?;
    let found = ctx
        .host_mut::<TlsRegistry>()
        .and_then(|registry| registry.listeners.get(&id))
        .map(|entry| {
            (
                entry.listener.clone(),
                entry.closed.clone(),
                entry.context.clone(),
            )
        });
    let Some((listener, closed, context)) = found else {
        CallbackQueue::enqueue(ctx, &resolve, &[Value::Null])?;
        return Ok(Value::Undefined);
    };
    let task = TaskRegistry::register_callbacks(ctx, &resolve, Some(&reject), decode_accept)?;
    completions(ctx).run_blocking(task, move || {
        let result = match listener.accept() {
            Ok((tcp, peer)) if !closed.load(Ordering::SeqCst) => {
                let local = tcp
                    .local_addr()
                    .unwrap_or_else(|_| listener.local_addr().unwrap());
                tcp.set_read_timeout(Some(Duration::from_millis(100))).ok();
                match trust_js_tls::TlsStream::accept_with_context(tcp, &context) {
                    Ok(stream) => {
                        let protocol = stream.protocol();
                        let cipher = stream.cipher();
                        let alpn = stream.alpn_protocol();
                        AcceptResult::Connected(Box::new(Connected {
                            stream,
                            local,
                            peer,
                            protocol,
                            cipher,
                            alpn,
                        }))
                    }
                    Err(error) => AcceptResult::Error(error),
                }
            }
            _ => AcceptResult::Closed,
        };
        Box::new(result)
    });
    Ok(Value::Undefined)
}

fn decode_accept<'s>(
    ctx: &mut Ctx<'s>,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value<'s>>, Value<'s>> {
    match *payload
        .downcast::<AcceptResult>()
        .expect("TLS accept payload")
    {
        AcceptResult::Connected(connected) => register_connected(ctx, *connected),
        AcceptResult::Closed => Ok(vec![Value::Null]),
        AcceptResult::Error(message) => Err(ctx.make_error("Error", message)),
    }
}

fn op_close_server<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let id = args.first().and_then(Value::as_num_opt).unwrap_or(0.0) as u64;
    let entry = ctx
        .host_mut::<TlsRegistry>()
        .and_then(|registry| registry.listeners.remove(&id));
    if let Some(entry) = entry {
        entry.closed.store(true, Ordering::SeqCst);
        wake_listener(entry.local);
    }
    Ok(Value::Undefined)
}

fn op_read<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let id = args.first().and_then(Value::as_num_opt).unwrap_or(0.0) as u64;
    let (resolve, reject) = callbacks(ctx, args.get(1), args.get(2))?;
    let found = ctx
        .host_mut::<TlsRegistry>()
        .and_then(|registry| registry.sockets.get(&id))
        .map(|entry| (entry.stream.clone(), entry.closed.clone()));
    let Some((stream, closed)) = found else {
        CallbackQueue::enqueue(ctx, &resolve, &[Value::Null])?;
        return Ok(Value::Undefined);
    };
    let task = TaskRegistry::register_callbacks(ctx, &resolve, Some(&reject), decode_read)?;
    completions(ctx).run_blocking(task, move || {
        let result = loop {
            if closed.load(Ordering::SeqCst) {
                break Ok(Vec::new());
            }
            let mut bytes = vec![0u8; 65536];
            // Drop the stream lock before yielding on SSL_WANT_READ. Keeping the guard alive
            // through the match arm lets the next read immediately reacquire it and can starve a
            // concurrent write forever (for example, the first Redis command after TLS setup).
            let read = {
                let mut stream = stream.lock().unwrap();
                stream.read(&mut bytes)
            };
            match read {
                Ok(0) => break Ok(Vec::new()),
                Ok(length) => {
                    bytes.truncate(length);
                    break Ok(bytes);
                }
                // RFC 8446 §5 permits handshake/control records between application-data
                // records. OpenSSL consequently documents SSL_ERROR_WANT_READ/WANT_WRITE as
                // retry conditions even for a blocking BIO; trust-js-tls maps those to WouldBlock
                // and Interrupted. The socket timeout is only our cancellation poll interval.
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::Interrupted
                            | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(error) => break Err(error.to_string()),
            }
        };
        Box::new(result)
    });
    Ok(Value::Undefined)
}

fn decode_read<'s>(
    ctx: &mut Ctx<'s>,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value<'s>>, Value<'s>> {
    match *payload
        .downcast::<Result<Vec<u8>, String>>()
        .expect("TLS read payload")
    {
        Ok(bytes) if bytes.is_empty() => Ok(vec![Value::Null]),
        Ok(bytes) => Ok(vec![ctx.make_uint8array(&bytes)?]),
        Err(message) => Err(ctx.make_error("Error", message)),
    }
}

fn op_write<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let id = args.first().and_then(Value::as_num_opt).unwrap_or(0.0) as u64;
    let bytes = ctx
        .typed_array_bytes(args.get(1).unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "TLS write expects bytes"))?;
    let (resolve, reject) = callbacks(ctx, args.get(2), args.get(3))?;
    let stream = ctx
        .host_mut::<TlsRegistry>()
        .and_then(|registry| registry.sockets.get(&id))
        .map(|entry| entry.stream.clone());
    let Some(stream) = stream else {
        return Err(ctx.make_error("Error", "TLS socket is closed"));
    };
    let task = TaskRegistry::register_callbacks(ctx, &resolve, Some(&reject), decode_write)?;
    completions(ctx).run_blocking(task, move || {
        let result = stream
            .lock()
            .unwrap()
            .write_all(&bytes)
            .map(|()| bytes.len())
            .map_err(|error| error.to_string());
        Box::new(result)
    });
    Ok(Value::Undefined)
}

fn decode_write<'s>(
    ctx: &mut Ctx<'s>,
    payload: Box<dyn std::any::Any + Send>,
) -> Result<Vec<Value<'s>>, Value<'s>> {
    match *payload
        .downcast::<Result<usize, String>>()
        .expect("TLS write payload")
    {
        Ok(length) => Ok(vec![Value::Num(length as f64)]),
        Err(message) => Err(ctx.make_error("Error", message)),
    }
}

fn op_close<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let id = args.first().and_then(Value::as_num_opt).unwrap_or(0.0) as u64;
    if let Some(entry) = ctx
        .host_mut::<TlsRegistry>()
        .and_then(|registry| registry.sockets.remove(&id))
    {
        entry.closed.store(true, Ordering::SeqCst);
    }
    Ok(Value::Undefined)
}
