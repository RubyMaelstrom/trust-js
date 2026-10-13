//! A bounded, origin-keyed blocking HTTP/1.1 connection pool, used from the fetch threadpool.
//! RFC 9112 §9.3 permits reuse only after a self-delimited response has been consumed completely;
//! Fetch §2.6 additionally partitions connections by origin and credentials. HTTPS uses a matching
//! per-partition OpenSSL context so TLS session state cannot cross the credentials boundary.

use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::http_syntax::{
    self, body_framing, field_value_bytes, read_body, read_fields, read_line, single_header,
    MAX_HEADER_BYTES,
};
use crate::url;

pub(crate) struct HttpResponse {
    pub status: u16,
    pub status_text: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// Where the response actually came from (after redirects).
    pub url: String,
}

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_REDIRECTS: usize = 20;
const MAX_IDLE_CONNECTIONS: usize = 32;
const MAX_IDLE_PER_ORIGIN: usize = 2;
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

type Connection = BufReader<Box<dyn ReadWrite + Send>>;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct OriginKey {
    scheme: String,
    host: String,
    port: u16,
    credentials: bool,
}

struct IdleConnection {
    connection: Connection,
    since: Instant,
}

#[derive(Default)]
struct ConnectionPool {
    idle: HashMap<OriginKey, Vec<IdleConnection>>,
    count: usize,
}

impl ConnectionPool {
    fn take(&mut self, key: &OriginKey) -> Option<Connection> {
        self.purge_expired();
        let connection = self.idle.get_mut(key)?.pop()?.connection;
        self.count -= 1;
        if self.idle.get(key).is_some_and(Vec::is_empty) {
            self.idle.remove(key);
        }
        Some(connection)
    }

    fn put(&mut self, key: OriginKey, connection: Connection) {
        self.purge_expired();
        if self
            .idle
            .get(&key)
            .is_some_and(|connections| connections.len() >= MAX_IDLE_PER_ORIGIN)
        {
            return;
        }
        if self.count == MAX_IDLE_CONNECTIONS {
            self.evict_oldest();
        }
        self.idle.entry(key).or_default().push(IdleConnection {
            connection,
            since: Instant::now(),
        });
        self.count += 1;
    }

    fn purge_expired(&mut self) {
        let now = Instant::now();
        self.idle.retain(|_, connections| {
            connections.retain(|connection| now.duration_since(connection.since) < IDLE_TIMEOUT);
            !connections.is_empty()
        });
        self.count = self.idle.values().map(Vec::len).sum();
    }

    fn evict_oldest(&mut self) {
        let oldest = self
            .idle
            .iter()
            .flat_map(|(key, connections)| {
                connections
                    .iter()
                    .enumerate()
                    .map(move |(index, connection)| (key.clone(), index, connection.since))
            })
            .min_by_key(|(_, _, since)| *since)
            .map(|(key, index, _)| (key, index));
        if let Some((key, index)) = oldest {
            if let Some(connections) = self.idle.get_mut(&key) {
                connections.swap_remove(index);
                self.count -= 1;
                if connections.is_empty() {
                    self.idle.remove(&key);
                }
            }
        }
    }
}

struct HttpClientInner {
    pool: Mutex<ConnectionPool>,
    credentialless_tls: OnceLock<Result<trust_js_tls::ClientContext, String>>,
    credentialed_tls: OnceLock<Result<trust_js_tls::ClientContext, String>>,
}

/// One Fetch user-agent connection pool. Clones share idle transports and TLS session scopes.
#[derive(Clone)]
pub(crate) struct HttpClient(Arc<HttpClientInner>);

impl Default for HttpClient {
    fn default() -> Self {
        Self(Arc::new(HttpClientInner {
            pool: Mutex::new(ConnectionPool::default()),
            credentialless_tls: OnceLock::new(),
            credentialed_tls: OnceLock::new(),
        }))
    }
}

impl HttpClient {
    fn pool(&self) -> std::sync::MutexGuard<'_, ConnectionPool> {
        self.0
            .pool
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn tls_context(&self, credentials: bool) -> Result<trust_js_tls::ClientContext, String> {
        let slot = if credentials {
            &self.0.credentialed_tls
        } else {
            &self.0.credentialless_tls
        };
        slot.get_or_init(|| trust_js_tls::ClientContext::new(true))
            .clone()
    }

    pub(crate) fn request(
        &self,
        mut method: String,
        mut target: String,
        mut headers: Vec<(String, String)>,
        mut body: Option<Vec<u8>>,
        credentials: bool,
    ) -> Result<HttpResponse, String> {
        if body
            .as_ref()
            .is_some_and(|body| body.len() as u64 > http_syntax::MAX_BODY)
        {
            return Err("fetch: request body exceeds 32 MiB limit".into());
        }
        for _ in 0..=MAX_REDIRECTS {
            let u = url::parse(&target, None)?;
            match u.scheme.as_str() {
                "http" | "https" => {}
                other => return Err(format!("fetch: unsupported scheme '{other}'")),
            }
            let response = self.one_request(&method, &u, &headers, body.as_deref(), credentials)?;
            match response.status {
                301 | 302 | 303 | 307 | 308 => {
                    let Some(location) = single_header(&response.headers, "location")? else {
                        return Ok(HttpResponse {
                            url: u.href(),
                            ..response
                        });
                    };
                    let mut next = url::parse(&location, Some(&u.href()))?;
                    // Fetch's response location URL inherits the request fragment only when the
                    // Location value did not provide one (an explicit trailing `#` clears it).
                    if !location.contains('#') {
                        next.set_fragment(&u.fragment);
                    }
                    if u.origin() != next.origin() {
                        // Fetch HTTP-redirect fetch: credentials do not cross an origin boundary.
                        headers.retain(|(name, _)| !name.eq_ignore_ascii_case("authorization"));
                    }
                    // Fetch HTTP-redirect fetch: 301/302 rewrite POST; 303 rewrites everything
                    // except GET/HEAD, and deleting a body deletes its representation metadata.
                    if ((response.status == 301 || response.status == 302) && method == "POST")
                        || (response.status == 303 && method != "GET" && method != "HEAD")
                    {
                        method = "GET".to_string();
                        body = None;
                        headers.retain(|(name, _)| {
                            !matches!(
                                name.to_ascii_lowercase().as_str(),
                                "content-encoding"
                                    | "content-language"
                                    | "content-location"
                                    | "content-type"
                            )
                        });
                    }
                    target = next.href();
                }
                _ => {
                    return Ok(HttpResponse {
                        url: u.href(),
                        ..response
                    })
                }
            }
        }
        Err(format!("fetch '{target}': too many redirects"))
    }
}

#[cfg(test)]
pub(crate) fn request(
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
) -> Result<HttpResponse, String> {
    HttpClient::default().request(method, target, headers, body, false)
}

impl HttpClient {
    fn one_request(
        &self,
        method: &str,
        u: &url::Url,
        headers: &[(String, String)],
        body: Option<&[u8]>,
        credentials: bool,
    ) -> Result<HttpResponse, String> {
        if !http_syntax::is_token(method) {
            return Err(format!("fetch: invalid HTTP method {method:?}"));
        }
        http_syntax::validate_headers(headers)?;
        if body.is_some_and(|body| body.len() as u64 > http_syntax::MAX_BODY) {
            return Err("fetch: request body exceeds 32 MiB limit".into());
        }
        let port = u.port.unwrap_or(if u.scheme == "https" { 443 } else { 80 });
        let key = OriginKey {
            scheme: u.scheme.clone(),
            host: u.host.to_ascii_lowercase(),
            port,
            credentials,
        };
        let host_header = match u.port {
            Some(port) => format!("{}:{port}", u.host),
            None => u.host.clone(),
        };

        let request_target = format!("{}{}", u.path, u.query);
        if request_target.is_empty()
            || !request_target.is_ascii()
            || request_target
                .bytes()
                .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
        {
            return Err("fetch: URL did not serialize to a valid HTTP request target".into());
        }
        let mut request =
            format!("{method} {request_target} HTTP/1.1\r\nHost: {host_header}\r\n").into_bytes();
        let mut have_ua = false;
        for (name, value) in headers {
            if name.eq_ignore_ascii_case("host")
                || name.eq_ignore_ascii_case("connection")
                || name.eq_ignore_ascii_case("content-length")
                || name.eq_ignore_ascii_case("transfer-encoding")
            {
                continue; // the HTTP serializer owns routing and message framing
            }
            have_ua |= name.eq_ignore_ascii_case("user-agent");
            request.extend_from_slice(name.as_bytes());
            request.extend_from_slice(b": ");
            request.extend_from_slice(&field_value_bytes(value)?);
            request.extend_from_slice(b"\r\n");
        }
        if !have_ua {
            request.extend_from_slice(
                concat!("User-Agent: trust-js/", env!("CARGO_PKG_VERSION"), "\r\n").as_bytes(),
            );
        }
        if let Some(body) = body {
            request.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
        }
        request.extend_from_slice(b"\r\n");
        if request.len() > MAX_HEADER_BYTES {
            return Err("fetch: HTTP request metadata too large".into());
        }

        let pooled = self.pool().take(&key);
        let reused = pooled.is_some();
        let connection = match pooled {
            Some(connection) => connection,
            None => self.open_connection(u, port, credentials)?,
        };
        match exchange(connection, method, u, &request, body) {
            Ok(exchange) => {
                if let Some(connection) = exchange.connection {
                    self.pool().put(key, connection);
                }
                Ok(exchange.response)
            }
            Err(error) if reused && error.retryable && is_idempotent(method) => {
                // RFC 9110 §9.2.2: recover once from an asynchronously closed persistent
                // connection only when repeating the method is known to be idempotent.
                let connection = self.open_connection(u, port, credentials)?;
                let exchange = exchange(connection, method, u, &request, body)
                    .map_err(|error| error.message)?;
                if let Some(connection) = exchange.connection {
                    self.pool().put(key, connection);
                }
                Ok(exchange.response)
            }
            Err(error) => Err(error.message),
        }
    }

    fn open_connection(
        &self,
        u: &url::Url,
        port: u16,
        credentials: bool,
    ) -> Result<Connection, String> {
        let hostname = u.host.trim_matches(['[', ']']);
        let stream = TcpStream::connect((hostname, port))
            .map_err(|error| format!("fetch '{}': connect: {error}", u.href()))?;
        stream.set_read_timeout(Some(TIMEOUT)).ok();
        stream.set_write_timeout(Some(TIMEOUT)).ok();
        let stream: Box<dyn ReadWrite + Send> = if u.scheme == "https" {
            let context = self
                .tls_context(credentials)
                .map_err(|error| format!("fetch '{}': {error}", u.href()))?;
            Box::new(
                trust_js_tls::TlsStream::connect_with_context(stream, hostname, &[], &context)
                    .map_err(|error| format!("fetch '{}': {error}", u.href()))?,
            )
        } else {
            Box::new(stream)
        };
        Ok(BufReader::new(stream))
    }
}

struct Exchange {
    response: HttpResponse,
    connection: Option<Connection>,
}

struct ExchangeError {
    message: String,
    /// True only when an idle connection failed before yielding any response octet.
    retryable: bool,
}

fn exchange(
    mut reader: Connection,
    method: &str,
    u: &url::Url,
    request: &[u8],
    body: Option<&[u8]>,
) -> Result<Exchange, ExchangeError> {
    reader
        .get_mut()
        .write_all(request)
        .and_then(|()| body.map_or(Ok(()), |body| reader.get_mut().write_all(body)))
        .and_then(|()| reader.get_mut().flush())
        .map_err(|error| ExchangeError {
            message: format!("fetch '{}': write: {error}", u.href()),
            retryable: true,
        })?;

    let mut metadata = 0;
    let (version, status, status_text, headers) = loop {
        let status_line =
            read_line(&mut reader, &mut metadata, MAX_HEADER_BYTES).map_err(|error| {
                ExchangeError {
                    message: format!("fetch '{}': {error}", u.href()),
                    retryable: metadata == 0,
                }
            })?;
        let (version, status, status_text) =
            parse_status_line(&status_line).map_err(|error| ExchangeError {
                message: format!("fetch '{}': {error}", u.href()),
                retryable: false,
            })?;
        let fields =
            read_fields(&mut reader, &mut metadata, true).map_err(|error| ExchangeError {
                message: format!("fetch '{}': {error}", u.href()),
                retryable: false,
            })?;
        if (100..200).contains(&status) && status != 101 {
            continue;
        }
        if status == 101 {
            return Err(ExchangeError {
                message: format!("fetch '{}': unexpected protocol upgrade", u.href()),
                retryable: false,
            });
        }
        break (version, status, status_text, fields);
    };

    // RFC 9112 §6.3 gives HEAD/204/304 special wire framing. A 205 is a Fetch null-body status,
    // but it is not one of those HTTP framing exceptions: consume any declared wire body before
    // reuse, then discard it below. This keeps a violating origin from desynchronizing the pool.
    let framing_forbidden = method == "HEAD" || matches!(status, 204 | 304);
    let discard_body = status == 205;
    let framing =
        body_framing(&headers, false, framing_forbidden).map_err(|error| ExchangeError {
            message: format!("fetch '{}': {error}", u.href()),
            retryable: false,
        })?;
    let follows_redirect = matches!(status, 301 | 302 | 303 | 307 | 308)
        && single_header(&headers, "location")
            .map_err(|error| ExchangeError {
                message: format!("fetch '{}': {error}", u.href()),
                retryable: false,
            })?
            .is_some();
    let reusable = framing != http_syntax::BodyFraming::CloseDelimited
        && response_allows_persistence(version, &headers);
    let body = if follows_redirect && !reusable {
        // A transport that must close cannot be reused, so a redirect need not wait for a body it
        // will never expose. Reusable redirects are consumed completely below before pooling.
        Vec::new()
    } else {
        let body = read_body(&mut reader, framing).map_err(|error| ExchangeError {
            message: format!("fetch '{}': body: {error}", u.href()),
            retryable: false,
        })?;
        if follows_redirect || discard_body {
            Vec::new()
        } else {
            body
        }
    };
    Ok(Exchange {
        response: HttpResponse {
            status,
            status_text,
            headers,
            body,
            url: String::new(), // stamped by the redirect loop
        },
        connection: reusable.then_some(reader),
    })
}

trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HttpVersion {
    Http10,
    Http11,
}

fn response_allows_persistence(version: HttpVersion, headers: &[(String, String)]) -> bool {
    let mut close = false;
    let mut keep_alive = false;
    for value in http_syntax::header_values(headers, "connection") {
        for member in value.split(',') {
            let token = member.trim_matches([' ', '\t']);
            // A malformed Connection field cannot safely authorize another response on the same
            // transport. RFC 9112 §9.3 otherwise defaults HTTP/1.1 to persistent and HTTP/1.0 to
            // non-persistent.
            if !http_syntax::is_token(token) {
                return false;
            }
            close |= token.eq_ignore_ascii_case("close");
            keep_alive |= token.eq_ignore_ascii_case("keep-alive");
        }
    }
    !close
        && match version {
            HttpVersion::Http11 => true,
            HttpVersion::Http10 => keep_alive,
        }
}

fn is_idempotent(method: &str) -> bool {
    matches!(
        method,
        "GET" | "HEAD" | "PUT" | "DELETE" | "OPTIONS" | "TRACE"
    )
}

fn parse_status_line(line: &[u8]) -> Result<(HttpVersion, u16, String), String> {
    let first_space = line
        .iter()
        .position(|byte| *byte == b' ')
        .ok_or_else(|| "malformed HTTP status line".to_string())?;
    let second_space = line[first_space + 1..]
        .iter()
        .position(|byte| *byte == b' ')
        .map(|index| first_space + 1 + index)
        .ok_or_else(|| "malformed HTTP status line".to_string())?;
    let version = match &line[..first_space] {
        b"HTTP/1.1" => HttpVersion::Http11,
        b"HTTP/1.0" => HttpVersion::Http10,
        _ => return Err("unsupported HTTP response version".into()),
    };
    let status_bytes = &line[first_space + 1..second_space];
    if status_bytes.len() != 3 || !status_bytes.iter().all(u8::is_ascii_digit) {
        return Err("invalid HTTP response status".into());
    }
    let status = std::str::from_utf8(status_bytes).unwrap().parse().unwrap();
    if !(100..=599).contains(&status) {
        return Err("HTTP response status is outside 100..=599".into());
    }
    let reason = http_syntax::latin1(&line[second_space + 1..]);
    if reason
        .bytes()
        .any(|byte| byte.is_ascii_control() && byte != b'\t')
    {
        return Err("invalid HTTP reason phrase".into());
    }
    Ok((version, status, reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use std::thread;

    fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        request
    }

    fn request_from_raw_response(response: &[u8]) -> Result<HttpResponse, String> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let response = response.to_vec();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_request(&mut stream);
            stream.write_all(&response).unwrap();
        });
        let result = request("GET".into(), format!("http://{address}/"), vec![], None);
        server.join().unwrap();
        result
    }

    #[test]
    fn chunked_decoding() {
        let raw = b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        let mut r = std::io::BufReader::new(&raw[..]);
        assert_eq!(
            read_body(&mut r, http_syntax::BodyFraming::Chunked).unwrap(),
            b"Wikipedia"
        );
    }

    #[test]
    fn response_framing_rejects_ambiguous_incomplete_and_oversized_messages() {
        for response in [
            &b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nxx"[..],
            &b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n0\r\n\r\n"
                [..],
            &b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n"[..],
            &b"HTTP/1.1 200 OK\r\nContent-Length: 33554433\r\n\r\n"[..],
            &b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nxx"[..],
        ] {
            assert!(request_from_raw_response(response).is_err());
        }
    }

    #[test]
    fn client_consumes_informational_responses_and_unfolds_legacy_fields() {
        let response = request_from_raw_response(
            b"HTTP/1.1 100 Continue\r\nX-Ignored: yes\r\n\r\n\
              HTTP/1.1 200 OK\r\nX-Folded: one\r\n two\r\nContent-Length: 2\r\n\r\nok",
        )
        .unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(
            http_syntax::first_header(&response.headers, "x-folded").unwrap(),
            "one two"
        );
        assert_eq!(response.body, b"ok");
    }

    #[test]
    fn redirects_rewrite_post_and_its_representation_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut second = Vec::new();
            for index in 0..2 {
                let request = read_request(&mut stream);
                if index == 0 {
                    stream.read_exact(&mut [0]).unwrap();
                    stream
                        .write_all(
                            b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\n\r\n",
                        )
                        .unwrap();
                } else {
                    second = request;
                    stream
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                        .unwrap();
                }
            }
            String::from_utf8(second).unwrap()
        });
        let headers = vec![
            ("Authorization".into(), "token".into()),
            ("Content-Type".into(), "text/plain".into()),
            ("Content-Language".into(), "en".into()),
            ("X-Keep".into(), "yes".into()),
        ];
        let response = request(
            "POST".into(),
            format!("http://{address}/start#kept"),
            headers,
            Some(b"x".to_vec()),
        )
        .unwrap();
        assert_eq!(response.url, format!("http://{address}/next#kept"));
        let second = server.join().unwrap().to_ascii_lowercase();
        assert!(second.starts_with("get /next http/1.1\r\n"));
        assert!(second.contains("authorization: token\r\n"));
        assert!(second.contains("x-keep: yes\r\n"));
        assert!(!second.contains("content-type:"));
        assert!(!second.contains("content-language:"));
        assert!(!second.contains("content-length:"));
    }

    #[test]
    fn redirects_strip_authorization_when_the_origin_changes() {
        let destination = TcpListener::bind("127.0.0.1:0").unwrap();
        let destination_address = destination.local_addr().unwrap();
        let destination_server = thread::spawn(move || {
            let (mut stream, _) = destination.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            String::from_utf8(request).unwrap()
        });

        let source = TcpListener::bind("127.0.0.1:0").unwrap();
        let source_address = source.local_addr().unwrap();
        let source_server = thread::spawn(move || {
            let (mut stream, _) = source.accept().unwrap();
            let mut byte = [0];
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            write!(
                stream,
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{destination_address}/next\r\nContent-Length: 0\r\n\r\n"
            )
            .unwrap();
        });
        request(
            "cUsToM".into(),
            format!("http://{source_address}/start"),
            vec![("Authorization".into(), "secret".into())],
            None,
        )
        .unwrap();
        source_server.join().unwrap();
        let second = destination_server.join().unwrap();
        assert!(second.starts_with("cUsToM /next HTTP/1.1\r\n"));
        assert!(!second.to_ascii_lowercase().contains("authorization:"));
    }

    #[test]
    fn reuses_fully_consumed_self_delimited_responses() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let first = read_request(&mut stream);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\none")
                .unwrap();
            let second = read_request(&mut stream);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\ntwo")
                .unwrap();
            (first, second)
        });
        let client = HttpClient::default();
        let url = format!("http://{address}/");
        assert_eq!(
            client
                .request("GET".into(), url.clone(), vec![], None, false)
                .unwrap()
                .body,
            b"one"
        );
        assert_eq!(
            client
                .request("GET".into(), url, vec![], None, false)
                .unwrap()
                .body,
            b"two"
        );
        let (first, second) = server.join().unwrap();
        assert!(!first
            .to_ascii_lowercase()
            .windows(19)
            .any(|window| window == b"connection: close\r\n"));
        assert!(second.starts_with(b"GET / HTTP/1.1\r\n"));
    }

    #[test]
    fn reset_content_discards_declared_bytes_before_connection_reuse() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_request(&mut stream);
            // Fetch exposes a null body for 205, but RFC 9112 does not make 205 a wire-framing
            // exception. Consume these nonconforming bytes so they cannot prefix the next reply.
            stream
                .write_all(b"HTTP/1.1 205 Reset Content\r\nContent-Length: 4\r\n\r\nevil")
                .unwrap();
            let second = read_request(&mut stream);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .unwrap();
            second
        });

        let client = HttpClient::default();
        let url = format!("http://{address}/");
        let first = client
            .request("GET".into(), url.clone(), vec![], None, false)
            .unwrap();
        assert_eq!(first.status, 205);
        assert!(first.body.is_empty());
        assert_eq!(
            client
                .request("GET".into(), url, vec![], None, false)
                .unwrap()
                .body,
            b"ok"
        );
        assert!(server.join().unwrap().starts_with(b"GET / HTTP/1.1\r\n"));
    }

    #[test]
    fn connection_close_and_credentials_partition_prevent_reuse() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for index in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                read_request(&mut stream);
                let connection = if index == 0 {
                    "Connection: close\r\n"
                } else {
                    ""
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\n{connection}Content-Length: 1\r\n\r\n{index}"
                )
                .unwrap();
            }
        });
        let client = HttpClient::default();
        let url = format!("http://{address}/");
        assert_eq!(
            client
                .request("GET".into(), url.clone(), vec![], None, false)
                .unwrap()
                .body,
            b"0"
        );
        assert_eq!(
            client
                .request("GET".into(), url.clone(), vec![], None, false)
                .unwrap()
                .body,
            b"1"
        );
        assert_eq!(
            client
                .request("GET".into(), url, vec![], None, true)
                .unwrap()
                .body,
            b"2"
        );
        server.join().unwrap();
    }

    #[test]
    fn retries_stale_idle_connection_once_for_idempotent_method() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            read_request(&mut first);
            first
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\na")
                .unwrap();
            drop(first); // race the client's next request against an asynchronous idle close
            let (mut retry, _) = listener.accept().unwrap();
            let request = read_request(&mut retry);
            retry
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nb")
                .unwrap();
            request
        });
        let client = HttpClient::default();
        let url = format!("http://{address}/");
        assert_eq!(
            client
                .request("GET".into(), url.clone(), vec![], None, false)
                .unwrap()
                .body,
            b"a"
        );
        assert_eq!(
            client
                .request("GET".into(), url, vec![], None, false)
                .unwrap()
                .body,
            b"b"
        );
        assert!(server.join().unwrap().starts_with(b"GET / HTTP/1.1\r\n"));
    }

    #[test]
    #[ignore = "requires external network and a system OpenSSL trust store"]
    fn https_uses_verified_tls() {
        let response = request("GET".into(), "https://example.com/".into(), vec![], None).unwrap();
        assert_eq!(response.status, 200);
    }
}
