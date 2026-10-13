//! WHATWG URL parsing and serialization shared by the JavaScript API and every network client.
//!
//! The state machine comes from the pure-Rust Servo `url` implementation. Its IDNA path uses
//! non-transitional UTS #46 processing with the URL Standard's forbidden-domain-code-point list.
//! Keeping the parsed record here prevents fetch, WebSocket, EventSource, and `URL` from applying
//! subtly different authority, IPv4/IPv6, percent-encoding, or relative-resolution rules.

use whatwg_url::{quirks, Position, Url as ParsedUrl};

const ASCII_DOMAIN_PLACEHOLDER: &str = "trust-js-invalid-domain.invalid";

/// Servo `url` 2.5.8 predates the URL Standard's opaque-path trailing-space rule. In the opaque
/// path state, only the last space immediately before `?` or `#` is percent-encoded; preceding
/// spaces remain literal. Normalize that one parser transition before handing input to the crate.
fn normalize_opaque_path_space(input: &str) -> Option<String> {
    let mut normalized: String = input
        .chars()
        .filter(|character| !matches!(character, '\t' | '\n' | '\r'))
        .collect();
    let stripped_units = normalized.len() != input.len();
    let colon = match normalized.find(':') {
        Some(colon) => colon,
        None => return stripped_units.then_some(normalized),
    };
    let scheme = &normalized[..colon];
    if matches!(
        scheme.to_ascii_lowercase().as_str(),
        "ftp" | "file" | "http" | "https" | "ws" | "wss"
    ) || normalized.as_bytes().get(colon + 1) == Some(&b'/')
    {
        return stripped_units.then_some(normalized);
    }
    let delimiter = match normalized[colon + 1..]
        .find(['?', '#'])
        .map(|offset| colon + 1 + offset)
    {
        Some(delimiter) => delimiter,
        None => return stripped_units.then_some(normalized),
    };
    if delimiter == 0 || normalized.as_bytes()[delimiter - 1] != b' ' {
        return stripped_units.then_some(normalized);
    }
    normalized.replace_range(delimiter - 1..delimiter, "%20");
    Some(normalized)
}

fn normalize_hierarchical_path_caret(input: &str) -> Option<String> {
    let colon = input.find(':');
    let mut path_start = 0;
    if let Some(colon) = colon {
        let scheme = &input[..colon];
        let valid_scheme = !scheme.is_empty()
            && scheme.as_bytes()[0].is_ascii_alphabetic()
            && scheme[1..]
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'));
        if valid_scheme {
            if input[colon + 1..].starts_with("//") {
                path_start = input[colon + 3..]
                    .find('/')
                    .map_or(input.len(), |offset| colon + 3 + offset);
            } else if !is_special_scheme(&scheme.to_ascii_lowercase())
                && input.as_bytes().get(colon + 1) != Some(&b'/')
            {
                return None;
            } else {
                path_start = colon + 1;
            }
        }
    }
    let path_end = input.find(['?', '#']).unwrap_or(input.len());
    if path_start >= path_end || !input[path_start..path_end].contains('^') {
        return None;
    }
    let mut normalized = input[..path_start].to_string();
    normalized.push_str(&input[path_start..path_end].replace('^', "%5E"));
    normalized.push_str(&input[path_end..]);
    Some(normalized)
}

fn normalize_special_relative_authority(input: &str, base: &ParsedUrl) -> Option<String> {
    if base.scheme() == "file" || !is_special_scheme(base.scheme()) {
        return None;
    }
    let prefix_units = input
        .chars()
        .take_while(|character| matches!(character, '/' | '\\'))
        .count();
    if prefix_units < 3 {
        return None;
    }
    let byte_offset = input
        .char_indices()
        .nth(prefix_units)
        .map_or(input.len(), |(offset, _)| offset);
    Some(format!("//{}", &input[byte_offset..]))
}

fn resolve_non_special_backslash(input: &str, base: &ParsedUrl) -> Option<ParsedUrl> {
    if is_special_scheme(base.scheme()) || base.cannot_be_a_base() || !input.starts_with('\\') {
        return None;
    }
    let path = base.path();
    let directory_end = path.rfind('/').map_or(0, |position| position + 1);
    let absolute = format!(
        "{}{}{}",
        &base[..Position::BeforePath],
        &path[..directory_end],
        input
    );
    ParsedUrl::parse(&absolute).ok()
}

/// Servo 2.5.8 applies its Windows-drive path-pop guard to the last segment of every hierarchical
/// URL. The URL Standard limits that guard to a one-item `file` path. Substitute an ordinary
/// segment while resolving a leading double-dot so the normal relative algorithm can shorten it.
fn resolve_non_file_drive_parent(input: &str, base: &ParsedUrl) -> Option<ParsedUrl> {
    if base.scheme() == "file"
        || base.cannot_be_a_base()
        || input.starts_with(['/', '\\'])
        || !is_double_dot_segment(input.split(['/', '?', '#']).next().unwrap_or_default())
    {
        return None;
    }
    let path = base.path();
    let without_trailing_slash = path.strip_suffix('/')?;
    let segment_start = without_trailing_slash
        .rfind('/')
        .map_or(0, |index| index + 1);
    if !is_windows_drive_segment(&without_trailing_slash[segment_start..]) {
        return None;
    }
    let mut substitute_path = without_trailing_slash[..segment_start].to_string();
    substitute_path.push_str("trust-js-drive-segment/");
    let mut substitute = base.clone();
    substitute.set_path(&substitute_path);
    substitute.join(input).ok()
}

/// Parsed components. `port` is `None` when absent or equal to the scheme's default port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Url {
    pub scheme: String,
    pub username: String,
    pub password: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: String,    // includes leading '?' when present
    pub fragment: String, // includes leading '#' when present
    serialized: String,
    ascii_domain_override: Option<String>,
    parsed: ParsedUrl,
}

impl Url {
    fn from_parsed(parsed: ParsedUrl) -> Self {
        let serialized = parsed.as_str().to_string();
        let query = parsed.query().map_or_else(String::new, |value| {
            let mut output = String::with_capacity(value.len() + 1);
            output.push('?');
            output.push_str(value);
            output
        });
        let fragment = parsed.fragment().map_or_else(String::new, |value| {
            let mut output = String::with_capacity(value.len() + 1);
            output.push('#');
            output.push_str(value);
            output
        });
        Self {
            scheme: parsed.scheme().to_string(),
            username: parsed.username().to_string(),
            password: parsed.password().unwrap_or_default().to_string(),
            host: parsed.host_str().unwrap_or_default().to_string(),
            port: parsed.port(),
            path: parsed.path().to_string(),
            query,
            fragment,
            serialized,
            ascii_domain_override: None,
            parsed,
        }
    }

    pub fn href(&self) -> String {
        self.serialized.clone()
    }

    pub fn origin(&self) -> String {
        // URL Standard §6.1: a blob URL only inherits the origin of an inner HTTP(S) or file
        // URL. Servo 2.5.8 still recurses through every tuple-origin scheme (including blob,
        // FTP, and WebSocket), which no longer matches the Living Standard.
        if self.scheme == "blob" {
            let Ok(inner) = parse(&self.path, None) else {
                return "null".to_string();
            };
            return match inner.scheme.as_str() {
                "http" | "https" | "file" => inner.origin(),
                _ => "null".to_string(),
            };
        }
        let mut origin = self.parsed.origin().ascii_serialization();
        if let Some(domain) = &self.ascii_domain_override {
            if let Some(position) = origin.find(ASCII_DOMAIN_PLACEHOLDER) {
                origin.replace_range(position..position + ASCII_DOMAIN_PLACEHOLDER.len(), domain);
            }
        }
        origin
    }

    pub fn set_fragment(&mut self, fragment: &str) {
        let fragment = fragment.strip_prefix('#').unwrap_or(fragment);
        self.parsed
            .set_fragment((!fragment.is_empty()).then_some(fragment));
        self.fragment = self.parsed.fragment().map_or_else(String::new, |value| {
            let mut output = String::with_capacity(value.len() + 1);
            output.push('#');
            output.push_str(value);
            output
        });
        let fragment_start = self.serialized.find('#').unwrap_or(self.serialized.len());
        self.serialized.truncate(fragment_start);
        self.serialized.push_str(&self.fragment);
    }
}

fn is_special_scheme(scheme: &str) -> bool {
    matches!(scheme, "ftp" | "file" | "http" | "https" | "ws" | "wss")
}

fn contains_forbidden_opaque_host_code_point(host: &str) -> bool {
    host.chars().any(|character| {
        character <= '\u{20}'
            || character == '\u{7f}'
            || matches!(
                character,
                '#' | '/' | ':' | '<' | '>' | '?' | '@' | '[' | '\\' | ']' | '^' | '|'
            )
    })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn ascii_domain_ends_in_number(domain: &[u8]) -> bool {
    let mut parts: Vec<&[u8]> = domain.split(|byte| *byte == b'.').collect();
    if parts.len() > 1 && parts.last().is_some_and(|part| part.is_empty()) {
        parts.pop();
    }
    let Some(last) = parts.last().copied() else {
        return false;
    };
    if !last.is_empty() && last.iter().all(u8::is_ascii_digit) {
        return true;
    }
    if last.len() >= 2 && last[0] == b'0' && matches!(last[1], b'x' | b'X') {
        return last[2..].iter().all(u8::is_ascii_hexdigit);
    }
    if last.len() >= 2 && last[0] == b'0' {
        return last[1..].iter().all(|byte| matches!(byte, b'0'..=b'7'));
    }
    false
}

/// URL Standard §3.3 deliberately accepts an ASCII domain (with `beStrict` false) even when
/// strict UTS #46 reports invalid Punycode or STD3 errors. `idna` as used by Servo `url` 2.5.8
/// predates that compatibility rule, so retain such domains behind an otherwise equivalent parsed
/// record. The forbidden-domain-code-point and IPv4 branches remain failures.
fn current_ascii_domain(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = hex_value(*bytes.get(index + 1)?)?;
            let low = hex_value(*bytes.get(index + 2)?)?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    if decoded.is_empty() || !decoded.is_ascii() {
        return None;
    }
    if decoded.iter().any(|byte| {
        *byte <= 0x20
            || *byte == 0x7f
            || matches!(
                *byte,
                b'#' | b'/'
                    | b':'
                    | b'<'
                    | b'>'
                    | b'?'
                    | b'@'
                    | b'['
                    | b'\\'
                    | b']'
                    | b'^'
                    | b'|'
                    | b'%'
            )
    }) {
        return None;
    }
    // Leave anything that might enter the IPv4 parser to Servo rather than accepting an invalid
    // numeric address as an opaque ASCII domain.
    if ascii_domain_ends_in_number(&decoded) {
        return None;
    }
    Some(String::from_utf8(decoded).ok()?.to_ascii_lowercase())
}

fn apply_ascii_domain(url: &mut Url, domain: Option<String>) {
    let Some(domain) = domain else { return };
    let position = url.parsed[..Position::BeforeHost].len();
    let range = position..position + ASCII_DOMAIN_PLACEHOLDER.len();
    if url.serialized.get(range.clone()) != Some(ASCII_DOMAIN_PLACEHOLDER) {
        return;
    }
    url.serialized.replace_range(range, &domain);
    url.host = domain.clone();
    url.ascii_domain_override = Some(domain);
}

fn with_ascii_domain(parsed: ParsedUrl, domain: Option<String>) -> Url {
    let mut url = Url::from_parsed(parsed);
    apply_ascii_domain(&mut url, domain);
    url
}

/// Parse an absolute URL whose special-scheme host hits the post-2.5.8 ASCII-domain rule. The
/// placeholder is never exposed; it only lets Servo continue handling credentials, ports, paths,
/// queries, fragments, and origins with its existing URL-record offsets.
fn parse_ascii_domain_backing(input: &str) -> Option<(ParsedUrl, String)> {
    let cleaned: String = input
        .chars()
        .filter(|character| !matches!(character, '\t' | '\n' | '\r'))
        .collect();
    let colon = cleaned.find(':')?;
    let scheme = cleaned[..colon].to_ascii_lowercase();
    if !is_special_scheme(&scheme) || !cleaned[colon + 1..].starts_with("//") {
        return None;
    }
    let authority_start = colon + 3;
    let authority_end = cleaned[authority_start..]
        .find(['/', '\\', '?', '#'])
        .map_or(cleaned.len(), |offset| authority_start + offset);
    let authority = &cleaned[authority_start..authority_end];
    let host_port_start = authority
        .rfind('@')
        .map_or(authority_start, |at| authority_start + at + 1);
    let host_port = &cleaned[host_port_start..authority_end];
    if host_port.starts_with('[') {
        return None;
    }
    let host_end = host_port
        .find(':')
        .map_or(authority_end, |offset| host_port_start + offset);
    let domain = current_ascii_domain(&cleaned[host_port_start..host_end])?;
    let mut candidate = cleaned;
    candidate.replace_range(host_port_start..host_end, ASCII_DOMAIN_PLACEHOLDER);
    let parsed = ParsedUrl::parse(&candidate).ok()?;
    Some((parsed, domain))
}

fn parse_backing(input: &str) -> Result<(ParsedUrl, Option<String>), String> {
    match ParsedUrl::parse(input) {
        Ok(parsed)
            if !is_special_scheme(parsed.scheme())
                && parsed.host_str().is_some_and(|host| {
                    !(host.starts_with('[') && host.ends_with(']'))
                        && contains_forbidden_opaque_host_code_point(host)
                }) =>
        {
            Err(format!(
                "invalid URL '{input}': forbidden opaque-host code point"
            ))
        }
        Ok(parsed) => Ok((parsed, None)),
        Err(error) => parse_ascii_domain_backing(input)
            .map(|(parsed, domain)| (parsed, Some(domain)))
            .ok_or_else(|| format!("invalid URL '{input}': {error}")),
    }
}

fn is_single_dot_segment(segment: &str) -> bool {
    segment == "." || segment.eq_ignore_ascii_case("%2e")
}

fn is_double_dot_segment(segment: &str) -> bool {
    matches!(segment, "..")
        || segment.eq_ignore_ascii_case(".%2e")
        || segment.eq_ignore_ascii_case("%2e.")
        || segment.eq_ignore_ascii_case("%2e%2e")
}

fn is_windows_drive_segment(segment: &str) -> bool {
    let bytes = segment.as_bytes();
    bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && matches!(bytes[1], b':' | b'|')
}

fn starts_with_windows_drive_segment(input: &str) -> bool {
    let mut characters = input.chars();
    matches!(characters.next(), Some(character) if character.is_ascii_alphabetic())
        && matches!(characters.next(), Some(':' | '|'))
        && matches!(characters.next(), None | Some('/' | '\\' | '?' | '#'))
}

fn encode_path_segment(segment: &str) -> String {
    let mut output = String::with_capacity(segment.len());
    for character in segment.chars() {
        let encode = character <= '\u{1f}'
            || character > '~'
            || matches!(
                character,
                ' ' | '"' | '#' | '<' | '>' | '?' | '^' | '`' | '{' | '}'
            );
        if encode {
            let mut bytes = [0; 4];
            for byte in character.encode_utf8(&mut bytes).bytes() {
                output.push('%');
                output.push(
                    char::from_digit(u32::from(byte >> 4), 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                output.push(
                    char::from_digit(u32::from(byte & 0xf), 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        } else {
            output.push(character);
        }
    }
    output
}

fn with_path_serialization(parsed: ParsedUrl, path: String, serialized_path: &str) -> Url {
    let mut prefix = parsed[..Position::BeforePath].to_string();
    if !is_special_scheme(parsed.scheme())
        && !parsed.has_authority()
        && parsed.path().starts_with("//")
        && prefix.ends_with("/.")
    {
        prefix.truncate(prefix.len() - 2);
    }
    let suffix = &parsed[Position::AfterPath..];
    let serialized = format!("{prefix}{serialized_path}{suffix}");
    let mut url = Url::from_parsed(parsed);
    url.path = path;
    url.serialized = serialized;
    url
}

fn file_path_segments(path: &str) -> Vec<String> {
    path.strip_prefix('/')
        .unwrap_or(path)
        .split('/')
        .map(str::to_string)
        .collect()
}

fn shorten_file_path(segments: &mut Vec<String>) {
    if segments.len() == 1 && is_windows_drive_segment(&segments[0]) {
        return;
    }
    segments.pop();
}

/// Run the URL Standard's path state for a file URL. Unlike Servo 2.5.8, the current algorithm
/// retains empty path items and does not discard a non-empty host when the first item is a drive
/// letter.
fn append_file_path(segments: &mut Vec<String>, input: &str) {
    let path_end = input.find(['?', '#']).unwrap_or(input.len());
    let normalized = input[..path_end].replace('\\', "/");
    let raw_segments: Vec<&str> = normalized.split('/').collect();
    for (index, segment) in raw_segments.iter().enumerate() {
        let at_end = index + 1 == raw_segments.len();
        if is_double_dot_segment(segment) {
            shorten_file_path(segments);
            if at_end {
                segments.push(String::new());
            }
        } else if is_single_dot_segment(segment) {
            if at_end {
                segments.push(String::new());
            }
        } else {
            let mut encoded = encode_path_segment(segment);
            if segments.is_empty() && is_windows_drive_segment(&encoded) {
                encoded.replace_range(1..2, ":");
            }
            segments.push(encoded);
        }
    }
}

fn parse_current_file_host(input: &str) -> Result<(String, Option<String>), String> {
    if input.is_empty() {
        return Ok((String::new(), None));
    }
    let candidate = format!("http://{input}/");
    let (parsed, domain) = parse_backing(&candidate)?;
    if !parsed.username().is_empty() || parsed.password().is_some() || parsed.port().is_some() {
        return Err(format!("invalid file URL host '{input}'"));
    }
    let host = domain
        .clone()
        .unwrap_or_else(|| parsed.host_str().unwrap_or_default().to_string());
    if host.eq_ignore_ascii_case("localhost") {
        Ok((String::new(), None))
    } else {
        Ok((host, domain))
    }
}

fn build_current_file_url(
    host: String,
    domain: Option<String>,
    segments: Vec<String>,
    generic: Url,
) -> Result<Url, String> {
    let backing_host = if domain.is_some() {
        ASCII_DOMAIN_PLACEHOLDER
    } else {
        &host
    };
    let mut parsed = ParsedUrl::parse(&format!("file://{backing_host}/"))
        .map_err(|error| format!("invalid file URL host '{host}': {error}"))?;
    let path = format!("/{}", segments.join("/"));
    parsed.set_path(&path);
    parsed.set_query(generic.parsed.query());
    parsed.set_fragment(generic.parsed.fragment());
    let mut url = with_path_serialization(parsed, path.clone(), &path);
    apply_ascii_domain(&mut url, domain);
    Ok(url)
}

/// Adapter for the current file, file-slash, file-host, and path states. The maintained Servo
/// parser remains the authority for host/query/fragment parsing; this only supplies the two
/// Living-Standard transitions that its 2.5.8 release explicitly lists as expected failures.
fn apply_current_file_states(input: &str, base: Option<&Url>, generic: Url) -> Result<Url, String> {
    let trimmed = input.trim_matches(|character| character <= '\u{20}');
    let cleaned: String = trimmed
        .chars()
        .filter(|character| !matches!(character, '\t' | '\n' | '\r'))
        .collect();
    let state_input = cleaned
        .find(':')
        .filter(|colon| cleaned[..*colon].eq_ignore_ascii_case("file"))
        .map_or(cleaned.as_str(), |colon| &cleaned[colon + 1..]);
    let file_base = base.filter(|base| base.scheme == "file");

    let mut host = String::new();
    let mut domain = None;
    let mut segments = Vec::new();

    if matches!(state_input.as_bytes().first(), Some(b'/' | b'\\')) {
        let after_first = &state_input[1..];
        if matches!(after_first.as_bytes().first(), Some(b'/' | b'\\')) {
            let after_second = &after_first[1..];
            let host_end = after_second
                .find(['/', '\\', '?', '#'])
                .unwrap_or(after_second.len());
            let host_input = &after_second[..host_end];
            let after_host = &after_second[host_end..];
            if is_windows_drive_segment(host_input) {
                append_file_path(&mut segments, after_second);
            } else {
                (host, domain) = parse_current_file_host(host_input)?;
                let path_input = after_host.strip_prefix(['/', '\\']).unwrap_or(after_host);
                append_file_path(&mut segments, path_input);
            }
        } else {
            if let Some(base) = file_base {
                host.clone_from(&base.host);
                domain.clone_from(&base.ascii_domain_override);
                if !starts_with_windows_drive_segment(after_first) {
                    let base_segments = file_path_segments(&base.path);
                    if base_segments
                        .first()
                        .is_some_and(|segment| is_windows_drive_segment(segment))
                    {
                        segments.push(base_segments[0].clone());
                    }
                }
            }
            append_file_path(&mut segments, after_first);
        }
    } else if let Some(base) = file_base {
        host.clone_from(&base.host);
        domain.clone_from(&base.ascii_domain_override);
        if state_input.is_empty() || state_input.starts_with(['?', '#']) {
            segments = file_path_segments(&base.path);
        } else {
            if !starts_with_windows_drive_segment(state_input) {
                segments = file_path_segments(&base.path);
                shorten_file_path(&mut segments);
            }
            append_file_path(&mut segments, state_input);
        }
    } else {
        append_file_path(&mut segments, state_input);
    }

    build_current_file_url(host, domain, segments, generic)
}

/// Run the URL Standard's path-start/path state override over an empty path. Servo `url` 2.5.8's
/// structural `set_path` cannot represent an empty hierarchical path, file-path empty segments,
/// or the `/.` serialization guard for a non-special URL with a null host.
fn set_pathname(parsed: ParsedUrl, value: &str) -> Url {
    if parsed.cannot_be_a_base() {
        return Url::from_parsed(parsed);
    }

    let special = is_special_scheme(parsed.scheme());
    let cleaned: String = value
        .chars()
        .filter(|character| !matches!(character, '\t' | '\n' | '\r'))
        .map(|character| {
            if special && character == '\\' {
                '/'
            } else {
                character
            }
        })
        .collect();

    let mut segments = Vec::<String>::new();
    if cleaned.is_empty() {
        // Path-start EOF appends an empty segment for special URLs and URLs with a null host. A
        // non-special URL with an explicitly empty host is allowed to retain a zero-item path.
        if special || !parsed.has_authority() {
            segments.push(String::new());
        }
    } else {
        let path_input = cleaned.strip_prefix('/').unwrap_or(&cleaned);
        let raw_segments: Vec<&str> = path_input.split('/').collect();
        for (index, segment) in raw_segments.iter().enumerate() {
            let at_end = index + 1 == raw_segments.len();
            if is_double_dot_segment(segment) {
                let drive_root = parsed.scheme() == "file"
                    && segments.len() == 1
                    && is_windows_drive_segment(&segments[0]);
                if !drive_root {
                    segments.pop();
                }
                if at_end {
                    segments.push(String::new());
                }
            } else if is_single_dot_segment(segment) {
                if at_end {
                    segments.push(String::new());
                }
            } else {
                let mut encoded = encode_path_segment(segment);
                if parsed.scheme() == "file"
                    && segments.is_empty()
                    && is_windows_drive_segment(&encoded)
                {
                    encoded.replace_range(1..2, ":");
                }
                segments.push(encoded);
            }
        }
    }

    let mut path = String::new();
    for segment in &segments {
        path.push('/');
        path.push_str(segment);
    }
    let serialized_path = if !special && !parsed.has_authority() && path.starts_with("//") {
        format!("/.{path}")
    } else {
        path.clone()
    };
    with_path_serialization(parsed, path, &serialized_path)
}

/// Parse `input` on its own, or against `base` when it is relative.
pub(crate) fn parse(input: &str, base: Option<&str>) -> Result<Url, String> {
    let normalized_opaque_input = normalize_opaque_path_space(input);
    let input = normalized_opaque_input.as_deref().unwrap_or(input);
    let normalized_caret_input = normalize_hierarchical_path_caret(input);
    let input = normalized_caret_input.as_deref().unwrap_or(input);
    let normalized_opaque_base = base.and_then(normalize_opaque_path_space);
    let base = normalized_opaque_base.as_deref().or(base);
    let normalized_caret_base = base.and_then(normalize_hierarchical_path_caret);
    let base = normalized_caret_base.as_deref().or(base);
    let mut parsed_base_url = None;
    let (parsed, ascii_domain) = match base {
        Some(base) => {
            let base_url =
                parse(base, None).map_err(|error| format!("invalid base URL '{base}': {error}"))?;
            let parsed_base = base_url.parsed.clone();
            let base_domain = base_url.ascii_domain_override.clone();
            let normalized_relative = normalize_special_relative_authority(input, &parsed_base);
            let relative = normalized_relative.as_deref().unwrap_or(input);
            let joined = resolve_non_file_drive_parent(relative, &parsed_base)
                .or_else(|| resolve_non_special_backslash(relative, &parsed_base))
                .map(Ok)
                .unwrap_or_else(|| parsed_base.join(relative));
            let result = match joined {
                Ok(parsed) => {
                    let inherited = (parsed.host_str() == Some(ASCII_DOMAIN_PLACEHOLDER))
                        .then_some(base_domain)
                        .flatten();
                    (parsed, inherited)
                }
                Err(error) => parse_ascii_domain_backing(input)
                    .map(|(parsed, domain)| (parsed, Some(domain)))
                    .ok_or_else(|| format!("invalid URL '{input}': {error}"))?,
            };
            parsed_base_url = Some(base_url);
            result
        }
        None => parse_backing(input)?,
    };
    let url = with_ascii_domain(parsed, ascii_domain);
    if url.scheme == "file" {
        apply_current_file_states(input, parsed_base_url.as_ref(), url)
    } else {
        Ok(url)
    }
}

/// Apply one URL API setter using a parsed URL record. Setter parse failures are ignored, as the
/// URL Standard requires, while the `href` setter continues to use `parse` and throw on failure.
pub(crate) fn mutate(href: &str, component: &str, value: &str) -> Result<Url, String> {
    let current =
        parse(href, None).map_err(|error| format!("invalid current URL '{href}': {error}"))?;
    let mut parsed = current.parsed;
    let mut ascii_domain = current.ascii_domain_override;
    match component {
        "protocol" => {
            let _ = quirks::set_protocol(&mut parsed, value);
        }
        "username" => {
            let _ = quirks::set_username(&mut parsed, value);
        }
        "password" => {
            let _ = quirks::set_password(&mut parsed, value);
        }
        "host" => {
            if quirks::set_host(&mut parsed, value).is_ok() {
                ascii_domain = None;
            } else if let Some(domain) = current_ascii_domain(value) {
                if quirks::set_host(&mut parsed, ASCII_DOMAIN_PLACEHOLDER).is_ok() {
                    ascii_domain = Some(domain);
                }
            }
        }
        "hostname" => {
            let old_path = parsed.path().to_string();
            let escaped_null_host_path = !is_special_scheme(parsed.scheme())
                && !parsed.has_authority()
                && old_path.starts_with("//");
            if quirks::set_hostname(&mut parsed, value).is_ok() {
                ascii_domain = None;
            } else if let Some(domain) = current_ascii_domain(value) {
                if quirks::set_hostname(&mut parsed, ASCII_DOMAIN_PLACEHOLDER).is_ok() {
                    ascii_domain = Some(domain);
                }
            }
            if escaped_null_host_path {
                let mut url = with_ascii_domain(parsed, ascii_domain);
                let search_start = url.scheme.len() + 3;
                if let Some(relative) = url.serialized[search_start..].find("/.//") {
                    let guard = search_start + relative;
                    url.serialized.replace_range(guard..guard + 2, "");
                }
                url.path = old_path;
                return Ok(url);
            }
        }
        "port" => {
            let cleaned: String = value
                .chars()
                .filter(|character| !matches!(character, '\t' | '\n' | '\r'))
                .collect();
            // Port-state EOF with an empty buffer is failure under a state override. An actually
            // empty setter value is handled separately by the API and clears the port.
            if value.is_empty() || !cleaned.is_empty() {
                let _ = quirks::set_port(&mut parsed, &cleaned);
            }
        }
        "pathname" => {
            let mut url = set_pathname(parsed, value);
            apply_ascii_domain(&mut url, ascii_domain);
            return Ok(url);
        }
        "search" => quirks::set_search(&mut parsed, value),
        "hash" => quirks::set_hash(&mut parsed, value),
        _ => return Err(format!("unknown URL component '{component}'")),
    }
    Ok(with_ascii_domain(parsed, ascii_domain))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(input: &str) -> Url {
        parse(input, None).unwrap()
    }

    #[test]
    fn parses_serializes_and_resolves_special_urls() {
        let url = p("HTTP://User:Pw@Example.COM:8080/a/b/../c?q=1#frag");
        assert_eq!(url.href(), "http://User:Pw@example.com:8080/a/c?q=1#frag");
        assert_eq!(url.origin(), "http://example.com:8080");
        assert_eq!(url.username, "User");
        assert_eq!(url.password, "Pw");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, Some(8080));
        assert_eq!(url.path, "/a/c");
        assert_eq!(
            parse("../d", Some("http://example.com/a/b/c?old#f"))
                .unwrap()
                .href(),
            "http://example.com/a/d"
        );
        assert_eq!(p("https:example.org").href(), "https://example.org/");
        assert_eq!(p("https://example.org\\a").path, "/a");
        assert_eq!(
            parse("../../q", Some("abc://x/y/z/C:/")).unwrap().href(),
            "abc://x/y/q"
        );
    }

    #[test]
    fn implements_uts46_ipv4_and_ipv6_host_algorithms() {
        assert_eq!(p("https://faß.example/").host, "xn--fa-hia.example");
        assert_eq!(p("https://①.②.③.④/").host, "1.2.3.4");
        assert_eq!(p("http://0x7f.1/").host, "127.0.0.1");
        assert_eq!(p("http://[2001:0db8::1]/").host, "[2001:db8::1]");
        assert!(parse("https://exa%23mple.org/", None).is_err());
        assert!(parse("http://[1::1::1]/", None).is_err());

        // URL Standard §3.3 (2026): with beStrict false, an ASCII domain is lowercased and kept
        // even when strict UTS #46 reports invalid Punycode.
        let compatible = p("http://A.B.C.XN--pokxncvks/");
        assert_eq!(compatible.href(), "http://a.b.c.xn--pokxncvks/");
        assert_eq!(compatible.host, "a.b.c.xn--pokxncvks");
        assert_eq!(compatible.origin(), "http://a.b.c.xn--pokxncvks");
        assert_eq!(p("file://xn--/p").href(), "file://xn--/p");
    }

    #[test]
    fn preserves_opaque_paths_and_opaque_origins() {
        let url = p("mailto:some one@example.org?q=hello world#fragment");
        assert_eq!(
            url.href(),
            "mailto:some one@example.org?q=hello%20world#fragment"
        );
        assert_eq!(url.origin(), "null");
        let url = p("data:space    ?test#fragment");
        assert_eq!(url.href(), "data:space   %20?test#fragment");
        assert_eq!(url.path, "space   %20");
        let cleared = mutate(&url.href(), "search", "").unwrap();
        assert_eq!(cleared.href(), "data:space   %20#fragment");
        assert_eq!(
            p("blob:https://example.org/id").origin(),
            "https://example.org"
        );
        assert_eq!(p("blob:blob:https://example.org/id").origin(), "null");
        assert_eq!(p("blob:wss://example.org/id").origin(), "null");
        assert!(parse("sc://a^b", None).is_err());
    }

    #[test]
    fn setters_use_component_state_overrides_and_ignore_invalid_input() {
        let url = mutate("http://example.com/a", "pathname", "next value").unwrap();
        assert_eq!(url.href(), "http://example.com/next%20value");
        let url = mutate(&url.href(), "host", "bücher.example:443").unwrap();
        assert_eq!(url.host, "xn--bcher-kva.example");
        assert_eq!(url.port, Some(443));
        let unchanged = mutate(&url.href(), "port", "70000").unwrap();
        assert_eq!(unchanged.href(), url.href());

        // URL Standard host state step 3.2: an empty hostname setter is a parse failure while a
        // port is present. Servo `url` 2.5.8 otherwise creates an internally invalid URL here.
        let unchanged = mutate("sc://test:12/", "hostname", "").unwrap();
        assert_eq!(unchanged.href(), "sc://test:12/");
        assert_eq!(unchanged.host, "test");
        assert_eq!(unchanged.port, Some(12));

        let unchanged = mutate("https://example.test:3000/", "port", "\n\t").unwrap();
        assert_eq!(unchanged.href(), "https://example.test:3000/");

        let hosted = mutate("non-spec:/.//p", "hostname", "h").unwrap();
        assert_eq!(hosted.href(), "non-spec://h//p");
        assert_eq!(hosted.path, "//p");
        let guarded = mutate("non-spec:/", "pathname", "//p").unwrap();
        assert_eq!(guarded.href(), "non-spec:/.//p");
        assert_eq!(guarded.path, "//p");
        let file = mutate("file://monkey/", "pathname", "\\\\").unwrap();
        assert_eq!(file.href(), "file://monkey//");
        assert_eq!(file.path, "//");
        let empty = mutate("foo:///some/path", "pathname", "").unwrap();
        assert_eq!(empty.href(), "foo://");
        assert_eq!(empty.path, "");

        let compatible = mutate("https://example.test/", "hostname", "XN--").unwrap();
        assert_eq!(compatible.href(), "https://xn--/");
        assert_eq!(compatible.origin(), "https://xn--");
        let compatible = mutate(&compatible.href(), "username", "me").unwrap();
        assert_eq!(compatible.href(), "https://me@xn--/");
    }
}
