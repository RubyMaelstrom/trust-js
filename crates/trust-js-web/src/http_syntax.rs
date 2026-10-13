//! Shared HTTP/1.1 field and message-framing primitives.
//!
//! RFC 9112 requires framing to be derived from the complete field section with one consistent
//! precedence algorithm. Keeping that algorithm here prevents the client and server from making
//! different Content-Length/Transfer-Encoding decisions (the root of request smuggling bugs).

use std::io::{BufRead, Read};

pub(crate) const MAX_HEADER_BYTES: usize = 64 << 10;
pub(crate) const MAX_BODY: u64 = 32 << 20;
const MAX_FIELDS: usize = 1024;
const MAX_LIST_MEMBERS: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BodyFraming {
    None,
    Chunked,
    Length(u64),
    CloseDelimited,
}

pub(crate) fn is_token(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(is_token_byte)
}

/// Fetch header values are byte strings without NUL/CR/LF and without surrounding SP/HTAB.
pub(crate) fn validate_header(name: &str, value: &str) -> Result<(), String> {
    if !is_token(name) {
        return Err(format!("invalid HTTP field name {name:?}"));
    }
    if value.starts_with([' ', '\t']) || value.ends_with([' ', '\t']) {
        return Err(format!("HTTP field {name:?} has unnormalized whitespace"));
    }
    if value.chars().any(|character| {
        character == '\0' || character == '\r' || character == '\n' || character > '\u{ff}'
    }) {
        return Err(format!("invalid HTTP field value for {name:?}"));
    }
    Ok(())
}

pub(crate) fn validate_headers(headers: &[(String, String)]) -> Result<(), String> {
    if headers.len() > MAX_FIELDS {
        return Err("too many HTTP fields".into());
    }
    let mut wire_bytes = 2usize; // terminating CRLF
    for (name, value) in headers {
        validate_header(name, value)?;
        if value
            .bytes()
            .any(|byte| !matches!(byte, b'\t' | b' ' | 0x21..=0x7e | 0x80..=0xff))
        {
            return Err(format!("invalid HTTP/1 field value for {name:?}"));
        }
        wire_bytes = wire_bytes
            .checked_add(name.len())
            .and_then(|length| length.checked_add(value.chars().count()))
            .and_then(|length| length.checked_add(4)) // colon, SP, CRLF
            .ok_or_else(|| "HTTP metadata too large".to_string())?;
        if wire_bytes > MAX_HEADER_BYTES {
            return Err("HTTP metadata too large".into());
        }
    }
    Ok(())
}

pub(crate) fn header_values<'a>(
    headers: &'a [(String, String)],
    name: &'a str,
) -> impl Iterator<Item = &'a str> {
    headers
        .iter()
        .filter(move |(candidate, _)| candidate.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

pub(crate) fn first_header(headers: &[(String, String)], name: &str) -> Option<String> {
    header_values(headers, name).next().map(str::to_string)
}

pub(crate) fn single_header(
    headers: &[(String, String)],
    name: &str,
) -> Result<Option<String>, String> {
    let first = first_header(headers, name);
    if header_values(headers, name).nth(1).is_some() {
        return Err(format!("multiple {name} fields where only one is allowed"));
    }
    Ok(first)
}

/// Parse RFC 9110 §8.6 / RFC 9112 §6.3 Content-Length, including repeated or combined
/// identical values. RFC 9110 §5.6.1.2 requires recipients to ignore a reasonable number of
/// empty list elements; disagreement, non-digits, overflow, and an effectively empty value are
/// fatal.
pub(crate) fn content_length(headers: &[(String, String)]) -> Result<Option<u64>, String> {
    let mut parsed = None;
    let mut saw_field = false;
    let mut members = 0;
    for field in header_values(headers, "content-length") {
        saw_field = true;
        for member in field.split(',') {
            members += 1;
            if members > MAX_LIST_MEMBERS {
                return Err("too many Content-Length list members".into());
            }
            let member = member.trim_matches([' ', '\t']);
            if member.is_empty() {
                continue;
            }
            if !member.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err("invalid Content-Length field".into());
            }
            let value = member
                .parse::<u64>()
                .map_err(|_| "Content-Length is too large".to_string())?;
            if parsed.is_some_and(|previous| previous != value) {
                return Err("conflicting Content-Length fields".into());
            }
            parsed = Some(value);
        }
    }
    if saw_field && parsed.is_none() {
        return Err("invalid Content-Length field".into());
    }
    Ok(parsed)
}

fn transfer_codings(headers: &[(String, String)]) -> Result<Option<Vec<String>>, String> {
    let fields: Vec<_> = header_values(headers, "transfer-encoding").collect();
    if fields.is_empty() {
        return Ok(None);
    }
    let mut codings = Vec::new();
    let mut members = 0;
    for field in fields {
        for member in field.split(',') {
            members += 1;
            if members > MAX_LIST_MEMBERS {
                return Err("too many Transfer-Encoding list members".into());
            }
            let member = member.trim_matches([' ', '\t']);
            if member.is_empty() {
                continue;
            }
            let (name, parameters) = member.split_once(';').unwrap_or((member, ""));
            let name = name.trim_matches([' ', '\t']);
            if !is_token(name) || name.eq_ignore_ascii_case("chunked") && !parameters.is_empty() {
                return Err("invalid Transfer-Encoding field".into());
            }
            codings.push(name.to_ascii_lowercase());
        }
    }
    if codings.is_empty() {
        return Err("empty Transfer-Encoding field".into());
    }
    Ok(Some(codings))
}

/// RFC 9112 §6.3 message body length, narrowed to the transfer codings TRust JS implements.
pub(crate) fn body_framing(
    headers: &[(String, String)],
    request: bool,
    body_forbidden: bool,
) -> Result<BodyFraming, String> {
    if body_forbidden {
        return Ok(BodyFraming::None);
    }
    let codings = transfer_codings(headers)?;
    let length = content_length(headers)?;
    if codings.is_some() && length.is_some() {
        return Err("message contains both Transfer-Encoding and Content-Length".into());
    }
    if let Some(codings) = codings {
        if codings.last().is_none_or(|coding| coding != "chunked") {
            let kind = if request { "request" } else { "response" };
            return Err(format!(
                "unsupported {kind} Transfer-Encoding (final coding is not chunked)"
            ));
        }
        if codings.len() != 1 {
            return Err(format!(
                "unsupported transfer coding before chunked: {}",
                codings[..codings.len() - 1].join(", ")
            ));
        }
        return Ok(BodyFraming::Chunked);
    }
    if let Some(length) = length {
        return Ok(BodyFraming::Length(length));
    }
    Ok(if request {
        BodyFraming::None
    } else {
        BodyFraming::CloseDelimited
    })
}

/// Read one HTTP protocol line as octets, accepting CRLF or the RFC 9112 §2.2 bare-LF
/// robustness exception, while enforcing the aggregate metadata budget before allocation grows.
pub(crate) fn read_line(
    reader: &mut impl BufRead,
    consumed: &mut usize,
    limit: usize,
) -> Result<Vec<u8>, String> {
    if *consumed >= limit {
        return Err("HTTP metadata too large".into());
    }
    let remaining = limit - *consumed;
    let mut line = Vec::new();
    let read = reader
        .take((remaining + 1) as u64)
        .read_until(b'\n', &mut line)
        .map_err(|error| format!("read HTTP line: {error}"))?;
    *consumed = consumed.saturating_add(read);
    if read == 0 {
        return Err("incomplete HTTP field section".into());
    }
    if *consumed > limit {
        return Err("HTTP metadata too large".into());
    }
    if line.pop() != Some(b'\n') {
        return Err("HTTP line is not terminated".into());
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    if line.contains(&b'\r') {
        return Err("bare CR in HTTP protocol element".into());
    }
    Ok(line)
}

pub(crate) fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

fn trim_ows(mut value: &[u8]) -> &[u8] {
    while value
        .first()
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        value = &value[1..];
    }
    while value
        .last()
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        value = &value[..value.len() - 1];
    }
    value
}

/// Parse a complete field section. User agents unfold obs-fold to SP; servers reject it, both
/// choices explicitly permitted/required by RFC 9112 §5.2.
pub(crate) fn read_fields(
    reader: &mut impl BufRead,
    consumed: &mut usize,
    allow_obs_fold: bool,
) -> Result<Vec<(String, String)>, String> {
    let mut headers: Vec<(String, String)> = Vec::new();
    loop {
        let line = read_line(reader, consumed, MAX_HEADER_BYTES)?;
        if line.is_empty() {
            return Ok(headers);
        }
        if headers.len() >= MAX_FIELDS {
            return Err("too many HTTP fields".into());
        }
        if line
            .first()
            .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
        {
            if !allow_obs_fold || headers.is_empty() {
                return Err("obsolete HTTP field line folding".into());
            }
            let continuation = latin1(trim_ows(&line));
            let previous = &mut headers.last_mut().unwrap().1;
            previous.push(' ');
            previous.push_str(&continuation);
            continue;
        }
        let colon = line
            .iter()
            .position(|byte| *byte == b':')
            .ok_or_else(|| "malformed HTTP field line".to_string())?;
        let name_bytes = &line[..colon];
        if !name_bytes.is_ascii() {
            return Err("non-ASCII HTTP field name".into());
        }
        let name = std::str::from_utf8(name_bytes).unwrap().to_string();
        let value_bytes = trim_ows(&line[colon + 1..]);
        if value_bytes
            .iter()
            .any(|byte| !matches!(byte, b'\t' | b' ' | 0x21..=0x7e | 0x80..=0xff))
        {
            return Err("invalid byte in HTTP/1 field value".into());
        }
        let value = latin1(value_bytes);
        validate_header(&name, &value)?;
        headers.push((name, value));
    }
}

pub(crate) fn read_body(
    reader: &mut impl BufRead,
    framing: BodyFraming,
) -> Result<Vec<u8>, String> {
    match framing {
        BodyFraming::None => Ok(Vec::new()),
        BodyFraming::Chunked => read_chunked(reader),
        BodyFraming::Length(length) => {
            if length > MAX_BODY {
                return Err("HTTP body too large".into());
            }
            let mut body = vec![0; length as usize];
            reader
                .read_exact(&mut body)
                .map_err(|error| format!("incomplete HTTP body: {error}"))?;
            Ok(body)
        }
        BodyFraming::CloseDelimited => {
            let mut body = Vec::new();
            reader
                .take(MAX_BODY + 1)
                .read_to_end(&mut body)
                .map_err(|error| format!("read close-delimited body: {error}"))?;
            if body.len() as u64 > MAX_BODY {
                return Err("HTTP body too large".into());
            }
            Ok(body)
        }
    }
}

fn read_chunked(reader: &mut impl BufRead) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    let mut metadata = 0;
    loop {
        let line = read_line(reader, &mut metadata, MAX_HEADER_BYTES)?;
        let (mut size, extension) = line
            .iter()
            .position(|byte| *byte == b';')
            .map_or((line.as_slice(), None), |separator| {
                (&line[..separator], Some(&line[separator + 1..]))
            });
        // `chunk-ext` begins with BWS before its semicolon (RFC 9112 §7.1.1), while leading
        // whitespace and trailing whitespace without an extension remain invalid chunk-size.
        if extension.is_some() {
            while size.last().is_some_and(|byte| matches!(byte, b' ' | b'\t')) {
                size = &size[..size.len() - 1];
            }
        }
        if size.is_empty() || !size.iter().all(u8::is_ascii_hexdigit) {
            return Err("invalid chunk size".into());
        }
        if let Some(extension) = extension {
            validate_chunk_extensions(extension)?;
        }
        let size_text = std::str::from_utf8(size).unwrap();
        let size = u64::from_str_radix(size_text, 16)
            .map_err(|_| "chunk size is too large".to_string())?;
        if size == 0 {
            let _trailers = read_fields(reader, &mut metadata, false)?;
            return Ok(body);
        }
        let new_length = (body.len() as u64)
            .checked_add(size)
            .filter(|length| *length <= MAX_BODY)
            .ok_or_else(|| "HTTP body too large".to_string())?;
        let start = body.len();
        body.resize(new_length as usize, 0);
        reader
            .read_exact(&mut body[start..])
            .map_err(|error| format!("incomplete chunk data: {error}"))?;
        let mut terminator = [0; 2];
        reader
            .read_exact(&mut terminator)
            .map_err(|error| format!("incomplete chunk terminator: {error}"))?;
        if terminator != *b"\r\n" {
            return Err("invalid chunk data terminator".into());
        }
    }
}

fn validate_chunk_extensions(extension: &[u8]) -> Result<(), String> {
    let mut cursor = 0;
    loop {
        skip_bws(extension, &mut cursor);
        let name_start = cursor;
        while extension
            .get(cursor)
            .is_some_and(|byte| is_token_byte(*byte))
        {
            cursor += 1;
        }
        if cursor == name_start {
            return Err("invalid chunk extension name".into());
        }
        skip_bws(extension, &mut cursor);
        if extension.get(cursor) == Some(&b'=') {
            cursor += 1;
            skip_bws(extension, &mut cursor);
            if extension.get(cursor) == Some(&b'"') {
                cursor += 1;
                loop {
                    match extension.get(cursor).copied() {
                        Some(b'"') => {
                            cursor += 1;
                            break;
                        }
                        Some(b'\\') => {
                            cursor += 1;
                            let Some(escaped) = extension.get(cursor).copied() else {
                                return Err("unterminated quoted chunk extension".into());
                            };
                            if !matches!(escaped, b'\t' | b' ' | 0x21..=0x7e | 0x80..=0xff) {
                                return Err("invalid quoted chunk extension".into());
                            }
                            cursor += 1;
                        }
                        Some(b'\t' | b' ' | b'!' | 0x23..=0x5b | 0x5d..=0x7e | 0x80..=0xff) => {
                            cursor += 1;
                        }
                        _ => return Err("invalid quoted chunk extension".into()),
                    }
                }
            } else {
                let value_start = cursor;
                while extension
                    .get(cursor)
                    .is_some_and(|byte| is_token_byte(*byte))
                {
                    cursor += 1;
                }
                if cursor == value_start {
                    return Err("invalid chunk extension value".into());
                }
            }
            skip_bws(extension, &mut cursor);
        }
        if cursor == extension.len() {
            return Ok(());
        }
        if extension.get(cursor) != Some(&b';') {
            return Err("invalid chunk extension separator".into());
        }
        cursor += 1;
    }
}

fn skip_bws(value: &[u8], cursor: &mut usize) {
    while value
        .get(*cursor)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        *cursor += 1;
    }
}

fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

pub(crate) fn field_value_bytes(value: &str) -> Result<Vec<u8>, String> {
    value
        .chars()
        .map(|character| {
            u8::try_from(character as u32).map_err(|_| "HTTP field is not a byte string".into())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn duplicate_content_lengths_must_be_identical() {
        let equal = vec![
            ("Content-Length".into(), "42".into()),
            ("content-length".into(), ", 42, , 042,".into()),
        ];
        assert_eq!(content_length(&equal).unwrap(), Some(42));
        let conflicting = vec![("Content-Length".into(), "42, 43".into())];
        assert!(content_length(&conflicting).is_err());
    }

    #[test]
    fn content_length_rejects_adversarial_values_and_excessive_lists() {
        for value in [
            "",
            " , \t, ",
            "-1",
            "+1",
            "0x10",
            "1 0",
            "1.0",
            "４２",
            "18446744073709551616",
            "42, 43",
        ] {
            assert!(
                content_length(&[("Content-Length".into(), value.into())]).is_err(),
                "accepted {value:?}"
            );
        }

        let excessive = ",".repeat(MAX_LIST_MEMBERS);
        assert!(content_length(&[("Content-Length".into(), excessive)]).is_err());
    }

    #[test]
    fn transfer_encoding_and_content_length_is_rejected() {
        let headers = vec![
            ("Transfer-Encoding".into(), "chunked".into()),
            ("Content-Length".into(), "3".into()),
        ];
        assert!(body_framing(&headers, true, false).is_err());
        assert!(body_framing(&headers, false, false).is_err());
    }

    #[test]
    fn body_framing_follows_rfc_9112_precedence() {
        assert_eq!(body_framing(&[], true, false).unwrap(), BodyFraming::None);
        assert_eq!(
            body_framing(&[], false, false).unwrap(),
            BodyFraming::CloseDelimited
        );
        assert_eq!(
            body_framing(&[("Content-Length".into(), "7".into())], false, false).unwrap(),
            BodyFraming::Length(7)
        );
        assert_eq!(
            body_framing(
                &[("Transfer-Encoding".into(), ", ChUnKeD, ".into())],
                true,
                false
            )
            .unwrap(),
            BodyFraming::Chunked
        );

        // Responses forbidden from carrying a body terminate at the field section regardless
        // of misleading framing fields (RFC 9112 §6.3 steps 1 and 2).
        let ambiguous = vec![
            ("Transfer-Encoding".into(), "chunked".into()),
            ("Content-Length".into(), "3".into()),
        ];
        assert_eq!(
            body_framing(&ambiguous, false, true).unwrap(),
            BodyFraming::None
        );
    }

    #[test]
    fn transfer_encoding_order_and_grammar_are_unambiguous() {
        for value in [
            "",
            " , \t, ",
            "chunked;parameter=value",
            "chunked, chunked",
            "gzip, chunked",
            "chunked, gzip",
            "not a coding",
        ] {
            assert!(
                body_framing(&[("Transfer-Encoding".into(), value.into())], true, false).is_err(),
                "accepted {value:?}"
            );
        }

        let excessive = ",".repeat(MAX_LIST_MEMBERS);
        assert!(body_framing(&[("Transfer-Encoding".into(), excessive)], true, false).is_err());
    }

    #[test]
    fn chunked_requires_crlf_and_enforces_limits() {
        let mut valid = BufReader::new(
            &b"4 \t; name=value; quoted=\"a\\\";b\"; observed=\"\x80\"\r\nWiki\r\n\
               00; done\r\nX-Ok: yes\r\n\r\n"[..],
        );
        assert_eq!(
            read_body(&mut valid, BodyFraming::Chunked).unwrap(),
            b"Wiki"
        );

        for raw in [
            &b"1\r\nxXX0\r\n\r\n"[..],
            &b"1\nx\n0\n\n"[..], // chunk-data still requires its exact CRLF delimiter
            &b" 1\r\nx\r\n0\r\n\r\n"[..],
            &b"1 \r\nx\r\n0\r\n\r\n"[..],
            &b"+1\r\nx\r\n0\r\n\r\n"[..],
            &b"0x1\r\nx\r\n0\r\n\r\n"[..],
            &b"1;=bad\r\nx\r\n0\r\n\r\n"[..],
            &b"1; name=\r\nx\r\n0\r\n\r\n"[..],
            &b"1; name=\"unterminated\r\nx\r\n0\r\n\r\n"[..],
            &b"1; name=\"bad\x01\"\r\nx\r\n0\r\n\r\n"[..],
            &b"1; first;;second\r\nx\r\n0\r\n\r\n"[..],
            &b"1; trailing;\r\nx\r\n0\r\n\r\n"[..],
            &b"1\r\nx\r\n0\r\nBad : trailer\r\n\r\n"[..],
            &b"1\r\nx\r\n0\r\n Folded: trailer\r\n\r\n"[..],
            &b"fffffffffffffffff\r\n"[..],
            &b"2000001\r\n"[..],
        ] {
            let mut reader = BufReader::new(raw);
            assert!(
                read_body(&mut reader, BodyFraming::Chunked).is_err(),
                "accepted {raw:?}"
            );
        }
    }

    #[test]
    fn field_parser_rejects_smuggling_syntax() {
        for raw in [
            &b"Bad : value\r\n\r\n"[..],
            &b" Fold: first\r\n\r\n"[..],
            &b"NoColon\r\n\r\n"[..],
            &b"X: value\rY\r\n\r\n"[..],
            &b"X: one\x01two\r\n\r\n"[..],
        ] {
            let mut reader = BufReader::new(raw);
            assert!(read_fields(&mut reader, &mut 0, false).is_err());
        }
    }

    #[test]
    fn field_parser_handles_only_the_permitted_line_robustness() {
        let mut bare_lf = BufReader::new(&b"One: 1\nTwo:\t2 \t\n\n"[..]);
        assert_eq!(
            read_fields(&mut bare_lf, &mut 0, false).unwrap(),
            vec![("One".into(), "1".into()), ("Two".into(), "2".into())]
        );

        let mut folded = BufReader::new(&b"X: first\r\n\t second \r\n\r\n"[..]);
        assert_eq!(
            read_fields(&mut folded, &mut 0, true).unwrap(),
            vec![("X".into(), "first second".into())]
        );

        let mut folded_request = BufReader::new(&b"X: first\r\n second\r\n\r\n"[..]);
        assert!(read_fields(&mut folded_request, &mut 0, false).is_err());
    }

    #[test]
    fn metadata_and_body_budgets_fail_before_unbounded_reads() {
        let mut oversized_line = vec![b'x'; MAX_HEADER_BYTES];
        oversized_line.push(b'\n');
        assert!(read_line(
            &mut BufReader::new(oversized_line.as_slice()),
            &mut 0,
            MAX_HEADER_BYTES
        )
        .is_err());

        let mut fields = Vec::new();
        for _ in 0..=MAX_FIELDS {
            fields.extend_from_slice(b"X: y\r\n");
        }
        fields.extend_from_slice(b"\r\n");
        assert!(read_fields(&mut BufReader::new(fields.as_slice()), &mut 0, false).is_err());

        assert!(read_body(
            &mut BufReader::new(&b""[..]),
            BodyFraming::Length(MAX_BODY + 1)
        )
        .is_err());
    }
}
