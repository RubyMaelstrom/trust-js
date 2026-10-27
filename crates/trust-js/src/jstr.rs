//! UTF-16 string semantics over Rust strings.
//!
//! JS strings are sequences of UTF-16 code units, including lone surrogates, which a Rust `str`
//! cannot hold. Lone surrogates are *smuggled* as the plane-16 private-use scalars
//! `U+10F800 + (unit - 0xD800)` (the same trick as Python's surrogateescape), so any unit
//! sequence round-trips through `Value::Str` while remaining a valid `str`. Everything that
//! speaks spec semantics — `length`, indexing, `charCodeAt`, JSON escapes, relational
//! comparison — goes through this module. A *real* character in the smuggle range
//! (U+10F800..U+10FFFF) is itself represented as its smuggled surrogate PAIR, so the encoding is
//! total: adjacent smuggled high+low is the canonical form of those two code points' character,
//! and a solitary smuggled scalar is always a lone surrogate.

/// First smuggled scalar: encodes the lone surrogate U+D800.
pub const SMUGGLE_BASE: u32 = 0x10F800;

/// If `c` is a smuggled lone surrogate, the surrogate code unit it encodes.
#[inline]
pub fn smuggled(c: char) -> Option<u16> {
    let v = c as u32;
    if (SMUGGLE_BASE..SMUGGLE_BASE + 0x800).contains(&v) {
        Some((v - SMUGGLE_BASE + 0xD800) as u16)
    } else {
        None
    }
}

/// Smuggle a surrogate code unit (0xD800..=0xDFFF) into its private-use scalar.
#[inline]
pub fn smuggle(unit: u16) -> char {
    debug_assert!((0xD800..0xE000).contains(&(unit as u32)));
    char::from_u32(SMUGGLE_BASE + (unit as u32 - 0xD800)).unwrap()
}

/// Whether `s` may hold a scalar in the smuggle range. Each such scalar's UTF-8 form begins with
/// byte 0xF4 (U+10F800 is `F4 8F A0 80`), so most strings are cleared by one byte search.
#[inline]
fn may_hold_smuggle_range(s: &str) -> bool {
    s.as_bytes().contains(&0xF4)
}

/// Host text (a Rust `str`, which holds Unicode scalar values only) in the engine
/// representation: every character in the smuggle range becomes its canonical smuggled pair, so
/// the JS string has the same code points as `s`. The result never contains a lone surrogate.
/// Script source and string values supplied by an embedder enter the engine through this.
pub(crate) fn from_text(s: &str) -> std::borrow::Cow<'_, str> {
    if !may_hold_smuggle_range(s) || !s.chars().any(|c| smuggled(c).is_some()) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if smuggled(c).is_some() {
            let v = c as u32 - 0x10000;
            out.push(smuggle(0xD800 + (v >> 10) as u16));
            out.push(smuggle(0xDC00 + (v & 0x3FF) as u16));
        } else {
            out.push(c);
        }
    }
    std::borrow::Cow::Owned(out)
}

/// [`to_text`] of engine text that may be owned already, copying only what it must.
pub(crate) fn to_text_cow(s: std::borrow::Cow<'_, str>) -> std::borrow::Cow<'_, str> {
    match s {
        std::borrow::Cow::Borrowed(text) => to_text(text),
        std::borrow::Cow::Owned(text) => match to_text(&text) {
            std::borrow::Cow::Borrowed(_) => std::borrow::Cow::Owned(text),
            std::borrow::Cow::Owned(converted) => std::borrow::Cow::Owned(converted),
        },
    }
}

/// An engine string as host text: the conversion of a JS string to a Web IDL `USVString`
/// (§3.2.11). Surrogate pairs, including the smuggled pairs of smuggle-range characters, become
/// their characters, and every lone surrogate becomes U+FFFD.
pub(crate) fn to_text(s: &str) -> std::borrow::Cow<'_, str> {
    if !may_hold_smuggle_range(s) || !s.chars().any(|c| smuggled(c).is_some()) {
        return std::borrow::Cow::Borrowed(s);
    }
    std::borrow::Cow::Owned(
        CodePointIter::new(s)
            .map(|point| char::from_u32(point).unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect(),
    )
}

/// The UTF-16 code units of `s` (smuggled scalars decode to their lone surrogates).
pub fn units(s: &str) -> Vec<u16> {
    // ASCII fast path: units are exactly the bytes (no surrogates, no smuggling possible).
    if s.is_ascii() {
        return s.as_bytes().iter().map(|&b| b as u16).collect();
    }
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        match smuggled(c) {
            Some(u) => out.push(u),
            None => {
                let mut buf = [0u16; 2];
                out.extend_from_slice(c.encode_utf16(&mut buf));
            }
        }
    }
    out
}

/// The UTF-16 length of `s` without materializing the units.
pub fn unit_len(s: &str) -> usize {
    if s.is_ascii() {
        return s.len();
    }
    s.chars()
        .map(|c| {
            if smuggled(c).is_some() {
                1
            } else {
                c.len_utf16()
            }
        })
        .sum()
}

/// Decode code units lazily for bounded character reads. Smuggled surrogates
/// remain individual units; ordinary astral scalars yield their high/low pair.
/// Unlike `units`, this view allocates no storage.
pub(crate) fn unit_iter(s: &str) -> impl Iterator<Item = u16> + '_ {
    s.chars().flat_map(|c| {
        let mut buffer = [0u16; 2];
        let len = match smuggled(c) {
            Some(unit) => {
                buffer[0] = unit;
                1
            }
            None => c.encode_utf16(&mut buffer).len(),
        };
        buffer.into_iter().take(len)
    })
}

pub(crate) fn code_point_from_pair(first: u16, second: Option<u16>) -> u32 {
    if (0xD800..0xDC00).contains(&first) {
        if let Some(second) = second.filter(|unit| (0xDC00..0xE000).contains(unit)) {
            return 0x10000 + ((first as u32 - 0xD800) << 10) + (second as u32 - 0xDC00);
        }
    }
    first as u32
}

/// Rebuild a string from code units: valid surrogate pairs combine into their code point, lone
/// surrogates are smuggled.
pub fn from_units(units: &[u16]) -> String {
    // ASCII fast path: no pairs or smuggling below 0x80.
    if units.iter().all(|&u| u < 0x80) {
        let bytes: Vec<u8> = units.iter().map(|&u| u as u8).collect();
        return String::from_utf8(bytes).unwrap();
    }
    let mut out = String::with_capacity(units.len());
    let mut i = 0;
    while i < units.len() {
        let u = units[i] as u32;
        if (0xD800..0xDC00).contains(&u)
            && i + 1 < units.len()
            && (0xDC00..0xE000).contains(&(units[i + 1] as u32))
        {
            let c = 0x10000 + ((u - 0xD800) << 10) + (units[i + 1] as u32 - 0xDC00);
            if c < SMUGGLE_BASE {
                out.push(char::from_u32(c).unwrap());
            } else {
                // A real character in the smuggle range is stored as its smuggled pair.
                out.push(smuggle(units[i]));
                out.push(smuggle(units[i + 1]));
            }
            i += 2;
            continue;
        }
        if (0xD800..0xE000).contains(&u) {
            out.push(smuggle(units[i]));
        } else {
            out.push(char::from_u32(u).unwrap());
        }
        i += 1;
    }
    out
}

/// The code points of a unit sequence: a surrogate pair is one, and any other unit, a lone
/// surrogate included, is its own (as [`CodePointIter`] reads engine text).
pub(crate) fn code_points_of_units(units: &[u16]) -> Vec<u32> {
    let mut out = Vec::with_capacity(units.len());
    let mut index = 0;
    while index < units.len() {
        let point = code_point_from_pair(units[index], units.get(index + 1).copied());
        index += if point >= 0x10000 { 2 } else { 1 };
        out.push(point);
    }
    out
}

/// Unicode Default Case Conversion of code units (ECMA-262 §22.1.3.28, §22.1.3.30), through
/// `char`'s mappings: each code point maps, one-to-many included, and a lone surrogate stays.
/// `None` when lowering units that hold U+03A3, whose Final_Sigma context `str::to_lowercase`
/// resolves.
pub(crate) fn map_case_units(units: &[u16], upper: bool) -> Option<Vec<u16>> {
    let mut out = Vec::with_capacity(units.len());
    let mut index = 0;
    let mut buffer = [0u16; 2];
    while index < units.len() {
        // ASCII maps within ASCII, one to one.
        if let Ok(byte) = u8::try_from(units[index]) {
            if byte.is_ascii() {
                let mapped = if upper {
                    byte.to_ascii_uppercase()
                } else {
                    byte.to_ascii_lowercase()
                };
                out.push(u16::from(mapped));
                index += 1;
                continue;
            }
        }
        let point = code_point_from_pair(units[index], units.get(index + 1).copied());
        index += if point >= 0x10000 { 2 } else { 1 };
        let Some(c) = char::from_u32(point) else {
            out.push(point as u16);
            continue;
        };
        if upper {
            for mapped in c.to_uppercase() {
                out.extend_from_slice(mapped.encode_utf16(&mut buffer));
            }
        } else if c == '\u{03A3}' {
            return None;
        } else {
            for mapped in c.to_lowercase() {
                out.extend_from_slice(mapped.encode_utf16(&mut buffer));
            }
        }
    }
    Some(out)
}

/// The single-unit string for one code unit.
pub fn unit_str(unit: u16) -> String {
    from_units(&[unit])
}

/// Canonical array-index strings below this bound are interned (see [`index_lstr`]).
pub(crate) const INDEX_STRINGS: u32 = 1024;

/// The canonical decimal string of array index `index` (ECMA-262 Number::toString), shared for
/// indices below [`INDEX_STRINGS`]. Engine strings are immutable values; an interned one has no
/// spare capacity, so an append to it copies.
pub(crate) fn index_lstr(index: u32) -> crate::lstr::LStr {
    if index >= INDEX_STRINGS {
        return crate::lstr::LStr::from(index.to_string().as_str());
    }
    crate::value::interned_string(128 + index as usize, || {
        crate::lstr::LStr::from(index.to_string().as_str())
    })
}

/// The engine string for one UTF-16 code unit, interned for ASCII.
pub fn unit_lstr(unit: u16) -> crate::lstr::LStr {
    if unit >= 128 {
        return crate::lstr::LStr::from_utf16(&[unit]);
    }
    crate::value::interned_string(unit as usize, || crate::lstr::LStr::from_utf16(&[unit]))
}

/// Iterator over `s`'s UTF-16 code units.
pub struct UnitIter<'a> {
    chars: std::str::Chars<'a>,
    trail: Option<u16>,
}

/// Iterator over `s`'s ECMAScript code points. A valid leading/trailing surrogate pair is one
/// item; an unpaired surrogate is returned as its code-unit value. This is the sequence consumed
/// by ECMA-262 `CodePointAt` users such as the String iterator.
pub struct CodePointIter<'a> {
    units: UnitIter<'a>,
    pending: Option<u16>,
}

impl<'a> CodePointIter<'a> {
    pub fn new(s: &'a str) -> Self {
        Self {
            units: UnitIter::new(s),
            pending: None,
        }
    }
}

impl Iterator for CodePointIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let first = self.pending.take().or_else(|| self.units.next())?;
        if (0xD800..0xDC00).contains(&(first as u32)) {
            if let Some(second) = self.units.next() {
                if (0xDC00..0xE000).contains(&(second as u32)) {
                    return Some(
                        0x10000 + ((first as u32 - 0xD800) << 10) + (second as u32 - 0xDC00),
                    );
                }
                self.pending = Some(second);
            }
        }
        Some(first as u32)
    }
}

impl<'a> UnitIter<'a> {
    pub fn new(s: &'a str) -> Self {
        UnitIter {
            chars: s.chars(),
            trail: None,
        }
    }
}

impl Iterator for UnitIter<'_> {
    type Item = u16;
    fn next(&mut self) -> Option<u16> {
        if let Some(t) = self.trail.take() {
            return Some(t);
        }
        let c = self.chars.next()?;
        if let Some(u) = smuggled(c) {
            return Some(u);
        }
        let mut buf = [0u16; 2];
        let enc = c.encode_utf16(&mut buf);
        if enc.len() == 2 {
            self.trail = Some(enc[1]);
        }
        Some(enc[0])
    }
}

/// Whether `s` contains a lone surrogate: a smuggled scalar NOT part of an adjacent
/// high+low pair (which canonically encodes a real smuggle-range character).
pub fn has_lone_surrogate(s: &str) -> bool {
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if smuggled_high(c).is_some() {
            if chars.peek().copied().and_then(smuggled_low).is_some() {
                chars.next();
                continue;
            }
            return true;
        }
        if smuggled(c).is_some() {
            return true;
        }
    }
    false
}

/// If `a` and `b` are a smuggled high+low pair, the real character they encode.
pub fn paired_char(a: char, b: char) -> Option<char> {
    let hi = smuggled_high(a)?;
    let lo = smuggled_low(b)?;
    char::from_u32(0x10000 + ((hi as u32 - 0xD800) << 10) + (lo as u32 - 0xDC00))
}

/// Whether `c` is a smuggled HIGH surrogate (U+D800..U+DBFF).
#[inline]
fn smuggled_high(c: char) -> Option<u16> {
    smuggled(c).filter(|u| (0xD800..0xDC00).contains(&(*u as u32)))
}

/// Whether `c` is a smuggled LOW surrogate (U+DC00..U+DFFF).
#[inline]
fn smuggled_low(c: char) -> Option<u16> {
    smuggled(c).filter(|u| (0xDC00..0xE000).contains(&(*u as u32)))
}

/// String concatenation with the canonical-form fix-up: a smuggled high surrogate at the end of
/// `a` followed by a smuggled low surrogate at the start of `b` combines into the astral scalar,
/// so `"\uD834" + "\uDF06"` equals the literal `"𝌆"`.
pub fn concat(a: &str, b: &str) -> String {
    if let (Some(last), Some(first)) = (a.chars().next_back(), b.chars().next()) {
        if let (Some(hi), Some(lo)) = (smuggled_high(last), smuggled_low(first)) {
            let cp = 0x10000 + ((hi as u32 - 0xD800) << 10) + (lo as u32 - 0xDC00);
            // Adjacent smuggled high+low IS the canonical form for smuggle-range characters.
            if cp < SMUGGLE_BASE {
                let mut out = String::with_capacity(a.len() + b.len());
                out.push_str(&a[..a.len() - last.len_utf8()]);
                out.push(char::from_u32(cp).unwrap());
                out.push_str(&b[first.len_utf8()..]);
                return out;
            }
        }
    }
    let mut out = String::with_capacity(a.len() + b.len());
    out.push_str(a);
    out.push_str(b);
    out
}

/// Whether joining `a + b` needs the smuggled-surrogate canonicalization fix-up (a smuggled
/// high surrogate ending `a` meeting a smuggled low starting `b` — see [`concat`]). Callers
/// building the result allocation directly must take the slow [`concat`] path when this holds.
pub fn needs_join_fixup(a: &str, b: &str) -> bool {
    match (a.chars().next_back(), b.chars().next()) {
        (Some(last), Some(first)) => smuggled_high(last).is_some() && smuggled_low(first).is_some(),
        _ => false,
    }
}

/// Canonicalize any adjacent smuggled high+low surrogate pairs inside `s` (needed after building
/// a string from independently produced pieces, e.g. `join` or `repeat`).
pub fn canonicalize(s: &str) -> Option<String> {
    let mut prev_high = false;
    for c in s.chars() {
        if prev_high && smuggled_low(c).is_some() {
            // Slow path: rebuild through the unit round-trip.
            return Some(from_units(&units(s)));
        }
        prev_high = smuggled_high(c).is_some();
    }
    None
}

/// The code points of `s`, with lone surrogates as their surrogate values (paired smuggled
/// high+low decode to the real character they canonically encode — see module docs).
pub fn code_points(s: &str) -> Vec<u32> {
    CodePointIter::new(s).collect()
}

/// Append one ECMAScript code point to an engine string: a surrogate value as a lone surrogate,
/// a smuggle-range character as its canonical smuggled pair, any other character as itself.
pub(crate) fn push_code_point(out: &mut String, cp: u32) {
    if (0xD800..0xE000).contains(&cp) {
        out.push(smuggle(cp as u16));
    } else if cp >= SMUGGLE_BASE {
        let v = cp - 0x10000;
        out.push(smuggle(0xD800 + (v >> 10) as u16));
        out.push(smuggle(0xDC00 + (v & 0x3FF) as u16));
    } else {
        out.push(char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER));
    }
}

/// Rebuild the engine representation of one ECMAScript code point. Surrogate values remain lone
/// code units and real characters in the smuggling range retain the canonical two-scalar form.
pub fn from_code_point(cp: u32) -> String {
    from_code_points(&[cp])
}

/// Rebuild a string from code points (surrogate values become lone surrogates).
pub fn from_code_points(cps: &[u32]) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(cps.len());
    for &cp in cps {
        if cp < 0x10000 {
            units.push(cp as u16);
        } else {
            let v = cp - 0x10000;
            units.push(0xD800 + (v >> 10) as u16);
            units.push(0xDC00 + (v & 0x3FF) as u16);
        }
    }
    from_units(&units)
}
