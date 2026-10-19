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

/// Sparse UTF-16-to-UTF-8 index for repeated substrings of an immutable engine string. ASCII
/// runs need no entries: each entry records the cumulative extra bytes after a non-ASCII
/// scalar. An index inside an astral pair maps to a continuation byte and is rejected, leaving
/// the caller's code-unit reconstruction to produce the appropriate lone surrogate.
pub(crate) struct Utf8Index {
    changes: Vec<(u32, u32)>,
    copy_safe: bool,
}

impl Utf8Index {
    /// `max_bytes` is the remaining budget of the existing owning string cache. Keep a small
    /// declined index on exhaustion so repeated slices do not retry an oversized index build.
    #[cold]
    #[inline(never)]
    pub(crate) fn new(s: &crate::lstr::LStr, max_bytes: usize) -> Self {
        let mut index = Self {
            changes: Vec::new(),
            copy_safe: true,
        };
        let max_entries = max_bytes / std::mem::size_of::<(u32, u32)>();
        let mut extra = 0;
        let mut previous_high = false;
        for (byte, c) in s.char_indices() {
            if previous_high && smuggled_low(c).is_some() {
                // Conservatively retain from_units canonicalization for smuggled pairs,
                // including host-created non-canonical input.
                index.copy_safe = false;
                break;
            }
            previous_high = smuggled_high(c).is_some();
            if c.is_ascii() {
                continue;
            }
            let units = if smuggled(c).is_some() {
                1
            } else {
                c.len_utf16()
            };
            extra += c.len_utf8() - units;
            if index.changes.len() == max_entries {
                index.copy_safe = false;
                break;
            }
            if index.changes.len() == index.changes.capacity() {
                let additional =
                    (max_entries - index.changes.len()).min(index.changes.len().max(16));
                index.changes.reserve_exact(additional);
            }
            let byte_end = byte + c.len_utf8();
            // LStr's byte length and capacity fit below its u32 ASCII-hint bit.
            index
                .changes
                .push(((byte_end - extra) as u32, extra as u32));
        }
        if !index.copy_safe {
            index.changes = Vec::new();
        }
        index
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.changes.capacity() * std::mem::size_of::<(u32, u32)>()
    }

    /// Invert the sparse boundary map for an already validated UTF-8 slice of the source.
    pub(crate) fn units_at_byte(&self, s: &str, byte: usize) -> Option<usize> {
        if !self.copy_safe || !s.is_char_boundary(byte) {
            return None;
        }
        let past = self
            .changes
            .partition_point(|&(end, extra)| end as usize + extra as usize <= byte);
        Some(
            byte - past
                .checked_sub(1)
                .map_or(0, |n| self.changes[n].1 as usize),
        )
    }

    /// Only copy complete UTF-8 scalars. Both offsets have already been clamped to the source's
    /// UTF-16 length; checking Rust's boundaries also rejects an interior surrogate-pair index.
    pub(crate) fn range<'a>(&self, s: &'a str, start: usize, end: usize) -> Option<&'a str> {
        if !self.copy_safe {
            return None;
        }
        let byte_offset = |unit| {
            let past = self
                .changes
                .partition_point(|&(end, _)| end as usize <= unit);
            unit + past
                .checked_sub(1)
                .map_or(0, |n| self.changes[n].1 as usize)
        };
        s.get(byte_offset(start)..byte_offset(end))
    }
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
    #[cfg(feature = "gc-heap")]
    return crate::value::interned_string(128 + index as usize, || {
        crate::lstr::LStr::from(index.to_string().as_str())
    });
}

/// The engine string for one UTF-16 code unit, interned for ASCII.
pub fn unit_lstr(unit: u16) -> crate::lstr::LStr {
    if unit >= 128 {
        return unit_str(unit).into();
    }
    #[cfg(feature = "gc-heap")]
    return crate::value::interned_string(unit as usize, || unit_str(unit).into());
}

/// The spec's code-unit-wise string comparison (differs from `str` byte order for strings mixing
/// supplementary-plane characters with U+E000..U+FFFF, and for smuggled surrogates).
pub fn cmp_units(a: &str, b: &str) -> std::cmp::Ordering {
    let mut ia = UnitIter::new(a);
    let mut ib = UnitIter::new(b);
    loop {
        match (ia.next(), ib.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(x), Some(y)) => {
                if x != y {
                    return x.cmp(&y);
                }
            }
        }
    }
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

/// Whether repeating `s` can create an adjacent smuggled surrogate pair that requires the
/// canonicalizing `from_units` pass.  The scan also rejects a source that already contains such a
/// pair; retaining the conservative fallback keeps direct byte copies limited to canonical input.
pub fn needs_repeat_fixup(s: &str, count: usize) -> bool {
    if count == 0 {
        return false;
    }
    let mut prev_high = false;
    for c in s.chars() {
        if prev_high && smuggled_low(c).is_some() {
            return true;
        }
        prev_high = smuggled_high(c).is_some();
    }
    count > 1 && needs_join_fixup(s, s)
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
