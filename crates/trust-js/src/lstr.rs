//! The engine string: UTF-16 code units with spare capacity, in a GC cell.
//!
//! `Value::Str` used to hold `Rc<str>` — immutable and exactly sized, which makes an append loop
//! (`s += x`, astring's `this.output += e`) inherently O(n²): every step materializes a fresh
//! allocation of the whole accumulated string. `LStr` is the classic engine fix (QuickJS's
//! JSString): a `{frontier, len, cap}` header + units, one thin pointer.
//!
//! A string is a cell of the active Agent's heap (P1 S2): a handle is the cell's address,
//! cloning copies it and nothing is counted. Word 0 of a flat buffer records its *frontier*, how
//! many units have been written into it. An append to a string that ends at its buffer's
//! frontier, with spare capacity, writes past the frontier and returns a view of the same
//! buffer; units a string can read never change afterwards, so aliases keep their own length
//! and no uniqueness is needed (Decision 4 as revised at step 4d). Any other append copies into
//! a new buffer with geometric capacity.
//!
//! ECMAScript strings are sequences of UTF-16 code units (ECMA-262 §6.1.4), and a buffer stores
//! them at a width fixed when it is allocated, as V8, SpiderMonkey and JavaScriptCore do: one
//! byte a unit when every unit is ASCII, two otherwise (a lone surrogate is an ordinary unit).
//! A concatenation or in-place append stays one-byte only when both sides are one-byte; a
//! two-byte suffix on a one-byte buffer copies into a two-byte buffer. (The browsers' one-byte
//! width holds Latin-1; here it holds ASCII until the byte-level fast paths read storage by
//! width: the P2 plan's W7, step S4.) Capacity beyond `len` is invisible to every reader, since
//! the readers ([`LStr::units`], [`LStr::as_ascii`], [`LStr::text`]) slice to `len`.
//!
//! Large substrings may instead hold a bounded view of a flat buffer, at its width. A view's
//! header has the same length offset, but no ASCII hint: generated code reads a flat one-byte
//! buffer's bytes only under the hint, and takes its checked path for anything else.
//!
//! `LStr` is neither `Send` nor `Sync`: the engine is one thread per realm.

use std::borrow::Cow;
use std::cell::Cell;
use std::ptr::NonNull;

#[repr(C)]
struct Header {
    /// A flat buffer's frontier, in units (see the module docs); unused by a view.
    frontier: Cell<usize>,
    /// The length in UTF-16 code units.
    len: Cell<u32>,
    /// A flat buffer's capacity in units below its width and hint bits; a view's is
    /// [`VIEW_CAP`].
    cap: Cell<u32>,
    // `cap` units of one or two bytes follow.
}

#[repr(C)]
struct View {
    header: Header,
    owner: LStr,
    /// The first unit's index in the owner.
    offset: usize,
}

/// See the module docs. `repr(transparent)`: one pointer, which the embedder's scoped strings
/// (`scope::JsString`) are too.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct LStr {
    p: NonNull<Header>,
}

/// A string's stored code units: one byte a unit (ASCII) or two.
#[derive(Clone, Copy)]
pub(crate) enum Units<'a> {
    Ascii(&'a [u8]),
    Utf16(&'a [u16]),
}

const HDR: usize = std::mem::size_of::<Header>();

/// Byte offset of the length within the header — the JIT's inline equality/truthiness templates
/// read `len` from machine code through the stored pointer.
pub(crate) const LEN_OFF: usize = std::mem::offset_of!(Header, len);
/// Byte offset of `cap` (which carries [`ASCII_HINT`] in its top bit) — the JIT's charCodeAt
/// intrinsic tests the hint from machine code.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) const CAP_OFF: usize = std::mem::offset_of!(Header, cap);
/// Byte offset of the first content unit in a flat buffer. Views require checked resolution.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) const DATA_OFF: usize = HDR;
/// Top bit of `cap`: a flat one-byte buffer, whose units are ASCII bytes (byte index == UTF-16
/// unit index, and every byte IS its unit). Set on every one-byte buffer and on nothing else; a
/// view of one does not carry it.
pub(crate) const ASCII_HINT: u32 = 1 << 31;
/// A flat two-byte buffer.
const TWO_BYTE: u32 = 1 << 30;
/// The capacity bits of a flat buffer's `cap`.
const CAP_BITS: u32 = TWO_BYTE - 1;

// Reserve one capacity value, rather than changing the native count/length layout. No flat
// buffer reaches it: a two-byte buffer's capacity is at most MAX_LEN, below CAP_BITS, and a
// one-byte buffer's carries ASCII_HINT. A view is never ASCII-hinted even when its units are.
const VIEW_CAP: u32 = ASCII_HINT - 1;
/// The longest content, in units, an engine string can hold; constructing a longer one panics.
pub(crate) const MAX_LEN: usize = CAP_BITS as usize - 1;
/// Slices shorter than this many bytes of content copy.
const MIN_VIEW_BYTES: usize = 256;
/// A view may keep an owner of at most this size alive regardless of the slice's length; larger
/// owners are pinned only by slices covering at least half of them. Copying every short slice
/// of a moderately sized document made incremental scanners (repeated `substring(0, n)` +
/// sticky RegExp exec) quadratic in both the copy and the matcher's text preparation.
const VIEW_PIN_BYTES: usize = 1 << 20;
const _: () = assert!(MIN_VIEW_BYTES > 10);

/// The descriptor of string cells: a view's one edge is its flat owner.
pub(crate) fn cell_layout() -> crate::gc_core::LayoutDesc {
    fn trace(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
        // SAFETY: string cells are initialized before any collection can trace them; a view
        // is recognized by its reserved capacity.
        unsafe {
            let header = &*cell.cast::<Header>();
            if header.cap.get() == VIEW_CAP {
                visit((*cell.cast::<View>()).owner.p.as_ptr().cast::<u8>());
            }
        }
    }
    crate::gc_core::LayoutDesc {
        name: "string",
        trace,
        mark_trace: None,
        finalize: None,
        // A view's owner is set when the view is made.
        carded: false,
    }
}

impl<'a> Units<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            Units::Ascii(bytes) => bytes.len(),
            Units::Utf16(units) => units.len(),
        }
    }

    /// Code unit `index`, if in range.
    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<u16> {
        match self {
            Units::Ascii(bytes) => bytes.get(index).map(|&byte| u16::from(byte)),
            Units::Utf16(units) => units.get(index).copied(),
        }
    }

    /// Units `start..end`.
    #[inline]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Units<'a> {
        match *self {
            Units::Ascii(bytes) => Units::Ascii(&bytes[start..end]),
            Units::Utf16(units) => Units::Utf16(&units[start..end]),
        }
    }

    /// The units as a UTF-16 slice: borrowed from two-byte storage, widened from one-byte.
    pub(crate) fn to_utf16(&self) -> Cow<'a, [u16]> {
        match *self {
            Units::Ascii(bytes) => Cow::Owned(bytes.iter().map(|&byte| u16::from(byte)).collect()),
            Units::Utf16(units) => Cow::Borrowed(units),
        }
    }

    /// The units one by one.
    #[inline]
    pub(crate) fn iter(&self) -> UnitsIter<'a> {
        match *self {
            Units::Ascii(bytes) => UnitsIter::Ascii(bytes.iter()),
            Units::Utf16(units) => UnitsIter::Utf16(units.iter()),
        }
    }

    /// Whether every unit is ASCII.
    fn is_ascii(&self) -> bool {
        match self {
            Units::Ascii(_) => true,
            Units::Utf16(units) => units.iter().all(|&unit| unit < 0x80),
        }
    }
}

/// [`Units::iter`].
#[derive(Clone)]
pub(crate) enum UnitsIter<'a> {
    Ascii(std::slice::Iter<'a, u8>),
    Utf16(std::slice::Iter<'a, u16>),
}

impl Iterator for UnitsIter<'_> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        match self {
            UnitsIter::Ascii(bytes) => bytes.next().map(|&byte| u16::from(byte)),
            UnitsIter::Utf16(units) => units.next().copied(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            UnitsIter::Ascii(bytes) => bytes.size_hint(),
            UnitsIter::Utf16(units) => units.size_hint(),
        }
    }
}

impl ExactSizeIterator for UnitsIter<'_> {}

impl LStr {
    /// An empty flat buffer for `cap` units at the given width.
    fn alloc_flat(two_byte: bool, cap: usize) -> LStr {
        assert!(cap <= MAX_LEN, "string too large");
        let bytes = HDR + if two_byte { cap * 2 } else { cap };
        unsafe {
            let p = crate::value::alloc_string_cell(bytes) as *mut Header;
            let p = NonNull::new(p).expect("allocation failed");
            let flags = if two_byte { TWO_BYTE } else { ASCII_HINT };
            p.as_ptr().write(Header {
                frontier: Cell::new(0),
                len: Cell::new(0),
                cap: Cell::new(cap as u32 | flags),
            });
            LStr { p }
        }
    }

    /// Write `units` at unit `at` of this flat buffer, widening ASCII bytes into a two-byte one.
    ///
    /// # Safety
    /// `at + units.len()` is within capacity, and no string reads units at or past `at`: the
    /// buffer is fresh, or `at` is its frontier.
    unsafe fn write_units(&self, at: usize, units: Units<'_>) {
        let data = self.p.as_ptr().cast::<u8>().add(HDR);
        if self.hdr().cap.get() & TWO_BYTE != 0 {
            let dst = data.cast::<u16>().add(at);
            match units {
                Units::Ascii(bytes) => {
                    for (index, &byte) in bytes.iter().enumerate() {
                        dst.add(index).write(u16::from(byte));
                    }
                }
                Units::Utf16(units) => {
                    std::ptr::copy_nonoverlapping(units.as_ptr(), dst, units.len())
                }
            }
        } else {
            match units {
                Units::Ascii(bytes) => {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.add(at), bytes.len())
                }
                Units::Utf16(_) => unreachable!("two-byte units into a one-byte buffer"),
            }
        }
    }

    /// Set a fresh, unpublished flat string's length after writing its units, and its frontier
    /// to match.
    #[inline]
    fn set_fresh_len(&self, len: usize) {
        let h = self.hdr();
        h.len.set(len as u32);
        h.frontier.set(len);
    }

    /// A flat string of `units`, at the narrowest width that holds them.
    fn from_units_at_width(units: Units<'_>) -> LStr {
        let two_byte = !units.is_ascii();
        let s = LStr::alloc_flat(two_byte, units.len());
        match units {
            Units::Utf16(wide) if !two_byte => {
                // SAFETY: fresh, sized for every unit; each is ASCII.
                let data = unsafe { s.p.as_ptr().cast::<u8>().add(HDR) };
                for (index, &unit) in wide.iter().enumerate() {
                    unsafe { data.add(index).write(unit as u8) };
                }
            }
            // SAFETY: fresh and sized for `units`.
            _ => unsafe { s.write_units(0, units) },
        }
        s.set_fresh_len(units.len());
        s
    }

    /// The string of these UTF-16 code units.
    pub(crate) fn from_utf16(units: &[u16]) -> LStr {
        LStr::from_units_at_width(Units::Utf16(units))
    }

    /// The string of engine text `s` (UTF-8 with lone surrogates smuggled; see
    /// [`crate::jstr`]).
    fn from_text(s: &str) -> LStr {
        if s.is_ascii() {
            LStr::from_units_at_width(Units::Ascii(s.as_bytes()))
        } else {
            LStr::from_utf16(&crate::jstr::units(s))
        }
    }

    /// Whether the content is KNOWN all-ASCII (see [`ASCII_HINT`]): a flat one-byte buffer.
    #[inline]
    pub(crate) fn ascii_hint(&self) -> bool {
        self.hdr().cap.get() & ASCII_HINT != 0
    }

    /// Whether the units are ASCII: the string is one-byte, a flat buffer or a view of one.
    #[inline]
    pub(crate) fn known_ascii(&self) -> bool {
        !self.is_two_byte()
    }

    /// Whether the string's units are stored two bytes each.
    #[inline]
    fn is_two_byte(&self) -> bool {
        self.flat().0.hdr().cap.get() & TWO_BYTE != 0
    }

    /// The concatenation of engine-text pieces into one allocation.
    fn concat_text(parts: &[&str]) -> LStr {
        if parts.iter().all(|part| part.is_ascii()) {
            let total = parts
                .iter()
                .try_fold(0usize, |total, part| total.checked_add(part.len()))
                .expect("string too large");
            let s = LStr::alloc_flat(false, total);
            let mut at = 0;
            for part in parts {
                // SAFETY: fresh, sized for every part.
                unsafe { s.write_units(at, Units::Ascii(part.as_bytes())) };
                at += part.len();
            }
            s.set_fresh_len(total);
            return s;
        }
        let units: Vec<u16> = parts
            .iter()
            .flat_map(|part| crate::jstr::unit_iter(part))
            .collect();
        LStr::from_utf16(&units)
    }

    /// The three pieces concatenated into one engine allocation.
    #[inline]
    pub(crate) fn concat3(a: &str, b: &str, c: &str) -> LStr {
        LStr::concat_text(&[a, b, c])
    }

    /// The four pieces concatenated into one engine allocation.
    #[inline]
    pub(crate) fn concat4(a: &str, b: &str, c: &str, d: &str) -> LStr {
        LStr::concat_text(&[a, b, c, d])
    }

    /// Concatenate already-coerced strings, `total` units in all, into one engine allocation.
    pub(crate) fn concat_many(parts: &[LStr], total: usize) -> LStr {
        let s = LStr::alloc_flat(parts.iter().any(LStr::is_two_byte), total);
        let mut at = 0;
        for part in parts {
            let units = part.units();
            assert!(at + units.len() <= total, "parts exceed the total");
            // SAFETY: fresh, and the check above keeps every part within capacity.
            unsafe { s.write_units(at, units) };
            at += units.len();
        }
        debug_assert_eq!(at, total);
        s.set_fresh_len(at);
        s
    }

    #[inline]
    fn hdr(&self) -> &Header {
        unsafe { self.p.as_ref() }
    }

    #[inline]
    fn view(&self) -> Option<&View> {
        (self.hdr().cap.get() == VIEW_CAP).then(|| unsafe { &*self.p.as_ptr().cast::<View>() })
    }

    /// The flat buffer this string reads and its first unit's index there.
    #[inline]
    fn flat(&self) -> (LStr, usize) {
        match self.view() {
            // Views always own a flat buffer; nested slices flatten at construction.
            Some(view) => (view.owner, view.offset),
            None => (*self, 0),
        }
    }

    pub(crate) fn view_owner(&self) -> Option<&LStr> {
        self.view().map(|view| &view.owner)
    }

    /// A view's owner and its unit range there.
    pub(crate) fn view_range(&self) -> Option<(&LStr, usize, usize)> {
        self.view()
            .map(|view| (&view.owner, view.offset, view.offset + self.raw_len()))
    }

    /// Every unit written to this flat buffer (up to its frontier), as one string, which each
    /// of the buffer's views reads a range of, when in-place appends have run past the
    /// buffer's own length; `None` when the buffer is its own extent.
    pub(crate) fn buffer_extent(&self) -> Option<LStr> {
        let (buffer, _) = self.flat();
        let frontier = buffer.hdr().frontier.get();
        (frontier != buffer.raw_len()).then(|| LStr::new_view(buffer, 0, frontier))
    }

    /// Whether in-place appends have written this flat buffer past its own length (see
    /// [`Self::buffer_extent`]).
    pub(crate) fn written_past_length(&self) -> bool {
        let (buffer, _) = self.flat();
        buffer.hdr().frontier.get() != buffer.raw_len()
    }

    /// Units `start..end`. Copy small slices (at the narrowest width) and ranges that would pin
    /// a large owner (over `VIEW_PIN_BYTES`) more than twice their size; other slices share
    /// their flat source owner, at its width.
    pub(crate) fn slice_units(&self, start: usize, end: usize) -> LStr {
        assert!(start <= end && end <= self.raw_len(), "slice out of range");
        if start == 0 && end == self.raw_len() {
            return *self;
        }
        let (owner, offset) = self.flat();
        let bytes = if owner.is_two_byte() {
            (end - start) * 2
        } else {
            end - start
        };
        let owner_bytes = owner.allocation_requested_bytes();
        if bytes < MIN_VIEW_BYTES
            || (owner_bytes > bytes.saturating_mul(2) && owner_bytes > VIEW_PIN_BYTES)
        {
            return LStr::from_units_at_width(self.units().slice(start, end));
        }
        LStr::new_view(owner, offset + start, end - start)
    }

    /// A view of `len` units of flat `owner` from unit `offset`.
    fn new_view(owner: LStr, offset: usize, len: usize) -> LStr {
        debug_assert!(owner.view().is_none(), "views own flat buffers");
        let view = View {
            header: Header {
                frontier: Cell::new(0),
                len: Cell::new(len as u32),
                cap: Cell::new(VIEW_CAP),
            },
            owner,
            offset,
        };
        let p = {
            let cell = crate::value::alloc_string_cell(std::mem::size_of::<View>()).cast::<View>();
            // SAFETY: a fresh cell sized and aligned for a view.
            unsafe { cell.write(view) };
            cell.cast::<Header>()
        };
        LStr {
            p: unsafe { NonNull::new_unchecked(p) },
        }
    }

    /// The string's cell, as a GC edge.
    #[inline]
    pub(crate) fn cell_word(&self) -> u64 {
        self.p.as_ptr() as u64
    }

    /// The stored code units, borrowed from the cell.
    #[inline]
    pub(crate) fn units(&self) -> Units<'_> {
        let (buffer, offset) = self.flat();
        let len = self.raw_len();
        // SAFETY: a flat buffer's units below its frontier are initialized and never change;
        // the buffer and each of its views read within them.
        unsafe {
            let data = buffer.p.as_ptr().cast::<u8>().add(HDR);
            if buffer.hdr().cap.get() & TWO_BYTE != 0 {
                Units::Utf16(std::slice::from_raw_parts(
                    data.cast::<u16>().add(offset),
                    len,
                ))
            } else {
                Units::Ascii(std::slice::from_raw_parts(data.add(offset), len))
            }
        }
    }

    /// Code unit `index`, if in range.
    #[inline]
    pub(crate) fn unit_at(&self, index: usize) -> Option<u16> {
        self.units().get(index)
    }

    /// The code point at unit `index` (CodePointAt's value, ECMA-262 §11.1.4): a surrogate pair
    /// starting there decodes, and any other unit, a lone surrogate included, is its own.
    pub(crate) fn code_point_at(&self, index: usize) -> Option<u32> {
        let units = self.units();
        let first = units.get(index)?;
        Some(crate::jstr::code_point_from_pair(
            first,
            units.get(index + 1),
        ))
    }

    /// The order of IsLessThan for two Strings (ECMA-262 §7.2.13): code unit by code unit, a
    /// proper prefix first.
    pub(crate) fn cmp_units(&self, other: &LStr) -> std::cmp::Ordering {
        match (self.units(), other.units()) {
            (Units::Ascii(a), Units::Ascii(b)) => a.cmp(b),
            (Units::Utf16(a), Units::Utf16(b)) => a.cmp(b),
            (a, b) => a.iter().cmp(b.iter()),
        }
    }

    /// The content as engine text: UTF-8 with lone surrogates smuggled as plane-16 scalars (see
    /// [`crate::jstr`]). Borrowed from a one-byte string; a two-byte string converts.
    #[inline]
    pub fn text(&self) -> Cow<'_, str> {
        match self.units() {
            // SAFETY: one-byte units are ASCII.
            Units::Ascii(bytes) => Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(bytes) }),
            Units::Utf16(units) => Cow::Owned(crate::jstr::from_units(units)),
        }
    }

    /// The content as `&str`, without copying, when the string is one-byte: then engine text,
    /// Latin-1 and UTF-16 code units agree, byte for unit. `None` for a two-byte string, even
    /// one whose units happen to be ASCII.
    #[inline]
    pub fn as_ascii(&self) -> Option<&str> {
        match self.units() {
            // SAFETY: one-byte units are ASCII.
            Units::Ascii(bytes) => Some(unsafe { std::str::from_utf8_unchecked(bytes) }),
            Units::Utf16(_) => None,
        }
    }

    /// The length in UTF-16 code units (ECMAScript's `length`).
    #[inline]
    pub fn unit_len(&self) -> usize {
        self.raw_len()
    }

    /// Whether the string is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hdr().len.get() == 0
    }

    /// The length, in units.
    #[inline]
    fn raw_len(&self) -> usize {
        self.hdr().len.get() as usize
    }

    /// The flat buffer this string reads: a view's owner, else itself. An in-place append keeps
    /// it (the result is a new view of the same buffer).
    #[cfg(test)]
    pub(crate) fn buffer_ptr(&self) -> *const u8 {
        self.view_owner().unwrap_or(self).as_ptr()
    }

    /// Pointer identity (cache keys): the string's cell address.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.p.as_ptr() as *const u8
    }

    #[inline]
    pub fn ptr_eq(a: &LStr, b: &LStr) -> bool {
        a.p == b.p
    }

    #[inline]
    #[track_caller]
    pub fn strong_count(&self) -> usize {
        panic!("LStr::strong_count: gc-heap strings have no reference counts");
    }

    /// Requested bytes retained by this allocation, excluding allocator rounding.
    ///
    /// This includes TRust JS's stable string header because [`LStr`] allocates that header and its
    /// unit capacity as one explicit allocation. It intentionally does not include allocator
    /// metadata outside that request.
    pub(crate) fn retained_requested_bytes(&self) -> usize {
        self.allocation_requested_bytes().saturating_add(
            self.view_owner()
                .map_or(0, LStr::allocation_requested_bytes),
        )
    }

    /// This allocation alone. Memory census follows a view's owner separately to deduplicate
    /// multiple views and other roots; cache admission charges the entire retained owner.
    pub(crate) fn allocation_requested_bytes(&self) -> usize {
        if self.view().is_some() {
            return std::mem::size_of::<View>();
        }
        let cap = self.hdr().cap.get();
        let units = (cap & CAP_BITS) as usize;
        HDR.saturating_add(if cap & TWO_BYTE != 0 {
            units * 2
        } else {
            units
        })
    }

    /// Append in place when this string ends at its buffer's frontier, the capacity suffices
    /// and the buffer's width holds the units: they go past the frontier, where no string reads,
    /// and `self` becomes a view of the longer content. Returns false (without modifying
    /// anything) otherwise — the caller copies.
    fn append_units_in_place(&mut self, right: Units<'_>) -> bool {
        // Frontier append: only the string ending where its buffer's writes end may extend it.
        let (buffer, offset) = self.flat();
        let len = self.raw_len();
        let h = buffer.hdr();
        let frontier = h.frontier.get();
        let cap = h.cap.get();
        if offset + len != frontier
            || frontier + right.len() > (cap & CAP_BITS) as usize
            || (cap & TWO_BYTE == 0 && matches!(right, Units::Utf16(_)))
        {
            return false;
        }
        // SAFETY: past the frontier, within capacity: no string can read these units yet. A
        // right operand sharing this buffer reads below the frontier.
        unsafe { buffer.write_units(frontier, right) };
        h.frontier.set(frontier + right.len());
        *self = LStr::new_view(buffer, offset, len + right.len());
        true
    }

    /// `self` + `right` in a new buffer with growth capacity, two-byte when either is. Doubles
    /// (at least) so a rebuilt accumulator amortizes the next appends.
    fn grown(&self, right: Units<'_>) -> LStr {
        let len = self.raw_len();
        let need = len.checked_add(right.len()).expect("string too large");
        assert!(need <= MAX_LEN, "string too large");
        let two_byte = self.is_two_byte() || matches!(right, Units::Utf16(_));
        let s = LStr::alloc_flat(two_byte, need.saturating_mul(2).clamp(32, MAX_LEN));
        // SAFETY: fresh, with capacity for both.
        unsafe {
            s.write_units(0, self.units());
            s.write_units(len, right);
        }
        s.set_fresh_len(need);
        s
    }

    /// Append engine text in place (see [`Self::append_units_in_place`]).
    pub fn append_in_place(&mut self, x: &str) -> bool {
        if x.is_ascii() {
            self.append_units_in_place(Units::Ascii(x.as_bytes()))
        } else {
            self.append_units_in_place(Units::Utf16(&crate::jstr::units(x)))
        }
    }

    /// `self + x` with growth capacity, for engine text `x`.
    pub fn concat_grown(&self, x: &str) -> LStr {
        if x.is_ascii() {
            self.grown(Units::Ascii(x.as_bytes()))
        } else {
            self.grown(Units::Utf16(&crate::jstr::units(x)))
        }
    }

    /// ECMAScript code-unit concatenation after the caller's coercions and length check.
    /// Shared by all execution tiers and String#concat. A left string that ends at its buffer's
    /// frontier appends there, amortized by geometric capacity, whatever aliases it has
    /// (`s + s` included).
    pub(crate) fn concat_owned(mut self, right: &LStr) -> LStr {
        self.append_or_replace(right);
        self
    }

    /// Append without temporarily removing an owning binding. Build a replacement before
    /// assigning it, so allocation failure cannot strand a cleared binding. A string that ends
    /// at its buffer's frontier, with spare capacity of a width that holds the right operand,
    /// writes past it (units no string reads) and becomes a view of the longer content; any
    /// other keeps its contents and is replaced.
    pub(crate) fn append_or_replace(&mut self, right: &LStr) {
        if right.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = *right;
            return;
        }
        let units = right.units();
        if !self.append_units_in_place(units) {
            *self = self.grown(units);
        }
    }

    /// Repeat the units directly into one engine allocation, at this string's width.
    ///
    /// The caller has already performed the observable `String.prototype.repeat` coercions and
    /// length check (ECMA-262 §22.1.3.18).  Keeping this representation-local avoids first
    /// building a Rust `String` and then copying it into an `LStr`.  The doubling copy uses
    /// `ptr::copy`, which is explicitly overlap-safe as each completed prefix becomes the source
    /// for the next block.
    pub(crate) fn repeat_direct(&self, count: usize) -> LStr {
        let source_len = self.raw_len();
        let total = source_len
            .checked_mul(count)
            .expect("repeat length checked by caller");
        let two_byte = self.is_two_byte();
        let repeated = LStr::alloc_flat(two_byte, total);
        if total == 0 {
            return repeated;
        }
        let width = if two_byte { 2 } else { 1 };
        unsafe {
            repeated.write_units(0, self.units());
            let destination = repeated.p.as_ptr().cast::<u8>().add(HDR);
            let mut filled = source_len;
            while filled < total {
                let copy_len = filled.min(total - filled);
                std::ptr::copy(
                    destination,
                    destination.add(filled * width),
                    copy_len * width,
                );
                filled += copy_len;
            }
        }
        repeated.set_fresh_len(total);
        repeated
    }

    /// Build an ASCII-padded string directly in its final engine allocation. The caller has
    /// already applied StringPad's UTF-16 length and coercion rules, and both operands are known
    /// ASCII, so repeating/truncating the fill cannot split a surrogate pair.
    pub(crate) fn pad_ascii(source: &str, pad: &str, need: usize, at_start: bool) -> LStr {
        debug_assert!(source.is_ascii() && pad.is_ascii() && !pad.is_empty());
        let total = source.len() + need;
        let padded = LStr::alloc_flat(false, total);
        unsafe {
            let destination = padded.p.as_ptr().cast::<u8>().add(HDR);
            let (fill_start, source_start) = if at_start {
                (0, need)
            } else {
                (source.len(), 0)
            };
            let mut offset = fill_start;
            while offset < fill_start + need {
                let copy_len = (fill_start + need - offset).min(pad.len());
                std::ptr::copy_nonoverlapping(pad.as_ptr(), destination.add(offset), copy_len);
                offset += copy_len;
            }
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                destination.add(source_start),
                source.len(),
            );
        }
        padded.set_fresh_len(total);
        padded
    }

    /// Map ASCII letters directly into one engine allocation. The caller must have already
    /// selected the ASCII representation; non-ASCII case mappings (including one-to-many and
    /// context-sensitive mappings) remain on the Unicode implementation path.
    pub(crate) fn map_ascii_case(&self, upper: bool) -> LStr {
        let Units::Ascii(source) = self.units() else {
            panic!("map_ascii_case of a two-byte string");
        };
        let mapped = LStr::alloc_flat(false, source.len());
        unsafe {
            let destination = mapped.p.as_ptr().cast::<u8>().add(HDR);
            for (index, &byte) in source.iter().enumerate() {
                let byte = if upper {
                    byte.to_ascii_uppercase()
                } else {
                    byte.to_ascii_lowercase()
                };
                destination.add(index).write(byte);
            }
        }
        mapped.set_fresh_len(source.len());
        mapped
    }
}

impl From<&str> for LStr {
    fn from(s: &str) -> LStr {
        LStr::from_text(s)
    }
}

impl From<String> for LStr {
    fn from(s: String) -> LStr {
        LStr::from_text(&s)
    }
}

impl From<std::rc::Rc<str>> for LStr {
    fn from(s: std::rc::Rc<str>) -> LStr {
        LStr::from_text(&s)
    }
}

impl From<&String> for LStr {
    fn from(s: &String) -> LStr {
        LStr::from_text(s)
    }
}

impl From<char> for LStr {
    fn from(c: char) -> LStr {
        LStr::from_text(c.encode_utf8(&mut [0u8; 4]))
    }
}

impl PartialEq for LStr {
    fn eq(&self, other: &LStr) -> bool {
        if LStr::ptr_eq(self, other) {
            return true;
        }
        if self.raw_len() != other.raw_len() {
            return false;
        }
        match (self.units(), other.units()) {
            (Units::Ascii(a), Units::Ascii(b)) => a == b,
            (Units::Utf16(a), Units::Utf16(b)) => a == b,
            (Units::Ascii(a), Units::Utf16(b)) | (Units::Utf16(b), Units::Ascii(a)) => a
                .iter()
                .zip(b)
                .all(|(&byte, &unit)| u16::from(byte) == unit),
        }
    }
}
impl Eq for LStr {}

/// Equality with engine text.
impl PartialEq<str> for LStr {
    fn eq(&self, other: &str) -> bool {
        match self.units() {
            Units::Ascii(bytes) => bytes == other.as_bytes(),
            Units::Utf16(units) => units.iter().copied().eq(crate::jstr::unit_iter(other)),
        }
    }
}

impl std::hash::Hash for LStr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Equal strings hash equally at either width: ASCII content goes in as its bytes and
        // any other as little-endian units, in the same blocks whichever way it is stored.
        const BLOCK: usize = 64;
        let mut block = [0u8; BLOCK];
        match self.units() {
            Units::Ascii(bytes) => {
                for chunk in bytes.chunks(BLOCK) {
                    state.write(chunk);
                }
            }
            Units::Utf16(units) if units.iter().all(|&unit| unit < 0x80) => {
                for chunk in units.chunks(BLOCK) {
                    for (byte, &unit) in block.iter_mut().zip(chunk) {
                        *byte = unit as u8;
                    }
                    state.write(&block[..chunk.len()]);
                }
            }
            Units::Utf16(units) => {
                for chunk in units.chunks(BLOCK / 2) {
                    for (index, &unit) in chunk.iter().enumerate() {
                        block[2 * index..2 * index + 2].copy_from_slice(&unit.to_le_bytes());
                    }
                    state.write(&block[..chunk.len() * 2]);
                }
            }
        }
        state.write_u8(0xff);
    }
}

impl std::fmt::Display for LStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text())
    }
}

impl std::fmt::Debug for LStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.text(), f)
    }
}

impl From<&LStr> for std::rc::Rc<str> {
    fn from(s: &LStr) -> std::rc::Rc<str> {
        std::rc::Rc::from(&*s.text())
    }
}

impl From<LStr> for std::rc::Rc<str> {
    fn from(s: LStr) -> std::rc::Rc<str> {
        std::rc::Rc::from(&*s.text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-byte flat buffer holding `text` with `cap` units of capacity.
    fn ascii_with_capacity(text: &str, cap: usize) -> LStr {
        let s = LStr::alloc_flat(false, cap);
        unsafe { s.write_units(0, Units::Ascii(text.as_bytes())) };
        s.set_fresh_len(text.len());
        s
    }

    fn hash_of(s: &LStr) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut state = crate::fasthash::FxHasher::default();
        s.hash(&mut state);
        state.finish()
    }

    #[test]
    fn owned_concat_reuses_unique_capacity_and_preserves_aliases() {
        let mut s = LStr::from("a").concat_owned(&LStr::from("b"));
        let p = s.buffer_ptr();
        s = s.concat_owned(&LStr::from("c"));
        assert_eq!(s.buffer_ptr(), p);
        assert_eq!(s.text(), "abc");
        // A frontier append returns a view, which carries no hint.
        assert!(s.ascii_hint() || s.view_owner().is_some());
        let alias = s;
        s = s.concat_owned(&LStr::from("d"));
        assert!(!LStr::ptr_eq(&s, &alias));
        assert_eq!(alias.text(), "abc");
        assert_eq!(s.text(), "abcd");
        let right = s;
        s = s.concat_owned(&right);
        assert_eq!(right.text(), "abcd");
        assert_eq!(s.text(), "abcdabcd");
    }

    /// P1-S0e oracle: JS `s += x` accumulation must stay amortized after
    /// the GC string migration (plan Decision 4). Geometric capacity
    /// growth is the observable proxy: a quadratic replacement that
    /// reallocates exact-size per append fails the capacity bound, and a
    /// uniqueness-based one fails the aliasing assertions above.
    #[test]
    fn accumulation_scaling_oracle_geometric_capacity_and_aliasing() {
        for chunk in [LStr::from("abcdefgh"), LStr::from("abcdefg\u{e9}")] {
            let mut s = LStr::from("");
            let mut alias_at_1k = None;
            for i in 0..10_000 {
                s = s.concat_owned(&chunk);
                if i == 999 {
                    alias_at_1k = Some(s);
                }
            }
            assert_eq!(s.unit_len(), 80_000);
            assert!(s.text().starts_with(&*chunk.text()));
            assert!(s.text().ends_with(&*chunk.text()));
            // The buffer the accumulated string reads (a view's owner).
            let buffer = s.view_owner().unwrap_or(&s);
            let cap = (buffer.hdr().cap.get() & CAP_BITS) as usize;
            assert!(
                cap <= 4 * s.unit_len(),
                "capacity {cap} not geometric vs len {}",
                s.unit_len()
            );
            let alias = alias_at_1k.unwrap();
            assert_eq!(
                alias.unit_len(),
                8_000,
                "alias must not observe later appends"
            );
        }
        // Doubling self-append through an alias stays correct.
        let a = LStr::from("xy");
        let b = a;
        let a = a.concat_owned(&b);
        assert_eq!(a.text(), "xyxy");
        assert_eq!(b.text(), "xy");
    }

    /// The width is fixed when a buffer is allocated: ASCII text is one-byte, any other text
    /// two-byte, and a two-byte suffix on a one-byte buffer copies into a two-byte one, after
    /// which appends are in place again. One-byte units widen into a two-byte buffer in place.
    #[test]
    fn widths_follow_the_content_and_appends_keep_them() {
        let ascii = LStr::from("abc");
        assert!(ascii.ascii_hint() && ascii.known_ascii() && ascii.as_ascii() == Some("abc"));
        let wide = LStr::from("ab\u{e9}");
        assert!(!wide.known_ascii() && wide.as_ascii().is_none());
        assert_eq!(wide.unit_len(), 3);
        assert_eq!(wide.unit_at(2), Some(0xe9));
        let mut s = ascii_with_capacity("start", 64);
        let p = s.buffer_ptr();
        s.append_or_replace(&LStr::from(":ab"));
        assert_eq!(s.buffer_ptr(), p, "one-byte onto one-byte appends in place");
        assert!(s.known_ascii());
        let before = s;
        s.append_or_replace(&wide);
        assert_ne!(s.buffer_ptr(), p, "two-byte onto one-byte copies");
        assert!(!s.known_ascii());
        assert_eq!(s.text(), "start:abab\u{e9}");
        assert_eq!(before.text(), "start:ab");
        let q = s.buffer_ptr();
        s.append_or_replace(&LStr::from("xyz"));
        assert_eq!(s.buffer_ptr(), q, "one-byte onto two-byte appends in place");
        assert_eq!(s.text(), "start:abab\u{e9}xyz");
        assert_eq!(s.unit_len(), 14);
    }

    /// Strings compare and hash by code units, whatever their width: a two-byte view whose
    /// units are ASCII equals and hashes like the one-byte string of the same text.
    #[test]
    fn equality_and_hashes_ignore_the_width() {
        let owner = LStr::from(format!("{}\u{4e2d}", "x".repeat(400)));
        assert!(!owner.known_ascii());
        let view = owner.slice_units(0, 300);
        assert!(view.view_owner().is_some() && !view.known_ascii());
        let narrow = LStr::from("x".repeat(300));
        assert!(narrow.known_ascii());
        assert!(view == narrow && narrow == view);
        assert_eq!(hash_of(&view), hash_of(&narrow));
        assert!(view == *"x".repeat(300).as_str());
        let tail = owner.slice_units(100, 401);
        let copy = LStr::from(format!("{}\u{4e2d}", "x".repeat(300)));
        assert!(tail == copy);
        assert_eq!(hash_of(&tail), hash_of(&copy));
        assert!(tail != narrow);
        assert!(LStr::from("ab") != LStr::from("a\u{e9}"));
    }

    /// Lone surrogates are ordinary two-byte units; engine text smuggles them and a surrogate
    /// pair joined from two halves is the character.
    #[test]
    fn surrogates_are_units_and_join_across_appends() {
        let high = LStr::from_utf16(&[0xD834]);
        let low = LStr::from_utf16(&[0xDF06]);
        assert_eq!(high.unit_len(), 1);
        assert_eq!(high.text(), crate::jstr::from_units(&[0xD834]));
        assert_eq!(high.concat_owned(&low).text(), "𝌆");
        let mut joined = high;
        joined.append_or_replace(&low);
        assert_eq!(joined.text(), "𝌆");
        assert_eq!(joined.unit_len(), 2);
        assert!(LStr::from("𝌆") == joined);
        assert_eq!(std::mem::size_of::<LStr>(), std::mem::size_of::<usize>());
        assert_eq!(LEN_OFF, std::mem::size_of::<usize>());
        assert_eq!(CAP_OFF, LEN_OFF + 4);
    }

    #[test]
    fn owned_concat_empty_boundaries() {
        let s = LStr::from("abc");
        let p = s.as_ptr();
        let s = s.concat_owned(&LStr::from(""));
        assert_eq!(s.as_ptr(), p);
        assert!(LStr::ptr_eq(&LStr::from("").concat_owned(&s), &s));
    }

    #[test]
    fn populated_append_reuses_unique_capacity_and_keeps_aliases_immutable() {
        let mut value = ascii_with_capacity("a", 16);
        let pointer = value.buffer_ptr();
        value.append_or_replace(&LStr::from("b"));
        assert_eq!(value.buffer_ptr(), pointer);
        assert_eq!(value.text(), "ab");
        let snapshot = value;
        value.append_or_replace(&LStr::from("c"));
        assert_eq!(snapshot.text(), "ab");
        assert_eq!(value.text(), "abc");
        assert!(!LStr::ptr_eq(&value, &snapshot));
        let rhs = value;
        value.append_or_replace(&rhs);
        assert_eq!(rhs.text(), "abc");
        assert_eq!(value.text(), "abcabc");
    }

    #[test]
    fn bounded_views_flatten_and_preserve_root_immutability() {
        for unit in ["x", "\u{4e2d}"] {
            let source = LStr::from(unit.repeat(1024));
            let mut prefix = source.slice_units(0, 900);
            let nested = prefix.slice_units(100, 800);
            assert!(LStr::ptr_eq(nested.view_owner().unwrap(), &source));
            assert!(!prefix.ascii_hint());
            assert_eq!(prefix.unit_len(), 900);
            assert_eq!(nested.text(), unit.repeat(700));
            let mut source = source;
            assert!(!source.append_in_place("changed"));
            assert!(!prefix.append_in_place("changed"));
            let alias = prefix;
            let appended = prefix.concat_owned(&LStr::from("!"));
            assert_eq!(appended.text(), format!("{}!", unit.repeat(900)));
            assert_eq!(alias.unit_len(), 900);
        }
        assert_eq!(std::mem::offset_of!(View, header), 0);
    }

    #[test]
    fn small_or_disproportionate_slices_copy_and_release_source() {
        // Small tokens always copy; a moderately sized owner may be shared by any other slice.
        let source = LStr::from("x".repeat(2048));
        for (start, end) in [(0, 10), (1024, 1030)] {
            let slice = source.slice_units(start, end);
            assert!(slice.view_owner().is_none());
            assert_eq!(slice.text(), "x".repeat(end - start));
        }
        let shared = source.slice_units(0, 900);
        assert!(shared.view_owner().is_some());
        drop(shared);
        // A large owner is pinned only by a slice covering at least half of it.
        let large = LStr::from("x".repeat(VIEW_PIN_BYTES + 4096));
        for (start, end) in [(0, 900), (1024, 300_000)] {
            let slice = large.slice_units(start, end);
            assert!(slice.view_owner().is_none());
            assert_eq!(slice.unit_len(), end - start);
        }
        assert!(large
            .slice_units(0, large.unit_len() / 2 + 1024)
            .view_owner()
            .is_some());
        // The pinning bound charges the owner's allocation, including spare capacity, rather
        // than its visible length.
        let base = LStr::from("x".repeat(VIEW_PIN_BYTES - 4096));
        let spare = base.concat_grown("x");
        assert!(spare.unit_len() <= VIEW_PIN_BYTES);
        assert!(spare.allocation_requested_bytes() > VIEW_PIN_BYTES);
        let slice = spare.slice_units(0, 1100);
        assert!(
            slice.view_owner().is_none(),
            "charge spare capacity, not visible length"
        );
        // A two-byte owner counts its content's bytes: 128 units of it already make a view, and
        // a short copy of ASCII units narrows to one-byte.
        let unicode = LStr::from(format!("{}😀\u{e9}", "x".repeat(1024)));
        assert_eq!(unicode.unit_len(), 1027);
        let tail = unicode.slice_units(200, unicode.unit_len());
        assert!(tail.view_owner().is_some());
        drop(unicode);
        assert_eq!(tail.text(), format!("{}😀\u{e9}", "x".repeat(824)));
        assert_eq!(tail.repeat_direct(2).unit_len(), tail.unit_len() * 2);
        let short = tail.slice_units(0, 10);
        assert!(short.view_owner().is_none() && short.known_ascii());
        assert_eq!(short.text(), "x".repeat(10));
    }
}
