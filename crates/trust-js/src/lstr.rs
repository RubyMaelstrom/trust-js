//! The engine string: a thin refcounted UTF-8 buffer with spare capacity.
//!
//! `Value::Str` used to hold `Rc<str>` — immutable and exactly sized, which makes an append loop
//! (`s += x`, astring's `this.output += e`) inherently O(n²): every step materializes a fresh
//! allocation of the whole accumulated string. `LStr` is the classic engine fix (QuickJS's
//! JSString): `{strong, len, cap}` header + bytes, one thin pointer. When a string is *uniquely
//! referenced* an append writes in place (amortized by capacity doubling); shared strings copy
//! first, exactly like `Rc::make_mut`.
//!
//! The payload is a single 8-byte pointer with the strong count at offset 0 — the same shape the
//! JIT's inline templates already assume for refcounted payloads (`Rc`'s RcBox), so the machine
//! code that bumps/decrements tag-6 values is unchanged. Logical content is always `len` bytes of
//! valid UTF-8 (lone surrogates smuggled, as before — see [`crate::jstr`]); capacity beyond `len`
//! is invisible to every reader because `Deref` slices to `len`. Large substrings may instead
//! hold a bounded view of a flat buffer. Their header has the same count/length offsets, but
//! no ASCII hint: native byte readers must take their checked path for a view.
//!
//! Like `Rc`, `LStr` is neither `Send` nor `Sync` (non-atomic count; the engine is one thread
//! per realm).
//!
//! With `gc-heap` (P1 S2) a string is a cell of the active Agent's heap: a handle is the cell's
//! address, cloning copies it and nothing is counted. Word 0 of a flat buffer then records its
//! *frontier*, how many bytes have been written into it. An append to a string that ends at its
//! buffer's frontier, with spare capacity, writes past the frontier and returns a view of the
//! same buffer; bytes a string can read never change afterwards, so aliases keep their own
//! length and no uniqueness is needed (Decision 4 as revised at step 4d). Any other append
//! copies into a new buffer with geometric capacity.

#[cfg(not(feature = "gc-heap"))]
use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::ptr::NonNull;

#[repr(C)]
struct Header {
    /// Strong count — MUST stay the first field (the JIT bumps it at payload offset 0). With
    /// `gc-heap`, a flat buffer's frontier (see the module docs); unused by a view.
    strong: Cell<usize>,
    len: Cell<u32>,
    cap: Cell<u32>,
    // `cap` bytes of UTF-8 follow.
}

#[repr(C)]
struct View {
    header: Header,
    owner: LStr,
    offset: usize,
}

/// See the module docs. `repr(transparent)`-thin: one pointer.
pub struct LStr {
    p: NonNull<Header>,
}

const HDR: usize = std::mem::size_of::<Header>();

/// Byte offset of the length within the header — the JIT's inline equality/truthiness templates
/// read `len` from machine code through the stored pointer (the strong count stays at offset 0).
pub(crate) const LEN_OFF: usize = std::mem::offset_of!(Header, len);
/// Byte offset of `cap` (which carries [`ASCII_HINT`] in its top bit) — the JIT's charCodeAt
/// intrinsic tests the hint from machine code.
pub(crate) const CAP_OFF: usize = std::mem::offset_of!(Header, cap);
/// Byte offset of the first content byte in a flat buffer. Views require checked resolution.
pub(crate) const DATA_OFF: usize = HDR;
/// Top bit of `cap`: the content is KNOWN all-ASCII (byte index == UTF-16 unit index, and every
/// byte IS its unit). Purely a hint — never set for non-ASCII content, may be clear for ASCII
/// content. Maintained by every constructor/mutator; capacity readers mask it off.
pub(crate) const ASCII_HINT: u32 = 1 << 31;

// Reserve one capacity value, rather than changing the native count/length layout. A view is
// never ASCII-hinted even when its bytes are ASCII. The optimizing tier's decimal-key reader
// also rejects it by its <=10-byte length guard before accessing inline bytes.
const VIEW_CAP: u32 = ASCII_HINT - 1;
/// The longest content, in bytes, an engine string can hold; constructing a longer one panics.
pub(crate) const MAX_LEN: usize = VIEW_CAP as usize - 1;
const MIN_VIEW_BYTES: usize = 256;
/// A view may keep an owner of at most this size alive regardless of the slice's length; larger
/// owners are pinned only by slices covering at least half of them. Copying every short slice
/// of a moderately sized document made incremental scanners (repeated `substring(0, n)` +
/// sticky RegExp exec) quadratic in both the copy and the matcher's text preparation.
const VIEW_PIN_BYTES: usize = 1 << 20;
const _: () = assert!(MIN_VIEW_BYTES > 10);

#[cfg(not(feature = "gc-heap"))]
fn layout(cap: u32) -> Layout {
    Layout::from_size_align(HDR + cap as usize, std::mem::align_of::<Header>())
        .expect("string too large")
}

/// The descriptor of string cells (gc-heap): a view's one edge is its flat owner.
#[cfg(feature = "gc-heap")]
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
        weak: None,
        finalize: None,
    }
}

impl LStr {
    /// Allocate with `cap` bytes of capacity, seeding `content` (must fit).
    fn alloc(content: &str, cap: u32) -> LStr {
        Self::alloc_with_hint(content, cap, content.is_ascii())
    }

    /// Reuse representation metadata when copying an engine string. A conservative false
    /// hint is valid; never rescan a growing prefix just to rediscover that it is ASCII.
    fn alloc_with_hint(content: &str, cap: u32, ascii: bool) -> LStr {
        debug_assert!(content.len() <= cap as usize);
        assert!(cap < VIEW_CAP, "capacity claims representation bits");
        debug_assert!(!ascii || content.is_ascii());
        unsafe {
            #[cfg(not(feature = "gc-heap"))]
            let p = alloc(layout(cap)) as *mut Header;
            #[cfg(feature = "gc-heap")]
            let p = crate::value::alloc_string_cell(HDR + cap as usize) as *mut Header;
            let p = NonNull::new(p).expect("allocation failed");
            // The hint holds for `content`; constructors that append more bytes afterwards
            // re-AND it with the extra bytes' ASCII-ness (see concat2/concat_grown).
            let hint = if ascii { ASCII_HINT } else { 0 };
            p.as_ptr().write(Header {
                // A count of one, or (gc-heap) the frontier: the content's end.
                strong: Cell::new(if cfg!(feature = "gc-heap") {
                    content.len()
                } else {
                    1
                }),
                len: Cell::new(content.len() as u32),
                cap: Cell::new(cap | hint),
            });
            let data = (p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(content.as_ptr(), data, content.len());
            LStr { p }
        }
    }

    /// Set a fresh, unpublished flat string's length after writing its bytes (and, with
    /// `gc-heap`, its frontier to match).
    #[inline]
    fn set_fresh_len(&self, len: usize) {
        let h = self.hdr();
        h.len.set(len as u32);
        #[cfg(feature = "gc-heap")]
        h.strong.set(len);
    }

    /// Whether the content is KNOWN all-ASCII (see [`ASCII_HINT`]).
    #[inline]
    pub(crate) fn ascii_hint(&self) -> bool {
        self.hdr().cap.get() & ASCII_HINT != 0
    }

    #[inline]
    fn and_ascii(&self, extra_is_ascii: bool) {
        if !extra_is_ascii {
            let h = self.hdr();
            h.cap.set(h.cap.get() & !ASCII_HINT);
        }
    }

    /// The two halves concatenated (a single copy of each into the result).
    pub fn concat2(a: &str, b: &str) -> LStr {
        let total = a.len() + b.len();
        let s = LStr::alloc("", u32::try_from(total).expect("string too large"));
        unsafe {
            let data = (s.p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(a.as_ptr(), data, a.len());
            std::ptr::copy_nonoverlapping(b.as_ptr(), data.add(a.len()), b.len());
            s.set_fresh_len(total);
        }
        s.and_ascii(a.is_ascii() && b.is_ascii());
        s
    }

    /// The three pieces concatenated into one engine allocation.
    #[inline]
    pub(crate) fn concat3(a: &str, b: &str, c: &str) -> LStr {
        let total = a
            .len()
            .checked_add(b.len())
            .and_then(|total| total.checked_add(c.len()))
            .expect("string too large");
        let s = LStr::alloc("", u32::try_from(total).expect("string too large"));
        unsafe {
            let data = (s.p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(a.as_ptr(), data, a.len());
            std::ptr::copy_nonoverlapping(b.as_ptr(), data.add(a.len()), b.len());
            std::ptr::copy_nonoverlapping(c.as_ptr(), data.add(a.len() + b.len()), c.len());
            s.set_fresh_len(total);
        }
        s.and_ascii(a.is_ascii() && b.is_ascii() && c.is_ascii());
        s
    }

    /// The four pieces concatenated into one engine allocation.
    #[inline]
    pub(crate) fn concat4(a: &str, b: &str, c: &str, d: &str) -> LStr {
        let total = a
            .len()
            .checked_add(b.len())
            .and_then(|total| total.checked_add(c.len()))
            .and_then(|total| total.checked_add(d.len()))
            .expect("string too large");
        let s = LStr::alloc("", u32::try_from(total).expect("string too large"));
        unsafe {
            let data = (s.p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(a.as_ptr(), data, a.len());
            std::ptr::copy_nonoverlapping(b.as_ptr(), data.add(a.len()), b.len());
            std::ptr::copy_nonoverlapping(c.as_ptr(), data.add(a.len() + b.len()), c.len());
            std::ptr::copy_nonoverlapping(
                d.as_ptr(),
                data.add(a.len() + b.len() + c.len()),
                d.len(),
            );
            s.set_fresh_len(total);
        }
        s.and_ascii(a.is_ascii() && b.is_ascii() && c.is_ascii() && d.is_ascii());
        s
    }

    /// Concatenate already-coerced strings into one engine allocation.
    pub(crate) fn concat_many(parts: &[LStr], total: usize) -> LStr {
        let s = LStr::alloc("", u32::try_from(total).expect("string too large"));
        let mut offset = 0;
        unsafe {
            let data = (s.p.as_ptr() as *mut u8).add(HDR);
            for part in parts {
                let bytes = part.as_str().as_bytes();
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.add(offset), bytes.len());
                offset += bytes.len();
            }
        }
        debug_assert_eq!(offset, total);
        s.set_fresh_len(total);
        s.and_ascii(parts.iter().all(LStr::ascii_hint));
        s
    }

    #[inline]
    fn hdr(&self) -> &Header {
        unsafe { self.p.as_ref() }
    }

    #[inline]
    fn data(&self) -> *const u8 {
        unsafe {
            if let Some(view) = self.view() {
                // Views always own a flat buffer; nested slices flatten at construction.
                (view.owner.p.as_ptr() as *const u8)
                    .add(HDR)
                    .add(view.offset)
            } else {
                (self.p.as_ptr() as *const u8).add(HDR)
            }
        }
    }

    #[inline]
    fn view(&self) -> Option<&View> {
        (self.hdr().cap.get() == VIEW_CAP).then(|| unsafe { &*self.p.as_ptr().cast::<View>() })
    }

    pub(crate) fn view_owner(&self) -> Option<&LStr> {
        self.view().map(|view| &view.owner)
    }

    pub(crate) fn view_range(&self) -> Option<(&LStr, usize, usize)> {
        self.view()
            .map(|view| (&view.owner, view.offset, view.offset + self.len()))
    }

    /// Slice already-proven UTF-8 boundaries. Copy small tokens, and ranges that would pin a
    /// large owner (over `VIEW_PIN_BYTES`) more than twice their byte length; other slices
    /// share their flat source owner.
    pub(crate) fn slice_bytes(&self, start: usize, end: usize) -> LStr {
        let bytes = &self.as_str()[start..end];
        if start == 0 && end == self.len() {
            return self.clone();
        }
        let (owner, offset) = self
            .view()
            .map_or((self, start), |view| (&view.owner, view.offset + start));
        let owner_bytes = owner.allocation_requested_bytes();
        if bytes.len() < MIN_VIEW_BYTES
            || (owner_bytes > bytes.len().saturating_mul(2) && owner_bytes > VIEW_PIN_BYTES)
        {
            return LStr::from(bytes);
        }
        LStr::new_view(owner.clone(), offset, bytes.len())
    }

    /// A view of `len` bytes of flat `owner` from `offset`.
    fn new_view(owner: LStr, offset: usize, len: usize) -> LStr {
        debug_assert!(owner.view().is_none(), "views own flat buffers");
        let view = View {
            header: Header {
                strong: Cell::new(if cfg!(feature = "gc-heap") { 0 } else { 1 }),
                len: Cell::new(len as u32),
                cap: Cell::new(VIEW_CAP),
            },
            owner,
            offset,
        };
        #[cfg(not(feature = "gc-heap"))]
        let p = Box::into_raw(Box::new(view)).cast::<Header>();
        #[cfg(feature = "gc-heap")]
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

    /// The string's cell, as a GC edge (gc-heap).
    #[cfg(feature = "gc-heap")]
    #[inline]
    pub(crate) fn cell_word(&self) -> u64 {
        self.p.as_ptr() as u64
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe {
            let bytes = std::slice::from_raw_parts(self.data(), self.hdr().len.get() as usize);
            std::str::from_utf8_unchecked(bytes)
        }
    }

    /// The flat buffer this string reads: a view's owner, else itself. An in-place append keeps
    /// it (with `gc-heap` the result is a new view of the same buffer).
    #[cfg(test)]
    pub(crate) fn buffer_ptr(&self) -> *const u8 {
        self.view_owner().unwrap_or(self).as_ptr()
    }

    /// Pointer identity (cache keys — same contract as `Rc::as_ptr`).
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
        #[cfg(feature = "gc-heap")]
        panic!("LStr::strong_count: gc-heap strings have no reference counts");
        #[cfg(not(feature = "gc-heap"))]
        self.hdr().strong.get()
    }

    /// Requested bytes retained by this allocation, excluding allocator rounding.
    ///
    /// This includes TRust JS's stable string header because [`LStr`] allocates that header and its
    /// byte capacity as one explicit allocation. It intentionally does not include allocator
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
            std::mem::size_of::<View>()
        } else {
            HDR.saturating_add((self.hdr().cap.get() & !ASCII_HINT) as usize)
        }
    }

    /// Append in place when this is the ONLY reference and capacity suffices. Returns false
    /// (without modifying anything) otherwise — the caller copies. The unique-owner requirement
    /// is what makes the mutation invisible: no other handle can observe the content, and the
    /// caller must not hold a `&str` borrow of `self` across the call (enforced by `&mut self`).
    pub fn append_in_place(&mut self, x: &str) -> bool {
        self.append_with_hint(x, x.is_ascii())
    }

    #[cfg(feature = "gc-heap")]
    fn append_with_hint(&mut self, x: &str, ascii: bool) -> bool {
        // Frontier append: only the string ending where its buffer's writes end may extend it.
        let (buffer, offset) = match self.view() {
            Some(view) => (view.owner.clone(), view.offset),
            None => (self.clone(), 0),
        };
        let len = self.len();
        let h = buffer.hdr();
        let frontier = h.strong.get();
        if offset + len != frontier || frontier + x.len() > (h.cap.get() & !ASCII_HINT) as usize {
            return false;
        }
        // SAFETY: past the frontier, within capacity: no string can read these bytes yet.
        unsafe {
            let data = (buffer.p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(x.as_ptr(), data.add(frontier), x.len());
        }
        h.strong.set(frontier + x.len());
        // The buffer's hint covers only the flat string's own bytes, so clearing it for a
        // non-ASCII suffix is conservative.
        buffer.and_ascii(ascii);
        *self = LStr::new_view(buffer, offset, len + x.len());
        true
    }

    #[cfg(not(feature = "gc-heap"))]
    fn append_with_hint(&mut self, x: &str, ascii: bool) -> bool {
        let h = self.hdr();
        if h.strong.get() != 1 || h.cap.get() == VIEW_CAP {
            return false;
        }
        let len = h.len.get() as usize;
        if len + x.len() > (h.cap.get() & !ASCII_HINT) as usize {
            return false;
        }
        unsafe {
            let data = (self.p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(x.as_ptr(), data.add(len), x.len());
        }
        h.len.set((len + x.len()) as u32);
        self.and_ascii(ascii);
        true
    }

    /// `self + x` with growth capacity: used by the fused append ops when in-place didn't apply.
    /// Doubles (at least) so a rebuilt accumulator amortizes the next appends.
    pub fn concat_grown(&self, x: &str) -> LStr {
        self.grow_with_hint(x, x.is_ascii())
    }

    fn grow_with_hint(&self, x: &str, ascii: bool) -> LStr {
        let need = self.len().checked_add(x.len()).expect("string too large");
        assert!(need < VIEW_CAP as usize, "string too large");
        let cap = need.saturating_mul(2).max(32).min((VIEW_CAP - 1) as usize) as u32;
        let s = LStr::alloc_with_hint(self.as_str(), cap, self.ascii_hint());
        unsafe {
            let data = (s.p.as_ptr() as *mut u8).add(HDR);
            std::ptr::copy_nonoverlapping(x.as_ptr(), data.add(self.as_str().len()), x.len());
        }
        s.set_fresh_len(need);
        s.and_ascii(ascii);
        s
    }

    /// ECMAScript code-unit concatenation after the caller's coercions and length check.
    /// Shared by all execution tiers and String#concat. Cached ASCII hints eliminate prefix
    /// rescans; moving the left handle also permits amortized appends to unique temporaries.
    /// Aliases (including `s + s`) keep the strong count above one and force a separate buffer.
    pub(crate) fn concat_owned(mut self, right: &LStr) -> LStr {
        self.append_or_replace(right);
        self
    }

    /// Append without temporarily removing an owning binding. Build a replacement before
    /// assigning it, so allocation failure cannot strand a cleared binding. Unique buffers
    /// with capacity mutate only while this mutable borrow excludes every other observer;
    /// shared strings and views retain the original contents and allocate a replacement.
    pub(crate) fn append_or_replace(&mut self, right: &LStr) {
        if right.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = right.clone();
            return;
        }
        if !self.ascii_hint()
            && !right.ascii_hint()
            && crate::jstr::needs_join_fixup(self.as_str(), right)
        {
            let replacement = crate::jstr::concat(self.as_str(), right).into();
            *self = replacement;
            return;
        }
        if !self.append_with_hint(right.as_str(), right.ascii_hint()) {
            let replacement = self.grow_with_hint(right.as_str(), right.ascii_hint());
            *self = replacement;
        }
    }

    /// Repeat the byte representation directly into one engine allocation.
    ///
    /// The caller has already performed the observable `String.prototype.repeat` coercions and
    /// length check (ECMA-262 §22.1.3.18).  Keeping this representation-local avoids first
    /// building a Rust `String` and then copying it into an `LStr`.  The doubling copy uses
    /// `ptr::copy`, which is explicitly overlap-safe as each completed prefix becomes the source
    /// for the next block.
    pub(crate) fn repeat_direct(&self, count: usize) -> LStr {
        let source_len = self.hdr().len.get() as usize;
        let total = source_len
            .checked_mul(count)
            .expect("repeat length checked by caller");
        let repeated = LStr::alloc("", u32::try_from(total).expect("string too large"));
        if source_len == 0 || count == 0 {
            return repeated;
        }
        unsafe {
            let destination = repeated.p.as_ptr().cast::<u8>().add(HDR);
            std::ptr::copy_nonoverlapping(self.data(), destination, source_len);
            let mut filled = source_len;
            while filled < total {
                let copy_len = filled.min(total - filled);
                std::ptr::copy(destination, destination.add(filled), copy_len);
                filled += copy_len;
            }
            repeated.set_fresh_len(total);
        }
        repeated.and_ascii(self.ascii_hint());
        repeated
    }

    /// Repeat an ASCII string directly into one engine allocation.
    #[inline]
    pub(crate) fn repeat_ascii(&self, count: usize) -> LStr {
        debug_assert!(self.ascii_hint());
        self.repeat_direct(count)
    }

    /// Build an ASCII-padded string directly in its final engine allocation. The caller has
    /// already applied StringPad's UTF-16 length and coercion rules, and both operands are known
    /// ASCII, so repeating/truncating the fill cannot require surrogate fixup.
    pub(crate) fn pad_ascii(source: &str, pad: &str, need: usize, at_start: bool) -> LStr {
        debug_assert!(source.is_ascii() && pad.is_ascii() && !pad.is_empty());
        let total = source.len() + need;
        let padded = LStr::alloc("", u32::try_from(total).expect("string too large"));
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
        padded.and_ascii(true);
        padded
    }

    /// Map ASCII letters directly into one engine allocation. The caller must have already
    /// selected the ASCII representation; non-ASCII case mappings (including one-to-many and
    /// context-sensitive mappings) remain on the Unicode implementation path.
    pub(crate) fn map_ascii_case(&self, upper: bool) -> LStr {
        debug_assert!(self.ascii_hint());
        let source = self.as_str().as_bytes();
        let mapped = LStr::alloc("", u32::try_from(source.len()).expect("string too large"));
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

impl Clone for LStr {
    #[inline]
    fn clone(&self) -> LStr {
        #[cfg(not(feature = "gc-heap"))]
        {
            let h = self.hdr();
            h.strong.set(h.strong.get() + 1);
        }
        LStr { p: self.p }
    }
}

/// Nothing to release under gc-heap: the collector frees unreachable cells. Kept as a type
/// with drop glue so code written for counted strings (an explicit `drop` that releases a
/// buffer early, scoped temporaries) reads the same in both builds.
#[cfg(feature = "gc-heap")]
impl Drop for LStr {
    #[inline]
    fn drop(&mut self) {}
}

#[cfg(not(feature = "gc-heap"))]
impl Drop for LStr {
    #[inline]
    fn drop(&mut self) {
        let h = self.hdr();
        let s = h.strong.get();
        if s == 1 {
            if h.cap.get() == VIEW_CAP {
                // Drops the one flat owner, then releases the view allocation.
                unsafe { drop(Box::from_raw(self.p.as_ptr().cast::<View>())) };
            } else {
                let cap = h.cap.get() & !ASCII_HINT;
                unsafe { dealloc(self.p.as_ptr() as *mut u8, layout(cap)) };
            }
        } else {
            h.strong.set(s - 1);
        }
    }
}

impl std::ops::Deref for LStr {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl std::borrow::Borrow<str> for LStr {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for LStr {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for LStr {
    fn from(s: &str) -> LStr {
        LStr::alloc(s, u32::try_from(s.len()).expect("string too large"))
    }
}

impl From<String> for LStr {
    fn from(s: String) -> LStr {
        LStr::from(s.as_str())
    }
}

impl From<std::rc::Rc<str>> for LStr {
    fn from(s: std::rc::Rc<str>) -> LStr {
        LStr::from(&*s)
    }
}

impl From<&String> for LStr {
    fn from(s: &String) -> LStr {
        LStr::from(s.as_str())
    }
}

impl From<char> for LStr {
    fn from(c: char) -> LStr {
        LStr::from(c.encode_utf8(&mut [0u8; 4]) as &str)
    }
}

impl PartialEq for LStr {
    fn eq(&self, other: &LStr) -> bool {
        LStr::ptr_eq(self, other) || self.as_str() == other.as_str()
    }
}
impl Eq for LStr {}

impl PartialEq<str> for LStr {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl std::hash::Hash for LStr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl std::fmt::Display for LStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Debug for LStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl From<&LStr> for std::rc::Rc<str> {
    fn from(s: &LStr) -> std::rc::Rc<str> {
        std::rc::Rc::from(s.as_str())
    }
}

impl From<LStr> for std::rc::Rc<str> {
    fn from(s: LStr) -> std::rc::Rc<str> {
        std::rc::Rc::from(s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_concat_reuses_unique_capacity_and_preserves_aliases() {
        let mut s = LStr::from("a").concat_owned(&LStr::from("b"));
        let p = s.buffer_ptr();
        s = s.concat_owned(&LStr::from("c"));
        assert_eq!(s.buffer_ptr(), p);
        assert_eq!(s.as_str(), "abc");
        // A frontier append (gc-heap) returns a view, which carries no hint.
        assert!(s.ascii_hint() || s.view_owner().is_some());
        let alias = s.clone();
        s = s.concat_owned(&LStr::from("d"));
        assert!(!LStr::ptr_eq(&s, &alias));
        assert_eq!(alias.as_str(), "abc");
        assert_eq!(s.as_str(), "abcd");
        let right = s.clone();
        s = s.concat_owned(&right);
        assert_eq!(right.as_str(), "abcd");
        assert_eq!(s.as_str(), "abcdabcd");
    }

    /// P1-S0e oracle: JS `s += x` accumulation must stay amortized after
    /// the GC string migration (plan Decision 4). Geometric capacity
    /// growth is the observable proxy: a quadratic replacement that
    /// reallocates exact-size per append fails the capacity bound, and a
    /// uniqueness-based one fails the aliasing assertions above.
    #[test]
    fn accumulation_scaling_oracle_geometric_capacity_and_aliasing() {
        let chunk = LStr::from("abcdefgh");
        let mut s = LStr::from("");
        let mut alias_at_1k = None;
        for i in 0..10_000 {
            s = s.concat_owned(&chunk);
            if i == 999 {
                alias_at_1k = Some(s.clone());
            }
        }
        assert_eq!(s.len(), 80_000);
        assert!(s.as_str().starts_with("abcdefgh"));
        assert!(s.as_str().ends_with("abcdefgh"));
        // The buffer the accumulated string reads (a view's owner under gc-heap).
        let buffer = s.view_owner().unwrap_or(&s);
        let cap = (buffer.hdr().cap.get() & !ASCII_HINT) as usize;
        assert!(
            cap <= 4 * s.len(),
            "capacity {cap} not geometric vs len {}",
            s.len()
        );
        let alias = alias_at_1k.unwrap();
        assert_eq!(alias.len(), 8_000, "alias must not observe later appends");
        // Doubling self-append through an alias stays correct.
        let a = LStr::from("xy");
        let b = a.clone();
        let a = a.concat_owned(&b);
        assert_eq!(a.as_str(), "xyxy");
        assert_eq!(b.as_str(), "xy");
    }

    #[test]
    fn owned_concat_retains_conservative_hints_and_clears_for_unicode() {
        let s = LStr::alloc_with_hint("abc", 3, false);
        let s = s.concat_owned(&LStr::from("d"));
        assert!(!s.ascii_hint(), "growth must not rescan the prefix");
        assert_eq!(s.as_str(), "abcd");
        let s = LStr::from("abc").concat_owned(&LStr::alloc_with_hint("d", 1, false));
        assert!(!s.ascii_hint(), "right hint must also be preserved");
        let s = LStr::from("a").concat_owned(&LStr::from("b"));
        let p = s.buffer_ptr();
        let s = s.concat_owned(&LStr::from("é"));
        assert_eq!(s.buffer_ptr(), p);
        assert!(!s.ascii_hint());
        assert_eq!(s.as_str(), "abé");
    }

    #[test]
    fn owned_concat_empty_and_surrogate_boundaries() {
        let s = LStr::from("abc");
        let p = s.as_ptr();
        let s = s.concat_owned(&LStr::from(""));
        assert_eq!(s.as_ptr(), p);
        assert!(LStr::ptr_eq(&LStr::from("").concat_owned(&s), &s));
        let high = LStr::from(crate::jstr::from_units(&[0xD834]));
        let low = LStr::from(crate::jstr::from_units(&[0xDF06]));
        assert_eq!(high.concat_owned(&low).as_str(), "𝌆");
        assert_eq!(std::mem::size_of::<LStr>(), std::mem::size_of::<usize>());
        assert_eq!(LEN_OFF, std::mem::size_of::<usize>());
        assert_eq!(CAP_OFF, LEN_OFF + 4);
    }

    #[test]
    fn populated_append_reuses_unique_capacity_and_keeps_aliases_immutable() {
        let mut value = LStr::alloc_with_hint("a", 16, true);
        let pointer = value.buffer_ptr();
        value.append_or_replace(&LStr::from("b"));
        assert_eq!(value.buffer_ptr(), pointer);
        assert_eq!(value.as_str(), "ab");
        let snapshot = value.clone();
        value.append_or_replace(&LStr::from("c"));
        assert_eq!(snapshot.as_str(), "ab");
        assert_eq!(value.as_str(), "abc");
        assert!(!LStr::ptr_eq(&value, &snapshot));
        let rhs = value.clone();
        value.append_or_replace(&rhs);
        assert_eq!(rhs.as_str(), "abc");
        assert_eq!(value.as_str(), "abcabc");
        let mut high = LStr::from(crate::jstr::from_units(&[0xd834]));
        high.append_or_replace(&LStr::from(crate::jstr::from_units(&[0xdf06])));
        assert_eq!(high.as_str(), "𝌆");
        assert!(!high.ascii_hint());
    }

    #[test]
    fn bounded_views_flatten_and_preserve_root_immutability() {
        let mut source = LStr::from("x".repeat(1024));
        let mut prefix = source.slice_bytes(0, 900);
        let nested = prefix.slice_bytes(100, 800);
        assert!(LStr::ptr_eq(nested.view_owner().unwrap(), &source));
        assert!(!prefix.ascii_hint());
        assert_eq!(prefix.len(), 900);
        assert_eq!(nested.as_str(), &source[100..800]);
        assert!(!source.append_in_place("changed"));
        assert!(!prefix.append_in_place("changed"));
        let alias = prefix.clone();
        let appended = prefix.concat_owned(&LStr::from("!"));
        assert_eq!(appended.as_str(), format!("{}!", "x".repeat(900)));
        assert_eq!(alias.len(), 900);
        // Reference counts release the owner as its views go; gc-heap has none to observe.
        #[cfg(not(feature = "gc-heap"))]
        {
            let count = source.strong_count();
            let packed = crate::value::PackedValue::pack(crate::value::Value::Str(nested));
            let cloned = packed.clone();
            drop(packed);
            drop(cloned);
            assert_eq!(source.strong_count(), count - 1);
            drop(alias);
            if cfg!(not(feature = "gc-heap")) {
                assert_eq!(source.strong_count(), 1);
            }
        }
        assert_eq!(std::mem::offset_of!(View, header), 0);
    }

    #[test]
    fn small_or_disproportionate_slices_copy_and_release_source() {
        // Small tokens always copy; a moderately sized owner may be shared by any other slice.
        let source = LStr::from("x".repeat(2048));
        for (start, end) in [(0, 10), (1024, 1030)] {
            let slice = source.slice_bytes(start, end);
            assert!(slice.view_owner().is_none());
            assert_eq!(slice.as_str(), &source[start..end]);
            if cfg!(not(feature = "gc-heap")) {
                assert_eq!(source.strong_count(), 1);
            }
        }
        let shared = source.slice_bytes(0, 900);
        assert!(shared.view_owner().is_some());
        drop(shared);
        if cfg!(not(feature = "gc-heap")) {
            assert_eq!(source.strong_count(), 1);
        }
        // A large owner is pinned only by a slice covering at least half of it.
        let large = LStr::from("x".repeat(VIEW_PIN_BYTES + 4096));
        for (start, end) in [(0, 900), (1024, 300_000)] {
            let slice = large.slice_bytes(start, end);
            assert!(slice.view_owner().is_none());
            assert_eq!(slice.as_str(), &large[start..end]);
            if cfg!(not(feature = "gc-heap")) {
                assert_eq!(large.strong_count(), 1);
            }
        }
        assert!(large
            .slice_bytes(0, large.len() / 2 + 1024)
            .view_owner()
            .is_some());
        // The pinning bound charges the owner's allocation, including spare capacity, rather
        // than its visible length.
        let base = LStr::from("x".repeat(VIEW_PIN_BYTES - 4096));
        let spare = base.concat_grown("x");
        assert!(spare.len() <= VIEW_PIN_BYTES);
        assert!(spare.allocation_requested_bytes() > VIEW_PIN_BYTES);
        let slice = spare.slice_bytes(0, 1100);
        assert!(
            slice.view_owner().is_none(),
            "charge spare capacity, not visible length"
        );
        let unicode = LStr::from(format!("{}😀é", "x".repeat(1024)));
        let tail = unicode.slice_bytes(200, unicode.len());
        drop(unicode);
        assert_eq!(tail.as_str(), format!("{}😀é", "x".repeat(824)));
        assert_eq!(tail.repeat_direct(2).len(), tail.len() * 2);
    }
}
