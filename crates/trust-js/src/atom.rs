//! Atoms and property keys (P2 W2a; the plan's "W2a and W2b: atoms and scope descriptors").
//!
//! An atom is an engine string that is the only one of its text in its Agent heap, or in the
//! process-wide static table, so atoms compare and hash by address. A heap atom is an exact-size
//! flat string cell of that heap ([`LStr::is_exact`]): nothing is ever written past its units,
//! so it reads like any engine string and stays immutable. Each heap's [`AtomTable`] holds its
//! atoms for the heap's whole life, as V8's string table and SpiderMonkey's atoms zone hold
//! theirs until their collection runs; collecting atoms is a later series.
//!
//! The static names are atoms the engine itself spells. They live outside every heap, once per
//! process, and every heap's table starts with them, so a static name is the same atom in every
//! heap. A collector ignores their addresses, as it ignores any word that names no cell of the
//! heap it collects (see [`crate::gc_trace::GcSink::word`]).
//!
//! A [`PropKey`] is a property key in one word: an atom, or a canonical array index tagged in
//! its low bit, as SpiderMonkey's jsid. Text becomes a key through [`PropKey::from_units`]
//! alone, so equal keys are the same word.

// The plan's step 2 brings atoms without users: binding names (step 3) and property keys
// (step 6) take them up.
#![cfg_attr(not(test), allow(dead_code))]

use crate::lstr::{LStr, Units};
use std::hash::{BuildHasher, Hasher};
use std::num::NonZeroU64;
use std::ptr::NonNull;

/// An atom: the only string of its text in its heap, or a static name (see the module docs).
/// Two atoms are equal exactly when they are the same cell.
#[derive(Clone, Copy)]
pub(crate) struct Atom(LStr);

impl PartialEq for Atom {
    #[inline]
    fn eq(&self, other: &Atom) -> bool {
        LStr::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Atom {}

impl std::hash::Hash for Atom {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.0.as_ptr() as usize);
    }
}

impl std::fmt::Debug for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Atom({:?})", self.0)
    }
}

/// The atom's text.
impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl Atom {
    /// The atom as an engine string. A static name's string is no heap cell, so it never
    /// becomes a [`crate::value::Value`] (see [`Atom::is_static`]).
    #[inline]
    pub(crate) fn as_lstr(self) -> LStr {
        self.0
    }

    /// The atom's address, the word a collector traces.
    #[inline]
    pub(crate) fn cell_word(self) -> u64 {
        self.0.cell_word()
    }

    /// The atom whose address is `word`.
    ///
    /// # Safety
    /// `word` is an atom's [`Atom::cell_word`], and the atom's heap is alive.
    #[inline]
    pub(crate) unsafe fn from_cell_word(word: u64) -> Atom {
        Atom(LStr::from_cell(NonNull::new_unchecked(
            word as usize as *mut u8,
        )))
    }

    /// Whether this is a static name, which lives outside every heap.
    pub(crate) fn is_static(self) -> bool {
        static_names().contains(self)
    }

    /// The atom's engine text, borrowed for a one-byte atom.
    #[inline]
    pub(crate) fn text(&self) -> std::borrow::Cow<'_, str> {
        self.0.text()
    }

    /// Whether the atom's text is `text`.
    #[inline]
    pub(crate) fn is(&self, text: &str) -> bool {
        self.0 == *text
    }
}

/// The atom of engine text `text` in the active heap, made if the heap has none yet.
pub(crate) fn atomize(text: &str) -> Atom {
    crate::value::with_atoms(|atoms| atoms.atomize_text(text))
}

/// The active heap's atom of engine text `text`, if it has one.
pub(crate) fn lookup(text: &str) -> Option<Atom> {
    crate::value::with_atoms(|atoms| atoms.lookup_text(text))
}

/// What becomes an atom of the active heap: an atom, or engine text atomized there.
pub(crate) trait IntoAtom {
    fn into_atom(self) -> Atom;
}

impl IntoAtom for Atom {
    #[inline]
    fn into_atom(self) -> Atom {
        self
    }
}

impl IntoAtom for &str {
    fn into_atom(self) -> Atom {
        atomize(self)
    }
}

impl IntoAtom for String {
    fn into_atom(self) -> Atom {
        atomize(&self)
    }
}

impl IntoAtom for &String {
    fn into_atom(self) -> Atom {
        atomize(self)
    }
}

impl IntoAtom for std::rc::Rc<str> {
    fn into_atom(self) -> Atom {
        atomize(&self)
    }
}

impl IntoAtom for &std::rc::Rc<str> {
    fn into_atom(self) -> Atom {
        atomize(self)
    }
}

/// A property key in one word: an atom, or a canonical array index (0 to 2^32 - 2) shifted left
/// and tagged in the low bit, which no atom's address has.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PropKey(NonZeroU64);

impl PropKey {
    /// The largest array index, 2^32 - 2 (ECMA-262 §6.1.7 "array index").
    pub(crate) const MAX_INDEX: u32 = u32::MAX - 1;

    /// The key of array index `index`.
    #[inline]
    pub(crate) fn index(index: u32) -> PropKey {
        debug_assert!(index <= PropKey::MAX_INDEX);
        PropKey(NonZeroU64::new(u64::from(index) << 1 | 1).expect("tagged"))
    }

    /// The key of `atom`.
    #[inline]
    pub(crate) fn atom(atom: Atom) -> PropKey {
        let word = atom.cell_word();
        debug_assert_eq!(word & 1, 0, "string cells are aligned");
        PropKey(NonZeroU64::new(word).expect("string cells are not null"))
    }

    /// The key whose text is `units` in `table`'s heap: an index when the text is a canonical
    /// array index, else its atom, made if it has none yet.
    pub(crate) fn from_units(table: &mut AtomTable, units: Units<'_>) -> PropKey {
        match canonical_index(units) {
            Some(index) => PropKey::index(index),
            None => PropKey::atom(table.atomize(units)),
        }
    }

    /// The existing key whose text is `units`: an index, or an atom `table` already has. Text
    /// with no atom names no property (V8's LookupExisting), so a read needs no new atom.
    pub(crate) fn lookup(table: &AtomTable, units: Units<'_>) -> Option<PropKey> {
        match canonical_index(units) {
            Some(index) => Some(PropKey::index(index)),
            None => table.lookup(units).map(PropKey::atom),
        }
    }

    #[inline]
    pub(crate) fn as_index(self) -> Option<u32> {
        (self.0.get() & 1 == 1).then(|| (self.0.get() >> 1) as u32)
    }

    #[inline]
    pub(crate) fn as_atom(self) -> Option<Atom> {
        (self.0.get() & 1 == 0).then(|| {
            // SAFETY: an untagged key holds the address of an atom's cell (`PropKey::atom`),
            // which lives as long as its heap.
            Atom(unsafe {
                LStr::from_cell(NonNull::new_unchecked(self.0.get() as usize as *mut u8))
            })
        })
    }

    /// The word a collector traces: an atom's address, none for an index.
    #[inline]
    pub(crate) fn cell_word(self) -> Option<u64> {
        (self.0.get() & 1 == 0).then_some(self.0.get())
    }
}

impl std::fmt::Debug for PropKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.as_index(), self.as_atom()) {
            (Some(index), _) => write!(f, "PropKey({index})"),
            (_, Some(atom)) => write!(f, "PropKey({:?})", atom.0),
            _ => unreachable!("a key is an index or an atom"),
        }
    }
}

/// The array index `units` spell canonically (ECMA-262 §6.1.7: CanonicalNumericIndexString of
/// an integer below 2^32 - 1), as [`crate::value::canonical_index`] reads engine text.
pub(crate) fn canonical_index(units: Units<'_>) -> Option<u32> {
    let first = units.get(0)?;
    if !(u16::from(b'0')..=u16::from(b'9')).contains(&first) {
        return None;
    }
    if first == u16::from(b'0') {
        return (units.len() == 1).then_some(0);
    }
    // 4294967294 has ten digits.
    if units.len() > 10 {
        return None;
    }
    let mut value = 0u64;
    for unit in units.iter() {
        let digit = unit.wrapping_sub(u16::from(b'0'));
        if digit > 9 {
            return None;
        }
        value = value * 10 + u64::from(digit);
    }
    (value <= u64::from(PropKey::MAX_INDEX)).then_some(value as u32)
}

/// One slot of an [`AtomTable`]: an atom and the low bits of its hash, or empty.
#[derive(Clone, Copy, Default)]
struct Slot {
    hash: u32,
    atom: Option<Atom>,
}

/// A heap's atoms, by text: open addressing over a hash of their code units seeded per table,
/// since key text comes from scripts.
pub(crate) struct AtomTable {
    /// A power of two, at most half full.
    slots: Vec<Slot>,
    len: usize,
    /// The hash's seeds, random per table.
    seed: (u64, u64),
    /// Recent lookups of ASCII engine text, by the text's address: names mostly come from the
    /// same AST text again and again. A hit compares the atom's bytes, so a reused address can
    /// only miss.
    recent: Box<[(usize, Option<Atom>); RECENT]>,
    /// The heap atoms made since a collection's marking last reported this table's roots: a
    /// minor collection marks these, the older ones being old cells (see
    /// [`AtomTable::trace_roots`]).
    young: Vec<Atom>,
    /// The heap atoms' bytes, units and headers (the GC log's census).
    bytes: usize,
}

/// The entries of [`AtomTable::recent`].
const RECENT: usize = 256;

impl Default for AtomTable {
    fn default() -> Self {
        let random = || {
            std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish()
        };
        let mut table = AtomTable {
            slots: vec![Slot::default(); 1024],
            len: 0,
            seed: (random(), random()),
            recent: Box::new([(0, None); RECENT]),
            young: Vec::new(),
            bytes: 0,
        };
        for &atom in &static_names().atoms {
            let units = atom.0.units();
            let hash = table.hash(units);
            if let Err(slot) = table.find(hash, units) {
                table.slots[slot] = Slot {
                    hash,
                    atom: Some(atom),
                };
                table.len += 1;
            }
        }
        table
    }
}

impl AtomTable {
    /// The atom whose text is `units`, made if the heap has none yet: a copy at the narrowest
    /// width, exactly sized.
    pub(crate) fn atomize(&mut self, units: Units<'_>) -> Atom {
        let hash = self.hash(units);
        match self.find(hash, units) {
            Ok(atom) => atom,
            Err(slot) => self.insert(slot, hash, LStr::exact(units)),
        }
    }

    /// [`AtomTable::atomize`] of a string's units, taking the string itself as the atom when it
    /// is already exact (V8 internalizes such a string in place).
    pub(crate) fn atomize_lstr(&mut self, string: LStr) -> Atom {
        let units = string.units();
        let hash = self.hash(units);
        match self.find(hash, units) {
            Ok(atom) => atom,
            Err(slot) if string.is_exact() => self.insert(slot, hash, string),
            Err(slot) => self.insert(slot, hash, LStr::exact(units)),
        }
    }

    /// The atom whose text is `units`, if the heap has one.
    pub(crate) fn lookup(&self, units: Units<'_>) -> Option<Atom> {
        self.find(self.hash(units), units).ok()
    }

    /// [`AtomTable::atomize`] of engine text (see [`crate::jstr`]).
    pub(crate) fn atomize_text(&mut self, text: &str) -> Atom {
        if !text.is_ascii() {
            return self.atomize(Units::Utf16(&crate::jstr::units(text)));
        }
        if let Some(atom) = self.recent(text) {
            return atom;
        }
        let atom = self.atomize(Units::Ascii(text.as_bytes()));
        self.remember(text, atom);
        atom
    }

    /// [`AtomTable::lookup`] of engine text.
    pub(crate) fn lookup_text(&mut self, text: &str) -> Option<Atom> {
        if !text.is_ascii() {
            return self.lookup(Units::Utf16(&crate::jstr::units(text)));
        }
        if let Some(atom) = self.recent(text) {
            return Some(atom);
        }
        let atom = self.lookup(Units::Ascii(text.as_bytes()))?;
        self.remember(text, atom);
        Some(atom)
    }

    /// The atom [`AtomTable::recent`] holds for ASCII `text`'s address, if it is `text`'s.
    #[inline]
    fn recent(&self, text: &str) -> Option<Atom> {
        let (address, atom) = self.recent[Self::recent_slot(text)];
        let atom = atom.filter(|_| address == text.as_ptr() as usize)?;
        (atom.0.as_ascii() == Some(text)).then_some(atom)
    }

    #[inline]
    fn remember(&mut self, text: &str, atom: Atom) {
        self.recent[Self::recent_slot(text)] = (text.as_ptr() as usize, Some(atom));
    }

    #[inline]
    fn recent_slot(text: &str) -> usize {
        (text.as_ptr() as usize >> 3) % RECENT
    }

    /// The number of atoms, static names included.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The bytes of the heap atoms' cells.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.bytes
    }

    /// Report the atoms a collection of `kind` traces: every heap atom for a major collection,
    /// those made since the last marking for a minor one. Either way the record of new atoms
    /// starts anew, so only a marking calls this.
    pub(crate) fn trace_roots(
        &mut self,
        kind: crate::gc_core::CollectionKind,
        sink: &mut dyn crate::gc_trace::GcSink,
    ) {
        match kind {
            crate::gc_core::CollectionKind::Minor => {
                for atom in &self.young {
                    sink.word(atom.cell_word());
                }
            }
            crate::gc_core::CollectionKind::Major => {
                for atom in self.slots.iter().filter_map(|slot| slot.atom) {
                    sink.word(atom.cell_word());
                }
            }
        }
        self.young.clear();
    }

    /// Every atom, static names included (heap verification and tests).
    pub(crate) fn atoms(&self) -> impl Iterator<Item = Atom> + '_ {
        self.slots.iter().filter_map(|slot| slot.atom)
    }

    /// Heap verification, after a minor collection's marking: no atom is a young cell the
    /// marking left unmarked, so the minor collection, which traced only the atoms made since
    /// the last one ([`AtomTable::trace_roots`]), missed none.
    #[cfg(debug_assertions)]
    pub(crate) fn check_marked(&self, core: &crate::gc_core::Heap) {
        for atom in self.atoms() {
            assert!(
                core.cell_age(atom.cell_word()) != Some(crate::gc_core::CellAge::Young),
                "the atom table holds the young atom {atom:?} after a minor collection left it \
                 unmarked"
            );
        }
    }

    fn insert(&mut self, slot: usize, hash: u32, string: LStr) -> Atom {
        let atom = Atom(string);
        self.slots[slot] = Slot {
            hash,
            atom: Some(atom),
        };
        self.len += 1;
        self.young.push(atom);
        self.bytes += string_bytes(string);
        if self.len * 2 > self.slots.len() {
            self.grow();
        }
        atom
    }

    fn grow(&mut self) {
        let doubled = vec![Slot::default(); self.slots.len() * 2];
        let old = std::mem::replace(&mut self.slots, doubled);
        let mask = self.slots.len() - 1;
        for slot in old.into_iter().filter(|slot| slot.atom.is_some()) {
            let mut index = slot.hash as usize & mask;
            while self.slots[index].atom.is_some() {
                index = (index + 1) & mask;
            }
            self.slots[index] = slot;
        }
    }

    /// The atom of `units` (whose hash is `hash`), or the empty slot where it belongs.
    fn find(&self, hash: u32, units: Units<'_>) -> Result<Atom, usize> {
        let mask = self.slots.len() - 1;
        let mut index = hash as usize & mask;
        loop {
            let slot = self.slots[index];
            match slot.atom {
                None => return Err(index),
                Some(atom) if slot.hash == hash && atom.0.units().same_units(&units) => {
                    return Ok(atom)
                }
                Some(_) => index = (index + 1) & mask,
            }
        }
    }

    /// The seeded hash of `units`, the same at either width: units that all fit a byte hash as
    /// those bytes, any other as their UTF-16 code units' little-endian bytes.
    fn hash(&self, units: Units<'_>) -> u32 {
        match units {
            Units::Ascii(bytes) => self.hash_bytes(bytes),
            Units::Utf16(wide) if wide.iter().all(|&unit| unit < 0x100) => {
                let narrow = |buffer: &mut [u8]| {
                    for (byte, &unit) in buffer.iter_mut().zip(wide) {
                        *byte = unit as u8;
                    }
                };
                if wide.len() <= 256 {
                    let mut buffer = [0u8; 256];
                    narrow(&mut buffer[..wide.len()]);
                    self.hash_bytes(&buffer[..wide.len()])
                } else {
                    let mut buffer = vec![0u8; wide.len()];
                    narrow(&mut buffer);
                    self.hash_bytes(&buffer)
                }
            }
            Units::Utf16(wide) => {
                let bytes: Vec<u8> = wide.iter().flat_map(|unit| unit.to_le_bytes()).collect();
                self.hash_bytes(&bytes)
            }
        }
    }

    /// A hash of `bytes` that folds each 8-byte word into the state with a 128-bit multiply
    /// (wyhash's and rapidhash's mixing), from the table's random seeds.
    fn hash_bytes(&self, bytes: &[u8]) -> u32 {
        const P0: u64 = 0xa076_1d64_78bd_642f;
        const P1: u64 = 0xe703_7ed1_a0b4_28db;
        let fold = |a: u64, b: u64| {
            let product = u128::from(a) * u128::from(b);
            product as u64 ^ (product >> 64) as u64
        };
        let mut state = self.seed.0 ^ (bytes.len() as u64).wrapping_mul(P0);
        let (words, rest) = bytes.as_chunks::<8>();
        for &word in words {
            state = fold(state ^ u64::from_le_bytes(word), P1 ^ self.seed.1);
        }
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        state = fold(state ^ u64::from_le_bytes(tail), P1 ^ self.seed.1);
        fold(state, P0) as u32
    }
}

/// The bytes an atom's cell takes: its header and units.
fn string_bytes(string: LStr) -> usize {
    match string.units() {
        Units::Ascii(bytes) => LStr::exact_ascii_size(bytes.len()),
        Units::Utf16(units) => LStr::exact_ascii_size(units.len() * 2),
    }
}

/// The static names: the atoms of [`STATIC_NAMES`], in one block outside every heap.
pub(crate) struct StaticNames {
    /// By [`StaticName`] index.
    atoms: Vec<Atom>,
    /// The block's address range, for [`StaticNames::contains`].
    span: std::ops::Range<usize>,
}

// SAFETY: the block is written once, before the table is published, and never again; atoms are
// exact, so no append writes into it.
unsafe impl Send for StaticNames {}
unsafe impl Sync for StaticNames {}

impl StaticNames {
    fn contains(&self, atom: Atom) -> bool {
        self.span.contains(&(atom.0.as_ptr() as usize))
    }
}

/// The process's static names, laid out at first use.
pub(crate) fn static_names() -> &'static StaticNames {
    static NAMES: std::sync::OnceLock<StaticNames> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        #[derive(Clone, Copy)]
        #[repr(align(16))]
        struct Unit {
            _bytes: [u8; 16],
        }
        const ALIGN: usize = std::mem::size_of::<Unit>();
        let size = |name: &str| LStr::exact_ascii_size(name.len()).next_multiple_of(ALIGN);
        let total: usize = STATIC_NAMES.iter().map(|name| size(name)).sum();
        // Kept for the process's life: the names outlive every heap.
        let block: &'static mut [Unit] =
            Vec::leak(vec![Unit { _bytes: [0; ALIGN] }; total / ALIGN]);
        let base = block.as_mut_ptr().cast::<u8>();
        let mut offset = 0;
        let atoms = STATIC_NAMES
            .iter()
            .map(|name| {
                debug_assert!(
                    name.is_ascii() && canonical_index(Units::Ascii(name.as_bytes())).is_none()
                );
                // SAFETY: the block is aligned and sized for every name, each at its own offset,
                // and never freed or written again.
                let atom = Atom(unsafe {
                    LStr::write_exact_ascii(
                        NonNull::new_unchecked(base.add(offset)),
                        name.as_bytes(),
                    )
                });
                offset += size(name);
                atom
            })
            .collect();
        StaticNames {
            atoms,
            span: base as usize..base as usize + total,
        }
    })
}

/// A static name, by its index in [`STATIC_NAMES`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StaticName(u16);

impl StaticName {
    /// The name's atom.
    #[inline]
    pub(crate) fn atom(self) -> Atom {
        static_names().atoms[self.0 as usize]
    }

    #[inline]
    pub(crate) fn key(self) -> PropKey {
        PropKey::atom(self.atom())
    }
}

/// Declares the static names: each constant's index in [`STATIC_NAMES`], whose order is theirs.
macro_rules! static_names {
    ($($name:ident = $text:literal,)*) => {
        /// The names the engine itself spells, made atoms once per process (see the module docs).
        pub(crate) const STATIC_NAMES: &[&str] = &[$($text,)*];

        /// The static names, as [`StaticName`]s. A name serves through every table's lookup
        /// even where no code names its constant.
        #[allow(dead_code)]
        pub(crate) mod names {
            use super::StaticName;
            static_names!(@consts 0u16; $($name,)*);
        }
    };
    (@consts $index:expr; $name:ident, $($rest:ident,)*) => {
        pub(crate) const $name: StaticName = StaticName($index);
        static_names!(@consts $index + 1u16; $($rest,)*);
    };
    (@consts $index:expr;) => {};
}

static_names! {
    LENGTH = "length",
    PROTOTYPE = "prototype",
    CONSTRUCTOR = "constructor",
    NAME = "name",
    MESSAGE = "message",
    CAUSE = "cause",
    STACK = "stack",
    TO_STRING = "toString",
    VALUE_OF = "valueOf",
    TO_JSON = "toJSON",
    VALUE = "value",
    DONE = "done",
    NEXT = "next",
    RETURN = "return",
    THROW = "throw",
    THEN = "then",
    GET = "get",
    SET = "set",
    WRITABLE = "writable",
    ENUMERABLE = "enumerable",
    CONFIGURABLE = "configurable",
    ARGUMENTS = "arguments",
    CALLEE = "callee",
    CALLER = "caller",
    PROTO = "__proto__",
    INDEX = "index",
    INPUT = "input",
    GROUPS = "groups",
    INDICES = "indices",
    LAST_INDEX = "lastIndex",
    SOURCE = "source",
    FLAGS = "flags",
    EXEC = "exec",
    RAW = "raw",
    DEFAULT = "default",
    CALL = "call",
    APPLY = "apply",
    BIND = "bind",
    JOIN = "join",
    PUSH = "push",
    KEYS = "keys",
    VALUES = "values",
    ENTRIES = "entries",
    SIZE = "size",
    HAS = "has",
    ADD = "add",
    DELETE = "delete",
    OWN_KEYS = "ownKeys",
    DEFINE_PROPERTY = "defineProperty",
    GET_OWN_PROPERTY_DESCRIPTOR = "getOwnPropertyDescriptor",
    GET_PROTOTYPE_OF = "getPrototypeOf",
    SET_PROTOTYPE_OF = "setPrototypeOf",
    IS_EXTENSIBLE = "isExtensible",
    PREVENT_EXTENSIONS = "preventExtensions",
    DELETE_PROPERTY = "deleteProperty",
    CONSTRUCT = "construct",
    UNDEFINED = "undefined",
    THIS = "this",
    HOME_OBJECT = "%homeobject%",
    SUPER_PROTO = "%superproto%",
    SUPER_CLASS = "%superclass%",
    IMPORT_META = "%importmeta%",
    THIS_CTOR = "%thisctor%",
    SUPER_PROP_OK = "%superpropok%",
    SUPER_CALL_OK = "%supercallok%",
    NEW_TARGET = "%newtarget%",
    FIELD_INIT = "%fieldinit%",
    NAN = "NaN",
    INFINITY = "Infinity",
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    #[test]
    fn equal_text_is_one_atom_at_either_width() {
        let mut table = AtomTable::default();
        let a = table.atomize(Units::Ascii(b"width"));
        let units = wide("width");
        assert!(table.atomize(Units::Utf16(&units)) == a);
        assert!(a.as_lstr().is_exact() && a.as_lstr().known_ascii());
        let b = table.atomize(Units::Utf16(&wide("wïdth")));
        assert!(b != a && !b.as_lstr().known_ascii());
        assert!(table.lookup(Units::Utf16(&wide("wïdth"))) == Some(b));
        assert!(table.lookup(Units::Ascii(b"absent")).is_none());
        // A lone surrogate is text like any other.
        let lone = table.atomize(Units::Utf16(&[0x61, 0xd800]));
        assert!(table.atomize(Units::Utf16(&[0x61, 0xd800])) == lone);
        assert!(table.lookup(Units::Utf16(&[0x61, 0xdc00])).is_none());
    }

    #[test]
    fn exact_strings_become_atoms_in_place_and_others_are_copied() {
        let mut table = AtomTable::default();
        let exact = LStr::exact(Units::Ascii(b"in place"));
        assert!(LStr::ptr_eq(&table.atomize_lstr(exact).as_lstr(), &exact));
        let mut grown = LStr::from("grow");
        grown.append_or_replace(&LStr::from("n buffer"));
        let atom = table.atomize_lstr(grown);
        assert!(!LStr::ptr_eq(&atom.as_lstr(), &grown) && atom.as_lstr().is_exact());
        assert!(atom.as_lstr() == grown);
        // A two-byte buffer of ASCII units becomes a one-byte atom.
        let wide_units = wide("ascii in two bytes");
        let two_byte = LStr::from("é")
            .concat_owned(&LStr::from_utf16(&wide_units))
            .slice_units(1, wide_units.len() + 1);
        let atom = table.atomize_lstr(two_byte);
        assert!(atom.as_lstr().known_ascii() && atom.as_lstr() == two_byte);
    }

    #[test]
    fn the_table_grows_and_keeps_every_atom() {
        let mut table = AtomTable::default();
        let base = table.len();
        let atoms: Vec<Atom> = (0..5000)
            .map(|n| table.atomize(Units::Ascii(format!("name{n}").as_bytes())))
            .collect();
        assert_eq!(table.len(), base + 5000);
        for (n, &atom) in atoms.iter().enumerate() {
            assert!(table.lookup(Units::Ascii(format!("name{n}").as_bytes())) == Some(atom));
        }
        assert!(table.slots.len() >= table.len() * 2);
    }

    #[test]
    fn static_names_are_every_tables_atoms() {
        let first = AtomTable::default();
        let second = AtomTable::default();
        for (index, name) in STATIC_NAMES.iter().enumerate() {
            let atom = StaticName(index as u16).atom();
            assert!(atom.is_static() && atom.as_lstr() == **name);
            assert!(first.lookup(Units::Ascii(name.as_bytes())) == Some(atom));
            assert!(second.lookup(Units::Ascii(name.as_bytes())) == Some(atom));
        }
        assert!(names::LENGTH.atom().as_lstr() == *"length");
        let mut table = AtomTable::default();
        assert!(table.atomize(Units::Utf16(&wide("prototype"))) == names::PROTOTYPE.atom());
        assert!(!table.atomize(Units::Ascii(b"lengths")).is_static());
        assert_eq!(table.heap_bytes(), LStr::exact_ascii_size(7));
    }

    #[test]
    fn keys_are_indices_or_atoms() {
        let mut table = AtomTable::default();
        for (text, index) in [
            ("0", Some(0)),
            ("7", Some(7)),
            ("4294967294", Some(PropKey::MAX_INDEX)),
            ("4294967295", None),
            ("42949672940", None),
            ("01", None),
            ("-1", None),
            ("1.0", None),
            ("", None),
            (" 1", None),
        ] {
            let key = PropKey::from_units(&mut table, Units::Ascii(text.as_bytes()));
            assert_eq!(key.as_index(), index, "{text:?}");
            assert_eq!(key.as_atom().is_some(), index.is_none(), "{text:?}");
            assert_eq!(index, crate::value::canonical_index(text), "{text:?}");
            assert!(PropKey::lookup(&table, Units::Utf16(&wide(text))) == Some(key));
            assert_eq!(key.cell_word().is_some(), index.is_none());
        }
        let key = PropKey::from_units(&mut table, Units::Ascii(b"length"));
        assert!(key == names::LENGTH.key() && key.as_atom() == Some(names::LENGTH.atom()));
        assert!(PropKey::lookup(&table, Units::Ascii(b"never made")).is_none());
        assert!(PropKey::index(5) != PropKey::from_units(&mut table, Units::Ascii(b"5x")));
    }
}
