//! The element census (`TRUST_JS_GC_CENSUS=1`; W2c's step 1, the plan's "W2c: elements as
//! 8-byte words"): after each collection, every live object with element storage, and every
//! array, is classified and its element storage's bytes added up by allocated size, beside a
//! model of the same objects under W2c's word representation, as one `[gc-elements]` line. It
//! walks the heap at each collection: a diagnostic only, never on unless asked for.
use super::*;

pub(crate) fn enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var_os("TRUST_JS_GC_CENSUS").is_some())
}

/// The allocated size of a block of `bytes`: its small-cell class, or its own size when it is
/// large.
fn class_bytes(bytes: usize) -> u64 {
    if bytes == 0 {
        return 0;
    }
    match crate::gc_core::small_cell_class(bytes) {
        Some((_, size)) => size as u64,
        None => bytes as u64,
    }
}

const FAMILIES: [&str; 3] = ["array", "arguments", "other"];
/// Element storage parts, then the W2c model of the same objects.
const PARTS: [&str; 11] = [
    "cell",
    "sidecar",
    "packed_box",
    "values",
    "dirty",
    "slot_map",
    "mirror",
    "index",
    "fields_spill",
    "layout",
    "w2c",
];

#[derive(Default)]
struct Family {
    objects: u64,
    /// Element slots: the element words' length.
    slots: u64,
    elements: u64,
    holes: u64,
    /// Elements whose map is plain: no accessor, no attributes but the defaults, no far key;
    /// of them, those with holes (or a `length` past the slots).
    plain: u64,
    holey: u64,
    /// Plain maps whose elements are all Numbers, and of them all exact int32s.
    numbers: u64,
    int32: u64,
    bytes: [u64; PARTS.len()],
}

/// Add one object's element storage to its family; an ordinary object without index keys or
/// element storage is not counted.
fn record(families: &mut [Family; 3], object: &Object) {
    let family = match object.exotic {
        Exotic::Array => 0,
        Exotic::Arguments(_) => 1,
        _ => 2,
    };
    let p = &object.props;
    let index_entries = p
        .entries
        .keys()
        .iter()
        .filter(|key| key.as_index().is_some())
        .count();
    let words = p.elems.words();
    if family == 2 && words.is_none() && index_entries == 0 {
        return;
    }
    let f = &mut families[family];
    f.objects += 1;
    let mut slots = 0u64;
    let mut elements = 0u64;
    let mut nonplain = false;
    let mut all_number = true;
    let mut all_int32 = true;
    let mut visit = |element: Element<'_>| {
        elements += 1;
        let plain_attributes = match element {
            Element::Word(..) => true,
            Element::Entry(property) => property.has_plain_element_attributes(),
        };
        if element.accessor() || !plain_attributes {
            nonplain = true;
        }
        match element.number_value() {
            Some(n) if !element.accessor() => {
                if !f64_exact_i32(n) {
                    all_int32 = false;
                }
            }
            _ => {
                all_number = false;
                all_int32 = false;
            }
        }
    };
    if let Some(words) = words {
        slots = words.len() as u64;
        for n in 0..words.len() {
            if let Some(value) = words.get(n) {
                visit(Element::Word(value, words.integrity()));
            }
        }
    }
    // The index keys in the named entries (non-plain, far or frozen elements).
    for (key, property) in p.entries.iter() {
        if key.as_index().is_some() {
            visit(Element::entry(property));
        }
    }
    let holes = slots.saturating_sub(elements.min(slots));
    let length = (family == 0)
        .then(|| p.length_property().and_then(Property::number_value))
        .flatten();
    let far = p.has_far.get();
    let plain = !nonplain && !far;
    f.slots += slots;
    f.elements += elements;
    f.holes += holes;
    if plain {
        f.plain += 1;
        if holes > 0 || length.is_some_and(|length| length > slots as f64) {
            f.holey += 1;
        }
        if elements > 0 && all_number {
            f.numbers += 1;
            if all_int32 {
                f.int32 += 1;
            }
        }
    }
    // Bytes by allocated size (small-cell classes; a large block counts its own bytes).
    let mut b = [0u64; PARTS.len()];
    b[0] = class_bytes(std::mem::size_of::<RefCell<Object>>());
    let index = p.elems.index();
    let mut capacity = 0usize;
    if let Some(words) = words {
        // The word block: its header, front room, capacity and remembered record (its dirty
        // bitmap; the packed box and the dirty column stay zero with words).
        capacity = words.capacity();
        b[3] = class_bytes(words.block_bytes());
    }
    // The sidecar (b[1]), the slot map (b[5]) and the mirror (b[6]) are gone (W2c step 3); their
    // columns stay, at zero, so the lines compare with step 1's.
    if let Some(index) = index {
        b[7] = class_bytes(std::mem::size_of::<KeyIndex>()) + class_bytes(index.bytes());
    }
    if p.entries.fields.spilled() {
        b[8] = class_bytes(p.entries.fields.capacity() * std::mem::size_of::<Property>());
    }
    if let Some(layout) = p.entries.layout.as_ref().filter(|l| !l.is_shared()) {
        b[9] = class_bytes(std::mem::size_of::<PropertyLayoutData>())
            + class_bytes(layout.capacity() * 8);
    }
    // The W2c model: a plain map keeps its cell and puts its elements in one block of 8-byte
    // words behind a 16-byte header, at its word capacity or its slots; the sidecar, the
    // mirror, the slot map and the index entries' named storage go, and a block over 64 words
    // keeps its remembered record after its words. Others keep today's bytes (the dictionary).
    let today: u64 = b[..PARTS.len() - 1].iter().sum();
    b[PARTS.len() - 1] = if plain {
        let words = if p.elems.has_words() {
            capacity
        } else {
            slots as usize
        };
        let named = p.entries.len() - index_entries;
        let mut model = b[0];
        if words != 0 {
            model += class_bytes(16 + 8 * (words + word_block::record_words(words)));
        }
        if named > INLINE_NAMED_SLOTS {
            model += class_bytes(named * std::mem::size_of::<Property>());
        }
        if index.is_some() {
            model += b[7];
        }
        model
    } else {
        today
    };
    for (sum, part) in f.bytes.iter_mut().zip(b) {
        *sum += part;
    }
}

/// One `[gc-elements]` line for the heap after a collection of `kind`.
pub(crate) fn sample(heap: &GcHeap, kind: &str) {
    eprintln!("{}", line(kind, &census(heap)));
}

/// The live heap's objects with elements, by family.
fn census(heap: &GcHeap) -> [Family; 3] {
    let cells = heap.core.borrow().live_cells(heap.object_desc);
    let mut families: [Family; 3] = Default::default();
    for cell in cells {
        // SAFETY: a live object cell is initialized; safepoints hold no mutable borrow.
        if let Ok(object) = unsafe { (*(cell as *const RefCell<Object>)).try_borrow_unguarded() } {
            record(&mut families, object);
        }
    }
    families
}

fn line(kind: &str, families: &[Family; 3]) -> String {
    let mut line = format!("[gc-elements] kind={kind}");
    for (name, f) in FAMILIES.iter().zip(families) {
        let parts = PARTS
            .iter()
            .zip(f.bytes)
            .map(|(part, bytes)| format!("{part}:{bytes}"))
            .collect::<Vec<_>>()
            .join(",");
        line.push_str(&format!(
            " {name}=objects:{},slots:{},elements:{},holes:{},plain:{},holey:{},numbers:{},int32:{},{parts}",
            f.objects, f.slots, f.elements, f.holes, f.plain, f.holey, f.numbers, f.int32,
        ));
    }
    line
}

#[cfg(test)]
mod tests {
    use crate::{Completion, Engine};

    #[test]
    fn the_census_counts_live_elements_by_family_and_fact() {
        let mut engine = Engine::new();
        let source =
            "var keep=[[1,2,3],[1.5,2],['a',,'c'],[1,{}],(function(){return arguments})(4,5)];\
            Object.defineProperty(keep[3],0,{get(){return 1}});var o={0:'x',1:'y'};'ok'";
        assert!(matches!(engine.eval(source, false), Ok(Completion::Value(v)) if v == "ok"));
        engine.interp.collect_garbage_for_host();
        let families = super::census(&engine.interp.gc_heap);
        let line = super::line("major", &families);
        let [arrays, arguments, other] = &families;
        // The five literal arrays and `keep` itself, among the realm's own.
        assert!(arrays.objects >= 6, "{line}");
        // [1,2,3] is plain, numeric and int32; [1.5,2] plain and numeric; ['a',,'c'] plain and
        // holey; [1,{}] holds an accessor at 0 and is not plain.
        assert!(arrays.int32 >= 1 && arrays.numbers > arrays.int32, "{line}");
        assert!(arrays.holey >= 1 && arrays.holes >= 1, "{line}");
        assert!(arrays.objects > arrays.plain, "{line}");
        assert!(arguments.objects >= 1 && arguments.elements >= 2, "{line}");
        assert!(other.objects >= 1 && other.elements >= 2, "{line}");
        assert!(
            line.starts_with("[gc-elements] kind=major array=objects:"),
            "{line}"
        );
        for f in &families {
            // The model never adds a block today's storage would not have: a plain map's words
            // take at most its packed values' and sidecar's room.
            assert!(
                f.bytes[super::PARTS.len() - 1]
                    <= f.bytes[..super::PARTS.len() - 1].iter().sum::<u64>(),
                "{line}"
            );
        }
    }
}
