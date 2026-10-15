//! `Intl.Segmenter` using the Unicode 17.0 UAX #29 default grapheme, word, and sentence rules.
//!
//! Every locale segments with these default rules: TRust JS ships no locale tailorings and no
//! dictionary-based word segmentation, so runs of Thai, Lao, Khmer, Myanmar or CJK text break as
//! UAX #29 alone defines (ECMA-402 leaves the boundaries implementation-dependent).

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
    resolve_locale,
};
use super::{
    ab, arg, canonicalize_locale_list, get_options_object as coerce_options, make_service,
};
use crate::interpreter::Interp;
use crate::value::{set_data, set_internal_slot, Gc, Object, Property, Value};
use std::rc::Rc;

pub fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "Segmenter", 0, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "segment", 1, |i, this, a| {
        segment(i, &this, &arg(a, 0))
    });
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
    install_segments_prototypes(it);
}

/// ECMA-402 #sec-%intlsegmentsprototype%-object and #sec-%intlsegmentiteratorprototype%-object.
fn install_segments_prototypes(it: &mut Interp) {
    let segments_proto = Object::new(Some(it.object_proto.clone()));
    it.def_method(&segments_proto, "containing", 1, segments_containing);
    if let Some(symbol) = it.iterator_sym.clone() {
        let iterator = it.make_native("[Symbol.iterator]", 0, segments_iterator);
        segments_proto.borrow_mut().props.insert(
            Interp::sym_key(&symbol),
            Property::builtin(Value::Obj(iterator)),
        );
    }
    let iterator_proto = Object::new(it.extra_protos.get("%IteratorPrototype%").cloned());
    it.def_method(&iterator_proto, "next", 0, segment_iterator_next);
    if let Some(key) = crate::builtins::to_string_tag_key(it) {
        iterator_proto.borrow_mut().props.insert(
            key,
            Property::data(Value::lstr("Segmenter String Iterator"), false, false, true),
        );
    }
    let protos = Rc::make_mut(&mut it.extra_protos);
    protos.insert("%IntlSegmentsPrototype%", segments_proto);
    protos.insert("%IntlSegmentIteratorPrototype%", iterator_proto);
}

fn construct(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    if !i.constructing {
        return Err(i.make_error("TypeError", "Intl.Segmenter requires 'new'"));
    }
    let requested = canonicalize_locale_list(i, &arg(a, 0))?;
    let options = coerce_options(i, &arg(a, 1))?;
    read_locale_matcher(i, &options)?;
    let granularity = get_option(
        i,
        &options,
        "granularity",
        &["grapheme", "word", "sentence"],
        Some("grapheme"),
    )?
    .unwrap();
    let resolved = resolve_locale(i, &requested, &[]);
    let obj = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.Segmenter")? {
        obj.borrow_mut().proto = Some(proto);
    }
    set_internal_slot(&obj, "#\u{0}sg", Value::Bool(true));
    set_internal_slot(&obj, "#\u{0}sg_locale", Value::lstr(resolved.locale));
    set_internal_slot(&obj, "#\u{0}sg_granularity", Value::lstr(granularity));
    Ok(Value::Obj(obj))
}

#[derive(Clone, Copy)]
struct CodePoint {
    offset: usize,
    value: u32,
}

fn decode_utf16(input: &[u16]) -> Vec<CodePoint> {
    let mut code_points = Vec::with_capacity(input.len());
    let mut offset = 0;
    while offset < input.len() {
        let first = input[offset];
        if (0xD800..=0xDBFF).contains(&first)
            && offset + 1 < input.len()
            && (0xDC00..=0xDFFF).contains(&input[offset + 1])
        {
            code_points.push(CodePoint {
                offset,
                value: 0x10000
                    + (((first as u32 - 0xD800) << 10) | (input[offset + 1] as u32 - 0xDC00)),
            });
            offset += 2;
        } else {
            // ECMAScript strings may contain an unpaired surrogate. It has no UAX #29 property
            // and therefore follows each algorithm's `Other` fallback as one code unit.
            code_points.push(CodePoint {
                offset,
                value: first as u32,
            });
            offset += 1;
        }
    }
    code_points
}

fn in_ranges(r: Option<&'static [(u32, u32)]>, cp: u32) -> bool {
    match r {
        Some(ranges) => ranges
            .binary_search_by(|&(lo, hi)| {
                if cp < lo {
                    std::cmp::Ordering::Greater
                } else if cp > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok(),
        None => false,
    }
}

fn grapheme_gb9c(
    properties: &[crate::unicode_segment::IndicConjunctBreak],
    boundary: usize,
) -> bool {
    use crate::unicode_segment::IndicConjunctBreak::{Consonant, Extend, Linker};
    if properties[boundary] != Consonant {
        return false;
    }
    let mut index = boundary;
    let mut saw_linker = false;
    while index > 0 {
        index -= 1;
        match properties[index] {
            Extend => {}
            Linker => saw_linker = true,
            Consonant => return saw_linker,
            _ => return false,
        }
    }
    false
}

/// UAX #29 extended grapheme cluster boundaries (GB1–GB999), in UTF-16 code units.
#[allow(clippy::if_same_then_else)]
fn grapheme_boundaries(s: &[u16]) -> Vec<(usize, bool)> {
    use crate::unicode_segment::GraphemeBreak as G;
    let code_points = decode_utf16(s);
    if code_points.is_empty() {
        return Vec::new();
    }
    let properties: Vec<G> = code_points
        .iter()
        .map(|point| crate::unicode_segment::grapheme_break(point.value))
        .collect();
    let conjunct: Vec<_> = code_points
        .iter()
        .map(|point| crate::unicode_segment::indic_conjunct_break(point.value))
        .collect();
    let mut output = vec![(0, false)];
    let mut regional_indicators = usize::from(properties[0] == G::RegionalIndicator);
    for boundary in 1..code_points.len() {
        let a = properties[boundary - 1];
        let b = properties[boundary];
        let no_break = if a == G::Cr && b == G::Lf {
            true // GB3
        } else if matches!(a, G::Control | G::Cr | G::Lf) || matches!(b, G::Control | G::Cr | G::Lf)
        {
            false // GB4 / GB5
        } else if a == G::L && matches!(b, G::L | G::V | G::Lv | G::Lvt) {
            true // GB6
        } else if matches!(a, G::Lv | G::V) && matches!(b, G::V | G::T) {
            true // GB7
        } else if matches!(a, G::Lvt | G::T) && b == G::T {
            true // GB8
        } else if matches!(b, G::Extend | G::Zwj) {
            true // GB9
        } else if b == G::Spacingmark {
            true // GB9a
        } else if a == G::Prepend {
            true // GB9b
        } else if grapheme_gb9c(&conjunct, boundary) {
            true // GB9c
        } else if a == G::Zwj
            && crate::unicode_segment::is_extended_pictographic(code_points[boundary].value)
        {
            // GB11: Extended_Pictographic Extend* ZWJ × Extended_Pictographic.
            let mut index = boundary - 1;
            while index > 0 && properties[index - 1] == G::Extend {
                index -= 1;
            }
            index > 0
                && crate::unicode_segment::is_extended_pictographic(code_points[index - 1].value)
        } else {
            a == G::RegionalIndicator && b == G::RegionalIndicator && regional_indicators % 2 == 1
            // GB12/13
        };
        if !no_break {
            output.push((code_points[boundary].offset, false));
        }
        regional_indicators = if b == G::RegionalIndicator {
            if a == G::RegionalIndicator {
                regional_indicators + 1
            } else {
                1
            }
        } else {
            0
        };
    }
    output
}

fn word_ignored(property: crate::unicode_segment::WordBreak) -> bool {
    use crate::unicode_segment::WordBreak as W;
    matches!(property, W::Extend | W::Format | W::Zwj)
}

fn word_newline(property: crate::unicode_segment::WordBreak) -> bool {
    use crate::unicode_segment::WordBreak as W;
    matches!(property, W::Cr | W::Lf | W::Newline)
}

fn word_previous(properties: &[crate::unicode_segment::WordBreak], before: usize) -> Option<usize> {
    if before == 0 {
        return None;
    }
    let mut index = before - 1;
    while word_ignored(properties[index]) {
        if index == 0 || word_newline(properties[index - 1]) {
            return Some(index);
        }
        index -= 1;
    }
    Some(index)
}

fn word_next(properties: &[crate::unicode_segment::WordBreak], after: usize) -> Option<usize> {
    let mut index = after + 1;
    while index < properties.len() && word_ignored(properties[index]) {
        index += 1;
    }
    (index < properties.len()).then_some(index)
}

fn word_ah_letter(property: crate::unicode_segment::WordBreak) -> bool {
    use crate::unicode_segment::WordBreak as W;
    matches!(property, W::Aletter | W::HebrewLetter)
}

fn word_mid_letter(property: crate::unicode_segment::WordBreak) -> bool {
    use crate::unicode_segment::WordBreak as W;
    matches!(property, W::Midletter | W::Midnumlet | W::SingleQuote)
}

fn word_mid_number(property: crate::unicode_segment::WordBreak) -> bool {
    use crate::unicode_segment::WordBreak as W;
    matches!(property, W::Midnum | W::Midnumlet | W::SingleQuote)
}

fn word_boundary(
    code_points: &[CodePoint],
    properties: &[crate::unicode_segment::WordBreak],
    boundary: usize,
) -> bool {
    use crate::unicode_segment::WordBreak as W;
    let raw_left = properties[boundary - 1];
    let right = properties[boundary];
    if raw_left == W::Cr && right == W::Lf {
        return false; // WB3
    }
    if word_newline(raw_left) || word_newline(right) {
        return true; // WB3a/WB3b
    }
    if raw_left == W::Zwj
        && crate::unicode_segment::is_extended_pictographic(code_points[boundary].value)
    {
        return false; // WB3c
    }
    if raw_left == W::Wsegspace && right == W::Wsegspace {
        return false; // WB3d
    }
    if word_ignored(right) {
        return false; // WB4
    }
    let Some(left_index) = word_previous(properties, boundary) else {
        return true;
    };
    let left = if word_ignored(properties[left_index]) {
        W::Other
    } else {
        properties[left_index]
    };
    if word_ah_letter(left) && word_ah_letter(right) {
        return false; // WB5
    }
    let next = word_next(properties, boundary).map(|index| properties[index]);
    if word_ah_letter(left) && word_mid_letter(right) && next.is_some_and(word_ah_letter) {
        return false; // WB6
    }
    let previous = word_previous(properties, left_index).map(|index| properties[index]);
    if word_mid_letter(left) && word_ah_letter(right) && previous.is_some_and(word_ah_letter) {
        return false; // WB7
    }
    if left == W::HebrewLetter && right == W::SingleQuote {
        return false; // WB7a
    }
    if left == W::HebrewLetter && right == W::DoubleQuote && next == Some(W::HebrewLetter) {
        return false; // WB7b
    }
    if left == W::DoubleQuote && right == W::HebrewLetter && previous == Some(W::HebrewLetter) {
        return false; // WB7c
    }
    if left == W::Numeric && right == W::Numeric {
        return false; // WB8
    }
    if word_ah_letter(left) && right == W::Numeric {
        return false; // WB9
    }
    if left == W::Numeric && word_ah_letter(right) {
        return false; // WB10
    }
    if word_mid_number(left) && right == W::Numeric && previous == Some(W::Numeric) {
        return false; // WB11
    }
    if left == W::Numeric && word_mid_number(right) && next == Some(W::Numeric) {
        return false; // WB12
    }
    if left == W::Katakana && right == W::Katakana {
        return false; // WB13
    }
    if matches!(
        left,
        W::Aletter | W::HebrewLetter | W::Numeric | W::Katakana | W::Extendnumlet
    ) && right == W::Extendnumlet
    {
        return false; // WB13a
    }
    if left == W::Extendnumlet
        && matches!(
            right,
            W::Aletter | W::HebrewLetter | W::Numeric | W::Katakana
        )
    {
        return false; // WB13b
    }
    if left == W::RegionalIndicator && right == W::RegionalIndicator {
        let mut count = 0;
        let mut cursor = Some(left_index);
        while let Some(index) = cursor {
            if properties[index] != W::RegionalIndicator {
                break;
            }
            count += 1;
            cursor = word_previous(properties, index);
        }
        if count % 2 == 1 {
            return false; // WB15/WB16
        }
    }
    true // WB999
}

fn word_like(
    code_points: &[CodePoint],
    properties: &[crate::unicode_segment::WordBreak],
    alphabetic: Option<&'static [(u32, u32)]>,
    ideographic: Option<&'static [(u32, u32)]>,
) -> bool {
    use crate::unicode_segment::WordBreak as W;
    code_points.iter().zip(properties).any(|(point, property)| {
        matches!(
            property,
            W::Aletter | W::HebrewLetter | W::Numeric | W::Katakana | W::Extendnumlet
        ) || in_ranges(alphabetic, point.value)
            || in_ranges(ideographic, point.value)
    })
}

fn word_boundaries(input: &[u16]) -> Vec<(usize, bool)> {
    let code_points = decode_utf16(input);
    if code_points.is_empty() {
        return Vec::new();
    }
    let properties: Vec<_> = code_points
        .iter()
        .map(|point| crate::unicode_segment::word_break(point.value))
        .collect();
    // Resolve the generated binary-property tables once for the entire string. `lookup` performs
    // loose-name canonicalization, so doing it per code point would allocate on the hot path.
    let alphabetic = crate::unicode_props::lookup("alphabetic", None);
    let ideographic = crate::unicode_props::lookup("ideographic", None);
    let mut starts = vec![0usize];
    for boundary in 1..code_points.len() {
        if word_boundary(&code_points, &properties, boundary) {
            starts.push(boundary);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let end = starts.get(index + 1).copied().unwrap_or(code_points.len());
            (
                code_points[start].offset,
                word_like(
                    &code_points[start..end],
                    &properties[start..end],
                    alphabetic,
                    ideographic,
                ),
            )
        })
        .collect()
}

fn sentence_ignored(property: crate::unicode_segment::SentenceBreak) -> bool {
    use crate::unicode_segment::SentenceBreak as S;
    matches!(property, S::Extend | S::Format)
}

fn sentence_para(property: crate::unicode_segment::SentenceBreak) -> bool {
    use crate::unicode_segment::SentenceBreak as S;
    matches!(property, S::Sep | S::Cr | S::Lf)
}

fn sentence_previous(
    properties: &[crate::unicode_segment::SentenceBreak],
    before: usize,
) -> Option<usize> {
    let mut index = before.checked_sub(1)?;
    while sentence_ignored(properties[index]) {
        if index == 0 || sentence_para(properties[index - 1]) {
            return Some(index);
        }
        index -= 1;
    }
    Some(index)
}

fn sentence_skip_ignored(
    properties: &[crate::unicode_segment::SentenceBreak],
    mut index: usize,
) -> usize {
    while index < properties.len() && sentence_ignored(properties[index]) {
        index += 1;
    }
    index
}

fn sentence_boundaries(input: &[u16]) -> Vec<(usize, bool)> {
    use crate::unicode_segment::SentenceBreak as S;
    let code_points = decode_utf16(input);
    if code_points.is_empty() {
        return Vec::new();
    }
    let properties: Vec<_> = code_points
        .iter()
        .map(|point| crate::unicode_segment::sentence_break(point.value))
        .collect();
    let mut starts = vec![0usize];
    let mut index = 0;
    while index < properties.len() {
        let property = properties[index];
        if property == S::Cr && properties.get(index + 1) == Some(&S::Lf) {
            if index + 2 < code_points.len() {
                starts.push(index + 2); // SB3/SB4
            }
            index += 2;
            continue;
        }
        if sentence_para(property) {
            if index + 1 < code_points.len() {
                starts.push(index + 1); // SB4
            }
            index += 1;
            continue;
        }
        if !matches!(property, S::Aterm | S::Sterm) {
            index += 1;
            continue;
        }

        let next = sentence_skip_ignored(&properties, index + 1);
        let previous = sentence_previous(&properties, index);
        if property == S::Aterm && properties.get(next) == Some(&S::Numeric) {
            index += 1; // SB6
            continue;
        }
        if property == S::Aterm
            && previous.is_some_and(|p| matches!(properties[p], S::Upper | S::Lower))
            && properties.get(next) == Some(&S::Upper)
        {
            index += 1; // SB7
            continue;
        }

        let mut tail = next;
        while properties.get(tail) == Some(&S::Close) {
            tail = sentence_skip_ignored(&properties, tail + 1);
        }
        while properties.get(tail) == Some(&S::Sp) {
            tail = sentence_skip_ignored(&properties, tail + 1);
        }

        let mut suppress = false;
        if property == S::Aterm {
            let mut lookahead = tail;
            while let Some(candidate) = properties.get(lookahead) {
                if *candidate == S::Lower {
                    suppress = true; // SB8
                    break;
                }
                if matches!(
                    candidate,
                    S::Oletter | S::Upper | S::Cr | S::Lf | S::Sep | S::Aterm | S::Sterm
                ) {
                    break;
                }
                lookahead = sentence_skip_ignored(&properties, lookahead + 1);
            }
        }
        if matches!(
            properties.get(tail),
            Some(S::Scontinue | S::Aterm | S::Sterm)
        ) {
            suppress = true; // SB8a
        }
        if suppress {
            index += 1;
            continue;
        }

        // SB9–SB11 include Close*, Sp*, and one optional paragraph separator in this sentence.
        let mut end = tail;
        if properties.get(end) == Some(&S::Cr) && properties.get(end + 1) == Some(&S::Lf) {
            end += 2;
        } else if properties.get(end).is_some_and(|p| sentence_para(*p)) {
            end += 1;
        }
        if end < code_points.len() {
            starts.push(end);
        }
        index += 1;
    }
    starts.sort_unstable();
    starts.dedup();
    starts
        .into_iter()
        .map(|start| (code_points[start].offset, false))
        .collect()
}

fn boundaries(s: &[u16], granularity: &str) -> Vec<(usize, bool)> {
    match granularity {
        "grapheme" => grapheme_boundaries(s),
        "word" => word_boundaries(s),
        _ => sentence_boundaries(s),
    }
}

/// ECMA-402 #sec-intl.segmenter.prototype.segment.
fn segment(i: &mut Interp, this: &Value, input: &Value) -> Result<Value, Value> {
    let segmenter = brand_slot(i, this, "#\u{0}sg")?;
    let string = ab(i.to_string(input))?;
    // CreateSegmentsObject: [[SegmentsSegmenter]] and [[SegmentsString]].
    let segments = Object::new(i.extra_protos.get("%IntlSegmentsPrototype%").cloned());
    set_internal_slot(&segments, SEGMENTER_SLOT, Value::Obj(segmenter));
    set_internal_slot(&segments, STRING_SLOT, Value::Str(string));
    Ok(Value::Obj(segments))
}

/// Segments instances' [[SegmentsSegmenter]] and [[SegmentsString]].
const SEGMENTER_SLOT: &str = "#\u{0}segments_segmenter";
const STRING_SLOT: &str = "#\u{0}segments_string";
/// Segment Iterators' [[IteratingSegmenter]] and [[IteratedString]].
const ITERATING_SEGMENTER_SLOT: &str = "#\u{0}segment_iterator_segmenter";
const ITERATED_STRING_SLOT: &str = "#\u{0}segment_iterator_string";
/// The string's boundaries under the segmenter, computed on first use and shared by the
/// Segments object's iterators: a hidden ArrayBuffer of [`RECORD`]-byte records (u32 UTF-16
/// start, u32 byte start in the engine string, u8 word-like), one per segment, then a sentinel
/// record holding the string's lengths.
const BOUNDARIES_SLOT: &str = "#\u{0}segments_boundaries";
const NEXT_INDEX_SLOT: &str = "#\u{0}segments_next_index";
const RECORD: usize = 9;

/// A Segments instance's or Segment Iterator's internal slots, after RequireInternalSlot.
struct SegmentsSlots {
    holder: Gc,
    segmenter: Gc,
    string: crate::lstr::LStr,
}

fn segments_slots(i: &mut Interp, this: &Value) -> Result<SegmentsSlots, Value> {
    slots_of(this, SEGMENTER_SLOT, STRING_SLOT)
        .ok_or_else(|| i.make_error("TypeError", "receiver is not a Segments object"))
}

fn slots_of(this: &Value, segmenter_slot: &str, string_slot: &str) -> Option<SegmentsSlots> {
    this.as_obj().and_then(|holder| {
        let object = holder.borrow();
        let segmenter = object.internal_slots().get(segmenter_slot)?.value();
        let string = object.internal_slots().get(string_slot)?.value();
        match (segmenter, string) {
            (Value::Obj(segmenter), Value::Str(string)) => Some(SegmentsSlots {
                holder: holder.clone(),
                segmenter,
                string,
            }),
            _ => None,
        }
    })
}

fn granularity_of(segmenter: &Gc) -> String {
    match segmenter
        .borrow()
        .internal_slots()
        .get("#\u{0}sg_granularity")
        .map(|p| p.value())
    {
        Some(Value::Str(s)) => s.to_string(),
        _ => "grapheme".to_string(),
    }
}

/// The boundary records of the slots' string, computing and caching them on first use.
fn boundary_records(i: &mut Interp, slots: &SegmentsSlots) -> crate::interpreter::ArrayBufferBytes {
    let cached = slots
        .holder
        .borrow()
        .internal_slots()
        .get(BOUNDARIES_SLOT)
        .map(|p| p.value());
    if let Some(Value::Obj(buffer)) = cached {
        if let Some(bytes) = i.local_buffer(Gc::as_ptr(&buffer) as usize) {
            return bytes.clone();
        }
    }
    let text: &str = &slots.string;
    // jstr units: lone surrogates must round-trip (each is its own single-unit segment).
    let units = crate::jstr::units(text);
    let starts = boundaries(&units, &granularity_of(&slots.segmenter));
    let mut bytes = Vec::with_capacity((starts.len() + 1) * RECORD);
    let mut push = |unit: usize, byte: usize, word_like: bool| {
        bytes.extend_from_slice(&(unit as u32).to_le_bytes());
        bytes.extend_from_slice(&(byte as u32).to_le_bytes());
        bytes.push(u8::from(word_like));
    };
    // Boundaries fall between code points, so each is a character boundary of the engine string.
    let mut next = starts.iter().peekable();
    let (mut unit, mut byte) = (0usize, 0usize);
    for character in text.chars() {
        while let Some(&&(start, word_like)) = next.peek() {
            if start != unit {
                break;
            }
            push(unit, byte, word_like);
            next.next();
        }
        unit += if crate::jstr::smuggled(character).is_some() {
            1
        } else {
            character.len_utf16()
        };
        byte += character.len_utf8();
    }
    push(unit, byte, false);
    let storage = Rc::new(std::cell::RefCell::new(bytes));
    let (buffer, _) = crate::builtins::array_buffer_with_storage(i, storage.clone());
    set_internal_slot(&slots.holder, BOUNDARIES_SLOT, buffer);
    storage
}

/// Record `index`: (UTF-16 start, byte start, word-like).
fn record_at(bytes: &[u8], index: usize) -> (usize, usize, bool) {
    let at = index * RECORD;
    let word = |from: usize| u32::from_le_bytes(bytes[from..from + 4].try_into().unwrap());
    (word(at) as usize, word(at + 4) as usize, bytes[at + 8] != 0)
}

/// The number of segments (records before the sentinel) and the string's UTF-16 length.
fn segment_count_and_length(bytes: &[u8]) -> (usize, usize) {
    let count = bytes.len() / RECORD - 1;
    (count, record_at(bytes, count).0)
}

/// The segment containing code unit `n` (< length): the last boundary at or before `n`
/// (FindBoundary ~before~); the next record, possibly the sentinel, is its end (~after~).
fn segment_at(bytes: &[u8], n: usize) -> usize {
    let (count, _) = segment_count_and_length(bytes);
    let (mut low, mut high) = (0, count);
    while low < high {
        let middle = low + (high - low) / 2;
        if record_at(bytes, middle).0 <= n {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low - 1
}

/// ECMA-402 #sec-createsegmentdataobject for segment `index`.
fn segment_data(i: &mut Interp, slots: &SegmentsSlots, bytes: &[u8], index: usize) -> Value {
    let (start, start_byte, word_like) = record_at(bytes, index);
    let (_, end_byte, _) = record_at(bytes, index + 1);
    let result = i.new_object();
    set_data(
        &result,
        "segment",
        Value::Str(slots.string.slice_bytes(start_byte, end_byte)),
    );
    set_data(&result, "index", Value::Num(start as f64));
    set_data(&result, "input", Value::Str(slots.string.clone()));
    if granularity_of(&slots.segmenter) == "word" {
        set_data(&result, "isWordLike", Value::Bool(word_like));
    }
    Value::Obj(result)
}

/// ECMA-402 #sec-%intlsegmentsprototype%.containing.
fn segments_containing(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    let slots = segments_slots(i, &this)?;
    // ToIntegerOrInfinity(index).
    let number = ab(i.to_number(&arg(a, 0)))?;
    let n = if number.is_nan() { 0.0 } else { number.trunc() };
    let storage = boundary_records(i, &slots);
    let bytes = storage.borrow();
    let (_, length) = segment_count_and_length(&bytes);
    if n < 0.0 || n >= length as f64 {
        return Ok(Value::Undefined);
    }
    let index = segment_at(&bytes, n as usize);
    Ok(segment_data(i, &slots, &bytes, index))
}

/// ECMA-402 #sec-%intlsegmentsprototype%-%symbol.iterator%: CreateSegmentIterator.
fn segments_iterator(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let slots = segments_slots(i, &this)?;
    let iterator = Object::new(
        i.extra_protos
            .get("%IntlSegmentIteratorPrototype%")
            .cloned(),
    );
    // [[IteratingSegmenter]], [[IteratedString]] and [[IteratedStringNextSegmentCodeUnitIndex]];
    // the iterator shares its Segments object's boundary cache.
    set_internal_slot(
        &iterator,
        ITERATING_SEGMENTER_SLOT,
        Value::Obj(slots.segmenter.clone()),
    );
    set_internal_slot(
        &iterator,
        ITERATED_STRING_SLOT,
        Value::Str(slots.string.clone()),
    );
    set_internal_slot(&iterator, NEXT_INDEX_SLOT, Value::Num(0.0));
    boundary_records(i, &slots);
    let buffer = slots
        .holder
        .borrow()
        .internal_slots()
        .get(BOUNDARIES_SLOT)
        .map(|p| p.value());
    if let Some(buffer) = buffer {
        set_internal_slot(&iterator, BOUNDARIES_SLOT, buffer);
    }
    Ok(Value::Obj(iterator))
}

/// ECMA-402 #sec-%intlsegmentiteratorprototype%.next.
fn segment_iterator_next(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let slots = slots_of(&this, ITERATING_SEGMENTER_SLOT, ITERATED_STRING_SLOT)
        .ok_or_else(|| i.make_error("TypeError", "receiver is not a Segment Iterator"))?;
    let start = match slots
        .holder
        .borrow()
        .internal_slots()
        .get(NEXT_INDEX_SLOT)
        .map(|p| p.value())
    {
        Some(Value::Num(start)) => start as usize,
        _ => 0,
    };
    let storage = boundary_records(i, &slots);
    let bytes = storage.borrow();
    let (_, length) = segment_count_and_length(&bytes);
    if start >= length {
        return Ok(i.iter_result_obj(Value::Undefined, true));
    }
    let index = segment_at(&bytes, start);
    let (end, _, _) = record_at(&bytes, index + 1);
    set_internal_slot(&slots.holder, NEXT_INDEX_SLOT, Value::Num(end as f64));
    let data = segment_data(i, &slots, &bytes, index);
    drop(bytes);
    Ok(i.iter_result_obj(data, false))
}

fn resolved_options(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = brand_slot(i, &this, "#\u{0}sg")?;
    let get = |k: &str| {
        o.borrow()
            .internal_slots()
            .get(k)
            .map(|p| p.value())
            .unwrap_or(Value::Undefined)
    };
    let res = i.new_object();
    set_data(&res, "locale", get("#\u{0}sg_locale"));
    set_data(&res, "granularity", get("#\u{0}sg_granularity"));
    Ok(Value::Obj(res))
}

#[cfg(test)]
mod tests {
    use super::boundaries;

    const GRAPHEME_TESTS: &str = include_str!("../../tests/unicode-17.0.0/GraphemeBreakTest.txt");
    const WORD_TESTS: &str = include_str!("../../tests/unicode-17.0.0/WordBreakTest.txt");
    const SENTENCE_TESTS: &str = include_str!("../../tests/unicode-17.0.0/SentenceBreakTest.txt");

    fn parse_case(line: &str) -> Option<(Vec<u16>, Vec<usize>)> {
        let body = line.split('#').next()?.trim();
        if body.is_empty() {
            return None;
        }
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for token in body.split_ascii_whitespace() {
            match token {
                "÷" => expected.push(input.len()),
                "×" => {}
                code_point => {
                    let value = u32::from_str_radix(code_point, 16).ok()?;
                    let character = char::from_u32(value)?;
                    let mut units = [0; 2];
                    input.extend_from_slice(character.encode_utf16(&mut units));
                }
            }
        }
        if expected.last() == Some(&input.len()) {
            expected.pop();
        }
        Some((input, expected))
    }

    fn assert_break_test(data: &str, granularity: &str) {
        for (line_number, line) in data.lines().enumerate() {
            let Some((input, expected)) = parse_case(line) else {
                continue;
            };
            let actual: Vec<usize> = boundaries(&input, granularity)
                .into_iter()
                .map(|(offset, _)| offset)
                .collect();
            assert_eq!(
                actual,
                expected,
                "Unicode 17.0 {granularity} break test line {}: {line}",
                line_number + 1
            );
        }
    }

    #[test]
    fn unicode_17_grapheme_break_conformance() {
        assert_break_test(GRAPHEME_TESTS, "grapheme");
    }

    #[test]
    fn unicode_17_word_break_conformance() {
        assert_break_test(WORD_TESTS, "word");
    }

    #[test]
    fn unicode_17_sentence_break_conformance() {
        assert_break_test(SENTENCE_TESTS, "sentence");
    }
}
