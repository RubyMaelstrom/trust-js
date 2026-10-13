//! Compact Unicode 17 DUCET reader for UTS #10 collation-element matching.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Element {
    /// Primary weights reserve the low 32 bits for CLDR tailoring insertions. DUCET and
    /// algorithmic implicit weights occupy the high half, so a locale rule can place a weight
    /// strictly after an existing primary without renumbering the generated root table.
    pub primary: u64,
    pub secondary: u16,
    pub tertiary: u16,
    pub variable: bool,
}

#[cfg(test)]
const DUCET: &[u8] = include_bytes!("unicode_collation.bin");
const CLDR_ROOT: &[u8] = include_bytes!("cldr_root_collation.bin");
const HEADER_SIZE: usize = 24;
const SINGLE_SIZE: usize = 12;
const CONTRACTION_SIZE: usize = 20;

fn u16_at(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn u64_at(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn counts(data: &[u8]) -> (usize, usize, usize, usize) {
    debug_assert!(matches!(&data[..8], b"LUCA17\0\x01" | b"LCLR17\0\x01"));
    (
        u32_at(data, 8) as usize,
        u32_at(data, 12) as usize,
        u32_at(data, 16) as usize,
        u32_at(data, 20) as usize,
    )
}

fn single_base() -> usize {
    HEADER_SIZE
}

fn contraction_base(data: &[u8]) -> usize {
    let (singles, _, _, _) = counts(data);
    single_base() + singles * SINGLE_SIZE
}

fn sequence_base(data: &[u8]) -> usize {
    let (_, contractions, _, _) = counts(data);
    contraction_base(data) + contractions * CONTRACTION_SIZE
}

fn weight_base(data: &[u8]) -> usize {
    let (_, _, sequences, _) = counts(data);
    sequence_base(data) + sequences * 4
}

fn append_weights(data: &[u8], offset: usize, length: usize, output: &mut Vec<Element>) {
    let base = weight_base(data) + offset * 8;
    output.reserve(length);
    for index in 0..length {
        let packed = u64_at(data, base + index * 8);
        output.push(Element {
            primary: ((packed >> 32) & 0xffff) << 32,
            secondary: ((packed >> 16) & 0xffff) as u16,
            tertiary: (packed & 0xffff) as u16,
            variable: packed & (1 << 48) != 0,
        });
    }
}

fn append_single(data: &[u8], code_point: u32, output: &mut Vec<Element>) -> bool {
    let (count, _, _, _) = counts(data);
    let mut low = 0;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = single_base() + middle * SINGLE_SIZE;
        match u32_at(data, base).cmp(&code_point) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => {
                append_weights(
                    data,
                    u32_at(data, base + 4) as usize,
                    u16_at(data, base + 8) as usize,
                    output,
                );
                return true;
            }
        }
    }
    false
}

/// The first non-zero primary weight for a singleton DUCET mapping. Locale tailorings use this
/// as a stable reset anchor (UTS #35 `&anchor<tailored`) without allocating a temporary CE list.
pub(crate) fn first_primary(code_point: u32) -> Option<u64> {
    let data = CLDR_ROOT;
    let (count, _, _, _) = counts(data);
    let mut low = 0;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = single_base() + middle * SINGLE_SIZE;
        match u32_at(data, base).cmp(&code_point) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => {
                let offset = u32_at(data, base + 4) as usize;
                let length = u16_at(data, base + 8) as usize;
                return (0..length)
                    .map(|index| u64_at(data, weight_base(data) + (offset + index) * 8))
                    .map(|packed| (packed >> 32) & 0xffff)
                    .find(|primary| *primary != 0)
                    .map(|primary| primary << 32);
            }
        }
    }
    None
}

fn contraction_source(data: &[u8], entry: usize, index: usize) -> u32 {
    let base = contraction_base(data) + entry * CONTRACTION_SIZE;
    let offset = u32_at(data, base + 4) as usize;
    u32_at(data, sequence_base(data) + (offset + index) * 4)
}

/// UTS #10 S2.1 longest matching includes discontiguous contractions across unblocked
/// non-starters. Return the consumed input span and the skipped marks that remain after the
/// contraction mapping.
fn contraction_match(
    data: &[u8],
    entry: usize,
    input: &[u32],
    at: usize,
) -> Option<(usize, Vec<u32>)> {
    let base = contraction_base(data) + entry * CONTRACTION_SIZE;
    let source_length = u16_at(data, base + 8) as usize;
    let mut position = at;
    let mut skipped = Vec::new();
    for source_index in 0..source_length {
        let wanted = contraction_source(data, entry, source_index);
        if source_index == 0 {
            if input.get(position) != Some(&wanted) {
                return None;
            }
            position += 1;
            continue;
        }
        let wanted_ccc = crate::unicode_norm_impl::ccc(wanted);
        loop {
            let candidate = *input.get(position)?;
            if candidate == wanted {
                position += 1;
                break;
            }
            let candidate_ccc = crate::unicode_norm_impl::ccc(candidate);
            if wanted_ccc == 0 || candidate_ccc == 0 || candidate_ccc >= wanted_ccc {
                return None;
            }
            skipped.push(candidate);
            position += 1;
        }
    }
    Some((position - at, skipped))
}

fn append_contraction(
    data: &[u8],
    input: &[u32],
    at: usize,
    output: &mut Vec<Element>,
) -> Option<usize> {
    let first = *input.get(at)?;
    let (_, count, _, _) = counts(data);
    let mut low = 0;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let code_point = u32_at(data, contraction_base(data) + middle * CONTRACTION_SIZE);
        if code_point < first {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let mut best: Option<(usize, usize, usize, Vec<u32>)> = None;
    let mut entry = low;
    while entry < count {
        let base = contraction_base(data) + entry * CONTRACTION_SIZE;
        if u32_at(data, base) != first {
            break;
        }
        let source_length = u16_at(data, base + 8) as usize;
        if best
            .as_ref()
            .is_none_or(|(_, best_length, _, _)| source_length > *best_length)
        {
            if let Some((span, skipped)) = contraction_match(data, entry, input, at) {
                best = Some((entry, source_length, span, skipped));
            }
        }
        entry += 1;
    }
    let (entry, _, span, skipped) = best?;
    let base = contraction_base(data) + entry * CONTRACTION_SIZE;
    append_weights(
        data,
        u32_at(data, base + 12) as usize,
        u16_at(data, base + 16) as usize,
        output,
    );
    for code_point in skipped {
        if !append_single(data, code_point, output) {
            implicit(code_point, output);
        }
    }
    Some(span)
}

fn is_core_han(code_point: u32) -> bool {
    // UTS #10, Table 16: Unified_Ideograph=True intersected with the CJK Unified
    // Ideographs and CJK Compatibility Ideographs blocks.  The sparse compatibility
    // set is intentional; most code points in that block are not Unified_Ideograph.
    matches!(
        code_point,
        0x4e00..=0x9fff
            | 0xfa0e..=0xfa0f
            | 0xfa11
            | 0xfa13..=0xfa14
            | 0xfa1f
            | 0xfa21
            | 0xfa23..=0xfa24
            | 0xfa27..=0xfa29
    )
}

fn is_other_han(code_point: u32) -> bool {
    // Unicode 17 PropList.txt, Unified_Ideograph=True, excluding the core ranges above.
    matches!(
        code_point,
        0x3400..=0x4dbf
            | 0x20000..=0x2a6df
            | 0x2a700..=0x2b81d
            | 0x2b820..=0x2cead
            | 0x2ceb0..=0x2ebe0
            | 0x2ebf0..=0x2ee5d
            | 0x30000..=0x3134a
            | 0x31350..=0x33479
    )
}

fn implicit(code_point: u32, output: &mut Vec<Element>) {
    let special = [
        (0x17000, 0x187ff, 0xfb00, 0x17000),
        (0x18800, 0x18aff, 0xfb01, 0x18800),
        (0x18d00, 0x18d7f, 0xfb00, 0x17000),
        (0x18d80, 0x18dff, 0xfb01, 0x18800),
        (0x1b170, 0x1b2ff, 0xfb02, 0x1b170),
        (0x18b00, 0x18cff, 0xfb03, 0x18b00),
    ];
    let (first, second) = special
        .iter()
        .find(|(start, end, _, _)| (*start..=*end).contains(&code_point))
        .map(|(_, _, primary, origin)| (*primary, (code_point - origin) | 0x8000))
        .unwrap_or_else(|| {
            let primary = if is_core_han(code_point) {
                0xfb40 + (code_point >> 15)
            } else if is_other_han(code_point) {
                0xfb80 + (code_point >> 15)
            } else {
                0xfbc0 + (code_point >> 15)
            };
            (primary, (code_point & 0x7fff) | 0x8000)
        });
    output.push(Element {
        primary: u64::from(first) << 32,
        secondary: 0x20,
        tertiary: 2,
        variable: false,
    });
    output.push(Element {
        primary: u64::from(second) << 32,
        secondary: 0,
        tertiary: 0,
        variable: false,
    });
}

/// Append the longest DUCET mapping at `at` and return the number of consumed NFD code points.
fn append_from(data: &[u8], input: &[u32], at: usize, output: &mut Vec<Element>) -> usize {
    if let Some(length) = append_contraction(data, input, at, output) {
        return length;
    }
    let code_point = input[at];
    if !append_single(data, code_point, output) {
        implicit(code_point, output);
    }
    1
}

/// Append the longest CLDR-root mapping at `at` for locale-aware ECMA-402 comparison.
pub(crate) fn append_mapping(input: &[u32], at: usize, output: &mut Vec<Element>) -> usize {
    append_from(CLDR_ROOT, input, at, output)
}

/// [`append_mapping`]'s elements for an ASCII code point followed by the end of the input or
/// by another ASCII code point, when no such context forms a contraction with it. The table is
/// derived from [`append_mapping`] itself, checking every ASCII successor, so it can only skip
/// the table searches, never change a result.
pub(crate) fn ascii_mapping(code_point: u32) -> Option<&'static [Element]> {
    static TABLE: std::sync::OnceLock<Vec<Option<Vec<Element>>>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            (0..0x80)
                .map(|first| {
                    let mut alone = Vec::new();
                    if append_mapping(&[first], 0, &mut alone) != 1 {
                        return None;
                    }
                    let mut pair = Vec::new();
                    (0..0x80)
                        .all(|next| {
                            pair.clear();
                            append_mapping(&[first, next], 0, &mut pair) == 1 && pair == alone
                        })
                        .then_some(alone)
                })
                .collect()
        })
        .get(code_point as usize)?
        .as_deref()
}

#[cfg(test)]
pub(crate) fn append_ducet_mapping(input: &[u32], at: usize, output: &mut Vec<Element>) -> usize {
    append_from(DUCET, input, at, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_decodes_singletons_expansions_and_contractions() {
        let mut output = Vec::new();
        assert_eq!(append_ducet_mapping(&['a' as u32], 0, &mut output), 1);
        assert_eq!(output[0].primary, 0x23ec_0000_0000);

        output.clear();
        assert_eq!(append_ducet_mapping(&[0x00df], 0, &mut output), 1);
        assert!(output.len() >= 2);

        output.clear();
        assert_eq!(append_ducet_mapping(&[0x006c, 0x00b7], 0, &mut output), 2);
        assert_eq!(output[0].primary, 0x2528_0000_0000);
    }
}
