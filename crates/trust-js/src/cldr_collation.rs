//! Compact CLDR 48 Japanese/Chinese starred-primary tailoring maps.

const DATA: &[u8] = include_bytes!("cldr_collation.bin");
const HEADER_SIZE: usize = 24;
const ROW_SIZE: usize = 8;

#[derive(Clone, Copy)]
pub(crate) enum HanOrder {
    Japanese,
    Pinyin,
    Stroke,
    Zhuyin,
}

fn u32_at(offset: usize) -> u32 {
    u32::from_le_bytes(DATA[offset..offset + 4].try_into().unwrap())
}

fn table(order: HanOrder) -> (usize, usize) {
    debug_assert_eq!(&DATA[..8], b"LCLD48\0\x01");
    let index = order as usize;
    let count = u32_at(8 + index * 4) as usize;
    let preceding = (0..index)
        .map(|previous| u32_at(8 + previous * 4) as usize)
        .sum::<usize>();
    (HEADER_SIZE + preceding * ROW_SIZE, count)
}

pub(crate) fn primary_rank(order: HanOrder, code_point: u32) -> Option<u32> {
    let (base, count) = table(order);
    let mut low = 0;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let row = base + middle * ROW_SIZE;
        match u32_at(row).cmp(&code_point) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Some(u32_at(row + 4)),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_han_orders_are_distinct_and_searchable() {
        let east = '東' as u32;
        let capital = '京' as u32;
        assert!(primary_rank(HanOrder::Japanese, east).is_some());
        assert_ne!(
            primary_rank(HanOrder::Pinyin, east),
            primary_rank(HanOrder::Stroke, east)
        );
        assert_ne!(
            primary_rank(HanOrder::Pinyin, capital),
            primary_rank(HanOrder::Zhuyin, capital)
        );
    }
}
