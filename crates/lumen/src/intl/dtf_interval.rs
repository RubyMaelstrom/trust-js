//! Date-time interval patterns (UTS #35 Part 4 intervalFormats) for `formatRange`: the range
//! patterns ECMA-402 PartitionDateTimeRangePattern selects by the greatest differing field.
//!
//! For a formatter's pattern this follows the UTS #35 procedure as ICU's DateIntervalFormat
//! implements it: the pattern's skeleton is split into date and time skeletons; the closest
//! intervalFormatItem skeleton is chosen (fields must agree; widths may differ) and its patterns'
//! field widths are adjusted to the request; a missing greatest difference is looked up in the
//! skeleton extended by that field (year differences of "MMMd" use "yMMMd"); a date with a time
//! combines the date pattern with the time's interval pattern through the dateTimeFormat, and
//! falls back to two full date-time patterns when the date differs; a time alone shows the date
//! (yMd) when the days differ. Fields without a pattern format both dates with the fallback
//! (intervalFormatFallback) pattern.

use super::dtf_pattern;
use crate::cldr_datetime_patterns as data;

/// A calendar field, in ECMA-402's range pattern field order (Table: Range pattern fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Field {
    Era,
    Year,
    Month,
    Day,
    AmPm,
    Hour,
    Minute,
    Second,
    Millisecond,
}

impl Field {
    const ALL: [Field; 9] = [
        Field::Era,
        Field::Year,
        Field::Month,
        Field::Day,
        Field::AmPm,
        Field::Hour,
        Field::Minute,
        Field::Second,
        Field::Millisecond,
    ];

    /// The pattern letter naming the field in skeletons and greatestDifference ids.
    fn letter(self) -> char {
        match self {
            Field::Era => 'G',
            Field::Year => 'y',
            Field::Month => 'M',
            Field::Day => 'd',
            Field::AmPm => 'a',
            Field::Hour => 'h',
            Field::Minute => 'm',
            Field::Second => 's',
            Field::Millisecond => 'S',
        }
    }

    /// The calendar field level: the larger, the smaller the unit.
    fn level(self) -> i32 {
        match self {
            Field::Era => 0,
            Field::Year => 10,
            Field::Month => 20,
            Field::Day => 30,
            Field::AmPm => 40,
            Field::Hour => 50,
            Field::Minute => 60,
            Field::Second => 70,
            Field::Millisecond => 80,
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The calendar field level of a pattern letter; -1 for letters that are no calendar unit.
fn letter_level(letter: char) -> i32 {
    match letter {
        'G' | 'O' | 'V' | 'X' | 'Z' | 'g' | 'l' | 'v' | 'x' | 'z' => 0,
        'U' | 'Y' | 'r' | 'u' | 'y' => 10,
        'D' | 'L' | 'M' | 'Q' | 'q' | 'w' => 20,
        'E' | 'F' | 'W' | 'c' | 'd' | 'e' => 30,
        'A' | 'a' => 40,
        'H' | 'K' | 'h' | 'k' => 50,
        'm' => 60,
        's' => 70,
        'S' => 80,
        _ => -1,
    }
}

/// Whether a field is finer than every field of a pattern or skeleton, so a difference in it
/// alone does not show.
pub(super) fn field_unit_ignored(pattern: &str, field: Field) -> bool {
    dtf_pattern::tokenize(pattern)
        .into_iter()
        .all(|token| match token {
            dtf_pattern::Token::Field(letter, _) => field.level() > letter_level(letter),
            dtf_pattern::Token::Literal(_) => true,
        })
}

/// The pattern for one greatest difference.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum RangePattern {
    /// An interval pattern split at its first repeated field. The first part formats the
    /// earlier date unless `later_first`.
    Split {
        first: String,
        second: String,
        later_first: bool,
    },
    /// No interval pattern: both dates formatted with this full pattern, joined by the
    /// intervalFormatFallback.
    Fallback(String),
}

/// A formatter's range patterns.
#[derive(Debug)]
pub(super) struct RangePatterns {
    patterns: [Option<RangePattern>; 9],
    /// The date and time halves of a date-time pattern (ICU's fallback "date + time range").
    date_pattern: Option<String>,
    time_pattern: Option<String>,
    /// The dateTimeFormat combining a date with a time (or a time range).
    pub date_time: String,
    /// The single-date pattern of the skeleton, which a range without an interval pattern
    /// shows twice.
    pub full: String,
    /// The intervalFormatFallback, with `{0}` the start and `{1}` the end.
    pub fallback: String,
}

impl RangePatterns {
    pub(super) fn get(&self, field: Field) -> Option<&RangePattern> {
        self.patterns[field.index()].as_ref()
    }

    /// The date and time patterns when a difference within one day is formatted as the date
    /// with a time range (ICU fallbackFormat's date-plus-time-range form).
    pub(super) fn date_and_time(&self) -> Option<(&str, &str)> {
        Some((self.date_pattern.as_deref()?, self.time_pattern.as_deref()?))
    }
}

/// Per-field widths of a skeleton or pattern (ICU DateIntervalInfo::parseSkeleton).
fn widths(skeleton: &str) -> [u8; 58] {
    let mut widths = [0u8; 58];
    for letter in skeleton.chars() {
        if letter.is_ascii_alphabetic() {
            let slot = &mut widths[letter as usize - 'A' as usize];
            *slot = slot.saturating_add(1);
        }
    }
    widths
}

/// How a best interval skeleton differs from the request: 0 exactly, 1 in field widths, 2 in
/// alternate letters (z/v, K/h, k/H, a, b), -1 in fields.
type Difference = i8;

struct Builder<'a> {
    locale: &'a str,
    calendar: &'a str,
    decimal: &'a str,
    hour_letter: char,
    /// The calendar's intervalFormats rows (skeleton, greatest difference, pattern), sorted.
    rows: &'static [(&'static str, &'static str, &'static str)],
    patterns: [Option<RangePattern>; 9],
    fallback_later_first: bool,
}

impl Builder<'_> {
    /// ICU DateTimePatternGenerator::getBestPattern for a skeleton, in the formatter's hour cycle.
    fn best_pattern(&self, skeleton: &str) -> String {
        let pattern =
            dtf_pattern::best_pattern(self.locale, self.calendar, skeleton, false, self.decimal);
        dtf_pattern::with_hour_letter(&pattern, self.hour_letter)
    }

    /// The intervalFormatItem skeletons.
    fn skeletons(&self) -> impl Iterator<Item = &'static str> + '_ {
        let mut previous = "";
        self.rows.iter().filter_map(move |row| {
            if row.0 == previous {
                return None;
            }
            previous = row.0;
            Some(row.0)
        })
    }

    /// The closest intervalFormatItem skeleton (ICU DateIntervalInfo::getBestSkeleton): fields
    /// must match; widths add their difference, a numeric month against a named one 0x100.
    fn best_skeleton(&self, skeleton: &str) -> Option<(&'static str, Difference)> {
        const DIFFERENT_FIELD: i32 = 0x1000;
        const STRING_NUMERIC_DIFFERENCE: i32 = 0x100;
        // The data has no skeletons with z, K, k, a or b: match v, h and H instead.
        let alternate = skeleton.contains(['z', 'k', 'K', 'a', 'b']);
        let input: String = skeleton
            .chars()
            .filter_map(|letter| match letter {
                'z' => Some('v'),
                'k' => Some('H'),
                'K' => Some('h'),
                'a' | 'b' => None,
                other => Some(other),
            })
            .collect();
        let input = widths(&input);
        let mut best: Option<(&'static str, i32, Difference)> = None;
        for candidate in self.skeletons() {
            let candidate_widths = widths(candidate);
            let mut distance = 0;
            let mut difference: Difference = 1;
            for (index, (&requested, &available)) in input.iter().zip(&candidate_widths).enumerate()
            {
                if requested == available {
                    continue;
                }
                if requested == 0 || available == 0 {
                    difference = -1;
                    distance += DIFFERENT_FIELD;
                } else if index == ('M' as usize - 'A' as usize)
                    && ((requested <= 2) != (available <= 2))
                {
                    distance += STRING_NUMERIC_DIFFERENCE;
                } else {
                    distance += (i32::from(requested) - i32::from(available)).abs();
                }
            }
            if best.is_none_or(|(_, best_distance, _)| distance < best_distance) {
                best = Some((candidate, distance, difference));
            }
            if distance == 0 {
                best = Some((candidate, 0, 0));
                break;
            }
        }
        best.map(|(candidate, _, difference)| {
            (
                candidate,
                if alternate && difference != -1 {
                    2
                } else {
                    difference
                },
            )
        })
    }

    /// The pattern of an intervalFormatItem for a greatest difference (`h` and `H` both give
    /// the hour; `a` and `B` the day period).
    fn interval(&self, skeleton: &str, field: Field) -> Option<&'static str> {
        let ids: &[&str] = match field {
            Field::Era => &["G"],
            Field::Year => &["y"],
            Field::Month => &["M"],
            Field::Day => &["d"],
            Field::AmPm => &["a", "B"],
            Field::Hour => &["h", "H"],
            Field::Minute => &["m"],
            Field::Second | Field::Millisecond => &[],
        };
        ids.iter()
            .find_map(|id| data::interval(self.locale, self.calendar, skeleton, id))
    }

    /// Store an interval pattern: an `earliestFirst:`/`latestFirst:` prefix sets the order, and
    /// the pattern splits at its first repeated field.
    fn set_interval(&mut self, field: Field, pattern: &str, later_first: bool) {
        let (pattern, later_first) = if let Some(rest) = pattern.strip_prefix("latestFirst:") {
            (rest, true)
        } else if let Some(rest) = pattern.strip_prefix("earliestFirst:") {
            (rest, false)
        } else {
            (pattern, later_first)
        };
        let pattern = dtf_pattern::with_hour_letter(pattern, self.hour_letter);
        let split = split_point(&pattern);
        self.patterns[field.index()] = Some(RangePattern::Split {
            first: pattern[..split].to_string(),
            second: pattern[split..].to_string(),
            later_first,
        });
    }

    /// ICU DateIntervalFormat::setIntervalPattern: the pattern for `field` from the best
    /// skeleton, widened to the request; when it has none (and the field shows), from the
    /// skeleton extended by the field. Returns whether the extension supplied it.
    fn set_from_best(
        &mut self,
        field: Field,
        skeleton: &str,
        best: &str,
        difference: Difference,
        extended: Option<&mut (String, String)>,
    ) -> bool {
        let mut best = best.to_string();
        let mut difference = difference;
        let mut pattern = self.interval(&best, field);
        let mut extended_used = false;
        if pattern.is_none() {
            if field_unit_ignored(&best, field) {
                return false;
            }
            if field == Field::AmPm {
                // A 24-hour skeleton's data may lack the day-period difference: the hour one.
                if let Some(hour) = self.interval(&best, Field::Hour) {
                    let adjusted = adjust_widths(skeleton, &best, hour, difference);
                    self.set_interval(field, &adjusted, self.fallback_later_first);
                }
                return false;
            }
            if let Some(extension) = extended {
                extension.0 = format!("{}{skeleton}", field.letter());
                extension.1 = format!("{}{best}", field.letter());
                extended_used = true;
                pattern = self.interval(&extension.1, field);
                if pattern.is_none() && difference == 0 {
                    if let Some((nearest, nearest_difference)) = self.best_skeleton(&extension.1) {
                        if nearest_difference != -1 {
                            pattern = self.interval(nearest, field);
                            best = nearest.to_string();
                            difference = nearest_difference;
                        }
                    }
                }
            }
        }
        let Some(pattern) = pattern else {
            return false;
        };
        let adjusted = if difference != 0 {
            adjust_widths(skeleton, &best, pattern, difference)
        } else {
            pattern.to_string()
        };
        self.set_interval(field, &adjusted, self.fallback_later_first);
        extended_used
    }
}

/// Where an interval pattern splits: before the first field letter that repeats an earlier one
/// (UTS #35: "broken up into two pieces by determining the first repeating field"); a stand-alone
/// month repeats a format month.
fn split_point(pattern: &str) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut chars = pattern.char_indices().peekable();
    let mut quoted = false;
    let mut previous: Option<char> = None;
    while let Some((index, character)) = chars.next() {
        if character == '\'' {
            if chars.peek().is_some_and(|&(_, next)| next == '\'') {
                chars.next();
            } else {
                quoted = !quoted;
            }
            previous = None;
            continue;
        }
        if quoted || !character.is_ascii_alphabetic() {
            previous = None;
            continue;
        }
        if previous == Some(character) {
            continue;
        }
        previous = Some(character);
        let key = if character == 'L' { 'M' } else { character };
        if !seen.insert(key) {
            return index;
        }
    }
    pattern.len()
}

/// ICU DateIntervalFormat::adjustFieldWidth: lengthen each field of an interval pattern whose
/// width equals the best skeleton's to the requested width, and restore the request's alternate
/// letters (z for v, K for h, k for H, b for a).
fn adjust_widths(skeleton: &str, best: &str, pattern: &str, difference: Difference) -> String {
    let requested = widths(skeleton);
    let mut available = widths(best);
    let mut pattern = pattern.to_string();
    if difference == 2 {
        let replace = |pattern: &str, from: char, to: char| -> String {
            dtf_pattern::replace_letter(pattern, from, to)
        };
        if skeleton.contains('z') {
            pattern = replace(&pattern, 'v', 'z');
        }
        if skeleton.contains('K') {
            pattern = replace(&pattern, 'h', 'K');
        }
        if skeleton.contains('k') {
            pattern = replace(&pattern, 'H', 'k');
        }
        if skeleton.contains('b') {
            pattern = replace(&pattern, 'a', 'b');
        }
    }
    let tokens = dtf_pattern::tokenize(&pattern);
    for letter in ['a', 'b'] {
        let slot = letter as usize - 'A' as usize;
        if available[slot] == 0
            && tokens.iter().any(
                |token| matches!(token, dtf_pattern::Token::Field(found, _) if *found == letter),
            )
        {
            available[slot] = 1;
        }
    }
    let tokens: Vec<dtf_pattern::Token> = tokens
        .into_iter()
        .map(|token| match token {
            dtf_pattern::Token::Field(letter, len) => {
                let key = if letter == 'L' { 'M' } else { letter };
                let slot = key as usize - 'A' as usize;
                let (want, have) = (usize::from(requested[slot]), usize::from(available[slot]));
                if have == len && want > have {
                    dtf_pattern::Token::Field(letter, want)
                } else {
                    dtf_pattern::Token::Field(letter, len)
                }
            }
            literal => literal,
        })
        .collect();
    dtf_pattern::untokenize(&tokens)
}

/// ICU DateIntervalFormat::getDateTimeSkeleton: the date and time halves of a skeleton and their
/// normalized forms (one y-run, M or MMM+, E or EEEE+, one d; the hour letter, m, z, v, and the
/// other time letters except a).
fn split_skeleton(skeleton: &str) -> (String, String, String, String) {
    let (mut date, mut time) = (String::new(), String::new());
    let (mut normalized_date, mut normalized_time) = (String::new(), String::new());
    let (mut e, mut d, mut month, mut y, mut m, mut v, mut z) = (0, 0, 0, 0, 0, 0, 0);
    let mut hour: Option<char> = None;
    for letter in skeleton.chars() {
        match letter {
            'E' => {
                date.push(letter);
                e += 1;
            }
            'd' => {
                date.push(letter);
                d += 1;
            }
            'M' => {
                date.push(letter);
                month += 1;
            }
            'y' => {
                date.push(letter);
                y += 1;
            }
            'G' | 'Y' | 'u' | 'Q' | 'q' | 'L' | 'l' | 'W' | 'w' | 'D' | 'F' | 'g' | 'e' | 'c'
            | 'U' | 'r' => {
                normalized_date.push(letter);
                date.push(letter);
            }
            'h' | 'H' | 'k' | 'K' => {
                time.push(letter);
                hour.get_or_insert(letter);
            }
            'm' => {
                time.push(letter);
                m += 1;
            }
            'z' => {
                time.push(letter);
                z += 1;
            }
            'v' => {
                time.push(letter);
                v += 1;
            }
            'a' => time.push(letter),
            'V' | 'Z' | 'j' | 's' | 'S' | 'A' | 'b' | 'B' => {
                time.push(letter);
                normalized_time.push(letter);
            }
            _ => {}
        }
    }
    normalized_date.extend(std::iter::repeat_n('y', y));
    if month > 0 {
        normalized_date.extend(std::iter::repeat_n(
            'M',
            if month < 3 { 1 } else { month.min(5) },
        ));
    }
    if e > 0 {
        normalized_date.extend(std::iter::repeat_n('E', if e <= 3 { 1 } else { e.min(6) }));
    }
    if d > 0 {
        normalized_date.push('d');
    }
    if let Some(hour) = hour {
        normalized_time.push(hour);
    }
    if m > 0 {
        normalized_time.push('m');
    }
    if z > 0 {
        normalized_time.push('z');
    }
    if v > 0 {
        normalized_time.push('v');
    }
    (date, normalized_date, time, normalized_time)
}

/// The range patterns of a formatter whose single-date pattern is `pattern` (ICU
/// DateIntervalFormat::initializePattern over the CLDR intervalFormats of `calendar`).
pub(super) fn range_patterns(
    locale: &str,
    calendar: &str,
    pattern: &str,
    decimal: &str,
) -> RangePatterns {
    let calendar_data = data::calendar_patterns(locale, calendar);
    let fallback = calendar_data.fallback.to_string();
    let hour_letter = dtf_pattern::tokenize(pattern)
        .into_iter()
        .find_map(|token| match token {
            dtf_pattern::Token::Field(letter @ ('h' | 'H' | 'k' | 'K'), _) => Some(letter),
            _ => None,
        })
        .unwrap_or('h');
    let mut builder = Builder {
        locale,
        calendar,
        decimal,
        hour_letter,
        rows: calendar_data.intervals,
        patterns: Default::default(),
        // "{1} – {0}" puts the later date first.
        fallback_later_first: fallback.find("{1}") < fallback.find("{0}"),
    };
    let skeleton = dtf_pattern::pattern_skeleton(pattern);
    let (date, normalized_date, time, normalized_time) = split_skeleton(&skeleton);
    let date_time = data::date_time_glue(locale).to_string();
    let mut date_pattern = None;
    let mut time_pattern = None;

    // ICU setSeparateDateTimePtn: a time skeleton finds the time interval patterns, otherwise
    // the date skeleton finds the date ones.
    let separate = if time.is_empty() {
        &normalized_date
    } else {
        &normalized_time
    };
    let mut found = false;
    if let Some((best, difference)) = builder.best_skeleton(separate) {
        if !date.is_empty() {
            date_pattern = Some(builder.best_pattern(&date));
        }
        if !time.is_empty() {
            time_pattern = Some(builder.best_pattern(&time));
        }
        if difference != -1 {
            found = true;
            if time.is_empty() {
                let mut extension = (String::new(), String::new());
                builder.set_from_best(Field::Day, separate, best, difference, Some(&mut extension));
                let mut skeleton = separate.clone();
                let mut best = best.to_string();
                if builder.set_from_best(
                    Field::Month,
                    &skeleton,
                    &best,
                    difference,
                    Some(&mut extension),
                ) {
                    skeleton = extension.0.clone();
                    best = extension.1.clone();
                }
                builder.set_from_best(
                    Field::Year,
                    &skeleton,
                    &best,
                    difference,
                    Some(&mut extension),
                );
                builder.set_from_best(
                    Field::Era,
                    &skeleton,
                    &best,
                    difference,
                    Some(&mut extension),
                );
            } else {
                builder.set_from_best(Field::Minute, separate, best, difference, None);
                builder.set_from_best(Field::Hour, separate, best, difference, None);
                builder.set_from_best(Field::AmPm, separate, best, difference, None);
            }
        }
    }

    let time_with_date = |builder: &mut Builder| {
        // A time alone shows the date (the short date skeleton yMd) when the days differ.
        let with_date = builder.best_pattern(&format!("yMd{time}"));
        for field in [Field::Day, Field::Month, Field::Year] {
            builder.patterns[field.index()] = Some(RangePattern::Fallback(with_date.clone()));
        }
        let with_era = builder.best_pattern(&format!("GyMd{time}"));
        builder.patterns[Field::Era.index()] = Some(RangePattern::Fallback(with_era));
    };
    if !found {
        if !time.is_empty() && date.is_empty() {
            time_with_date(&mut builder);
        }
    } else if !time.is_empty() {
        if date.is_empty() {
            time_with_date(&mut builder);
        } else {
            // A differing date shows both full date-times; the skeleton gains the date fields it
            // lacks for the larger differences.
            let mut extended = skeleton.clone();
            for field in [Field::Day, Field::Month, Field::Year, Field::Era] {
                if !date.contains(field.letter()) {
                    extended.insert(0, field.letter());
                    let pattern = builder.best_pattern(&extended);
                    builder.patterns[field.index()] = Some(RangePattern::Fallback(pattern));
                }
            }
            // A difference within the day shows the date once with the time's interval pattern.
            let date_only = builder.best_pattern(&date);
            for field in [Field::AmPm, Field::Hour, Field::Minute] {
                if let Some(RangePattern::Split {
                    first,
                    second,
                    later_first,
                }) = builder.patterns[field.index()].clone()
                {
                    let combined = date_time
                        .replace("{0}", &format!("{first}{second}"))
                        .replace("{1}", &date_only);
                    builder.set_interval(field, &combined, later_first);
                }
            }
        }
    }
    RangePatterns {
        full: builder.best_pattern(&skeleton),
        patterns: builder.patterns,
        date_pattern,
        time_pattern,
        date_time,
        fallback,
    }
}

/// The greatest field in which two dates differ, in [`Field`] order.
pub(super) fn greatest_difference(start: &[i64; 9], end: &[i64; 9]) -> Option<Field> {
    Field::ALL
        .into_iter()
        .find(|field| start[field.index()] != end[field.index()])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(pattern: &str) -> (&str, &str) {
        let point = split_point(pattern);
        (&pattern[..point], &pattern[point..])
    }

    #[test]
    fn interval_patterns_split_at_the_first_repeated_field() {
        assert_eq!(split("MMM d – d, y"), ("MMM d – ", "d, y"));
        assert_eq!(split("MMM d – MMM d, y"), ("MMM d – ", "MMM d, y"));
        assert_eq!(split("LLL – MMM y"), ("LLL – ", "MMM y"));
        assert_eq!(split("h:mm – h:mm a"), ("h:mm – ", "h:mm a"));
        assert_eq!(split("d.–d. MMMM"), ("d.–", "d. MMMM"));
        assert_eq!(split("'d' d"), ("'d' d", ""));
    }

    #[test]
    fn skeletons_split_into_normalized_date_and_time() {
        assert_eq!(
            split_skeleton("yyMMdd"),
            ("yyMMdd".into(), "yyMd".into(), String::new(), String::new())
        );
        assert_eq!(
            split_skeleton("yMMMMEEEEd"),
            (
                "yMMMMEEEEd".into(),
                "yMMMMEEEEd".into(),
                String::new(),
                String::new()
            )
        );
        assert_eq!(
            split_skeleton("yMMMdahmmz"),
            ("yMMMd".into(), "yMMMd".into(), "ahmmz".into(), "hmz".into())
        );
        assert_eq!(split_skeleton("Hmmss").3, "ssHm");
    }

    #[test]
    fn widths_follow_the_request() {
        assert_eq!(
            adjust_widths("yMMMMd", "yMMMd", "MMM d – d, y", 1),
            "MMMM d – d, y"
        );
        assert_eq!(
            adjust_widths("yMMMMEEEEd", "yMMMEd", "E, MMM d – E, MMM d, y", 1),
            "EEEE, MMMM d – EEEE, MMMM d, y"
        );
        assert_eq!(
            adjust_widths("yyMd", "yMd", "M/d/y – M/d/y", 1),
            "M/d/yy – M/d/yy"
        );
        assert_eq!(
            adjust_widths("hmz", "hmv", "h:mm – h:mm a v", 2),
            "h:mm – h:mm a z"
        );
        assert_eq!(
            adjust_widths("Km", "hm", "h:mm – h:mm a", 2),
            "K:mm – K:mm a"
        );
    }

    #[test]
    fn ignored_fields_are_finer_than_the_pattern() {
        assert!(field_unit_ignored("MMM d", Field::Hour));
        assert!(!field_unit_ignored("MMM d", Field::Year));
        assert!(field_unit_ignored("y", Field::Month));
        assert!(!field_unit_ignored("h:mm a", Field::AmPm));
        assert!(field_unit_ignored("h:mm a", Field::Second));
    }

    #[test]
    fn range_patterns_cover_date_time_and_extension() {
        let en = |pattern: &str, field| {
            range_patterns("en", "gregory", pattern, ".")
                .get(field)
                .cloned()
        };
        assert_eq!(
            en("MMMM d", Field::Day),
            Some(RangePattern::Split {
                first: "MMMM d\u{2009}–\u{2009}".into(),
                second: "d".into(),
                later_first: false
            })
        );
        // A year difference of "MMM d" uses yMMMd's pattern.
        assert_eq!(
            en("MMM d", Field::Year),
            Some(RangePattern::Split {
                first: "MMM d, y\u{2009}–\u{2009}".into(),
                second: "MMM d, y".into(),
                later_first: false
            })
        );
        // The date once, with the time's interval pattern.
        assert_eq!(
            en("MMM d, y, h:mm\u{202f}a", Field::Hour),
            Some(RangePattern::Split {
                first: "MMM d, y, h:mm\u{2009}–\u{2009}".into(),
                second: "h:mm\u{202f}a".into(),
                later_first: false
            })
        );
        assert_eq!(en("MMM d, y, h:mm\u{202f}a", Field::Day), None);
        // A time alone gains the date when the day differs.
        assert_eq!(
            en("h:mm\u{202f}a", Field::Day),
            Some(RangePattern::Fallback("M/d/y, h:mm\u{202f}a".into()))
        );
    }
}
