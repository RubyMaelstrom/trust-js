//! UTS #35 Part 4 skeleton matching for `Intl.DateTimeFormat`'s BestFitFormatMatcher
//! (ECMA-402 #sec-bestfitformatmatcher), following the algorithm of
//! [Matching Skeletons](https://unicode.org/reports/tr35/tr35-dates.html#Matching_Skeletons):
//! find the closest availableFormats (or date/time style, or single-field) pattern, adjust its
//! field widths to the request, append any missing fields with the locale's appendItems, and
//! join a date half and a time half with the dateTimeFormat chosen by the requested month width.
//!
//! The distance metric and width-adjustment rules reproduce ICU's DateTimePatternGenerator, which
//! other ECMA-402 implementations use, so locale data selects the same patterns everywhere.

use crate::cldr_datetime_patterns as data;

pub(super) const ERA: usize = 0;
pub(super) const YEAR: usize = 1;
const QUARTER: usize = 2;
pub(super) const MONTH: usize = 3;
const WEEK_OF_YEAR: usize = 4;
const WEEK_OF_MONTH: usize = 5;
pub(super) const WEEKDAY: usize = 6;
const DAY_OF_YEAR: usize = 7;
const DAY_OF_WEEK_IN_MONTH: usize = 8;
pub(super) const DAY: usize = 9;
pub(super) const DAY_PERIOD: usize = 10;
pub(super) const HOUR: usize = 11;
pub(super) const MINUTE: usize = 12;
pub(super) const SECOND: usize = 13;
pub(super) const FRACTIONAL: usize = 14;
pub(super) const ZONE: usize = 15;
const FIELD_COUNT: usize = 16;
/// The date fields (era through day); the rest are time fields.
const DATE_MASK: u32 = (1 << DAY_PERIOD) - 1;
const TIME_MASK: u32 = ((1 << FIELD_COUNT) - 1) & !DATE_MASK;

const NARROW: i32 = -0x101;
const SHORTER: i32 = -0x102;
const SHORT: i32 = -0x103;
const LONG: i32 = -0x104;
const NUMERIC: i32 = 0x100;
const DELTA: i32 = 0x10;
const MISSING_FIELD: i32 = 0x1000;
const EXTRA_FIELD: i32 = 0x10000;

/// A pattern letter's field and type class. A field's type is `kind + length` for a numeric
/// row (`kind > 0`) and `kind` for a text row; the distance between two skeletons sums the type
/// differences, so numeric/text mismatches cost far more than width differences.
struct TypeRow {
    letter: char,
    field: usize,
    kind: i32,
    min_len: usize,
}

const fn row(letter: char, field: usize, kind: i32, min_len: usize) -> TypeRow {
    TypeRow {
        letter,
        field,
        kind,
        min_len,
    }
}

/// Rows grouped by letter in ascending `min_len`.
static TYPES: &[TypeRow] = &[
    row('G', ERA, SHORT, 1),
    row('G', ERA, LONG, 4),
    row('G', ERA, NARROW, 5),
    row('y', YEAR, NUMERIC, 1),
    row('Y', YEAR, NUMERIC + DELTA, 1),
    row('u', YEAR, NUMERIC + 2 * DELTA, 1),
    row('r', YEAR, NUMERIC + 3 * DELTA, 1),
    row('U', YEAR, SHORT, 1),
    row('U', YEAR, LONG, 4),
    row('U', YEAR, NARROW, 5),
    row('Q', QUARTER, NUMERIC, 1),
    row('Q', QUARTER, SHORT, 3),
    row('Q', QUARTER, LONG, 4),
    row('Q', QUARTER, NARROW, 5),
    row('q', QUARTER, NUMERIC + DELTA, 1),
    row('q', QUARTER, SHORT - DELTA, 3),
    row('q', QUARTER, LONG - DELTA, 4),
    row('q', QUARTER, NARROW - DELTA, 5),
    row('M', MONTH, NUMERIC, 1),
    row('M', MONTH, SHORT, 3),
    row('M', MONTH, LONG, 4),
    row('M', MONTH, NARROW, 5),
    row('L', MONTH, NUMERIC + DELTA, 1),
    row('L', MONTH, SHORT - DELTA, 3),
    row('L', MONTH, LONG - DELTA, 4),
    row('L', MONTH, NARROW - DELTA, 5),
    row('l', MONTH, NUMERIC + DELTA, 1),
    row('w', WEEK_OF_YEAR, NUMERIC, 1),
    row('W', WEEK_OF_MONTH, NUMERIC, 1),
    row('E', WEEKDAY, SHORT, 1),
    row('E', WEEKDAY, LONG, 4),
    row('E', WEEKDAY, NARROW, 5),
    row('E', WEEKDAY, SHORTER, 6),
    row('c', WEEKDAY, NUMERIC + 2 * DELTA, 1),
    row('c', WEEKDAY, SHORT - 2 * DELTA, 3),
    row('c', WEEKDAY, LONG - 2 * DELTA, 4),
    row('c', WEEKDAY, NARROW - 2 * DELTA, 5),
    row('c', WEEKDAY, SHORTER - 2 * DELTA, 6),
    row('e', WEEKDAY, NUMERIC + DELTA, 1),
    row('e', WEEKDAY, SHORT - DELTA, 3),
    row('e', WEEKDAY, LONG - DELTA, 4),
    row('e', WEEKDAY, NARROW - DELTA, 5),
    row('e', WEEKDAY, SHORTER - DELTA, 6),
    row('d', DAY, NUMERIC, 1),
    row('d', DAY, SHORT, 3),
    row('g', DAY, NUMERIC + DELTA, 1),
    row('D', DAY_OF_YEAR, NUMERIC, 1),
    row('F', DAY_OF_WEEK_IN_MONTH, NUMERIC, 1),
    row('a', DAY_PERIOD, SHORT, 1),
    row('a', DAY_PERIOD, LONG, 4),
    row('a', DAY_PERIOD, NARROW, 5),
    row('b', DAY_PERIOD, SHORT - DELTA, 1),
    row('b', DAY_PERIOD, LONG - DELTA, 4),
    row('b', DAY_PERIOD, NARROW - DELTA, 5),
    // b is closer to a than to B.
    row('B', DAY_PERIOD, SHORT - 3 * DELTA, 1),
    row('B', DAY_PERIOD, LONG - 3 * DELTA, 4),
    row('B', DAY_PERIOD, NARROW - 3 * DELTA, 5),
    row('H', HOUR, NUMERIC + 10 * DELTA, 1),
    row('k', HOUR, NUMERIC + 11 * DELTA, 1),
    row('h', HOUR, NUMERIC, 1),
    row('K', HOUR, NUMERIC + DELTA, 1),
    row('m', MINUTE, NUMERIC, 1),
    row('s', SECOND, NUMERIC, 1),
    row('A', SECOND, NUMERIC + DELTA, 1),
    row('S', FRACTIONAL, NUMERIC, 1),
    row('v', ZONE, SHORT - 2 * DELTA, 1),
    row('v', ZONE, LONG - 2 * DELTA, 4),
    row('z', ZONE, SHORT, 1),
    row('z', ZONE, LONG, 4),
    row('Z', ZONE, NARROW - DELTA, 1),
    row('Z', ZONE, LONG - DELTA, 4),
    row('Z', ZONE, SHORT - DELTA, 5),
    row('O', ZONE, SHORT - DELTA, 1),
    row('O', ZONE, LONG - DELTA, 4),
    row('V', ZONE, SHORT - DELTA, 1),
    row('V', ZONE, LONG - DELTA, 2),
    row('V', ZONE, LONG - 1 - DELTA, 3),
    row('V', ZONE, LONG - 2 - DELTA, 4),
    row('X', ZONE, NARROW - DELTA, 1),
    row('X', ZONE, SHORT - DELTA, 2),
    row('X', ZONE, LONG - DELTA, 4),
    row('x', ZONE, NARROW - DELTA, 1),
    row('x', ZONE, SHORT - DELTA, 2),
    row('x', ZONE, LONG - DELTA, 4),
];

/// The row for a run of `len` copies of `letter`: the last of its rows whose minimum length
/// the run reaches.
fn type_row(letter: char, len: usize) -> Option<&'static TypeRow> {
    let mut best = None;
    for row in TYPES.iter().filter(|row| row.letter == letter) {
        if best.is_none() || row.min_len <= len {
            best = Some(row);
        }
    }
    best
}

/// A pattern's fields and literals, in order. Quoted text and every non-letter is literal.
#[derive(Debug, PartialEq)]
pub(super) enum Token {
    Field(char, usize),
    Literal(String),
}

pub(super) fn tokenize(pattern: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut literal = String::new();
    let mut chars = pattern.chars().peekable();
    let mut quoted = false;
    while let Some(character) = chars.next() {
        if character == '\'' {
            if chars.peek() == Some(&'\'') {
                chars.next();
                literal.push('\'');
            } else {
                quoted = !quoted;
            }
            continue;
        }
        if !quoted && character.is_ascii_alphabetic() {
            let mut len = 1;
            while chars.peek() == Some(&character) {
                chars.next();
                len += 1;
            }
            if !literal.is_empty() {
                tokens.push(Token::Literal(std::mem::take(&mut literal)));
            }
            tokens.push(Token::Field(character, len));
        } else {
            literal.push(character);
        }
    }
    if !literal.is_empty() {
        tokens.push(Token::Literal(literal));
    }
    tokens
}

/// Write a token sequence back as a pattern, quoting literal letters and apostrophes.
pub(super) fn untokenize(tokens: &[Token]) -> String {
    let mut pattern = String::new();
    for token in tokens {
        match token {
            Token::Field(letter, len) => {
                pattern.extend(std::iter::repeat_n(*letter, *len));
            }
            Token::Literal(text) => {
                let mut quoted = false;
                for character in text.chars() {
                    if character == '\'' {
                        // Two apostrophes are a literal one, inside or outside quoted text.
                        pattern.push_str("''");
                    } else if character.is_ascii_alphabetic() {
                        if !quoted {
                            pattern.push('\'');
                            quoted = true;
                        }
                        pattern.push(character);
                    } else {
                        if quoted {
                            pattern.push('\'');
                            quoted = false;
                        }
                        pattern.push(character);
                    }
                }
                if quoted {
                    pattern.push('\'');
                }
            }
        }
    }
    pattern
}

/// A skeleton (or a pattern read as one): each field's letter and length as written, and its
/// type value (zero when absent).
#[derive(Clone)]
struct Skeleton {
    original: [Option<(char, usize)>; FIELD_COUNT],
    /// Each field's canonical letter at its type's minimum length (ICU's base pattern).
    base: [Option<(char, usize)>; FIELD_COUNT],
    types: [i32; FIELD_COUNT],
}

impl Skeleton {
    fn parse(text: &str) -> Skeleton {
        let mut skeleton = Skeleton {
            original: [None; FIELD_COUNT],
            base: [None; FIELD_COUNT],
            types: [0; FIELD_COUNT],
        };
        for token in tokenize(text) {
            let Token::Field(letter, len) = token else {
                continue;
            };
            let Some(row) = type_row(letter, len) else {
                continue;
            };
            skeleton.set(row, letter, len);
        }
        // Minutes with fractional seconds but no seconds use seconds.
        if skeleton.original[MINUTE].is_some()
            && skeleton.original[FRACTIONAL].is_some()
            && skeleton.original[SECOND].is_none()
        {
            skeleton.set(type_row('s', 1).expect("seconds row"), 's', 1);
        }
        // A 12-hour hour implies a day period ("a"); a 24-hour hour ignores one.
        match skeleton.original[HOUR] {
            Some(('h' | 'K', _)) if skeleton.original[DAY_PERIOD].is_none() => {
                skeleton.set(type_row('a', 1).expect("day period row"), 'a', 1);
            }
            Some(('H' | 'k', _)) => {
                skeleton.original[DAY_PERIOD] = None;
                skeleton.base[DAY_PERIOD] = None;
                skeleton.types[DAY_PERIOD] = 0;
            }
            _ => {}
        }
        skeleton
    }

    fn set(&mut self, row: &TypeRow, letter: char, len: usize) {
        self.original[row.field] = Some((letter, len));
        self.base[row.field] = Some((row.letter, row.min_len));
        self.types[row.field] = if row.kind > 0 {
            row.kind + len as i32
        } else {
            row.kind
        };
    }

    /// ICU's base pattern: the base fields in field order.
    fn base_pattern(&self) -> String {
        self.base
            .iter()
            .flatten()
            .flat_map(|&(letter, len)| std::iter::repeat_n(letter, len))
            .collect()
    }

    fn mask(&self) -> u32 {
        (0..FIELD_COUNT)
            .filter(|&field| self.types[field] != 0)
            .fold(0, |mask, field| mask | (1 << field))
    }
}

/// The distance from the requested fields in `mask` to a candidate, with the fields the
/// candidate lacks and the fields it adds.
fn distance(request: &Skeleton, candidate: &Skeleton, mask: u32) -> (i32, u32, u32) {
    let (mut result, mut missing, mut extra) = (0, 0, 0);
    for field in 0..FIELD_COUNT {
        let mine = if mask & (1 << field) == 0 {
            0
        } else {
            request.types[field]
        };
        let other = candidate.types[field];
        if mine == other {
            continue;
        }
        if mine == 0 {
            result += EXTRA_FIELD;
            extra |= 1 << field;
        } else if other == 0 {
            result += MISSING_FIELD;
            missing |= 1 << field;
        } else {
            result += (mine - other).abs();
        }
    }
    (result, missing, extra)
}

/// A selectable pattern: its skeleton, the pattern, and whether the skeleton was specified by
/// availableFormats (which governs width adjustment) rather than derived from a pattern.
struct Candidate {
    skeleton: Skeleton,
    pattern: String,
    specified: bool,
}

/// UTS #35 Part 4 single-field patterns that always exist for matching (ICU adds the same set).
const CANONICAL_ITEMS: &[char] = &[
    'G', 'y', 'Q', 'M', 'w', 'W', 'E', 'D', 'F', 'd', 'a', 'H', 'm', 's', 'S', 'v',
];

/// The pattern skeleton of a pattern: its field letters and lengths without literals.
pub(super) fn pattern_skeleton(pattern: &str) -> String {
    let mut skeleton = String::new();
    for token in tokenize(pattern) {
        if let Token::Field(letter, len) = token {
            skeleton.extend(std::iter::repeat_n(letter, len));
        }
    }
    skeleton
}

/// The locale's patterns in the order ICU's DateTimePatternGenerator holds them: the canonical
/// single fields, the time and date styles (without overriding), then the availableFormats,
/// which replace a derived entry of the same skeleton. Entries iterate grouped by their base
/// pattern's first letter (A–Z, then a–z), in insertion order within a group.
fn candidates(locale: &str, calendar: &str) -> Vec<Candidate> {
    struct Entry {
        base: String,
        candidate: Candidate,
    }
    let mut entries: Vec<Entry> = Vec::new();
    let add = |entries: &mut Vec<Entry>, pattern: String, key: Option<&str>| {
        let skeleton = Skeleton::parse(
            key.map_or_else(|| pattern_skeleton(&pattern), str::to_string)
                .as_str(),
        );
        let base = skeleton.base_pattern();
        let specified = key.is_some();
        let override_existing = specified;
        let base_conflict = entries
            .iter()
            .any(|entry| entry.base == base && !entry.candidate.specified);
        if base_conflict && !override_existing {
            return;
        }
        if let Some(index) = entries
            .iter()
            .position(|entry| entry.candidate.skeleton.original == skeleton.original)
        {
            if !override_existing || entries[index].candidate.specified {
                return;
            }
        }
        if let Some(entry) = entries
            .iter_mut()
            .find(|entry| entry.base == base && entry.candidate.skeleton.types == skeleton.types)
        {
            entry.candidate.pattern = pattern;
            entry.candidate.specified = specified;
            entry.candidate.skeleton = skeleton;
            return;
        }
        entries.push(Entry {
            base,
            candidate: Candidate {
                skeleton,
                pattern,
                specified,
            },
        });
    };
    for letter in CANONICAL_ITEMS {
        add(&mut entries, letter.to_string(), None);
    }
    let patterns = data::calendar_patterns(locale, calendar);
    for style in ["full", "long", "medium", "short"] {
        add(
            &mut entries,
            data::time_style(locale, style).to_string(),
            None,
        );
    }
    for (date, _) in patterns.dates {
        add(&mut entries, date.to_string(), None);
    }
    for (key, pattern, _) in data::available_formats(locale, calendar) {
        add(&mut entries, pattern.to_string(), Some(key));
    }
    let bucket = |base: &str| match base.chars().next() {
        Some(letter @ 'A'..='Z') => letter as u32 - 'A' as u32,
        Some(letter @ 'a'..='z') => 26 + letter as u32 - 'a' as u32,
        _ => 52,
    };
    entries.sort_by_key(|entry| bucket(&entry.base));
    entries.into_iter().map(|entry| entry.candidate).collect()
}

/// ICU's yearLengthIsCompatible: a two-digit-year pattern serves only a two-digit request.
fn year_length_compatible(request: &Skeleton, candidate: &Skeleton) -> bool {
    if request.types[YEAR] <= 0 || candidate.types[YEAR] <= 0 {
        return true;
    }
    candidate.original[YEAR].map(|(_, len)| len) != Some(2)
        || request.original[YEAR].map(|(_, len)| len) == Some(2)
}

struct Matcher<'a> {
    locale: &'a str,
    calendar: &'a str,
    request: Skeleton,
    candidates: Vec<Candidate>,
    match_hour_length: bool,
    decimal: &'a str,
}

impl Matcher<'_> {
    /// The closest candidate for the requested fields in `mask`, with the fields it lacks.
    fn best_raw(&self, mask: u32) -> Option<(&Candidate, u32, u32)> {
        let mut best: Option<(&Candidate, i32, u32, u32)> = None;
        for candidate in &self.candidates {
            if !year_length_compatible(&self.request, &candidate.skeleton) {
                continue;
            }
            let (distance, missing, extra) = distance(&self.request, &candidate.skeleton, mask);
            // Equal distances favour the larger missing-field mask, as ICU does.
            let better = match &best {
                None => true,
                Some((_, best_distance, best_missing, _)) => {
                    distance < *best_distance
                        || (distance == *best_distance && *best_missing < missing)
                }
            };
            if better {
                best = Some((candidate, distance, missing, extra));
                if distance == 0 {
                    break;
                }
            }
        }
        best.map(|(candidate, _, missing, extra)| (candidate, missing, extra))
    }

    /// Adjust the found pattern's field widths and letters to the request (UTS #35 "Matching
    /// Skeletons" adjustments as ICU's adjustFieldTypes performs them), optionally appending
    /// the requested fractional seconds to its seconds field.
    fn adjust(&self, candidate: &Candidate, fix_fraction: bool) -> String {
        let specified = candidate.specified.then_some(&candidate.skeleton);
        self.adjust_pattern(&candidate.pattern, specified, fix_fraction)
    }

    fn adjust_pattern(
        &self,
        pattern: &str,
        specified: Option<&Skeleton>,
        fix_fraction: bool,
    ) -> String {
        // The requested fractional seconds attach to the found seconds field ("s.SS").
        let extended;
        let pattern = match (fix_fraction, self.request.original[FRACTIONAL]) {
            (true, Some((fraction, count))) => {
                let decimal = quote_literal(self.decimal);
                extended = replace_first_field(pattern, SECOND, |letter, len| {
                    let mut text: String = std::iter::repeat_n(letter, len).collect();
                    text.push_str(&decimal);
                    text.extend(std::iter::repeat_n(fraction, count));
                    text
                });
                extended.as_str()
            }
            _ => pattern,
        };
        let mut tokens = tokenize(pattern);
        for token in &mut tokens {
            let Token::Field(letter, len) = token else {
                continue;
            };
            let Some(row) = type_row(*letter, *len) else {
                continue;
            };
            let field = row.field;
            let Some((requested, mut requested_len)) = self.request.original[field] else {
                continue;
            };
            if requested == 'E' && requested_len < 3 {
                requested_len = 3;
            }
            let mut adjusted_len = requested_len;
            if (field == HOUR && !self.match_hour_length)
                || field == MINUTE
                || field == SECOND
                || field == FRACTIONAL
            {
                adjusted_len = *len;
            } else if let Some(specified) = specified {
                if requested != 'c' && requested != 'e' {
                    let skeleton_len = specified.original[field].map_or(0, |(_, len)| len);
                    let pattern_numeric = row.kind > 0;
                    let requested_numeric = self.request.types[field] > 0;
                    if skeleton_len == requested_len || pattern_numeric != requested_numeric {
                        adjusted_len = *len;
                    }
                }
            }
            let mut out_letter = if field != HOUR
                && field != MONTH
                && field != WEEKDAY
                && (field != YEAR || requested == 'Y')
            {
                requested
            } else {
                *letter
            };
            if out_letter == 'E' && adjusted_len < 3 {
                out_letter = 'e';
            }
            *letter = out_letter;
            *len = adjusted_len;
        }
        untokenize(&tokens)
    }

    /// UTS #35's appending of missing fields: match the fields in `mask`, then repeatedly match
    /// the still-missing ones and append them with the locale's appendItems pattern.
    fn best_appending(&self, mask: u32) -> String {
        if mask == 0 {
            return String::new();
        }
        let Some((candidate, mut missing, _)) = self.best_raw(mask) else {
            return String::new();
        };
        let first_specified = candidate.specified.then(|| candidate.skeleton.clone());
        let mut result = self.adjust(candidate, false);
        let mut last_missing = 0;
        while missing != 0 {
            if missing == last_missing {
                break;
            }
            // The requested fractional seconds attach to a found seconds field.
            let seconds = (1 << SECOND) | (1 << FRACTIONAL);
            if missing & seconds == 1 << FRACTIONAL && mask & seconds == seconds {
                result = self.adjust_pattern(&result, first_specified.as_ref(), true);
                missing &= !(1 << FRACTIONAL);
                continue;
            }
            last_missing = missing;
            let Some((candidate, still_missing, _)) = self.best_raw(missing) else {
                break;
            };
            let addition = self.adjust(candidate, false);
            let found = missing & !still_missing;
            let top = 31 - found.leading_zeros() as usize;
            if found != 0 {
                if let Some(append) =
                    data::append_item(self.locale, self.calendar, append_field_key(top))
                {
                    let name = append_field_name(self.locale, top);
                    result = append
                        .replace("{0}", &result)
                        .replace("{1}", &addition)
                        .replace("{2}", &quote_literal(&name));
                } else {
                    result = format!("{result} {addition}");
                }
            }
            missing = still_missing;
        }
        result
    }
}

/// Replace the first field of `field` in `pattern` by the pattern text `replace` builds from it.
fn replace_first_field(
    pattern: &str,
    field: usize,
    replace: impl FnOnce(char, usize) -> String,
) -> String {
    let mut replace = Some(replace);
    let mut result = String::new();
    for token in tokenize(pattern) {
        match token {
            Token::Field(letter, len)
                if replace.is_some()
                    && type_row(letter, len).is_some_and(|row| row.field == field) =>
            {
                result.push_str(&(replace.take().unwrap())(letter, len));
            }
            other => result.push_str(&untokenize(std::slice::from_ref(&other))),
        }
    }
    result
}

fn quote_literal(text: &str) -> String {
    untokenize(&[Token::Literal(text.to_string())])
}

/// The CLDR appendItems key for a field.
fn append_field_key(field: usize) -> &'static str {
    match field {
        ERA => "Era",
        YEAR => "Year",
        QUARTER => "Quarter",
        MONTH => "Month",
        WEEK_OF_YEAR => "Week",
        WEEKDAY => "Day-Of-Week",
        DAY => "Day",
        HOUR => "Hour",
        MINUTE => "Minute",
        SECOND => "Second",
        ZONE => "Timezone",
        _ => "*",
    }
}

/// A field's localized display name (CLDR dateFields), the appendItems `{2}` argument.
fn append_field_name(locale: &str, field: usize) -> String {
    let code = match field {
        ERA => "era",
        YEAR => "year",
        QUARTER => "quarter",
        MONTH => "month",
        WEEK_OF_YEAR => "weekOfYear",
        WEEKDAY => "weekday",
        DAY => "day",
        DAY_PERIOD => "dayPeriod",
        HOUR => "hour",
        MINUTE => "minute",
        SECOND => "second",
        _ => "timeZoneName",
    };
    crate::cldr_display_names::name(locale, "dateTimeField", "long", code)
        .unwrap_or(code)
        .to_string()
}

/// The best pattern for a requested skeleton in a locale and calendar (UTS #35 Part 4 Matching
/// Skeletons). `match_hour_length` adjusts the hour width to the request, as ECMA-402's 2-digit
/// hour requires. `decimal` is the locale's decimal separator for fractional seconds.
pub(super) fn best_pattern(
    locale: &str,
    calendar: &str,
    skeleton: &str,
    match_hour_length: bool,
    decimal: &str,
) -> String {
    let matcher = Matcher {
        locale,
        calendar,
        request: Skeleton::parse(skeleton),
        candidates: candidates(locale, calendar),
        match_hour_length,
        decimal,
    };
    let all = matcher.request.mask();
    let Some((candidate, missing, extra)) = matcher.best_raw(all) else {
        return String::new();
    };
    if missing == 0 && extra == 0 {
        return matcher.adjust(candidate, false);
    }
    let date = matcher.best_appending(all & DATE_MASK);
    let time = matcher.best_appending(all & TIME_MASK);
    if date.is_empty() {
        return time;
    }
    if time.is_empty() {
        return date;
    }
    // The dateTimeFormat for the requested month width: long with a weekday is full, long is
    // long, abbreviated is medium, otherwise short.
    let month_len = matcher.request.original[MONTH].map_or(0, |(_, len)| len);
    let has_weekday = matcher.request.original[WEEKDAY].is_some();
    let style = match month_len {
        4 if has_weekday => "full",
        4 => "long",
        3 => "medium",
        _ => "short",
    };
    data::connector(locale, calendar, style)
        .replace("{1}", &date)
        .replace("{0}", &time)
}

/// Replace the field letter `from` by `to` throughout a pattern (outside quoted text).
pub(super) fn replace_letter(pattern: &str, from: char, to: char) -> String {
    let tokens: Vec<Token> = tokenize(pattern)
        .into_iter()
        .map(|token| match token {
            Token::Field(letter, len) if letter == from => Token::Field(to, len),
            other => other,
        })
        .collect();
    untokenize(&tokens)
}

/// Replace a pattern's hour letters by the formatter's hour-cycle letter (h11 → K, h12 → h,
/// h23 → H, h24 → k), as ECMA-402's [[HourCycle]] requires of a pattern of the same clock.
pub(super) fn with_hour_letter(pattern: &str, letter: char) -> String {
    let tokens: Vec<Token> = tokenize(pattern)
        .into_iter()
        .map(|token| match token {
            Token::Field('h' | 'H' | 'K' | 'k', len) => Token::Field(letter, len),
            other => other,
        })
        .collect();
    untokenize(&tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best(locale: &str, calendar: &str, skeleton: &str) -> String {
        best_pattern(locale, calendar, skeleton, true, ".")
    }

    #[test]
    fn widths_adjust_to_the_requested_skeleton() {
        // en has yMMMd but not yMMMMd: the found pattern's text month widens.
        assert_eq!(best("en", "gregory", "yMMMMd"), "MMMM d, y");
        assert_eq!(best("en", "gregory", "MMMM"), "LLLL");
        assert_eq!(best("en", "gregory", "MMdd"), "MM/dd");
        // ja's numeric-month patterns for text skeletons keep their numeric month.
        assert_eq!(best("ja", "gregory", "yMMMM"), "y年M月");
        assert_eq!(best("ja", "gregory", "MMMMd"), "M月d日");
    }

    #[test]
    fn missing_fields_split_and_append() {
        assert_eq!(
            best("en", "gregory", "yMMMMdhm"),
            "MMMM d, y 'at' h:mm\u{202f}a"
        );
        assert_eq!(best("en", "gregory", "yMMMdhm"), "MMM d, y, h:mm\u{202f}a");
        assert_eq!(best("en", "gregory", "yMMMzzzz"), "MMM y, zzzz");
        assert_eq!(best("en", "gregory", "yd"), "y ('day': d)");
        assert_eq!(best("en", "gregory", "sSS"), "s.SS");
        assert_eq!(best("en", "gregory", "mm"), "m");
    }

    #[test]
    fn calendars_supply_their_own_patterns() {
        assert_eq!(best("en", "buddhist", "yMMMd"), "MMM d, y G");
        assert_eq!(best("en", "chinese", "y"), "r(U)");
        assert_eq!(best("en", "japanese", "yMMMd"), "MMM d, y G");
    }

    #[test]
    fn patterns_round_trip_through_tokens() {
        for pattern in [
            "EEEE, d 'de' MMMM 'de' y",
            "h 'o''clock' a",
            "d.M.y",
            "HH'h'mm",
        ] {
            assert_eq!(untokenize(&tokenize(pattern)), pattern);
        }
        assert_eq!(with_hour_letter("h:mm a", 'K'), "K:mm a");
        assert_eq!(quote_literal(" o'clock"), " 'o''clock'");
        assert_eq!(
            tokenize(&quote_literal("it's d")),
            vec![Token::Literal("it's d".into())]
        );
        assert_eq!(pattern_skeleton("HH 'Uhr'"), "HH");
    }
}
