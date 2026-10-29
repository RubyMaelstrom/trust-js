//! The native RegExp tier against the matcher (see `regex_native.rs`): an instruction corpus, a
//! fuzzer whose differences are reduced to a minimal pattern and subject, interruption, the
//! backtracking state's limit, and the tier-up rules.

use super::super::{test_poll_word, Inst, MatchError, ReInput, Regex, MAX_BACKTRACK_STATE};
use super::{decline, note_execution, Width, BACKEND};
use std::sync::Arc;

include!("regex_native_generated.rs");

/// A subject of one byte a unit (as one-byte storage holds them, Latin-1 bytes included) or two.
#[derive(Clone, Debug)]
enum Subject {
    One(Vec<u8>),
    Two(Vec<u16>),
}

impl Subject {
    fn len(&self) -> usize {
        match self {
            Subject::One(bytes) => bytes.len(),
            Subject::Two(units) => units.len(),
        }
    }

    fn width(&self) -> Width {
        match self {
            Subject::One(_) => Width::One,
            Subject::Two(_) => Width::Two,
        }
    }

    fn without(&self, at: usize) -> Subject {
        match self {
            Subject::One(bytes) => {
                let mut bytes = bytes.clone();
                bytes.remove(at);
                Subject::One(bytes)
            }
            Subject::Two(units) => {
                let mut units = units.clone();
                units.remove(at);
                Subject::Two(units)
            }
        }
    }

    fn of(text: &str, two: bool) -> Subject {
        if two {
            Subject::Two(text.encode_utf16().collect())
        } else {
            Subject::One(text.chars().map(|c| c as u32 as u8).collect())
        }
    }
}

type Found = Result<Option<Vec<Option<(usize, usize)>>>, MatchError>;

/// Run `re` over `subject` from `start` through exec's path, which runs the pattern's native
/// code for the subject's width when it has some. In Unicode mode a start inside a surrogate
/// pair begins at the pair, as RegExpBuiltinExec's (`Regex::exec_at`) does.
fn run(re: &Regex, subject: &Subject, start: usize, control: &crate::RuntimeInterrupt) -> Found {
    let word = test_poll_word(control);
    let found = match subject {
        Subject::One(bytes) => re.exec_impl(&bytes[..], start, control, &word),
        Subject::Two(units) if re.unicode => {
            let input = super::super::Paired(units);
            re.exec_impl(input, input.boundary(start), control, &word)
        }
        Subject::Two(units) => re.exec_impl(&units[..], start, control, &word),
    };
    found.map(|found| found.map(|captures| captures.to_vec()))
}

/// `pattern` with `flags` twice: compiled for `width` at once, and kept on the matcher. `None`
/// if it does not parse.
fn pair(pattern: &str, flags: &str, width: Width) -> Option<(Regex, Regex)> {
    let mut native = Regex::new(pattern, flags).ok()?;
    native.tier_up_at = 1;
    note_execution(&native, width, 0);
    let mut matcher = Regex::new(pattern, flags).ok()?;
    matcher.tier_up_at = 0;
    Some((native, matcher))
}

/// Whether the tier takes `re` for `width`, and if so that it made code for it.
fn compiled(re: &Regex, width: Width) -> bool {
    let taken = decline(re, width).is_none();
    if taken && BACKEND {
        assert!(
            re.native.get(width).is_some(),
            "/{}/{} compiles for {width:?}",
            re.source,
            re.flags
        );
    }
    taken && BACKEND
}

/// Compare the native code with the matcher at every start of `subject` (and one past it).
fn compare(pattern: &str, flags: &str, subject: &Subject) -> Option<(usize, Found, Found)> {
    compare_from(pattern, flags, subject, 0..=subject.len() + 1)
}

/// Compare the native code with the matcher from each of `starts`.
fn compare_from(
    pattern: &str,
    flags: &str,
    subject: &Subject,
    starts: impl IntoIterator<Item = usize>,
) -> Option<(usize, Found, Found)> {
    let (native, matcher) = pair(pattern, flags, subject.width())?;
    let control = crate::RuntimeInterrupt::default();
    // Catastrophic backtracking is in the fuzzer's grammar: each execution has a deadline. One
    // the matcher misses leaves nothing to compare, so the native code runs only after it; one
    // the native code misses where the matcher kept its own is a difference.
    let deadline = |millis| {
        control.set_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_millis(millis),
        ))
    };
    let late = |found: &Found| {
        *found
            == Err(MatchError::Interrupted(
                crate::InterruptReason::DeadlineExceeded,
            ))
    };
    for start in starts {
        deadline(250);
        let reference = run(&matcher, subject, start, &control);
        if late(&reference) {
            break;
        }
        deadline(4000);
        let found = run(&native, subject, start, &control);
        if found != reference {
            return Some((start, found, reference));
        }
    }
    None
}

/// The shortest pattern and subject, by deleting characters of each, that still show a
/// difference between the native code and the matcher.
fn reduce(pattern: &str, flags: &str, subject: &Subject) -> (String, Subject) {
    let mut pattern: Vec<char> = pattern.chars().collect();
    let mut subject = subject.clone();
    loop {
        let mut shorter = false;
        for at in 0..pattern.len() {
            let mut candidate = pattern.clone();
            candidate.remove(at);
            let text: String = candidate.iter().collect();
            if compare(&text, flags, &subject).is_some() {
                pattern = candidate;
                shorter = true;
                break;
            }
        }
        for at in 0..subject.len() {
            let candidate = subject.without(at);
            let text: String = pattern.iter().collect();
            if compare(&text, flags, &candidate).is_some() {
                subject = candidate;
                shorter = true;
                break;
            }
        }
        if !shorter {
            return (pattern.into_iter().collect(), subject);
        }
    }
}

/// Fail with the reduced difference, if `pattern` shows one over `subject`.
fn check(pattern: &str, flags: &str, subject: &Subject) {
    if let Some((start, found, reference)) = compare(pattern, flags, subject) {
        let (small, small_subject) = reduce(pattern, flags, subject);
        let detail = compare(&small, flags, &small_subject);
        panic!(
            "native code and matcher differ: /{pattern}/{flags} over {subject:?} from {start}: \
             {found:?} against {reference:?}\nreduced: /{small}/{flags} over {small_subject:?}: \
             {detail:?}"
        );
    }
}

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    /// The generator of `seed`'s stream: splitmix64 mixes the seed into a nonzero state, so that
    /// every seed has a stream of its own.
    fn seeded(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Rng((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// The fuzzer's atoms: characters (one-byte, Latin-1 and two-byte ones, line terminators), `.`,
/// classes with members on either side of 256, and escapes.
const ATOMS: &[&str] = &[
    "a",
    "b",
    "c",
    "x",
    " ",
    "\\n",
    "\\r",
    "é",
    "Ω",
    "\\u2028",
    ".",
    "[ab]",
    "[^a]",
    "[a-c]",
    "\\d",
    "\\w",
    "\\s",
    "\\W",
    "\\S",
    "[\\s\\S]",
    "[é-ü]",
    "[^\\n]",
    "[Ω-Ϋ]",
    "[\\u2000-\\u200a]",
    "[^\\u0100-\\uffff]",
    "0",
    "_",
    "-",
    "[\\w-]",
    "[^\\s\\d]",
    "\\x00",
    "[\\u0000-\\u00ff]",
    // Case variants, and the case relation's exceptions under `i`: characters whose uppercase is
    // ASCII (ſ, ı) or not one character (ß), the Kelvin sign, Latin-1 letters whose case partner
    // is outside Latin-1 (ÿ, µ), and groups of three and four (ǅ, Σ, ι).
    "A",
    "K",
    "s",
    "ß",
    "ÿ",
    "µ",
    "\\u212a",
    "\\u017f",
    "\\u0131",
    "\\u01c5",
    "Σ",
    "\\u0345",
    "[A-Z]",
    "[^a-z]",
    "[à-þ]",
    "[k-s]",
    "[ǅ-ǆ]",
    // Past the BMP and its surrogates: in Unicode mode a character, a class of them or a lone
    // surrogate (a pair's half that the subject may hold alone); elsewhere their units. The
    // Deseret letters are a case pair under `iu`.
    "😀",
    "\\u{1F600}",
    "\\ud83d",
    "\\ude00",
    "[😀-😂]",
    "[^😀]",
    "[\\u{10000}-\\u{10ffff}]",
    "[\\ud800-\\udbff]",
    "[\\udc00-\\udfff]",
    "\\u{10400}",
    "\\u{10428}",
    "\\p{ASCII}",
    "\\P{ASCII}",
];

/// Near misses: programs one instruction away from what the tier takes (`v`-mode class strings,
/// under `v`), which run on the matcher.
const NEAR_MISSES: &[&str] = &["[\\q{ab|c}]", "[\\q{}a]", "[\\q{😀a}]"];

const QUANTIFIERS: &[&str] = &[
    "*", "+", "?", "{2}", "{1,3}", "{0,}", "{2,}", "{0,2}", "{3,5}",
];

/// Counts too large to unroll, which make a group a counted loop.
const COUNTED: &[&str] = &["{1001}", "{0,1001}", "{1001,}", "{2,1200}"];

/// Backreferences: to groups the pattern may have or not, before or inside them, and by name.
const BACKREFS: &[&str] = &["\\1", "\\2", "\\3", "\\k<n>", "\\k<d>"];

/// Inline modifier groups' flags.
const MODIFIERS: &[&str] = &["?i", "?-i", "?m", "?-m", "?s", "?-s", "?i-s", "?ms-i"];

fn gen_alternation(rng: &mut Rng, depth: usize) -> String {
    let count = if rng.chance(30) { 2 + rng.below(3) } else { 1 };
    (0..count)
        .map(|_| gen_sequence(rng, depth))
        .collect::<Vec<_>>()
        .join("|")
}

fn gen_sequence(rng: &mut Rng, depth: usize) -> String {
    let count = rng.below(4) + usize::from(depth == 0);
    (0..count).map(|_| gen_term(rng, depth)).collect()
}

fn gen_term(rng: &mut Rng, depth: usize) -> String {
    let (atom, quantifiable) = match rng.below(34) {
        0..=12 => (rng.pick(ATOMS).to_string(), true),
        13 => ("^".to_string(), false),
        14 => ("$".to_string(), false),
        15 => ("\\b".to_string(), false),
        16 => ("\\B".to_string(), false),
        17..=19 if depth < 3 => (format!("({})", gen_alternation(rng, depth + 1)), true),
        20..=21 if depth < 3 => (format!("(?:{})", gen_alternation(rng, depth + 1)), true),
        // Lookaheads, quantifiable outside Unicode mode (Annex B).
        22..=23 if depth < 3 => {
            let kind = if rng.chance(50) { "?=" } else { "?!" };
            (format!("({kind}{})", gen_alternation(rng, depth + 1)), true)
        }
        24..=25 => (rng.pick(BACKREFS).to_string(), false),
        // Named groups, a name repeated across alternatives.
        26 if depth < 3 => {
            let name = if rng.chance(50) { "n" } else { "d" };
            let group = format!("(?<{name}>{})", gen_alternation(rng, depth + 1));
            if name == "d" {
                (
                    format!("(?:{group}|(?<d>{}))", gen_sequence(rng, depth + 1)),
                    true,
                )
            } else {
                (group, true)
            }
        }
        27..=28 if depth < 3 => (
            format!(
                "({}:{})",
                rng.pick(MODIFIERS),
                gen_alternation(rng, depth + 1)
            ),
            true,
        ),
        // Lookbehinds, which no quantifier takes.
        29..=30 if depth < 3 => {
            let kind = if rng.chance(50) { "?<=" } else { "?<!" };
            (
                format!("({kind}{})", gen_alternation(rng, depth + 1)),
                false,
            )
        }
        31 if rng.chance(25) => (rng.pick(NEAR_MISSES).to_string(), false),
        _ => (rng.pick(ATOMS).to_string(), true),
    };
    if quantifiable && rng.chance(35) {
        let lazy = if rng.chance(40) { "?" } else { "" };
        let count = if rng.chance(8) {
            rng.pick(COUNTED)
        } else {
            rng.pick(QUANTIFIERS)
        };
        format!("{atom}{count}{lazy}")
    } else {
        atom
    }
}

fn gen_flags(rng: &mut Rng) -> String {
    let mut flags = String::new();
    for (flag, percent) in [
        ('d', 5),
        ('g', 20),
        ('m', 30),
        ('s', 25),
        ('y', 20),
        ('i', 30),
    ] {
        if rng.chance(percent) {
            flags.push(flag);
        }
    }
    // Unicode mode: `u`, or now and then `v`.
    if rng.chance(30) {
        flags.push(if rng.chance(15) { 'v' } else { 'u' });
    }
    flags
}

const ONE_BYTE_UNITS: &[u16] = &[
    b'a' as u16,
    b'b' as u16,
    b'c' as u16,
    b'x' as u16,
    b' ' as u16,
    b'\n' as u16,
    b'\r' as u16,
    0xe9,
    b'0' as u16,
    b'1' as u16,
    b'_' as u16,
    b'-' as u16,
    0xfc,
    0,
    b'A' as u16,
    b'K' as u16,
    b'S' as u16,
    b'X' as u16,
    b'k' as u16,
    b's' as u16,
    0xc9,
    0xdf,
    0xff,
    0xb5,
];

const TWO_BYTE_UNITS: &[u16] = &[
    0x3a9, 0x2028, 0x2029, 0xd83d, 0xde00, 0x3b0, 0x100, 0x2005, 0x3c9, 0x178, 0x17f, 0x212a,
    0x39c, 0x3bc, 0x1e9e, 0x131, 0x130, 0x3a3, 0x3c3, 0x3c2, 0x1c4, 0x1c5, 0x1c6, 0x345, 0x399,
    0x3b9, 0x1fbe,
];

/// Surrogate pairs a two-byte subject may hold: 😀 and its neighbour, and the Deseret letters.
const PAIRS: &[[u16; 2]] = &[
    [0xd83d, 0xde00],
    [0xd83d, 0xde01],
    [0xd801, 0xdc00],
    [0xd801, 0xdc28],
];

fn gen_subject(rng: &mut Rng, two: bool) -> Subject {
    let len = if rng.chance(10) {
        20 + rng.below(40)
    } else {
        rng.below(13)
    };
    let mut units: Vec<u16> = Vec::with_capacity(len + 2);
    while units.len() < len {
        if two && rng.chance(12) {
            units.extend_from_slice(&PAIRS[rng.below(PAIRS.len())]);
        } else if two && rng.chance(30) {
            units.push(TWO_BYTE_UNITS[rng.below(TWO_BYTE_UNITS.len())]);
        } else {
            units.push(ONE_BYTE_UNITS[rng.below(ONE_BYTE_UNITS.len())]);
        }
    }
    if two {
        Subject::Two(units)
    } else {
        Subject::One(units.iter().map(|&unit| unit as u8).collect())
    }
}

/// What a fuzzing run covered: the cases run, the patterns compiled for a width, those the tier
/// compiled, and of these the ones in Unicode mode, with a lookbehind, with a loop that checks
/// for an empty iteration and with a loop that does not (Decision 25).
#[derive(Default)]
struct Fuzzed {
    cases: usize,
    patterns: usize,
    native: usize,
    unicode: usize,
    lookbehind: usize,
    checked_loops: usize,
    unchecked_loops: usize,
}

/// Whether program `prog`, with its lookaround bodies, has a loop that checks for an empty
/// iteration, and one that does not: a loop's back edge or count that no check precedes.
fn loop_checks(prog: &[Inst]) -> (bool, bool) {
    let mut checked = false;
    let mut unchecked = false;
    for (pc, inst) in prog.iter().enumerate() {
        let after_check = pc > 0
            && matches!(
                prog[pc - 1],
                Inst::CheckProgress(_) | Inst::CheckCountedProgress { .. }
            );
        match inst {
            Inst::CheckProgress(_) | Inst::CheckCountedProgress { .. } => checked = true,
            Inst::Jmp(target) if *target < pc && !after_check => unchecked = true,
            Inst::CounterIncr(_) if !after_check => unchecked = true,
            Inst::Look { prog: body, .. } | Inst::LookBehind { prog: body, .. } => {
                let (body_checked, body_unchecked) = loop_checks(body);
                checked |= body_checked;
                unchecked |= body_unchecked;
            }
            _ => {}
        }
    }
    (checked, unchecked)
}

/// Fuzz `cases` executions (subject and start) from `seed`: random patterns over the tier's
/// grammar and its near misses, with random flags, against one-byte and two-byte subjects at
/// every start.
fn fuzz(seed: u64, cases: usize) -> Fuzzed {
    let mut rng = Rng::seeded(seed);
    let trace = std::env::var_os("TRUST_JS_REGEXP_FUZZ_TRACE").is_some();
    let mut fuzzed = Fuzzed::default();
    while fuzzed.cases < cases {
        let pattern = gen_alternation(&mut rng, 0);
        let flags = gen_flags(&mut rng);
        if Regex::new(&pattern, &flags).is_err() {
            continue;
        }
        for two in [false, true] {
            let width = if two { Width::Two } else { Width::One };
            if let Some((native, _)) = pair(&pattern, &flags, width) {
                fuzzed.patterns += 1;
                if compiled(&native, width) {
                    let (checked, unchecked) = loop_checks(&native.prog);
                    fuzzed.native += 1;
                    fuzzed.unicode += usize::from(native.unicode);
                    fuzzed.lookbehind +=
                        usize::from(pattern.contains("(?<=") || pattern.contains("(?<!"));
                    fuzzed.checked_loops += usize::from(checked);
                    fuzzed.unchecked_loops += usize::from(unchecked);
                }
            }
            for _ in 0..4 {
                let subject = gen_subject(&mut rng, two);
                fuzzed.cases += subject.len() + 2;
                if trace {
                    eprintln!("[fuzz] /{pattern}/{flags} {subject:?}");
                }
                check(&pattern, &flags, &subject);
            }
        }
    }
    fuzzed
}

/// The fuzzer at the lib suite's size; `TRUST_JS_REGEXP_FUZZ_CASES=N` (with
/// `TRUST_JS_REGEXP_FUZZ_SEED=S`) runs the differential gate's (at least 1M a stage).
#[test]
fn fuzzed_programs_match_as_the_matcher_does() {
    let cases = std::env::var("TRUST_JS_REGEXP_FUZZ_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(40_000);
    let seed = std::env::var("TRUST_JS_REGEXP_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x9e37_79b9_7f4a_7c15);
    let fuzzed = fuzz(seed, cases);
    eprintln!(
        "[regexp-native-fuzz] seed {seed}: {} cases, {} of {} programs native, {} of them in \
         Unicode mode, {} with a lookbehind, {} with a loop checking for an empty iteration and \
         {} with a loop that need not",
        fuzzed.cases,
        fuzzed.native,
        fuzzed.patterns,
        fuzzed.unicode,
        fuzzed.lookbehind,
        fuzzed.checked_loops,
        fuzzed.unchecked_loops,
    );
    if BACKEND {
        // The grammar makes loops whose bodies can match empty and loops whose bodies cannot.
        assert!(fuzzed.native > 0 && fuzzed.checked_loops > 0 && fuzzed.unchecked_loops > 0);
    }
}

/// A pattern over subjects of both widths, from every start.
fn check_both(pattern: &str, flags: &str, texts: &[&str]) {
    for text in texts {
        for two in [false, true] {
            if !two && text.chars().any(|c| c as u32 > 0xff) {
                continue;
            }
            check(pattern, flags, &Subject::of(text, two));
        }
    }
}

#[test]
fn core_programs_match_as_the_matcher_does() {
    let texts = [
        "",
        "a",
        "abc",
        "xxaXcyy abc ab",
        "a\nb\r\nc",
        "  trailing   ",
        "foo bar_baz 12 x-y",
        "http://user:pw@host:80/path?q#frag",
        "aaaaab",
        "é ü Ω\u{2028}z",
    ];
    for (pattern, flags) in [
        ("a.c", ""),
        ("a.c", "s"),
        ("(a)|b", ""),
        ("(a|ab)(c|bcd)(d*)", ""),
        ("^\\s*|\\s*$", "g"),
        ("\\s+$", ""),
        ("$", "m"),
        ("^b", "m"),
        ("\\b\\w+\\b", "g"),
        ("\\B\\w", ""),
        ("(\\d*)(\\D*)", "g"),
        (
            "^(?:([\\w+\\-.]+):)?\\/\\/(?:(\\w+:\\w+)@)?([\\w.-]*)(?::(\\d+))?(.*)$",
            "",
        ),
        ("(a*)*b", ""),
        ("(a*?)+?c", ""),
        ("(?:a|b)*?c", ""),
        ("(z)((a+)?(b+)?(c))*", ""),
        ("x*y+$", "my"),
        ("[^\\s]+", "y"),
        (".{2,4}?b", ""),
        ("a{0,3}a{2}", ""),
        ("([a-c]){2,3}", "d"),
        ("(?:)", ""),
        ("()", ""),
        ("(|a)+b", ""),
        ("[é-ü]+.", ""),
        ("[Ω-Ϋ\\u2028]", ""),
        ("[^Ω]+", ""),
        ("\\u2028|\\n$", "m"),
        ("ba{3000000000}|b", ""),
        ("ba{3000000000}?|b", ""),
        ("b(?:a{2147483648,})?", ""),
    ] {
        check_both(pattern, flags, &texts);
    }
}

/// Programs under `i`, closed over case, over subjects with case variants: the census's case-
/// insensitive patterns (Octane's cookie readers, its browser sniffers and alternations of words,
/// CoffeeScript's number lexer) and the case relation's exceptions.
#[test]
fn case_closed_programs_match_as_the_matcher_does() {
    let texts = [
        "",
        "HFUCJRN = abc; x",
        "a hfucjrn=1;hFuCjRn  =  two",
        "xHfUcJrN = 2",
        "Fnsnev xbadhrebe XUGZY fnsNEV",
        "0B101 0b2 0XfF 0x 1E5 1.5e-3 .5 0O17",
        "?a=b&SEVRAQVQ=x#y ?q PunaaryVQ=",
        "x.RONL qri-ehf3.wbg.FGBERF",
        "ÿŸ µΜμ ßẞ SSs ſ K k İı ǄǅǆΣσς",
        "AbCdEfGhIjKlMnOpQrStUvWxYz É é Ω ω",
        "ǅeep σΣς ιΙ\u{345}\u{1fbe}x",
    ];
    for (pattern, flags) in [
        ("\\bhfucjrn\\s*=\\s*([^;]*)", "i"),
        ("Fnsnev|Xbadhrebe|XUGZY", "gi"),
        (
            "^0b[01]+|^0o[0-7]+|^0x[\\da-f]+|^\\d*\\.?\\d+(?:e[+-]?\\d+)?",
            "i",
        ),
        (
            "\\?[\\w\\W]*(sevraqvq|punaaryvq|tebhcvq)=([^\\&\\?#]*)",
            "i",
        ),
        ("(^|.)(ronl|qri-ehf3.wbg)(|fgberf|zbgbef)", "i"),
        ("[a-z]+", "i"),
        ("[^a-z]+", "i"),
        ("\\w+\\b", "i"),
        ("[à-þ]+", "gi"),
        ("ÿ|µ|ß|ſ|\\u212a", "gi"),
        ("ss|s+", "i"),
        ("k+", "gi"),
        ("\\u01c5+\\u03c3*", "i"),
        ("[ǅσ]{2,}", "i"),
        ("(?:ι|θ)+x", "i"),
        ("é{2}|Ω.?ω", "im"),
        ("^[a-f]*?Z", "im"),
        ("\\B[a-z]\\b", "gi"),
    ] {
        let (native, _) = pair(pattern, flags, Width::Two).unwrap();
        assert!(
            compiled(&native, Width::Two) || !BACKEND,
            "/{pattern}/{flags} is in the tier's instruction set"
        );
        check_both(pattern, flags, &texts);
    }
}

/// Lookaheads, backreferences, counted loops and inline modifier groups (W5b's 3c): CoffeeScript's
/// lexer patterns, which put a fifth of its matcher work behind lookaheads and backreferences,
/// and the cases of their rules: captures inside lookaheads, positive and negative, nested and in
/// loops; backreferences to unset, empty, enclosing and duplicate named groups, under `i` too;
/// counted loops' mandatory and optional iterations; flags changed for a group's instructions.
#[test]
fn lookahead_backreference_counted_and_modifier_programs_match_as_the_matcher_does() {
    let texts = [
        "",
        "foo: bar",
        "$x_1:: y = 2 # note",
        "  ### block ### x",
        "# a\n  # b\n##c",
        "a --> b ++ c -= d >>>= e ?. f ?:: g ... h",
        "/re[/]x\\/y/g z",
        "'a\\x4g' \"s #{x} \\u{12} \\u12z \\0 \\8\"",
        "0x1E 1E5 0xE",
        "abab:abab aaa ab ab",
        "aAaA bB ÿŸ µΜμ kK\u{212a} ß",
        "imguy gimg yy",
        "aab aaab abc abcabc",
        "\u{2028}x\ny\r\n",
    ];
    for (pattern, flags) in [
        ("^(?!\\d)((?:(?!\\s)[$\\w\\x7f-\\uffff])+)([^\\n\\S]*:(?!:))?", ""),
        ("^(?![\\d<])((?:(?!\\s)[\\.\\-$\\w\\x7f-\\uffff])+)", ""),
        ("^\\s*###([^#][\\s\\S]*?)(?:###[^\\n\\S]*|###$)|^(?:\\s*#(?!##[^#]).*)+", ""),
        ("^(.?)\\1*$", ""),
        (
            "^(?:[-=]>|[-+*\\/%<>&|^!?=]=|>>>=?|([-+:])\\1|([&|<>*\\/%])\\2=?|\\?(\\.|::)|\\.{2,3})",
            "",
        ),
        ("^\\/(?!\\/)((?:[^[\\/\\n\\\\]|\\\\[^\\n]|\\[(?:\\\\[^\\n]|[^\\]\\n\\\\])*\\])*)(\\/)?", ""),
        ("^(?:[^\\\\\\/#\\s]|\\\\[\\s\\S]|\\/(?!\\/\\/)|\\#(?!\\{)|\\s+(?:#(?!\\{).*)?)*", ""),
        ("^\\s*(?:,|\\??\\.(?![.\\d])|::)", ""),
        ("^(?:[^\\\\\"#]|\\\\[\\s\\S]|\\#(?!\\{))*", ""),
        ("(\\\\\\\\)|(\\\\0(?=[1-7]))|\\\\?(\\/)|\\\\?(?:(\\n)|(\\r)|(\\u2028)|(\\u2029))|(\\\\.)", "g"),
        ("^(?!0x).*E", ""),
        ("\\s+(#(?!{).*)", "g"),
        ("^(?!.*(.).*\\1)[imguy]*$", ""),
        (
            "((?:^|[^\\\\])(?:\\\\\\\\)*)\\\\(?:(0[0-7]|[1-7])|(x(?![\\da-fA-F]{2}).{0,2})|(u\\{(?![\\da-fA-F]{1,}\\})[^}]*\\}?)|(u(?!\\{|[\\da-fA-F]{4}).{0,4}))",
            "",
        ),
        // Captures inside lookaheads: kept after a positive one, unset after a negative one,
        // unset again when backtracking past either, in loops and nested.
        ("(?=(a+))a*b\\1", ""),
        ("(?=(\\w+))\\1:", ""),
        ("(?!(a)b)\\w(\\1)", ""),
        ("(?:(?=(\\w))\\w)+\\1", ""),
        ("(?:(?=(a))a|b)*c\\1", ""),
        ("(?=(?!(a)b)(a|c))\\w\\2?\\1", ""),
        ("(?=a|(b))\\1x?", "g"),
        ("(?=(a)(?=(b)))?\\w\\1\\2", ""),
        ("(?!(?=(a))a)b|\\1c", ""),
        ("x(?=(y)|z)", "y"),
        ("(?=(?:a|ab)(c))\\w+", ""),
        // Backreferences.
        ("(a)|\\1b", ""),
        ("(a\\1)", ""),
        ("(?:(a)|b)\\1", "g"),
        ("(a*)\\1", ""),
        ("(a*?)\\1$", "m"),
        ("^(a+)\\1+$", ""),
        ("(\\w)\\1(\\w)\\2", "g"),
        ("(?<n>a|b)\\k<n>", ""),
        ("(?:(?<n>a)|(?<n>b))\\k<n>", "g"),
        ("([a-z])\\1", "gi"),
        ("(\\w+) \\1", "i"),
        ("(ÿ|µ|k|ß)\\1", "gi"),
        ("(.)\\1", "i"),
        ("\\2(a)(b)", ""),
        // Counted loops, too long to unroll.
        ("(?:ab){1001}", ""),
        ("(?:a|bc){2,1500}d", ""),
        ("(a|b){0,1200}?c", ""),
        ("(?:(a)|b){1001,}\\1?", ""),
        ("(?:a?){1500}b", ""),
        ("((?:a|b)c?){1002}", ""),
        // Inline modifier groups.
        ("a(?i:b)c", ""),
        ("(?i:a(?-i:b)c)", ""),
        ("(?m:^b)", ""),
        ("x(?-m:$)|y$", "m"),
        ("(?s:a.b)|a.c", ""),
        ("(?-s:a.b)", "s"),
        ("(?i:(a)\\1)", ""),
        ("(?-i:(a)\\1)", "i"),
        ("(?i:[a-z]+)(?-i:B)", ""),
        ("(?i:\\bk+)", ""),
        ("(?ims:^.a$)", ""),
    ] {
        for two in [false, true] {
            let width = if two { Width::Two } else { Width::One };
            let (native, _) = pair(pattern, flags, width).unwrap();
            assert!(
                compiled(&native, width) || !BACKEND,
                "/{pattern}/{flags} is in the tier's instruction set for {width:?}"
            );
        }
        check_both(pattern, flags, &texts);
    }
    // Counted loops over subjects long enough to leave them.
    let long = [
        "ab".repeat(1001),
        format!("{}d", "abc".repeat(600)),
        format!("{}c", "ab".repeat(700)),
        format!("{}b", "a".repeat(1600)),
        "abc".repeat(1100),
    ];
    for (pattern, flags) in [
        ("(?:ab){1001}", ""),
        ("(?:a|bc){2,1500}d", ""),
        ("(a|b){0,1200}?c", ""),
        ("(?:(a)|b){1001,}\\1?", ""),
        ("(?:a?){1500}b", ""),
        ("((?:a|b)c?){1002}", ""),
    ] {
        for text in &long {
            for two in [false, true] {
                let subject = Subject::of(text, two);
                if let Some(difference) = compare_from(pattern, flags, &subject, [0, 1, 7]) {
                    panic!(
                        "/{pattern}/{flags} over {} units: {difference:?}",
                        text.len()
                    );
                }
            }
        }
    }
}

/// Unicode mode and lookbehind (W5b's 3d): strings.js's lexers over subjects whose elements are
/// units and surrogate pairs (lone surrogates too), the matcher's `Paired` input; characters,
/// classes and loops past the BMP, `.`, negated classes and lone surrogates, decoded where they
/// are; case-insensitive Unicode matching (simple case folding, `\w` and `\b` with the long s
/// and the Kelvin sign); backreferences that must not end inside a pair; lookbehinds of every
/// kind, reading backward through pairs, with captures, backreferences, loops and nested
/// lookarounds.
#[test]
fn unicode_and_lookbehind_programs_match_as_the_matcher_does() {
    let texts = [
        "",
        "var x = foo(bar, 42); // a comment about things\n",
        "var x = f\u{e9}o(bar, 42); // a \u{4e2d} comment\n",
        "var \u{1f600}x = 1; // \u{1f600}\u{1f601} smile\n",
        "\u{1f600}\u{1f600}a\u{1f601}b\u{10400}\u{10428}c\u{1f600}",
        "aaa\u{1f600}bbb\u{1f600}\u{1f600}ccc",
        "x\u{2028}y\r\n\u{1f600}\n",
        "abc abc ab\u{1f600}ab\u{1f600}",
        "K\u{212a}k \u{17f}sS \u{10400}\u{10428} \u{3a3}\u{3c3}\u{3c2}",
        "word \u{1f600}word w\u{17f}rd",
        "12ab 3\u{1f600}4 56",
    ];
    for (pattern, flags) in [
        // strings.js's sticky lexer and its anchored per-chunk matcher.
        ("[A-Za-z_]\\w*|\\d+|\\s+|\\/\\/[^\\n]*|.", "uy"),
        ("^(?:[A-Za-z_]\\w*|\\d+|\\s+|.)", "u"),
        ("[A-Za-z_]\\w*|\\d+|\\s+|\\/\\/[^\\n]*|.", "gu"),
        // Elements past the BMP and lone surrogates.
        ("(.)", "gu"),
        ("..", "gu"),
        ("\\u{1f600}+", "gu"),
        ("\\u{1f600}{2}a", "u"),
        ("[\\u{1f600}-\\u{1f64f}]+\\w", "gu"),
        ("[^a]{2,3}?b", "u"),
        ("[^\\n]*$", "mu"),
        ("(\\ud83d)", "gu"),
        ("\\ude00|x", "gu"),
        ("([\\ud800-\\udbff])", "gu"),
        ("[\\udc00-\\udfff]\\w", "u"),
        ("\\P{ASCII}+", "gu"),
        ("\\p{ASCII}+\\u{1f600}", "u"),
        (".*?\\u{1f601}", "u"),
        ("(.)(.)\\2\\1", "u"),
        ("a.{0,2}?c", "su"),
        // Case-insensitive Unicode mode: simple case folding, \w and \b.
        ("\\u{10400}+", "giu"),
        ("k\\w", "giu"),
        ("\\bs\\w*", "giu"),
        ("\\w+\\B", "iu"),
        ("(\\u{10400}|\\u{3a3})\\1", "giu"),
        ("[a-z\\u{10428}]+", "iu"),
        // Backreferences over pairs.
        ("(..)\\1", "gu"),
        ("(\\ud83d)\\1", "u"),
        ("(a\\u{1f600}?)\\1", "gu"),
        // Lookbehinds.
        ("(?<=a)b", "g"),
        ("(?<!a)b", "g"),
        ("(?<=\\u{1f600})\\w", "gu"),
        ("(?<!\\u{1f600})[a-z]", "gu"),
        ("(?<=(\\d+))\\w", "g"),
        ("(?<=\\b\\w+)\\s", "g"),
        ("(?<=^a*)b", "gm"),
        ("(?<=[^\\n]*)x", "u"),
        ("(?<=.{2})\\w", "gu"),
        ("(?<=(.)\\1)\\w", "gu"),
        ("(?<=\\1(.))x", "u"),
        ("(?<=(?=ab)a)b", "g"),
        ("(?=(?<=a)b)\\w", "g"),
        ("(?<=a(?<!ba))c", "g"),
        ("(?<=(a|b)+?)c", "g"),
        ("(?<=(?:ab){1001})", "y"),
        ("(?<!(a)b)c\\1", "g"),
        ("(?<=\\u{1f600}{2})a", "u"),
        ("(?<=[\\u{1f600}-\\u{1f64f}]*)b", "gu"),
        ("(?<=\\ud83d)x", "gu"),
        ("(?<=(?i:K))\\w", "gu"),
        ("(?<=\\w)\\b", "gu"),
    ] {
        for two in [false, true] {
            let width = if two { Width::Two } else { Width::One };
            let (native, _) = pair(pattern, flags, width).unwrap();
            assert!(
                compiled(&native, width) || !BACKEND,
                "/{pattern}/{flags} is in the tier's instruction set for {width:?}"
            );
        }
        check_both(pattern, flags, &texts);
    }
}

/// The word characters `\\b` tests under `iu` beyond `[0-9A-Za-z_]` (WordCharacters with the
/// characters that fold into it), which the native code compares in line, are the long s and
/// the Kelvin sign, of every two-byte element.
#[test]
fn unicode_case_insensitive_word_characters_are_the_long_s_and_the_kelvin_sign() {
    let extra: Vec<u32> = (0..=0xffffu32)
        .filter(|&c| super::super::is_word_ic(c, true, true) && !super::super::is_word(c))
        .collect();
    assert_eq!(extra, [0x17f, 0x212a]);
}

/// In Unicode mode, every element of the two-byte range, as a unit, a lone surrogate or the
/// halves of a pair, decodes as the matcher's `Paired` input steps it: each unit, then elements
/// past the BMP at the edges of each plane, each followed by `x`, under `.`, a negated class
/// and an astral class, forward and in a lookbehind.
#[test]
fn unicode_mode_decodes_every_element_as_the_matcher_does() {
    let mut units: Vec<u16> = Vec::new();
    for unit in 0..=0xffffu32 {
        units.push(unit as u16);
        units.push(b'x' as u16);
    }
    for plane in 1..=16u32 {
        for low in [0u32, 1, 0x3ff, 0x400, 0xfffe, 0xffff] {
            let c = char::from_u32(plane << 16 | low).unwrap();
            let mut pair = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut pair));
            units.push(b'x' as u16);
        }
    }
    let subject = Subject::Two(units);
    for pattern in [
        ".x",
        "[^a]x",
        "[\\u{10000}-\\u{10ffff}]x",
        "(?<=.)x",
        "(?<=[^a])x",
    ] {
        let (native, matcher) = pair(pattern, "u", Width::Two).unwrap();
        assert!(
            compiled(&native, Width::Two) || !BACKEND,
            "/{pattern}/u compiles"
        );
        let control = crate::RuntimeInterrupt::default();
        let mut start = 0;
        loop {
            let found = run(&native, &subject, start, &control);
            let reference = run(&matcher, &subject, start, &control);
            assert_eq!(found, reference, "/{pattern}/u from {start}");
            match found {
                Ok(Some(spans)) => start = spans[0].unwrap().1,
                _ => break,
            }
        }
    }
}

/// A backreference under `i` compares Canonicalize's (the legacy one, outside Unicode mode): for
/// every two-byte element, its case partners match it and a neighbour does not.
#[test]
fn case_insensitive_backreferences_compare_every_element_as_the_matcher_does() {
    let (native, matcher) = pair("(.)\\1", "i", Width::Two).unwrap();
    compiled(&native, Width::Two);
    let control = crate::RuntimeInterrupt::default();
    let mut units = Vec::new();
    for unit in 0..=0xffffu32 {
        let partners: Vec<u32> = super::super::case_group(unit, false)
            .into_iter()
            .chain([unit ^ 1, unit])
            .filter(|&other| other <= 0xffff)
            .collect();
        for other in partners {
            units.extend_from_slice(&[unit as u16, other as u16, b'\n' as u16]);
        }
    }
    let subject = Subject::Two(units);
    let mut start = 0;
    loop {
        let found = run(&native, &subject, start, &control);
        let reference = run(&matcher, &subject, start, &control);
        assert_eq!(found, reference, "/(.)\\1/i from {start}");
        match found {
            Ok(Some(spans)) => start = spans[0].unwrap().1,
            _ => break,
        }
    }
}

/// Programs the fuzzer seldom makes: more capture slots than the root floor's logged-slot word
/// has bits (each attempt unsets them instead), many loops with and without marks, long straight
/// runs, and code past AArch64's 32 KB test-and-branch reach.
#[test]
fn wide_and_long_programs_match_as_the_matcher_does() {
    let wide = format!("{}(y|z)*q", "(x)?".repeat(36));
    // A wide program's lookaheads: their captures, at the root floor and above it, and a
    // backreference to them.
    let wide_look = format!("{}(?=(y)|(z))(?:\\36|.)(?!(x))", "(x)?".repeat(34));
    // Many loops: bodies that cannot match empty, which take no mark, and bodies that can, each
    // with its mark (Decision 25).
    let loops = "(?:(?:a|b)*c)*(?:(?:d|e)+f)*(?:g|h)*".to_string();
    let marks = "(?:(?:a|b?)*c?)*(?:(?:d|)+f?)*(?:g|h|\\b)*".to_string();
    assert_eq!(Regex::new(&loops, "").unwrap().nmarks, 0);
    assert_eq!(Regex::new(&marks, "").unwrap().nmarks, 5);
    let long = format!("{}.{}", "ab".repeat(300), "cd".repeat(300));
    // Code far past AArch64's 32 KB test-and-branch reach, ending in the candidate loop's
    // outlines.
    let alternatives: Vec<String> = (0..250).map(|k| format!("x{k}y")).collect();
    let big = format!("(?:{})z", alternatives.join("|"));
    let texts = [
        "xxxyzq".to_string(),
        "xyzyzyzx".to_string(),
        format!("{}yzq", "x".repeat(40)),
        "abcabcdefdeffgh".to_string(),
        "aabbcdddeeffhgx".to_string(),
        format!("{}X{}", "ab".repeat(300), "cd".repeat(300)),
        format!("z{}X{}", "ab".repeat(300), "cd".repeat(299)),
        "ax17yzx249yz x3y".to_string(),
    ];
    for (pattern, flags) in [
        (wide.as_str(), ""),
        (wide.as_str(), "y"),
        (wide_look.as_str(), ""),
        (wide_look.as_str(), "g"),
        (loops.as_str(), ""),
        (loops.as_str(), "g"),
        (marks.as_str(), ""),
        (marks.as_str(), "g"),
        (long.as_str(), ""),
        (big.as_str(), ""),
    ] {
        for text in &texts {
            for two in [false, true] {
                let subject = Subject::of(text, two);
                let (native, _) = pair(pattern, flags, subject.width()).unwrap();
                compiled(&native, subject.width());
                if let Some(difference) = compare_from(pattern, flags, &subject, [0, 1, 3]) {
                    panic!("/{pattern}/{flags} over {text:?}: {difference:?}");
                }
            }
        }
    }
}

/// Every element of the two-byte range against classes with members on both sides of 256, as
/// the matcher tests them: the byte table below 256 and the ranges above.
#[test]
fn classes_test_every_two_byte_element_as_the_matcher_does() {
    for (pattern, flags) in [
        ("[a-z\\u0100-\\u017f\\u2000-\\u200a]", ""),
        ("[^\\u0100-\\uffff]", ""),
        ("\\s", ""),
        ("\\S", ""),
        ("[\\s\\d]", ""),
        ("[^\\w\\u3000-\\u303f]", ""),
        (
            "[\\u0101\\u0103\\u0105\\u0107\\u0109\\u010b\\u010d\\u010f\\u0111\\u0113]",
            "",
        ),
        (".", ""),
        ("[ab]", ""),
        ("[\\r\\n]", ""),
        ("[+, ]", ""),
        ("[Aa]", ""),
        // Closed over case: pairs that differ in one bit (a, é) or more (ÿ), groups of three and
        // four (µ, ǅ, Σ, ι), characters with no partner under the legacy Canonicalize (ß, ſ, the
        // Kelvin sign), classes and their complements.
        ("a", "i"),
        ("k", "i"),
        ("é", "i"),
        ("ÿ", "i"),
        ("µ", "i"),
        ("ß", "i"),
        ("\\u017f", "i"),
        ("\\u212a", "i"),
        ("\\u01c5", "i"),
        ("Σ", "i"),
        ("\\u0399", "i"),
        ("\\u1e9e", "i"),
        ("[a-z]", "i"),
        ("[^a-z]", "i"),
        ("[à-þ]", "i"),
        ("[\\w\\u0100-\\u017f]", "i"),
        ("[^\\u01c4]", "i"),
        ("\\W", "i"),
    ] {
        let (native, matcher) = pair(&format!("{pattern}x"), flags, Width::Two).unwrap();
        compiled(&native, Width::Two);
        let control = crate::RuntimeInterrupt::default();
        let mut units = Vec::new();
        for unit in 0..=0xffffu32 {
            units.push(unit as u16);
            units.push(b'x' as u16);
        }
        // Each unit followed by `x`: one execution finds the first match, and the starts walk
        // through the rest.
        let subject = Subject::Two(units);
        let mut start = 0;
        loop {
            let found = run(&native, &subject, start, &control);
            let reference = run(&matcher, &subject, start, &control);
            assert_eq!(found, reference, "/{pattern}x/{flags} from {start}");
            match found {
                Ok(Some(spans)) => start = spans[0].unwrap().1,
                _ => break,
            }
        }
    }
}

/// Deep backtracking runs on the native stack as it grows, past its initial capacity, and
/// unwinds through every record.
#[test]
fn deep_backtracking_grows_the_native_stack() {
    let text = format!("{}c", "ab".repeat(5000));
    let unmatched = format!("{}X", "ab".repeat(5000));
    for (pattern, flags, text) in [
        ("(?:a|ab)*c", "", &text),
        ("((a)|(ab))*c", "y", &text),
        ("(?:a|b)*?c", "", &text),
        ("(?:a|b)*$", "y", &unmatched),
        ("((a)|(b))*$", "y", &unmatched),
        ("(?:(a)|b)*?X", "", &unmatched),
    ] {
        for two in [false, true] {
            let subject = Subject::of(text, two);
            let (native, _) = pair(pattern, flags, subject.width()).unwrap();
            compiled(&native, subject.width());
            if let Some(difference) = compare_from(pattern, flags, &subject, [0, 1, 9_000]) {
                panic!("/{pattern}/{flags}: {difference:?}");
            }
        }
    }
}

/// Cancellation from another thread ends native backtracking with the matcher's error, through
/// the poll word.
#[test]
fn native_backtracking_answers_cancellation() {
    if !BACKEND {
        return;
    }
    let (native, _) = pair("(?:a|aa)*[cd]", "", Width::One).unwrap();
    assert!(compiled(&native, Width::One));
    let subject = Subject::One(vec![b'a'; 80]);
    let control = Arc::new(crate::RuntimeInterrupt::default());
    let remote = control.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        remote.cancel();
    });
    let started = std::time::Instant::now();
    let found = run(&native, &subject, 0, &control);
    canceller.join().unwrap();
    assert_eq!(
        found,
        Err(MatchError::Interrupted(crate::InterruptReason::Cancelled))
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

/// A deadline ends a native scan and native backtracking with the matcher's error, once the
/// watchdog sets the poll word's interrupt bit.
#[test]
fn native_scans_and_backtracking_answer_deadlines() {
    if !BACKEND {
        return;
    }
    // A loop's scan over a long run: polled between stretches.
    let (scan, _) = pair("[a-z]*[01]", "", Width::One).unwrap();
    assert!(compiled(&scan, Width::One));
    let long = Subject::One(vec![b'a'; 1 << 26]);
    // Backtracking with no end in sight.
    let (backtrack, _) = pair("(?:a|aa)*[cd]", "", Width::Two).unwrap();
    assert!(compiled(&backtrack, Width::Two));
    let deep = Subject::Two(vec![b'a' as u16; 80]);
    for (re, subject) in [(&scan, &long), (&backtrack, &deep)] {
        let control = crate::RuntimeInterrupt::default();
        control.set_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_millis(5),
        ));
        assert_eq!(
            run(re, subject, 0, &control),
            Err(MatchError::Interrupted(
                crate::InterruptReason::DeadlineExceeded
            )),
            "/{}/",
            re.source
        );
    }
    // A deadline already past, and a navigation request.
    let control = crate::RuntimeInterrupt::default();
    control.set_deadline(Some(
        std::time::Instant::now() - std::time::Duration::from_secs(1),
    ));
    assert_eq!(
        run(&backtrack, &deep, 0, &control),
        Err(MatchError::Interrupted(
            crate::InterruptReason::DeadlineExceeded
        ))
    );
    let control = crate::RuntimeInterrupt::default();
    control.request_user_navigation();
    assert_eq!(
        run(&scan, &long, 0, &control),
        Err(MatchError::Interrupted(
            crate::InterruptReason::UserNavigation
        ))
    );
}

/// The native stack is bounded by the matcher's backtracking state limit, and exhausting it is
/// the matcher's resource error, not a failure to match.
#[test]
fn native_stack_exhaustion_is_the_matcher_s_resource_error() {
    // Every iteration leaves the loop's exit to try, until `$` fails at the `X`.
    let text = format!("{}X", "ab".repeat(20_000));
    for two in [false, true] {
        let subject = Subject::of(&text, two);
        let (native, matcher) = pair("(?:a|b)*$", "y", subject.width()).unwrap();
        compiled(&native, subject.width());
        let control = crate::RuntimeInterrupt::default();
        super::super::set_backtrack_state_limit(1 << 10);
        let found = run(&native, &subject, 0, &control);
        let reference = run(&matcher, &subject, 0, &control);
        super::super::set_backtrack_state_limit(MAX_BACKTRACK_STATE);
        assert_eq!(found, Err(MatchError::ResourceExhausted));
        assert_eq!(reference, Err(MatchError::ResourceExhausted));
        assert_eq!(run(&native, &subject, 0, &control), Ok(None));
    }
}

/// A backreference's long compare answers a deadline, unit by unit and under `i` through the
/// helper, on both widths: a lookahead captures the whole subject at once (`.` under `s` takes
/// the rest of it in one step), so the time goes to the compare. Backtracking inside a lookahead
/// answers cancellation, and exhausting the backtracking state inside a lookahead or a counted
/// loop is the matcher's resource error.
#[test]
fn native_lookaheads_and_backreferences_answer_interruption_and_limits() {
    if !BACKEND {
        return;
    }
    for flags in ["s", "is"] {
        for two in [false, true] {
            let width = if two { Width::Two } else { Width::One };
            let (re, _) = pair("(?=(.*))\\1", flags, width).unwrap();
            assert!(compiled(&re, width));
            let subject = if two {
                Subject::Two(vec![b'a' as u16; 1 << 25])
            } else {
                Subject::One(vec![b'a'; 1 << 26])
            };
            let control = crate::RuntimeInterrupt::default();
            control.set_deadline(Some(
                std::time::Instant::now() + std::time::Duration::from_millis(5),
            ));
            assert_eq!(
                run(&re, &subject, 0, &control),
                Err(MatchError::Interrupted(
                    crate::InterruptReason::DeadlineExceeded
                )),
                "/(?=(.*))\\1/{flags} over {width:?}"
            );
            // Without a deadline the compare runs to its end and matches.
            let small = Subject::of(&"a".repeat(40_000), two);
            let found = run(&re, &small, 0, &crate::RuntimeInterrupt::default());
            assert_eq!(found, Ok(Some(vec![Some((0, 40_000)), Some((0, 40_000))])));
        }
    }
    // Backtracking with no end in sight, inside a lookahead's body.
    let (looking, _) = pair("(?=(?:a|aa)*[cd])", "", Width::One).unwrap();
    assert!(compiled(&looking, Width::One));
    let control = Arc::new(crate::RuntimeInterrupt::default());
    let remote = control.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        remote.cancel();
    });
    let found = run(&looking, &Subject::One(vec![b'a'; 80]), 0, &control);
    canceller.join().unwrap();
    assert_eq!(
        found,
        Err(MatchError::Interrupted(crate::InterruptReason::Cancelled))
    );
    // Every iteration leaves the loop's exit to try, until `$` fails at the `X`.
    let text = format!("{}X", "ab".repeat(20_000));
    for (pattern, flags) in [("(?=(?:a|b)*$)", "y"), ("(?:a|b){0,100000}$", "y")] {
        for two in [false, true] {
            let subject = Subject::of(&text, two);
            let (native, matcher) = pair(pattern, flags, subject.width()).unwrap();
            assert!(compiled(&native, subject.width()));
            let control = crate::RuntimeInterrupt::default();
            super::super::set_backtrack_state_limit(1 << 10);
            let found = run(&native, &subject, 0, &control);
            let reference = run(&matcher, &subject, 0, &control);
            super::super::set_backtrack_state_limit(MAX_BACKTRACK_STATE);
            assert_eq!(
                found,
                Err(MatchError::ResourceExhausted),
                "/{pattern}/{flags}"
            );
            assert_eq!(
                reference,
                Err(MatchError::ResourceExhausted),
                "/{pattern}/{flags}"
            );
            assert_eq!(
                run(&native, &subject, 0, &control),
                Ok(None),
                "/{pattern}/{flags}"
            );
        }
    }
}

/// Unicode mode's loops of decoded elements and a lookbehind's backward reading answer
/// interruption and limits as the forward code does: a deadline ends scans of elements past the
/// BMP, forward and backward, counted or not, a backward scan of units, and a long backward
/// backreference compare (in line and under `i` through the helper, over units and over pairs);
/// cancellation ends backtracking inside a lookbehind; and exhausting the backtracking state
/// inside a lookbehind, in a loop or a counted loop, is the matcher's resource error.
#[test]
fn unicode_and_lookbehind_code_answers_interruption_and_limits() {
    if !BACKEND {
        return;
    }
    let deadline = || {
        let control = crate::RuntimeInterrupt::default();
        control.set_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_millis(5),
        ));
        control
    };
    let late = || -> Found {
        Err(MatchError::Interrupted(
            crate::InterruptReason::DeadlineExceeded,
        ))
    };
    let units = Subject::One(vec![b'a'; 1 << 26]);
    let pairs = Subject::Two([0xd83d_u16, 0xde00].repeat(1 << 24));
    // Scans: elements past the BMP, forward and backward (from the subject's end, sticky), and
    // units backward.
    for (pattern, flags, subject) in [
        ("[^x]*[01]", "u", &pairs),
        ("[^x]{0,40000000}[01]", "u", &pairs),
        ("(?<=[01][^x]*)", "uy", &pairs),
        ("(?<=[01][^x]{0,40000000})", "uy", &pairs),
        ("(?<=[01][a-z]*)", "y", &units),
    ] {
        let (re, _) = pair(pattern, flags, subject.width()).unwrap();
        assert!(compiled(&re, subject.width()), "/{pattern}/{flags}");
        let start = if flags.contains('y') {
            subject.len()
        } else {
            0
        };
        assert_eq!(
            run(&re, subject, start, &deadline()),
            late(),
            "/{pattern}/{flags}"
        );
    }
    // A backward backreference's long compare: the lookahead takes the subject's second half at
    // once (`.` under `s`), and the lookbehind compares it with the first half.
    let two_units = Subject::Two(vec![b'a' as u16; 1 << 25]);
    for (flags, subject, small) in [
        ("sy", &units, Subject::of(&"a".repeat(40_000), false)),
        ("isy", &units, Subject::of(&"a".repeat(40_000), false)),
        ("sy", &two_units, Subject::of(&"a".repeat(40_000), true)),
        ("isy", &two_units, Subject::of(&"a".repeat(40_000), true)),
        (
            "suy",
            &pairs,
            Subject::of(&"\u{1f600}".repeat(20_000), true),
        ),
        (
            "isuy",
            &pairs,
            Subject::of(&"\u{1f600}".repeat(20_000), true),
        ),
    ] {
        let width = subject.width();
        let (re, _) = pair("(?=(.*))(?<=\\1)", flags, width).unwrap();
        assert!(compiled(&re, width), "/(?=(.*))(?<=\\1)/{flags}");
        assert_eq!(
            run(&re, subject, subject.len() / 2, &deadline()),
            late(),
            "/(?=(.*))(?<=\\1)/{flags} over {width:?}"
        );
        // Without a deadline the compare runs to its end and matches.
        let found = run(&re, &small, 20_000, &crate::RuntimeInterrupt::default());
        assert_eq!(
            found,
            Ok(Some(vec![Some((20_000, 20_000)), Some((20_000, 40_000))])),
            "/(?=(.*))(?<=\\1)/{flags} over {width:?}"
        );
    }
    // Backtracking with no end in sight, inside a lookbehind's body.
    for (flags, subject) in [
        ("y", Subject::One(vec![b'a'; 80])),
        ("uy", Subject::Two(vec![b'a' as u16; 80])),
    ] {
        let (looking, _) = pair("(?<=[cd](?:a|aa)*)", flags, subject.width()).unwrap();
        assert!(
            compiled(&looking, subject.width()),
            "/(?<=[cd](?:a|aa)*)/{flags}"
        );
        let control = Arc::new(crate::RuntimeInterrupt::default());
        let remote = control.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            remote.cancel();
        });
        let found = run(&looking, &subject, 80, &control);
        canceller.join().unwrap();
        assert_eq!(
            found,
            Err(MatchError::Interrupted(crate::InterruptReason::Cancelled)),
            "/(?<=[cd](?:a|aa)*)/{flags}"
        );
    }
    // Reading backward, every iteration leaves the loop's exit to try, until `^` fails at the
    // `X`.
    let text = format!("X{}", "ab".repeat(20_000));
    for (pattern, flags) in [
        ("(?<=^(?:a|b)*)", "y"),
        ("(?<=^(?:a|b){0,100000})", "y"),
        ("(?<=^(?:a|b)*)", "uy"),
    ] {
        for two in [false, true] {
            let subject = Subject::of(&text, two);
            let end = subject.len();
            let (native, matcher) = pair(pattern, flags, subject.width()).unwrap();
            assert!(compiled(&native, subject.width()), "/{pattern}/{flags}");
            let control = crate::RuntimeInterrupt::default();
            super::super::set_backtrack_state_limit(1 << 10);
            let found = run(&native, &subject, end, &control);
            let reference = run(&matcher, &subject, end, &control);
            super::super::set_backtrack_state_limit(MAX_BACKTRACK_STATE);
            assert_eq!(
                found,
                Err(MatchError::ResourceExhausted),
                "/{pattern}/{flags}"
            );
            assert_eq!(
                reference,
                Err(MatchError::ResourceExhausted),
                "/{pattern}/{flags}"
            );
            assert_eq!(
                run(&native, &subject, end, &control),
                Ok(None),
                "/{pattern}/{flags}"
            );
        }
    }
}

/// A pattern compiles for a width at its threshold's execution over a subject of that width,
/// or at once over a long subject (V8's rule), each width on its first use.
#[test]
fn hot_patterns_compile_each_width_on_its_first_use() {
    let control = crate::RuntimeInterrupt::default();
    let mut re = Regex::new("a(b|c)d", "").unwrap();
    re.tier_up_at = 64;
    let ascii = crate::lstr::LStr::from("xxacdyy");
    let wide = crate::lstr::LStr::from("xxacdé");
    for _ in 0..63 {
        assert!(re.exec_polled(&ascii, 0, &control).unwrap().is_some());
    }
    let (ticks, _, widths) = re.tier_feedback();
    assert_eq!(ticks, 63);
    assert_eq!(widths, [(false, false), (false, false)]);
    re.exec_polled(&ascii, 0, &control).unwrap();
    let (_, _, widths) = re.tier_feedback();
    assert_eq!(widths, [(true, BACKEND), (false, false)]);
    let found = re.exec_polled(&wide, 0, &control).unwrap().unwrap();
    assert_eq!(found.to_vec(), vec![Some((2, 5)), Some((3, 4))]);
    let (_, _, widths) = re.tier_feedback();
    assert_eq!(widths, [(true, BACKEND), (true, BACKEND)]);

    let mut long = Regex::new("a(b|c)d", "").unwrap();
    long.tier_up_at = 64;
    let text = crate::lstr::LStr::from(format!("{}acd", "x".repeat(1000)));
    long.exec_polled(&text, 0, &control).unwrap();
    assert_eq!(long.tier_feedback().2[0], (true, BACKEND));

    // Patterns the tier does not take (`v`-mode class strings) try once per width and stay on
    // the matcher.
    for (pattern, flags) in [("a[\\q{bc|d}]", "v"), ("(?=[\\q{bc}])", "v")] {
        let mut declined = Regex::new(pattern, flags).unwrap();
        declined.tier_up_at = 1;
        declined.exec_polled(&ascii, 0, &control).unwrap();
        declined.exec_polled(&wide, 0, &control).unwrap();
        let (_, _, widths) = declined.tier_feedback();
        assert_eq!(widths, [(true, false), (true, false)], "/{pattern}/{flags}");
    }
    // A case-insensitive literal runs on one-byte subjects by its folding search, and compiles,
    // closed over case, for two-byte ones.
    let mut literal = Regex::new("ab", "i").unwrap();
    literal.tier_up_at = 1;
    literal.exec_polled(&ascii, 0, &control).unwrap();
    literal.exec_polled(&wide, 0, &control).unwrap();
    assert_eq!(literal.tier_feedback().2, [(true, false), (true, BACKEND)]);
    // Backreferences, lookarounds, counted loops, modifier groups and Unicode mode compile.
    for (pattern, flags) in [
        ("a(b)\\1", ""),
        ("a(?=b)", ""),
        ("(?:a|bc){1001}", ""),
        ("a(?i:b)", ""),
        ("a(?<=b)", ""),
        ("a(?=(?<!b))", ""),
        ("a.", "u"),
        ("[a-z]+", "v"),
    ] {
        let mut taken = Regex::new(pattern, flags).unwrap();
        taken.tier_up_at = 1;
        taken.exec_polled(&ascii, 0, &control).unwrap();
        taken.exec_polled(&wide, 0, &control).unwrap();
        let (_, _, widths) = taken.tier_feedback();
        assert_eq!(
            widths,
            [(true, BACKEND), (true, BACKEND)],
            "/{pattern}/{flags}"
        );
    }
}

/// The perf map (`TRUST_JS_PERF_MAP`) lists a pattern's native code under its source and flags.
#[test]
fn perf_map_names_native_regexp_programs_by_source_and_flags() {
    if !BACKEND {
        return;
    }
    let capture = crate::jit::profiler::PerfMapCapture::start("regexp");
    let (re, _) = pair("a.c", "g", Width::One).unwrap();
    let code = re.native.get(Width::One).expect("a.c compiles");
    let lines = capture.lines();
    let line = (
        code.executable.as_ptr() as usize,
        code.executable.len(),
        "RegExp:/a.c/g".to_string(),
    );
    assert!(lines.contains(&line), "{line:x?} in {lines:x?}");
}
