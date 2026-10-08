//! IANA time-zone lookups over the generated [`crate::tzdata`] offset tables.

use crate::tzdata::{Zone, LINKS, ZONES};

/// The canonical registry name for a case-insensitive IANA zone id (`None` if unknown).
pub fn canonicalize(name: &str) -> Option<&'static str> {
    // The UTC aliases canonicalize to "UTC"; the GMT/Greenwich family stays as "Etc/GMT" (resolved
    // below through the link table), and Etc/GMT+N remain distinct fixed-offset zones.
    let lc = name.to_ascii_lowercase();
    if matches!(
        lc.as_str(),
        "utc"
            | "etc/utc"
            | "etc/uct"
            | "uct"
            | "universal"
            | "etc/universal"
            | "zulu"
            | "etc/zulu"
            | "gmt"
            | "etc/gmt"
            | "etc/gmt0"
            | "gmt0"
            | "gmt+0"
            | "gmt-0"
            | "etc/gmt+0"
            | "etc/gmt-0"
            | "etc/greenwich"
            | "greenwich"
    ) {
        return Some("UTC");
    }
    if let Some(z) = ZONES.iter().find(|z| z.name.eq_ignore_ascii_case(name)) {
        return Some(z.name);
    }
    LINKS
        .iter()
        .find(|(a, _)| a.eq_ignore_ascii_case(name))
        .map(|(_, canon)| *canon)
}

/// The registry name for a case-insensitive zone id, PRESERVING aliases (an input of "Asia/Calcutta"
/// stays "Asia/Calcutta", not the canonical "Asia/Kolkata"). Temporal keeps the identifier as given;
/// only `equals`/`compare` canonicalize. `None` if the id is unknown.
pub fn registry_name(name: &str) -> Option<&'static str> {
    if name.eq_ignore_ascii_case("UTC") {
        return Some("UTC");
    }
    if let Some(z) = ZONES.iter().find(|z| z.name.eq_ignore_ascii_case(name)) {
        return Some(z.name);
    }
    LINKS
        .iter()
        .find(|(a, _)| a.eq_ignore_ascii_case(name))
        .map(|(a, _)| *a)
}

/// Every canonical IANA zone name in the registry (for `Intl.supportedValuesOf("timeZone")`).
#[cfg(feature = "intl")]
pub fn canonical_zone_names() -> Vec<&'static str> {
    ZONES.iter().map(|z| z.name).collect()
}

/// The primary zones in common use in an ISO 3166-1 region (IANA zone.tab), sorted; empty for
/// a region zone.tab does not list, such as a UN M.49 area code.
#[cfg(feature = "intl")]
pub fn region_zones(region: &str) -> &'static [&'static str] {
    let regions = crate::tzdata::REGION_ZONES;
    regions
        .binary_search_by(|(candidate, _)| (*candidate).cmp(region))
        .map_or(&[], |index| regions[index].1)
}

fn zone(name: &str) -> Option<&'static Zone> {
    let canon = canonicalize(name)?;
    ZONES.iter().find(|z| z.name == canon)
}

/// The UTC offset (seconds) in effect at `epoch_sec` for a named zone.
///
/// A zone's table lists its transitions through its last explicitly recorded one (2037 for
/// zones with ongoing daylight saving time). RFC 9636 section 3.3: local time on or after the
/// last transition is specified by the zone's TZif footer TZ string ([`crate::tzrules`]).
pub fn offset_at(name: &str, epoch_sec: i64) -> Option<i32> {
    let z = zone(name)?;
    let idx = z.transitions.partition_point(|&(t, _)| t <= epoch_sec);
    if idx == z.transitions.len() {
        if let Some(rule) = rule(z.name) {
            return Some(rule.offset_at(epoch_sec));
        }
    }
    Some(if idx == 0 {
        z.initial
    } else {
        z.transitions[idx - 1].1
    })
}

/// Whether daylight saving time is in effect at `epoch_sec` in a named zone: the TZif
/// `tt_isdst` of the local time type then (RFC 9636 section 3.2), and past the zone's last
/// transition, whether its footer rule is in daylight time (RFC 9636 section 3.3).
#[cfg(feature = "intl")]
pub fn is_dst(name: &str, epoch_sec: i64) -> Option<bool> {
    let z = zone(name)?;
    let idx = z.transitions.partition_point(|&(t, _)| t <= epoch_sec);
    let changes = crate::tzdata::DST_CHANGES
        .binary_search_by(|(zone, _)| (*zone).cmp(z.name))
        .map_or(&[][..], |index| crate::tzdata::DST_CHANGES[index].1);
    let last_transition = z
        .transitions
        .last()
        .map(|&(t, _)| t)
        .max(changes.last().map(|&(t, _)| t));
    if last_transition.is_none_or(|last| epoch_sec >= last) {
        if let Some(rule) = rule(z.name) {
            return Some(rule.state_at(epoch_sec).1);
        }
    }
    let mut since = i64::MIN;
    let mut dst = false;
    if idx > 0 {
        let k = idx - 1;
        since = z.transitions[k].0;
        dst = z
            .dst
            .get(k / 64)
            .is_some_and(|word| word >> (k % 64) & 1 == 1);
    }
    let j = changes.partition_point(|&(t, _)| t <= epoch_sec);
    if j > 0 && changes[j - 1].0 > since {
        dst = changes[j - 1].1;
    }
    Some(dst)
}

/// The epoch-second of the next (`forward`) or previous offset transition strictly after/before
/// `epoch_sec`, or `None` when the zone has no further transition in that direction. Past the
/// table, transitions come from the zone's footer rule, so a zone that keeps observing daylight
/// saving time always has a next transition.
pub fn next_transition(name: &str, epoch_sec: i64, forward: bool) -> Option<i64> {
    let z = zone(name)?;
    let ts = z.transitions;
    let last = ts.last().map(|&(t, _)| t);
    if forward {
        if let Some(&(t, _)) = ts.iter().find(|&&(t, _)| t > epoch_sec) {
            return Some(t);
        }
        // The rule governs only instants after the last tabulated transition.
        let from = last.map_or(epoch_sec, |last| last.max(epoch_sec));
        rule(z.name)?.next_change(from, true)
    } else {
        if last.is_none_or(|last| epoch_sec > last) {
            if let Some(rule) = rule(z.name) {
                if let Some(t) = rule.next_change(epoch_sec, false) {
                    if last.is_none_or(|last| t > last) {
                        return Some(t);
                    }
                }
            }
        }
        ts.iter()
            .rev()
            .find(|&&(t, _)| t < epoch_sec)
            .map(|&(t, _)| t)
    }
}

/// The zone's TZif footer rule, parsed once per process.
fn rule(canonical: &str) -> Option<PosixTz> {
    static PARSED: std::sync::OnceLock<Box<[Option<PosixTz>]>> = std::sync::OnceLock::new();
    let rules = crate::tzrules::RULES;
    let index = rules
        .binary_search_by(|(zone, _)| (*zone).cmp(canonical))
        .ok()?;
    PARSED.get_or_init(|| rules.iter().map(|(_, text)| PosixTz::parse(text)).collect())[index]
}

/// A POSIX TZ string (POSIX.1-2017 Base Definitions section 8.3, "TZ"), as used in TZif footers
/// with the RFC 9636 section 3.3.2 extension that lets rule times range over -167..=167 hours.
/// Offsets are stored as seconds east of UTC (POSIX writes them west of UTC).
#[derive(Debug, Clone, Copy, PartialEq)]
struct PosixTz {
    standard: i32,
    daylight: Option<PosixDst>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PosixDst {
    offset: i32,
    start: (PosixDate, i32),
    end: (PosixDate, i32),
}

/// A rule's `date` field.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PosixDate {
    /// `Jn`: day n (1..=365) of the year, never counting February 29.
    Julian(i64),
    /// `n`: zero-based day n (0..=365) of the year, counting February 29.
    Zero(i64),
    /// `Mm.w.d`: day d (0 = Sunday) of week w (1..=5, 5 = the last) of month m.
    Month(i64, i64, i64),
}

impl PosixTz {
    fn parse(text: &str) -> Option<PosixTz> {
        let mut rest = text.as_bytes();
        skip_name(&mut rest)?;
        let standard = -parse_hms(&mut rest)?;
        if rest.is_empty() {
            return Some(PosixTz {
                standard,
                daylight: None,
            });
        }
        skip_name(&mut rest)?;
        // The daylight offset defaults to one hour ahead of standard time.
        let offset = if rest.first() == Some(&b',') || rest.is_empty() {
            standard + 3600
        } else {
            -parse_hms(&mut rest)?
        };
        // TZif footers always spell out the rule when daylight time is observed.
        let start = parse_rule_date(&mut rest)?;
        let end = parse_rule_date(&mut rest)?;
        rest.is_empty().then_some(PosixTz {
            standard,
            daylight: Some(PosixDst { offset, start, end }),
        })
    }

    /// The UTC offset at `epoch_sec`.
    fn offset_at(&self, epoch_sec: i64) -> i32 {
        self.state_at(epoch_sec).0
    }

    /// The UTC offset at `epoch_sec` and whether it is daylight saving time.
    fn state_at(&self, epoch_sec: i64) -> (i32, bool) {
        let Some(dst) = self.daylight else {
            return (self.standard, false);
        };
        let year = civil_year(epoch_sec);
        // The latest event at or before the instant among the surrounding years decides; an event
        // of the previous year always precedes it, so the initial value is never the answer.
        let mut latest = (i64::MIN, 0u8, self.standard);
        for year in year - 1..=year + 1 {
            for event in self.year_events(dst, year) {
                if event.0 <= epoch_sec && (event.0, event.1) > (latest.0, latest.1) {
                    latest = event;
                }
            }
        }
        // `order` 1 is a daylight-time start.
        (latest.2, latest.1 == 1)
    }

    /// The first instant strictly after (`forward`) or the last strictly before `epoch_sec` at
    /// which the offset changes. Coinciding events (RFC 9636 section 3.3.1 all-year daylight
    /// time ends and restarts at the same instant) are not a change.
    fn next_change(&self, epoch_sec: i64, forward: bool) -> Option<i64> {
        let dst = self.daylight?;
        if dst.offset == self.standard {
            return None;
        }
        let year = civil_year(epoch_sec);
        let mut instants: Vec<i64> = self
            .events(dst, year - 2, year + 2)
            .into_iter()
            .map(|(instant, _)| instant)
            .filter(|&instant| {
                if forward {
                    instant > epoch_sec
                } else {
                    instant < epoch_sec
                }
            })
            .collect();
        instants.dedup();
        if !forward {
            instants.reverse();
        }
        instants
            .into_iter()
            .find(|&instant| self.offset_at(instant - 1) != self.offset_at(instant))
    }

    /// The rule's (instant, new offset) events for the years `from..=to`, in time order; an end
    /// sorts before a start at the same instant so all-year daylight time stays in effect.
    fn events(&self, dst: PosixDst, from: i64, to: i64) -> Vec<(i64, i32)> {
        let mut events: Vec<_> = (from..=to)
            .flat_map(|year| self.year_events(dst, year))
            .collect();
        events.sort_unstable();
        events
            .into_iter()
            .map(|(instant, _, offset)| (instant, offset))
            .collect()
    }

    /// A year's daylight-time start and end as (instant, order, new offset); `order` places an
    /// end before a start at the same instant.
    fn year_events(&self, dst: PosixDst, year: i64) -> [(i64, u8, i32); 2] {
        // The start time is given in standard local time, the end time in daylight time.
        let start = rule_day(dst.start.0, year) * 86_400 + i64::from(dst.start.1)
            - i64::from(self.standard);
        let end = rule_day(dst.end.0, year) * 86_400 + i64::from(dst.end.1) - i64::from(dst.offset);
        [(start, 1, dst.offset), (end, 0, self.standard)]
    }
}

/// Skip a `std`/`dst` designation: alphabetic, or quoted in `<...>`.
fn skip_name(rest: &mut &[u8]) -> Option<()> {
    if rest.first() == Some(&b'<') {
        let close = rest.iter().position(|&b| b == b'>')?;
        *rest = &rest[close + 1..];
        return Some(());
    }
    let len = rest.iter().take_while(|b| b.is_ascii_alphabetic()).count();
    (len >= 3).then(|| *rest = &rest[len..])
}

/// `[+-]hh[:mm[:ss]]` in seconds.
fn parse_hms(rest: &mut &[u8]) -> Option<i32> {
    let mut sign = 1;
    if let Some(&b) = rest.first() {
        if b == b'+' || b == b'-' {
            sign = if b == b'-' { -1 } else { 1 };
            *rest = &rest[1..];
        }
    }
    let mut seconds = 0i32;
    for (index, scale) in [3600, 60, 1].into_iter().enumerate() {
        if index > 0 {
            if rest.first() != Some(&b':') {
                break;
            }
            *rest = &rest[1..];
        }
        let len = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        if len == 0 || len > 3 {
            return None;
        }
        let value: i32 = std::str::from_utf8(&rest[..len]).ok()?.parse().ok()?;
        *rest = &rest[len..];
        seconds += value * scale;
    }
    Some(sign * seconds)
}

/// `,date[/time]`; the time defaults to 02:00:00.
fn parse_rule_date(rest: &mut &[u8]) -> Option<(PosixDate, i32)> {
    if rest.first() != Some(&b',') {
        return None;
    }
    *rest = &rest[1..];
    let number = |rest: &mut &[u8]| -> Option<i64> {
        let len = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        let value = std::str::from_utf8(&rest[..len]).ok()?.parse().ok()?;
        *rest = &rest[len..];
        Some(value)
    };
    let date = match rest.first()? {
        b'J' => {
            *rest = &rest[1..];
            PosixDate::Julian(number(rest)?)
        }
        b'M' => {
            *rest = &rest[1..];
            let month = number(rest)?;
            (rest.first() == Some(&b'.')).then(|| *rest = &rest[1..])?;
            let week = number(rest)?;
            (rest.first() == Some(&b'.')).then(|| *rest = &rest[1..])?;
            let day = number(rest)?;
            if !(1..=12).contains(&month) || !(1..=5).contains(&week) || !(0..=6).contains(&day) {
                return None;
            }
            PosixDate::Month(month, week, day)
        }
        _ => PosixDate::Zero(number(rest)?),
    };
    let time = if rest.first() == Some(&b'/') {
        *rest = &rest[1..];
        parse_hms(rest)?
    } else {
        7200
    };
    Some((date, time))
}

/// Days from 1970-01-01 to the day a rule date names in `year`.
fn rule_day(date: PosixDate, year: i64) -> i64 {
    let january_1 = days_from_epoch(year, 1, 1);
    match date {
        PosixDate::Julian(n) => {
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
            january_1 + n - 1 + i64::from(leap && n >= 60)
        }
        PosixDate::Zero(n) => january_1 + n,
        PosixDate::Month(month, week, weekday) => {
            let first = days_from_epoch(year, month, 1);
            let following = if month == 12 {
                days_from_epoch(year + 1, 1, 1)
            } else {
                days_from_epoch(year, month + 1, 1)
            };
            // 1970-01-01 was a Thursday (weekday 4).
            let mut day = first + (weekday - (first + 4)).rem_euclid(7) + (week - 1) * 7;
            while day >= following {
                day -= 7;
            }
            day
        }
    }
}

/// The proleptic Gregorian year containing `epoch_sec` (UTC).
fn civil_year(epoch_sec: i64) -> i64 {
    let days = epoch_sec.div_euclid(86_400);
    // civil_from_days (H. Hinnant), year only.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    yoe + era * 400 + i64::from(mp >= 10)
}

fn days_from_epoch(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(y: i64, m: i64, d: i64, h: i64) -> i64 {
        days_from_epoch(y, m, d) * 86_400 + h * 3600
    }

    #[test]
    fn posix_tz_strings_parse_per_posix_8_3_and_rfc_9636() {
        let berlin = PosixTz::parse("CET-1CEST,M3.5.0,M10.5.0/3").unwrap();
        assert_eq!(berlin.standard, 3600);
        let dst = berlin.daylight.unwrap();
        assert_eq!(
            dst.offset, 7200,
            "the daylight offset defaults to one hour ahead"
        );
        assert_eq!(dst.start, (PosixDate::Month(3, 5, 0), 7200));
        assert_eq!(dst.end, (PosixDate::Month(10, 5, 0), 10800));
        // Quoted designations, explicit daylight offsets and RFC 9636 3.3.2 hours past 24.
        let santiago = PosixTz::parse("<-04>4<-03>,M9.1.6/24,M4.1.6/24").unwrap();
        assert_eq!(santiago.standard, -14400);
        assert_eq!(santiago.daylight.unwrap().offset, -10800);
        let gaza = PosixTz::parse("EET-2EEST,M3.4.4/50,M10.4.4/50").unwrap();
        assert_eq!(gaza.daylight.unwrap().start.1, 50 * 3600);
        let negative = PosixTz::parse("<-03>3<-02>,M3.5.0/-2,M10.5.0/-1").unwrap();
        assert_eq!(negative.daylight.unwrap().start.1, -7200);
        assert_eq!(PosixTz::parse("<+0530>-5:30").unwrap().standard, 19800);
        assert_eq!(PosixTz::parse("UTC0").unwrap().daylight, None);
        assert_eq!(
            PosixTz::parse("J60").map(|tz| tz.standard),
            None,
            "a designation needs three letters"
        );
        assert_eq!(PosixTz::parse("EST5EDT,M13.1.0,M11.1.0"), None);
    }

    #[test]
    fn julian_and_zero_based_days_differ_only_in_leap_years() {
        assert_eq!(
            rule_day(PosixDate::Julian(60), 2024),
            days_from_epoch(2024, 3, 1)
        );
        assert_eq!(
            rule_day(PosixDate::Zero(59), 2024),
            days_from_epoch(2024, 2, 29)
        );
        assert_eq!(
            rule_day(PosixDate::Julian(60), 2025),
            days_from_epoch(2025, 3, 1)
        );
        assert_eq!(
            rule_day(PosixDate::Zero(59), 2025),
            days_from_epoch(2025, 3, 1)
        );
        // Week 5 is the month's last such weekday: the last Sunday of March 2038 is the 28th.
        assert_eq!(
            rule_day(PosixDate::Month(3, 5, 0), 2038),
            days_from_epoch(2038, 3, 28)
        );
    }

    #[test]
    fn all_year_daylight_time_has_no_transitions() {
        // RFC 9636 section 3.3.1's example: perpetually four hours west of UT.
        let tz = PosixTz::parse("XXX3EDT4,0/0,J365/23").unwrap();
        for instant in [
            utc(2040, 1, 1, 0),
            utc(2040, 6, 1, 0),
            utc(2040, 12, 31, 23),
        ] {
            assert_eq!(tz.offset_at(instant), -14400);
        }
        assert_eq!(tz.next_change(utc(2040, 6, 1, 0), true), None);
        assert_eq!(tz.next_change(utc(2040, 6, 1, 0), false), None);
    }

    #[test]
    fn footer_rules_continue_daylight_saving_time_after_the_tables() {
        assert_eq!(
            offset_at("America/New_York", utc(2050, 7, 1, 0)),
            Some(-14400)
        );
        assert_eq!(
            offset_at("America/New_York", utc(2050, 1, 1, 0)),
            Some(-18000)
        );
        assert_eq!(
            offset_at("Australia/Sydney", utc(2045, 1, 15, 0)),
            Some(39600)
        );
        assert_eq!(
            offset_at("Australia/Sydney", utc(2045, 7, 15, 0)),
            Some(36000)
        );
        assert_eq!(offset_at("Asia/Tokyo", utc(2100, 7, 1, 0)), Some(32400));
        // The last tabulated Berlin transition is 2037-10-25; the rule supplies the next one.
        assert_eq!(
            next_transition("Europe/Berlin", utc(2037, 12, 1, 0), true),
            Some(utc(2038, 3, 28, 1))
        );
        assert_eq!(
            next_transition("Europe/Berlin", utc(2040, 1, 1, 0), false),
            Some(utc(2039, 10, 30, 1))
        );
        // A backward search that crosses back into the table uses the table.
        assert_eq!(
            next_transition("Europe/Berlin", utc(2038, 1, 1, 0), false),
            Some(utc(2037, 10, 25, 1))
        );
        // Santiago's rule changes at 24:00 local time on the first Saturday of the month.
        assert_eq!(
            next_transition("America/Santiago", utc(2045, 6, 1, 4), true),
            Some(utc(2045, 9, 3, 4))
        );
        assert_eq!(
            next_transition("America/Santiago", utc(2045, 10, 1, 4), true),
            Some(utc(2046, 4, 8, 3))
        );
        // A zone without daylight time has no transition after its table.
        assert_eq!(
            next_transition("Asia/Tokyo", utc(2040, 1, 1, 0), true),
            None
        );
        assert_eq!(next_transition("UTC", utc(2040, 1, 1, 0), true), None);
    }

    #[cfg(feature = "intl")]
    #[test]
    fn daylight_saving_time_follows_tt_isdst_and_the_footer_rule() {
        const JAN_2024: i64 = 1_704_369_600;
        const JUL_2024: i64 = 1_720_094_400;
        assert_eq!(is_dst("Europe/London", JAN_2024), Some(false));
        assert_eq!(is_dst("Europe/London", JUL_2024), Some(true));
        assert_eq!(is_dst("America/New_York", JUL_2024), Some(true));
        assert_eq!(is_dst("America/Phoenix", JUL_2024), Some(false));
        assert_eq!(is_dst("Australia/Sydney", JAN_2024), Some(true));
        assert_eq!(is_dst("Australia/Sydney", JUL_2024), Some(false));
        assert_eq!(is_dst("UTC", JUL_2024), Some(false));
        // Past the tables, from the footer rule (2100-07-01 and 2100-01-01).
        assert_eq!(is_dst("America/New_York", 4_118_083_200), Some(true));
        assert_eq!(is_dst("America/New_York", 4_102_444_800), Some(false));
        // British Standard Time (1968-10-27 to 1971-10-31) kept +01:00 as standard time; the
        // change from British Summer Time altered only tt_isdst.
        assert_eq!(offset_at("Europe/London", 0), Some(3600));
        assert_eq!(is_dst("Europe/London", 0), Some(false));
        assert_eq!(is_dst("Europe/London", -40_000_000), Some(true));
        // The TZ database's main format gives Ireland negative daylight saving time in winter.
        assert_eq!(is_dst("Europe/Dublin", JAN_2024), Some(true));
        assert_eq!(is_dst("Europe/Dublin", JUL_2024), Some(false));
        assert_eq!(is_dst("Not/AZone", JAN_2024), None);
    }

    #[test]
    fn every_footer_rule_agrees_with_its_tabulated_transitions() {
        // RFC 9636 section 3.3 requires the footer to be consistent with the last transition;
        // the generator verified it, and this keeps the tables and rules from drifting apart.
        for zone in crate::tzdata::ZONES {
            let rule = rule(zone.name).unwrap_or_else(|| panic!("{} has no rule", zone.name));
            if let Some(&(last, offset)) = zone.transitions.last() {
                assert_eq!(rule.offset_at(last), offset, "{}", zone.name);
            }
            let last = zone.transitions.last().map_or(0, |&(t, _)| t);
            // Zones with explicitly predicted transitions beyond 2037 (Morocco, Palestine) follow
            // their rule only after those; the others tabulate their rule's own changes.
            if rule.daylight.is_none() || last >= utc(2038, 1, 1, 0) {
                continue;
            }
            let mut instant = utc(2030, 1, 1, 0);
            while let Some(next) = rule.next_change(instant, true) {
                if next > last {
                    break;
                }
                assert!(
                    zone.transitions.iter().any(|&(t, _)| t == next),
                    "{}: rule transition {next} is not tabulated",
                    zone.name
                );
                instant = next;
            }
        }
    }
}
