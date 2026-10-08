//! Time zone display names: the ILD String FormatDateTimePattern (ECMA-402
//! #sec-formatdatetimepattern) gives a `timeZoneName` field in the form its [[TimeZone]]
//! format requests, built from CLDR (`crate::cldr_tz_names`) by UTS #35 Part 4 "Using Time
//! Zone Names".
//!
//! The styles map to UTS #35 formats: `short`/`long` are the specific non-location formats
//! (z, zzzz), `shortOffset`/`longOffset` the localized GMT formats (O, OOOO), and
//! `shortGeneric`/`longGeneric` the generic non-location formats (v, vvvv), which fall back to
//! the generic location format (VVVV). Each falls back to the localized GMT format.
//!
//! Where UTS #35 leaves room, the choices follow ICU, the CLDR reference implementation that
//! other ECMAScript engines use: a generic name is qualified by a location (the generic partial
//! location format) only when the zone's offset differs from that of the metazone's preferred
//! zone for the locale's region, and Type Fallback's 184-day rule chooses a standard name for a
//! zone that observes no daylight saving time around the instant.

use crate::cldr_tz_names::{self as data, text, Id, LocaleNames, MetazoneUse};
use std::collections::HashMap;
use std::sync::OnceLock;

/// The UTS #35 Type Fallback interval around the formatted instant (184 days), in seconds.
const DST_CHECK_RANGE: i64 = 184 * 86_400;

/// Name kinds, indexing [`data::Names`] after the width offset (long 0, short 3).
const GENERIC: usize = 0;
const STANDARD: usize = 1;
const DAYLIGHT: usize = 2;

/// The `timeZoneName` of `identifier` (a DateTimeFormat [[TimeZone]]: a named zone or a UTC
/// offset `±HH:MM`) at `epoch_sec` for the CLDR locale `locale`, whose region (explicit or
/// likely) is `region`, in a `timeZoneName` style.
pub(super) fn display_name(
    locale: &str,
    region: &str,
    identifier: &str,
    style: &str,
    epoch_sec: i64,
) -> String {
    let names = locale_names(locale);
    let long = matches!(style, "long" | "longOffset" | "longGeneric");
    let zone = if identifier.starts_with(['+', '-']) {
        // An offset time zone is the fixed zone of that offset (ECMA-402 formats it as the
        // equivalent Etc/GMT∓N). Zero is CLDR's Etc/GMT, in the GMT metazone; the other fixed
        // zones have no location or names (UTS #35 Part 4 Generic location format note), so
        // every style is their localized GMT format.
        let offset = offset_zone_seconds(identifier);
        if offset != 0 {
            return localized_gmt(names, offset, long);
        }
        Zone {
            identifier: "UTC",
            cldr: "Etc/GMT",
            epoch_sec,
            offset: 0,
            metazone: metazone_use("Etc/GMT", epoch_sec),
            daylight: false,
        }
    } else {
        let offset = crate::tz::offset_at(identifier, epoch_sec).unwrap_or(0);
        Zone::new(identifier, epoch_sec, offset)
    };
    let offset = zone.offset;
    let formatter = Formatter {
        names,
        locale,
        region,
    };
    let name = match style {
        "short" | "long" => formatter.specific(&zone, long),
        "shortGeneric" | "longGeneric" => formatter
            .generic(&zone, long)
            .or_else(|| formatter.generic_location(&zone)),
        _ => None,
    };
    name.unwrap_or_else(|| localized_gmt(names, offset, long))
}

/// The `en` specific non-location long name of a named zone at an instant, if CLDR has one.
pub(super) fn english_long_name(identifier: &str, epoch_sec: i64) -> Option<String> {
    let offset = crate::tz::offset_at(identifier, epoch_sec)?;
    let zone = Zone::new(identifier, epoch_sec, offset);
    Formatter {
        names: locale_names("en"),
        locale: "en",
        region: "US",
    }
    .specific(&zone, true)
}

/// A named zone at an instant: its CLDR identifier, UTC offset, metazone and daylight flag.
struct Zone<'a> {
    /// The identifier the TZ database functions take.
    identifier: &'a str,
    cldr: &'static str,
    epoch_sec: i64,
    offset: i32,
    metazone: Option<&'static MetazoneUse>,
    daylight: bool,
}

impl<'a> Zone<'a> {
    fn new(identifier: &'a str, epoch_sec: i64, offset: i32) -> Self {
        let cldr = cldr_id(identifier);
        let metazone = metazone_use(cldr, epoch_sec);
        let daylight = daylight_at(identifier, metazone, epoch_sec, offset);
        Zone {
            identifier,
            cldr,
            epoch_sec,
            offset,
            metazone,
            daylight,
        }
    }
}

/// Whether daylight time is in effect: CLDR's explicit standard and daylight offsets for the
/// metazone interval decide when given (UTS #35 Part 4 Metazone Names, `usesMetazone` stdOffset
/// and dstOffset), else the TZ database's `tt_isdst`.
fn daylight_at(
    identifier: &str,
    metazone: Option<&MetazoneUse>,
    epoch_sec: i64,
    offset: i32,
) -> bool {
    match metazone.and_then(|usage| usage.offsets) {
        Some((_, daylight)) => offset == daylight,
        None => crate::tz::is_dst(identifier, epoch_sec).unwrap_or(false),
    }
}

struct Formatter<'a> {
    names: &'static LocaleNames,
    locale: &'a str,
    region: &'a str,
}

impl Formatter<'_> {
    /// The specific non-location format (UTS #35 Part 4 Using Time Zone Names, non-location
    /// formats steps 1 and 2): the zone's own standard or daylight name, else its metazone's.
    fn specific(&self, zone: &Zone, long: bool) -> Option<String> {
        let kind = if zone.daylight { DAYLIGHT } else { STANDARD };
        self.zone_or_metazone_name(zone, kind, long)
    }

    fn zone_or_metazone_name(&self, zone: &Zone, kind: usize, long: bool) -> Option<String> {
        let width = if long { 0 } else { 3 };
        if let Some(name) = names_of(self.names.zones, zone.cldr).map(|names| names[width + kind]) {
            if name != 0 {
                return Some(text(name).to_string());
            }
        }
        let metazone = zone.metazone?;
        let name = names_of(self.names.metazones, text(metazone.metazone))?[width + kind];
        (name != 0).then(|| text(name).to_string())
    }

    /// The generic non-location format (UTS #35 Part 4 Using Time Zone Names, non-location
    /// formats, with Type Fallback and the generic partial location format).
    fn generic(&self, zone: &Zone, long: bool) -> Option<String> {
        let width = if long { 0 } else { 3 };
        if let Some(name) =
            names_of(self.names.zones, zone.cldr).map(|names| names[width + GENERIC])
        {
            if name != 0 {
                return Some(text(name).to_string());
            }
        }
        let usage = zone.metazone?;
        let metazone = text(usage.metazone);
        let metazone_generic = names_of(self.names.metazones, metazone)
            .map(|names| names[width + GENERIC])
            .filter(|&name| name != 0)
            .map(text);
        // Type Fallback: without daylight time within 184 days of the instant, the standard
        // name stands for the generic one (Mountain Standard Time for Phoenix), unless it only
        // repeats the generic name.
        if !zone.daylight && !daylight_nearby(zone) {
            if let Some(standard) = self.zone_or_metazone_name(zone, STANDARD, long) {
                if metazone_generic.is_none_or(|generic| !generic.eq_ignore_ascii_case(&standard)) {
                    return Some(standard);
                }
            }
        }
        let generic = metazone_generic?;
        // The generic partial location format when the metazone's preferred zone for the
        // locale's region keeps a different offset at this instant.
        if let Some(preferred) =
            reference_zone(metazone, self.region).filter(|&preferred| preferred != zone.cldr)
        {
            let preferred_offset = crate::tz::offset_at(preferred, zone.epoch_sec);
            let preferred_daylight = preferred_offset.map(|offset| {
                daylight_at(
                    preferred,
                    metazone_use(preferred, zone.epoch_sec),
                    zone.epoch_sec,
                    offset,
                )
            });
            if preferred_offset != Some(zone.offset) || preferred_daylight != Some(zone.daylight) {
                return Some(self.partial_location(zone, metazone, generic));
            }
        }
        Some(generic.to_string())
    }

    /// The generic partial location format: the metazone name qualified by the zone's country
    /// when it is the metazone's preferred zone there, else by its exemplar city.
    fn partial_location(&self, zone: &Zone, metazone: &str, name: &str) -> String {
        let location = match zone_region(zone.cldr) {
            Some(region) if reference_zone(metazone, region) == Some(zone.cldr) => {
                self.region_name(region)
            }
            _ => self
                .exemplar_city(zone.cldr)
                .unwrap_or_else(|| zone.cldr.to_string()),
        };
        let fallback = text(self.names.formats[5]);
        fallback.replace("{1}", name).replace("{0}", &location)
    }

    /// The generic location format (UTS #35 Part 4 Using Time Zone Names, location formats):
    /// the region format with the zone's country when it is the country's only or primary zone,
    /// else with its exemplar city. A zone outside any country (Etc/UTC) has none.
    fn generic_location(&self, zone: &Zone) -> Option<String> {
        let region = zone_region(zone.cldr)?;
        let location = if is_primary(zone, region) {
            self.region_name(region)
        } else {
            self.exemplar_city(zone.cldr)?
        };
        Some(text(self.names.formats[2]).replace("{0}", &location))
    }

    /// The localized country name, or the code itself when the locale has none (UTS #35
    /// Part 4 Composition).
    fn region_name(&self, region: &str) -> String {
        if let Some(name) = lookup_pair(self.names.regions, region) {
            return name.to_string();
        }
        let language = self.locale.split('-').next().unwrap_or(self.locale);
        crate::cldr_display_names::name(language, "region", "long", region)
            .unwrap_or(region)
            .to_string()
    }

    /// The zone's localized exemplar city, or the last field of its identifier with `_` as a
    /// space (UTS #35 Part 4 Composition); none for zones that are not locations.
    fn exemplar_city(&self, cldr: &str) -> Option<String> {
        if let Some(city) = lookup_pair(self.names.cities, cldr) {
            return Some(city.to_string());
        }
        if cldr.starts_with("Etc/") || cldr.starts_with("SystemV/") {
            return None;
        }
        let (_, city) = cldr.rsplit_once('/')?;
        (!city.is_empty()).then(|| city.replace('_', " "))
    }
}

/// Whether the zone observes daylight time within the Type Fallback interval: just before the
/// previous offset change or just after the next one, when either lies within 184 days.
fn daylight_nearby(zone: &Zone) -> bool {
    let near = |instant: i64| (instant - zone.epoch_sec).abs() < DST_CHECK_RANGE;
    let daylight = |instant: i64| {
        let offset = crate::tz::offset_at(zone.identifier, instant).unwrap_or(zone.offset);
        daylight_at(
            zone.identifier,
            metazone_use(zone.cldr, instant),
            instant,
            offset,
        )
    };
    let before = crate::tz::next_transition(zone.identifier, zone.epoch_sec + 1, false)
        .filter(|&instant| near(instant))
        .is_some_and(|instant| daylight(instant - 1));
    before
        || crate::tz::next_transition(zone.identifier, zone.epoch_sec, true)
            .filter(|&instant| near(instant))
            .is_some_and(daylight)
}

/// Whether the zone is its country's only zone (IANA zone.tab) or its primary zone.
fn is_primary(zone: &Zone, region: &str) -> bool {
    crate::tz::region_zones(region).len() == 1
        || data::PRIMARY_ZONES
            .binary_search_by(|row| text(row.0).cmp(region))
            .is_ok_and(|index| text(data::PRIMARY_ZONES[index].1) == zone.cldr)
}

/// The localized GMT format (UTS #35 Part 4 Time Zone Format Terminology): the gmtFormat
/// around the hourFormat's offset, with two-digit hours and minutes (long) or unpadded hours
/// and only nonzero minutes (short), and seconds only when nonzero. A zero offset is `+`.
fn localized_gmt(names: &LocaleNames, offset: i32, long: bool) -> String {
    let hour_format = text(names.formats[0]);
    let (positive, negative) = hour_format
        .split_once(';')
        .unwrap_or((hour_format, hour_format));
    let pattern = if offset < 0 { negative } else { positive };
    let magnitude = offset.unsigned_abs();
    let (hours, minutes, seconds) = (magnitude / 3600, magnitude / 60 % 60, magnitude % 60);
    // The pattern is prefix, H run, separator, m run, suffix.
    let hour_start = pattern.find('H').unwrap_or(pattern.len());
    let hour_end = hour_start
        + pattern[hour_start..]
            .chars()
            .take_while(|&c| c == 'H')
            .count();
    let minute_start = pattern[hour_end..]
        .find('m')
        .map_or(pattern.len(), |index| hour_end + index);
    let minute_end = minute_start
        + pattern[minute_start..]
            .chars()
            .take_while(|&c| c == 'm')
            .count();
    let (prefix, separator, suffix) = (
        &pattern[..hour_start],
        &pattern[hour_end..minute_start],
        &pattern[minute_end..],
    );
    let mut value = prefix.to_string();
    if long {
        value += &format!("{hours:02}{separator}{minutes:02}");
    } else {
        value += &hours.to_string();
        if minutes != 0 || seconds != 0 {
            value += &format!("{separator}{minutes:02}");
        }
    }
    if seconds != 0 {
        value += &format!("{separator}{seconds:02}");
    }
    value += suffix;
    text(names.formats[1]).replace("{0}", &value)
}

/// The signed seconds of a canonical `±HH:MM` offset time zone.
fn offset_zone_seconds(identifier: &str) -> i32 {
    let sign = if identifier.starts_with('-') { -1 } else { 1 };
    let mut fields = identifier[1..].split(':');
    let hours: i32 = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let minutes: i32 = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    sign * (hours * 3600 + minutes * 60)
}

/// The CLDR names of a locale (an advertised DateTimeFormat locale; English otherwise).
fn locale_names(locale: &str) -> &'static LocaleNames {
    let locales = data::LOCALES;
    let index = locales
        .binary_search_by(|(candidate, _)| (*candidate).cmp(locale))
        .or_else(|_| locales.binary_search_by(|(candidate, _)| (*candidate).cmp("en")))
        .unwrap_or(0);
    &locales[index].1
}

/// The CLDR canonical identifier (UTS #35 Part 1 Time Zone Identifiers) of a zone identifier.
/// The UTC family (Etc/UTC, Etc/GMT, GMT, …), whose ECMA-402 primary identifier is "UTC", is
/// Etc/UTC.
fn cldr_id(identifier: &str) -> &'static str {
    if crate::tz::canonicalize(identifier) == Some("UTC") {
        return "Etc/UTC";
    }
    let aliases = data::CLDR_IDS;
    if let Ok(index) = aliases.binary_search_by(|row| text(row.0).cmp(identifier)) {
        return text(aliases[index].1);
    }
    if let Some(name) = crate::tz::registry_name(identifier) {
        return name;
    }
    crate::tz::canonicalize(identifier).unwrap_or("Etc/Unknown")
}

/// The metazone interval in effect for a CLDR zone at an instant.
fn metazone_use(cldr: &str, epoch_sec: i64) -> Option<&'static MetazoneUse> {
    let uses = data::METAZONE_USES;
    let index = uses.binary_search_by(|row| text(row.0).cmp(cldr)).ok()?;
    uses[index]
        .1
        .iter()
        .find(|usage| usage.from <= epoch_sec && epoch_sec < usage.to)
}

/// The metazone's preferred zone for a region, else its golden zone (territory 001).
fn reference_zone(metazone: &str, region: &str) -> Option<&'static str> {
    let find = |territory: &str| {
        data::REFERENCE_ZONES
            .binary_search_by(|row| {
                text(row.0)
                    .cmp(metazone)
                    .then_with(|| text(row.1).cmp(territory))
            })
            .ok()
            .map(|index| text(data::REFERENCE_ZONES[index].2))
    };
    find(region).or_else(|| find("001"))
}

/// The country of a CLDR zone (IANA zone.tab), if it has one.
fn zone_region(cldr: &str) -> Option<&'static str> {
    static REGIONS: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    REGIONS
        .get_or_init(|| {
            let mut regions = HashMap::new();
            for (region, zones) in crate::tzdata::REGION_ZONES {
                for zone in *zones {
                    regions.insert(cldr_id(zone), *region);
                }
            }
            regions
        })
        .get(cldr)
        .copied()
}

fn names_of(rows: &'static [(Id, data::Names)], key: &str) -> Option<&'static data::Names> {
    rows.binary_search_by(|row| text(row.0).cmp(key))
        .ok()
        .map(|index| &rows[index].1)
}

fn lookup_pair(rows: &'static [(Id, Id)], key: &str) -> Option<&'static str> {
    rows.binary_search_by(|row| text(row.0).cmp(key))
        .ok()
        .map(|index| text(rows[index].1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const JAN_2024: i64 = 1_704_369_600; // 2024-01-04T12:00:00Z
    const JUL_2024: i64 = 1_720_094_400; // 2024-07-04T12:00:00Z

    fn name(locale: &str, region: &str, zone: &str, style: &str, at: i64) -> String {
        display_name(locale, region, zone, style, at)
    }

    #[test]
    fn generated_tables_are_sorted_for_binary_search() {
        let sorted = |keys: Vec<&str>| keys.windows(2).all(|pair| pair[0] < pair[1]);
        assert!(sorted(
            data::METAZONE_USES.iter().map(|row| text(row.0)).collect()
        ));
        assert!(sorted(
            data::PRIMARY_ZONES.iter().map(|row| text(row.0)).collect()
        ));
        assert!(sorted(
            data::CLDR_IDS.iter().map(|row| text(row.0)).collect()
        ));
        assert!(data::REFERENCE_ZONES
            .windows(2)
            .all(|pair| (text(pair[0].0), text(pair[0].1)) < (text(pair[1].0), text(pair[1].1))));
        assert!(data::LOCALES.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for (_, names) in data::LOCALES {
            assert!(sorted(
                names.metazones.iter().map(|row| text(row.0)).collect()
            ));
            assert!(sorted(names.zones.iter().map(|row| text(row.0)).collect()));
            assert!(sorted(names.cities.iter().map(|row| text(row.0)).collect()));
            assert!(sorted(
                names.regions.iter().map(|row| text(row.0)).collect()
            ));
        }
    }

    #[test]
    fn specific_names_follow_daylight_time_and_metazones() {
        let ny = |style, at| name("en", "US", "America/New_York", style, at);
        assert_eq!(ny("short", JAN_2024), "EST");
        assert_eq!(ny("short", JUL_2024), "EDT");
        assert_eq!(ny("long", JAN_2024), "Eastern Standard Time");
        assert_eq!(ny("long", JUL_2024), "Eastern Daylight Time");
        assert_eq!(
            name("de", "DE", "America/New_York", "long", JAN_2024),
            "Nordamerikanische Ostküsten-Normalzeit"
        );
        // A locale without the short metazone name uses the short localized GMT format.
        assert_eq!(
            name("de", "DE", "America/New_York", "short", JAN_2024),
            "GMT-5"
        );
        assert_eq!(name("de", "DE", "Europe/Berlin", "short", JUL_2024), "MESZ");
        assert_eq!(
            name("en", "US", "Asia/Kolkata", "long", JAN_2024),
            "India Standard Time"
        );
        assert_eq!(
            name("en", "US", "Asia/Calcutta", "short", JAN_2024),
            "GMT+5:30"
        );
        // Zone-specific names: British Summer Time, and Ireland's daylight name in summer with
        // CLDR's explicit standard/daylight offsets (the TZ database marks winter as daylight).
        assert_eq!(
            name("en", "US", "Europe/London", "long", JUL_2024),
            "British Summer Time"
        );
        assert_eq!(
            name("en", "US", "Europe/London", "long", JAN_2024),
            "Greenwich Mean Time"
        );
        assert_eq!(
            name("en", "US", "Europe/Dublin", "long", JAN_2024),
            "Greenwich Mean Time"
        );
        assert_eq!(
            name("en", "US", "Europe/Dublin", "long", JUL_2024),
            "Irish Standard Time"
        );
        assert_eq!(
            name("en", "US", "Europe/Dublin", "short", JUL_2024),
            "GMT+1"
        );
        // Indiana/Knox used Eastern time between 1991 and 2006.
        assert_eq!(
            name("en", "US", "America/Indiana/Knox", "long", 1_000_000_000),
            "Eastern Standard Time"
        );
        assert_eq!(
            name("en", "US", "America/Indiana/Knox", "long", JAN_2024),
            "Central Standard Time"
        );
    }

    #[test]
    fn utc_and_offset_zones() {
        assert_eq!(
            name("en", "US", "UTC", "long", JAN_2024),
            "Coordinated Universal Time"
        );
        assert_eq!(
            name("de", "DE", "UTC", "long", JAN_2024),
            "Koordinierte Weltzeit"
        );
        assert_eq!(name("fr", "FR", "UTC", "short", JAN_2024), "UTC");
        assert_eq!(
            name("fr", "FR", "Etc/GMT", "long", JAN_2024),
            "temps universel coordonné"
        );
        assert_eq!(name("en", "US", "UTC", "shortOffset", JAN_2024), "GMT+0");
        assert_eq!(name("en", "US", "UTC", "longOffset", JAN_2024), "GMT+00:00");
        assert_eq!(name("fr", "FR", "UTC", "shortOffset", JAN_2024), "UTC+0");
        assert_eq!(
            name("en", "US", "UTC", "longGeneric", JAN_2024),
            "GMT+00:00"
        );
        // The zero offset zone is Etc/GMT.
        assert_eq!(name("en", "US", "+00:00", "short", JAN_2024), "GMT");
        assert_eq!(
            name("en", "US", "+00:00", "long", JAN_2024),
            "Greenwich Mean Time"
        );
        assert_eq!(name("en", "US", "+00:00", "shortOffset", JAN_2024), "GMT+0");
        assert_eq!(name("fr", "FR", "+00:00", "short", JAN_2024), "UTC+0");
        assert_eq!(name("en", "US", "+05:30", "short", JAN_2024), "GMT+5:30");
        assert_eq!(
            name("en", "US", "+05:30", "longGeneric", JAN_2024),
            "GMT+05:30"
        );
        assert_eq!(name("fr", "FR", "-03:00", "long", JAN_2024), "UTC−03:00");
        assert_eq!(name("en", "US", "Etc/GMT+5", "long", JAN_2024), "GMT-05:00");
        assert_eq!(
            name("en", "US", "Etc/GMT+5", "shortGeneric", JAN_2024),
            "GMT-5"
        );
    }

    #[test]
    fn generic_names_use_type_fallback_and_locations() {
        let en = |zone, style| name("en", "US", zone, style, JAN_2024);
        assert_eq!(en("America/Los_Angeles", "longGeneric"), "Pacific Time");
        assert_eq!(en("America/Los_Angeles", "shortGeneric"), "PT");
        // No daylight time within 184 days: the standard name.
        assert_eq!(
            en("America/Phoenix", "longGeneric"),
            "Mountain Standard Time"
        );
        assert_eq!(en("America/Phoenix", "shortGeneric"), "MST");
        assert_eq!(en("Asia/Kolkata", "longGeneric"), "India Standard Time");
        // Without a generic metazone name: the generic location format.
        assert_eq!(en("Asia/Kolkata", "shortGeneric"), "India Time");
        assert_eq!(en("Europe/London", "longGeneric"), "United Kingdom Time");
        assert_eq!(en("Europe/Dublin", "shortGeneric"), "Ireland Time");
        assert_eq!(en("Africa/Abidjan", "longGeneric"), "Greenwich Mean Time");
        assert_eq!(en("Europe/Moscow", "shortGeneric"), "Moscow Time");
        assert_eq!(en("Asia/Shanghai", "shortGeneric"), "China Time");
        assert_eq!(en("Europe/Busingen", "shortGeneric"), "Büsingen Time");
        assert_eq!(en("Europe/Berlin", "shortGeneric"), "Germany Time");
        assert_eq!(en("Atlantic/Reykjavik", "shortGeneric"), "GMT");
        assert_eq!(
            name("de", "DE", "Europe/Busingen", "shortGeneric", JAN_2024),
            "MEZ"
        );
        assert_eq!(
            name("de", "DE", "America/Indiana/Knox", "shortGeneric", JAN_2024),
            "Knox, Indiana (Ortszeit)"
        );
        assert_eq!(
            name("fr", "FR", "Atlantic/Reykjavik", "shortGeneric", JAN_2024),
            "heure : Islande"
        );
    }

    #[test]
    fn english_long_names_for_date_to_string() {
        assert_eq!(
            english_long_name("Europe/Berlin", JUL_2024).as_deref(),
            Some("Central European Summer Time")
        );
        assert_eq!(
            english_long_name("America/Argentina/Buenos_Aires", JAN_2024).as_deref(),
            Some("Argentina Standard Time")
        );
        assert_eq!(
            english_long_name("Pacific/Auckland", JAN_2024).as_deref(),
            Some("New Zealand Daylight Time")
        );
        assert_eq!(english_long_name("Etc/GMT+5", JAN_2024), None);
        assert_eq!(english_long_name("Not/AZone", JAN_2024), None);
    }

    #[test]
    fn localized_gmt_uses_the_hour_format() {
        let en = locale_names("en");
        assert_eq!(localized_gmt(en, 0, false), "GMT+0");
        assert_eq!(localized_gmt(en, -3 * 3600 - 1800, false), "GMT-3:30");
        assert_eq!(localized_gmt(en, 5 * 3600 + 45 * 60, true), "GMT+05:45");
        assert_eq!(
            localized_gmt(en, -(3600 + 23 * 60 + 5), true),
            "GMT-01:23:05"
        );
        assert_eq!(localized_gmt(locale_names("fr"), -7200, false), "UTC−2");
    }
}
