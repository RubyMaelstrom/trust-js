//! CLDR supplemental locale information used by `Intl.Locale`.
//!
//! The grouped tables below transcribe CLDR's `calendarPreferenceData`, `timeData`, and the
//! first-day/weekend portion of `weekData`. They come from Unicode CLDR commit
//! 83a0a8e8ee0db41a6ee3caaa19c47ba4007070d5 (UTS #35, version 48.2). Grouping territories exactly
//! as CLDR does keeps this small enough to audit while still covering the complete data set.

const AVAILABLE_CALENDARS: &[&str] = &[
    "buddhist",
    "chinese",
    "coptic",
    "dangi",
    "ethioaa",
    "ethiopic",
    "gregory",
    "hebrew",
    "indian",
    "islamic-civil",
    "islamic-tbla",
    "islamic-umalqura",
    "iso8601",
    "japanese",
    "persian",
    "roc",
];

const CALENDAR_PREFERENCES: &[(&str, &[&str])] = &[
    ("001", &["gregorian"]),
    (
        "BD DJ DZ EH ER ID IQ JO KM LB LY MA MR MY NE OM PK PS SD SY TD TN YE",
        &["gregorian", "islamic", "islamic-civil", "islamic-tbla"],
    ),
    (
        "AL AZ MV TJ TM TR UZ XK",
        &["gregorian", "islamic-civil", "islamic-tbla"],
    ),
    (
        "AE BH KW QA",
        &[
            "gregorian",
            "islamic-umalqura",
            "islamic",
            "islamic-civil",
            "islamic-tbla",
        ],
    ),
    (
        "AF IR",
        &[
            "persian",
            "gregorian",
            "islamic",
            "islamic-civil",
            "islamic-tbla",
        ],
    ),
    ("CN CX HK MO SG", &["gregorian", "chinese"]),
    (
        "EG",
        &[
            "gregorian",
            "coptic",
            "islamic",
            "islamic-civil",
            "islamic-tbla",
        ],
    ),
    ("ET", &["gregorian", "ethiopic"]),
    (
        "IL",
        &[
            "gregorian",
            "hebrew",
            "islamic",
            "islamic-civil",
            "islamic-tbla",
        ],
    ),
    ("IN", &["gregorian", "indian"]),
    ("JP", &["gregorian", "japanese"]),
    ("KR", &["gregorian", "dangi"]),
    (
        "SA",
        &["gregorian", "islamic-umalqura", "islamic", "islamic-rgsa"],
    ),
    ("TH", &["buddhist", "gregorian"]),
    ("TW", &["gregorian", "roc", "chinese"]),
];

const TIME_DATA: &[(&str, &str)] = &[
    ("AX BQ CP CZ DK FI ID IS ML NE RU SE SJ SK", "H"),
    ("001 BI BY FO GL HU MG MT MU MV NO PL TH TJ TM VN ZW", "H h"),
    (
        "AC AI BW BZ CC CK CX DG FK GB GG GI GS IE IM IO JE LT MK MN MS NF NG NR NU PN SH SX TA ZA en_IL",
        "H h hb hB",
    ),
    ("CF CM LU NP PF SC SM SN TF VA ca_ES fr_CA gl_ES it_CH it_IT", "H h hB"),
    ("AR CL EA IC KG KM LK MA PY UY af_ZA es_BR es_ES es_GQ", "H h hB hb"),
    ("JP", "H K h"),
    ("AF LA", "H hb hB h"),
    (
        "AD AM AO AT AW BE BF BJ BL BR CG CI CV CW DE EE FR GA GF GN GP GW HR HT IL IT KZ MC MD MF MQ MZ NC NL PM PT RE RO SI SR ST TG TR WF YT ZM ku_SY",
        "H hB",
    ),
    ("AZ BA BG CH GE LI ME RS UA UZ XK", "H hB h"),
    ("ES GQ", "H hB h hb"),
    ("CN LV TL zu_ZA", "H hB hb h"),
    ("CD IR", "H hB"),
    ("KE MM RW TZ UG", "H hB hb h"),
    ("AS BT DJ ER GH IN LS PG PW SO TO VU WS", "h H"),
    ("CY GR", "h H hb hB"),
    ("AL TD", "h H hB"),
    ("419 BO CO CR CU DO EC GT HN KP KR MX NI NA PA PE PR SV VE", "h H hB hb"),
    (
        "AG AU BB BM BS CA DM FJ FM GD GM GU GY JM KI KN KY LC LR MH MP MW NZ SB SG SL SS SZ TC TT UM US VC VG VI en_001 en_HK en_MY",
        "h hb H hB",
    ),
    ("BD PK", "h hB H"),
    ("AE BH DZ EG EH HK IQ JO KW LB LY MO MR OM PH PS QA SA SD SY TN YE ar_001", "h hB hb H"),
    ("BN MY", "h hb hB H"),
    ("hi_IN kn_IN ml_IN te_IN", "hB h H"),
    ("KH", "hB h H hb"),
    ("ta_IN", "hB h hb H"),
    ("TW ET gu_IN mr_IN pa_IN", "hB hb h H"),
];

const FIRST_DAY_DATA: &[(&str, &str)] = &[
    (
        "mon",
        "001 AD AE AI AL AM AN AR AT AU AX AZ BA BE BG BM BN BY CH CL CM CN CR CY CZ DE DK EC EE ES FI FJ FO FR GB GE GF GP GR HR HU IE IT KG KZ LB LI LK LT LU LV MC MD ME MK MN MQ MY NL NO NZ PL RE RO RS RU SE SI SK SM TJ TM TR UA UY UZ VA VN XK",
    ),
    ("fri", "MV"),
    ("sat", "AF BH DJ DZ EG IQ IR JO KW LY OM QA SD SY"),
    (
        "sun",
        "AG AS BD BR BS BT BW BZ CA CO DM DO ET GT GU HK HN ID IL IN IS JM JP KE KH KR LA MH MM MO MT MX MZ NI NP PA PE PH PK PR PT PY SA SG SV TH TT TW UM US VE VI WS YE ZA ZW",
    ),
];

const WEEKEND_START_DATA: &[(&str, &str)] = &[
    ("thu", "AF"),
    ("fri", "BH DZ EG IL IQ IR JO KW LY OM QA SA SD SY YE"),
    ("sat", "001"),
    ("sun", "IN UG"),
];

const WEEKEND_END_DATA: &[(&str, &str)] = &[
    ("fri", "AF IR"),
    ("sat", "BH DZ EG IL IQ JO KW LY OM QA SA SD SY YE"),
    ("sun", "001"),
];

fn contains_key(keys: &str, key: &str) -> bool {
    keys.split_ascii_whitespace()
        .any(|candidate| candidate == key)
}

/// Calendar preference data explicitly available for a region, filtered to calendars implemented
/// by this engine. The caller performs ECMA-402's ordered region fallback before using `001`.
pub fn calendar_preferences(region: &str) -> Option<Vec<&'static str>> {
    let (_, calendars) = CALENDAR_PREFERENCES
        .iter()
        .find(|(regions, _)| contains_key(regions, region))?;
    Some(
        calendars
            .iter()
            .filter_map(|calendar| {
                let canonical = if *calendar == "gregorian" {
                    "gregory"
                } else {
                    calendar
                };
                AVAILABLE_CALENDARS
                    .contains(&canonical)
                    .then_some(canonical)
            })
            .collect(),
    )
}

/// Time preference data explicitly available for a language-region key or region. CLDR pattern
/// letters map to ECMA-402 hour-cycle identifiers; flexible day periods retain the cycle selected
/// by their leading `h`/`H` pattern letter.
pub fn hour_cycles(key: &str) -> Option<Vec<&'static str>> {
    let (_, allowed) = TIME_DATA.iter().find(|(keys, _)| contains_key(keys, key))?;
    let mut cycles = Vec::new();
    for pattern in allowed.split_ascii_whitespace() {
        let cycle = match pattern.as_bytes().first() {
            Some(b'h') => "h12",
            Some(b'H') => "h23",
            Some(b'K') => "h11",
            Some(b'k') => "h24",
            _ => continue,
        };
        if !cycles.contains(&cycle) {
            cycles.push(cycle);
        }
    }
    Some(cycles)
}

fn weekday(day: &str) -> u8 {
    match day {
        "mon" => 1,
        "tue" => 2,
        "wed" => 3,
        "thu" => 4,
        "fri" => 5,
        "sat" => 6,
        "sun" => 7,
        _ => unreachable!("CLDR weekday table contains an invalid identifier"),
    }
}

fn region_day(data: &[(&str, &str)], region: &str, default: u8) -> u8 {
    data.iter()
        .find(|(_, regions)| contains_key(regions, region))
        .map(|(day, _)| weekday(day))
        .unwrap_or(default)
}

/// The CLDR week-information record for a region, with inherited `001` values applied. Weekend
/// days are returned in the ascending ISO weekday order required by ECMA-402.
pub fn week_info(region: &str) -> (u8, Vec<u8>) {
    let first_day = region_day(FIRST_DAY_DATA, region, 1);
    let weekend_start = region_day(WEEKEND_START_DATA, region, 6);
    let weekend_end = region_day(WEEKEND_END_DATA, region, 7);
    let mut weekend = if weekend_start <= weekend_end {
        (weekend_start..=weekend_end).collect::<Vec<_>>()
    } else {
        (weekend_start..=7)
            .chain(1..=weekend_end)
            .collect::<Vec<_>>()
    };
    weekend.sort_unstable();
    (first_day, weekend)
}

/// The scripts whose default general ordering of characters within a line is right-to-left
/// (`RTL: YES`) in UTS #35 Part 1 Script Metadata, CLDR common/properties/scriptMetadata.txt
/// at commit 1c6bc010ee9a36ec11c108b7f5e015d08b86e389. Sorted.
const RTL_SCRIPTS: &[&str] = &[
    "Adlm", "Arab", "Armi", "Avst", "Chrs", "Cprt", "Elym", "Gara", "Hatr", "Hebr", "Hung", "Khar",
    "Lydi", "Mand", "Mani", "Mend", "Merc", "Mero", "Narb", "Nbat", "Nkoo", "Orkh", "Ougr", "Palm",
    "Phli", "Phlp", "Phnx", "Prti", "Rohg", "Samr", "Sarb", "Sidt", "Sogd", "Sogo", "Syrc", "Thaa",
    "Yezi",
];

/// The scripts the same file marks left-to-right (`RTL: NO`). Scripts it marks `UNKNOWN`
/// (Brai, Zinh, Zyyy, Zzzz) and scripts it does not list have no known direction. Sorted.
const LTR_SCRIPTS: &[&str] = &[
    "Aghb", "Ahom", "Armn", "Bali", "Bamu", "Bass", "Batk", "Beng", "Berf", "Bhks", "Bopo", "Brah",
    "Bugi", "Buhd", "Cakm", "Cans", "Cari", "Cham", "Cher", "Copt", "Cpmn", "Cyrl", "Deva", "Diak",
    "Dogr", "Dsrt", "Dupl", "Egyp", "Elba", "Ethi", "Geor", "Glag", "Gong", "Gonm", "Goth", "Gran",
    "Grek", "Gujr", "Gukh", "Guru", "Hanb", "Hang", "Hani", "Hano", "Hans", "Hant", "Hira", "Hluw",
    "Hmng", "Hmnp", "Ital", "Jamo", "Java", "Jpan", "Jurc", "Kali", "Kana", "Kawi", "Khmr", "Khoj",
    "Kits", "Knda", "Kore", "Krai", "Kthi", "Lana", "Laoo", "Latn", "Lepc", "Limb", "Lina", "Linb",
    "Lisu", "Lyci", "Mahj", "Maka", "Marc", "Medf", "Mlym", "Modi", "Mong", "Mroo", "Mtei", "Mult",
    "Mymr", "Nagm", "Nand", "Newa", "Nshu", "Ogam", "Olck", "Onao", "Orya", "Osge", "Osma", "Pauc",
    "Pcun", "Perm", "Phag", "Plrd", "Rjng", "Runr", "Saur", "Seal", "Sgnw", "Shaw", "Shrd", "Sidd",
    "Sind", "Sinh", "Sora", "Soyo", "Sund", "Sunu", "Sylo", "Tagb", "Takr", "Tale", "Talu", "Taml",
    "Tang", "Tavt", "Tayo", "Telu", "Tfng", "Tglg", "Thai", "Tibt", "Tirh", "Tnsa", "Todr", "Tols",
    "Toto", "Tutg", "Ugar", "Vaii", "Vith", "Wara", "Wcho", "Xpeo", "Xsux", "Yiii", "Zanb",
];

/// The default line direction of a (title-case) ISO 15924 script: `"rtl"`, `"ltr"`, or `None`
/// when Script Metadata does not determine it.
pub fn script_direction(script: &str) -> Option<&'static str> {
    if RTL_SCRIPTS.binary_search(&script).is_ok() {
        Some("rtl")
    } else if LTR_SCRIPTS.binary_search(&script).is_ok() {
        Some("ltr")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplemental_data_keeps_locale_and_region_distinctions() {
        assert_eq!(calendar_preferences("TH").unwrap(), ["buddhist", "gregory"]);
        assert_eq!(hour_cycles("fr_CA").unwrap(), ["h23", "h12"]);
        assert_eq!(hour_cycles("CA").unwrap(), ["h12", "h23"]);
        assert_eq!(week_info("AF"), (6, vec![4, 5]));
        assert_eq!(week_info("IN"), (7, vec![7]));
        assert_eq!(week_info("001"), (1, vec![6, 7]));
    }

    #[test]
    fn script_metadata_directions_are_sorted_and_disjoint() {
        assert!(RTL_SCRIPTS.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(LTR_SCRIPTS.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(RTL_SCRIPTS
            .iter()
            .all(|script| !LTR_SCRIPTS.contains(script)));
        assert_eq!(script_direction("Arab"), Some("rtl"));
        assert_eq!(script_direction("Hebr"), Some("rtl"));
        assert_eq!(script_direction("Latn"), Some("ltr"));
        assert_eq!(script_direction("Zyyy"), None);
        assert_eq!(script_direction("Qaaa"), None);
    }
}
