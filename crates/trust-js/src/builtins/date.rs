//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

fn now_ms(i: &Interp) -> f64 {
    i.wall_now_ms().trunc()
}

/// ECMA-262 #sec-systemtimezoneidentifier: the host environment's current time
/// zone, as a primary identifier or a UTC offset string. `TZ` names it first (an
/// IANA identifier, optionally `:`-prefixed, or a zoneinfo path; empty is UTC),
/// then Windows' zone (mapped through CLDR windowsZones), or elsewhere the
/// `/etc/localtime` link, then `/etc/timezone`. A zone TRust JS's offset tables do
/// not know is UTC. Unit tests run in UTC unless they choose a zone.
pub(crate) fn system_time_zone() -> &'static str {
    #[cfg(test)]
    {
        TEST_TIME_ZONE.with(std::cell::Cell::get).unwrap_or("UTC")
    }
    #[cfg(not(test))]
    {
        static ZONE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        ZONE.get_or_init(detect_time_zone)
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_TIME_ZONE: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(not(test))]
fn detect_time_zone() -> &'static str {
    let known = |name: &str| crate::tz::canonicalize(name.trim().trim_start_matches(':'));
    let from_path = |path: &str| {
        path.rsplit_once("zoneinfo/")
            .and_then(|(_, name)| known(name))
    };
    if let Some(tz) = std::env::var_os("TZ") {
        let tz = tz.to_string_lossy();
        if tz.trim().is_empty() {
            return "UTC";
        }
        if let Some(zone) = known(&tz).or_else(|| from_path(&tz)) {
            return zone;
        }
    }
    #[cfg(windows)]
    {
        windows_zone::detect().unwrap_or("UTC")
    }
    #[cfg(not(windows))]
    {
        std::fs::read_link("/etc/localtime")
            .ok()
            .and_then(|target| from_path(&target.to_string_lossy()))
            .or_else(|| {
                std::fs::read_to_string("/etc/timezone")
                    .ok()
                    .and_then(|name| known(&name))
            })
            .unwrap_or("UTC")
    }
}

/// The system time zone for Windows' current zone: its time zone key name (and the user's
/// region) mapped through CLDR windowsZones (UTS #35 Part 4, Windows Zones), as a primary
/// identifier; or, when "adjust for daylight saving time automatically" is off, Windows' fixed
/// standard offset (`bias_minutes` west of UTC).
#[cfg(any(windows, test))]
fn windows_time_zone(
    key_name: &str,
    region: Option<&str>,
    fixed_bias_minutes: Option<i32>,
) -> Option<&'static str> {
    if let Some(bias) = fixed_bias_minutes {
        return Some(fixed_offset_zone(-bias));
    }
    crate::cldr_windows_zones::unicode_zone(key_name, region).and_then(crate::tz::canonicalize)
}

/// A fixed UTC offset (`minutes` east of UTC) as a system time zone: UTC, else the IANA
/// `Etc/GMT∓h` zone of a whole-hour offset (whose sign is inverted, POSIX style), else the
/// offset string ECMA-262 #sec-systemtimezoneidentifier permits.
#[cfg(any(windows, test))]
fn fixed_offset_zone(minutes: i32) -> &'static str {
    if minutes == 0 {
        return "UTC";
    }
    if minutes % 60 == 0 {
        let hours = -minutes / 60;
        if let Some(zone) = crate::tz::canonicalize(&format!("Etc/GMT{hours:+}")) {
            return zone;
        }
    }
    let sign = if minutes < 0 { '-' } else { '+' };
    let (hours, minutes) = (minutes.abs() / 60, minutes.abs() % 60);
    // Computed once per process (system_time_zone caches it).
    Box::leak(format!("{sign}{hours:02}:{minutes:02}").into_boxed_str())
}

/// The minutes east of UTC of a `±HH:MM` offset time zone identifier.
fn offset_zone_minutes(zone: &str) -> Option<i32> {
    let sign = match zone.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let (hours, minutes) = zone[1..].split_once(':')?;
    Some(sign * (hours.parse::<i32>().ok()? * 60 + minutes.parse::<i32>().ok()?))
}

#[cfg(windows)]
mod windows_zone {
    #[repr(C)]
    #[derive(Clone, Copy, Default, PartialEq)]
    struct SystemTime {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        milliseconds: u16,
    }

    /// DYNAMIC_TIME_ZONE_INFORMATION (the API fills fields detection does not read).
    #[repr(C)]
    #[allow(dead_code)]
    struct DynamicTimeZoneInformation {
        bias: i32,
        standard_name: [u16; 32],
        standard_date: SystemTime,
        standard_bias: i32,
        daylight_name: [u16; 32],
        daylight_date: SystemTime,
        daylight_bias: i32,
        time_zone_key_name: [u16; 128],
        dynamic_daylight_time_disabled: u8,
    }

    const TIME_ZONE_ID_INVALID: u32 = u32::MAX;
    const GEOCLASS_NATION: u32 = 16;
    const GEOID_NOT_AVAILABLE: i32 = -1;
    const GEO_ISO2: u32 = 4;
    const HKEY_LOCAL_MACHINE: isize = 0x8000_0002_u32 as i32 as isize;
    const KEY_READ: u32 = 0x2_0019;
    const RRF_RT_REG_SZ: u32 = 0x2;
    const ERROR_SUCCESS: i32 = 0;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDynamicTimeZoneInformation(information: *mut DynamicTimeZoneInformation) -> u32;
        fn GetUserGeoID(class: u32) -> i32;
        fn GetGeoInfoW(location: i32, kind: u32, data: *mut u16, length: i32, language: u16)
            -> i32;
    }

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegOpenKeyExW(
            key: isize,
            sub_key: *const u16,
            options: u32,
            sam: u32,
            result: *mut isize,
        ) -> i32;
        fn RegEnumKeyExW(
            key: isize,
            index: u32,
            name: *mut u16,
            name_length: *mut u32,
            reserved: *mut u32,
            class: *mut u16,
            class_length: *mut u32,
            last_write: *mut u64,
        ) -> i32;
        fn RegGetValueW(
            key: isize,
            sub_key: *const u16,
            value: *const u16,
            flags: u32,
            kind: *mut u32,
            data: *mut u8,
            data_length: *mut u32,
        ) -> i32;
        fn RegCloseKey(key: isize) -> i32;
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain([0]).collect()
    }

    fn until_nul(text: &[u16]) -> &[u16] {
        &text[..text
            .iter()
            .position(|&unit| unit == 0)
            .unwrap_or(text.len())]
    }

    pub(super) fn detect() -> Option<&'static str> {
        // SAFETY: an all-zero DYNAMIC_TIME_ZONE_INFORMATION is a valid value of the plain-data
        // structure, which the call fills in.
        let mut information: DynamicTimeZoneInformation = unsafe { std::mem::zeroed() };
        // SAFETY: the pointer is to a writable structure laid out as the API requires.
        if unsafe { GetDynamicTimeZoneInformation(&mut information) } == TIME_ZONE_ID_INVALID {
            return None;
        }
        let key = until_nul(&information.time_zone_key_name);
        // With automatic daylight saving time adjustment switched off, Windows keeps the
        // zone's standard time all year: the dynamic information then reports no transition
        // dates (or, without a key name, the redirected zone's dates with identical start
        // and end), as the Date and Time settings themselves test.
        let no_dates = information.standard_date == SystemTime::default();
        let fixed = information.dynamic_daylight_time_disabled != 0
            && information.standard_date == information.daylight_date
            && (key.is_empty() != no_dates);
        // Local standard time is UTC - (Bias + StandardBias).
        let fixed_bias = fixed.then_some(information.bias + information.standard_bias);
        let key = if key.is_empty() && !fixed {
            // A remote session with time zone redirection gives no key name: the zone is the
            // registry entry whose standard name is the one reported.
            key_for_standard_name(until_nul(&information.standard_name))?
        } else {
            String::from_utf16_lossy(key)
        };
        super::windows_time_zone(&key, user_region().as_deref(), fixed_bias)
    }

    /// The user's ISO 3166 region (GEO_ISO2 of the user's nation).
    fn user_region() -> Option<String> {
        // SAFETY: GetUserGeoID takes a GEOCLASS by value.
        let nation = unsafe { GetUserGeoID(GEOCLASS_NATION) };
        if nation == GEOID_NOT_AVAILABLE {
            return None;
        }
        let mut buffer = [0u16; 16];
        // SAFETY: the buffer is writable for the length passed.
        let written = unsafe {
            GetGeoInfoW(
                nation,
                GEO_ISO2,
                buffer.as_mut_ptr(),
                buffer.len() as i32,
                0,
            )
        };
        let region = String::from_utf16(until_nul(&buffer[..written.max(0) as usize])).ok()?;
        (region.len() == 2 && region.bytes().all(|byte| byte.is_ascii_uppercase()))
            .then_some(region)
    }

    /// The Time Zones registry key whose `Std` name is `standard_name`.
    fn key_for_standard_name(standard_name: &[u16]) -> Option<String> {
        let path = wide(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones");
        let mut zones: isize = 0;
        // SAFETY: the path is NUL-terminated and the result pointer writable.
        if unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_READ, &mut zones) }
            != ERROR_SUCCESS
        {
            return None;
        }
        let std_value = wide("Std");
        let mut found = None;
        for index in 0.. {
            let mut name = [0u16; 256];
            let mut name_length = name.len() as u32;
            // SAFETY: the name buffer is writable for name_length units; the optional
            // outputs are null.
            let status = unsafe {
                RegEnumKeyExW(
                    zones,
                    index,
                    name.as_mut_ptr(),
                    &mut name_length,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if status != ERROR_SUCCESS {
                break;
            }
            let mut sub_key = name[..name_length as usize].to_vec();
            sub_key.push(0);
            let mut value = [0u16; 128];
            let mut value_length = std::mem::size_of_val(&value) as u32;
            // SAFETY: the strings are NUL-terminated and the data buffer is writable for
            // value_length bytes.
            let status = unsafe {
                RegGetValueW(
                    zones,
                    sub_key.as_ptr(),
                    std_value.as_ptr(),
                    RRF_RT_REG_SZ,
                    std::ptr::null_mut(),
                    value.as_mut_ptr().cast(),
                    &mut value_length,
                )
            };
            if status == ERROR_SUCCESS && until_nul(&value) == standard_name {
                found = String::from_utf16(&sub_key[..name_length as usize]).ok();
                break;
            }
        }
        // SAFETY: the key was opened above.
        unsafe { RegCloseKey(zones) };
        found
    }
}

/// GetNamedTimeZoneOffsetNanoseconds for the system time zone at time value
/// `t`, in milliseconds.
fn offset_ms(t: f64) -> f64 {
    let zone = system_time_zone();
    if zone == "UTC" || !t.is_finite() {
        return 0.0;
    }
    if let Some(minutes) = offset_zone_minutes(zone) {
        return f64::from(minutes) * 60_000.0;
    }
    crate::tz::offset_at(zone, (t / 1000.0).floor() as i64).map_or(0.0, |s| f64::from(s) * 1000.0)
}

/// ECMA-262 #sec-localtime.
fn local_time(t: f64) -> f64 {
    t + offset_ms(t)
}

/// ECMA-262 #sec-utc-t: the time value of local time `t`. A repeated local time
/// (a negative transition) is its earlier instant; a skipped one (a positive
/// transition) is read with the offset in effect before the transition.
fn utc_time(t: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let before = offset_ms(t - 86_400_000.0);
    let after = offset_ms(t + 86_400_000.0);
    let offset = [before, after]
        .into_iter()
        .find(|&offset| offset_ms(t - offset) == offset)
        .unwrap_or(before);
    t - offset
}

/// ECMA-262 #sec-timezoneestring: `GMT±hhmm` and the zone's name.
fn time_zone_string(t: f64) -> String {
    let offset = offset_ms(t).trunc() as i64;
    let sign = if offset >= 0 { '+' } else { '-' };
    let minutes = offset.abs() / 60_000;
    let (hours, minutes) = (minutes / 60, minutes % 60);
    // The zone's English CLDR name, else (or without Intl's locale data) its offset.
    #[cfg(feature = "intl")]
    let name = crate::intl::time_zone_long_name(system_time_zone(), (t / 1000.0).floor() as i64)
        .unwrap_or_else(|| format!("GMT{sign}{hours:02}:{minutes:02}"));
    #[cfg(not(feature = "intl"))]
    let name = format!("GMT{sign}{hours:02}:{minutes:02}");
    format!("GMT{sign}{hours:02}{minutes:02} ({name})")
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// (year, month0, day, hour, minute, second, millisecond, weekday[0=Sun]).
const WDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn year_str(y: i64) -> String {
    if y < 0 {
        format!("-{:04}", -y)
    } else {
        format!("{y:04}")
    }
}

fn date_str_part(t: f64) -> Option<String> {
    if !t.is_finite() {
        return None;
    }
    let (y, mo, d, _, _, _, _, wd) = ms_to_parts(t);
    Some(format!(
        "{} {} {:02} {}",
        WDAYS[wd as usize],
        MONTHS[mo as usize],
        d,
        year_str(y)
    ))
}
/// The `toTimeString` form of time value `t`: its local time and zone.
fn time_str_part(t: f64) -> Option<String> {
    if !t.is_finite() {
        return None;
    }
    let (_, _, _, h, mi, s, _, _) = ms_to_parts(local_time(t));
    Some(format!("{h:02}:{mi:02}:{s:02} {}", time_zone_string(t)))
}
fn utc_string(t: f64) -> Option<String> {
    if !t.is_finite() {
        return None;
    }
    let (y, mo, d, h, mi, s, _, wd) = ms_to_parts(t);
    Some(format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        WDAYS[wd as usize],
        d,
        MONTHS[mo as usize],
        year_str(y),
        h,
        mi,
        s
    ))
}

fn ms_to_parts(t: f64) -> (i64, i64, i64, i64, i64, i64, i64, i64) {
    let ms = t as i64;
    let days = ms.div_euclid(86_400_000);
    let mut rem = ms.rem_euclid(86_400_000);
    let milli = rem % 1000;
    rem /= 1000;
    let sec = rem % 60;
    rem /= 60;
    let min = rem % 60;
    rem /= 60;
    let hour = rem;
    let (y, m, d) = civil_from_days(days);
    let weekday = (days.rem_euclid(7) + 4) % 7; // 1970-01-01 was a Thursday (4)
    (y, m - 1, d, hour, min, sec, milli, weekday)
}

#[allow(clippy::too_many_arguments)]
/// MakeTime (spec-ordered f64 arithmetic, so precision matches the standard exactly).
fn make_time(h: f64, m: f64, s: f64, ms: f64) -> f64 {
    h.trunc() * 3_600_000.0 + m.trunc() * 60_000.0 + s.trunc() * 1000.0 + ms.trunc()
}

/// MakeDay: the day number for (year, month, date), with month overflow carried into the year.
fn make_day(y: f64, m: f64, d: f64) -> f64 {
    if !y.is_finite() || !m.is_finite() || !d.is_finite() {
        return f64::NAN;
    }
    let (y, m, d) = (y.trunc(), m.trunc(), d.trunc());
    let ym = y + (m / 12.0).floor();
    let mn = m.rem_euclid(12.0);
    // Outside this range no finite time value can exist anyway (the clip is ±8.64e15 ms).
    if !(-300_000.0..=300_000.0).contains(&ym) {
        return f64::NAN;
    }
    days_from_civil(ym as i64, mn as i64 + 1, 1) as f64 + (d - 1.0)
}

fn make_date(day: f64, time: f64) -> f64 {
    let t = day * 86_400_000.0 + time;
    if t.is_finite() {
        t
    } else {
        f64::NAN
    }
}

/// TimeClip: NaN outside ±8.64e15 ms; -0 normalizes to +0.
fn time_clip(t: f64) -> f64 {
    if !t.is_finite() || t.abs() > 8.64e15 {
        f64::NAN
    } else {
        t.trunc() + 0.0
    }
}

fn parts_to_ms(y: i64, mo0: i64, d: i64, h: i64, mi: i64, s: i64, ml: i64) -> f64 {
    // Normalize the month into 0..11 with a year carry so e.g. month 13 rolls over.
    let y = y + mo0.div_euclid(12);
    let mo = mo0.rem_euclid(12);
    let days = days_from_civil(y, mo + 1, d);
    (days * 86_400_000 + h * 3_600_000 + mi * 60_000 + s * 1000 + ml) as f64
}

fn date_ms(i: &mut Interp, this: &Value) -> Result<f64, Value> {
    // thisTimeValue: the receiver must be a Date (carry the internal time slot), else TypeError.
    match &(this).view() {
        ValueView::Obj(o) if o.borrow().internal_slots().contains("#\u{0}date_ms") => {}
        _ => return Err(i.make_error("TypeError", "this is not a Date object")),
    }
    Ok(
        match (ab(i.get_internal_slot(this, "#\u{0}date_ms"))?).view() {
            ValueView::Num(n) => n,
            _ => f64::NAN,
        },
    )
}

fn date_get(i: &mut Interp, this: &Value, sel: u8, local: bool) -> Result<Value, Value> {
    let t = date_ms(i, this)?;
    if t.is_nan() {
        return Ok(Value::Num(f64::NAN));
    }
    let (y, mo, d, h, mi, s, ml, wd) = ms_to_parts(if local { local_time(t) } else { t });
    let v = match sel {
        0 => y,
        1 => mo,
        2 => d,
        3 => wd,
        4 => h,
        5 => mi,
        6 => s,
        _ => ml,
    };
    Ok(Value::Num(v as f64))
}

/// A multi-argument Date setter (setHours, setFullYear, …). Per spec, ALL provided arguments are
/// ToNumber-coerced first, in order, before any field is applied; `start_sel` is the field of the
/// first argument (the field order skips the unused selector 3). `setFullYear`/`setUTCFullYear`
/// (start 0) treat a NaN receiver as the epoch; the time setters leave it NaN.
fn date_set_multi(
    i: &mut Interp,
    this: &Value,
    start_sel: u8,
    args: &[Value],
    n_max: usize,
    local: bool,
) -> Result<Value, Value> {
    const ORDER: [u8; 7] = [0, 1, 2, 4, 5, 6, 7];
    let start_idx = ORDER.iter().position(|&f| f == start_sel).unwrap();
    let count = args.len().clamp(1, n_max);
    // thisTimeValue validation precedes argument coercion (a non-Date receiver throws before any
    // argument's valueOf runs); then coerce every read argument up front, in order.
    let t = date_ms(i, this)?;
    let mut vals = Vec::with_capacity(count);
    for k in 0..count {
        vals.push(ab(i.to_number(&arg(args, k)))?);
    }
    let nan_to_zero = start_sel == 0;
    // A NaN stored time (for setters that don't zero it) yields NaN and leaves [[DateValue]]
    // untouched — so an argument's valueOf side-effect on the receiver persists.
    if t.is_nan() && !nan_to_zero {
        return Ok(Value::Num(f64::NAN));
    }
    let mut any_nan = t.is_nan() && !nan_to_zero;
    // setFullYear: a NaN time is +0 in local time already.
    let base = if t.is_nan() {
        0.0
    } else if local {
        local_time(t)
    } else {
        t
    };
    let (py, pmo, pd, ph, pmi, ps, pml, _) = ms_to_parts(base);
    // Fields default to their current value, held as f64 so an out-of-range assignment overflows
    // to a non-finite MakeDay/MakeTime intermediate (NaN) rather than wrapping an i64.
    let mut f = [
        py as f64, pmo as f64, pd as f64, ph as f64, pmi as f64, ps as f64, pml as f64,
    ];
    for (k, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            any_nan = true;
        }
        let slot = match ORDER[start_idx + k] {
            0 => 0,
            1 => 1,
            2 => 2,
            4 => 3,
            5 => 4,
            6 => 5,
            _ => 6,
        };
        f[slot] = v.trunc();
    }
    let day = make_day(f[0], f[1], f[2]);
    let time = f[3] * 3_600_000.0 + f[4] * 60_000.0 + f[5] * 1000.0 + f[6];
    let ms = if any_nan {
        f64::NAN
    } else {
        let date = make_date(day, time);
        time_clip(if local { utc_time(date) } else { date })
    };
    if let ValueView::Obj(o) = &(this).view() {
        set_internal(o, "#\u{0}date_ms", Value::Num(ms));
    }
    Ok(Value::Num(ms))
}

/// The implementation-specific date formats ECMA-262 #sec-date.parse permits when a string is not
/// in the Date Time String Format: the `toString`/`toUTCString`/`toDateString` renderings and the
/// RFC 5322 section 3.3 date-time with its numeric and section 4.3 obsolete zones (e.g.
/// "Thu, 01 Jan 1970 00:00:00 GMT", "Wed Jul 28 1993 14:39:07 GMT-0600 (…)", "… 10:00 PST"),
/// plus the slash dates other engines read. NaN for anything else.
fn parse_rfc(s: &str) -> f64 {
    const MONTHS: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let (mut year, mut month, mut day): (Option<i64>, Option<i64>, Option<i64>) =
        (None, None, None);
    let (mut hh, mut mm, mut ss) = (0i64, 0i64, 0i64);
    let mut offset: i64 = 0; // minutes east of UTC
    let mut zoned = false;
    let mut got_time = false;
    for tok in s.split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')')) {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let low = tok.to_lowercase();
        // RFC 5322 section 4.3 obs-zone: "UT"/"GMT" and the North American zone names.
        let obs_zone = match low.as_str() {
            "ut" | "gmt" => Some(0),
            "edt" => Some(-4 * 60),
            "est" | "cdt" => Some(-5 * 60),
            "cst" | "mdt" => Some(-6 * 60),
            "mst" | "pdt" => Some(-7 * 60),
            "pst" => Some(-8 * 60),
            _ => None,
        };
        if let Some(minutes) = obs_zone {
            if got_time {
                offset = minutes;
                zoned = true;
                continue;
            }
        }
        // Slash dates, as in other engines: year first (`2027/01/31`) or the
        // US month/day/year order (`1/31/2027`).
        // A first number above 31 can only be the year (`50/1/31`).
        if tok.contains('/') && year.is_none() && month.is_none() {
            let fields: Vec<&str> = tok.split('/').collect();
            let parts: Vec<i64> = match fields.iter().map(|p| p.parse()).collect() {
                Ok(parts) => parts,
                Err(_) => return f64::NAN,
            };
            let (y, mo, d, y_field) = match parts[..] {
                [y, mo, d] if fields[0].len() >= 3 || y > 31 => (y, mo, d, fields[0]),
                [mo, d, y] => (y, mo, d, fields[2]),
                _ => return f64::NAN,
            };
            if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
                return f64::NAN;
            }
            (year, month, day) = (Some(expand_year(y_field, y)), Some(mo - 1), Some(d));
            continue;
        }
        if let Some(idx) = MONTHS.iter().position(|m| low.starts_with(m)) {
            month = Some(idx as i64);
        } else if got_time
            && matches!(tok.as_bytes().first(), Some(b'+' | b'-'))
            && tok.len() == 5
            && tok[1..].bytes().all(|b| b.is_ascii_digit())
        {
            // RFC 2822 #section-3.3 zone: `+hhmm`/`-hhmm` after the time
            // (`-0000` is UTC), not a year.
            let mag: i64 =
                tok[1..3].parse::<i64>().unwrap_or(0) * 60 + tok[3..5].parse::<i64>().unwrap_or(0);
            offset = if tok.starts_with('-') { -mag } else { mag };
            zoned = true;
        } else if tok.contains(':') && !got_time {
            let mut p = tok.split(':');
            hh = p.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            mm = p.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            ss = p.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            got_time = true;
        } else if let Ok(n) = tok.parse::<i64>() {
            if (tok.len() >= 4 || n > 31 || day.is_some()) && year.is_some() {
                // A second year (`may 1999 1999`).
                return f64::NAN;
            }
            if tok.len() >= 4 || n > 31 || day.is_some() {
                year = Some(expand_year(tok, n));
            } else if (1..=31).contains(&n) {
                day = Some(n);
            } else {
                return f64::NAN;
            }
        } else if matches!(low.as_str(), "z" | "ut" | "utc") {
            zoned = true;
        } else if low.starts_with("gmt") || low.starts_with('+') || low.starts_with('-') {
            zoned = true;
            let rest = low.trim_start_matches("gmt");
            let sign = rest.chars().next();
            let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 4 {
                let oh: i64 = digits[..2].parse().unwrap_or(0);
                let om: i64 = digits[2..4].parse().unwrap_or(0);
                let mag = oh * 60 + om;
                offset = if sign == Some('-') { -mag } else { mag };
            }
        }
    }
    match (year, month, day) {
        // A string without a zone is local time, as in other engines.
        (Some(y), Some(mo), Some(d)) if !zoned => utc_time(parts_to_ms(y, mo, d, hh, mm, ss, 0)),
        (Some(y), Some(mo), Some(d)) => {
            parts_to_ms(y, mo, d, hh, mm, ss, 0) - (offset as f64) * 60000.0
        }
        _ => f64::NAN,
    }
}

/// A one- or two-digit year in the implementation-specific formats, as other
/// engines read it: 0–49 are 2000–2049 and 50–99 are 1950–1999.
fn expand_year(field: &str, y: i64) -> i64 {
    match y {
        0..=49 if field.len() <= 2 => 2000 + y,
        50..=99 if field.len() <= 2 => 1900 + y,
        _ => y,
    }
}

/// ECMA-262 #sec-date.parse: the Date Time String Format first, then the
/// implementation-specific formats every browser engine accepts.
fn parse_date_string(s: &str) -> f64 {
    let v = parse_iso(s);
    if v.is_nan() {
        parse_rfc(s)
    } else {
        v
    }
}

fn parse_iso(s: &str) -> f64 {
    let s = s.trim();
    // The 'T' separator demands strict ISO component widths; the web-reality space separator
    // (SpiderMonkey's NOTE-datetime extension) is lenient (1-2 digit fields, `±hh` offsets).
    let (date_part, time_part, strict) = match s.split_once('T') {
        Some((d, t)) => (d, Some(t), true),
        None => match s.split_once(' ') {
            Some((d, t)) => (d, Some(t), false),
            None => (s, None, true),
        },
    };
    // Extended years carry an explicit sign and six digits (`+275760-…`); "-000000" is rejected.
    let (neg_year, signed_year, date_part) = match date_part.strip_prefix('-') {
        Some(r) => (true, true, r),
        None => match date_part.strip_prefix('+') {
            Some(r) => (false, true, r),
            None => (false, false, date_part),
        },
    };
    let mut dp = date_part.splitn(3, '-');
    let ystr = dp.next().unwrap_or("");
    if !ystr.bytes().all(|b| b.is_ascii_digit()) || ystr.is_empty() {
        return f64::NAN;
    }
    if strict {
        let want = if signed_year { 6 } else { 4 };
        if ystr.len() != want {
            return f64::NAN;
        }
    }
    if neg_year && ystr.bytes().all(|b| b == b'0') {
        return f64::NAN;
    }
    let y: i64 = match ystr.parse::<i64>() {
        Ok(v) => {
            if neg_year {
                -v
            } else {
                v
            }
        }
        Err(_) => return f64::NAN,
    };
    let field = |txt: Option<&str>, strict: bool| -> Option<i64> {
        let t = txt?;
        if t.is_empty() || !t.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if strict && t.len() != 2 {
            return None;
        }
        if !strict && t.len() > 2 {
            return None;
        }
        t.parse().ok()
    };
    let (mo_txt, d_txt) = (dp.next(), dp.next());
    let mo = match mo_txt {
        None => 1,
        some => match field(some, strict) {
            Some(v) => v,
            None => return f64::NAN,
        },
    };
    let d = match d_txt {
        None => 1,
        some => match field(some, strict) {
            Some(v) => v,
            None => return f64::NAN,
        },
    };
    let (mut h, mut mi, mut sec, mut ml) = (0i64, 0i64, 0i64, 0i64);
    let mut offset_min: Option<i64> = None;
    if let Some(tp) = time_part {
        if tp.is_empty() {
            return f64::NAN;
        }
        // Split a trailing zone: 'Z', or ±hh[:mm] / ±hhmm (strict requires ±hh:mm).
        let (hms_frac, zone) = if let Some(rest) = tp.strip_suffix('Z') {
            (rest, Some(0i64))
        } else if let Some(pos) = tp.rfind(['+', '-']) {
            let (a, z) = tp.split_at(pos);
            let sign = if z.starts_with('-') { -1 } else { 1 };
            let z = &z[1..];
            let (zh, zm) = if let Some((zh, zm)) = z.split_once(':') {
                (zh, zm)
            } else if z.len() == 4 {
                if strict {
                    return f64::NAN;
                }
                z.split_at(2)
            } else if z.len() == 2 {
                if strict {
                    return f64::NAN;
                }
                (z, "")
            } else {
                return f64::NAN;
            };
            if zh.len() != 2
                || !(zm.is_empty() || zm.len() == 2)
                || !zh.bytes().all(|b| b.is_ascii_digit())
                || !zm.bytes().all(|b| b.is_ascii_digit())
            {
                return f64::NAN;
            }
            let mins = zh.parse::<i64>().unwrap_or(0) * 60 + zm.parse::<i64>().unwrap_or(0);
            (a, Some(sign * mins))
        } else {
            (tp, None)
        };
        let (hms, frac) = match hms_frac.split_once('.') {
            Some((a, b)) => (a, Some(b)),
            None => (hms_frac, None),
        };
        let mut parts = hms.splitn(3, ':');
        h = match field(parts.next(), strict) {
            Some(v) => v,
            None => return f64::NAN,
        };
        mi = match parts.next() {
            // A lone hour ("T11" / "1997-03-08 11") parses as HH:00 in both forms.
            None => 0,
            some => match field(some, strict) {
                Some(v) => v,
                None => return f64::NAN,
            },
        };
        sec = match parts.next() {
            None => 0,
            some => match field(some, strict) {
                Some(v) => v,
                None => return f64::NAN,
            },
        };
        if let Some(f) = frac {
            if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
                return f64::NAN;
            }
            let f3: String = f
                .chars()
                .take(3)
                .chain(std::iter::repeat('0'))
                .take(3)
                .collect();
            ml = f3.parse().unwrap_or(0);
        }
        offset_min = zone;
    }
    let base = parts_to_ms(y, mo - 1, d, h, mi, sec, ml);
    // #sec-date-time-string-format: without an offset, a date-only form is UTC
    // and a date-time form is local time.
    let adjusted = match offset_min {
        Some(mins) => base - (mins as f64) * 60_000.0,
        None if time_part.is_some() => utc_time(base),
        None => base,
    };
    time_clip(adjusted)
}

pub(super) fn date_to_string(t: f64) -> String {
    match (date_str_part(local_time(t)), time_str_part(t)) {
        (Some(d), Some(tm)) => format!("{d} {tm}"),
        _ => "Invalid Date".to_string(),
    }
}

fn date_ctor(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    // Called as a function (no `new`), Date ignores its arguments and returns the
    // current time as a string.
    if !matches!((i.new_target).view(), ValueView::Obj(_)) {
        return Ok(Value::lstr(date_to_string(now_ms(i))));
    }
    let ms = match args.len() {
        0 => now_ms(i),
        1 => {
            let __value_match_original = &args[0];
            match &(__value_match_original).view() {
                // A Date argument clones its time value directly (no valueOf call).
                ValueView::Obj(o) if o.borrow().internal_slots().contains("#\u{0}date_ms") => {
                    match (o
                        .borrow()
                        .internal_slots()
                        .get("#\u{0}date_ms")
                        .map(|p| p.value()))
                    .map(|__value| (__value).view())
                    {
                        Some(ValueView::Num(n)) => n,
                        _ => f64::NAN,
                    }
                }
                _ => {
                    let v = __value_match_original;
                    {
                        let prim = ab(i.to_primitive(v, crate::eval::Hint::Default))?;
                        match prim.view() {
                            // #sec-date-constructor step 4.b.ii: parse exactly as
                            // Date.parse does, then TimeClip.
                            ValueView::Str(s) => time_clip(parse_date_string(&s)),
                            _ => time_clip(ab(i.to_number(&prim))?),
                        }
                    }
                }
            }
        }
        _ => {
            // All components coerce as f64 (NaN propagates through MakeDay/MakeTime).
            let mut y = ab(i.to_number(&args[0]))?;
            if !y.is_nan() && (0.0..=99.0).contains(&y.trunc()) {
                y = 1900.0 + y.trunc();
            }
            let read = |i: &mut Interp, k: usize, dflt: f64| -> Result<f64, Value> {
                if args.len() > k {
                    ab(i.to_number(&args[k]))
                } else {
                    Ok(dflt)
                }
            };
            let mo = read(i, 1, 0.0)?;
            let d = read(i, 2, 1.0)?;
            let h = read(i, 3, 0.0)?;
            let mi = read(i, 4, 0.0)?;
            let sec = read(i, 5, 0.0)?;
            let ml = read(i, 6, 0.0)?;
            time_clip(utc_time(make_date(
                make_day(y, mo, d),
                make_time(h, mi, sec, ml),
            )))
        }
    };
    let obj = new_from_ctor(i, "Date")?;
    set_internal(&obj, "#\u{0}date_ms", Value::Num(ms));
    Ok(Value::Obj(obj))
}

fn iso_string(t: f64) -> Option<String> {
    if !t.is_finite() {
        return None;
    }
    let (y, mo, d, h, mi, s, ml, _) = ms_to_parts(t);
    // Years outside 0..=9999 use the signed six-digit extended form.
    let ys = if (0..=9999).contains(&y) {
        format!("{y:04}")
    } else if y < 0 {
        format!("-{:06}", -y)
    } else {
        format!("+{y:06}")
    };
    Some(format!(
        "{ys}-{:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{ml:03}Z",
        mo + 1
    ))
}

pub(super) fn install_date(it: &mut Interp) {
    let proto = Object::new(Some(it.object_proto.clone()));
    Rc::make_mut(&mut it.extra_protos).insert("Date", proto.clone());

    it.def_method(&proto, "getTime", 0, |i, this, _| {
        Ok(Value::Num(date_ms(i, &this)?))
    });
    it.def_method(&proto, "valueOf", 0, |i, this, _| {
        Ok(Value::Num(date_ms(i, &this)?))
    });
    it.def_method(&proto, "setTime", 1, |i, this, a| {
        date_ms(i, &this)?; // thisTimeValue brand check
        let v = time_clip(ab(i.to_number(&arg(a, 0)))?);
        if let ValueView::Obj(o) = &(this).view() {
            set_internal(o, "#\u{0}date_ms", Value::Num(v));
        }
        Ok(Value::Num(v))
    });
    it.def_method(&proto, "getTimezoneOffset", 0, |i, this, _| {
        let t = date_ms(i, &this)?;
        // (t − LocalTime(t)) / msPerMinute.
        Ok(Value::Num(if t.is_nan() {
            f64::NAN
        } else {
            -offset_ms(t) / 60_000.0
        }))
    });
    it.def_method(&proto, "toTemporalInstant", 0, |i, this, _| {
        // RequireInternalSlot([[DateValue]]) then a Temporal.Instant at ms×10^6 ns.
        let ms = date_ms(i, &this)?;
        crate::temporal::instant_from_epoch_ms(i, ms)
    });
    // Local accessors read LocalTime(t); the UTC ones the time value itself.
    for (name, sel) in [
        ("getFullYear", 0u8),
        ("getMonth", 1),
        ("getDate", 2),
        ("getDay", 3),
        ("getHours", 4),
        ("getMinutes", 5),
        ("getSeconds", 6),
        ("getMilliseconds", 7),
    ] {
        let utc = format!("getUTC{}", &name[3..]);
        macro_rules! getters {
            ($sel:literal) => {{
                it.def_method(&proto, name, 0, |i, this, _| date_get(i, &this, $sel, true));
                it.def_method(&proto, &utc, 0, |i, this, _| {
                    date_get(i, &this, $sel, false)
                });
            }};
        }
        match sel {
            0 => getters!(0),
            1 => getters!(1),
            2 => getters!(2),
            3 => getters!(3),
            4 => getters!(4),
            5 => getters!(5),
            6 => getters!(6),
            _ => getters!(7),
        }
    }
    it.def_method(&proto, "setFullYear", 3, |i, this, a| {
        date_set_multi(i, &this, 0, a, 3, true)
    });
    // Annex B legacy getYear/setYear (years offset from 1900).
    it.def_method(&proto, "getYear", 0, |i, this, _| {
        let f = ab(i.get_member_key(&this, "getFullYear"))?;
        let yv = ab(i.call(f, this.clone(), &[]))?;
        let y = ab(i.to_number(&yv))?;
        Ok(Value::Num(if y.is_nan() { f64::NAN } else { y - 1900.0 }))
    });
    it.def_method(&proto, "setYear", 1, |i, this, a| {
        let y = ab(i.to_number(&arg(a, 0)))?;
        let full = if y.is_nan() {
            f64::NAN
        } else {
            let yi = y.trunc() as i64;
            if (0..=99).contains(&yi) {
                1900.0 + yi as f64
            } else {
                y
            }
        };
        date_set_multi(i, &this, 0, &[Value::Num(full)], 1, true)?;
        // TimeClip: setYear reports the stored (possibly NaN) time value.
        date_ms(i, &this).map(Value::Num)
    });
    it.def_method(&proto, "setMonth", 2, |i, this, a| {
        date_set_multi(i, &this, 1, a, 2, true)
    });
    it.def_method(&proto, "setDate", 1, |i, this, a| {
        date_set_multi(i, &this, 2, a, 1, true)
    });
    it.def_method(&proto, "setHours", 4, |i, this, a| {
        date_set_multi(i, &this, 4, a, 4, true)
    });
    it.def_method(&proto, "setMinutes", 3, |i, this, a| {
        date_set_multi(i, &this, 5, a, 3, true)
    });
    it.def_method(&proto, "setSeconds", 2, |i, this, a| {
        date_set_multi(i, &this, 6, a, 2, true)
    });
    it.def_method(&proto, "setMilliseconds", 1, |i, this, a| {
        date_set_multi(i, &this, 7, a, 1, true)
    });
    it.def_method(&proto, "setUTCFullYear", 3, |i, this, a| {
        date_set_multi(i, &this, 0, a, 3, false)
    });
    it.def_method(&proto, "setUTCMonth", 2, |i, this, a| {
        date_set_multi(i, &this, 1, a, 2, false)
    });
    it.def_method(&proto, "setUTCDate", 1, |i, this, a| {
        date_set_multi(i, &this, 2, a, 1, false)
    });
    it.def_method(&proto, "setUTCHours", 4, |i, this, a| {
        date_set_multi(i, &this, 4, a, 4, false)
    });
    it.def_method(&proto, "setUTCMinutes", 3, |i, this, a| {
        date_set_multi(i, &this, 5, a, 3, false)
    });
    it.def_method(&proto, "setUTCSeconds", 2, |i, this, a| {
        date_set_multi(i, &this, 6, a, 2, false)
    });
    it.def_method(&proto, "setUTCMilliseconds", 1, |i, this, a| {
        date_set_multi(i, &this, 7, a, 1, false)
    });
    it.def_method(&proto, "toISOString", 0, |i, this, _| {
        let t = date_ms(i, &this)?;
        match iso_string(t) {
            Some(s) => Ok(Value::lstr(s)),
            None => Err(i.make_error("RangeError", "Invalid time value")),
        }
    });
    it.def_method(&proto, "toJSON", 1, |i, this, _| {
        // Generic: ToObject, ToPrimitive(number); a non-finite time is null; else Invoke toISOString.
        let o = to_object_arg(i, this.clone(), "Date.prototype.toJSON")?;
        let ov = Value::Obj(o);
        let tv = ab(i.to_primitive(&ov, crate::eval::Hint::Number))?;
        if let ValueView::Num(n) = &(tv).view() {
            if !n.is_finite() {
                return Ok(Value::Null);
            }
        }
        let iso = ab(i.get_member_key(&ov, "toISOString"))?;
        if !iso.is_callable() {
            return Err(i.make_error("TypeError", "toISOString is not callable"));
        }
        ab(i.call(iso, ov, &[]))
    });
    it.def_method(&proto, "toString", 0, |i, this, _| {
        let t = date_ms(i, &this)?;
        Ok(Value::lstr(date_to_string(t)))
    });
    it.def_method(&proto, "toDateString", 0, |i, this, _| {
        let t = date_ms(i, &this)?;
        Ok(Value::lstr(
            date_str_part(local_time(t)).unwrap_or_else(|| "Invalid Date".to_string()),
        ))
    });
    it.def_method(&proto, "toTimeString", 0, |i, this, _| {
        let t = date_ms(i, &this)?;
        Ok(Value::lstr(
            time_str_part(t).unwrap_or_else(|| "Invalid Date".to_string()),
        ))
    });
    it.def_method(&proto, "toUTCString", 0, |i, this, _| {
        let t = date_ms(i, &this)?;
        Ok(Value::lstr(
            utc_string(t).unwrap_or_else(|| "Invalid Date".to_string()),
        ))
    });
    // toGMTString IS toUTCString (the very same function object).
    let utc = proto.borrow().props.get("toUTCString").cloned();
    if let Some(p) = utc {
        proto.borrow_mut().props.insert("toGMTString", p);
    }
    // toLocale* route through Intl.DateTimeFormat (which exists now).
    it.def_method(&proto, "toLocaleString", 0, |i, this, args| {
        let t = date_ms(i, &this)?;
        if !t.is_finite() {
            return Ok(Value::lstr("Invalid Date"));
        }
        // ToDateTimeOptions(options, "any", "all"): default to date AND time unless the caller
        // already requested a date or time component (or a dateStyle/timeStyle).
        let opts = date_all_default(i, &arg(args, 1))?;
        intl_delegate(
            i,
            "DateTimeFormat",
            arg(args, 0),
            opts,
            "format",
            &[Value::Num(t)],
        )
    });
    it.def_method(&proto, "toLocaleDateString", 0, |i, this, args| {
        let t = date_ms(i, &this)?;
        if !t.is_finite() {
            return Ok(Value::lstr("Invalid Date"));
        }
        let opts = date_style_default(i, &arg(args, 1), true)?;
        intl_delegate(
            i,
            "DateTimeFormat",
            arg(args, 0),
            opts,
            "format",
            &[Value::Num(t)],
        )
    });
    it.def_method(&proto, "toLocaleTimeString", 0, |i, this, args| {
        let t = date_ms(i, &this)?;
        if !t.is_finite() {
            return Ok(Value::lstr("Invalid Date"));
        }
        let opts = date_style_default(i, &arg(args, 1), false)?;
        intl_delegate(
            i,
            "DateTimeFormat",
            arg(args, 0),
            opts,
            "format",
            &[Value::Num(t)],
        )
    });

    let ctor = it.make_native("Date", 7, date_ctor);
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto.clone()), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
    it.def_method(&ctor, "now", 0, |i, _t, _a| Ok(Value::Num(now_ms(i))));
    it.def_method(&ctor, "parse", 1, |i, _t, a| {
        let s = ab(i.to_string(&arg(a, 0)))?;
        Ok(Value::Num(parse_date_string(&s)))
    });
    it.def_method(&ctor, "UTC", 7, |i, _t, a| {
        // The year is always read; later components only if supplied. Coerce all reads first; any
        // non-finite component makes the whole result NaN.
        let count = a.len().clamp(1, 7);
        let defaults = [f64::NAN, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut vals = defaults;
        let mut nan = false;
        for k in 0..count {
            let v = ab(i.to_number(&arg(a, k)))?;
            if !v.is_finite() {
                nan = true;
            }
            vals[k] = v;
        }
        let _ = nan;
        let mut y = vals[0];
        if !y.is_nan() && (0.0..=99.0).contains(&y.trunc()) {
            y = 1900.0 + y.trunc();
        }
        Ok(Value::Num(time_clip(make_date(
            make_day(y, vals[1], vals[2]),
            make_time(vals[3], vals[4], vals[5], vals[6]),
        ))))
    });
    // Date.prototype[@@toPrimitive]: "number" hint uses valueOf, "string"/"default" use toString.
    let prim = it.make_native("[Symbol.toPrimitive]", 1, |i, this, a| {
        if !matches!((this).view(), ValueView::Obj(_)) {
            return Err(i.make_error(
                "TypeError",
                "Date.prototype[Symbol.toPrimitive] on non-object",
            ));
        }
        // The hint must be a primitive String; OrdinaryToPrimitive then tries the two methods in
        // order, returning the first non-object result.
        let hint = match (arg(a, 0)).view() {
            ValueView::Str(s) => s.to_string(),
            _ => return Err(i.make_error("TypeError", "invalid Symbol.toPrimitive hint")),
        };
        let methods: &[&str] = match hint.as_str() {
            "number" => &["valueOf", "toString"],
            "string" | "default" => &["toString", "valueOf"],
            _ => return Err(i.make_error("TypeError", "invalid Symbol.toPrimitive hint")),
        };
        for m in methods {
            let f = ab(i.get_member_key(&this, m))?;
            if f.is_callable() {
                let r = ab(i.call(f, this.clone(), &[]))?;
                if !matches!((r).view(), ValueView::Obj(_)) {
                    return Ok(r);
                }
            }
        }
        Err(i.make_error("TypeError", "cannot convert Date to a primitive value"))
    });
    if let Some(key) = well_known_key(it, "toPrimitive") {
        // @@toPrimitive is non-writable (unlike ordinary builtin methods).
        proto
            .borrow_mut()
            .props
            .insert(key, Property::data(Value::Obj(prim), false, false, true));
    }
    set_builtin(&it.global, "Date", Value::Obj(ctor));
}

#[cfg(test)]
mod windows_zone_tests {
    use super::{fixed_offset_zone, offset_zone_minutes, windows_time_zone};

    #[test]
    fn windows_zones_map_through_cldr_to_primary_identifiers() {
        // UTS #35 Part 4 Windows Zones: the territory's first listed zone, else territory 001's;
        // the result is the IANA primary identifier (CLDR's Asia/Calcutta is a link).
        let zone = |key, region| windows_time_zone(key, region, None);
        assert_eq!(
            zone("Pacific Standard Time", None),
            Some("America/Los_Angeles")
        );
        assert_eq!(
            zone("Pacific Standard Time", Some("CA")),
            Some("America/Vancouver")
        );
        assert_eq!(
            zone("Pacific Standard Time", Some("MX")),
            Some("America/Los_Angeles")
        );
        assert_eq!(
            zone("Pacific Standard Time", Some("FR")),
            Some("America/Los_Angeles")
        );
        assert_eq!(
            zone("India Standard Time", Some("IN")),
            Some("Asia/Kolkata")
        );
        assert_eq!(
            zone("W. Europe Standard Time", Some("DE")),
            Some("Europe/Berlin")
        );
        assert_eq!(
            zone("W. Europe Standard Time", Some("CH")),
            Some("Europe/Zurich")
        );
        assert_eq!(
            zone("SA Pacific Standard Time", Some("ZZ")),
            Some("Etc/GMT+5")
        );
        assert_eq!(zone("UTC", None), Some("UTC"));
        assert_eq!(zone("No Such Standard Time", Some("US")), None);
    }

    #[test]
    fn disabled_daylight_adjustment_is_a_fixed_offset_zone() {
        // Windows reports the bias west of UTC; IANA's Etc/GMT zones invert the sign too.
        assert_eq!(
            windows_time_zone("Pacific Standard Time", Some("US"), Some(480)),
            Some("Etc/GMT+8")
        );
        assert_eq!(
            windows_time_zone("AUS Eastern Standard Time", None, Some(-600)),
            Some("Etc/GMT-10")
        );
        assert_eq!(
            windows_time_zone("GMT Standard Time", None, Some(0)),
            Some("UTC")
        );
        assert_eq!(fixed_offset_zone(-210), "-03:30");
        assert_eq!(fixed_offset_zone(570), "+09:30");
        assert_eq!(offset_zone_minutes("-03:30"), Some(-210));
        assert_eq!(offset_zone_minutes("+09:30"), Some(570));
        assert_eq!(offset_zone_minutes("America/New_York"), None);
    }
}
