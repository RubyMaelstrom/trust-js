//! The engine shell's `performance.now()` (docs/v8-chase-p3-plan-2026-11-08.txt, "The fixtures'
//! timer"), as d8's and JSC's shells define one: a plain, non-enumerable `performance` object
//! whose `now()` is the milliseconds, with a fraction, on the monotonic clock since the shell
//! started.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(tier: &str, script: &str) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_trust-js"))
        .arg(format!("--tier={tier}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{tier}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

const TIERS: [&str; 3] = ["interp", "bytecode", "jit"];

#[test]
fn shell_performance_is_a_plain_object_with_now() {
    let script = r#"
        var d = Object.getOwnPropertyDescriptor(globalThis, "performance");
        var n = Object.getOwnPropertyDescriptor(performance, "now");
        console.log([typeof performance, Object.getPrototypeOf(performance) === Object.prototype,
            d.enumerable, d.writable, d.configurable, typeof performance.now,
            performance.now.length, performance.now.name, n.enumerable,
            Object.keys(globalThis).indexOf("performance")].join(" "));
    "#;
    for tier in TIERS {
        assert_eq!(
            run(tier, script).trim(),
            "object true false true true function 0 now false -1",
            "{tier}"
        );
    }
}

/// Monotonic, finer than a millisecond, measured from the shell's start, and in milliseconds:
/// a busy wait of 20 ms by `Date.now()` reads at least 19 ms on it (each `Date.now()` reading
/// loses under a millisecond), and well under a second unless the process is held off the CPU.
#[test]
fn shell_performance_now_is_monotonic_milliseconds_since_start() {
    // `tick` is called hot so that the compiled tiers call the native from compiled code.
    let script = r#"
        function tick() { return performance.now(); }
        var first = tick();
        var prev = first, back = 0, steps = 0, fine = 0, numbers = 0;
        for (var i = 0; i < 10000; i++) {
            var t = tick();
            if (typeof t === "number") numbers++;
            if (t < prev) back++;
            if (t > prev) { steps++; if (t - prev < 1) fine++; }
            prev = t;
        }
        var p0 = performance.now(), d0 = Date.now(), d1 = d0;
        while (d1 - d0 < 20) d1 = Date.now();
        var p1 = performance.now();
        console.log(JSON.stringify({first: first, numbers: numbers, back: back, steps: steps,
            fine: fine, date: d1 - d0, perf: p1 - p0}));
    "#;
    for tier in TIERS {
        let out = run(tier, script);
        let field = |name: &str| -> f64 {
            let key = format!("\"{name}\":");
            let rest =
                &out[out.find(&key).unwrap_or_else(|| panic!("{tier}: {out}")) + key.len()..];
            let end = rest.find([',', '}']).unwrap();
            rest[..end].parse().unwrap()
        };
        let (first, numbers, back, fine) = (
            field("first"),
            field("numbers"),
            field("back"),
            field("fine"),
        );
        let (date, perf) = (field("date"), field("perf"));
        assert!(
            (0.0..60_000.0).contains(&first),
            "{tier}: origin not the shell's start: {out}"
        );
        assert_eq!(numbers, 10_000.0, "{tier}: {out}");
        assert_eq!(back, 0.0, "{tier}: went backwards: {out}");
        assert!(fine > 0.0, "{tier}: no step under a millisecond: {out}");
        assert!(perf >= date - 1.0, "{tier}: not milliseconds: {out}");
        assert!(
            perf < 10.0 * date + 1000.0,
            "{tier}: not milliseconds: {out}"
        );
    }
}
