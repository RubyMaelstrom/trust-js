//! CreateByteDataBlock (ECMA-262 §6.2.9.1) throws a RangeError when the data block cannot be
//! allocated. Run the engine shell under an address-space limit so allocation really fails: an
//! infallible allocation aborted the whole process instead.
#![cfg(target_os = "linux")]

use std::io::Write;
use std::process::{Command, Stdio};

const SCRIPT: &[u8] = br#"
    function fill(make) {
        var kept = [], outcome = 'no failure';
        try { for (var k = 0; k < 4096; k++) kept.push(make()); }
        catch (e) { outcome = e.name + ': ' + e.message; }
        return outcome + ' after ' + (kept.length > 0 ? 'some' : 'none');
    }
    var mib = 1024 * 1024;
    // The first fill's buffers are unreachable once it returns; a failed allocation collects
    // them before giving up, so the second fill succeeds as often again.
    console.log(fill(function () { return new Uint8Array(8 * mib); }));
    console.log(fill(function () { return new Float64Array(mib); }));
"#;

#[test]
fn exhausted_memory_throws_range_error_for_array_buffers() {
    for tier in ["interp", "bytecode", "jit"] {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("ulimit -v 196608 && exec \"$0\" --tier=\"$1\"")
            .arg(env!("CARGO_BIN_EXE_lumen"))
            .arg(tier)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(SCRIPT).unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{tier}: {:?}\n{stdout}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            stdout.lines().collect::<Vec<_>>(),
            ["RangeError: Array buffer allocation failed after some"; 2],
            "{tier}"
        );
    }
}

/// A new ArrayBuffer's data comes from the allocator already zeroed, so a large one is fresh
/// pages that become resident only when written (asm.js and emscripten heaps of hundreds of MiB
/// are allocated up front). Filling the block with zeros made all of it resident at once.
#[test]
fn large_array_buffers_stay_lazily_zeroed() {
    const SCRIPT: &[u8] = br#"
        var buffer = new ArrayBuffer(256 * 1024 * 1024);
        new Uint8Array(buffer)[12345678] = 1;
        var until = Date.now() + 1500;
        while (Date.now() < until) {}
    "#;
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(SCRIPT).unwrap();
    // The peak resident set (VmHWM) only grows, so any reading while the script spins after
    // the allocation shows whether the block became resident.
    let status = format!("/proc/{}/status", child.id());
    let mut peak_kib = 0u64;
    while child.try_wait().unwrap().is_none() {
        let hwm = std::fs::read_to_string(&status).ok().and_then(|text| {
            let line = text.lines().find(|line| line.starts_with("VmHWM:"))?;
            line.split_whitespace().nth(1)?.parse::<u64>().ok()
        });
        peak_kib = peak_kib.max(hwm.unwrap_or(0));
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(peak_kib > 0, "no reading of the child's memory");
    assert!(peak_kib < 128 * 1024, "peak resident set {peak_kib} KiB");
}
