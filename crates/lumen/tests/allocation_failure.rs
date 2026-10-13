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
