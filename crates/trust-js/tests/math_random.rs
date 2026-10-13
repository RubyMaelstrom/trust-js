//! ECMA-262 #sec-math.random (local snapshot e28783d5fc9d): randomly seeded
//! independent executions must not replay one process-global fixed sequence.

use std::io::Write;
use std::process::{Command, Stdio};

fn sample(tier: &str) -> String {
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
        .write_all(b"console.log(Array.from({length: 8}, () => Math.random()).join(','));")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = String::from_utf8(output.stdout).unwrap();
    let values: Vec<f64> = output
        .trim()
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    assert_eq!(values.len(), 8);
    assert!(values
        .iter()
        .all(|n| *n >= 0.0 && *n < 1.0 && !n.is_sign_negative()));
    output
}

#[test]
fn math_random_fresh_processes_do_not_replay_a_fixed_seed() {
    for tier in ["interp", "bytecode", "jit"] {
        assert_ne!(
            sample(tier),
            sample(tier),
            "repeated Math.random stream in {tier}"
        );
    }
}
