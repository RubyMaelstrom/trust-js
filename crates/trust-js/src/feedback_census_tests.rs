//! The feedback census (W4 step 2): the VM's dispatch counts by pc and the production caches'
//! snapshot, read back from the census text.
use crate::bytecode::Tier;
use crate::Engine;

/// The census text after running `source` on a fresh engine of `tier` (threshold 0).
fn census(tier: Tier, source: &str) -> String {
    crate::feedback_census::enable_for_test();
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
        .eval(source, false)
        .expect("the census fixture parses");
    crate::feedback_census::take_for_test()
}

/// The `O` lines (kind, VM dispatches, state, operation) of the chunk whose label starts with
/// `name@`, and that chunk's `C` line fields (chunks merged, chunks with native code).
fn chunk_ops(text: &str, name: &str) -> ((u32, u32), Vec<(u8, u64, u32, String)>) {
    let mut key = None;
    let mut chunk = (0, 0);
    let mut ops = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        match fields[0] {
            "C" if fields[5].starts_with(&format!("{name}@")) && fields[2] == "0" => {
                key = Some(fields[1].to_string());
                chunk = (fields[3].parse().unwrap(), fields[4].parse().unwrap());
            }
            "O" if key.as_deref() == Some(fields[1]) && fields[2] == "0" => ops.push((
                fields[4].parse().unwrap(),
                fields[5].parse().unwrap(),
                u32::from_str_radix(fields[6], 16).unwrap(),
                fields[7].to_string(),
            )),
            _ => {}
        }
    }
    assert!(key.is_some(), "no chunk for {name} in:\n{text}");
    (chunk, ops)
}

const SOURCE: &str = "
    function f(o) { return o.x + 1; }
    var total = 0;
    for (var i = 0; i < 10; i++) total += f({x: i});
    total";

#[test]
fn the_vm_counts_each_operation_and_the_named_cache_shows_its_way() {
    let text = census(Tier::Bytecode, SOURCE);
    let ((chunks, native), ops) = chunk_ops(&text, "f");
    assert_eq!((chunks, native), (1, 0), "{text}");
    let named: Vec<_> = ops.iter().filter(|op| op.0 == 1).collect();
    assert_eq!(named.len(), 1, "one named load in f: {ops:?}");
    assert_eq!(named[0].1, 10, "ten dispatches of o.x: {ops:?}");
    assert_eq!(named[0].2 & 7, 1, "one filled way: {ops:?}");
    let arithmetic: Vec<_> = ops.iter().filter(|op| op.0 == 7).collect();
    assert_eq!(arithmetic.len(), 1, "one addition in f: {ops:?}");
    assert_eq!(arithmetic[0].1, 10, "{ops:?}");
}

#[test]
fn a_native_chunk_is_marked_and_its_vm_dispatches_stay_uncounted() {
    let text = census(Tier::Jit, SOURCE);
    let ((_, native), ops) = chunk_ops(&text, "f");
    if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        assert_eq!(native, 1, "{text}");
        assert!(
            ops.iter().all(|op| op.1 == 0),
            "native code dispatches nothing through the VM's loop: {ops:?}"
        );
    }
    let named: Vec<_> = ops.iter().filter(|op| op.0 == 1).collect();
    assert_eq!(named.len(), 1, "{ops:?}");
    assert_eq!(
        named[0].2 & 7,
        1,
        "the native code's fill shows too: {ops:?}"
    );
}

#[test]
fn the_kinds_cover_comparisons_names_and_allocations() {
    let text = census(
        Tier::Bytecode,
        "
        var g = 1;
        function h(a, b) { var o = {p: a}; return (a < b) && typeof o === 'object' ? g : 0; }
        for (var i = 0; i < 4; i++) h(i, 2);",
    );
    let (_, ops) = chunk_ops(&text, "h");
    let kinds: std::collections::BTreeSet<u8> = ops.iter().map(|op| op.0).collect();
    for kind in [
        10,
        crate::bytecode::census::CENSUS_COMPARISON,
        crate::bytecode::census::CENSUS_NAME,
    ] {
        assert!(kinds.contains(&kind), "kind {kind} in {ops:?}");
    }
    let allocation = ops.iter().find(|op| op.0 == 10).expect("the literal");
    assert_eq!(allocation.2 & 1, 1, "the literal's image is made: {ops:?}");
}
