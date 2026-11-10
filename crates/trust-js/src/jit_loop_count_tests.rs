//! Back edges counted toward the checkpoints (P3 stage 2, step 8; the plan's "Step 8, back edges
//! counted toward the checkpoints"), with V8's weight (step 11): with the optimizing tier on, each
//! loop iteration of template code counts its loop's weight toward its chunk's checkpoint word
//! while a checkpoint is pending, up to one below it (`jit::emit_loop_count`), in every loop
//! shape a first stage keeps: a loop run in canonical code at its header's poll, a general region
//! and a linked scan at their back edges, a helper-free rotated loop by the iterations its
//! countdown ran, at its polls, exits and bails. The weight is the loop's share of an entry's K
//! (`opt::ENTRY_WEIGHT`), max(1, round(K x B / L)), B its span (its header to its last back edge)
//! and L the chunk's operations; an entry counts K. With the tier off, or under
//! TRUST_JS_OPT_NO_LOOP_COUNT, the code counts nothing and holds no counting code. That the
//! checkpoints the loops reach compile at the next entry is `opt::tier_tests`'.
//!
//! Each shape's evidence is the test counter of its own code's entries: `regions::
//! executed_regions` for a general region, `TEST_LINKED_SCAN_ENTRIES` and
//! `TEST_NUMERIC_REGION_ENTRIES` for a linked scan and a rotated loop; a loop in canonical code
//! enters none. One entry for a whole run of the loop means its iterations ran in that code. A
//! loop of n iterations counts n weights, its header once more (and once after each of a rotated
//! loop's polls), and its entry K (an entry counts in Rust, `Interp::note_native_entry`).

use crate::bytecode::Tier;
use crate::opt::TestSwitches;
use crate::Engine;

use super::traced_tests::{eval, with_chunk};

/// A loop shape: the global function `name` (defined by `source`), made native by `warm`, whose
/// call `call` runs its loop `iterations` times in that shape's code.
struct Shape {
    what: &'static str,
    name: &'static str,
    source: &'static str,
    warm: &'static str,
    call: &'static str,
    iterations: u32,
}

/// The shapes (see the module's doc).
const SHAPES: [Shape; 4] = [
    // Its 16-element literal is deeper than a region's operand stack (`jit::poll_tests`).
    Shape {
        what: "canonical",
        name: "wide",
        source: "function wide(n) { var sum = 0, row = null; for (var i = 0; i < n; i++) { \
                 row = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]; sum += row[0]; } \
                 return sum; }",
        warm: "wide(3)",
        call: "wide(3000)",
        iterations: 3000,
    },
    // A named read in the loop: no rotated loop admits it. It allocates nothing, so that no
    // collection makes its back edge bail.
    Shape {
        what: "general region",
        name: "region",
        source: "function region(o, n) { var sum = 0; for (var i = 0; i < n; i++) sum += o.v; \
                 return sum; }\n\
                 var holder = {v: 2};",
        warm: "region(holder, 3)",
        call: "region(holder, 3000)",
        iterations: 3000,
    },
    // Compiled after the VM filled the link's cache (see `prepare`); one literal makes every
    // node, so that the cache's one shape is every node's.
    Shape {
        what: "linked scan",
        name: "last",
        source: "function last(next) { var peek; while ((peek = next.link) != null) next = peek; \
                 return next; }\n\
                 function cons(l) { return {link: l}; }\n\
                 var list = null; for (var k = 0; k < 20001; k++) list = cons(list);",
        warm: "last(cons(cons(null)))",
        call: "last(list).link",
        iterations: 20000,
    },
    // Past 16,384 iterations: its countdown's polls and its exit both count.
    Shape {
        what: "rotated loop",
        name: "spin",
        source: "function spin(n) { var sum = 0; for (var i = 0; i < n; i++) sum = (sum + i) | 0; \
                 return sum; }",
        warm: "spin(3)",
        call: "spin(100000)",
        iterations: 100000,
    },
];

/// The entries of each shape's own code a run of its call makes ([`entries`]): one, and a
/// rotated loop's once more after each of its polls.
fn expected_entries(shape: &Shape) -> [u64; 3] {
    match shape.what {
        "canonical" => [0, 0, 0],
        "general region" => [1, 0, 0],
        "linked scan" => [0, 1, 0],
        _ => [
            0,
            0,
            1 + u64::from(shape.iterations / super::REGION_POLL_COUNTDOWN),
        ],
    }
}

/// The tier on, not eager, counting unless `counting` is false.
fn tier(counting: bool) -> TestSwitches {
    TestSwitches {
        on: true,
        no_loop_count: !counting,
        ..Default::default()
    }
}

/// An engine under `switches` in which `shape`'s first stage is made by its warm-up call (a
/// linked scan's after the VM's runs filled its cache, so that it is compiled with the scan),
/// with its checkpoint at `u32::MAX`: pending and never due.
fn prepare(switches: TestSwitches, shape: &Shape) -> Engine {
    switches.set();
    let mut engine = Engine::new();
    engine.set_tier_threshold(0);
    if shape.what == "linked scan" {
        engine.set_tier(Tier::Bytecode);
        eval(&mut engine, shape.source);
        for _ in 0..3 {
            eval(&mut engine, shape.warm);
        }
    } else {
        engine.set_tier(Tier::Jit);
        eval(&mut engine, shape.source);
    }
    engine.set_tier(Tier::Jit);
    eval(&mut engine, shape.warm);
    with_chunk(&mut engine, shape.name, |chunk| {
        assert!(
            chunk.jit.get().flatten().is_some(),
            "{}: native",
            shape.what
        );
        chunk.set_inline_retry_at(u32::MAX);
    });
    engine
}

fn runs(engine: &mut Engine, name: &str) -> u32 {
    with_chunk(engine, name, |chunk| chunk.jit_runs.get())
}

/// Each loop header of `chunk`'s operations with its weight with the tier on (step 11):
/// max(1, round(K x B / L)), B the operations from the header to its last back edge (a jump to
/// a pc at or before its own) and L the chunk's, ascending by header.
fn loop_weights(chunk: &crate::bytecode::Chunk) -> Vec<(usize, u32)> {
    use crate::bytecode::Op;
    let ops = chunk.jit_ops();
    let mut spans = std::collections::BTreeMap::new();
    for (pc, op) in ops.iter().enumerate() {
        if let Op::Jump(target)
        | Op::AbruptJump(target, _)
        | Op::JumpIfFalse(target)
        | Op::JumpIfFalsePeek(target)
        | Op::JumpIfTruePeek(target)
        | Op::JumpIfNotNullishPeek(target) = *op
        {
            let header = target as usize;
            if header <= pc {
                let span = spans.entry(header).or_insert(0usize);
                *span = (*span).max(pc - header + 1);
            }
        }
    }
    let k = u64::from(crate::opt::ENTRY_WEIGHT);
    let len = ops.len() as u64;
    spans
        .into_iter()
        .map(|(header, span)| (header, ((k * span as u64 + len / 2) / len).max(1) as u32))
        .collect()
}

/// The weight of the one loop of the global function `name`.
fn weight(engine: &mut Engine, name: &str) -> u32 {
    let weights = with_chunk(engine, name, loop_weights);
    assert_eq!(weights.len(), 1, "{name}: one loop, {weights:?}");
    weights[0].1
}

/// The entries of each shape's own code on this thread so far: general regions, linked scans and
/// rotated loops.
fn entries() -> [u64; 3] {
    [
        super::regions::executed_regions(),
        super::TEST_LINKED_SCAN_ENTRIES.with(std::cell::Cell::get) as u64,
        super::TEST_NUMERIC_REGION_ENTRIES.with(std::cell::Cell::get) as u64,
    ]
}

/// The native runs a run of `call` counted on `name`'s first stage, and the entries of each
/// shape's own code it made ([`entries`]).
fn counted(engine: &mut Engine, name: &str, call: &str) -> (u32, [u64; 3]) {
    let before = (runs(engine, name), entries());
    eval(engine, call);
    let after = (runs(engine, name), entries());
    (
        after.0 - before.0,
        std::array::from_fn(|k| after.1[k] - before.1[k]),
    )
}

#[test]
fn every_loop_shape_counts_its_iterations_with_the_tier_on() {
    let k = crate::opt::ENTRY_WEIGHT;
    for shape in &SHAPES {
        let mut engine = prepare(tier(true), shape);
        let w = weight(&mut engine, shape.name);
        let (runs, entered) = counted(&mut engine, shape.name, shape.call);
        eprintln!(
            "{}: {runs} runs, weight {w}, entries {entered:?}",
            shape.what
        );
        assert_eq!(entered, expected_entries(shape), "{}", shape.what);
        assert!(w > 1 && w < k, "{}: a share of an entry, {w}", shape.what);
        // The entry's K, and n iterations' weights, the header once more at entry and after
        // each of the rotated loop's polls.
        let polls = shape.iterations / super::REGION_POLL_COUNTDOWN;
        assert!(
            (k + shape.iterations * w..=k + (shape.iterations + polls + 1) * w).contains(&runs),
            "{}: {runs} runs for {} iterations of weight {w}",
            shape.what,
            shape.iterations
        );
    }
}

#[test]
fn loops_count_nothing_with_the_tier_off_or_the_counting_off() {
    // The tier off, an entry counts one, as P2's; on, without the loops' counting, K.
    for (label, switches, entry) in [
        ("off", TestSwitches::default(), 1),
        ("no count", tier(false), crate::opt::ENTRY_WEIGHT),
    ] {
        for shape in &SHAPES {
            let mut engine = prepare(switches, shape);
            let (runs, entered) = counted(&mut engine, shape.name, shape.call);
            assert_eq!(runs, entry, "{label}, {}: the entry alone", shape.what);
            assert_eq!(entered, expected_entries(shape), "{label}, {}", shape.what);
        }
    }
}

#[test]
fn with_no_checkpoint_pending_the_count_stays() {
    // A chunk whose second stage was built, or whose next checkpoint would pass the word: its
    // word is 0.
    for shape in &SHAPES {
        let mut engine = prepare(tier(true), shape);
        with_chunk(&mut engine, shape.name, |chunk| {
            chunk.set_inline_retry_at(0)
        });
        let (runs, _) = counted(&mut engine, shape.name, shape.call);
        assert_eq!(
            runs,
            crate::opt::ENTRY_WEIGHT,
            "{}: the entry alone",
            shape.what
        );
    }
}

/// Two loops, one inside the other, run in canonical code (the 16-element literal is deeper than
/// a region's operand stack).
const NESTED: &str = "function nest(m, n) { var sum = 0, row = null; \
     for (var i = 0; i < m; i++) { sum += i; sum -= i; sum += i; sum -= i; \
     for (var j = 0; j < n; j++) { row = [i, j, i, j, i, j, i, j, i, j, i, j, i, j, i, j]; \
     sum += row[0]; } } return sum; }";

#[test]
fn a_nested_loops_back_edge_charges_its_own_span() {
    // The inner loop's iterations count the inner loop's weight, the outer's its own, larger:
    // the inner header passes m(n + 1) times, the outer m + 1.
    let k = u64::from(crate::opt::ENTRY_WEIGHT);
    let shape = Shape {
        what: "nested",
        name: "nest",
        source: NESTED,
        warm: "nest(2, 2)",
        call: "nest(30, 100)",
        iterations: 3000,
    };
    let mut engine = prepare(tier(true), &shape);
    let weights = with_chunk(&mut engine, "nest", loop_weights);
    let [(outer_h, outer), (inner_h, inner)] = weights[..] else {
        panic!("two loops: {weights:?}")
    };
    assert!(outer_h < inner_h && outer > inner, "{weights:?}");
    let (runs, entered) = counted(&mut engine, "nest", shape.call);
    eprintln!("nested: {runs} runs, weights {weights:?}, entries {entered:?}");
    assert_eq!(entered, [0, 0, 0], "canonical code");
    let (m, n) = (30u64, 100u64);
    let (outer, inner) = (u64::from(outer), u64::from(inner));
    assert_eq!(
        u64::from(runs),
        k + m * (n + 1) * inner + (m + 1) * outer,
        "each back edge its own loop's weight"
    );
}

#[test]
fn the_count_saturates_one_below_the_checkpoint() {
    for shape in &SHAPES {
        // A checkpoint 1,000 runs away: the loop counts to one below it, where the next entry's
        // gate takes it.
        let mut engine = prepare(tier(true), shape);
        let at = runs(&mut engine, shape.name) + 1000;
        with_chunk(&mut engine, shape.name, |chunk| {
            chunk.set_inline_retry_at(at)
        });
        eval(&mut engine, shape.call);
        assert_eq!(runs(&mut engine, shape.name), at - 1, "{}", shape.what);
        with_chunk(&mut engine, shape.name, |chunk| {
            assert!(
                chunk.inline_retry_due(chunk.jit_runs.get() + 1),
                "{}",
                shape.what
            );
        });
        // A word at its top: the count stops one below it, whose gate's `runs + K` the gate
        // compares in 64 bits; a count at the word (an entry's saturation) counts nothing more.
        let mut engine = prepare(tier(true), shape);
        with_chunk(&mut engine, shape.name, |chunk| {
            chunk.jit_runs.set(u32::MAX - 500);
            chunk.set_inline_retry_at(u32::MAX);
        });
        eval(&mut engine, shape.call);
        assert_eq!(
            runs(&mut engine, shape.name),
            u32::MAX - 1,
            "{}: saturated",
            shape.what
        );
        with_chunk(&mut engine, shape.name, |chunk| {
            chunk.jit_runs.set(u32::MAX);
            chunk.set_inline_retry_at(u32::MAX);
        });
        eval(&mut engine, shape.call);
        assert_eq!(runs(&mut engine, shape.name), u32::MAX, "{}", shape.what);
    }
}

#[test]
fn a_direct_entry_at_the_top_of_the_word_takes_the_helper_and_never_wraps() {
    // The direct sequence's gate adds K and compares the count with the word in 64 bits (step
    // 11): a count within K of u32::MAX takes the call helper, which saturates and takes the
    // checkpoint, where a 32-bit sum would wrap to a small count and store it. The checkpoint
    // after it would pass the word: none is armed, and the later entries count nothing.
    tier(true).set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function leaf(x) { return x + 1; }\n\
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t = leaf(t); return t; }",
    );
    assert_eq!(eval(&mut engine, "drive(3)"), "3");
    let direct =
        |engine: &mut Engine| with_chunk(engine, "leaf", |chunk| chunk.test_direct_returns.get());
    with_chunk(&mut engine, "leaf", |chunk| {
        // Its callers' direct sequences wait for the checkpoints' first attempt.
        chunk.inline_attempted.set(true);
        chunk.set_inline_retry_at(u32::MAX);
    });
    // A run that makes the site's way a direct entry and takes it, the gate counting far below
    // the word; then the count within K of the word, which the next entry's gate meets.
    assert_eq!(eval(&mut engine, "drive(3)"), "3");
    let before = direct(&mut engine);
    assert!(before > 0, "the direct sequence enters leaf");
    with_chunk(&mut engine, "leaf", |chunk| {
        chunk.jit_runs.set(u32::MAX - 10)
    });
    assert_eq!(eval(&mut engine, "drive(50)"), "50");
    assert_eq!(
        runs(&mut engine, "leaf"),
        u32::MAX,
        "saturated, not wrapped"
    );
    with_chunk(&mut engine, "leaf", |chunk| {
        assert_eq!(
            chunk.inline_retry_at.get(),
            0,
            "no checkpoint past the word"
        );
    });
    assert!(
        direct(&mut engine) > before,
        "the calls took the direct sequence"
    );
}

#[test]
fn a_rotated_loop_counts_the_iterations_before_its_bails() {
    // The rotated loop's element read bails past the array's 2,000 elements, at the loop's
    // 2,001st iteration: its countdown's 2,000 iterations count at the bail, canonical code
    // finishes the iteration and its header counts, and each later iteration enters the loop
    // again and bails at once.
    const SOURCE: &str = "function sumTo(a, n) { var s = 0; for (var i = 0; i < n; i++) \
                          s = (s + a[i]) | 0; return s; }\n\
                          var ones = []; for (var k = 0; k < 2000; k++) ones.push(1);";
    tier(true).set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, SOURCE);
    eval(&mut engine, "sumTo(ones, 3)");
    with_chunk(&mut engine, "sumTo", |chunk| {
        chunk.set_inline_retry_at(u32::MAX)
    });
    let w = weight(&mut engine, "sumTo");
    let (runs, entered) = counted(&mut engine, "sumTo", "sumTo(ones, 2100)");
    eprintln!("rotated loop with bails: {runs} runs, weight {w}, entries {entered:?}");
    assert!(
        entered[2] >= 2,
        "the loop bailed and entered again: {entered:?}"
    );
    let k = crate::opt::ENTRY_WEIGHT;
    assert!(
        (k + 2100 * w..=k + 2103 * w).contains(&runs),
        "{runs} runs for 2,100 iterations of weight {w}"
    );
}

/// The counting sites in `code`: its `ldr w17, [x16, #retry]`, the second instruction of each
/// count (`emit_loop_count`, `emit_loop_count_countdown`), which no other template emits.
fn counting_sites(code: &super::JitCode) -> usize {
    let retry = {
        let mut a = super::asm::Asm::new();
        let offset = std::mem::offset_of!(crate::bytecode::Chunk, inline_retry_at);
        a.ldr_w_imm(17, 16, offset as u32);
        a.buf[0]
    };
    // SAFETY: the code's `len` bytes are mapped and readable while `code` lives.
    let words = unsafe { std::slice::from_raw_parts(code.mem as *const u32, code.len / 4) };
    words.iter().filter(|&&word| word == retry).count()
}

#[test]
fn template_code_holds_counting_code_only_with_the_tier_on_and_counting() {
    // The tier off, and on under TRUST_JS_OPT_NO_LOOP_COUNT: no counting site, as the code was
    // before step 8. Counting: the header's, and a region's or a scan's back edge's, a rotated
    // loop's polls, exit and bails.
    for shape in &SHAPES {
        let sites = |switches: TestSwitches| {
            let mut engine = prepare(switches, shape);
            with_chunk(&mut engine, shape.name, |chunk| {
                counting_sites(&chunk.jit.get().flatten().expect("native"))
            })
        };
        assert_eq!(sites(TestSwitches::default()), 0, "{}", shape.what);
        assert_eq!(sites(tier(false)), 0, "{}", shape.what);
        let counting = sites(tier(true));
        let least = match shape.what {
            "canonical" => 1,
            "rotated loop" => 3,
            _ => 2,
        };
        assert!(counting >= least, "{}: {counting} sites", shape.what);
    }
}
