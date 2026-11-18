# Benchmark records

This directory retains the checked-in benchmark manifests, fixtures, and regression
policy.

## Regression policy

- Prioritize Speedometer 3.1 segments and actual browser pages using widely used frameworks
  for engine performance acceptance. Selected suites are sufficient for iteration; a full
  Speedometer run is optional.
- Standards, tier-differential, and real-site failures are release blockers regardless
  of performance.
- Compare interleaved distributions from the same manifest and machine. Do not promote a single
  sample, best run, or historical directional ratio as a result.
- Investigate component scores, wall/CPU time, peak and post-GC memory, pauses, compilation time,
  and code size independently. An aggregate improvement does not excuse a visible DOM, latency,
  memory, startup, or completion regression.
- A minor regression (roughly 5% or less on a component, with no visible regression of the
  kinds above) is accepted only if it is a step towards a major engine improvement. Anything
  larger, or any visible regression, is meaningful: it requires a written explanation and
  explicit user acceptance. Otherwise rework or revert the responsible optimization.
- If keeping a local, gitignored `accepted-production.json`, update it only after the user
  approves and promotes exact release artifacts. Preserve the preceding report rather than
  overwriting history.

## Measurement method

What the recorded investigations learned about measuring this engine. A figure that goes
into a record follows these; a quick bisection may relax them, and says so.

Builds

- Compare fat-LTO release builds of the change and of its immediate parent, never of an
  older snapshot of main. Builds without LTO move by about 1% between unrelated changes,
  and an older base charges the change with whatever landed in between. Name the base
  commit in the record.
- Build each revision from its own export into a fresh target directory, and check that
  the binaries differ before measuring them. An export's old file times make cargo hand
  back another revision's artifact without an error.
- A changed Cargo feature list reorders the binary's text by itself (symbol hashes
  change). To compare across a feature's removal, re-declare it as an empty feature.

Instruction counts (`perf stat -e instructions:u`)

- Pin the run to one core type with `taskset` outside `perf`, switch address
  randomization off (`setarch -R`), set `TRUST_JS_GC_PACING=0`, and take the median of
  three. Unpinned on a machine with two core types, the count is not a count.
- Fixed addresses make one binary's runs agree; they do not make two binaries
  comparable to that precision. With `setarch -R` three runs of a binary agree to
  0.0002%, but each binary has its own addresses, and a workload that hashes by address
  moves with them: a run's standard deviation at randomized addresses is 0.06% to 0.09%
  on Vue (fixture and steady state), 0.1% on PdfJS, 0.08% to 0.23% on TypeScript, 0.02%
  to 0.05% on GBEMU, Splay and Djot's steady state, and 0.012% or less on the other
  fixtures and Octane benchmarks at five runs. A difference between two binaries that
  is under about three of those deviations is not a result. Confirm one that matters
  with the mean of at least eight runs a side without `setarch -R`, the binaries
  interleaved, and report the mean, the deviation and the standard error. A +0.09% on
  Vue's steady state, three runs a side agreeing to four decimals, was 0.007% in 24-run
  means.
- Run both sides through the same script and invocation. The environment's layout moves
  a count through the conservative stack scan, so a same-binary comparison toggled by an
  environment variable passes an unread variable of equal length on the other side.
- Check every run's exit status and its driver's completion line. A fixed-run driver
  keeps its state out of globals a benchmark shares and resets Octane's random numbers
  as Octane's own runner does: a driver that did neither ran zlib once in place of five
  times and failed RegExp's checksum, and no record noticed until the driver was audited.
- Octane at five fixed runs is mostly warm-up on the short benchmarks. A change to
  tiering, inlining or exits also needs long fixed runs (200 or 2,000) or timed Octane:
  one that traded warm-up exits for steady-state speed read -12.5% instructions on
  Richards at five runs and -6.6% in its timed score.
- The collecting benchmarks (Earley-Boyer, Splay) and small fixtures move with the
  incremental marker and with what the conservative scan finds. Count them on idle
  CPUs. The marker's schedule (how many slices a cycle takes, where its first major
  collection ends) is a property of the binary: it repeats to 0.001% for one build
  at fixed addresses and moved Splay by 1.6% to 1.9% with the optimizing tier on, and
  Earley-Boyer by 0.3%, between builds whose work was equal. So a row of these two is
  charged to a change only when the recount with `TRUST_JS_GC_INCREMENTAL=0` shows
  it too; compare the collector's log (`TRUST_JS_GC_LOG=1`: the major collections'
  object counts and slice counts) before looking for a cause in the change. This
  holds for a change to the collector as much as for any other.
- PdfJS yields after a time slice of the real clock, so its count follows the machine's
  speed. Give it a deterministic clock before drawing a verdict from it.
- Under about 1.5%, a difference can be layout: a template's size flipping an admission
  decision, address-keyed tables colliding differently, LTO inlining changing in an
  untouched function. Compare generated-code sizes and decisions and disassemble the hot
  function before charging it to the change.

Time

- Interleave rounds of the builds on one pinned core with the rest of the machine quiet.
  Record the delivered clock for each phase and discard a phase that was capped or
  disturbed: frequency capping moves ratios, not only absolute times.
- Name the core type. Times that split into modes at equal instruction counts on one
  core type are a code-placement effect until shown otherwise.
- With the optimizing tier on, the engine has a compile thread. Give the engine and
  every reference (Node and its tier-limited modes) the same CPUs, and count the tier's
  main-thread compile work beside the run time.
- Instruction counts do not show what a memory layout costs. A change that moves where
  data lives (a field out of line behind a pointer, a record split or merged, a table
  relocated) is measured in cycles before it is decided, with the counters that say
  where the cycles went (`branch-misses`, `L1-dcache-load-misses`, the back-end stall
  events). Moving a call site's 168-byte entry out of line saved instructions and cost
  RayTrace 2% in cycles: one more dependent load on each construction.

Memory

- Peak RSS includes the binary's file-backed pages, which depend on how the file was
  last written. Copy every binary being compared into a new directory at the same time,
  measure the copies, and include an empty-script run as a control.
- Before charging a peak-RSS difference between two builds to a data structure, split it
  into anonymous and file-backed memory (`RssAnon` and `RssFile` in `/proc/<pid>/status`,
  sampled through the run) and count the structure: instances times bytes. Two builds'
  layouts touch different pages of the binary, which moves peak RSS by a few hundred KiB
  either way; a +1.5% reading on a small fixture was that, where the structure's own
  growth was 1 KB.
- The largest live-bytes figure among the major collections is a sample, not the peak:
  majors fire at byte thresholds, so builds whose objects differ in size are sampled at
  different points. Sample densely (`TRUST_JS_GC_STRESS=100000 TRUST_JS_GC_STRESS_MAJOR=1
  TRUST_JS_GC_LOG=1 TRUST_JS_GC_PACING=0`) and compare at equal object counts.

## Fixtures

Each `.js` fixture documents in its header what it isolates and how to run it. In
general, run them with the release engine shell (or Node, where the header says so):

```sh
cargo build --release -p trust-js --features embed --bin trust-js
target/release/trust-js --tier=jit benchmarks/<fixture>.js
```

Compare interleaved fresh baseline and candidate processes with the same CPU affinity,
build settings, and input; verify every checksum and retain process wall/CPU time and
memory separately. A timing rounded to zero milliseconds does not establish an exact
speedup. These focused workloads do not establish overall engine or application speed.

## Records

The dated `.json`, `.csv`, and `.txt` files in this directory are frozen investigation
records, not performance acceptance or installed-artifact promotion evidence. Narrative
reviews of past investigations live in `../docs/`.
