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
