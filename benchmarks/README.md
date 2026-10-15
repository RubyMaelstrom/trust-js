# Benchmark records

This directory retains the checked-in benchmark manifests, fixtures, and historical performance
policy. The local investigation runners and provisioning helpers have been removed.

## Regression policy

- Prioritize Speedometer 3.1 segments and actual browser pages using widely used frameworks
  for engine performance acceptance. Selected suites are sufficient for iteration; a full
  Speedometer run is optional. Record the exact subset and per-suite results, and never label
  a partial run as a full Speedometer score. Synthetic tests provide diagnostic controls.
- Standards, Test262/WPT, tier-differential, and real-site failures are release blockers regardless
  of performance.
- Compare interleaved distributions from the same manifest and machine. Do not promote a single
  sample, best run, or historical directional ratio as a result.
- A stable component is provisionally regressed when its 95% comparison interval excludes parity
  in the slower direction and the median loss exceeds 3%. Noisy components use a threshold fixed
  from their Phase 0 variance before an optimization is measured.
- Investigate component scores, wall/CPU time, peak and post-GC memory, pauses, compilation time,
  and code size independently. An aggregate improvement does not excuse a visible DOM, latency,
  memory, startup, or completion regression.
- A meaningful regression requires a written explanation and explicit user acceptance. Otherwise
  rework or revert the responsible optimization.
- If keeping a local, gitignored `accepted-production.json`, update it only after the user
  approves and promotes exact release artifacts. Preserve the preceding report rather than
  overwriting history.

## Renderer-shaped JavaScript

`webgl-engine-kernels.js` isolates ordinary arrays, typed-array copying, computed property reads,
prototype misses, enumeration, and function calls without browser or graphics APIs. Arithmetic,
an indexed numerical stencil, and recursive calls provide controls against unrelated regressions.
Run it with the release engine shell:

```sh
cargo build --release -p trust-js --features embed --bin trust-js
target/release/trust-js --tier=jit benchmarks/webgl-engine-kernels.js
```

Each kernel warms up and prints all five timings plus checksums. Compare fresh baseline and
candidate processes in interleaved order, using the same CPU affinity, build settings, and input;
verify every checksum. Kernel speedups do not predict a scene's frame rate.

`webgl-engine-2026-09-19.json` records the ARM64 comparison, release artifact hashes, raw kernel
samples, real-WebGL browser measurements, validation scope, and remaining limitations. Its
baseline already includes the preceding browser-call and WebGL optimizations. It is an
investigation record, not an installed-artifact promotion record.

## Builtin scaling and native calls

`group-by-scaling.js` checks Object/Map grouping with repeated and distinct keys, using arrays
and generators at increasing sizes. It verifies group order and every member outside the timed
interval. A preceding script can set `GROUP_BY_CONFIG` to select sizes, input kinds or samples.

`string-replace-split-scaling.js` measures literal replacement callbacks and empty-separator
splitting, including limited output and full-output controls. `native-math-control.js` exercises
ordinary Math/isNaN receivers to detect costs from native-call ownership guards.
`array-sparse-recovery-scaling.js` compares pristine dense arrays with arrays restored after a
temporary far-index write, measuring push/pop and numeric reads at increasing sizes. Each fixture
prints every timing and checksums, and runs on the release engine shell or Node:

```sh
target/release/trust-js --tier=jit benchmarks/group-by-scaling.js
target/release/trust-js --tier=jit benchmarks/string-replace-split-scaling.js
target/release/trust-js --tier=jit benchmarks/native-math-control.js
target/release/trust-js --tier=jit benchmarks/array-sparse-recovery-scaling.js
```

Use interleaved fresh processes and matching build settings for comparisons. A timing rounded
to zero milliseconds does not establish an exact speedup; retain process wall/CPU time and
memory separately. These focused workloads do not establish overall engine or application speed.

The 2026-10-14 investigation is documented in
[the performance review](../docs/performance-review-2026-10-14.txt) and
[the pinned Ladybird source comparison](../docs/ladybird-comparison-2026-10-14.txt).
The `review-*-2026-10-14.json` files retain versioned samples and validation evidence;
earlier candidates remain in the record when a regression required further work.

The [compiler checkpoint](../docs/compiler-review-2026-10-14.txt) records the separate
inlining, range-analysis and property-proof work, its exact validation artifacts, and
selected browser methodology. Correctness records and diagnostic profiles do not by
themselves establish application performance acceptance.

The [declarative string accumulation record](../docs/string-accumulation.txt) and
[validation manifest](review-string-accumulation-validation-2026-10-15.json) retain
the native x86-64 React 19.2.7 SSR comparison: seven fixed alternating pairs, with
separate correctness evidence for the integration. The measured rendering-duration
reduction is scoped to that authored SSR workload.

The [inline named-field experiment](review-inline-named-fields-2026-10-15.json)
records a separate seven-pair React SSR comparison. It found no convincing speed
gain; the object-layout patch remains an isolated experiment and was not promoted.

The [borrowed-operands comparison](review-borrowed-operands-react-2026-10-15.txt)
records a separate native x86-64 React SSR result: 2.65% less rendering-and-equality
time with 2.44% higher peak RSS. It covers the exact frozen compiler probe and
keeps the optional optimization disabled by default.

The [held activation and native-call candidate](review-activation-native-calls-validation-2026-10-15.json)
passed scoped correctness checks but retains four Vue component concerns in a descriptive
three-pair browser screen. The combined engine source remains isolated and is not promoted
to main; preserving a review checkpoint does not change that decision.
The [43-component paired results](review-activation-speedometer-components-2026-10-15.csv)
retain all three baseline/candidate values and cost ratios without editing the frozen CSV.
