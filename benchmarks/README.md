# Benchmark records

This directory retains the checked-in benchmark manifests, fixtures, and historical performance
policy. The local investigation runners and provisioning helpers have been removed.

## Regression policy

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
