# Working on TRust JS

This file collects the implementation and testing guidance that used to live in
README.md. Keep README.md a short introduction for people discovering the project.

## Project and workspace

TRust JS is a JavaScript engine written in Rust and the JavaScript backend used by
TRust. The workspace also provides a standalone runtime. Keep the engine independent
of a particular host; use the `embed` feature's curated API for host integration.

| Crate | Responsibility |
| --- | --- |
| `trust-js` | Lexer, parser, interpreter, bytecode VM, native JIT, builtins, embedding API |
| `trust-js-host` | Host state, resource tables, extensions, worker and callback primitives |
| `trust-js-timers` | Timers, immediates, microtask integration |
| `trust-js-fs` | Synchronous and asynchronous filesystem operations |
| `trust-js-web` | Web APIs and their JS glue, HTTP, Streams, WebSocket, WebAssembly support |
| `trust-js-node` | Node compatibility, CommonJS resolution, native addons, Bun compatibility APIs |
| `trust-js-tls` | TLS transport using dynamically loaded system OpenSSL |
| `trust-js-runtime` | Event loop, module loading, console/process, assembly of host extensions |
| `trust-js-repl` | Interactive shell with a persistent realm |
| `trust-js-cli` | Runtime command-line entrypoint |
| `trust-js-typescript` | TypeScript parsing and checking |
| `trust-js-wasm` | WebAssembly build of the JavaScript engine |
| `test262-runner` | Test262 conformance harness |
| `trust-js-difftest` | Differential testing across execution tiers |
| `wasm-spec-runner` | Official WebAssembly test harness |

The main layering is engine → host substrate → op crates → runtime → REPL → CLI.
Keep this dependency direction acyclic. Consult the manifests for the complete graph.

The workspace is no longer zero-dependency. The native engine uses `stacker`;
web support uses URL/IDNA and encoding crates; Wasm bindings and test tooling have
their own dependencies. Check Cargo.toml and Cargo.lock before making dependency
claims. System libraries and the vendored Streams polyfill are separate from Cargo
dependencies.

Linux builds link with packed relative relocations (RELR; see `.cargo/config.toml`), so
their binaries need glibc 2.36 or later where they run, and a linker that supports
`-z pack-relative-relocs` (GNU ld 2.38 or later, LLVM lld 15 or later, or mold).

## Performance work

Optimize the established bytecode/native JIT pipeline and shared runtime as one engine.
The optional Cranelift JavaScript backend has been removed; its dated investigation
records are historical evidence, not an active implementation roadmap.

Use release builds and record the machine, revision, tier, flags, and workload.
Follow [benchmarks/README.md](benchmarks/README.md) for comparison and regression
policy. Check correctness as well as throughput, memory, startup, and compilation
costs. Investigation tools and reports may exist locally without being tracked;
do not assume a fresh clone includes them.

The tracked benchmark runners (Octane, Web Tooling, ARES-6) and their
invocation are listed in [DIAGNOSTICS.md](DIAGNOSTICS.md). How to take a figure that
goes into a record (builds, instruction counts, time, memory) is in
[benchmarks/README.md](benchmarks/README.md), "Measurement method".

The current program is [docs/v8-chase-2026-10-15.txt](docs/v8-chase-2026-10-15.txt).
Its Phase 3, the speculative optimizing tier (`TRUST_JS_OPT=1`, ARM64, off by
default), is planned and recorded step by step in
[docs/v8-chase-p3-plan-2026-11-08.txt](docs/v8-chase-p3-plan-2026-11-08.txt);
[docs/v8-chase-p3-handoff-2026-11-18.txt](docs/v8-chase-p3-handoff-2026-11-18.txt)
says where that work stands, what is half done on which branch, which questions wait
for the owner, and what a fresh clone lacks. Write a design note with its exit before
code, count what a change can reach before building it, and record what was measured
and what stayed open when it lands. A change with no measured effect does not land.

## Design decisions

The owner settles design forks. Bring each with a recommendation, grounded like this:

- V8, SpiderMonkey and JSC are the reference designs. Read how each handles the case
  in its current source (memory and articles go stale), cite file and line, and lead
  with the option that matches them.
- Give the reason behind each engine's choice and say whether it applies here. A
  baseline tier that skips work because an optimizing tier does it later is a reason
  that may not transfer. Where TRust JS must depart (a conservative collector that does
  not move objects, NaN-boxed double values, generated code written once), say so and why.
- Never recommend an option because it is easier to build. When an exemplar's design
  looks costly, find out what it costs them before proposing a shortcut.
- The optimizing tier is off by default only until it is complete; it will be the
  default. Weigh a design for that end state as well as for today's default: do not
  argue for a first-stage special form because the tier is off, and do not let one
  hide from the tier what its own handling of the case will need.
- Collection scheduling and host GC policy follow the major browsers' engines' current
  behaviour. Judge a collector change by correctness first (bounded memory), then by
  costs that stay meaningful as generated code gets faster (per allocated and live
  megabyte, the longest pause). A memory bound is a reading, not the goal.

## Engineering rules

Each of these cost a crash, a miscompile or a measured regression before it was
written down.

- Rooting. A `Value` held in ordinary Rust memory (a `Vec`, a struct, an enum payload)
  across anything that can allocate or run JavaScript is not a root; the collector may
  free it. Root it the way the surrounding code does, and run the paths you touch under
  `TRUST_JS_GC_STRESS=1`: conservative scanning of stale stack words hides most of these
  in ordinary runs.
- The collector is conservative and non-moving. It scans the native stack, the
  callee-saved registers and active JIT records, so generated code needs no stack maps,
  but a stale word keeps garbage alive.
- Shared stubs are process-wide. Code a shared stub can hold must not embed the address
  of a heap, an Agent, collector state or a thread-local: load it through the context. A
  test with two engines alive on one thread catches a violation.
- Hot types keep their ABI. A second scalar in a hot enum variant made every
  `Option<Element>` return go through memory at the same size (Vue +0.56%). Check the
  disassembly of an out-of-line callee, not `size_of`.
- Offsets that generated code embeds come from a release build's layout. The test build
  of `Interp` has a test-only field, so its later offsets differ.
- The tier off stays identical. A change to the optimizing tier leaves the default
  configuration's generated code byte-identical unless its design says otherwise; show
  it with masked code hashes or equal instruction counts.
- A speculation needs a teacher. A check whose only feedback is its own exit flag is
  relearnt one site per discard, and a function with many such sites is discarded once
  per site (zlib ran +291% in the always-wrong case). Give each new speculation an input
  the compile reads (a record the first stage fills, or an engine-wide bit), and measure
  the case where it is always wrong.
- Freed code addresses stay out of use. On Cortex-X925, code mapped at the address of
  code just freed mispredicts persistently, at equal instruction counts (the plan's
  stage 2, step 9).
- A second thread changes `malloc`. Any second thread takes glibc's allocator off its
  single-thread path (CodeLoad +10% from an idle thread alone). When work that parses or
  allocates slows down with the compile thread present, test for that first.

## Testing

Tests are a diagnostic tool. Running every suite at every commit stalled this project
once; run what tells you something about the change. A change's check before it lands
takes minutes, not hours: quick checks, then land, and fix forward if something breaks
later. Count the hours a check plan implies before starting it.

- While developing: `cargo fmt --check`, `cargo clippy --offline --locked -p trust-js
  --lib --tests --features embed -- -D warnings`, and the tests nearest the change.
- For the optimizing tier: its modules (`cargo test -p trust-js --lib --features embed
  opt::`) with the tier off, with `TRUST_JS_OPT=1`, and with `TRUST_JS_OPT=1
  TRUST_JS_OPT_EAGER=1`; add forced exits (`TRUST_JS_OPT_EXIT_EVERY=100`) and GC stress
  over the new tests where a change touches exits or the collector's contract. A new or
  changed switch is checked over the whole library suite, not `opt::` alone.
- Release builds do not run the tier's graph verifier. Before reporting a tier change,
  run every Octane benchmark at five fixed runs and the fixtures with `TRUST_JS_OPT=1
  TRUST_JS_OPT_VERIFY=1 TRUST_JS_OPT_LOG=1` at both `TRUST_JS_OPT_INLINE` modes and count
  `verify ... failed` lines. One fixed run misses compiles that only later runs reach.
- Every rule gets a test that fails when the rule is broken: mutate the rule and name
  the test that catches it. An uncaught mutation is a missing test.
- Full validation (the library suite in every configuration, Test262, difftest, the
  fixtures' outputs, the other architecture) runs at the end of a stage or after a
  large change to what it measures, not per commit.
- A failure is pre-existing only if it fails without the change: on the change's parent,
  or with the change switched off on the same binary.
- Shared code is built and linted for x86-64 and Windows before it is called done;
  x86-64-only code never compiles on an ARM64 host.
- Traps met before: `test262-runner` takes every argument as a test path and writes
  `test262-report/` into the current directory, so run it from a scratch directory. Do
  not build or test in a target directory while a test run from it is going (tests spawn
  `current_exe()`). A revision exported with `git archive` needs a fresh target
  directory, because its old file times make cargo reuse stale artifacts. `cargo test |
  grep | head && git commit` commits on a failed test. Never edit a shell script that is
  still running.

## Documentation

Keep useful public setup instructions in READMEs,
diagnostic commands and environment variables in `DIAGNOSTICS.md`, and
third-party provenance beside the vendored source. General Markdown workfiles
and the retired changelog are ignored and kept locally. Do not make builds or
required contributor instructions depend on those untracked documents.