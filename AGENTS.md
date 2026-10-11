# Working on Lumen

This file collects the implementation and testing guidance that used to live in
README.md. Keep README.md a short introduction for people discovering the project.

## Project and workspace

Lumen is a JavaScript engine written in Rust and the JavaScript backend used by
TRust. The workspace also provides a standalone runtime. Keep the engine independent
of a particular host; use the `embed` feature's curated API for host integration.

| Crate | Responsibility |
| --- | --- |
| `lumen` | Lexer, parser, interpreter, bytecode VM, native JIT, builtins, embedding API |
| `lumen-host` | Host state, resource tables, extensions, worker and callback primitives |
| `lumen-timers` | Timers, immediates, microtask integration |
| `lumen-fs` | Synchronous and asynchronous filesystem operations |
| `lumen-web` | Web APIs and their JS glue, HTTP, Streams, WebSocket, WebAssembly support |
| `lumen-node` | Node compatibility, CommonJS resolution, native addons, Bun compatibility APIs |
| `lumen-tls` | TLS transport using dynamically loaded system OpenSSL |
| `lumen-runtime` | Event loop, module loading, console/process, assembly of host extensions |
| `lumen-repl` | Interactive shell with a persistent realm |
| `lumen-cli` | Runtime command-line entrypoint |
| `lumen-typescript` | TypeScript parsing and checking |
| `lumen-wasm` | WebAssembly build of the JavaScript engine |
| `test262-runner` | Test262 conformance harness |
| `lumen-difftest` | Differential testing across execution tiers |
| `wasm-spec-runner` | Official WebAssembly test harness |

The main layering is engine → host substrate → op crates → runtime → REPL → CLI.
Keep this dependency direction acyclic. Consult the manifests for the complete graph.

The workspace is no longer zero-dependency. The native engine uses `stacker`;
web support uses URL/IDNA and encoding crates; Wasm bindings and test tooling have
their own dependencies. Check Cargo.toml and Cargo.lock before making dependency
claims. System libraries and the vendored Streams polyfill are separate from Cargo
dependencies.

## Execution and correctness

The tree-walking interpreter is the reference implementation for differential
testing. All three tiers must agree on observable behavior: completion values,
errors, side effects, and final state. Use the language specification to resolve
semantic questions; agreement between tiers alone does not prove correctness.

Eligible functions compile to bytecode, with inline caches, object shapes, and
dense-array fast paths. The native template JIT lowers bytecode to ARM64 or x86-64
on supported desktop platforms, with checked helpers for slow paths. Other targets
fall back to bytecode. Backend coverage differs; check both `jit.rs` and
`jit_x64.rs` when changing shared operations or layouts.

The default tier is JIT. Useful diagnostic controls:

- `LUMEN_TIER=interp|bytecode|jit` selects the tier.
- `LUMEN_TIER_THRESHOLD=N` changes the call threshold for tiering up; loop-containing
  bodies can tier up immediately.
- `LUMEN_INLINE_AT=N` changes the speculative inline recompile threshold (default:
  100 machine-code runs); zero disables inlining.
- `LUMEN_JIT_NO_DIRECT_CALLS=1` disables ARM64 direct calls (the inline call sequence
  that runs the callee on a pooled frame record) while retaining ordinary JIT calls.
- `LUMEN_JIT_INLINE_STUBS=1` emits ARM64 name-cache validation, whole free-name reads,
  slow-path helper calls, generic property-cache probes, secondary call-cache probes and
  direct-call sequences at every site instead of as per-chunk shared stubs (by default only
  loop bodies keep name validation, whole name reads and direct calls in line, and every
  slow-path helper call is a stub call). It is a code-size/performance ablation with the
  same behavior. Compiled DOOM (archive.org's asm.js DOSBox) is bound by instruction-cache
  pressure: making sites larger, by inlining stubs or sequences, slowed it by 10-18% in
  paired runs.
- `LUMEN_JIT_NO_JSCVT=1` makes ARM64 code use the portable guarded ToInt32 sequence even
  when the CPU implements FEAT_JSCVT (`fjcvtzs`).
- `LUMEN_TAIL_CALL_DEPTH=N` sets the execution-context depth below which a strict
  tail call runs as an ordinary call (default 1024); at or past it the call is staged as a
  proper tail call, so unbounded tail recursion stays bounded. `0` stages every tail call.
- `LUMEN_JIT_HOT_LOOP_OPS=N` changes the 512-op bound on loops that keep ARM64 name checks
  and direct-call sequences in line; larger (flattened dispatch) loops use shared stubs.
- `LUMEN_JIT_CHUNK_STUBS=1` gives each ARM64 chunk its own copy of the shared stubs instead
  of veneers to the process-wide stub table. It is a code-size ablation with the same behavior.
- `LUMEN_TIER_LOG=1` helps diagnose compilation bailouts.
- `LUMEN_JIT_DUMP=<names>` prints the bytecode of chunks whose first four slot names contain
  the text, and `LUMEN_JIT_CODEDUMP=<names>[@<offset>]` the finished ARM64 words of the same
  chunks (hex, one per line; assemble them with `clang --target=aarch64-linux-gnu -c` and
  `objdump -d`). `@<offset>` also requires the function's source start offset, the `@N` that
  `LUMEN_JIT_MAP=1` prints in its `[jit-map-range]` lines, which singles out one function among
  many that share names. Any value of `LUMEN_JIT_CODEDUMP` adds a `[jit-map]` line (runtime base
  and length) per compiled chunk.
- `LUMEN_JIT_NO_NAME_ELEM=1` compiles `LoadName K; <pure key>; GetElem/SetElemDrop` register
  regions with the ordinary LoadName and element templates instead of reading a typed array
  straight from its binding (see `NameElementPlan` in `jit_regions.rs`).
- `LUMEN_JIT_TYPED_STUBS=0` carries the native typed-array element sequence in line at every
  site instead of calling the process-wide shared stub, and `LUMEN_JIT_NO_NAME_ELEM_STUB=1` does
  the same for the fused name/element access (the sites then keep only the name probe shared).
  Compiled applications such as DOOM's emulator are bound by instruction fetch: shared stubs
  cost a call and return per access and save more in instruction cache misses.
- `LUMEN_JIT_NO_INT_REGS=1` keeps every int32 result of a register region (bitwise operators,
  shifts, `Math.imul`) in a floating-point register instead of leaving it in an integer
  register for the next operator, comparison or branch (`Operand::Int` in `jit_regions.rs`).
- On GNU/Linux, `LUMEN_JIT_GPROFNG=1 gprofng collect app -o profile.er ...`
  registers live generated-code symbols with the already-loaded gprofng collector.
  Labels show the chunk's first local names, not source function names. This is
  optional compile/drop metadata; it neither instruments native instructions nor
  loads a profiler library. Keep it disabled for uninstrumented timing comparisons.

The optional `optimizing-jit` feature adds an in-development whole-function Cranelift backend
on native ARM64/x86-64. It uses Cranelift 0.136.2, matching TRust's owned Wasmi integration,
and requires Rust 1.96 or newer when enabled. `LUMEN_OPT_JIT=1` explicitly selects this
development backend for eligible ordinary functions; unsupported entries retain the template
JIT/VM. `LUMEN_OPT_JIT=hot` instead starts with template code and admits frequently called
ordinary functions using four bounded entry samples, guarded Number/Boolean input facts,
numeric own-data field results, and an estimated compilation/removed-check work budget.
Field sampling only inspects an already warmed ordinary descriptor; it invokes no getter,
proxy or conversion. Native reads guard live receiver/descriptor/value state on each access
and resume before that access on a miss. Generic bodies, environment-homed inputs and
bodies above 256 bytecodes remain on the established tier. `LUMEN_OPT_JIT_HOT_AT`
overrides the admission distance for diagnostics; it does not waive positive semantic benefit
or the IR budget. A candidate must compile successfully before second-stage publication and
call-cache invalidation. Hot mode preserves the established bytecode inliner for unadmitted
functions. Both transformations check the same second-stage slot; a published inlined body
is outside the optimizer's single-frame recovery contract and stays on the template tier.
This whole-function policy does not migrate active frames. The separate opt-in
loop-continuation experiment below handles a restricted active-frame case.
Eight specialization misses retire that version: future dispatch stops publishing its address,
active native frames and Rust leases finish before reclamation, and later calls use the template
tier. Successful native entries gain no retirement counter. This is a bounded one-version
policy, not adaptive reoptimization.
Its cost estimate and real application repayment still require validation. The feature and runtime switch
are both off by default. `LUMEN_OPT_JIT_LOG=1` reports
declines, and `LUMEN_OPT_JIT_DUMP=1` prints bytecode/CLIF for diagnosis. Do not equate this
backend integration with a completed JavaScript optimizing tier or a performance acceptance.
See `docs/optimizing-tier.txt` for its contracts, remaining work and real-application gates.
With the optional feature compiled, `LUMEN_OPT_JIT_OSR=1` separately enables
guarded continuations at hot loop headers. It continues the existing invocation,
without repeating parameter binding or prefix effects. `LUMEN_OPT_JIT_OSR_AT=N`
sets the header-visit threshold (default 65,536, minimum 2). Eligible bodies have
at most 512 operations and at most eight tracked headers; unsupported frames,
detailed-feedback bodies and whole-function optimized bodies keep their existing
execution. Canonical owners remain live across entry, callbacks and recovery.
Effect-scoped Number fields can remain in registers between observers; changed
proofs recover at the correct before- or after-effect bytecode boundary. Eight
entry misses retire that continuation without replacing the primary body.
This experiment is off by default and has not improved measured application
performance. Its isolated numeric-loop gain does not justify enabling it.
`LUMEN_OPT_JIT_DIAGNOSTICS=1` adds bounded per-native-body compilation, entry,
helper, heap-guard, owner-transfer, frame-publication and specialization-bailout observations. It emits
compiler/drop records and includes live records in performance metrics. Disabled
emission adds no native instructions. `LUMEN_OPT_JIT_DIAGNOSTICS=compile` records
compilation/static counts and emits the ordinary uninstrumented body. Instrumented
IR is larger and still subject
to the same compiler limits; diagnostic runs cannot be acceptance timings.
The separate `architecture-diagnostics` feature records allocation sites/lifetimes
in the live Rc heap and call-cache epoch/refill traffic. It adds no object owners;
site and registry-slot limits report untracked allocations. Initial exotic family
is recorded at allocation, before later function/exotic initialization. Lifetimes
use heap allocation ordinals and completed collections, not elapsed time. Live
families at collection are a separate population sample. These probes compile out
of ordinary builds; use them only for diagnostics, never performance acceptance.
`LUMEN_OPT_JIT_DEOPT_AT=N` is a development stress hook: an eligible optimized function
exits to its ordinary VM continuation when it reaches bytecode boundary N, before that operation.
It is unset by default, is not a type-specialization policy, and must be absent during
performance measurements. Suspended or baseline-inlined frames are not admitted by this hook.
The experimental ARM64 call-stub integration reuses the template tier's guarded shared-frame
calls. `LUMEN_OPT_JIT_CALL_STUBS=0` is its diagnostic ablation switch; it retains checked calls,
not a different JavaScript behavior. Stubs are weak-directory cached and pinned by referring
native bodies; all platforms retain the checked path when no supported stub can be emitted.
`LUMEN_OPT_JIT_SPARSE_RELOADS=0` is a compiler-emission diagnostic ablation: reload all
tracked local register copies after observers instead of the bounded liveness plan. It does
not remove canonical frame owners, change GC roots, or disable state publication. Keep the
setting explicit in measurement manifests; the optimizing tier itself remains opt-in.
`LUMEN_OPT_JIT_LOCAL_EFFECTS=0` retains the all-clobber helper analysis for a diagnostic
comparison. The development default separates canonical publication from private-slot
write effects: preserved register words do not imply immutable objects or environments.
Full-reload mode (`LUMEN_OPT_JIT_SPARSE_RELOADS=0`) still forces all tracked reloads.
`LUMEN_OPT_JIT_VALUE_FACTS=0` disables the experimental whole-function value-category
analysis and its guard/owner-check elisions. The analysis proves normal-result categories,
not immutable heap contents or numeric ranges; unknown effects and completion landings
retain dynamic checks. Budget exhaustion also retains those checks. This switch skips
analysis itself, unlike the reload-emission-only ablation above.
`LUMEN_OPT_JIT_HEAP_OPS=0` disables the experimental native own-element reads/overwrites
and ordinary own-data property writes, retaining their checked helpers. The native path
keeps live descriptor/exotic guards, canonical numeric mirrors and alias-aware owner
transfers; exotic/observable slow paths still execute the full runtime algorithm.
This is an emission diagnostic within the opt-in tier, not a production default change.
`LUMEN_OPT_JIT_CREATION=0` separately disables the native named-field creation path while
retaining the other heap operations. With creation enabled, a live creation-cache proof can
append into a small ordinary receiver's reserved field storage, retaining its shared key
layout. Extensibility/prototype changes, exotics, storage growth, sidecar maintenance and
last-owned layout destruction stay checked. Computed non-index creation and object allocation
are not covered by this path. `LUMEN_OPT_JIT_HEAP_OPS=0` also disables creation.

The standalone engine shell accepts `--tier=interp|bytecode|jit`. The runtime CLI's
current argument parser only accepts `--tier=interp|bytecode`; use `LUMEN_TIER=jit`
or the default for runtime JIT execution.

Some built-ins are self-hosted: `crates/lumen/src/self_hosted/*.js` defines them as strict
JavaScript methods, evaluated per Realm in a private environment whose only bindings are
the intrinsics documented in `self_hosted.rs` (ECMA-262 abstract operations such as
`ToObject`, `LengthOfArrayLike`, `ArraySpeciesCreate`, `CreateDataPropertyOrThrow` and
`Call`). The tree-walker calls the intrinsics' native forms; the bytecode compiler emits
`Op::Abstract` or an ordinary method call, including where the inliner splices a
self-hosted body into a caller. Write self-hosted source step for step from the
specification: it may observe nothing the algorithm does not (no author-reachable global,
prototype method or `Function.prototype` member), and its methods receive their specified
`length` at installation. They render as NativeFunction, are not constructors and compile
on their first call. Add Node-verified tests across every tier for each change.

Preserve activation and GC boundaries in generated calls: bind thread-local GC
state at execution time and discard callee-owned exception handlers before restoring
the caller. Keep raw object-layout assumptions validated against live Rust types
and retain checked fallbacks. Executable memory must follow platform write/execute
protections and instruction-cache synchronization requirements. Do not describe
the interpreter or VM as entirely safe Rust; they also contain unsafe code.

An AST, and the bytecode and native code cached on its functions, can serve every
Realm of an Agent: self-hosted built-ins and snapshots evaluated through
`eval_shared_classic_snapshot_interruptible` (one decode per interpreter, used by
browser Window Realms) share them. Compiled code must own no JavaScript objects and
must take intrinsics, the global object and environments from the running Realm.
Every cache keyed by an object, environment or Realm identity must validate it at
use; inline splices also require their planning Realm's global environment. Never
share an AST between interpreters: its caches hold one heap's shapes.

Such a shared snapshot keeps each function body encoded in the `&'static` snapshot bytes
until the function runs or compiles (`ast::FunctionBody`, `snapshot::decode_static`); the
process's first decode of the bytes still validates every body. Code that only analyzes a
function that is not about to run (closure creation, an enclosing function's capture scan)
must use `FunctionBody::with`, never materialize it; compiling an enclosing function
replays an encoded inner function's `bytecode::CaptureSummary` instead of walking it, and
test builds check each summary against the walk. Memory accounting reads only
materialized bodies. The encoder marks a snapshot's run-once bootstrap functions
(`Function::run_once`: immediately invoked from top-level code or another such function,
outside loops and classes); after a static decode their bytecode stays on the VM and never
gets native code, since they run once per Realm.

Engine strings store a lone surrogate as the scalar U+10F800 + (unit − 0xD800) and a
character in U+10F800..=U+10FFFF as its pair of such scalars (`jstr.rs`). Rust text
enters through `Value::str`, `Value::from_string` and the host source entry points,
which store those characters as pairs, and leaves through `Ctx::coerce_string` and
`Value::as_text`, the USVString conversion (lone surrogates become U+FFFD).
`Value::from_utf16` and `Ctx::coerce_utf16` keep lone surrogates. A host that stores
Web IDL `DOMString`s as Rust strings (a DOM) keeps them as engine text instead:
`Ctx::coerce_engine_text`, `Value::as_engine_text` and `Value::from_engine_text` pass it
unchanged, and `embed::text_to_engine`/`embed::engine_to_text` convert at that host's own
text boundaries (decoded documents in, rendering and USVString consumers out). Engine code
builds strings already in the engine representation with `Value::lstr` and appends code
points through `jstr`, never through `Value::str`.

TypeErrors for non-callable call or `new` targets name the target's source expression
(`callee_name.rs`). Compiled chunks keep these names in a side table read only when such
an error is thrown; every tier must produce the same message.

`Error.prototype.stack` prints V8-style frames with source positions (`stack_trace.rs`):
`at name (url:line:column)`, `at url:line:column` for anonymous functions and top-level
code, `at new name (…)` for constructors, `at eval (<anonymous>:line:column)` for eval code
and `at name (<anonymous>)` for self-hosted built-ins; host-snapshot platform functions are
left out. Columns count UTF-16 units. Embedders name scripts with `SourceOrigin`
(`*_with_origin` entry points; module keys name modules); code without one reports
`<anonymous>`. Positions are byte offsets on call, `new`, property-access, tagged-template
and assignment nodes; compiled chunks map operations to them (`SourcePositions`). Nothing is
recorded on fast paths: an explicit call stores its position in `Interp::call_site` only for
the duration of the call (the JIT's direct call writes it into the callee's `FnFrame`), a
frame push moves it into `FnFrame::call_site` and the pop restores it, and an error the engine
creates is positioned by the innermost positioned node or operation it propagates through
(`Interp::place_thrown`). A new call path must keep that discipline, and a compiler change
must attribute operations to the same nodes the tree-walker positions, or tiers diverge.
`Error.stackTraceLimit` (default 10) and `Error.captureStackTrace` follow V8.

The default-on `intl` feature supplies ECMA-402 and CLDR data. For the engine,
`--no-default-features` removes `Intl` and gives `toLocale*` methods their
locale-independent behavior. The `embed` feature exposes the host API; `bench`
exposes internal measurement entrypoints. The `heap-bridge` feature is an opt-in
migration facility, not the default production object representation.

## Runtime and integration

One event-loop thread owns the non-Send engine. Blocking work goes through a std
thread pool and returns through completion channels. Preserve microtask, callback,
timer, and I/O ordering when changing scheduling.

The runtime supports CommonJS and ESM. The CLI chooses ESM for `.mjs`, CommonJS for
`.cjs`, and uses the nearest package.json `"type"` for `.js`. Preserve module-graph
linking, top-level await, `node:` imports, CommonJS interop, and
`require.main === module` for CommonJS entrypoints.

Web API code lives in `crates/lumen-web/src/js/` and its Rust host operations.
`crates/lumen-web/build.rs` assembles the JS in order and precompiles an AST snapshot.
Keep the source bundle and snapshot consistent through that build path.

HTTPS client support uses system OpenSSL on Unix through `lumen-tls`, with
certificate and hostname verification. Availability depends on the platform and
installed libraries. `Lumen.serve` is the runtime's HTTP server API; check
`crates/lumen-web/src/server.rs` for current transport and buffering limits.

Streams is implemented. Preserve
[STREAMS_UPSTREAM.md](crates/lumen-web/src/js/STREAMS_UPSTREAM.md) and
[STREAMS_LICENSE](crates/lumen-web/src/js/STREAMS_LICENSE).
The upstream note documents the vendored bundle, its rebuild procedure, and the
private `_readSync` hook used by the buffered Fetch adapter.

Native addons enter through the dynamic library loader and N-API implementation
in `lumen-node`. Keep their ABI and lifecycle behavior covered by focused tests.
Package compatibility examples include Hono, React SSR, Vite/Rollup, and native
addons; their READMEs explain setup. Check implementation and tests before claiming
full Node, N-API, or web API compatibility: some module-header checklists are stale.

## Build and test

Run commands from the repository root. Build the runtime and engine shell separately:

```sh
cargo build --release -p lumen-cli
./target/release/lumen-cli -e 'console.log("Hello from Lumen!")'
./target/release/lumen-cli repl
./target/release/lumen-cli file.js

cargo build --release -p lumen --bin lumen
./target/release/lumen --tier=interp file.js
```

The engine shell lacks the runtime's host APIs. The REPL supports multiline input
and top-level await; line editing is line-buffered, with `rlwrap` usable for history.

Choose checks appropriate to the change. Common commands are:

```sh
cargo check --workspace --all-targets
cargo test -p lumen
cargo test -p lumen-runtime
cargo run --release -p lumen-difftest -- --count 2000
```

Run checks explicitly. Do not add Git hooks or hook installation machinery to this
project.

Differential tests compare completions, errors, side-effect traces, and final global
state, and minimize divergences into a regression corpus. Add focused regressions
for changed semantics and exercise the affected tiers. A native check alone does
not establish cross-platform or wasm32 correctness.

For Test262:

```sh
scripts/test262-clone.sh
scripts/run-test262.sh language/expressions/addition
LUMEN_TIER=jit scripts/run-test262.sh .
```

Without arguments the runner covers only language expressions and statements;
`.` selects the full test directory. Paths are relative to `test262/test`.
The runner writes `test262-report/summary.json`, including uncapped failure and
skip paths/reasons independently of optional console samples. Preserve the full
report. The process can exit successfully despite test failures; inspect the
totals and `all_executed_passed`, not just its exit status. Check its source for
current worker controls and limits before a broad run.

The old README recorded 53,574 passes, zero failures, and four skips on the default
JIT tier on 2026-08-26, against Test262 revision
`d86b2294eb0a17eaa281ff12c73c473ec864c72f`. This is a historical measurement, not
evidence about the current checkout. Report the tested engine revision, suite
revision, tier, scope, failures, and skips with new conformance results.

## Performance work

Use release builds and record the machine, revision, tier, flags, and workload.
Follow [benchmarks/README.md](benchmarks/README.md) for comparison and regression
policy. Check correctness as well as throughput, memory, startup, and compilation
costs. Investigation tools and reports may exist locally without being tracked;
do not assume a fresh clone includes them.

The tracked benchmark runners use external checkouts:

| Runner | Default checkout | Override | Focused run |
| --- | --- | --- | --- |
| `scripts/run-octane.sh` | `../octane` | `OCTANE` | `scripts/run-octane.sh richards crypto` |
| `scripts/run-web-tooling.sh` | `../web-tooling-benchmark` | `WEB_TOOLING_BENCHMARK_DIR` | `scripts/run-web-tooling.sh --only babel` |
| `scripts/run-ares6.sh` | `../ARES-6` | `ARES6` | `scripts/run-ares6.sh air basic` |

Octane uses the chromium/octane repository; Web Tooling uses v8/web-tooling-benchmark
and requires `npm install` in that checkout to build its CLI bundle. ARES-6 sources
are not vendored. All three runners accept `LUMEN_BIN` to use an existing binary.

Web Tooling's `--only` rebuilds the external bundle using webpack's build-time
selector; it is not a runtime filter. Full suites can be lengthy, so start with
selected workloads. Octane scores are higher-is-better; ARES-6 reports a geomean in
milliseconds, lower-is-better. A selected ARES-6 run is a partial result, not the
official full-suite score. Missing completion markers or failed workloads invalidate
a run.

## Documentation

Keep useful public setup instructions in READMEs, implementation guidance here,
and third-party provenance beside the vendored source. General Markdown workfiles
and the retired changelog are ignored and kept locally. Do not make builds or
required contributor instructions depend on those untracked documents.
