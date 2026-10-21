# TRust JS diagnostics and diagnostic inputs

This is the central index for TRust JS's developer diagnostics. It consolidates
the environment variables, diagnostic features, and benchmark runners that are
otherwise described in module comments and experiment documents. It is an
engineering aid, not part of the normal runtime interface.

The source remains authoritative when a diagnostic's output or defaults change.
When adding or changing an input, update this file and the nearby source
comment/example in the same change.

## Conventions

* Value inputs are normally parsed once at process startup. Set them before
  launching the engine shell, runtime, or test process; changing the
  environment while it is running is not reliable.
* Instrumented or diagnostic runs are not performance acceptance timings.
  Keep diagnostic switches out of measurement manifests unless a switch's own
  entry below says to record it explicitly.

## Execution tier and JIT diagnostic controls

- `TRUST_JS_TIER=interp|bytecode|jit` selects the tier.
- `TRUST_JS_TIER_THRESHOLD=N` changes the call threshold for tiering up. Cold ordinary
  synchronous author functions with loops obey it too; async/generator coroutine bytecode
  remains eager, and self-hosted methods compile on their first call. A long-running
  synchronous invocation can tier up after 128 executed
  backedges without repeating its parameter binding, hoisting or prefix effects. Eligible
  body-level loops transfer their whole function frame. Captures, mapped arguments and eval
  retain their live environments; unsupported activation projections or active outer author
  cleanup use the existing loop-fragment path. Ordinary/base constructor frames can transfer
  after `this` is initialized; derived constructors retain their contextual fragment path.
  Sloppy slot-backed formals preserve legacy argument maps through scoped shared slot Cells;
  borrowed native observer boundaries publish and reload those homes. Function-owned fragments
  can project uncaptured bindings, publishing every completion before parked AST cleanup runs.
  Every projected binding transfers its Value owner to its slot while the AST record is parked;
  scoped teardown restores its current value before unregistering or recycling frame storage.
  Retained AST evaluator plans that copy a projected alias keep environment-backed fragments.
- `TRUST_JS_JIT_ADMIT_RUNS=N` sets how many bytecode VM runs a loop-free function makes before
  it is compiled to native code (default 0: the first VM run compiles it). A body with a loop
  skips that delay once admitted to bytecode: "first run" means its first VM/native-admission
  invocation, after the call threshold or live-frame transfer, not its first JavaScript call.
  Borrowed Script/loop-fragment entries use their own executed-backedge/work admission policy.
  `TRUST_JS_TIER_LOG=1` reports each native compile.
- `TRUST_JS_INLINE_AT=N` changes the speculative inline recompile threshold (default:
  100 machine-code runs); zero disables inlining.
- `TRUST_JS_JIT_NO_DIRECT_CALLS=1` disables ARM64 direct calls (the inline call sequence
  that runs the callee on a pooled frame record) while retaining ordinary JIT calls.
- `TRUST_JS_JIT_INLINE_STUBS=1` emits ARM64 name-cache validation, whole free-name reads,
  slow-path helper calls, generic property-cache probes, secondary call-cache probes and
  direct-call sequences at every site instead of as per-chunk shared stubs (by default only
  loop bodies keep name validation, whole name reads and direct calls in line, and every
  slow-path helper call is a stub call). It is a code-size/performance ablation with the
  same behavior. Compiled DOOM (archive.org's asm.js DOSBox) is bound by instruction-cache
  pressure: making sites larger, by inlining stubs or sequences, slowed it by 10-18% in
  paired runs.
- `TRUST_JS_JIT_NO_JSCVT=1` makes ARM64 code use the portable guarded ToInt32 sequence even
  when the CPU implements FEAT_JSCVT (`fjcvtzs`).
- `TRUST_JS_TAIL_CALL_DEPTH=N` sets the execution-context depth below which a strict
  tail call runs as an ordinary call (default 1024); at or past it the call is staged as a
  proper tail call, so unbounded tail recursion stays bounded. `0` stages every tail call.
- `TRUST_JS_JIT_HOT_LOOP_OPS=N` changes the 512-op bound on loops that keep ARM64 name checks
  and direct-call sequences in line; larger (flattened dispatch) loops use shared stubs.
- `TRUST_JS_JIT_CHUNK_STUBS=1` gives each ARM64 chunk its own copy of the shared stubs instead
  of veneers to the process-wide stub table. It is a code-size ablation with the same behavior.
- `TRUST_JS_TIER_LOG=1` helps diagnose compilation bailouts.
- `TRUST_JS_JIT_DUMP=<names>` prints the bytecode of chunks whose first four slot names contain
  the text, and `TRUST_JS_JIT_CODEDUMP=<names>[@<offset>]` the finished ARM64 words of the same
  chunks (`*` dumps every chunk; hex, one per line; assemble them with `clang
  --target=aarch64-linux-gnu -c` and `objdump -d`). `@<offset>` also requires the function's
  source start offset, the `@N` that `TRUST_JS_JIT_MAP=1` prints in its `[jit-map-range]` lines,
  which singles out one function among many that share names. Any value of
  `TRUST_JS_JIT_CODEDUMP` adds a `[jit-map]` line (runtime base and length) per compiled chunk.
- `TRUST_JS_JIT_NO_NAME_ELEM=1` compiles `LoadName K; <pure key>; GetElem/SetElemDrop` register
  regions with the ordinary LoadName and element templates instead of reading a typed array
  straight from its binding (see `NameElementPlan` in `jit_regions.rs`).
- `TRUST_JS_JIT_TYPED_STUBS=0` carries the native typed-array element sequence in line at every
  site instead of calling the process-wide shared stub, and `TRUST_JS_JIT_NO_NAME_ELEM_STUB=1` does
  the same for the fused name/element access (the sites then keep only the name probe shared).
  Compiled applications such as DOOM's emulator are bound by instruction fetch: shared stubs
  cost a call and return per access and save more in instruction cache misses.
- `TRUST_JS_JIT_NO_INT_REGS=1` keeps every int32 result of a register region (bitwise operators,
  shifts, `Math.imul`) in a floating-point register instead of leaving it in an integer
  register for the next operator, comparison or branch (`Operand::Int` in `jit_regions.rs`).
- `TRUST_JS_PERF_METRICS=1` prints one JSON JIT/collector performance summary at normal
  exit (compile attempts/seconds/bytes, inline attempts, lex/parse/bytecode-compile time,
  native-call time, GC pauses and post-GC managed memory). The shell collects once more at
  exit, before printing it, so the GC counts and pauses include that collection (cause
  `explicit`, a major collection). Collections are counted by cause (`gc_causes`) and by kind
  (`gc_kinds`: `minor`, `major`), and their pauses are also reported by cause
  (`gc_pause_seconds_by_cause`) and by kind (`gc_pause_seconds_by_kind`). Diagnostic only;
  excluded from acceptance timing.
- Helper-traffic tallies print their top entries to stderr at exit and are diagnostic only:
  `TRUST_JS_JIT_OPSTAT=1` counts ops that reach a JIT helper (`=2` adds the function and
  pc), `TRUST_JS_JIT_CALLSTAT=1` counts why calls reaching the call helper missed the
  direct-call sequence, `TRUST_JS_JIT_NATIVESTAT=1` counts those calls by native callee,
  and `TRUST_JS_JIT_PROPSTAT=1` counts why named reads reaching a JIT property helper were
  not served by the inline template (proxy, exotic/side-table, accessor, array holder,
  depth, absent, or uncached/cached data; `=2` adds the site). `TRUST_JS_JIT_PROFILE_AFTER_MS=N`
  delays the OPSTAT/CALLSTAT/PROPSTAT counting window past startup.
- `TRUST_JS_JIT_MAP=1` prints `[jit-map-range]`/`[jit-map-pc]`/`[jit-map-mark]` lines mapping
  generated code ranges to chunk source offsets, bytecode pcs and attribution families, for
  joining sampling-profiler addresses back to operations.
- `TRUST_JS_FEEDBACK_PROFILE=1` enables the detailed per-site feedback vector; it changes
  the execution paths that collect it, so its observations are not observations of ordinary
  optimized execution and it is never an acceptance setting.
- `TRUST_JS_GC_LOG=1` logs collections.
- Collections are minor or major. A minor collection traces from the roots and from the old
  cells (cells an earlier collection kept) whose cards the write barrier set since the last
  collection, and sweeps only the pages that received cells since then, so an old object that
  died stays allocated until a major collection, which marks and sweeps everything.
  Allocation safepoints and task boundaries collect minorly unless a major collection is due;
  explicit and idle collections, the shell's exit collection, ArrayBuffer pressure and the
  live-object limit collect majorly. For collector tuning, `TRUST_JS_GC_NURSERY_BYTES=N`
  replaces the 2 MiB nursery budget: a minor collection is due once N bytes of cells have
  been allocated since the last collection (0 turns minor collections off: every collection
  is then major, due once the allocated bytes pass the byte budget).
  `TRUST_JS_GC_MIN_HEAP_BYTES=N` replaces the 8 MiB floor of the heap's byte budget (twice
  the bytes a major collection leaves, at least the floor): a major collection is due once
  the heap holds more than the budget after a minor collection, its old cells counted whether
  they live or not. Outside GC stress, cell allocation makes collections due through these
  budgets alone, and the live-object count only enforces the engine's live-object limit.
- `TRUST_JS_GC_STRESS=N` collects at the first safepoint after every N cell allocations and
  polls at every call (N=1 finds unrooted values; run each test in its own process, since a
  use-after-free can take down the whole run). Its collections are minor but for every Mth,
  which is major (`TRUST_JS_GC_STRESS_MAJOR=M`, 8 by default; 1 makes every one major). Debug
  builds also check the write barrier as every collection starts, minor or major, so stress
  checks it at every safepoint: an old cell (one a collection kept) that holds a cell
  allocated since the last collection must have had its card set when it was written, and a
  panic `write barrier: old cell …` names both cells.
  `TRUST_JS_GC_LOG=1` lines name each collection's kind and report live cells, live bytes,
  small-cell pages (64 KiB each) and released pages kept for reuse (`spare`) after it; a
  minor collection's line adds the young pages it swept, the old cells it traced because
  their cards were set (`card_cells`) and the bytes it promoted. `TRUST_JS_GC_CENSUS=1`
  adds a `gc_layout_census` to the `TRUST_JS_PERF_METRICS` record: named-property counts of the
  live objects after every collection and of objects as they die, by family (it walks the
  heap at each collection; diagnostic only). In test builds, `TRUST_JS_GC_WHY=1` records
  why each cell was marked (tests print the path with `gc_test::why`/`why_address`; a
  stack root names its frame's return address, for `addr2line`, and its depth below that
  frame's record), and `TRUST_JS_SEGV_BACKTRACE=1` prints a backtrace on a fault.
- On GNU/Linux, `TRUST_JS_JIT_GPROFNG=1 gprofng collect app -o profile.er ...`
  registers live generated-code symbols with the already-loaded gprofng collector.
  Labels show the chunk's first local names, not source function names. This is
  optional compile/drop metadata; it neither instruments native instructions nor
  loads a profiler library. Keep it disabled for uninstrumented timing comparisons.

## Architecture diagnostics

The `architecture-diagnostics` feature records allocation sites/lifetimes
in the live object heap and call-cache epoch/refill traffic. It adds no object owners;
site and registry-slot limits report untracked allocations. Initial exotic family
is recorded at allocation, before later function/exotic initialization. Lifetimes
use heap allocation ordinals and completed collections, not elapsed time. Live
families at collection are a separate population sample. These probes compile out
of ordinary builds; use them only for diagnostics, never performance acceptance.

## Benchmark runners

The tracked benchmark runners use external checkouts:

| Runner | Default checkout | Override | Focused run |
| --- | --- | --- | --- |
| `scripts/run-octane.sh` | `../octane` | `OCTANE` | `scripts/run-octane.sh richards crypto` |
| `scripts/run-web-tooling.sh` | `../web-tooling-benchmark` | `WEB_TOOLING_BENCHMARK_DIR` | `scripts/run-web-tooling.sh --only babel` |
| `scripts/run-ares6.sh` | `../ARES-6` | `ARES6` | `scripts/run-ares6.sh air basic` |

Octane uses the chromium/octane repository; Web Tooling uses v8/web-tooling-benchmark
and requires `npm install` in that checkout to build its CLI bundle. ARES-6 sources
are not vendored. All three runners accept `TRUST_JS_BIN` to use an existing binary.

Web Tooling's `--only` rebuilds the external bundle using webpack's build-time
selector; it is not a runtime filter. Full suites can be lengthy, so start with
selected workloads. Octane scores are higher-is-better; ARES-6 reports a geomean in
milliseconds, lower-is-better. A selected ARES-6 run is a partial result, not the
official full-suite score. Missing completion markers or failed workloads invalidate
a run.
