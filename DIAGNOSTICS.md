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
  native-call time, GC pauses and managed-memory availability metadata). The shell collects once more at
  exit, before printing it, so the GC counts and pauses include that collection (cause
  `explicit`, a major collection). Collections are counted by cause (`gc_causes`) and by kind
  (`gc_kinds`: `minor`, `major`), and their pauses are also reported by cause
  (`gc_pause_seconds_by_cause`) and by kind (`gc_pause_seconds_by_kind`). Production GC does
  not record a managed-memory snapshot: `managed_memory` reports `safepoint: null`,
  `complete: false`, and unavailable requested/external byte totals, not measured zeros.
  Live rooted-handle counts (`handles`: `strong` and `weak`) remain available independently;
  a leaked strong handle keeps its value alive. Diagnostic only; excluded from acceptance timing.
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
- `TRUST_JS_GC_LOG=1` logs collections, each line with its phases' times in microseconds:
  `mark` and `sweep` as the engine drives them, `engine` for what the engine does around the
  heap's phases, the heap's phases (`begin`: cards and marks; `roots`; `stack`; `side`:
  side-table edges and side-slot owners; `drain`: tracing; `finalize`; `free`), the engine's
  steps (`scrub`: zeroing the native stack below the collection's frames; `client`: building
  the marking client; `dead`: listing the dead objects for the tables' sweeps; `weak`:
  clearing weak handles and weak cells; `tables`: evicting the dead from the engine's tables;
  `after`: the Agent's bookkeeping after the sweep), then the cells finalized, by kind, and
  the time their finalizers took (`fin:<kind>=count fin_us:<kind>=microseconds`).
- Collections are minor or major. A minor collection traces from the roots and from the old
  cells (cells an earlier collection kept) whose cards the write barrier set since the last
  collection, and of the engine's side tables (internal slots kept outside objects, WeakMap and
  WeakSet entries, built-ins' [[Realm]]) only the entries written since then, and sweeps only
  the pages that received cells since then, so an old object that died stays allocated until a
  major collection, which marks and sweeps everything. A host may likewise report only its
  young state to a minor collection (`HostGcVisitor::is_minor`), declaring its old native
  identities live (`HostGcVisitor::native_old_generation`).
  Allocation safepoints and task boundaries collect minorly unless a major collection is due;
  explicit and idle collections, the shell's exit collection, ArrayBuffer pressure, the
  live-object limit and the memory reducer collect majorly.
- The heap sizes itself as V8 does (`gc_pacing.rs`). A minor collection is due once the
  nursery budget's bytes of cells have been allocated since the last collection: 2 MiB at
  first, doubling up to 32 MiB whenever the bytes promoted since it last grew pass it, and back
  to 2 MiB while allocation is slower than 1 KB/ms. A major collection leaves a byte budget of
  the bytes it kept times V8's dynamic growing factor (1.1 to 4: smaller the faster marking is
  than the program's promotion, so that major collections take about 3% of the time), at least
  8 MiB more than it kept, plus the nursery budget, and at most halfway from the bytes it kept
  to the heap's maximum, V8's maximum old generation size: half of the physical memory as V8
  reads it (`sysconf` on Linux, `hw.memsize` on macOS, `GlobalMemoryStatusEx` on Windows), at
  least 256 MiB and at most 4 GiB (256 MiB where the system does not say). The factor reaches 4
  only with a maximum of 2 GiB or more (4 GiB of memory); a smaller maximum caps it, as V8's
  `MaxGrowingFactor` does, at 1.3 up to 256 MiB, rising linearly toward 2 just below 2 GiB. At
  the factor of 4 with a 4 GiB maximum, the bound binds above about 585 MiB kept. V8 fails
  with out-of-memory at its maximum; the heap does not, and like V8's global limit the bound
  never leaves less room than 8 MiB, so a heap past its maximum keeps collecting at that room.
  V8's memory reducer runs when the host
  calls `Engine::run_memory_reducer` at `Engine::memory_reducer_deadline`: 8 s after a major
  collection that left the heap holding at least 10% and 10 MiB more than the reducer's last
  run (30 s after `Engine::notify_possible_garbage`, or after the heap first grew 1 MiB past its
  bootstrap without a major collection), it collects majorly to reduce memory if the program
  allocates slowly, at most twice, 0.5 s apart; such a collection grows the budget by 1.1,
  returns the nursery to 2 MiB and keeps no spare pages beyond the minimum, and while the
  reducer is done the budget grows by at most 1.3. `TRUST_JS_GC_PACING=0` fixes the budgets
  instead (the engine's unit tests' default; `1` turns pacing on there): a 2 MiB nursery and a
  byte budget of twice the bytes a major collection kept. For collector tuning,
  `TRUST_JS_GC_NURSERY_BYTES=N` fixes the nursery budget at N bytes (0 turns minor collections
  off: every collection is then major, due once the allocated bytes pass the byte budget).
  Objects' property and element storage and environments' binding storage are cells too.
  Bytes that cells own outside the heap (Map and Set tables, hash indexes, internal slots,
  bound arguments, stack traces) count as well: a collection is due once a nursery budget of
  them was acquired since the last one, and those still held count with the cells against the
  byte budget. `TRUST_JS_GC_MIN_HEAP_BYTES=N` replaces the 8 MiB floor of the heap's byte
  budget: a major collection is due once the heap holds more than the budget after a minor
  collection, its old cells counted whether they live or not. `TRUST_JS_GC_MAX_HEAP_BYTES=N`
  replaces the heap's maximum (pacing only), and with it the growing factor's cap, as V8's
  `--max-old-space-size` does (0 keeps V8's). Outside GC stress, cell allocation makes
  collections due through these budgets alone, and the live-object count only enforces the
  engine's live-object limit.
- `TRUST_JS_GC_STRESS=N` collects at the first safepoint after every N cell allocations and
  polls at every call (N=1 finds unrooted values; run each test in its own process, since a
  use-after-free can take down the whole run). Its collections are minor but for every Mth,
  which is major (`TRUST_JS_GC_STRESS_MAJOR=M`, 8 by default; 1 makes every one major). With
  `TRUST_JS_GC_VERIFY=1`, a debug build under stress also verifies the heap at every
  safepoint.
- `TRUST_JS_GC_VERIFY=1` verifies the heap at every collection, minor or major, in a debug
  build. The engine's unit tests always do; any other debug build only with it set (to
  anything but `0`), and release builds never. Each check walks the whole heap, so a verifying
  minor collection costs what a major one does; checks proportional to a collection's own
  work run in every debug build. As each collection starts it checks the write barrier: an
  old cell (one a collection kept) that holds a cell allocated since the last collection must
  have had its card set when it was written, and a panic `write barrier: old cell …` names
  both cells. The side tables' records are checked the same way: a side-table entry that
  holds such a cell, or belongs to such a cell, must have been recorded when it was written,
  and a panic `side-slot table …`, `weak_collection_data: …` or `native_function_realms: …`
  names the table, the owner and the value. So are the rooted handles' (the values host code
  keeps): a handle not written since the last collection must name a cell that collection
  kept (or, if weak, nothing), and a panic `handle table of heap …` names the entry and the
  cell. After marking, every edge of a marked cell must name a marked cell
  (`heap verifier: …`), and after a minor collection's marking the engine's caches and code
  roots and the atom table must hold no unmarked young cell.
- `TRUST_JS_GC_LOG=1` lines name each collection's kind and report live cells, live bytes,
  small-cell pages (64 KiB each) and released pages kept for reuse (`spare`) after it, the
  heap's collections so far, those that were due but postponed (`deferred`), the bytes its
  cells own outside it, not yet released (`external`), the byte budget, which each major
  collection sets (`budget`), and the heap's atoms, static names included, with the bytes of
  its own (`atoms`, `atom_bytes`); a minor collection's line adds
  the young pages it swept, the old cells it traced because their cards were set
  (`card_cells`) and the bytes it promoted. `TRUST_JS_GC_CENSUS=1`
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
- `TRUST_JS_PERF_MAP=1` (any value but `0`) lists every piece of code the engine
  generates in perf's JIT map, `/tmp/perf-<pid>.map`, the file V8's `--perf-basic-prof`
  writes: one `START SIZE name` line each, hexadecimal without `0x`, so `perf report`
  names samples in generated code instead of showing bare addresses. A compiled chunk is
  named as V8 names baseline code, `JS:^<name> <url>:<line>:<column>` at its function's
  first token (`<anonymous>` for a function without a name; code without a script, such
  as Script and module bodies, loop fragments and self-hosted built-ins, names its first
  locals instead: `JS:^<name> locals(a,b)`), a shared stub `Stub:<kind>` and a native
  RegExp program `RegExp:/<source>/<flags>`. The file is appended to, never truncated, and
  freed code stays listed (the format has no unload record); delete the file once the
  profile is reported. Off, it costs one check per piece of code; keep it off for
  uninstrumented timing comparisons. Call graphs need `--user-callchains` where
  `kernel.perf_event_paranoid` is 2: it refuses the kernel call chains perf otherwise
  requests. On a host with two PMU clusters (a Cortex-X925/A725 system, for one), pin perf
  with the workload to one core, or perf also reports a spurious count from the other
  cluster:

  ```sh
  TRUST_JS_PERF_MAP=1 taskset -c 5 perf record -e instructions:u --call-graph fp \
    --user-callchains -o vue.data -- target/release/trust-js benchmarks/fixtures/vue.js
  perf report -i vue.data --no-children --sort symbol --stdio
  ```

  While another perf session holds the user's locked-memory budget
  (`perf_event_mlock_kb`), `perf record` fails to open its events; smaller ring buffers
  (`-m 16`) let it open them.

## RegExp diagnostic controls

- `TRUST_JS_REGEXP_PROF=1` reports at exit the matcher's executed instructions by kind, the
  positions its prescans skipped, its attempts and backtracking entries, and the native one-byte
  tier's promotions, runs and code size. It costs a check per instruction: diagnostic only.
- `TRUST_JS_REGEXP_TIER_UP_AT=N` promotes an eligible pattern to the native one-byte tier after
  N executions (default 64); 0 keeps every pattern on the backtracking matcher.
- `TRUST_JS_REGEXP_NO_PREFILTER=1` compiles patterns without their prefilter: the literal every
  match contains and the range of its offset from the match start. Exec scans for that literal
  and tries only the starts that put an occurrence in range, and a sticky regex, or one anchored
  at the input's start, enters the matcher only when the literal lies in range of its one start.
  With the switch set, exec tries every start the first-element filter lets through.
- `TRUST_JS_REGEXP_NO_SIMD=1` keeps the subject scans (for the prefilter's literal, and for a
  match's first element by byte or by table) on their portable loops instead of AArch64 vector
  code.
- `TRUST_JS_REGEXP_NO_PROTECTOR=1` keeps the RegExp protector from holding. The protector
  proves for the running Realm that %RegExp.prototype%'s flag getters, `exec`, symbol methods
  and `constructor`, and %RegExp%'s @@species getter, are the intrinsics, and lets a RegExp of
  the shape and prototype RegExpAlloc gave it skip their lookups: String.prototype's `match`,
  `matchAll`, `replace`, `replaceAll`, `search` and `split` then call the RegExp algorithms
  directly, RegExpExec calls RegExpBuiltinExec, and the direct replace, split, test, flags and
  dead-result paths need no other guard. A RegExp of that shape holding a plain Number in
  `lastIndex` reads and writes it at its slot. With the switch set, every one of them takes the
  observable steps.

The fast paths' switches exist so that a workload can run with each forced off and be compared,
by instruction counts (`perf stat -e instructions:u`); none changes a result.

## Architecture diagnostics

The `architecture-diagnostics` feature records allocation sites/lifetimes
in the live object heap and call-cache epoch/refill traffic. It adds no object owners;
site and tracked-live-allocation limits report untracked allocations. Initial exotic family
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


### Optional Miri checks for root and interpreter access storage

The standalone harnesses import the actual engine modules without building the engine.
They are optional diagnostics and are not workspace build dependencies. The recorded
probe toolchain is `nightly-2026-10-23` (`rustc 1.101.0-nightly`, commit `7189d39d9`,
Miri `0.1.0`, LLVM `23.1.4`). Install that nightly's `miri` and `rust-src` components
separately if using these commands; normal builds do not require them.

```sh
MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly-2026-10-23 miri test --offline --manifest-path tools/agent-access-miri/Cargo.toml
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-tree-borrows" cargo +nightly-2026-10-23 miri test --offline --manifest-path tools/agent-access-miri/Cargo.toml
MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly-2026-10-23 miri test --offline --manifest-path tools/rooting-miri/Cargo.toml
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-tree-borrows" cargo +nightly-2026-10-23 miri test --offline --manifest-path tools/rooting-miri/Cargo.toml
```

The access harness exercises current-borrow publication, nested/reentrant access,
root movement, marking/sweep phase separation, quarantine and unwind. The rooting
harness uses inert Value/trace interfaces around the directly imported module;
it exercises raw ownership, registration, short mutation loans, argument/iterator
reentry, reallocation, transfer, registry-slot reuse and teardown. Passing these
checks covers those module call shapes, not the JIT, native stack scanner or the
whole JavaScript engine.

This explicit negative control must run one selected test and fail with Miri's
undefined-behavior diagnostic under either model; a compile failure or zero selected
tests is not a successful negative control:

```sh
MIRIFLAGS="-Zmiri-strict-provenance" cargo +nightly-2026-10-23 miri test --offline --manifest-path tools/agent-access-miri/Cargo.toml --features stale-suspension --lib tests::stale_suspension_is_a_miri_negative_control -- --exact
```
