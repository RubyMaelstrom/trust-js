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
  chunks (hex, one per line; assemble them with `clang --target=aarch64-linux-gnu -c` and
  `objdump -d`). `@<offset>` also requires the function's source start offset, the `@N` that
  `TRUST_JS_JIT_MAP=1` prints in its `[jit-map-range]` lines, which singles out one function among
  many that share names. Any value of `TRUST_JS_JIT_CODEDUMP` adds a `[jit-map]` line (runtime base
  and length) per compiled chunk.
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
- On GNU/Linux, `TRUST_JS_JIT_GPROFNG=1 gprofng collect app -o profile.er ...`
  registers live generated-code symbols with the already-loaded gprofng collector.
  Labels show the chunk's first local names, not source function names. This is
  optional compile/drop metadata; it neither instruments native instructions nor
  loads a profiler library. Keep it disabled for uninstrumented timing comparisons.

## Optimizing-JIT diagnostics and ablations

`TRUST_JS_OPT_JIT_LOG=1` reports
declines, and `TRUST_JS_OPT_JIT_DUMP=1` prints bytecode/CLIF for diagnosis.

`TRUST_JS_OPT_JIT_DIAGNOSTICS=1` adds bounded per-native-body compilation, entry,
helper, heap-guard, owner-transfer, frame-publication and specialization-bailout observations. It emits
compiler/drop records and includes live records in performance metrics. Disabled
emission adds no native instructions. `TRUST_JS_OPT_JIT_DIAGNOSTICS=compile` records
compilation/static counts and emits the ordinary uninstrumented body. Instrumented
IR is larger and still subject
to the same compiler limits; diagnostic runs cannot be acceptance timings.
`TRUST_JS_OPT_JIT_DIAGNOSTICS=execution` separately records physical optimized entries,
normal/error exits, terminal VM recoveries, specialization retirement and the first eight
emitted committed data-property read/existing-overwrite, literal-comparison, ordered-case
or borrowed-owner-window sites. One dispatch event counts its group; one borrowed event
counts a local copy or owning call-input materialization, not its loads or call completion.
Normal exits include recovered VM completions;
native control flow may still call checked helpers or author code. Loop-continuation and
whole-function retirement use separate miss states. Creation, computed elements and scalar
loop-field accesses are outside the site counters. Execution counters saturate; the legacy
Full-mode counters keep their existing behavior. Shared-owner leaf counters are Full-only
and have distinct storage from execution sites. Dynamic diagnostic metadata capacities are
included in retained memory, and the directory holds weak references.

The separate `architecture-diagnostics` feature records allocation sites/lifetimes
in the live Rc heap and call-cache epoch/refill traffic. It adds no object owners;
site and registry-slot limits report untracked allocations. Initial exotic family
is recorded at allocation, before later function/exotic initialization. Lifetimes
use heap allocation ordinals and completed collections, not elapsed time. Live
families at collection are a separate population sample. These probes compile out
of ordinary builds; use them only for diagnostics, never performance acceptance.

`TRUST_JS_OPT_JIT_DEOPT_AT=N` is a development stress hook: an eligible optimized function
exits to its ordinary VM continuation when it reaches bytecode boundary N, before that operation.
It is unset by default, is not a type-specialization policy, and must be absent during
performance measurements. Suspended frames are not admitted by this hook; eligible flattened
inline bodies continue in their canonical remapped slots and logical source frames.

`TRUST_JS_OPT_JIT_SPARSE_RELOADS=0` is a compiler-emission diagnostic ablation: reload all
tracked local register copies after observers instead of the bounded liveness plan. It does
not remove canonical frame owners, change GC roots, or disable state publication. Keep the
setting explicit in measurement manifests; the optimizing tier itself remains opt-in.
`TRUST_JS_OPT_JIT_LOCAL_EFFECTS=0` retains the all-clobber helper analysis for a diagnostic
comparison. The development default separates canonical publication from private-slot
write effects: preserved register words do not imply immutable objects or environments.
Full-reload mode (`TRUST_JS_OPT_JIT_SPARSE_RELOADS=0`) still forces all tracked reloads.
`TRUST_JS_OPT_JIT_VALUE_FACTS=0` disables the experimental whole-function value-category
analysis and its guard/owner-check elisions. The analysis proves normal-result categories,
not immutable heap contents or numeric ranges; unknown effects and completion landings
retain dynamic checks. Budget exhaustion also retains those checks. This switch skips
analysis itself, unlike the reload-emission-only ablation above.

Release optimizing compilation validates finalized CLIF once before backend work.
`TRUST_JS_OPT_JIT_VERIFY=1`, debug builds and tests additionally enable Cranelift's repeated
internal verifier passes. This diagnostic does not change JavaScript guards or recovery.

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
