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
- `TRUST_JS_TIER_THRESHOLD=N` sets the tier-up budget of ordinary synchronous author
  functions: N calls, where each backedge the tree-walker runs in a function's own loops
  counts as a sixteenth of a call (16·N backedges, or any mix). A function of 8 KB or more of
  source (or of its statements' snapshot encoding) needs proportionally more backedges, 16·N
  per 4 KB, as V8 scales its budget by bytecode length. Async/generator coroutine bytecode
  remains eager, and self-hosted methods compile on their first call. Once its function has
  tiered up, a call
  still on the tree-walker leaves it at its next loop header, and any loop leaves after 128
  backedges of one execution, without repeating its parameter binding, hoisting or prefix
  effects. A for, while or do-while loop the body reaches through blocks, ifs, labels and other
  such loops transfers the whole function frame (before its budget is spent, a function of
  8 KB or more transfers only at its body-level loops); a loop under try, catch, finally,
  switch, with, for-in or for-of enters a compiled fragment. Captures,
  mapped arguments and eval
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
  `TRUST_JS_TIER_LOG=1` reports each native compile, and each bytecode tier-up compile with its
  reason (calls, loop budget or loop transfer).
- `TRUST_JS_INLINE_AT=N` changes the speculative inline recompile threshold (default:
  100 machine-code runs); zero disables inlining.
- `TRUST_JS_JIT_NO_CHAIN_REOPTIMIZE=1` keeps a function on its chunk when its operator chains'
  property reads and stores keep bailing. By default the generic helper counts each property
  operation a chain's bail replays while its site's cache holds the receiver's shape, and at
  100 the function recompiles once without property chains (`FunctionRuntime::code3`);
  `TRUST_JS_TIER_LOG=1` reports each such recompile with its replay count, its inline sites
  and its bytecode compile time, or why it was declined.
- `TRUST_JS_JIT_NO_TDZ_PROOF=1` keeps the ARM64 TDZ test of every local read. By default a
  compile proves which `LoadLocal`s find their slot initialized on every path
  (`Cfg::initialized_reads`: parameters, `var`s, temporaries, and lexicals a store or an
  earlier read reaches) and emits those reads, and the `LoadLocal; Pop` pair that guards a
  lexical's assignment, without the test.
- `TRUST_JS_JIT_NO_DIRECT_CALLS=1` disables ARM64 direct calls and constructions (the inline
  call and `new` sequences that run the callee on a pooled frame record) while retaining
  ordinary JIT calls.
- `TRUST_JS_JIT_KEEP_FP_SAVES=1` keeps every ARM64 chunk's save and restore of d8-d15, which
  chunks whose code writes none of them otherwise skip, with a 64-byte native frame for 128.
- `TRUST_JS_JIT_INLINE_STUBS=1` emits ARM64 name-cache validation, whole free-name reads,
  slow-path helper calls, generic property-cache probes, secondary call-cache probes and
  direct call and `new` sequences at every site instead of as per-chunk shared stubs (by
  default only loop bodies keep name validation, whole name reads and direct sequences in
  line, and every slow-path helper call is a stub call). It is a code-size/performance
  ablation with the same behavior. Compiled DOOM (archive.org's asm.js DOSBox) is bound by
  instruction-cache pressure: making sites larger, by inlining stubs or sequences, slowed it
  by 10-18% in paired runs.
- `TRUST_JS_JIT_NO_JSCVT=1` makes ARM64 code use the portable guarded ToInt32 sequence even
  when the CPU implements FEAT_JSCVT (`fjcvtzs`).
- `TRUST_JS_FOR_OF_STEP=helper` compiles every synchronous `for…of` step to the one iteration
  operation whose helper steps the loop in Rust, instead of calling a `next` method that is not a
  built-in function through a call cache and reading its result's `done` and `value` through
  property caches; `TRUST_JS_JIT_NO_ITER_STEP=1` keeps ARM64's in-line Array step out of
  generated code. Both are ablations with the same behavior.
- `TRUST_JS_TAIL_CALL_DEPTH=N` sets the execution-context depth below which a strict
  tail call runs as an ordinary call (default 1024); at or past it the call is staged as a
  proper tail call, so unbounded tail recursion stays bounded. `0` stages every tail call.
- `TRUST_JS_JIT_HOT_LOOP_OPS=N` changes the 512-op bound on loops that keep ARM64 name checks
  and direct-call sequences in line; larger (flattened dispatch) loops use shared stubs.
- `TRUST_JS_JIT_CHUNK_STUBS=1` gives each ARM64 chunk its own copy of the shared stubs instead
  of veneers to the process-wide stub table. It is a code-size ablation with the same behavior.
- `TRUST_JS_TIER_LOG=1` helps diagnose compilation bailouts.
- `TRUST_JS_SUMMARY_SOURCE=<script file>` names the Script that the ignored lib test
  `lazy_snapshot_body_tests::summary_source_checks` checks the capture summaries of a static
  snapshot on (`cargo test -p trust-js --lib --features embed -- --ignored summary_source`),
  such as a browser's assembled platform prelude: every function's stored summary equals walking
  the parsed function, every function plans alike from stored summaries, from summaries computed
  by scanning and from the parsed source, and the bootstrap function's plan decodes no body. It
  prints the snapshot's size and its summaries' share.
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
- `TRUST_JS_JIT_FIRST_STAGE_SPANS=1` gives first stages their acyclic register regions back. By
  default a function's first native compile has loop regions but no acyclic spans: those come
  with its second stage, the inline recompile of a function that has run (a borrowed frame's
  compile, a loop transfer or a resumable body, keeps them).
- `TRUST_JS_JIT_UNRUN_REGIONS=1` gives second stages the register regions of spans that had not run
  when they were compiled. By default a second stage emits a region only where a named read's
  cache or a call site's targets, seeded from the stage that ran, show its operations ran (a span
  with neither keeps its region).
- `TRUST_JS_JIT_NO_NATIVE_STUB=1` keeps a call site's call-hit form on the call-hit helper. By
  default (ARM64, shared stubs on, at most eight arguments) the form calls a shared stub that runs
  a native of a light class on light operands (`bytecode::native_operands`) through its fast
  entry, `bytecode::jit_native_light`, without the helper's decode of the call.
- `TRUST_JS_JIT_PUBLISH_EPOCH=1` moves the call caches' process-wide epoch at each new code a
  function gets (an inline recompile, a recompile without property chains), so that every call
  site re-resolves, as before per-callee pickup. By default the new code supersedes the chunk it
  replaces, and only that chunk's callers refill their ways.
- `TRUST_JS_JIT_OVERFLOW_RENEWAL=<n>` renews a megamorphic call site's admissions to the call
  caches' overflow table after `n` of its calls went through the full guards for want of an entry
  (1 to 65535; by default 1024, `CALL_SITE_OVERFLOW_RENEWAL`).
- `TRUST_JS_JIT_NO_INT_REGS=1` keeps every int32 result of a register region (bitwise operators,
  shifts, `Math.imul`) in a floating-point register instead of leaving it in an integer
  register for the next operator, comparison or branch (`Operand::Int` in `jit_regions.rs`).
- `TRUST_JS_PERF_METRICS=1` prints one JSON JIT/collector performance summary at normal
  exit (compile attempts/seconds/bytes, inline attempts, lex/parse/bytecode-compile time,
  native-call time, GC pauses and managed-memory availability metadata). The shell collects once more at
  exit, before printing it, so the GC counts and pauses include that collection (cause
  `explicit`, a major collection). Collections are counted by cause (`gc_causes`) and by kind
  (`gc_kinds`: `minor`, `major`), and their pauses are also reported by cause
  (`gc_pause_seconds_by_cause`) and by kind (`gc_pause_seconds_by_kind`); a marking cycle's
  slices (its start among them) are not collections, and are counted apart (`gc_slices`,
  `gc_slice_pause_seconds`, `gc_max_slice_pause_seconds`). Production GC does
  not record a managed-memory snapshot: `managed_memory` reports `safepoint: null`,
  `complete: false`, and unavailable requested/external byte totals, not measured zeros.
  Live rooted-handle counts (`handles`: `strong` and `weak`) remain available independently;
  a leaked strong handle keeps its value alive. Diagnostic only; excluded from acceptance timing.
- Helper-traffic tallies print their top entries to stderr at exit and are diagnostic only:
  `TRUST_JS_JIT_OPSTAT=1` counts ops that reach a JIT helper (`=2` adds the function and pc),
  `TRUST_JS_JIT_CALLSTAT=1` counts why calls reaching the call helper missed the direct-call
  sequence (`=2`, in builds with the `architecture-diagnostics` feature, adds `[jit-callpath]`
  counts of calls by the path that entered the callee: the direct sequence, the call-hit helper
  for user callees and for natives with and without the native-call boundary, the generic call
  helper, the intrinsic helper, cached calls from Rust and `Interp::call`),
  `TRUST_JS_JIT_NATIVESTAT=1` counts those calls by native callee, and `TRUST_JS_JIT_PROPSTAT=1`
  counts why named reads reaching a JIT property helper were not served by the inline template
  (proxy, exotic/side-table, accessor, array holder, depth, absent, or uncached/cached data;
  `=2` adds the site). `TRUST_JS_JIT_PROFILE_AFTER_MS=N` delays the OPSTAT/CALLSTAT/PROPSTAT
  counting window past startup.
- `TRUST_JS_JIT_MAP=1` prints `[jit-map-range]`/`[jit-map-pc]`/`[jit-map-mark]` lines mapping
  generated code ranges to chunk source offsets, bytecode pcs and attribution families, for
  joining sampling-profiler addresses back to operations.
- `TRUST_JS_FEEDBACK_PROFILE=1` enables the detailed per-site feedback vector; it changes
  the execution paths that collect it, so its observations are not observations of ordinary
  optimized execution and it is never an acceptance setting.
- `TRUST_JS_FEEDBACK_CENSUS=<path>` (builds with the `architecture-diagnostics` feature; ordinary
  builds compile it out) writes the feedback census to `<path>` at the shell's exit: the
  operations the VM's loop dispatched, by chunk and bytecode pc, and each site's production
  feedback (its property, call, construct and name caches and allocation images, snapshotted when
  its chunk drops or at exit). Run a program twice, once with `--tier=bytecode` (every
  compiled-code operation passes the VM's loop, so its counts are executions) and once with the
  production `--tier=jit`, and `python3 -I scripts/feedback-census.py <bytecode file> <jit file>`
  prints, by operation kind, the executions, the share whose site has usable feedback (coverage)
  and the share whose feedback names exactly what the operation saw (precision), reading the
  always-on records the file's header names (`records=`; `--records values` reads the files as
  if the build wrote only those) by crates/trust-js/src/feedback-schema.txt's sections 4 and 5. A
  function's first calls in the tree-walker have no pcs and are not counted.
- `TRUST_JS_GC_LOG=1` logs collections, each line with its phases' times in microseconds:
  `pause`, the whole collection, the time the program waited for it; `mark` and `sweep` as the
  engine drives them, `engine` for what the engine does around the heap's phases; for a major
  collection, which is the final pause of its marking cycle, `cycle_mark`, the cycle's whole
  marking time (its slices' and this pause's marking); the heap's phases (`begin`: cards and marks; `roots`; `stack`; `side`:
  side-table edges and side-slot owners; `drain`: tracing; `finalize`; `free`), the engine's
  steps (`scrub`: zeroing the native stack below the collection's frames; `client`: building
  the marking client; `dead`: listing the dead objects for the tables' sweeps; `weak`:
  clearing weak handles and weak cells; `tables`: evicting the dead from the engine's tables;
  `after`: the Agent's bookkeeping after the sweep), then the cells finalized, by kind, and
  the time their finalizers took (`fin:<kind>=count fin_us:<kind>=microseconds`).
- `TRUST_JS_GC_INCREMENTAL=0` turns off incremental marking (W6), which is on by default but in
  wasm32 builds (a slice's time needs a clock): every major collection then marks in one pause.
  With it on, the heap marks the major collections it asks for on its own (its byte budget, a
  stress interval, the memory reducer) in a cycle between whose slices the program runs, once
  the heap holds more than 8 MiB (in release builds): a minor
  collection that leaves the room below the byte budget at one nursery's budget or less starts
  it (V8's soft limit). The cycle's start marks the roots, but not the native stack, which the
  final pause scans (V8's start), and each slice traces part of what they reach, with the
  engine's side tables (internal slots kept outside objects, WeakMap and WeakSet entries,
  discovered as their collections are marked), at the first safepoint after every 64 KiB
  allocated: the bytes V8's schedule asks (those due by now if the cycle's marking took 500 ms,
  or by the time it has allocated half its room before the margin, whichever is more, and at
  least 64 KiB), for at most a millisecond. The words of the element blocks its traces read
  count in a slice's bytes (8 each), and its clock is read every 128 cells and every 4,096 such
  words; a block of more words than that (32 KiB) is traced 4,096 of them at a time, its owner
  queued again for the next chunk (as V8 does), and a take of the owner's card (see below) traces
  it again from its first word. A host's event loop advances a cycle while it waits
  (`Engine::advance_incremental_marking`), as the runtime does. Each time nothing is left to
  trace, a slice precleans: the marked cells written since their trace (their cards set) are
  traced again while the program runs, until a preclean finds fewer than 7,000 such cells and
  what they reach is under 1 MiB; a final pause once nothing is left after that marks the roots,
  the stack and the cells written since (their cards) again, with the side-table entries written
  during the cycle, then sweeps as a major collection does; the heap's growth counts the cycle's
  whole marking time. No minor collection runs while a cycle marks; the cycle's allocation past
  a margin over the byte budget (V8's), and every requested collection (explicit, the exit's,
  the live-object limit, ArrayBuffer pressure, an explicit minor one) finish it at once; an
  explicit major collection that finishes one is followed by another, which frees what the cycle
  marked before it died. `TRUST_JS_GC_INCREMENTAL=force` makes the slices a few cells each, at every
  safepoint while a cycle marks, and a block's chunks 16 words, or a larger block's eighth, each
  slice ending with the next 16 pages of a preclean's pass (which leaves the cards of chunked
  blocks' owners set, for a pass begun out of work or the final pause to take, as an ordinary
  cycle's precleans, out of work, never meet a block between its chunks), and with
  `TRUST_JS_GC_VERIFY` each take of the cards (a preclean's, the final pause's) is checked against a
  walk over the marks of their pages, the slices' ends (the first 64 of a cycle, then every 64th)
  and the final pause's start check that no traced cell with a clean card holds an edge to an
  unmarked one, and the final pause's end that every side-table entry of a marked owner names marked
  cells. `TRUST_JS_GC_LOG=1` prints a `[gc-slice]` line for the cycle's start (`phase=start`) and
  each slice: the slices so far (`n`), the bytes of the cells it traced (`traced`), the bytes it
  marked and the cycle's so far (`cycle_marked`), the cells left in the worklist, the bytes
  allocated since the cycle started (`cycle_alloc`), the cycle's precleans so far and the cells the
  last one found (`precleans`, `precleaned`), whether the marking is complete but for the final
  pause, and its `pause` in microseconds, the start's with the heap's phases in it (`begin`:
  clearing the marks and cards, `roots`, `side`: the side tables' edges; the rest is the engine's);
  the final pause's `[gc-heap]` line is a major collection's.
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
  V8's memory reducer runs when the host calls `Engine::run_memory_reducer` at
  `Engine::memory_reducer_deadline`: 8 s after a major collection that left the heap holding at
  least 10% and 10 MiB more than the reducer's last run (30 s after
  `Engine::notify_possible_garbage`, or after the heap first grew 1 MiB past its bootstrap
  without a major collection), it collects majorly to reduce memory if the program allocates
  slowly, at most twice, 0.5 s apart; such a collection grows the budget by 1.1, returns the
  nursery to 2 MiB and keeps no spare pages beyond the minimum, and while the reducer is done
  the budget grows by at most 1.3. The pages a collection frees beyond the spare pages the heap
  keeps (a nursery's budget of them, at least 16, or without a nursery its budget's room) stay
  pooled for reuse, out of its pause, and no longer count in its committed memory; they go back
  to the system when the host calls `Engine::run_memory_reducer` 8 s after the pool formed
  (`Engine::memory_reducer_deadline` then names that time too), or at once in a collection that
  reduces memory (V8's memory pool); a host that never calls it keeps them. Without pacing,
  every collection gives them back as it ends. `TRUST_JS_GC_PACING=0` fixes the budgets
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
  and a panic `side-slot table …`, `weak collection: …` or `native_function_realms: …`
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
  (`card_cells`) and the bytes it promoted, and a major collection's the slices that marked
  before its final pause (`slices`), the marked cells the final pause traced again because their
  cards were set (`card_cells`) and the bytes allocated from its cycle's start to that pause
  (`cycle_alloc`): 0 for a major collection that marks in one pause. `TRUST_JS_GC_CENSUS=1`
  adds a `gc_layout_census` to the `TRUST_JS_PERF_METRICS` record: named-property counts of the
  live objects after every collection and of objects as they die, by family (it walks the
  heap at each collection; diagnostic only), and prints a `[gc-elements]` line after each
  collection: for arrays, arguments objects and other objects with index keys or element
  storage, their count, element slots, elements and holes, the plain ones (no accessor, no
  attributes but the defaults, no far key) and of those the holey, all-Number and all-int32
  ones, and the bytes of each element storage part (element words are the `values` part: the
  block with its header and remembered record; a map's own key index is `index`; `sidecar`,
  `packed_box`, `dirty`, `slot_map` and `mirror` stay zero, kept so lines compare with older
  ones), beside the same objects' bytes under W2c's model of 8-byte words (`w2c`). The
  population census's `sidecars` column counts maps with a key index of their own. In test
  builds,
  `TRUST_JS_GC_WHY=1` records
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
  positions its prescans skipped, its attempts and backtracking entries, and the native tier's
  compilations (tried and made), runs, code size and compile time. It costs a check per
  instruction: diagnostic only. `TRUST_JS_REGEXP_PROF=2` adds a census by pattern
  (`[regexp-census]` lines): for the 40 source and flag pairs with the most work (dispatched
  instructions, four per attempt or native run and one per four scanned positions), their share
  of those counters, their executions, matches, executions on two-byte subjects, native runs,
  native code bytes and compile time (`native_us`), and their programs' features (length, groups,
  instructions by kind with lookaround bodies, atomic loops, leading run, first-element filter,
  prefilter, single-element and literal paths). Each execution then also looks its pattern up by
  source and flags.
- `TRUST_JS_REGEXP_TIER_UP_AT=N` compiles a pattern to native code for a subject width (one
  byte or two a unit) after N executions (default 64), or at once over a subject of 1,000 units
  or more, as V8 does, each width on its first use; 0 keeps every pattern on the backtracking
  matcher, and 1 compiles at the first execution. The native tier (AArch64 and x86-64, on
  Unix) takes programs of characters, `.`, classes, captures, alternation, `^`, `$`, `\b`, `\B`,
  loops and groups' repetitions, lookaheads and lookbehinds, backreferences, inline modifier
  groups and counted loops, under `i` once they are closed over case (see
  `TRUST_JS_REGEXP_NO_CASE_CLOSURE`), and in Unicode mode, where it steps a two-byte subject's
  surrogate pair as one element; `v`-mode class strings stay on the matcher.
- `TRUST_JS_REGEXP_NATIVE_VERIFY=1` runs the matcher after every native execution and aborts,
  printing the pattern, flags, subject and start, if the result, a capture span or the error
  kind differs (an interruption that only one of the two saw is no difference). With
  `TRUST_JS_REGEXP_TIER_UP_AT=1` it checks every execution of every pattern the tier takes.
  The lib tests' fuzzer (`regex::native::tests::fuzzed_programs_match_as_the_matcher_does`)
  runs `TRUST_JS_REGEXP_FUZZ_CASES` executions (40,000 by default) from the seed
  `TRUST_JS_REGEXP_FUZZ_SEED`, reducing any difference to a minimal pattern and subject.
- `TRUST_JS_REGEXP_CODEDUMP=<text>` prints the native code of each pattern whose source
  contains the text (`*`: every pattern) as it compiles: `[regexp-codedump]` lines with the
  pattern, its width, base address and size, then its words (AArch64; assemble them as
  `TRUST_JS_JIT_CODEDUMP`'s) or bytes (x86-64).
- `TRUST_JS_REGEXP_NO_PREFILTER=1` compiles patterns without their prefilter: the literal every
  match contains and the range of its offset from the match start. Exec scans for that literal
  and tries only the starts that put an occurrence in range, and a sticky regex, or one anchored
  at the input's start, enters the matcher only when the literal lies in range of its one start.
  With the switch set, exec tries every start the first-element filter lets through.
- `TRUST_JS_REGEXP_NO_SINGLE_ELEMENT=1` keeps patterns that match exactly one element (one
  character, `.` or one class, as in `/[&<>]/`) on the matcher. Otherwise exec finds the starts
  the first-element filter admits and tests the element at each, without entering the matcher.
- `TRUST_JS_REGEXP_NO_ATOMIC_LOOPS=1` keeps every greedy loop's retries. Otherwise a loop whose
  shorter runs could only fail keeps no record to retry one: a loop followed by the input's end
  (`\s+$` outside multiline mode), by an element its body cannot match (`[\w+\-.]+:`), by a word
  boundary after at least one word character (`\b\w+\b`), or by nothing left to test
  (`(\d*)(\D*)`). Patterns with an inline modifier keep every retry, as do `i` patterns under
  `TRUST_JS_REGEXP_NO_CASE_CLOSURE=1`.
- `TRUST_JS_REGEXP_NO_RUN_SKIP=1` resumes the scan after a failed attempt at the next start.
  Otherwise, for a pattern that begins with a greedy loop of unbounded count (as `/\s+$/` and
  `/[\w+\-.]+:/` do), it resumes where that loop's first run ended: no start inside the run can
  match.
- `TRUST_JS_REGEXP_NO_ASSERTION_FILTER=1` gives patterns whose paths may begin with an
  assertion other than `^` no first-element filter: exec then tries every start. Otherwise a path
  that begins with `$` may begin only at the input's end or, in multiline mode, before a line
  terminator (as in `/$/m` and trimming's `/^\s*|\s*$/g`), and one that begins with `\b`, `\B`
  or a lookaround only where the path after it may.
- `TRUST_JS_REGEXP_NO_CASE_CLOSURE=1` leaves the characters, classes and loops of a pattern's
  case-insensitive parts (under `i`, or in a `(?i:…)` group) to fold case as they match.
  Otherwise they are closed over case when the pattern is compiled: a character becomes the
  class of the characters with its Canonicalize, a class takes in those of its members, and
  they match with no folding, which also lets the first-element filter take an exact leading
  byte, the loop analysis prove loops of `i` patterns atomic and the native tier take them.
  Backreferences, word boundaries and `v`-mode class strings still compare case-insensitively
  as they match.
- `TRUST_JS_REGEXP_NO_SIMD=1` keeps the subject scans (for the prefilter's literal, and for a
  match's first element by byte or by table) on their portable loops instead of AArch64 vector
  code.
- `TRUST_JS_REGEXP_NO_PROTECTOR=1` keeps the RegExp protector from holding. The protector
  proves for the running Realm that %RegExp.prototype%'s flag getters, `exec`, symbol methods
  and `constructor`, and %RegExp%'s @@species getter, are the intrinsics, and lets a RegExp of
  the shape and prototype RegExpAlloc gave it skip their lookups: String.prototype's `match`,
  `matchAll`, `replace`, `replaceAll`, `search` and `split` then call the RegExp algorithms
  directly, RegExpExec calls RegExpBuiltinExec, and the direct replace, split, test, flags,
  global match, search, matchAll and dead-result paths need no other guard. A RegExp of that
  shape holding a plain Number in `lastIndex` reads and writes it at its slot. With the switch
  set, every one of them takes the observable steps.

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
