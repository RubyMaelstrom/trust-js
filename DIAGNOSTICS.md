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
* Two forms of switch behave differently. One written `=1` below and read through
  `env_flag!` is on when the variable is set to anything, `0` and the empty string
  included; a switch's own entry says where only particular values count (`TRUST_JS_PERF_MAP`,
  "any value but `0`"). One written `=0` is on by default, and only the exact value `0`
  turns it off. A cached switch is read at its first use, and a few entries below say they
  are read at each event, by each `Runtime`, or when the binary is compiled.
* Test-process markers are not inputs. A lib test that needs a process of its own (a switch read
  once per process, a process-wide epoch or code budget, an exit path) runs the test binary again
  with `--exact` and its own name and a marker variable set; the copy runs the test's body and
  the parent checks its exit status and output. The parent often sets the switches under test on
  the child too. Nobody sets a marker by hand: with one set, its test takes the child's branch in
  the first process, without those switches. The markers, all in `crates/trust-js/src` test code:
  `TRUST_JS_FASTALLOC_EXIT_CHILD`, `TRUST_JS_OPT_D8_CHILD`, `TRUST_JS_OPT_ONE_CPU_CHILD`,
  `TRUST_JS_PROPSTAT_CHILD`, `TRUST_JS_REMEMBERED_NATIVE_APPEND_CHILD`,
  `TRUST_JS_STRING_ACCUMULATION_OPSTAT_CHILD`, `TRUST_JS_STRING_ACCUMULATION_REPLAY_CHILD`,
  `TRUST_JS_STUB_FLAG_CHILD` and, with the prefix `TRUST_JS_TEST_`, `ADDRESS_REUSE_CHILD`,
  `CACHED_CALLBACK_ENTRY_ROOT_CHILD`, `CACHE_OWNER_DEATH_CHILD`, `CALL_EPOCH_MAX_CHILD`,
  `CALL_RECORD_CHILD`, `CODE_EPOCH_CHILD`, `CODE_PRESSURE_CHILD`, `CODE_RECLAIM_CHILD`,
  `CONSTRUCTOR_ARGUMENT_LOAN_CHILD`, `CONSTRUCT_CHILD`, `CREATE_TIER_CHILD`,
  `CREATION_DIRECTORY_CHILD`, `DIRECT_RECLAIM_CHILD` (its value names the child's mode, `stable`
  or `stale`), `ELEMENT_RECORD_INLINE_CHILD`, `FAST_CALLBACK_ENTRY_ROOT_CHILD`,
  `FOREIGN_RUNTIME_RETIREMENT_CHILD`, `GENV_PIN_CHILD`, `METADATA_ADMISSION_CHILD`,
  `NATIVE_LAYOUT_CARD_CHILD`, `OPT_DEFERRED_DISCARD_CHILD`, `OPT_LOOP_LIMIT_LOG_CHILD`,
  `PACKED_NEW_TARGET_CHILD`, `PREPARED_INLINE_CHILD`, `PRIMITIVE_MATH_INTRINSICS_CHILD`,
  `REGION_VOCABULARY_CHILD`, `ROOTING_STRESS_CHILD`, `SCOPED_NATIVE_SIGNATURE_CHILD` and
  `SPLICE_RULE_CHILD`. The Node compatibility layer marks the processes it starts the same way:
  `child_process.fork` sets `TRUST_JS_FORK_IPC=1` on its child, whose `process` then carries the
  IPC channel over stdin and stdout, and `cluster.fork` sets `TRUST_JS_CLUSTER_WORKER=1`, which
  makes `cluster.isWorker` true there.

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
  100 machine-code runs; with the optimizing tier on, 100 invocations, each K = 64 native runs,
  so that N keeps meaning invocations); zero disables inlining.
- `TRUST_JS_OPT=1` turns on the speculative optimizing tier (ARM64 only; off by default until a
  stage exit measures a net win; docs/v8-chase-p3-plan-2026-11-08.txt, "Stage 1"). At the second
  stage's checkpoints (about 100, 500, 2,100 and 8,500 invocations, then one every 500
  invocations), once a function's feedback fingerprint equals its previous checkpoint's, the
  second stage is built as without the tier, a function with nothing to splice getting its own
  chunk again, and the tier generates its native code in place of the template JIT: nothing
  forces a compile (V8's wait after an IC change), so a function whose feedback keeps changing
  keeps its template code. The fingerprint leaves out a generic named site's ways and a
  megamorphic call site's (their refills), not their transitions. A refused second stage keeps
  the template JIT's code, and a refused chunk with nothing spliced is dropped, the function
  keeping its first stage (a stale chain's recompile, `code3`, is compiled by the tier too); the
  tier refuses a chunk of more than 15,000 operations before its snapshot, and one with nothing
  to splice before its bytecode recompile (V8's `max_optimized_bytecode_size`). With the tier on
  the inline plan splices no call site whose ways hold more than one distinct callee unless the
  call is a named method call (`o.m(...)`, whose callee `GetMethod` reads, as the bytecode
  compiler marks the site), and none that met another closure of a way's function (its first
  code-keyed hit, which the site records once by writing its code hint): Maglev inlines no
  polymorphic call but continues a polymorphic named load's arms through the call of the loaded
  function, Warp inlines none, and the exemplars guard closures by their code; with the tier off
  the plans are as before. `TRUST_JS_TIER_LOG=1` names each site the rule declines, and each
  named method call's site of several callees it admits (`[tier] inline skip site|site <n> (pc
  <pc> of <name>, depth <d>, call <source name>): several callees, not a named method
  call|another closure of its callee's function|several callees at a named method call,
  admitted, the tier's rule; <n> distinct callee(s): <name>|<first stage>|<cell address>;
  ...`). A failed
  speculation exits to the bytecode VM, which finishes the call. After 100 << R exits of one
  code object (R, the function's discards so far, up to 13) the code is discarded: new entries
  take the template JIT's code (a compile deferred for executable capacity leaves the chunk to
  compile at its next entry), activations still running it finish in it, each exit site but
  throws and unsupported operations is flagged in the function's exit profile, R grows, and the
  second stage's warm-up begins: checkpoints (100, 400, 1,600, 6,400) << R invocations apart,
  then one every 500 << R, at which, once the feedback fingerprint is unchanged since the last,
  the tier compiles the chunk again, declining what the profile flagged (the generic operation
  for NotNumber, f64 for NotInt32 and Overflow, the generic access for WrongShape unless the
  site's ways grew, the guard's mismatch edge for WrongCallee, the block for NeverRan, an f64
  loop phi for PhiNotInt32 and a tagged one for PhiNotNumber, the two exits of a loop's entry
  check of a word entering an untagged phi, at its header, and the checks in the loop for
  HoistingFailed, the exit of a loop invariant's check or of a typed array's checks moved to
  the loop's entry, at its header; the stores' book for ObservedBuffer, the exit of a typed
  store's TypedData whose buffer an embedder observed through `array_buffer_version` or
  `take_array_buffer_dirty_ranges`, at the store's pc or, for a loop's entry check, its
  header); at a bitwise operation whose record
  holds undefined, null or Booleans, a NotNumber flag declines the operation only once its
  truncation of them exited (a Number check's exits there were oddballs the VM recorded, which
  the truncation takes). An exit whose own reason the exiting code's compile had already flagged
  at its site with the same data (a shape check's ways no more than the flag's, a free name's
  proof, a truncation's mark) shows the form that flag left exits for its reason too: the
  discard flags the site Generic as well, and the recompile builds the operation's generic form there before any
  other decline (`Generic: the generic operation`; not for throws, unsupported operations,
  forced exits, NeverRan, WrongCallee or the exits kept at a loop's header). A code whose
  exited frames loop is
  discarded sooner, as JSC's from its baseline loop: once its exits reach 5 << R (never more
  than its limit), each exit that does not discard starts a warm-up of its chunk's template loop
  iterations (100 << R invocations' worth, a discard's warm-up's first distance) that entries do
  not count, and when a loop's count reaches it (an exited frame's loop in the chunk's
  borrowed-frame code, which every exited frame moves into at its first back edge) the loops
  count nothing more and the next native entry discards the code as at its limit. A discard
  whose exits were all throws or unsupported operations arms no recompile. Each discard and
  each recompile moves the call caches' epoch. With the tier on, the checkpoints count V8's
  weight: a native entry counts K = 64 native runs (an invocation), and a loop iteration of
  template code (a first stage's, a discarded code's fallback) its loop's share of one,
  max(1, round(K x B / L)), B the loop's span in operations (its header to its last back edge)
  and L its chunk's (V8's back edge charges its loop's length against invocations times the
  bytecode's length): each iteration counts toward its chunk's pending checkpoint or warm-up,
  up to one below it, after which its loops count nothing, and the next native entry takes the
  checkpoint; a checkpoint that would pass the 32-bit count is not armed.
  `TRUST_JS_OPT_NO_LOOP_COUNT=1` counts native entries alone, K each; with the tier off an entry
  counts one, and no iteration counts. Under
  `TRUST_JS_OPT_EAGER=1` the tier compiles at the first checkpoint, a recompile's too, without
  waiting for the feedback to settle (the eager gate: V8's `--stress-maglev`).
  `TRUST_JS_OPT_EXIT_EVERY=N` compiles every check and effect of the tier's code with a
  countdown that exits at every Nth one (the forced-exit gate: V8's `--deopt-every-n-times`);
  forced exits neither count toward the limit nor flag anything.
  `TRUST_JS_OPT_EXIT_LIMIT=N` (a positive N) sets the limit's base in place of 100: 1 discards
  each code at its first exit, the configuration that cycles reoptimization.
  `TRUST_JS_OPT_LOOP_EXIT_LIMIT=N` (a positive N) sets the base of the limit for exited frames
  that loop in place of 5 (JSC's `osrExitCountForReoptimizationFromLoop`); it never passes the
  exit limit.
  `TRUST_JS_OPT_POISON=1` compiles a store of a poison Number (`0x8badf00ddeadbeef`) before
  each generic node of the tier's code (P3 stage 3 step 3a: an operation the tier does not lower,
  run on the canonical operand words with explicit operands) into every frame word its state does
  not hold current: the slots it does not name whose values live in registers or spill slots or
  that are dead, and the operand words below its operands; a template or helper that read one
  would compute with the poison (a diagnostic of what they read of the frame; results differing
  from the tier off's mean a hidden reader). Before each canonical-frame effect without a handler
  it stores the poison into the operand words below the effect's operands as well. Under the
  switch, the tier's code checks after each generic node's template that the words it poisoned
  still hold the poison (one that does not was written by a template or helper: a slot write the
  operation's slot use does not list), and the helper that runs the VM on a copy of the operand
  stack checks the words it copies; each fault is written once to standard error as an
  `[opt-poison]` line naming the function, the pc, the operation and the word. An inlined frame's
  generic node (P3 stage 3 step 3c) runs its template on a slot window under the switch, every
  word of it but its named slots' holding the poison, the operand words below its operands too,
  and, where its operation reads no `this`, `this` the poison while it runs; the check after its
  template covers the window and those operand words.
  `TRUST_JS_OPT_VERIFY=1` verifies each graph the tier builds in a release build (its SSA
  dominance, representations, frame states and terminators) and refuses a graph that fails,
  logging why under `TRUST_JS_OPT_LOG`, which with it also logs `[opt] verified <name>: <k>
  headers kept heap facts, <l> loops checked` for a graph whose loop headers kept their entry's
  heap facts (the loops the union rule was checked on); debug builds and the library's tests always verify, and
  panic at a failure (Maglev, SpiderMonkey and JSC verify their graphs in debug builds only).
  The tier compiles off the main thread: at a checkpoint the main thread takes the snapshot and
  the chunk's frozen view, and the process's compile thread (`trust-js-opt`, started at the first
  compile) builds, allocates registers, emits and maps the code, making any process-wide shared
  stub the code calls that is still missing (under the stub table's lock, as the main thread
  makes one); the next poll of generated code
  after it finished installs it (a function's second stage is published then: until then the
  function runs the code it has). The install drops a result whose realm no longer runs, whose
  exit profile a discard changed, whose function was collected or whose chunk was superseded or
  published over meanwhile, and the function tiers up again. A function whose inline plan
  splices nothing gets no second chunk: the tier compiles its own chunk and installs the code in
  the chunk's slot over the template JIT's code, which stays as the fallback a discard returns
  to without a compile (one is compiled where the code cache's CLOCK evicted it); the chunk's
  entry word, which every call way copies and, with the tier on, every entry compares, sends the
  ways filled before the install to the new code without moving the call caches' epoch. `TRUST_JS_OPT_SYNC=1` compiles on
  the main thread, as do a process with one CPU available (its affinity mask counts) and one
  that cannot start the thread; the library's tests whose switches read a compile's results at
  once (`TestSwitches::asynchronous` false) do too.
  `TRUST_JS_OPT_NO_REOPT=1` keeps stage 1's fallback for good: a discard flags nothing, adds
  nothing to R and arms no recompile. `TRUST_JS_OPT_NO_EXIT_PROFILE=1` flags nothing, so that
  recompiles decline nothing (the feedback alone guides them). `TRUST_JS_OPT_LOG=1` prints, at
  the shell's end of run, the census of the realms' global objects' reported writes (`[opt] global
  stores: checked <n>, constants changed <n>, same value <n>, fills refused <n>, reference hints
  refused <n>`: value writes that reached Rust, those that made a Constant property NotConstant
  and of them the same-value stores, and the writing caches' fills and Reference hints refused
  for a property not yet NotConstant), and one
  stderr line per compile (with its time and code range, `at <base> <length>` in hex), refusal
  (with its reason and pc; `over the size limit` for a chunk past 15,000 operations, with `(<n>
  ops, before its recompile)` where it had nothing to splice), install of the tier's code in a
  chunk's own slot (`[opt] installed <name> in place at <base> over <template base> <template
  length>`, or `, no template code kept` where the slot held none), decline
  (`[opt] declined <name> pc <pc> <reason>: <form built>`), wait for settled feedback (with the
  function and checkpoint, a warm-up's marked `(reoptimization)`), exit (with its pc, reason
  and kind, and for a shape check's the shapes it expected and the receiver's, or what the
  receiver is when it is no object), discard (`[opt] discarded <name> after <n> exits (limit
  <l>, R <r> -> <r'>) for <what new entries run>: pc <pc> <reason> x<count>, ...`; `the
  template JIT's kept code` where it returned to the code an install kept; ending `; at its
  loop limit: <n> exits >= <loop limit>` for a discard at the loop limit), new exit
  site (`[opt] flag <name> pc <pc> <reason> own|at <baseline name> pc <pc>[ (spliced)]`), discard
  of unfixable exits only (`[opt] not reoptimized <name>: unfixable exits only`) or of a warm-up
  past the count (`[opt] not reoptimized <name>: its warm-up would pass the checkpoint word`),
  recompile (`[opt] reoptimizing <name> at checkpoint <k> (R <r>)`) and epoch move (`[opt]
  epoch moved (discard|reoptimization), <n> by the tier`), and one per template JIT compile of
  a second stage, or of a first stage after the tier's code was installed in it: `[opt]
  template <base> <length> <name> (second stage|fallback)`, the code of a second stage the tier
  refused (or of a loop's borrowed-frame entry, or of a slot CLOCK evicted) or of a discard
  whose kept code was gone. A
  profile's samples join to these ranges, since the perf map names a second stage's template
  code `JS:^`, as its first stage's. A compile's line and a refusal's (whose `(<ms> ms)` is the
  compile's) end with where its time was spent: `on the main thread` or `on the compile
  thread` (`on a scoped thread` in the tests' TRUST_JS_OPT_SCOPED_COMPILE check), `queued <ms>
  ms` (its wait in the compile thread's queue), `installed <ms> ms after its snapshot` and, for
  a compile whose snapshot of the shared stubs lacked some, `, met <n> missing stubs, made <m>`
  (it looked them up again in the stub table and made those still missing, in its one run); a
  dropped result prints `[opt] dropped <name>: <reason> (compiled in <ms> ms on the compile
  thread, ...)`. The parts of a tier compile that run on the main thread besides the compile
  line's own time each print `[opt] main-thread <part> <name> in <ms> ms`: `recompile`, the
  bytecode recompile that makes the tier's chunk (its second stage, or a stale chain's
  recompile; none for a function with nothing to splice, whose own chunk the tier compiles);
  `snapshot`, the snapshot, the chunk's frozen view and the job's hand-off to the
  compile thread; `install`, the code's link and
  installation after its compile; `fallback`, the template JIT's compile of a discarded code's
  fallback, at the discard or, deferred there for executable capacity, at a later entry. Each
  part's time leaves out the log's own printing. The main thread's share of the tier's compiles
  is then the main-thread lines and the compile and refusal lines `on the main thread`.
  `TRUST_JS_OPT_FOLD_GLOBALS=0` turns off the tier's fold of global properties (F4 part 2;
  measurement only): by default a compile reads a global-object property whose value is a
  Number and which the realm's global fuse holds Constant as a constant of its code, under a
  dependency the property's change, removal or shadowing fires (the code exits `Invalidated`
  at its next test, once, and is discarded; three such discards and the function folds no
  global). `TRUST_JS_OPT_LOG` prints `[opt] folds <n> globals of <name>` per compile and, at
  exit, the invalidations among the global stores' census.
  At `graph`, a call the tier does not inline whose site's record names one user function that
  the template JIT's call would enter directly is the known callee's direct call (P3 stage 3
  step 7): its identity guarded as an inlined callee's is (a WrongCallee exit, after which the
  site's recompile calls generically), the callee's current code read at every call from its
  chunk's code-slot entry word, which every install, discard, reoptimization, supersession and
  eviction writes (zero: the call's template runs). `TRUST_JS_OPT_LOG` prints `[opt] calls
  directly <n> known callee(s) of <name>` per compile that has such sites in its own frame.
  `TRUST_JS_OPT_INLINE` chooses how the tier inlines (P3 stage 3). `graph`, the default since
  2026-11-19 (any value but `splice` too), inlines in the tier's graph; `splice` keeps the
  splices above for every plan. At `graph`: a function whose inline plan's ways all call
  nothing (no call or `new` in a callee) is compiled on its first stage, as one with nothing to
  splice, each such callee built into its graph from the callee's own first stage behind the
  guard a splice has, an inlined frame whose exits rebuild the callee's frame and its caller's;
  an operation of a callee the tier does not lower is a generic node of the inlined frame (P3
  stage 3 step 3c: its template run with the callee's chunk, inline position, `this`,
  environment and named slots written where its helpers read them, and the caller's restored
  after it), but one that needs the whole canonical frame (it may read any slot, or its helper
  runs the VM on the frame's slots) or a call, which declines its callee (E0: its call stays),
  as does a callee whose slot window would put its caller past a pooled activation record (`an
  inlined frame past its record`); every other plan splices as before. A reoptimization of such a function inlines the ways
  its first compile took (while their callees live), as a second stage's reoptimization keeps its
  splices; a function the tier refuses splices at its next checkpoint (`[opt] graph inlining refused <name>: its
  next checkpoint splices`), as a refused second stage keeps its splices. Its
  log adds `[opt] oracle <name>: <n> sites, <w> ways, <c> callee chunks in <ms> ms` (the inline
  decisions and the callees' snapshots, inside the `snapshot` part), `[opt] inlined <name> pc
  <pc> way <k>: <callee>`, `[opt] not inlined <name> pc <pc> way <k> <callee>: <reason>[ at pc
  <m> (<op>)]` and, for a plan left to the splice, `[opt] spliced <name>: a way's callee calls`;
  a decline in an inlined frame ends with ` (inlined frame <k>)`. An exit through an inlined
  frame whose callee's first stage has tried to enter a loop (its borrowed-frame code) discards
  the code at its loop limit at once (`; at its loop limit through an inlined frame that loops`),
  as JSC's exits through inlined code that loops do.
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
  default (ARM64, shared stubs on, at most eight arguments) the form calls a shared stub that
  enters a native without the helper's decode of the call: one of a light class on light operands
  (`bytecode::native_operands`) on the caller's window (`bytecode::jit_native_light`), Reflect.get,
  has and set answered on their operands (`bytecode::jit_native_reflect`), and any other engine
  native through the lean native-call boundary (`bytecode::jit_native_boundary`), which polls no
  interrupt and makes no safepoint around the native (generated code's polls do).
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
  incremental marking on, a major one may start a marking cycle (in release builds once the
  heap holds more than 8 MiB), and the next one is that cycle's final pause, as V8's,
  SpiderMonkey's and JSC's stress collections are whole ones (`TRUST_JS_GC_INCREMENTAL=force`
  keeps the cycle's slices). With
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
- `TRUST_JS_PERF_JITDUMP=<directory>` writes perf's jitdump, `<directory>/jit-<pid>.dump`
  (Linux `tools/perf/Documentation/jitdump-specification.txt`), as V8's `--perf-prof` does: one
  load record per piece of code the perf map lists, with its bytes, under the perf map's name,
  and with the optimizing tier on, a second stage's template code marked `(second stage)` or
  `(fallback)` (see `TRUST_JS_OPT_LOG`); a first stage's template code compiled before the
  tier's code was installed in its slot keeps its name (the install's log line gives its
  range). Each record is stamped with `CLOCK_MONOTONIC`, so a
  profile recorded with `perf record -k mono` attributes a sample to the code that ran at its
  address then, where the perf map gives an address that freed code's memory left to later code
  to its last entry. `perf inject --jit -i perf.data -o perf.jit.data` writes one ELF file per
  record beside the dump; report from `perf.jit.data`. GNU/Linux only; the file lives until it
  is deleted, and the map's switch may be on or off with it.

- `TRUST_JS_AST_HOT=1` reports functions whose calls keep running on the tree-walker: each call
  that `Interp::call_user_inner` does not hand to a compiled chunk counts, and at a function's
  1,000th, 10,000th, 100,000th and 1,000,000th it prints `[ast-hot] <name> ×<n>: <the first 60
  characters of its source>` to standard error. It counts in the per-function call counter that
  the tier-up budget reads, so it can bring a function's tier-up forward: diagnostic only.
- `TRUST_JS_INLINE_BUDGET=N` and `TRUST_JS_INLINE_SMALL_BUDGET=N` set the inline planner's two
  budgets of callee operations for one optimized body (`bytecode::plan_inlines`): the shared
  budget, N less the caller chunk's own operations (default 320), which every splice draws on
  first, and the small-callee budget (default 384), which admits a callee of at most 16 operations
  once the shared budget cannot, if the callee calls nothing or its site repeats in a loop. Both
  at 0 leave every plan empty. `TRUST_JS_INLINE_MAX_OPS=N` sets the largest callee chunk a site
  may splice, in operations (default 96); `TRUST_JS_TIER_LOG=1` names a larger callee's site as
  `callee size/shape`, the reason it prints for the other limits on a callee's shape too. A value
  that does not parse keeps the default.
- `TRUST_JS_JIT_NO_CACHE_SEED=1` compiles a second stage (`bytecode::compile_with_inlines`) with
  empty property, name and call caches instead of seeding them from the chunk that ran. Its
  caches then hold no evidence of what ran, so it also keeps every register region, as under
  `TRUST_JS_JIT_UNRUN_REGIONS=1`.

- `TRUST_JS_JIT_FAST=<mask>` sets the ARM64 templates' feature mask, a decimal `u32` (every bit
  by default; a value that does not parse keeps the default). A cleared bit leaves its templates
  out and the operations take their generic forms. Bits: 1 arithmetic and bitwise operators; 2
  comparisons, equality and `typeof` tests; 4 `JumpIfFalse`; 8, 16 and 32 local loads, stores and
  updates; 64 `Pop`, `Dup` and `ToPropKey`; 128 constants; 256 named reads; 512 `GetMethod`; 1024
  element reads and `in`; 2048 and 4096 element stores (`SetElemDrop`, `SetElem`); 8192 free-name
  and captured-binding accesses; 16384 numeric register chains; 32768 register regions (loop and
  acyclic) and `this` loads; 65536 named stores and updates; 131072 `!`; 262144 a loop region's
  access to its receivers' element words; 524288 the in-line call-cache probe; 1048576 direct
  calls (which need 524288); 2097152 the linked-scan loop region. The optimizing tier's effects
  read the same mask, and `TRUST_JS_FEEDBACK_PROFILE=1` clears fourteen of the bits itself
  (`TRUST_JS_JIT_FAST=4291167192` is those fourteen off with nothing observed).
- `TRUST_JS_JIT_NO_ARRAY_INTRINSICS=1` leaves the in-line forms of `Array.prototype.push` and
  `pop` out of ARM64 call sites, and `TRUST_JS_JIT_NO_FUNCTION_CALL_INTRINSIC=1` that of
  `Function.prototype.call` (sites of one to eight arguments): those calls then take the call-hit
  form, as other natives do. The template JIT and the optimizing tier's code both read them.
- `TRUST_JS_JIT_NO_APPLY_FORM=1` turns off the apply form: a body whose only uses of `arguments`
  besides `arguments.length` and `arguments[k]` are `<f>.apply(<t>, arguments)` then makes its
  arguments object at entry and calls `apply` with it, where the form calls `f` with the actual
  arguments and makes none (`Op::CallApplyFrameArgs`). `TRUST_JS_JIT_NO_FORWARDER=1` turns off
  the recognized forwarding constructor `this.<init>.apply(this, arguments)` of part A2
  (`Chunk::jit_arguments_apply_forwarder` recognizes none): its Rust forwarder and the direct
  construct sequence's forward form, which only an entry the forwarder filled reaches. Both are
  read once, at every tier; with both set, such a body runs the general path alone.
- `TRUST_JS_JIT_NO_PROP_CHAIN=1` builds no ARM64 numeric register chain with a property read or
  store, as a chunk recompiled after its chains bailed has none (see
  `TRUST_JS_JIT_NO_CHAIN_REOPTIMIZE`); `TRUST_JS_JIT_NO_PROP_STORE_CHAIN=1` keeps the reads and
  leaves out the stores. `TRUST_JS_JIT_PROP_CHAIN_LOG=1` prints each chain that holds a property
  read as it is planned: `[jit-prop-chain] <pc>:<operation> | ...`.
- `TRUST_JS_JIT_NO_CFG_REGION=1` keeps ARM64 loops of the linked-scan shape, `while ((peek =
  next.link) != null) next = peek`, off their own region (`plan_linked_scan`); such a loop then
  compiles as any other. Despite the name, the general CFG regions stay on.
- `TRUST_JS_JIT_LOOPLOG=1` prints the ARM64 helper-free loop plan's decisions (`plan_loop`) as
  `[jit-loop] head <pc>:` lines: `reject: <why>` (an unchainable operation with its pc, operand
  depth, receivers, register pressure), `demote I slot <n>` for a local taken out of the integer
  slots, and for a planned loop `CHAINED <n> ops, <n> slots (<n> I), <n> receivers (<n> vec
  pins), <n> names, memo elem <r>r/<u>u conv <r>r/<u>u`. `TRUST_JS_JIT_REGIONLOG=1` prints the
  register regions' as `[jit-region] head <pc>:` lines: a loop head's region IR (`<n> blocks, <n>
  ops, <n> values, <n> params, <n> exits`, or `reject <error>`; a head that is no loop prints
  nothing), `EMITTED linked scan`, `EMITTED general CFG (<n> ops, <n> bytes, <n> saved-work
  units)`, `EMITTED acyclic CFG (<n> ops, <n> bytes, <n> ops in all)`, or `retained baseline
  (...)` with the reason: its optional bytes and saved work (not profitable), `<n> ops past the
  region budget`, or `its operations have not run`. Under it every loop head builds a region IR
  for the log alone.
- `TRUST_JS_JIT_LOOP_ALIGNMENT=N` aligns the body of an ARM64 helper-free loop region to N bytes:
  16, 32, 64 (the default) or 128, or 0 for no padding; any other value keeps 64. It is for A/B
  comparisons of code placement in release builds.
- `TRUST_JS_SHARED_NATIVE_OPERATIONS=0` makes ARM64 register regions emit their effects from the
  regions' own, older copies of the operation templates (`regions::emit_effect`) instead of the
  templates baseline code uses (`jit_operations.rs`).
- `TRUST_JS_JIT_INLINE_ALLOC=0` makes generated code take no cell in line: environment records,
  objects (a direct construct's `this` among them) and element blocks then come from the
  allocation helpers.
- `TRUST_JS_JIT_NO_PACKED_HOLES=1` (ARM64 builds) makes `Array(n)` with a Number n of 1 to 8
  reserve no element words. By default the constructor reserves n hole words in the new array's
  element block, so that the indexed writes that usually follow fill them without growing it.
- `TRUST_JS_JIT_CODE_BUDGET_MB=N` sets the process-wide cap on live executable code, in MiB (1 to
  1024; default 128, and any other value keeps it), which the JIT and the native RegExp tier
  share. It counts the bytes requested for live mappings, not code ever emitted. A request that
  does not fit first reclaims inactive code, and failing that the function stays on the bytecode
  VM. The process's code region reserves 1 GiB of addresses, or twice a budget above 512 MiB.
- `TRUST_JS_ITERATOR_RESULTS=0` makes every step of a built-in Array, String, Map or Set iterator
  call its `next` and allocate the iterator result object. By default an iterator whose `next` is
  the intrinsic steps without one (`Interp::try_intrinsic_iterator_step`), and ARM64 code steps
  an Array iterator in line (see `TRUST_JS_JIT_NO_ITER_STEP`).

- `TRUST_JS_SCOPE_PLANS=0` turns off the per-function environment plans (`bytecode_scope_plan.rs`):
  every function's plan is opaque, a Script's inner functions get no outer contexts, and no free
  name gets a static (hops, slot) address, so each tier makes its records its own way and names
  keep their caches. `TRUST_JS_SCOPE_LOG=1` prints `[scope-plan] <name>@<source start>
  Public|Private|Opaque <why> prefix=<n>` as each function's plan is made, and from the bytecode
  compiler `[scope] <its first three slot names> <name>: static hops=<h> slot=<s>|global
  hops=<h>|dynamic: <where the resolution stopped>` for each free-name read and `[scope] store
  <name>: static hops=<h> slot=<s>|dynamic` for each free-name store.
- `TRUST_JS_ACTIVATION_PLANS=0` builds a compiled function's activation record without its
  chunk's activation plan, binding by binding (`Chunk::make_unplanned_activation`), and no chunk
  then has a fixed activation (`Chunk::jit_fixed_activation`): generated code reaches captured
  bindings through its checked caches instead of fixed offsets.
- `TRUST_JS_COLD_ENV_LAYOUTS=0` keeps the tree-walker's function records off shared binding
  layouts. By default, once a call has instantiated its parameters, hoisted declarations and
  lexicals, its variable record (and a separate parameter record, where it has one) takes the
  layout that its function's calls share; a record under `with` never does.
- `TRUST_JS_LAYOUT_NAME_IC=0` keeps every name cache and captured-binding cache keyed by one
  environment record. By default a site that meets a second record of the same binding layout
  changes to the layout's slot index, which then serves every record of that layout.
- `TRUST_JS_ELIDE_SELF_ENV=0` gives every named function expression's closure the environment
  record that binds its own name. By default one whose body cannot observe the binding
  (`bytecode::named_self_binding_may_be_observed`) gets none.

- `TRUST_JS_FN_PROFILE=1` times the calls that pass `Interp::call_user_inner`, the tree-walker's
  function entry, and keeps per function and Realm its calls, self and inclusive time and
  callers. Generated code's direct call sequences do not pass it: a callee reached only that way
  has no entry and its time is its caller's self time. A dump prints the 32 entries with the
  most self time to standard error and clears the profile: `[fn-profile] self=<s>s
  inclusive=<s>s calls=<n> [realm=<address> function=<address>] <name> top_caller_calls=<n>
  top_caller=<its description>`, then `[fn-profile-state] description=<...>
  tier=cold|bytecode|bytecode-bail|jit|jit-bail|jit-budget-wait strict= arrow= with= env_depth=
  same_realm= prepared= activation= code_used= code_limit= code_denials=`, the function's state
  at its first call of the batch. The profile dumps when its Agent's interpreter is dropped: at
  the shell's normal exit, but not after an uncaught exception, which exits without dropping it.
  `TRUST_JS_FN_PROFILE_FLUSH=<name>` also dumps each time an outermost profiled call of a
  function of that name returns, and `TRUST_JS_FN_PROFILE_FLUSH_SECONDS=<seconds>` each time an
  outermost profiled call took that long or longer (a finite number, 0 or more).
  `TRUST_JS_FN_PROFILE_SOURCE=1` adds the first 100 characters of each function's source to its
  description and a `[fn-profile-source]` line with up to 16,384 characters of it; without it the
  output holds no source text.
- `TRUST_JS_PERF_METRICS_INTERVAL_SECONDS=<seconds>` sets the interval of the embedding API's
  periodic `TRUST_JS_PERF_METRICS` record (`PerformanceMetricsSampler::from_env`, whose
  `maybe_dump` collects and returns one JSON envelope per interval): a finite positive number,
  10 by default, anything else ignored. Only a host that calls `from_env` reads it; the shell and
  `trust-js-cli` do not.

- `TRUST_JS_SCRATCH=<file>` makes the lib test `tests::scratch_eval_file` evaluate the file in a
  fresh engine and print its console lines and its value or throw (`cargo test scratch_eval_file
  -- --nocapture`); without it the test does nothing. `TRUST_JS_SCRATCH_PRE=<file>` evaluates a
  preamble script first, `TRUST_JS_SCRATCH_STRICT=1` evaluates the file as strict code and
  `TRUST_JS_SCRATCH_MODULE=1` as a module whose relative imports resolve against its directory.
  `TRUST_JS_SIZES=1` makes `tests::debug_type_sizes` print the sizes of `Stmt`, `Expr`, `Token`
  and `Tok`.

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

- `TRUST_JS_REGEXP_FUZZ_TRACE=1` makes the lib tests' fuzzer print each case before it checks it,
  `[fuzz] /<pattern>/<flags> <subject>`, so that a crash or a hang names its case. On x86-64,
  `TRUST_JS_REGEXP_X64_ENCODINGS=1` makes the assembler's test
  `regex::native::x64::tests::encodings_disassemble_as_intended` print each encoding it emits
  with the instruction it should be (`<hex bytes>` and a tab, then the text) instead of comparing
  it, for checking against a disassembler: the test then asserts nothing.

## Architecture diagnostics

The `architecture-diagnostics` feature records allocation sites/lifetimes
in the live object heap and call-cache epoch/refill traffic. It adds no object owners;
site and tracked-live-allocation limits report untracked allocations. Initial exotic family
is recorded at allocation, before later function/exotic initialization. Lifetimes
use heap allocation ordinals and completed collections, not elapsed time. Live
families at collection are a separate population sample. These probes compile out
of ordinary builds; use them only for diagnostics, never performance acceptance.

## Error and reflection tracing

Each of these prints bounded metadata to standard error. They read native state only: no getter,
Proxy trap or conversion runs for them.

- `TRUST_JS_TRACE_ERRORS=<text>` prints each error object `Interp::make_error` makes, caught or
  not, whose message or stack contains the text (`1`: every one): `trust-js-error: <kind>: Error:
  <message><stack>`, the message's first 512 characters and the stack's first 2,048, at most
  1,024 errors a process. The text is matched against the message and the stack, not the kind.
  After each error come the histories the next two switches record, when they are on.
  `TRUST_JS_TRACE_ERROR_SOURCE=<n>` adds `trust-js-error-source:` lines with the first n
  characters of the source of the four innermost running functions (800 for a value that is no
  number above 1, such as `1`; at most 16,384), and, when the filter is `1` or contains `not a
  function`, a `trust-js-call-error: callee=<preview> this-type=<type> argument-types=[...]` line
  for each of the first 32 calls of a value that is no object.
- `TRUST_JS_TRACE_UNDEFINED_PROPERTIES=1` keeps the last 32 slow-path property reads
  (`Interp::get_member_key`) that gave undefined, which a traced error prints as
  `trust-js-error-undefined-property: function=<running function's name> base=<type> key=<key>`,
  with `kind=`, `constructor=` and the first eight own keys (`own=`) for an object read by a key
  that is no index. A read that gave undefined is not proof of absence: an accessor or a Proxy
  may return it.
- `TRUST_JS_TRACE_BOUND_CALLS=1` keeps the last 256 events of bound function calls
  (`Interp::call_dispatch`), which a traced error prints as `trust-js-error-bound-call:` lines:
  `enter id=<n> parent=<id> target=<preview> saved=[bound arguments] args=[arguments]` and `exit
  id=<n> return|throw <preview>`. With it, `TRUST_JS_TRACE_BOUND_ARRAYS=1` adds, at each such
  call's entry and exit, the elements that changed in the arrays of 16 to 512 elements that the
  first bound argument holds in its first 16 properties (up to four arrays): `array id=<call>
  phase=enter|exit object=<address> field=<key> changes=[(index, preview), ...]`.
- `TRUST_JS_NULLISH_TRACE=1` prints `TRust JS: nullish computed-property read site=<site>
  key=<type>:<value>` when a computed read's base is undefined or null, before the TypeError;
  the site is `eval-index`, `bytecode-update-elem`, `bytecode-profiled-get-elem`,
  `get-method-elem` or `jit-update-elem`. It is read at each such failure.
- `TRUST_JS_PARSE_TRACE=1` prints each error `Parser::err` makes, with its token: `TRust JS
  parser: line <n> chars <start>..<end> token <kind>: <message>`. Errors of the checks that run
  after a construct is consumed (`err_semantic`) are not printed. It is read at each error.
- `TRUST_JS_TRACE_CODE_POINT_ERRORS=1` prints, for the first 32 arguments `String.fromCodePoint`
  refuses, `trust-js-code-point: argument_index=<i> converted=<number> argument_count=<n>`.
- `TRUST_JS_TRACE_FUNCTION_TOSTRING=1` prints, for the first 512 calls of
  `Function.prototype.toString` on a callable, `trust-js-function-source: name=<name>
  source_bytes=<n> host_ops=<bool>` (whether the source contains `__dom_`, `__host_` or
  `__trust`): which functions a page inspects, without their source.
- `TRUST_JS_TRACE_LEGACY_REFLECTION=1` reports whether a program reads `f.arguments` or `f.caller`
  of a sloppy function: `[legacy-reflection] arguments: inactive`, `arguments: active,
  materialized`, `arguments: active, lazily available`, `arguments: active, unavailable`,
  `caller: inactive` or `caller: active`, each at most once a thread.
- `TRUST_JS_DBG=1` prints three ShadowRealm debugging lines: `DBG cross enter realm=<address>
  parent=<...> self=<address> tkind=<target kind>` at each call of a host function wrapped for a
  ShadowRealm, `DBG cross err: <value>` when that call throws and `DBG cws err: <value>` when a
  function wrapped from a ShadowRealm throws, each before the TypeError that replaces the throw.
  It is read at every such call, not once.

## Runtime thread pools

- Each `Runtime` reads these as it is constructed, a worker's too; a value that does not parse
  keeps the default and one out of range is clamped. `TRUST_JS_BLOCKING_THREADS=N` sets the
  worker threads of the blocking pool, which runs file-system and other finite blocking work
  (default 4; 1 to 64), `TRUST_JS_BLOCKING_QUEUE_CAPACITY=N` the tasks its queue holds before a
  submission fails with `blocking executor queue is full` (default 64 a thread; 1 to 65,536) and
  `TRUST_JS_BLOCKING_STACK_BYTES=N` each worker's stack (default 1 MiB; 64 KiB to 64 MiB).
  `TRUST_JS_DEDICATED_THREAD_LIMIT=N` sets how many dedicated threads may run at once, one for
  each operation that may wait without bound, such as a child's stdio (default 256; 1 to 4,096),
  past which an operation fails with `dedicated executor thread limit reached`, and
  `TRUST_JS_DEDICATED_STACK_BYTES=N` their stacks (default 512 KiB; 64 KiB to 64 MiB).

## Build and installer inputs

- `TRUST_JS_VERSION_SUFFIX=<text>` is read when the binaries are compiled (`option_env!`), not
  when they run: the text follows the package version in `trust-js --version` and in
  `trust-js-cli --version` and its REPL banner.
- `scripts/install.sh` takes `TRUST_JS_INSTALL` (the install prefix, `$HOME/.trust-js` by
  default; the binary goes to its `bin`) and `TRUST_JS_RELEASE` (the release tag to download,
  `nightly` by default).

## Measurement and profiling recipes

The method (which builds to compare, how to count and time) is in
`benchmarks/README.md`, "Measurement method". These are the commands and traps that go
with it; `TRUST_JS_PERF_MAP` and `TRUST_JS_PERF_JITDUMP` above name generated code for
`perf`.

* Call graphs without privileges: `perf record -g` also asks for kernel call chains and
  fails with `EACCES` at `kernel.perf_event_paranoid=2`. Add `--user-callchains`. Isolate
  any other `perf` failure with `-vv` before concluding that access is blocked.
* Pin `perf` with its workload (`taskset -c <cpus> perf stat ... <command>`). On a
  machine with two core types each type has its own PMU, and an unpinned run reports
  two partial counts whose sum varies from run to run.
* A second `perf record` beside a long one fails on the per-user ring-buffer budget; add
  `-m 16`. `perf` resolves symbols from the binary now at the recorded path, so profile a
  copy if the build directory will be rebuilt.
* Pick a run's map by the engine's process id (`/tmp/perf-<pid>.map`), never the newest
  file: concurrent runs take each other's maps.
* To find a broad regression in a fat-LTO build on ARM64, sample with the Statistical
  Profiling Extension, not the PMU, whose skid moves about 1% between generated code and
  Rust: `perf record -e arm_spe_0/load_filter=0,store_filter=0,branch_filter=0,jitter=1/u
  -c 4096`, then `perf report --itrace=i1i`. `nm --size-sort -S` on both binaries shows
  functions whose inlining flipped.
* Struct layouts that generated code depends on: build the release engine with
  `CARGO_PROFILE_RELEASE_DEBUG=2 CARGO_PROFILE_RELEASE_STRIP=none` (debug information
  does not change layout) and read the type with `llvm-dwarfdump --name=<Type>
  --show-parents`, picking the definition by its namespace. The library's test binary
  lays `Interp` out differently, and `pahole -C Object` finds an AST variant of the same
  name.
* Rooting defects that the unit tests' GC stress misses: run Test262 with a dev-profile
  `test262-runner` under `TRUST_JS_GC_STRESS=1 TRUST_JS_TIER=jit
  TRUST_JS_TIER_THRESHOLD=0`, from a scratch directory (the runner writes
  `test262-report/` into the current directory). For a suspected site, write a
  reproducer that drops the program's own references, churns the heap, then uses the
  held value; a control with collections suppressed (`TRUST_JS_GC_NURSERY_BYTES=0
  TRUST_JS_GC_MIN_HEAP_BYTES=4000000000`) tells a rooting defect from another fault.
* The tier's graph verifier in release builds: `TRUST_JS_OPT=1 TRUST_JS_OPT_VERIFY=1
  TRUST_JS_OPT_LOG=1`, each workload at five fixed runs, once with
  `TRUST_JS_OPT_INLINE=splice` and once with `graph`; search the log with `grep -a`
  (it holds non-text bytes) for `verify ... failed`. A clean sweep covers only the
  compiles the runs reach.

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
