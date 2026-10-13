**Architecture implementation record — 29 September 2026**

Final source and release validation are complete. This supplements the original
[independent reassessment](architecture-reassessment-20260928.md), whose evidence
and rejected approaches remain preserved. Complete component results, input
hashes, artifact identities and limitations are in the
[release record](../benchmarks/architecture-release-2026-09-29.json).

The default release improves full Speedometer score by **20.27%**
over the frozen original template-mode creation17 browser, in four balanced
fresh-process pairs. The paired log-t 95% score interval is
14.17%–26.68%.
Complete-process elapsed time falls 14.34% and observed
CPU falls 14.90%. All 20 whole-suite duration
estimates improve; every run completes all 58 actions and leaves zero owned
browser processes. No component crosses the predeclared regression gate.

ES5 TodoMVC takes 40.75% less time, Observable
Plot 25.01% less, and Chart.js
5.86% less. The former Plot asynchronous-phase blocker
has baseline/candidate medians 75.60/73.80ms,
geometric duration ratio 0.9969 and interval
0.7863–1.2640. It no longer crosses the gate; this wide phase interval
is not evidence of a precise phase speedup.

Observed kernel peak memory has ratio
0.9074, interval
0.8068–1.0205. All four observed candidate peaks are
lower, but the interval includes parity. Do not claim a proven whole-workload
memory reduction or post-GC retained-memory result. Kernel VmHWM has declared
precedence over two-second sampled RSS; every observation remains retained.

The separate persistent-document comparison loads three pinned applications
once each and executes six workflows per application. All 48 observed application
states match in every process. Later workflows take
14.96% less time, with duration-ratio interval
0.8137–0.8887; complete-process elapsed time falls
12.65%. Observed kernel peak memory is
10.83% lower, with ratio interval
0.8707–0.9132. No component crosses its declared gate. This
is a separate repeated-interaction workload, not an official Speedometer score.

All 14 collection families complete four balanced pairs with exact checksums and
no blocked component. String Map/Set duration ratios are
1.0084/0.9646.
Numeric and churn kernels retain larger isolated improvements; those are not
whole-engine speedup claims. All intervals and raw internal samples are retained.

The selected improvements are active by default:

- Dense arrays retain contiguous descriptors through construction and growth,
  avoiding decimal-key allocation and a second element directory. Sparse indices,
  holes, accessors and Array length behavior retain their checked paths.
- Closed native iterator consumers avoid unobservable result-object allocation.
  Public next calls still produce distinct results, and custom iterators,
  getters and other observable behavior follow the full algorithms.
- Compiled invocations share immutable binding plans and wide-scope name indexes,
  while owning separate values and initialization state. Structural mutation
  invalidates fixed-layout proofs. Indexed lookup repairs the measured Chart.js
  regression from the earlier planned linear representation.
- The mixed native/JavaScript graph supplies DOM wrapper identity. Single-edge
  traversal and live child collections remove child-array reconstruction and
  duplicate JavaScript wrapper-management ownership.
- Bounded cascade/computation graphs and typed style records reuse equal complete
  inputs across nodes and mutations. Logical axes, inherited computation,
  variables, transitions and resource revisions remain explicit dependencies.
  Deferred invalidation flushes at observable style/layout reads.
- Baseline and region ARM64 emission share 50 existing native operations.
  Retiring an embedder settings context requests a coalesced task-boundary
  collection independently of allocation pressure, while respecting host
  deferral and keeping retained Document/Window objects alive.

Cranelift and its loop-continuation experiment remain optional and disabled in
the shipping build. Bounded live feedback, effects, ownership, frame publication,
precise VM recovery, semantic admission and version retirement are implemented.
The unchanged scalar field-loop kernel improves about 4.52×, but the application
screen is negative: these compiler capabilities do not justify enabling this
backend. Broader inlining and tail-transfer experiments were flat and remain
parked. Neither a moving collector nor an LLVM replacement is justified by the
measurements. The actual gains above come from the default implementation.

Disposition of the six reassessment goals:

| Goal | Result |
| --- | --- |
| G1: causal attribution | Separate compiler/IR/lifetime, allocation-family, call-cache and native-profile evidence; diagnostic counters excluded from timing. Inclusive times are not summed as exclusive CPU costs. |
| G2: connected specialization | Implemented bounded feedback, stable operation identities, effects, ownership, guards and VM recovery. Keep the compiler experimental because application repayment is not demonstrated. |
| G3: admission/publication | Benefit/IR budgets, successful-compilation-before-publication, miss retirement and active-code leases implemented. Default execution pays no optional compiler admission cost. |
| G4: actual allocation | Dense descriptors, direct closed-native consumption, shared activation metadata and document-retirement collection operate on the live production graph. Ownership/lifetime checks and full workloads pass. |
| G5: repeated browser work | Native wrappers/collections, deferred invalidation and bounded style computations integrated in both frontends, with mutation/lifetime/CSS coverage and final live rendering. |
| G6: broad validation | Final debug/release and strict Clippy pass; all final performance gates pass. Earlier cross-platform/conformance evidence and its known failures remain explicitly identified below. |

Final checks use source-clean63-v2 and the ordinary primary release artifacts:

- Engine debug and release each pass 1,758 tests. Browser debug and release each
  pass 2,193 library, 53 desktop, 14 headless and one replay test, with 35 existing
  ignores. Three Test262-report unit tests pass.
- Both repositories pass default and all-feature/all-target Clippy with
  -D warnings; TRust JS covers its entire workspace. Formatting and whitespace
  checks pass. cargo build --release builds all three shipping browser binaries.
- The final desktop and terminal releases load Archive.org's 21,819-result
  LibriVox collection, including covers, titles and further rows after scrolling.
  Initial incomplete states and final screenshots remain preserved. This does
  not establish playback, downloads, login or whole-site fidelity.
- Steam's earlier integration60 desktop/terminal interaction evidence retains its
  source identity. No final63 Steam execution is claimed. The known clipped
  desktop logo and blank initial terminal hero remain documented.
- Earlier integration60 Test262 reports 53,572 passes, two known legacy-reflection
  failures and four skips at revision d86b2294eb0a17eaa281ff12c73c473ec864c72f.
  Its complete report is byte-identical to the historical report; this is not
  clean conformance. CSS WPT retains 170 passes, 38 known failures and zero
  incomplete pages, with identical outcomes/messages. Linux-musl x86-64 under
  QEMU passes 266 selected checks; that is correctness evidence, not native timing.
  These broad suites were not rerun after the retirement request and lint cleanup.

Failures and rejected conclusions remain preserved. Integration60's score gain
of 17.61% did not excuse its 73.05/107.00 ms Plot phase regression. A chart-only
history manifest incorrectly claimed Chart.js preceded Plot; the actual subset
ran Plot first and failed to preserve the full workload prefix. Its correction
is retained rather than treating that subset as causal evidence.

Full-context profiles 61 include about 44 ms current versus 16 ms baseline inclusive
collection in the problematic phase. The baseline releases 18,155 old objects
between TipTap and Plot, while the candidate carries 19,591 into Plot's collection.
The candidate window uses a recorded post-run MONOTONIC_RAW approximation;
baseline correlation was captured during the run. Buffered console ordering is
not timing evidence. The general retirement-pressure request follows HTML
discard-a-document (local e5071a20) and ECMA-262 liveness/WeakRef invariants
(e28783d5), without site or benchmark detection.

Validation also retains the old dense-storage assertion failure, the invalid
total-DOM-memory monotonicity assertion, the unsuccessful cross-workspace Cargo
invocation, intermediate Clippy failures, and the 62 helper exit 143. The revised
class-token test isolates its cache's memory accounting while retaining all
matching/invalidation checks. Final full suites pass. The 62 Steam interaction
overclaim and the 63 controller's source-path typo each have explicit corrections.
No failed attempt or source version was replaced by a passing-only history.

Primary integration preserves all outstanding iframe/media changes in addition
to the Archive grammar repair and selected optimization work. The original
handoff, reassessment, every failed experiment, and the pre-final outcome record
remain preserved. README files and installed executables were not changed.
The release artifacts are under /big/Code/TRust/target/release; their exact
SHA256 values and validation commands are in the machine-readable release record.

Evidence root:
/big/Code/TRust/target/architecture-rework-20260928-gGJhNCNs

The chronological [work plan](architecture-work-plan-20260928.txt) records the
standards, intermediate decisions and corrections. The final comparison records
are clean63-v1-official/, clean63-v1-persistent/ and clean63-v1-collections/;
clean63-validation.json identifies successful checks and retained failures.
