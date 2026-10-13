**Independent architecture reassessment — 28 September 2026**

My recommendation is to stop expanding the experimental tier opcode by opcode and
change the unit of work to a complete, profitable specialization path. Retain the
template JIT as the production tier. Keep Cranelift available as a backend, but
make the next investment in the JavaScript representation and feedback supplied
to it. The evidence supports rejecting the measured candidate; it does not yet
identify a single cause of the regression, establish a V8 gap, or justify a 60x
prediction.

This is an architectural review, not another optimizer revision. I read the
[handoff](/big/Code/trust-js/docs/optimization-handoff-20260928.txt), the development
records and source in both repositories, and reanalysed the saved measurements.
The original artifacts and source were left intact. The independent
[evidence record](/big/Code/trust-js/benchmark-results/architecture-reassessment-20260928/evidence.json)
contains input hashes, raw-run derivations, source identity and validation status.
Its [analysis script](/big/Code/trust-js/benchmark-results/architecture-reassessment-20260928/reanalyse.py)
requires a new output filename on every invocation.

**What the measurements actually establish.** All 16 VALUES15 runs independently
passed the recorded completeness checks: 20 suites, 58 steps and one iteration.
Recomputing the paired score and wall-time ratios reproduces the original
analysis. The following phase breakdown is newly derived from those existing
results; no new browser timing was performed.

| Measure | Template tier median | VALUES15 hot median | Median paired candidate/baseline |
| --- | ---: | ---: | ---: |
| Official score, higher is better | 0.335996 | 0.226284 | 0.67347 |
| Completion wall time | 100.930 s | 134.345 s | 1.32714 |
| Sum of measured synchronous steps | 45.717 s | 69.003 s | 1.50935 |
| Sum of measured asynchronous steps | 28.790 s | 34.229 s | 1.18941 |

These are medians of different distributions; sums of medians need not match a
median total. Phase sums are descriptive elapsed interaction time, not the
geometrically aggregated official score or exclusive CPU attribution.

All 20 suite-duration comparisons regress. React Complex DOM's synchronous
ratio is 2.706 while its asynchronous ratio is approximately 1.000; Redux is
2.742 and 0.992 respectively. Vue regresses in both phases, 2.036 and 2.363.
This directs investigation toward execution and compilation during interactions.
It cannot identify pure JavaScript cost: synchronous work includes host calls
and forced layout, and asynchronous work can execute framework JavaScript.

The [official fixture runner](/big/Code/trust-js/benchmark-results/performance-implementation-20260926-bEP6pk/speedometer-official/resources/benchmark-runner.mjs:383)
recreates its frame across iterations and navigates it for each suite. Therefore
“run more iterations” does not by itself isolate warmed application execution.
Measure long-lived application interactions separately, verify actual code reuse,
and retain the unmodified official workload as an acceptance gate. Loading and
compilation outside scored steps still matter to startup and completion time.

The sampled compiler/runtime/DOM buckets remain useful leads, with their original
unwinding, overlap and attribution limitations. The candidate profile's roughly
18.2% compiler bucket cannot simply be subtracted from controlled wall time.
Likewise, a small generated-code bucket does not mean JavaScript is cheap: its
work includes runtime helpers and allocation. The earlier inlining-policy and
call-stub ablations already failed to explain the large regression. Repeating
them without a new hypothesis would add little.

**The compiler currently optimizes too late in the representation.**
[Compilation](/big/Code/trust-js/crates/trust-js/src/jit_optimizing.rs:1059) builds control
flow, local/stack plans and category facts, then translates individual bytecode
operations into CLIF. It initially creates a block for each bytecode boundary;
guards, ownership transactions and checked continuations create more blocks.
Existing instruction/block budgets bound an attempt, but do not establish that
its cost will be recovered during this page's lifetime.

[Value analysis](/big/Code/trust-js/crates/trust-js/src/jit_optimizing_values.rs:287)
starts entry states at ANY. Property results usually remain unknown. It can
eliminate checks when categories follow from the program, but does not specialize
parameters or property-result types from runtime observations. Compact property
guards already consume some warm IC state, so describing this tier as having
*no feedback at all* would also be wrong.

Cranelift consequently sees much of the machinery for executing dynamic
operations before TRust JS has proved which machinery is unnecessary across a
sequence of operations. Its machine-level optimizer cannot invent the missing
JavaScript assumptions and recovery rules. This is a source-supported explanation
for limited optimization opportunity, and a hypothesis for compilation/runtime
cost. It is not a measured causal decomposition of the 32.65% score loss.

**Existing feedback is useful infrastructure, but is not yet the feedback loop
this compiler needs.** [FeedbackVector](/big/Code/trust-js/crates/trust-js/src/feedback.rs:705)
has stable site identities and bounded observation categories. Ordinary property
and call ICs also contain useful live observations. However, detailed feedback
is gated by `TRUST_JS_FEEDBACK_PROFILE`; it is diagnostic, not a cheap continuous
specialization policy. On ARM64, that mode
[disables many template fast paths](/big/Code/trust-js/crates/trust-js/src/jit.rs:3330).
The optimizer declines chunks carrying detailed feedback, while
[recompiled chunks receive unbound feedback](/big/Code/trust-js/crates/trust-js/src/bytecode.rs:5846)
with detailed writes disabled. A profile collected this way must not be presented
as an observation of the ordinary optimized execution path.

The missing connection is stable operation identity through transformations,
cheap live observations, guarded specialization, and bounded recovery/retry.
Reuse the schema and ICs; do not assume that enabling the existing diagnostic
switch supplies this connection. Collecting elaborate types everywhere can itself
be expensive, so prefer observations already needed to make baseline execution
fast, adding narrowly justified value-result feedback.

**Shapes and ownership explain why native operations remain complicated.**
[Props shapes](/big/Code/trust-js/crates/trust-js/src/value.rs:3113) primarily identify
ordered keys. They do not prove descriptor kind, writability, prototype identity
or the type of a field's current value. The property lowerer correctly checks
live descriptors and prototype links. Adding more native property opcodes will
not make a shape test prove those missing facts.

For repeated operations, the compiler needs explicit facts with validity scopes:
receiver layout, descriptor state, prototype dependencies and value type. Start
by reusing proven checks within regions where effects cannot invalidate them.
Consider richer immutable shape metadata or separate versioned dependencies only
with an audit of every mutation path and their memory cost. A key-only shape
cannot safely become a descriptor proof by convention.

[Packed ownership lowering](/big/Code/trust-js/crates/trust-js/src/jit_optimizing_ownership.rs:191)
retains heap values for loads/duplicates and releases them for stores/discards.
Moving those operations into native code removes a helper call, but usually
retains the ownership work. [Calls and observers](/big/Code/trust-js/crates/trust-js/src/jit_optimizing.rs:520)
still publish canonical owners, with selective local reloads and operand reloads.
Local SSA therefore does not imply that the function has stopped behaving like
an owning stack machine at these boundaries.

An intermediate representation should distinguish borrowed values, owned values,
effects and recovery state. That permits a proof that an intermediate retain/drop
pair is unnecessary, or that a check can serve several operations. Begin inside
regions with an explicit surviving owner and safe fallback. Do not remove roots,
last-owner destruction, or publication simply because a value exists in a register.

**Tiering needs a benefit model, and code publication needs narrower dependencies.**
The [hot threshold](/big/Code/trust-js/crates/trust-js/src/jit_optimizing.rs:126) is entry
count plus bytecode size. It does not account for expected remaining execution,
type stability, slow-path frequency or expanded IR cost. The
[upgrade path](/big/Code/trust-js/crates/trust-js/src/jit_optimizing.rs:154) recompiles
bytecode from the function with an empty inline plan, publishes `code2`, and bumps
a [process-wide call-cache epoch](/big/Code/trust-js/crates/trust-js/src/bytecode.rs:806).
Native compilation also occurs synchronously on the execution path. Every upgrade
can therefore invalidate unrelated raw call-cache entries. Its cost is unmeasured;
zero executable-cache evictions do not mean zero invalidation churn.

Admission should estimate whether future savings can repay compilation and
invalidation. Instrument those costs before changing the threshold. Per-code or
per-callee generations, or stable entry cells with safe retirement, merit study
if refill traffic is significant. Background compilation comes later: it needs
an immutable snapshot and safe publication, not moving the current live Rc/IC
graph onto a worker thread. Moving expensive work off-thread also does not make
unprofitable generated code profitable.

**A heap rewrite is a separate hypothesis, not the immediate prerequisite.**
[Object allocation](/big/Code/trust-js/crates/trust-js/src/value.rs:1490) creates an Rc
object, maintains a weak registry and tracks young membership. These operations,
property storage and string ownership may be important costs. The existing
[literal helpers](/big/Code/trust-js/crates/trust-js/src/bytecode.rs:19108) already move
ordinary templated object and array operands directly into their storage. Adding
those helpers again is not an optimization.

The [central heap](/big/Code/trust-js/crates/trust-js/src/heap.rs:1) is a checked migration
nucleus, not the live production heap. Its moving nursery exercises a restricted
tagged-field family. The `heap-bridge` retains Rc authority. More tests of that
nucleus alone will not change application allocation cost.

First measure allocations, lifetimes, retained bytes and ownership traffic by
object family/site. Test ownership elision and compact ordinary-object allocation
without requiring a complete moving collector first. If results justify a heap
migration, it must replace costs for a real object family, with host roots, weak
references, barriers, teardown and relocation fully connected. A second shadow
heap can add bookkeeping while preserving the original cost. No measured result
here proves that Rc alone prevents a material gain or that a moving GC supplies
one.

**The browser remains an independent part of the objective.** Baseline profiles
place substantial samples in DOM/style work. The implementation of
[`tag_name`](/big/Code/TRust/src/dom.rs:3104) is already a small indexed lookup;
its prominence suggests investigating callers, repeated walks and data locality
before inventing a faster spelling of that lookup.
[`computed_cache_get`](/big/Code/TRust/src/dom.rs:4541) clones string values,
and typed layout snapshots consume DOM-computed style. Measure lookup count,
allocation and invalidation fanout before selecting a representation change.
The dirty per-Document invalidation work is already present and must be assessed
on its own evidence.

The current [live extraction path](/big/Code/TRust/src/trust_js_backend.rs:3410)
uses the canonical DOM and retained layout fragments. HTML serialization is
diagnostic/test-only there; the controller consumes the rendered result. Thus
“remove routine serialize/reparse” and “introduce retained geometry” are not new
solutions to this path. Rebuilding after an observer mutation can be required.
Trace the first repeated or divergent stage, preserve synchronous geometry and
observer ordering, and respect the desktop/shared-engine boundary. Do not use
render skipping or scheduling changes to hide execution costs.

**The next experiment should join the missing pieces in a small scope.** I would
first add low-overhead diagnostic attribution, then select a frequent operation
sequence from the full workload. Build one path from live IC/value feedback to
specialized JavaScript operations, explicit effects/owner state, guard reuse and
compact CLIF, with single-frame recovery. Existing guarded fallback remains for
unsupported behavior. Multi-frame inlining and moving GC need not precede this
experiment.

A useful model is:

```text
bytecode + bounded live IC/value observations
    -> specialized JS operations + effects + owner/recovery state
    -> eliminate repeated checks and unnecessary ownership work
    -> compact CLIF -> machine code
```

Reuse the existing CFG, frame-state tests, cache lifetimes and heap contracts.
Avoid a broad rewrite of all template emitters as a prerequisite. Over time, a
shared semantic recipe for an IC operation could serve the baseline emitters and
optimizing frontend, reducing the current need to maintain equivalent fast paths
in several places. Mozilla's historical
[Warp design](https://hacks.mozilla.org/2020/11/warp-improved-js-performance-in-firefox-83/)
provides a concrete example of consuming baseline IC recipes in the optimizing
frontend. V8's historical [Maglev design](https://v8.dev/blog/maglev) shows another
relevant choice: specialize during graph construction and keep compilation work
small. These are architectural precedents, not TRust JS performance predictions or
instructions to adopt another engine's implementation.

Before implementing the pilot, record a savings prediction based on its measured
frequency and current cost. A helper-count target alone is insufficient. These
experiments distinguish the competing explanations:

| Question | Observation/experiment | Decision it enables |
| --- | --- | --- |
| Does compilation repay its cost? | Separate bytecode rebuilding, lowering, verification, register allocation and emission; associate them with native entries and code lifetime. Report distributions, not only totals. | Restrict admission or simplify IR when recovery is implausible. |
| Is code faster after compilation? | Compare unchanged full Speedometer plus repeated interactions in a verified long-lived application realm; record compilation during each window. | Distinguish startup debt from continuing execution loss. |
| Is ownership/guard expansion dominant? | Measure runtime fallback reasons and emitted IR/guards/owner transactions in the selected sequence; compare the complete specialization path. | Continue only if less generated machinery also saves application time. |
| Do upgrades disturb unrelated calls? | Correlate upgrade epochs with IC misses/refills using bounded counters, independently of eviction. | Justify finer publication dependencies or reject that explanation. |
| Is browser work repeated unnecessarily? | Attribute style lookups, invalidated nodes, geometry requests and actual rebuilds to actions/documents. | Choose a browser change with a measured cause and terminal regressions covered. |

Keep diagnostic and uninstrumented timing runs separate. Compare exact frozen
release artifacts in balanced fresh processes, with no owned builds/tests/profile
jobs overlapping timing. Inspect all suites, memory, compilation latency and
visual/functionality gates. More iterations can improve sampling; they do not
automatically remove compilation. A switch to LLVM, disabling the verifier, or
adding another backend should wait for evidence that a compact specialized input
still encounters a backend limitation.

The first milestone is a material full-workload improvement over the template
tier under the existing regression policy. Recovering part of the experimental
loss is useful research, but does not meet that milestone. The 60x objective
requires an end-to-end cost model and repeated broad gains; these profiles are
not precise enough to provide an Amdahl bound or a numerical promise.

**Standards constraints and freedom.** I used the local web-standards skill and
official snapshots, without refreshing them: ECMA-262 `e28783d5`, DOM `a2331a45`,
HTML `e5071a20`, and CSSWG `81c27f68`, fetched 6 September 2026. These are local
draft/living-standard snapshots, not verified-current upstream text.

ECMA-262 [execution contexts](/big/web-standards/repositories/tc39/ecma262/spec.html:12222)
are an abstract semantic mechanism; the standard does not mandate the current
canonical Rust frame representation
([official clause](https://tc39.es/ecma262/#sec-execution-contexts)).
[OrdinaryGet/OrdinarySet](/big/web-standards/repositories/tc39/ecma262/spec.html:13389)
require preserving descriptors, receiver behavior, observable calls and completion
ordering ([official clause](https://tc39.es/ecma262/#sec-ordinarysetwithowndescriptor)).
[WeakRef liveness](/big/web-standards/repositories/tc39/ecma262/spec.html:12875)
allows implementation freedom while preserving observable identity and kept-alive
targets ([official clause](https://tc39.es/ecma262/#sec-weakref-processing-model)).
These support changing representation through explicit proofs, rather than
treating existing storage contracts as language requirements.

For browser work I also read local DOM mutation-observer delivery at `dom.bs:3961`,
HTML rendering and microtask checkpoints at `source:123182` and `source:123479`,
CSSOM computed style at `cssom-1/Overview.bs:3380`, and CSSOM View bounding boxes at
`cssom-view-1/Overview.bs:1329`. Reducing repeated computation must retain those
observable results and scheduling boundaries. Any implementation needs the
focused conformance tests and broader checks required by the repository guides.

**Evidence and unfinished gates retained.** TRust JS HEAD remains
`91da1ade662d5a3e8dcdb2a2def189529575d945`; TRust remains
`5fa5a3031df744b108d0c9b6efd726f410ab4f3a`. All 37 TRust JS and two TRust files in the
frozen CREATION17 changed-file manifest still match their recorded hashes. The
VALUES15 timing binary and all four frozen CREATION17 binaries were hash-checked.
This verifies those files, not a clean tree or completed acceptance.

HEAP16's formerly running bailout Test262 job completed at 06:33:02 UTC: 53,572
pass, two fail, four skip, with `all_executed_passed=false`. Its full report is
byte-identical to the forced result, hash
`ed3407449599727ca94b0a1ec7fe98f77180da9c7304092272f7e75a0325dbcf`.
Both failing paths and all skip reasons are copied into the separate evidence
record. CREATION17's release unit-test *build* completed; that is not test
execution. Its x64 report records only creation 10 and heap 17 passes, 27 of the
handoff's expected 107, with no completed remainder established. The listed
controller/server PIDs were absent when inspected; no replacement timing or
validation job was started for this review.

The failed functional start and singleton-refused conformance attempt remain
failed attempts. CREATION17 still lacks controlled application timing and the
unfinished correctness/visual gates in the handoff. The prior CSS WPT 170/38
result, Steam visual/scroll acceptance, string Map/Set regressions and unexplained
idle-callback failure remain open records. This review changes no engine code,
installs nothing, and makes no promotion, commit or push.
