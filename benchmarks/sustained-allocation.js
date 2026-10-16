// Sustained allocation over a large retained heap (the P1 plan's sustained-allocation workload,
// docs/v8-chase-p1-plan-2026-10-15.txt).
//
// Isolates allocation and collection cost: each round allocates short-lived two-property
// objects, arrays and closures while a retained set of 200k objects stays live, and replaces a
// slice of that set so old objects die after surviving collections. A tracing collector marks
// the retained heap at every collection; reference counting frees each temporary at once.
// Prints a checksum; compare a baseline and a candidate by interleaved runs, e.g.
//
//   target/release/trust-js --tier=bytecode benchmarks/sustained-allocation.js
//
// with the same build settings, tier and CPU affinity (see README.md). For the collector's
// share, add TRUST_JS_PERF_METRICS=1 (pause total, maximum and histogram); under the gc-heap
// feature TRUST_JS_GC_LOG=1 adds live bytes and pages after each collection.

function churn(rounds, retained) {
    let sum = 0;
    for (let r = 0; r < rounds; r++) {
        for (let i = 0; i < 20000; i++) {
            const point = { x: i, y: r };
            const pair = [point, i];
            const read = () => pair[0].x + pair[1];
            sum = (sum + read() + point.y) | 0;
        }
        // Replace a slice of the retained set so old objects die after surviving collections.
        for (let i = 0; i < 2000; i++) {
            const slot = (r * 2000 + i) % retained.length;
            retained[slot] = { id: slot, round: r, next: retained[(slot + 1) % retained.length] };
        }
    }
    return sum;
}

const retained = [];
for (let i = 0; i < 200000; i++) retained.push({ id: i, round: -1, next: null });
const result = churn(20, retained);
let links = 0;
for (let i = 0; i < retained.length; i++) if (retained[i].next) links++;
console.log("checksum", result, links);
