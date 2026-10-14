// Active author cleanup keeps its established context while a hot inner loop changes tier.
"use strict";
let cleanups = 0;
function run(n) {
    let sum = 0;
    try {
        for (let index = 0; index < n; ++index) sum += index & 255;
        return sum;
    } finally {
        ++cleanups;
    }
}
const start = Date.now();
const checksum = run(32000000);
if (checksum !== 4080000000 || cleanups !== 1) throw new Error("cleanup loop checksum " + checksum);
console.log(JSON.stringify({ workload: "outer-cleanup-first-call", milliseconds: Date.now() - start, checksum, cleanups }));
