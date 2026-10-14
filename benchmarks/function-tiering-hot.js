// A long first invocation must enter native code without repeating its prefix.
"use strict";
let prefixes = 0;
function run(n) {
    ++prefixes;
    let sum = 0;
    for (let index = 0; index < n; ++index) sum += index & 255;
    return sum;
}
const start = Date.now();
const checksum = run(32000000);
if (checksum !== 4080000000 || prefixes !== 1) throw new Error("hot loop checksum " + checksum);
console.log(JSON.stringify({ workload: "hot-first-call", milliseconds: Date.now() - start, checksum, prefixes }));
