// Prefix-created closures keep their original live function environment across tier-up.
"use strict";
function run(n) {
    let sum = 0;
    const read = () => sum;
    for (let index = 0; index < n; ++index) {
        sum += index & 255;
        if ((index & 1023) === 1023 && read() !== sum) throw new Error("capture alias");
    }
    return read();
}
const start = Date.now();
const checksum = run(4000000);
if (checksum !== 510000000) throw new Error("captured loop checksum " + checksum);
console.log(JSON.stringify({ workload: "captured-first-call", milliseconds: Date.now() - start, checksum }));
