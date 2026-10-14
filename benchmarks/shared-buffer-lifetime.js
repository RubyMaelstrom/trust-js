// Fixed-work lifetime probe: unreachable shared blocks should not accumulate.
// Run with the engine shell, or Node with --expose-gc.
"use strict";
const collect = typeof $262 === "object" ? $262.gc : gc;
const rounds = 32;
const bytesPerRound = 16 * 1024 * 1024;
function allocateAndTouch(value) {
    const bytes = new Uint8Array(new SharedArrayBuffer(bytesPerRound));
    for (let index = 0; index < bytes.length; index += 4096) bytes[index] = value;
    return bytes[0] + bytes[bytes.length - 4096];
}
let checksum = 0;
for (let round = 0; round < rounds; ++round) {
    checksum += allocateAndTouch(round);
    collect();
}
if (checksum !== rounds * (rounds - 1)) throw Error("shared bytes changed");
console.log(JSON.stringify({ rounds, bytesPerRound, checksum }));
