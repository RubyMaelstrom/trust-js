// A parked AST binding must transfer its owner to the live compiled frame. Once
// the slot is overwritten and GC runs, two touched 64 MiB buffers must not overlap.
"use strict";
const size = 64 * 1024 * 1024;
function run() {
    let old = new Uint8Array(size);
    for (let page = 0; page < size; page += 4096) old[page] = 1;
    for (let index = 0; index < 3000; index++) {
        if (index === 200) { old = null; $262.gc(); }
        if (index === 500) {
            let next = new Uint8Array(size);
            for (let page = 0; page < size; page += 4096) next[page] = 2;
            return next[0] + next[size - 4096];
        }
    }
}
const checksum = run();
if (checksum !== 4) throw new Error("owner retirement checksum " + checksum);
console.log(checksum);
