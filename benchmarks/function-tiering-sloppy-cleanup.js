// A non-strict loop inside author cleanup must retain fast local parameter access.
let cleanups = 0;
function run(count) {
    let sum = 0;
    try {
        for (let index = 0; index < count; ++index) sum += index & 255;
        return sum;
    } finally {
        ++cleanups;
    }
}
const start = Date.now();
const checksum = run(32000000);
if (checksum !== 4080000000 || cleanups !== 1) throw Error("sloppy cleanup state");
console.log(JSON.stringify({ workload: "sloppy-cleanup-first-call", milliseconds: Date.now() - start, checksum, cleanups }));
