// Sloppy function parameters remain reflectable through function.arguments during the call.
function run(n) {
    let sum = 0;
    for (let index = 0; index < n; ++index) sum += index & 255;
    return sum;
}
const start = Date.now();
const checksum = run(32000000);
if (checksum !== 4080000000) throw new Error("parameter bound checksum " + checksum);
console.log(JSON.stringify({ workload: "sloppy-bound-first-call", milliseconds: Date.now() - start, checksum }));
