// An ordinary constructor's long first invocation must retain live this/new.target.
let prefixes = 0;
function Counter(value) {
    ++prefixes;
    for (let index = 0; index < 32000000; ++index) ++value;
    this.value = value;
    this.correctTarget = new.target === Counter;
}
const start = Date.now();
const result = new Counter(0);
if (result.value !== 32000000 || !result.correctTarget || prefixes !== 1) throw Error("constructor state");
console.log(JSON.stringify({ workload: "constructor-first-call", milliseconds: Date.now() - start, checksum: result.value, prefixes }));
