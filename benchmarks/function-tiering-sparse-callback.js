// Parameter updates must stay fast when a long loop occasionally calls out.
// The callback observes the value passed to it; this fixture does not depend on
// the engine-specific legacy function.arguments extension.
let observed = -1;
function observe(value) { observed = value; }
function run(value) {
    for (let index = 0; index < 32000000; ++index) {
        ++value;
        if (index === 16000000) observe(value);
    }
    return value;
}
const start = Date.now();
const checksum = run(0);
if (checksum !== 32000000 || observed !== 16000001)
    throw new Error("sparse callback checksum " + checksum + ":" + observed);
console.log(JSON.stringify({ workload: "sloppy-parameter-sparse-callback", milliseconds: Date.now() - start, checksum, observed }));
