// A sloppy simple parameter can be reflected through function.arguments during its live call.
// This measures the migration representation required to keep that ParameterMap intact.
function run(value) {
    for (let index = 0; index < 32000000; ++index) ++value;
    return value;
}
const start = Date.now();
const checksum = run(0);
if (checksum !== 32000000) throw new Error("parameter loop checksum " + checksum);
console.log(JSON.stringify({ workload: "sloppy-parameter-first-call", milliseconds: Date.now() - start, checksum }));
