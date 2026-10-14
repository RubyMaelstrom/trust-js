// Short one-shot loop functions should remain interpreted. The source and result count are
// fixed; run fresh release processes and keep interleaved baseline/candidate samples.
"use strict";
let checksum = 0;
const start = Date.now();
for (let index = 0; index < 2000; ++index) {
    const run = new Function("n", "let sum=0;for(let k=0;k<n;k++)sum+=k;return sum;");
    checksum += run(index % 16);
}
if (checksum !== 70000) throw new Error("cold loop checksum " + checksum);
console.log(JSON.stringify({ workload: "cold-functions", milliseconds: Date.now() - start, checksum }));
