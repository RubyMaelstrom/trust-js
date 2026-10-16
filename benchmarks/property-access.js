// Property-heavy loop on shape-stable objects (P1 S2 perf tripwire).
//
// Isolates named property reads and writes, prototype method calls and string-keyed computed
// access on a small set of long-lived objects: little allocation, so it measures the object
// model's access paths rather than the collector. Prints a checksum; compare a baseline and a
// candidate by interleaved runs, e.g.
//
//   target/release/trust-js --tier=bytecode benchmarks/property-access.js
//
// with the same build settings, tier and CPU affinity (see README.md).

class Vec3 {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    dot(other) {
        return this.x * other.x + this.y * other.y + this.z * other.z;
    }
    scaleInPlace(k) {
        this.x *= k;
        this.y *= k;
        this.z *= k;
    }
}

const points = [];
for (let i = 0; i < 64; i++) points.push(new Vec3(i, i + 1, i + 2));
const record = { alpha: 1, beta: 2, gamma: 3, delta: 4 };
const keys = ["alpha", "beta", "gamma", "delta"];

let acc = 0;
for (let round = 0; round < 8000; round++) {
    for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const q = points[(i + round) & 63];
        acc = (acc + p.dot(q)) % 1000003;
        p.x = (p.x + 1) % 97;
        q.z = (q.z + p.y) % 89;
    }
    const key = keys[round & 3];
    record[key] = (record[key] + round) % 101;
    acc = (acc + record.alpha + record[keys[(round + 1) & 3]]) % 1000003;
    if ((round & 1023) === 0) points[round & 63].scaleInPlace(1);
}
console.log("checksum", acc, record.alpha + record.beta + record.gamma + record.delta);
