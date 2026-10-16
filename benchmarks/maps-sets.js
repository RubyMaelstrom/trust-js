// Map and Set operations over many small collections (P1 S3 payload-move measurement).
//
// Isolates [[MapData]]/[[SetData]] access: every get/set/has/delete and iteration step reaches
// the collection behind a Map or Set object. 2,000 live collections keep that lookup from
// being a single hot cache line; the values are small integers and short strings, so little
// else allocates. Prints a checksum; compare a baseline and a candidate by interleaved runs, e.g.
//
//   target/release/trust-js --tier=bytecode benchmarks/maps-sets.js
//
// with the same build settings, tier and CPU affinity (see README.md).

const maps = [];
const sets = [];
for (let i = 0; i < 2000; i++) {
    const map = new Map();
    const set = new Set();
    for (let k = 0; k < 8; k++) {
        map.set(k, i + k);
        set.add("s" + ((i + k) % 64));
    }
    maps.push(map);
    sets.push(set);
}

let sum = 0;
for (let round = 0; round < 60; round++) {
    for (let i = 0; i < maps.length; i++) {
        const map = maps[(i * 7 + round) % maps.length];
        const set = sets[(i * 13 + round) % sets.length];
        const key = (i + round) & 7;
        sum = (sum + map.get(key)) | 0;
        map.set(key, (map.get(key) + 1) | 0);
        if (set.has("s" + ((i + round) % 64))) sum = (sum + 1) | 0;
        if ((round & 7) === 0) {
            set.delete("s" + (i % 64));
            set.add("s" + (i % 64));
        }
    }
    if ((round & 15) === 0) {
        for (const map of maps) for (const [k, v] of map) sum = (sum + k + v) | 0;
        sets[round % sets.length].forEach((value) => { sum = (sum + value.length) | 0; });
    }
}
console.log("checksum", sum, maps[5].size, sets[9].size);
