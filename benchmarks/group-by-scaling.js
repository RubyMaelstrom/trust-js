// GroupBy grouping and iterator costs. Run with the release engine shell or Node.
// GROUP_BY_CONFIG may select cases/sizes/samples before loading this file.
// Source arrays and keys are prepared outside the timed interval. The generator
// yields the same source values; every result's key order and members are checked.
// These components do not predict an application's complete execution time.
(function () {
    "use strict";
    const config = typeof GROUP_BY_CONFIG === "undefined" ? {} : GROUP_BY_CONFIG;
    const sizes = config.sizes || [1024, 4096, 16384, 65536];
    const samples = config.samples || 3;
    const backends = config.backends || ["Object", "Map"];
    const inputs = config.inputs || ["array", "generator"];
    const cardinalities = config.cardinalities || ["recurring", "distinct"];

    function* values(source) {
        for (let index = 0; index < source.length; index++) yield source[index];
    }

    function group(backend, input, source, callback) {
        const items = input === "array" ? source : values(source);
        return backend === "Object" ? Object.groupBy(items, callback) : Map.groupBy(items, callback);
    }

    function verify(backend, result, keys, size, groupCount) {
        const actualKeys = backend === "Object" ? Object.keys(result) : Array.from(result.keys());
        if (actualKeys.length !== groupCount)
            throw new Error("wrong number of groups: " + actualKeys.length);
        if (backend === "Object" && Object.getPrototypeOf(result) !== null)
            throw new Error("grouped object has a prototype");
        let checksum = 0, members = 0;
        for (let offset = 0; offset < groupCount; offset++) {
            const key = actualKeys[offset];
            if (key !== keys[offset]) throw new Error("wrong group encounter order");
            const membersInGroup = backend === "Object" ? result[key] : result.get(key);
            const expectedLength = Math.floor((size - 1 - offset) / groupCount) + 1;
            if (membersInGroup.length !== expectedLength) throw new Error("wrong group length");
            for (let index = 0; index < membersInGroup.length; index++) {
                const member = membersInGroup[index];
                if (member !== offset + index * groupCount) throw new Error("wrong group member");
                checksum += member;
                members++;
            }
        }
        if (members !== size || checksum !== size * (size - 1) / 2)
            throw new Error("wrong independent member sum");
        return checksum;
    }

    for (const backend of backends) {
        for (const input of inputs) {
            for (const cardinality of cardinalities) {
                for (const size of sizes) {
                    const groupCount = cardinality === "distinct" ? size : Math.min(size, 16);
                    const source = [], keys = [];
                    for (let index = 0; index < size; index++) source.push(index);
                    for (let index = 0; index < groupCount; index++) keys.push("group_" + index);
                    const callback = value => keys[value % groupCount];
                    for (let warm = 0; warm < 3; warm++)
                        group(backend, input, source.slice(0, Math.min(size, 64)), callback);
                    const timings = [], checksums = [];
                    const sampleCount = cardinality === "distinct" && size >= 65536 ?
                        (config.largeDistinctSamples || 1) : samples;
                    for (let sample = 0; sample < sampleCount; sample++) {
                        const started = Date.now();
                        const result = group(backend, input, source, callback);
                        timings.push(Date.now() - started);
                        checksums.push(verify(backend, result, keys, size, groupCount));
                    }
                    console.log(JSON.stringify({workload: "group-by", backend: backend,
                        input: input, cardinality: cardinality, size: size, groups: groupCount,
                        samples_ms: timings, checksums: checksums}));
                }
            }
        }
    }
    console.log("group-by-scaling complete");
})();
