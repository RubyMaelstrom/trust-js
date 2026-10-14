// Recovery after a temporary sparse write, with pristine dense arrays as controls.
// Run in fresh baseline/candidate engine shells or Node with matching CPU affinity.
// Construction, sparse writes, repair and validation stay outside each timed interval.
// A preceding script can set SPARSE_RECOVERY_CONFIG to select sizes/rounds/samples.
"use strict";

var config = typeof SPARSE_RECOVERY_CONFIG === "object" ? SPARSE_RECOVERY_CONFIG : {};
var sizes = config.sizes || [128, 1024, 4096];
var rounds = config.rounds || 131072;
var numericOperations = config.numericOperations || 4194304;
var samples = config.samples || 5;

function makeArray(size, repair) {
    var array = [];
    for (var index = 0; index < size; index++) array.push(index);
    if (repair !== "pristine") {
        array[1000000] = 7;
        if (repair === "delete") delete array[1000000];
        array.length = size;
    }
    return array;
}

function tail(array, count) {
    var checksum = 0;
    for (var index = 0; index < count; index++) {
        array.push(index);
        checksum += array.pop();
    }
    return checksum;
}

function numericRead(array, count) {
    var checksum = 0;
    for (var repeat = 0; repeat < count; repeat++) {
        for (var index = 0; index < array.length; index++) checksum += array[index];
    }
    return checksum;
}

function validate(array, size) {
    if (array.length !== size || array[0] !== 0 || array[size - 1] !== size - 1 ||
        Object.hasOwn(array, 1000000) || Object.keys(array).length !== size)
        throw new Error("invalid recovered array");
}

var operations = ["push-pop", "numeric-read"];
var repairs = ["pristine", "delete", "truncate"];
for (var operationIndex = 0; operationIndex < operations.length; operationIndex++) {
    var name = operations[operationIndex];
    var operation = name === "push-pop" ? tail : numericRead;
    for (var sizeIndex = 0; sizeIndex < sizes.length; sizeIndex++) {
        var size = sizes[sizeIndex];
        var count = name === "push-pop" ? rounds : Math.max(16, Math.floor(numericOperations / size));
        var expected = name === "push-pop" ? count * (count - 1) / 2 : count * size * (size - 1) / 2;
        for (var repairIndex = 0; repairIndex < repairs.length; repairIndex++) {
            var repair = repairs[repairIndex];
            for (var warm = 0; warm < 8; warm++) operation(makeArray(32, repair), 32);
            var milliseconds = [], checksums = [];
            for (var sample = 0; sample < samples; sample++) {
                var array = makeArray(size, repair);
                validate(array, size);
                var start = Date.now();
                var checksum = operation(array, count);
                milliseconds.push(Date.now() - start);
                if (checksum !== expected) throw new Error(name + "/" + repair + "/" + size + " checksum");
                checksums.push(checksum);
                validate(array, size);
            }
            console.log(JSON.stringify({workload: name, repair: repair, size: size,
                operations: name === "numeric-read" ? count * size : count,
                samples_ms: milliseconds, checksums: checksums}));
        }
    }
}
console.log("ARRAY_SPARSE_RECOVERY_COMPLETE");
