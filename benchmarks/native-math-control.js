// Ordinary Math receiver controls for inline native-call guards. All sums have
// independent closed-form expectations for an integral number of input periods.
// Each kernel emits every warmed timing and checksum. Run with Node or the
// release engine shell, using interleaved fresh processes on the same CPU.
"use strict";

function absFround(count) {
    let sum = 0;
    for (let index = 0; index < count; index++)
        sum += Math.fround(Math.abs((index & 255) - 128) * 0.5);
    return sum;
}

function imulClz32(count) {
    let sum = 0;
    for (let index = 0; index < count; index++)
        sum += Math.imul(index & 255, 3) + Math.clz32((index & 15) + 1);
    return sum;
}

function minMax(count) {
    let sum = 0;
    for (let index = 0; index < count; index++)
        sum += Math.min(index & 31, 11) + Math.max(index & 15, 5);
    return sum;
}

function rounding(count) {
    let sum = 0;
    for (let index = 0; index < count; index++) {
        const value = (index & 63) - 31.25;
        sum += Math.floor(value) + Math.ceil(value) + Math.trunc(value);
    }
    return sum;
}

function squareRoot(count) {
    let sum = 0;
    for (let index = 0; index < count; index++) {
        const value = index & 15;
        sum += Math.sqrt(value * value);
    }
    return sum;
}

function numericIsNaN(count) {
    let sum = 0;
    for (let index = 0; index < count; index++) {
        sum += isNaN(index) ? 1 : 0;
        sum += isNaN(NaN) ? 1 : 0;
    }
    return sum;
}

const count = 4194304;
const kernels = [
    ["abs-fround", absFround, 134217728],
    ["imul-clz32", imulClz32, 1724383232],
    ["min-max", minMax, 72876032],
    ["floor-ceil-trunc", rounding, 0],
    ["sqrt", squareRoot, 31457280],
    ["isNaN", numericIsNaN, count]
];
for (const [name, kernel, expected] of kernels) {
    for (let warm = 0; warm < 3; warm++) kernel(4096);
    const samples = [], checksums = [];
    for (let sample = 0; sample < 5; sample++) {
        const start = Date.now();
        const checksum = kernel(count);
        samples.push(Date.now() - start);
        checksums.push(checksum);
        if (checksum !== expected) throw new Error(name + " checksum: " + checksum);
    }
    console.log(JSON.stringify({workload: name, count: count,
        samples_ms: samples, checksums: checksums}));
}
console.log("NATIVE_MATH_CONTROL_COMPLETE");
