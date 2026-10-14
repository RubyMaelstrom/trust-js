// Literal callback replacement and limited empty-separator split scaling.
// Run with Node or the release engine shell. Each row prints five samples and checksums.
// Compare interleaved fresh processes with the same binary build flags and CPU affinity.
"use strict";

function replacement(source) {
    let checksum = 0;
    const result = source.replaceAll("a", function (match, offset, whole) {
        if (whole !== source || match !== "a") throw new Error("callback input");
        checksum += offset;
        return "b";
    });
    return result.length + checksum;
}

function limitedSplit(source) {
    let checksum = 0;
    for (let repeat = 0; repeat < 8; ++repeat) {
        const result = source.split("", 1);
        if (result.length !== 1) throw new Error("split limit");
        checksum += result[0].charCodeAt(0);
    }
    return checksum;
}

function fullSplit(source, repetitions = 4) {
    let checksum = 0;
    for (let repeat = 0; repeat < repetitions; ++repeat) {
        const result = source.split("");
        if (result.length !== source.length) throw new Error("full split length");
        checksum += result.length + result[0].charCodeAt(0) +
            result[result.length - 1].charCodeAt(0);
    }
    return checksum;
}

function fullSplitAscii(source) {
    // Produce one million code units per sample to reduce clock quantization.
    return fullSplit(source, 1048576 / source.length);
}

function measure(name, size, source, operation, expected) {
    for (let warm = 0; warm < 2; ++warm) {
        if (operation(source) !== expected) throw new Error(name + " warm checksum");
    }
    const samples = [], checksums = [];
    for (let sample = 0; sample < 5; ++sample) {
        const start = Date.now();
        const checksum = operation(source);
        samples.push(Date.now() - start);
        checksums.push(checksum);
        if (checksum !== expected) throw new Error(name + " checksum");
    }
    console.log(JSON.stringify({workload: name, size, samples_ms: samples, checksums}));
}

for (const size of [4096, 16384, 65536]) {
    const ascii = "a".repeat(size), unicode = "é😀a".repeat(size);
    measure("replace_callback_ascii", size, ascii, replacement,
        size + size * (size - 1) / 2);
    measure("replace_callback_unicode", size, unicode, replacement,
        4 * size + 2 * size * (size - 1) + 3 * size);
}
for (const size of [16384, 65536, 262144]) {
    measure("split_empty_limit_ascii", size, "a".repeat(size), limitedSplit, 8 * 97);
    measure("split_empty_limit_unicode", size, "😀é".repeat(size), limitedSplit, 8 * 0xd83d);
}
for (const size of [4096, 16384, 65536]) {
    measure("split_empty_full_ascii", size, "a".repeat(size), fullSplitAscii,
        (1048576 / size) * (size + 194));
    measure("split_empty_full_unicode", size, "😀é".repeat(size), fullSplit,
        4 * (3 * size + 0xd83d + 233));
}
for (const size of [4096, 16384, 65536]) {
    const view = ("a".repeat(size) + "!").slice(0, size);
    measure("split_empty_full_ascii_view", size, view, fullSplitAscii,
        (1048576 / size) * (size + 194));
}
console.log("STRING_SCALING_COMPLETE");
