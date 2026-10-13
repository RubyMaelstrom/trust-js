// Standalone CPU kernels for renderer-shaped JavaScript. No WebGL or browser APIs.
// Run with a release engine shell: target/release/trust-js benchmarks/webgl-engine-kernels.js
// Compare interleaved fresh processes; retain every sample and verify checksums.
// These isolate engine costs and do not predict a whole scene's frame rate.
"use strict";

function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0, n = a.length; i !== n; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function copy(a, b) {
    a.length = b.length;
    for (let i = 0, n = b.length; i !== n; ++i) a[i] = b[i];
}

const source = [1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25,
    9.25, 10.25, 11.25, 12.25, 13.25, 14.25, 15.25, 16.25];
const destination = [];
function arrayCopyCompare(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) {
        copy(destination, source);
        if (arraysEqual(destination, source)) ++sum;
    }
    return sum;
}

const small = [1.25, 2.25, 3.25];
function smallArray(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) {
        small[0] = i + 0.5;
        sum += small[0] + small[1] + small[2];
    }
    return sum;
}

const typed = new Float32Array(16);
function typedCopy(n) {
    for (let i = 0; i !== n; ++i) typed.set(source);
    return typed[0] + typed[15];
}

class Setter {
    constructor() { this.value = 0; }
    setValue(gl, value) {
        // Keep this call distinct from speculative body inlining.
        try { this.value = value + gl.bias; return this.value; } finally { }
    }
}
const setter = new Setter(), gl = {bias: 0.5}, textures = {id: 1};
function exactArguments(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) sum += setter.setValue(gl, i);
    return sum;
}
function surplusArguments(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) sum += setter.setValue(gl, i, textures);
    return sum;
}

const uniforms = {red: 1, green: 2, blue: 3, alpha: 4};
const names = ["red", "green", "blue", "alpha"];
function computedReads(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) sum += uniforms[names[i & 3]];
    return sum;
}
function enumeration(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) {
        for (const name in uniforms) sum += uniforms[name];
    }
    return sum;
}

const base = Object.create(null);
let instance = base;
for (let i = 0; i < 5; ++i) instance = Object.create(instance);
function absentProperties(n) {
    let sum = 0;
    for (let i = 0; i !== n; ++i) {
        if (instance.isMesh === undefined) ++sum;
        if (instance.isMaterial === undefined) ++sum;
    }
    return sum;
}

function arithmetic(n) {
    let sum = 0;
    for (let i = 0; i < n; ++i) { sum += i * 2 - 1; sum ^= i & 7; }
    return sum;
}
const stencilA = [], stencilB = [];
for (let i = 0; i < 512; ++i) stencilA[i] = stencilB[i] = i / 512;
function stencil(a, b, n) {
    for (let pass = 0; pass < n; ++pass) {
        for (let k = 1; k < 511; ++k) {
            b[k] = (a[k - 1] + a[k] + a[k + 1]) * (1 / 3);
        }
        const temporary = a; a = b; b = temporary;
    }
    return a[256];
}
function indexedNumericArray(n) { return stencil(stencilA, stencilB, n); }
function fibonacci(n) { return n < 2 ? n : fibonacci(n - 1) + fibonacci(n - 2); }

function measure(name, fn, count, warmCount) {
    for (let i = 0; i < 200; ++i) fn(warmCount);
    const samples = [], checksums = [];
    for (let i = 0; i < 5; ++i) {
        const start = Date.now();
        const checksum = fn(count);
        samples.push(Date.now() - start);
        checksums.push(checksum);
    }
    print(JSON.stringify({name, count, samples_ms: samples, checksums}));
}

measure("array-copy-compare", arrayCopyCompare, 500000, 100);
measure("small-array", smallArray, 3000000, 100);
measure("typed-array-set", typedCopy, 500000, 100);
measure("exact-arguments", exactArguments, 300000, 100);
measure("surplus-arguments", surplusArguments, 300000, 100);
measure("computed-reads", computedReads, 2000000, 100);
measure("enumeration", enumeration, 100000, 100);
measure("absent-properties", absentProperties, 3000000, 100);
measure("arithmetic", arithmetic, 10000000, 100);
measure("indexed-numeric-array", indexedNumericArray, 100000, 2);
measure("fibonacci", fibonacci, 30, 10);
