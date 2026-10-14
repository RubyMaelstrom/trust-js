// DataView call/access costs. Run the release engine shell with --tier=jit or --tier=bytecode.
// Five warmed samples and their checksums are emitted for every kernel; compare fresh processes
// in alternating baseline/candidate order on the same CPU. Times are milliseconds.
(function () {
    const count = 1000000;
    const local = new DataView(new ArrayBuffer(64));
    const floats = new DataView(new ArrayBuffer(64));
    const shared = new DataView(new SharedArrayBuffer(64));
    for (let k = 0; k < 16; k++) {
        local.setUint32(k * 4, 0x1020304 + k, true);
        shared.setUint32(k * 4, 0x1020304 + k, true);
    }
    for (let k = 0; k < 8; k++) floats.setFloat64(k * 8, k + 0.5, true);
    function readUint8(view, n) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += view.getUint8(k & 63);
        return sum;
    }
    function readUint32Little(view, n) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += view.getUint32((k & 15) * 4, true);
        return sum;
    }
    function readUint32Big(view, n) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += view.getUint32((k & 15) * 4);
        return sum;
    }
    function readFloat64(view, n) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += view.getFloat64((k & 7) * 8, true);
        return sum;
    }
    function writeUint32(view, n) {
        for (let k = 0; k < n; k++) view.setUint32((k & 15) * 4, k, true);
        return view.getUint32(((n - 1) & 15) * 4, true);
    }
    function writeFloat64(view, n) {
        for (let k = 0; k < n; k++) view.setFloat64((k & 7) * 8, k * 0.125);
        return view.getFloat64(((n - 1) & 7) * 8);
    }
    function numericControl(view, n) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += (k & 15) * 4;
        return sum;
    }
    const kernels = [
        ['getUint8', readUint8, local],
        ['getUint32_le', readUint32Little, local],
        ['getUint32_be', readUint32Big, local],
        ['getFloat64_le', readFloat64, floats],
        ['shared_getUint32_le', readUint32Little, shared],
        ['setUint32_le', writeUint32, local],
        ['setFloat64_be', writeFloat64, floats],
        ['shared_setUint32_le', writeUint32, shared],
        ['numeric_control', numericControl, null]
    ];
    for (const [name, kernel, view] of kernels) {
        for (let k = 0; k < 3; k++) kernel(view, 50000);
        const n = name === 'numeric_control' ? count * 32 : count;
        const samples = [], checksums = [];
        for (let sample = 0; sample < 5; sample++) {
            const start = Date.now();
            const checksum = kernel(view, n);
            samples.push(Date.now() - start);
            checksums.push(checksum);
        }
        console.log(JSON.stringify({ kernel: name, count: n, samples_ms: samples, checksums: checksums }));
    }
})();
