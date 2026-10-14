// Dense Array end-operation scaling. Run the release engine shell with an
// explicit tier; each JSON record contains all independent timed samples.
// Construction and warm-up happen outside the interval. Checksums/lengths
// validate each run, including relocations after front or tail space runs out.
var rounds = 32768;
var sizes = [256, 2048, 16384];
var samples = 5;

function list(size) {
    var array = [];
    for (var index = 0; index < size; index++) array.push(index);
    return array;
}

function shiftPush(array, count) {
    var sum = 0, size = array.length;
    for (var index = 0; index < count; index++) {
        sum += array.shift();
        array.push(size + index);
    }
    return sum;
}

function unshiftPop(array, count) {
    var sum = 0;
    for (var index = 0; index < count; index++) {
        array.unshift(-index - 1);
        sum += array.pop();
    }
    return sum;
}

function prependShift(array, count) {
    var sum = 0;
    for (var index = 0; index < count; index++) {
        array.unshift(index);
        sum += array.shift();
    }
    return sum;
}

function drain(array, count) {
    var sum = 0;
    for (var index = 0; index < count; index++) sum += array.shift();
    return sum;
}

var cases = [
    {name: 'shift-push', run: shiftPush},
    {name: 'unshift-pop', run: unshiftPop},
    {name: 'unshift-shift', run: prependShift},
    {name: 'drain', run: drain}
];

for (var ci = 0; ci < cases.length; ci++) {
    var test = cases[ci];
    for (var warm = 0; warm < 40; warm++) test.run(list(64), 64);
    for (var si = 0; si < sizes.length; si++) {
        var size = sizes[si], count = test.name === 'drain' ? size : rounds;
        var timings = [], checksums = [];
        for (var sample = 0; sample < samples; sample++) {
            var array = list(size);
            var start = Date.now();
            var checksum = test.run(array, count);
            var elapsed = Date.now() - start;
            var expected;
            if (test.name === 'unshift-pop') {
                var original = Math.min(size, count), rest = count - original;
                expected = original * (2 * size - original - 1) / 2 - rest * (rest + 1) / 2;
            } else {
                expected = count * (count - 1) / 2;
            }
            if (checksum !== expected || array.length !== (test.name === 'drain' ? 0 : size))
                throw new Error('invalid ' + test.name + '/' + size + ': ' + checksum);
            timings.push(elapsed);
            checksums.push(checksum);
        }
        print(JSON.stringify({workload: test.name, size: size, operations: count,
            milliseconds: timings, checksums: checksums}));
    }
}
print('array-end-operations complete');
