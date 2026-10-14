/*---
description: Nested third-Realm callbacks and collection preserve derived-constructor caller errors.
flags: [noStrict]
features: [class, cross-realm]
---*/

// ECMA-262 e28783d5, [[Construct]]: restore callerContext before validating a
// derived constructor's primitive return; preserve an existing throw completion.
var constructorRealm = $262.createRealm().global;
var callbackRealm = $262.createRealm().global;
var trace = [];
var outerErrors = 0;
var callbackErrors = 0;
var bounceCalls = 0;
var rounds = 32;

function assertRealm(condition, message) {
    if (!condition) throw new Error('nested Realm regression: ' + message);
}
function mark(message) { trace.push(message); }
function callerBounce() {
    'use strict';
    mark('A:bounce');
    assertRealm(new.target === undefined, 'callback must be an ordinary call');
    $262.gc();
    assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'A intrinsics after callback GC');
    bounceCalls++;
    return [new TypeError('A marker')];
}
class InnerCtor extends Object {
    constructor() {
        super();
        mark('A:inner');
        assertRealm(new.target === InnerCtor, 'inner new.target');
        $262.gc();
        assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'A intrinsics after inner GC');
        return null;
    }
}

constructorRealm.mainRealm = globalThis;
constructorRealm.callbackRealm = callbackRealm;
callbackRealm.mainRealm = globalThis;
callbackRealm.constructorRealm = constructorRealm;
callbackRealm.eval(`
    globalThis.nestedErrors = 0;
    globalThis.callbackC = function callbackC(inner, bounce, mode) {
        mainRealm.mark('C:enter');
        mainRealm.assertRealm(this === globalThis && new.target === undefined, 'C ordinary call binding');
        $262.gc();
        mainRealm.assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'C intrinsics after entry GC');
        var caught;
        try { new inner(); } catch (error) { caught = error; }
        mainRealm.assertRealm(caught && caught.constructor === TypeError, 'inner invalid return creates C TypeError');
        mainRealm.assertRealm(!(caught instanceof mainRealm.TypeError), 'inner error is not A TypeError');
        mainRealm.assertRealm(!(caught instanceof constructorRealm.TypeError), 'inner error is not B TypeError');
        nestedErrors++;
        mainRealm.mark('C:inner-error');
        var fromA = bounce();
        mainRealm.assertRealm(Object.getPrototypeOf(fromA) === mainRealm.Array.prototype, 'A callback array Realm');
        mainRealm.assertRealm(fromA[0].constructor === mainRealm.TypeError, 'A callback error Realm');
        mainRealm.assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'C restored after A callback');
        mainRealm.mark('C:exit');
        if (mode) throw new RangeError('C throw');
        return [new TypeError('C marker')];
    };
`);
constructorRealm.eval(`
    globalThis.OuterCtor = class OuterCtor extends Object {
        constructor(callback, inner, bounce, mode) {
            super();
            mainRealm.mark('B:enter');
            mainRealm.assertRealm(new.target === OuterCtor, 'outer new.target');
            var fromC = callback(inner, bounce, mode);
            mainRealm.assertRealm(Object.getPrototypeOf(fromC) === callbackRealm.Array.prototype, 'C callback array Realm');
            mainRealm.assertRealm(fromC[0].constructor === callbackRealm.TypeError, 'C callback error Realm');
            mainRealm.assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'B restored after C callback');
            mainRealm.mark('B:after');
            return null;
        }
    };
`);

assertRealm(TypeError !== constructorRealm.TypeError && TypeError !== callbackRealm.TypeError,
    'three distinct intrinsic identities');
assertRealm(constructorRealm.TypeError !== callbackRealm.TypeError, 'B and C distinct intrinsics');

function drive(mode) {
    'use strict';
    trace = ['A:start'];
    var caught;
    try {
        new constructorRealm.OuterCtor(callbackRealm.callbackC, InnerCtor, callerBounce, mode);
    } catch (error) { caught = error; }
    if (mode) {
        assertRealm(caught && caught.constructor === callbackRealm.RangeError, 'C throw completion retains its Realm');
        assertRealm(!(caught instanceof TypeError), 'callback error was not replaced by outer return validation');
        callbackErrors++;
        mark('A:callback-error');
    } else {
        assertRealm(caught && caught.constructor === TypeError, 'outer invalid return creates A TypeError');
        assertRealm(!(caught instanceof constructorRealm.TypeError), 'outer error is not B TypeError');
        assertRealm(!(caught instanceof callbackRealm.TypeError), 'nested caller Realm does not leak into outer error');
        outerErrors++;
        mark('A:outer-error');
    }
    $262.gc();
    assertRealm(new.target === undefined, 'driver remains an ordinary call');
    assertRealm(Number(3) === 3, 'native ordinary call after constructor completion');
    assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'A array intrinsics restored');
    assertRealm((new TypeError('after')).constructor === TypeError, 'A error intrinsics restored');
    var fresh = new Error('after completion');
    if (typeof fresh.stack === 'string') {
        assertRealm(fresh.stack.indexOf('at new OuterCtor') === -1, 'no stranded outer constructor frame');
        assertRealm(fresh.stack.indexOf('at new InnerCtor') === -1, 'no stranded inner constructor frame');
        assertRealm(fresh.stack.indexOf('at callbackC') === -1, 'no stranded third-Realm callback frame');
    }
    mark('A:restored');
    var expected = 'A:start,B:enter,C:enter,A:inner,C:inner-error,A:bounce,C:exit,' +
        (mode ? 'A:callback-error,' : 'B:after,A:outer-error,') + 'A:restored';
    assertRealm(trace.join(',') === expected, 'exact side-effect trace: ' + trace.join(','));
}
for (var iteration = 0; iteration < rounds; iteration++) {
    drive(0);
    drive(1);
}
assertRealm(outerErrors === rounds && callbackErrors === rounds, 'both completion paths repeat exactly once');
assertRealm(callbackRealm.nestedErrors === rounds * 2 && bounceCalls === rounds * 2,
    'nested constructor/callback effects repeat exactly once');
class GoodCtor {
    constructor(value) { this.value = value; }
}
var completed = new GoodCtor(7);
assertRealm(completed.value === 7 && Object.getPrototypeOf(completed) === GoodCtor.prototype,
    'successful constructor after repeated abrupt completions');
assertRealm(Object.getPrototypeOf([]) === Array.prototype, 'A remains active after all rounds');
var result = {status: 'ok', rounds: rounds, outerErrors: outerErrors,
    callbackErrors: callbackErrors, nestedErrors: callbackRealm.nestedErrors, bounceCalls: bounceCalls};
if (typeof console !== 'undefined' && typeof console.log === 'function') {
    console.log(JSON.stringify(result));
} else if (typeof print === 'function') {
    print(JSON.stringify(result));
}
