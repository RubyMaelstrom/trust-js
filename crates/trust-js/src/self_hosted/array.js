// TRust JS self-hosted built-ins: Array.prototype iteration methods (ECMA-262 §23.1.3).
//
// This source is evaluated once per Realm in a private environment whose only bindings are
// the intrinsics defined in `self_hosted.rs`. It reaches no author-visible global, prototype
// or Function.prototype member except where the specification itself observes one (Get and
// HasProperty on the receiver, ArraySpeciesCreate's `constructor`/@@species reads). Each
// method follows its algorithm step for step; the intrinsics compile to operations (see
// `self_hosted.rs`). Every parameter a method reads is declared; installation gives each
// method its specified `length`, which counts only the required ones.
"use strict";
({
  // §23.1.3.15 Array.prototype.forEach ( callback [ , thisArg ] )
  forEach(callback, thisArg) {
    const obj = ToObject(this, "Array.prototype.forEach");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) ThrowTypeError("Array.prototype.forEach callback is not callable");
    for (let k = 0; k < length; k++) {
      if (k in obj) {
        const kValue = obj[k];
        Call(callback, thisArg, kValue, k, obj);
      }
    }
    return void 0;
  },

  // §23.1.3.21 Array.prototype.map ( callback [ , thisArg ] )
  map(callback, thisArg) {
    const obj = ToObject(this, "Array.prototype.map");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) ThrowTypeError("Array.prototype.map callback is not callable");
    const array = ArraySpeciesCreate(obj, length);
    for (let k = 0; k < length; k++) {
      if (k in obj) {
        const kValue = obj[k];
        const mappedValue = Call(callback, thisArg, kValue, k, obj);
        CreateDataPropertyOrThrow(array, k, mappedValue);
      }
    }
    return array;
  },

  // §23.1.3.8 Array.prototype.filter ( callback [ , thisArg ] )
  filter(callback, thisArg) {
    const obj = ToObject(this, "Array.prototype.filter");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) ThrowTypeError("Array.prototype.filter callback is not callable");
    const array = ArraySpeciesCreate(obj, 0);
    let to = 0;
    for (let k = 0; k < length; k++) {
      if (k in obj) {
        const kValue = obj[k];
        const selected = Call(callback, thisArg, kValue, k, obj);
        if (selected) {
          CreateDataPropertyOrThrow(array, to, kValue);
          to++;
        }
      }
    }
    return array;
  },

  // §23.1.3.29 Array.prototype.some ( callback [ , thisArg ] )
  some(callback, thisArg) {
    const obj = ToObject(this, "Array.prototype.some");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) ThrowTypeError("predicate is not callable");
    for (let k = 0; k < length; k++) {
      if (k in obj) {
        const kValue = obj[k];
        const testResult = Call(callback, thisArg, kValue, k, obj);
        if (testResult) return true;
      }
    }
    return false;
  },

  // §23.1.3.6 Array.prototype.every ( callback [ , thisArg ] )
  every(callback, thisArg) {
    const obj = ToObject(this, "Array.prototype.every");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) ThrowTypeError("predicate is not callable");
    for (let k = 0; k < length; k++) {
      if (k in obj) {
        const kValue = obj[k];
        const testResult = Call(callback, thisArg, kValue, k, obj);
        if (!testResult) return false;
      }
    }
    return true;
  },

  // §23.1.3.9 Array.prototype.find ( predicate [ , thisArg ] ) — FindViaPredicate ascending
  find(predicate, thisArg) {
    const obj = ToObject(this, "Array.prototype.find");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(predicate)) ThrowTypeError("predicate is not callable");
    for (let k = 0; k < length; k++) {
      const kValue = obj[k];
      const testResult = Call(predicate, thisArg, kValue, k, obj);
      if (testResult) return kValue;
    }
    return void 0;
  },

  // §23.1.3.10 Array.prototype.findIndex ( predicate [ , thisArg ] )
  findIndex(predicate, thisArg) {
    const obj = ToObject(this, "Array.prototype.findIndex");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(predicate)) ThrowTypeError("predicate is not callable");
    for (let k = 0; k < length; k++) {
      const kValue = obj[k];
      const testResult = Call(predicate, thisArg, kValue, k, obj);
      if (testResult) return k;
    }
    return -1;
  },

  // §23.1.3.11 Array.prototype.findLast ( predicate [ , thisArg ] ) — FindViaPredicate descending
  findLast(predicate, thisArg) {
    const obj = ToObject(this, "Array.prototype.findLast");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(predicate)) ThrowTypeError("predicate is not callable");
    for (let k = length - 1; k >= 0; k--) {
      const kValue = obj[k];
      const testResult = Call(predicate, thisArg, kValue, k, obj);
      if (testResult) return kValue;
    }
    return void 0;
  },

  // §23.1.3.12 Array.prototype.findLastIndex ( predicate [ , thisArg ] )
  findLastIndex(predicate, thisArg) {
    const obj = ToObject(this, "Array.prototype.findLastIndex");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(predicate)) ThrowTypeError("predicate is not callable");
    for (let k = length - 1; k >= 0; k--) {
      const kValue = obj[k];
      const testResult = Call(predicate, thisArg, kValue, k, obj);
      if (testResult) return k;
    }
    return -1;
  },

  // §23.1.3.24 Array.prototype.reduce ( callback [ , initialValue ] ). Whether initialValue
  // is present (not merely undefined) is observable: `arguments.length` compiles to the number
  // of actual arguments.
  reduce(callback, initialValue) {
    const obj = ToObject(this, "Array.prototype.reduce");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) ThrowTypeError("Array.prototype.reduce callback is not callable");
    const initialValuePresent = arguments.length > 1;
    if (length === 0 && !initialValuePresent) {
      ThrowTypeError("Reduce of empty array with no initial value");
    }
    let k = 0;
    let accumulator = void 0;
    if (initialValuePresent) {
      accumulator = initialValue;
    } else {
      let kPresent = false;
      while (!kPresent && k < length) {
        kPresent = k in obj;
        if (kPresent) accumulator = obj[k];
        k++;
      }
      if (!kPresent) ThrowTypeError("Reduce of empty array with no initial value");
    }
    for (; k < length; k++) {
      if (k in obj) {
        const kValue = obj[k];
        accumulator = Call(callback, void 0, accumulator, kValue, k, obj);
      }
    }
    return accumulator;
  },

  // §23.1.3.25 Array.prototype.reduceRight ( callback [ , initialValue ] )
  reduceRight(callback, initialValue) {
    const obj = ToObject(this, "Array.prototype.reduceRight");
    const length = LengthOfArrayLike(obj);
    if (!IsCallable(callback)) {
      ThrowTypeError("Array.prototype.reduceRight callback is not callable");
    }
    const initialValuePresent = arguments.length > 1;
    if (length === 0 && !initialValuePresent) {
      ThrowTypeError("Reduce of empty array with no initial value");
    }
    let k = length - 1;
    let accumulator = void 0;
    if (initialValuePresent) {
      accumulator = initialValue;
    } else {
      let kPresent = false;
      while (!kPresent && k >= 0) {
        kPresent = k in obj;
        if (kPresent) accumulator = obj[k];
        k--;
      }
      if (!kPresent) ThrowTypeError("Reduce of empty array with no initial value");
    }
    for (; k >= 0; k--) {
      if (k in obj) {
        const kValue = obj[k];
        accumulator = Call(callback, void 0, accumulator, kValue, k, obj);
      }
    }
    return accumulator;
  },
})
