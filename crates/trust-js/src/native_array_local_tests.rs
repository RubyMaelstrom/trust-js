//! ECMA-262 e28783d5 (local official snapshot, 2026-09-06): GetValue,
//! ToPropertyKey, OrdinaryGet, OrdinaryHasProperty and the `in` operator.
//! The local-source JIT region may skip operand materialization only for a
//! live ordinary own dense data element; every uncertain case replays the
//! original LoadLocal/LoadLocal/op sequence.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use crate::{bytecode::Tier, value::Value, Completion, Engine};

struct ForeignArrayWriteOnDrop {
    engine: Rc<RefCell<Engine>>,
    receiver: Value,
    drops: Rc<Cell<usize>>,
}

struct ForeignArrayDropState {
    engine: Rc<RefCell<Engine>>,
    drops: Rc<Cell<usize>>,
}

impl Drop for ForeignArrayWriteOnDrop {
    fn drop(&mut self) {
        let mut engine = self.engine.borrow_mut();
        let _agent = engine.interp.enter_agent();
        if engine
            .interp
            .member_set(&self.receiver, "1", Value::Num(99.0))
            .is_ok()
        {
            self.drops.set(self.drops.get() + 1);
        }
    }
}

fn install_foreign_array_dropper(
    engine: &mut Engine,
    foreign: Rc<RefCell<Engine>>,
    drops: Rc<Cell<usize>>,
) {
    engine.interp.op_state().put(ForeignArrayDropState {
        engine: foreign,
        drops,
    });
    engine.interp.def_method(
        &engine.interp.global,
        "makeForeignArrayDropper",
        1,
        make_foreign_array_dropper,
    );
}

fn make_foreign_array_dropper(
    interp: &mut crate::interpreter::Interp,
    _: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let state = interp
        .host::<ForeignArrayDropState>()
        .expect("foreign array dropper state");
    let payload = ForeignArrayWriteOnDrop {
        engine: state.engine.clone(),
        receiver: args.first().cloned().unwrap_or(Value::Undefined),
        drops: state.drops.clone(),
    };
    Ok(Value::Obj(interp.make_native_closure(
        "foreign-array-dropper",
        0,
        Rc::new(move |_, _, _| {
            let _keep_payload = &payload;
            Ok(Value::Undefined)
        }),
    )))
}

struct NativeTdzDropWitness(Rc<Cell<usize>>);

impl Drop for NativeTdzDropWitness {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("native-array fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    let tdz_drops = Rc::new(Cell::new(0usize));
    engine.interp.op_state().put(tdz_drops);
    engine.interp.def_method(
        &engine.interp.global,
        "collectNativeArrayLocalTest",
        0,
        |interp, _, _| {
            interp.gc_collect();
            Ok(crate::value::Value::Undefined)
        },
    );
    engine.interp.def_method(
        &engine.interp.global,
        "makeNativeTdzWitness",
        0,
        |interp, _, _| {
            let witness = NativeTdzDropWitness(
                interp
                    .host::<Rc<Cell<usize>>>()
                    .expect("TDZ witness counter")
                    .clone(),
            );
            Ok(crate::value::Value::Obj(interp.make_native_closure(
                "tdz-witness",
                0,
                Rc::new(move |_, _, _| {
                    let _keep_alive = &witness;
                    Ok(crate::value::Value::Undefined)
                }),
            )))
        },
    );
    engine.interp.def_method(
        &engine.interp.global,
        "nativeTdzDropCount",
        0,
        |interp, _, _| {
            Ok(crate::value::Value::Num(
                interp
                    .host::<Rc<Cell<usize>>>()
                    .expect("TDZ witness counter")
                    .get() as f64,
            ))
        },
    );
    engine
}

fn check_all_tiers(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
    }
}

#[test]
fn tdz_region_refreshes_tagged_home_before_reference_error() {
    let source = r#"
        function readUninitializedAfterTdz() {
            let index = 0, count = 0;
            while (index < 10) {
                let current = index === 9 ? current : { value: index };
                if (current && current && current && current) count++;
                index++;
            }
            return count;
        }
        let referenceError = false;
        try { readUninitializedAfterTdz(); }
        catch (error) { referenceError = error instanceof ReferenceError; }
        referenceError;
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        let before = super::test_tdz_effect_paths();
        assert_eq!(eval(&mut engine, source), "true", "{tier:?}");
        if tier == Tier::Jit {
            let after = super::test_tdz_effect_paths();
            assert!(
                after[0] > before[0],
                "no actual tagged Tdz Write while a numeric home is live: before={before:?}, after={after:?}"
            );
        }
    }
}

#[test]
fn tdz_tagged_write_releases_last_owner_before_collection() {
    let source = r#"
        var tdzValues = [makeNativeTdzWitness(), {}];
        var tdzWeak = new WeakRef(tdzValues[0]);
        function releaseAtNextTdz() {
            let index = 0, released = false;
            while (index < 2) {
                let current = tdzValues[index];
                if (index === 0) tdzValues[0] = null;
                if (current && current && current && current) current;
                if (index === 1) {
                    collectNativeArrayLocalTest();
                    released = tdzWeak.deref() === undefined && nativeTdzDropCount() === 1;
                }
                index++;
            }
            return released;
        }
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, source), "undefined", "{tier:?}");
        let before = super::test_tdz_effect_paths();
        assert_eq!(eval(&mut engine, "releaseAtNextTdz();"), "true", "{tier:?}");
        if tier == Tier::Jit {
            let after = super::test_tdz_effect_paths();
            assert!(
                after[0] > before[0],
                "no actual last-owner tagged Tdz Write while a numeric home is live: before={before:?}, after={after:?}"
            );
        }
    }
}

#[test]
fn local_local_getelem_and_in_compile_and_take_native_own_element_hits() {
    let mut engine = engine(Tier::Jit);
    let before = super::test_local_array_fusion_hits();
    assert_eq!(
        eval(
            &mut engine,
            r#"
            function localGetLoop() {
                // `values` is lexical, so the bytecode keeps the explicit
                // LoadLocal/LoadLocal/GetElem sequence. `index++` makes its
                // numeric home eligible for the DReg-key region path.
                let values = [1, 2, 3, 4];
                let index = 0, total = 0;
                while (index < 4) {
                    total += values[index];
                    index++;
                }
                return total;
            }
            function localInLoop() {
                let values = [1, 2, 3, 4];
                let index = 0, present = 0;
                while (index < 4) {
                    if (index in values) present++;
                    index++;
                }
                return present;
            }
            var result;
            for (var warm = 0; warm < 8; warm++) {
                result = localGetLoop() + ':' + localInLoop();
            }
            result;
        "#
        ),
        "10:4"
    );
    let after = super::test_local_array_fusion_hits();
    assert!(
        after[0] > before[0],
        "no successful 3-op GetElem hit with a numeric DReg key"
    );
    assert!(
        after[2] > before[2],
        "no successful 3-op In hit with a numeric DReg key"
    );
}

#[test]
fn self_hosted_filter_pair_hits_and_rereads_callback_mutation() {
    let source = r#"
        function filterPair(values) {
            let seen = '';
            let result = values.filter(function(value, index, receiver) {
                if (index === 0) receiver[1] = 99;
                seen += (seen ? ',' : '') + value;
                return true;
            });
            return seen + '|' + result.join(',');
        }
        let values = [11, 22, 33];
        let answer = '';
        for (let warm = 0; warm < 8; warm++) answer = filterPair(values);
        answer;
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        let before = super::test_tdz_array_pair_hits();
        assert_eq!(eval(&mut engine, source), "11,99,33|11,99,33", "{tier:?}");
        if tier == Tier::Jit {
            let after = super::test_tdz_array_pair_hits();
            assert!(
                after[0] > before[0],
                "self-hosted filter did not execute the 3-op DReg-key pair: before={before:?}, after={after:?}"
            );
        }
    }
}

#[test]
fn pair_misses_preserve_inherited_and_own_accessors() {
    check_all_tiers(
        r#"
        let values = [3, 4, 5];
        delete values[1];
        let inheritedCalls = 0, inheritedReceiver = false;
        let ownCalls = 0, ownReceiver = false;
        Object.defineProperty(Array.prototype, '1', {
            configurable: true,
            get() {
                inheritedCalls++;
                inheritedReceiver = this === values;
                return 7;
            }
        });
        Object.defineProperty(values, '2', {
            configurable: true,
            get() {
                ownCalls++;
                ownReceiver = this === values;
                return 8;
            }
        });
        let seen = '';
        try {
            values.filter(function(value) {
                seen += (seen ? ',' : '') + value;
                return false;
            });
        } finally {
            delete Array.prototype[1];
        }
        seen + ':' + inheritedCalls + ':' + inheritedReceiver + ':' +
            ownCalls + ':' + ownReceiver;
    "#,
        "3,7,8:1:true:1:true",
    );
}

#[test]
fn pair_proxy_and_coercible_key_fallback_preserve_has_get_conversion_order() {
    check_all_tiers(
        r#"
        let events = '';
        let target = [7];
        let receiver = new Proxy(target, {
            has(target, key) {
                if (key === '0') events += 'has:' + key + ';';
                return Reflect.has(target, key);
            },
            get(target, key, receiver) {
                if (key === '0') events += 'get:' + key + ';';
                return Reflect.get(target, key, receiver);
            }
        });
        let key = { [Symbol.toPrimitive](hint) {
            events += 'key:' + hint + ';';
            return '0';
        }};
        function readPair(receiver, key) {
            let index = 0, result = 0;
            while (index < 1) {
                if (key in receiver) {
                    let value = receiver[key];
                    let mapped = value;
                    result = mapped;
                }
                index++;
            }
            return result;
        }
        readPair(receiver, key) + ':' + events;
    "#,
        "7:key:string;has:0;key:string;get:0;",
    );
}

#[test]
fn pair_final_owner_release_falls_back_before_foreign_engine_mutation() {
    let mut engine = engine(Tier::Jit);
    let foreign = Rc::new(RefCell::new(Engine::new()));
    let drops = Rc::new(Cell::new(0usize));
    install_foreign_array_dropper(&mut engine, foreign, drops.clone());
    assert_eq!(
        eval(
            &mut engine,
            r#"
            let pairInput = [0, 22];
            pairInput[0] = makeForeignArrayDropper(pairInput);
            void 0;
            "#
        ),
        "undefined"
    );

    let before = super::test_tdz_array_pair_hits();
    assert_eq!(
        eval(
            &mut engine,
            r#"
            function runForeignDropPair(input) {
                let seen = '';
                let result = input.filter(function(value, index, receiver) {
                    if (index === 0) {
                        delete receiver[0];
                        return false;
                    }
                    seen += value;
                    return true;
                });
                return seen + '|' + result.join(',') + '|' + input[1];
            }
            runForeignDropPair(pairInput);
            "#
        ),
        "99|99|99"
    );
    let after = super::test_tdz_array_pair_hits();
    assert_eq!(drops.get(), 1, "the native capture destructor did not run");
    assert_eq!(
        after[0] - before[0],
        1,
        "index 0 should hit; index 1's last-owner TDZ must take the checked path: before={before:?}, after={after:?}"
    );
}

#[test]
fn pair_same_object_two_owners_falls_back_and_three_owners_admit() {
    for keep_third_owner in [false, true] {
        let mut engine = engine(Tier::Jit);
        let foreign = Rc::new(RefCell::new(Engine::new()));
        let drops = Rc::new(Cell::new(0usize));
        install_foreign_array_dropper(&mut engine, foreign, drops.clone());
        let setup = if keep_third_owner {
            r#"
                let pairKeep = pairInput[0];
            "#
        } else {
            ""
        };
        assert_eq!(
            eval(
                &mut engine,
                &format!(
                    r#"
                    let pairInput = [0, 22];
                    pairInput[0] = makeForeignArrayDropper(pairInput);
                    let pairSink = new Proxy({{}}, {{ defineProperty() {{ return true; }} }});
                    pairInput.constructor = {{
                        [Symbol.species]: function PairSink() {{ return pairSink; }}
                    }};
                    {setup}
                    void 0;
                    "#
                )
            ),
            "undefined",
            "keep_third_owner={keep_third_owner}"
        );
        let before = super::test_tdz_array_pair_hits();
        assert_eq!(
            eval(
                &mut engine,
                if keep_third_owner {
                    r#"
                    function runPairOwnerCase(input) {
                        let seen = '';
                        input.map(function(value, index, receiver) {
                            if (index === 0) {
                                delete receiver[0];
                                return value;
                            }
                            seen += value;
                            return value;
                        });
                        let beforeRelease = input[1];
                        pairKeep = null;
                        return seen + '|' + beforeRelease + '|' + input[1];
                    }
                    runPairOwnerCase(pairInput);
                    "#
                } else {
                    r#"
                    function runPairOwnerCase(input) {
                        let seen = '';
                        input.map(function(value, index, receiver) {
                            if (index === 0) {
                                delete receiver[0];
                                return value;
                            }
                            seen += value;
                            return value;
                        });
                        return seen + '|' + input[1];
                    }
                    runPairOwnerCase(pairInput);
                    "#
                }
            ),
            if keep_third_owner {
                "22|22|99"
            } else {
                "99|99"
            },
            "keep_third_owner={keep_third_owner}"
        );
        let after = super::test_tdz_array_pair_hits();
        assert_eq!(drops.get(), 1, "native capture destructor count");
        let expected_pair_hits = if keep_third_owner { 2 } else { 1 };
        assert_eq!(
            after[0] - before[0],
            expected_pair_hits,
            "same-object owner admission with keep_third_owner={keep_third_owner}: before={before:?}, after={after:?}"
        );
    }
}

#[test]
fn local_array_holes_inherited_getters_gc_and_live_prototype_mutation_fall_back() {
    check_all_tiers(
        r#"
        var backing = [2, 4, 8], getterCalls = 0, getterReceiver = false;
        function readLocalArray() {
            let values = backing;
            let index = 0, total = 0;
            while (index < 3) {
                total += values[index];
                index++;
            }
            return total;
        }
        var before = readLocalArray();
        delete backing[1];
        Object.defineProperty(Array.prototype, '1', {
            configurable: true,
            get() {
                getterCalls++;
                getterReceiver = this === backing;
                collectNativeArrayLocalTest();
                backing[2] = 9;
                return 5;
            }
        });
        var after;
        try {
            after = readLocalArray();
        } finally {
            delete Array.prototype[1];
        }
        before + ':' + after + ':' + getterCalls + ':' + getterReceiver;
    "#,
        "14:16:1:true",
    );
}

#[test]
fn local_array_in_sees_inherited_accessor_without_calling_it() {
    check_all_tiers(
        r#"
        var getterCalls = 0;
        Object.defineProperty(Array.prototype, '1', {
            configurable: true,
            get() { getterCalls++; return 9; }
        });
        function localHasHole() {
            let values = [];
            values[0] = 3;
            values[2] = 7;
            let index = 1, present = 0, attempts = 0;
            while (attempts < 1) {
                if (index in values) present++;
                attempts++;
            }
            return present;
        }
        var present;
        try {
            present = localHasHole();
        } finally {
            delete Array.prototype[1];
        }
        present + ':' + getterCalls;
    "#,
        "1:0",
    );
}

#[test]
fn local_proxy_reads_and_has_preserve_conversion_trap_order() {
    check_all_tiers(
        r#"
        var events = '';
        function proxyGetLoop() {
            let values = new Proxy([7], {
                get(target, key, receiver) {
                    events += 'get:' + key + ';';
                    return Reflect.get(target, key, receiver);
                }
            });
            let key = { [Symbol.toPrimitive](hint) {
                events += 'key:' + hint + ';';
                return 0;
            }};
            let i = 0, total = 0;
            while (i < 2) {
                total += values[key];
                i++;
            }
            return total;
        }
        let getValue = proxyGetLoop(), getEvents = events;
        events = '';
        function proxyHasLoop() {
            let values = new Proxy({}, {
                has(target, key) { events += 'has:' + key + ';'; return true; }
            });
            let key = { toString() { events += 'string;'; return 'x'; } };
            let i = 0, count = 0;
            while (i < 2) {
                if (key in values) count++;
                i++;
            }
            return count;
        }
        let hasValue = proxyHasLoop();
        getValue + ':' + getEvents + '|' + hasValue + ':' + events;
    "#,
        "14:key:string;get:0;key:string;get:0;|2:string;has:x;string;has:x;",
    );
}

#[test]
fn local_index_guards_preserve_negative_zero_fraction_nan_large_keys_and_symbols() {
    check_all_tiers(
        r#"
        let values = [];
        values[0] = 'zero';
        values['1.5'] = 'fraction';
        values.NaN = 'nan';
        values.Infinity = 'infinity';
        values['4294967295'] = 'large';
        values.x = 'coerced';
        let symbolKey = Symbol('element');
        values[symbolKey] = 'symbol';
        let conversions = 0;
        let objectKey = { [Symbol.toPrimitive](hint) {
            if (hint !== 'string') throw 'wrong hint';
            conversions++;
            return 'x';
        }};
        function convertedRead(rawKey) {
            let array = values, key = rawKey, n = 0, result;
            while (n < 1) {
                result = array[key];
                n++;
            }
            return result;
        }
        convertedRead(-0) + '|' + convertedRead(1.5) + '|' +
        convertedRead(NaN) + '|' + convertedRead(Infinity) + '|' +
        convertedRead(4294967295) + '|' + convertedRead(objectKey) + '|' +
        convertedRead(symbolKey) + ':' + conversions;
    "#,
        "zero|fraction|nan|infinity|large|coerced|symbol:1",
    );
}

#[test]
fn local_get_and_in_slow_paths_preserve_null_order_throw_identity_and_tdz() {
    check_all_tiers(
        r#"
        var events = '', sentinel = { id: 'sentinel' }, coercions = 0;
        let key = { [Symbol.toPrimitive]() { coercions++; events += 'key;'; return 'x'; } };
        function nullRead() {
            let receiver = null, localKey = key, n = 0;
            while (n < 1) { receiver[localKey]; n++; }
        }
        function nullIn() {
            let receiver = null, localKey = key, n = 0;
            while (n < 1) { localKey in receiver; n++; }
        }
        var nullReadName = '', nullInName = '';
        try { nullRead(); } catch (error) { nullReadName = error.name; }
        try { nullIn(); } catch (error) { nullInName = error.name; }
        var nullOrder = events + ':' + coercions;

        function throwingRead() {
            let receiver = [1];
            let localKey = { [Symbol.toPrimitive]() { throw sentinel; } };
            let n = 0;
            while (n < 1) { receiver[localKey]; n++; }
        }
        var sameThrown = false;
        try { throwingRead(); } catch (error) { sameThrown = error === sentinel; }

        function tdzRead() {
            let receiver = [1], total = 0;
            for (let n = 0; n < 1; n++) total += receiver[lateKey];
            let lateKey = 0;
            return total;
        }
        var tdzName = '';
        try { tdzRead(); } catch (error) { tdzName = error.name; }
        nullReadName + ':' + nullInName + ':' + nullOrder + ':' +
        sameThrown + ':' + tdzName;
    "#,
        "TypeError:TypeError::0:true:ReferenceError",
    );
}
