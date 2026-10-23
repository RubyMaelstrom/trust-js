//! A64 encodings of the instructions the inline Math intrinsics and Uint8Clamped stores use, against the Arm ARM
//! (100076_0100): FRINTM/FRINTP/FABS (C7.2), FMIN/FMAX (C7.2.51-52), CLZ (C6.2.57), MUL (MADD
//! with WZR, C6.2.194). The expected words are the assembler's output for the same operands.

use super::asm;
use crate::{bytecode::Tier, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("Math fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn assert_native_intrinsic(engine: &mut Engine, name: &str, intrinsic: u8) -> u32 {
    let env = engine.interp.global_env;
    let function = engine.interp.get_var(name, &env).ok().unwrap();
    let function_handle = function.as_obj().unwrap();
    let object = function_handle.borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} must be an author function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
    assert!(
        chunk.jit.get().flatten().is_some(),
        "{name} must run natively"
    );
    assert!(
        chunk.jit_runs.get() > 0,
        "{name} must enter the native body"
    );
    chunk
        .jit_ops()
        .iter()
        .find_map(|op| {
            let crate::bytecode::Op::CallWithThis(_, site) = op else {
                return None;
            };
            let entries =
                chunk.jit_call_cache_ptr(*site) as *const std::cell::Cell<crate::bytecode::CallIc>;
            (0..crate::bytecode::CALL_IC_WAYS).find_map(|way| {
                // The chunk retains this site's complete fixed-size cache array.
                let entry = unsafe { (*entries.add(way)).get() };
                (entry.callee != 0 && entry.intrinsic == intrinsic).then_some(entry.epoch)
            })
        })
        .unwrap_or_else(|| panic!("{name} must warm the exact builtin intrinsic"))
}

/// Math methods ignore their receiver (ECMA-262 e28783d5, #sec-math.abs,
/// #sec-math.min, #sec-math.max, #sec-math.sqrt). A callable may also be the
/// receiver, in both operand words at once. Owners carry no count: the callable survives a
/// collection after every run, intact.
#[test]
fn math_intrinsics_release_aliased_and_distinct_operand_owners() {
    use crate::bytecode::{
        INTRINSIC_IS_NAN, INTRINSIC_MATH_ABS, INTRINSIC_MATH_MAX, INTRINSIC_MATH_MIN,
        INTRINSIC_MATH_SQRT,
    };
    for (expression, arguments, result, intrinsic) in [
        ("Math.abs", "-9", 9, INTRINSIC_MATH_ABS),
        ("Math.min", "-9,81", -9, INTRINSIC_MATH_MIN),
        ("Math.max", "-9,81", 81, INTRINSIC_MATH_MAX),
        ("Math.sqrt", "81", 9, INTRINSIC_MATH_SQRT),
        ("isNaN", "NaN", 1, INTRINSIC_IS_NAN),
        ("Number.isNaN", "NaN", 1, INTRINSIC_IS_NAN),
    ] {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            evaluate(&mut engine, &format!(
                "var builtin={expression}; builtin.same=builtin;\n\
                 var receiver={{same:builtin}};\n\
                 function alias(n){{var total=0;for(var k=0;k<n;k++)total+=builtin.same({arguments});return total;}}\n\
                 function distinct(n){{var total=0;for(var k=0;k<n;k++)total+=receiver.same({arguments});return total;}}"
            ));
            for name in ["alias", "distinct"] {
                assert_eq!(
                    evaluate(&mut engine, &format!("{name}(8)")),
                    (result * 8).to_string()
                );
                if tier == Tier::Jit {
                    assert_native_intrinsic(&mut engine, name, intrinsic);
                }
            }
            for name in ["alias", "distinct", "alias", "distinct"] {
                assert_eq!(
                    evaluate(&mut engine, &format!("{name}(256)")),
                    (result * 256).to_string()
                );
                engine.interp.gc_collect();
                assert_eq!(
                    evaluate(
                        &mut engine,
                        &format!(
                            "builtin.same===builtin&&receiver.same===builtin&&\
                             +builtin.same({arguments})==={result}"
                        )
                    ),
                    "true",
                    "{tier:?} {expression} {name}"
                );
            }
        }
    }
}

#[test]
fn math_intrinsics_destroy_the_final_aliased_operand_owner() {
    use crate::bytecode::{INTRINSIC_IS_NAN, INTRINSIC_MATH_ABS, INTRINSIC_MATH_SQRT};
    for (expression, argument, expected, intrinsic) in [
        ("Math.abs", "-9", "9", INTRINSIC_MATH_ABS),
        ("Math.sqrt", "81", "9", INTRINSIC_MATH_SQRT),
        ("isNaN", "NaN", "true", INTRINSIC_IS_NAN),
        ("Number.isNaN", "NaN", "true", INTRINSIC_IS_NAN),
    ] {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            evaluate(&mut engine, &format!(
                "var builtin={expression},holder={{value:builtin}},sentinel={{}}; builtin.sentinel=sentinel;\n\
                 Object.defineProperty(builtin,'same',{{get:function(){{return this;}},configurable:true}});\n\
                 function invoke(clear){{return holder.value.same(clear?(delete holder.value,delete {expression},{argument}):{argument});}}"
            ));
            // The strong handles live only in this helper frame (the collector scans the stack).
            let (weak, sentinel_weak) = crate::gc_test::in_frame(|| {
                let env = engine.interp.global_env;
                let value = engine.interp.get_var("builtin", &env).ok().unwrap();
                let sentinel = engine.interp.get_var("sentinel", &env).ok().unwrap();
                (
                    crate::value::Gc::downgrade(&value.as_obj().unwrap()),
                    crate::value::Gc::downgrade(&sentinel.as_obj().unwrap()),
                )
            });
            evaluate(&mut engine, "builtin=undefined; sentinel=undefined");
            for _ in 0..8 {
                assert_eq!(evaluate(&mut engine, "invoke(false)"), expected);
            }
            if tier == Tier::Jit {
                assert_native_intrinsic(&mut engine, "invoke", intrinsic);
            }
            assert_eq!(evaluate(&mut engine, "invoke(true)"), expected);
            // The next collection finds the last owner dead and destroys it.
            engine.interp.gc_collect();
            assert_eq!(weak.strong_count(), 0, "{tier:?} {expression} final owner");
            // A bare decrement to zero would make the callable Weak appear
            // dead without running its destructor. Releasing its property is
            // independent evidence that the last-owner drop really ran.
            assert_eq!(
                sentinel_weak.strong_count(),
                0,
                "{tier:?} {expression} must destroy its owned property"
            );
        }
    }
}

fn word(build: impl FnOnce(&mut asm::Asm)) -> u32 {
    let mut a = asm::Asm::new();
    build(&mut a);
    let words = a.finish();
    assert_eq!(words.len(), 1);
    words[0]
}

#[test]
fn math_intrinsic_instruction_encodings() {
    assert_eq!(word(|a| a.frintm(3, 5)), 0x1e65_40a3); // frintm d3, d5
    assert_eq!(word(|a| a.frintp(3, 5)), 0x1e64_c0a3); // frintp d3, d5
    assert_eq!(word(|a| a.fabs(3, 5)), 0x1e60_c0a3); // fabs d3, d5
    assert_eq!(word(|a| a.fminmax(false, 3, 5, 7)), 0x1e67_58a3); // fmin d3, d5, d7
    assert_eq!(word(|a| a.fminmax(true, 3, 5, 7)), 0x1e67_48a3); // fmax d3, d5, d7
    assert_eq!(word(|a| a.clz_w(3, 5)), 0x5ac0_10a3); // clz w3, w5
    assert_eq!(word(|a| a.mul_w(3, 5, 7)), 0x1b07_7ca3); // mul w3, w5, w7
    assert_eq!(word(|a| a.fcvtns_w_d(3, 5)), 0x1e60_00a3); // fcvtns w3, d5 (Uint8Clamped stores)
}

#[test]
fn exact_math_and_isnan_intrinsics_accept_primitive_receivers_natively() {
    const CHILD: &str = "TRUST_JS_TEST_PRIMITIVE_MATH_INTRINSICS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Every call checks the process-global IC epoch. Other tests retiring code can
        // invalidate this warmed site mid-loop without changing its JavaScript results.
        // Isolate the exact native-hit proof; keep inherited GC-stress settings intact.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "jit::math_intrinsic_tests::exact_math_and_isnan_intrinsics_accept_primitive_receivers_natively",
                "--exact", "--nocapture", "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use crate::bytecode::{INTRINSIC_IS_NAN, INTRINSIC_MATH_ABS, INTRINSIC_MATH_SQRT};
    for (builtin, argument, result, intrinsic) in [
        ("Math.abs", "-9", 9, INTRINSIC_MATH_ABS),
        ("Math.sqrt", "81", 9, INTRINSIC_MATH_SQRT),
        ("isNaN", "NaN", 1, INTRINSIC_IS_NAN),
        ("Number.isNaN", "NaN", 1, INTRINSIC_IS_NAN),
    ] {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            evaluate(&mut engine, &format!(
                "String.prototype.operation={builtin};Number.prototype.operation={builtin};Boolean.prototype.operation={builtin};\n\
                 function stringReceiver(n){{var sum=0;for(var k=0;k<n;k++)sum+='text'.operation({argument});return sum;}}\n\
                 function numberReceiver(n){{var sum=0;for(var k=0;k<n;k++)sum+=(1).operation({argument});return sum;}}\n\
                 function booleanReceiver(n){{var sum=0;for(var k=0;k<n;k++)sum+=true.operation({argument});return sum;}}"));
            for name in ["stringReceiver", "numberReceiver", "booleanReceiver"] {
                assert_eq!(
                    evaluate(&mut engine, &format!("{name}(16)")),
                    (result * 16).to_string()
                );
                if tier == Tier::Jit {
                    assert_native_intrinsic(&mut engine, name, intrinsic);
                    // Establish that this exact one-iteration input can hit before testing
                    // the stale entry; otherwise an unrelated fallback could mask the guard.
                    super::TEST_FAST_MATH_INTRINSICS.with(|count| count.set(0));
                    assert_eq!(
                        evaluate(&mut engine, &format!("{name}(1)")),
                        result.to_string()
                    );
                    assert_eq!(
                        super::TEST_FAST_MATH_INTRINSICS.with(|count| count.get()),
                        1,
                        "{builtin} with {name} must be eligible before epoch invalidation"
                    );
                    // A stale epoch must take the checked path even for an exact Math
                    // builtin. Prove the miss keeps the result and does not count as a hit.
                    crate::bytecode::invalidate_call_caches();
                    super::TEST_FAST_MATH_INTRINSICS.with(|count| count.set(0));
                    assert_eq!(
                        evaluate(&mut engine, &format!("{name}(1)")),
                        result.to_string()
                    );
                    assert_eq!(
                        super::TEST_FAST_MATH_INTRINSICS.with(|count| count.get()),
                        0,
                        "{builtin} with {name} must reject the stale intrinsic entry"
                    );
                    assert_eq!(
                        evaluate(&mut engine, &format!("{name}(16)")),
                        (result * 16).to_string()
                    );
                }
                let cached_epoch = (tier == Tier::Jit)
                    .then(|| assert_native_intrinsic(&mut engine, name, intrinsic));
                let epoch_before =
                    crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
                super::TEST_FAST_MATH_INTRINSICS.with(|count| count.set(0));
                assert_eq!(
                    evaluate(&mut engine, &format!("{name}(32)")),
                    (result * 32).to_string()
                );
                if tier == Tier::Jit {
                    let hits = super::TEST_FAST_MATH_INTRINSICS.with(|count| count.get());
                    let epoch_after =
                        crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
                    assert!(
                        hits >= 32,
                        "{builtin} with {name} must commit through the native intrinsic: \
                         {hits} hits, cached epoch {cached_epoch:?}, process epoch \
                         {epoch_before}->{epoch_after} (including post-call task collection)"
                    );
                }
            }
        }
    }
}
