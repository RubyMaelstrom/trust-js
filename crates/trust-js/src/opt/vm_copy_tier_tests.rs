//! Canonical-frame effects whose helper runs the VM on a copy of the operand stack
//! (`bytecode::jit_runs_on_vm_copy`; P3 stage 3 step 3a's follow-up), end to end: the helper copies
//! only the operation's operands, so that the words below them, which the tier's code does not
//! write, reach neither the operation nor the collector's precise trace of the copy.

use super::{eager, eval, jit_engine, run, tier_chunk, with_switches};
use crate::bytecode::Tier;
use crate::opt::TestSwitches;

/// Programs whose hot functions run such effects with values held in registers below their
/// operands: an array literal's spread and pushes, a call's and a construction's argument array, an
/// object rest, a tagged template's object and an array literal's hole, and a spread in a try region
/// (whose state's words the frame holds whole).
const VM_COPY_PROGRAMS: &[&str] = &[
    "function sp(n, a) { return n * 2 + [...a, n].length; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += sp(k, [k, 1, 2]); t",
    "function ca(n, g, a) { return n * 3 + g(...a, n); }\n\
     function sum(x, y, z, w) { return x + y + z + w; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += ca(k, sum, [1, k, 2]); t",
    "function nw(n, C, a) { return n * 5 + new C(...a, 1).v; }\n\
     function C(x, y, z) { this.v = x + y + z; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += nw(k, C, [k, 2]); t",
    "function rs(n, o) { var {a, ...rest} = o; return n * 3 + a + Object.keys(rest).length; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += rs(k, {a: k, b: 1, c: 2}); t",
    "function tt(n, tag) { return n * 7 + tag`a${n}b${n + 1}`; }\n\
     function tag(s, x, y) { return s.length + x - y; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += tt(k, tag); t",
    "function ho(n, a) { return n * 2 + [n, , ...a].length; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += ho(k, [k]); t",
    "function th(n, a) { var s = n * 2; try { s = s + [...a].length; } catch (e) { s = -1; } \
     return s; }\n\
     var t = 0; for (var k = 0; k < 3000; k++) t += th(k, k % 3 ? [k, 1] : null); t",
];

/// VM_COPY_PROGRAMS with the tier on, eager and forcing exits, each with the poison switch, which
/// stores the poison into the operand words below a canonical-frame effect's operands: the results
/// are the VM's, and the helper copies no poisoned word.
#[test]
fn vm_copy_effects_results_equal_the_vms_and_copy_no_word_their_code_did_not_write() {
    let configurations = [
        (
            "on",
            TestSwitches {
                on: true,
                poison: true,
                ..Default::default()
            },
        ),
        (
            "eager",
            TestSwitches {
                poison: true,
                ..eager()
            },
        ),
        (
            "forced exits",
            TestSwitches {
                on: true,
                eager: true,
                exit_every: 7,
                poison: true,
                ..Default::default()
            },
        ),
    ];
    let mut compiles = 0;
    for program in VM_COPY_PROGRAMS {
        let expected = run(Tier::Bytecode, program);
        assert!(!expected.starts_with("throw"), "{program}: {expected}");
        for (label, switches) in configurations {
            super::compiled();
            crate::opt::take_poison_faults();
            assert_eq!(
                with_switches(switches, program),
                expected,
                "{label}: {program}"
            );
            assert_eq!(
                crate::opt::take_poison_faults(),
                0,
                "{label}: the helper copied no poisoned word: {program}"
            );
            compiles += super::compiled().len();
        }
    }
    assert!(compiles > 0, "the tier compiled the programs' functions");
}

/// What a record's operand words below such an effect's operands hold: what an earlier activation
/// of the record left there, since the tier's code does not write them. A test hook fills every
/// pooled record's words above `f`'s slots with the word of an object the program dropped and a
/// precise major collection freed; `f`, compiled by the tier, then runs an array literal's spread
/// and a call's argument array with `n * 2` and a sum held in registers below their operands. The
/// helper's copy holds none of those words while the cell is free (before the fix it copied them,
/// and with TRUST_JS_GC_STRESS=1 collections ran with the freed cell's word in the copy's precise
/// trace; under the stress a new cell can take the address, whose word an operand then is), and the
/// result is the VM's.
#[test]
fn a_vm_copy_effect_copies_no_word_an_earlier_activation_left_in_its_record() {
    use crate::gc_core::CollectionKind;
    let program = "function f(n, a) { return n * 2 + [...a].length + g(...a, n); }\n\
         function g(x, y, z, w) { return x + y + z + w; }\n\
         var t = 0; for (var k = 0; k < 300; k++) t += f(k, [k, 1, 2]); t";
    let expected = run(Tier::Bytecode, &format!("{program}; f(5, [1, 2, 3])"));
    eager().set();
    let mut engine = jit_engine();
    eval(&mut engine, program);
    let chunk = tier_chunk(&mut engine, "f").expect("the tier compiled f");
    eval(&mut engine, "var dead = {d: [1, 2, 3]}; 0");
    let global_env = engine.interp.global_env;
    let word = engine
        .interp
        .get_var("dead", &global_env)
        .unwrap_or_else(|_| panic!("the program made dead"))
        .bits();
    eval(&mut engine, "dead = undefined; 0");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert_eq!(
        engine.interp.gc_heap.core.borrow().cell_age(word),
        None,
        "the dropped object's cell is freed"
    );
    crate::jit::fill_pooled_words_for_test(&engine.interp, chunk.jit_frame().1, word);
    crate::bytecode::TEST_COPY_SEEN.with(|seen| seen.set(0));
    crate::bytecode::TEST_COPY_WATCH.with(|watch| watch.set(Some(word)));
    let result = eval(&mut engine, "f(5, [1, 2, 3])");
    crate::bytecode::TEST_COPY_WATCH.with(|watch| watch.set(None));
    assert_eq!(result, expected);
    assert_eq!(
        crate::bytecode::TEST_COPY_SEEN.with(std::cell::Cell::get),
        0,
        "the helper copied no word of the freed cell"
    );
}

/// The poison switch's reader check itself: an operand whose word is the poison (a program can
/// make that Number) is copied, and the check reports it, under the parameter names `fn(pz0,pz1,
/// pz2)`, which a whole suite's run under the switch shows (and only this test's).
#[test]
fn the_poison_reader_check_reports_a_copied_operand_that_is_the_poison() {
    let program =
        "var p = new Float64Array(new BigUint64Array([0x8badf00ddeadbeefn]).buffer)[0];\n\
         function pu(pz0, pz1, pz2) { return pz0 * 2 + [...pz1, pz2].length; }\n\
         var t = 0; for (var k = 0; k < 300; k++) t += pu(k, [k], p); t";
    let expected = run(Tier::Bytecode, program);
    TestSwitches {
        poison: true,
        ..eager()
    }
    .set();
    crate::opt::take_poison_faults();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, program), expected);
    assert!(
        tier_chunk(&mut engine, "pu").is_some(),
        "the tier compiled pu"
    );
    assert!(
        crate::opt::take_poison_faults() > 0,
        "the check reports the copied operand"
    );
}
