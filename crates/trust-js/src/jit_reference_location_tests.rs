//! A global Reference's hint is three words, the shape, the slot and the slot's location word,
//! and the template's in-line ResolveNameRef copies all three from the resolution site's
//! descriptor. The site is the chunk's for the name, shared by every ResolveNameRef of it, each
//! with its own Reference: when a helper run of one of them refills the descriptor after the
//! name's slot moved, another one's in-line capture proves the new shape, and a location word
//! left from its own last helper run would name the old slot, another property's (a store to
//! `x` wrote `b`). Shared by the ARM64 and x86-64 templates' tests.
use super::TEST_NATIVE_REFERENCES;
use crate::{bytecode::Tier, Completion, Engine};

fn eval(tier: Tier, source: &str) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(32);
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}\n{source}"),
    }
}

/// `source` gives `expected` at every tier; with the template, its in-line ResolveNameRef ran
/// where `inline` says it must.
fn check(source: &str, expected: &str, inline: bool) {
    for tier in [Tier::Interp, Tier::Bytecode] {
        assert_eq!(eval(tier, source), expected, "{tier:?}\n{source}");
    }
    TEST_NATIVE_REFERENCES.with(|counts| counts.iter().for_each(|count| count.set(0)));
    assert_eq!(eval(Tier::Jit, source), expected, "Jit\n{source}");
    let captures = TEST_NATIVE_REFERENCES.with(|counts| counts[0].get());
    if inline {
        assert!(
            captures > 100,
            "the template's in-line ResolveNameRef ran ({captures})\n{source}"
        );
    }
}

#[test]
fn reference_location_store_after_the_name_is_deleted_and_added_again() {
    // `x` moves past `b`: the second assignment's helper refills the shared site, then the
    // first's in-line capture takes the new shape and slot.
    check(
        "globalThis.a=1;globalThis.x=2;globalThis.b=3;
         for(var i=0;i<2000;i++){x=i;if(i==1000){delete globalThis.x;globalThis.x=0;}x=i+0.5;}
         [a,x,b].join()",
        "1,1999.5,3",
        true,
    );
}

#[test]
fn reference_location_store_in_a_function_with_a_direct_eval() {
    check(
        "globalThis.a=1;globalThis.x=2;globalThis.b=3;
         function f(n){eval('');
           for(var i=0;i<n;i++){x=i;if(i==1000){delete globalThis.x;globalThis.x=0;}x=i+0.5;}}
         f(2000);[a,x,b].join()",
        "1,1999.5,3",
        true,
    );
}

#[test]
fn reference_location_load_reads_the_name_not_its_neighbour() {
    // The move comes first in the body, so the first assignment's helper refills the site and
    // the logical assignment's in-line capture reads through its own Reference: a LoadRef with
    // no store (`x` is truthy), so the neighbour's value would show in `seen` alone.
    check(
        "globalThis.a=1;globalThis.x=2;globalThis.b=3;var seen=0;
         for(var i=1;i<=2000;i++){if(i==1000){delete globalThis.x;globalThis.x=0;}x=i;seen+=(x||=99);}
         [a,x,b,seen].join()",
        "1,2000,3,2001000",
        true,
    );
}

#[test]
fn reference_location_store_after_an_earlier_property_is_deleted() {
    // Removing `p` closes the gap: `x` and `b` move down one slot.
    check(
        "globalThis.p=0;globalThis.x=2;globalThis.b=3;
         for(var i=0;i<2000;i++){x=i;if(i==1000){delete globalThis.p;}x=i+0.5;}
         [x,b,typeof p].join()",
        "1999.5,3,undefined",
        true,
    );
}

#[test]
fn reference_location_store_after_an_earlier_named_index_is_deleted() {
    // An accessor at an index is a named entry of the global object, in insertion order.
    check(
        "Object.defineProperty(globalThis,'7',{get(){return 7},configurable:true});
         globalThis.x=2;globalThis.b=3;
         for(var i=0;i<2000;i++){x=i;if(i==1000){delete globalThis[7];}x=i+0.5;}
         [x,b,globalThis[7]].join()",
        "1999.5,3,",
        true,
    );
}

#[test]
fn reference_location_store_through_a_with_object() {
    // A `with` object's names have no resolution site (its proof refuses a with scope): the
    // helper's References carry their three words; the results agree.
    check(
        "var o={p:0,x:2,b:3};
         with(o){for(var i=0;i<2000;i++){x=i;if(i==1000){delete o.p;}x=i+0.5;}}
         [o.x,o.b,i].join()",
        "1999.5,3,2000",
        false,
    );
}

#[test]
fn reference_location_store_to_a_direct_evals_var() {
    // A direct eval's `var` in global code is a configurable global property; in a function it
    // is a binding of the function's variables, deleted and declared again.
    check(
        "globalThis.p=0;eval('var x=2');globalThis.b=3;
         for(var i=0;i<2000;i++){x=i;if(i==1000){delete globalThis.p;}x=i+0.5;}
         [x,b].join()",
        "1999.5,3",
        true,
    );
    check(
        "function f(n){eval('var p=0,x=2,b=3');
           for(var i=0;i<n;i++){x=i;if(i==1000){delete p;delete x;eval('var x=0');}x=i+0.5;}
           return [x,b,typeof p].join();}
         f(2000)",
        "1999.5,3,undefined",
        false,
    );
}
