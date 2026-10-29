//! W3 item 3's step 5 (Decisions 20c and 20g): the native code a fresh frame enters reads and
//! writes the captured bindings of an activation its chunk's plan made at fixed slots, its
//! prologue copies the captured parameters and `this` into them, and a read of the function's own
//! captured lexical tests the empty word once in each basic block. ECMA-262 e28783d5:
//! FunctionDeclarationInstantiation (spec.html:14247), and the declarative Environment Record's
//! InitializeBinding, SetMutableBinding and GetBindingValue (10828, 10850, 10887). Expected
//! results are Node's (v24.22.0).
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 8] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            match engine
                .eval(source, false)
                .expect("fixed activation fixture parses")
            {
                Completion::Value(actual) => {
                    assert_eq!(actual, expected, "{tier:?}/{threshold}: {source}");
                }
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{threshold}: {name}: {message}\n{source}");
                }
            }
        }
    }
}

#[test]
fn fixed_activations_copy_parameters_this_and_functions_in_order() {
    check(
        r#"
        function params(a, b, c) { var read = () => a + ':' + b + ':' + c; a = a * 2; return read(); }
        function shadowed(a) { function a() { return 'fn'; } var read = () => typeof a; return read(); }
        function duplicate(a, a) { var read = () => a; return read(); }
        function receiver(x) { var read = () => this.n + x; return read(); }
        function missing(a, b) { var read = () => String(b); return read(); }
        function redeclared(a) { var a; var read = () => a; return read(); }
        function mapped(a, b) {
            var read = () => a + ':' + b;
            var before = arguments[0];
            arguments[0] = 9; b = 4;
            return before + ',' + read() + ',' + arguments[1];
        }
        function run() {
            var out = [];
            for (var k = 0; k < 40; k++) {
                out.push([params(k, 'b', k * 3), shadowed(k), duplicate(1, k),
                    receiver.call({n: k}, 1), missing(k), redeclared(k), mapped(k, 2)].join(','));
            }
            return out[0] + '|' + out[39];
        }
        run();
        "#,
        "0:b:0,function,0,1,undefined,0,0,9:4,4|78:b:117,function,39,40,undefined,39,39,9:4,4",
    );
}

#[test]
fn fixed_activations_copy_the_bound_receiver() {
    // OrdinaryCallBindThis (spec.html:13887): a sloppy callee's captured `this` is the coerced
    // receiver, a strict callee's the receiver itself.
    check(
        r#"
        function sloppy() { var read = () => typeof this + ':' + (this === globalThis); return read(); }
        function strict() { 'use strict'; var read = () => typeof this + ':' + String(this); return read(); }
        function run() {
            var out = [];
            for (var k = 0; k < 40; k++) {
                out.push([sloppy(), sloppy.call(k), strict(), strict.call(k), sloppy.call('s'),
                    strict.call(null)].join(','));
            }
            return out[0] + '|' + out[39];
        }
        run();
        "#,
        "object:true,object:false,undefined:undefined,number:0,object:false,object:null|\
         object:true,object:false,undefined:undefined,number:39,object:false,object:null",
    );
}

#[test]
fn fixed_activations_check_their_own_lexicals_once_a_block() {
    // A merge point starts a block: `own(false)` reaches the second read without the first.
    check(
        r#"
        function own(c) {
            var read = () => x;
            var r = '';
            try { if (c) { r += x; } r += x; } catch (e) { r += e.name; }
            let x = 1;
            r += x; r += x;
            return r + read();
        }
        function stores() {
            var read = () => y + ':' + z + ':' + w;
            var r = [];
            try { y = 5; } catch (e) { r.push(e.name); }
            try { z++; } catch (e) { r.push(e.name); }
            try { r.push(w); } catch (e) { r.push(e.name); }
            let y = 1; let z = 10; const w = 3;
            y = 7; z++; ++z; z--;
            r.push(read());
            return r.join(',');
        }
        function loop(n) {
            var read = () => x;
            var r = 0;
            for (var k = 0; k < n; k++) {
                if (k > 2) r += x; else { try { r += x; } catch (e) { r -= 100; } }
            }
            let x = 5;
            return r + ':' + read();
        }
        function fact(n) {
            var read = () => n;
            if (n <= 1) return 1;
            var r = n * fact(n - 1);
            n = r;
            return read();
        }
        function counter() { var n = 0; var inc = () => ++n; for (var k = 0; k < 50; k++) { n++; inc(); } return n; }
        function run() {
            var out = [];
            for (var k = 0; k < 40; k++) {
                out.push([own(k % 2 === 0), stores(), loop(2), fact(10), counter()].join('/'));
            }
            return out[0] + '|' + out[39];
        }
        run();
        "#,
        "ReferenceError111/ReferenceError,ReferenceError,ReferenceError,7:11:3/-200:5/3628800/100|\
         ReferenceError111/ReferenceError,ReferenceError,ReferenceError,7:11:3/-200:5/3628800/100",
    );
}

/// A fixed slot's store into an old activation marks the record's card: a minor collection then
/// keeps a young referent whose only edge is that slot.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn fixed_slot_stores_mark_an_old_activations_card() {
    use crate::gc_core::{CellAge, CollectionKind};
    use crate::value::{Callable, Value, ValueView};
    use std::cell::Cell;
    thread_local! {
        static FRESH: Cell<usize> = const { Cell::new(0) };
        static KEPT: Cell<bool> = const { Cell::new(false) };
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine.interp.gc_heap.agent.task_deferred.set(true);
    let global = engine.interp.global;
    engine.interp.def_method(&global, "promote", 0, |i, _, _| {
        i.gc_collect_precise_for_test(CollectionKind::Major);
        Ok(Value::Undefined)
    });
    engine.interp.def_method(&global, "fresh", 0, |i, _, _| {
        let object = Value::Obj(i.new_object());
        let cell = object.cell_word().unwrap() as usize;
        assert_eq!(
            i.gc_heap.core.borrow().cell_age(cell as u64),
            Some(CellAge::Young)
        );
        FRESH.with(|fresh| fresh.set(cell));
        Ok(object)
    });
    engine.interp.def_method(&global, "minor", 0, |i, _, _| {
        let cell = FRESH.with(Cell::get);
        let weak = crate::gc_handle::WeakRef::to(cell);
        assert_eq!(
            i.gc_heap.core.borrow().cell_age(cell as u64),
            Some(CellAge::Young),
            "no intervening promotion may mask the card"
        );
        i.gc_collect_precise_for_test(CollectionKind::Minor);
        let heap = i.gc_heap.clone();
        KEPT.with(|kept| {
            kept.set(
                weak.live_ptr() == Some(cell)
                    && heap.core.borrow().owns_cell(cell, heap.object_desc),
            )
        });
        Ok(Value::Undefined)
    });
    let setups = crate::bytecode::TEST_FRAME_SETUPS.with(Cell::get);
    let Completion::Value(result) = engine
        .eval(
            r#"
            function keep() {
                var held = null;
                var read = () => held;
                promote();
                held = fresh();
                minor();
                return read;
            }
            var read = keep();
            typeof read();
            "#,
            false,
        )
        .expect("fixture parses")
    else {
        panic!("fixture throws")
    };
    assert_eq!(result, "object");
    assert!(
        KEPT.with(Cell::get),
        "the young referent survived the minor collection"
    );
    assert!(
        crate::bytecode::TEST_FRAME_SETUPS.with(Cell::get) > setups,
        "keep ran in a fresh native frame"
    );
    // The store went to an old record, through a fixed slot of native code.
    let env = engine.interp.global_env;
    let Ok(keep) = engine.interp.get_var("keep", &env) else {
        panic!("keep is defined")
    };
    let ValueView::Obj(keep) = keep.view() else {
        panic!("keep is a function")
    };
    let keep = keep.borrow();
    let Callable::User(user) = &keep.call else {
        panic!("keep is a user function")
    };
    let chunk = user
        .func
        .runtime()
        .execution_code()
        .and_then(Option::as_ref)
        .expect("compiled keep")
        .clone();
    assert!(
        chunk.jit.get().is_some_and(|code| code.is_some()),
        "keep is native"
    );
    assert!(chunk.jit_fixed_activation());
    let Ok(read) = engine.interp.get_var("read", &env) else {
        panic!("read is defined")
    };
    let ValueView::Obj(read) = read.view() else {
        panic!("read is a function")
    };
    let read = read.borrow();
    let Callable::User(user) = &read.call else {
        panic!("read is a user function")
    };
    assert_eq!(
        engine
            .interp
            .gc_heap
            .core
            .borrow()
            .cell_age(crate::interpreter::Env::as_ptr(&user.env) as u64),
        Some(CellAge::Old)
    );
}
