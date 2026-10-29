//! Actual x64 machine-entry proofs; run under native x64 or the isolated QEMU runner.
//! ECMA-262 e28783d5 GetIdentifierReference/GetValue/PutValue and ToNumeric.
use super::*;
use crate::interpreter::{new_binding_layout_id, new_scope, Binding, Env};
use crate::value::Value;
use crate::value::ValueView;
use crate::{bytecode::Tier, Completion, Engine};
use std::rc::Rc;

fn compile(engine: &mut Engine, source: &str) -> (Rc<Chunk>, JitCode) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("source parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).expect("bytecode");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(valid(&layout), "validated native name layout");
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = crate::jit::compile(&chunk, &layout, &ilayout).expect("native code");
    (chunk, code)
}

fn scope(
    parent: Option<Env>,
    key: &str,
    value: f64,
    layout: impl Into<crate::interpreter::TestLayout>,
) -> Env {
    let scope = new_scope(parent);
    scope
        .borrow_mut()
        .vars
        .insert(key, Binding::data(Value::Num(value), true, true));
    scope.borrow_mut().vars.publish_layout(layout);
    scope
}

fn run(engine: &mut Engine, chunk: &Rc<Chunk>, code: &JitCode, env: &Env) -> Value {
    crate::jit::run(&mut engine.interp, chunk, code, env, Value::Undefined, &[])
        .unwrap_or_else(|_| panic!("native fixture throws"))
}

fn number(value: Value) -> f64 {
    match (value).view() {
        ValueView::Num(number) => number,
        _ => panic!("expected Number"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn native_deep_names_use_fresh_holder_layouts_and_live_flags() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
    let layouts = [
        new_binding_layout_id(),
        new_binding_layout_id(),
        new_binding_layout_id(),
    ];
    let holder = scope(None, "outer", 11.0, layouts[0]);
    let child = scope(
        Some(scope(Some(holder), "middle", 0.0, layouts[1])),
        "local",
        0.0,
        layouts[2],
    );
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 11.0);
    let before = TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 11.0);
    let other = scope(None, "outer", 29.0, layouts[0]);
    let fresh = scope(
        Some(scope(Some(other), "middle", 0.0, layouts[1])),
        "local",
        0.0,
        layouts[2],
    );
    assert_eq!(number(run(&mut engine, &chunk, &code, &fresh)), 29.0);
    assert_eq!(
        TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get),
        before + 2
    );
    let operations = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
    let tdz_outer = other
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .enter_tdz_for_test();
    assert!(crate::jit::run(
        &mut engine.interp,
        &chunk,
        &code,
        &fresh,
        Value::Undefined,
        &[]
    )
    .is_err());
    assert_eq!(
        TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get),
        operations,
        "TDZ is not a fast read"
    );
    other
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .initialize(tdz_outer);
    // A layout identity (a shared descriptor) proves the key set whatever the generation; a
    // record with its own names proves it only by identity and generation.
    other.borrow_mut().vars.set_generation_for_test(u32::MAX);
    assert_eq!(number(run(&mut engine, &chunk, &code, &fresh)), 29.0);
    let operations = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
    other
        .borrow_mut()
        .vars
        .set_attributes_for_test("outer", None, None, Some(true));
    assert_eq!(other.borrow().vars.layout_id(), 0);
    other.borrow_mut().vars.set_generation_for_test(u32::MAX);
    assert_eq!(number(run(&mut engine, &chunk, &code, &fresh)), 29.0);
    assert_eq!(
        TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get),
        operations,
        "exhausted generation must not publish a raw address proof"
    );
}

/// A fresh holder's binding is found from its published layout in place and spilled: an
/// environment cell keeps its first binding, and more spill (here the name is the third).
#[test]
fn native_deep_names_find_fresh_holder_bindings_in_place_and_spilled() {
    for before in [0usize, 2] {
        let mut engine = Engine::new();
        let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
        let layouts = [
            new_binding_layout_id(),
            new_binding_layout_id(),
            new_binding_layout_id(),
        ];
        let chain = |value: f64| {
            let holder = new_scope(None);
            {
                let mut holder_scope = holder.borrow_mut();
                let vars = &mut holder_scope.vars;
                for k in 0..before {
                    vars.insert(format!("b{k}"), Binding::data(Value::Num(0.0), true, true));
                }
                vars.insert("outer", Binding::data(Value::Num(value), true, true));
                vars.publish_layout(layouts[0]);
            }
            scope(
                Some(scope(Some(holder), "middle", 0.0, layouts[1])),
                "local",
                0.0,
                layouts[2],
            )
        };
        let child = chain(11.0);
        assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 11.0);
        // Resolved targets, and reads that pushed their value natively.
        let counts = || {
            (
                TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get),
                TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get),
            )
        };
        let (targets, reads) = counts();
        assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 11.0);
        let fresh = chain(29.0);
        assert_eq!(number(run(&mut engine, &chunk, &code, &fresh)), 29.0);
        assert_eq!(
            counts(),
            (targets + 2, reads + 2),
            "{} bindings in the holder: the exact and the fresh holder are read natively",
            before + 1
        );
    }
}

#[test]
fn native_deep_globals_load_store_and_all_numeric_update_kinds() {
    let mut engine = Engine::new();
    engine.eval("var outer=0;", false).unwrap();
    let (chunk, code) = compile(&mut engine,
        "function f(){outer=10;let a=outer++;let b=++outer;let c=outer--;let d=--outer;outer++;outer--;return a+b+c+d+outer;}");
    let child = new_scope(Some(new_scope(Some(engine.interp.global_env))));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 54.0);
    let before = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 54.0);
    assert!(
        TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get) >= before + 7,
        "numeric update/load operations must enter native name templates"
    );
    check("var value=-0;function f(){let a=value++;value=Infinity;let b=--value;value=NaN;value++;return [1/a,b,Number.isNaN(value)].join('|')}f()",
        "-Infinity|Infinity|true");
}

#[test]
fn native_name_loads_and_stores_all_execution_word_kinds() {
    check(
        r#"
        var shared, values=[undefined,null,false,true,-0,NaN,Infinity,-Infinity,42.25,
            'text',Symbol('same'),{identity:7},12345678901234567890n];
        function global(value){shared=value;return shared;}
        function make(){let held;return function(value){held=value;return held;};}
        var capture=make(),okay=true;
        for(let j=0;j<100;j++)for(let value of values){
            if(!Object.is(global(value),value)||!Object.is(capture(value),value))okay=false;
        }
        for(let j=0;j<100;j++){global({last:j});capture({last:j});}
        [okay,global('last'),capture('last')].join('|');
    "#,
        "true|last|last",
    );
}

#[test]
fn native_name_store_rejects_internal_words_before_target_stack_or_card_commit() {
    use crate::gc_core::{CARD_SHIFT, CARD_TABLE_OFFSET, PAGE_SIZE};
    let _engine = Engine::new();
    let owner = new_scope(None);
    let owner_word = Env::as_ptr(&owner) as usize;
    let card = ((owner_word & !(PAGE_SIZE - 1))
        + CARD_TABLE_OFFSET
        + ((owner_word & (PAGE_SIZE - 1)) >> CARD_SHIFT)) as *mut u8;
    let mut a = Asm::new();
    let slow = a.label();
    let done = a.label();
    a.bytes(&[0x41, 0x55]); // push r13: the engine operand top is callee-saved.
    #[cfg(target_os = "windows")]
    {
        a.nr_mov(SP, R8); // top, before DX is used by the guard
        a.nr_mov(R9, DX); // actual owner
    }
    #[cfg(not(target_os = "windows"))]
    {
        a.nr_mov(SP, DX); // top
        a.nr_mov(R9, 6); // rsi = actual owner
        a.nr_mov(CX, 7); // rdi = target word address
    }
    store_value(&mut a, slow);
    a.jmp(done);
    a.bind(slow);
    a.bind(done);
    a.nr_mov(AX, SP);
    a.bytes(&[0x41, 0x5d, 0xc3]); // pop r13; ret
    let code = crate::jit::ExecutableBuffer::from_bytes(&a.finish()).expect("store guard probe");
    let probe: unsafe extern "C" fn(*mut u64, usize, *mut u64) -> usize =
        unsafe { std::mem::transmute(code.as_ptr()) };
    for (incoming, accepted) in [
        (7.0f64.to_bits(), true),
        (crate::value::PACK_EMPTY, false),
        (crate::value::PACK_LAZY_PROTO | 0x5a5a, false),
        (0xffff_0000_0000_0000, false),
    ] {
        // Raw byte/word buffers deliberately avoid constructing an invalid Value.
        let mut target = [11.0f64.to_bits(), 0x55aa_55aa_55aa_55aa];
        let mut operands = [0x1122_3344_5566_7788, incoming, 0x8877_6655_4433_2211];
        let original_target = target;
        let original_operands = operands;
        let top = unsafe { operands.as_mut_ptr().add(2) };
        // SAFETY: the owner is a live, small carded Env; no safepoint or edge mutation occurs.
        unsafe { card.write(0) };
        let returned_top = unsafe { probe(target.as_mut_ptr(), owner_word, top) };
        assert_eq!(operands, original_operands);
        assert_eq!(target[1], original_target[1]);
        if accepted {
            assert_eq!(target[0], incoming);
            assert_eq!(returned_top, unsafe { top.sub(1) } as usize);
            assert_eq!(
                unsafe { card.read() },
                1,
                "valid store reaches the actual owner card"
            );
        } else {
            assert_eq!(
                target, original_target,
                "guard before target commit: {incoming:#x}"
            );
            assert_eq!(returned_top, top as usize, "guard before operand pop");
            assert_eq!(unsafe { card.read() }, 0, "guard before card commit");
        }
    }
}

#[test]
fn native_name_import_mutability_and_parent_changes_stay_live() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
    let holder = scope(None, "outer", 1.0, new_binding_layout_id());
    let middle = new_scope(Some(holder));
    let child = new_scope(Some(middle));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 1.0);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 1.0);
    let exporter = scope(None, "source", 23.0, new_binding_layout_id());
    holder
        .borrow_mut()
        .vars
        .set_import_for_test("outer", Some((exporter, crate::atom::atomize("source"))));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 23.0);
    exporter
        .borrow_mut()
        .vars
        .get_mut("source")
        .unwrap()
        .set_value(Value::Num(24.0));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 24.0);
    let replacement = scope(None, "outer", 31.0, new_binding_layout_id());
    middle.borrow_mut().parent = Some(replacement);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 31.0);
    check(
        "const x=1;function f(){x=2;}var output;try{f()}catch(e){output=e.name;}output",
        "TypeError",
    );
}

#[test]
fn native_references_hot_fragment_reuses_capture_load_and_store() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(32);
    TEST_NATIVE_REFERENCES.with(|counts| counts.iter().for_each(|count| count.set(0)));
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    crate::bytecode::TEST_REFERENCE_HELPERS.with(|count| count.set(0));
    assert!(matches!(engine.eval(
        "var total=0;for(var index=0;index<2000;index++)total+=index;total;", false),
        Ok(Completion::Value(value)) if value=="1999000"));
    assert!(crate::jit::TEST_OSR_ENTRIES.with(std::cell::Cell::get) > 0);
    TEST_NATIVE_REFERENCES.with(|counts| {
        for count in counts {
            assert!(
                count.get() > 100,
                "actual native capture/load/store required"
            );
        }
    });
    assert!(
        crate::bytecode::TEST_REFERENCE_HELPERS.with(std::cell::Cell::get) < 16,
        "warm canonical references must not repeat slow-path walks"
    );
}

#[test]
fn native_references_preserve_rhs_coercion_tdz_and_saved_base() {
    check(
        r#"
        var x='outer', trace=[], box={x:1};
        function deleted(){return eval('var x=1;x+=(delete x,4);x;');}
        var first=deleted();
        with(box){x+=(delete box.x,Object.defineProperty(box,Symbol.unscopables,{value:{x:true}}),4);}
        try { t=(trace.push('rhs'),1);let t; } catch(e){trace.push(e.name);}
        const c=1;try{c=(trace.push('const'),2);}catch(e){trace.push(e.name);}
        var coercing={x:{valueOf(){delete coercing.x;return 8;}}};
        with(coercing){x++;}
        [first,box.x,coercing.x,x,trace.join(',')].join('|');
    "#,
        "5|5|9|outer|rhs,ReferenceError,const,TypeError",
    );
}

#[test]
fn native_references_keep_heap_owners_across_collection_and_hint_misses() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        engine.interp.def_method(
            &engine.interp.global,
            "collectReferenceTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        let source = r#"
            var values=[undefined,null,true,1,-0,NaN,'a',Symbol(),{},1n],held,count=0;
            for(var j=0;j<2000;j++){
                held=values[j%values.length];count++;
                if(j%200===0)collectReferenceTest();
            }
            [count,typeof held].join('|');
        "#;
        assert!(
            matches!(engine.eval(source,false),Ok(Completion::Value(value)) if value=="2000|bigint"),
            "{tier:?}"
        );
    }
}

thread_local! {
    /// The native name-op and native StoreRef counts at each `collect()` of the barrier test.
    static AT_COLLECTS: std::cell::RefCell<Vec<(usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// S6 step 10b: `store_value`'s write barrier (the store's owner's card, its address taken
/// before `decode_rhs` overwrites R9 and the card set after the store). Each scenario warms a
/// store site, collects inside its loop (`k === 50`), making the holder old, and has the same
/// site store values allocated since; the next collection's barrier check (debug builds)
/// passes, and the counts show the stores after the first collection ran in native code: a
/// global property and wide bindings from free-name caches (direct, enclosing and deeper
/// environments), a captured binding (`StoreCap`), and a Reference's global property and
/// binding (`StoreRef`: a script's stores to a global variable and to a global lexical binding
/// an earlier script declared).
#[test]
fn native_name_stores_record_their_owners_in_cards() {
    let scenarios = [
        (
            "global",
            false,
            "var g=null;function setg(v){g=v;}\
             for(var k=0;k<60;k++){if(k===50)collect();setg({k});}collect();g.k",
        ),
        (
            "enclosing",
            false,
            "function outer(){let x=null;function set(v){x=v;}return [set,()=>x];}var p=outer();\
             for(var k=0;k<60;k++){if(k===50)collect();p[0]({k});}collect();p[1]().k",
        ),
        (
            "parent",
            false,
            "function outer(){let x=null;function set(v){let s=v;const keep=()=>s;x=s;}\
             return [set,()=>x];}var p=outer();\
             for(var k=0;k<60;k++){if(k===50)collect();p[0]({k});}collect();p[1]().k",
        ),
        (
            "deep",
            false,
            "function outer(){let x=null;function middle(){let m=0;\
             return function(v){let s=v;const keep=()=>s+m;x=s;};}return [middle(),()=>x];}\
             var p=outer();for(var k=0;k<60;k++){if(k===50)collect();p[0]({k});}\
             collect();p[1]().k",
        ),
        (
            "captured",
            false,
            "function own(){let x=null;const get=()=>x;\
             for(let k=0;k<60;k++){if(k===50)collect();x={k};}collect();return get().k;}own()",
        ),
        (
            "reference property",
            true,
            "var r=null;for(var k=0;k<60;k++){if(k===50)collect();r={k};}collect();r.k",
        ),
        (
            // A script's store to a global lexical binding an earlier script declared.
            "reference binding",
            true,
            "for(var k=0;k<60;k++){if(k===50)collect();h={k};}collect();h.k",
        ),
    ];
    for (name, reference, source) in scenarios {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine
            .interp
            .def_method(&engine.interp.global, "collect", 0, |interp, _, _| {
                interp.gc_collect();
                let counts = (
                    TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get),
                    TEST_NATIVE_REFERENCES.with(|counts| counts[2].get()),
                );
                AT_COLLECTS.with(|at| at.borrow_mut().push(counts));
                Ok(Value::Undefined)
            });
        engine.eval("let h=null;", false).expect("prelude parses");
        AT_COLLECTS.with(|at| at.borrow_mut().clear());
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => assert_eq!(value, "59", "{name}"),
            Completion::Throw {
                name: error,
                message,
            } => panic!("{name}: {error}: {message}"),
        }
        let at = AT_COLLECTS.with(|at| at.borrow().clone());
        assert_eq!(at.len(), 2, "{name}: two collections");
        let after = if reference {
            at[1].1 - at[0].1
        } else {
            at[1].0 - at[0].0
        };
        assert!(
            after >= 10,
            "{name}: {after} native stores after the holder became old"
        );
    }
}

#[test]
fn native_name_overwrites_do_not_read_displaced_bigints_or_deferred_words() {
    let mut engine = Engine::new();
    engine
        .eval("var outer=0;function source(){};", false)
        .unwrap();
    let (chunk, code) = compile(&mut engine, "function write(){outer=7;return 0;}");
    let env = new_scope(Some(engine.interp.global_env));
    run(&mut engine, &chunk, &code, &env);
    let global = engine.interp.global;
    for deferred in [false, true] {
        if deferred {
            let globals = engine.interp.global_env;
            let source = engine.interp.get_var("source", &globals).ok().unwrap();
            let owner = source.as_obj().unwrap();
            let mut template = crate::value::Props::new();
            template.insert(
                "constructor",
                crate::value::Property::builtin(Value::Undefined),
            );
            // Install a property-only word directly to prove that a numeric PutValue never
            // observes/materializes the value it displaces. Ordinary reads still guard it.
            let mut property = global.borrow().props.get("outer").unwrap().clone();
            property.defer_function_prototype(&owner, engine.interp.object_proto, &template);
            // Reinsert the same descriptor class: a mutable descriptor escape would demote
            // the shape and test a semantic IC miss rather than the displaced-word guard.
            global.borrow_mut().props.insert("outer", property);
        } else {
            assert!(global.borrow_mut().props.set_value(
                "outer",
                Value::BigInt(crate::bigint::JsBigInt::from_u64(99)),
            ));
        }
        let before = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
        assert_eq!(number(run(&mut engine, &chunk, &code, &env)), 0.0);
        assert!(
            TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get) > before,
            "the native store must replace displaced deferred={deferred} without a helper"
        );
        assert_eq!(
            global
                .borrow()
                .props
                .get("outer")
                .unwrap()
                .value()
                .as_num_opt(),
            Some(7.0)
        );
    }
}

#[test]
fn native_name_bigint_stores_decode_the_word_and_record_an_old_holders_card() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function write(value){outer=value;return 0;}");
    let holder = scope(None, "outer", 0.0, new_binding_layout_id());
    let child = new_scope(Some(holder));
    for _ in 0..2 {
        run(&mut engine, &chunk, &code, &child);
    }
    engine.interp.gc_collect();
    let weak = crate::gc_test::in_frame(|| {
        let value = Value::BigInt(crate::bigint::JsBigInt::from_u64(123456789));
        let weak = crate::gc_handle::WeakRef::to(value.cell_word().unwrap() as usize);
        let before = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
        let result = crate::jit::run(
            &mut engine.interp,
            &chunk,
            &code,
            &child,
            Value::Undefined,
            &[value],
        )
        .unwrap_or_else(|_| panic!("native BigInt store throws"));
        assert_eq!(number(result), 0.0);
        assert!(
            TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get) > before,
            "the incoming BigInt must take the native name-store path"
        );
        weak
    });
    engine.interp.gc_collect_minor();
    assert!(crate::gc_test::alive(&weak));
    engine.interp.gc_collect();
    assert!(crate::gc_test::alive(&weak));
    assert!(
        matches!(&(holder.borrow().vars.get("outer").unwrap().value()).view(), ValueView::BigInt(value) if value.to_string_radix(10) == "123456789")
    );
}
