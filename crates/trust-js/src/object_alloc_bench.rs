//! Ignored release-mode measurements of the raw object allocation/destruction cost, isolated from
//! JavaScript dispatch. Run with
//! `cargo test --release -p trust-js --lib object_alloc_bench -- --ignored --nocapture`.

use crate::value::{Exotic, Object, Property, Props, Value};
use std::rc::Rc;

fn time_per_op(label: &str, iterations: u32, mut op: impl FnMut(u32)) {
    for i in 0..iterations / 10 {
        op(i);
    }
    let started = std::time::Instant::now();
    for i in 0..iterations {
        op(i);
    }
    let nanos = started.elapsed().as_nanos() as f64 / f64::from(iterations);
    println!("{label:40} {nanos:8.1} ns/op");
}

#[test]
#[ignore]
fn object_alloc_bench() {
    let heap = crate::value::new_gc_heap();
    let symbols = crate::value::new_symbol_agent();
    let _active = crate::value::enter_agent(&heap, &symbols);
    let proto = Object::new(None);
    const N: u32 = 5_000_000;

    time_per_op("Object::new(proto) + drop", N, |_| {
        std::hint::black_box(Object::new(Some(proto)));
    });

    let mut template = Props::new();
    template.insert(Rc::from("x"), Property::plain(Value::Undefined));
    template.insert(Rc::from("y"), Property::plain(Value::Undefined));
    time_per_op("templated {x,y} + drop", N, |i| {
        let props =
            template.instantiate_plain([(Value::Num(f64::from(i))), (Value::Num(1.0))].into_iter());
        std::hint::black_box(Object::new_with_parts(Some(proto), props, Exotic::None));
    });

    let mut kept = Vec::with_capacity(N as usize);
    time_per_op("Object::new(proto) retained", N, |_| {
        kept.push(Object::new(Some(proto)));
    });
    let started = std::time::Instant::now();
    drop(kept);
    println!(
        "{:40} {:8.1} ns/op",
        "drop retained",
        started.elapsed().as_nanos() as f64 / f64::from(N)
    );

    time_per_op("Rc::new(RefCell<[u8;136]>) + drop", N, |_| {
        std::hint::black_box(Rc::new(std::cell::RefCell::new([0u8; 136])));
    });
    time_per_op("Gc::downgrade + drop weak", N, |_| {
        std::hint::black_box(crate::value::Gc::downgrade(&proto));
    });
    #[cfg(not(target_arch = "wasm32"))]
    time_per_op("stacker::remaining_stack()", N, |_| {
        std::hint::black_box(stacker::remaining_stack());
    });
    time_per_op("comparison: gc_heap Rc clone+drop", N, |_| {
        std::hint::black_box(heap.clone());
    });
    time_per_op("Props::new() + drop", N, |_| {
        std::hint::black_box(Props::new());
    });
    time_per_op("active_gc_heap()", N, |_| {
        std::hint::black_box(crate::value::active_gc_heap());
    });
}

#[test]
#[ignore]
fn object_layout_sizes() {
    use crate::value::*;
    println!(
        "Object {} Props {} Callable {} Exotic {} Property {} RefCell<Object> {}",
        std::mem::size_of::<Object>(),
        std::mem::size_of::<Props>(),
        std::mem::size_of::<Callable>(),
        std::mem::size_of::<Exotic>(),
        std::mem::size_of::<Property>(),
        std::mem::size_of::<std::cell::RefCell<Object>>()
    );
    println!(
        "FunctionMaps {} ast::Function {} OnceCell<Props> {}",
        std::mem::size_of::<crate::ast::FunctionMaps>(),
        std::mem::size_of::<crate::ast::Function>(),
        std::mem::size_of::<std::cell::OnceCell<Props>>()
    );
}

/// Collector throughput on a mixed heap: a retained heap, then new objects, arrays, closures
/// (environment records) and cyclic garbage. Run with
/// `cargo test --release -p trust-js --lib gc_collection_bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn gc_collection_bench() {
    use crate::{Completion, Engine};
    fn run(engine: &mut Engine, source: &str) {
        match engine.eval(source, false).expect("benchmark source parses") {
            Completion::Value(_) => {}
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
    let mut engine = Engine::new();
    run(
        &mut engine,
        r#"
        var old = [];
        for (var i = 0; i < 200000; i++) old.push({id: i, next: null, tag: 'x', list: [i, i + 1]});
        for (var i = 1; i < old.length; i++) old[i].next = old[i - 1];
        function maker(n) { var captured = {n: n}; return function () { return captured.n; }; }
        var closures = [];
        for (var i = 0; i < 20000; i++) closures.push(maker(i));
        true
    "#,
    );
    engine.interp.gc_collect();
    let young_source = r#"
        var young = [];
        for (var i = 0; i < 60000; i++) {
            var o = {a: i, b: {c: i, d: 'y'}, e: [i]};
            if (i % 3 == 0) { o.self = o; o.b.parent = o; } else { young.push(o); }
            if (i % 10 == 0) young.push(maker(i));
        }
        true
    "#;
    let mut major = Vec::new();
    for _ in 0..3 {
        run(&mut engine, young_source);
        let started = std::time::Instant::now();
        engine.interp.gc_collect();
        major.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let live = crate::value::heap_live_objects(&engine.interp.gc_heap);
    println!("gc bench: live objects after {live}");
    println!("major ms {major:.2?}");
}
