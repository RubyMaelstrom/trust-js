//! Shared ARM64/x64 proofs of packed binding traffic, measured after successful native ops.
//! ECMA-262 e28783d5 GetBindingValue/SetMutableBinding and GetIdentifierReference: value
//! representation does not replace initialization, mutability or with-presence checks.

use crate::bytecode::Chunk;
use crate::interpreter::{new_binding_layout_id, new_scope, Binding, Env};
use crate::jit::JitCode;
use crate::value::Value;
use crate::Engine;
use std::rc::Rc;

fn native(engine: &mut Engine, source: &str) -> (Rc<Chunk>, JitCode) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("fixture parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("fixture function")
    };
    let chunk = crate::bytecode::compile(function).expect("bytecode");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interp_layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = crate::jit::compile(&chunk, &layout, &interp_layout).expect("native image");
    (chunk, code)
}

fn invoke(
    engine: &mut Engine,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    env: &Env,
    args: &[Value],
) -> Value {
    crate::jit::run(&mut engine.interp, chunk, code, env, Value::Undefined, args)
        .unwrap_or_else(|_| panic!("native fixture throws"))
}

fn commits() -> [usize; 3] {
    super::TEST_NATIVE_NAME_COMMITS
        .with(|counts| [counts[0].get(), counts[1].get(), counts[2].get()])
}

fn holder(padding: usize) -> Env {
    holder_with_layout(padding, new_binding_layout_id())
}

fn holder_with_layout(padding: usize, layout: impl Into<crate::interpreter::TestLayout>) -> Env {
    let holder = new_scope(None);
    let mut scope = holder.borrow_mut();
    for index in 0..padding {
        scope.vars.insert(
            format!("pad{index}"),
            Binding::data(Value::Null, true, true),
        );
    }
    scope
        .vars
        .insert("outer", Binding::data(Value::Undefined, true, true));
    scope.vars.publish_layout(layout);
    drop(scope);
    holder
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Global,
    Current,
    Fixed,
    ParentLayout,
    ParentGeneration,
    Deep,
}

const MODES: [Mode; 6] = [
    Mode::Global,
    Mode::Current,
    Mode::Fixed,
    Mode::ParentLayout,
    Mode::ParentGeneration,
    Mode::Deep,
];

struct Case {
    mode: Mode,
    env: Env,
    target: Option<Env>,
    path: Vec<Env>,
    first: Option<Env>,
}

impl Case {
    fn new(engine: &Engine, padding: usize, mode: Mode) -> Self {
        if mode == Mode::Global {
            return Self {
                mode,
                env: engine.interp.global_env,
                target: None,
                path: vec![engine.interp.global_env],
                first: None,
            };
        }
        let layout = if matches!(mode, Mode::Fixed | Mode::Deep) {
            new_binding_layout_id()
        } else {
            crate::interpreter::TestLayout::default()
        };
        let target = holder_with_layout(padding, layout);
        let first = (mode == Mode::Fixed).then(|| holder_with_layout(padding, layout));
        let mut path = vec![target];
        let depth = match mode {
            Mode::ParentLayout | Mode::ParentGeneration => 1,
            Mode::Deep => 3,
            _ => 0,
        };
        for _ in 0..depth {
            let child = new_scope(Some(*path.last().unwrap()));
            if mode == Mode::ParentLayout {
                // A real layout proof of absence in the running child (a descriptor of its
                // names); the parent has no layout.
                let mut scope = child.borrow_mut();
                scope
                    .vars
                    .insert("childpad", Binding::data(Value::Undefined, true, true));
                scope.vars.publish_layout(new_binding_layout_id());
            } else if mode == Mode::ParentGeneration {
                // The prepared-entry fixture below is a real own activation with these captured
                // names. A generation proof of that fixed factory establishes child absence.
                let mut scope = child.borrow_mut();
                scope
                    .vars
                    .insert("own", Binding::data(Value::Undefined, true, true));
                scope
                    .vars
                    .insert("keep", Binding::data(Value::Undefined, true, true));
            }
            path.push(child);
        }
        Self {
            mode,
            env: *path.last().unwrap(),
            target: Some(target),
            path,
            first,
        }
    }

    fn warm(&self, engine: &mut Engine, chunk: &Rc<Chunk>, code: &JitCode, args: &[Value]) {
        if let Some(first) = self.first {
            for _ in 0..3 {
                invoke(engine, chunk, code, &first, args);
            }
        }
        for _ in 0..3 {
            invoke(engine, chunk, code, &self.env, args);
        }
        self.assert_mode(chunk);
    }

    fn assert_mode(&self, chunk: &Chunk) {
        use crate::bytecode::Op;
        let mut seen = 0;
        for op in chunk.test_ops() {
            let cache = match *op {
                Op::LoadName(_, cache)
                | Op::LoadNameForCall(_, cache)
                | Op::StoreNameCached(_, cache)
                | Op::UpdateNameCached(_, cache, _) => cache,
                _ => continue,
            };
            let ic = chunk.test_name_cache(cache);
            match self.mode {
                Mode::Global => assert_eq!(ic.env & 7, 1),
                Mode::Current => assert_eq!(ic.env, Env::as_ptr(&self.env) as usize),
                Mode::Fixed => assert_eq!(ic.env, crate::bytecode::FIXED_NAME_IC),
                Mode::ParentLayout | Mode::ParentGeneration => {
                    let tag = if self.mode == Mode::ParentLayout {
                        6
                    } else {
                        2
                    };
                    assert_eq!(ic.env, Env::as_ptr(&self.target.unwrap()) as usize | tag);
                }
                Mode::Deep => assert_eq!(ic.env, crate::bytecode::lexical_cache::DEEP_NAME_IC),
            }
            seen += 1;
        }
        assert!(
            seen > 0,
            "a real name value site must publish {:?}",
            self.mode
        );
    }
}

fn native_for_mode(engine: &mut Engine, source: &str, mode: Mode) -> (Rc<Chunk>, JitCode) {
    if mode != Mode::ParentGeneration {
        return native(engine, source);
    }
    let source = source.replacen('{', "{var own;function keep(){return own;}", 1);
    let statements = crate::parser::parse_script(&source, false)
        .ok()
        .expect("prepared fixture");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).expect("prepared bytecode");
    let chunk = Chunk::rewrite_owned_for_test(chunk, |chunk| chunk.prepared_entry = true);
    assert!(
        !chunk
            .test_ops()
            .iter()
            .any(|op| matches!(op, crate::bytecode::Op::StoreCap(_))),
        "the prepared activation must not add a store that could mask the target store proof"
    );
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interp_layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = crate::jit::compile(&chunk, &layout, &interp_layout).expect("prepared native image");
    (chunk, code)
}

#[test]
fn every_execution_kind_commits_native_name_loads_and_stores_for_each_owner() {
    let expressions = [
        "undefined",
        "null",
        "false",
        "true",
        "0",
        "-0",
        "NaN",
        "Infinity",
        "-Infinity",
        "5e-324",
        "42.25",
        "'noninterned packed binding text'",
        "Symbol('identity')",
        "({identity:7})",
        "12345678901234567890n",
    ];
    for padding in [0, 3] {
        for mode in MODES {
            if padding != 0 && mode == Mode::Global {
                continue;
            }
            let mut engine = Engine::new();
            engine
                .eval("var outer,inputWord;", false)
                .expect("global prelude");
            let (chunk, code) = native_for_mode(
                &mut engine,
                "function write(input){outer=input;return outer;}",
                mode,
            );
            let case = Case::new(&engine, padding, mode);
            case.warm(&mut engine, &chunk, &code, &[Value::Undefined]);
            for expression in expressions {
                engine
                    .eval(&format!("var inputWord={expression};"), false)
                    .expect("value prelude");
                let globals = engine.interp.global_env;
                let input = engine
                    .interp
                    .get_var("inputWord", &globals)
                    .ok()
                    .expect("rooted input");
                // A separate script can redefine global descriptors even while preserving its
                // value. Fill the current shape after that prelude, then measure this kind.
                invoke(&mut engine, &chunk, &code, &case.env, &[input]);
                case.assert_mode(&chunk);
                let before = commits();
                let result = invoke(&mut engine, &chunk, &code, &case.env, &[input]);
                let after = commits();
                assert!(
                    after[0] > before[0],
                    "native load: padding={padding}, mode={mode:?}, {expression}"
                );
                assert!(
                    after[1] > before[1],
                    "native store: padding={padding}, mode={mode:?}, {expression}"
                );
                assert_eq!((result).bits(), (input).bits(), "{expression}");
                case.assert_mode(&chunk);
                let stored = if let Some(target) = case.target {
                    let scope = target.borrow();
                    let binding = scope.vars.get("outer").unwrap();
                    assert!(binding.initialized() && binding.mutable() && !binding.is_import());
                    binding.value()
                } else {
                    engine.interp.get_var("outer", &globals).ok().unwrap()
                };
                assert_eq!(stored.bits(), (input).bits(), "stored {expression}");
            }
        }
    }
}

#[test]
fn fixed_layout_flags_and_stale_layouts_decline_before_native_commit() {
    for scenario in [
        "stale layout",
        "TDZ",
        "import",
        "strict immutable",
        "sloppy immutable",
    ] {
        let mut engine = Engine::new();
        let (reader, read_code) = native(&mut engine, "function read(){return outer;}");
        let (writer, write_code) =
            native(&mut engine, "function write(input){outer=input;return 0;}");
        let case = Case::new(&engine, 0, Mode::Fixed);
        let target = case.target.unwrap();
        let _root = crate::rooting::Rooted::new(case.env);
        for scope in [case.first.unwrap(), target] {
            scope
                .borrow_mut()
                .vars
                .get_mut("outer")
                .unwrap()
                .set_value(Value::Num(17.0));
        }
        case.warm(&mut engine, &reader, &read_code, &[]);
        case.warm(&mut engine, &writer, &write_code, &[Value::Num(17.0)]);
        case.assert_mode(&reader);
        case.assert_mode(&writer);

        match scenario {
            "stale layout" => {
                let mut scope = target.borrow_mut();
                let old_layout = scope.vars.layout_id();
                scope.vars.clear();
                scope
                    .vars
                    .insert("decoy", Binding::data(Value::Num(99.0), true, true));
                scope
                    .vars
                    .insert("outer", Binding::data(Value::Num(17.0), true, true));
                scope.vars.publish_layout(new_binding_layout_id());
                assert_ne!(scope.vars.layout_id(), old_layout);
                assert_eq!(
                    scope.vars.fixed_binding_at(0).unwrap().value().as_num_opt(),
                    Some(99.0),
                    "the stale slot remains in bounds and names a different binding"
                );
            }
            "TDZ" => {
                target
                    .borrow_mut()
                    .vars
                    .get_mut("outer")
                    .unwrap()
                    .enter_tdz_for_test();
            }
            "import" => {
                let exporter = new_scope(None);
                exporter
                    .borrow_mut()
                    .vars
                    .insert("source", Binding::data(Value::Num(99.0), true, true));
                // CreateImportBinding's replacement gives the record its own names.
                target.borrow_mut().vars.insert(
                    "outer",
                    Binding::live_import(exporter, crate::atom::atomize("source")),
                );
            }
            _ => target.borrow_mut().vars.set_attributes_for_test(
                "outer",
                Some(false),
                Some(scenario == "strict immutable"),
                None,
            ),
        }

        let before = commits();
        let read = crate::jit::run(
            &mut engine.interp,
            &reader,
            &read_code,
            &case.env,
            Value::Undefined,
            &[],
        );
        if scenario == "TDZ" {
            assert!(read.is_err(), "{scenario}");
        } else {
            assert_eq!(
                read.unwrap_or_else(|_| panic!("{scenario}: checked read throws"))
                    .as_num_opt(),
                Some(if scenario == "import" { 99.0 } else { 17.0 }),
                "{scenario}"
            );
        }
        // Every scenario changes the record's names (attributes live with them), so no proof
        // filled before holds.
        assert_eq!(
            commits()[0],
            before[0],
            "{scenario} must reject native read"
        );
        if scenario.ends_with("immutable") {
            // The reader's checked read refilled its cache: an initialized immutable binding is
            // readable natively.
            let before = commits();
            let read = crate::jit::run(
                &mut engine.interp,
                &reader,
                &read_code,
                &case.env,
                Value::Undefined,
                &[],
            );
            assert_eq!(read.ok().and_then(|value| value.as_num_opt()), Some(17.0));
            assert!(
                commits()[0] > before[0],
                "an initialized immutable binding is readable"
            );
        }

        // The reader's checked refill does not alter the independently warmed writer cache.
        case.assert_mode(&writer);
        let before = commits();
        let write = crate::jit::run(
            &mut engine.interp,
            &writer,
            &write_code,
            &case.env,
            Value::Undefined,
            &[Value::Num(23.0)],
        );
        if matches!(scenario, "TDZ" | "import" | "strict immutable") {
            assert!(write.is_err(), "{scenario}");
        } else {
            assert_eq!(
                write
                    .unwrap_or_else(|_| panic!("{scenario}: checked write throws"))
                    .as_num_opt(),
                Some(0.0),
                "{scenario}"
            );
        }
        assert_eq!(
            commits()[1],
            before[1],
            "{scenario} must reject native store"
        );
        let scope = target.borrow();
        let binding = scope.vars.get("outer").unwrap();
        assert_eq!(
            binding.value().as_num_opt(),
            if scenario == "import" || scenario == "TDZ" {
                // An import's slot holds nothing; a binding in TDZ holds the empty word.
                None
            } else if scenario == "stale layout" {
                Some(23.0)
            } else {
                Some(17.0)
            },
            "only the checked stale-layout store may change this binding"
        );
        if scenario == "stale layout" {
            assert_eq!(
                scope.vars.get("decoy").unwrap().value().as_num_opt(),
                Some(99.0)
            );
        }
    }
}

#[test]
fn null_with_words_reject_native_names_and_empty_words_restore_them() {
    for mode in MODES {
        let mut engine = Engine::new();
        engine.eval("var outer=17;", false).unwrap();
        let (chunk, code) = native_for_mode(&mut engine, "function read(){return outer;}", mode);
        let case = Case::new(&engine, 0, mode);
        for target in case.target.into_iter().chain(case.first) {
            target
                .borrow_mut()
                .vars
                .get_mut("outer")
                .unwrap()
                .set_value(Value::Num(17.0));
        }
        case.warm(&mut engine, &chunk, &code, &[]);
        for &scope in &case.path {
            scope.borrow_mut().set_with_object(Some(Value::Null));
            assert!(scope.borrow().has_with_object());
            let before = commits();
            let result = invoke(&mut engine, &chunk, &code, &case.env, &[]);
            assert_eq!(result.as_num_opt(), Some(17.0));
            assert_eq!(
                commits(),
                before,
                "non-EMPTY with word in {mode:?} must reject"
            );
            scope.borrow_mut().set_with_object(None);
            // A checked access may discard its old proof; permit one cold refill after reset.
            invoke(&mut engine, &chunk, &code, &case.env, &[]);
            let before = commits();
            assert_eq!(
                invoke(&mut engine, &chunk, &code, &case.env, &[]).as_num_opt(),
                Some(17.0)
            );
            assert!(
                commits()[0] > before[0],
                "EMPTY restores native read in {mode:?}"
            );
            case.assert_mode(&chunk);
        }
    }
}

#[test]
fn native_binding_stores_mark_the_actual_old_owners_card_for_every_reference_kind() {
    use crate::gc_core::{CellAge, CollectionKind};
    for padding in [0, 3] {
        for mode in MODES.into_iter().filter(|mode| *mode != Mode::Global) {
            let mut engine = Engine::new();
            engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
            engine.interp.gc_heap.agent.task_deferred.set(true);
            let (chunk, code) = native_for_mode(
                &mut engine,
                "function write(input){outer=input;return 0;}",
                mode,
            );
            let case = Case::new(&engine, padding, mode);
            let target = case.target.unwrap();
            let root = crate::rooting::Rooted::new(case.env);
            for traversed in case.path.iter().skip(1) {
                assert_ne!(
                    (Env::as_ptr(&target) as usize) >> crate::gc_core::CARD_SHIFT,
                    (Env::as_ptr(traversed) as usize) >> crate::gc_core::CARD_SHIFT,
                    "marking a traversed scope must not accidentally mark the target's card"
                );
            }
            case.warm(&mut engine, &chunk, &code, &[Value::Undefined]);
            let heap = engine.interp.gc_heap.clone();
            for kind in 0..4 {
                // Reset cards and promote the entire scope chain before allocating the next
                // referent. Only the generated store may dirty the actual binding owner.
                engine
                    .interp
                    .gc_collect_precise_for_test(CollectionKind::Major);
                assert_eq!(
                    heap.core.borrow().cell_age(Env::as_ptr(&target) as u64),
                    Some(CellAge::Old)
                );
                let (input, descriptor) = match kind {
                    0 => (
                        Value::lstr("sole packed binding string ".repeat(40)),
                        heap.string_desc,
                    ),
                    1 => (
                        engine
                            .interp
                            .new_symbol(Some(Rc::from("sole packed binding Symbol"))),
                        heap.symbol_desc,
                    ),
                    2 => (
                        Value::BigInt(crate::bigint::JsBigInt::from_u64(123456789012345)),
                        heap.bigint_desc,
                    ),
                    _ => (Value::Obj(engine.interp.new_object()), heap.object_desc),
                };
                let cell = input.cell_word().unwrap() as usize;
                let weak = crate::gc_handle::WeakRef::to(cell);
                assert_eq!(
                    heap.core.borrow().cell_age(cell as u64),
                    Some(CellAge::Young)
                );
                let before = commits();
                assert_eq!(
                    invoke(&mut engine, &chunk, &code, &root, &[input]).as_num_opt(),
                    Some(0.0)
                );
                assert!(
                    commits()[1] > before[1],
                    "committed native reference store {kind}, mode={mode:?}, padding={padding}"
                );
                case.assert_mode(&chunk);
                assert_eq!(
                    heap.core.borrow().cell_age(cell as u64),
                    Some(CellAge::Young),
                    "no intervening promotion may mask the card"
                );
                // Precise marking excludes the Rust input/argument temporaries and cache weak
                // pins. The old scope's new word is now this referent's sole strong edge.
                for collection in [CollectionKind::Minor, CollectionKind::Major] {
                    engine.interp.gc_collect_precise_for_test(collection);
                    assert_eq!(weak.live_ptr(), Some(cell));
                    assert!(heap.core.borrow().owns_cell(cell, descriptor));
                }
                target
                    .borrow_mut()
                    .vars
                    .get_mut("outer")
                    .unwrap()
                    .set_value(Value::Undefined);
                engine
                    .interp
                    .gc_collect_precise_for_test(CollectionKind::Major);
                assert_eq!(
                    weak.live_ptr(),
                    None,
                    "cleared binding releases kind={kind}"
                );
            }
        }
    }
}

#[test]
fn native_binding_updates_commit_all_six_forms_and_numeric_edges() {
    for (expression, discard, post, decrement) in [
        ("++outer", false, false, false),
        ("outer++", false, true, false),
        ("--outer", false, false, true),
        ("outer--", false, true, true),
        ("outer++", true, false, false),
        ("outer--", true, false, true),
    ] {
        let mut engine = Engine::new();
        let source = if discard {
            format!("function update(){{{expression};return 9;}}")
        } else {
            format!("function update(){{return {expression};}}")
        };
        let (chunk, code) = native(&mut engine, &source);
        let target = holder(3);
        let env = new_scope(Some(new_scope(Some(target))));
        target
            .borrow_mut()
            .vars
            .get_mut("outer")
            .unwrap()
            .set_value(Value::Num(1.0));
        for _ in 0..3 {
            invoke(&mut engine, &chunk, &code, &env, &[]);
        }
        for input in [
            -0.0,
            0.0,
            1.25,
            -1.0,
            f64::from_bits(1),
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            target
                .borrow_mut()
                .vars
                .get_mut("outer")
                .unwrap()
                .set_value(Value::Num(input));
            let before = commits();
            let result = invoke(&mut engine, &chunk, &code, &env, &[]);
            assert!(
                commits()[2] > before[2],
                "committed native {expression}, discard={discard}, input={input:?}"
            );
            let next = if decrement { input - 1.0 } else { input + 1.0 };
            let expected = if discard {
                9.0
            } else if post {
                input
            } else {
                next
            };
            assert_eq!((result).bits(), (Value::Num(expected)).bits());
            assert_eq!(
                target.borrow().vars.get("outer").unwrap().value().bits(),
                (Value::Num(next)).bits()
            );
        }
    }
}
