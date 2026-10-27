//! The perf map (`TRUST_JS_PERF_MAP`, see [`super`]): compiled functions and shared stubs each
//! add a line that parses as perf parses it and names their code (the native RegExp tier's
//! programs are checked beside that tier, in `regex`), and nothing is written while the switch
//! is off.

use super::{perf_map_enabled, perf_map_line, PerfMapCapture};
use crate::ast::{Expr, Function, Stmt};
use std::rc::Rc;

/// `demo` starts at line 2, column 1, and the anonymous function expression at line 3, column
/// 17. `demo` has a loop, so it also gets a borrowed (OSR) entry.
const FIXTURE: &str = "\nfunction demo(value) { let sum = 0; for (let k = 0; k < value; k++) sum += k; return sum; }\nvar anonymous = function (value) { return value * 2; };\n";
const URL: &str = "perf-map-fixture.js";
const DEMO: &str = "JS:^demo perf-map-fixture.js:2:1";
const ANONYMOUS: &str = "JS:^<anonymous> perf-map-fixture.js:3:17";

fn parse(source: &str) -> Vec<Stmt> {
    let origin = crate::SourceOrigin::new(URL);
    crate::parser::parse_script_from(
        source,
        false,
        crate::stack_trace::SourceKind::Script,
        Some(&origin),
    )
    .ok()
    .expect("fixture parses")
    .0
}

/// The fixture's `demo` and anonymous function.
fn functions(body: &[Stmt]) -> (&Rc<Function>, &Rc<Function>) {
    let Stmt::FuncDecl(demo) = &body[0] else {
        panic!("demo is a declaration")
    };
    let Stmt::VarDecl { decls, .. } = &body[1] else {
        panic!("anonymous is a variable's initializer")
    };
    let Some(Expr::Func(anonymous)) = &decls[0].1 else {
        panic!("anonymous is a function expression")
    };
    (demo, anonymous)
}

#[test]
fn perf_map_lines_are_lowercase_hexadecimal_and_end_at_their_newline() {
    assert_eq!(
        perf_map_line(0xffff_8a3c_1000, 0x4c, DEMO),
        "ffff8a3c1000 4c JS:^demo perf-map-fixture.js:2:1\n"
    );
    // A line terminator in a name (a URL, a RegExp source) would end perf's line early.
    assert_eq!(
        perf_map_line(0x10, 0x8, "RegExp:/a\nb\r/"),
        "10 8 RegExp:/a\\nb\\r/\n"
    );
}

#[test]
fn perf_map_labels_name_a_chunk_by_function_and_first_token_or_by_its_locals() {
    let _capture = PerfMapCapture::start("labels");
    assert!(perf_map_enabled());
    let body = parse(FIXTURE);
    let (demo, anonymous) = functions(&body);
    let chunk = crate::bytecode::compile(demo).expect("demo compiles");
    assert_eq!(chunk.perf_map_label.as_deref(), Some(DEMO));
    let chunk = crate::bytecode::compile(anonymous).expect("anonymous compiles");
    assert_eq!(chunk.perf_map_label.as_deref(), Some(ANONYMOUS));
    // Self-hosted code has no script: its first locals name it.
    let body = crate::parser::parse_self_hosted("function hosted(first, second) { return first; }")
        .ok()
        .expect("self-hosted fixture parses");
    let Stmt::FuncDecl(hosted) = &body[0] else {
        panic!("hosted is a declaration")
    };
    let chunk = crate::bytecode::compile(hosted).expect("hosted compiles");
    let label = chunk.perf_map_label.as_deref().expect("labelled");
    assert!(
        label.starts_with("JS:^hosted locals(first,second"),
        "{label}"
    );
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn perf_map_lists_each_compiled_entry_of_a_function_where_its_code_is() {
    use crate::jit::JitCompileOutcome;
    let capture = PerfMapCapture::start("chunks");
    let mut engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
    let body = parse(FIXTURE);
    let (demo, _) = functions(&body);
    let chunk = crate::bytecode::compile(demo).expect("demo compiles");
    let JitCompileOutcome::Compiled(fresh) =
        crate::jit::compile_profiled(&chunk, &layout, &interpreter_layout)
    else {
        panic!("demo has native code")
    };
    let JitCompileOutcome::Compiled(borrowed) =
        crate::jit::compile_borrowed_profiled(&chunk, &layout, &interpreter_layout)
    else {
        panic!("demo's loop has a borrowed entry")
    };
    let lines = capture.lines();
    for code in [&fresh, &borrowed] {
        let line = (code.mem as usize, code.len, DEMO.to_string());
        assert!(lines.contains(&line), "{line:x?} in {lines:x?}");
    }
}

/// Through the engine's own tier-up: each function is listed when it is compiled.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn perf_map_lists_the_functions_an_engine_compiles() {
    let capture = PerfMapCapture::start("engine");
    let mut engine = crate::Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    let source = format!(
        "{FIXTURE}var total = 0;\nfor (var i = 0; i < 200; i++) total += demo(i % 7) + anonymous(i);\ntotal"
    );
    let outcome = engine
        .eval_interruptible_with_origin(&source, false, Some(&crate::SourceOrigin::new(URL)))
        .expect("fixture parses");
    let crate::ExecutionOutcome::Value(total) = outcome else {
        panic!("fixture completes")
    };
    assert_eq!(total, "40784");
    let names: Vec<String> = capture.lines().into_iter().map(|line| line.2).collect();
    for name in [DEMO, ANONYMOUS] {
        assert!(
            names.iter().any(|listed| listed == name),
            "{name} in {names:?}"
        );
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn perf_map_lists_each_shared_stub_by_kind() {
    use crate::jit::{shared_stubs_enabled, with_global_stubs, SharedStub};
    // Test builds keep a stub table per thread, so this thread assembles its own stubs.
    let capture = PerfMapCapture::start("stubs");
    let source = r#"
        var base = 40, text = 'abc';
        function name() { return base + 1; }
        function prop(o) { return o.x + o.y; }
        function leaf(v) { return v * 2; }
        function direct(v) { var r = leaf(v); return r; }
        function intr(s, n) { return s.charCodeAt(n); }
        var t = 0;
        for (var k = 0; k < 50; k++) {
            t += name() + prop({ x: k, y: 1 }) + direct(k) + intr(text, k % 3);
        }
        t
    "#;
    let mut engine = crate::Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    match engine.eval(source, false).expect("stub fixture parses") {
        crate::Completion::Value(value) => assert_eq!(value, "10674"),
        crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    if env_flag!("TRUST_JS_JIT_CHUNK_STUBS") || !shared_stubs_enabled() {
        return;
    }
    let stubs: Vec<(u32, u64)> = with_global_stubs(|table| {
        let table = table
            .as_ref()
            .expect("compiled chunks requested shared stubs");
        table
            .addresses
            .iter()
            .map(|(&key, &at)| (key, at))
            .collect()
    })
    .unwrap();
    assert!(!stubs.is_empty());
    let lines = capture.lines();
    for (key, address) in stubs {
        let name = SharedStub::from_key(key).perf_map_name();
        assert!(
            lines
                .iter()
                .any(|(start, size, listed)| *start as u64 == address
                    && *size > 0
                    && *listed == name),
            "{name} at {address:#x} in {lines:x?}"
        );
    }
    // A direct-call stub's buffer ends with its teardown stub, listed on its own right after
    // the stubs before it.
    let (finish, _, _) = lines
        .iter()
        .find(|(_, _, name)| name == "Stub:DirectFinish")
        .expect("a direct call was compiled");
    assert!(lines.iter().any(|(start, size, _)| start + size == *finish));
}

#[test]
fn perf_map_labels_nothing_while_off() {
    let capture = PerfMapCapture::start("labels-off");
    let body = parse(FIXTURE);
    let (demo, anonymous) = functions(&body);
    let labelled = crate::bytecode::compile(demo).expect("demo compiles");
    assert_eq!(labelled.perf_map_label.as_deref(), Some(DEMO));
    capture.stop();
    // Off for this thread. The process's own switch is off too unless this test run sets
    // TRUST_JS_PERF_MAP, and then only the process's map is written.
    let unlabelled = crate::bytecode::compile(anonymous).expect("anonymous compiles");
    assert_eq!(
        unlabelled.perf_map_label.is_some(),
        perf_map_enabled(),
        "a label is built only while a map is on"
    );
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn perf_map_lists_nothing_while_off() {
    use crate::jit::JitCompileOutcome;
    let capture = PerfMapCapture::start("off");
    let mut engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
    let compile = |chunk: &crate::bytecode::Chunk| match crate::jit::compile_profiled(
        chunk,
        &layout,
        &interpreter_layout,
    ) {
        JitCompileOutcome::Compiled(code) => code,
        _ => panic!("the fixture has native code"),
    };
    let body = parse(FIXTURE);
    let (demo, anonymous) = functions(&body);
    let labelled = crate::bytecode::compile(demo).expect("demo compiles");
    let on = compile(&labelled);
    let before = capture.lines();
    let line = (on.mem as usize, on.len, DEMO.to_string());
    assert!(before.contains(&line), "{line:x?} in {before:x?}");
    capture.stop();
    let unlabelled = crate::bytecode::compile(anonymous).expect("anonymous compiles");
    let _off = [compile(&labelled), compile(&unlabelled)];
    assert_eq!(
        capture.lines(),
        before,
        "nothing is listed while this thread's map is off, labelled or not"
    );
}
