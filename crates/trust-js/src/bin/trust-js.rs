//! Minimal shell: evaluate JS files in one engine, printing console output as it appears.
//! Usage: trust-js [--module] [--tier=interp|bytecode|jit] [--tier-threshold=N] [file.js ...]
//!        trust-js -h | --help       print the help menu (all flags and env vars)
//!        trust-js -v | --version    print the version
//!
//! With no files, reads from stdin: an interactive REPL when stdin is a terminal (one realm
//! for the whole session, incomplete input continues on the next line), otherwise the whole
//! stream is evaluated as one script.
//!
//! With `--module` each file is evaluated as an ES module; relative import specifiers resolve
//! against the importing file on disk (what a test262 host like test262.fyi expects).

use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use trust_js::{Completion, Engine, ExecutionOutcome};

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static GLOBAL_ALLOC: trust_js::fastalloc::ClassAlloc = trust_js::fastalloc::ClassAlloc;

fn main() {
    let mut module = false;
    let mut interactive = false;
    let mut tier = None;
    let mut threshold = None;
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| {
            if a == "-h" || a == "--help" {
                print_help();
                std::process::exit(0);
            }
            if a == "-v" || a == "--version" {
                // The nightly workflow sets TRUST_JS_VERSION_SUFFIX to "-nightly (<short sha>)".
                println!(
                    "TRust JS {}{}",
                    env!("CARGO_PKG_VERSION"),
                    option_env!("TRUST_JS_VERSION_SUFFIX").unwrap_or("")
                );
                std::process::exit(0);
            }
            if a == "--module" {
                module = true;
                false
            } else if a == "-i" || a == "--interactive" {
                interactive = true;
                false
            } else if let Some(t) = a.strip_prefix("--tier=") {
                tier = Some(match t {
                    "bytecode" => trust_js::bytecode::Tier::Bytecode,
                    "jit" => trust_js::bytecode::Tier::Jit,
                    "interp" => trust_js::bytecode::Tier::Interp,
                    other => {
                        eprintln!("error: unknown tier '{other}' (interp|bytecode|jit)");
                        std::process::exit(2);
                    }
                });
                false
            } else if let Some(n) = a.strip_prefix("--tier-threshold=") {
                threshold = n.parse::<u32>().ok();
                false
            } else {
                true
            }
        })
        .collect();
    let mut engine = Engine::new();
    if let Some(t) = tier {
        engine.set_tier(t);
    }
    if let Some(n) = threshold {
        engine.set_tier_threshold(n);
    }
    if args.is_empty() {
        if module {
            eprintln!("error: --module requires a file");
            std::process::exit(2);
        }
        if interactive || std::io::stdin().is_terminal() {
            repl(&mut engine);
        } else {
            eval_stdin(&mut engine);
        }
        finalize_performance_metrics(&mut engine);
        return;
    }
    for path in &args {
        let src = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read {path}: {e}");
                std::process::exit(2);
            }
        };
        let result = if module {
            let key = normalize_path(Path::new(path))
                .to_string_lossy()
                .into_owned();
            engine.eval_module(&src, &key, load_module)
        } else {
            // Stack frames name the file, like Node's.
            let origin = trust_js::SourceOrigin::new(path);
            engine
                .eval_interruptible_with_origin(&src, false, Some(&origin))
                .map(|outcome| match outcome {
                    ExecutionOutcome::Value(value) => Completion::Value(value),
                    ExecutionOutcome::Throw { name, message } => {
                        Completion::Throw { name, message }
                    }
                    ExecutionOutcome::Interrupted { reason } => Completion::Throw {
                        name: String::new(),
                        message: reason.message().to_string(),
                    },
                })
        };
        for line in engine.take_console() {
            println!("{line}");
        }
        match result {
            Ok(Completion::Value(_)) => {}
            Ok(Completion::Throw { name, message }) => {
                eprintln!("uncaught {name}: {message}");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("SyntaxError in {path}: {}", e.message);
                std::process::exit(1);
            }
        }
    }
    finalize_performance_metrics(&mut engine);
}

/// The managed-memory contract is explicitly post-collection. Keep this diagnostic collection in
/// the shell (and behind the existing opt-in switch) so library embedders retain control of their
/// own idle/collection boundaries and normal CLI runs do no extra work.
fn finalize_performance_metrics(engine: &mut Engine) {
    if let Some(report) = trust_js::unstable_regexp_prof_report() {
        eprintln!("[regexp-prof] {report}");
    }
    if std::env::var_os("TRUST_JS_PERF_METRICS").is_some() {
        engine.unstable_collect_for_performance_metrics();
        if let Some(metrics) = trust_js::unstable_performance_metrics_json(engine) {
            eprintln!("[trust-js-perf] {metrics}");
        }
    }
}

/// Print the `--help` menu: every flag the shell parses plus the runtime env vars it honours.
/// Keep this in sync with the argument loop in `main` and the tier env vars read by the engine.
fn print_help() {
    let version = env!("CARGO_PKG_VERSION");
    print!(
        "\
TRust JS {version} — a JavaScript engine

Usage:
  trust-js [options] [file.js ...]   evaluate each file in one engine
  trust-js [options] < script.js     evaluate stdin as one script
  trust-js [-i] [options]            interactive REPL (default when stdin is a tty)

Options:
  -h, --help              Print this help and exit
  -v, --version           Print the version and exit
      --module            Evaluate each file as an ES module (relative imports
                          resolve against the importing file on disk)
  -i, --interactive       Force the REPL even when stdin is not a terminal
      --tier=TIER         Execution tier: interp | bytecode | jit
                          (default: jit; falls back to the bytecode VM where the
                          JIT is unavailable)
      --tier-threshold=N  Calls before an eligible function tiers up to a
                          compiled tier; 0 = immediately (default: 8)
                          Long-running loops can tier up during the current call
                          after 128 executed backedges.

Environment:
  TRUST_JS_TIER=TIER         Default execution tier (--tier overrides it)
  TRUST_JS_TIER_THRESHOLD=N  Default tier-up threshold (--tier-threshold overrides it)
  TRUST_JS_TAGGED_ARITHMETIC=1  Enable the opt-in tagged Number frame migration slice
                            (arithmetic and comparison operators in the bytecode VM)

Diagnostics (env, unstable):
  TRUST_JS_TIER_LOG=1        Report the AST construct a compile bail came from
  TRUST_JS_JIT_DUMP=SUBSTR   Dump the op stream of JIT'd chunks whose slot names match
  TRUST_JS_JIT_CODEDUMP=SUB  Dump the finished machine code of matching chunks (hex)
  TRUST_JS_JIT_OPSTAT=1      Tally ops that reach the JIT slow path (top at exit; =2 pinpoints sites)
  TRUST_JS_REGEXP_PROF=1     Report matcher instruction/scan/attempt/backtrack/tier counts at exit
  TRUST_JS_REGEXP_TIER_UP_AT=N  Promote eligible ASCII matchers after N calls; 0 disables
  TRUST_JS_JIT_CALLSTAT=1    Tally calls that reach the inline-cache call helper
  TRUST_JS_JIT_CODE_BUDGET_MB=N  Process-wide live executable-code cap, 1..1024 MiB
  TRUST_JS_JIT_LOOPLOG=1     Trace JIT loop back-edge compilation
  TRUST_JS_PERF_METRICS=1    Print one JSON JIT/collector performance summary at normal exit
"
    );
}

/// Interactive prompt: one engine (realm) for the whole session. Input that parses but for
/// running out of source (`at_eof`) continues on a `..` line; Ctrl-D exits.
fn repl(engine: &mut Engine) {
    println!("TRust JS {} (Ctrl-D to exit)", env!("CARGO_PKG_VERSION"));
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    loop {
        print!("{}", if buffer.is_empty() { "> " } else { ".. " });
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                return;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("error: {e}");
                return;
            }
        }
        buffer.push_str(&line);
        if buffer.trim().is_empty() {
            buffer.clear();
            continue;
        }
        match engine.eval(&buffer, false) {
            Err(e) if e.at_eof => continue, // incomplete input: keep reading
            result => {
                buffer.clear();
                report(engine, result, true);
            }
        }
    }
}

/// Non-terminal stdin: evaluate the whole stream as one script (e.g. `echo 1+1 | trust-js`).
fn eval_stdin(engine: &mut Engine) {
    let mut src = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut src) {
        eprintln!("error: cannot read stdin: {e}");
        std::process::exit(2);
    }
    let result = engine.eval(&src, false);
    let ok = matches!(result, Ok(Completion::Value(_)));
    report(engine, result, false);
    if !ok {
        std::process::exit(1);
    }
}

/// Print buffered console output, then the completion: the result value (REPL only), an
/// `Uncaught ...` line for a throw, or the SyntaxError.
fn report(engine: &mut Engine, result: Result<Completion, trust_js::ParseError>, echo_value: bool) {
    for line in engine.take_console() {
        println!("{line}");
    }
    match result {
        Ok(Completion::Value(v)) => {
            if echo_value && !v.is_empty() {
                println!("{v}");
            }
        }
        Ok(Completion::Throw { name, message }) => {
            if name.is_empty() {
                eprintln!("Uncaught {message}");
            } else {
                eprintln!("Uncaught {name}: {message}");
            }
        }
        Err(e) => eprintln!("SyntaxError: {}", e.message),
    }
}

/// Resolve `spec` against the importing module's path and read it off disk. Non-UTF-8 sources are
/// decoded latin-1 style (test262 `type: "bytes"` fixtures).
fn load_module(spec: &str, referrer: &str) -> Option<(String, String)> {
    let base = Path::new(referrer).parent()?;
    let resolved = normalize_path(&base.join(spec));
    let bytes = std::fs::read(&resolved).ok()?;
    let text: String = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => e.into_bytes().iter().map(|&b| b as char).collect(),
    };
    Some((resolved.to_string_lossy().into_owned(), text))
}

/// Lexically resolve `.` / `..` so a module that imports itself via a roundabout specifier maps to
/// the same registration key.
fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c.as_os_str()),
        }
    }
    out
}
