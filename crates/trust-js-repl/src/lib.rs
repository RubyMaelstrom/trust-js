//! trust-js-repl — the interactive shell.
//!
//! One realm for the whole session. Each submitted input is evaluated **without** a microtask
//! checkpoint (`Ctx::eval_value`), then the runtime's event loop runs to quiescence, so
//! awaited promises and timers settle (and their console output streams) *before* the result
//! prints and the next prompt appears. Incomplete input — an unclosed block, template, or
//! paren — is detected by the parser's `at_eof` flag and keeps reading on a `..` continuation
//! prompt.
//!
//! Top-level `await` is REPL sugar, not script syntax: input that fails to parse directly and
//! mentions `await` is retried wrapped in an async IIFE (expression form first, then statement
//! form) whose settlement is captured and printed. Statement-form bindings (`const x = await
//! ...`) stay inside the wrapper — a known limitation, matching the simple half of Node's
//! behavior.
//!
//! Caveat, documented on purpose: quiescence-before-prompt means an unbounded `setInterval`
//! holds the prompt hostage until cleared (Ctrl-C kills the process; there is no per-line
//! signal handling in pure std).

use trust_js_host::{Ctx, Handle, ParseError, Value};
use trust_js_runtime::{describe_error, render_value, Runtime};

/// An input's completion, its value (or the value it threw) held by a handle across the loop's
/// run to quiescence.
type Held = Result<Result<Handle, Handle>, ParseError>;

/// Evaluate `src` in a scope of `ctx`'s engine, holding the completion.
fn eval_held(ctx: &mut Ctx<'_>, src: &str) -> Held {
    fn hold<'s>(ctx: &Ctx<'s>, value: Value<'s>) -> Handle {
        Handle::new(ctx, &value).expect("the engine's own value")
    }
    Ok(match ctx.eval_value(src)? {
        Ok(value) => Ok(hold(ctx, value)),
        Err(thrown) => Err(hold(ctx, thrown)),
    })
}

pub struct Repl {
    runtime: Runtime,
    /// Accumulated continuation lines awaiting a complete parse.
    buffer: String,
}

/// What a submitted line produced.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    /// Input is incomplete: show the continuation prompt and keep reading.
    More,
    /// A complete evaluation (result, error report, or dot-command output) to print.
    Done(String),
    /// `.exit` (or Ctrl-D at the driver level).
    Exit,
}

const HELP: &str = "\
.help   this text
.exit   leave the repl
_       the last result
await   allowed at the top level (expression form)";

impl Repl {
    pub fn new(runtime: Runtime) -> Repl {
        Repl {
            runtime,
            buffer: String::new(),
        }
    }

    pub fn runtime(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// Whether the next prompt is a continuation (`.. `) rather than a fresh `> `.
    pub fn continuing(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Submit one input line.
    pub fn feed(&mut self, line: &str) -> Step {
        // Dot commands are single-line only — a `.exit` inside a half-typed function is code.
        if self.buffer.is_empty() {
            match line.trim() {
                "" => return Step::More,
                ".exit" => return Step::Exit,
                ".help" => return Step::Done(HELP.into()),
                cmd if cmd.starts_with('.') && !cmd.starts_with("..") => {
                    return Step::Done(format!("unknown command {cmd} (try .help)"));
                }
                _ => {}
            }
        }
        self.buffer.push_str(line);
        self.buffer.push('\n');
        let src = self.buffer.clone();

        let completion = self.runtime.engine().scope(|cx| eval_held(cx, &src));
        match completion {
            Err(e) if e.at_eof => Step::More, // incomplete: keep the buffer
            Err(e) => {
                self.buffer.clear();
                if src.contains("await") {
                    if let Some(step) = self.try_top_level_await(&src) {
                        return step;
                    }
                }
                Step::Done(format!("SyntaxError: {}", e.message))
            }
            Ok(completion) => {
                self.buffer.clear();
                self.runtime.run_to_completion();
                match completion {
                    Ok(v) => self.finish_value(&v),
                    Err(e) => self.finish_thrown(&e),
                }
            }
        }
    }

    /// Retry `src` as REPL-sugar top-level await: an async IIFE whose settlement lands in a
    /// temporary global this side of the loop-to-quiescence run. `None` = the wrapping didn't
    /// parse either; report the original error.
    fn try_top_level_await(&mut self, src: &str) -> Option<Step> {
        let expr = format!(
            "globalThis.__repl_settled = undefined;
             (async () => ( {src} ))().then(
                 (v) => {{ globalThis.__repl_settled = {{ ok: true, value: v }}; }},
                 (e) => {{ globalThis.__repl_settled = {{ ok: false, value: e }}; }});"
        );
        let stmt = format!(
            "globalThis.__repl_settled = undefined;
             (async () => {{ {src} }})().then(
                 (v) => {{ globalThis.__repl_settled = {{ ok: true, value: v }}; }},
                 (e) => {{ globalThis.__repl_settled = {{ ok: false, value: e }}; }});"
        );
        let completion = [expr, stmt].into_iter().find_map(|wrapped| {
            self.runtime
                .engine()
                .scope(|cx| eval_held(cx, &wrapped))
                .ok()
        })?;
        if let Err(e) = completion {
            // A sync throw from the wrapper itself (rare; async bodies reject instead).
            return Some(self.finish_thrown(&e));
        }
        self.runtime.run_to_completion();

        // The settlement: `None` when the awaited promise never settled, else whether it was
        // fulfilled and its value or reason, held.
        let settled = self.runtime.engine().scope(|ctx| {
            let global = ctx.global_this();
            let settled = ctx.get_member(&global, "__repl_settled").ok()?;
            let _ = ctx.set_member(&global, "__repl_settled", Value::Undefined);
            if matches!(settled, Value::Undefined) {
                return Some(None);
            }
            let ok = matches!(ctx.get_member(&settled, "ok"), Ok(Value::Bool(true)));
            let value = ctx.get_member(&settled, "value").ok()?;
            Some(Some((
                ok,
                Handle::new(ctx, &value).expect("the engine's own value"),
            )))
        })?;
        let Some((ok, value)) = settled else {
            // The awaited promise never settled (e.g. `await new Promise(() => {})`): the
            // loop went idle with it still pending. Say so instead of printing a value.
            return Some(Step::Done("[promise never settled]".into()));
        };
        Some(if ok {
            self.finish_value(&value)
        } else {
            self.finish_thrown(&value)
        })
    }

    /// Record `_` and render the result line.
    fn finish_value(&mut self, held: &Handle) -> Step {
        let text = self.runtime.engine().scope(|ctx| {
            let v = held.get(ctx).expect("the engine's own handle");
            let global = ctx.global_this();
            let _ = ctx.set_member(&global, "_", v);
            match &v {
                // Quote strings so `"42"` and `42` are distinguishable, as every REPL does.
                Value::Str(_) => format!("'{}'", v.as_text().unwrap_or_default()),
                other => render_value(ctx, other),
            }
        });
        Step::Done(text)
    }

    fn finish_thrown(&mut self, held: &Handle) -> Step {
        let text = self.runtime.engine().scope(|ctx| {
            let e = held.get(ctx).expect("the engine's own handle");
            describe_error(ctx, &e)
        });
        Step::Done(format!("Uncaught {text}"))
    }

    /// The line-buffered driver: prompts on `out`, reads `input` until `.exit` or EOF.
    pub fn run(mut self, input: &mut dyn std::io::BufRead, out: &mut dyn std::io::Write) {
        loop {
            let prompt = if self.continuing() { ".. " } else { "> " };
            let _ = write!(out, "{prompt}");
            let _ = out.flush();
            let mut line = String::new();
            match input.read_line(&mut line) {
                Ok(0) | Err(_) => {
                    let _ = writeln!(out);
                    return;
                }
                Ok(_) => {}
            }
            match self.feed(&line) {
                Step::More => {}
                Step::Done(text) => {
                    if !text.is_empty() {
                        let _ = writeln!(out, "{text}");
                    }
                }
                Step::Exit => return,
            }
        }
    }
}

#[cfg(test)]
mod tests;
