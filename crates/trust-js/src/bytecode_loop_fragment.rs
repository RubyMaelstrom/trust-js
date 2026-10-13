//! Settled AST loop backedges enter a new baseline chunk without replaying a head.
//!
//! The enclosing Rust evaluator retains its scopes, handlers and resources. This chunk owns
//! only operations after the transfer, including each newly entered iteration's cleanup.
//! ECMA-262 §§14.7.2.2, 14.7.3.2, 14.7.4.3 and 14.7.5.7 (local snapshot e28783d5).
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RootKind {
    While,
    DoWhile,
    For,
    ForIn,
    ForOf,
}

/// Borrowed syntax only: the cold path neither clones the AST nor analyzes the subtree.
pub(crate) enum Source<'a> {
    While(&'a Expr, &'a Stmt),
    DoWhile(&'a Stmt, &'a Expr),
    For(
        Option<&'a ForInit>,
        Option<&'a Expr>,
        Option<&'a Expr>,
        &'a Stmt,
    ),
    ForInOf(Option<DeclKind>, &'a Pattern, &'a Stmt, bool),
}

impl Source<'_> {
    fn kind(&self) -> RootKind {
        match self {
            Self::While(..) => RootKind::While,
            Self::DoWhile(..) => RootKind::DoWhile,
            Self::For(..) => RootKind::For,
            Self::ForInOf(_, _, _, true) => RootKind::ForOf,
            Self::ForInOf(_, _, _, false) => RootKind::ForIn,
        }
    }

    fn owned(&self) -> (RootKind, Stmt) {
        match self {
            Self::While(test, body) => (
                RootKind::While,
                Stmt::While {
                    test: (*test).clone(),
                    body: Box::new(LoopBody::new((*body).clone())),
                },
            ),
            Self::DoWhile(body, test) => (
                RootKind::DoWhile,
                Stmt::DoWhile {
                    body: Box::new(LoopBody::new((*body).clone())),
                    test: (*test).clone(),
                },
            ),
            Self::For(init, test, update, body) => (
                RootKind::For,
                Stmt::For {
                    init: init.map(|init| Box::new(init.clone())),
                    test: test.cloned(),
                    update: update.cloned(),
                    body: Box::new(LoopBody::new((*body).clone())),
                },
            ),
            Self::ForInOf(decl, left, body, of) => (
                if *of {
                    RootKind::ForOf
                } else {
                    RootKind::ForIn
                },
                Stmt::ForInOf {
                    decl: *decl,
                    left: (*left).clone(),
                    right: Expr::Undefined,
                    of: *of,
                    is_await: false,
                    body: Box::new(LoopBody::new((*body).clone())),
                },
            ),
        }
    }
}

pub(crate) enum Seed<'a> {
    None,
    ForIn(&'a Value, &'a crate::eval::ForInKeys, u32),
    ForOf(&'a Value, &'a Value),
}

/// One bounded admission attempt per dynamic loop execution. Failed compilation leaves the
/// settled AST phase untouched; no source-text or process-global executable cache is involved.
pub(crate) struct Admission {
    remaining: u32,
}

/// Only execution-policy facts are copied. This is not a new ECMAScript execution context:
/// ScriptOrModule, Realm, PrivateEnvironment, FunctionEnvironment (this/super/new.target),
/// VariableEnvironment and the active with/eval chain remain those of the parked evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Context {
    strict: bool,
    async_generator: bool,
    tail_calls: bool,
}

pub(super) struct Plan {
    pub(super) chunk: Rc<Chunk>,
    pub(super) source_bytes: usize,
    pub(super) completion_slot: u16,
    pub(super) seed_slots: Vec<u16>,
    pub(super) active: std::cell::Cell<usize>,
}

impl Context {
    pub(super) fn current(i: &Interp) -> Self {
        Self {
            strict: i.strict,
            async_generator: i.in_async_gen_body && crate::coroutine::in_async_gen(),
            tail_calls: i.tco_ok && !i.using_stack.iter().any(|frame| !frame.is_empty()),
        }
    }
}

impl Admission {
    pub(crate) fn new(i: &Interp) -> Self {
        // The first visit is the initial header, not an executed backedge.
        Self {
            remaining: if matches!(i.tier, Tier::Interp) {
                0
            } else {
                129
            },
        }
    }

    #[inline]
    pub(crate) fn ready(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        self.remaining == 0
    }
}

/// Called only after Admission::ready at an exact settled phase. An error before commitment
/// belongs to the parked AST loop; after Some, the fragment owns all remaining loop cleanup.
pub(crate) fn enter(
    i: &mut Interp,
    site: &LoopSite,
    source: Source<'_>,
    labels: &[&str],
    env: &Env,
    completion: &Value,
    seed: Seed<'_>,
) -> Result<Option<crate::interpreter::Completion>, Abrupt> {
    // Hosts may change tier policy from a callback during this loop. Admission evidence is
    // not authority to compile after an explicit switch to the interpreter-only tier.
    if matches!(i.tier, Tier::Interp) {
        return Ok(None);
    }
    i.interrupt_poll_force()?;
    let token = site.token();
    let realm = i.global.clone();
    let context = Context::current(i);
    let kind = source.kind();
    let plan = match i
        .fragment_cache
        .lookup(token, &realm, kind, context, labels)
    {
        super::fragment_cache::Lookup::Unsupported => return Ok(None),
        super::fragment_cache::Lookup::Plan(plan) => {
            #[cfg(test)]
            TEST_FRAGMENT_CACHE_HITS.with(|counter| counter.set(counter.get() + 1));
            plan
        }
        super::fragment_cache::Lookup::Miss => {
            #[cfg(test)]
            TEST_FRAGMENT_COMPILES.with(|counter| counter.set(counter.get() + 1));
            let started = crate::jit::perf_stage_start();
            let compiled = compile(source, labels, context);
            crate::jit::perf_bytecode_compile_end(started, compiled.is_some());
            i.interrupt_poll_force()?;
            let Some((chunk, completion_slot, seed_slots)) = compiled else {
                i.fragment_cache
                    .unsupported(token, &realm, kind, context, labels);
                return Ok(None);
            };
            let source_bytes = crate::memory::fragment_plan_source_charge(&chunk);
            let plan = Rc::new(Plan {
                chunk,
                source_bytes,
                completion_slot,
                seed_slots,
                active: std::cell::Cell::new(0),
            });
            i.fragment_cache
                .running(token, &realm, kind, context, labels, &plan);
            plan
        }
    };
    plan.active.set(plan.active.get() + 1);
    let chunk = &plan.chunk;
    let completion_slot = plan.completion_slot;
    let seed_slots = &plan.seed_slots;
    #[cfg(test)]
    TEST_FRAGMENT_ENTRIES.with(|counter| counter.set(counter.get() + 1));

    // No make_run_env, ScriptEvaluation, FDI, binding initialization, iterator acquisition or
    // receiver binding occurs here. Rust-owned outer environments remain collector roots.
    let mut current_env = env.clone();
    let cap_env = env.clone();
    let (mut slots, mut stack) = i.vm_pool.pop().unwrap_or_default();
    debug_assert!(slots.is_empty() && stack.is_empty());
    slots.resize_with(chunk.n_slots, || PackedValue::pack(Value::Undefined));
    slots.write_value(completion_slot as usize, completion.clone());
    match seed {
        Seed::None => debug_assert!(seed_slots.is_empty()),
        Seed::ForIn(source, keys, cursor) => {
            slots.write_value(seed_slots[0] as usize, source.clone());
            slots.write_value(seed_slots[1] as usize, keys.clone().into_value());
            slots.write_value(seed_slots[2] as usize, Value::Num(cursor as f64));
        }
        Seed::ForOf(iterator, next) => {
            slots.write_value(seed_slots[0] as usize, iterator.clone());
            slots.write_value(seed_slots[1] as usize, next.clone());
        }
    }
    let mut pc = 0;
    let mut handlers = Vec::new();
    let mut disposal_frames = Vec::new();
    let mut class_states = (0..chunk.class_plans.len())
        .map(|_| None)
        .collect::<Vec<_>>();
    let mut references = (0..chunk.n_refs)
        .map(|_| crate::eval::PreparedReferenceSlot::default())
        .collect::<Vec<_>>();
    let mut tiering = crate::tiering::VmTiering::new();
    // LoadThis is lexical in every fragment, including a not-yet-initialized derived receiver.
    let native = matches!(i.tier, Tier::Jit);
    let result = drive_vm(
        i,
        chunk,
        &mut current_env,
        &cap_env,
        &mut references,
        &mut slots,
        &mut stack,
        &mut pc,
        &Value::Undefined,
        &mut handlers,
        &mut disposal_frames,
        &mut class_states,
        None,
        false,
        native.then_some(&mut tiering),
    );
    slots.clear();
    stack.clear();
    if i.vm_pool.len() < 64 {
        i.vm_pool.push((slots, stack));
    }
    if matches!(&result, Err(Abrupt::Interrupt(_))) {
        // Cancellation must not initiate an extra metadata walk; no structural failure is cached.
        super::fragment_cache::Cache::abandon(&plan);
    } else {
        i.fragment_cache
            .finish(token, &realm, kind, context, labels, &plan);
    }
    let result = match result {
        Ok(VmStep::FragmentExit(FragmentExitKind::Normal, value)) => Ok(value),
        Ok(VmStep::FragmentExit(FragmentExitKind::Break(name), value)) => Err(Abrupt::Break(
            Some(chunk.names[name as usize].to_string()),
            value,
        )),
        Ok(VmStep::FragmentExit(FragmentExitKind::Continue(name), value)) => Err(Abrupt::Continue(
            Some(chunk.names[name as usize].to_string()),
            value,
        )),
        Ok(VmStep::Return(value) | VmStep::ResumeReturn(value)) => Err(Abrupt::Return(value)),
        Ok(VmStep::BareReturn) => Err(Abrupt::Return(Value::Undefined)),
        Err(error) => Err(error),
        Ok(_) => unreachable!("synchronous loop fragment escaped without a typed completion"),
    };
    Ok(Some(result))
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_FRAGMENT_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static TEST_FRAGMENT_COMPILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static TEST_FRAGMENT_CACHE_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn compile(
    source: Source<'_>,
    labels: &[&str],
    context: Context,
) -> Option<(Rc<Chunk>, u16, Vec<u16>)> {
    let (kind, source) = source.owned();
    let mut c = Compiler {
        // Fragments borrow a tree-walker frame and never know its function. Self-hosted
        // built-ins compile on their first call, so they reach the tree-walker only in the
        // interpreter-only tier, which compiles no fragments; their intrinsics' native forms
        // would in any case have the same semantics.
        self_hosted: false,
        lean_new_target: false,
        rest_slot: None,
        arguments_length_only: false,
        strict: context.strict,
        fragment_entry: true,
        fragment_root: Some(kind),
        lexical_this: true,
        fragment_async_generator: context.async_generator,
        fragment_tail_calls: context.tail_calls,
        direct_eval: true,
        reuse_activation: true,
        pending_labels: labels.iter().map(|label| (*label).to_owned()).collect(),
        ..Compiler::default()
    };
    c.push_compile_scope();
    // A non-source control context anchors external cleanup at handler depth zero. It cannot
    // capture unlabelled break/continue, and is never a runtime destination itself.
    c.loops.push(LoopCtx {
        is_label_block: true,
        ..LoopCtx::default()
    });
    let completion = c.fresh_slot("%fragment-completion%");
    c.script_completion = Some(completion);
    c.stmt(&source).ok()?;
    c.emit(Op::LoadLocal(completion));
    c.emit(Op::FragmentExit(FragmentExitKind::Normal));
    for (jump, kind) in std::mem::take(&mut c.fragment_exits) {
        c.patch(jump);
        c.emit(Op::LoadLocal(completion));
        c.emit(Op::FragmentExit(kind));
    }
    // Actual own suspension cannot cross a parked Rust AST activation. Nested functions are
    // opaque MakeClosure constants, so async/generator declarations do not disqualify a loop.
    if c.ops.iter().any(|op| {
        matches!(
            op,
            Op::Await
                | Op::Yield
                | Op::YieldStar
                | Op::GetAsyncIter
                | Op::AsyncIterStepL(..)
                | Op::AsyncIterCloseL(..)
                | Op::AddDisposable(true)
        )
    }) {
        return None;
    }
    let seeds = std::mem::take(&mut c.fragment_seed_slots);
    Some((finish_chunk(c, None, true, false, None)?, completion, seeds))
}

impl Compiler {
    pub(super) fn fragment_using_for(
        &mut self,
        labels: Vec<String>,
        kind: DeclKind,
        decls: &[(Pattern, Option<Expr>)],
        test: Option<&Expr>,
        update: Option<&Expr>,
        body: &Stmt,
    ) -> CResult {
        let mut bindings = Vec::new();
        for (pattern, _) in decls {
            let Pattern::Ident(name) = pattern else {
                return Err(Bail);
            };
            bindings.push((name.clone(), true));
        }
        let scope = self.push_runtime_lexical_scope(bindings);
        self.environment_scope(
            |compiler| {
                compiler.disposal_scope(|compiler| {
                    for (pattern, initializer) in decls {
                        let Pattern::Ident(name) = pattern else {
                            unreachable!()
                        };
                        compiler.named_expr(initializer.as_ref().ok_or(Bail)?, name)?;
                        compiler.emit(Op::AddDisposable(matches!(kind, DeclKind::AwaitUsing)));
                        let name = compiler.name_idx(name);
                        compiler.emit(Op::InitLex(name));
                    }
                    compiler.without_tail(|compiler| {
                        compiler.for_loop_core(labels, None, test, update, body, None)
                    })
                })
            },
            scope,
        )
    }

    pub(super) fn fragment_external_exit(&mut self, name: &str, is_continue: bool) -> CResult {
        if !self.fragment_entry {
            return Err(Bail);
        }
        let name = self.name_idx(name);
        // Local finally/disposal/iterator cleanup uses the same real-PC landing pads as any
        // ordinary jump. Only the final terminal returns a labelled Completion to the AST.
        let jump = self.emit_exit_jump(0, is_continue)?;
        self.fragment_exits.push((
            jump,
            if is_continue {
                FragmentExitKind::Continue(name)
            } else {
                FragmentExitKind::Break(name)
            },
        ));
        Ok(())
    }

    pub(super) fn fragment_for_remainder(
        &mut self,
        init: Option<&ForInit>,
        test: Option<&Expr>,
        update: Option<&Expr>,
        body: &Stmt,
    ) -> CResult {
        let labels = std::mem::take(&mut self.pending_labels);
        let per_iteration = if let Some(ForInit::VarDecl {
            kind: DeclKind::Let,
            decls,
        }) = init
        {
            let mut names = crate::fasthash::FastSet::default();
            for (pattern, _) in decls {
                pat_idents(pattern, &mut names);
            }
            let mut names: Vec<_> = names.into_iter().collect();
            names.sort();
            // The AST already owns this exact per-iteration Environment Record. Register its
            // binding schema without PushLex or the initial CreatePerIterationEnvironment.
            let index = self.lexical_scopes.len() as u32;
            for name in &names {
                self.lexical_env_bind(name, false);
            }
            self.lexical_scopes.push(
                names
                    .into_iter()
                    .map(|name| LexicalBinding {
                        name: Rc::from(name),
                        is_const: false,
                    })
                    .collect(),
            );
            Some(index)
        } else {
            None
        };
        let start = self.ops.len();
        let exit = if let Some(test) = test {
            self.expr(test)?;
            Some(self.emit(Op::JumpIfFalse(0)))
        } else {
            None
        };
        self.loops.push(LoopCtx {
            labels,
            entry_try_depth: self.try_depth,
            ..LoopCtx::default()
        });
        self.stmt(body)?;
        let context = self.loops.pop().expect("fragment for context");
        for jump in context.continues {
            self.patch(jump);
        }
        if let Some(scope) = per_iteration {
            self.emit(Op::CloneLex(scope));
        }
        if let Some(update) = update {
            self.expr_stmt(update)?;
        }
        self.emit(Op::Jump(start as u32));
        if let Some(exit) = exit {
            self.patch(exit);
        }
        for jump in context.breaks {
            self.patch(jump);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_implicit_async_generator_await_is_an_own_suspension() {
        for (body, eligible) in [
            ("return 17", false),
            ("return", true),
            ("async function nested(){await 1} break", true),
            ("yield 17", false),
            ("await 17", false),
        ] {
            let source = format!("async function* f(){{while(true){{{body};}}}}");
            let ast = crate::parser::parse_script(&source, false)
                .ok()
                .expect("generator parses");
            let Stmt::FuncDecl(function) = &ast[0] else {
                panic!("function")
            };
            let Stmt::While { test, body, .. } = &function.body[0] else {
                panic!("loop")
            };
            let compiled = compile(
                Source::While(test, body),
                &[],
                Context {
                    strict: true,
                    async_generator: true,
                    tail_calls: false,
                },
            );
            assert_eq!(compiled.is_some(), eligible, "{source}");
            if let Some((chunk, _, _)) = compiled {
                assert!(!chunk.resumable);
                for (pc, op) in chunk.ops.iter().enumerate() {
                    if matches!(op, Op::FragmentExit(_)) {
                        assert_eq!(chunk.jit_stack_effect(pc), None);
                        assert_eq!(chunk.jit_borrowed_stack_effect(pc), Some((1, 0)));
                    }
                }
            }
        }
    }

    #[test]
    fn fragment_precompile_cancellation_keeps_parked_state_uncommitted() {
        let mut engine = crate::Engine::new();
        engine.set_tier(Tier::Bytecode);
        let env = engine.interp.global_env.clone();
        let source = crate::parser::parse_script("while(true){throw 17}", false)
            .ok()
            .expect("loop parses");
        let Stmt::While { test, body } = &source[0] else {
            panic!("loop")
        };
        let site = &body.site;
        TEST_FRAGMENT_ENTRIES.with(|counter| counter.set(0));
        engine
            .interp
            .runtime_interrupt
            .set_deadline(Some(std::time::Instant::now()));
        assert!(matches!(
            enter(
                &mut engine.interp,
                site,
                Source::While(test, body),
                &[],
                &env,
                &Value::Num(23.0),
                Seed::None
            ),
            Err(Abrupt::Interrupt(_))
        ));
        assert_eq!(TEST_FRAGMENT_ENTRIES.with(|counter| counter.get()), 0);
        engine.interp.runtime_interrupt.set_deadline(None);
        assert!(matches!(
            enter(
                &mut engine.interp,
                site,
                Source::While(test, body),
                &[],
                &env,
                &Value::Num(23.0),
                Seed::None
            ),
            Ok(Some(Err(Abrupt::Throw(Value::Num(17.0)))))
        ));
        assert_eq!(TEST_FRAGMENT_ENTRIES.with(|counter| counter.get()), 1);
    }
}
