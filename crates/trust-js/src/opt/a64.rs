//! The optimizing tier's ARM64 code generator (P3 stage 1; the plan's Design points 6, 10, 11 and
//! 14 and the "Stage 1" record's back end). A child of `jit`, as the regions and the operation
//! emitters are: it emits through their encoder, templates and frame (`opt::a64` names it).
//!
//! [`compile`] turns a graph over a chunk into a FreshFrame [`JitCode`] that every existing
//! caller enters unchanged (contracts 5.4): the first stage's prologue and epilogues
//! (`emit_entry_prologue`, `emit_entry_epilogues`), with the frame area below the header holding
//! the spill slots and the saves of the x23-x28 the allocation uses; the first stage's
//! `max_stack` and `needs_global`.
//!
//! Registers between nodes (see `regalloc::ARM64`): x19 the context, x21 the helper table and
//! x22 the slots, pinned; x20 set to `stack_base + 8 * depth(pc)` before each effect and
//! otherwise unused; x9, x16, x17 and x30 scratch, d30 and d31 FP scratch; x18 unused; x29 the
//! caller's. A value's register holds a tagged word (Tagged), an object's cell address (Cell),
//! 0 or 1 (Bool) or an int32 in its low 32 bits (I32), or an f64 in a d register.
//!
//! Effects run the operation's baseline template on the canonical frame (contracts 4): the
//! operands written to the canonical words below `stack_base + 8 * depth(pc)`, x20 set there,
//! the template emitted through `operations::emit` (`emit_exec` where it declines) with the
//! tier's unwind, the results loaded from below the returned x20 by the projections that follow.
//! A throw leaves through the throw epilogue (ctx.error is the helper's), or, at a pc with a
//! handler, exits to the VM with the throw pending; for that the code writes the effect's whole
//! frame state to the frame before the effect, since the template clobbers the caller-saved
//! registers. Values living across an effect have callee-saved homes or spill slots
//! (`regalloc`), which templates and helpers preserve.
//!
//! Exits (`opt::exit`): each failing check branches to its exit's site, one `bl` to the code's
//! shared exit stub placed after the body; the stub saves x0-x29 and d0-d31, calls the
//! deoptimizer with the return address, which names the exit, and leaves through the epilogue
//! the completion selects. The code embeds its exit table's address; the table is freed with the
//! code. The code's `pc_offsets` name a trap: no pc of this code is enterable, and no unwinder
//! or completion router lands in it (its handlers live in its exits' recipes).

use super::*;
use crate::bytecode::Op;
use crate::opt::exit::{ExitKind, ExitRecord, ExitTable, Source, SAVE_BYTES};
use crate::opt::ir::*;
use crate::opt::regalloc::{self, Allocation, Home};

// Condition codes.
const EQ: u32 = 0;
const NE: u32 = 1;
const LO: u32 = 3;
const MI: u32 = 4;
const VS: u32 = 6;
const VC: u32 = 7;
const HI: u32 = 8;
const LS: u32 = 9;
const GE: u32 = 10;
const LT: u32 = 11;
const GT: u32 = 12;
const LE: u32 = 13;

// Scratch registers.
const X9: u32 = 9;
const X16: u32 = 16;
const X17: u32 = 17;
const X30: u32 = 30;
const D30: u32 = 30;
const D31: u32 = 31;

/// What a code generation includes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Options {
    /// The forced-exit gate's countdown at every check and effect (TRUST_JS_OPT_EXIT_EVERY).
    pub(crate) forced_exits: bool,
    /// The sequences for a core without FEAT_JSCVT where it has it (tests; the switch
    /// TRUST_JS_JIT_NO_JSCVT does the same for a process through jscvt_available).
    pub(crate) no_jscvt: bool,
}

impl Options {
    /// The options the switches select.
    pub(crate) fn from_env() -> Self {
        Options {
            forced_exits: crate::opt::exit_every() > 0,
            no_jscvt: false,
        }
    }
}

/// Generate `graph`'s code for `chunk` (the switches' options).
pub(crate) fn compile(
    chunk: &Chunk,
    graph: &Graph,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
) -> Result<Rc<JitCode>, Refusal> {
    compile_with(chunk, graph, layout, ilayout, Options::from_env())
}

/// Generate `graph`'s code for `chunk` with `options`.
pub(crate) fn compile_with(
    chunk: &Chunk,
    graph: &Graph,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    options: Options,
) -> Result<Rc<JitCode>, Refusal> {
    if !layout.valid || !ilayout.valid {
        return Err(Refusal::whole("no validated object or interpreter layout"));
    }
    let ops = chunk.jit_ops();
    if ops.len() > 0xFFFF {
        return Err(Refusal::whole("a chunk past 65,535 operations"));
    }
    if chunk.jit_is_resumable() {
        return Err(Refusal::whole("a resumable chunk"));
    }
    let cfg = crate::jit_ir::Cfg::build(chunk)
        .map_err(|_| Refusal::whole("the chunk's stack depths do not settle"))?;
    check_graph(chunk, graph, &cfg)?;
    let alloc = regalloc::allocate(graph, &regalloc::ARM64)?;
    let frame = entry_frame(&alloc)?;
    let mut gen = Gen::new(chunk, graph, alloc, layout, ilayout, cfg, options, frame);
    gen.emit()?;
    gen.finish()
}

/// The checks that keep the code generator's assumptions: one entry without predecessors, a
/// terminator ending every block and nowhere else, a poll opening every loop header, entry
/// slots read in the entry block, a state on every node that can exit, and effects only on
/// operations whose baseline template the canonical frame can run.
fn check_graph(chunk: &Chunk, graph: &Graph, cfg: &crate::jit_ir::Cfg) -> Result<(), Refusal> {
    let entry = graph
        .entry
        .ok_or(Refusal::whole("a graph without an entry block"))?;
    if !graph.block(entry).preds.is_empty() {
        return Err(Refusal::whole("an entry block with predecessors"));
    }
    let ops = chunk.jit_ops();
    for (index, block) in graph.blocks.iter().enumerate() {
        let Some((&last, rest)) = block.body.split_last() else {
            return Err(Refusal::whole("an empty block"));
        };
        if !graph.node(last).op.is_terminator()
            || rest.iter().any(|&n| graph.node(n).op.is_terminator())
        {
            return Err(Refusal::at(
                block.first_pc as usize,
                "a block not ended by one terminator",
            ));
        }
        if block.loop_header
            && !matches!(block.body.first(), Some(&n) if graph.node(n).op == NodeOp::Poll)
        {
            return Err(Refusal::at(
                block.first_pc as usize,
                "a loop header without a poll",
            ));
        }
        for &n in &block.body {
            let node = graph.node(n);
            if node.op.can_exit() && node.state.is_none() {
                return Err(Refusal::at(
                    node.pc as usize,
                    "an exit without a frame state",
                ));
            }
            match node.op {
                NodeOp::EntrySlot(_) if index != entry.index() => {
                    return Err(Refusal::at(
                        node.pc as usize,
                        "an entry slot outside the entry block",
                    ));
                }
                NodeOp::Effect { pc, pops, pushes } => {
                    let op = ops
                        .get(pc as usize)
                        .ok_or(Refusal::at(pc as usize, "an effect past the chunk"))?;
                    if !effect_supported(op) {
                        return Err(Refusal::at(
                            pc as usize,
                            "an effect on a control-flow operation",
                        ));
                    }
                    let depth = cfg
                        .stack_depth_at(pc as usize)
                        .ok_or(Refusal::at(pc as usize, "an effect at an unreachable pc"))?;
                    if (pops as usize) > depth || node.inputs.len() != pops as usize {
                        return Err(Refusal::at(pc as usize, "an effect's operands"));
                    }
                    if let Some(after) = cfg.stack_depth_at(pc as usize + 1) {
                        if !matches!(op, Op::Throw)
                            && depth - pops as usize + pushes as usize != after
                        {
                            return Err(Refusal::at(pc as usize, "an effect's stack effect"));
                        }
                    }
                }
                NodeOp::CheckShape(ref shapes)
                    if shapes.is_empty()
                        || shapes.len() > 4
                        || !shapes.iter().all(|&s| crate::value::is_cacheable_shape(s)) =>
                {
                    return Err(Refusal::at(
                        node.pc as usize,
                        "a shape check without proving shapes",
                    ));
                }
                _ => {}
            }
            if let Some(state) = node.state {
                // The VM resumes at the state's pc on the settled stack there.
                let state = graph.state(state);
                if cfg.stack_depth_at(state.pc as usize) != Some(state.stack.len()) {
                    return Err(Refusal::at(
                        state.pc as usize,
                        "a frame state's operand depth",
                    ));
                }
                if state.slots.len() != graph.n_slots {
                    return Err(Refusal::at(state.pc as usize, "a frame state's slots"));
                }
                // A record's depth is the operand depth at its push, which its region may run
                // below: a for-of pushes its handler with the step's value on the stack, and the
                // binding's store pops it. An unwind keeps the smaller depth, as the VM's and the
                // first stage's do (and the deoptimizer's for a pending throw).
                let pushes_handler = |&(push_pc, depth): &(u32, u32)| {
                    matches!(
                        ops.get(push_pc as usize),
                        Some(Op::PushHandler(_) | Op::PushFinally(..) | Op::PushIterator(..))
                    ) && cfg.stack_depth_at(push_pc as usize) == Some(depth as usize)
                };
                if !state.handlers.iter().all(pushes_handler) {
                    return Err(Refusal::at(state.pc as usize, "a frame state's handlers"));
                }
            }
        }
        for &phi in &block.phis {
            let node = graph.node(phi);
            if node.inputs.len() != block.preds.len()
                || node
                    .inputs
                    .iter()
                    .any(|&input| graph.node(input).rep != node.rep)
            {
                return Err(Refusal::at(block.first_pc as usize, "a phi's inputs"));
            }
        }
    }
    Ok(())
}

/// Whether an operation can run as an effect: its baseline template, or `jit_exec`, runs it on
/// the canonical frame at its pc (contracts 4.2). Control flow cannot.
fn effect_supported(op: &Op) -> bool {
    !matches!(
        op,
        Op::Jump(_)
            | Op::AbruptJump(..)
            | Op::JumpIfFalse(_)
            | Op::JumpIfFalsePeek(_)
            | Op::JumpIfTruePeek(_)
            | Op::JumpIfNotNullishPeek(_)
            | Op::InlineGuard(..)
            | Op::Return
            | Op::ReturnBare
            | Op::ReturnUndef
            | Op::ResumeReturn
            | Op::ResumeJump
            | Op::Await
            | Op::Yield
            | Op::YieldStar
            | Op::PushHandler(_)
            | Op::PushFinally(..)
            | Op::PushIterator(..)
            | Op::PopHandler
            | Op::FragmentExit(_)
            | Op::AsyncIterStepL(..)
            | Op::AsyncIterCloseL(..)
            | Op::DisposeNormal
            | Op::DisposeThrow
            | Op::DisposeReturn
            | Op::DisposeBareReturn
            | Op::DisposeResumeReturn
            | Op::DisposeJump
            | Op::PushDisposeFrame
            | Op::AddDisposable(_)
            | Op::TailEvalCallArgsArray
    )
}

/// The frame area below the header: the spill slots, then the saves of the callee-saved x
/// registers the allocation uses, in pairs.
fn entry_frame(alloc: &Allocation) -> Result<EntryFrame, Refusal> {
    let mut saves = Vec::new();
    let mut offset = alloc.spill_bytes.next_multiple_of(16);
    for pair in alloc.callee_saved_gp.chunks(2) {
        saves.push((
            u32::from(pair[0]),
            pair.get(1).map(|&r| u32::from(r)),
            offset,
        ));
        offset += 16;
    }
    if offset >= 4096 {
        return Err(Refusal::whole("a frame area past 4,095 bytes"));
    }
    Ok(EntryFrame {
        extra: offset,
        saves,
    })
}

/// Out-of-line code, emitted after the body.
enum Cold {
    /// CheckNumber's NaN case: a NaN word is a Number only when it is the canonical NaN.
    NanNumber {
        label: usize,
        word: u32,
        exit: usize,
        resume: usize,
    },
    /// TruncateToInt32 without FEAT_JSCVT, when the conversion to 64 bits saturated: |x| is 2^63
    /// or more, or infinite. The low word of ToInt32 into `dst` from the f64 in `d`.
    Truncate {
        label: usize,
        d: u32,
        dst: u32,
        resume: usize,
    },
    /// A poll's slow path.
    Poll {
        label: usize,
        node: NodeIx,
        resume: usize,
        throw: usize,
    },
    /// ToBoolean of a BigInt or of an object that may be [[IsHTMLDDA]]: the word in `word`, the
    /// Boolean to `dst`.
    ToBoolean {
        label: usize,
        node: NodeIx,
        word: u32,
        dst: u32,
        resume: usize,
    },
    /// The moves of an edge into a merge from a block with two successors.
    Edge {
        label: usize,
        from: BlockIx,
        to: BlockIx,
    },
    /// A branch to a label (an effect's throw into its exit).
    Jump { label: usize, target: usize },
}

/// A location of a parallel move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    Gp(u32),
    Fp(u32),
    Spill(u32),
    /// A constant node's value.
    Const(NodeIx),
}

struct Gen<'a> {
    a: asm::Asm,
    chunk: &'a Chunk,
    graph: &'a Graph,
    alloc: Allocation,
    layout: &'a crate::value::JitLayout,
    ilayout: &'a crate::interpreter::InterpLayout,
    cfg: crate::jit_ir::Cfg,
    options: Options,
    frame: EntryFrame,
    blocks: Vec<usize>,
    l_ret_ok: usize,
    l_ret_throw: usize,
    l_direct_finish: usize,
    l_exit_stub: usize,
    l_trap: usize,
    l_dsave: usize,
    /// The epilogues' d8..d15 restore groups.
    drestore: (usize, usize),
    /// (site label, return label, record) per exit.
    exits: Vec<(usize, usize, ExitRecord)>,
    cold: Vec<Cold>,
    fast: u32,
    array_intrinsics_on: bool,
    function_call_intrinsic_on: bool,
    direct_on: bool,
    in_loop: Vec<bool>,
    /// The effect the projections that follow it read.
    last_effect: Option<NodeIx>,
    /// The exit table, whose address the exit stub embeds.
    table: Box<ExitTable>,
}

impl<'a> Gen<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        chunk: &'a Chunk,
        graph: &'a Graph,
        alloc: Allocation,
        layout: &'a crate::value::JitLayout,
        ilayout: &'a crate::interpreter::InterpLayout,
        cfg: crate::jit_ir::Cfg,
        options: Options,
        frame: EntryFrame,
    ) -> Self {
        let ops = chunk.jit_ops();
        let mut a = asm::Asm::new();
        a.set_stubs_ok(true);
        a.set_first_stage(!chunk.jit_second_stage());
        a.set_fixed_activation(chunk.jit_fixed_activation());
        if !env_flag!("TRUST_JS_JIT_NO_TDZ_PROOF") {
            a.set_initialized_reads(cfg.initialized_reads(ops, true));
        }
        let fast = template_mask(chunk);
        let blocks = (0..graph.blocks.len()).map(|_| a.new_label()).collect();
        let l_ret_ok = a.new_label();
        let l_ret_throw = a.new_label();
        let l_direct_finish = a.new_label();
        let l_exit_stub = a.new_label();
        let l_trap = a.new_label();
        let name = chunk
            .jit_slot_names()
            .iter()
            .take(4)
            .map(|s| &**s)
            .collect::<Vec<_>>()
            .join(",");
        Gen {
            a,
            chunk,
            graph,
            alloc,
            layout,
            ilayout,
            cfg,
            options,
            frame,
            blocks,
            l_ret_ok,
            l_ret_throw,
            l_direct_finish,
            l_exit_stub,
            l_trap,
            l_dsave: 0,
            drestore: (0, 0),
            exits: Vec::new(),
            cold: Vec::new(),
            fast,
            array_intrinsics_on: !env_flag!("TRUST_JS_JIT_NO_ARRAY_INTRINSICS"),
            function_call_intrinsic_on: !env_flag!("TRUST_JS_JIT_NO_FUNCTION_CALL_INTRINSIC"),
            direct_on: fast & (1 << 20) != 0 && crate::bytecode::direct_shared_context_enabled(),
            in_loop: loop_body_mask(ops),
            last_effect: None,
            table: Box::new(ExitTable {
                name: format!("fn({name})").into_boxed_str(),
                limit: crate::opt::exit_limit(chunk),
                ..ExitTable::default()
            }),
        }
    }

    fn emit(&mut self) -> Result<(), Refusal> {
        // The entry's throws (the poll, the frame setup) find no handler at pc 0.
        self.l_dsave = emit_entry_prologue(
            &mut self.a,
            self.chunk,
            false,
            &self.frame,
            self.l_ret_throw,
        );
        self.a.set_end_polls(true);
        let order = self.alloc.order.clone();
        for (k, &b) in order.iter().enumerate() {
            let next = order.get(k + 1).copied();
            self.a.bind(self.blocks[b.index()]);
            self.last_effect = None;
            for &n in &self.graph.block(b).body {
                self.lower(n, b, next)?;
            }
        }
        self.emit_cold()?;
        self.drestore =
            emit_entry_epilogues(&mut self.a, &self.frame, self.l_ret_ok, self.l_ret_throw);
        if !self.exits.is_empty() {
            self.emit_exit_stub();
        }
        self.a.bind(self.l_trap);
        self.a.brk(0x4f50);
        // The shared stubs the effects' templates requested, and the direct-call teardown.
        emit_shared_stubs(
            &mut self.a,
            &StubContext {
                layout: self.layout,
                ilayout: Some(self.ilayout),
                direct: self.direct_on.then(|| DirectCallContext {
                    attempted_off: self.chunk.jit_inline_attempted_off(),
                    runs_off: self.chunk.jit_runs_off(),
                    retry_off: self.chunk.jit_inline_retry_at_off(),
                    finish_stub: self.l_direct_finish,
                }),
            },
        );
        self.a.bind(self.l_direct_finish);
        if self.direct_on && self.a.label_referenced(self.l_direct_finish) {
            emit_direct_finish_stub(&mut self.a, self.ilayout, self.layout.valid);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Rc<JitCode>, Refusal> {
        if self.alloc.fp_callee_across_effect
            && STUB_WRITES_CALLEE_SAVED_FP.load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(Refusal::whole(
                "a shared stub writes d8-d15, which hold values across effects",
            ));
        }
        let ops = self.chunk.jit_ops();
        let mut exported = vec![self.l_trap];
        exported.extend(self.exits.iter().map(|&(_, ret, _)| ret));
        let (l_drestore_ok, l_drestore_throw) = self.drestore;
        exported.extend([self.l_dsave, l_drestore_ok, l_drestore_throw]);
        let fp_callee_saved = self.a.writes_callee_saved_fp();
        if self.a.failed().is_some() {
            return Err(Refusal::whole("an operand no instruction encodes"));
        }
        let a = std::mem::replace(&mut self.a, asm::Asm::new());
        let (mut words, offsets) = a.finish_with_offsets(&exported);
        let n_exits = self.exits.len();
        let trap = offsets[0];
        let fp_saves: Vec<usize> = offsets[1 + n_exits..]
            .iter()
            .map(|&offset| offset as usize / 4)
            .collect();
        skip_unused_callee_saved_fp(&mut words, &fp_saves, fp_callee_saved);
        let mut exits: Vec<(u32, ExitRecord)> = offsets[1..1 + n_exits]
            .iter()
            .copied()
            .zip(self.exits.drain(..).map(|(_, _, record)| record))
            .collect();
        exits.sort_by_key(|&(offset, _)| offset);
        let mut table = self.table;
        table.return_offsets = exits.iter().map(|&(offset, _)| offset).collect();
        table.exits = exits.into_iter().map(|(_, record)| record).collect();
        if let Some(pat) = env_value!("TRUST_JS_JIT_CODEDUMP") {
            if pat == "*" || table.name.contains(pat) {
                eprintln!("[opt-codedump] {} {} words", table.name, words.len());
                for w in &words {
                    eprintln!("[opt-codedump] {w:08x}");
                }
            }
        }
        let executable = ExecutableBuffer::from_bytes(unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4)
        })
        .ok_or(Refusal::whole("no executable memory"))?;
        let mem = executable.as_ptr() as *mut u8;
        let len = executable.len();
        table.code_base = mem as usize;
        crate::opt::exit::register(mem as usize, table);
        let code = JitCode {
            entry_kind: NativeEntryKind::FreshFrame,
            osr_entry_depths: Vec::new(),
            resume_depths: Vec::new(),
            needs_global: ops.iter().any(|o| {
                matches!(
                    o,
                    Op::LoadName(..)
                        | Op::LoadNameForCall(..)
                        | Op::LoadNameIn(..)
                        | Op::LoadNameForCallIn(..)
                )
            }),
            mem,
            len,
            pc_offsets: vec![trap; ops.len()],
            max_stack: self.cfg.jit_stack_capacity(),
            executable,
            referenced: Cell::new(0),
        };
        profiler::register_optimized(self.chunk, mem, len);
        Ok(Rc::new(code))
    }
}

/// The runtime's `%` on Numbers (ECMA-262 Number::remainder: the C library's fmod, the result
/// taking the dividend's sign), which F64Mod calls.
/// ToBoolean's slow path (`lower_to_boolean`: a BigInt, an object that may be [[IsHTMLDDA]]): the
/// interpreter's ToBoolean, which runs no JavaScript and does not collect.
unsafe extern "C" fn to_boolean(ctx: *mut JitCtx, word: u64) -> u64 {
    let i = unsafe { &*(*ctx).interp };
    u64::from(i.to_boolean(&unsafe { crate::value::Value::from_bits(word) }))
}

extern "C" fn f64_remainder(dividend: f64, divisor: f64) -> f64 {
    dividend % divisor
}

/// The exit reason a check reports when its node names none.
fn default_reason(op: &NodeOp) -> ExitReason {
    match op {
        NodeOp::CheckNumber => ExitReason::NotNumber,
        NodeOp::CheckInt32 => ExitReason::NotInt32,
        NodeOp::CheckObject => ExitReason::NotObject,
        NodeOp::CheckShape(_) => ExitReason::WrongShape,
        NodeOp::CheckBoolean => ExitReason::NotBoolean,
        NodeOp::CheckCallee { .. } => ExitReason::WrongCallee,
        NodeOp::I32Add | NodeOp::I32Sub | NodeOp::I32Mul | NodeOp::I32UShr => ExitReason::Overflow,
        NodeOp::Exit(reason) => *reason,
        _ => ExitReason::Forced,
    }
}

impl Gen<'_> {
    fn home(&self, v: NodeIx) -> Home {
        self.alloc.homes[v.index()]
    }

    /// A constant node's execution word (what the frame holds for it).
    fn const_word(&self, v: NodeIx) -> u64 {
        use crate::value::Value;
        match self.graph.node(v).op {
            NodeOp::ConstTagged(bits) => bits,
            NodeOp::ConstF64(bits) => Value::Num(f64::from_bits(bits)).bits(),
            NodeOp::ConstI32(x) => Value::Num(f64::from(x)).bits(),
            NodeOp::ConstBool(b) => Value::Bool(b).bits(),
            _ => unreachable!("a constant node"),
        }
    }

    /// A constant node's bits in its own representation (what a register of its class holds).
    fn const_bits(&self, v: NodeIx) -> u64 {
        match self.graph.node(v).op {
            NodeOp::ConstTagged(bits) | NodeOp::ConstF64(bits) => bits,
            NodeOp::ConstI32(x) => u64::from(x as u32),
            NodeOp::ConstBool(b) => u64::from(b),
            _ => unreachable!("a constant node"),
        }
    }

    /// The x register that holds `v` here: its home, or `scratch`, loaded from its spill slot or
    /// with its constant.
    fn gp(&mut self, v: NodeIx, scratch: u32) -> u32 {
        match self.home(v) {
            Home::Gp(r) => u32::from(r),
            Home::Spill(offset) => {
                self.a.ldr_imm(scratch, 31, offset);
                scratch
            }
            Home::Const => {
                let bits = self.const_bits(v);
                self.a.mov_imm64(scratch, bits);
                scratch
            }
            Home::Fp(_) | Home::None => unreachable!("a read value has an x home"),
        }
    }

    /// The d register that holds `v` here (as [`Gen::gp`]; a constant goes through x30).
    fn fp(&mut self, v: NodeIx, scratch: u32) -> u32 {
        match self.home(v) {
            Home::Fp(r) => u32::from(r),
            Home::Spill(offset) => {
                self.a.ldr_d_imm(scratch, 31, offset);
                scratch
            }
            Home::Const => {
                let bits = self.const_bits(v);
                if bits == 0 {
                    self.a.fmov_d_x(scratch, 31); // xzr
                } else {
                    self.a.mov_imm64(X30, bits);
                    self.a.fmov_d_x(scratch, X30);
                }
                scratch
            }
            Home::Gp(_) | Home::None => unreachable!("a read value has a d home"),
        }
    }

    /// `v`'s execution word in an x register (`scratch` unless it is a tagged word in its home):
    /// a Number for an f64 (NaN canonicalized) or an int32, the Boolean or Object word for a Bool
    /// or a cell. Clobbers x30, d31 and the flags.
    fn boxed(&mut self, v: NodeIx, scratch: u32) -> u32 {
        if self.home(v) == Home::Const {
            let word = self.const_word(v);
            self.a.mov_imm64(scratch, word);
            return scratch;
        }
        match self.graph.node(v).rep {
            Rep::Tagged => self.gp(v, scratch),
            Rep::F64 => {
                let d = self.fp(v, D31);
                self.a.fmov_x_d(scratch, d);
                self.a.fcmp(d, d);
                self.a.movz(X30, 0x7ff8, 3); // PACK_CANON_NAN
                self.a.csel_x(scratch, X30, scratch, VS);
                scratch
            }
            Rep::I32 => {
                let w = self.gp(v, scratch);
                self.a.scvtf_d_w(D31, w);
                self.a.fmov_x_d(scratch, D31);
                scratch
            }
            Rep::Bool => {
                let b = self.gp(v, scratch);
                self.a.movz(X30, (crate::value::PACK_BOOL >> 48) as u32, 3);
                self.a.logic_x(1, scratch, X30, b);
                scratch
            }
            Rep::Cell => {
                let c = self.gp(v, scratch);
                self.a.movz(X30, (crate::value::PACK_OBJ >> 48) as u32, 3);
                self.a.logic_x(1, scratch, X30, c);
                scratch
            }
            Rep::Ptr => unreachable!("a frame state holds no address (verify.rs)"),
            Rep::None => unreachable!("a frame state holds values"),
        }
    }

    /// Where `n` writes its x value: its home register, or x9 for a spill slot or no home.
    /// Whether to use FJCVTZS (ToInt32 itself, FEAT_JSCVT).
    fn jscvt(&self) -> bool {
        jscvt_available() && !self.options.no_jscvt
    }

    fn gp_dst(&self, n: NodeIx) -> u32 {
        match self.home(n) {
            Home::Gp(r) => u32::from(r),
            _ => X9,
        }
    }

    /// Where `n` writes its d value: its home register, or d30.
    fn fp_dst(&self, n: NodeIx) -> u32 {
        match self.home(n) {
            Home::Fp(r) => u32::from(r),
            _ => D30,
        }
    }

    /// Spill `n`'s value at its definition, when its home is a slot.
    fn gp_done(&mut self, n: NodeIx, reg: u32) {
        if let Home::Spill(offset) = self.home(n) {
            self.a.str_imm(reg, 31, offset);
        }
    }

    fn fp_done(&mut self, n: NodeIx, reg: u32) {
        if let Home::Spill(offset) = self.home(n) {
            self.a.str_d_imm(reg, 31, offset);
        }
    }

    /// Where an exit finds `v`.
    fn source(&self, v: NodeIx, pc: u32) -> Result<Source, Refusal> {
        let rep = self.graph.node(v).rep;
        Ok(match self.home(v) {
            Home::Gp(reg) => Source::Gp { reg, rep },
            Home::Fp(reg) => Source::Fp { reg },
            Home::Spill(offset) => Source::Spill { offset, rep },
            Home::Const => Source::Const(self.const_word(v)),
            Home::None => {
                return Err(Refusal::at(
                    pc as usize,
                    "a frame state's value without a home",
                ));
            }
        })
    }

    /// A new exit with `n`'s frame state, of `kind`: the label of its site, where a failing
    /// check branches.
    fn exit(&mut self, n: NodeIx, kind: ExitKind) -> Result<usize, Refusal> {
        let graph = self.graph;
        let node = graph.node(n);
        let state = graph.state(node.state.expect("checked: a state on every exit"));
        let reason = match kind {
            ExitKind::Check => node.reason.unwrap_or_else(|| default_reason(&node.op)),
            ExitKind::PollThrow | ExitKind::EffectThrow => ExitReason::Throw,
        };
        let mut slots = Vec::new();
        for (slot, source) in state.slots.iter().enumerate() {
            if let SlotSource::Value(v) = *source {
                slots.push((slot as u16, self.source(v, state.pc)?));
            }
        }
        let stack = state
            .stack
            .iter()
            .map(|&v| self.source(v, state.pc))
            .collect::<Result<Vec<_>, _>>()?;
        let record = ExitRecord {
            pc: state.pc,
            kind,
            reason,
            slots: slots.into(),
            stack: stack.into(),
            handlers: state.handlers.clone(),
            shape_check: None,
            count: Cell::new(0),
        };
        let site = self.a.new_label();
        let ret = self.a.new_label();
        self.exits.push((site, ret, record));
        Ok(site)
    }

    /// The forced-exit gate: decrement the countdown, and take `site` at zero.
    fn forced(&mut self, site: usize) {
        if !self.options.forced_exits {
            return;
        }
        let countdown = std::mem::offset_of!(crate::interpreter::Interp, opt_exit_countdown) as u32;
        self.a
            .ldr_imm(X9, 19, std::mem::offset_of!(JitCtx, interp) as u32);
        self.a.ldr_w_imm(X16, X9, countdown);
        self.a.subs_imm_w(X16, X16, 1);
        self.a.str_w_imm(X16, X9, countdown);
        self.a.b_cond(EQ, site);
    }

    /// `n`'s exit for a check: made, and the forced-exit gate placed before the check.
    fn check_exit(&mut self, n: NodeIx) -> Result<usize, Refusal> {
        let site = self.exit(n, ExitKind::Check)?;
        self.forced(site);
        Ok(site)
    }

    /// Whether `v`'s word is never a reference (a store of it needs no card mark).
    fn scalar_word(&self, v: NodeIx) -> bool {
        match self.graph.node(v).op {
            NodeOp::F64ToTagged | NodeOp::I32ToTagged | NodeOp::BoolToTagged => true,
            NodeOp::ConstTagged(bits) => {
                let tag = bits >> 48;
                !((crate::value::PACK_BIGINT >> 48..=crate::value::PACK_SYM >> 48).contains(&tag)
                    || tag == crate::value::PACK_OBJ >> 48)
            }
            _ => false,
        }
    }

    /// The Object of the cell in `cell` into `obj`. No bound check follows: the shape the field's
    /// CheckShape proved fixes the key set and the capacity class, and shape ids are never reused
    /// (property_shapes.rs), as V8, SpiderMonkey and JSC rely on after a map check; the
    /// baseline's check guards a stale cache, which a live-checked constant shape is not (the
    /// plan's Stage 1 record, "Shapes").
    fn field_object(&mut self, cell: u32, obj: u32) {
        self.a.add_imm(obj, cell, self.layout.obj_from_cell as u32);
    }

    /// The address base of a field whose location word is `location`, and the offset from it:
    /// the Object for a slot in the cell, the record for one past it (W2d step 3; value.rs's
    /// `location_word`). `obj` becomes the record pointer in the second case.
    fn field_base(&mut self, obj: u32, location: u16) -> (u32, u32) {
        let offset = u32::from(location >> 1);
        if location & 1 != 0 {
            let layout = self.layout;
            self.a
                .ldr_imm(obj, obj, (layout.obj_props + layout.props_record) as u32);
        }
        (obj, offset)
    }
}

impl Gen<'_> {
    fn lower(&mut self, n: NodeIx, block: BlockIx, next: Option<BlockIx>) -> Result<(), Refusal> {
        let graph = self.graph;
        let node = graph.node(n);
        let input = |k: usize| node.inputs[k];
        let unused = self.home(n) == Home::None;
        match &node.op {
            NodeOp::ConstTagged(_)
            | NodeOp::ConstF64(_)
            | NodeOp::ConstI32(_)
            | NodeOp::ConstBool(_)
            | NodeOp::Phi => {}
            NodeOp::EntrySlot(slot) | NodeOp::FrameSlot(slot) => {
                if !unused {
                    let dst = self.gp_dst(n);
                    self.a.ldr_imm(dst, 22, u32::from(*slot) * 8);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::This => {
                if !unused {
                    let dst = self.gp_dst(n);
                    self.a
                        .ldr_imm(dst, 19, std::mem::offset_of!(JitCtx, this_word) as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::Projection(k) => {
                let effect = input(0);
                let NodeOp::Effect { pushes, .. } = graph.node(effect).op else {
                    return Err(Refusal::at(node.pc as usize, "a projection of no effect"));
                };
                if self.last_effect != Some(effect) || *k >= pushes {
                    return Err(Refusal::at(
                        node.pc as usize,
                        "a projection apart from its effect",
                    ));
                }
                if !unused {
                    let dst = self.gp_dst(n);
                    self.a.ldur(dst, 20, -8 * i32::from(pushes - k));
                    self.gp_done(n, dst);
                }
            }
            NodeOp::CheckNumber => {
                let site = self.check_exit(n)?;
                let word = self.gp(input(0), X16);
                let d = self.fp_dst(n);
                let nan = self.a.new_label();
                let resume = self.a.new_label();
                // Every tag is a NaN; of the NaNs, only the canonical one is a Number (value.rs:
                // a word is a Number unless its bits 48-62 are 0x7ff9 or more).
                self.a.fmov_d_x(d, word);
                self.a.fcmp(d, d);
                self.a.b_cond(VS, nan);
                self.a.bind(resume);
                self.cold.push(Cold::NanNumber {
                    label: nan,
                    word,
                    exit: site,
                    resume,
                });
                self.fp_done(n, d);
            }
            NodeOp::CheckInt32 => {
                let site = self.check_exit(n)?;
                let d = self.fp(input(0), D31);
                let w = self.gp_dst(n);
                if self.jscvt() {
                    // Z is set exactly when the conversion is exact and the input is not -0
                    // (FEAT_JSCVT; FPToFixedJS clears Z for -0).
                    self.a.fjcvtzs_w_d(w, d);
                    self.a.b_cond(NE, site);
                } else {
                    self.a.fcvtzs_w_d(w, d);
                    self.a.scvtf_d_w(D30, w);
                    self.a.fcmp(D30, d);
                    self.a.b_cond(NE, site);
                    // -0 converts exactly to 0, which is not its value.
                    let ok = self.a.new_label();
                    self.a.cbnz(w, false, ok);
                    self.a.fmov_x_d(X17, d);
                    self.a.tbnz(X17, 63, site);
                    self.a.bind(ok);
                }
                self.gp_done(n, w);
            }
            NodeOp::CheckObject => {
                let site = self.check_exit(n)?;
                let word = self.gp(input(0), X16);
                let t = if word == X16 { X17 } else { X16 };
                // The Object tag 0xfff9 is -7 as a signed half.
                self.a.asr_imm(t, word, 48);
                self.a.cmn_imm_x(t, 7);
                self.a.b_cond(NE, site);
                let dst = self.gp_dst(n);
                self.a.ubfx(dst, word, 0, 48);
                self.gp_done(n, dst);
            }
            NodeOp::CheckShape(shapes) => {
                let site = self.check_exit(n)?;
                // For the log: what a WrongShape exit expected, and where its receiver is.
                let pc = graph.state(node.state.expect("checked")).pc;
                let receiver = self.source(input(0), pc)?;
                if let Some((_, _, record)) = self.exits.last_mut() {
                    record.shape_check = Some(Box::new(crate::opt::exit::ShapeCheck {
                        receiver,
                        shapes: shapes.clone().into(),
                    }));
                }
                let word = self.gp(input(0), X16);
                let t = if word == X16 { X17 } else { X16 };
                self.a.asr_imm(t, word, 48);
                self.a.cmn_imm_x(t, 7);
                self.a.b_cond(NE, site);
                let cell = self.gp_dst(n);
                self.a.ubfx(cell, word, 0, 48);
                // As the baseline's property templates test a receiver (`emit_prop_way_probe`):
                // an ordinary object whose IC-plain flag is set, then its shape.
                let layout = self.layout;
                self.a.add_imm(X16, cell, layout.obj_from_cell as u32);
                self.a.ldrb_imm(X17, X16, layout.obj_kind as u32);
                self.a
                    .cmp_imm_w(X17, crate::value::KIND_ORDINARY_LAST as u32);
                self.a.b_cond(HI, site);
                self.a.ldrb_imm(X17, X16, layout.obj_flags as u32);
                self.a.tbz(X17, flag_bit(crate::value::FLAG_IC_PLAIN), site);
                self.a
                    .ldr_w_imm(X17, X16, (layout.obj_props + layout.props_shape) as u32);
                let ok = self.a.new_label();
                for (k, &shape) in shapes.iter().enumerate() {
                    if shape < 4096 {
                        self.a.cmp_imm_w(X17, shape);
                    } else {
                        self.a.mov_imm64(X16, u64::from(shape));
                        self.a.cmp_reg_w(X17, X16);
                    }
                    if k + 1 == shapes.len() {
                        self.a.b_cond(NE, site);
                    } else {
                        self.a.b_cond(EQ, ok);
                    }
                }
                self.a.bind(ok);
                self.gp_done(n, cell);
            }
            NodeOp::CheckBoolean => {
                let site = self.check_exit(n)?;
                let word = self.gp(input(0), X16);
                let t = if word == X16 { X17 } else { X16 };
                self.a.movz(t, (crate::value::PACK_BOOL >> 48) as u32, 3);
                self.a.logic_x(2, t, word, t);
                self.a.cmp_imm_x(t, 1);
                self.a.b_cond(HI, site);
                let dst = self.gp_dst(n);
                self.a.mov_w(dst, t);
                self.gp_done(n, dst);
            }
            NodeOp::CheckCallee { target } => self.lower_check_callee(n, *target)?,
            NodeOp::CalleeMatches { target } => self.lower_callee_matches(n, *target)?,
            NodeOp::I32ToF64 => {
                if !unused {
                    let w = self.gp(input(0), X16);
                    let d = self.fp_dst(n);
                    self.a.scvtf_d_w(d, w);
                    self.fp_done(n, d);
                }
            }
            NodeOp::TruncateToInt32 => {
                if !unused {
                    let d = self.fp(input(0), D31);
                    let w = self.gp_dst(n);
                    if self.jscvt() {
                        // ToInt32 itself (FEAT_JSCVT), as V8's TruncateDoubleToI uses it.
                        self.a.fjcvtzs_w_d(w, d);
                    } else {
                        // Below 2^63 the conversion to 64 bits is exact and its low word is
                        // ToInt32 (NaN converts to 0). INT64_MIN and INT64_MAX mean it saturated,
                        // which subtracting or adding 1 detects as signed overflow (V8's
                        // TryConvertDoubleToInt64); the cold path computes the low word then.
                        let saturated = self.a.new_label();
                        let resume = self.a.new_label();
                        self.a.fcvtzs_x_d(w, d);
                        self.a.cmp_imm_x(w, 1);
                        self.a.ccmn_imm_x(w, 1, 0b0001, VC);
                        self.a.b_cond(VS, saturated);
                        self.a.mov_w(w, w);
                        self.a.bind(resume);
                        self.cold.push(Cold::Truncate {
                            label: saturated,
                            d,
                            dst: w,
                            resume,
                        });
                    }
                    self.gp_done(n, w);
                }
            }
            NodeOp::F64ToTagged => {
                if !unused {
                    let d = self.fp(input(0), D31);
                    let x = self.gp_dst(n);
                    self.a.fmov_x_d(x, d);
                    self.a.fcmp(d, d);
                    self.a.movz(X17, 0x7ff8, 3); // PACK_CANON_NAN
                    self.a.csel_x(x, X17, x, VS);
                    self.gp_done(n, x);
                }
            }
            NodeOp::I32ToTagged => {
                if !unused {
                    let w = self.gp(input(0), X16);
                    let x = self.gp_dst(n);
                    self.a.scvtf_d_w(D31, w);
                    self.a.fmov_x_d(x, D31);
                    self.gp_done(n, x);
                }
            }
            NodeOp::BoolToTagged => {
                if !unused {
                    let b = self.gp(input(0), X16);
                    let x = self.gp_dst(n);
                    self.a.movz(X17, (crate::value::PACK_BOOL >> 48) as u32, 3);
                    self.a.logic_x(1, x, X17, b);
                    self.gp_done(n, x);
                }
            }
            NodeOp::ToBoolean => {
                if !unused {
                    self.lower_to_boolean(n);
                }
            }
            NodeOp::I32Add | NodeOp::I32Sub => {
                let site = self.check_exit(n)?;
                let a = self.gp(input(0), X16);
                let b = self.gp(input(1), X17);
                let w = self.gp_dst(n);
                if node.op == NodeOp::I32Add {
                    self.a.adds_reg_w(w, a, b);
                } else {
                    self.a.subs_reg_w(w, a, b);
                }
                self.a.b_cond(VS, site);
                self.gp_done(n, w);
            }
            NodeOp::I32Mul => {
                let site = self.check_exit(n)?;
                let a = self.gp(input(0), X16);
                let b = self.gp(input(1), X17);
                let w = self.gp_dst(n);
                self.a.smull(w, a, b);
                self.a.cmp_sxtw(w, w);
                self.a.b_cond(NE, site);
                // A zero product of a negative operand is -0.
                let nonzero = self.a.new_label();
                self.a.cbnz(w, false, nonzero);
                self.a.logic_w(1, X30, a, b);
                self.a.tbnz(X30, 31, site);
                self.a.bind(nonzero);
                self.gp_done(n, w);
            }
            NodeOp::I32And | NodeOp::I32Or | NodeOp::I32Xor => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let b = self.gp(input(1), X17);
                    let w = self.gp_dst(n);
                    let op = match node.op {
                        NodeOp::I32And => 0,
                        NodeOp::I32Or => 1,
                        _ => 2,
                    };
                    self.a.logic_w(op, w, a, b);
                    self.gp_done(n, w);
                }
            }
            NodeOp::I32Shl | NodeOp::I32Sar => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let b = self.gp(input(1), X17);
                    let w = self.gp_dst(n);
                    // The count is taken mod 32, as JavaScript takes it.
                    let op = if node.op == NodeOp::I32Shl { 0 } else { 2 };
                    self.a.shift_w(op, w, a, b);
                    self.gp_done(n, w);
                }
            }
            NodeOp::I32UShr => {
                let site = self.check_exit(n)?;
                let a = self.gp(input(0), X16);
                let b = self.gp(input(1), X17);
                let w = self.gp_dst(n);
                self.a.shift_w(1, w, a, b);
                self.a.tbnz(w, 31, site); // past int32
                self.gp_done(n, w);
            }
            NodeOp::I32UShrToF64 => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let b = self.gp(input(1), X17);
                    self.a.shift_w(1, X16, a, b);
                    let d = self.fp_dst(n);
                    self.a.ucvtf_d_w(d, X16);
                    self.fp_done(n, d);
                }
            }
            NodeOp::I32Not => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let w = self.gp_dst(n);
                    self.a.mvn_w(w, a);
                    self.gp_done(n, w);
                }
            }
            NodeOp::F64Add | NodeOp::F64Sub | NodeOp::F64Mul | NodeOp::F64Div => {
                if !unused {
                    let a = self.fp(input(0), D30);
                    let b = self.fp(input(1), D31);
                    let d = self.fp_dst(n);
                    let op = match node.op {
                        NodeOp::F64Add => 0,
                        NodeOp::F64Sub => 1,
                        NodeOp::F64Mul => 2,
                        _ => 3,
                    };
                    self.a.f_arith(op, d, a, b);
                    self.fp_done(n, d);
                }
            }
            NodeOp::F64Mod => {
                // A call: the allocator gave nothing live across it a caller-saved home.
                let a = self.fp(input(0), D30);
                let b = self.fp(input(1), D31);
                self.a.fmov_d_d(D30, a);
                self.a.fmov_d_d(D31, b);
                self.a.fmov_d_d(0, D30);
                self.a.fmov_d_d(1, D31);
                self.a.mov_imm64(X16, f64_remainder as *const () as u64);
                self.a.blr(X16);
                if !unused {
                    let d = self.fp_dst(n);
                    self.a.fmov_d_d(d, 0);
                    self.fp_done(n, d);
                }
            }
            NodeOp::F64Neg => {
                if !unused {
                    let a = self.fp(input(0), D30);
                    let d = self.fp_dst(n);
                    self.a.fneg(d, a);
                    self.fp_done(n, d);
                }
            }
            NodeOp::I32Cmp(cond) => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let b = self.gp(input(1), X17);
                    let w = self.gp_dst(n);
                    self.a.cmp_reg_w(a, b);
                    let code = match cond {
                        Cond::Lt => LT,
                        Cond::Le => LE,
                        Cond::Gt => GT,
                        Cond::Ge => GE,
                        Cond::Eq => EQ,
                        Cond::Ne => NE,
                    };
                    self.a.cset_w(w, code);
                    self.gp_done(n, w);
                }
            }
            NodeOp::F64Cmp(cond) => {
                if !unused {
                    let a = self.fp(input(0), D30);
                    let b = self.fp(input(1), D31);
                    let w = self.gp_dst(n);
                    self.a.fcmp(a, b);
                    // Unordered sets C and V and clears N and Z: of these only NE holds.
                    let code = match cond {
                        Cond::Lt => MI,
                        Cond::Le => LS,
                        Cond::Gt => GT,
                        Cond::Ge => GE,
                        Cond::Eq => EQ,
                        Cond::Ne => NE,
                    };
                    self.a.cset_w(w, code);
                    self.gp_done(n, w);
                }
            }
            NodeOp::TaggedSame => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let b = self.gp(input(1), X17);
                    let w = self.gp_dst(n);
                    self.a.cmp_reg_x(a, b);
                    self.a.cset_w(w, EQ);
                    self.gp_done(n, w);
                }
            }
            NodeOp::BoolNot => {
                if !unused {
                    let b = self.gp(input(0), X16);
                    let w = self.gp_dst(n);
                    let one = asm::logical_imm_w(1).expect("1 is a logical immediate");
                    self.a.logic_imm_w(2, w, b, one);
                    self.gp_done(n, w);
                }
            }
            NodeOp::LoadField { location, .. } => {
                if !unused {
                    let cell = self.gp(input(0), X16);
                    self.field_object(cell, X17);
                    let (base, offset) = self.field_base(X17, *location);
                    let dst = self.gp_dst(n);
                    self.a.ldr_imm(dst, base, offset);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::StoreField { location, .. } => {
                let cell = self.gp(input(0), X16);
                let value = self.gp(input(1), X17);
                self.field_object(cell, X9);
                let (base, offset) = self.field_base(X9, *location);
                self.a.str_imm(value, base, offset);
                if !self.scalar_word(input(1)) {
                    // The receiver's card, after the store and before any safepoint, as the
                    // baseline's store marks it (`emit_card_mark`, `emit_set_prop_inline`): a
                    // minor collection then finds an old object's edge to a young value.
                    emit_card_mark(&mut self.a, cell, X9, X30);
                }
            }
            NodeOp::StoreSlot(slot) => {
                let value = self.boxed(input(0), X16);
                self.a.str_imm(value, 22, u32::from(*slot) * 8);
            }
            NodeOp::Poll => {
                let slow = self.a.new_label();
                let resume = self.a.new_label();
                self.a
                    .ldr_imm(X9, 19, std::mem::offset_of!(JitCtx, poll_word) as u32);
                self.a.ldr_imm(X9, X9, 0);
                self.a.cbnz(X9, true, slow);
                self.a.bind(resume);
                let state = graph.state(node.state.expect("checked"));
                let throw = if state.handlers.is_empty() {
                    self.l_ret_throw
                } else {
                    self.exit(n, ExitKind::PollThrow)?
                };
                self.cold.push(Cold::Poll {
                    label: slow,
                    node: n,
                    resume,
                    throw,
                });
            }
            NodeOp::Effect { pc, pops, .. } => {
                self.lower_effect(n, *pc as usize, *pops as usize)?;
            }
            NodeOp::Jump(target) => {
                self.emit_moves(block, *target)?;
                if next != Some(*target) {
                    self.a.b(self.blocks[target.index()]);
                }
            }
            NodeOp::Branch {
                then_block,
                else_block,
            } => {
                let cond = self.gp(input(0), X16);
                let then_label = self.edge_target(block, *then_block);
                let else_label = self.edge_target(block, *else_block);
                let then_direct = then_label == self.blocks[then_block.index()];
                let else_direct = else_label == self.blocks[else_block.index()];
                if next == Some(*else_block) && else_direct {
                    self.a.cbnz(cond, false, then_label);
                } else if next == Some(*then_block) && then_direct {
                    self.a.cbz(cond, false, else_label);
                } else {
                    self.a.cbnz(cond, false, then_label);
                    self.a.b(else_label);
                }
            }
            NodeOp::Return => {
                let value = self.gp(input(0), X16);
                self.a
                    .str_imm(value, 19, std::mem::offset_of!(JitCtx, ret) as u32);
                // An empty operand stack: what a Rust entry drops from `final_sp`.
                self.a
                    .ldr_imm(20, 19, std::mem::offset_of!(JitCtx, stack_base) as u32);
                self.a.b(self.l_ret_ok);
            }
            NodeOp::Exit(_) => {
                let site = self.exit(n, ExitKind::Check)?;
                self.a.b(site);
            }
        }
        Ok(())
    }

    /// A spliced callee's guard, as the first stage's InlineGuard template tests it.
    fn lower_check_callee(&mut self, n: NodeIx, target: u32) -> Result<(), Refusal> {
        let node = self.graph.node(n);
        let site = self.check_exit(n)?;
        let it = self.chunk.jit_inline_target(target);
        // The expected function's stored word, read out of an Option<Gc> as the first stage's
        // InlineGuard reads it; a dead callee always exits.
        let stored = it.pin.upgrade().map(|object| {
            let some: Option<crate::value::Gc> = Some(object);
            unsafe { *(&some as *const Option<crate::value::Gc> as *const usize) }
        });
        let Some(stored) = stored else {
            self.a.b(site);
            return Ok(());
        };
        let callee = self.gp(node.inputs[0], X16);
        if node.rep == Rep::Tagged && self.home(n) != Home::None {
            let dst = self.gp_dst(n);
            self.a.mov(dst, callee);
            self.gp_done(n, dst);
        }
        if it.expected_env != 0 {
            self.a
                .ldr_imm(X17, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
            self.a.mov_imm64(X30, it.expected_env as u64);
            self.a.cmp_reg_x(X17, X30);
            self.a.b_cond(NE, site);
        }
        // Same-Realm proof: the splice runs in the planning Realm only.
        self.a
            .ldr_imm(X17, 19, std::mem::offset_of!(JitCtx, genv) as u32);
        self.a.mov_imm64(X30, it.expected_genv as u64);
        self.a.cmp_reg_x(X17, X30);
        self.a.b_cond(NE, site);
        self.a
            .mov_imm64(X17, crate::value::PACK_OBJ | stored as u64);
        self.a.cmp_reg_x(callee, X17);
        self.a.b_cond(NE, site);
        if it.check_this {
            let receiver = *node.inputs.get(1).ok_or(Refusal::at(
                node.pc as usize,
                "a callee check of `this` without the receiver",
            ))?;
            let word = self.gp(receiver, X16);
            self.a.asr_imm(X17, word, 48);
            self.a.cmn_imm_x(X17, 7);
            self.a.b_cond(NE, site);
        }
        Ok(())
    }

    /// A guard's test without an exit (CalleeMatches): `lower_check_callee`'s comparisons, each
    /// mismatch giving false. The result is written after every input is read, since its home
    /// may be an input's whose range ends here.
    fn lower_callee_matches(&mut self, n: NodeIx, target: u32) -> Result<(), Refusal> {
        let node = self.graph.node(n);
        let it = self.chunk.jit_inline_target(target);
        let mismatch = self.a.new_label();
        let done = self.a.new_label();
        // A dead callee never matches.
        let stored = it.pin.upgrade().map(|object| {
            let some: Option<crate::value::Gc> = Some(object);
            unsafe { *(&some as *const Option<crate::value::Gc> as *const usize) }
        });
        if let Some(stored) = stored {
            let callee = self.gp(node.inputs[0], X16);
            if it.expected_env != 0 {
                self.a
                    .ldr_imm(X17, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
                self.a.mov_imm64(X30, it.expected_env as u64);
                self.a.cmp_reg_x(X17, X30);
                self.a.b_cond(NE, mismatch);
            }
            self.a
                .ldr_imm(X17, 19, std::mem::offset_of!(JitCtx, genv) as u32);
            self.a.mov_imm64(X30, it.expected_genv as u64);
            self.a.cmp_reg_x(X17, X30);
            self.a.b_cond(NE, mismatch);
            self.a
                .mov_imm64(X17, crate::value::PACK_OBJ | stored as u64);
            self.a.cmp_reg_x(callee, X17);
            self.a.b_cond(NE, mismatch);
            if it.check_this {
                let receiver = *node.inputs.get(1).ok_or(Refusal::at(
                    node.pc as usize,
                    "a callee test of `this` without the receiver",
                ))?;
                let word = self.gp(receiver, X16);
                self.a.asr_imm(X17, word, 48);
                self.a.cmn_imm_x(X17, 7);
                self.a.b_cond(NE, mismatch);
            }
        } else {
            self.a.b(mismatch);
        }
        let w = self.gp_dst(n);
        self.a.movz(w, 1, 0);
        self.a.b(done);
        self.a.bind(mismatch);
        self.a.movz(w, 0, 0);
        self.a.bind(done);
        self.gp_done(n, w);
        Ok(())
    }

    /// An effect: the operation's baseline template on the canonical frame (see the module
    /// header).
    fn lower_effect(&mut self, n: NodeIx, pc: usize, pops: usize) -> Result<(), Refusal> {
        let graph = self.graph;
        let node = graph.node(n);
        let state = graph.state(node.state.expect("checked"));
        let handlers = !state.handlers.is_empty();
        if self.options.forced_exits {
            let site = self.exit(n, ExitKind::Check)?;
            self.forced(site);
        }
        let depth = self.cfg.stack_depth_at(pc).expect("checked");
        self.a
            .ldr_imm(X9, 19, std::mem::offset_of!(JitCtx, stack_base) as u32);
        if handlers {
            // A throw exits with this state, which the template's clobbers would lose: the frame
            // holds all of it before the template runs.
            for (k, &v) in state.stack.iter().enumerate() {
                let word = self.boxed(v, X16);
                self.a.str_imm(word, X9, 8 * k as u32);
            }
            for (slot, source) in state.slots.iter().enumerate() {
                if let SlotSource::Value(v) = *source {
                    let word = self.boxed(v, X16);
                    self.a.str_imm(word, 22, 8 * slot as u32);
                }
            }
        }
        for (k, &v) in node.inputs.iter().enumerate() {
            let word = self.boxed(v, X16);
            self.a.str_imm(word, X9, (8 * (depth - pops + k)) as u32);
        }
        self.a.add_imm(20, X9, (8 * depth) as u32);
        let unwind = if handlers {
            let site = self.exit(n, ExitKind::EffectThrow)?;
            let label = self.a.new_label();
            self.cold.push(Cold::Jump {
                label,
                target: site,
            });
            label
        } else {
            self.l_ret_throw
        };
        self.a.set_hot(self.in_loop[pc]);
        let emitted = operations::emit(
            &mut self.a,
            self.chunk,
            self.layout,
            self.ilayout,
            pc,
            self.fast,
            self.array_intrinsics_on,
            self.function_call_intrinsic_on,
            unwind,
            self.l_direct_finish,
        );
        if !emitted {
            match self.chunk.jit_ops()[pc] {
                Op::CallSpread(..) | Op::CallSpreadThis(..) => {
                    emit_op_helper(&mut self.a, H_CALL_SPREAD, pc as u32, unwind);
                }
                _ => emit_exec(&mut self.a, pc as u32, unwind),
            }
        }
        self.a.set_hot(false);
        self.last_effect = Some(n);
        Ok(())
    }

    /// The label a branch from `from` to `to` takes: the block's, or a trampoline that makes the
    /// edge's moves into its phis first.
    fn edge_target(&mut self, from: BlockIx, to: BlockIx) -> usize {
        if self.graph.block(to).phis.is_empty() {
            return self.blocks[to.index()];
        }
        let label = self.a.new_label();
        self.cold.push(Cold::Edge { label, from, to });
        label
    }

    /// The parallel moves of the edge `from` -> `to` into `to`'s phis.
    fn emit_moves(&mut self, from: BlockIx, to: BlockIx) -> Result<(), Refusal> {
        let graph = self.graph;
        let data = graph.block(to);
        if data.phis.is_empty() {
            return Ok(());
        }
        let index = data
            .preds
            .iter()
            .position(|&p| p == from)
            .ok_or(Refusal::whole("an edge its successor does not list"))?;
        let loc = |home: Home, v: NodeIx| match home {
            Home::Gp(r) => Some(Loc::Gp(u32::from(r))),
            Home::Fp(r) => Some(Loc::Fp(u32::from(r))),
            Home::Spill(offset) => Some(Loc::Spill(offset)),
            Home::Const => Some(Loc::Const(v)),
            Home::None => None,
        };
        // (destination, source, whether the class is FP)
        let mut moves: Vec<(Loc, Loc, bool)> = Vec::new();
        for &phi in &data.phis {
            let Some(dst) = loc(self.home(phi), phi) else {
                continue; // a phi nothing reads
            };
            let input = graph.node(phi).inputs[index];
            let src = loc(self.home(input), input)
                .ok_or(Refusal::whole("a phi's input without a home"))?;
            if src != dst {
                moves.push((dst, src, graph.node(phi).rep == Rep::F64));
            }
        }
        while !moves.is_empty() {
            let free =
                (0..moves.len()).find(|&k| moves.iter().all(|&(_, src, _)| src != moves[k].0));
            match free {
                Some(k) => {
                    let (dst, src, fp) = moves.remove(k);
                    self.emit_move(dst, src, fp);
                }
                None => {
                    // A cycle: park the first destination's value in a scratch register, which
                    // its readers read instead.
                    let (dst, _, fp) = moves[0];
                    let scratch = if fp { Loc::Fp(D31) } else { Loc::Gp(X17) };
                    self.emit_move(scratch, dst, fp);
                    for m in &mut moves {
                        if m.1 == dst {
                            m.1 = scratch;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_move(&mut self, dst: Loc, src: Loc, fp: bool) {
        match (dst, src) {
            (Loc::Gp(d), Loc::Gp(s)) => self.a.mov(d, s),
            (Loc::Gp(d), Loc::Spill(o)) => self.a.ldr_imm(d, 31, o),
            (Loc::Gp(d), Loc::Const(v)) => {
                let bits = self.const_bits(v);
                self.a.mov_imm64(d, bits);
            }
            (Loc::Fp(d), Loc::Fp(s)) => self.a.fmov_d_d(d, s),
            (Loc::Fp(d), Loc::Spill(o)) => self.a.ldr_d_imm(d, 31, o),
            (Loc::Fp(d), Loc::Const(v)) => {
                let r = self.fp(v, d);
                debug_assert_eq!(r, d);
            }
            (Loc::Spill(o), Loc::Gp(s)) => self.a.str_imm(s, 31, o),
            (Loc::Spill(o), Loc::Fp(s)) => self.a.str_d_imm(s, 31, o),
            (Loc::Spill(o), Loc::Spill(p)) => {
                self.a.ldr_imm(X16, 31, p);
                self.a.str_imm(X16, 31, o);
            }
            (Loc::Spill(o), Loc::Const(v)) => {
                let bits = self.const_bits(v);
                self.a.mov_imm64(X16, bits);
                self.a.str_imm(X16, 31, o);
            }
            (Loc::Gp(_), Loc::Fp(_)) | (Loc::Fp(_), Loc::Gp(_)) | (Loc::Const(_), _) => {
                unreachable!("a move within one class (fp: {fp})")
            }
        }
    }

    /// The out-of-line code, then the exits' sites.
    fn emit_cold(&mut self) -> Result<(), Refusal> {
        let cold = std::mem::take(&mut self.cold);
        for item in cold {
            match item {
                Cold::NanNumber {
                    label,
                    word,
                    exit,
                    resume,
                } => {
                    self.a.bind(label);
                    self.a.mov_imm64(X17, crate::value::PACK_CANON_NAN);
                    self.a.cmp_reg_x(word, X17);
                    self.a.b_cond(NE, exit);
                    self.a.b(resume);
                }
                Cold::Truncate {
                    label,
                    d,
                    dst,
                    resume,
                } => {
                    // |x| >= 2^63, so its exponent's shift e - 1075 is at least 11: the implicit
                    // bit and every bit above the mantissa's land at bit 63 or above, and the low
                    // word of the bits shifted by it is that of |x| (zero from a shift of 32 up,
                    // 2^84 and the infinities among them); the sign negates it modulo 2^32.
                    let zero = self.a.new_label();
                    let positive = self.a.new_label();
                    self.a.bind(label);
                    self.a.fmov_x_d(X16, d);
                    self.a.ubfx(X17, X16, 52, 11);
                    self.a.sub_imm(X17, X17, 1075);
                    self.a.cmp_imm_x(X17, 32);
                    self.a.b_cond(GE, zero);
                    self.a.lsl_reg(X30, X16, X17);
                    self.a.tbz(X16, 63, positive);
                    self.a.neg_w(X30, X30);
                    self.a.bind(positive);
                    self.a.mov_w(dst, X30);
                    self.a.b(resume);
                    self.a.bind(zero);
                    self.a.mov_w(dst, 31);
                    self.a.b(resume);
                }
                Cold::Jump { label, target } => {
                    self.a.bind(label);
                    self.a.b(target);
                }
                Cold::Edge { label, from, to } => {
                    self.a.bind(label);
                    self.emit_moves(from, to)?;
                    self.a.b(self.blocks[to.index()]);
                }
                Cold::Poll {
                    label,
                    node,
                    resume,
                    throw,
                } => self.emit_poll_slow(label, node, resume, throw),
                Cold::ToBoolean {
                    label,
                    node,
                    word,
                    dst,
                    resume,
                } => {
                    self.a.bind(label);
                    let saved = self.save_live_caller_saved(node);
                    // The word first: it may be in x0.
                    self.a.mov(1, word);
                    self.a.mov(0, 19);
                    self.a.mov_imm64(X16, to_boolean as *const () as u64);
                    self.a.blr(X16);
                    self.a.mov_w(X9, 0);
                    self.restore_live_caller_saved(&saved);
                    self.a.mov_w(dst, X9);
                    self.a.b(resume);
                }
            }
        }
        // One `bl` per exit: its return address names the exit.
        let sites: Vec<(usize, usize)> = self.exits.iter().map(|&(s, r, _)| (s, r)).collect();
        for (site, ret) in sites {
            self.a.bind(site);
            self.a.bl_label(self.l_exit_stub);
            self.a.bind(ret);
        }
        Ok(())
    }

    /// Save on the native stack, where the collector scans it, each caller-saved register that
    /// holds a value live at `node`, for a call from a node that is no effect (a slow path the
    /// allocator did not see as a call). Returns what `restore_live_caller_saved` restores.
    fn save_live_caller_saved(&mut self, node: NodeIx) -> (Vec<(bool, u32)>, u32) {
        let file = &regalloc::ARM64;
        let live: Vec<(bool, u32)> = self
            .alloc
            .live_at(node)
            .into_iter()
            .filter_map(|v| match self.home(v) {
                Home::Gp(r) if file.gp_caller.contains(&r) => Some((false, u32::from(r))),
                Home::Fp(r) if file.fp_caller.contains(&r) => Some((true, u32::from(r))),
                _ => None,
            })
            .collect();
        let size = (8 * live.len() as u32).next_multiple_of(16);
        if size > 0 {
            self.a.sub_imm(31, 31, size);
        }
        for (k, &(fp, r)) in live.iter().enumerate() {
            if fp {
                self.a.str_d_imm(r, 31, 8 * k as u32);
            } else {
                self.a.str_imm(r, 31, 8 * k as u32);
            }
        }
        (live, size)
    }

    fn restore_live_caller_saved(&mut self, (live, size): &(Vec<(bool, u32)>, u32)) {
        for (k, &(fp, r)) in live.iter().enumerate() {
            if fp {
                self.a.ldr_d_imm(r, 31, 8 * k as u32);
            } else {
                self.a.ldr_imm(r, 31, 8 * k as u32);
            }
        }
        if *size > 0 {
            self.a.add_imm(31, 31, *size);
        }
    }

    /// A poll's slow path: the live caller-saved registers saved, `H_INTERRUPT` called as the
    /// first stage's loop poll calls it, the registers restored, and a throw taken to `throw`.
    fn emit_poll_slow(&mut self, label: usize, node: NodeIx, resume: usize, throw: usize) {
        let depth = self
            .graph
            .state(self.graph.node(node).state.expect("checked"))
            .stack
            .len() as u32;
        self.a.bind(label);
        let saved = self.save_live_caller_saved(node);
        let a = &mut self.a;
        a.ldr_imm(20, 19, std::mem::offset_of!(JitCtx, stack_base) as u32);
        a.add_imm(20, 20, 8 * depth);
        a.mov(0, 19);
        a.movz(1, POLL_LOOP, 0);
        a.mov(2, 20);
        a.ldr_imm(X16, 21, (H_INTERRUPT * 8) as u32);
        a.blr(X16);
        a.mov(20, 0);
        a.mov_w(X9, 1);
        self.restore_live_caller_saved(&saved);
        self.a.cbnz(X9, false, throw);
        self.a.b(resume);
    }

    /// ToBoolean (ECMA-262 §7.1.2) of the word in line, as the baseline's ToBoolean stub tests
    /// it (`emit_to_boolean_stub`): a Boolean's payload; a Number unless ±0 or NaN; false for
    /// undefined, null and the Empty sentinel; a String unless empty; true for a Symbol and for
    /// an IC-plain object, which cannot be [[IsHTMLDDA]]. A BigInt and any other object take
    /// `to_boolean` out of line, which runs no JavaScript and does not collect.
    fn lower_to_boolean(&mut self, n: NodeIx) {
        use crate::value::{PACK_BOOL, PACK_OBJ, PACK_STR, PACK_SYM};
        let word = self.gp(self.graph.node(n).inputs[0], X16);
        let w = self.gp_dst(n);
        let not_bool = self.a.new_label();
        let tagged = self.a.new_label();
        let string = self.a.new_label();
        let truthy = self.a.new_label();
        let falsy = self.a.new_label();
        let slow = self.a.new_label();
        let done = self.a.new_label();
        let a = &mut self.a;
        // A Boolean's word EOR its tag is its payload, 0 or 1; every other word is larger.
        a.movz(X17, (PACK_BOOL >> 48) as u32, 3);
        a.logic_x(2, X17, word, X17);
        a.cmp_imm_x(X17, 1);
        a.b_cond(HI, not_bool);
        a.mov_w(w, X17);
        a.b(done);
        a.bind(not_bool);
        // A Number compares with zero; NaN and every tag compare unordered.
        a.fmov_d_x(D31, word);
        a.fcmp_zero(D31);
        a.b_cond(VS, tagged);
        a.cset_w(w, NE);
        a.b(done);
        a.bind(tagged);
        a.lsr_imm(X17, word, 48);
        // Below the Boolean tag: the canonical NaN, undefined, the Empty sentinel and null.
        a.movz(X30, (PACK_BOOL >> 48) as u32, 0);
        a.cmp_reg_w(X17, X30);
        a.b_cond(LO, falsy);
        a.movz(X30, (PACK_STR >> 48) as u32, 0);
        a.cmp_reg_w(X17, X30);
        a.b_cond(EQ, string);
        a.b_cond(LO, slow); // a BigInt
        a.movz(X30, (PACK_SYM >> 48) as u32, 0);
        a.cmp_reg_w(X17, X30);
        a.b_cond(EQ, truthy);
        a.movz(X30, (PACK_OBJ >> 48) as u32, 0);
        a.cmp_reg_w(X17, X30);
        a.b_cond(NE, slow);
        a.ubfx(X30, word, 0, 48);
        a.add_imm(X30, X30, self.layout.obj_from_cell as u32);
        a.ldrb_imm(X30, X30, self.layout.obj_flags as u32);
        a.tbz(X30, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
        a.bind(truthy);
        a.movz(w, 1, 0);
        a.b(done);
        a.bind(string);
        a.ubfx(X30, word, 0, 48);
        a.ldr_w_imm(X30, X30, crate::lstr::LEN_OFF as u32);
        a.cmp_imm_w(X30, 0);
        a.cset_w(w, NE);
        a.b(done);
        a.bind(falsy);
        a.movz(w, 0, 0);
        a.bind(done);
        self.cold.push(Cold::ToBoolean {
            label: slow,
            node: n,
            word,
            dst: w,
            resume: done,
        });
        self.gp_done(n, w);
    }

    /// The code's exit stub (see `opt::exit`): save x0-x29 and d0-d31, call the deoptimizer with
    /// the exit's return address from x30, and leave through the epilogue its result selects,
    /// with an empty operand stack.
    fn emit_exit_stub(&mut self) {
        let table = &*self.table as *const ExitTable as u64;
        let a = &mut self.a;
        a.bind(self.l_exit_stub);
        a.sub_imm(31, 31, SAVE_BYTES);
        for r in (0..30).step_by(2) {
            a.stp_off(r, r + 1, 8 * r as i32);
        }
        for r in (0..32).step_by(2) {
            a.stp_d_off(r, r + 1, 8 * (crate::opt::exit::SAVED_X as i32 + r as i32));
        }
        a.mov(0, 19);
        a.mov_imm64(1, table);
        a.mov(2, X30);
        a.add_imm(3, 31, 0); // mov x3, sp
        a.mov_imm64(X16, crate::opt::exit::deoptimize as *const () as u64);
        a.blr(X16);
        a.add_imm(31, 31, SAVE_BYTES);
        a.ldr_imm(20, 19, std::mem::offset_of!(JitCtx, stack_base) as u32);
        a.cbz(0, false, self.l_ret_throw);
        a.b(self.l_ret_ok);
    }
}

#[cfg(test)]
#[path = "a64_tests.rs"]
mod tests;
