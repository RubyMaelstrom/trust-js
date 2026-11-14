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
//! Effects and generic nodes run the operation's baseline template on the canonical operand
//! words (contracts 4): the operands written to the canonical words below `stack_base + 8 *
//! depth(pc)`, a generic node's named slots whose words are stale written to them (P3 stage 3
//! step 3a), x20 set there, the template emitted through `operations::emit` (`emit_exec` where it
//! declines) with the tier's unwind, the results loaded from below the returned x20 by the
//! projections that follow and a written slot's word by its SlotResult. A throw leaves through
//! the throw epilogue (ctx.error is the helper's), or, at a pc with a handler, exits to the VM
//! with the throw pending: a generic node's exit reads its recipe's values from the homes its call
//! preserves (`Graph::throw_recipe`), and a canonical-frame effect's code writes its whole frame
//! state to the frame before it, since the template clobbers the caller-saved registers. Values
//! living across an effect or a generic node have callee-saved homes or spill slots
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
use crate::opt::view::FrozenChunk;

// Condition codes.
const EQ: u32 = 0;
const NE: u32 = 1;
const HS: u32 = 2;
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

/// The condition on (b, a) that holds where `code` holds on (a, b): a comparison's operands
/// swapped, which reverses an order and keeps an equality (P3 stage 2 step 7).
fn swapped(code: u32) -> u32 {
    match code {
        LT => GT,
        GT => LT,
        LE => GE,
        GE => LE,
        LO => HI,
        HI => LO,
        HS => LS,
        LS => HS,
        EQ | NE => code,
        _ => unreachable!("a comparison's condition"),
    }
}

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
    /// The poison before each generic node (TRUST_JS_OPT_POISON; P3 stage 3 step 3a): the frame's
    /// stale words, the slots it does not name whose words the state does not hold current and
    /// the operand words below its operands, overwritten with [`POISON`], so that a template or
    /// helper that read one would compute with it, which the tests' results show; before each
    /// canonical-frame effect without a handler, the operand words below its operands; and after
    /// each generic node's template, the check that it wrote none of its poisoned words
    /// (`Gen::check_poison`; the step's follow-up).
    pub(crate) poison: bool,
}

#[cfg(test)]
thread_local! {
    /// The inlined frames' generic nodes this thread's code generations lowered, by their
    /// operation (tests: P3 stage 3 step 3c).
    pub(crate) static TEST_INLINED_GENERIC: std::cell::RefCell<Vec<Op>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The word the poison switch writes (`Options::poison`): a Number by value.rs's encoding (a
/// tiny negative double), which names no cell and which no program computes in passing.
pub(crate) const POISON: u64 = 0x8bad_f00d_dead_beef;

/// The poison check's code for an operand word below a generic node's operands (`BELOW | k`),
/// where a slot's is its number ([`poison_written`]).
const BELOW: u64 = 1 << 32;

/// The poison check's report (`Gen::check_poison`): the word `code` names, which the poison
/// switch wrote before the generic node at `pc` of `ctx`'s chunk, no longer holds the poison after
/// the node's template.
unsafe extern "C" fn poison_written(ctx: *mut JitCtx, pc: u64, code: u64) {
    // SAFETY: the running code's context, whose chunk is live while its code runs.
    let chunk = unsafe { &*(*ctx).chunk };
    let what = if code & BELOW != 0 {
        format!(
            "its template wrote operand word {} below its operands",
            code & !BELOW
        )
    } else {
        let name = chunk
            .jit_slot_names()
            .get(code as usize)
            .map_or("?", |name| &**name);
        format!("its template wrote slot {code} ({name}), which its operation does not name")
    };
    crate::opt::poison_fault(chunk, pc as usize, &what);
}

impl Options {
    /// The options the switches select (main thread: TRUST_JS_OPT_EXIT_EVERY is a test's switch
    /// there).
    pub(crate) fn from_env() -> Self {
        Options {
            forced_exits: crate::opt::exit_every() > 0,
            no_jscvt: false,
            poison: crate::opt::poison(),
        }
    }
}

/// The process switches and the chunk's facts the code generator reads, read on the main thread
/// with the snapshot (P3 stage 2, step 12; the plan's point 2), so that a compile reads none.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Switches {
    /// The templates' feature mask (`template_mask`, TRUST_JS_JIT_FAST).
    pub(crate) fast: u32,
    /// The stack analysis's TDZ proof (off under TRUST_JS_JIT_NO_TDZ_PROOF).
    pub(crate) tdz_proof: bool,
    /// Off under TRUST_JS_JIT_NO_ARRAY_INTRINSICS and TRUST_JS_JIT_NO_FUNCTION_CALL_INTRINSIC.
    pub(crate) array_intrinsics: bool,
    pub(crate) function_call_intrinsic: bool,
    /// The direct call sequence (off under TRUST_JS_JIT_NO_DIRECT_CALLS).
    pub(crate) direct_shared_context: bool,
    /// FEAT_JSCVT, unless TRUST_JS_JIT_NO_JSCVT (`jscvt_available`).
    pub(crate) jscvt: bool,
    /// The code's exit limit (`opt::exit_limit`: 100 << R at the snapshot).
    pub(crate) exit_limit: u32,
    /// The code's loop limit (`opt::loop_exit_limit`: 5 << R at the snapshot, never above the
    /// exit limit) and its warm-up (`opt::loop_warm_up`: a discard's warm-up's first distance at
    /// the snapshot's R; stage 2, step 13).
    pub(crate) loop_exit_limit: u32,
    pub(crate) loop_warm_up: u32,
    /// The native runs an entry counts (`opt::entry_weight`), which the direct sequence's shared
    /// stubs hold as a process constant.
    pub(crate) entry_weight: u32,
    /// Whether the running Agent's heap has the in-line allocation site of class-0 cells, which
    /// a direct construct's in-line `this` uses (`value::object_inline_alloc_site`).
    pub(crate) inline_construct: bool,
    /// Tests: the stubs a compile off the main thread makes find no executable memory, or write
    /// d8 (`opt::TestSwitches`; P3 stage 2, step 17).
    #[cfg(test)]
    pub(crate) stub_memory_refused: bool,
    #[cfg(test)]
    pub(crate) stubs_write_d8: bool,
}

impl Switches {
    /// The switches as the process sets them, for `chunk`'s code (main thread).
    pub(crate) fn read(chunk: &Chunk) -> Self {
        prime_switches();
        Switches {
            fast: template_mask(chunk),
            tdz_proof: !env_flag!("TRUST_JS_JIT_NO_TDZ_PROOF"),
            array_intrinsics: !env_flag!("TRUST_JS_JIT_NO_ARRAY_INTRINSICS"),
            function_call_intrinsic: !env_flag!("TRUST_JS_JIT_NO_FUNCTION_CALL_INTRINSIC"),
            direct_shared_context: crate::bytecode::direct_shared_context_enabled(),
            jscvt: jscvt_available(),
            exit_limit: crate::opt::exit_limit(chunk),
            loop_exit_limit: crate::opt::loop_exit_limit(chunk),
            loop_warm_up: crate::opt::loop_warm_up(chunk),
            entry_weight: crate::opt::entry_weight(),
            inline_construct: crate::value::object_inline_alloc_site(0).is_some(),
            #[cfg(test)]
            stub_memory_refused: crate::opt::stub_test_switches().0,
            #[cfg(test)]
            stubs_write_d8: crate::opt::stub_test_switches().1,
        }
    }

    /// What a shared stub that a compile off the main thread makes reads (P3 stage 2, step 17):
    /// these switches, read on the main thread with the snapshot.
    pub(crate) fn stub_reads(&self) -> StubReads {
        StubReads {
            entry_weight: self.entry_weight,
            inline_construct: self.inline_construct,
            #[cfg(test)]
            memory_refused: self.stub_memory_refused,
            #[cfg(test)]
            writes_d8: self.stubs_write_d8,
        }
    }
}

/// Generate `graph`'s code for `chunk` (the switches' options), from the chunk's Cfg and
/// initialized reads as its build made them (`build::Built`), on the calling thread, and map it:
/// the tests' path, which installs the code by hand.
#[cfg(test)]
pub(crate) fn compile_with(
    chunk: &Chunk,
    graph: &Graph,
    cfg: &crate::jit_ir::Cfg,
    initialized: crate::jit_ir::InitializedReads,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    options: Options,
) -> Result<Rc<JitCode>, Refusal> {
    compile_frames_with(
        chunk,
        graph,
        cfg,
        &[],
        initialized,
        layout,
        ilayout,
        options,
    )
}

/// [`compile_with`] of a graph with inlined frames (stage 3), whose chunks `inlined` gives with
/// their Cfgs, by `InlinedFrame::chunk`: the code holds them (`exit::Inlined`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_frames_with(
    chunk: &Chunk,
    graph: &Graph,
    cfg: &crate::jit_ir::Cfg,
    inlined: &[(Rc<Chunk>, crate::jit_ir::Cfg)],
    initialized: crate::jit_ir::InitializedReads,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    options: Options,
) -> Result<Rc<JitCode>, Refusal> {
    let view = chunk.freeze();
    let views: Vec<FrozenChunk> = inlined.iter().map(|(chunk, _)| chunk.freeze()).collect();
    let frames: Vec<crate::opt::verify::FrameChunk<'_>> = views
        .iter()
        .zip(inlined)
        .map(|(view, (_, cfg))| (view, cfg))
        .collect();
    let switches = Switches::read(chunk);
    let mut stubs = Stubs::live();
    let assembled = assemble(
        &view,
        graph,
        cfg,
        &frames,
        initialized,
        layout,
        ilayout,
        options,
        &switches,
        &mut stubs,
    )?;
    let executable = ExecutableBuffer::from_bytes(assembled.bytes())
        .ok_or(Refusal::whole("no executable memory"))?;
    let held = crate::opt::exit::Inlined {
        chunks: inlined.iter().map(|(chunk, _)| chunk.clone()).collect(),
        pins: Box::new([]),
        sites: Default::default(),
    };
    Ok(link(assembled, executable, chunk, held))
}

/// What a code generation's emission does with the process-wide shared stubs its sites request
/// (P3 stage 2, steps 12 and 17; the plan's point 3): on the main thread, find or make each in
/// the table as the template JIT does (`live`); in a job, find each in the main thread's table as
/// it stood at the snapshot (`jit::StubTableSnapshot`), and one the snapshot lacks in the table
/// itself, under its lock, made there where still missing, with the job's reads (`frozen`), so
/// that the job runs once.
pub(crate) struct Stubs<'s> {
    table: Option<&'s StubTableSnapshot>,
    /// The stubs the snapshot's table lacked: found under the table's lock, made there, or kept
    /// as copies where their assembly failed.
    pub(crate) met: MissingStubs,
}

impl<'s> Stubs<'s> {
    /// Find or make each stub in this thread's table (main thread only).
    pub(crate) fn live() -> Self {
        Stubs {
            table: None,
            met: MissingStubs::default(),
        }
    }

    /// Find each stub in `table`, and one it lacks in the table it was taken of.
    pub(crate) fn frozen(table: &'s StubTableSnapshot) -> Self {
        Stubs {
            table: Some(table),
            met: MissingStubs::default(),
        }
    }
}

/// A code generation's words and exit table, made without executable memory or any registry, so
/// that it crosses threads: the caller maps the words (`ExecutableBuffer`) and links the code
/// (`link`) on the main thread or maps them on the compile thread (P3 stage 2, step 12).
pub(crate) struct Assembled {
    pub(crate) words: Vec<u32>,
    pub(crate) table: Box<ExitTable>,
    /// Every pc's code offset: the trap's.
    pc_offsets: Vec<u32>,
    needs_global: bool,
    max_stack: usize,
}

impl Assembled {
    /// The code's bytes.
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the words' bytes, in the words' memory.
        unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.words.len() * 4)
        }
    }
}

/// The code `assembled` now that `executable` holds its bytes (main thread): its exit table
/// registered by its code's base with what the code holds for its inlined frames (`inlined`:
/// their chunks and the identities it embeds, stage 3; none without them), the JitCode, the perf
/// map's entry under `chunk`'s name.
pub(crate) fn link(
    assembled: Assembled,
    executable: ExecutableBuffer,
    chunk: &Chunk,
    inlined: crate::opt::exit::Inlined,
) -> Rc<JitCode> {
    let Assembled {
        mut table,
        pc_offsets,
        needs_global,
        max_stack,
        ..
    } = assembled;
    let mem = executable.as_ptr() as *mut u8;
    let len = executable.len();
    table.code_base = mem as usize;
    debug_assert!(
        table
            .frames
            .iter()
            .all(|frame| (frame.chunk as usize) < inlined.chunks.len()),
        "a code holds the chunks of its inlined frames"
    );
    crate::opt::exit::hold(mem as usize, inlined);
    crate::opt::exit::register(mem as usize, table);
    let code = JitCode {
        entry_kind: NativeEntryKind::FreshFrame,
        osr_entry_depths: Vec::new(),
        resume_depths: Vec::new(),
        needs_global,
        mem,
        len,
        pc_offsets,
        max_stack,
        executable,
        referenced: Cell::new(0),
    };
    profiler::register_optimized(chunk, mem, len);
    Rc::new(code)
}

/// Generate `graph`'s code for `chunk` with `options` and `switches`, its stubs from `stubs`,
/// into words (see [`Assembled`]): reads nothing but its arguments and process constants, so
/// that it runs on any thread. `inlined` gives the chunks of the graph's inlined frames with
/// their Cfgs, by `InlinedFrame::chunk` (stage 3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble(
    chunk: &FrozenChunk,
    graph: &Graph,
    cfg: &crate::jit_ir::Cfg,
    inlined: &[crate::opt::verify::FrameChunk<'_>],
    initialized: crate::jit_ir::InitializedReads,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    options: Options,
    switches: &Switches,
    stubs: &mut Stubs<'_>,
) -> Result<Assembled, Refusal> {
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
    check_graph(chunk, graph, cfg, inlined)?;
    if !words_inlinable(layout) && graph.nodes.iter().any(|node| reads_words(&node.op)) {
        return Err(Refusal::whole(
            "element words in a layout the code cannot address",
        ));
    }
    let creates = graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::StoreElement(Store::Create));
    if creates && !define_elem_inlinable(layout) {
        // The template's in-line append bakes the same offsets in (`emit_word_create`).
        return Err(Refusal::whole(
            "an Array's append in a layout the code cannot address",
        ));
    }
    let creations = graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::StoreNew { .. }));
    if creations && !set_prop_inlinable(layout) {
        // The template's creation probe and commit bake the same offsets in
        // (`emit_prop_create_probe`, `emit_set_prop_inline`).
        return Err(Refusal::whole(
            "a property's creation in a layout the code cannot address",
        ));
    }
    let alloc = regalloc::allocate(graph, &regalloc::ARM64)?;
    // An inlined frame's generic node saves the compiled frame's context words it switches
    // (P3 stage 3 step 3c) in the frame area, which the collector scans with the native stack.
    let inlined_generic = graph
        .nodes
        .iter()
        .any(|node| !node.frame.is_compiled() && matches!(node.op, NodeOp::Generic { .. }));
    let (frame, save_area) = entry_frame(&alloc, inlined_generic)?;
    let mut gen = Gen::new(
        chunk,
        graph,
        alloc,
        layout,
        ilayout,
        cfg,
        initialized,
        options,
        switches,
        frame,
    );
    gen.inlined = inlined;
    gen.save_area = save_area;
    gen.emit(stubs)?;
    gen.finish()
}

/// The checks that keep the code generator's assumptions: one entry without predecessors, a
/// terminator ending every block and nowhere else, a poll opening every loop header, entry
/// slots read in the entry block, a state on every node that can exit, and effects only on
/// operations whose baseline template the canonical frame can run; and with inlined frames
/// (stage 3), the frames' rules the deoptimizer relies on (`verify::frames`), each node of its
/// frame, and each state against its frame's chunk.
fn check_graph(
    chunk: &FrozenChunk,
    graph: &Graph,
    cfg: &crate::jit_ir::Cfg,
    inlined: &[crate::opt::verify::FrameChunk<'_>],
) -> Result<(), Refusal> {
    let entry = graph
        .entry
        .ok_or(Refusal::whole("a graph without an entry block"))?;
    if !graph.block(entry).preds.is_empty() {
        return Err(Refusal::whole("an entry block with predecessors"));
    }
    if !graph.frames.is_empty() {
        crate::opt::verify::frames(graph, (chunk, cfg), inlined)
            .map_err(|_| Refusal::whole("an inlined frame no exit can rebuild"))?;
    }
    // A frame's chunk and Cfg, by its index; None for a frame the graph has not.
    let frame_chunk = |frame: FrameIx| -> Option<crate::opt::verify::FrameChunk<'_>> {
        let Some(k) = frame.index().checked_sub(1) else {
            return Some((chunk, cfg));
        };
        inlined.get(graph.frames.get(k)?.chunk as usize).copied()
    };
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
            if !node.frame.is_compiled()
                && (frame_chunk(node.frame).is_none() || node.op.reads_compiled_frame())
            {
                return Err(Refusal::at(
                    node.pc as usize,
                    "an inlined frame's node that reads the compiled frame",
                ));
            }
            match node.op {
                NodeOp::EntrySlot(_) if index != entry.index() => {
                    return Err(Refusal::at(
                        node.pc as usize,
                        "an entry slot outside the entry block",
                    ));
                }
                NodeOp::Effect { pc, pops, pushes } | NodeOp::Generic { pc, pops, pushes } => {
                    // The node's frame's chunk and Cfg: an inlined frame's generic node's are
                    // its callee's (P3 stage 3 step 3c), a canonical-frame effect's the
                    // compiled function's only.
                    let Some((frame_view, frame_cfg)) = frame_chunk(node.frame).filter(|_| {
                        node.frame.is_compiled() || !matches!(node.op, NodeOp::Effect { .. })
                    }) else {
                        return Err(Refusal::at(pc as usize, "an effect in an inlined frame"));
                    };
                    let ops = frame_view.jit_ops();
                    let op = ops
                        .get(pc as usize)
                        .ok_or(Refusal::at(pc as usize, "an effect past the chunk"))?;
                    if !effect_supported(op) {
                        return Err(Refusal::at(
                            pc as usize,
                            "an effect on a control-flow operation",
                        ));
                    }
                    let depth = frame_cfg
                        .stack_depth_at(pc as usize)
                        .ok_or(Refusal::at(pc as usize, "an effect at an unreachable pc"))?;
                    let n_frame_slots = if node.frame.is_compiled() {
                        graph.n_slots
                    } else {
                        frame_view.jit_frame().1
                    };
                    let stale = if matches!(node.op, NodeOp::Generic { .. }) {
                        stale_slots(frame_view, n_frame_slots, graph, node).len()
                    } else {
                        0
                    };
                    if (pops as usize) > depth || node.inputs.len() != pops as usize + stale {
                        return Err(Refusal::at(pc as usize, "an effect's operands"));
                    }
                    // An inlined frame's generic node publishes its frame's inline site, whose
                    // caller position is the compiled frame's call's (the oracle's): a frame
                    // inlined into an inlined frame (no plan builds one today: the oracle keeps
                    // leaf callees) would need its site to chain its caller's.
                    if graph
                        .inlined(node.frame)
                        .is_some_and(|frame| !graph.state(frame.parent).frame.is_compiled())
                    {
                        return Err(Refusal::at(
                            pc as usize,
                            "a generic node of a frame inlined into an inlined frame",
                        ));
                    }
                    // An inlined frame's node's operand words and slot window lie in the
                    // compiled frame's record, from its operand base (`lower_inlined_generic`).
                    if !node.frame.is_compiled()
                        && graph.n_slots + frame_cfg.jit_stack_capacity() + n_frame_slots
                            > crate::jit::FRAME_BUF
                    {
                        return Err(Refusal::at(pc as usize, "an inlined frame past its record"));
                    }
                    if let Some(after) = frame_cfg.stack_depth_at(pc as usize + 1) {
                        if !matches!(op, Op::Throw)
                            && depth - pops as usize + pushes as usize != after
                        {
                            return Err(Refusal::at(pc as usize, "an effect's stack effect"));
                        }
                    }
                }
                NodeOp::StoreNew { shape, .. } if !crate::value::is_cacheable_shape(shape) => {
                    return Err(Refusal::at(
                        node.pc as usize,
                        "a creation without a proving landing shape",
                    ));
                }
                NodeOp::CheckShape(ref shapes)
                | NodeOp::CheckProto { ref shapes, .. }
                | NodeOp::CheckStringProto { ref shapes, .. }
                | NodeOp::ShapeIs(ref shapes)
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
                // The VM resumes at the state's pc on the settled stack there, in the node's
                // frame (an inlined frame's chunk and Cfg).
                let state = graph.state(state);
                let Some((state_chunk, cfg)) =
                    frame_chunk(state.frame).filter(|_| state.frame == node.frame)
                else {
                    return Err(Refusal::at(state.pc as usize, "a frame state's frame"));
                };
                let ops = state_chunk.jit_ops();
                if cfg.stack_depth_at(state.pc as usize) != Some(state.stack.len()) {
                    return Err(Refusal::at(
                        state.pc as usize,
                        "a frame state's operand depth",
                    ));
                }
                let n_slots = if state.frame.is_compiled() {
                    graph.n_slots
                } else {
                    state_chunk.jit_frame().1
                };
                if state.slots.len() != n_slots
                    || !state.frame.is_compiled() && state.slots.contains(&SlotSource::Frame)
                {
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

/// The slots generic node `node` writes before its template, in the order of its slot inputs
/// (P3 stage 3 step 3a, `build::Builder::effect`): those its operation names that its state holds
/// as values, their frame words stale; `chunk` and `n_slots` its frame's (an inlined frame's
/// node writes them into its slot window: step 3c).
fn stale_slots(chunk: &FrozenChunk, n_slots: usize, graph: &Graph, node: &Node) -> Vec<u16> {
    let Some(state) = node.state else {
        return Vec::new();
    };
    let state = graph.state(state);
    crate::opt::liveness::frame_slots(&chunk.jit_ops()[node.pc as usize], n_slots)
        .into_iter()
        .filter(|&(slot, _)| {
            matches!(
                state.slots.get(usize::from(slot)),
                Some(SlotSource::Value(_))
            )
        })
        .map(|(slot, _)| slot)
        .collect()
}

/// Whether a node addresses element words or their header, whose offsets the layout must let
/// the code reach (`words_inlinable`, the element templates' gate).
fn reads_words(op: &NodeOp) -> bool {
    matches!(
        op,
        NodeOp::LoadElements
            | NodeOp::LoadElementsUsed
            | NodeOp::LoadElement(_)
            | NodeOp::LoadElementOrUndefined
            | NodeOp::StoreElement(_)
    )
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
/// The frame area: the spill slots, then where `save_area` asks for it the inlined frames' save
/// area (`SAVE_AREA_BYTES` at the returned offset: P3 stage 3 step 3c), then the saves of the
/// callee-saved registers the code writes.
fn entry_frame(alloc: &Allocation, save_area: bool) -> Result<(EntryFrame, Option<u32>), Refusal> {
    let mut saves = Vec::new();
    let mut offset = alloc.spill_bytes.next_multiple_of(16);
    let area = save_area.then_some(offset);
    if save_area {
        offset += SAVE_AREA_BYTES;
    }
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
    Ok((
        EntryFrame {
            extra: offset,
            saves,
        },
        area,
    ))
}

/// The inlined frames' save area (P3 stage 3 step 3c): the compiled frame's chunk, `this`,
/// running environment and its parent, and slots base, while an inlined frame's generic node
/// runs with its own (`Gen::lower_inlined_generic`), at these byte offsets from its start.
const SAVE_AREA_BYTES: u32 = 48;
const SAVE_CHUNK: u32 = 0;
const SAVE_THIS: u32 = 8;
const SAVE_ENV: u32 = 16;
const SAVE_ENV_PARENT: u32 = 24;
const SAVE_SLOTS: u32 = 32;

/// Out-of-line code, emitted after the body.
enum Cold {
    /// A boxed f64's NaN case (`box_f64`): the canonical NaN's word into `dst`.
    CanonNan {
        label: usize,
        dst: u32,
        resume: usize,
    },
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
    /// TruncateNumberOrOddball's NaN case, the word in `word`: the canonical NaN, undefined and
    /// null give 0 and a Boolean its bit, into `dst`; any other word (a String, a Symbol, a
    /// BigInt, an object) takes `exit`.
    NumberOrOddball {
        label: usize,
        word: u32,
        dst: u32,
        exit: usize,
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
    /// A test node's object its in-line code does not decide (P3 stage 3 step L1: one that may be
    /// [[IsHTMLDDA]], a proxy): `helper(ctx, word, arg)`, which runs no JavaScript, allocates
    /// nothing and does not collect, its 0 or 1 to `dst`.
    TestCall {
        label: usize,
        node: NodeIx,
        word: u32,
        dst: u32,
        resume: usize,
        helper: u64,
        arg: TestArg,
    },
    /// The moves of an edge into a merge from a block with two successors.
    Edge {
        label: usize,
        from: BlockIx,
        to: BlockIx,
    },
    /// An element read's hole or index past the used words (P3 stage 2 step 2): undefined into
    /// `dst` where no element can come from the prototype chain of `node`'s receiver
    /// (`emit_no_inherited_elements`), else `exit`; with `negative`, a negative index in its w
    /// register takes its exit first.
    ElementUndefined {
        label: usize,
        node: NodeIx,
        dst: u32,
        resume: usize,
        exit: usize,
        negative: Option<(u32, usize)>,
    },
    /// An int32 element store's kind bits when INT32 is clear or the words are unwritable (the
    /// flags in x9): unwritable words take `unwritable`; otherwise the int32 may replace the word
    /// that broke a fact, so KIND_SCANNED goes, and the store resumes.
    ElementKinds {
        label: usize,
        out: u32,
        resume: usize,
        unwritable: usize,
    },
    /// An element store's unwritable words (in `out`): frozen ones exit, a shared literal's
    /// block is made the object's own by a pure call (`unshare_elements`), whose words `out`
    /// takes before the store runs again from `retry`.
    Unwritable {
        label: usize,
        node: NodeIx,
        out: u32,
        retry: usize,
        exit: usize,
    },
    /// An element store of an edge into words that keep a remembered record: the dirty bit of
    /// its run by a pure call (`remember_element`), then `resume`.
    Remember {
        label: usize,
        node: NodeIx,
        out: u32,
        resume: usize,
    },
    /// An Array's hole fill or append (StoreElement(Create)): `emit_element_create`.
    ElementCreate {
        label: usize,
        node: NodeIx,
        out: u32,
        done: usize,
        exits: CreateExits,
    },
    /// A branch to a label (an effect's throw into its exit).
    Jump { label: usize, target: usize },
    /// An inlined frame's generic node's throw (P3 stage 3 step 3c): the compiled frame's
    /// context words it switched restored, then a branch to the throw epilogue or its exit.
    Restore {
        label: usize,
        switched: Switched,
        target: usize,
    },
    /// A creation the in-line commit cannot make (StoreNew, P3 stage 3 step L3): the insert by a
    /// pure call (`store_new_slow`), the Object in x17, then `done`, or `exit` where it did not
    /// insert.
    CreateSlow {
        label: usize,
        node: NodeIx,
        done: usize,
        exit: usize,
    },
}

/// A test call's second argument (`Cold::TestCall`): an immediate, or a register's word.
#[derive(Clone, Copy)]
enum TestArg {
    Imm(u64),
    Reg(u32),
}

/// The exits of an Array's hole fill or append: a receiver, prototype or `length` the in-line
/// path does not take, a gap past the next word, and growth that failed.
#[derive(Clone, Copy)]
struct CreateExits {
    wrong: usize,
    gap: usize,
    grow: usize,
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
    chunk: &'a FrozenChunk,
    graph: &'a Graph,
    alloc: Allocation,
    layout: &'a crate::value::JitLayout,
    ilayout: &'a crate::interpreter::InterpLayout,
    cfg: &'a crate::jit_ir::Cfg,
    options: Options,
    switches: &'a Switches,
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
    /// The moved checks' exits (P3 stage 3 step L1c): each one's place in `exits` and the checks
    /// it stands for (`ExitTable::moved`).
    moved_exits: Vec<(usize, Box<[crate::opt::exit::MovedSite]>)>,
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
    /// Each inlined frame's index in the table's frames, once an exit passes through it (stage
    /// 3), by the graph's frame index; and the frames' recipes so far.
    frame_index: Vec<Option<u16>>,
    frames: Vec<crate::opt::exit::FrameRecipe>,
    /// The chunks of the graph's inlined frames, with their Cfgs, by `InlinedFrame::chunk`.
    inlined: &'a [crate::opt::verify::FrameChunk<'a>],
    /// The frame area's save area for inlined frames' generic nodes (`SAVE_AREA_BYTES` at this
    /// byte offset), where the graph has one.
    save_area: Option<u32>,
    /// The block whose code is being emitted, whose loop caches its reads take (P3 stage 3 step
    /// R1); None in out-of-line code but an edge's moves, which read the slots.
    cur_block: Option<BlockIx>,
    /// Each loop header with caches: the label after their loads, which its loop's own edges
    /// take.
    after_loads: Vec<Option<usize>>,
}

/// What an inlined frame's generic node switched of the context (P3 stage 3 step 3c), which its
/// return and its throw restore (`Gen::restore_context`).
#[derive(Clone, Copy)]
struct Switched {
    this: bool,
    env: bool,
    slots: bool,
    /// A call node (P3 stage 3 step 3c-2, piece B2): its call template switches `JitCtx::chunk`,
    /// `Interp::call_site` and x21 on the paths that reach a helper only, and restores them where
    /// those paths end (`jit::emit_inlined_call_switch`, `jit::emit_inlined_call_restore`), so
    /// that the node saves no chunk word and its unwind writes constants; any other node switches
    /// the chunk and publishes its frame's site before its template.
    call: bool,
}

impl<'a> Gen<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        chunk: &'a FrozenChunk,
        graph: &'a Graph,
        alloc: Allocation,
        layout: &'a crate::value::JitLayout,
        ilayout: &'a crate::interpreter::InterpLayout,
        cfg: &'a crate::jit_ir::Cfg,
        initialized: crate::jit_ir::InitializedReads,
        options: Options,
        switches: &'a Switches,
        frame: EntryFrame,
    ) -> Self {
        let ops = chunk.jit_ops();
        let mut a = asm::Asm::new();
        a.set_stubs_ok(true);
        a.set_first_stage(!chunk.jit_second_stage());
        a.set_fixed_activation(chunk.jit_fixed_activation());
        a.set_entry_weight(switches.entry_weight);
        a.set_inline_construct(switches.inline_construct);
        if switches.tdz_proof {
            a.set_initialized_reads(initialized);
        }
        let fast = switches.fast;
        let blocks = (0..graph.blocks.len()).map(|_| a.new_label()).collect();
        let mut after_loads = vec![None; graph.blocks.len()];
        for c in &alloc.caches {
            if after_loads[c.header.index()].is_none() {
                after_loads[c.header.index()] = Some(a.new_label());
            }
        }
        let l_ret_ok = a.new_label();
        let l_ret_throw = a.new_label();
        let l_direct_finish = a.new_label();
        let l_exit_stub = a.new_label();
        let l_trap = a.new_label();
        Gen {
            a,
            chunk,
            graph,
            alloc,
            layout,
            ilayout,
            cfg,
            options,
            switches,
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
            moved_exits: Vec::new(),
            cold: Vec::new(),
            fast,
            array_intrinsics_on: switches.array_intrinsics,
            function_call_intrinsic_on: switches.function_call_intrinsic,
            direct_on: fast & (1 << 20) != 0 && switches.direct_shared_context,
            in_loop: loop_body_mask(ops),
            last_effect: None,
            table: Box::new(ExitTable {
                name: chunk.label().into(),
                limit: switches.exit_limit,
                loop_limit: switches.loop_exit_limit,
                loop_warm_up: switches.loop_warm_up,
                ..ExitTable::default()
            }),
            frame_index: vec![None; graph.frames.len()],
            frames: Vec::new(),
            inlined: &[],
            save_area: None,
            cur_block: None,
            after_loads,
        }
    }

    fn emit(&mut self, stubs: &mut Stubs<'_>) -> Result<(), Refusal> {
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
            self.load_caches(b);
            self.cur_block = Some(b);
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
        let context = StubContext {
            layout: self.layout,
            ilayout: Some(self.ilayout),
            direct: self.direct_on.then(|| DirectCallContext {
                attempted_off: self.chunk.jit_inline_attempted_off(),
                runs_off: self.chunk.jit_runs_off(),
                retry_off: self.chunk.jit_inline_retry_at_off(),
                finish_stub: self.l_direct_finish,
            }),
        };
        match stubs.table {
            None => emit_shared_stubs(&mut self.a, &context),
            Some(table) => emit_shared_stubs_from(
                &mut self.a,
                &context,
                table,
                &self.switches.stub_reads(),
                &mut stubs.met,
            ),
        }
        self.a.bind(self.l_direct_finish);
        if self.direct_on && self.a.label_referenced(self.l_direct_finish) {
            emit_direct_finish_stub(&mut self.a, self.ilayout, self.layout.valid);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Assembled, Refusal> {
        // Read after the emission found or made its stubs (`emit`), on this thread: a stub that
        // writes d8-d15 set the flag before it entered the table (P3 stage 2, step 17).
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
        // Each moved check's exit by its index among the sorted records (step L1c).
        let mut moved: Vec<(u32, Box<[crate::opt::exit::MovedSite]>)> = self
            .moved_exits
            .drain(..)
            .map(|(place, sites)| {
                let offset = offsets[1 + place];
                let index = table
                    .return_offsets
                    .binary_search(&offset)
                    .expect("each exit's return offset");
                (index as u32, sites)
            })
            .collect();
        moved.sort_by_key(|&(index, _)| index);
        table.moved = moved.into_boxed_slice();
        table.frames = self.frames.into_boxed_slice();
        let names_global = |o: &Op| {
            matches!(
                o,
                Op::LoadName(..)
                    | Op::LoadNameForCall(..)
                    | Op::LoadNameIn(..)
                    | Op::LoadNameForCallIn(..)
            )
        };
        // An inlined frame's generic nodes run their templates on the compiled frame's record
        // (P3 stage 3 step 3c): its operand words and slot window, which every entry's stack
        // must hold, and the global names its templates read.
        let mut max_stack = self.cfg.jit_stack_capacity();
        let mut inlined_global = false;
        for node in &self.graph.nodes {
            let (Some(frame), NodeOp::Generic { .. }) = (self.graph.inlined(node.frame), &node.op)
            else {
                continue;
            };
            let (view, cfg) = self.inlined[frame.chunk as usize];
            max_stack = max_stack.max(cfg.jit_stack_capacity() + view.jit_frame().1);
            inlined_global |= names_global(&view.jit_ops()[node.pc as usize]);
        }
        Ok(Assembled {
            words,
            table,
            // Made here, on the compile's thread.
            pc_offsets: vec![trap; ops.len()],
            needs_global: inlined_global || ops.iter().any(names_global),
            max_stack,
        })
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

/// NullOrLikeUndefined's cold path (P3 stage 3 step L1): whether the object `word`, whose
/// ic_plain is clear, is the [[IsHTMLDDA]] object. It runs no JavaScript, allocates nothing and
/// does not collect.
unsafe extern "C" fn emulates_undefined(ctx: *mut JitCtx, word: u64, _: u64) -> u64 {
    let i = unsafe { &*(*ctx).interp };
    u64::from(i.is_htmldda(&unsafe { crate::value::Value::from_bits(word) }))
}

/// StringEqual's cold path (P3 stage 3 step L1): IsStrictlyEqual of the Strings `a` and `b` of
/// equal lengths, their contents compared (`LStr`'s equality). It runs no JavaScript, allocates
/// nothing and does not collect.
unsafe extern "C" fn strings_equal(ctx: *mut JitCtx, a: u64, b: u64) -> u64 {
    let i = unsafe { &*(*ctx).interp };
    // SAFETY: two Strings' words, which the code's frame keeps alive.
    let (a, b) = unsafe {
        (
            crate::value::Value::from_bits(a),
            crate::value::Value::from_bits(b),
        )
    };
    u64::from(i.strict_equals(&a, &b))
}

/// TypeofIs's cold path (P3 stage 3 step L1): the interpreter's typeof_matches of `word` for
/// the test whose discriminant is `test` (a proxy's callability is a field of it; an object
/// that may be [[IsHTMLDDA]] is looked up). It runs no JavaScript, allocates nothing and does
/// not collect.
unsafe extern "C" fn typeof_matches(ctx: *mut JitCtx, word: u64, test: u64) -> u64 {
    use crate::value::TypeofTest;
    let i = unsafe { &*(*ctx).interp };
    let test = [
        TypeofTest::Undefined,
        TypeofTest::Boolean,
        TypeofTest::Number,
        TypeofTest::BigInt,
        TypeofTest::String,
        TypeofTest::Symbol,
        TypeofTest::Object,
        TypeofTest::Function,
    ]
    .into_iter()
    .find(|&t| u64::from(t as u8) == test)
    .unwrap_or(TypeofTest::Never);
    u64::from(i.typeof_matches(&unsafe { crate::value::Value::from_bits(word) }, test))
}

extern "C" fn f64_remainder(dividend: f64, divisor: f64) -> f64 {
    dividend % divisor
}

/// The elements protector's proof at the live prototype epoch (`emit_no_inherited_elements`, P3
/// stage 2 step 2): `Interp::array_prototypes_unshadowed`, which renews a lapsed proof as the
/// VM's Rust callers renew it (a realm switch or a marked prototype's change lapses it). It runs
/// no JavaScript, allocates nothing in the heap and does not collect.
unsafe extern "C" fn renew_elements_protector(ctx: *mut JitCtx) -> u64 {
    let i = unsafe { &*(*ctx).interp };
    u64::from(i.array_prototypes_unshadowed())
}

/// A shared literal's block made the object's own before an element store writes it in place
/// (P3 stage 2 step 2; Maglev's EnsureWritableFastElements, V8's CopyElementsOnWrite): the
/// object's mutable borrow, which marks its card for the new block (gc_storage.rs), and
/// `Props::own_words_address`. It allocates a storage block and never collects (the engine's heap
/// collects at its safepoints only: value.rs, `new_cell_heap`), and runs no JavaScript. The
/// object's words after it, or 0 where the object is borrowed, which none is while generated code
/// runs.
unsafe extern "C" fn unshare_elements(cell: u64) -> u64 {
    let word = unsafe { crate::value::Value::from_bits(crate::value::PACK_OBJ | cell) };
    let Some(object) = word.as_obj() else {
        return 0;
    };
    let Ok(mut object) = object.try_borrow_mut() else {
        return 0;
    };
    object.props.own_words_address() as u64
}

/// Room for one more element word of an Array whose in-line append found its words full or the
/// static empty block (P3 stage 2 step 2; Warp's addDenseElementPure, Maglev's
/// MaybeGrowFastElements): `Props::grown_words_address` through the object's mutable borrow,
/// which marks its card for the new block. It allocates a storage block and never collects, and
/// runs no JavaScript. The object's words after it, or 0 where they cannot grow (the most words
/// a header counts) or the object is borrowed.
unsafe extern "C" fn grow_elements(cell: u64) -> u64 {
    let word = unsafe { crate::value::Value::from_bits(crate::value::PACK_OBJ | cell) };
    let Some(object) = word.as_obj() else {
        return 0;
    };
    let Ok(mut object) = object.try_borrow_mut() else {
        return 0;
    };
    object
        .props
        .grown_words_address()
        .map_or(0, |words| words as u64)
}

/// A creation's insert where StoreNew's in-line commit cannot make it (P3 stage 3 step L3): a key
/// index of the object's own, a named count of 8 or more, no key layout for the slot, or a record
/// without room. `Props::append_new`, the insert the VM's creation hit makes (Interp::try_ic_set),
/// of a key the receiver's shape proved absent, landing on `shape`: it maintains the key index
/// and the layout and grows the record. It allocates and neither collects (engine heaps collect at
/// their safepoints) nor runs JavaScript; the object's mutable borrow marks its card for the new
/// edge and any new block. 1, or 0 where the object is borrowed (nothing written).
unsafe extern "C" fn store_new_slow(cell: u64, key: u64, value: u64, shape: u64) -> u64 {
    let word = unsafe { crate::value::Value::from_bits(crate::value::PACK_OBJ | cell) };
    let Some(object) = word.as_obj() else {
        return 0;
    };
    let Ok(mut object) = object.try_borrow_mut() else {
        return 0;
    };
    // SAFETY: the node's key word, the site's name's key in this heap.
    let key = unsafe { crate::atom::PropKey::from_word(key) };
    let value = unsafe { crate::value::Value::from_bits(value) };
    object
        .props
        .append_new(key, crate::value::Property::plain(value), shape as u32);
    1
}

/// `out` where a call clobbers it: a node's own value, which no save of the live values keeps
/// (an element store's words), unless a callee-saved register holds it.
fn caller_saved(out: u32) -> Vec<u32> {
    if regalloc::ARM64.gp_callee.contains(&(out as u8)) {
        Vec::new()
    } else {
        vec![out]
    }
}

/// The dirty bit of an edge an element store wrote below a large block's traced length (P3 stage
/// 2 step 2; Warp's post barrier for large arrays, jit/VMFunctions.cpp:1054-1076):
/// `Words::remember_edge_at`. Allocates nothing.
unsafe extern "C" fn remember_element(words: u64, index: u64) {
    unsafe {
        crate::value::word_block::Words::remember_edge_at(words as *mut u64, index as u32 as usize)
    };
}

/// The exit reason a check reports when its node names none.
fn default_reason(op: &NodeOp) -> ExitReason {
    match op {
        NodeOp::CheckNumber | NodeOp::TruncateNumberOrOddball => ExitReason::NotNumber,
        NodeOp::CheckInt32 => ExitReason::NotInt32,
        NodeOp::CheckObject => ExitReason::NotObject,
        NodeOp::CheckShape(_)
        | NodeOp::CheckProto { .. }
        | NodeOp::CheckStringProto { .. }
        | NodeOp::CheckCreatable { .. }
        | NodeOp::CheckNoProto { .. }
        | NodeOp::StoreNew { .. }
        | NodeOp::StringLength => ExitReason::WrongShape,
        NodeOp::CheckBoolean => ExitReason::NotBoolean,
        NodeOp::CheckCallee { .. } => ExitReason::WrongCallee,
        NodeOp::CheckInitialized => ExitReason::Tdz,
        NodeOp::CheckString => ExitReason::NotString,
        NodeOp::CheckObjectOrNullish => ExitReason::NotObjectOrNullish,
        NodeOp::CheckEnvIs { .. }
        | NodeOp::CheckCellIs { .. }
        | NodeOp::CheckGlobalEpoch { .. }
        | NodeOp::CheckEnvGeneration { .. } => ExitReason::WrongName,
        NodeOp::I32Add | NodeOp::I32Sub | NodeOp::I32Mul | NodeOp::I32UShr => ExitReason::Overflow,
        NodeOp::CheckElements(_) => ExitReason::WrongElements,
        NodeOp::CheckIndex => ExitReason::NotInt32,
        NodeOp::CheckBounds | NodeOp::LoadElementOrUndefined => ExitReason::OutOfBounds,
        NodeOp::LoadElement(_) => ExitReason::Hole,
        NodeOp::StoreElement(_) => ExitReason::WrongElements,
        NodeOp::TypedData { .. } => ExitReason::Detached,
        NodeOp::TypedElement { .. } => ExitReason::OutOfBounds,
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
            // No frame state holds an environment (verify.rs): its address as it is.
            NodeOp::ConstEnv(address) => address,
            _ => unreachable!("a constant node"),
        }
    }

    /// A constant node's bits in its own representation (what a register of its class holds).
    fn const_bits(&self, v: NodeIx) -> u64 {
        match self.graph.node(v).op {
            NodeOp::ConstTagged(bits) | NodeOp::ConstF64(bits) | NodeOp::ConstEnv(bits) => bits,
            NodeOp::ConstI32(x) => u64::from(x as u32),
            NodeOp::ConstBool(b) => u64::from(b),
            _ => unreachable!("a constant node"),
        }
    }

    /// The register that holds a copy of spilled value `v` in the block being emitted, if a loop
    /// cache does (P3 stage 3 step R1, `regalloc::loop_caches`).
    fn cached(&self, v: NodeIx) -> Option<Home> {
        if !matches!(self.home(v), Home::Spill(_)) {
            return None;
        }
        self.alloc.cache_of(self.cur_block?, v)
    }

    /// The loads of a loop header's caches from their values' slots, then the label its loop's
    /// own edges take, so that the loads run once per entry and the back edges run nothing.
    fn load_caches(&mut self, header: BlockIx) {
        let Some(after) = self.after_loads[header.index()] else {
            return;
        };
        for k in 0..self.alloc.caches.len() {
            let cache = self.alloc.caches[k];
            if cache.header != header {
                continue;
            }
            let Home::Spill(offset) = self.home(cache.value) else {
                unreachable!("a cache copies a spilled value")
            };
            match cache.reg {
                Home::Gp(r) => self.a.ldr_imm(u32::from(r), 31, offset),
                Home::Fp(r) => self.a.ldr_d_imm(u32::from(r), 31, offset),
                _ => unreachable!("a cache is a register"),
            }
        }
        self.a.bind(after);
    }

    /// The label a branch from `from` into `to` takes: `to`'s, or after its caches' loads for an
    /// edge of `to`'s loop.
    fn block_label(&self, from: BlockIx, to: BlockIx) -> usize {
        match self.after_loads[to.index()] {
            Some(after) if self.alloc.enters_loaded(from, to) => after,
            _ => self.blocks[to.index()],
        }
    }

    /// The x register that holds `v` here: its home, or `scratch`, loaded from its spill slot or
    /// with its constant.
    fn gp(&mut self, v: NodeIx, scratch: u32) -> u32 {
        if let Some(Home::Gp(r)) = self.cached(v) {
            return u32::from(r);
        }
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

    /// The d register that holds `v` here (as [`Gen::gp`]; a constant by FMOV's 8-bit immediate
    /// where it encodes, else through x30).
    fn fp(&mut self, v: NodeIx, scratch: u32) -> u32 {
        if let Some(Home::Fp(r)) = self.cached(v) {
            return u32::from(r);
        }
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
                } else if let Some(imm8) = asm::fmov_imm8_d(bits) {
                    self.a.fmov_d_imm(scratch, imm8);
                } else {
                    self.a.mov_imm64(X30, bits);
                    self.a.fmov_d_x(scratch, X30);
                }
                scratch
            }
            Home::Gp(_) | Home::None => unreachable!("a read value has a d home"),
        }
    }

    /// `v`'s value where it is an int32 constant, which an instruction may take as its immediate
    /// (P3 stage 2 step 7; Maglev's TryGetInt32ConstantInput). No register holds a constant
    /// (`regalloc`): one whose immediate does not encode is loaded into a scratch register at
    /// its use, as before.
    fn const_i32(&self, v: NodeIx) -> Option<i32> {
        match self.graph.node(v).op {
            NodeOp::ConstI32(x) => Some(x),
            _ => None,
        }
    }

    /// The immediate a constant operand of `node`, an operation on two int32s, takes as `encode`
    /// gives it, and the operand that stays in a register: the right operand's, else, where the
    /// operation commutes, the left one's, the operands swapped. Maglev's ARM64 code generator
    /// takes the right operand's (TryGetAddImmediateInt32ConstantInput and
    /// TryGetLogicalImmediateInt32ConstantInput, maglev-ir-arm64.cc).
    fn imm_operand<T>(
        &self,
        node: &Node,
        commutes: bool,
        encode: impl Fn(i32) -> Option<T>,
    ) -> Option<(NodeIx, T)> {
        let (left, right) = (node.inputs[0], node.inputs[1]);
        if let Some(imm) = self.const_i32(right).and_then(&encode) {
            return Some((left, imm));
        }
        let imm = self.const_i32(left).filter(|_| commutes).and_then(encode)?;
        Some((right, imm))
    }

    /// An int32 add (`sub` clear) or subtract of the constant `c` by ADD/SUB immediate
    /// (`asm::add_sub_imm`): whether the instruction subtracts, and its immediate. A negative
    /// constant takes the opposite operation by its magnitude, as V8's and SpiderMonkey's
    /// macro-assemblers take one; its flags are the same (AddWithCarry sums x + c either way,
    /// so N, Z, C and V agree, V the overflow exit's and N, Z and V a comparison's). INT32_MIN's
    /// magnitude encodes in neither form.
    fn add_sub_imm(c: i32, sub: bool) -> Option<(bool, u32)> {
        let magnitude = c.unsigned_abs();
        asm::add_sub_imm(magnitude).then_some((sub != (c < 0), magnitude))
    }

    /// Compare the int32s `a` and `b` (CMP) and give the condition to test for `code`, a
    /// condition on (a, b): a constant `b` by CMP's immediate (a negative one by CMN's, by its
    /// magnitude), a constant `a` likewise with the operands swapped and the condition reversed,
    /// any other pair in registers.
    fn cmp_i32(&mut self, a: NodeIx, b: NodeIx, code: u32) -> u32 {
        let imm = |c| Self::add_sub_imm(c, true);
        let (x, (subtract, imm), code) = if let Some(i) = self.const_i32(b).and_then(imm) {
            (a, i, code)
        } else if let Some(i) = self.const_i32(a).and_then(imm) {
            (b, i, swapped(code))
        } else {
            let ra = self.gp(a, X16);
            let rb = self.gp(b, X17);
            self.a.cmp_reg_w(ra, rb);
            return code;
        };
        let r = self.gp(x, X16);
        if subtract {
            self.a.cmp_imm_w(r, imm);
        } else {
            self.a.cmn_imm_w(r, imm);
        }
        code
    }

    /// `rd` = the int32 in `rn` shifted by the int32 `count` (op: 0 LSL, 1 LSR, 2 ASR), which
    /// JavaScript takes modulo 32 as the register forms do: a constant count as the immediate,
    /// its low five bits.
    fn shift_i32(&mut self, op: u32, rd: u32, rn: u32, count: NodeIx) {
        let Some(c) = self.const_i32(count) else {
            let rm = self.gp(count, X17);
            return self.a.shift_w(op, rd, rn, rm);
        };
        let c = c as u32 & 31;
        match op {
            0 => self.a.lsl_imm_w(rd, rn, c),
            1 => self.a.lsr_imm_w(rd, rn, c),
            _ => self.a.asr_imm_w(rd, rn, c),
        }
    }

    /// `x` = the Number word of the f64 in `d`: its bits, a NaN's replaced by the canonical NaN
    /// out of line (any other NaN reads as a tag), as the first stage's templates box a sum and
    /// SpiderMonkey canonicalizes (canonicalizeDoubleNaN, jit/MacroAssembler-inl.h:971-976): a
    /// branch over the replacement, which a Number's word never takes, so that the word is ready
    /// when its FMOV is, where a CSEL on the comparison's flags waited for both and put the
    /// comparison on the word's path (a small callee's result, on its caller's loop-carried
    /// chain). Clobbers the flags.
    fn box_f64(&mut self, d: u32, x: u32) {
        let nan = self.a.new_label();
        let resume = self.a.new_label();
        self.a.fmov_x_d(x, d);
        self.a.fcmp(d, d);
        self.a.b_cond(VS, nan);
        self.a.bind(resume);
        self.cold.push(Cold::CanonNan {
            label: nan,
            dst: x,
            resume,
        });
    }

    /// `v`'s execution word in an x register (`scratch` unless it is a tagged word in its home):
    /// a Number for an f64 (NaN canonicalized, `box_f64`) or an int32, the Boolean or Object word
    /// for a Bool or a cell. Clobbers x30, d31 and the flags.
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
                self.box_f64(d, scratch);
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
            Rep::Env => unreachable!("a frame state holds no environment (verify.rs)"),
            Rep::None => unreachable!("a frame state holds values"),
        }
    }

    /// Where `n` writes its x value: its home register, or x9 for a spill slot or no home.
    /// Whether to use FJCVTZS (ToInt32 itself, FEAT_JSCVT).
    fn jscvt(&self) -> bool {
        self.switches.jscvt && !self.options.no_jscvt
    }

    /// ToInt32 of the f64 in `d` into the w register `w` (TruncateToInt32's code, and
    /// TruncateNumberOrOddball's for a Number).
    fn truncate(&mut self, d: u32, w: u32) {
        if self.jscvt() {
            // ToInt32 itself (FEAT_JSCVT), as V8's TruncateDoubleToI uses it.
            self.a.fjcvtzs_w_d(w, d);
        } else {
            // Below 2^63 the conversion to 64 bits is exact and its low word is ToInt32 (NaN
            // converts to 0). INT64_MIN and INT64_MAX mean it saturated, which subtracting or
            // adding 1 detects as signed overflow (V8's TryConvertDoubleToInt64); the cold path
            // computes the low word then.
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
        let proof = match node.op {
            NodeOp::CheckEnvIs { proof, .. }
            | NodeOp::CheckCellIs { proof, .. }
            | NodeOp::CheckGlobalEpoch { proof, .. }
            | NodeOp::CheckEnvGeneration { proof, .. } => proof,
            NodeOp::TruncateNumberOrOddball => TRUNCATION_EXIT,
            // A graph-inlined site's guard's place in its chain (P3 stage 3, step 2).
            NodeOp::CheckCallee { target } => self.chunk.jit_inline_target(target).guard,
            _ => 0,
        };
        let frame = self.frame_recipe(state.frame)?;
        let record = ExitRecord {
            pc: state.pc,
            kind,
            reason,
            slots: slots.into(),
            stack: stack.into(),
            handlers: state.handlers.clone(),
            shape_check: None,
            proof,
            count: Cell::new(0),
            frame,
        };
        // A moved check's exit keeps the checks it stands for (P3 stage 3 step L1c), each in its
        // chunk: the code's own, or the inlined chunk its frame runs.
        if kind == ExitKind::Check {
            if let Ok(at) = graph.moved.binary_search_by_key(&n, |check| check.node) {
                let sites = graph.moved[at]
                    .sites
                    .iter()
                    .map(|site| crate::opt::exit::MovedSite {
                        pc: site.pc,
                        chunk: graph
                            .inlined(site.frame)
                            .map_or(crate::opt::exit::OWN_CHUNK, |inlined| inlined.chunk),
                        reason: site.reason,
                        anyway: site.anyway,
                    })
                    .collect();
                self.moved_exits.push((self.exits.len(), sites));
            }
        }
        let site = self.a.new_label();
        let ret = self.a.new_label();
        self.exits.push((site, ret, record));
        Ok(site)
    }

    /// The table's index of inlined frame `frame`'s recipe (0 for the compiled function's frame),
    /// made at the first exit through the frame, after its parent's (stage 3): its callee's and
    /// receiver's sources and its parent state's recipe, which every exit through the frame
    /// shares, each value having one home for its whole range (`regalloc`).
    fn frame_recipe(&mut self, frame: FrameIx) -> Result<u16, Refusal> {
        let graph = self.graph;
        let Some(inlined) = graph.inlined(frame) else {
            return Ok(0);
        };
        if let Some(index) = self.frame_index[frame.index() - 1] {
            return Ok(index);
        }
        let parent = graph.state(inlined.parent);
        let parent_frame = self.frame_recipe(parent.frame)?;
        let mut slots = Vec::new();
        for (slot, source) in parent.slots.iter().enumerate() {
            if let SlotSource::Value(v) = *source {
                slots.push((slot as u16, self.source(v, parent.pc)?));
            }
        }
        let stack = parent
            .stack
            .iter()
            .map(|&v| self.source(v, parent.pc))
            .collect::<Result<Vec<_>, _>>()?;
        let recipe = crate::opt::exit::FrameRecipe {
            chunk: inlined.chunk,
            call_pc: inlined.call_pc,
            parent: parent_frame,
            callee: self.source(inlined.callee, parent.pc)?,
            receiver: self.source(inlined.receiver, parent.pc)?,
            slots: slots.into(),
            stack: stack.into(),
            handlers: parent.handlers.clone(),
        };
        self.frames.push(recipe);
        let index = u16::try_from(self.frames.len())
            .map_err(|_| Refusal::whole("more inlined frames than an exit can name"))?;
        self.frame_index[frame.index() - 1] = Some(index);
        Ok(index)
    }

    /// An effect's or a generic node's exit for a throw at a pc with a handler: a canonical-frame
    /// effect's recipe writes nothing (its code wrote the state to the frame before it), a
    /// generic node's is its `Graph::throw_recipe`, each value in a home its call preserves (P3
    /// stage 3 step 3a: the allocator reads them after the call), the operand words above it left
    /// as the template left them.
    fn exit_throw(&mut self, n: NodeIx) -> Result<usize, Refusal> {
        let graph = self.graph;
        let node = graph.node(n);
        let state = graph.state(node.state.expect("checked: a state on every exit"));
        let (slots, stack) = graph.throw_recipe(n);
        let preserved = |source: &Source| match *source {
            Source::Gp { reg, .. } => regalloc::ARM64.gp_callee.contains(&reg),
            Source::Fp { reg } => regalloc::ARM64.fp_callee.contains(&reg),
            Source::Spill { .. } | Source::Const(_) | Source::Frame => true,
        };
        let mut recipe_slots = Vec::with_capacity(slots.len());
        for (slot, v) in slots {
            recipe_slots.push((slot, self.source(v, state.pc)?));
        }
        let mut recipe_stack = Vec::with_capacity(state.stack.len());
        if matches!(node.op, NodeOp::Generic { .. }) {
            for v in stack {
                recipe_stack.push(match v {
                    Some(v) => self.source(v, state.pc)?,
                    None => Source::Frame,
                });
            }
        } else {
            recipe_stack.resize(state.stack.len(), Source::Frame);
        }
        // An inlined frame's throw (P3 stage 3 step 3c) also reads its chain's recipes, the
        // frames' callees, receivers and parent states, after the call.
        let frame = self.frame_recipe(state.frame)?;
        let mut chain = Vec::new();
        let mut at = frame;
        while at != 0 {
            let recipe = &self.frames[usize::from(at) - 1];
            chain.push(recipe.callee);
            chain.push(recipe.receiver);
            chain.extend(recipe.slots.iter().map(|&(_, source)| source));
            chain.extend(recipe.stack.iter().copied());
            at = recipe.parent;
        }
        if !recipe_slots
            .iter()
            .map(|(_, source)| source)
            .chain(&recipe_stack)
            .chain(&chain)
            .all(preserved)
        {
            return Err(Refusal::at(
                node.pc as usize,
                "a throw's value in a register its call clobbers",
            ));
        }
        let record = ExitRecord {
            pc: state.pc,
            kind: ExitKind::EffectThrow,
            reason: ExitReason::Throw,
            slots: recipe_slots.into(),
            stack: recipe_stack.into(),
            handlers: state.handlers.clone(),
            shape_check: None,
            proof: 0,
            count: Cell::new(0),
            frame,
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

    /// Branch to `site` unless the shape in `shape` (a w register) is one of `shapes` (X16
    /// scratch): an immediate compare where the id fits 12 bits, a built one otherwise.
    fn shape_compare(&mut self, shape: u32, shapes: &[u32], site: usize) {
        let ok = self.a.new_label();
        for (k, &expected) in shapes.iter().enumerate() {
            if expected < 4096 {
                self.a.cmp_imm_w(shape, expected);
            } else {
                self.a.mov_imm64(X16, u64::from(expected));
                self.a.cmp_reg_w(shape, X16);
            }
            if k + 1 == shapes.len() {
                self.a.b_cond(NE, site);
            } else {
                self.a.b_cond(EQ, ok);
            }
        }
        self.a.bind(ok);
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
            | NodeOp::ConstEnv(_)
            | NodeOp::Phi => {}
            NodeOp::Env
            | NodeOp::EnvParent
            | NodeOp::LoadBinding { .. }
            | NodeOp::CheckInitialized
            | NodeOp::StoreBinding { .. }
            | NodeOp::GlobalObject
            | NodeOp::CheckEnvIs { .. }
            | NodeOp::CheckCellIs { .. }
            | NodeOp::CheckGlobalEpoch { .. }
            | NodeOp::CheckEnvGeneration { .. }
            | NodeOp::LoadGlobalLexical { .. } => self.lower_name(n)?,
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
            NodeOp::SlotResult(slot) => {
                // The template's write of the slot, read right after its node (and its
                // projections).
                if self.last_effect != Some(input(0)) {
                    return Err(Refusal::at(
                        node.pc as usize,
                        "a slot's result apart from its node",
                    ));
                }
                if !unused {
                    let dst = self.gp_dst(n);
                    match graph.inlined(node.frame) {
                        None => self.a.ldr_imm(dst, 22, u32::from(*slot) * 8),
                        Some(frame) => {
                            // An inlined frame's node's slot window, above its frame's operand
                            // capacity from the operand base (`lower_inlined_generic`).
                            let (_, cfg) = self.inlined[frame.chunk as usize];
                            let window = cfg.jit_stack_capacity() as u32;
                            self.a.ldr_imm(
                                X16,
                                19,
                                std::mem::offset_of!(JitCtx, stack_base) as u32,
                            );
                            self.a.ldr_imm(dst, X16, 8 * (window + u32::from(*slot)));
                        }
                    }
                    self.gp_done(n, dst);
                }
            }
            NodeOp::Projection(k) => {
                let effect = input(0);
                let (NodeOp::Effect { pushes, .. } | NodeOp::Generic { pushes, .. }) =
                    graph.node(effect).op
                else {
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
                        ways: shapes.len().min(usize::from(u8::MAX)) as u8,
                        hop: false,
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
                self.shape_compare(X17, shapes, site);
                self.gp_done(n, cell);
            }
            NodeOp::CheckProto { shapes, ways } => {
                let site = self.check_exit(n)?;
                let pc = graph.state(node.state.expect("checked")).pc;
                let object = self.source(input(0), pc)?;
                if let Some((_, _, record)) = self.exits.last_mut() {
                    record.shape_check = Some(Box::new(crate::opt::exit::ShapeCheck {
                        receiver: object,
                        shapes: shapes.clone().into(),
                        ways: *ways,
                        hop: true,
                    }));
                }
                // As the templates' hop (`emit_baked`): the [[Prototype]] read from the object,
                // non-null, an ordinary kind, IC_PLAIN, then its shape.
                let layout = self.layout;
                let cell = self.gp(input(0), X16);
                let proto = self.gp_dst(n);
                self.a.add_imm(X17, cell, layout.obj_from_cell as u32);
                self.a.ldr_imm(proto, X17, layout.obj_proto as u32);
                self.a.cbz(proto, true, site);
                self.a.add_imm(X16, proto, layout.obj_from_cell as u32);
                self.a.ldrb_imm(X17, X16, layout.obj_kind as u32);
                self.a
                    .cmp_imm_w(X17, crate::value::KIND_ORDINARY_LAST as u32);
                self.a.b_cond(HI, site);
                self.a.ldrb_imm(X17, X16, layout.obj_flags as u32);
                self.a.tbz(X17, flag_bit(crate::value::FLAG_IC_PLAIN), site);
                self.a
                    .ldr_w_imm(X17, X16, (layout.obj_props + layout.props_shape) as u32);
                self.shape_compare(X17, shapes, site);
                self.gp_done(n, proto);
            }
            NodeOp::CheckStringProto { shapes, ways } => {
                let site = self.check_exit(n)?;
                let pc = graph.state(node.state.expect("checked")).pc;
                let receiver = self.source(input(0), pc)?;
                if let Some((_, _, record)) = self.exits.last_mut() {
                    record.shape_check = Some(Box::new(crate::opt::exit::ShapeCheck {
                        receiver,
                        shapes: shapes.clone().into(),
                        ways: *ways,
                        hop: false,
                    }));
                }
                let offset = self.ilayout.string_proto;
                if !self.ilayout.valid || !offset.is_multiple_of(8) || offset / 8 >= 4096 {
                    return Err(Refusal::at(pc as usize, "a String.prototype out of reach"));
                }
                // As the templates' string receiver (`emit_packed_probe_receiver`): the word's
                // tag, then the realm's String.prototype through ctx.interp, its IC_PLAIN and its
                // shape (its kind is the intrinsic's, StrWrap, which nothing changes).
                let layout = self.layout;
                let word = self.gp(input(0), X16);
                self.a.lsr_imm(X17, word, 48);
                self.a.movz(X16, (crate::value::PACK_STR >> 48) as u32, 0);
                self.a.cmp_reg_w(X17, X16);
                self.a.b_cond(NE, site);
                let proto = self.gp_dst(n);
                self.a
                    .ldr_imm(X17, 19, std::mem::offset_of!(JitCtx, interp) as u32);
                self.a.ldr_imm(proto, X17, offset as u32);
                self.a.add_imm(X16, proto, layout.obj_from_cell as u32);
                self.a.ldrb_imm(X17, X16, layout.obj_flags as u32);
                self.a.tbz(X17, flag_bit(crate::value::FLAG_IC_PLAIN), site);
                self.a
                    .ldr_w_imm(X17, X16, (layout.obj_props + layout.props_shape) as u32);
                self.shape_compare(X17, shapes, site);
                self.gp_done(n, proto);
            }
            NodeOp::ShapeIs(shapes) => {
                if !unused {
                    let layout = self.layout;
                    let cell = self.gp(input(0), X16);
                    self.a.add_imm(X17, cell, layout.obj_from_cell as u32);
                    self.a
                        .ldr_w_imm(X17, X17, (layout.obj_props + layout.props_shape) as u32);
                    let dst = self.gp_dst(n);
                    let yes = self.a.new_label();
                    let done = self.a.new_label();
                    for &shape in shapes.iter() {
                        if shape < 4096 {
                            self.a.cmp_imm_w(X17, shape);
                        } else {
                            self.a.mov_imm64(X16, u64::from(shape));
                            self.a.cmp_reg_w(X17, X16);
                        }
                        self.a.b_cond(EQ, yes);
                    }
                    self.a.movz(dst, 0, 0);
                    self.a.b(done);
                    self.a.bind(yes);
                    self.a.movz(dst, 1, 0);
                    self.a.bind(done);
                    self.gp_done(n, dst);
                }
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
                    self.truncate(d, w);
                    self.gp_done(n, w);
                }
            }
            NodeOp::TruncateNumberOrOddball => {
                let site = self.check_exit(n)?;
                let word = self.gp(input(0), X16);
                let w = self.gp_dst(n);
                let nan = self.a.new_label();
                let resume = self.a.new_label();
                // A Number's word is its f64 (CheckNumber's test); every NaN goes out of line,
                // where the canonical one, undefined, null and the Booleans convert.
                self.a.fmov_d_x(D31, word);
                self.a.fcmp(D31, D31);
                self.a.b_cond(VS, nan);
                self.truncate(D31, w);
                self.a.bind(resume);
                self.cold.push(Cold::NumberOrOddball {
                    label: nan,
                    word,
                    dst: w,
                    exit: site,
                    resume,
                });
                self.gp_done(n, w);
            }
            NodeOp::F64ToTagged => {
                if !unused {
                    let d = self.fp(input(0), D31);
                    let x = self.gp_dst(n);
                    self.box_f64(d, x);
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
            NodeOp::F64ToBoolean => {
                if !unused {
                    // ±0 compares equal to zero and NaN unordered: true where the comparison
                    // is not equal and ordered (Maglev's Float64ToBoolean).
                    let d = self.fp(input(0), D31);
                    let w = self.gp_dst(n);
                    self.a.fcmp_zero(d);
                    self.a.cset_w(w, NE);
                    self.a.csel_x(w, 31, w, VS);
                    self.gp_done(n, w);
                }
            }
            NodeOp::I32Add | NodeOp::I32Sub => {
                let site = self.check_exit(n)?;
                let sub = node.op == NodeOp::I32Sub;
                let w = self.gp_dst(n);
                // A constant that encodes takes the immediate, an addition's on either side.
                match self.imm_operand(node, !sub, |c| Self::add_sub_imm(c, sub)) {
                    Some((x, (true, imm))) => {
                        let a = self.gp(x, X16);
                        self.a.subs_imm_w(w, a, imm);
                    }
                    Some((x, (false, imm))) => {
                        let a = self.gp(x, X16);
                        self.a.adds_imm_w(w, a, imm);
                    }
                    None => {
                        let a = self.gp(input(0), X16);
                        let b = self.gp(input(1), X17);
                        if sub {
                            self.a.subs_reg_w(w, a, b);
                        } else {
                            self.a.adds_reg_w(w, a, b);
                        }
                    }
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
                    let w = self.gp_dst(n);
                    let op = match node.op {
                        NodeOp::I32And => 0,
                        NodeOp::I32Or => 1,
                        _ => 2,
                    };
                    // A constant that encodes takes the logical immediate, on either side.
                    match self.imm_operand(node, true, |c| asm::logical_imm_w(c as u32)) {
                        Some((x, field)) => {
                            let a = self.gp(x, X16);
                            self.a.logic_imm_w(op, w, a, field);
                        }
                        None => {
                            let a = self.gp(input(0), X16);
                            let b = self.gp(input(1), X17);
                            self.a.logic_w(op, w, a, b);
                        }
                    }
                    self.gp_done(n, w);
                }
            }
            NodeOp::I32Shl | NodeOp::I32Sar => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    let w = self.gp_dst(n);
                    // The count is taken mod 32, as JavaScript takes it.
                    let op = if node.op == NodeOp::I32Shl { 0 } else { 2 };
                    self.shift_i32(op, w, a, input(1));
                    self.gp_done(n, w);
                }
            }
            NodeOp::I32UShr => {
                let site = self.check_exit(n)?;
                let a = self.gp(input(0), X16);
                let w = self.gp_dst(n);
                self.shift_i32(1, w, a, input(1));
                // Past int32, unless a constant count of 1 to 31 clears bit 31 (the builder's
                // only kind, whose exit then never runs; the forced-exit gate keeps it).
                if self.const_i32(input(1)).is_none_or(|c| c & 31 == 0) {
                    self.a.tbnz(w, 31, site);
                }
                self.gp_done(n, w);
            }
            NodeOp::I32UShrToF64 => {
                if !unused {
                    let a = self.gp(input(0), X16);
                    self.shift_i32(1, X16, a, input(1));
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
                    let w = self.gp_dst(n);
                    let code = match cond {
                        Cond::Lt => LT,
                        Cond::Le => LE,
                        Cond::Gt => GT,
                        Cond::Ge => GE,
                        Cond::Eq => EQ,
                        Cond::Ne => NE,
                    };
                    let code = self.cmp_i32(input(0), input(1), code);
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
            NodeOp::NullOrLikeUndefined => {
                if !unused {
                    self.lower_null_or_like_undefined(n);
                }
            }
            NodeOp::TypeofIs(test) => {
                if !unused {
                    self.lower_typeof_is(n, *test);
                }
            }
            NodeOp::CheckString => {
                let site = self.check_exit(n)?;
                let word = self.gp(input(0), X16);
                self.a.lsr_imm(X17, word, 48);
                self.a.movz(X30, (crate::value::PACK_STR >> 48) as u32, 0);
                self.a.cmp_reg_w(X17, X30);
                self.a.b_cond(NE, site);
            }
            NodeOp::StringEqual => {
                if !unused {
                    self.lower_string_equal(n);
                }
            }
            NodeOp::CheckObjectOrNullish => {
                let site = self.check_exit(n)?;
                let word = self.gp(input(0), X16);
                let ok = self.a.new_label();
                // An object: the tag 0xfff9 is -7 as a signed half.
                self.a.asr_imm(X30, word, 48);
                self.a.cmn_imm_x(X30, 7);
                self.a.b_cond(EQ, ok);
                self.nullish_or(word, site);
                self.a.bind(ok);
            }
            NodeOp::ObjectOrNullishEqual => {
                if !unused {
                    self.lower_object_or_nullish_equal(n);
                }
            }
            NodeOp::WordAsF64 => {
                if !unused {
                    let word = self.gp(input(0), X16);
                    let d = self.fp_dst(n);
                    self.a.fmov_d_x(d, word);
                    self.fp_done(n, d);
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
            NodeOp::CheckCreatable { ways } | NodeOp::CheckNoProto { ways } => {
                let site = self.check_exit(n)?;
                let pc = graph.state(node.state.expect("checked")).pc;
                let receiver = self.source(input(0), pc)?;
                let hop = matches!(node.op, NodeOp::CheckNoProto { .. });
                if let Some((_, _, record)) = self.exits.last_mut() {
                    record.shape_check = Some(Box::new(crate::opt::exit::ShapeCheck {
                        receiver,
                        shapes: Box::new([]),
                        ways: *ways,
                        hop,
                    }));
                }
                let layout = self.layout;
                let cell = self.gp(input(0), X16);
                self.a.add_imm(X17, cell, layout.obj_from_cell as u32);
                if hop {
                    // The chain's end: a null [[Prototype]], read from the object as the
                    // templates' hop reads it.
                    self.a.ldr_imm(X16, X17, layout.obj_proto as u32);
                    self.a.cbnz(X16, true, site);
                } else {
                    // As the template's creation probe (`emit_prop_create_probe`): extensible
                    // and no marked prototype.
                    self.a.ldrb_imm(X16, X17, layout.obj_flags as u32);
                    self.a
                        .tbz(X16, flag_bit(crate::value::FLAG_EXTENSIBLE), site);
                    self.a.tbnz(X16, flag_bit(crate::value::FLAG_PROTO), site);
                }
            }
            NodeOp::StoreNew {
                location,
                shape,
                key,
                ways,
            } => self.lower_store_new(n, *location, *shape, *key, *ways)?,
            NodeOp::StringLength => {
                let site = self.check_exit(n)?;
                let pc = graph.state(node.state.expect("checked")).pc;
                let receiver = self.source(input(0), pc)?;
                if let Some((_, _, record)) = self.exits.last_mut() {
                    record.shape_check = Some(Box::new(crate::opt::exit::ShapeCheck {
                        receiver,
                        shapes: Box::new([]),
                        ways: 0,
                        hop: false,
                    }));
                }
                // The String tag, as the templates test it (`emit_ascii_string_length`), then
                // the header's length (a u32 of every string, flat or a view).
                let word = self.gp(input(0), X16);
                self.a.lsr_imm(X17, word, 48);
                self.a.movz(X9, (crate::value::PACK_STR >> 48) as u32, 0);
                self.a.cmp_reg_w(X17, X9);
                self.a.b_cond(NE, site);
                if !unused {
                    self.a.ubfx(X17, word, 0, 48);
                    let dst = self.gp_dst(n);
                    self.a.ldr_w_imm(dst, X17, crate::lstr::LEN_OFF as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::StoreSlot(slot) => {
                let value = self.boxed(input(0), X16);
                self.a.str_imm(value, 22, u32::from(*slot) * 8);
            }
            NodeOp::CheckElements(kind) => {
                let site = self.check_exit(n)?;
                let cell = self.gp(input(0), X16);
                let layout = self.layout;
                if let ElemKind::Typed(element) = *kind {
                    // A typed array's tag, as the typed templates test it (`emit_access`); its
                    // IC_PLAIN is clear, as every typed array's is.
                    self.a
                        .ldrb_imm(X17, cell, (layout.obj_from_cell + layout.obj_kind) as u32);
                    self.a
                        .cmp_imm_w(X17, crate::value::KIND_TYPED_ARRAY_TAG as u32);
                    self.a.b_cond(NE, site);
                    if let Some(element) = element {
                        // Then its view's kind byte, which a declined view's (u8::MAX, its buffer
                        // record null) never matches.
                        use crate::native_typed_array::NativeTypedArray as View;
                        let view = layout.obj_from_cell + self.native()?.object_view;
                        self.a.ldr_imm(X17, cell, view as u32);
                        self.a
                            .ldrb_imm(X17, X17, std::mem::offset_of!(View, kind) as u32);
                        self.a.cmp_imm_w(X17, element as u32);
                        self.a.b_cond(NE, site);
                    }
                    return Ok(());
                }
                // The kind tag, then IC_PLAIN, as the element templates test a receiver
                // (`emit_get_elem_inline`): an ordinary kind's or the Array's tag, whose tag
                // follows them, and no side-table internal method.
                self.a
                    .ldrb_imm(X17, cell, (layout.obj_from_cell + layout.obj_kind) as u32);
                self.a.cmp_imm_w(X17, crate::value::KIND_ARRAY_TAG as u32);
                let other = if *kind == ElemKind::Array { NE } else { HI };
                self.a.b_cond(other, site);
                self.a
                    .ldrb_imm(X17, cell, (layout.obj_from_cell + layout.obj_flags) as u32);
                self.a.tbz(X17, flag_bit(crate::value::FLAG_IC_PLAIN), site);
            }
            NodeOp::LoadElements => {
                if !unused {
                    // The words pointer, never null (`emit_words_base`).
                    let cell = self.gp(input(0), X16);
                    let dst = self.gp_dst(n);
                    let layout = self.layout;
                    let words = layout.obj_from_cell + layout.obj_props + layout.props_words;
                    self.a.ldr_imm(dst, cell, words as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::LoadElementsUsed => {
                if !unused {
                    let words = self.gp(input(0), X16);
                    let dst = self.gp_dst(n);
                    self.a.ldur_w(dst, words, -(self.layout.words_used as i32));
                    self.gp_done(n, dst);
                }
            }
            NodeOp::CheckIndex => {
                let site = self.check_exit(n)?;
                let d = self.fp(input(0), D31);
                let w = self.gp_dst(n);
                // An exact int32 converts back to itself, -0 to +0, which compares equal to it
                // (the index -0 names is 0), and NaN compares unordered.
                self.a.fcvtzs_w_d(w, d);
                self.a.scvtf_d_w(D30, w);
                self.a.fcmp(D30, d);
                self.a.b_cond(NE, site);
                self.gp_done(n, w);
            }
            NodeOp::CheckBounds => {
                let site = self.check_exit(n)?;
                // Unsigned: a negative index is past every length. A constant index (`a[3]`)
                // takes the immediate, compared from the length's side.
                let code = self.cmp_i32(input(0), input(1), HS);
                self.a.b_cond(code, site);
            }
            NodeOp::LoadElement(holes) => self.lower_load_element(n, *holes)?,
            NodeOp::LoadElementOrUndefined => self.lower_load_element_or_undefined(n)?,
            NodeOp::StoreElement(mode) => self.lower_store_element(n, *mode)?,
            NodeOp::LoadArrayLength => {
                if !unused {
                    // Slot 0, an Array's first entry (`first_slot_addr`).
                    let cell = self.gp(input(0), X16);
                    self.a.add_imm(X17, cell, self.layout.obj_from_cell as u32);
                    first_slot_addr(&mut self.a, self.layout, X17, X17);
                    let dst = self.gp_dst(n);
                    self.a.ldr_imm(dst, X17, 0);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::TypedView => {
                if !unused {
                    let native = self.native()?;
                    let cell = self.gp(input(0), X16);
                    let dst = self.gp_dst(n);
                    let view = self.layout.obj_from_cell + native.object_view;
                    self.a.ldr_imm(dst, cell, view as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::TypedLength => {
                if !unused {
                    use crate::native_typed_array::NativeTypedArray as View;
                    // At most 2^31: its low word compares unsigned with an index.
                    let view = self.gp(input(0), X16);
                    let dst = self.gp_dst(n);
                    let limit = std::mem::offset_of!(View, limit);
                    self.a.ldr_imm(dst, view, limit as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::TypedData { write } => self.lower_typed_data(n, *write)?,
            NodeOp::LoadTyped(kind) => {
                if !unused {
                    self.lower_load_typed(n, *kind);
                }
            }
            NodeOp::LoadTypedOrUndefined(kind) => {
                if !unused {
                    self.lower_load_typed_or_undefined(n, *kind);
                }
            }
            NodeOp::StoreTyped(kind) => self.lower_store_typed(n, *kind),
            NodeOp::ClampToUint8 => {
                if !unused {
                    // ToUint8Clamp: FCVTNS rounds half to even, saturates, and takes NaN to 0;
                    // then the clamp to [0, 255], as the template stores Uint8Clamped.
                    let d = self.fp(input(0), D31);
                    let w = self.gp_dst(n);
                    let low = self.a.new_label();
                    let done = self.a.new_label();
                    self.a.fcvtns_w_d(w, d);
                    self.a.cmp_imm_w(w, 0);
                    self.a.b_cond(LT, low);
                    self.a.cmp_imm_w(w, 255);
                    self.a.b_cond(LS, done);
                    self.a.movz(w, 255, 0);
                    self.a.b(done);
                    self.a.bind(low);
                    self.a.movz(w, 0, 0);
                    self.a.bind(done);
                    self.gp_done(n, w);
                }
            }
            NodeOp::TypedElement { write } => self.lower_typed_element(n, *write)?,
            NodeOp::IsTypedArray => {
                if !unused {
                    let cell = self.gp(input(0), X16);
                    let w = self.gp_dst(n);
                    let kind = self.layout.obj_from_cell + self.layout.obj_kind;
                    self.a.ldrb_imm(X17, cell, kind as u32);
                    self.a
                        .cmp_imm_w(X17, crate::value::KIND_TYPED_ARRAY_TAG as u32);
                    self.a.cset_w(w, EQ);
                    self.gp_done(n, w);
                }
            }
            NodeOp::Poll => {
                let slow = self.a.new_label();
                let resume = self.a.new_label();
                self.a
                    .ldr_imm(X9, 19, std::mem::offset_of!(JitCtx, poll_word) as u32);
                self.a.ldr_imm(X9, X9, 0);
                self.a.cbnz(X9, true, slow);
                self.a.bind(resume);
                // A handler in its frame or a frame its exit rebuilds (stage 3) takes the throw.
                let throw = if !graph.chain_has_handlers(node.state.expect("checked")) {
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
            NodeOp::Effect { pc, pops, .. } | NodeOp::Generic { pc, pops, .. } => {
                self.lower_effect(n, *pc as usize, *pops as usize)?;
            }
            NodeOp::Jump(target) => {
                self.emit_moves(block, *target)?;
                let label = self.block_label(block, *target);
                if next != Some(*target) || label != self.blocks[target.index()] {
                    self.a.b(label);
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
        let it = *self.chunk.jit_inline_target(target);
        // The expected function's stored word, read out of an Option<Gc> as the first stage's
        // InlineGuard reads it (the frozen view upgraded its pin on the main thread); a dead
        // callee always exits.
        let Some(stored) = it.callee else {
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
        let it = *self.chunk.jit_inline_target(target);
        let mismatch = self.a.new_label();
        let done = self.a.new_label();
        // A dead callee never matches.
        if let Some(stored) = it.callee {
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
        if !node.frame.is_compiled() {
            return self.lower_inlined_generic(n, pc, pops);
        }
        let state = graph.state(node.state.expect("checked"));
        let handlers = !state.handlers.is_empty();
        if self.options.forced_exits {
            let site = self.exit(n, ExitKind::Check)?;
            self.forced(site);
        }
        let depth = self.cfg.stack_depth_at(pc).expect("checked");
        let generic = matches!(node.op, NodeOp::Generic { .. });
        self.a
            .ldr_imm(X9, 19, std::mem::offset_of!(JitCtx, stack_base) as u32);
        if handlers && !generic {
            // A throw exits with this state, which the template's clobbers would lose: the frame
            // holds all of it before the template runs (a canonical-frame effect's; a generic
            // node's throw exit reads its recipe's homes, which its call preserves).
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
        for (k, &v) in node.inputs[..pops].iter().enumerate() {
            let word = self.boxed(v, X16);
            self.a.str_imm(word, X9, (8 * (depth - pops + k)) as u32);
        }
        // A generic node's named slots whose words are stale (P3 stage 3 step 3a): the template
        // reads them in the frame.
        if matches!(node.op, NodeOp::Generic { .. }) {
            let slots = stale_slots(self.chunk, graph.n_slots, graph, node);
            for (&slot, &v) in slots.iter().zip(&node.inputs[pops..]) {
                let word = self.boxed(v, X16);
                self.a.str_imm(word, 22, 8 * u32::from(slot));
            }
        }
        // The poison switch: before a generic node, its stale words; before a canonical-frame
        // effect that has no handler (with one, the frame holds the whole state, above), the
        // operand words below its operands, which the code does not write either (the helpers
        // that copy the operand stack to run the VM among them: `bytecode::jit_runs_on_vm_copy`).
        let poisoned = (self.options.poison && (generic || !handlers))
            .then(|| self.poison(node, state, depth - pops, generic));
        self.a.add_imm(20, X9, (8 * depth) as u32);
        let unwind = if handlers {
            let site = self.exit_throw(n)?;
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
        if let Some(slots) = poisoned.filter(|_| generic) {
            self.check_poison(pc, &slots, depth - pops);
        }
        self.a.set_hot(false);
        self.last_effect = Some(n);
        Ok(())
    }

    /// An inlined frame's generic node (P3 stage 3 step 3c; step3c-impl's design note): the
    /// operation's template, emitted from its frame's chunk, run with its explicit inputs where
    /// the template ABI's helpers read them, as JSC's DFG stores its node's code origin into the
    /// frame before each call out (dfg/DFGJITCompiler.h:133-143, dfg/DFGSpeculativeJIT.h:
    /// 1078-1097): its operands in the compiled frame's operand words at the operation's own
    /// depth in its frame's chunk (x20 there); its chunk in `JitCtx::chunk`, its template passing
    /// its pc; its frame's inline site in `Interp::call_site`, as a splice's checked operation
    /// publishes its implicit position; for a kind that reads them (`liveness::context_use`), the
    /// frame's receiver in `this_word` and its closure's environment in `env_raw` and
    /// `env_parent_raw`; for a kind that names a slot, a slot window above the callee's operand
    /// capacity that x22, `JitCtx::slots` and `n_slots` name, the named slots' values in it. The
    /// compiled frame's words are saved in the save area before and restored after, on its
    /// return and on its throw, which leaves by the throw epilogue or, with a handler in a frame
    /// of its chain, by its exit. Its property helpers' direct accessor calls take the published
    /// site (`Asm::inlined_frame_helpers`), its TDZ tests all kept (its pcs are another chunk's).
    fn lower_inlined_generic(&mut self, n: NodeIx, pc: usize, pops: usize) -> Result<(), Refusal> {
        let graph = self.graph;
        let node = graph.node(n);
        let state = node.state.expect("checked");
        let frame = graph.inlined(node.frame).expect("an inlined frame's node");
        let &(view, cfg) = self
            .inlined
            .get(frame.chunk as usize)
            .ok_or(Refusal::at(pc, "an inlined frame's chunk the code lacks"))?;
        let save = self.save_area.ok_or(Refusal::at(
            pc,
            "an inlined frame's generic node without a save area",
        ))?;
        if self.options.forced_exits {
            let site = self.exit(n, ExitKind::Check)?;
            self.forced(site);
        }
        let op = &view.jit_ops()[pc];
        #[cfg(test)]
        TEST_INLINED_GENERIC.with(|ops| ops.borrow_mut().push(*op));
        let depth = cfg.stack_depth_at(pc).expect("checked");
        let (_, n_slots) = view.jit_frame();
        let capacity = cfg.jit_stack_capacity();
        let uses = crate::opt::liveness::context_use(op);
        let named = crate::opt::liveness::frame_slots(op, n_slots);
        let stale = stale_slots(view, n_slots, graph, node);
        let poison = self.options.poison;
        // An explicit call (P3 stage 3 step 3c-2): its position is its inline site in this frame,
        // which the way's view gives its template (the direct sequence's and the stubs' words)
        // and its helper paths publish for the call helpers, which take it from
        // `Interp::call_site` through the inlined call helpers' table.
        let call_site = match *op {
            Op::Call(_, cache) | Op::CallWithThis(_, cache) => Some(view.call_position(cache)),
            _ => None,
        };
        let switched = Switched {
            this: uses.this || poison,
            env: uses.env,
            slots: !named.is_empty() || poison,
            call: call_site.is_some(),
        };
        let ctx = |field: usize| field as u32;
        #[cfg(test)]
        self.check_node_context(pc);
        let stack_base = ctx(std::mem::offset_of!(JitCtx, stack_base));
        // The operands, at the operation's depth from the compiled frame's operand base.
        self.a.ldr_imm(X9, 19, stack_base);
        for (k, &v) in node.inputs[..pops].iter().enumerate() {
            let word = self.boxed(v, X16);
            self.a.str_imm(word, X9, (8 * (depth - pops + k)) as u32);
        }
        // The slot window above the callee's capacity: its named slots' values, and under the
        // poison switch the poison in every other word of it.
        if switched.slots {
            if poison {
                self.a.mov_imm64(X16, POISON);
                for slot in (0..n_slots as u16).filter(|slot| !stale.contains(slot)) {
                    self.a
                        .str_imm(X16, X9, 8 * (capacity as u32 + u32::from(slot)));
                }
            }
            for (&slot, &v) in stale.iter().zip(&node.inputs[pops..]) {
                let word = self.boxed(v, X16);
                self.a
                    .str_imm(word, X9, 8 * (capacity as u32 + u32::from(slot)));
            }
        }
        if poison {
            // The operand words below its operands (`poison`'s for the compiled frame's nodes).
            self.a.mov_imm64(X16, POISON);
            for k in 0..depth - pops {
                self.a.str_imm(X16, X9, 8 * k as u32);
            }
        }
        // The compiled frame's context words saved, the node's written (a call node's chunk by
        // its template's helper paths alone).
        if !switched.call {
            let chunk_off = ctx(std::mem::offset_of!(JitCtx, chunk));
            self.a.ldr_imm(X16, 19, chunk_off);
            self.a.str_imm(X16, 31, save + SAVE_CHUNK);
            self.a.mov_imm64(X16, view.chunk_addr as u64);
            self.a.str_imm(X16, 19, chunk_off);
        }
        let this_off = ctx(std::mem::offset_of!(JitCtx, this_word));
        if switched.this {
            self.a.ldr_imm(X16, 19, this_off);
            self.a.str_imm(X16, 31, save + SAVE_THIS);
            let word = if uses.this {
                self.gp(frame.receiver, X16)
            } else {
                // A kind that reads no `this` finds the poison there (a Number: the collector
                // traces none).
                self.a.mov_imm64(X16, POISON);
                X16
            };
            self.a.str_imm(word, 19, this_off);
        }
        if switched.env {
            if frame.env == 0 {
                return Err(Refusal::at(pc, "an inlined frame's environment unknown"));
            }
            let env_off = ctx(std::mem::offset_of!(JitCtx, env_raw));
            let parent_off = ctx(std::mem::offset_of!(JitCtx, env_parent_raw));
            self.a.ldr_imm(X16, 19, env_off);
            self.a.str_imm(X16, 31, save + SAVE_ENV);
            self.a.ldr_imm(X16, 19, parent_off);
            self.a.str_imm(X16, 31, save + SAVE_ENV_PARENT);
            // The closure's environment and its parent, as the direct sequence installs both
            // (`emit_direct_sequence`: the parent's word read from the record).
            self.a.mov_imm64(X16, frame.env as u64);
            self.a.str_imm(X16, 19, env_off);
            self.a.ldr_imm(X17, X16, self.layout.scope_parent as u32);
            self.a.str_imm(X17, 19, parent_off);
        }
        if switched.slots {
            self.a.str_imm(22, 31, save + SAVE_SLOTS);
            self.a.add_imm(22, X9, 8 * capacity as u32);
            self.a
                .str_imm(22, 19, ctx(std::mem::offset_of!(JitCtx, slots)));
            self.a.movz(X16, n_slots as u32, 0);
            self.a
                .str_imm(X16, 19, ctx(std::mem::offset_of!(JitCtx, n_slots)));
        }
        // The frame's inline site, published as a splice's checked operation publishes its own; a
        // call's, its own site, as a spliced body's call publishes it (explicit), by its call
        // template's helper paths alone (`jit::InlinedCall`): its direct path writes the callee's
        // frame from the call's own word and reads no publication.
        let frame_site = if frame.position == crate::stack_trace::NO_POSITION {
            frame.position
        } else {
            crate::stack_trace::implicit_inline_position(frame.position)
        };
        let inlined_call = match call_site {
            Some(site) => {
                if !crate::stack_trace::is_inline_position(site) {
                    return Err(Refusal::at(
                        pc,
                        "an inlined frame's call without its inline site",
                    ));
                }
                Some(crate::jit::InlinedCall {
                    callee_chunk: view.chunk_addr,
                    own_chunk: self.chunk.chunk_addr,
                    site,
                    frame_site,
                    call_site_off: self.ilayout.call_site as u32,
                })
            }
            None => {
                self.a
                    .ldr_imm(X17, 19, ctx(std::mem::offset_of!(JitCtx, interp)));
                self.a.mov_imm64(X16, u64::from(frame_site));
                self.a.str_w_imm(X16, X17, self.ilayout.call_site as u32);
                None
            }
        };
        self.a.add_imm(20, X9, (8 * depth) as u32);
        // A throw restores the context first.
        let target = if graph.chain_has_handlers(state) {
            self.exit_throw(n)?
        } else {
            self.l_ret_throw
        };
        let unwind = self.a.new_label();
        self.cold.push(Cold::Restore {
            label: unwind,
            switched,
            target,
        });
        let hot = self.in_loop[self.compiled_pc(node.frame) as usize] || self.loop_mask(view)[pc];
        self.a.set_hot(hot);
        let reads = self.a.replace_initialized_reads(Default::default());
        // A property node's helpers take the published site (piece A's variant); a call node's
        // template switches for its helpers itself.
        let helpers = self.a.set_inlined_frame_helpers(inlined_call.is_none());
        let outer_call = self.a.set_inlined_call(inlined_call);
        let fixed = self.a.fixed_activation();
        self.a.set_fixed_activation(view.jit_fixed_activation());
        let emitted = operations::emit(
            &mut self.a,
            view,
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
            // The VM's operation through H_EXEC reads the callee's chunk and, for a call, the
            // published site: switched around it.
            crate::jit::emit_inlined_call_switch(&mut self.a);
            emit_exec(&mut self.a, pc as u32, unwind);
            crate::jit::emit_inlined_call_restore(&mut self.a);
        }
        self.a.set_fixed_activation(fixed);
        self.a.set_inlined_frame_helpers(helpers);
        self.a.set_inlined_call(outer_call);
        self.a.replace_initialized_reads(reads);
        if poison {
            // The words it poisoned, still the poison: the window's other slots (x22 names the
            // window still) and the operand words below its operands.
            let slots: Vec<u16> = (0..n_slots as u16)
                .filter(|slot| !named.iter().any(|&(name, _)| name == *slot))
                .collect();
            self.check_poison(pc, &slots, depth - pops);
        }
        self.restore_context(switched, false);
        self.a.set_hot(false);
        self.last_effect = Some(n);
        Ok(())
    }

    /// The compiled frame's context words an inlined frame's generic node switched, back from
    /// the save area (`lower_inlined_generic`), and `Interp::call_site` NO_POSITION again, the
    /// value it has between operations (contracts 4.3). A call node's chunk, x21 and publication
    /// are its call template's to restore on the paths that switched them; its unwind (`unwind`,
    /// the node's `Cold::Restore`), which a path that switched nothing takes too, writes their
    /// constants (`jit::emit_inlined_call_restore`'s). x16 and x17 scratch; x20 and x0-x1 kept.
    fn restore_context(&mut self, switched: Switched, unwind: bool) {
        let save = self.save_area.expect("a save area");
        let ctx = |field: usize| field as u32;
        if switched.call {
            if unwind {
                self.a.mov_imm64(X16, self.chunk.chunk_addr as u64);
            }
        } else {
            self.a.ldr_imm(X16, 31, save + SAVE_CHUNK);
        }
        if !switched.call || unwind {
            self.a
                .str_imm(X16, 19, ctx(std::mem::offset_of!(JitCtx, chunk)));
        }
        if switched.this {
            self.a.ldr_imm(X16, 31, save + SAVE_THIS);
            self.a
                .str_imm(X16, 19, ctx(std::mem::offset_of!(JitCtx, this_word)));
        }
        if switched.env {
            self.a.ldr_imm(X16, 31, save + SAVE_ENV);
            self.a
                .str_imm(X16, 19, ctx(std::mem::offset_of!(JitCtx, env_raw)));
            self.a.ldr_imm(X16, 31, save + SAVE_ENV_PARENT);
            self.a
                .str_imm(X16, 19, ctx(std::mem::offset_of!(JitCtx, env_parent_raw)));
        }
        if switched.slots {
            self.a.ldr_imm(22, 31, save + SAVE_SLOTS);
            self.a
                .str_imm(22, 19, ctx(std::mem::offset_of!(JitCtx, slots)));
            self.a.movz(X16, self.graph.n_slots as u32, 0);
            self.a
                .str_imm(X16, 19, ctx(std::mem::offset_of!(JitCtx, n_slots)));
        }
        if switched.call && !unwind {
            return;
        }
        if switched.call {
            // The activation's own helper table again (`JitCtx::helpers`, as its entry loads it).
            self.a
                .ldr_imm(21, 19, ctx(std::mem::offset_of!(JitCtx, helpers)));
        }
        self.a
            .ldr_imm(X17, 19, ctx(std::mem::offset_of!(JitCtx, interp)));
        self.a
            .mov_imm64(X16, u64::from(crate::stack_trace::NO_POSITION));
        self.a.str_w_imm(X16, X17, self.ilayout.call_site as u32);
    }

    /// Test builds: at an inlined frame's generic node, `JitCtx::chunk` is the compiled chunk's
    /// own (a call node's restore and unwind write that constant rather than a saved word: the
    /// code is installed for that chunk alone, and every entry runs with it there) and x21 the
    /// activation's own table (`JitCtx::helpers`), or the code traps (`brk`). x16, x17 scratch.
    #[cfg(test)]
    fn check_node_context(&mut self, pc: usize) {
        let ok = self.a.new_label();
        self.a
            .ldr_imm(X16, 19, std::mem::offset_of!(JitCtx, chunk) as u32);
        self.a.mov_imm64(X17, self.chunk.chunk_addr as u64);
        self.a.cmp_reg_x(X16, X17);
        self.a.b_cond(EQ, ok);
        self.a.brk(0xb2c0 | (pc as u32 & 0x3f));
        self.a.bind(ok);
        let ok = self.a.new_label();
        self.a
            .ldr_imm(X16, 19, std::mem::offset_of!(JitCtx, helpers) as u32);
        self.a.cmp_reg_x(X16, 21);
        self.a.b_cond(EQ, ok);
        self.a.brk(0xb210 | (pc as u32 & 0x3f));
        self.a.bind(ok);
    }

    /// The pc in the compiled chunk of the call that `frame` (an inlined frame, or the compiled
    /// frame's own 0) descends from.
    fn compiled_pc(&self, mut frame: FrameIx) -> u32 {
        let mut pc = 0;
        while let Some(inlined) = self.graph.inlined(frame) {
            pc = inlined.call_pc;
            frame = self.graph.state(inlined.parent).frame;
        }
        pc
    }

    /// The loop bodies of an inlined frame's chunk (`loop_body_mask`).
    fn loop_mask(&self, view: &FrozenChunk) -> Vec<bool> {
        loop_body_mask(view.jit_ops())
    }

    /// The poison switch's stores before effect `node` (`Options::poison`): before a generic node
    /// (`slots`), every slot its operation does not name whose word `state` does not say current
    /// (a value's, stale; a dead slot's); and the `below` operand words under its operands, x9
    /// holding `stack_base`. Every exit writes what it needs of them from its recipe (a throw's
    /// too), and the VM writes a dead slot before it reads it. Returns the slots it poisoned.
    fn poison(&mut self, node: &Node, state: &FrameState, below: usize, slots: bool) -> Vec<u16> {
        let named = crate::opt::liveness::frame_slots(
            &self.chunk.jit_ops()[node.pc as usize],
            self.graph.n_slots,
        );
        self.a.mov_imm64(X16, POISON);
        let mut poisoned = Vec::new();
        for (slot, source) in state.slots.iter().enumerate().filter(|_| slots) {
            let stale = matches!(source, SlotSource::Value(_) | SlotSource::Dead);
            if stale && !named.iter().any(|&(name, _)| usize::from(name) == slot) {
                self.a.str_imm(X16, 22, 8 * slot as u32);
                poisoned.push(slot as u16);
            }
        }
        for k in 0..below {
            self.a.str_imm(X16, X9, 8 * k as u32);
        }
        poisoned
    }

    /// The poison switch's check after generic node at `pc`'s template has returned
    /// (`Options::poison`): every word [`poison`](Self::poison) wrote before it, the `slots` it
    /// does not name and the `below` operand words under its operands, still holds the poison;
    /// one that does not was written by the template or a helper it called, a write
    /// `liveness::slot_use` does not list, which [`poison_written`] reports. The template's call
    /// left nothing live in a caller-saved register.
    fn check_poison(&mut self, pc: usize, slots: &[u16], below: usize) {
        if slots.is_empty() && below == 0 {
            return;
        }
        let done = self.a.new_label();
        let common = self.a.new_label();
        let mut faults = Vec::new();
        self.a
            .ldr_imm(X9, 19, std::mem::offset_of!(JitCtx, stack_base) as u32);
        self.a.mov_imm64(X17, POISON);
        let words = slots
            .iter()
            .map(|&slot| (22, u32::from(slot), u64::from(slot)))
            .chain((0..below).map(|k| (X9, k as u32, BELOW | k as u64)));
        for (base, index, code) in words {
            let fault = self.a.new_label();
            self.a.ldr_imm(X16, base, 8 * index);
            self.a.cmp_reg_x(X16, X17);
            self.a.b_cond(NE, fault);
            faults.push((fault, code));
        }
        self.a.b(done);
        for (fault, code) in faults {
            self.a.bind(fault);
            self.a.mov_imm64(2, code);
            self.a.b(common);
        }
        self.a.bind(common);
        self.a.mov(0, 19);
        self.a.mov_imm64(1, pc as u64);
        self.a.mov_imm64(X16, poison_written as *const () as u64);
        self.a.blr(X16);
        self.a.bind(done);
    }

    /// The label a branch from `from` to `to` takes: the block's, or a trampoline that makes the
    /// edge's moves into its phis first.
    fn edge_target(&mut self, from: BlockIx, to: BlockIx) -> usize {
        if self.graph.block(to).phis.is_empty() {
            return self.block_label(from, to);
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
            // A spilled input's loop cache in the edge's source block (step R1) is a copy of it.
            let home = self.cached(input).unwrap_or(self.home(input));
            let src = loc(home, input).ok_or(Refusal::whole("a phi's input without a home"))?;
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
        // Out-of-line code reads spilled values from their slots, an edge's moves aside.
        self.cur_block = None;
        // Until none is left: a cold path may itself push one (a boxed f64's NaN case).
        loop {
            let cold = std::mem::take(&mut self.cold);
            if cold.is_empty() {
                break;
            }
            for item in cold {
                match item {
                    Cold::CanonNan { label, dst, resume } => {
                        self.a.bind(label);
                        self.a.movz(dst, 0x7ff8, 3); // PACK_CANON_NAN
                        self.a.b(resume);
                    }
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
                    Cold::NumberOrOddball {
                        label,
                        word,
                        dst,
                        exit,
                        resume,
                    } => {
                        use crate::value::{PACK_BOOL, PACK_CANON_NAN, PACK_NULL, PACK_UNDEFINED};
                        let boolean = self.a.new_label();
                        let zero = self.a.new_label();
                        self.a.bind(label);
                        // A Boolean's word less its tag is its bit (CheckBoolean's test).
                        self.a.movz(X17, (PACK_BOOL >> 48) as u32, 3);
                        self.a.logic_x(2, X17, word, X17);
                        self.a.cmp_imm_x(X17, 1);
                        self.a.b_cond(LS, boolean);
                        // Each of the others is one word: its tag over a zero payload.
                        for oddball in [PACK_UNDEFINED, PACK_NULL, PACK_CANON_NAN] {
                            self.a.movz(X17, (oddball >> 48) as u32, 3);
                            self.a.cmp_reg_x(word, X17);
                            self.a.b_cond(EQ, zero);
                        }
                        self.a.b(exit);
                        self.a.bind(boolean);
                        self.a.mov_w(dst, X17);
                        self.a.b(resume);
                        self.a.bind(zero);
                        self.a.movz(dst, 0, 0);
                        self.a.b(resume);
                    }
                    Cold::Jump { label, target } => {
                        self.a.bind(label);
                        self.a.b(target);
                    }
                    Cold::Restore {
                        label,
                        switched,
                        target,
                    } => {
                        self.a.bind(label);
                        self.restore_context(switched, true);
                        self.a.b(target);
                    }
                    Cold::ElementUndefined {
                        label,
                        node,
                        dst,
                        resume,
                        exit,
                        negative,
                    } => {
                        self.a.bind(label);
                        if let Some((index, negative_exit)) = negative {
                            // A negative index names a String key ("-1"), never an element.
                            self.a.tbnz(index, 31, negative_exit);
                        }
                        self.emit_no_inherited_elements(node, false, &[], exit);
                        self.a.mov_imm64(dst, crate::value::PACK_UNDEFINED);
                        self.a.b(resume);
                    }
                    Cold::ElementKinds {
                        label,
                        out,
                        resume,
                        unwritable,
                    } => {
                        use crate::value::word_block::{KIND_SCANNED, KIND_UNWRITABLE};
                        self.a.bind(label);
                        let imm = |bits| asm::logical_imm_w(bits).expect("a run of kind bits");
                        self.a.logic_imm_w(0, X17, X9, imm(KIND_UNWRITABLE));
                        self.a.cbnz(X17, false, unwritable);
                        self.a.tbz(X9, KIND_SCANNED.trailing_zeros(), resume);
                        self.a.logic_imm_w(0, X9, X9, imm(!KIND_SCANNED));
                        self.a.stur_w(X9, out, -(self.layout.words_flags as i32));
                        self.a.b(resume);
                    }
                    Cold::Unwritable {
                        label,
                        node,
                        out,
                        retry,
                        exit,
                    } => {
                        use crate::value::word_block::KIND_FROZEN;
                        self.a.bind(label);
                        // A frozen block takes no write; a shared literal's is copied, then written.
                        self.a.ldur_w(X9, out, -(self.layout.words_flags as i32));
                        self.a.tbnz(X9, KIND_FROZEN.trailing_zeros(), exit);
                        let cell = self.gp(self.graph.node(node).inputs[0], X17);
                        let saved = self.save_live_caller_saved(node);
                        self.a.mov(0, cell);
                        self.a.mov_imm64(X16, unshare_elements as *const () as u64);
                        self.a.blr(X16);
                        self.a.mov(X9, 0);
                        self.restore_live_caller_saved(&saved);
                        self.a.cbz(X9, true, exit);
                        self.a.mov(out, X9);
                        self.a.b(retry);
                    }
                    Cold::Remember {
                        label,
                        node,
                        out,
                        resume,
                    } => {
                        self.a.bind(label);
                        // The arguments through scratch registers, which no home is: either may be
                        // in x0 or x1.
                        let index = self.gp(self.graph.node(node).inputs[2], X17);
                        self.a.mov(X9, out);
                        if index != X17 {
                            self.a.mov_w(X17, index);
                        }
                        // The words are the store's output, live after the call wherever a register
                        // the call clobbers holds them.
                        let keep = caller_saved(out);
                        let saved = self.save_live_caller_saved_and(node, &keep);
                        self.a.mov(0, X9);
                        self.a.mov_w(1, X17);
                        self.a.mov_imm64(X16, remember_element as *const () as u64);
                        self.a.blr(X16);
                        self.restore_live_caller_saved(&saved);
                        self.a.b(resume);
                    }
                    Cold::ElementCreate {
                        label,
                        node,
                        out,
                        done,
                        exits,
                    } => {
                        self.a.bind(label);
                        self.emit_element_create(node, out, done, exits);
                    }
                    Cold::CreateSlow {
                        label,
                        node,
                        done,
                        exit,
                    } => {
                        // store_new_slow(cell, key, value, shape) after every check of the
                        // creation, before any write: it runs no JavaScript and does not collect.
                        // The live caller-saved values are kept across it; the value is read
                        // into x9 before they are saved, since a spill slot's offset is the stack
                        // pointer's, which the save moves.
                        self.a.bind(label);
                        let layout = self.layout;
                        let (value, key, shape) = match self.graph.node(node).op {
                            NodeOp::StoreNew { key, shape, .. } => {
                                (self.graph.node(node).inputs[1], key, shape)
                            }
                            _ => unreachable!("a creation's slow path"),
                        };
                        let word = self.gp(value, X9);
                        if word != X9 {
                            self.a.mov(X9, word);
                        }
                        let saved = self.save_live_caller_saved(node);
                        self.a.sub_imm(0, X17, layout.obj_from_cell as u32);
                        self.a.mov_imm64(1, key);
                        self.a.mov(2, X9);
                        self.a.mov_imm64(3, u64::from(shape));
                        self.a.mov_imm64(X16, store_new_slow as *const () as u64);
                        self.a.blr(X16);
                        self.a.mov(X9, 0);
                        self.restore_live_caller_saved(&saved);
                        self.a.cbz(X9, true, exit);
                        self.a.b(done);
                    }
                    Cold::Edge { label, from, to } => {
                        self.a.bind(label);
                        // The moves run on the edge out of `from`, whose loop caches hold.
                        self.cur_block = Some(from);
                        self.emit_moves(from, to)?;
                        self.cur_block = None;
                        self.a.b(self.block_label(from, to));
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
                    Cold::TestCall {
                        label,
                        node,
                        word,
                        dst,
                        resume,
                        helper,
                        arg,
                    } => {
                        self.a.bind(label);
                        let saved = self.save_live_caller_saved(node);
                        // x1 = the word, x2 = the argument, x0 = the context, in an order that
                        // reads each register before it is written.
                        let other = match arg {
                            TestArg::Reg(1) => {
                                self.a.mov(X17, 1);
                                Some(X17)
                            }
                            TestArg::Reg(r) => Some(r),
                            TestArg::Imm(_) => None,
                        };
                        if word != 1 {
                            self.a.mov(1, word);
                        }
                        match (arg, other) {
                            (_, Some(r)) => self.a.mov(2, r),
                            (TestArg::Imm(imm), None) => self.a.mov_imm64(2, imm),
                            (TestArg::Reg(_), None) => unreachable!("a register's argument"),
                        }
                        self.a.mov(0, 19);
                        self.a.mov_imm64(X16, helper);
                        self.a.blr(X16);
                        self.a.mov_w(X9, 0);
                        self.restore_live_caller_saved(&saved);
                        self.a.mov_w(dst, X9);
                        self.a.b(resume);
                    }
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
        let mut live: Vec<(bool, u32)> = self
            .alloc
            .live_at(node)
            .into_iter()
            .filter_map(|v| match self.home(v) {
                Home::Gp(r) if file.gp_caller.contains(&r) => Some((false, u32::from(r))),
                Home::Fp(r) if file.fp_caller.contains(&r) => Some((true, u32::from(r))),
                _ => None,
            })
            .collect();
        live.extend(self.caller_saved_caches(node));
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

    /// The caller-saved registers of the loop caches valid in `node`'s block (P3 stage 3 step
    /// R1): a cold path's call clobbers them, and the loop reads them after it, so the call keeps
    /// them with the live values. A callee-saved cache needs nothing.
    fn caller_saved_caches(&self, node: NodeIx) -> Vec<(bool, u32)> {
        let file = &regalloc::ARM64;
        self.alloc
            .caches_in(self.graph.node(node).block)
            .filter_map(|c| match c.reg {
                Home::Gp(r) if file.gp_caller.contains(&r) => Some((false, u32::from(r))),
                Home::Fp(r) if file.fp_caller.contains(&r) => Some((true, u32::from(r))),
                _ => None,
            })
            .collect()
    }

    /// [`Gen::save_live_caller_saved`], with the x registers in `scratch` saved too: values of a
    /// node's own cold path that its call would clobber (an element store's words in x16, which
    /// its output takes after the call). `restore_live_caller_saved` restores them all.
    fn save_live_caller_saved_and(
        &mut self,
        node: NodeIx,
        scratch: &[u32],
    ) -> (Vec<(bool, u32)>, u32) {
        let file = &regalloc::ARM64;
        let mut live: Vec<(bool, u32)> = self
            .alloc
            .live_at(node)
            .into_iter()
            .filter_map(|v| match self.home(v) {
                Home::Gp(r) if file.gp_caller.contains(&r) => Some((false, u32::from(r))),
                Home::Fp(r) if file.fp_caller.contains(&r) => Some((true, u32::from(r))),
                _ => None,
            })
            .collect();
        live.extend(self.caller_saved_caches(node));
        live.extend(scratch.iter().map(|&r| (false, r)));
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

    /// Save on the native stack the values a call made on `node`'s own path clobbers
    /// (`regalloc::clobbers`) and the code after the call still reads: those live past `node`
    /// (none, as the allocator homes them: kept as a guard) and those of `node`'s frame state,
    /// which its exit after the call reads. Its inputs, read before the call, are not. Maglev's
    /// register snapshot of a node likewise takes its eager deopt's inputs
    /// (maglev/maglev-regalloc.cc:1708-1730). Returns what `restore_live_caller_saved` restores.
    fn save_live_clobbered(&mut self, node: NodeIx) -> (Vec<(bool, u32)>, u32) {
        let (gp, fp) = regalloc::clobbers(&self.graph.node(node).op);
        let at = self.alloc.pos[node.index()];
        // What the node's exit reads: its state's values and, in an inlined frame, its chain's
        // (stage 3), the values the allocator's reads hold at the node (`for_each_exit_value`).
        let mut reads = Vec::new();
        if let Some(state) = self.graph.node(node).state {
            self.graph.for_each_exit_value(state, |v| reads.push(v));
        }
        let in_state = |v: NodeIx| reads.contains(&v);
        let live: Vec<(bool, u32)> = self
            .alloc
            .live_at(node)
            .into_iter()
            .filter(|&v| self.alloc.ranges[v.index()].1 > at || in_state(v))
            .filter_map(|v| match self.home(v) {
                Home::Gp(r) if gp & (1 << r) != 0 => Some((false, u32::from(r))),
                Home::Fp(r) if fp & (1 << r) != 0 => Some((true, u32::from(r))),
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
        // The compiled frame's operand top, which the helper keeps as its record's live top
        // (`JitCtx::poll_top`): its state's depth, or in an inlined frame (stage 3) the depth of
        // the compiled frame's state at the outermost call, whose frame the record holds.
        let mut state = self
            .graph
            .state(self.graph.node(node).state.expect("checked"));
        while let Some(inlined) = self.graph.inlined(state.frame) {
            state = self.graph.state(inlined.parent);
        }
        let depth = state.stack.len() as u32;
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

    /// Branch to `not_object` unless `word` is an object's; otherwise x30 = its object's address
    /// (the cell plus `obj_from_cell`), for its kind and flag bytes. Clobbers x17 and x30.
    fn object_or(&mut self, word: u32, not_object: usize) {
        // The Object tag 0xfff9 is -7 as a signed half.
        self.a.asr_imm(X17, word, 48);
        self.a.cmn_imm_x(X17, 7);
        self.a.b_cond(NE, not_object);
        self.a.ubfx(X30, word, 0, 48);
        self.a.add_imm(X30, X30, self.layout.obj_from_cell as u32);
    }

    /// NullOrLikeUndefined (P3 stage 3 step L1): undefined and null by `tag | 2`, an object whose
    /// ic_plain is set false, one whose ic_plain is clear by the cold call of
    /// `Interp::is_htmldda`, any other word false. Each path reads the word before it writes the
    /// result, which may share its register.
    fn lower_null_or_like_undefined(&mut self, n: NodeIx) {
        use crate::value::PACK_NULL;
        let word = self.gp(self.graph.node(n).inputs[0], X16);
        let w = self.gp_dst(n);
        let nullish = self.a.new_label();
        let no = self.a.new_label();
        let slow = self.a.new_label();
        let done = self.a.new_label();
        // Undefined's and null's tags differ in bit 1 alone: `tag | 2` is null's for both.
        self.a.lsr_imm(X17, word, 48);
        let two = asm::logical_imm_w(2).expect("2 is a logical immediate");
        self.a.logic_imm_w(1, X17, X17, two);
        self.a.movz(X30, (PACK_NULL >> 48) as u32, 0);
        self.a.cmp_reg_w(X17, X30);
        self.a.b_cond(EQ, nullish);
        self.object_or(word, no);
        self.a.ldrb_imm(X30, X30, self.layout.obj_flags as u32);
        self.a.tbz(X30, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
        self.a.bind(no);
        self.a.movz(w, 0, 0);
        self.a.b(done);
        self.a.bind(nullish);
        self.a.movz(w, 1, 0);
        self.a.bind(done);
        self.cold.push(Cold::TestCall {
            label: slow,
            node: n,
            word,
            dst: w,
            resume: done,
            helper: emulates_undefined as *const () as u64,
            arg: TestArg::Imm(0),
        });
        self.gp_done(n, w);
    }

    /// Branch to `not_nullish` unless `word` is undefined or null (`tag | 2` is null's tag for those
    /// two alone, their tags differing in bit 1, as NullOrLikeUndefined and the templates' `==
    /// null` test it; the TDZ marker's 0x7ffa stays 0x7ffa). Clobbers x9 and x30.
    fn nullish_or(&mut self, word: u32, not_nullish: usize) {
        self.a.lsr_imm(X30, word, 48);
        let two = asm::logical_imm_w(2).expect("2 is a logical immediate");
        self.a.logic_imm_w(1, X30, X30, two);
        self.a.movz(X9, (crate::value::PACK_NULL >> 48) as u32, 0);
        self.a.cmp_reg_w(X30, X9);
        self.a.b_cond(NE, not_nullish);
    }

    /// ObjectOrNullishEqual (P3 stage 3 step L1b): IsLooselyEqual of two words each an object or
    /// nullish (their checks made), TurboFan's form (compiler/js-typed-lowering.cc:1009-1060):
    /// the same word true; else, where one is nullish, whether the other is nullish or
    /// [[IsHTMLDDA]] (an object of ic_plain set is not, one of ic_plain clear takes the cold call
    /// of `Interp::is_htmldda`, as NullOrLikeUndefined decides); else (two objects) false. The
    /// inputs are read before the result is written; only x9 and x30 are written before that.
    fn lower_object_or_nullish_equal(&mut self, n: NodeIx) {
        let node = self.graph.node(n);
        let x = self.gp(node.inputs[0], X16);
        let y = self.gp(node.inputs[1], X17);
        let w = self.gp_dst(n);
        let yes = self.a.new_label();
        let no = self.a.new_label();
        let done = self.a.new_label();
        let x_not = self.a.new_label();
        let slow_x = self.a.new_label();
        let slow_y = self.a.new_label();
        self.a.cmp_reg_x(x, y);
        self.a.b_cond(EQ, yes);
        self.nullish_or(x, x_not);
        // x nullish: equal where y is nullish, or an object that is [[IsHTMLDDA]].
        self.not_nullish_object_test(y, yes, no, slow_y);
        self.a.bind(x_not);
        // x an object: equal where y is nullish and x [[IsHTMLDDA]]; two objects differ.
        self.nullish_or(y, no);
        self.htmldda_test(x, no, slow_x);
        self.a.bind(no);
        self.a.movz(w, 0, 0);
        self.a.b(done);
        self.a.bind(yes);
        self.a.movz(w, 1, 0);
        self.a.bind(done);
        for (label, word) in [(slow_x, x), (slow_y, y)] {
            self.cold.push(Cold::TestCall {
                label,
                node: n,
                word,
                dst: w,
                resume: done,
                helper: emulates_undefined as *const () as u64,
                arg: TestArg::Imm(0),
            });
        }
        self.gp_done(n, w);
    }

    /// For `word`, an object or nullish: to `nullish` where it is nullish, else as
    /// [`Gen::htmldda_test`]. Clobbers x9 and x30.
    fn not_nullish_object_test(&mut self, word: u32, nullish: usize, no: usize, slow: usize) {
        let object = self.a.new_label();
        self.nullish_or(word, object);
        self.a.b(nullish);
        self.a.bind(object);
        self.htmldda_test(word, no, slow);
    }

    /// For `word`, an object: to `no` where its ic_plain is set (no [[IsHTMLDDA]] object has it),
    /// else to `slow`, the cold call that asks. Clobbers x30.
    fn htmldda_test(&mut self, word: u32, no: usize, slow: usize) {
        self.a.ubfx(X30, word, 0, 48);
        self.a.add_imm(X30, X30, self.layout.obj_from_cell as u32);
        self.a.ldrb_imm(X30, X30, self.layout.obj_flags as u32);
        self.a.tbz(X30, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
        self.a.b(no);
    }

    /// StringEqual (P3 stage 3 step L1): two proven Strings, the same cell equal, lengths that
    /// differ unequal, equal lengths the cold call of the contents' compare (IsStrictlyEqual, as
    /// the shared Equal stub and the checked helper compare them). The inputs are read before the
    /// result is written.
    fn lower_string_equal(&mut self, n: NodeIx) {
        let node = self.graph.node(n);
        let x = self.gp(node.inputs[0], X16);
        let y = self.gp(node.inputs[1], X17);
        let w = self.gp_dst(n);
        let yes = self.a.new_label();
        let no = self.a.new_label();
        let slow = self.a.new_label();
        let done = self.a.new_label();
        self.a.cmp_reg_x(x, y);
        self.a.b_cond(EQ, yes);
        let len = crate::lstr::LEN_OFF as u32;
        self.a.ubfx(X30, x, 0, 48);
        self.a.ldr_w_imm(X30, X30, len);
        self.a.ubfx(X9, y, 0, 48);
        self.a.ldr_w_imm(X9, X9, len);
        self.a.cmp_reg_w(X30, X9);
        self.a.b_cond(EQ, slow);
        self.a.bind(no);
        self.a.movz(w, 0, 0);
        self.a.b(done);
        self.a.bind(yes);
        self.a.movz(w, 1, 0);
        self.a.bind(done);
        self.cold.push(Cold::TestCall {
            label: slow,
            node: n,
            word: x,
            dst: w,
            resume: done,
            helper: strings_equal as *const () as u64,
            arg: TestArg::Reg(y),
        });
        self.gp_done(n, w);
    }

    /// TypeofIs (P3 stage 3 step L1): the interpreter's typeof_matches. A primitive by its tag;
    /// an object by its kind tag (the callables' range, KIND_CALLABLE_FIRST..=KIND_EXOTIC_NONE_LAST,
    /// and the proxy's) and ic_plain: "object" is true for null and an object of no callable
    /// kind ([[IsHTMLDDA]] is a callable native), "function" for a callable kind with ic_plain
    /// set, "undefined" for undefined (and the TDZ marker, as typeof_matches takes it); a proxy,
    /// and an object that may be [[IsHTMLDDA]] where that decides, take the cold call of
    /// typeof_matches. Each path reads the word before it writes the result.
    fn lower_typeof_is(&mut self, n: NodeIx, test: crate::value::TypeofTest) {
        use crate::value::{
            TypeofTest, KIND_CALLABLE_FIRST, KIND_EXOTIC_NONE_LAST, KIND_PROXY_TAG, PACK_BIGINT,
            PACK_BOOL, PACK_NULL, PACK_STR, PACK_SYM, PACK_UNDEFINED,
        };
        let word = self.gp(self.graph.node(n).inputs[0], X16);
        let w = self.gp_dst(n);
        let yes = self.a.new_label();
        let no = self.a.new_label();
        let slow = self.a.new_label();
        let done = self.a.new_label();
        let callables = u32::from(KIND_EXOTIC_NONE_LAST - KIND_CALLABLE_FIRST);
        let proxy = u32::from(KIND_PROXY_TAG - KIND_CALLABLE_FIRST);
        // The object's kind tag less the first callable's: at most `callables` for a callable kind.
        let kind = |g: &mut Self| {
            g.a.ldrb_imm(X17, X30, g.layout.obj_kind as u32);
            g.a.sub_imm(X17, X17, u32::from(KIND_CALLABLE_FIRST));
        };
        match test {
            TypeofTest::Number => {
                // Every word but a Number's is a NaN, and the canonical NaN is a Number.
                self.a.fmov_d_x(D31, word);
                self.a.fcmp(D31, D31);
                self.a.b_cond(VC, yes);
                self.a.mov_imm64(X17, crate::value::PACK_CANON_NAN);
                self.a.cmp_reg_x(word, X17);
                self.a.b_cond(EQ, yes);
                self.a.b(no);
            }
            TypeofTest::Boolean => {
                // A Boolean's word EOR its tag is its payload, 0 or 1; every other word is larger.
                self.a.movz(X17, (PACK_BOOL >> 48) as u32, 3);
                self.a.logic_x(2, X17, word, X17);
                self.a.cmp_imm_x(X17, 1);
                self.a.b_cond(LS, yes);
                self.a.b(no);
            }
            TypeofTest::String | TypeofTest::Symbol | TypeofTest::BigInt => {
                let tag = match test {
                    TypeofTest::String => PACK_STR,
                    TypeofTest::Symbol => PACK_SYM,
                    _ => PACK_BIGINT,
                };
                self.a.lsr_imm(X17, word, 48);
                self.a.movz(X30, (tag >> 48) as u32, 0);
                self.a.cmp_reg_w(X17, X30);
                self.a.b_cond(EQ, yes);
                self.a.b(no);
            }
            TypeofTest::Undefined => {
                // Undefined's tag, or the TDZ marker's after it.
                self.a.lsr_imm(X17, word, 48);
                self.a.movz(X30, (PACK_UNDEFINED >> 48) as u32, 0);
                self.a.sub_reg(X17, X17, X30);
                self.a.cmp_imm_x(X17, 1);
                self.a.b_cond(LS, yes);
                self.object_or(word, no);
                self.a.ldrb_imm(X30, X30, self.layout.obj_flags as u32);
                self.a.tbz(X30, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
                self.a.b(no);
            }
            TypeofTest::Object => {
                self.a.mov_imm64(X17, PACK_NULL);
                self.a.cmp_reg_x(word, X17);
                self.a.b_cond(EQ, yes);
                self.object_or(word, no);
                kind(self);
                self.a.cmp_imm_x(X17, callables);
                self.a.b_cond(LS, no);
                self.a.cmp_imm_x(X17, proxy);
                self.a.b_cond(EQ, slow);
                self.a.b(yes);
            }
            TypeofTest::Function => {
                self.object_or(word, no);
                self.a.ldrb_imm(X9, X30, self.layout.obj_flags as u32);
                kind(self);
                self.a.cmp_imm_x(X17, proxy);
                self.a.b_cond(EQ, slow);
                self.a.cmp_imm_x(X17, callables);
                self.a.b_cond(HI, no);
                // A callable: [[IsHTMLDDA]] (a native whose ic_plain is clear) is "undefined".
                self.a.tbz(X9, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
                self.a.b(yes);
            }
            // The builder folds a test no typeof result matches.
            TypeofTest::Never => self.a.b(no),
        }
        self.a.bind(no);
        self.a.movz(w, 0, 0);
        self.a.b(done);
        self.a.bind(yes);
        self.a.movz(w, 1, 0);
        self.a.bind(done);
        self.cold.push(Cold::TestCall {
            label: slow,
            node: n,
            word,
            dst: w,
            resume: done,
            helper: typeof_matches as *const () as u64,
            arg: TestArg::Imm(u64::from(test as u8)),
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

/// Element accesses (P3 stage 2 step 2): Maglev's element nodes on TRust's element words (ir.rs).
impl Gen<'_> {
    /// The typed arrays' native layout, which the typed nodes read through.
    fn native(&self) -> Result<crate::native_typed_array::Layout, Refusal> {
        self.layout
            .native_typed_array
            .ok_or(Refusal::whole("typed arrays without a probed layout"))
    }

    /// TypedData: as the template's access tests a typed array (`typed_array::emit_access`), the
    /// buffer attached, writable for a store, its bytes borrowed by no Rust code, and reaching
    /// the view's end (a resizable buffer that shrank fails); then the elements' address, the
    /// bytes' plus the view's offset. Native accesses reload the bytes' header, never keeping its
    /// address (native_typed_array.rs), and this one keeps it for its effect epoch only.
    fn lower_typed_data(&mut self, n: NodeIx, write: bool) -> Result<(), Refusal> {
        use crate::native_typed_array::{NativeBuffer as Buffer, NativeTypedArray as View};
        use std::mem::offset_of;
        let native = self.native()?;
        let site = self.check_exit(n)?;
        let view = self.gp(self.graph.node(n).inputs[1], X16);
        self.a
            .ldr_imm(X17, view, offset_of!(View, buffer_ptr) as u32);
        self.a
            .ldrb_imm(X9, X17, offset_of!(Buffer, attached) as u32);
        self.a.cbz(X9, false, site);
        if write {
            self.a
                .ldrb_imm(X9, X17, offset_of!(Buffer, writable) as u32);
            self.a.cbz(X9, false, site);
        }
        self.a
            .ldr_imm(X17, X17, offset_of!(Buffer, storage_ptr) as u32);
        // The RefCell's borrow flag, first: no Rust borrow while the code holds the address.
        self.a.ldr_imm(X9, X17, 0);
        self.a.cbnz(X9, true, site);
        self.a
            .ldr_imm(X9, X17, (native.refcell_value + native.vec_len) as u32);
        self.a.ldr_imm(X30, view, offset_of!(View, end) as u32);
        self.a.cmp_reg_x(X9, X30);
        self.a.b_cond(LO, site);
        let dst = self.gp_dst(n);
        self.a
            .ldr_imm(X9, X17, (native.refcell_value + native.vec_ptr) as u32);
        self.a.ldr_imm(X30, view, offset_of!(View, offset) as u32);
        self.a.add_shifted(dst, X9, X30, 0);
        self.gp_done(n, dst);
        Ok(())
    }

    /// LoadTyped: one load of the element's width at the zero-extended index, an int32 kind's
    /// sign- or zero-extended, Uint32's converted unsigned, Float32's widened (exactly).
    fn lower_load_typed(&mut self, n: NodeIx, kind: TypedKind) {
        let node = self.graph.node(n);
        let data = self.gp(node.inputs[1], X16);
        let index = self.gp(node.inputs[2], X17);
        self.a.mov_w(X9, index);
        if kind.rep(false) == Rep::I32 {
            let w = self.gp_dst(n);
            match kind {
                TypedKind::U8 | TypedKind::U8Clamped => self.a.ldrb_reg(w, data, X9),
                TypedKind::I8 => self.a.ldrsb_reg(w, data, X9),
                TypedKind::U16 => self.a.ldrh_lsl1(w, data, X9),
                TypedKind::I16 => self.a.ldrsh_lsl1(w, data, X9),
                _ => self.a.ldr_w_lsl2(w, data, X9),
            }
            self.gp_done(n, w);
            return;
        }
        let d = self.fp_dst(n);
        match kind {
            TypedKind::U32 => {
                self.a.ldr_w_lsl2(X9, data, X9);
                self.a.ucvtf_d_w(d, X9);
            }
            TypedKind::F32 => {
                self.a.ldr_s_lsl2(D31, data, X9);
                self.a.fcvt_float_width(d, D31, false);
            }
            _ => self.a.ldr_d_lsl3(d, data, X9),
        }
        self.fp_done(n, d);
    }

    /// LoadTypedOrUndefined: the unsigned bounds check as a branch to undefined, then LoadTyped's
    /// element boxed, an int32 kind's and Uint32's converted exactly, a float's NaN canonicalized
    /// (its bytes may hold any NaN, which would read as a tag), as the template boxes it
    /// (`emit_float_result`).
    fn lower_load_typed_or_undefined(&mut self, n: NodeIx, kind: TypedKind) {
        let node = self.graph.node(n);
        let data = self.gp(node.inputs[1], X16);
        let index = self.gp(node.inputs[2], X17);
        let length = self.gp(node.inputs[3], X30);
        let dst = self.gp_dst(n);
        let past = self.a.new_label();
        let done = self.a.new_label();
        self.a.cmp_reg_w(index, length);
        self.a.b_cond(HS, past);
        self.a.mov_w(X30, index);
        match kind {
            TypedKind::U32 => {
                self.a.ldr_w_lsl2(X30, data, X30);
                self.a.ucvtf_d_w(D31, X30);
            }
            TypedKind::F32 => {
                self.a.ldr_s_lsl2(D31, data, X30);
                self.a.fcvt_float_width(D31, D31, false);
            }
            TypedKind::F64 => self.a.ldr_d_lsl3(D31, data, X30),
            _ => {
                match kind {
                    TypedKind::U8 | TypedKind::U8Clamped => self.a.ldrb_reg(X30, data, X30),
                    TypedKind::I8 => self.a.ldrsb_reg(X30, data, X30),
                    TypedKind::U16 => self.a.ldrh_lsl1(X30, data, X30),
                    TypedKind::I16 => self.a.ldrsh_lsl1(X30, data, X30),
                    _ => self.a.ldr_w_lsl2(X30, data, X30),
                }
                self.a.scvtf_d_w(D31, X30);
            }
        }
        self.a.fmov_x_d(dst, D31);
        if matches!(kind, TypedKind::F32 | TypedKind::F64) {
            self.a.fcmp(D31, D31);
            self.a.movz(X30, 0x7ff8, 3); // PACK_CANON_NAN
            self.a.csel_x(dst, X30, dst, VS);
        }
        self.a.b(done);
        self.a.bind(past);
        self.a.mov_imm64(dst, crate::value::PACK_UNDEFINED);
        self.a.bind(done);
        self.gp_done(n, dst);
    }

    /// StoreTyped: the value's low bits (an integer kind's) or its float at the zero-extended
    /// index, then the buffer's dirty interval and write count for embedders, as the template's
    /// `emit_dirty` keeps them: [view offset + index << shift, + width), one write.
    fn lower_store_typed(&mut self, n: NodeIx, kind: TypedKind) {
        use crate::native_typed_array::{NativeBuffer as Buffer, NativeTypedArray as View};
        use std::mem::offset_of;
        let node = self.graph.node(n);
        let (view, data, index, value) = (
            node.inputs[1],
            node.inputs[2],
            node.inputs[3],
            node.inputs[4],
        );
        let data = self.gp(data, X17);
        let index = self.gp(index, X9);
        self.a.mov_w(X9, index);
        // log2 of the element's bytes.
        let shift = match kind {
            TypedKind::U8 | TypedKind::I8 | TypedKind::U8Clamped => 0,
            TypedKind::U16 | TypedKind::I16 => 1,
            TypedKind::U32 | TypedKind::I32 | TypedKind::F32 => 2,
            TypedKind::F64 => 3,
        };
        match kind {
            TypedKind::F32 => {
                let d = self.fp(value, D30);
                self.a.fcvt_float_width(D31, d, true);
                self.a.str_s_lsl2(D31, data, X9);
            }
            TypedKind::F64 => {
                let d = self.fp(value, D30);
                self.a.str_d_lsl3(d, data, X9);
            }
            _ => {
                let w = self.gp(value, X30);
                match shift {
                    0 => self.a.strb_reg(w, data, X9),
                    1 => self.a.strh_lsl1(w, data, X9),
                    _ => self.a.str_w_lsl2(w, data, X9),
                }
            }
        }
        let view = self.gp(view, X16);
        self.a.ldr_imm(X17, view, offset_of!(View, offset) as u32);
        self.a.add_shifted(X9, X17, X9, shift);
        self.a
            .ldr_imm(X17, view, offset_of!(View, buffer_ptr) as u32);
        self.a.add_imm(X30, X9, 1 << shift);
        let start_done = self.a.new_label();
        let end_done = self.a.new_label();
        let start = offset_of!(Buffer, dirty_start) as u32;
        let end = offset_of!(Buffer, dirty_end) as u32;
        let writes = offset_of!(Buffer, writes) as u32;
        self.a.ldr_imm(X16, X17, start);
        self.a.cmp_reg_x(X9, X16);
        self.a.b_cond(HS, start_done);
        self.a.str_imm(X9, X17, start);
        self.a.bind(start_done);
        self.a.ldr_imm(X16, X17, end);
        self.a.cmp_reg_x(X30, X16);
        self.a.b_cond(LS, end_done);
        self.a.str_imm(X30, X17, end);
        self.a.bind(end_done);
        self.a.ldr_imm(X16, X17, writes);
        self.a.add_imm(X16, X16, 1);
        self.a.str_imm(X16, X17, writes);
    }

    /// TypedElement: the template's TypedElement stub on its register contract (x11 the Object,
    /// x9 the index, w12 the kind tag, d1 a store's value -> w13 1 or 0, x12 a load's word; it
    /// clobbers x9, x12-x17, d1, d2 and x30: jit_typed_array.rs). The allocator gives no value
    /// live across the call a register the call clobbers (`regalloc::clobbers`), so the call
    /// saves only the values its exit reads that sit in one (P3 stage 3 step 0b: saving every
    /// live caller-saved register, as a cold path's call does, cost 13% of zlib's instructions).
    /// A 0 exits.
    fn lower_typed_element(&mut self, n: NodeIx, write: bool) -> Result<(), Refusal> {
        let node = self.graph.node(n);
        let site = self.check_exit(n)?;
        let cell = self.gp(node.inputs[0], X16);
        let index = self.gp(node.inputs[1], X17);
        let value = if write {
            Some(self.fp(node.inputs[2], D30))
        } else {
            None
        };
        let saved = self.save_live_clobbered(n);
        // The index first: x11 may be its home.
        self.a.mov_w(X9, index);
        self.a.add_imm(11, cell, self.layout.obj_from_cell as u32);
        self.a
            .movz(12, crate::value::KIND_TYPED_ARRAY_TAG as u32, 0);
        if let Some(d) = value {
            self.a.fmov_d_d(1, d);
        }
        let stub = self.a.shared_stub(SharedStub::TypedElement { write }.key());
        self.a.bl_label(stub);
        self.a.mov(X16, 13);
        self.a.mov(X17, 12);
        self.restore_live_caller_saved(&saved);
        self.a.cbz(X16, false, site);
        if !write && self.home(n) != Home::None {
            let dst = self.gp_dst(n);
            self.a.mov(dst, X17);
            self.gp_done(n, dst);
        }
        Ok(())
    }

    /// A further exit of `n`'s frame state, with `reason`: a node one of whose paths fails for
    /// another reason than its own (an element read past the used words beside its hole path).
    fn exit_for(&mut self, n: NodeIx, reason: ExitReason) -> Result<usize, Refusal> {
        let site = self.exit(n, ExitKind::Check)?;
        if let Some((_, _, record)) = self.exits.last_mut() {
            record.reason = reason;
        }
        Ok(site)
    }

    /// LoadElement: the word at the index (below the used length, as a check proved), the index
    /// register zero-extended (`uxtw`: an int32's register guarantees its low 32 bits only); a
    /// hole exits, or reads undefined where the cold path proves no element can come from the
    /// prototype chain.
    fn lower_load_element(&mut self, n: NodeIx, holes: Holes) -> Result<(), Refusal> {
        let node = self.graph.node(n);
        let site = self.check_exit(n)?;
        let words = self.gp(node.inputs[1], X16);
        let index = self.gp(node.inputs[2], X17);
        let dst = self.gp_dst(n);
        self.a.ldr_x_uxtw3(dst, words, index);
        self.a.mov_imm64(X30, crate::value::PACK_EMPTY);
        self.a.cmp_reg_x(dst, X30);
        match holes {
            Holes::Exit => self.a.b_cond(EQ, site),
            Holes::Undefined => {
                let hole = self.a.new_label();
                let resume = self.a.new_label();
                self.a.b_cond(EQ, hole);
                self.a.bind(resume);
                self.cold.push(Cold::ElementUndefined {
                    label: hole,
                    node: n,
                    dst,
                    resume,
                    exit: site,
                    negative: None,
                });
            }
        }
        self.gp_done(n, dst);
        Ok(())
    }

    /// LoadElementOrUndefined: an index below the used length reads as LoadElement(Undefined)
    /// does; at or past it the cold path reads undefined under the same proof, a negative index
    /// exiting [OutOfBounds] (Warp's LoadDenseElementHoleResult bails on one) and a hole whose
    /// proof fails [Hole].
    fn lower_load_element_or_undefined(&mut self, n: NodeIx) -> Result<(), Refusal> {
        let node = self.graph.node(n);
        let past = self.check_exit(n)?;
        let hole_exit = self.exit_for(n, ExitReason::Hole)?;
        let words = self.gp(node.inputs[1], X16);
        let index = self.gp(node.inputs[2], X17);
        let used = self.gp(node.inputs[3], X9);
        let dst = self.gp_dst(n);
        let past_used = self.a.new_label();
        let hole = self.a.new_label();
        let resume = self.a.new_label();
        // The used length first: it may share x9 with the destination.
        self.a.cmp_reg_w(index, used);
        self.a.b_cond(HS, past_used);
        self.a.ldr_x_uxtw3(dst, words, index);
        self.a.mov_imm64(X30, crate::value::PACK_EMPTY);
        self.a.cmp_reg_x(dst, X30);
        self.a.b_cond(EQ, hole);
        self.a.bind(resume);
        self.cold.push(Cold::ElementUndefined {
            label: hole,
            node: n,
            dst,
            resume,
            exit: hole_exit,
            negative: None,
        });
        self.cold.push(Cold::ElementUndefined {
            label: past_used,
            node: n,
            dst,
            resume,
            exit: past,
            negative: Some((index, past)),
        });
        self.gp_done(n, dst);
        Ok(())
    }

    /// StoreElement (ir.rs): the words in `out` (the output's register, or x16), which a copy of
    /// a shared literal's block replaces before the store runs again; under Existing the bounds
    /// and the hole checked first; then the value's store by its representation, every check
    /// before the first write. The cell, the first input, keeps the words alive, and marks its
    /// card for an edge.
    /// StoreNew (P3 stage 3 step L3): the template's creation commit (`emit_prop_create_probe`
    /// and `emit_set_prop_inline`'s commit, jit.rs) on the tier's scratch registers, its shapes
    /// and location the node's constants: its conditions first (no key index of its own, a named
    /// count below 8, the key layout, room in a record slot), any failing one taking the insert's
    /// pure call (`store_new_slow`, cold) with nothing written, where the template takes its
    /// helper; then the commit with no call between its writes: the layout where one is
    /// installed, the value, the card (the Object's address shares its cell's card), the landing
    /// shape and the count, as the template orders them (the count last: the collector reads the
    /// slots below it). The node exits only where the call could not borrow the object.
    fn lower_store_new(
        &mut self,
        n: NodeIx,
        location: u16,
        shape: u32,
        key: u64,
        ways: u8,
    ) -> Result<(), Refusal> {
        use crate::value::{
            SHAPE_LAYOUT_PAGE_BITS, SHAPE_LAYOUT_PAGE_COUNT, SHAPE_LAYOUT_PAGE_SIZE,
        };
        let node = self.graph.node(n);
        let (cell, value) = (node.inputs[0], node.inputs[1]);
        let layout = self.layout;
        let site = self.check_exit(n)?;
        let pc = self.graph.state(node.state.expect("checked")).pc;
        let receiver = self.source(cell, pc)?;
        if let Some((_, _, record)) = self.exits.last_mut() {
            record.shape_check = Some(Box::new(crate::opt::exit::ShapeCheck {
                receiver,
                shapes: Box::new([]),
                ways,
                hop: false,
            }));
        }
        // A record slot's index in the record, which the location names past the record's header.
        let record_index = if location & 1 != 0 {
            let offset = usize::from(location >> 1);
            if offset < layout.record_slots || !(offset - layout.record_slots).is_multiple_of(8) {
                return Err(Refusal::at(pc as usize, "a creation's record slot"));
            }
            Some((offset - layout.record_slots) / 8)
        } else {
            None
        };
        let c = self.gp(cell, X16);
        self.a.add_imm(X17, c, layout.obj_from_cell as u32);
        let slow = self.a.new_label();
        let done = self.a.new_label();
        self.cold.push(Cold::CreateSlow {
            label: slow,
            node: n,
            done,
            exit: site,
        });
        // 1. A small map: no key index of its own, fewer than 8 named properties (the 9th needs
        //    one, value.rs ensure_fallback_index): x9 the count.
        guard_no_key_index(&mut self.a, layout, X17, X16, slow);
        named_len(&mut self.a, layout, X9, X17);
        self.a.cmp_imm_x(X9, 8);
        self.a.b_cond(HS, slow);
        // 2. The key layout for the next slot: the object's own prediction holding the key with
        //    plain attributes (x16 = 0, nothing to install), else the landing shape's cached
        //    layout from the shape-layout directory (x16), whose entry must be the shape's.
        let transition = self.a.new_label();
        let laid = self.a.new_label();
        let plain = self.a.new_label();
        self.a
            .ldr_imm(X16, X17, (layout.obj_props + layout.props_layout) as u32);
        self.a.cbz(X16, true, transition);
        self.a.ldr_w_imm(X30, X16, layout.layout_keys_len as u32);
        self.a.cmp_reg_x(X9, X30);
        self.a.b_cond(HS, transition);
        self.a.ldr_w_imm(X30, X16, layout.layout_attrs_len as u32);
        self.a.cbz(X30, false, plain);
        self.a.ldr_imm(X30, X16, layout.layout_attrs_ptr as u32);
        self.a.ldrb_reg(X30, X30, X9);
        self.a.cmp_imm_w(X30, crate::value::PROP_PLAIN as u32);
        self.a.b_cond(NE, transition);
        self.a.bind(plain);
        self.a.ldr_imm(X16, X16, layout.layout_keys_ptr as u32);
        self.a.ldr_x_lsl3(X16, X16, X9);
        self.a.mov_imm64(X30, key);
        self.a.cmp_reg_x(X16, X30);
        self.a.b_cond(NE, transition);
        self.a.movz(X16, 0, 0);
        self.a.b(laid);
        self.a.bind(transition);
        // The directory's candidate for the shape (property_shapes.rs ShapeLayouts::index).
        let index = shape as usize & (SHAPE_LAYOUT_PAGE_SIZE * SHAPE_LAYOUT_PAGE_COUNT - 1);
        let (page, entry) = (
            index >> SHAPE_LAYOUT_PAGE_BITS,
            index & ((1 << SHAPE_LAYOUT_PAGE_BITS) - 1),
        );
        self.a
            .ldr_imm(X16, 19, std::mem::offset_of!(JitCtx, shape_layouts) as u32);
        self.a.ldr_imm(X16, X16, (8 * page) as u32);
        self.a.cbz(X16, true, slow);
        self.a
            .add_imm(X16, X16, (entry * layout.shape_layout_entry_size) as u32);
        self.a
            .ldr_w_imm(X30, X16, layout.shape_layout_entry_id as u32);
        if shape < 4096 {
            self.a.cmp_imm_w(X30, shape);
        } else {
            self.a.mov_imm64(X9, u64::from(shape));
            self.a.cmp_reg_w(X30, X9);
        }
        self.a.b_cond(NE, slow);
        self.a
            .ldr_imm(X16, X16, layout.shape_layout_entry_keys as u32);
        self.a.cbz(X16, true, slow);
        named_len(&mut self.a, layout, X9, X17);
        self.a.ldr_w_imm(X30, X16, layout.layout_keys_len as u32);
        self.a.cmp_reg_x(X9, X30);
        self.a.b_cond(HS, slow);
        self.a.bind(laid);
        // 3. Room for the slot: the landing shape proves a cell slot's; a record slot needs the
        //    record and its capacity (`guard_loc_room`), else the slow path grows it.
        if let Some(index) = record_index {
            self.a
                .ldr_imm(X30, X17, (layout.obj_props + layout.props_record) as u32);
            self.a.cbz(X30, true, slow);
            self.a.ldr_w_imm(X30, X30, layout.record_cap as u32);
            self.a.cmp_imm_w(X30, index as u32);
            self.a.b_cond(LS, slow);
        }
        // 4. The commit.
        let kept = self.a.new_label();
        self.a.cbz(X16, true, kept);
        self.a
            .str_imm(X16, X17, (layout.obj_props + layout.props_layout) as u32);
        self.a.bind(kept);
        slot_addr_loc_const(&mut self.a, layout, X30, X17, location);
        let word = self.gp(value, X9);
        self.a.str_imm(word, X30, 0);
        // The write barrier (the template's site B): the Object can be old (a constructor's
        // `this` made before the callee's entry poll), and an installed layout is an edge too.
        debug_assert_object_in_cell_card(layout);
        emit_card_mark(&mut self.a, X17, X9, X30);
        if shape < 4096 {
            self.a.movz(X9, shape, 0);
        } else {
            self.a.mov_imm64(X9, u64::from(shape));
        }
        self.a
            .str_w_imm(X9, X17, (layout.obj_props + layout.props_shape) as u32);
        named_len(&mut self.a, layout, X9, X17);
        self.a.add_imm(X9, X9, 1);
        named_set_len(&mut self.a, layout, X9, X17);
        self.a.bind(done);
        Ok(())
    }

    fn lower_store_element(&mut self, n: NodeIx, mode: Store) -> Result<(), Refusal> {
        use crate::value::word_block::{KIND_INT32, KIND_UNWRITABLE};
        let node = self.graph.node(n);
        let &[_, words, index, value] = &node.inputs[..] else {
            return Err(Refusal::at(node.pc as usize, "an element store's operands"));
        };
        let layout = self.layout;
        let flags = -(layout.words_flags as i32);
        let wrong = self.check_exit(n)?;
        let out = match self.home(n) {
            Home::Gp(r) => u32::from(r),
            _ => X16,
        };
        let retry = self.a.new_label();
        let unwritable = self.a.new_label();
        let done = self.a.new_label();
        // Where an index past the used words, and a hole, go: an exit, or for an Array's store
        // the hole fill or append.
        let checks = match mode {
            Store::Existing => Some((
                self.exit_for(n, ExitReason::OutOfBounds)?,
                self.exit_for(n, ExitReason::Hole)?,
            )),
            Store::Create => {
                let create = self.a.new_label();
                let exits = CreateExits {
                    wrong,
                    gap: self.exit_for(n, ExitReason::OutOfBounds)?,
                    grow: self.exit_for(n, ExitReason::CouldNotGrow)?,
                };
                self.cold.push(Cold::ElementCreate {
                    label: create,
                    node: n,
                    out,
                    done,
                    exits,
                });
                Some((create, create))
            }
            Store::Present => None,
        };
        let input = self.gp(words, X16);
        if input != out {
            self.a.mov(out, input);
        }
        self.a.bind(retry);
        if let Some((past, hole)) = checks {
            self.a.ldur_w(X9, out, -(layout.words_used as i32));
            let at = self.gp(index, X17);
            self.a.cmp_reg_w(at, X9);
            self.a.b_cond(HS, past);
            self.a.ldr_x_uxtw3(X9, out, at);
            self.a.mov_imm64(X30, crate::value::PACK_EMPTY);
            self.a.cmp_reg_x(X9, X30);
            self.a.b_cond(EQ, hole);
        }
        match self.graph.node(value).rep {
            Rep::I32 => {
                // An int32's word is its f64 bits (SCVTF), which keep every kind bit while INT32
                // holds; the cold path takes the rest (`emit_word_number_kind_unfrozen`'s rule).
                let kinds = self.a.new_label();
                let store = self.a.new_label();
                self.a.ldur_w(X9, out, flags);
                let int32 = asm::logical_imm_w(KIND_INT32 | KIND_UNWRITABLE)
                    .expect("INT32, FROZEN and COW are one run of bits");
                self.a.logic_imm_w(0, X17, X9, int32);
                self.a.cmp_imm_w(X17, KIND_INT32);
                self.a.b_cond(NE, kinds);
                self.a.bind(store);
                let int = self.gp(value, X9);
                self.a.scvtf_d_w(D31, int);
                let at = self.gp(index, X17);
                self.a.str_d_uxtw3(D31, out, at);
                self.cold.push(Cold::ElementKinds {
                    label: kinds,
                    out,
                    resume: store,
                    unwritable,
                });
            }
            Rep::F64 => {
                let d = self.fp(value, D30);
                emit_word_number_kind_unfrozen(
                    &mut self.a,
                    layout,
                    out,
                    d,
                    X9,
                    X17,
                    X30,
                    D31,
                    unwritable,
                );
                // Its NaN canonicalized, as F64ToTagged boxes it: any other NaN reads as a tag.
                self.a.fmov_x_d(X9, d);
                self.a.fcmp(d, d);
                self.a.movz(X30, 0x7ff8, 3); // PACK_CANON_NAN
                self.a.csel_x(X9, X30, X9, VS);
                let at = self.gp(index, X17);
                self.a.str_x_uxtw3(X9, out, at);
            }
            _ => self.lower_store_word(n, out, unwritable, wrong, done),
        }
        self.a.bind(done);
        self.cold.push(Cold::Unwritable {
            label: unwritable,
            node: n,
            out,
            retry,
            exit: wrong,
        });
        self.gp_done(n, out);
        Ok(())
    }

    /// Whether `v`, a word, is never the hole marker (an element store's value would make a
    /// hole of it, `emit_word_value_guard`): a boxed scalar or a constant other than it.
    fn never_hole(&self, v: NodeIx) -> bool {
        match self.graph.node(v).op {
            NodeOp::ConstTagged(bits) => bits != crate::value::PACK_EMPTY,
            NodeOp::F64ToTagged | NodeOp::I32ToTagged | NodeOp::BoolToTagged => true,
            _ => false,
        }
    }

    /// StoreElement's store of a word (the value input): a Number takes the f64 store's kind
    /// rule and its word as it is, any other word clears NUMBERS and INT32, which keeps the kind
    /// bits exact (the replaced word was no hole and it is none: KIND_SCANNED stays), and, unless
    /// it is a scalar, marks the owner's card and, below a large block's traced length, its run's
    /// dirty bit by a pure call; the template's ElemStore (`emit_word_store_body`) declines such a
    /// block to its helper. The hole marker as the value exits (the template's value guard).
    fn lower_store_word(
        &mut self,
        n: NodeIx,
        out: u32,
        unwritable: usize,
        wrong: usize,
        done: usize,
    ) {
        use crate::value::word_block::{KIND_INT32, KIND_NUMBERS, KIND_UNWRITABLE, REMEMBER_MIN};
        let node = self.graph.node(n);
        let (cell, index, value) = (node.inputs[0], node.inputs[2], node.inputs[3]);
        let layout = self.layout;
        let flags = -(layout.words_flags as i32);
        // A Boolean's word or a constant's tag is no Number (value.rs: a word is a Number unless
        // its bits 48-62 are 0x7ff9 or more).
        let known_word = match self.graph.node(value).op {
            NodeOp::ConstTagged(bits) => (bits >> 48) as u16 & 0x7fff >= 0x7ff9,
            NodeOp::BoolToTagged => true,
            _ => false,
        };
        if !known_word {
            let number = self.a.new_label();
            let tagged = self.a.new_label();
            let reference = self.a.new_label();
            let word = self.gp(value, X9);
            // A Number's f64 bits compare ordered; of the NaN patterns only the canonical NaN is
            // a Number (CheckNumber's test).
            self.a.fmov_d_x(D30, word);
            self.a.fcmp(D30, D30);
            self.a.b_cond(VS, tagged);
            self.a.bind(number);
            emit_word_number_kind_unfrozen(
                &mut self.a,
                layout,
                out,
                D30,
                X9,
                X17,
                X30,
                D31,
                unwritable,
            );
            self.a.fmov_x_d(X9, D30);
            let at = self.gp(index, X17);
            self.a.str_x_uxtw3(X9, out, at);
            self.a.b(done);
            self.a.bind(tagged);
            let word = self.gp(value, X9);
            self.a.movz(X30, 0x7ff8, 3); // PACK_CANON_NAN
            self.a.cmp_reg_x(word, X30);
            self.a.b_cond(EQ, number);
            self.a.bind(reference);
        }
        let word = self.gp(value, X9);
        if !self.never_hole(value) {
            emit_word_value_guard(&mut self.a, word, X30, X17, wrong);
        }
        let imm = |bits| asm::logical_imm_w(bits).expect("a run of kind bits");
        self.a.ldur_w(X30, out, flags);
        self.a.logic_imm_w(0, X17, X30, imm(KIND_UNWRITABLE));
        self.a.cbnz(X17, false, unwritable);
        self.a
            .logic_imm_w(0, X30, X30, imm(!(KIND_NUMBERS | KIND_INT32)));
        self.a.stur_w(X30, out, flags);
        let at = self.gp(index, X17);
        self.a.str_x_uxtw3(word, out, at);
        if !self.scalar_word(value) {
            // The owner's card, before any safepoint, as every generated store of a reference
            // marks it (`emit_card_mark`); a large block's dirty bit after it.
            let owner = self.gp(cell, X30);
            emit_card_mark(&mut self.a, owner, X30, X17);
            let remember = self.a.new_label();
            self.a.ldur_w(X30, out, -(layout.words_capacity as i32));
            self.a.ldur_w(X17, out, -(layout.words_shifted as i32));
            self.a.add_shifted(X30, X30, X17, 0);
            self.a.cmp_imm_x(X30, REMEMBER_MIN as u32);
            self.a.b_cond(HI, remember);
            self.cold.push(Cold::Remember {
                label: remember,
                node: n,
                out,
                resume: done,
            });
        }
    }

    /// StoreElement(Create)'s cold path, from a hole below the used words or an index at or past
    /// them, `out` the words: an Array's hole fill or append, as the template's in-line set does
    /// it (`emit_word_create`, whose checks and commit this follows on the tier's registers), the
    /// receiver's kind tag tested too (the site may have seen ordinary objects as well). Every
    /// check precedes the first write but growth, which changes nothing a JavaScript observer
    /// sees. Then `done`.
    fn emit_element_create(&mut self, n: NodeIx, out: u32, done: usize, exits: CreateExits) {
        use crate::value::{FLAG_ELEM_MODE, FLAG_EXTENSIBLE, FLAG_HAS_FAR, FLAG_PROTO};
        let node = self.graph.node(n);
        let (cell, index, value) = (node.inputs[0], node.inputs[2], node.inputs[3]);
        let layout = self.layout;
        let used = -(layout.words_used as i32);
        let keep = caller_saved(out);
        // 1. An extensible Array that is no marked prototype (whose structural changes must move
        //    the prototype epoch), with no index among its named entries.
        let c = self.gp(cell, X17);
        self.a
            .ldrb_imm(X9, c, (layout.obj_from_cell + layout.obj_kind) as u32);
        self.a.cmp_imm_w(X9, crate::value::KIND_ARRAY_TAG as u32);
        self.a.b_cond(NE, exits.wrong);
        self.a
            .ldrb_imm(X9, c, (layout.obj_from_cell + layout.obj_flags) as u32);
        guard_flags(
            &mut self.a,
            X9,
            FLAG_EXTENSIBLE | FLAG_PROTO | FLAG_ELEM_MODE | FLAG_HAS_FAR,
            FLAG_EXTENSIBLE | FLAG_ELEM_MODE,
            exits.wrong,
        );
        // 2. Its prototype the realm's Array.prototype, and the protector's proof live: no index
        //    on the chain intercepts the creation (OrdinarySet).
        self.emit_no_inherited_elements(n, true, &keep, exits.wrong);
        // 3. A value that would be a hole.
        if self.graph.node(value).rep == Rep::Tagged && !self.never_hole(value) {
            let word = self.gp(value, X17);
            emit_word_value_guard(&mut self.a, word, X30, X9, exits.wrong);
        }
        // 4. A hole below the used words, or the next word; past it, a gap of holes.
        let fill = self.a.new_label();
        self.a.ldur_w(X9, out, used);
        let at = self.gp(index, X17);
        self.a.cmp_reg_w(at, X9);
        self.a.b_cond(LO, fill);
        self.a.b_cond(NE, exits.gap);
        // 5. An append: `length` (slot 0, an Array's first entry, a Number) writable where the
        //    index reaches it.
        let below = self.a.new_label();
        let c = self.gp(cell, X30);
        self.a.add_imm(X30, c, layout.obj_from_cell as u32);
        first_slot_addr(&mut self.a, layout, X9, X30);
        self.a.ldr_d_imm(D31, X9, 0);
        let at = self.gp(index, X17);
        self.a.ucvtf_d_w(D30, at);
        self.a.fcmp(D30, D31);
        self.a.b_cond(LO, below);
        // XZR is slot index 0.
        guard_slot_writable_data(&mut self.a, layout, X30, 31, X9, X17, exits.wrong);
        self.a.bind(below);
        // 6. Room for it: a full block, or the static empty one, grows by a pure call.
        let room = self.a.new_label();
        self.a.ldur_w(X9, out, used);
        self.a.ldur_w(X17, out, -(layout.words_capacity as i32));
        self.a.cmp_reg_w(X9, X17);
        self.a.b_cond(LO, room);
        let c = self.gp(cell, X17);
        let saved = self.save_live_caller_saved(n);
        self.a.mov(0, c);
        self.a.mov_imm64(X16, grow_elements as *const () as u64);
        self.a.blr(X16);
        self.a.mov(X9, 0);
        self.restore_live_caller_saved(&saved);
        self.a.cbz(X9, true, exits.grow);
        self.a.mov(out, X9);
        self.a.bind(room);
        // 7. The commit: the word and the kind bits it keeps (an append keeps NO_HOLES and
        //    KIND_SCANNED: no cleared fact becomes true), the owner's card for an edge (an append
        //    lies past the traced length, so no run's bit), the used length, and `length` where
        //    the index reached it.
        self.emit_created_word(n, out, false);
        self.a.ldur_w(X9, out, used);
        self.a.add_imm(X9, X9, 1);
        self.a.stur_w(X9, out, used);
        let kept = self.a.new_label();
        let c = self.gp(cell, X30);
        self.a.add_imm(X30, c, layout.obj_from_cell as u32);
        first_slot_addr(&mut self.a, layout, X17, X30);
        self.a.ldr_d_imm(D31, X17, 0);
        let at = self.gp(index, X30);
        self.a.ucvtf_d_w(D30, at);
        self.a.fcmp(D30, D31);
        self.a.b_cond(LO, kept);
        self.a.ucvtf_d_w(D31, X9);
        self.a.str_d_imm(D31, X17, 0);
        self.a.bind(kept);
        self.a.b(done);
        // 8. A hole fill, below `length` already: the word, the kind bits it keeps less
        //    KIND_SCANNED (NO_HOLES may hold again), the card and the run's bit for an edge.
        self.a.bind(fill);
        self.emit_created_word(n, out, true);
        self.a.b(done);
    }

    /// The commit of an Array's created element (`emit_element_create`): StoreElement's value
    /// (input 3) written at its index (input 2) of `out`, the kind bits ANDed with those the value
    /// keeps (and KIND_SCANNED cleared for a hole `fill`), and for a word that is no Number the
    /// owner's card and, in a hole fill below a large block's traced length, its run's dirty bit
    /// by a pure call.
    fn emit_created_word(&mut self, n: NodeIx, out: u32, fill: bool) {
        use crate::value::word_block::{KIND_INT32, KIND_NUMBERS, KIND_SCANNED, REMEMBER_MIN};
        let node = self.graph.node(n);
        let (cell, index, value) = (node.inputs[0], node.inputs[2], node.inputs[3]);
        let layout = self.layout;
        let flags = -(layout.words_flags as i32);
        let imm = |bits| asm::logical_imm_w(bits).expect("a run of kind bits");
        let rep = self.graph.node(value).rep;
        if rep == Rep::I32 {
            // An int32 breaks no kind bit.
            let int = self.gp(value, X9);
            self.a.scvtf_d_w(D31, int);
            let at = self.gp(index, X17);
            self.a.str_d_uxtw3(D31, out, at);
            if fill {
                self.a.ldur_w(X9, out, flags);
                self.a.logic_imm_w(0, X9, X9, imm(!KIND_SCANNED));
                self.a.stur_w(X9, out, flags);
            }
            return;
        }
        // w30 = the kind bits the value keeps: all but INT32 for a Number that is no exact
        // int32 (bit-identical through an i32), all but NUMBERS and INT32 for any other word.
        let exact = self.a.new_label();
        let kinds = self.a.new_label();
        let d = if rep == Rep::F64 {
            self.fp(value, D30)
        } else {
            let other = self.a.new_label();
            let number = self.a.new_label();
            let word = self.gp(value, X9);
            self.a.fmov_d_x(D30, word);
            self.a.fcmp(D30, D30);
            self.a.b_cond(VC, number);
            self.a.movz(X17, 0x7ff8, 3); // PACK_CANON_NAN, a Number
            self.a.cmp_reg_x(word, X17);
            self.a.b_cond(NE, other);
            self.a.bind(number);
            self.a.mov_imm64(X30, u64::from(u32::MAX));
            self.emit_int32_exact(D30, exact);
            self.a.logic_imm_w(0, X30, X30, imm(!KIND_INT32));
            self.a.b(kinds);
            self.a.bind(other);
            self.a
                .mov_imm64(X30, u64::from(!(KIND_NUMBERS | KIND_INT32)));
            self.a.b(kinds);
            D30
        };
        if rep == Rep::F64 {
            self.a.mov_imm64(X30, u64::from(u32::MAX));
            self.emit_int32_exact(d, exact);
            self.a.logic_imm_w(0, X30, X30, imm(!KIND_INT32));
            self.a.b(kinds);
        }
        self.a.bind(exact);
        self.a.bind(kinds);
        // The word: an f64's with its NaN canonicalized (any other NaN reads as a tag).
        let word = if rep == Rep::F64 {
            self.a.fmov_x_d(X9, d);
            self.a.fcmp(d, d);
            self.a.movz(X17, 0x7ff8, 3); // PACK_CANON_NAN
            self.a.csel_x(X9, X17, X9, VS);
            X9
        } else {
            self.gp(value, X9)
        };
        let at = self.gp(index, X17);
        self.a.str_x_uxtw3(word, out, at);
        self.a.ldur_w(X17, out, flags);
        self.a.logic_w(0, X17, X17, X30);
        if fill {
            self.a.logic_imm_w(0, X17, X17, imm(!KIND_SCANNED));
        }
        self.a.stur_w(X17, out, flags);
        if rep == Rep::Tagged && !self.scalar_word(value) {
            // A Number kept NUMBERS: no edge, no card.
            let number = self.a.new_label();
            self.a.tbnz(X30, KIND_NUMBERS.trailing_zeros(), number);
            let owner = self.gp(cell, X9);
            emit_card_mark(&mut self.a, owner, X9, X17);
            if fill {
                // A hole of a large block may lie below its traced length.
                self.a.ldur_w(X9, out, -(layout.words_capacity as i32));
                self.a.ldur_w(X17, out, -(layout.words_shifted as i32));
                self.a.add_shifted(X9, X9, X17, 0);
                self.a.cmp_imm_x(X9, REMEMBER_MIN as u32);
                self.a.b_cond(LS, number);
                let at = self.gp(index, X17);
                self.a.mov(X9, out);
                if at != X17 {
                    self.a.mov_w(X17, at);
                }
                let keep = caller_saved(out);
                let saved = self.save_live_caller_saved_and(n, &keep);
                self.a.mov(0, X9);
                self.a.mov_w(1, X17);
                self.a.mov_imm64(X16, remember_element as *const () as u64);
                self.a.blr(X16);
                self.restore_live_caller_saved(&saved);
            }
            self.a.bind(number);
        }
    }

    /// Branch to `exact` when the f64 in `d` is an exact int32 (bit-identical through an i32, so
    /// not -0), as `emit_word_number_kind` tests it; clobbers x9, x17 and d31.
    fn emit_int32_exact(&mut self, d: u32, exact: usize) {
        self.a.fcvtzs_w_d(X9, d);
        self.a.scvtf_d_w(D31, X9);
        self.a.fmov_x_d(X9, D31);
        self.a.fmov_x_d(X17, d);
        self.a.cmp_reg_x(X9, X17);
        self.a.b_cond(EQ, exact);
    }

    /// Branch to `fail` unless no element of `node`'s receiver (its cell, the node's first input)
    /// can come from its prototype chain at an index its words do not hold: no index among its
    /// named entries (HAS_FAR clear), its prototype the active realm's Array.prototype (or, unless
    /// `array_only`, Object.prototype), and the elements protector's proof live at the current
    /// prototype epoch: Maglev's CanTreatHoleAsUndefined (maglev-graph-builder.cc:3993-4008), whose
    /// prototype maps V8's map check implies and TRust reads here, with the protector read at run
    /// time until dependencies exist, as the template's append reads it (`emit_word_create`). A
    /// lapsed proof is renewed by a pure call (`renew_elements_protector`) where the template
    /// takes its helper. Clobbers x9, x17 and x30 (and, across the call, the caller-saved
    /// registers but the live ones and `keep`).
    fn emit_no_inherited_elements(
        &mut self,
        node: NodeIx,
        array_only: bool,
        keep: &[u32],
        fail: usize,
    ) {
        use crate::value::{FLAG_HAS_FAR, FLAG_IC_PLAIN};
        let layout = self.layout;
        let cell = self.gp(self.graph.node(node).inputs[0], X17);
        self.a
            .ldrb_imm(X9, cell, (layout.obj_from_cell + layout.obj_flags) as u32);
        self.a.tbnz(X9, flag_bit(FLAG_HAS_FAR), fail);
        // IC_PLAIN again, from the same byte: the element check's proof is a value fact under the
        // contract that no object JavaScript has seen loses it (ElemKind), and this is where a
        // breach of it would read undefined for an index a side table answers (a host's indexed
        // properties installed on an object that was not fresh).
        self.a.tbz(X9, flag_bit(FLAG_IC_PLAIN), fail);
        self.a
            .ldr_imm(X9, cell, (layout.obj_from_cell + layout.obj_proto) as u32);
        self.a
            .ldr_imm(X30, 19, std::mem::offset_of!(JitCtx, interp) as u32);
        self.a.ldr_imm(X17, X30, INTERP_ARRAY_PROTO as u32);
        self.a.cmp_reg_x(X9, X17);
        if array_only {
            self.a.b_cond(NE, fail);
        } else {
            let either = self.a.new_label();
            self.a.b_cond(EQ, either);
            self.a.ldr_imm(X17, X30, INTERP_OBJECT_PROTO as u32);
            self.a.cmp_reg_x(X9, X17);
            self.a.b_cond(NE, fail);
            self.a.bind(either);
        }
        // The proof is clean at the live epoch (a cached epoch is never u32::MAX, which no proof
        // accepts), or renewed.
        let renew = self.a.new_label();
        let proven = self.a.new_label();
        self.a.ldrb_imm(X9, X30, INTERP_ELEMS_CLEAN as u32);
        self.a.cbz(X9, false, renew);
        self.a.ldr_w_imm(X9, X30, INTERP_ELEMS_EPOCH as u32);
        self.a
            .mov_imm64(X17, crate::value::proto_epoch_ptr() as usize as u64);
        self.a.ldr_w_imm(X17, X17, 0);
        self.a.cmp_reg_w(X9, X17);
        self.a.b_cond(EQ, proven);
        self.a.bind(renew);
        let saved = self.save_live_caller_saved_and(node, keep);
        self.a.mov(0, 19);
        self.a
            .mov_imm64(X16, renew_elements_protector as *const () as u64);
        self.a.blr(X16);
        self.a.mov_w(X9, 0);
        self.restore_live_caller_saved(&saved);
        self.a.cbz(X9, false, fail);
        self.a.bind(proven);
    }
}

#[path = "a64_names.rs"]
mod names;

#[cfg(test)]
#[path = "a64_tests.rs"]
mod tests;
