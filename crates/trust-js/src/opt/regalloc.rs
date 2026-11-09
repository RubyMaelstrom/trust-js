//! The optimizing tier's register allocator (P3 stage 1; the plan's Design point 10 and the
//! "Stage 1" record's back end): Maglev's single forward pass over the scheduled blocks.
//!
//! Live ranges and next uses come first: a dataflow liveness over the graph, frame states'
//! values counting as uses of the nodes that carry them, and each value's range the hull of the
//! positions where it is live in code order. One pass in code order then gives each value its
//! home at its definition: a free register of its class; a callee-saved one (x23-x28, d8-d15) or
//! a spill slot when the value lives across an effect or a call, which clobber every
//! caller-saved register (contracts 4.4: helpers and templates preserve x19-x28 and d8-d15, and
//! no baseline template touches x23-x28 or writes d8-d15). When no register is free, the held
//! value whose next use is the farthest gives up its register and is spilled at its definition,
//! or the new value is when its own next use is farther (Maglev's choice). A home is the value's
//! for its whole range, so an exit's recipe names one place per value; registers and slots free
//! again where ranges end. Phis take their inputs by parallel moves on the edges into their blocks
//! (`a64`). Constants have no home: each use rematerializes them, and a recipe records them as
//! constants.
//!
//! Raw cell pointers and tagged words in callee-saved registers and spill slots are roots: the
//! collector scans the native stack and the callee-saved registers conservatively and resolves
//! interior pointers (contracts 4.4; JSC's design point), so the homes need no maps.

use super::ir::{BlockIx, Graph, NodeIx, NodeOp, Refusal, Rep, SlotSource};

/// Where a value lives for its whole range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Home {
    /// No value (`Rep::None`), or a value nothing reads.
    None,
    /// A constant node: rematerialized at each use.
    Const,
    /// An x register (Tagged, I32, Bool, Cell).
    Gp(u8),
    /// A d register (F64).
    Fp(u8),
    /// A spill slot: its byte offset from the frame area's base (`sp` between nodes).
    Spill(u32),
}

/// A register class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Gp,
    Fp,
}

/// The class a value of `rep` lives in, if it has one.
pub(crate) fn class_of(rep: Rep) -> Option<Class> {
    match rep {
        Rep::Tagged | Rep::I32 | Rep::Bool | Rep::Cell => Some(Class::Gp),
        // An address (P3 stage 2 step 2): in an x register like a cell, and a root only through
        // the owner's cell, which the node that reads the address also takes (Rep::Ptr).
        Rep::Ptr => Some(Class::Gp),
        Rep::F64 => Some(Class::Fp),
        Rep::None => None,
    }
}

/// Whether a node is a constant, which no register holds.
pub(crate) fn rematerializable(op: &NodeOp) -> bool {
    matches!(
        op,
        NodeOp::ConstTagged(_) | NodeOp::ConstF64(_) | NodeOp::ConstI32(_) | NodeOp::ConstBool(_)
    )
}

/// The registers the allocator hands out, by class and by whether calls preserve them.
pub(crate) struct RegisterFile {
    pub(crate) gp_caller: &'static [u8],
    pub(crate) gp_callee: &'static [u8],
    pub(crate) fp_caller: &'static [u8],
    pub(crate) fp_callee: &'static [u8],
}

/// ARM64 (the Stage 1 record's back end): x19 the context, x21 the helper table and x22 the
/// slots stay pinned, as effects need them; x20, the operand top, is set before each effect and
/// allocated to nothing; x9, x16, x17 and x30 are the code generator's scratch registers (x16 and
/// x17 the veneers' too), x18 is unused and x29 keeps the caller's frame pointer. d30 and d31 are
/// the code generator's FP scratch registers, which a spilled or constant F64 input needs.
pub(crate) const ARM64: RegisterFile = RegisterFile {
    gp_caller: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
    gp_callee: &[23, 24, 25, 26, 27, 28],
    fp_caller: &[
        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    ],
    fp_callee: &[8, 9, 10, 11, 12, 13, 14, 15],
};

/// A graph's registers and code order.
#[derive(Debug)]
pub(crate) struct Allocation {
    /// The reachable blocks in code order: reverse postorder from the entry, so every block's
    /// dominators precede it.
    pub(crate) order: Vec<BlockIx>,
    /// Each node's home.
    pub(crate) homes: Vec<Home>,
    /// Each scheduled node's position: its inputs (and its state) are read at `pos`, its value
    /// is defined at `pos + 1`. A phi's position is its block's head.
    pub(crate) pos: Vec<u32>,
    /// Each value's live range `[start, end]`, positions in code order (`start` its definition).
    pub(crate) ranges: Vec<(u32, u32)>,
    /// The spill area's bytes.
    pub(crate) spill_bytes: u32,
    /// The callee-saved x registers some home uses, ascending (the frame saves them).
    pub(crate) callee_saved_gp: Vec<u8>,
    /// Whether a value living across an effect has a d8..d15 home (`a64` then checks that no
    /// shared stub writes those registers).
    pub(crate) fp_callee_across_effect: bool,
}

impl Allocation {
    /// The values live where `node` reads its inputs and its state and still needed there or
    /// after it, which `node` did not define: what a call made at `node` must keep.
    pub(crate) fn live_at(&self, node: NodeIx) -> Vec<NodeIx> {
        let at = self.pos[node.index()];
        (0..self.homes.len())
            .filter(|&v| {
                let (start, end) = self.ranges[v];
                !matches!(self.homes[v], Home::None | Home::Const) && start < at && end >= at
            })
            .map(|v| NodeIx(v as u32))
            .collect()
    }
}

/// The values a node reads where it runs: its inputs (a phi's are read on its edges) and the
/// values of its frame state, which its exit writes.
fn reads(graph: &Graph, node: NodeIx) -> impl Iterator<Item = NodeIx> + '_ {
    let n = graph.node(node);
    let inputs: &[NodeIx] = if n.op == NodeOp::Phi { &[] } else { &n.inputs };
    let state = n.state.map(|s| graph.state(s));
    inputs
        .iter()
        .copied()
        .chain(state.into_iter().flat_map(|s| {
            s.stack
                .iter()
                .copied()
                .chain(s.slots.iter().filter_map(|slot| match slot {
                    SlotSource::Value(v) => Some(*v),
                    _ => None,
                }))
        }))
}

/// A bit set over node indices.
#[derive(Clone, PartialEq, Eq)]
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Self {
        Bits(vec![0; n.div_ceil(64)])
    }
    fn insert(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
    fn remove(&mut self, i: usize) {
        self.0[i / 64] &= !(1 << (i % 64));
    }
    fn union(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= *b;
        }
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(k, &word)| {
            (0..64)
                .filter(move |bit| word & (1 << bit) != 0)
                .map(move |bit| k * 64 + bit)
        })
    }
}

/// Reverse postorder from the entry over `succs`.
fn reverse_postorder(graph: &Graph, entry: BlockIx) -> Vec<BlockIx> {
    let mut seen = vec![false; graph.blocks.len()];
    let mut post = Vec::with_capacity(graph.blocks.len());
    let mut stack = vec![(entry, 0usize)];
    seen[entry.index()] = true;
    while let Some(&mut (block, ref mut next)) = stack.last_mut() {
        let succs = &graph.block(block).succs;
        if let Some(&succ) = succs.get(*next) {
            *next += 1;
            if !seen[succ.index()] {
                seen[succ.index()] = true;
                stack.push((succ, 0));
            }
        } else {
            post.push(block);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// The free registers and spill slots of one class during the pass.
struct Pool {
    caller: Vec<u8>,
    callee: Vec<u8>,
}

impl Pool {
    fn take(list: &mut Vec<u8>) -> Option<u8> {
        // The lowest-numbered free register, for stable code.
        let (index, _) = list.iter().enumerate().min_by_key(|&(_, &r)| r)?;
        Some(list.swap_remove(index))
    }
}

/// Allocate `graph`'s values to `file`'s registers and spill slots.
pub(crate) fn allocate(graph: &Graph, file: &RegisterFile) -> Result<Allocation, Refusal> {
    let entry = graph
        .entry
        .ok_or(Refusal::whole("a graph without an entry block"))?;
    let order = reverse_postorder(graph, entry);
    let n = graph.nodes.len();

    // Positions: a block's head (its phis' definitions at head + 1), then two per body node.
    let mut pos = vec![u32::MAX; n];
    let mut head = vec![u32::MAX; graph.blocks.len()];
    let mut tail = vec![u32::MAX; graph.blocks.len()];
    let mut next = 0u32;
    for &b in &order {
        head[b.index()] = next;
        for &phi in &graph.block(b).phis {
            pos[phi.index()] = next;
        }
        next += 2;
        for &node in &graph.block(b).body {
            pos[node.index()] = next;
            next += 2;
        }
        tail[b.index()] = next - 1;
    }

    // Liveness over the blocks (values that are not constants), to a fixed point.
    let value = |v: NodeIx| {
        let node = graph.node(v);
        node.rep != Rep::None && !rematerializable(&node.op)
    };
    let mut live_in: Vec<Bits> = vec![Bits::new(n); graph.blocks.len()];
    let mut live_out: Vec<Bits> = vec![Bits::new(n); graph.blocks.len()];
    let edge_inputs = |pred: BlockIx, succ: BlockIx| -> Result<Vec<NodeIx>, Refusal> {
        let data = graph.block(succ);
        let mut found = data.preds.iter().enumerate().filter(|&(_, &p)| p == pred);
        let Some((index, _)) = found.next() else {
            return Err(Refusal::whole("an edge its successor does not list"));
        };
        if found.next().is_some() && !data.phis.is_empty() {
            return Err(Refusal::whole("two edges from one block into a merge"));
        }
        data.phis
            .iter()
            .map(|&phi| {
                graph
                    .node(phi)
                    .inputs
                    .get(index)
                    .copied()
                    .ok_or(Refusal::whole(
                        "a phi without an input for each predecessor",
                    ))
            })
            .collect()
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let data = graph.block(b);
            let mut live = Bits::new(n);
            for &s in &data.succs {
                let mut from = live_in[s.index()].clone();
                for &phi in &graph.block(s).phis {
                    from.remove(phi.index());
                }
                live.union(&from);
                for input in edge_inputs(b, s)? {
                    if value(input) {
                        live.insert(input.index());
                    }
                }
            }
            if live != live_out[b.index()] {
                live_out[b.index()] = live.clone();
            }
            for &node in data.body.iter().rev() {
                live.remove(node.index());
                for read in reads(graph, node) {
                    if value(read) {
                        live.insert(read.index());
                    }
                }
            }
            for &phi in &data.phis {
                live.remove(phi.index());
            }
            if live != live_in[b.index()] {
                live_in[b.index()] = live;
                changed = true;
            }
        }
    }
    if live_in[entry.index()].iter().next().is_some() {
        return Err(Refusal::whole("a value read before its definition"));
    }

    // Ranges: the hull of each value's definition, its reads and the blocks it lives out of.
    let mut ranges = vec![(u32::MAX, 0u32); n];
    for &b in &order {
        let data = graph.block(b);
        for &phi in &data.phis {
            ranges[phi.index()] = (head[b.index()] + 1, head[b.index()] + 1);
        }
        for &node in &data.body {
            let p = pos[node.index()];
            ranges[node.index()] = (p + 1, p + 1);
        }
    }
    let extend = |v: NodeIx, at: u32, ranges: &mut Vec<(u32, u32)>| {
        let range = &mut ranges[v.index()];
        range.1 = range.1.max(at);
    };
    for &b in &order {
        let data = graph.block(b);
        for &node in &data.body {
            for read in reads(graph, node) {
                if value(read) {
                    if pos[read.index()] == u32::MAX {
                        return Err(Refusal::whole("a value defined in no reachable block"));
                    }
                    extend(read, pos[node.index()], &mut ranges);
                }
            }
        }
        for v in live_out[b.index()].iter() {
            extend(NodeIx(v as u32), tail[b.index()], &mut ranges);
        }
    }

    // Next uses: each value's reads in code order (a phi's input is read at its edge's block end).
    let mut uses: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &b in &order {
        let data = graph.block(b);
        for &node in &data.body {
            for read in reads(graph, node) {
                if value(read) {
                    uses[read.index()].push(pos[node.index()]);
                }
            }
        }
        for &s in &data.succs {
            for input in edge_inputs(b, s)? {
                if value(input) {
                    uses[input.index()].push(tail[b.index()]);
                }
            }
        }
    }
    for list in &mut uses {
        list.sort_unstable();
        list.dedup();
    }
    // The first read of `v` after `at`, or the end of its range (a read in a loop's next turn).
    let next_use = |v: NodeIx, at: u32| {
        let list = &uses[v.index()];
        list.get(list.partition_point(|&p| p <= at))
            .copied()
            .unwrap_or(ranges[v.index()].1)
    };

    // The positions of effects and calls, which clobber the caller-saved registers.
    let mut calls: Vec<u32> = order
        .iter()
        .flat_map(|&b| graph.block(b).body.iter())
        .filter(|&&node| graph.node(node).op.is_effectful())
        .map(|&node| pos[node.index()])
        .collect();
    calls.sort_unstable();
    let crosses_call = |(start, end): (u32, u32)| {
        // A call at `p` with start < p < end: the value is defined before it and read after it.
        let first = calls.partition_point(|&p| p <= start);
        calls.get(first).is_some_and(|&p| p < end)
    };

    // The pass: a home at each definition, freed where the range ends.
    let mut homes = vec![Home::None; n];
    let mut gp = Pool {
        caller: file.gp_caller.to_vec(),
        callee: file.gp_callee.to_vec(),
    };
    let mut fp = Pool {
        caller: file.fp_caller.to_vec(),
        callee: file.fp_callee.to_vec(),
    };
    let mut free_slots: Vec<u32> = Vec::new();
    let mut spill_bytes = 0u32;
    let mut used_callee_gp: Vec<u8> = Vec::new();
    let mut fp_callee_across_effect = false;
    // (range end, value) of the values that hold a register or a slot.
    let mut active: Vec<(u32, NodeIx)> = Vec::new();
    let is_callee = |file: &RegisterFile, class: Class, r: u8| match class {
        Class::Gp => file.gp_callee.contains(&r),
        Class::Fp => file.fp_callee.contains(&r),
    };
    let release = |home: Home, gp: &mut Pool, fp: &mut Pool, free_slots: &mut Vec<u32>| match home {
        Home::Gp(r) if file.gp_callee.contains(&r) => gp.callee.push(r),
        Home::Gp(r) => gp.caller.push(r),
        Home::Fp(r) if file.fp_callee.contains(&r) => fp.callee.push(r),
        Home::Fp(r) => fp.caller.push(r),
        Home::Spill(offset) => free_slots.push(offset),
        Home::None | Home::Const => {}
    };
    let expire = |active: &mut Vec<(u32, NodeIx)>,
                  upto: u32,
                  homes: &Vec<Home>,
                  gp: &mut Pool,
                  fp: &mut Pool,
                  free_slots: &mut Vec<u32>| {
        active.retain(|&(end, v)| {
            if end <= upto {
                release(homes[v.index()], gp, fp, free_slots);
                false
            } else {
                true
            }
        });
    };
    let mut assign = |v: NodeIx,
                      homes: &mut Vec<Home>,
                      active: &mut Vec<(u32, NodeIx)>,
                      gp: &mut Pool,
                      fp: &mut Pool,
                      free_slots: &mut Vec<u32>,
                      spill_bytes: &mut u32| {
        let node = graph.node(v);
        if rematerializable(&node.op) {
            homes[v.index()] = Home::Const;
            return;
        }
        let Some(class) = class_of(node.rep) else {
            return;
        };
        let range = ranges[v.index()];
        if range.1 == range.0 {
            return; // nothing reads it: the code writes it to a scratch register
        }
        let across = crosses_call(range);
        let pool = match class {
            Class::Gp => &mut *gp,
            Class::Fp => &mut *fp,
        };
        let mut register = if across {
            Pool::take(&mut pool.callee)
        } else {
            Pool::take(&mut pool.caller).or_else(|| Pool::take(&mut pool.callee))
        };
        if register.is_none() {
            // No register is free: as Maglev chooses, the held value whose next use is the
            // farthest gives up its register and is spilled at its definition, unless this
            // value's own next use is farther. Its range began before now, so its slot is a new
            // one, which nothing else has held.
            let def = range.0;
            let fits = |home: Home| match (home, class) {
                (Home::Gp(r), Class::Gp) => !across || file.gp_callee.contains(&r),
                (Home::Fp(r), Class::Fp) => !across || file.fp_callee.contains(&r),
                _ => false,
            };
            let victim = active
                .iter()
                .filter(|&&(_, u)| fits(homes[u.index()]))
                .map(|&(_, u)| (next_use(u, def), u))
                .max();
            if let Some((far, u)) = victim {
                if far > next_use(v, def) {
                    let (Home::Gp(r) | Home::Fp(r)) = homes[u.index()] else {
                        unreachable!("the victim holds a register")
                    };
                    homes[u.index()] = Home::Spill(*spill_bytes);
                    *spill_bytes += 8;
                    register = Some(r);
                }
            }
        }
        let home = match (register, class) {
            (Some(r), Class::Gp) => Home::Gp(r),
            (Some(r), Class::Fp) => Home::Fp(r),
            (None, _) => Home::Spill(free_slots.pop().unwrap_or_else(|| {
                let offset = *spill_bytes;
                *spill_bytes += 8;
                offset
            })),
        };
        if let (Some(r), Class::Gp) = (register, class) {
            if is_callee(file, class, r) && !used_callee_gp.contains(&r) {
                used_callee_gp.push(r);
            }
        }
        if let (Some(r), Class::Fp) = (register, class) {
            if across && is_callee(file, class, r) {
                fp_callee_across_effect = true;
            }
        }
        homes[v.index()] = home;
        active.push((range.1, v));
    };
    for &b in &order {
        let data = graph.block(b);
        expire(
            &mut active,
            head[b.index()],
            &homes,
            &mut gp,
            &mut fp,
            &mut free_slots,
        );
        for &phi in &data.phis {
            assign(
                phi,
                &mut homes,
                &mut active,
                &mut gp,
                &mut fp,
                &mut free_slots,
                &mut spill_bytes,
            );
        }
        for &node in &data.body {
            let p = pos[node.index()];
            // A node that can exit takes its home while everything it reads is still held: its
            // exit may run after it wrote its value (an overflow check), and its recipe reads
            // the values it read.
            if graph.node(node).op.can_exit() {
                assign(
                    node,
                    &mut homes,
                    &mut active,
                    &mut gp,
                    &mut fp,
                    &mut free_slots,
                    &mut spill_bytes,
                );
                expire(&mut active, p, &homes, &mut gp, &mut fp, &mut free_slots);
            } else {
                expire(&mut active, p, &homes, &mut gp, &mut fp, &mut free_slots);
                assign(
                    node,
                    &mut homes,
                    &mut active,
                    &mut gp,
                    &mut fp,
                    &mut free_slots,
                    &mut spill_bytes,
                );
            }
        }
    }
    // Constants that nothing schedules still have no register.
    for (v, node) in graph.nodes.iter().enumerate() {
        if rematerializable(&node.op) {
            homes[v] = Home::Const;
        }
    }
    used_callee_gp.sort_unstable();
    Ok(Allocation {
        order,
        homes,
        pos,
        ranges,
        spill_bytes,
        callee_saved_gp: used_callee_gp,
        fp_callee_across_effect,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::opt::ir::*;

    fn state(g: &mut Graph, pc: u32, stack: &[NodeIx]) -> StateIx {
        g.add_state(FrameState {
            pc,
            slots: vec![SlotSource::Frame; g.n_slots].into(),
            stack: stack.into(),
            handlers: Box::new([]),
        })
    }

    /// No two values whose ranges overlap share a register or a slot, values living across a
    /// call have callee-saved homes or slots, and every read value has a home.
    pub(crate) fn assert_consistent(graph: &Graph, alloc: &Allocation) {
        let calls: Vec<u32> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(v, node)| node.op.is_effectful() && alloc.pos[*v] != u32::MAX)
            .map(|(v, _)| alloc.pos[v])
            .collect();
        let held: Vec<usize> = (0..graph.nodes.len())
            .filter(|&v| matches!(alloc.homes[v], Home::Gp(_) | Home::Fp(_) | Home::Spill(_)))
            .collect();
        for (k, &a) in held.iter().enumerate() {
            let (sa, ea) = alloc.ranges[a];
            if calls.iter().any(|&p| sa < p && p < ea) {
                assert!(
                    match alloc.homes[a] {
                        Home::Gp(r) => ARM64.gp_callee.contains(&r),
                        Home::Fp(r) => ARM64.fp_callee.contains(&r),
                        _ => true,
                    },
                    "value {a} lives across a call in {:?}",
                    alloc.homes[a]
                );
            }
            for &b in &held[k + 1..] {
                let (sb, eb) = alloc.ranges[b];
                if sa <= eb && sb <= ea {
                    assert_ne!(
                        alloc.homes[a], alloc.homes[b],
                        "values {a} {:?} and {b} {:?} overlap",
                        alloc.ranges[a], alloc.ranges[b]
                    );
                }
            }
        }
        for (v, node) in graph.nodes.iter().enumerate() {
            let read = alloc.ranges[v].1 > alloc.ranges[v].0;
            if read && node.rep != Rep::None && alloc.pos[v] != u32::MAX {
                assert_ne!(alloc.homes[v], Home::None, "value {v} is read");
            }
        }
    }

    #[test]
    fn values_get_registers_and_die_where_their_ranges_end() {
        // a + b, then the sum boxed: a and b die at the add, whose result may take a's register.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let b = g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
        let s = state(&mut g, 2, &[a, b]);
        let fa = g.add_node(entry, NodeOp::CheckNumber, &[a], Rep::F64, 2);
        g.node_mut(fa).state = Some(s);
        let fb = g.add_node(entry, NodeOp::CheckNumber, &[b], Rep::F64, 2);
        g.node_mut(fb).state = Some(s);
        let sum = g.add_node(entry, NodeOp::F64Add, &[fa, fb], Rep::F64, 2);
        let boxed = g.add_node(entry, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 2);
        g.add_node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert_eq!(alloc.homes[a.index()], Home::Gp(0));
        assert_eq!(alloc.homes[b.index()], Home::Gp(1));
        // The checks keep both tagged words held while they can exit.
        assert_eq!(alloc.homes[fa.index()], Home::Fp(0));
        assert_eq!(alloc.homes[fb.index()], Home::Fp(1));
        // fa and fb die at the add: its result reuses d0.
        assert_eq!(alloc.homes[sum.index()], Home::Fp(0));
        assert_eq!(alloc.homes[boxed.index()], Home::Gp(0));
        assert_eq!(alloc.spill_bytes, 0);
        assert!(alloc.callee_saved_gp.is_empty());
    }

    #[test]
    fn a_value_live_across_an_effect_gets_a_callee_saved_home() {
        let mut g = Graph::new(1);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let s = state(&mut g, 1, &[a]);
        let call = g.add_node(
            entry,
            NodeOp::Effect {
                pc: 1,
                pops: 0,
                pushes: 1,
            },
            &[],
            Rep::None,
            1,
        );
        g.node_mut(call).state = Some(s);
        let r = g.add_node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 1);
        let same = g.add_node(entry, NodeOp::TaggedSame, &[a, r], Rep::Bool, 2);
        let boxed = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
        g.add_node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert_eq!(alloc.homes[a.index()], Home::Gp(23));
        assert_eq!(alloc.callee_saved_gp, vec![23]);
        // The projection is defined after the effect: a caller-saved register will do.
        assert_eq!(alloc.homes[r.index()], Home::Gp(0));
        assert_eq!(alloc.live_at(call), vec![a]);
    }

    #[test]
    fn values_spill_at_their_definition_when_no_register_is_free() {
        // Sixteen entry slots live to one return: fifteen caller-saved and six callee-saved x
        // registers hold twenty-one; the rest spill.
        let count = 24u16;
        let mut g = Graph::new(count as usize);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let slots: Vec<NodeIx> = (0..count)
            .map(|k| g.add_node(entry, NodeOp::EntrySlot(k), &[], Rep::Tagged, 0))
            .collect();
        let mut acc = slots[0];
        for &v in &slots[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, v], Rep::Bool, 1);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 1);
        }
        // Keep every slot live to the end.
        let s = state(&mut g, 2, &slots);
        let exit = g.add_node(
            entry,
            NodeOp::Exit(ExitReason::Unsupported),
            &[],
            Rep::None,
            2,
        );
        g.node_mut(exit).state = Some(s);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        // The slots, read only at the end, give their registers up to the accumulator's
        // values, each read at once: no accumulator value spills.
        let spilled = slots
            .iter()
            .filter(|v| matches!(alloc.homes[v.index()], Home::Spill(_)))
            .count();
        assert!(spilled >= count as usize - 21, "{spilled} slots spilled");
        assert!(g.nodes.iter().enumerate().all(|(v, node)| {
            node.op != NodeOp::BoolToTagged || matches!(alloc.homes[v], Home::Gp(_) | Home::None)
        }));
        assert!(alloc.spill_bytes >= 8 * spilled as u32);
    }

    #[test]
    fn a_loop_keeps_values_read_inside_it_until_its_back_edge() {
        // entry: n = slot 0; jump header. header: i = phi(0, i'); poll; branch i < n.
        // body: i' = i + 1; jump header. exit: return.
        let mut g = Graph::new(1);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body);
        g.add_edge(header, exit);
        g.add_edge(body, header);
        let n = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let s = state(&mut g, 0, &[]);
        let fn_ = g.add_node(entry, NodeOp::CheckNumber, &[n], Rep::F64, 0);
        g.node_mut(fn_).state = Some(s);
        let zero = g.add_node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::F64, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let lt = g.add_node(header, NodeOp::F64Cmp(Cond::Lt), &[i, fn_], Rep::Bool, 1);
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        let one = g.add_node(body, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 2);
        let next = g.add_node(body, NodeOp::F64Add, &[i, one], Rep::F64, 2);
        g.node_mut(i).inputs.push(next);
        g.add_node(body, NodeOp::Jump(header), &[], Rep::None, 2);
        let boxed = g.add_node(exit, NodeOp::F64ToTagged, &[i], Rep::Tagged, 3);
        g.add_node(exit, NodeOp::Return, &[boxed], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        let (_, n_end) = alloc.ranges[fn_.index()];
        let body_end = alloc.pos[g.block(body).body.last().unwrap().index()] + 1;
        assert!(n_end >= body_end, "n lives through the back edge");
        assert_ne!(alloc.homes[fn_.index()], alloc.homes[i.index()]);
        assert_ne!(alloc.homes[fn_.index()], alloc.homes[next.index()]);
        assert_eq!(alloc.homes[zero.index()], Home::Const);
        assert_eq!(alloc.live_at(poll), vec![fn_, i]);
    }
}
