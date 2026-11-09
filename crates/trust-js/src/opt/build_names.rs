//! The builder's free names (P3 stage 2, step 4; the plan's "Step 4", its input
//! target/p3/stage2/names-input.txt): every name operation was an effect; a static address
//! (Decision 21's: closure variables, and a fixed activation's captured bindings) is now a walk of
//! parent links and a slot's load, without a guard, as V8's context slots, SpiderMonkey's
//! GetAliasedVar and JSC's GetClosureVar are read.
//!
//! - The running environment (`ctx.env_raw`, [`NodeOp::Env`]) is hoisted to the entry where no
//!   operation of the chunk moves it (`moves_env`); otherwise it is read where a site first needs
//!   it and forgotten at an operation that moves it, an effect. A call never moves its caller's.
//! - A splice's static read (`Op::LoadNameIn`) starts from its pinned [[Environment]], its hops
//!   folded at the compile into the holder record ([`NodeOp::ConstEnv`]; JSC's SkipScope on a
//!   constant base, V8's constant context), alive while the splice's guard holds.
//! - A lexical binding's word is checked not empty ([`NodeOp::CheckInitialized`], exiting with
//!   Tdz), as SpiderMonkey's LexicalCheck and JSC's CheckNotEmpty: once checked, a (record, slot)
//!   is initialized for the rest of the function, since a record's binding never returns to its
//!   TDZ (a loop's fresh bindings are new records, which effects make).
//! - Stores and updates at a static address (StoreNameCached, UpdateNameCached, StoreCap,
//!   StoreCapInit, UpdateCap) are nodes: the old word's TDZ test for a lexical binding, an
//!   update's arithmetic by its value record, the store, and the record's card unless the word is
//!   no reference ([`NodeOp::StoreBinding`]), as Maglev's StoreContextSlot, Warp's SetAliasedVar
//!   and JSC's PutClosureVar.
//! - A global-object property's read (a static global, or a site in global code, whose cache is
//!   in global-object mode) is SpiderMonkey's GetGName (jit/CacheIR.cpp:3467-3482): the search's
//!   environment is the realm's and the global object the compile's ([`NodeOp::CheckEnvIs`],
//!   [`NodeOp::CheckCellIs`], value facts per activation; a splice's guard proved its realm), the
//!   global property epoch holds ([`NodeOp::CheckGlobalEpoch`], a heap fact: an effect makes it
//!   be tested again, as a call does SpiderMonkey's movable guard), and the slot's load
//!   ([`NodeOp::LoadField`] on the global object, a heap fact per location). Each check exits
//!   with WrongName.
//! - A script-level lexical's read (the cache in scope mode on the global environment) takes the
//!   cache's own proof, the plan's Q4 having found that the global declarative record does not
//!   only append: the record is the realm's ([`NodeOp::CheckEnvIs`]), its generation the cache's
//!   ([`NodeOp::CheckEnvGeneration`], a heap fact), and the binding's word at the address the
//!   proof keeps ([`NodeOp::LoadGlobalLexical`]), with no TDZ test.
//! - The exit profile (`Rules::name_lowers`, which the loop prepass reads too): a Tdz flag at a
//!   lexical binding's site takes the effect, whose template throws in place (SpiderMonkey keeps
//!   its checks pinned after such a bailout); a WrongName flag takes the effect where the site's
//!   proof is the discarded code's (its byte in the profile, `NameView::proof`), and checks
//!   again where the epoch moved or the cache refilled since, as V8 and SpiderMonkey recompile
//!   on what changed.
//! - Facts ([`Facts`]), as the exemplars' alias rules have them: parent links are value facts
//!   within an activation (the one later write, class evaluation re-parenting its fresh outer
//!   record, eval.rs's `eval_class_strict`, happens after its heritage and keys ran, when no
//!   running activation's chain holds that record); an immutable binding's value is a value fact
//!   once initialized (V8's loaded_context_constants_); a mutable one's a heap fact, reloaded
//!   after an effect, as all three reload mutable slots. A store kills a slot's facts on every
//!   record: a splice's folded holder and a chain from the running environment can name one
//!   record.

use super::*;
use crate::bytecode::StaticName;
use crate::fasthash::FastSet;
use crate::opt::snapshot::{Base, NameView, Search};

/// Whether `op` can change the running environment (`ctx.env_raw`): the VM's operations that
/// write it (`run_vm`'s `*env` stores, which `jit_vm_operation` and `jit_scope_op` copy back to
/// the native frame), every one an effect in the tier.
pub(super) fn moves_env(op: &Op) -> bool {
    matches!(
        op,
        Op::PushLex(_)
            | Op::PushCatchLex(_)
            | Op::CloneLex(_)
            | Op::PopEnv
            | Op::PushWith
            | Op::ClassStart(..)
            | Op::ClassHeritage(..)
            | Op::ClassFinish(_)
            | Op::ClassAbort(_)
    )
}

/// The free names' facts a state holds (see the module's doc).
#[derive(Clone, Default)]
pub(super) struct Facts {
    /// The running environment's node, in a chunk whose operations move it (in any other it is
    /// hoisted), until such an operation.
    running: Option<NodeIx>,
    /// Value facts: each record's outer record, by the record's node.
    parents: FastMap<NodeIx, NodeIx>,
    /// Value facts: the lexical bindings known initialized, by (record, slot).
    initialized: FastSet<(NodeIx, u16)>,
    /// Value facts: the values of initialized immutable bindings, by (record, slot).
    constants: FastMap<(NodeIx, u16), NodeIx>,
    /// Heap facts: the values of mutable bindings loaded or stored in this effect epoch.
    values: FastMap<(NodeIx, u16), NodeIx>,
    /// Value facts: the identity checks passed, by the checked node and the address it is.
    checked: FastSet<(NodeIx, u64)>,
    /// A heap fact: the global property epoch checked in this effect epoch.
    epoch: bool,
    /// Heap facts: the global environment's generation checked, by its record's node...
    generations: FastSet<(NodeIx, u32)>,
    /// ...and the global lexicals loaded, by their bindings' addresses.
    lexicals: FastMap<u64, NodeIx>,
}

impl Facts {
    /// An effect ends the heap facts.
    pub(super) fn clear_heap(&mut self) {
        self.values.clear();
        self.epoch = false;
        self.generations.clear();
        self.lexicals.clear();
    }

    /// An operation moved the running environment: the records it reaches are other nodes from
    /// now on, so no fact keyed on the old ones serves.
    pub(super) fn moved(&mut self) {
        *self = Facts::default();
    }

    /// The facts every predecessor holds.
    pub(super) fn meet(preds: &[(BlockIx, State)]) -> Facts {
        let (first, rest) = (&preds[0].1.names, &preds[1..]);
        let others = || rest.iter().map(|(_, st)| &st.names);
        Facts {
            running: first
                .running
                .filter(|&env| others().all(|names| names.running == Some(env))),
            parents: common(&first.parents, others().map(|names| &names.parents)),
            initialized: first
                .initialized
                .iter()
                .filter(|key| others().all(|names| names.initialized.contains(key)))
                .copied()
                .collect(),
            constants: common(&first.constants, others().map(|names| &names.constants)),
            values: common(&first.values, others().map(|names| &names.values)),
            checked: first
                .checked
                .iter()
                .filter(|key| others().all(|names| names.checked.contains(key)))
                .copied()
                .collect(),
            epoch: first.epoch && others().all(|names| names.epoch),
            generations: first
                .generations
                .iter()
                .filter(|key| others().all(|names| names.generations.contains(key)))
                .copied()
                .collect(),
            lexicals: common(&first.lexicals, others().map(|names| &names.lexicals)),
        }
    }

    /// The facts a loop header keeps from its entry's (`meet`): no heap fact (opt/build.rs's
    /// module doc), and no fact at all in a chunk whose operations move its environment, since a
    /// back edge may come from another one (a `for (let ...)` loop's CloneLex).
    pub(super) fn at_header(&self, moves_env: bool) -> Facts {
        if moves_env {
            return Facts::default();
        }
        Facts {
            values: FastMap::default(),
            epoch: false,
            generations: FastSet::default(),
            lexicals: FastMap::default(),
            ..self.clone()
        }
    }
}

/// The entries of `first` every map of `rest` holds alike.
fn common<'a, K: Copy + Eq + std::hash::Hash + 'a>(
    first: &FastMap<K, NodeIx>,
    rest: impl Iterator<Item = &'a FastMap<K, NodeIx>> + Clone,
) -> FastMap<K, NodeIx> {
    first
        .iter()
        .filter(|(key, value)| rest.clone().all(|map| map.get(key) == Some(value)))
        .map(|(&key, &value)| (key, value))
        .collect()
}

impl Rules<'_> {
    /// Whether the free-name operation at `pc` lowers (the loop prepass's question, and the
    /// builder's: see [`Rules`]): a static address reachable in place, unless a Tdz flag sends a
    /// lexical binding's site to the effect, and for an update a value record of Numbers; a global
    /// read, unless WrongName is flagged with the site's proof as it is (see the module's doc).
    pub(super) fn name_lowers(&self, pc: usize) -> bool {
        let snapshot = self.snapshot;
        let view = snapshot.name(pc);
        let reachable = |address: StaticName, base: Base| {
            usize::from(address.slot) < crate::interpreter::MAX_INLINE_SLOTS
                && base != Base::Unknown
        };
        let tdz = |address: StaticName| address.lexical && snapshot.declines(pc, ExitReason::Tdz);
        match (self.ops[pc], view) {
            (
                Op::LoadName(..)
                | Op::LoadNameForCall(..)
                | Op::LoadNameIn(..)
                | Op::LoadNameForCallIn(..)
                | Op::LoadCap(_),
                NameView::Static { address, base },
            ) => reachable(address, base) && !tdz(address),
            (
                Op::LoadName(..)
                | Op::LoadNameForCall(..)
                | Op::LoadNameIn(..)
                | Op::LoadNameForCallIn(..),
                NameView::Global { .. } | NameView::GlobalLexical { .. },
            ) => !self.stale_name(pc, &view),
            (Op::StoreNameCached(..) | Op::StoreCap(_), NameView::Static { address, base }) => {
                address.mutable && reachable(address, base) && !tdz(address)
            }
            (Op::StoreCapInit(_), NameView::Static { address, base }) => reachable(address, base),
            (Op::UpdateNameCached(..) | Op::UpdateCap(..), NameView::Static { address, base }) => {
                address.mutable
                    && reachable(address, base)
                    && !tdz(address)
                    && self.numbers(pc, false).is_some()
                    && !snapshot.declines(pc, ExitReason::NotNumber)
            }
            _ => false,
        }
    }

    /// WrongName flagged at `pc` where the site's proof is the one the discarded code's check
    /// expected (its byte, `NameView::proof`): the effect, as JSC's exit flags stop a
    /// speculation. A proof that differs (the epoch moved, the cache refilled) is checked again.
    fn stale_name(&self, pc: usize, view: &NameView) -> bool {
        self.snapshot.declines(pc, ExitReason::WrongName)
            && self.snapshot.discarded_proof(pc) == view.proof()
    }
}

impl Builder<'_> {
    /// The free-name site at `pc` takes the effect (`Rules::name_lowers`): list the exit
    /// profile's declines that sent it there.
    fn name_declines(&mut self, pc: usize) {
        match self.snapshot.name(pc) {
            NameView::Static { address, .. } if address.lexical => {
                self.declined(pc, ExitReason::Tdz, "the effect, which throws in place");
            }
            view @ (NameView::Global { .. } | NameView::GlobalLexical { .. })
                if self.rules().stale_name(pc, &view) =>
            {
                self.declined(pc, ExitReason::WrongName, "the effect");
            }
            _ => {}
        }
        if matches!(self.ops[pc], Op::UpdateNameCached(..) | Op::UpdateCap(..)) {
            self.declined(pc, ExitReason::NotNumber, "the generic operation");
        }
    }

    /// The running environment: hoisted where no operation of the chunk moves it.
    fn running_env(&mut self, st: &mut State, pc: usize) -> NodeIx {
        if !self.moves_env {
            return self.hoist(Hoisted::Env);
        }
        if let Some(env) = st.names.running {
            return env;
        }
        let env = self.add(NodeOp::Env, &[], Rep::Env, pc);
        st.names.running = Some(env);
        env
    }

    /// The record `hops` parent links from `record`, each link loaded once (a value fact).
    fn walk(&mut self, st: &mut State, mut record: NodeIx, hops: u16, pc: usize) -> NodeIx {
        for _ in 0..hops {
            record = match st.names.parents.get(&record) {
                Some(&parent) => parent,
                None => {
                    let parent = self.add(NodeOp::EnvParent, &[record], Rep::Env, pc);
                    st.names.parents.insert(record, parent);
                    parent
                }
            };
        }
        record
    }

    /// The record that holds the binding of a static address from `base`, None for a splice
    /// whose holder the compile could not fold. A slot past a record's in-place slots is no
    /// planned record's (they never spill): None as well.
    fn static_record(
        &mut self,
        st: &mut State,
        address: StaticName,
        base: Base,
        pc: usize,
    ) -> Option<NodeIx> {
        if usize::from(address.slot) >= crate::interpreter::MAX_INLINE_SLOTS {
            return None;
        }
        match base {
            Base::Running => {
                let env = self.running_env(st, pc);
                Some(self.walk(st, env, address.hops, pc))
            }
            Base::Holder(holder) => Some(self.hoist(Hoisted::ConstEnv(holder as u64))),
            Base::Unknown => None,
        }
    }

    /// The value of the binding at `slot` of `record`: a fact's, or loaded, its TDZ checked where
    /// the binding is lexical and neither `known_initialized` nor proven so.
    fn binding_value(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        record: NodeIx,
        address: StaticName,
        known_initialized: bool,
    ) -> NodeIx {
        let key = (record, address.slot);
        if let Some(&value) = st
            .names
            .constants
            .get(&key)
            .or_else(|| st.names.values.get(&key))
        {
            return value;
        }
        let pc = cx.pc;
        let value = self.add(
            NodeOp::LoadBinding { slot: address.slot },
            &[record],
            Rep::Tagged,
            pc,
        );
        if address.lexical {
            if !known_initialized && !st.names.initialized.contains(&key) {
                let state = self.before(cx, st);
                self.add_exit(
                    NodeOp::CheckInitialized,
                    &[value],
                    Rep::None,
                    pc,
                    state,
                    ExitReason::Tdz,
                );
            }
            st.names.initialized.insert(key);
        }
        if address.mutable {
            st.names.values.insert(key, value);
        } else {
            st.names.constants.insert(key, value);
        }
        value
    }

    /// Store `word` into the binding at `slot` of `record` (a mutable binding, or one being
    /// initialized), the binding's facts the word's from now on: every record's facts of the slot
    /// end first, since a splice's holder and a chain from the running environment can name one
    /// record.
    fn store_binding(
        &mut self,
        st: &mut State,
        record: NodeIx,
        address: StaticName,
        word: NodeIx,
        pc: usize,
    ) {
        self.add(
            NodeOp::StoreBinding { slot: address.slot },
            &[record, word],
            Rep::None,
            pc,
        );
        let key = (record, address.slot);
        st.names.values.retain(|&(_, slot), _| slot != address.slot);
        if address.lexical {
            st.names.initialized.insert(key);
        }
        if address.mutable {
            st.names.values.insert(key, word);
        } else {
            st.names.constants.insert(key, word);
        }
    }

    /// StoreNameCached at a static address (whose binding is mutable:
    /// `Compiler::note_static_store`) and StoreCap, the old word's TDZ tested for a lexical
    /// binding (SetMutableBinding throws in it), and StoreCapInit (`init`: InitializeBinding,
    /// which ends the TDZ): the incoming word stored in place (see [`NodeOp::StoreBinding`]); any
    /// other site, a global's among them, stays the effect.
    pub(super) fn name_store(
        &mut self,
        pc: usize,
        init: bool,
        st: &mut State,
    ) -> Result<(), Refusal> {
        if !self.rules().lowers(pc) {
            self.name_declines(pc);
            return self.effect(pc, st);
        }
        let NameView::Static { address, base } = self.snapshot.name(pc) else {
            unreachable!("the rules lower only a static address's store")
        };
        let record = self
            .static_record(st, address, base, pc)
            .expect("the rules found the record");
        let key = (record, address.slot);
        let known = matches!(self.ops[pc], Op::StoreCap(_)) && self.chunk.jit_cap_initialized(pc);
        if address.lexical
            && !init
            && !known
            && !st.names.initialized.contains(&key)
            && !st.names.values.contains_key(&key)
        {
            // The old word's TDZ, with the store's operand on the stack.
            let mut cx = Before::new(pc);
            let old = self.add(
                NodeOp::LoadBinding { slot: address.slot },
                &[record],
                Rep::Tagged,
                pc,
            );
            let state = self.before(&mut cx, st);
            self.add_exit(
                NodeOp::CheckInitialized,
                &[old],
                Rep::None,
                pc,
                state,
                ExitReason::Tdz,
            );
        }
        let value = st.stack.pop().expect("the stack analysis saw the operand");
        let word = self.tagged(st, value, pc);
        self.store_binding(st, record, address, word, pc);
        Ok(())
    }

    /// UpdateNameCached at a static address and UpdateCap (`++` and `--`) on a binding holding a
    /// Number: its old word (its TDZ tested for a lexical binding), int32 or f64 arithmetic by the
    /// value record as `update_local`'s, and the new word stored (see [`NodeOp::StoreBinding`]);
    /// any other site, or a record that saw another kind of value, stays the effect.
    pub(super) fn name_update(
        &mut self,
        pc: usize,
        kind: UpdKind,
        st: &mut State,
    ) -> Result<(), Refusal> {
        if !self.rules().lowers(pc) {
            self.name_declines(pc);
            return self.effect(pc, st);
        }
        let NameView::Static { address, base } = self.snapshot.name(pc) else {
            unreachable!("the rules lower only a static address's update")
        };
        let values = self
            .rules()
            .numbers(pc, false)
            .expect("the rules found the record's Numbers");
        let record = self
            .static_record(st, address, base, pc)
            .expect("the rules found the record");
        let known = matches!(self.ops[pc], Op::UpdateCap(..)) && self.chunk.jit_cap_initialized(pc);
        let mut cx = Before::new(pc);
        let old = self.binding_value(st, &mut cx, record, address, known);
        if !self.numeric(old) {
            // A constant the binding was given here, which is no Number.
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        let increment = matches!(
            kind,
            UpdKind::PreInc | UpdKind::PostInc | UpdKind::IncDiscard
        );
        let proven = self.known_int32(st, old);
        let int = (proven || (int32_hint(&values) && self.int32_operand(st, old)))
            && !self.declines_int32(pc, proven);
        let (number, new) = match int.then(|| self.int32(st, &mut cx, old)).flatten() {
            Some(x) => {
                let one = self.hoist(Hoisted::I32(1));
                let node = if increment {
                    NodeOp::I32Add
                } else {
                    NodeOp::I32Sub
                };
                let state = self.before(&mut cx, st);
                let new = self.add_exit(node, &[x, one], Rep::I32, pc, state, ExitReason::Overflow);
                (x, new)
            }
            None => {
                let Some(x) = self.number(st, &mut cx, old) else {
                    return Err(Refusal::at(pc, "a Number binding without an f64 form"));
                };
                let one = self.hoist(Hoisted::F64(1f64.to_bits()));
                let node = if increment {
                    NodeOp::F64Add
                } else {
                    NodeOp::F64Sub
                };
                (x, self.add(node, &[x, one], Rep::F64, pc))
            }
        };
        let word = self.tagged(st, new, pc);
        self.store_binding(st, record, address, word, pc);
        match kind {
            UpdKind::PreInc | UpdKind::PreDec => st.stack.push(new),
            UpdKind::PostInc | UpdKind::PostDec => st.stack.push(number),
            UpdKind::IncDiscard | UpdKind::DecDiscard => {}
        }
        Ok(())
    }

    /// LoadName, LoadNameForCall (`for_call`: the constant undefined `this` below the value, as
    /// all three exemplars call a free name; the `with` case never reaches the tier), their
    /// splices' forms, and LoadCap: a static address's binding (see the module's doc); any other
    /// site stays the effect.
    pub(super) fn name_load(
        &mut self,
        pc: usize,
        for_call: bool,
        st: &mut State,
    ) -> Result<(), Refusal> {
        if !self.rules().lowers(pc) {
            self.name_declines(pc);
            return self.effect(pc, st);
        }
        let mut cx = Before::new(pc);
        let value = match self.snapshot.name(pc) {
            NameView::Static { address, base } => {
                let record = self
                    .static_record(st, address, base, pc)
                    .expect("the rules found the record");
                // A captured binding checked earlier in the operation's block (`cap_tdz_known`).
                let known =
                    matches!(self.ops[pc], Op::LoadCap(_)) && self.chunk.jit_cap_initialized(pc);
                self.binding_value(st, &mut cx, record, address, known)
            }
            view @ NameView::Global { .. } => self.global_read(st, &mut cx, view),
            view @ NameView::GlobalLexical { .. } => self.global_lexical_read(st, &mut cx, view),
            _ => unreachable!("the rules lower only these views"),
        };
        if for_call {
            let undefined = self.word(PACK_UNDEFINED);
            st.stack.push(undefined);
        }
        st.stack.push(value);
        Ok(())
    }

    /// An identity check of `value` against `address` [WrongName], unless passed already (a
    /// value fact).
    fn check_identity(&mut self, st: &mut State, cx: &mut Before, op: NodeOp, value: NodeIx) {
        let address = match op {
            NodeOp::CheckEnvIs { env, .. } => env,
            NodeOp::CheckCellIs { cell, .. } => cell,
            _ => unreachable!("an identity check"),
        };
        if !st.names.checked.insert((value, address)) {
            return;
        }
        let state = self.before(cx, st);
        self.add_exit(op, &[value], Rep::None, cx.pc, state, ExitReason::WrongName);
    }

    /// A global-object property's read (see the module's doc), the site's view `view`.
    fn global_read(&mut self, st: &mut State, cx: &mut Before, view: NameView) -> NodeIx {
        let NameView::Global {
            search,
            genv,
            global,
            slot,
            location,
            epoch,
        } = view
        else {
            unreachable!("a global read's view")
        };
        let pc = cx.pc;
        let proof = view.proof();
        if let Search::Running(hops) = search {
            let env = self.running_env(st, pc);
            let record = self.walk(st, env, hops, pc);
            let check = NodeOp::CheckEnvIs {
                env: genv as u64,
                proof,
            };
            self.check_identity(st, cx, check, record);
        }
        let object = self.hoist(Hoisted::GlobalObject);
        let check = NodeOp::CheckCellIs {
            cell: global as u64,
            proof,
        };
        self.check_identity(st, cx, check, object);
        if !st.names.epoch {
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::CheckGlobalEpoch { epoch, proof },
                &[object],
                Rep::None,
                pc,
                state,
                ExitReason::WrongName,
            );
            st.names.epoch = true;
        }
        if let Some(&value) = st.fields.get(&(object, location)) {
            return value;
        }
        let value = self.add(
            NodeOp::LoadField { location, slot },
            &[object],
            Rep::Tagged,
            pc,
        );
        st.fields.insert((object, location), value);
        value
    }

    /// A script-level lexical's read (see the module's doc), the site's view `view`.
    fn global_lexical_read(&mut self, st: &mut State, cx: &mut Before, view: NameView) -> NodeIx {
        let NameView::GlobalLexical {
            search,
            genv,
            generation,
            binding,
        } = view
        else {
            unreachable!("a global lexical's view")
        };
        let pc = cx.pc;
        let proof = view.proof();
        let record = match search {
            Search::Running(hops) => {
                let env = self.running_env(st, pc);
                let record = self.walk(st, env, hops, pc);
                let check = NodeOp::CheckEnvIs {
                    env: genv as u64,
                    proof,
                };
                self.check_identity(st, cx, check, record);
                record
            }
            // A splice's pinned [[Environment]], which its guard keeps.
            Search::Pinned => self.hoist(Hoisted::ConstEnv(genv as u64)),
        };
        if st.names.generations.insert((record, generation)) {
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::CheckEnvGeneration { generation, proof },
                &[record],
                Rep::None,
                pc,
                state,
                ExitReason::WrongName,
            );
        }
        let binding = binding as u64;
        if let Some(&value) = st.names.lexicals.get(&binding) {
            return value;
        }
        let value = self.add(
            NodeOp::LoadGlobalLexical { binding },
            &[record],
            Rep::Tagged,
            pc,
        );
        st.names.lexicals.insert(binding, value);
        value
    }
}
