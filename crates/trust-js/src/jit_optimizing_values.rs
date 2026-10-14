//! Whole-function value-category proofs above machine-code lowering.
//!
//! ECMA-262 snapshot e28783d5: ToNumeric/ToNumber, ApplyStringOrNumericBinaryOperator,
//! update expressions, ToBoolean, and IsStrictlyEqual. Facts describe NORMAL results;
//! they never authorize skipping coercion, callbacks, exceptions, or heap effects.
//! Number includes NaN, infinities and BOTH zero signs; it is not an integer/range fact.
//! BigInt, String and Symbol are primitive language values but are OWNERS in our ABI.
//! https://tc39.es/ecma262/#sec-tonumeric
//! https://tc39.es/ecma262/#sec-applystringornumericbinaryoperator
//! https://tc39.es/ecma262/#sec-weakref-processing-model
//!
//! Joins union possible categories. Private words survive only the audited helper write
//! contract; captured/eval-visible bindings and property reads remain unknown. Every
//! completion landing starts unknown, independently of its ordinary CFG predecessors.
//! Opaque operand prefixes supply no facts. Compilation storage/work are bounded, and
//! exhaustion disables this analysis, retaining the complete dynamically checked tier.

use super::{effects, stack};
use crate::bytecode::{Chunk, Op};
use crate::jit_ir::Cfg;
use crate::value::Value;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Types(u16);

impl Types {
    const NONE: Self = Self(0);
    const UNDEFINED: Self = Self(1 << 0);
    const EMPTY: Self = Self(1 << 1);
    pub(super) const NULL: Self = Self(1 << 2);
    const BOOLEAN: Self = Self(1 << 3);
    pub(super) const NUMBER: Self = Self(1 << 4);
    const BIGINT: Self = Self(1 << 5);
    const STRING: Self = Self(1 << 6);
    const SYMBOL: Self = Self(1 << 7);
    pub(super) const OBJECT: Self = Self(1 << 8);
    // Internal packed representations, e.g. a lazy prototype, are never plain owners.
    const INTERNAL: Self = Self(1 << 9);
    pub(super) const ANY: Self = Self((1 << 10) - 1);
    const NUMERIC: Self = Self(Self::NUMBER.0 | Self::BIGINT.0);
    const COPYABLE: Self =
        Self(Self::UNDEFINED.0 | Self::EMPTY.0 | Self::NULL.0 | Self::BOOLEAN.0 | Self::NUMBER.0);

    /// Observations become facts only after the caller emits an entry guard. A mixture
    /// of Number refinements still needs IEEE Number arithmetic, including -0 and NaN.
    pub(super) fn guarded_class(bits: u32) -> Option<Self> {
        use crate::feedback::ValueClass;
        let numbers = ValueClass::NumberInt32.bit() | ValueClass::NumberDouble.bit();
        if bits != 0 && bits & !numbers == 0 {
            Some(Self::NUMBER)
        } else if bits == ValueClass::Boolean.bit() {
            Some(Self::BOOLEAN)
        } else {
            None
        }
    }

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub(super) fn is_number(self) -> bool {
        self == Self::NUMBER
    }

    pub(super) fn is_boolean(self) -> bool {
        self == Self::BOOLEAN
    }

    pub(super) fn is_object(self) -> bool {
        self == Self::OBJECT
    }

    pub(super) fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub(super) fn is_nullish(self) -> bool {
        self != Self::NONE && !self.intersects(Self::ANY.without(Self::UNDEFINED.union(Self::NULL)))
    }

    pub(super) fn is_copyable(self) -> bool {
        self != Self::NONE && !self.intersects(Self::ANY.without(Self::COPYABLE))
    }

    pub(super) fn initialized(self) -> bool {
        self != Self::NONE && !self.intersects(Self::EMPTY)
    }

    fn of(value: &Value) -> Self {
        match value {
            Value::Undefined => Self::UNDEFINED,
            Value::Empty => Self::EMPTY,
            Value::Null => Self::NULL,
            Value::Bool(_) => Self::BOOLEAN,
            Value::Num(_) => Self::NUMBER,
            Value::BigInt(_) => Self::BIGINT,
            Value::Str(_) => Self::STRING,
            Value::Sym(_) => Self::SYMBOL,
            Value::Obj(_) => Self::OBJECT,
        }
    }

    /// Possible successful ToNumeric results, not permission to elide ToNumeric itself.
    fn numeric(self) -> Self {
        let mut result = Self::NONE;
        if self.intersects(Self::COPYABLE.without(Self::EMPTY).union(Self::STRING)) {
            result = result.union(Self::NUMBER);
        }
        if self.intersects(Self::BIGINT) {
            result = result.union(Self::BIGINT);
        }
        if self.intersects(Self::OBJECT.union(Self::INTERNAL).union(Self::EMPTY)) {
            result = result.union(Self::NUMERIC);
        }
        result
    }

    fn numeric_pair(self, other: Self) -> Self {
        // Mixed Number/BigInt throws. An empty intersection cannot justify a guard elision.
        Self(self.numeric().0 & other.numeric().0)
    }

    fn addition(self, other: Self) -> Self {
        if self == Self::STRING || other == Self::STRING {
            return Self::STRING;
        }
        let may_concatenate = Self::STRING.union(Self::OBJECT).union(Self::INTERNAL);
        if self.intersects(may_concatenate) || other.intersects(may_concatenate) {
            self.numeric_pair(other).union(Self::STRING)
        } else {
            self.numeric_pair(other)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Inputs {
    pub(super) top: Types,
    pub(super) second: Types,
    /// The explicitly addressed private local, before any store/update/TDZ operation.
    pub(super) local: Types,
}

impl Default for Inputs {
    fn default() -> Self {
        Self {
            top: Types::ANY,
            second: Types::ANY,
            local: Types::ANY,
        }
    }
}

pub(super) struct ValuePlan {
    pub(super) at: Vec<Inputs>,
}

fn local_operand(op: &Op) -> Option<usize> {
    match op {
        Op::LoadLocal(slot) | Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) => {
            Some(usize::from(*slot))
        }
        _ => None,
    }
}

/// Applies a normal instruction to one reusable block state. Unmodelled results stay
/// unknown, but still kill the concrete helper's private writes. Operand prefixes are
/// preserved by the checked-operation ABI; only its described consumed suffix changes.
fn transfer(
    chunk: &Chunk,
    pc: usize,
    state: &mut [Types],
    slots: usize,
    stack_plan: &stack::StackPlan,
    property_results: &[(u32, Types)],
) -> Option<Inputs> {
    let op = &chunk.jit_ops()[pc];
    let before = stack_plan.at[pc];
    let (pops, pushes) = chunk.jit_stack_effect(pc)?;
    let kept = before.depth.checked_sub(pops)?;
    let input = |from_top| {
        if before.known() >= from_top {
            state[slots + before.depth - from_top]
        } else {
            Types::ANY
        }
    };
    let inputs = Inputs {
        top: input(1),
        second: input(2),
        local: match local_operand(op) {
            Some(slot) => *state.get(..slots)?.get(slot)?,
            None => Types::ANY,
        },
    };
    // Four is the maximum specialized push count (Dup2). Larger destructuring and
    // otherwise unmodelled results use the generic unknown-filled suffix below.
    let mut results = [Types::ANY; 4];
    match op {
        Op::Const(index) => {
            // Chunk pins its constant table throughout this compile-time read.
            results[0] = Types::of(unsafe { &*chunk.jit_const_ptr(*index) });
        }
        Op::Undef | Op::Void => results[0] = Types::UNDEFINED,
        Op::LoadLocal(_) => results[0] = inputs.local.without(Types::EMPTY),
        Op::Dup => results[..2].fill(inputs.top),
        Op::Dup2 => results = [inputs.second, inputs.top, inputs.second, inputs.top],
        Op::Add => results[0] = inputs.second.addition(inputs.top),
        Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Mod
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr => results[0] = inputs.second.numeric_pair(inputs.top),
        Op::UShr | Op::Plus => results[0] = Types::NUMBER,
        Op::Neg | Op::BitNot => results[0] = inputs.top.numeric(),
        Op::UpdateLocal(..) => results[0] = inputs.local.numeric(),
        Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::EqEq
        | Op::NotEq
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::Not
        | Op::TypeofIs(..)
        | Op::DeleteProp(..)
        | Op::DeleteElem(_)
        | Op::DeleteName(_) => results[0] = Types::BOOLEAN,
        Op::Typeof | Op::TypeofName(_) | Op::ToStr => results[0] = Types::STRING,
        Op::JumpIfFalsePeek(_)
        | Op::JumpIfTruePeek(_)
        | Op::JumpIfNotNullishPeek(_)
        | Op::RequireObject
        | Op::DestructureGuard
        | Op::SetProp(..)
        | Op::SetElem
        | Op::SetElemLocal(_) => results[0] = inputs.top,
        Op::GetMethod(..) => results[0] = inputs.top,
        Op::GetMethodElem => results[0] = inputs.second,
        // ToPropKey deliberately also retains internal numeric keys in this ABI;
        // the spec's String/Symbol result type alone would be an unsound machine fact.
        // Other unmodelled helpers likewise keep unknown results.
        _ => {}
    }
    if let Some(&(_, ty)) = property_results
        .iter()
        .find(|&&(site, _)| site as usize == pc)
    {
        // A successful own-data read has guarded the actual result. Any descriptor/type
        // miss finishes in the VM BEFORE the access and cannot reach this normal edge.
        if !matches!(op, Op::GetPropLocal(..) | Op::GetPropThis(..)) || pushes != 1 {
            return None;
        }
        results[0] = ty;
    }
    match effects::checked_writes(op) {
        effects::SlotWrites::None => {}
        effects::SlotWrites::One(slot) => *state[..slots].get_mut(usize::from(slot))? = Types::ANY,
        effects::SlotWrites::Range(start, count) => {
            let start = usize::from(start);
            state[..slots]
                .get_mut(start..start + usize::from(count))?
                .fill(Types::ANY);
        }
        effects::SlotWrites::All => state[..slots].fill(Types::ANY),
    }
    match op {
        Op::StoreLocal(slot) => state[usize::from(*slot)] = inputs.top,
        Op::UpdateLocal(slot, _) => state[usize::from(*slot)] = results[0],
        Op::LoadLocal(slot) => {
            // The normal successor has completed GetBindingValue's TDZ check.
            state[usize::from(*slot)] = results[0];
        }
        Op::Tdz(slot) => state[usize::from(*slot)] = Types::EMPTY,
        Op::ResetSlots(start, count) => {
            let start = usize::from(*start);
            state[..slots]
                .get_mut(start..start + usize::from(*count))?
                .fill(Types::UNDEFINED);
        }
        _ => {}
    }
    let written = state.get_mut(slots + kept..slots + kept + pushes)?;
    written.fill(Types::ANY);
    let modeled = written.len().min(results.len());
    written[..modeled].copy_from_slice(&results[..modeled]);
    Some(inputs)
}

impl ValuePlan {
    pub(super) fn build(chunk: &Chunk, cfg: &Cfg, stack_plan: &stack::StackPlan) -> Option<Self> {
        Self::with_inputs(chunk, cfg, stack_plan, &[], &[])
    }

    pub(super) fn with_inputs(
        chunk: &Chunk,
        cfg: &Cfg,
        stack_plan: &stack::StackPlan,
        inputs: &[(u16, Types)],
        properties: &[(u32, Types)],
    ) -> Option<Self> {
        Self::solve(
            chunk, cfg, stack_plan, inputs, properties, 0, false, 1_000_000, 8_000_000,
        )
    }

    /// An optimizing continuation starts at this live boundary, not at function
    /// entry. Prefix assignments have already happened and cannot prove values
    /// for a later invocation. Only guarded current private slots supply facts.
    pub(super) fn at_entry(
        chunk: &Chunk,
        cfg: &Cfg,
        stack_plan: &stack::StackPlan,
        pc: usize,
        inputs: &[(u16, Types)],
    ) -> Option<Self> {
        Self::at_entry_with_properties(chunk, cfg, stack_plan, pc, inputs, &[])
    }

    pub(super) fn at_entry_with_properties(
        chunk: &Chunk,
        cfg: &Cfg,
        stack_plan: &stack::StackPlan,
        pc: usize,
        inputs: &[(u16, Types)],
        properties: &[(u32, Types)],
    ) -> Option<Self> {
        Self::solve(
            chunk, cfg, stack_plan, inputs, properties, pc, true, 1_000_000, 8_000_000,
        )
    }

    #[cfg(test)]
    fn with_budget(
        chunk: &Chunk,
        cfg: &Cfg,
        stack_plan: &stack::StackPlan,
        cells_limit: usize,
        work: usize,
    ) -> Option<Self> {
        Self::solve(
            chunk,
            cfg,
            stack_plan,
            &[],
            &[],
            0,
            false,
            cells_limit,
            work,
        )
    }

    fn solve(
        chunk: &Chunk,
        cfg: &Cfg,
        stack_plan: &stack::StackPlan,
        inputs: &[(u16, Types)],
        properties: &[(u32, Types)],
        entry_pc: usize,
        continuation: bool,
        cells_limit: usize,
        mut work: usize,
    ) -> Option<Self> {
        let ops = chunk.jit_ops();
        let slots = chunk.jit_frame().1;
        let width = slots.checked_add(stack_plan.capacity)?;
        let blocks = cfg.blocks();
        // A completion landing at PC 0 bypasses the native entry guard. It cannot inherit
        // entry-only facts (ordinary backedges still merge their actual predecessor facts).
        if !inputs.is_empty()
            && (cfg
                .handler_roots()
                .iter()
                .any(|root| blocks[root.target.0 as usize].start == entry_pc)
                || ops.iter().any(
                    |op| matches!(op, Op::AbruptJump(target, _) if *target as usize == entry_pc),
                ))
        {
            return None;
        }
        let cells = width.checked_mul(blocks.len())?;
        if cells > cells_limit {
            return None;
        }
        // Block-entry facts, not a full frame clone at every opcode. The retained plan
        // contains only three queried inputs per operation; all solver storage dies here.
        let mut entries = vec![Types::NONE; cells];
        let mut reached = vec![false; blocks.len()];
        let mut queued = vec![false; blocks.len()];
        let mut roots = vec![false; blocks.len()];
        roots[cfg.block_at(entry_pc)?.0 as usize] = true;
        for root in cfg.handler_roots() {
            roots[root.target.0 as usize] = true;
        }
        for (index, block) in blocks.iter().enumerate() {
            // Only the selected entry and real completion roots are external.
            // Treating every loop header as an ANY entry destroys facts proved
            // by its actual predecessors, including the guarded continuation.
            if block.stack_in.is_some()
                && block.predecessors.is_empty()
                && (!continuation || block.start != 0)
            {
                roots[index] = true;
            }
        }
        for op in ops {
            if let Op::AbruptJump(target, _) = op {
                if let Some(block) = cfg.block_at(*target as usize) {
                    roots[block.0 as usize] = true;
                }
            }
        }
        let mut queue = VecDeque::new();
        for (index, root) in roots.into_iter().enumerate() {
            if root {
                entries[index * width..(index + 1) * width].fill(Types::ANY);
                reached[index] = true;
                queued[index] = true;
                queue.push_back(index);
            }
        }
        let entry = cfg.block_at(entry_pc)?.0 as usize * width;
        for &(slot, ty) in inputs {
            if (!continuation && usize::from(slot) >= chunk.jit_frame().0)
                || usize::from(slot) >= slots
            {
                return None;
            }
            entries[entry + usize::from(slot)] = ty;
        }
        let mut scratch = vec![Types::ANY; width];
        let mut at = vec![Inputs::default(); ops.len()];
        while let Some(index) = queue.pop_front() {
            queued[index] = false;
            let block = &blocks[index];
            work = work.checked_sub(width + 1)?;
            scratch.copy_from_slice(&entries[index * width..(index + 1) * width]);
            for pc in block.start..block.end {
                // Charge for even the worst full private-word kill, not just one opcode.
                work = work.checked_sub(width + 1)?;
                let floor = stack_plan.at[pc].floor;
                scratch[slots..slots + floor].fill(Types::ANY);
                at[pc] = transfer(chunk, pc, &mut scratch, slots, stack_plan, properties)?;
            }
            // Unlike the ordinary CFG, AbruptJump is not an edge carrying value facts:
            // a finally clause can run first, replacing locals and consuming stack owners.
            let last = block.end - 1;
            let (targets, count) = stack::successors(&ops[last], last);
            for &target in &targets[..count] {
                let Some(target_block) = cfg.block_at(target) else {
                    if target == ops.len() {
                        continue;
                    }
                    return None;
                };
                let target_index = target_block.0 as usize;
                let destination = stack_plan.at[target];
                let used = slots + destination.depth;
                work = work.checked_sub(used + 1)?;
                let target_state = &mut entries[target_index * width..target_index * width + used];
                let mut changed = !reached[target_index];
                reached[target_index] = true;
                for (word, current) in target_state.iter_mut().enumerate() {
                    let incoming = if word >= slots && word < slots + destination.floor {
                        Types::ANY
                    } else {
                        scratch[word]
                    };
                    let joined = current.union(incoming);
                    changed |= *current != joined;
                    *current = joined;
                }
                if changed && !queued[target_index] {
                    queued[target_index] = true;
                    queue.push_back(target_index);
                }
            }
        }
        Some(Self { at })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode;
    use std::rc::Rc;

    fn plan(source: &str) -> (Rc<Chunk>, Cfg, stack::StackPlan, ValuePlan) {
        let statements = crate::parser::parse_script(source, false).ok().unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function fixture")
        };
        let chunk = bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        let facts = ValuePlan::build(&chunk, &cfg, &stack).unwrap();
        (chunk, cfg, stack, facts)
    }

    #[test]
    fn language_primitives_are_not_the_same_as_copyable_machine_words() {
        for value in [
            Types::BIGINT,
            Types::STRING,
            Types::SYMBOL,
            Types::OBJECT,
            Types::INTERNAL,
        ] {
            assert!(!value.is_copyable());
            assert!(!Types::NUMBER.union(value).is_copyable());
        }
        assert!(Types::NUMBER
            .union(Types::BOOLEAN)
            .union(Types::NULL)
            .is_copyable());
        assert!(Types::EMPTY.is_copyable());
        assert!(!Types::EMPTY.initialized());
        assert!(!Types::NONE.is_copyable());
        assert!(!Types::NONE.initialized());
        assert!(!Types::ANY.is_number());
        assert_eq!(Types::NUMBER.numeric_pair(Types::BIGINT), Types::NONE);
        assert_eq!(Types::STRING.numeric(), Types::NUMBER);
        assert_eq!(Types::STRING.addition(Types::BIGINT), Types::STRING);
        assert_eq!(Types::OBJECT.numeric(), Types::NUMERIC);
    }

    #[test]
    fn widening_a_category_set_never_strengthens_a_conversion_proof() {
        // Enumerate the finite lattice, not only hand-selected Number cases. Every
        // one-bit widening must retain all possible old results, in either operand.
        // This protects the worklist's monotone fixed-point contract as rules expand.
        for mask in 0..=Types::ANY.0 {
            let before = Types(mask);
            for bit in 0..10 {
                let after = before.union(Types(1 << bit));
                assert_eq!(before.numeric().without(after.numeric()), Types::NONE);
                for other_mask in 0..=Types::ANY.0 {
                    let other = Types(other_mask);
                    assert_eq!(
                        before
                            .numeric_pair(other)
                            .without(after.numeric_pair(other)),
                        Types::NONE
                    );
                    assert_eq!(
                        before.addition(other).without(after.addition(other)),
                        Types::NONE
                    );
                    assert_eq!(
                        other.addition(before).without(other.addition(after)),
                        Types::NONE
                    );
                }
            }
        }
    }

    #[test]
    fn normal_loop_values_remain_numeric_across_private_preserving_calls() {
        let (chunk, _, _, facts) = plan(
            "function subject(n){var sum=0;for(var i=0;i<n;i++){tick();sum=sum+i;}return sum;}",
        );
        let mut additions = 0;
        let mut updates = 0;
        for (pc, op) in chunk.jit_ops().iter().enumerate() {
            match op {
                Op::Add => {
                    additions += 1;
                    assert!(facts.at[pc].top.is_number());
                    assert!(facts.at[pc].second.is_number());
                }
                Op::UpdateLocal(..) => {
                    updates += 1;
                    assert!(facts.at[pc].local.is_number());
                }
                _ => {}
            }
        }
        assert!(additions > 0 && updates > 0);
        assert!(chunk
            .jit_ops()
            .iter()
            .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
    }

    #[test]
    fn diamonds_and_backedges_union_instead_of_remembering_the_first_type() {
        for source in [
            "function subject(flag){var x=1;if(flag)x='s';return x+2;}",
            "function subject(n){var x=1;for(var i=0;i<n;i++){if(i===2)x='s';else x=x+1;}return x+2;}",
        ] {
            let (chunk, _, _, facts) = plan(source);
            let pc = chunk.jit_ops().iter().rposition(|op| matches!(op, Op::Add)).unwrap();
            assert_eq!(facts.at[pc].second, Types::NUMBER.union(Types::STRING));
            assert!(facts.at[pc].top.is_number());
        }
    }

    #[test]
    fn projected_expression_kills_private_facts_and_all_landings_start_unknown() {
        // ClassDefinitionEvaluation evaluates the computed key in the surrounding
        // environment. This is a real projected EvalExpr; assignment patterns now
        // lower directly and must not be mistaken for generic AssignTarget coverage.
        let (chunk, cfg, stack, facts) =
            plan("function subject(){var x=1;try{void class {[x='s'](){}};}catch(e){}return x+2;}");
        assert!(chunk
            .jit_ops()
            .iter()
            .any(|op| matches!(op, Op::EvalExpr(_))));
        let pc = chunk
            .jit_ops()
            .iter()
            .rposition(|op| matches!(op, Op::Add))
            .unwrap();
        assert_eq!(facts.at[pc].second, Types::ANY.without(Types::EMPTY));
        assert!(!cfg.handler_roots().is_empty());
        for root in cfg.handler_roots() {
            let pc = cfg.blocks()[root.target.0 as usize].start;
            assert_eq!(facts.at[pc].top, Types::ANY);
            assert_eq!(facts.at[pc].local, Types::ANY);
        }
        for (pc, state) in stack.at[..chunk.jit_ops().len()].iter().enumerate() {
            if state.known() == 0 {
                assert_eq!(facts.at[pc].top, Types::ANY);
            }
        }
    }

    #[test]
    fn abrupt_jump_cannot_carry_pre_finally_numeric_proofs_to_its_destination() {
        let (chunk, cfg, _, facts) = plan(
            "function subject(){var x=1;outer:while(true){try{break outer;}finally{x='f';}}return x+2;}",
        );
        let mut jumps = 0;
        for op in chunk.jit_ops() {
            if let Op::AbruptJump(target, _) = op {
                jumps += 1;
                let pc = cfg.blocks()[cfg.block_at(*target as usize).unwrap().0 as usize].start;
                assert_eq!(facts.at[pc].local, Types::ANY);
            }
        }
        assert!(jumps > 0);
        let pc = chunk
            .jit_ops()
            .iter()
            .rposition(|op| matches!(op, Op::Add))
            .unwrap();
        assert!(!facts.at[pc].second.is_number());
    }

    #[test]
    fn value_analysis_budget_exhaustion_returns_no_partial_proof() {
        let (chunk, cfg, stack, _) = plan("function subject(n){var x=1;while(n--)x=x+1;return x;}");
        assert!(ValuePlan::with_budget(&chunk, &cfg, &stack, 0, usize::MAX).is_none());
        assert!(ValuePlan::with_budget(&chunk, &cfg, &stack, usize::MAX, 0).is_none());
    }

    #[test]
    fn whole_function_value_proofs_remove_emission_without_changing_numeric_edges() {
        crate::jit::optimizing::TEST_VALUE_PROOFS.with(|count| count.set([0; 4]));
        crate::jit::optimizing::tests::check(
            r#"
            function subject(){
                var sum=0;
                for(var i=0;i<10;i++){tick();sum=sum+i;}
                var minus=-0, quotient=1/minus, bad=0/0;
                var yes=bad!==bad, equal=minus===0;
                return [sum,quotient,yes,equal,minus&&'wrong',!bad,1/(minus+minus)].join('|');
            }
            function tick(){return 7;}
            "#,
            "45|-Infinity|true|true|0|true|-Infinity",
        );
        let counts = crate::jit::optimizing::TEST_VALUE_PROOFS.with(|count| count.get());
        eprintln!(
            "[value-proofs] number operands={}, updates={}, owner sites={}, conditions={}",
            counts[0], counts[1], counts[2], counts[3]
        );
        if std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() == Ok("0") {
            assert_eq!(counts, [0; 4]);
        } else {
            assert!(
                counts.iter().all(|count| *count > 0),
                "must omit emitted checks, not merely compute facts"
            );
        }
    }

    #[test]
    fn value_proofs_keep_bigint_string_and_object_coercion_order() {
        crate::jit::optimizing::tests::check(
            r#"
            function subject(){
                var log=[], x='7', old=x++, integer=3n;
                var prior=integer++;
                var left={valueOf(){log.push('L');return 2n;}};
                var right={valueOf(){log.push('R');return 4n;}};
                var product=left*right;
                try{left+1;}catch(e){log.push(e.name);}
                try{+integer;}catch(e){log.push(e.name);}
                var text='a';
                for(var i=0;i<3;i++)text=text+i;
                return [old,x,prior,integer,product,text,log.join(',')].join('|');
            }
            "#,
            "7|8|3|4|8|a012|L,R,L,TypeError,TypeError",
        );
    }

    #[test]
    fn value_proofs_keep_tdz_and_type_changes_in_exceptional_control_flow() {
        crate::jit::optimizing::tests::check(
            r#"
            function subject(){
                var log=[], x=1;
                try{log.push(later);}catch(e){log.push(e.name);}
                let later=3;
                outer:while(true){try{break outer;}finally{x='f';}}
                log.push(x+2,later);
                for(var i=0;i<4;i++){
                    try {if(i===1)throw 'change';x=x+1;}
                    catch(e){x=20n;}
                    finally {if(i===2)x='after';}
                }
                return log.join('|')+'|'+x;
            }
            "#,
            "ReferenceError|f2|3|after1",
        );
    }

    #[test]
    fn value_proofs_discard_private_facts_after_projected_normal_and_throwing_writes() {
        // ClassDefinitionEvaluation / DefineMethod evaluate keys in source order.
        // The second bridge writes x before throwing; that write is not rolled back.
        // https://tc39.es/ecma262/#sec-runtime-semantics-classdefinitionevaluation
        crate::jit::optimizing::tests::check_real_call(
            r#"
            function subject(){
                var x=1, before=x+2;
                void class {[x='s'](){}};
                var after=x+2;
                try {void class {[x=3](){} [(null).key](){}};} catch(e) {}
                return [before,after,x+2].join('|');
            }
            "#,
            "3|s2|5",
        );
    }
}
