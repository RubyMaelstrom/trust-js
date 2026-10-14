//! Opt-in replacement of a running ordinary native frame at a hot loop header.
//!
//! ECMA-262 e28783d5 #sec-execution-contexts, #sec-forbodyevaluation,
//! #sec-getvalue, #sec-try-statement and #sec-weakref-processing-model: this
//! continues the existing invocation. It never repeats binding, prefix effects,
//! per-iteration environments, or cleanup, and retains every canonical owner.

use super::{frame, specialization, stack, value_facts, Cfg, JitCtx};
use crate::bytecode::{self, Chunk};
use crate::jit::{JitCompileOutcome, NativeEntryKind, SpFlag};
use crate::value::PackedValue;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::OnceLock;

fn threshold() -> Option<u32> {
    static AT: OnceLock<Option<u32>> = OnceLock::new();
    *AT.get_or_init(|| {
        (std::env::var("TRUST_JS_OPT_JIT_OSR").as_deref() == Ok("1")).then(|| {
            std::env::var("TRUST_JS_OPT_JIT_OSR_AT")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(65_536)
                .max(2)
        })
    })
}

/// A counter is emitted only for this independent experimental switch. Count
/// actual header visits, not process-wide interrupt ticks or enclosing calls.
pub(in crate::jit) fn counter(chunk: &Chunk, pc: usize) -> Option<*const Cell<u32>> {
    let threshold = threshold()?;
    if chunk.jit_ops().len() > 512
        || !bytecode::native_deopt::supported_loop_continuation(chunk)
        || chunk.jit_detailed_feedback_enabled()
        || super::selected(chunk)
    {
        return None;
    }
    let state = chunk.optimizing_loop.get_or_init(|| {
        let mut headers: Vec<_> = chunk
            .jit_ops()
            .iter()
            .enumerate()
            .filter_map(|(pc, op)| crate::jit_ir::jump_target(op).filter(|&target| target <= pc))
            .collect();
        headers.sort_unstable();
        headers.dedup();
        headers.truncate(8);
        Box::new(crate::tiering::OptimizingLoopState::new(headers, threshold))
    });
    state
        .counters
        .iter()
        .find(|(header, _)| *header == pc)
        .map(|(_, count)| count as *const _)
}

/// Failed entry guards retire this continuation alone, never the primary code
/// or its call caches. Already-active native frames/leases keep their mapping.
pub(super) unsafe extern "C" fn miss(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
    if let Some(state) = (&*(*ctx).chunk).optimizing_loop.get() {
        let misses = state.misses.get().saturating_add(1);
        state.misses.set(misses);
        if misses == bytecode::SPECIALIZATION_MISS_LIMIT {
            for (_, counter) in &state.counters {
                counter.set(0);
            }
            state.code.request_retirement();
        }
    }
    bytecode::native_deopt::resume_before(ctx, pc, sp)
}

pub(in crate::jit) unsafe fn enter(raw: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> SpFlag {
    let ctx = &mut *raw;
    let chunk = &*ctx.chunk;
    let keep = SpFlag { sp, flag: 0 };
    let Some(state) = chunk.optimizing_loop.get() else {
        return keep;
    };
    let pc = pc as usize;
    if state.selected.get().is_some_and(|selected| selected != pc)
        || state.misses.get() >= bytecode::SPECIALIZATION_MISS_LIMIT
    {
        return keep;
    }
    ctx.final_sp = sp;
    if !state.attempted.replace(true) {
        let log = std::env::var_os("TRUST_JS_OPT_JIT_LOG").is_some();
        if log {
            eprintln!(
                "[optimizing-loop] observe pc={pc} depth={}",
                sp.offset_from(ctx.stack_base)
            );
        }
        state.selected.set(Some(pc));
        for (_, counter) in &state.counters {
            counter.set(0);
        }
        let Ok(cfg) = Cfg::build(chunk) else {
            return keep;
        };
        if log {
            eprintln!(
                "[optimizing-loop] headers={:?}",
                cfg.loops()
                    .iter()
                    .map(|natural| cfg.blocks()[natural.header.0 as usize].start)
                    .collect::<Vec<_>>()
            );
        }
        let Some(natural) = cfg
            .loops()
            .iter()
            .find(|natural| cfg.blocks()[natural.header.0 as usize].start == pc)
        else {
            return keep;
        };
        if cfg.stack_depth_at(pc) != Some(sp.offset_from(ctx.stack_base) as usize) {
            return keep;
        }
        let Some(stack) = stack::StackPlan::build(chunk, &cfg) else {
            return keep;
        };
        let Some(frame) = frame::FramePlan::build(chunk, &cfg, true) else {
            return keep;
        };
        // Private slots are observed without materializing any Value or invoking
        // a getter. Environment-homed bindings remain in their checked operations.
        let inputs: Vec<_> = frame
            .tracked
            .iter()
            .copied()
            .take(64)
            .filter(|&slot| usize::from(slot) < ctx.n_slots)
            .filter_map(|slot| {
                value_facts::Types::guarded_class(super::sampled_class(
                    &*ctx.slots.add(usize::from(slot)),
                ))
                .map(|class| (slot, class))
            })
            .collect();
        let properties: Vec<_> = natural
            .blocks
            .iter()
            .map(|block| &cfg.blocks()[block.0 as usize])
            .flat_map(|block| block.start..block.end)
            .filter(|&at| super::loop_facts::sample(ctx, at))
            .take(64)
            .map(|at| at as u32)
            .collect();
        if log {
            eprintln!("[optimizing-loop] inputs={inputs:?}");
        }
        if inputs.is_empty() && properties.is_empty() {
            return keep;
        }
        let Some(generic) = value_facts::ValuePlan::at_entry(chunk, &cfg, &stack, pc, &[]) else {
            if log {
                eprintln!("[optimizing-loop] generic value proof unavailable");
            }
            return keep;
        };
        let Some(specialized) = value_facts::ValuePlan::at_entry(chunk, &cfg, &stack, pc, &inputs)
        else {
            if log {
                eprintln!("[optimizing-loop] guarded value proof unavailable");
            }
            return keep;
        };
        let removed: u32 = natural
            .blocks
            .iter()
            .map(|block| &cfg.blocks()[block.0 as usize])
            .flat_map(|block| block.start..block.end)
            .map(|at| {
                specialization::tests(&chunk.jit_ops()[at], generic.at[at]).saturating_sub(
                    specialization::tests(&chunk.jit_ops()[at], specialized.at[at]),
                )
            })
            .sum();
        if log {
            eprintln!("[optimizing-loop] removed={removed}");
        }
        if removed == 0 && properties.is_empty() {
            return keep;
        }
        let Some(permit) = state.code.begin_compile() else {
            return keep;
        };
        let interp = &mut *ctx.interp;
        let values = *interp
            .jit_layout
            .get_or_init(|| crate::value::jit_layout(&interp.object_proto));
        let layout = interp.interp_layout.get();
        let result = super::super::compile_profiled_with(chunk, || {
            let result = super::compile_entry_with_limits(
                chunk,
                &values,
                &layout,
                None,
                chunk
                    .jit_ops()
                    .len()
                    .saturating_mul(48)
                    .saturating_add(super::names::instruction_allowance(chunk))
                    .min(24_576),
                8192,
                Some((pc, &inputs, &properties)),
            );
            if std::env::var_os("TRUST_JS_OPT_JIT_LOG").is_some() {
                match &result {
                    Ok(_) => eprintln!(
                        "[optimizing-loop] pc={pc} ops={} guards={} removed={removed}",
                        chunk.jit_ops().len(),
                        inputs.len()
                    ),
                    Err(error) => eprintln!("[optimizing-loop] pc={pc} declined: {error}"),
                }
            }
            result.ok()
        });
        match result {
            JitCompileOutcome::Compiled(code) => permit.commit(Some(Rc::new(code))),
            _ => {
                permit.commit(None);
                return keep;
            }
        }
    }
    let Some(Some(code)) = state.code.get() else {
        return keep;
    };
    assert_eq!(code.entry_kind, NativeEntryKind::OptimizingContinuation);
    if let Some((_, counter)) = state.counters.iter().find(|(header, _)| *header == pc) {
        counter.set(1);
    }
    #[cfg(test)]
    state.entries.set(state.entries.get() + 1);
    // The lease and body's residency bracket protect this mapping through nested
    // calls/collections. Its terminal result consumes callee handlers exactly once.
    let entry: unsafe extern "C" fn(*mut JitCtx) -> u64 = std::mem::transmute(code.mem);
    let success = entry(raw);
    SpFlag {
        sp: (*raw).final_sp,
        flag: if success == 1 { 1 } else { 2 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bytecode::Tier, value::Callable, Completion, Engine};

    fn child(test: &str) -> bool {
        const CHILD: &str = "TRUST_JS_TEST_OPT_LOOP_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return false;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("jit::optimizing::loop_entry::tests::{test}"),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_OPT_JIT", "0")
            .env("TRUST_JS_OPT_JIT_OSR", "1")
            .env("TRUST_JS_OPT_JIT_OSR_AT", "16")
            .env("TRUST_JS_OPT_JIT_LOG", "1")
            .env("TRUST_JS_INLINE_AT", "0")
            .env_remove("TRUST_JS_OPT_JIT_DEOPT_AT")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        true
    }

    fn engine() -> Engine {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine
    }

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("loop fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn subject(engine: &mut Engine) -> Rc<Chunk> {
        let env = engine.interp.global_env.clone();
        let value = engine
            .interp
            .get_var("subject", &env)
            .ok()
            .expect("subject binding");
        let object = value.as_obj().expect("function").borrow();
        let Callable::User(user) = &object.call else {
            panic!("ordinary function")
        };
        user.func
            .code
            .get()
            .and_then(Option::as_ref)
            .expect("compiled subject")
            .clone()
    }

    fn entered(engine: &mut Engine) -> Rc<Chunk> {
        let chunk = subject(engine);
        let state = chunk.optimizing_loop.get().expect("loop metadata");
        assert!(
            state.entries.get() > 0,
            "fixture must enter optimizing continuation: attempted={} selected={:?} counters={:?}",
            state.attempted.get(),
            state.selected.get(),
            state.counters
        );
        assert!(state.selected.get().is_some_and(|pc| pc > 0));
        chunk
    }

    #[test]
    fn optimizing_loop_continues_one_invocation_without_replaying_prefix() {
        if child("optimizing_loop_continues_one_invocation_without_replaying_prefix") {
            return;
        }
        let mut engine = engine();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var prefixes=0;
            function ping(x){return x}
            function subject(n){prefixes++;var sum=0;for(var k=0;k<n;k++)sum+=ping(k);return sum}
            [subject(100),prefixes].join(':');
        "#
            ),
            "4950:1"
        );
        let chunk = entered(&mut engine);
        let primary = chunk.jit.get().flatten().expect("primary code");
        let optimized = chunk
            .optimizing_loop
            .get()
            .unwrap()
            .code
            .get()
            .flatten()
            .expect("continuation");
        assert!(!Rc::ptr_eq(&primary, &optimized));
        assert_eq!(primary.entry_kind, NativeEntryKind::FreshFrame);
        assert_eq!(
            optimized.entry_kind,
            NativeEntryKind::OptimizingContinuation
        );
        assert_eq!(
            eval(&mut engine, "[subject(60),prefixes].join(':')"),
            "1770:2"
        );
    }

    #[test]
    fn optimizing_loop_guard_miss_keeps_owners_coercion_and_retires_only_continuation() {
        if child("optimizing_loop_guard_miss_keeps_owners_coercion_and_retires_only_continuation") {
            return;
        }
        let mut engine = engine();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var prefixes=0,coercions=0;
            function ping(x){return x}
            function subject(n,seed,change){prefixes++;var sum=seed;
              for(var k=0;k<n;k++){if(change&&k===20)sum={valueOf(){coercions++;return 10}};sum+=1;ping(k)}
              return sum}
            [subject(100,0,true),prefixes,coercions].join(':');
        "#
            ),
            "90:1:1"
        );
        let chunk = entered(&mut engine);
        let primary = chunk.jit.get().flatten().unwrap();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var lengths=[];for(var r=0;r<9;r++)lengths.push(subject(100,'s',false).length);
            [lengths.join(','),prefixes,coercions].join(':');
        "#
            ),
            "101,101,101,101,101,101,101,101,101:10:1"
        );
        let state = chunk.optimizing_loop.get().unwrap();
        assert_eq!(state.misses.get(), bytecode::SPECIALIZATION_MISS_LIMIT);
        assert!(Rc::ptr_eq(&primary, &chunk.jit.get().flatten().unwrap()));
        assert!(state.counters.iter().all(|(_, counter)| counter.get() == 0));
    }

    #[test]
    fn optimizing_loop_preserves_preexisting_handlers_and_caller_completion() {
        if child("optimizing_loop_preserves_preexisting_handlers_and_caller_completion") {
            return;
        }
        let mut engine = engine();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var prefixes=0,finalized=0,caught=0;
            function ping(x){return x}
            function subject(n,fail){prefixes++;var sum=0;try{
              for(var k=0;k<n;k++){sum+=ping(k);if(fail&&k===40)throw 'stop'}
              return sum;
            }finally{finalized++}}
            function call(n,fail){try{return 7+subject(n,fail)}catch(e){caught++;return e}}
            [call(100,false),call(100,true),prefixes,finalized,caught].join(':');
        "#
            ),
            "4957:stop:2:2:1"
        );
        entered(&mut engine);
    }

    #[test]
    fn optimizing_loop_preserves_iteration_environments_and_live_roots() {
        if child("optimizing_loop_preserves_iteration_environments_and_live_roots") {
            return;
        }
        let mut engine = engine();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function subject(n){var result=[],owner={value:9},count=0;
              for(let k=0;k<n;k++){count+=1;result.push(()=>k);if(k===30)$262.gc()}
              return [result[0](),result[25](),result[79](),owner.value,count].join(':')}
            subject(80);
        "#
            ),
            "0:25:79:9:80"
        );
        entered(&mut engine);
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "subject(80)"), "0:25:79:9:80");
    }

    #[test]
    fn optimizing_loop_retains_operand_owners_and_iterator_close() {
        if child("optimizing_loop_retains_operand_owners_and_iterator_close") {
            return;
        }
        let mut engine = engine();
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var closed=0,prefixes=0;
            var iterable={ [Symbol.iterator](){let k=0;return {
              next(){return {done:k===100,value:{k:k++}}},
              return(){closed++;return {done:true}}
            }}};
            function subject(){prefixes++;var count=0,owner={value:7};
              for(var item of iterable){count+=1;if(count===30)$262.gc();
                if(item.k===70)return count+owner.value}
              return -1}
            [subject(),closed,prefixes].join(':');
            "#
            ),
            "78:1:1"
        );
        entered(&mut engine);
        assert_eq!(
            eval(&mut engine, "[subject(),closed,prefixes].join(':')"),
            "78:2:2"
        );
    }
}
