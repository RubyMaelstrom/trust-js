//! CLIF validation and backend compilation policy.
//!
//! Cranelift 0.136.2 `Context::compile_stencil` and `Context::optimize` run the
//! optional verifier before compilation and repeatedly between internal passes.
//! Its shared settings document this as slower development checking. Release
//! compilation still validates TRust's finalized IR once before backend work;
//! debug/test builds and an explicit diagnostic keep every internal check.
//! This changes no bytecode, ownership, heap, type, or deoptimization guard.

use cranelift_codegen::{isa::TargetIsa, CompiledCode, Context};

pub(super) fn verify_every_pass() -> bool {
    cfg!(debug_assertions)
        || cfg!(test)
        || std::env::var("TRUST_JS_OPT_JIT_VERIFY").as_deref() == Ok("1")
}

pub(super) fn compile<'a>(
    context: &'a mut Context,
    isa: &dyn TargetIsa,
) -> Result<&'a CompiledCode, String> {
    if !isa.flags().enable_verifier() {
        // Preserve the same input validation as compile_stencil's first check.
        // verify() checks types, CFG and dominance without mutating the function.
        context
            .verify(isa)
            .map_err(|errors| format!("IR validation: {errors}"))?;
    }
    context
        .compile(isa, &mut Default::default())
        .map_err(|error| format!("{error:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{
        condcodes::FloatCC, types, AbiParam, Function, InstBuilder, Signature, UserFuncName,
    };
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_codegen::timing::{self, Pass, Profiler};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use std::any::Any;
    use std::cell::Cell;
    use std::rc::Rc;

    fn target(repeated: bool) -> std::sync::Arc<dyn TargetIsa> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").unwrap();
        flags.set("enable_nan_canonicalization", "false").unwrap();
        flags
            .set("enable_verifier", if repeated { "true" } else { "false" })
            .unwrap();
        cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap()
    }

    fn function(isa: &dyn TargetIsa) -> Context {
        let mut signature = Signature::new(isa.default_call_conv());
        signature
            .params
            .extend([AbiParam::new(types::F64), AbiParam::new(types::F64)]);
        signature.returns.push(AbiParam::new(types::F64));
        let mut context = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, 0),
            signature,
        ));
        let mut frontend = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
        let entry = builder.create_block();
        let greater = builder.create_block();
        let other = builder.create_block();
        let join = builder.create_block();
        builder.append_block_param(join, types::F64);
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let left = builder.block_params(entry)[0];
        let right = builder.block_params(entry)[1];
        let condition = builder.ins().fcmp(FloatCC::GreaterThan, left, right);
        builder.ins().brif(condition, greater, &[], other, &[]);
        builder.switch_to_block(greater);
        let result = builder.ins().fmul(left, right);
        builder.ins().jump(join, &[result.into()]);
        builder.switch_to_block(other);
        let result = builder.ins().fadd(left, right);
        builder.ins().jump(join, &[result.into()]);
        builder.switch_to_block(join);
        let result = builder.block_params(join)[0];
        builder.ins().return_(&[result]);
        builder.seal_all_blocks();
        builder.finalize(isa.frontend_config());
        context
    }

    struct Counter(Rc<Cell<usize>>);
    impl Profiler for Counter {
        fn start_pass(&self, pass: Pass) -> Box<dyn Any> {
            if pass == Pass::verifier {
                self.0.set(self.0.get() + 1);
            }
            Box::new(())
        }
    }
    struct RestoreProfiler(Option<Box<dyn Profiler>>);
    impl Drop for RestoreProfiler {
        fn drop(&mut self) {
            let _ = timing::set_thread_profiler(self.0.take().unwrap());
        }
    }

    fn counted_compile(context: &mut Context, isa: &dyn TargetIsa) -> (Vec<u8>, usize) {
        let count = Rc::new(Cell::new(0));
        let _restore = RestoreProfiler(Some(timing::set_thread_profiler(Box::new(Counter(
            count.clone(),
        )))));
        let bytes = compile(context, isa).unwrap().code_buffer().to_vec();
        (bytes, count.get())
    }

    #[test]
    fn input_only_validation_rejects_malformed_ir_before_codegen() {
        let isa = target(false);
        let mut context = function(&*isa);
        // A well-formed frontend graph made malformed at the signature boundary:
        // its return is Float64 while the function contract requires Int64.
        context.func.signature.returns[0] = AbiParam::new(types::I64);
        let error = compile(&mut context, &*isa).unwrap_err();
        assert!(error.starts_with("IR validation:"), "{error}");
        assert!(context.compiled_code().is_none());
    }

    #[test]
    fn input_only_validation_emits_identical_code_with_fewer_verifier_passes() {
        let input = target(false);
        let repeated = target(true);
        let (input_code, input_checks) = counted_compile(&mut function(&*input), &*input);
        let (repeated_code, repeated_checks) =
            counted_compile(&mut function(&*repeated), &*repeated);
        assert_eq!(input_code, repeated_code);
        assert_eq!(input_checks, 1);
        assert!(repeated_checks > input_checks, "{repeated_checks}");
    }
}
