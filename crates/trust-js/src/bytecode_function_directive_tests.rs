//! ECMA-262 e28783d5: Directive Prologues, EvaluateFunctionBody, GeneratorStart,
//! AsyncFunctionStart and AsyncGeneratorStart discard normal body completions.
//! The observable fixtures are checked separately against Node 24.21.0.

use super::*;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("directive fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine, name: &str) -> Rc<Function> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object_handle = value.as_obj().unwrap();
    let object = object_handle.borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func.clone()
}

fn parsed_function(source: &str) -> Rc<Function> {
    let statements =
        crate::parser::parse_script(source, false).unwrap_or_else(|e| panic!("{}", e.message));
    let Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function declaration")
    };
    function.clone()
}

fn strings(chunk: &Chunk) -> Vec<String> {
    chunk
        .consts
        .iter()
        .filter_map(|value| value.as_engine_text().map(std::borrow::Cow::into_owned))
        .collect()
}

fn check(source: &str, observe: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        assert_eq!(eval(&mut engine, observe), expected, "{tier:?}");
    }
}

#[test]
fn function_prefix_eliminates_const_pop_without_mutating_ast_or_strictness() {
    let function = parsed_function("function subject(){'vendor';'use strict';return 7;}");
    assert!(function.is_strict);
    assert_eq!(function.body.len(), 3);
    let chunk = compile(&function).unwrap();
    assert!(chunk.strict);
    assert!(strings(&chunk).is_empty());
    assert_eq!(
        chunk.ops.len(),
        3,
        "Number return and the existing fallthrough epilogue remain"
    );
    assert!(matches!(
        chunk.ops.as_slice(),
        [Op::Const(_), Op::Return, Op::ReturnUndef]
    ));
    assert_eq!(function.body.len(), 3, "source AST must remain intact");

    let fallthrough = parsed_function("function subject(){'vendor';'use strict';}");
    let chunk = compile(&fallthrough).unwrap();
    assert!(strings(&chunk).is_empty());
    assert!(matches!(chunk.ops.as_slice(), [Op::ReturnUndef]));
    for source in [
        "function* subject(){'vendor';'use strict';yield 7;}",
        "async function subject(){'vendor';'use strict';return await 7;}",
        "async function* subject(){'vendor';'use strict';yield await 7;}",
    ] {
        let function = parsed_function(source);
        let chunk = compile(&function).expect("resumable body");
        assert!(chunk.resumable && chunk.strict);
        assert!(strings(&chunk).is_empty(), "{source}");
    }
}

#[test]
fn parser_controls_strictness_and_only_a_pure_leading_prefix_is_skipped() {
    for (source, strict, expected) in [
        (
            "function subject(){`use strict`;'use strict';return 7;}",
            false,
            vec![],
        ),
        (
            "function subject(){'use\\x20strict';return 7;}",
            false,
            vec![],
        ),
        (
            "function subject(){('use strict');'late';return 7;}",
            false,
            vec![],
        ),
        (
            "function subject(){'vendor';;'late';return 7;}",
            false,
            vec!["late"],
        ),
        (
            "function subject(){'vendor';observe();'late';return 7;}",
            false,
            vec!["late"],
        ),
    ] {
        let function = parsed_function(source);
        assert_eq!(function.is_strict, strict, "{source}");
        let chunk = compile(&function).unwrap();
        assert_eq!(strings(&chunk), expected, "{source}");
        assert_eq!(chunk.strict, strict);
    }
    let statements = crate::parser::parse_script("'use strict';", false)
        .unwrap_or_else(|e| panic!("{}", e.message));
    let script = compile_script(&statements, true).unwrap();
    assert_eq!(strings(&script), ["use strict"]);
    let module = compile_module(&statements, &[]).unwrap();
    assert_eq!(
        strings(&module),
        ["use strict"],
        "module policy is unchanged"
    );
}

const ORDINARY: &str = r#"
    var trace=[];
    function strict(a){'vendor';'use strict';arguments[0]=9;return [this===undefined,a,arguments[0]];}
    function escaped(a){'use\x20strict';arguments[0]=9;return [this===globalThis,a];}
    function template(){`use strict`;'use strict';return this===globalThis;}
    function parenthesized(){('use strict');'use strict';return this===globalThis;}
    function sideEffect(){'vendor';`${trace.push('template')}`;'late';return trace.length;}
    function empty(){'vendor';'use strict';}
    var arrow=()=>{'use strict';},concise=()=>'use strict',conciseTemplate=()=>`use strict`;
    function saved(){'vendor';'use strict';return 7;}
    var constructed=Function("'vendor';'use strict';return this===undefined;");
    var result=[strict(1),escaped(1),template(),parenthesized(),sideEffect(),trace,
        empty()===undefined,arrow()===undefined,concise(),conciseTemplate(),constructed(),
        saved.toString()==="function saved(){'vendor';'use strict';return 7;}"];
"#;

#[test]
fn ordinary_functions_keep_strictness_this_arguments_sources_and_effects() {
    check(
        ORDINARY,
        "JSON.stringify(result)",
        r#"[[true,1,9],[true,9],true,true,1,["template"],true,true,"use strict","use strict",true,true]"#,
    );
}

const DEFAULTS_AND_STATIC: &str = r#"
    var trace=[];
    function defaults(value=(trace.push('default'),eval("'use strict'"))){'vendor';return [value,eval("'use strict'")];}
    class Static {
        static { 'vendor';'use strict';trace.push(eval("'use strict'"));this.value='field'; }
        static method(){'vendor';'use strict';return this.value;}
    }
    function managed(){'vendor';'use strict';using resource = {[Symbol.dispose](){trace.push('dispose');}};trace.push('managed');return 13;}
    var result=[defaults(),Static.method(),managed(),trace];
"#;

#[test]
fn defaults_nested_eval_class_static_blocks_and_disposal_keep_their_semantics() {
    check(
        DEFAULTS_AND_STATIC,
        "JSON.stringify(result)",
        r#"[["use strict","use strict"],"field",13,["use strict","default","managed","dispose"]]"#,
    );
    let function =
        parsed_function("function subject(value=eval(\"'use strict'\")){'vendor';return value;}");
    let chunk = compile(&function).unwrap();
    assert!(chunk.prepared_entry);
    assert!(!strings(&chunk).iter().any(|text| text == "vendor"));
}

const COMPLETIONS: &str = r#"
    function direct(){'vendor';'use strict';return [eval("'use strict'"),eval("'vendor';'use strict'"),eval("'use strict';void 0")===undefined];}
    var result=[eval("'use strict'"),(0,eval)("'vendor';'use strict'"),direct(),
        eval("'use strict';;"),eval("'use strict';if(false){}")===undefined];
"#;

#[test]
fn script_and_direct_indirect_eval_keep_string_completion_values() {
    check(
        COMPLETIONS,
        "JSON.stringify(result)",
        r#"["use strict","use strict",["use strict","use strict",true],"use strict",true]"#,
    );
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, "'use strict';"), "use strict");
    }
}

const CONSTRUCTORS: &str = r#"
    var trace=[];
    function Ordinary(){'vendor';'use strict';this.value=7;trace.push(new.target===Ordinary);}
    class Base{constructor(){'base';this.value=2;trace.push('base');}}
    class Derived extends Base{constructor(){'derived';'use strict';super();this.value++;trace.push('derived');}}
    var ordinary=new Ordinary(),derived=new Derived();
    function cleanup(){'vendor';'use strict';try {trace.push('body');return 11;}finally {trace.push('finally');}}
    var result=[ordinary.value,derived.value,cleanup(),trace];
"#;

#[test]
fn constructors_and_finally_keep_observable_work_and_return_rules() {
    check(
        CONSTRUCTORS,
        "JSON.stringify(result)",
        r#"[7,3,11,[true,"base","derived","body","finally"]]"#,
    );
}

const COROUTINES: &str = r#"
    var trace=[],result;
    function* sync(){'vendor';'use strict';try {yield this===undefined;yield 7;}finally {trace.push('sync finally');}}
    async function asynchronous(){'vendor';'use strict';trace.push('async start');await 0;trace.push('async end');}
    async function* streaming(){'vendor';'use strict';try {yield await 9;}finally {trace.push('stream finally');}}
    var iterator=sync(),syncResults=[iterator.next(),iterator.next(),iterator.next()];
    (async function(){
        var value=await asynchronous(),stream=streaming();
        var first=await stream.next(),last=await stream.next();
        result=[syncResults,value===undefined,first,last,trace];
    })();
"#;

#[test]
fn generator_and_async_slices_keep_suspension_cleanup_and_undefined_fallthrough() {
    check(
        COROUTINES,
        "JSON.stringify(result)",
        r#"[[{"value":true,"done":false},{"value":7,"done":false},{"done":true}],true,{"value":9,"done":false},{"done":true},["sync finally","async start","async end","stream finally"]]"#,
    );
}

#[test]
fn live_function_loop_keeps_original_ast_site_and_current_bytecode_pc() {
    let source = "var prefixes=0;function subject(){'vendor';'use strict';prefixes++;let sum=0;for(let k=0;k<3000;k++)sum+=k;return sum+':'+prefixes;}";
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        eval(&mut engine, source);
        let function = function(&mut engine, "subject");
        let site = function.body.iter().find_map(function_osr_site).unwrap();
        TEST_FUNCTION_OSR_ENTRIES.with(|count| count.set(0));
        assert_eq!(eval(&mut engine, "subject()"), "4498500:1");
        assert!(TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()) > 0);
        let runtime = function.runtime();
        let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
        assert!(strings(chunk)
            .iter()
            .all(|value| *value != "vendor" && *value != "use strict"));
        let entry = chunk
            .function_osr_entries
            .iter()
            .find(|entry| entry.site == site)
            .unwrap();
        assert!(entry.pc < chunk.ops.len());
        assert_eq!(
            function.body.len(),
            6,
            "the two prefix expressions remain in the AST"
        );
    }
}

#[test]
fn positioned_operations_keep_original_source_offsets_after_prefix_removal() {
    let function = parsed_function("function subject(){\n'vendor';\n'use strict';\nnull.fail;\n}");
    let Stmt::Expr(Expr::Member { pos, .. }) = &function.body[2] else {
        panic!("positioned member")
    };
    let chunk = compile(&function).unwrap();
    let pc = chunk
        .ops
        .iter()
        .position(|op| matches!(op, Op::GetProp(..)))
        .unwrap();
    assert_eq!(chunk.positions.at(pc), *pos);
    assert!(strings(&chunk).is_empty());
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn prepared_caller_and_actual_inline_splice_skip_both_directive_prefixes() {
    for (tier, strict) in [
        (Tier::Bytecode, false),
        (Tier::Jit, false),
        (Tier::Bytecode, true),
        (Tier::Jit, true),
    ] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let directive = if strict { "'use strict';" } else { "" };
        let source = format!("var trace=[],holder={{bias:2,leaf:function leaf(n){{'leaf directive';{directive}trace.push('leaf');return this.bias+n;}}}};function subject(value){{'caller directive';{directive}var read=()=>value;try {{var r=holder.leaf(value);arguments[0]=value+1;eval('value+=2');return [r,read(),arguments[0]];}}finally {{trace.push('finally');}}}}");
        eval(&mut engine, &source);
        let function = function(&mut engine, "subject");
        let primary = compile(&function).unwrap();
        assert!(primary.prepared_entry);
        primary.inline_attempted.set(true);
        primary.inline_retry_at.set(0);
        assert!(function.runtime().code.set(Some(primary.clone())).is_ok());
        for _ in 0..8 {
            eval(&mut engine, "subject(5)");
        }
        let env = engine.interp.global_env;
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("prepared user")
        };
        let realm = user.realm;
        drop(object);
        let plan = plan_inlines(&primary, &function, realm, &engine.interp.global_env);
        assert!(
            !plan.is_empty(),
            "native cache must prove a real inline plan"
        );
        let chunk = compile_with_inlines(&function, &plan, &primary).unwrap();
        assert!(chunk.prepared_entry);
        assert!(chunk.ops.iter().any(|op| matches!(op, Op::InlineGuard(..))));
        for removed in ["caller directive", "leaf directive", "use strict"] {
            assert!(!strings(&chunk).iter().any(|text| text == removed));
        }
        let leaf = chunk
            .ops
            .iter()
            .find_map(|op| match op {
                Op::InlineGuard(target, _) => {
                    let callee = chunk.jit_inline_target(*target).pin.upgrade()?;
                    let is_leaf = matches!(&callee.borrow().call,
                        crate::value::Callable::User(user) if user.func.name.as_deref() == Some("leaf"));
                    is_leaf.then_some(callee)
                }
                _ => None,
            })
            .expect("the named leaf must actually have been spliced");
        let leaf_code = {
            let object = leaf.borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("leaf")
            };
            user.func
                .runtime()
                .code
                .get()
                .and_then(Option::as_ref)
                .unwrap()
                .jit
                .get()
                .flatten()
                .unwrap()
        };
        assert!(
            leaf_code.test_residency_marks(false).1 > 0,
            "original native leaf ran during warmup"
        );
        leaf_code.test_residency_marks(true);
        assert!(function.runtime().code2.set(Some(chunk.clone())).is_ok());
        assert!(Rc::ptr_eq(
            function
                .runtime()
                .execution_code_for(Env::as_ptr(&engine.interp.global_env) as usize)
                .and_then(Option::as_ref)
                .unwrap(),
            &chunk
        ));
        engine.set_tier(tier);
        assert_eq!(
            eval(&mut engine, "trace=[];JSON.stringify([subject(5),trace])"),
            if strict {
                "[[7,7,6],[\"leaf\",\"finally\"]]"
            } else {
                "[[7,8,8],[\"leaf\",\"finally\"]]"
            }
        );
        assert_eq!(
            leaf_code.test_residency_marks(false).1,
            0,
            "the splice must execute"
        );
        if tier == Tier::Jit {
            let code = chunk.jit.get().flatten().expect("native inline body");
            let (active, referenced) = code.test_residency_marks(false);
            assert!(referenced > 0);
            assert_eq!(active, 0);
        }
        assert_eq!(engine.interp.depth, 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}
