//! ECMA-262 LoopEvaluation, UpdateEmpty, IteratorClose and DisposeResources.
//! Actual fragment/native counters prevent an AST fallback from masking transfer bugs.
use crate::{bytecode::Tier, Completion, Engine};

fn enable_callback(tier: Tier) -> crate::value::NativeFn {
    match tier {
        Tier::Interp => |i, _, _| {
            i.tier = Tier::Interp;
            Ok(crate::value::Value::Undefined)
        },
        Tier::Bytecode => |i, _, _| {
            i.tier = Tier::Bytecode;
            Ok(crate::value::Value::Undefined)
        },
        Tier::Jit => |i, _, _| {
            i.tier = Tier::Jit;
            Ok(crate::value::Value::Undefined)
        },
    }
}

fn check(source: &str, expected: &str, minimum_fragments: usize, native: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.set(0));
        crate::jit::TEST_OSR_ENTRIES.with(|n| n.set(0));
        match engine.eval(source, false).expect("fragment fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}: {source}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}\n{source}"),
        }
        let entries = crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.get());
        if tier == Tier::Interp {
            assert_eq!(entries, 0);
        } else {
            assert!(
                entries >= minimum_fragments,
                "{tier:?}: {entries} fragment entries\n{source}"
            );
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if native && tier == Tier::Jit {
            assert!(
                crate::jit::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
                "actual native fragment entry required: {source}"
            );
        }
    }
}

#[test]
fn fragment_cold_loops_do_not_compile_or_clone_a_body() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        for source in [
            "for(var k=0;k<0;k++){}",
            "for(var k=0;k<1;k++){}",
            "for(var k=0;k<100;k++){}",
            "if(false){while(true){}}",
        ] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(32);
            crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.set(0));
            crate::bytecode::TEST_SCRIPT_ENTRIES.with(|n| n.set((0, 0)));
            engine.eval(source, false).expect("cold fixture parses");
            assert_eq!(
                crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.get()),
                0
            );
            assert_eq!(
                crate::bytecode::TEST_SCRIPT_ENTRIES.with(|n| n.get()),
                (0, 0)
            );
        }
    }
}

#[test]
fn fragment_all_loop_phases_transfer_without_replaying_observable_heads() {
    check(
        "var k=0,s=0,t=0;while((t++,k<1000)){s+=k++;} [k,s,t].join('|')",
        "1000|499500|1001",
        1,
        true,
    );
    check(
        "var k=0,t=0;do{k++;if(k%2)continue;}while((t++,k<1000));[k,t].join('|')",
        "1000|1000",
        1,
        true,
    );
    check("var init=0,t=0,u=0,s=0;for(var k=(init++,0);(t++,k<1000);(u++,k++)){s+=k;}[init,t,u,s].join('|')",
        "1|1001|1000|499500", 1, true);
    check("var o={},n=0,s=0;for(var k=0;k<300;k++)o[k]=k;for(var key in o){n++;s+=+key;if(key==='130')delete o[200];}[n,s].join('|')",
        "299|44650", 2, false);
    check("var n=0,closed=0,seen=0,it={next(){return {done:false,value:n++}},return(){closed++;return {}}};var src={[Symbol.iterator](){return it}};for(const x of src){seen++;if(x===127)it.next=()=>{throw 'replayed next lookup'};if(x===255)break;}[n,seen,closed].join('|')",
        "256|256|1", 1, false);
}

#[test]
fn fragment_external_labels_unwind_local_handlers_once_then_resume_outer_ast() {
    check(
        r#"var n=0,clean=0,after=0;outer:for(var a=0;a<2;a++){
        inner:for(var k=0;k<1000;k++){try{n++;if(k===800)continue outer;}finally{clean++;}}
        after++;
    }[n,clean,after].join('|')"#,
        "1602|1602|0",
        2,
        true,
    );
    check(
        r#"var n=0,clean=0;block:{for(var k=0;k<1000;k++){
        try{n++;if(k===800)break block;}finally{clean++;}
    }}[n,clean].join('|')"#,
        "801|801",
        1,
        true,
    );
    check(
        "var k=0;while(k++<1000){if(k===900){37;break;}}",
        "37",
        1,
        true,
    );
}

#[test]
fn fragment_preserves_live_environment_eval_with_and_per_iteration_closures() {
    check(
        r#"var f=[],n=0;for(let k=0;k<1000;k++){
        let x=k;if(k===0||k===127||k===128||k===999)f.push(()=>[k,x]);
        eval('n += 1');
    }[n,f.map(x=>x().join(':')).join(',')].join('|')"#,
        "1000|0:0,127:127,128:128,999:999",
        1,
        true,
    );
    check(
        r#"var o={n:0},outer=5;with(o){for(var k=0;k<1000;k++){
        n++;if(k===128)o.outer=7;if(k===129)delete o.outer;
        if(k===128&&outer!==7)throw 'bad with';
    }}[o.n,outer].join('|')"#,
        "1000|5",
        1,
        true,
    );
    check(
        r#"var first,last;for(var k=0;k<1000;k++){
        function f(){return 1}if(k===0)first=f;function f(){return 2}last=f;
    }[first(),last(),f(),first!==last].join('|')"#,
        "2|2|2|true",
        1,
        true,
    );
}

#[test]
fn fragment_iterator_step_failures_do_not_close_but_body_throw_does() {
    for operation in [
        "throw 'step'",
        "return {get done(){throw 'step'}}",
        "return {done:false,get value(){throw 'step'}}",
    ] {
        let source = format!("var n=0,c=0,msg='';var src={{[Symbol.iterator](){{return {{next(){{if(n===130){{{operation}}}return {{done:false,value:n++}}}},return(){{c++;return {{}}}}}}}}}};try{{for(const x of src){{}}}}catch(e){{msg=e}}[n,c,msg].join('|')");
        check(&source, "130|0|step", 1, false);
    }
    check("var n=0,c=0,msg='';var src={[Symbol.iterator](){return {next(){return {done:false,value:n++}},return(){c++;throw 'close'}}}};try{for(const x of src){if(x===130)throw 'body'}}catch(e){msg=e}[n,c,msg].join('|')",
        "131|1|body", 1, false);
}

#[test]
fn fragment_using_disposal_preserves_iteration_and_parked_head_ownership() {
    check(
        r#"var closed=0,disposed=0,n=0;var src={[Symbol.iterator](){return {
        next(){return {done:false,value:{[Symbol.dispose](){disposed++}}}},
        return(){closed++;return {}}
    }}};for(using x of src){if(++n===300)break;}[n,disposed,closed].join('|')"#,
        "300|300|1",
        1,
        true,
    );
    check("var disposed=0,n=0;for(using x={[Symbol.dispose](){disposed++}};n<1000;n++){}[n,disposed].join('|')",
        "1000|1", 1, true);
    check("var disposed=0,f=[],n=0;for(var k=0;k<1000;k++){for(using x={v:k,[Symbol.dispose](){disposed++}};n===k;){f.push(()=>x.v);n++;}}[disposed,f[0](),f[128](),f[999]()].join('|')",
        "1000|0|128|999", 1, true);
}

/// Enter the containing call on the AST oracle, then change execution policy from a native
/// callback before its loop. This proves transfer inside a genuine Function Environment,
/// instead of allowing the ordinary whole-function compiler to satisfy the fixture.
fn context_check(source: &str, expected: &str, minimum_fragments: usize) {
    use crate::value::{Property, Value};
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Interp);
        engine.set_tier_threshold(32);
        let enable = engine
            .interp
            .make_native("enableFragments", 0, enable_callback(tier));
        engine.interp.global.borrow_mut().props.insert(
            "enableFragments",
            Property::data(Value::Obj(enable), true, true, true),
        );
        crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.set(0));
        match engine.eval(source, false).expect("context fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}: {source}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}\n{source}"),
        }
        let entries = crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.get());
        if tier == Tier::Interp {
            assert_eq!(entries, 0);
        } else {
            assert!(
                entries >= minimum_fragments,
                "{tier:?}: {entries} fragment entries\n{source}"
            );
        }
    }
}

#[test]
fn fragment_function_context_preserves_this_super_newtarget_and_eval_bindings() {
    context_check(
        r#"class Base{constructor(){this.base=7}method(){return 3}}
        class Derived extends Base{constructor(){enableFragments();let caught=0,total=0;
            for(let k=0;k<1000;k++){
                if(k===128){try{this}catch(e){caught+=e instanceof ReferenceError}}
                if(k===129)super();
                if(k>=129)total+=super.method();
                if(new.target!==Derived)throw 'new.target';
            }this.answer=[caught,total,this.base].join('|');
        }}new Derived().answer"#,
        "1|2613|7",
        1,
    );
    context_check(
        r#"function f(a){enableFragments();var n=0;with({'this':{bad:true}}){
        for(var k=0;k<1000;k++){if(this.marker!==9)throw 'receiver';
            eval('a += 1');if(arguments[0]!==a)throw 'mapped';n++;}}
        return [a,n,this.marker].join('|');}f.call({marker:9},1)"#,
        "1001|1000|9",
        1,
    );
    context_check(
        r#"function outer(){'use strict';return ()=>{enableFragments();var n=0;
        for(var k=0;k<1000;k++){
            if(this!==17||eval('this')!==17)throw 'strict primitive receiver';n++;
        }return n;}}outer.call(17)()"#,
        "1000",
        1,
    );
    context_check(
        r#"function f(){var answer;with({'this':{bad:true}}){
        var arrow=()=>{'use strict';enableFragments();var n=0;
            for(var k=0;k<1000;k++){
                if(this.marker!==9||eval('this').marker!==9)throw 'lexical receiver';n++;
            }return n;};answer=arrow();}return answer;}f.call({marker:9})"#,
        "1000",
        1,
    );
}

#[test]
fn fragment_return_unwinds_owned_iterators_then_parked_ast_handlers_once() {
    context_check(
        r#"var log=[];function iterable(name){return {[Symbol.iterator](){
        let n=0;return {next(){return {done:false,value:n++}},return(){log.push(name);return {}}}
    }}}function f(){enableFragments();try{for(const a of iterable('outer')){
        for(const b of iterable('inner')){try{if(b===800)return 31;}finally{
            if(b===800)log.push('local finally');}}
    }}finally{log.push('outer finally')}}[f(),log.join(',')].join('|')"#,
        "31|local finally,inner,outer,outer finally",
        1,
    );
    context_check(
        r#"function f(blocked){enableFragments();var caught;
        for(var k=0;k<1000;k++){try{throw k}catch(x){var x=x+1;caught=x;}
            {function blocked(){return 7}}}
        return [blocked,caught,typeof x].join('|')};f(42)"#,
        "42|1000|undefined",
        1,
    );
}

#[test]
fn fragment_tail_return_preserves_pending_trampoline_transfer() {
    use crate::{
        ast::Stmt,
        bytecode::loop_fragment::{enter, Seed, Source},
        interpreter::Abrupt,
        value::Value,
    };
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        engine
            .eval(
                "var k=0,called=0;function target(x){called++;return x}",
                false,
            )
            .unwrap();
        let body = crate::parser::parse_script(
            "function source(){while(k++<2000){if(k===1999)return target(9)}}",
            false,
        )
        .ok()
        .expect("tail fixture parses");
        let Stmt::FuncDecl(function) = &body[0] else {
            panic!("function")
        };
        let Stmt::While { test, body } = &function.body[0] else {
            panic!("loop")
        };
        let site = &body.site;
        engine.interp.strict = true;
        engine.interp.tco_ok = true;
        let env = engine.interp.global_env.clone();
        // Past the ordinary-call depth (see `Op::TailDeep`) the tail call must be staged.
        let depth = engine.interp.depth;
        engine.interp.depth = crate::bytecode::ordinary_tail_call_depth();
        let outcome = enter(
            &mut engine.interp,
            site,
            Source::While(test, body),
            &[],
            &env,
            &Value::Undefined,
            Seed::None,
        )
        .ok()
        .expect("admitted")
        .expect("compiled");
        engine.interp.depth = depth;
        assert!(matches!(outcome, Err(Abrupt::Return(Value::Undefined))));
        let pending = engine
            .interp
            .pending_tail
            .take()
            .expect("tail call staged, not executed");
        assert!(matches!(
            engine.interp.get_var("called", &env),
            Ok(Value::Num(0.0))
        ));
        let (callee, receiver, arguments) = *pending;
        assert!(matches!(
            engine.interp.call(callee, receiver, &arguments),
            Ok(Value::Num(9.0))
        ));
        assert!(matches!(
            engine.interp.get_var("called", &env),
            Ok(Value::Num(1.0))
        ));
        // A shallow tail call runs as an ordinary call and returns its result directly.
        engine.eval("k=0", false).unwrap();
        let outcome = enter(
            &mut engine.interp,
            site,
            Source::While(test, body),
            &[],
            &env,
            &Value::Undefined,
            Seed::None,
        )
        .ok()
        .expect("admitted")
        .expect("compiled");
        assert!(matches!(outcome, Err(Abrupt::Return(Value::Num(9.0)))));
        assert!(engine.interp.pending_tail.is_none());
        assert!(matches!(
            engine.interp.get_var("called", &env),
            Ok(Value::Num(2.0))
        ));
    }
}

#[test]
fn fragment_declines_actual_suspension_but_not_nested_async_function_syntax() {
    context_check("function f(){enableFragments();var n=0;for(var k=0;k<1000;k++){async function nested(){await 1}n++;}return n}f()",
        "1000", 1);
    check("var n=0;for(var k=0;k<1000;k++){class C{[(async()=>{await 1})]() {}}n+=typeof C==='function'}n",
        "1000", 1, true);
    use crate::value::{Property, Value};
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Interp);
        let enable = engine
            .interp
            .make_native("enableFragments", 0, enable_callback(tier));
        engine.interp.global.borrow_mut().props.insert(
            "enableFragments",
            Property::data(Value::Obj(enable), true, true, true),
        );
        crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.set(0));
        engine.eval(r#"var log=[],answer='pending';async function* source(){enableFragments();
            try{for(var k=0;k<1000;k++){if(k===999)return {then(resolve){log.push('await');resolve(17)}}}}
            finally{log.push('finally')}
        }source().next().then(x=>answer=x.value+'|'+x.done);"#, false).unwrap();
        match engine.eval("answer+'|'+log.join(',')", false).unwrap() {
            Completion::Value(value) => assert_eq!(value, "17|true|await,finally", "{tier:?}"),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        assert_eq!(
            crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|n| n.get()),
            0,
            "implicit async-generator return Await must remain in its original coroutine"
        );
    }
}
