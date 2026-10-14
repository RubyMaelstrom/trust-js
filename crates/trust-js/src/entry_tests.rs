//! General compiled function entry must reuse FunctionDeclarationInstantiation's live bindings.
//! ECMA-262 e28783d5fc9dc12b3de905961e2c71410b38a202, spec.html:14247–14435.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("entry fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// Which compiled entry a function declaration must use. Rest and destructuring parameters,
/// ordinary `new.target`, and mapped `arguments` projections have lean entries; parameter
/// expressions that create closures, direct eval, and arrows that observe the activation still
/// need the general FunctionDeclarationInstantiation entry.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Entry {
    Prepared,
    Lean,
}

use Entry::{Lean, Prepared};

fn check(source: &str, expected: &str, functions: &[(&str, Entry)], native: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source), expected, "{tier:?}");
        if matches!(tier, Tier::Interp) {
            continue;
        }
        for &(name, entry) in functions {
            let env = engine.interp.global_env.clone();
            let value = engine
                .interp
                .get_var(name, &env)
                .unwrap_or_else(|_| panic!("missing {name}"));
            let crate::value::Value::Obj(object) = value else {
                panic!("{name} is not an object")
            };
            let object = object.borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("{name} is not a user function")
            };
            let chunk = user
                .func
                .code
                .get()
                .and_then(Option::as_ref)
                .unwrap_or_else(|| panic!("{name} stayed interpreted"));
            // Named expressions and arrows can use either entry now that their lexical
            // bindings are retained by closure creation.
            if !user.func.is_arrow && !user.func.is_fn_expr {
                let actual = if chunk.prepared_entry { Prepared } else { Lean };
                assert_eq!(
                    actual, entry,
                    "{name} used the wrong compiled entry ({tier:?})"
                );
            }
            if native
                && matches!(tier, Tier::Jit)
                && cfg!(any(target_arch = "aarch64", target_arch = "x86_64"))
            {
                assert!(
                    chunk.jit.get().is_some_and(|code| code.is_some()),
                    "{name} stayed in bytecode"
                );
            }
        }
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[test]
fn compiled_entry_rest_destructuring_and_parameter_default_order() {
    check(
        r#"
        var log=[];
        function rest(first,...tail){return first+tail.length+tail[0];}
        function destructure({x=4},[a,b=3]){return x+a+b;}
        function defaults(a=(log.push('a'),3),b=(log.push('b'),a+1)){return a+b;}
        function tdz(a=b,b=2){return a+b;}
        var out=[rest(2,3,4),destructure({},[2]),defaults()];
        try{tdz()}catch(e){out.push(e.name)}
        out.push(tdz(1,2));
        out.join('|')+'|'+log.join(',')
    "#,
        "7|9|7|ReferenceError|3|a,b",
        &[("rest", Lean), ("destructure", Lean), ("tdz", Prepared)],
        true,
    );
}

#[test]
fn compiled_entry_parameter_iterator_closing_happens_once_before_body() {
    check(
        r#"
        var log=[];
        function iterator(){return {[Symbol.iterator](){return this},
            next(){log.push('next');return {done:false,value:2}},
            return(){log.push('close');return {done:true}}};}
        function first([x]){log.push('body');return x;}
        function fail([x=(()=>{throw 9})()]){log.push('bad');}
        fail([1]);log=[];
        first(iterator());first(iterator());
        var it=iterator();it.next=function(){log.push('undefined');return {done:false}};
        try{fail(it)}catch(e){log.push(e)}
        log.join(',')
    "#,
        "next,close,body,next,close,body,undefined,close,9",
        &[("first", Lean), ("fail", Prepared)],
        true,
    );
}

#[test]
fn compiled_entry_arguments_mapping_and_parameter_body_cells() {
    check(
        r#"
        function mapped(a,b){a=4;arguments[1]=7;var a;return a+':'+arguments[0]+':'+b;}
        function duplicate(a,a){arguments[0]=8;arguments[1]=9;return a+':'+arguments[0];}
        function unmapped(a){'use strict';a=4;arguments[0]=9;return a+':'+arguments[0];}
        function complex(a=3){a=4;arguments[0]=9;return a+':'+arguments[0];}
        function separate(a=1,read=()=>a){var a=9;return read()+':'+a;}
        function captured(a){var read=()=>a;arguments[0]=7;return read()+a;}
        [mapped(1,2),duplicate(1,2),unmapped(1),complex(),separate(),captured(1)].join('|')
    "#,
        "4:4:7|9:8|4:9|4:9|1:9|14",
        &[
            ("mapped", Prepared),
            ("duplicate", Prepared),
            ("complex", Prepared),
            ("separate", Prepared),
            ("captured", Prepared),
        ],
        true,
    );
}

#[test]
fn missing_last_duplicate_formal_does_not_map_an_earlier_argument() {
    // ECMA-262 e28783d5 CreateMappedArgumentsObject scans every formal in reverse,
    // recording a name even when its last index lies beyond the actual argument list.
    // Check both directions of aliasing and the still-mapped, supplied last occurrence.
    check(
        r#"
        function duplicate(a,a){
            var original=arguments[0];
            arguments[0]=7;
            var afterIndex=a;
            a=9;
            return original+':'+afterIndex+':'+arguments[0]+':'+arguments[1]+':'+a;
        }
        function interleaved(a,b,a){
            arguments[0]=6;arguments[1]=8;
            var before=a;a=9;
            return before+':'+b+':'+arguments[0]+':'+arguments[1]+':'+(2 in arguments);
        }
        function define(a,a){
            Object.defineProperty(arguments,'0',{value:7,writable:false});
            var before=a;a=9;
            return before+':'+arguments[0]+':'+a;
        }
        function escaped(a,a){return {args:arguments,set:function(v){a=v},get:function(){return a}}}
        var e=escaped(3);e.set(9);e.args[0]=8;
        [duplicate(3),duplicate(3,4),duplicate(),interleaved(3,4),define(3),
            e.get()+':'+e.args[0]].join('|')
        "#,
        "3:undefined:7:undefined:9|3:4:7:9:9|undefined:undefined:7:undefined:9|undefined:8:6:8:false|undefined:7:9|9:8",
        &[("duplicate", Prepared), ("interleaved", Prepared), ("define", Prepared), ("escaped", Prepared)],
        true,
    );
}

#[test]
fn compiled_entry_hoists_and_block_capture_identity() {
    check(
        r#"
        function hoists(...args){var before=f;function f(){return args[0]};return before===f && f()===3;}
        function blocks(...args){var get;{let x=args[0];get=()=>x;x++;}return get();}
        function sibling(...args){var a,b;{let x=1;a=()=>x}{let x=args[0];b=()=>x}return a()+b();}
        function body(...args){let x=args[0];var read=()=>x;x++;return read();}
        function annex(...args){var before=f;{function f(){return args[0]}}return before===undefined && f()===3;}
        [hoists(3),blocks(3),sibling(3),body(3),annex(3)].join('|')
    "#,
        "true|4|4|4|true",
        &[
            ("hoists", Lean),
            ("blocks", Lean),
            ("sibling", Lean),
            ("body", Lean),
            ("annex", Lean),
        ],
        false,
    );
    // A closure in a parameter expression gives the body its own var environment (ECMA-262
    // §10.2.11 step 28), which keeps the same hoisting and block-capture semantics on the
    // general entry.
    check(
        r#"
        function hoists(v,p=()=>v){var before=f;function f(){return v};return before===f && f()===3;}
        function blocks(v,p=()=>v){var get;{let x=v;get=()=>x;x++;}return get();}
        function sibling(v,p=()=>v){var a,b;{let x=1;a=()=>x}{let x=v;b=()=>x}return a()+b();}
        function body(v,p=()=>v){let x=v;var read=()=>x;x++;return read();}
        function annex(v,p=()=>v){var before=f;{function f(){return v}}return before===undefined && f()===3;}
        [hoists(3),blocks(3),sibling(3),body(3),annex(3)].join('|')
    "#,
        "true|4|4|4|true",
        &[
            ("hoists", Prepared),
            ("blocks", Prepared),
            ("sibling", Prepared),
            ("body", Prepared),
            ("annex", Prepared),
        ],
        false,
    );
}

#[test]
fn compiled_entry_named_function_self_binding_and_shadowing() {
    check(
        r#"
        var named=function self(n){if(n)return self(n-1);self=3;return typeof self;};
        var shadow=function self(){var self=4;return self;};
        var strict=function self(){'use strict';try{self=4}catch(e){return e.name}};
        function make(value){return function same(){return same.value}}
        var one=make(1),two=make(2);one.value=10;two.value=20;
        [named(8),shadow(),strict(),one(),two(),one()].join('|')
    "#,
        "function|4|TypeError|10|20|10",
        &[
            ("named", Lean),
            ("shadow", Lean),
            ("strict", Lean),
            ("one", Lean),
            ("two", Lean),
        ],
        true,
    );
}

#[test]
fn compiled_entry_arrows_inherit_lexical_this_arguments_and_new_target() {
    check(
        r#"
        var arrow, argsArrow, targetArrow;
        function create(a){arrow=()=>this.value;argsArrow=()=>arguments[0];arguments[0]=a+1;}
        create.call({value:42},4);
        function C(){targetArrow=()=>new.target;}
        new C();
        var result=[arrow(),argsArrow(),targetArrow()===C];
        C();result.push(targetArrow()===undefined);
        function sloppy(...args){return args[0]?this===globalThis:Object.prototype.toString.call(this)}
        result.push(sloppy.call(3),sloppy.call(null,true));
        result.join('|')
    "#,
        "42|5|true|true|[object Number]|true",
        &[
            ("arrow", Lean),
            ("argsArrow", Lean),
            ("targetArrow", Lean),
            ("C", Prepared),
            ("sloppy", Lean),
        ],
        false,
    );
}

#[test]
fn compiled_entry_direct_eval_observes_and_updates_live_bindings() {
    check(
        r#"
        function evaluate(a){var x=2;eval('a=4;x=5;var added=6');return a+':'+x+':'+added+':'+arguments[0];}
        function lexical(...args){let x=3;eval('x=8');return x;}
        function localEval(eval,...args){return eval(args[0]);}
        function scoped(a=1, read=()=>a){var a=9;eval('a=10');return a+':'+read();}
        [evaluate(1),lexical(),localEval(x=>x+2,5),scoped()].join('|')
    "#,
        "4:5:6:4|8|7|10:1",
        &[
            ("evaluate", Prepared),
            ("lexical", Prepared),
            ("localEval", Prepared),
            ("scoped", Prepared),
        ],
        false,
    );
}

#[test]
fn compiled_entry_constructor_fallback_keeps_return_mapping() {
    check(
        r#"
        function C(...args){this.value=args[0];return args[1]}
        C.call({},1,2);
        var a=new C(3,9),b=new C(3,{value:4});
        a.value+':'+(a instanceof C)+':'+b.value+':'+(b instanceof C)
    "#,
        "3:true:4:false",
        &[("C", Lean)],
        true,
    );
    check(
        r#"
        function C(v,w,p=()=>v){this.value=v;return w}
        C.call({},1,2);
        var a=new C(3,9),b=new C(3,{value:4});
        a.value+':'+(a instanceof C)+':'+b.value+':'+(b instanceof C)
    "#,
        "3:true:4:false",
        &[("C", Prepared)],
        true,
    );
}

#[test]
fn compiled_entry_new_target_observers_keep_their_own_activation() {
    // ECMA-262 §9.4.5 GetNewTarget: an arrow, a class computed key, or a class heritage
    // expression reads the [[NewTarget]] of the nearest non-arrow function, even when an
    // enclosing function also has one and even after the activation has returned.
    check(
        r#"
        var named, keyed, heritage;
        function Named(){named=()=>new.target;var local=()=>new.target;return local()}
        function Keyed(){var k=class{[(()=>typeof new.target)()](){}};keyed=Object.getOwnPropertyNames(k.prototype)[1]}
        function Outer(){return function inner(){return class extends (new.target?Object:Array){}}}
        function Plain(){return new.target}
        var direct=new Named()===undefined;
        var after=named()===Named;Named();var cleared=named()===undefined;
        new Keyed();var constructed=keyed;Keyed();
        var inner=new Outer();heritage=[inner()===undefined, Object.getPrototypeOf(inner()) === Array,
            Object.getPrototypeOf(new inner()) === Object];
        [direct, after, cleared, constructed, keyed, heritage.join(','),
         Plain()===undefined, new Plain()===undefined].join('|')
    "#,
        "false|true|true|function|undefined|false,true,true|true|false",
        &[
            ("Named", Prepared),
            ("Keyed", Prepared),
            ("Outer", Lean),
            ("Plain", Lean),
        ],
        true,
    );
}

#[test]
fn compiled_lean_arrows_read_live_lexical_this_only_when_executed() {
    check(
        r#"
        var arrow, captured, before, error, original;
        class Base { constructor(){this.value=40} }
        class Derived extends Base {
            constructor(){
                arrow=read=>read?this.value:3;
                before=arrow(false);
                try{arrow(true)}catch(e){error=e.name}
                super();
                this.value+=2;
            }
        }
        original=new Derived();
        var first=arrow.call({value:99},true);
        original.value=43;
        function make(a){captured=()=>arguments[0];a=9}
        make(1);
        [before,error,first,arrow(true),captured()].join('|')
    "#,
        "3|ReferenceError|42|43|9",
        &[("arrow", Lean), ("captured", Lean)],
        true,
    );
}

#[test]
fn named_expression_self_environment_is_created_once_and_collectible() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                "var named=function self(){return self}; named()===named"
            ),
            "true"
        );
        let env = engine.interp.global_env.clone();
        let value = engine
            .interp
            .get_var("named", &env)
            .unwrap_or_else(|_| panic!("missing named function"));
        let crate::value::Value::Obj(object) = value else {
            panic!("missing function")
        };
        let weak = std::rc::Rc::downgrade(&object);
        {
            let borrowed = object.borrow();
            let crate::value::Callable::User(user) = &borrowed.call else {
                panic!("not a function")
            };
            let closure = user.env.borrow();
            let binding = closure
                .vars
                .get("self")
                .expect("self binding exists at closure creation");
            assert!(!binding.mutable && binding.initialized && !binding.strict_immutable);
            assert!(
                matches!(&binding.value, crate::value::Value::Obj(bound) if std::rc::Rc::ptr_eq(bound, &object))
            );
            if !matches!(tier, Tier::Interp) {
                assert!(
                    !user
                        .func
                        .code
                        .get()
                        .and_then(Option::as_ref)
                        .unwrap()
                        .prepared_entry
                );
            }
        }
        drop(object);
        evaluate(&mut engine, "named=null");
        // The function/prototype/self-environment cycle must not become a permanent root.
        engine.interp.gc_collect();
        assert!(
            weak.upgrade().is_none(),
            "unreachable named function survived collection"
        );
    }
}

#[test]
fn named_expression_unused_self_environment_is_elided_without_changing_identity() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine,
            "var named=function unused(value){return value+1};[named.name,named.length,named(3)].join('|')"),
            "unused|1|4");
        let env = engine.interp.global_env.clone();
        let value = engine
            .interp
            .get_var("named", &env)
            .unwrap_or_else(|_| panic!("named"));
        let object = value.as_obj().unwrap();
        let weak = std::rc::Rc::downgrade(object);
        let enabled = std::env::var("TRUST_JS_ELIDE_SELF_ENV").as_deref() != Ok("0");
        {
            let borrowed = object.borrow();
            let crate::value::Callable::User(user) = &borrowed.call else {
                panic!("user function")
            };
            assert_eq!(user.env.borrow().vars.get("unused").is_none(), enabled);
            // Activation scans run after creation; their cache must not erase the self query.
            user.func.scan_flags();
            assert_eq!(user.func.needs_self_environment(), !enabled);
        }
        drop(value);
        evaluate(&mut engine, "named=null;");
        if enabled {
            assert!(
                weak.upgrade().is_none(),
                "unused self binding kept a cycle alive"
            );
        }
        engine.interp.gc_collect();
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn named_expression_self_environment_keeps_nested_patterns_eval_and_classes() {
    let source = r#"
        var recursive=function Self(n){return n?Self(n-1):Self};
        var parameter=function Self([x=Self]=[]){return x};
        var key=function Self({[Self.name]:x}={Self:7}){return x};
        var nested=function Self(){return function(){return Self}};
        var arrow=function Self(){return ()=>Self};
        var evaluated=function Self(){return function(){return (eval)('Self')}};
        var withEnv=function Self(){with({})return eval('Self')};
        var klass=function Self(){return class{field=Self;get value(){return Self}}};
        var C=klass(),instance=new C;
        [recursive(3)===recursive,parameter()===parameter,key(),nested()()===nested,
         arrow()()===arrow,evaluated()()===evaluated,withEnv()===withEnv,
         instance.field===klass,instance.value===klass].join('|')
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(&mut engine, source),
            "true|true|7|true|true|true|true|true|true",
            "{tier:?}"
        );
    }
}

#[test]
fn named_expression_self_environment_elision_preserves_outer_closure_and_constructor() {
    let source = r#"
        function make(value){return function Unused(delta){return value+delta}}
        var one=make(10),two=make(20);
        var C=function Named(value){this.value=value};
        var a=new C(7),b=new C(8);
        [one(1),two(1),one!==two,a.value,b.value,a instanceof C,b instanceof C,
         C.prototype.constructor===C,one.name,two.name].join('|')
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(&mut engine, source),
            "11|21|true|7|8|true|true|true|Unused|Unused",
            "{tier:?}"
        );
    }
}
