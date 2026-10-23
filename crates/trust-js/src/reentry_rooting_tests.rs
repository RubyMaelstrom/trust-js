//! Rust worklists must retain their cells while author callbacks run.
//! ECMA-262 e28783d5 (2026-09-06): DisposeResources, IteratorZip, GroupBy,
//! ClassDefinitionEvaluation and InitializeInstanceElements.

use crate::bytecode::Tier;
use crate::gc_core::CollectionKind;
use crate::gc_handle::WeakRef;
use crate::gc_trace::GcSink;
use crate::value::{set_data, NativeCaptures, Value};
use crate::{Completion, Engine};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
enum Roots {
    Temporary,
    Coroutine,
    Heap,
}

#[derive(Default)]
struct Watched {
    cells: Vec<WeakRef>,
    checks: usize,
}

#[derive(Default)]
struct Words(HashSet<u64>);

impl GcSink for Words {
    fn word(&mut self, word: u64) {
        self.0.insert(word);
    }
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("rooting fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn eval_without_jobs(engine: &mut Engine, source: &str) {
    let body = crate::parser::parse_script(source, false)
        .unwrap_or_else(|_| panic!("suspended fixture parses: {source}"));
    engine
        .interp
        .run_program(&body)
        .unwrap_or_else(|_| panic!("suspended fixture throws before its checkpoint"));
}

fn run_microtasks(engine: &mut Engine) {
    // The embedding checkpoint has this same sequence. Keep the fixture
    // available in the default and minimal engine configurations as well.
    engine.interp.activate_gc_heap();
    engine.interp.drain_microtasks();
    engine.interp.gc_task_boundary();
}

fn setup(tier: Tier) -> (Engine, Rc<RefCell<Watched>>) {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    let watched = Rc::new(RefCell::new(Watched::default()));
    let state = watched.clone();
    let watch = engine.interp.new_native_fn(
        "__watchRoot",
        1,
        Rc::new(move |_, _, args| {
            let value = args[0];
            let word = value.cell_word().expect("watch a managed value");
            state.borrow_mut().cells.push(WeakRef::to(word as usize));
            Ok(value)
        }),
        NativeCaptures::None, // weak observers and Rust counters only
    );
    set_data(&engine.interp.global, "__watchRoot", watch);
    for (name, domain) in [
        ("__checkTemporaryRoots", Roots::Temporary),
        ("__checkCoroutineRoots", Roots::Coroutine),
        ("__collectRootFixture", Roots::Heap),
    ] {
        let state = watched.clone();
        let check = engine.interp.new_native_fn(
            name,
            0,
            Rc::new(move |interp, _, _| {
                let cells = state.borrow().cells.clone();
                if !matches!(domain, Roots::Heap) {
                    assert!(!cells.is_empty(), "the fixture must install witnesses");
                    let mut words = Words::default();
                    match domain {
                        Roots::Temporary => crate::rooting::trace_rooted(&mut words),
                        Roots::Coroutine => crate::gc_trace::trace_running_coroutines(&mut words),
                        Roots::Heap => unreachable!(),
                    }
                    for weak in &cells {
                        // Require the actual worklist/continuation edge before collecting.
                        // A stale native word or dead JIT slot cannot make this test pass.
                        assert!(weak.live_ptr().is_some(), "witness died before checkpoint");
                        assert!(
                            words.0.contains(&(weak.ptr() as u64)),
                            "{tier:?}/{domain:?}: missing explicit worklist edge"
                        );
                    }
                }
                for kind in [CollectionKind::Minor, CollectionKind::Major] {
                    let before = interp.gc_heap.core.borrow().stats().collections;
                    interp.gc_heap_collect_explicit(kind);
                    assert_eq!(interp.gc_heap.core.borrow().stats().collections, before + 1);
                    assert!(
                        cells.iter().all(|weak| weak.live_ptr().is_some()),
                        "{tier:?}/{domain:?}: a pending value was collected"
                    );
                }
                state.borrow_mut().checks += 1;
                Ok(Value::Undefined)
            }),
            NativeCaptures::None,
        );
        set_data(&engine.interp.global, name, check);
    }
    (engine, watched)
}

fn retired(engine: &mut Engine, watched: &RefCell<Watched>, minimum: usize) {
    assert!(watched.borrow().cells.len() >= minimum);
    // Precise collection is only used after JS and its native temporaries have unwound.
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert!(
        watched
            .borrow()
            .cells
            .iter()
            .all(|weak| weak.live_ptr().is_none()),
        "completed/abrupt worklists must unregister their roots"
    );
}

#[test]
fn disposal_records_keep_deleted_methods_across_reentrant_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        let source = r#"(function() {
            var seen=[];
            var first={tag:'first',[Symbol.dispose](){seen.push(this.tag)}};
            __watchRoot(first[Symbol.dispose]);
            var holder={first};
            var second={[Symbol.dispose](){
                delete holder.first[Symbol.dispose]; holder.first=null;
                __checkTemporaryRoots(); seen.push('second');
            }};
            { using a=holder.first; using b=second; first=null; }
            if(seen.join(',')!=='second,first')throw 'disposal order/receiver';
            return 'ok';
        })()"#;
        assert_eq!(eval(&mut engine, source), "ok", "{tier:?}");
        assert_eq!(watched.borrow().checks, 1);
        retired(&mut engine, &watched, 1);
    }
}

#[test]
fn disposal_records_keep_return_throw_and_suppressed_completions() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for completion in ["return", "throw"] {
            let (mut engine, watched) = setup(tier);
            let source = format!(
                r#"(function() {{
                    function body() {{
                        using resource={{[Symbol.dispose](){{__checkTemporaryRoots()}}}};
                        {completion} [__watchRoot({{tag:'pending'}})][0];
                    }}
                    var result,threw=false;
                    try {{result=body()}}catch(error){{threw=true;result=error}}
                    if(threw!=={throwing}||!result||result.tag!=='pending')throw 'completion lost';
                    return 'ok';
                }})()"#,
                throwing = completion == "throw",
            );
            assert_eq!(eval(&mut engine, &source), "ok", "{tier:?}/{completion}");
            assert_eq!(watched.borrow().checks, 1);
            retired(&mut engine, &watched, 1);
        }
        let (mut engine, watched) = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"(function(){
            try {
                using a={[Symbol.dispose](){__checkTemporaryRoots();throw {tag:'outer'}}};
                using b={[Symbol.dispose](){throw __watchRoot({tag:'inner'})}};
            }catch(error){
                if(!(error instanceof SuppressedError)||error.error.tag!=='outer'||
                   error.suppressed.tag!=='inner')throw 'suppression lost';
                return 'ok';
            }
            throw 'expected suppression';
        })()"#
            ),
            "ok",
            "{tier:?}"
        );
        assert_eq!(watched.borrow().checks, 1);
        retired(&mut engine, &watched, 1);
    }
}

#[test]
fn async_disposal_first_drive_publishes_resources_and_completions() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        eval(
            &mut engine,
            r#"
            globalThis.__rootResult='pending';
            (async function(){
                var seen=[];
                var first={tag:'first',[Symbol.dispose](){seen.push(this.tag)}};
                __watchRoot(first[Symbol.dispose]);
                var holder={first};
                var second={[Symbol.dispose](){
                    delete holder.first[Symbol.dispose];holder.first=null;
                    __checkCoroutineRoots();seen.push('second');
                }};
                { await using marker=null;using a=holder.first;using b=second;first=null; }
                if(seen.join(',')!=='second,first')throw 'async disposal order';
                __rootResult='ok';
            })().catch(error=>__rootResult='error:'+error);
            'started'
        "#,
        );
        run_microtasks(&mut engine);
        assert_eq!(eval(&mut engine, "__rootResult"), "ok", "{tier:?}");
        assert_eq!(watched.borrow().checks, 1);
        retired(&mut engine, &watched, 1);

        for completion in ["return", "throw"] {
            let (mut engine, watched) = setup(tier);
            let source = format!(
                r#"
                globalThis.__rootResult='waiting';
                (async function(){{
                    await using marker=null;
                    using resource={{[Symbol.dispose](){{__checkCoroutineRoots()}}}};
                    {completion} [__watchRoot({{tag:'pending'}})][0];
                }})().then(value=>{{__rootResult='return:'+value.tag}},error=>{{__rootResult='throw:'+error.tag}});
                'started'
            "#
            );
            eval(&mut engine, &source);
            run_microtasks(&mut engine);
            assert_eq!(
                eval(&mut engine, "__rootResult"),
                format!("{completion}:pending"),
                "{tier:?}/{completion}"
            );
            assert_eq!(watched.borrow().checks, 1);
            retired(&mut engine, &watched, 1);
        }
    }
}

#[test]
fn async_disposal_keeps_the_resource_parked_by_an_await_only_marker() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        eval_without_jobs(
            &mut engine,
            r#"
            globalThis.__rootResult='waiting';globalThis.__disposed=0;
            globalThis.__rootTask=(async function(){
                var resource={[Symbol.dispose](){__disposed++}};
                __watchRoot(resource[Symbol.dispose]);
                { using a=resource;await using marker=null;delete resource[Symbol.dispose];resource=null; }
            })().then(()=>__rootResult='done',error=>__rootResult='error:'+error);
            'started'
        "#,
        );
        assert_eq!(watched.borrow().cells.len(), 1);
        assert_eq!(
            engine
                .interp
                .global
                .borrow()
                .props
                .get("__disposed")
                .unwrap()
                .value()
                .as_num_opt(),
            Some(0.0),
            "the await-only marker must actually suspend before disposal"
        );
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Major);
        assert!(watched.borrow().cells[0].live_ptr().is_some());
        run_microtasks(&mut engine);
        assert_eq!(eval(&mut engine, "__rootResult+':'+__disposed"), "done:1");
        eval(&mut engine, "__rootTask=null;'released'");
        retired(&mut engine, &watched, 1);
    }
}

#[test]
fn async_disposal_retains_suspended_records_until_the_promise_resumes() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        eval(
            &mut engine,
            r#"
            globalThis.__rootResult='waiting';globalThis.__releaseRootTask=null;
            globalThis.__rootTask=(async function(){
                var seen=[];
                var first={[Symbol.dispose](){seen.push('first')}};
                __watchRoot(first[Symbol.dispose]);
                var holder={first};
                {
                    using a=first;
                    await using b={[Symbol.asyncDispose](){
                        delete holder.first[Symbol.dispose];holder.first=null;
                        __checkCoroutineRoots();seen.push('second');
                        return new Promise(resolve=>__releaseRootTask=resolve);
                    }};
                    first=null;
                }
                if(seen.join(',')!=='second,first')throw 'resume order';
            })().then(()=>__rootResult='done',error=>__rootResult='error:'+error);
            'started'
        "#,
        );
        run_microtasks(&mut engine);
        assert_eq!(eval(&mut engine, "__rootResult"), "waiting");
        assert_eq!(watched.borrow().checks, 1);
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Major);
        assert!(watched.borrow().cells[0].live_ptr().is_some());
        eval(
            &mut engine,
            "__releaseRootTask();__releaseRootTask=null;'released'",
        );
        run_microtasks(&mut engine);
        assert_eq!(eval(&mut engine, "__rootResult"), "done");
        eval(&mut engine, "__rootTask=null;'cleared'");
        retired(&mut engine, &watched, 1);
    }
}

#[test]
fn async_disposal_retains_object_completions_across_a_real_await() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for completion in ["return", "throw"] {
            let (mut engine, watched) = setup(tier);
            let source = format!(
                r#"
                globalThis.__rootResult='waiting';globalThis.__releaseRootTask=null;
                globalThis.__rootTask=(async function(){{
                    await using resource={{[Symbol.asyncDispose](){{
                        __checkCoroutineRoots();
                        return new Promise(resolve=>__releaseRootTask=resolve);
                    }}}};
                    {completion} [__watchRoot({{tag:'retained'}})][0];
                }})();
                __rootTask.then(value=>__rootResult='return:'+value.tag,
                                error=>__rootResult='throw:'+error.tag);
                'started'
            "#
            );
            eval(&mut engine, &source);
            assert_eq!(eval(&mut engine, "__rootResult"), "waiting");
            assert_eq!(watched.borrow().checks, 1);
            assert_eq!(watched.borrow().cells.len(), 1);
            engine
                .interp
                .gc_collect_precise_for_test(CollectionKind::Major);
            assert!(watched.borrow().cells[0].live_ptr().is_some());
            eval(
                &mut engine,
                "__releaseRootTask();__releaseRootTask=null;'released'",
            );
            run_microtasks(&mut engine);
            assert_eq!(
                eval(&mut engine, "__rootResult"),
                format!("{completion}:retained"),
                "{tier:?}/{completion}"
            );
            eval(&mut engine, "__rootTask=null;'cleared'");
            retired(&mut engine, &watched, 1);
        }
    }
}

#[test]
fn zip_partial_results_are_roots_during_later_iterator_steps() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for keyed in [false, true] {
            for mode in ["shortest", "longest", "strict"] {
                let (mut engine, watched) = setup(tier);
                let call = if keyed {
                    "Iterator.zipKeyed({a:first,b:second,c:last},options)"
                } else {
                    "Iterator.zip([first,second,last],options)"
                };
                let values = if keyed {
                    "row.value.a.tag+row.value.b.tag+row.value.c"
                } else {
                    "row.value[0].tag+row.value[1].tag+row.value[2]"
                };
                let source = format!(
                    r#"(function(){{
                    function fresh(tag){{var used=false;return {{
                        [Symbol.iterator](){{return this}},next(){{
                            if(used)return {{done:true}};used=true;
                            return {{value:__watchRoot({{tag}}),done:false}};
                        }}
                    }}}}
                    var first=fresh('a'),second=fresh('b'),used=false;
                    var last={{[Symbol.iterator](){{return this}},next(){{
                        if(used)return {{done:true}};used=true;__checkTemporaryRoots();
                        return {{value:7,done:false}};
                    }}}};
                    var options={{mode:'{mode}'}},zip={call},row=zip.next();
                    if(row.done||{values}!=='ab7'||!zip.next().done)throw 'zip result';
                    if({keyed}&&Object.getPrototypeOf(row.value)!==null)throw 'zip prototype';
                    return 'ok';
                }})()"#
                );
                assert_eq!(eval(&mut engine, &source), "ok", "{tier:?}/{keyed}/{mode}");
                assert_eq!(watched.borrow().checks, 1);
                retired(&mut engine, &watched, 2);
            }
        }
    }
}

#[test]
fn class_instance_symbol_keys_survive_later_decorators() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"(function(){
            function observe(value,context){__checkTemporaryRoots();return value}
            class C {
                [__watchRoot(Symbol('a'))]=1;
                [__watchRoot(Symbol('b'))]=2;
                [__watchRoot(Symbol('c'))]=3;
                @observe method(){}
            }
            __collectRootFixture();
            var object=new C(),keys=Object.getOwnPropertySymbols(object);
            if(keys.map(k=>k.description+object[k]).join(',')!=='a1,b2,c3')throw 'instance keys';
            return 'ok';
        })()"#
            ),
            "ok",
            "{tier:?}"
        );
        assert_eq!(watched.borrow().checks, 2);
        retired(&mut engine, &watched, 3);
    }
}

#[test]
fn class_static_symbol_keys_stay_rooted_through_blocks_and_initializers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for first in [
            "static start=(__checkTemporaryRoots(),0);",
            "static {__checkTemporaryRoots();}",
        ] {
            let (mut engine, watched) = setup(tier);
            let source = format!(
                r#"(function(){{
                class C {{
                    {first}
                    static [__watchRoot(Symbol('a'))]=(__checkTemporaryRoots(),1);
                    static [__watchRoot(Symbol('b'))]=(__checkTemporaryRoots(),2);
                }}
                var keys=Object.getOwnPropertySymbols(C);
                if(keys.map(k=>k.description+C[k]).join(',')!=='a1,b2')throw 'static keys';
                return 'ok';
            }})()"#
            );
            assert_eq!(eval(&mut engine, &source), "ok", "{tier:?}");
            assert_eq!(watched.borrow().checks, 3);
            retired(&mut engine, &watched, 2);
        }
    }
}

#[test]
fn class_published_fields_survive_a_static_throw_and_keep_transforms() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"(function(){
            var escaped;
            function transform(){return function(value){__collectRootFixture();return value+1}}
            try {
                class C {
                    #method(){return 40}
                    @transform [__watchRoot(Symbol('field'))]=1;
                    get answer(){return this.#method()}
                    static {escaped=this;__collectRootFixture();throw 73}
                }
            }catch(error){if(error!==73)throw error}
            __collectRootFixture();
            var object=new escaped(),keys=Object.getOwnPropertySymbols(object);
            if(keys.length!==1||keys[0].description!=='field'||object[keys[0]]!==2||object.answer!==40)
                throw 'published fields';
            return 'ok';
        })()"#
            ),
            "ok",
            "{tier:?}"
        );
        assert_eq!(watched.borrow().checks, 3);
        retired(&mut engine, &watched, 1);
    }
}

#[test]
fn class_private_methods_and_field_transforms_are_roots_before_publication() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"(function(){
                    function method(value){return __watchRoot(value)}
                    function initializer(){return __watchRoot(function(value){return value+2})}
                    function checkpoint(value){__checkTemporaryRoots();return value}
                    class C {
                        @method #private(){return 37}
                        @initializer field=3;
                        @checkpoint later(){}
                        answer(){return this.#private()+this.field}
                    }
                    if(new C().answer()!==42)throw 'pending private method/initializer';
                    return 'ok';
                })()"#
            ),
            "ok",
            "{tier:?}"
        );
        assert_eq!(watched.borrow().checks, 1);
        retired(&mut engine, &watched, 2);
    }
}

#[test]
fn object_groupby_retains_converted_symbol_keys_through_callbacks_and_close() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for abrupt in [false, true] {
            let (mut engine, watched) = setup(tier);
            let source = format!(
                r#"(function(){{
                var index=0,closed=0,items={{[Symbol.iterator](){{return {{
                    next(){{return index<3?{{value:index++,done:false}}:{{done:true}}}},
                    return(){{closed++;__checkTemporaryRoots();return {{}}}}
                }}}}}};
                var result;
                try {{
                    result=Object.groupBy(items,function(value){{
                        if(value===0)return {{[Symbol.toPrimitive](){{return __watchRoot(Symbol('group'))}}}};
                        __checkTemporaryRoots();
                        if({abrupt})return {{[Symbol.toPrimitive](){{throw 73}}}};
                        return 'rest';
                    }});
                }}catch(error){{if(!{abrupt}||error!==73)throw error}}
                if({abrupt}){{if(closed!==1)throw 'iterator close'}}
                else {{
                    var keys=Object.getOwnPropertySymbols(result);
                    if(keys.length!==1||keys[0].description!=='group'||result[keys[0]][0]!==0||
                       result.rest.join(',')!=='1,2'||Object.getPrototypeOf(result)!==null)throw 'group keys';
                }}
                return 'ok';
            }})()"#
            );
            assert_eq!(eval(&mut engine, &source), "ok", "{tier:?}/{abrupt}");
            assert_eq!(watched.borrow().checks, 2);
            retired(&mut engine, &watched, 1);
        }
    }
}

#[test]
fn object_groupby_keeps_current_elements_during_key_coercion() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let (mut engine, watched) = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"(function(){
            var index=0,items={[Symbol.iterator](){return {
                next(){return index<3?{value:__watchRoot({id:index++}),done:false}:{done:true}}
            }}};
            var result=Object.groupBy(items,()=>({[Symbol.toPrimitive](){__checkTemporaryRoots();return 'group'}}));
            if(result.group.map(value=>value.id).join(',')!=='0,1,2')throw 'group elements';
            return 'ok';
        })()"#
            ),
            "ok",
            "{tier:?}"
        );
        assert_eq!(watched.borrow().checks, 3);
        retired(&mut engine, &watched, 3);
    }
}
