use super::*;
use crate::{bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

fn child(name: &str) -> bool {
    const CHILD: &str = "TRUST_JS_TEST_LOOP_FACTS_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("jit::optimizing::loop_facts_tests::{name}"),
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("TRUST_JS_OPT_JIT", "0")
        .env("TRUST_JS_OPT_JIT_OSR", "1")
        .env("TRUST_JS_OPT_JIT_OSR_AT", "16")
        .env("TRUST_JS_OPT_JIT_LOOP_FACTS", "1")
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
    true
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn subject_chunk(engine: &mut Engine) -> Rc<Chunk> {
    let env = engine.interp.global_env.clone();
    let subject = engine
        .interp
        .get_var("subject", &env)
        .ok()
        .expect("subject");
    let object = subject.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone()
}

fn entered(engine: &mut Engine) -> Rc<Chunk> {
    assert!(
        loop_facts::TEST_PLANS.with(|count| count.get()) > 0,
        "must build a complete loop proof"
    );
    let chunk = subject_chunk(engine);
    let state = chunk.optimizing_loop.get().expect("loop metadata");
    assert!(state.entries.get() > 0, "must execute continuation");
    assert!(state.code.get().flatten().is_some(), "live native proof");
    chunk
}

// Separate children keep process-wide compiler controls independent. Each
// variant requires a real native entry; scalar-off and ordinary entry are
// controls for the publication contract, rather than fallback substitutes.
fn read_publication_children(name: &str) -> bool {
    const CHILD: &str = "TRUST_JS_TEST_LOOP_READ_PUBLICATION_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return false;
    }
    for variant in ["scalar-on", "scalar-off", "ordinary"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("jit::optimizing::loop_facts_tests::{name}"),
                "--nocapture",
            ])
            .env(CHILD, variant)
            .env(
                "TRUST_JS_OPT_JIT",
                if variant == "ordinary" { "1" } else { "0" },
            )
            .env(
                "TRUST_JS_OPT_JIT_OSR",
                if variant == "ordinary" { "0" } else { "1" },
            )
            .env("TRUST_JS_OPT_JIT_OSR_AT", "16")
            .env("TRUST_JS_OPT_JIT_LOOP_FACTS", "1")
            .env(
                "TRUST_JS_OPT_JIT_LOOP_SCALARS",
                if variant == "scalar-off" { "0" } else { "1" },
            )
            .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution")
            .env("TRUST_JS_INLINE_AT", "0")
            .env_remove("TRUST_JS_OPT_JIT_DEOPT_AT")
            .output()
            .unwrap();
        eprintln!(
            "{variant}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{variant}");
    }
    true
}

fn read_publication_record(engine: &mut Engine) -> Rc<diagnostics::Code> {
    let variant = std::env::var("TRUST_JS_TEST_LOOP_READ_PUBLICATION_CHILD").unwrap();
    let chunk = subject_chunk(engine);
    let code = if variant == "ordinary" {
        let code = chunk.jit.get().flatten().expect("ordinary native body");
        assert!(
            code.optimizing_diagnostics.is_some(),
            "ordinary control must use optimizing lowering"
        );
        assert_eq!(chunk.optimizing_misses.get(), 0);
        code
    } else {
        if variant == "scalar-on" {
            entered(engine);
        }
        let state = chunk.optimizing_loop.get().expect("continuation metadata");
        assert!(state.entries.get() > 0);
        state
            .code
            .get()
            .flatten()
            .expect("continuation native body")
    };
    let record = code.optimizing_diagnostics.as_ref().unwrap().clone();
    assert!(
        record.counters[0].get() > 0,
        "actual instrumented native entry"
    );
    record
}

#[test]
fn loop_fields_publish_before_unmodeled_name_this_capture_and_element_reads() {
    if read_publication_children(
        "loop_fields_publish_before_unmodeled_name_this_capture_and_element_reads",
    ) {
        return;
    }
    let fixtures = [
        "globalThis.x=0;function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=x;}return seen}subject(globalThis)+':'+globalThis.x",
        "function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=this.x;}return seen}var o={x:0};subject.call(o,o)+':'+o.x",
        "function subject(o){var p=o,seen=0;for(var k=0;k<100;k++){o.x=o.x+1;if(k>=80)seen=p.x;}return seen}var o={x:0};subject(o)+':'+o.x",
        "function factory(obj){return function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=obj.x;}return seen}}var o={x:0},subject=factory(o);subject(o)+':'+o.x",
        "function subject(o){var p=o,key='x',seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=p[key];}return seen}var o={x:0};subject(o)+':'+o.x",
    ];
    for (fixture, source) in fixtures.into_iter().enumerate() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = engine(tier);
            assert_eq!(eval(&mut engine, source), "100:100", "{tier:?}: {source}");
            if matches!(tier, Tier::Jit) {
                let record = read_publication_record(&mut engine);
                assert_eq!(
                    record.counters[diagnostics::VM_RECOVERY].get(),
                    0,
                    "no recovery substitutes for the native read"
                );
                let sites = record.site_stats();
                if matches!(fixture, 1 | 2) {
                    let chunk = subject_chunk(&mut engine);
                    assert!(
                        sites.iter().any(|(pc, kind, count)| {
                            *kind == "data_property_read"
                                && *count > 0
                                && match (&chunk.jit_ops()[*pc], fixture) {
                                    (Op::GetPropThis(name, _), 1) => chunk.jit_name(*name) == "x",
                                    (Op::GetPropLocal(slot, name, _), 2) => {
                                        chunk.jit_name(*name) == "x"
                                            && chunk.jit_slot_names()[*slot as usize].as_ref() == "p"
                                    }
                                    _ => false,
                                }
                        }),
                        "must commit the unmodeled alias read through native property lowering: {sites:?}"
                    );
                }
                eprintln!(
                    "alias fixture={fixture} entries={} sites={sites:?}",
                    record.counters[0].get()
                );
            }
        }
    }
}

#[test]
fn loop_read_publication_keeps_getter_throw_effects_and_error_position() {
    if read_publication_children(
        "loop_read_publication_keeps_getter_throw_effects_and_error_position",
    ) {
        return;
    }
    const SOURCE:&str="var reads=0;function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=this.read;}return seen}var o={x:0,get read(){this.x++;reads++;if(this.x===86)throw Error('stop');return this.x}};var message,trace;try{subject.call(o,o)}catch(error){message=error.message;trace=error.stack};message+':'+reads+':'+o.x";
    let getter_column = SOURCE.find("Error('stop')").unwrap() + 1;
    let read_column = SOURCE.find("this.read").unwrap() + "this.".len() + 1;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, SOURCE), "stop:43:86");
        if matches!(tier, Tier::Jit) {
            let record = read_publication_record(&mut engine);
            assert_eq!(
                record.counters[diagnostics::VM_RECOVERY].get(),
                0,
                "throw keeps the existing checked unwind path"
            );
        }
        let trace = eval(&mut engine, "trace");
        let frames: Vec<_> = trace.lines().collect();
        assert_eq!(frames[0], "Error: stop");
        assert_eq!(
            frames[1].trim(),
            format!("at get read (<anonymous>:1:{getter_column})"),
            "{tier:?}: {trace}"
        );
        assert_eq!(
            frames[2].trim(),
            format!("at subject (<anonymous>:1:{read_column})"),
            "{tier:?}: {trace}"
        );
        eprintln!(
            "getter throw {tier:?}: {} | {}",
            frames[1].trim(),
            frames[2].trim()
        );
    }
}

#[test]
fn loop_getter_descriptor_miss_recovers_after_the_single_completed_read() {
    if read_publication_children(
        "loop_getter_descriptor_miss_recovers_after_the_single_completed_read",
    ) {
        return;
    }
    const SOURCE:&str="var reads=0;function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=this.read;}return seen}var o={x:0,get read(){reads++;if(reads===40)Object.defineProperty(this,'x',{value:this.x+100,writable:false,configurable:true});return this.x}};subject.call(o,o)+':'+reads+':'+o.x";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(eval(&mut engine, SOURCE), "140:100:140");
        if matches!(tier, Tier::Jit) {
            let record = read_publication_record(&mut engine);
            // Disabling scalar registers retains live field-proof validation.
            // Both continuation controls recover after the descriptor change;
            // ordinary lowering has no continuation field proof.
            let expected = u64::from(
                std::env::var("TRUST_JS_TEST_LOOP_READ_PUBLICATION_CHILD").as_deref()
                    != Ok("ordinary"),
            );
            assert_eq!(
                record.counters[diagnostics::VM_RECOVERY].get(),
                expected,
                "scalar descriptor change recovers once after its getter result"
            );
        }
    }
}

#[test]
fn loop_facts_reuse_numeric_fields_and_borrowed_receiver_across_polls() {
    if child("loop_facts_reuse_numeric_fields_and_borrowed_receiver_across_polls") {
        return;
    }
    let mut engine = engine(Tier::Jit);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var prefixes=0;
        function subject(n,object){prefixes++;var sum=0;for(var k=0;k<n;k++){
            sum+=object.x+object.y;k&1?object.x=1:object.x=2;
        }return sum}
        subject(50000,{x:1,y:4})+':'+prefixes;
    "#
        ),
        "275000:1"
    );
    entered(&mut engine);
    engine.interp.gc_collect();
    assert_eq!(
        eval(&mut engine, "subject(100,{x:1,y:4})+':'+prefixes"),
        "550:2"
    );
}

#[test]
fn loop_facts_revalidate_fresh_instances_accessors_proxy_and_readonly_writes() {
    if child("loop_facts_revalidate_fresh_instances_accessors_proxy_and_readonly_writes") {
        return;
    }
    let mut engine = engine(Tier::Jit);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var prefixes=0,gets=0;
        function subject(n,object){'use strict';prefixes++;var sum=0;
            for(var k=0;k<n;k++){sum+=object.x+object.y;object.x=k&1}
            return sum}
        subject(100,{x:1,y:4});
    "#
        ),
        "450"
    );
    entered(&mut engine);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var o={x:1,y:4};Object.defineProperty(o,'y',{get(){gets++;return 4}});
        subject(100,o)+':'+gets+':'+prefixes;
    "#
        ),
        "450:100:2"
    );
    assert_eq!(
        eval(
            &mut engine,
            r#"
        gets=0;var p=new Proxy({x:1,y:4},{get(t,k,r){gets++;return Reflect.get(t,k,r)}});
        subject(100,p)+':'+gets+':'+prefixes;
    "#
        ),
        "450:200:3"
    );
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var ro={x:1,y:4};Object.defineProperty(ro,'x',{writable:false});
        try{subject(100,ro)}catch(e){e.name+':'+ro.x+':'+prefixes}
    "#
        ),
        "TypeError:1:4"
    );
    assert_eq!(
        eval(&mut engine, "subject(100,{x:9,y:8})+':'+prefixes"),
        "858:5"
    );
}

#[test]
fn loop_facts_preserve_aliasing_and_late_nonnumeric_values() {
    if child("loop_facts_preserve_aliasing_and_late_nonnumeric_values") {
        return;
    }
    let source = r#"
        function subject(n,a,b){var sum=0;for(var k=0;k<n;k++){
            sum+=a.x+b.y;a.x=k&7;b.y=k&3;
        }return sum}
        var a={x:1,y:4};var warm=subject(100,a,a);
        var b={x:'s',y:4};var later=subject(30,b,b);
        JSON.stringify([warm,later,b.x,b.y]);
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    entered(&mut engine);
}

#[test]
fn loop_facts_reject_receiver_replacement() {
    if child("loop_facts_reject_receiver_replacement") {
        return;
    }
    let source = r#"
        var calls=0,gets=0;
        function change(o,k){calls++;if(k===25)Object.defineProperty(o,'y',{get(){gets++;return 7}})}
        function subject(n,o){var sum=0;for(var k=0;k<n;k++){
            change(o,k);sum+=o.x+o.y;
            if(k===50)o={x:3,y:2};
        }return sum}
        [subject(100,{x:1,y:4}),calls,gets].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    assert_eq!(loop_facts::TEST_PLANS.with(|count| count.get()), 0);
}

#[test]
fn loop_effects_publish_callbacks_reload_values_and_keep_return_owners() {
    if child("loop_effects_publish_callbacks_reload_values_and_keep_return_owners") {
        return;
    }
    let source = r#"
        function subject(n,o,callback){var sum=0;for(var k=0;k<n;k++){
            o.x=o.x+1;callback(o,k);o.y=o.y+2;sum+=o.x+o.y;
        }return sum}
        var calls=0,seen=0,returns=[];
        function callback(o,k){calls++;seen+=o.x;o.x=k&7;
            var r={v:k};if(k===31)returns.push(r);return r}
        var o={x:1,y:4};var result=subject(300,o,callback);
        [result,o.x,o.y,calls,seen,returns[0].v].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    entered(&mut engine);
    engine.interp.gc_collect();
    assert_eq!(eval(&mut engine, "returns[0].v"), "31");
}

#[test]
fn loop_effects_resume_after_descriptor_change_without_replaying_call() {
    if child("loop_effects_resume_after_descriptor_change_without_replaying_call") {
        return;
    }
    let source = r#"
        function subject(n,o,callback){var sum=0;for(var k=0;k<n;k++){
            o.x=o.x+1;callback(o,k);sum+=o.x+o.y;
        }return sum}
        var calls=0,gets=0,seen=0;
        function callback(o,k){calls++;seen+=o.x;if(k===31){
            Object.defineProperty(o,'y',{get(){gets++;return 7}});
        }return {live:k}}
        var o={x:1,y:4};var result=subject(100,o,callback);
        [result,o.x,calls,gets,seen].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    let chunk = entered(&mut engine);
    assert!(chunk.optimizing_loop.get().unwrap().misses.get() > 0);
    engine.interp.gc_collect();
}

#[test]
fn loop_effects_publish_before_callback_throw_and_numeric_coercion() {
    if child("loop_effects_publish_before_callback_throw_and_numeric_coercion") {
        return;
    }
    let source = r#"
        function subject(n,o,values,callback){var sum=0;for(var k=0;k<n;k++){
            o.x=o.x+1;sum+=o.x+o.y;sum+=values[k&1];callback(o,k);
        }return sum}
        var calls=0,conversions=0,seen=0;
        var o={x:1,y:4};
        var value={valueOf(){conversions++;seen+=o.x;o.y=6;return 2}};
        function callback(o,k){calls++;if(k===31)throw {v:o.x}}
        var thrown;try{subject(100,o,[1,value],callback)}catch(e){thrown=e.v}
        [thrown,o.x,o.y,calls,conversions,seen].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    entered(&mut engine);
    engine.interp.gc_collect();
}

#[test]
fn loop_effects_array_length_elements_alias_growth_and_getter_reentry() {
    if child("loop_effects_array_length_elements_alias_growth_and_getter_reentry") {
        return;
    }
    let source = r#"
        function subject(n,a,alias,o){var sum=0;for(var k=0;k<n;k++){
            o.x=o.x+1;sum+=a.length;alias[k]=k;sum+=a.length+a[0];
        }return sum}
        var a=[2],o={x:1};var first=subject(100,a,a,o);
        var gets=0,b=[2],p={x:1};
        Object.defineProperty(b,'0',{get(){gets++;p.x+=2;return 2}});
        var other=[];var second=subject(100,b,other,p);
        [first,a.length,o.x,second,b.length,other.length,p.x,gets].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    entered(&mut engine);
    engine.interp.gc_collect();
}

#[test]
fn loop_effects_general_native_writes_refresh_aliased_scalar_field() {
    if child("loop_effects_general_native_writes_refresh_aliased_scalar_field") {
        return;
    }
    let source = r#"
        function subject(n,o,alias,key){var sum=0;for(var k=0;k<n;k++){
            o.x=o.x+1;alias.x=k&7;sum+=o.x+o.y;alias[key]=k&3;
        }return sum}
        var o={x:1,y:4};var first=subject(100,o,o,'x');
        var other={x:4};var second=subject(100,o,other,'x');
        [first,second,o.x,other.x].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    entered(&mut engine);
}

#[test]
fn loop_effects_array_length_is_a_guarded_field_in_both_storage_modes() {
    if child("loop_effects_array_length_is_a_guarded_field_in_both_storage_modes") {
        return;
    }
    let source = r#"
        function subject(n,a){var sum=0;for(var k=0;k<n;k++){
            sum+=a.length;a[k&7]=k;sum+=a.length+a[0];
        }return sum}
        var a=[2];var first=subject(100,a);
        var b=[1,2,3];var second=subject(100,b);
        [first,second,a.length,b.length,a[0],b[0]].join(':');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    entered(&mut engine);
}

#[test]
fn loop_scalars_publish_breaks_and_reject_overlapping_writable_field_aliases() {
    if child("loop_scalars_publish_breaks_and_reject_overlapping_writable_field_aliases") {
        return;
    }
    let source = r#"
        function subject(n,a,b){var sum=0;for(var k=0;k<n;k++){
            a.x=a.x+1;b.x=b.x+2;sum+=a.x+b.x;if(k===31)break;
        }return sum+':'+a.x+':'+b.x}
        var a={x:1},b={x:3};var first=subject(100,a,b);
        var both={x:1};var second=subject(100,both,both);
        [first,second,both.x].join('|');
    "#;
    let reference = eval(&mut engine(Tier::Interp), source);
    let mut engine = engine(Tier::Jit);
    assert_eq!(eval(&mut engine, source), reference);
    let chunk = entered(&mut engine);
    let misses = chunk.optimizing_loop.get().unwrap().misses.get();
    if std::env::var("TRUST_JS_OPT_JIT_LOOP_SCALARS").as_deref() == Ok("0") {
        assert_eq!(
            misses, 0,
            "eager memory operations support writable aliases"
        );
    } else {
        assert!(
            misses > 0,
            "overlapping writable addresses must reject scalar promotion"
        );
    }
}

#[test]
fn loop_scalars_publish_before_host_interruption() {
    if child("loop_scalars_publish_before_host_interruption") {
        return;
    }
    let mut engine = engine(Tier::Jit);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        function subject(n,o){var sum=0;for(var k=0;k<n;k++){
            o.x=o.x+1;o.y=o.x;sum+=o.y;
        }return sum}
        subject(100,{x:0,y:0});
    "#
        ),
        "5050"
    );
    entered(&mut engine);
    eval(&mut engine, "var interrupted={x:0,y:0}");
    let interrupt = engine.interrupt_handle();
    interrupt.set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(30),
    ));
    let outcome = engine
        .eval_interruptible("subject(1000000000,interrupted)", false)
        .expect("parse");
    assert!(matches!(
        outcome,
        crate::ExecutionOutcome::Interrupted { .. }
    ));
    interrupt.set_deadline(None);
    assert_eq!(
        eval(
            &mut engine,
            "interrupted.x>0 && interrupted.x===interrupted.y"
        ),
        "true"
    );
}

#[test]
fn native_bitwise_numbers_reduce_full_ieee_range_and_shift_modulo_32() {
    let source = r#"
        function subject(){var out=[];for(var i=0;i<numbers.length;i++)
          for(var j=0;j<numbers.length;j++){var a=numbers[i],b=numbers[j];
            out.push(a&b,a|b,a^b,a<<b,a>>b,a>>>b)}
          return out.join(',')}
        var numbers=[0,-0,0.5,-0.9,2147483647,2147483648,4294967295,4294967296,
          4294967297,9007199254740991,9223372036854775808,9223372036854777856,
          -9223372036854777856,18446744073709555712,19342813113834066795298816,
          NaN,Infinity,-Infinity];
    "#;
    let mut reference = engine(Tier::Interp);
    eval(&mut reference, source);
    let expected = eval(&mut reference, "subject()");
    super::tests::check_warmed(source, &expected);
}

#[test]
fn native_bitwise_slow_conversions_keep_order_bigint_and_dirty_owners() {
    super::tests::check_warmed(
        r#"
        function subject(){trace=[];var owner={value:41};var first=a&b;
          var second=9n<<2n;var mixed=false;
          try{second>>>1n}catch(e){mixed=e instanceof TypeError}
          return [first,String(second),mixed,trace.join(''),owner.value].join(':')}
        var trace=[];
        var a={valueOf(){trace.push('a');return 7}},
            b={valueOf(){trace.push('b');return 2}};
    "#,
        "2:36:true:ab:41",
    );
}
