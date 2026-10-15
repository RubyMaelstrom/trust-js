//! Shared bounded ArrayAccumulation / ArgumentListEvaluation expansion.
//!
//! ECMA-262 #sec-runtime-semantics-arrayaccumulation and
//! #sec-runtime-semantics-argumentlistevaluation use ?IteratorStepValue, not IteratorClose.
//! Snapshot e28783d5fc9dc12b3de905961e2c71410b38a202. Resource limits are host policy;
//! they must be uniform across tiers, and interruption must never run author cleanup.

use crate::interpreter::{Abrupt, Interp, MAX_ARGUMENTS_LEN, MAX_ARRAY_OP_LEN};
use crate::value::{Property, Value};

impl Interp {
    /// IteratorBindingInitialization / IteratorDestructuringAssignmentEvaluation rest drains.
    /// A step failure marks Done; a resource failure after a live value leaves Done false so
    /// the enclosing pattern performs IteratorClose with the original throw completion.
    pub(crate) fn drain_iterator_rest(
        &mut self,
        iterator: &Value,
        next: &Value,
        done: &mut bool,
    ) -> Result<Vec<Value>, Abrupt> {
        self.drain_iterator_rest_bounded(iterator, next, done, MAX_ARRAY_OP_LEN)
    }

    fn drain_iterator_rest_bounded(
        &mut self,
        iterator: &Value,
        next: &Value,
        done: &mut bool,
        limit: usize,
    ) -> Result<Vec<Value>, Abrupt> {
        let mut values = Vec::new();
        if *done {
            return Ok(values);
        }
        self.interrupt_poll_force()?;
        if crate::builtins::inert_array_iterator_remaining(self, iterator, next)
            .is_some_and(|remaining| remaining > limit)
        {
            return Err(self.throw("RangeError", "rest array exceeds engine allocation limit"));
        }
        loop {
            if values.len() & 255 == 0 {
                self.interrupt_poll_force()?;
                self.gc_check()?;
            }
            *done = true;
            let Some(value) = self.iterator_step(iterator, next)? else {
                return Ok(values);
            };
            *done = false;
            if values.len() >= limit {
                return Err(self.throw("RangeError", "rest array exceeds engine allocation limit"));
            }
            values.push(value);
        }
    }

    /// Append to a private fresh literal/argument-list Array with CreateDataProperty semantics.
    /// No prototype setters are invoked. All tiers use the same cumulative allocation bound.
    pub(crate) fn append_array_literal(
        &mut self,
        array: &Value,
        value: Option<Value>,
    ) -> Result<(), Abrupt> {
        let Value::Obj(array) = array else {
            unreachable!("literal builder retains an Array")
        };
        let index = self.array_length(array);
        if index >= MAX_ARRAY_OP_LEN {
            return Err(self.throw("RangeError", "array length exceeds engine limit"));
        }
        let mut object = array.borrow_mut();
        if let Some(value) = value {
            object
                .props
                .insert(index.to_string(), Property::plain(value));
        }
        assert!(
            object
                .props
                .set_value("length", Value::Num(index as f64 + 1.0)),
            "fresh Array has length"
        );
        Ok(())
    }

    pub(crate) fn append_spread_arguments(
        &mut self,
        spread: &Value,
        arguments: &mut Vec<Value>,
    ) -> Result<(), Abrupt> {
        self.expand_spread_bounded(spread, arguments.len(), MAX_ARGUMENTS_LEN, |_, value| {
            arguments.push(value);
            Ok(())
        })
    }

    pub(crate) fn append_argument(
        &mut self,
        arguments: &mut Vec<Value>,
        value: Value,
    ) -> Result<(), Abrupt> {
        if arguments.len() >= MAX_ARGUMENTS_LEN {
            return Err(self.throw(
                "RangeError",
                "argument list exceeds engine allocation limit",
            ));
        }
        arguments.push(value);
        Ok(())
    }

    /// The prefix and every yielded value are live owned roots across safepoints. Callers retain
    /// the output Array/Vec and source, and release any property borrow before invoking this.
    pub(crate) fn expand_spread(
        &mut self,
        spread: &Value,
        prefix: usize,
        append: impl FnMut(&mut Interp, Value) -> Result<(), Abrupt>,
    ) -> Result<(), Abrupt> {
        self.expand_spread_bounded(spread, prefix, MAX_ARRAY_OP_LEN, append)
    }

    fn expand_spread_bounded(
        &mut self,
        spread: &Value,
        mut count: usize,
        limit: usize,
        mut append: impl FnMut(&mut Interp, Value) -> Result<(), Abrupt>,
    ) -> Result<(), Abrupt> {
        self.interrupt_poll_force()?;
        if let Value::Str(text) = spread {
            if crate::builtins::intrinsic_string_iterator_is_unmodified(self) {
                for point in crate::jstr::CodePointIter::new(text) {
                    if count & 255 == 0 {
                        self.interrupt_poll_force()?;
                        self.gc_check()?;
                    }
                    if count >= limit {
                        return Err(
                            self.throw("RangeError", "spread exceeds engine allocation limit")
                        );
                    }
                    let value = if point < 0x80 {
                        Value::Str(crate::jstr::unit_lstr(point as u16))
                    } else {
                        Value::lstr(crate::jstr::from_code_point(point))
                    };
                    append(self, value)?;
                    count += 1;
                }
                return Ok(());
            }
        }
        // Only a proven intrinsic iterator permits using the Array's private length. An Array
        // with an own/custom iterator may have an enormous length but yield just one value.
        if crate::builtins::array_iterator_fast_path_is_safe(self, spread) {
            let Value::Obj(array) = spread else {
                unreachable!()
            };
            let length = self.array_length(array);
            let exceeds = count > limit || length > limit.saturating_sub(count);
            // A getter could shrink length before the nominally huge iterator is exhausted.
            // Prove those effects absent before rejecting from the initial length alone.
            let inert_elements = exceeds
                && self.array_append_unshadowed(array)
                && array
                    .borrow()
                    .props
                    .indexed_properties()
                    .all(|property| !property.accessor());
            if inert_elements {
                return Err(self.throw("RangeError", "spread exceeds engine allocation limit"));
            }
            // Check the bound BEFORE dense_array_snapshot reserves memory. Holes/accessors
            // miss the snapshot and use real IteratorStepValue, re-reading length each step.
            if let Some(values) = (!exceeds)
                .then(|| crate::builtins::dense_array_snapshot(self, spread))
                .flatten()
            {
                for value in values {
                    if count & 255 == 0 {
                        self.interrupt_poll_force()?;
                        self.gc_check()?;
                    }
                    append(self, value)?;
                    count += 1;
                }
                return Ok(());
            }
        }

        let (iterator, next) = self.get_iterator(spread)?;
        loop {
            if count & 255 == 0 {
                self.interrupt_poll_force()?;
                self.gc_check()?;
            }
            let Some(value) = self.iterator_step(&iterator, &next)? else {
                return Ok(());
            };
            // Consume IteratorStepValue before checking capacity: done=true at the exact limit
            // is permitted, and no speculative extra call is made after an overflowing value.
            if count >= limit {
                return Err(self.throw("RangeError", "spread exceeds engine allocation limit"));
            }
            append(self, value)?;
            count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bytecode::Tier, Completion, Engine};

    fn result(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).unwrap() {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}: {source}"),
        }
    }

    fn check(source: &str, expected: &str) {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.set((0, 0)));
            assert_eq!(result(&mut engine, source), expected, "{tier:?}");
            let (vm, native) = crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.get());
            if !matches!(tier, Tier::Interp) {
                assert!(
                    vm + native > 0,
                    "spread fixture must execute compiled Script"
                );
            }
            #[cfg(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(target_os = "linux", target_os = "macos", target_os = "windows")
            ))]
            if matches!(tier, Tier::Jit) {
                assert!(native > 0, "spread fixture must execute native Script");
            }
        }
    }

    #[test]
    fn spread_caps_intrinsic_array_and_all_argument_entry_shapes() {
        check(
            r#"
            function f(){return arguments.length;}
            var o={f}, calls=[
                ()=>[...Array(2**30)],
                ()=>f(...Array(2**30)),
                ()=>o.f(1,...Array(2**30)),
                ()=>f(...[1],...Array(2**30),2),
                function(){'use strict';return f(...Array(2**30));},
                ()=>new f(...Array(2**30))
            ];
            calls.map(fn=>{try{fn();return 'miss'}catch(e){return e.name}}).join('|');
        "#,
            "RangeError|RangeError|RangeError|RangeError|RangeError|RangeError",
        );
    }

    #[test]
    fn spread_preflight_never_skips_custom_iterators_or_index_getter_effects() {
        check(
            r#"
            var custom=Array(2**30);
            custom[Symbol.iterator]=function*(){yield 9;};
            var first=[...custom].join(',');
            var own=Array(2**30);
            Object.defineProperty(own,'0',{get(){own.length=1;return 7;}});
            var second=[...own].join(',');
            var inherited=Array(2**30);
            Object.defineProperty(Array.prototype,'0',{configurable:true,get(){inherited.length=0;return 8;}});
            var third=[...inherited];delete Array.prototype[0];
            var iteratorPrototype=Object.getPrototypeOf([][Symbol.iterator]());
            var saved=iteratorPrototype.next;iteratorPrototype.next=function(){return {done:true};};
            var fourth=[...Array(2**30)].length;iteratorPrototype.next=saved;
            [first,second,third.join(','),fourth].join('|');
        "#,
            "9|7|8|0",
        );
        check(
            r#"
            var a=[1,2,3];Object.defineProperty(a,'0',{get(){a.length=1;return 8;}});
            [...a].join(',');
        "#,
            "8",
        );
    }

    #[test]
    fn spread_step_errors_do_not_close_and_keep_original_throw_identity() {
        for body in [
            "throw marker;",
            "return {get done(){throw marker}};",
            "return {done:false,get value(){throw marker}};",
        ] {
            for expression in [
                "[...src]",
                "f(...src)",
                "holder.f(0,...src)",
                "f(...[],...src,1)",
            ] {
                check(
                    &format!(
                        r#"
                    var marker={{}},closed=0;
                    var src={{[Symbol.iterator](){{return {{next(){{{body}}},get return(){{closed++;throw 'close';}}}};}}}};
                    function f(){{throw 'called'}}var holder={{f}};
                    var same=false;try{{{expression};}}catch(e){{same=e===marker;}}
                    same+'|'+closed;
                "#
                    ),
                    "true|0",
                );
            }
        }
        check(
            r#"
            var closed=0,src={[Symbol.iterator](){return {next:1,get return(){closed++;}}}};
            var name;try{[...src]}catch(e){name=e.name}name+'|'+closed;
        "#,
            "TypeError|0",
        );
    }

    #[test]
    fn spread_preserves_holes_codepoints_argument_order_and_receivers() {
        check(
            r#"
            var trace=[];
            function* part(){trace.push('begin');yield 1;trace.push('end');}
            function item(){trace.push('item');return 2;}
            var holder={marker:9,f(){return this.marker+':'+Array.from(arguments).join(',');}};
            var call=holder.f(...part(),item(),...[3]);
            var a=[1,,...'a😀',,...[4]];
            [call,trace.join(','),a.length,Object.keys(a).join(','),a[3]].join('|');
        "#,
            "9:1,2,3|begin,end,item|6|0,2,3,5|😀",
        );
    }

    #[test]
    fn rest_caps_all_binding_assignment_and_parameter_drains() {
        check(
            r#"
            var calls=[
                ()=>{let [...a]=Array(2**30);},
                ()=>{let a;[...a]=Array(2**30);},
                ()=>{let [,...[a]]=Array(2**30);},
                ()=>{let a;[,...[a]]=Array(2**30);},
                ()=>{function f([...a]){}f(Array(2**30));},
                ()=>{function f(...[...a]){}f(...Array(2**30));}
            ];
            calls.map(fn=>{try{fn();return 'miss'}catch(e){return e.name}}).join('|');
        "#,
            "RangeError|RangeError|RangeError|RangeError|RangeError|RangeError",
        );
    }

    #[test]
    fn rest_preflight_preserves_custom_iteration_and_closes_resource_errors() {
        check(
            r#"
            var short=Array(2**30);short[Symbol.iterator]=function*(){yield 9;};
            let [...first]=short;
            var shrink=Array(2**30);
            Object.defineProperty(shrink,'0',{get(){shrink.length=1;return 7;}});
            var second;[...second]=shrink;
            var closed=0,marker={};
            function source(){
                var it=Array(2**30)[Symbol.iterator]();
                it.return=function(){closed++;throw marker;};
                return {[Symbol.iterator](){return it;}};
            }
            var errors=[];
            try{let [...a]=source();}catch(e){errors.push(e.name);}
            try{let a;[...a]=source();}catch(e){errors.push(e.name);}
            [first.join(','),second.join(','),closed,errors.join(',')].join('|');
        "#,
            "9|7|2|RangeError,RangeError",
        );
    }

    #[test]
    fn rest_step_failures_mark_done_and_reference_order_is_preserved() {
        for body in [
            "throw marker;",
            "return {get done(){throw marker}};",
            "return {done:false,get value(){throw marker}};",
        ] {
            for statement in ["let [...a]=src;", "var a;[...a]=src;"] {
                check(
                    &format!(
                        r#"
                        var marker={{}},closed=0;
                        var src={{[Symbol.iterator](){{return {{next(){{{body}}},return(){{closed++;throw 'close'}}}};}}}};
                        var same=false;try{{{statement}}}catch(e){{same=e===marker;}}
                        same+'|'+closed;
                    "#
                    ),
                    "true|0",
                );
            }
        }
        check(
            r#"
            var trace=[],marker={},holder={set x(v){trace.push('set');throw marker;}};
            function src(){return {[Symbol.iterator](){return {next(){trace.push('next');return {done:true}},return(){trace.push('close');return {}}}}};}
            function base(){trace.push('base');return holder;}
            function fail(){trace.push('fail');throw marker;}
            var same=0;
            try{[...base().x]=src();}catch(e){same+=e===marker;}
            try{[...fail().x]=src();}catch(e){same+=e===marker;}
            same+'|'+trace.join(',');
        "#,
            "2|base,next,set,fail,close",
        );
    }

    #[test]
    fn rest_bounded_drain_publishes_done_before_abrupt_propagation() {
        for (body, success, done, reads) in [
            ("return {done:reads>2,value:reads};", true, true, 3),
            ("return {done:false,value:reads};", false, false, 3),
            ("throw 'step';", false, true, 1),
            ("return {get done(){throw 'done'}};", false, true, 1),
            (
                "return {done:false,get value(){throw 'value'}};",
                false,
                true,
                1,
            ),
        ] {
            let mut engine = Engine::new();
            result(
                &mut engine,
                &format!(
                "var reads=0,src={{[Symbol.iterator](){{return {{next(){{reads++;{body}}}}};}}}};"
            ),
            );
            let env = engine.interp.global_env.clone();
            let source = engine.interp.get_var("src", &env).ok().unwrap();
            let (iterator, next) = engine.interp.get_iterator(&source).ok().unwrap();
            let mut actual_done = false;
            let completion =
                engine
                    .interp
                    .drain_iterator_rest_bounded(&iterator, &next, &mut actual_done, 2);
            assert_eq!(completion.is_ok(), success, "{body}");
            assert_eq!(actual_done, done, "{body}");
            assert_eq!(result(&mut engine, "reads"), reads.to_string());
        }
    }

    #[test]
    fn spread_bounded_prefix_consumes_no_more_than_one_overflowing_value() {
        for (values, expected_reads, expected_len, success) in [(2, 3, 2, true), (3, 3, 2, false)] {
            let mut engine = Engine::new();
            result(
                &mut engine,
                &format!(
                    r#"
                var reads=0,closed=0;
                var src={{[Symbol.iterator](){{return {{next(){{reads++;return {{done:reads>{values},value:reads}}}},return(){{closed++;return {{}}}}}};}}}};
            "#
                ),
            );
            let env = engine.interp.global_env.clone();
            let src = engine.interp.get_var("src", &env).ok().unwrap();
            let mut output = Vec::new();
            let completion = engine.interp.expand_spread_bounded(&src, 2, 4, |_, value| {
                output.push(value);
                Ok(())
            });
            assert_eq!(completion.is_ok(), success);
            assert_eq!(output.len(), expected_len);
            assert_eq!(
                result(&mut engine, "reads+'|'+closed"),
                format!("{expected_reads}|0")
            );
        }
    }

    #[test]
    fn spread_roots_all_yielded_values_across_collection() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let collect = engine.interp.make_native("collect", 0, |i, _, _| {
                i.gc_collect();
                Ok(Value::Undefined)
            });
            engine
                .interp
                .global
                .borrow_mut()
                .props
                .insert("collect", Property::plain(Value::Obj(collect)));
            assert_eq!(
                result(
                    &mut engine,
                    r#"
                var src={[Symbol.iterator](){var n=0;return {next(){
                    collect();if(n===300)return {done:true};var v={n:n++};v.self=v;
                    return {done:false,value:v};
                }}}};
                function f(first,...items){collect();return first.n+':'+items.length+':'+items[298].self.n;}
                var array=[...src];collect();
                let [...binding]=src;var assignment;[...assignment]=src;collect();
                [array.length,array[0].self.n,array[299].n,f(...src),
                    binding[299].self.n,assignment[0].self.n].join('|');
            "#
                ),
                "300|0|299|0:299:299|299|0",
                "{tier:?}"
            );
        }
    }

    #[test]
    fn spread_deadline_never_calls_return_or_author_catch_finally() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            for statement in ["[...src];", "let [...a]=src;", "var a;[...a]=src;"] {
                let mut engine = Engine::new();
                engine.set_tier(tier);
                result(&mut engine, "var closed=0,caught=0,finalized=0;");
                let interrupt = engine.interrupt_handle();
                interrupt.set_deadline(Some(
                    std::time::Instant::now() + std::time::Duration::from_millis(30),
                ));
                let outcome=engine.eval_interruptible(&format!(r#"
                var src={{[Symbol.iterator](){{return {{next(){{return {{done:false,value:1}}}},return(){{closed++;return {{}}}}}}}}}};
                try{{{statement}}}catch(e){{caught++}}finally{{finalized++}}
            "#),false).unwrap();
                assert!(
                    matches!(outcome, crate::ExecutionOutcome::Interrupted { .. }),
                    "{tier:?}"
                );
                interrupt.set_deadline(None);
                assert_eq!(
                    result(&mut engine, "closed+'|'+caught+'|'+finalized"),
                    "0|0|0"
                );
            }
        }
    }
}
