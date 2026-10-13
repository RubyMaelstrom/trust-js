//! Direct ownership and stack-boundary tests for the dedicated named-property JIT helper.
//! These enter `jit_get_prop` with canonical packed operands so a VM-level equivalence test
//! cannot hide a moved-out owner, a widened return, or a partially committed stack result.

use super::*;
use crate::{jit::JitCtx, Completion, Engine};
use std::mem::MaybeUninit;

#[derive(Clone, Copy, Debug)]
enum ReadKind {
    Stack,
    This,
    Local,
    Method,
}

impl ReadKind {
    fn is_method(self) -> bool {
        matches!(self, Self::Method)
    }
}

#[derive(Clone, Copy)]
struct ReadSite {
    pc: usize,
    slot: u16,
    name: u32,
    cache: u32,
}

fn read_chunk() -> (Rc<Chunk>, ReadSite) {
    let parsed =
        crate::parser::parse_script("function read(receiver){return receiver.value;}", false)
            .ok()
            .expect("named-read fixture parses");
    let crate::ast::Stmt::FuncDecl(function) = &parsed[0] else {
        panic!("named-read fixture is a function declaration")
    };
    let chunk = compile(function).expect("named-read fixture compiles");
    let site = chunk
        .ops
        .iter()
        .enumerate()
        .find_map(|(pc, op)| match *op {
            Op::GetPropLocal(slot, name, cache) => Some(ReadSite {
                pc,
                slot,
                name,
                cache,
            }),
            Op::GetProp(name, cache) => Some(ReadSite {
                pc,
                slot: 0,
                name,
                cache,
            }),
            _ => None,
        })
        .expect("fixture contains a named property read");
    assert_eq!(chunk.names[site.name as usize].as_ref(), "value");
    (chunk, site)
}

fn use_kind(chunk: &mut Rc<Chunk>, site: ReadSite, kind: ReadKind) {
    let chunk = Rc::get_mut(chunk).expect("the test owns the fixture chunk");
    chunk.ops[site.pc] = match kind {
        ReadKind::Stack => Op::GetProp(site.name, site.cache),
        ReadKind::This => Op::GetPropThis(site.name, site.cache),
        ReadKind::Local => Op::GetPropLocal(site.slot, site.name, site.cache),
        ReadKind::Method => Op::GetMethod(site.name, site.cache),
    };
}

struct Invocation {
    stack: Box<[MaybeUninit<PackedValue>; 4]>,
    initialized: [bool; 4],
    live: usize,
    prefix_len: usize,
    error: Option<Abrupt>,
}

impl Invocation {
    fn value(&mut self, index: usize) -> Value {
        assert!(index < self.live && self.initialized[index]);
        self.initialized[index] = false;
        // SAFETY: every live stack position is initialized by the helper, and this bit is
        // cleared before the owner is moved out so Drop cannot release it a second time.
        unsafe { self.stack[index].assume_init_read().into_value() }
    }

    fn packed(&self, index: usize) -> &PackedValue {
        assert!(index < self.live && self.initialized[index]);
        // SAFETY: the helper's returned SP covers precisely its still-owned packed cells.
        unsafe { self.stack[index].assume_init_ref() }
    }

    fn take_values(&mut self) -> Vec<Value> {
        (0..self.live).map(|index| self.value(index)).collect()
    }

    fn take_prefix(&mut self) -> Vec<Value> {
        (0..self.prefix_len)
            .map(|index| self.value(index))
            .collect()
    }

    fn drop_slot(&mut self, index: usize) {
        assert!(index < self.live && self.initialized[index]);
        self.initialized[index] = false;
        // SAFETY: this position still contains its one live packed owner.
        unsafe { self.stack[index].assume_init_drop() };
    }
}

impl Drop for Invocation {
    fn drop(&mut self) {
        for index in 0..self.initialized.len() {
            if self.initialized[index] {
                self.initialized[index] = false;
                // SAFETY: the flag tracks all packed operands/results that remain initialized.
                unsafe { self.stack[index].assume_init_drop() };
            }
        }
    }
}

/// Enter the exact checked helper called by generated named reads. `receiver` is moved into the
/// operand stack, local, or `this` binding according to the opcode. The returned stack remains
/// packed so callers can run collection before widening the property result.
fn invoke(
    engine: &mut Engine,
    chunk: &Rc<Chunk>,
    site: ReadSite,
    kind: ReadKind,
    receiver: Value,
    prefix: Value,
) -> Invocation {
    let mut stack: Box<[MaybeUninit<PackedValue>; 4]> =
        Box::new(std::array::from_fn(|_| MaybeUninit::uninit()));
    let base = stack.as_mut_ptr().cast::<PackedValue>();
    let prefix_len = 1;
    unsafe { base.write(PackedValue::pack(prefix)) };

    let mut slots: Vec<PackedValue> = (0..chunk.n_slots)
        .map(|_| PackedValue::pack(Value::Undefined))
        .collect();
    let mut this_value = Value::Undefined;
    let mut stack_len = prefix_len;
    match kind {
        ReadKind::Stack | ReadKind::Method => {
            unsafe { base.add(stack_len).write(PackedValue::pack(receiver)) };
            stack_len += 1;
        }
        ReadKind::Local => {
            slots[site.slot as usize] = PackedValue::pack(receiver);
        }
        ReadKind::This => this_value = receiver,
    }

    let env = engine.interp.global_env.clone();
    let interp = &mut *engine.interp;
    let mut ctx = JitCtx {
        helpers: interp.jit_helpers.as_ptr(),
        stack_base: base,
        final_sp: base,
        slots: slots.as_mut_ptr(),
        inline_ic_safe: &interp.inline_ic_safe as *const _ as *const u8,
        env_raw: Rc::as_ptr(&env).cast(),
        this_raw: std::ptr::null(),
        global_body: std::ptr::null(),
        genv: Rc::as_ptr(&env) as usize,
        interp,
        chunk: Rc::as_ptr(chunk),
        this_val: this_value,
        n_slots: slots.len(),
        handlers: Vec::new(),
        handler_floor: 0,
        code_base: std::ptr::null(),
        pc_offsets: std::ptr::null(),
        error: None,
        ret: PackedValue::pack(Value::Undefined),
        env_parent_raw: std::ptr::null(),
        opstat_enabled: false,
        callstat_enabled: false,
        inline_recompile_at: inline_recompile_at(),
        live_objects: crate::value::live_objects_ptr(&interp.gc_heap),
        activation: None,
        resume_activation: std::ptr::null_mut(),
        resume_pc: 0,
        resume_step: None,
        references_raw: std::ptr::null_mut(),
    };
    ctx.this_raw = &ctx.this_val;

    let result = unsafe { jit_get_prop(&mut ctx, site.pc as u32, base.add(stack_len)) };
    assert_eq!(result.flag != 0, ctx.error.is_some());
    let live = unsafe { result.sp.offset_from(base) as usize };
    let expected = if ctx.error.is_some() {
        prefix_len + usize::from(kind.is_method())
    } else {
        prefix_len + 1 + usize::from(kind.is_method())
    };
    assert_eq!(
        live,
        expected,
        "wrong canonical stack extent for {kind:?}, error={}",
        ctx.error.is_some()
    );
    let error = ctx.error.take();
    drop(ctx);
    drop(slots);
    Invocation {
        stack,
        initialized: std::array::from_fn(|index| index < live),
        live,
        prefix_len,
        error,
    }
}

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture source parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn binding(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env.clone();
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("fixture binding")
}

fn same_js_value(engine: &mut Engine, expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Num(expected), Value::Num(actual)) => {
            (expected.is_nan() && actual.is_nan()) || expected.to_bits() == actual.to_bits()
        }
        _ => engine.interp.strict_equals(expected, actual),
    }
}

#[test]
fn named_read_helper_round_trips_every_value_tag_and_method_receiver() {
    let mut engine = Engine::new();
    evaluate(&mut engine, "var namedToken=Symbol('named-read');");
    let mut values = vec![
        Value::Undefined,
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Num(-0.0),
        Value::Num(f64::from_bits(0xffff_ffff_ffff_ffff)),
        Value::Num(f64::INFINITY),
        Value::Num(-1.25),
        Value::Str("named result".into()),
        binding(&mut engine, "namedToken"),
        Value::Obj(crate::value::Object::new(None)),
        Value::BigInt(crate::bigint::JsBigInt::from_i128(i128::MAX)),
    ];

    let (mut chunk, site) = read_chunk();
    for kind in [
        ReadKind::Stack,
        ReadKind::This,
        ReadKind::Local,
        ReadKind::Method,
    ] {
        use_kind(&mut chunk, site, kind);
        for (tag, expected) in values.iter().enumerate() {
            let receiver = crate::value::Object::new(None);
            receiver
                .borrow_mut()
                .props
                .insert("value", crate::value::Property::plain(expected.clone()));
            let receiver_value = Value::Obj(receiver.clone());
            let mut outcome = invoke(
                &mut engine,
                &chunk,
                site,
                kind,
                receiver_value,
                Value::Str("prefix".into()),
            );
            assert!(outcome.error.is_none());
            let values = outcome.take_values();
            assert!(matches!(&values[0], Value::Str(s) if s.as_str() == "prefix"));
            let actual = values.last().expect("named read pushes its result");
            assert!(
                same_js_value(&mut engine, expected, actual),
                "wrong named result for {kind:?} at value tag {tag}"
            );
            if kind.is_method() {
                assert!(matches!(&values[1], Value::Obj(value) if Rc::ptr_eq(value, &receiver)));
                assert_eq!(values.len(), 3);
            } else {
                assert_eq!(values.len(), 2);
            }
        }
    }
    values.clear();
}

#[test]
fn named_read_packed_result_and_prefix_are_last_owners_across_collection() {
    let mut engine = Engine::new();
    let (mut chunk, site) = read_chunk();
    use_kind(&mut chunk, site, ReadKind::Local);

    let result = crate::value::Object::new(None);
    let weak_result = Rc::downgrade(&result);
    let receiver = crate::value::Object::new(None);
    let weak_receiver = Rc::downgrade(&receiver);
    receiver.borrow_mut().props.insert(
        "value",
        crate::value::Property::plain(Value::Obj(result.clone())),
    );
    let prefix = crate::value::Object::new(None);
    let weak_prefix = Rc::downgrade(&prefix);
    let mut outcome = invoke(
        &mut engine,
        &chunk,
        site,
        ReadKind::Local,
        Value::Obj(receiver.clone()),
        Value::Obj(prefix.clone()),
    );
    drop(result);
    drop(receiver);
    drop(prefix);
    assert!(outcome.error.is_none());
    assert!(
        weak_receiver.upgrade().is_none(),
        "the local slot was released"
    );

    engine.interp.gc_collect();
    assert!(
        weak_prefix.upgrade().is_some(),
        "the packed prefix remains rooted"
    );
    assert!(
        weak_result.upgrade().is_some(),
        "the packed result owns its value after the receiver and property are gone"
    );
    assert!(outcome.packed(0).with_object(|_| ()).is_some());
    assert!(outcome.packed(1).with_object(|_| ()).is_some());

    outcome.drop_slot(0);
    outcome.drop_slot(1);
    engine.interp.gc_collect();
    assert!(weak_prefix.upgrade().is_none());
    assert!(weak_result.upgrade().is_none());
}

fn collect_named_read(
    i: &mut crate::interpreter::Interp,
    _: Value,
    _: &[Value],
) -> Result<Value, Value> {
    i.gc_collect();
    Ok(Value::Undefined)
}

fn error_name(engine: &mut Engine, error: &Value) -> String {
    match engine
        .interp
        .get_member(error, "name")
        .ok()
        .expect("error name read")
    {
        Value::Str(name) => name.as_str().to_owned(),
        _ => panic!("error name is not a string"),
    }
}

fn assert_prefix(outcome: &mut Invocation, expected_prefix: &Value, engine: &mut Engine) {
    let mut prefix = outcome.take_prefix();
    assert!(same_js_value(engine, expected_prefix, &prefix[0]));
    prefix.clear();
}

#[test]
fn named_read_errors_preserve_prefix_tdz_order_and_release_last_receiver_owner() {
    let mut engine = Engine::new();
    engine.interp.def_method(
        &engine.interp.global.clone(),
        "collectNamedRead",
        0,
        collect_named_read,
    );
    evaluate(
        &mut engine,
        "var namedSentinel={}; var namedGetter=function(){collectNamedRead();throw namedSentinel;};",
    );
    let getter = binding(&mut engine, "namedGetter");
    let sentinel = binding(&mut engine, "namedSentinel");
    let (mut chunk, site) = read_chunk();

    // A throwing accessor reenters the engine and collects. Stack/local/this receivers are
    // consumed or frame-owned through the call; GetMethod deliberately leaves the receiver
    // packed until the caller's exceptional unwind drops it.
    for kind in [
        ReadKind::Stack,
        ReadKind::This,
        ReadKind::Local,
        ReadKind::Method,
    ] {
        use_kind(&mut chunk, site, kind);
        let receiver = crate::value::Object::new(None);
        receiver.borrow_mut().props.insert(
            "value",
            crate::value::Property::accessor_prop(Some(getter.clone()), None, true, true),
        );
        let weak_receiver = Rc::downgrade(&receiver);
        let receiver_value = Value::Obj(receiver.clone());
        drop(receiver);
        let prefix = crate::value::Object::new(None);
        let weak_prefix = Rc::downgrade(&prefix);
        let expected_prefix = Value::Obj(prefix.clone());
        let mut outcome = invoke(
            &mut engine,
            &chunk,
            site,
            kind,
            receiver_value,
            Value::Obj(prefix.clone()),
        );
        drop(prefix);
        let Some(Abrupt::Throw(error)) = outcome.error.as_ref() else {
            panic!("expected the accessor's exact thrown value for {kind:?}");
        };
        assert!(engine.interp.strict_equals(error, &sentinel));
        assert_prefix(&mut outcome, &expected_prefix, &mut engine);
        drop(expected_prefix);
        engine.interp.gc_collect();
        assert!(
            weak_prefix.upgrade().is_none(),
            "the consumed prefix was released"
        );
        assert_eq!(
            weak_receiver.upgrade().is_some(),
            kind.is_method(),
            "only GetMethod keeps its borrowed receiver until exceptional unwind"
        );
        drop(outcome);
        assert!(
            weak_receiver.upgrade().is_none(),
            "the receiver owner was released"
        );
    }

    // Nullish reads throw before any property access while retaining only the original prefix;
    // the method form keeps its borrowed receiver for the unwinder to discard.
    let null_prefix = crate::value::Object::new(None);
    let null_prefix_value = Value::Obj(null_prefix.clone());
    for kind in [
        ReadKind::Stack,
        ReadKind::This,
        ReadKind::Local,
        ReadKind::Method,
    ] {
        use_kind(&mut chunk, site, kind);
        let mut outcome = invoke(
            &mut engine,
            &chunk,
            site,
            kind,
            Value::Null,
            null_prefix_value.clone(),
        );
        assert_prefix(&mut outcome, &null_prefix_value, &mut engine);
        let Some(Abrupt::Throw(error)) = outcome.error.as_ref() else {
            panic!("nullish property read must throw for {kind:?}");
        };
        assert_eq!(error_name(&mut engine, error), "TypeError");
    }

    use_kind(&mut chunk, site, ReadKind::Local);
    let mut outcome = invoke(
        &mut engine,
        &chunk,
        site,
        ReadKind::Local,
        Value::Empty,
        null_prefix_value.clone(),
    );
    assert_prefix(&mut outcome, &null_prefix_value, &mut engine);
    let Some(Abrupt::Throw(error)) = outcome.error.as_ref() else {
        panic!("an empty local must remain in the temporal dead zone");
    };
    assert_eq!(error_name(&mut engine, error), "ReferenceError");
}

#[test]
fn named_read_proxy_reentry_runs_after_borrowed_lookup_and_keeps_method_receiver() {
    let mut engine = Engine::new();
    engine.interp.def_method(
        &engine.interp.global.clone(),
        "collectNamedRead",
        0,
        collect_named_read,
    );
    evaluate(
        &mut engine,
        "var namedProxy = new Proxy({value:31},{get(target,key,receiver){collectNamedRead();return Reflect.get(target,key,receiver);}})",
    );
    let proxy = binding(&mut engine, "namedProxy");
    let prefix = Value::Str("proxy-prefix".into());
    let (mut chunk, site) = read_chunk();
    for kind in [
        ReadKind::Stack,
        ReadKind::Local,
        ReadKind::This,
        ReadKind::Method,
    ] {
        use_kind(&mut chunk, site, kind);
        let mut outcome = invoke(
            &mut engine,
            &chunk,
            site,
            kind,
            proxy.clone(),
            prefix.clone(),
        );
        assert!(
            outcome.error.is_none(),
            "proxy property read failed for {kind:?}"
        );
        let values = outcome.take_values();
        assert!(matches!(&values[0], Value::Str(value) if value.as_str() == "proxy-prefix"));
        assert!(matches!(values.last(), Some(Value::Num(31.0))));
        if kind.is_method() {
            assert!(same_js_value(&mut engine, &proxy, &values[1]));
        }
    }
}
