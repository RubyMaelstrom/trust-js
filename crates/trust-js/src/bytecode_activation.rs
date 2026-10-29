//! Fixed captured-binding storage for compiled FunctionDeclarationInstantiation.
//!
//! ECMA-262 #sec-functiondeclarationinstantiation / #sec-declarative-environment-records:
//! binding names and attributes are static; values, TDZ state and function identities
//! are per activation. Compile the creation work once, then seed indexed slots.
//! Prepared entries and reused mapped-arguments environments retain their original
//! instantiation path. Structural eval mutations invalidate VarMap's layout proof.

use super::{CapInit, Chunk};
use crate::interpreter::{
    new_var_scope_with_bindings, Binding, BindingAttrs, BindingSlot, DescRef, Env, Interp,
    ScopeDesc, VarMap,
};
use crate::value::Value;
use std::rc::Rc;

pub(super) fn activation_plans_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TRUST_JS_ACTIVATION_PLANS").as_deref() != Ok("0"))
}

/// A chunk's plan, made in the chunk's heap: its names are that heap's atoms, in a descriptor
/// every activation shares with the bindings' attributes, so that making one copies binding
/// values only.
pub(super) struct ActivationPlan {
    bindings: Vec<BindingSlot>,
    desc: Rc<ScopeDesc>,
    /// `desc`, kept by the chunk's heap for its activations to name.
    kept_desc: DescRef,
    parameters: Vec<(usize, u16)>,
    /// A captured rest parameter: (binding index, first argument index).
    rest: Option<(usize, u16)>,
    functions: Vec<(usize, u16)>,
    this_slot: Option<usize>,
    arguments_slot: Option<usize>,
}

impl ActivationPlan {
    pub(super) fn new(chunk: &Chunk) -> Self {
        let mut names = Vec::with_capacity(chunk.cap_inits.len() + 2);
        let mut attrs = Vec::<BindingAttrs>::with_capacity(chunk.cap_inits.len() + 2);
        let mut plan = Self {
            bindings: Vec::with_capacity(chunk.cap_inits.len() + 2),
            desc: ScopeDesc::new(Vec::new(), Vec::new()),
            kept_desc: DescRef::empty(),
            parameters: Vec::new(),
            rest: None,
            functions: Vec::new(),
            this_slot: None,
            arguments_slot: None,
        };
        let mut slots = crate::fasthash::FastMap::<Rc<str>, usize>::default();
        let mut slot = |bindings: &mut Vec<BindingSlot>,
                        slots: &mut crate::fasthash::FastMap<Rc<str>, usize>,
                        name: &Rc<str>,
                        binding: Binding,
                        replace: bool|
         -> usize {
            if let Some(&slot) = slots.get(name) {
                if replace {
                    bindings[slot] = binding.slot();
                    attrs[slot] = binding.attrs();
                }
                return slot;
            }
            let index = bindings.len();
            slots.insert(name.clone(), index);
            names.push(crate::atom::atomize(name));
            bindings.push(binding.slot());
            attrs.push(binding.attrs());
            index
        };
        let mut functions = Vec::new();
        for init in &chunk.cap_inits {
            match init {
                CapInit::Param(parameter, name) => {
                    let index = slot(
                        &mut plan.bindings,
                        &mut slots,
                        name,
                        Binding::data(Value::Undefined, true, true),
                        true,
                    );
                    plan.parameters.push((index, *parameter));
                }
                CapInit::Rest(first, name) => {
                    let index = slot(
                        &mut plan.bindings,
                        &mut slots,
                        name,
                        Binding::data(Value::Undefined, true, true),
                        true,
                    );
                    plan.rest = Some((index, *first));
                }
                CapInit::Var(name) => {
                    slot(
                        &mut plan.bindings,
                        &mut slots,
                        name,
                        Binding::data(Value::Undefined, true, true),
                        false,
                    );
                }
                CapInit::Lexical(name, is_const) => {
                    // Lexical at the activation's level, as FunctionDeclarationInstantiation's
                    // separate lexical record would hold it (EvalDeclarationInstantiation).
                    slot(
                        &mut plan.bindings,
                        &mut slots,
                        name,
                        Binding::data(Value::Undefined, !is_const, false).lexical(true),
                        true,
                    );
                }
                CapInit::Fn(function, name) => functions.push((*function, name.clone())),
            }
        }
        if chunk.env_this && !chunk.lexical_this {
            plan.this_slot = Some(slot(
                &mut plan.bindings,
                &mut slots,
                &Rc::from("this"),
                Binding::data(Value::Undefined, false, true),
                true,
            ));
        }
        // Source-order function creation follows parameter/var instantiation.
        // Preserve all creations, including source-order replacement of names.
        for (function, name) in functions {
            let index = slot(
                &mut plan.bindings,
                &mut slots,
                &name,
                Binding::data(Value::Undefined, true, true),
                true,
            );
            plan.functions.push((index, function));
        }
        if chunk.env_arguments {
            plan.arguments_slot = Some(slot(
                &mut plan.bindings,
                &mut slots,
                &Rc::from("arguments"),
                Binding::data(Value::Undefined, true, true),
                true,
            ));
        }
        // ECMA-262 e28783d5 FunctionDeclarationInstantiation/GetBindingValue:
        // names and attributes are static here, while initialization and values
        // stay per invocation.
        drop(slot);
        plan.desc = ScopeDesc::new(names, attrs);
        plan.kept_desc = crate::value::keep_scope_desc(&plan.desc);
        plan
    }

    pub(super) fn instantiate(
        &self,
        chunk: &Chunk,
        interp: &mut Interp,
        parent: &Env,
        this_value: &Value,
        arguments: &[Value],
    ) -> Env {
        let mut vars =
            VarMap::from_fixed_bindings(self.bindings.iter().cloned().collect(), self.kept_desc);
        for &(slot, parameter) in &self.parameters {
            vars.initialize_fixed_value(
                slot,
                arguments
                    .get(usize::from(parameter))
                    .copied()
                    .unwrap_or(Value::Undefined),
            );
        }
        if let Some((slot, first)) = self.rest {
            vars.initialize_fixed_value(
                slot,
                interp.make_array(arguments.get(usize::from(first)..).unwrap_or(&[]).to_vec()),
            );
        }
        if let Some(slot) = self.this_slot {
            vars.initialize_fixed_value(slot, *this_value);
        }
        let activation = new_var_scope_with_bindings(Some(*parent), vars);
        // Every closure captures this invocation's actual environment. The
        // plan contains no JS object, closure environment, or argument owner.
        for &(slot, function) in &self.functions {
            let value =
                interp.make_function(chunk.funcs[usize::from(function)].clone(), activation);
            activation
                .borrow_mut()
                .vars
                .initialize_fixed_value(slot, value);
        }
        if let Some(slot) = self.arguments_slot {
            let value = Value::Obj(interp.make_compiled_arguments_object(arguments, &activation));
            activation
                .borrow_mut()
                .vars
                .initialize_fixed_value(slot, value);
        }
        activation
            .borrow_mut()
            .vars
            .publish_layout(chunk.binding_layout_id);
        activation
    }

    pub(super) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        visitor.scope_desc(&self.desc);
        // The names are atoms, which their heap's table accounts.
        std::mem::size_of::<Self>()
            + self.bindings.capacity() * std::mem::size_of::<BindingSlot>()
            + (self.parameters.capacity() + self.functions.capacity())
                * std::mem::size_of::<(usize, u16)>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ValueView;
    use crate::{bytecode::Tier, Completion, Engine};

    fn evaluate(engine: &mut Engine, source: &str) -> String {
        match engine
            .eval(source, false)
            .expect("activation fixture parses")
        {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    #[test]
    fn packed_activation_calls_keep_escaped_parameter_owners_across_collection() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            crate::bytecode::TEST_FRAME_SETUPS.with(|entries| entries.set(0));
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                var effects=0,escaped=[];
                function argument(n){effects++;return {n:n};}
                function capture(a,b,c,d,e,f){
                    b.n+=10;
                    return function(){return [a.n,f===undefined?'missing':f.n,b.n].join(':');};
                }
                function dispatch(n){return capture(argument(n),argument(n+1),
                    argument(n+2),'unused',false,argument(n+5),argument(n+6));}
                for(var n=0;n<30;n++)escaped.push(dispatch(n));
                var missing=capture(argument(50),argument(51));
                [effects,escaped[0](),escaped[29](),missing()].join('|');
            "#,
                ),
                "152|0:5:11|29:34:40|50:missing:61",
                "{tier:?}"
            );
            engine.interp.gc_collect();
            assert_eq!(
                evaluate(
                    &mut engine,
                    "[escaped[0](),escaped[29](),missing()].join('|')"
                ),
                "0:5:11|29:34:40|50:missing:61",
                "{tier:?}"
            );
            #[cfg(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(target_os = "linux", target_os = "macos", target_os = "windows")
            ))]
            if tier == Tier::Jit {
                // The callee's prologue made its activation (`bytecode::jit_frame_setup`).
                assert!(crate::bytecode::TEST_FRAME_SETUPS.with(|entries| entries.get()) > 0);
            }
        }
    }

    #[test]
    fn packed_activation_arguments_keep_surplus_values_and_parameter_alias_rules() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                function unmapped(a,b,c,d,e,f){
                    'use strict';
                    var read=()=>[a.n,f.n,arguments[6].n,arguments.length].join(':');
                    arguments[0]={n:99};
                    return read;
                }
                function mapped(a,a,b,c,d,e){
                    var read=()=>[a.n,arguments[0].n,arguments[1].n,arguments[6].n].join(':');
                    arguments[0]={n:10};arguments[1]={n:20};
                    return read;
                }
                function dispatch(fn){return fn({n:1},{n:2},null,undefined,false,{n:6},{n:7});}
                var strictReads=[],mappedReads=[];
                for(var n=0;n<30;n++){
                    strictReads.push(dispatch(unmapped));mappedReads.push(dispatch(mapped));
                }
                [strictReads[0](),strictReads[29](),mappedReads[0](),mappedReads[29]()].join('|');
            "#,
                ),
                "1:6:7:7|1:6:7:7|20:10:20:7|20:10:20:7",
                "{tier:?}"
            );
            engine.interp.gc_collect();
            assert_eq!(
                evaluate(
                    &mut engine,
                    "[strictReads[29](),mappedReads[0]()].join('|')"
                ),
                "1:6:7:7|20:10:20:7",
                "{tier:?}"
            );
        }
    }

    #[test]
    fn activation_plan_wide_closures_keep_fixed_slots_and_live_binding_guards() {
        for tier in [Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                function make(seed) {
                    let a=seed,b=seed+1,c=seed+2,d=seed+3,e=seed+4,f=seed+5;
                    let g=seed+6,h=seed+7,j=seed+8,k=seed+9,l=seed+10,m=seed+11;
                    return [()=>a+b+c+d+e+f+g+h+j+k+l+m,()=>++a];
                }
                var left=make(1),right=make(10),one=left[0];
                for(let n=0;n<40;n++) { if(one()!==78 || right[0]()!==186) throw 'alias'; }
                left[1](); one()===79 && right[0]()===186
            "#
                ),
                "true"
            );
            let global = engine.interp.global_env;
            let ValueView::Obj(one) = (engine
                .interp
                .get_var("one", &global)
                .unwrap_or_else(|_| panic!("missing closure")))
            .view() else {
                panic!("expected closure")
            };
            let activation = match &one.borrow().call {
                crate::value::Callable::User(user) => user.env,
                _ => panic!("expected user closure"),
            };
            if activation_plans_enabled() {
                let scope = activation.borrow();
                assert_ne!(scope.vars.layout_id(), 0);
                assert!(scope.vars.binding_slot("m").unwrap() >= 8);
            }
            // A dynamic insertion may replace the vector with a hash table.
            // Previously emitted slot/generation proofs must reject it.
            activation
                .borrow_mut()
                .vars
                .insert("added", Binding::data(Value::Num(7.), true, true));
            assert_eq!(activation.borrow().vars.layout_id(), 0);
            assert_eq!(evaluate(&mut engine, "one()"), "79");
            activation
                .borrow_mut()
                .vars
                .get_mut("m")
                .unwrap()
                .set_initialized(false);
            assert_eq!(
                evaluate(
                    &mut engine,
                    "try { one(); 'missed TDZ' } catch(e) { e.name }"
                ),
                "ReferenceError"
            );
            activation
                .borrow_mut()
                .vars
                .get_mut("m")
                .unwrap()
                .set_initialized(true);
            assert_eq!(evaluate(&mut engine, "one()"), "79");
        }
    }

    #[test]
    fn indexed_activation59_shares_wide_keys_but_keeps_values_and_mutations_private() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let declarations = (0..96)
                .map(|n| format!("let v{n}=seed+{n};"))
                .collect::<String>();
            let sum = (0..96)
                .map(|n| format!("v{n}"))
                .collect::<Vec<_>>()
                .join("+");
            let source = format!(
                "'use strict';function wide59(seed){{{declarations}return function(){{return {sum};}};}}
                 var left59=wide59(1),right59=wide59(2);
                 for(var n59=0;n59<30;n59++){{left59();right59();}}
                 left59()+'|'+right59();"
            );
            assert_eq!(evaluate(&mut engine, &source), "4656|4752", "{tier:?}");
            let environment = |engine: &mut Engine, name| {
                let global = engine.interp.global_env;
                let value = engine
                    .interp
                    .get_var(name, &global)
                    .unwrap_or_else(|_| panic!("missing fixture closure {name}"));
                let object_handle = value.as_obj().unwrap();
                let object = object_handle.borrow();
                let crate::value::Callable::User(user) = &object.call else {
                    panic!("fixture closure")
                };
                user.env
            };
            let left = environment(&mut engine, "left59");
            let right = environment(&mut engine, "right59");
            assert!(!Env::ptr_eq(&left, &right));
            if tier != Tier::Interp && activation_plans_enabled() {
                let l = left.borrow();
                let r = right.borrow();
                assert!(std::ptr::eq(
                    l.vars.shared_desc().unwrap(),
                    r.vars.shared_desc().unwrap()
                ));
                assert!(l.vars.binding_slot("v95").is_some());
            }
            left.borrow_mut()
                .vars
                .get_mut("v95")
                .unwrap()
                .set_value(Value::Num(1.));
            assert_eq!(evaluate(&mut engine, "left59()+'|'+right59()"), "4561|4752");
            left.borrow_mut()
                .vars
                .insert("added59", Binding::data(Value::Num(3.), true, true));
            assert_eq!(left.borrow().vars.layout_id(), 0);
            assert_eq!(evaluate(&mut engine, "left59()+'|'+right59()"), "4561|4752");
            right
                .borrow_mut()
                .vars
                .get_mut("v95")
                .unwrap()
                .set_initialized(false);
            assert_eq!(
                evaluate(&mut engine, "try{right59();'miss'}catch(e){e.name}"),
                "ReferenceError"
            );
            right
                .borrow_mut()
                .vars
                .get_mut("v95")
                .unwrap()
                .set_initialized(true);
            right.borrow_mut().vars.remove("v95");
            assert_eq!(right.borrow().vars.layout_id(), 0);
            assert_eq!(
                evaluate(&mut engine, "try{right59();'miss'}catch(e){e.name}"),
                "ReferenceError"
            );
            assert_eq!(evaluate(&mut engine, "left59()"), "4561");
            engine.interp.gc_collect();
        }
    }

    #[test]
    fn activation_plan_preserves_parameters_functions_this_and_arguments() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                function duplicate(a,a) { var a; return ()=>a; }
                function redeclare(a) { function a(){return 23;} return ()=>a(); }
                function lexical(x) {
                    var read=()=>x+y+z;
                    let y=x+2; const z=x+4;
                    return [read,()=>{try {z=99;}catch(e){return e.name;}}];
                }
                function receiver() {return ()=>this.n+arguments[0];}
                function early() {
                    var read=()=>late;
                    try {read();} catch(e) {return e.name;}
                    let late=1;
                }
                var a=lexical(2),b=lexical(20),f=receiver.call({n:7},9);
                [duplicate(1,8)(),redeclare(99)(),a[0](),b[0](),a[1](),f(),early()].join('|')
            "#
                ),
                "8|23|12|66|TypeError|16|ReferenceError",
                "{tier:?}"
            );
        }
    }

    #[test]
    fn activation_plan_does_not_replace_reused_or_eval_environments() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                function dynamic(a) {
                    var read=()=>a+injected;
                    eval('var injected=5');
                    arguments[0]=9;
                    return read();
                }
                function* generator(a) {
                    var read=()=>a;
                    arguments[0]=7; yield read();
                    a=11; yield arguments[0];
                }
                function defaults(a=()=>b) { var b=3; return a; }
                var it=generator(1),result=[dynamic(1),it.next().value,it.next().value];
                try { defaults()(); } catch(e) { result.push(e.name); }
                result.join('|')
            "#
                ),
                "14|7|11|ReferenceError",
                "{tier:?}"
            );
        }
    }
}
