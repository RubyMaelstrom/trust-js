//! The dependency registry's first producers and a test consumer (W4 step 10): a compilation
//! that relies on a prototype's shape and on the Array elements protector, whose code a change
//! of either invalidates.
use crate::dependency::{
    CodeDependencies, DesiredWatchpoints, WatchState, ARRAY_PROTECTOR, PROTOTYPE_SHAPE,
};
use crate::value::{Value, ValueView};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("dependency fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> crate::value::Gc {
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(object) = value.view() else {
        panic!("{name} is not an object")
    };
    object
}

/// The test consumer: a "compilation" of a method call on P's instances that relies on
/// P.prototype's shape and on the protector, installed at its end.
fn compile(engine: &mut Engine) -> (std::sync::Arc<CodeDependencies>, Result<(), &'static str>) {
    let prototype = object(engine, "proto");
    let mut desired = DesiredWatchpoints::default();
    desired.add(engine.interp.prototype_shape_set(&prototype));
    desired.add(engine.interp.array_protector_set());
    let code = CodeDependencies::new();
    let installed = desired.install(&code);
    (code, installed)
}

#[test]
fn a_prototypes_change_invalidates_the_code_relying_on_its_shape() {
    let mut engine = Engine::new();
    eval(
        &mut engine,
        "function P() {} P.prototype.m = function () { return 1; }; var proto = P.prototype;",
    );
    let (code, installed) = compile(&mut engine);
    assert_eq!(installed, Ok(()));
    assert!(code.is_valid());
    let prototype = object(&mut engine, "proto");
    assert_eq!(
        engine.interp.prototype_shape_set(&prototype).state(),
        WatchState::Watched,
        "the shared set"
    );
    eval(&mut engine, "new P().m();");
    assert!(
        code.is_valid(),
        "running relies on the facts, changes nothing"
    );
    eval(&mut engine, "P.prototype.n = 2;");
    assert!(!code.is_valid());
    assert_eq!(code.invalidated_by(), Some(PROTOTYPE_SHAPE));
    let (again, installed) = compile(&mut engine);
    assert_eq!(installed, Ok(()), "a fresh set for the changed prototype");
    assert!(again.is_valid());
}

#[test]
fn an_indexed_array_prototype_invalidates_the_code_relying_on_the_protector() {
    let mut engine = Engine::new();
    eval(&mut engine, "function P() {} var proto = P.prototype;");
    let (code, installed) = compile(&mut engine);
    assert_eq!(installed, Ok(()));
    eval(&mut engine, "Array.prototype[0] = 1;");
    assert!(!code.is_valid());
    assert_eq!(code.invalidated_by(), Some(ARRAY_PROTECTOR));
    let (late, installed) = compile(&mut engine);
    assert_eq!(
        installed,
        Err(ARRAY_PROTECTOR),
        "the protector no longer holds"
    );
    assert!(!late.is_valid());
}
