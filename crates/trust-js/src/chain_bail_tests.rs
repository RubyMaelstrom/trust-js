//! Fused numeric chains (`jit::emit_chain`) bail to their shared replay ladder at every
//! operation: each guard failure spills the chain's pre-op values and replays the remaining
//! operations through the checked helpers, including conversions that run author code.
//! Expected values are Node's.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("chain fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn numeric_chains_bail_at_every_operation() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
var log = [];
function rec(x) { log.push(Object.is(x, -0) ? '-0' : String(x)); }
var G = 3, H = 0.5;
function f(a, b, c, d) { return a + b * c - d; }
function g(x) { return G * x + H - x * G; }
function k(a, b) { return (a - b) * (a + b) / (b - 0.25); }
for (var r = 0; r < 300; r++) { f(r, 2, 3, 1); g(r); k(r, 2); }
var odd = [1, 2.5, '3', null, undefined, true, { valueOf: function () { log.push('vo'); return 7; } }, -0, NaN, 1099511627776, 'x', [4]];
for (var p = 0; p < 4; p++) {
  for (var q = 0; q < odd.length; q++) {
    var args = [5, 6, 7, 8];
    args[p] = odd[q];
    rec(f(args[0], args[1], args[2], args[3]));
  }
}
for (var q = 0; q < odd.length; q++) { rec(g(odd[q])); rec(k(odd[q], 3)); rec(k(4, odd[q])); }
G = 'G'; rec(g(2)); G = { valueOf: function () { log.push('Gvo'); return 2; } }; rec(g(3)); G = 3;
H = undefined; rec(g(4)); H = 0.5; rec(g(5));
log.join(',');
"#
            ),
            "35,36.5,334,34,NaN,35,vo,41,34,NaN,1099511627810,NaN,434,4,14.5,18,-3,NaN,4,vo,46,-3,NaN,7696581394429,NaN,25,3,12,15,-3,NaN,3,vo,39,-3,NaN,6597069766653,NaN,21,46,44.5,44,47,NaN,46,vo,40,47,NaN,-1099511627729,NaN,43,0.5,-2.909090909090909,20,0.5,-1,4.333333333333333,0.5,0,15.636363636363637,0.5,-3.272727272727273,-64,NaN,NaN,NaN,0.5,-2.909090909090909,20,vo,vo,0.5,vo,vo,14.545454545454545,vo,vo,vo,-4.888888888888889,0.5,-3.272727272727273,-64,NaN,NaN,NaN,0.5,4.3960938895077426e+23,-1099511627776.25,NaN,NaN,NaN,0.5,15.636363636363637,0,NaN,Gvo,Gvo,0.5,NaN,0.5",
            "{tier:?}"
        );
    }
}
