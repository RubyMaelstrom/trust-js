//! Local reads whose TDZ test the compiler drops (`Cfg::initialized_reads`): parameters, `var`s
//! and temporaries, and lexicals every path to the read initialized, read in straight-line code,
//! branches, loops (OSR'd ones too), switch cases, try/catch/finally, generators across their
//! yields, and functions inlined into their callers; and the reads that keep the test throwing
//! the same ReferenceError as before (a read before the declaration, in another switch case,
//! in a catch, after a yield, in a default parameter, an assignment). Every tier agrees; the
//! tree-walker's results are the expected ones.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

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

const PROGRAM: &str = r#"
var lines = [];
function rec(tag, f) {
  try { lines.push(tag + ":" + String(f())); }
  catch (e) { lines.push(tag + ":throw " + (e && e.constructor ? e.constructor.name + " " + e.message : typeof e)); }
}
function before() { var r = x; let x = 1; return r; }
function after(p) { let x = p + 1; var y = x * 2; return x + y; }
function branch(c) { let x; if (c) x = 1; else x = "two"; return x; }
function branchTdz(c) { if (c) { return y; } let y = 3; return y; }
function loopLet(n) { var s = 0; for (let i = 0; i < n; i++) { let j = i * 2; s += j; } return s; }
function loopBefore(n) { var s = 0; for (var i = 0; i < n; i++) { if (i > 0) s += z; } let z = 5; return s + z; }
function sw(v) { switch (v) { case 0: let a = 10; return a; case 1: return a; } return -1; }
function sw2(v) { switch (v) { case 0: let a = 1; case 1: a = 2; return a; } return 0; }
function thrower() { throw new TypeError("t"); }
function tc(c) { let t = 1; try { if (c) thrower(); t = 2; } catch (e) { return "caught " + t; } return t; }
function tc2(c) { try { if (c) throw 0; } catch (e) { return k; } let k = 1; return k; }
function fin(c) { let f = 0; try { f = 1; if (c) throw 2; f = 3; } catch (e) { f = f + 10; } finally { f = f + 100; } return f; }
function* gen() { let a = 1; yield a; let b = a + 1; yield b; return a + b; }
function* gen2() { yield 1; yield c; let c = 2; }
function drain(g) { var out = []; for (;;) { var step = g.next(); out.push(step.value); if (step.done) return out.join(); } }
function dp(a = b, b = 2) { return a + b; }
function cl() { var r = typeof C; class C {} return r; }
function cl2() { class D { m() { return 1; } } return new D().m(); }
function asg() { x2 = 1; let x2; return x2; }
function cmp() { y2 += 1; let y2 = 0; }
function tz() { return typeof u; let u; }
function clo() { let v = 1; function g() { return v; } return g() + v; }
function clo2() { var r = g(); let v = 2; function g() { return typeof v; } return r; }
function forof() { for (let x of [x]) {} }
function ds() { let [a, b = a] = [1]; return a + b; }
function ds2() { let [a = b, b] = [undefined, 1]; return a; }
function lab(n) { var s = 0; outer: for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) { if (j == i) continue outer; let k = i + j; s += k; } } return s; }
function osr(n) { let acc = 0; for (let i = 0; i < n; i++) { let t = i & 7; acc += t; } let tail = acc; return tail; }
function late(n) { if (n > 0) { var r = late(n - 1); } let m = n; return m + (r || 0); }
function nb(c) { { let q = c ? 1 : 2; if (q > 1) { return q * 10; } } return 0; }
function brk(n) { let x; for (var i = 0; i < n; i++) { x = i; if (i == 2) break; } return x; }
function cond(c) { let w; if (c) { w = 7; } return c ? w : "none"; }
function leaf(v) { const k = v * 3; return k + 1; }
function caller(n) { var s = 0; for (var i = 0; i < n; i++) s += leaf(i); return s; }
function leafTdz(v) { if (v > 1000) return later; let later = v; return later; }
function callerTdz(n) { var s = 0; for (var i = 0; i < n; i++) s += leafTdz(i); return s; }
for (var round = 0; round < 3; round++) {
  rec("before", before);
  rec("after", function () { return after(round); });
  rec("branch1", function () { return branch(true); });
  rec("branch0", function () { return branch(false); });
  rec("branchTdz1", function () { return branchTdz(true); });
  rec("branchTdz0", function () { return branchTdz(false); });
  rec("loopLet", function () { return loopLet(5); });
  rec("loopBefore1", function () { return loopBefore(1); });
  rec("loopBefore2", function () { return loopBefore(2); });
  rec("sw0", function () { return sw(0); });
  rec("sw1", function () { return sw(1); });
  rec("sw2", function () { return sw(2); });
  rec("sw2-0", function () { return sw2(0); });
  rec("sw2-1", function () { return sw2(1); });
  rec("tc0", function () { return tc(false); });
  rec("tc1", function () { return tc(true); });
  rec("tc2-0", function () { return tc2(false); });
  rec("tc2-1", function () { return tc2(true); });
  rec("fin0", function () { return fin(false); });
  rec("fin1", function () { return fin(true); });
  rec("gen", function () { return drain(gen()); });
  rec("gen2", function () { return drain(gen2()); });
  rec("dp", function () { return dp(); });
  rec("dp1", function () { return dp(1); });
  rec("cl", cl);
  rec("cl2", cl2);
  rec("asg", asg);
  rec("cmp", cmp);
  rec("tz", tz);
  rec("clo", clo);
  rec("clo2", clo2);
  rec("forof", forof);
  rec("ds", ds);
  rec("ds2", ds2);
  rec("lab", function () { return lab(4); });
  rec("osr", function () { return osr(20000); });
  rec("late", function () { return late(4); });
  rec("nb1", function () { return nb(true); });
  rec("nb0", function () { return nb(false); });
  rec("brk", function () { return brk(5); });
  rec("cond1", function () { return cond(true); });
  rec("cond0", function () { return cond(false); });
  rec("caller", function () { return caller(3000); });
  rec("callerTdz", function () { return callerTdz(3000); });
  rec("leafTdz", function () { return leafTdz(2000); });
}
lines.join("\n");
"#;

#[test]
fn initialized_reads_agree_with_the_tree_walker() {
    let expected = eval(&mut engine(Tier::Interp), PROGRAM);
    assert!(
        expected.lines().count() > 100 && expected.contains("cannot access 'x' before"),
        "the program ran: {expected}"
    );
    for tier in [Tier::Bytecode, Tier::Jit] {
        let actual = eval(&mut engine(tier), PROGRAM);
        let differ: Vec<String> = expected
            .lines()
            .zip(actual.lines())
            .filter(|(want, got)| got != want)
            .map(|(want, got)| format!("{want} / {got}"))
            .collect();
        assert!(differ.is_empty(), "{tier:?}: {differ:?}");
        assert_eq!(actual, expected, "{tier:?}");
    }
}
