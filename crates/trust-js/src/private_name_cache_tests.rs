//! ECMA-262 e28783d5 ResolvePrivateIdentifier, PrivateElementFind, PrivateGet and PrivateSet
//! through the per-site private-name caches (`bytecode::PrivateSite`). Expected values are
//! Node's.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("private-name fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// One class body evaluated twice shares its methods' code, so every private site sees two
/// environments and two Private Names; brand checks must still fail across them. Receivers
/// with different layouts, later additions and deletions move the element's slot, Proxies
/// carry their own elements, and methods, accessors and compound assignments take the
/// general path through the same warmed sites.
#[test]
fn private_sites_follow_environment_key_and_live_slot() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
var log = [];
function rec(x) { log.push(String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
// One class body evaluated twice: both classes share every method's code (and its private
// sites) but have distinct Private Names, so each site sees two environments and two keys.
function make(tag) {
  return class {
    #v; #w = tag + '-w';
    static #count = 0;
    constructor(v) { this.#v = v; }
    get v() { return this.#v; }
    set v(x) { this.#v = x; }
    has(o) { return #v in o; }
    peek(o) { return o.#v; }
    poke(o, x) { o.#v = x; return o.#v; }
    w() { return this.#w; }
    static bump() { return ++this.#count; }
  };
}
var A = make('a'), B = make('b');
var as = [], bs = [];
for (var i = 0; i < 40; i++) { as.push(new A(i)); bs.push(new B(100 + i)); }
var s = 0, hits = 0, misses = 0;
for (var r = 0; r < 50; r++) {
  for (var i = 0; i < 40; i++) {
    var a = as[i], b = bs[i];
    s += a.v + b.v;
    if (a.has(b)) hits++; else misses++;
    if (b.has(a)) hits++; else misses++;
    if (a.has(a) && b.has(b)) hits++;
    s += a.peek(a) + b.peek(b);
  }
}
rec(s); rec(hits); rec(misses);
rec(err(() => as[0].peek(bs[0]))); rec(err(() => bs[0].peek(as[0])));
rec(err(() => as[0].poke(bs[0], 1))); rec(as[0].poke(as[1], 'x')); rec(as[1].v);
rec(as[2].w() + bs[2].w());
// Different instance layouts behind the same site: the cached slot is only a hint.
class Base { constructor(o) { return o; } }
class Stamp extends Base { #s = 'stamped'; static read(o) { return o.#s; } static has(o) { return #s in o; } static write(o, v) { o.#s = v; } }
var objs = [{}, { a: 1 }, { a: 1, b: 2, c: 3 }, [1, 2, 3], function () {}];
objs.forEach(o => new Stamp(o));
for (var r = 0; r < 30; r++) objs.forEach((o, k) => { if (Stamp.read(o) !== 'stamped' && r === 0) rec('bad' + k); });
objs[1].z = 9; delete objs[2].a;
rec(objs.map(o => Stamp.read(o)).join('/'));
objs.forEach((o, k) => Stamp.write(o, 'w' + k));
rec(objs.map(o => Stamp.read(o) + Stamp.has(o)).join('/'));
rec(Stamp.has({}) + ',' + err(() => Stamp.read({})) + ',' + err(() => Stamp.write(5, 1)));
// A Proxy can carry its own private elements (return override) and is not its target.
var target = {}, proxy = new Proxy(target, {});
new Stamp(proxy);
rec(Stamp.has(proxy) + ',' + Stamp.has(target) + ',' + Stamp.read(proxy));
// Methods, accessors and setter-less accessors through warmed sites.
class M {
  #x = 1; #m() { return this.#x * 10; } get #g() { return this.#x + 100; } set #g(v) { this.#x = v; }
  get #ro() { return 7; }
  run() { var t = 0; for (var k = 0; k < 20; k++) { this.#g = k; t += this.#m() + this.#g; } return t; }
  ro() { return this.#ro; }
  roSet() { this.#ro = 1; }
  mSet() { this.#m = 1; }
  static cmp(o) { o.#x += 1; o.#x *= 2; o.#x ??= 5; o.#x ||= 6; o.#x &&= o.#x + 1; return o.#x; }
}
var m = new M();
rec(m.run()); rec(m.ro()); rec(err(() => m.roSet())); rec(err(() => m.mSet()));
for (var r = 0; r < 25; r++) M.cmp(m);
rec(M.cmp(m));
log.join(',');
"#
            ),
            "556000,2000,4000,TypeError,TypeError,TypeError,x,x,a-wb-w,stamped/stamped/stamped/stamped/stamped,w0true/w1true/w2true/w3true/w4true,false,TypeError,TypeError,true,false,stamped,4090,7,TypeError,TypeError,1476395005",
            "{tier:?}"
        );
    }
}
