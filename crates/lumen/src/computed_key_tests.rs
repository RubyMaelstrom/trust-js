//! Integer-keyed reads of Strings and Proxies, String-keyed `in`, and StringToNumber of
//! decimal digit strings. Local ECMA-262 e28783d5: §10.4.3.5 StringGetOwnProperty, §10.5.8
//! Proxy [[Get]], §7.1.19 ToPropertyKey, §13.10.1 RelationalExpression `in` (HasProperty),
//! §7.1.4.1.1 StringToNumber. Expected answers from Node.js 22.

use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("key fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn integer_and_string_keys_agree_across_tiers() {
    check(
        r##"
function run() {
  var out = [];
  // String receivers with integer keys
  var s = "abé😀z", ascii = "hello";
  function at(str, k) { return str[k]; }
  out.push([at(ascii, 0), at(ascii, 4), at(ascii, 5), at(ascii, -0), at(ascii, 1.5), at(ascii, -1),
            at(s, 2), at(s, 3) === "\ud83d", at(s, 4) === "\ude00", at(s, 5), at(s, 6), at("", 0)].map(String).join(","));
  // Proxy receivers with integer keys
  var seen = [];
  var target = ["t0", "t1"];
  var p = new Proxy(target, { get: function (t, k, r) { seen.push(typeof k + ":" + k + ":" + (r === p)); return k === "1" ? "one" : t[k]; } });
  function read(o, k) { return o[k]; }
  out.push([read(p, 0), read(p, 1), read(p, -0), read(p, -3), read(p, 4096), read(p, 2.5)].map(String).join(","));
  out.push(seen.join(" "));
  var frozen = Object.freeze(["x"]);
  var liar = new Proxy(frozen, { get: function () { return "y"; } });
  try { read(liar, 0); out.push("no"); } catch (e) { out.push(e.name); }
  var revocable = Proxy.revocable([1], {}); revocable.revoke();
  try { read(revocable.proxy, 0); out.push("no"); } catch (e) { out.push(e.name); }
  var getterTarget = {}; Object.defineProperty(getterTarget, 3, { get: function () { return this === fwd; } });
  var fwd = new Proxy(getterTarget, {});
  out.push(String(read(fwd, 3)));
  var keys = []; var echo = new Proxy({}, { get: function (t, k) { keys.push(k); return k; } });
  for (var i = 0; i < 3; i++) read(echo, 7);
  out.push(String(keys[0] === keys[2]) + ":" + keys.join());
  // `in` with String keys
  var proto = { inherited: 1 }, obj = Object.create(proto); obj.own = 2;
  var arr = [1, , 3];
  function has(k, o) { return k in o; }
  out.push([has("own", obj), has("inherited", obj), has("missing", obj), has("toString", obj),
            has("length", arr), has("0", arr), has("1", arr), has("2", arr), has("3", arr),
            has("x", Object.create(null)), has("1", new String("ab")), has("length", new String("")),
            has("0", new Uint8Array(2)), has("2", new Uint8Array(2)), has("call", function () {})].join(","));
  var trapped = []; var hp = new Proxy({}, { has: function (t, k) { trapped.push(k); return k === "yes"; } });
  var child = Object.create(hp);
  out.push([has("yes", child), has("no", child), trapped.join("/")].join(","));
  // StringToNumber
  out.push(["123", "007", "0", "", " 12 ", "1e3", "999999999999999", "9999999999999999",
            "12345678901234567890", "0x10", "-5", "+5", "1_000", " 42", "4.5", ".5", "5."]
           .map(function (t) { return String(Number(t)); }).join(","));
  return out.join("|");
}
var first = run(), same = true;
for (var i = 0; i < 40; i++) same = same && run() === first;
same + "#" + first;
"##,
        "true#h,o,undefined,h,undefined,undefined,é,true,true,z,undefined,undefined|\
         t0,one,t0,undefined,undefined,undefined|string:0:true string:1:true string:0:true \
         string:-3:true string:4096:true string:2.5:true|TypeError|TypeError|true|true:7,7,7|\
         true,true,false,true,true,true,false,true,false,false,true,true,true,false,true|\
         true,false,yes/no|123,7,0,0,12,1000,999999999999999,10000000000000000,\
         12345678901234567000,16,-5,5,NaN,42,4.5,0.5,5",
    );
}
