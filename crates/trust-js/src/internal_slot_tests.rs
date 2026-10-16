//! ECMA-262 6.1.7.2, PrivateElementFind/Get/Set and ECMA-402 internal slots,
//! local snapshots e28783d5fc9d / b1c961988b9a. Slots are separate from author keys.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("internal-slot fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
    }
}

#[test]
fn internal_slot_tests_author_strings_cannot_read_replace_or_forge_builtin_state() {
    check(
        r#"
        function error(f) { try { f(); return ''; } catch (e) { return e.name; } }
        var d = new Date(7), key = '#\0date_ms';
        var hidden = [d[key], key in d, Object.hasOwn(d, key)].join(':');
        d[key] = 200;
        Object.defineProperty(d, key, {get() { return 99; }, configurable: true});
        var own = [d[key], Object.getOwnPropertyNames(d)[0] === key, d.getTime()].join(':');
        delete d[key];
        Object.freeze(d);
        d.setTime(8);
        var fake = Object.create(Date.prototype); fake[key] = 12;
        var ab = new ArrayBuffer(2, {maxByteLength: 4});
        ab['#\0abResizable'] = false; ab['#\0abMaxByteLength'] = 999;
        ab.resize(4);
        var raw = JSON.rawJSON('12');
        var initial = raw['\0raw_json'];
        var forged = {'\0raw_json': '13'};
        [hidden, own, d.getTime(), Object.isFrozen(d), error(() => fake.getTime()),
         ab.byteLength, ab.maxByteLength, ab.resizable,
         initial, JSON.isRawJSON(forged), JSON.stringify({raw})].join('|')
    "#,
        ":false:false|99:true:7|8|true|TypeError|4|4|true||false|{\"raw\":12}",
    );
}

#[test]
fn internal_slot_tests_array_property_replacement_preserves_slots_and_copies_omit_brands() {
    check(
        r#"
        class A extends Array {
          #value = {marker: 'retained'};
          get() { return this.#value; }
          static has(value) { return #value in value; }
          static get [Symbol.species]() { return Array; }
        }
        var array = new A(3), saved = array.get();
        array.fill(7);
        var copied = [array.slice(), array.concat([]), Array.from(array)];
        [array.get() === saved, array.get().marker, array.join(','), A.has(array),
         copied.every(value => !A.has(value) && value.join(',') === '7,7,7'),
         Object.getOwnPropertyNames(array).join(',')].join('|')
        "#,
        "true|retained|7,7,7|true|true|0,1,2,length",
    );
}

#[test]
#[cfg(feature = "intl")]
fn internal_slot_tests_intl_state_does_not_invoke_author_accessors_or_proxy_traps() {
    check(
        r#"
        var c = new Intl.Collator('en-US', {numeric: true});
        var before = c.compare, reads = 0;
        c['#\0co_locale'] = 'fr'; c['#\0co_numeric'] = false;
        c['#\0co_bound'] = () => 123;
        Object.defineProperty(c, '#\0co', {get() { reads++; return false; }});
        var fake = Object.create(Intl.Collator.prototype); fake['#\0co'] = true;
        var failed = false; try { fake.resolvedOptions(); } catch(e) { failed = e.name === 'TypeError'; }
        var traps = 0, p = new Proxy(c, {get(t, k) { traps++; return t[k]; }});
        var proxyFailed = false;
        try { Intl.Collator.prototype.resolvedOptions.call(p); } catch(e) { proxyFailed = e.name === 'TypeError'; }
        var options = c.resolvedOptions();
        [options.locale, options.numeric, c.compare === before, c.compare('2', '10') < 0,
         reads, failed, proxyFailed, traps,
         Object.getOwnPropertyNames(c).indexOf('#\0co_locale') >= 0].join('|')
    "#,
        "en-US|true|true|true|0|true|true|0|true",
    );
}

#[test]
fn internal_slot_tests_private_names_and_autoaccessors_cannot_be_forged_by_strings() {
    check(
        r#"
        function error(f) { try { f(); return ''; } catch (e) { return e.name; } }
        class C {
          #x = 7; #method() { return this.#x; }
          get #value() { return this.#x; } set #value(v) { this.#x = v; }
          static #s = 11;
          read(o) { return o.#method() + o.#value; }
          has(o) { return #x in o; }
          update() { this.#x++; this.#value += 2; return this.#x; }
          static read() { return this.#s; }
        }
        var c = new C(), fake = {};
        for (var n = 1; n <= 80; n++) {
          c['#x\x01' + n] = 99; c['#method\x01' + n] = () => 100;
          fake['#x\x01' + n] = 99;
          C['#s\x01' + n] = 100;
        }
        var before = c.read(c), names = Object.keys(c).length;
        Object.freeze(c);
        class D { ['#x\x011'] = 5; ['#\0acc1'] = 6; }
        var ordinary = new D();
        class A { accessor value = 3; }
        var a = new A();
        for (var n = 1; n <= 100; n++) a['#\0acc' + n] = 123;
        Object.freeze(a); a.value = 8;
        [before, c.update(), c.has(fake), error(() => c.read(fake)), names,
         C.read(), Object.hasOwn(ordinary, '#x\x011'), ordinary['#x\x011'],
         Object.keys(ordinary).length, a.value].join('|')
    "#,
        "14|10|false|TypeError|160|11|true|5|2|8",
    );
}

#[test]
fn internal_slot_tests_iterator_state_remains_own_and_unobservable() {
    check(
        r#"
        var it = [1, 2].values().map(x => x * 2);
        it['#\0ih_done'] = true; it['#\0ih_fn'] = () => 99;
        var names = Object.keys(it).length;
        var inherited = Object.create(it), failed = false;
        try { inherited.next(); } catch(e) { failed = e.name === 'TypeError'; }
        var a = it.next(), b = it.next(), c = it.next();
        /(a)/.exec('xa');
        RegExp['#\0legacy_$1'] = 'spoof';
        [names, failed, a.value, b.value, c.done, RegExp.$1].join('|')
    "#,
        "2|true|2|4|true|a",
    );
}

#[test]
fn internal_slot_tests_ordinary_class_names_preserve_private_looking_strings() {
    check(
        r#"
        var field = '#field\x011', method = '#method\x011', getter = '#getter\x011';
        class C {
          [field] = function() {};
          [method]() {}
          get [getter]() { return 1; }
          static [field] = () => {};
        }
        var c = new C();
        [c[field].name === field, c[method].name === method,
         Object.getOwnPropertyDescriptor(C.prototype, getter).get.name === 'get ' + getter,
         C[field].name === field].join('|')
    "#,
        "true|true|true|true",
    );
}

#[test]
fn internal_slot_tests_preserve_nonextensible_private_proposal_and_proxy_effect_order() {
    // Stage 3 draft, 2026-10-07:
    // https://tc39.es/proposal-nonextensible-applies-to-private/
    // New private elements perform IsExtensible, including its Proxy trap. Existing
    // elements remain writable after freeze. Node v24.21 has the base-standard semantics
    // and intentionally differs on new stamping and the IsExtensible observations.
    check(
        r#"
        var log = [];
        function error(f) { try { f(); return ''; } catch (e) { return e.name; } }
        class Base { constructor(o) { return o; } }
        class Fields extends Base {
          #a = (log.push('init-a'), 1);
          #b = (log.push('init-b'), 2);
          static has(o) { return #a in o; }
          static get(o) { return o.#a; }
          static set(o, v) { o.#a = v; }
        }
        var plain = Object.preventExtensions({});
        var rejected = error(() => new Fields(plain));
        var initializer = log.join(',');
        log = [];
        var target = {}, proxy = new Proxy(target, { isExtensible(t) {
          log.push('ext'); return Reflect.isExtensible(t);
        }});
        new Fields(proxy);
        var fieldOrder = log.join(',');
        Object.freeze(proxy); Fields.set(proxy, 8);
        var frozenValue = Fields.get(proxy);
        log = [];
        var failedProxy = error(() => new Fields(proxy));
        var failedOrder = log.join(',');
        class Methods extends Base { #m() {} #n() {} }
        log = [];
        var m = new Proxy({}, { isExtensible(t) { log.push('methods'); return true; }});
        new Methods(m);
        var groupCount = log.length;
        var revoked = Proxy.revocable({}, {}); revoked.revoke();
        var revokedError = error(() => new Fields(revoked.proxy));
        [rejected, Fields.has(plain), initializer, fieldOrder, frozenValue,
         failedProxy, failedOrder, groupCount, revokedError].join('|')
    "#,
        "TypeError|false|init-a|init-a,ext,init-b,ext|8|TypeError|init-a,ext|1|TypeError",
    );
}

#[test]
#[cfg(feature = "intl")]
fn internal_slot_tests_gc_traces_live_slots_and_breaks_internal_slot_cycles() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            class Box { #value; constructor(v) { this.#value = v; } get() { return this.#value; } }
            var live = new Box({marker: 'alive'});
            var cycle = new Intl.Collator('en-US'), compare = cycle.compare;
        "#,
        );
        let env = engine.interp.global_env.clone();
        // Strong handles live only in helper frames (gc-heap scans the stack).
        let weak = crate::gc_test::in_frame(|| {
            let cycle = engine.interp.get_var("cycle", &env).ok().unwrap();
            crate::value::Gc::downgrade(cycle.as_obj().unwrap())
        });
        engine.interp.collect_garbage_for_host();
        assert_eq!(eval(&mut engine, "live.get().marker"), "alive", "{tier:?}");
        eval(&mut engine, "cycle = null; compare = null; live = null;");
        engine.interp.collect_garbage_for_host();
        assert!(
            !crate::gc_test::alive(&weak),
            "{tier:?}: slot/bound-function cycle retained"
        );
        eval(
            &mut engine,
            "cycle = new Intl.Collator('en-US'); compare = cycle.compare;",
        );
        let weak = crate::gc_test::in_frame(|| {
            let cycle = engine.interp.get_var("cycle", &env).ok().unwrap();
            crate::value::Gc::downgrade(cycle.as_obj().unwrap())
        });
        drop(env);
        drop(engine);
        assert!(
            !crate::gc_test::alive(&weak),
            "{tier:?}: slots retained at teardown"
        );
    }
}
