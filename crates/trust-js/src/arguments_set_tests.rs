//! ECMA-262 e28783d5fc9dc12b3de905961e2c71410b38a202: OrdinarySetWithOwnDescriptor
//! and arguments exotic [[Set]], [[DefineOwnProperty]], [[Delete]] and ParameterMap keys.
//! Expected results are checked against Node; use each function's own `arguments`.
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 32] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            match engine
                .eval(source, false)
                .expect("arguments set fixture parses")
            {
                Completion::Value(actual) => {
                    assert_eq!(actual, expected, "{tier:?}/{threshold}: {source}");
                }
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{threshold}: {name}: {message}\n{source}");
                }
            }
        }
    }
}

#[test]
fn reflect_set_arguments_updates_the_mapped_parameter_and_live_descriptor() {
    check(
        r#"
        function subject(a) {
            var own = arguments, replacement = {tag:7};
            a = 2;
            if (!Reflect.set(own, 0, replacement) || a !== replacement || own[0] !== a)
                throw 'mapped value';
            if (Object.getOwnPropertyDescriptor(own, '0').value !== replacement)
                throw 'mapped descriptor';
            a = {tag:9};
            if (own[0] !== a) throw 'map was severed';
            return a.tag;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "432",
    );
}

#[test]
fn reflect_set_distinct_targets_define_on_the_arguments_receiver() {
    check(
        r#"
        function subject(a) {
            var own = arguments;
            var targets = [{0:1}, Object.create({0:2}), {}, new Proxy({0:3}, {})];
            for (var k = 0; k < targets.length; k++) {
                var value = {tag:k};
                if (!Reflect.set(targets[k], '0', value, own) || a !== value || own[0] !== value)
                    throw 'foreign target receiver';
            }
            var home = {__proto__:{0:0}, put(value){super[0] = value;}};
            var last = {tag:4};
            home.put.call(own, last);
            if (a !== last || own[0] !== last) throw 'super arguments receiver';
            return a.tag;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "192",
    );
}

#[test]
fn arguments_set_foreign_and_primitive_receivers_do_not_write_the_target_map() {
    check(
        r#"
        function subject(a) {
            var own = arguments, proxy = new Proxy(own, {}), receiver = {};
            if (!Reflect.set(own, '0', 7, receiver) || receiver[0] !== 7 || a !== 1 || own[0] !== 1)
                throw 'direct foreign receiver';
            receiver = {};
            if (!Reflect.set(proxy, '0', 8, receiver) || receiver[0] !== 8 || a !== 1 || own[0] !== 1)
                throw 'proxy foreign receiver';
            var inherited = Object.create(own);
            inherited[0] = 9;
            if (inherited[0] !== 9 || a !== 1) throw 'inherited receiver';
            for (var primitive of [undefined, null, 0, false, 'text', Symbol('receiver')]) {
                if (Reflect.set(own, '0', 10, primitive) || Reflect.set(proxy, '0', 11, primitive))
                    throw 'primitive accepted';
                if (a !== 1 || own[0] !== 1) throw 'primitive mapped write';
            }
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "48",
    );
}

#[test]
fn reflect_set_arguments_preserves_attributes_and_allows_existing_nonextensible_keys() {
    check(
        r#"
        function subject(a) {
            var own = arguments;
            Object.defineProperty(own, '0', {enumerable:false, configurable:false});
            Object.preventExtensions(own);
            if (!Reflect.set(own, 0, 7) || !Reflect.set({0:0}, 0, 8, own) || a !== 8)
                throw 'existing nonextensible';
            var d = Object.getOwnPropertyDescriptor(own, '0');
            if (d.value !== 8 || !d.writable || d.enumerable || d.configurable)
                throw 'attributes';
            if (Reflect.set(own, 'new', 9) || Reflect.set({}, 'new', 10, own))
                throw 'new nonextensible';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "384",
    );
}

#[test]
fn reflect_set_new_arguments_keys_do_not_create_parameter_mappings() {
    check(
        r#"
        function subject(a) {
            var own = arguments, symbol = Symbol('key');
            if (!Reflect.set(own, 0, 7) || a !== undefined || own[0] !== 7)
                throw 'missing actual argument mapped';
            if (!Reflect.set({}, 'named', 8, own) || !Reflect.set(own, symbol, 9))
                throw 'new key';
            for (var key of ['0', 'named', symbol]) {
                var d = Object.getOwnPropertyDescriptor(own, key);
                if (!d.writable || !d.enumerable || !d.configurable) throw 'new attributes';
            }
            a = 10;
            if (own[0] !== 7) throw 'new property became mapped';
            return own.named + own[symbol];
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject();
        total
        "#,
        "816",
    );
}

#[test]
fn reflect_set_checks_target_and_receiver_descriptors_before_writing_arguments() {
    check(
        r#"
        function subject(a) {
            var own = arguments, calls = 0;
            var frozenTarget = Object.defineProperty({}, '0', {value:2, writable:false});
            if (Reflect.set(frozenTarget, '0', 7, own) || a !== 1) throw 'target writable';
            Object.defineProperty(own, '0', {
                get:function(){calls++; return 3;},
                set:function(v){if (this !== own || v !== 8) throw 'setter receiver'; calls += 10;},
                configurable:true
            });
            if (Reflect.set({0:0}, '0', 7, own) || calls !== 0 || a !== 1)
                throw 'receiver accessor invoked';
            if (!Reflect.set(own, '0', 8) || calls !== 10 || a !== 1)
                throw 'target setter omitted';
            return calls;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "480",
    );
}

#[test]
fn reflect_set_respects_argument_unmapping_after_nonwritable_accessor_and_delete() {
    check(
        r#"
        function nonwritable(a) {
            var own = arguments;
            a = 2;
            Object.defineProperty(own, '0', {writable:false});
            a = 3;
            if (Reflect.set(own, 0, 7) || Reflect.set({0:0}, 0, 8, own) || own[0] !== 2 || a !== 3)
                throw 'nonwritable';
            Object.defineProperty(own, '0', {writable:true});
            if (!Reflect.set(own, 0, 9) || own[0] !== 9 || a !== 3) throw 'remapped writable';
            return a;
        }
        function accessor(a) {
            var own = arguments;
            Object.defineProperty(own, '0', {get:function(){return 2;}, configurable:true});
            Object.defineProperty(own, '0', {value:3, writable:true});
            if (!Reflect.set(own, 0, 7) || a !== 1 || own[0] !== 7) throw 'remapped accessor';
            return a;
        }
        function removed(a) {
            var own = arguments;
            delete own[0];
            if (!Reflect.set(own, 0, 7) || a !== 1 || own[0] !== 7) throw 'remapped delete';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++)
            total += nonwritable(1) + accessor(1) + removed(1);
        total
        "#,
        "240",
    );
}

#[test]
fn arguments_set_proxy_receiver_runs_descriptor_traps_and_forwards_only_value() {
    check(
        r#"
        function subject(a) {
            var own = arguments, trace = [];
            var proxy = new Proxy(own, {
                getOwnPropertyDescriptor:function(t,k) {
                    trace.push('get:'+k); return Reflect.getOwnPropertyDescriptor(t,k);
                },
                defineProperty:function(t,k,d) {
                    trace.push('define:'+Object.keys(d).join(','));
                    return Reflect.defineProperty(t,k,d);
                }
            });
            if (!Reflect.set(proxy, '0', 7) || a !== 7 || own[0] !== 7)
                throw 'forwarded proxy receiver';
            if (trace.join('|') !== 'get:0|define:value') throw 'descriptor trap order';
            trace = [];
            if (!Reflect.set({0:1}, '0', 8, proxy) || a !== 8 || own[0] !== 8)
                throw 'foreign proxy receiver';
            if (trace.join('|') !== 'get:0|define:value') throw 'foreign descriptor trap order';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "384",
    );
}

#[test]
fn arguments_set_proxy_define_false_throw_and_success_without_forward_do_not_alias() {
    check(
        r#"
        function subject(a) {
            var own = arguments, trace = [], sentinel = {};
            for (var mode = 0; mode < 3; mode++) {
                var proxy = new Proxy(own, {
                    getOwnPropertyDescriptor:function(t,k) {
                        trace.push('get'); return Reflect.getOwnPropertyDescriptor(t,k);
                    },
                    defineProperty:function(t,k,d) {
                        trace.push('define');
                        if (mode === 1) throw sentinel;
                        return mode === 2;
                    }
                });
                var result, caught;
                try {result = Reflect.set(proxy, '0', 7);} catch(e) {caught = e;}
                if (mode === 1 ? caught !== sentinel : result !== (mode === 2))
                    throw 'proxy result';
                if (a !== 1 || own[0] !== 1) throw 'premature map write';
            }
            if (trace.join(',') !== 'get,define,get,define,get,define') throw 'trap order';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "48",
    );
}

#[test]
fn arguments_set_distinct_rejecting_receivers_leave_target_parameters_unchanged() {
    check(
        r#"
        function subject(a) {
            var own = arguments, setterCalls = 0;
            var receivers = [Object.preventExtensions({}),
                Object.defineProperty({}, '0', {value:2, writable:false}),
                Object.defineProperty({}, '0', {
                    get:function(){throw 'receiver getter';},
                    set:function(){setterCalls++;}, configurable:true
                })];
            for (var target of [own, new Proxy(own, {})]) {
                for (var receiver of receivers) {
                    if (Reflect.set(target, '0', 7, receiver) || a !== 1 || own[0] !== 1)
                        throw 'rejecting foreign receiver';
                }
            }
            if (setterCalls !== 0) throw 'receiver setter called';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "48",
    );
}

#[test]
fn arguments_set_proxy_descriptor_throw_or_reentrant_delete_never_writes_an_old_map() {
    check(
        r#"
        function subject(a) {
            var own = arguments, sentinel = {}, trace = [];
            var throwing = new Proxy(own, {getOwnPropertyDescriptor:function() {
                if (a !== 1) throw 'premature mapped write';
                throw sentinel;
            }});
            var caught;
            try {Reflect.set(throwing, '0', 7);} catch(e) {caught = e;}
            if (caught !== sentinel || a !== 1 || own[0] !== 1) throw 'descriptor throw';
            var disconnecting = new Proxy(own, {
                getOwnPropertyDescriptor:function(t,k) {
                    if (a !== 1) throw 'map written before descriptor';
                    trace.push('get');
                    var d = Reflect.getOwnPropertyDescriptor(t,k);
                    if (!Reflect.deleteProperty(t,k)) throw 'reentrant delete';
                    a = 3;
                    return d;
                },
                defineProperty:function(t,k,d) {
                    trace.push('define:'+Object.keys(d).join(','));
                    return Reflect.defineProperty(t,k,d);
                }
            });
            if (!Reflect.set(disconnecting, '0', 7) || a !== 3 || own[0] !== 7)
                throw 'reentrant define';
            a = 4;
            if (own[0] !== 7 || trace.join('|') !== 'get|define:value') throw 'old map reused';
            var d = Object.getOwnPropertyDescriptor(own, '0');
            if (d.writable || d.enumerable || d.configurable) throw 'absent target partial descriptor';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "192",
    );
}

#[test]
fn arguments_proxy_prototype_rejection_obeys_reflect_vs_assignment_strictness() {
    check(
        r#"
        function reflect(own) {
            'use strict';
            return Reflect.set(own, 'new', 7);
        }
        function assign(own) {
            'use strict';
            own.new = 7;
        }
        function subject(a) {
            var own = arguments;
            Object.setPrototypeOf(own, new Proxy({}, {}));
            Object.preventExtensions(own);
            if (reflect(own) !== false || a !== 1 || own[0] !== 1) throw 'strict Reflect throw';
            var caught;
            try {assign(own);} catch(e) {caught = e;}
            if (!caught || caught.name !== 'TypeError') throw 'strict assignment accepted';
            if (!Reflect.set(own, 0, 8) || a !== 8) throw 'strictness restoration';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "384",
    );
}

#[test]
fn arguments_set_exact_canonical_keys_leave_other_numeric_spellings_independent() {
    check(
        r#"
        function subject(a,b) {
            var own = arguments, proxy = new Proxy(own, {}), symbol = Symbol('0');
            var keys = ['00', '+0', '-0', '01', '1.0', '1e0', ' 0', '', '4294967294',
                '4294967295', '4294967296', '9007199254740991', symbol];
            for (var k = 0; k < keys.length; k++) {
                if (!Reflect.set(own, keys[k], k + 10) || a !== 1 || b !== 2)
                    throw 'noncanonical key mapped';
                if (!Reflect.set(proxy, keys[k], k + 20) || own[keys[k]] !== k + 20 || a !== 1 || b !== 2)
                    throw 'proxy noncanonical key';
            }
            if (!Reflect.set(own, '0', 7) || !Reflect.set({1:0}, '1', 8, own) || a !== 7 || b !== 8)
                throw 'canonical keys not mapped';
            return a + b;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1,2);
        total
        "#,
        "720",
    );
}

#[test]
fn arguments_noncanonical_deletes_and_descriptor_changes_do_not_unmap_parameters() {
    check(
        r#"
        function subject(a,b) {
            var own = arguments;
            for (var key of ['00', '+0', '-0', '01']) {
                own[key] = 3;
                if (!Reflect.deleteProperty(own, key)) throw 'ordinary reflected delete';
                own[key] = 3;
                delete own[key];
                Object.defineProperty(own, key, {value:4, writable:false, configurable:true});
                Object.defineProperty(own, key, {get:function(){return 5;}, configurable:true});
                if (own[key] !== 5) throw 'ordinary accessor';
            }
            if (!Reflect.set(own, '0', 7) || !Reflect.set(own, '1', 8) || a !== 7 || b !== 8)
                throw 'wrong map severed';
            return a + b;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1,2);
        total
        "#,
        "720",
    );
}

#[test]
fn reflect_set_duplicate_strict_and_nonsimple_arguments_keep_their_mapping_rules() {
    check(
        r#"
        function duplicate(a,a) {
            var own = arguments;
            if (!Reflect.set(own, 0, 7) || a !== 2 || own[0] !== 7) throw 'first duplicate';
            if (!Reflect.set(own, 1, 8) || a !== 8 || own[1] !== 8) throw 'last duplicate';
            return a;
        }
        function strict(a) {
            'use strict';
            var rejecting = new Proxy(arguments, {defineProperty:function(){return false;}});
            if (Reflect.set(rejecting, 0, 8) || a !== 1) throw 'strict proxy Reflect rejection';
            if (!Reflect.set(arguments, 0, 7) || a !== 1 || arguments[0] !== 7)
                throw 'strict mapped';
            Object.defineProperty(arguments, '0', {writable:false});
            if (Reflect.set(arguments, 0, 8) || Reflect.set(new Proxy(arguments, {}), 0, 8))
                throw 'strict Reflect rejection';
            return a;
        }
        function nonsimple(a = 0) {
            if (!Reflect.set(arguments, 0, 7) || a !== 1 || arguments[0] !== 7)
                throw 'nonsimple mapped';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++)
            total += duplicate(1,2) + strict(1) + nonsimple(1);
        total
        "#,
        "480",
    );
}

#[test]
fn reflect_set_argument_alias_survives_return_and_reentrant_key_conversion() {
    check(
        r#"
        function escaped(a) {
            return {own:arguments, read:function(){return a;}};
        }
        function reentrant(o) {
            var own = arguments, original = o;
            var key = {toString:function(){Reflect.set(own, 0, {value:22}); return 'x';}};
            var value = o[key];
            if (original === o || value !== 7 || o.value !== 22 || own[0] !== o)
                throw 'reentrant mapped receiver';
            return value + o.value;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) {
            var saved = escaped(1), value = {tag:9};
            if (!Reflect.set(saved.own, 0, value) || saved.read() !== value || saved.own[0] !== value)
                throw 'escaped mapping';
            total += reentrant({get x(){return 7;}, value:11}) + saved.read().tag;
        }
        total
        "#,
        "1824",
    );
}

#[test]
fn reflect_delete_arguments_success_severs_the_parameter_alias_permanently() {
    check(
        r#"
        function subject(a) {
            var own = arguments;
            if (!Reflect.deleteProperty(own, '0') || Object.hasOwn(own, '0') || own[0] !== undefined)
                throw 'own delete';
            if (!Reflect.deleteProperty(own, '0')) throw 'absent delete';
            if (!Reflect.set(own, '0', 7) || a !== 1 || own[0] !== 7) throw 'remapped delete';
            a = 9;
            if (own[0] !== 7) throw 'parameter still mapped';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "432",
    );
}

#[test]
fn reflect_delete_arguments_failure_preserves_mapping_even_in_strict_caller() {
    check(
        r#"
        function reject(own) {
            'use strict';
            if (Reflect.deleteProperty(own, '0')) throw 'nonconfigurable deleted';
            if (Reflect.deleteProperty(new Proxy(own, {}), '0')) throw 'proxy nonconfigurable deleted';
        }
        function subject(a) {
            var own = arguments;
            Object.defineProperty(own, '0', {configurable:false});
            reject(own);
            if (!Reflect.set(own, '0', 7) || a !== 7 || own[0] !== 7) throw 'failed delete unmapped';
            a = 9;
            if (own[0] !== 9) throw 'failed delete lost alias';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++) total += subject(1);
        total
        "#,
        "432",
    );
}

#[test]
fn reflect_delete_proxy_arguments_unmaps_only_when_the_target_is_actually_deleted() {
    check(
        r#"
        function subject(a) {
            var own = arguments, sentinel = {}, trace = [];
            for (var mode = 0; mode < 3; mode++) {
                var proxy = new Proxy(own, {deleteProperty:function(t,k) {
                    trace.push(k);
                    if (mode === 1) throw sentinel;
                    return mode === 2;
                }});
                var result, caught;
                try {result = Reflect.deleteProperty(proxy, '0');} catch(e) {caught = e;}
                if (mode === 1 ? caught !== sentinel : result !== (mode === 2)) throw 'proxy delete result';
                if (!Reflect.set(own, 0, 7 + mode) || a !== 7 + mode || own[0] !== a)
                    throw 'nondeleting trap unmapped';
            }
            if (trace.join(',') !== '0,0,0') throw 'delete trap order';
            var forwarding = new Proxy(own, {deleteProperty:function(t,k) {
                return Reflect.deleteProperty(t,k);
            }});
            if (!Reflect.deleteProperty(forwarding, '0') || Object.hasOwn(own, '0'))
                throw 'forwarded delete';
            if (!Reflect.set(own, 0, 10) || a !== 9 || own[0] !== 10)
                throw 'forwarded delete remapped';
            return a;
        }
        function transparent(a) {
            var own = arguments;
            if (!Reflect.deleteProperty(new Proxy(own, {}), '0')) throw 'transparent delete';
            if (!Reflect.set(own, 0, 7) || a !== 1 || own[0] !== 7) throw 'transparent remapped';
            return a;
        }
        var total = 0;
        for (var iteration = 0; iteration < 48; iteration++)
            total += subject(1) + transparent(1);
        total
        "#,
        "480",
    );
}
