//! ECMA-262 e28783d5fc9dc12b3de905961e2c71410b38a202: ArraySetLength,
//! Array [[DefineOwnProperty]], OrdinarySetWithOwnDescriptor and ToUint32.
//! These exercise the shared operation from assignment and both define APIs.
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 32] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            match engine
                .eval(source, false)
                .expect("array length fixture parses")
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
fn array_length_full_number_range_modulo_then_second_conversion() {
    check(
        r#"
        var cases = [
          [18446744073709551616, 0], [-18446744073709547520, 4096],
          [18446744073709559808, 8192], [Number.MAX_VALUE, 0],
          [Infinity, 0], [-Infinity, 0], [NaN, 0], [-0, 0],
          [4294967299, 3], [3.75, 3], [-0.75, 0]
        ];
        var total = 0;
        function perform(mode, pair) {
          var a = [1, 2, 3], calls = 0;
          var value = { [Symbol.toPrimitive](hint) {
            if (hint !== 'number') throw 'wrong hint';
            return pair[calls++];
          }};
          if (mode === 0) a.length = value;
          else if (mode === 1) Object.defineProperty(a, 'length', {value});
          else if (!Reflect.defineProperty(a, 'length', {value})) throw 'define failed';
          if (calls !== 2 || a.length !== pair[1] || Object.is(a.length, -0))
            throw 'wrong length or coercion';
          total++;
        }
        for (var mode = 0; mode < 3; mode++)
          for (var k = 0; k < cases.length; k++) perform(mode, cases[k]);
        total
        "#,
        "33",
    );
}

#[test]
fn array_length_primitive_numbers_cover_uint32_and_floating_point_boundaries() {
    check(
        r#"
        var valid = [0, -0, 1, 17, 2147483647, 2147483648, 4294967294, 4294967295];
        var invalid = [-1, -0.25, 0.25, 2147483647.5, 4294967295.5,
          4294967296, 4294967297, 9007199254740992, 18446744073709551616,
          Number.MAX_VALUE, Number.MIN_VALUE, -Number.MIN_VALUE, NaN, Infinity, -Infinity];
        var total = 0;
        function write(a, n, mode) {
          if (mode === 0) return a.length = n;
          if (mode === 1) return Object.defineProperty(a, 'length', {value:n});
          if (mode === 2) return Reflect.defineProperty(a, 'length', {value:n});
          return Reflect.set(a, 'length', n);
        }
        for (var mode = 0; mode < 4; mode++) {
          for (var n of valid) {
            var a = [7,8,9], result = write(a,n,mode);
            if (a.length !== n || Object.is(a.length,-0)) throw 'wrong normalized length';
            if (mode === 0 && !Object.is(result,n)) throw 'wrong assignment result';
            if (mode === 1 && result !== a) throw 'wrong define result';
            if (mode >= 2 && result !== true) throw 'wrong reflect result';
            if ((0 in a) !== (n > 0) || (1 in a) !== (n > 1) ||
                (2 in a) !== (n > 2) || 3 in a) throw 'wrong elements';
            total++;
          }
          for (var n of invalid) {
            var a = [7,8,9], error = '';
            try { write(a,n,mode); } catch(e) { error = e.name; }
            if (error !== 'RangeError' || a.length !== 3 || a.join(',') !== '7,8,9')
              throw 'wrong rejection';
            total++;
          }
        }
        total
        "#,
        "92",
    );
}

#[test]
fn array_length_unchanged_value_preserves_conversion_and_descriptor_changes() {
    check(
        r#"
        var total = 0;
        function exercise(mode) {
          var a = [7,8,9], calls = 0;
          var value = {[Symbol.toPrimitive](hint) {
            if (hint !== 'number') throw 'wrong hint';
            calls++;
            if (calls === 1) a[3] = 10;
            else a.length = 3;
            return 3;
          }};
          if (mode === 0) a.length = value;
          else if (mode === 1) Object.defineProperty(a,'length',{value});
          else if (mode === 2) Reflect.defineProperty(a,'length',{value});
          else Reflect.set(a,'length',value);
          if (calls !== 2 || a.length !== 3 || 3 in a) throw 'lost conversion';
          Object.preventExtensions(a);
          if (!Reflect.defineProperty(a,'length',{value:3,writable:true})) throw 'same value failed';
          if (Reflect.defineProperty(a,'length',{value:3,enumerable:true}) ||
              Reflect.defineProperty(a,'length',{value:3,configurable:true})) throw 'wrong attributes';
          if (!Reflect.defineProperty(a,'length',{value:3,writable:false})) throw 'freeze failed';
          if (Object.getOwnPropertyDescriptor(a,'length').writable) throw 'freeze skipped';
          if (!Reflect.defineProperty(a,'length',{value:3}) ||
              Reflect.defineProperty(a,'length',{value:3,writable:true}) ||
              Reflect.set(a,'length',3)) throw 'wrong frozen result';
          total++;
        }
        for (var mode = 0; mode < 4; mode++) exercise(mode);
        var zero = [];
        Object.defineProperty(zero,'length',{value:-0,writable:false});
        if (Object.is(zero.length,-0) || !Reflect.defineProperty(zero,'length',{value:-0}))
          throw 'wrong frozen zero';
        total
        "#,
        "4",
    );
}

#[test]
fn array_length_numeric_looking_names_never_grow_or_block_shrink() {
    check(
        r#"
        var names = ['00','01','+1','1e0','1.0','-0','4294967295','9007199254740991'];
        var result = [];
        for (var mode = 0; mode < 2; mode++) {
          var a = [];
          for (var k = 0; k < names.length; k++) {
            if (mode === 0) a[names[k]] = k + 10;
            else Object.defineProperty(a, names[k], {value:k + 10, configurable:false});
          }
          if (a.length !== 0) throw 'named property grew length';
          a[2] = 22;
          Object.defineProperty(a, 'length', {value:0});
          if (a.length !== 0 || 2 in a) throw 'index survived shrink';
          for (var k = 0; k < names.length; k++)
            if (a[names[k]] !== k + 10) throw 'named property deleted';
          Object.defineProperty(a, 'length', {writable:false});
          a['0004'] = 44;
          Object.defineProperty(a, '+5', {value:55});
          result.push(a.length + ':' + a['0004'] + ':' + a['+5']);
        }
        result.join('|')
        "#,
        "0:44:55|0:44:55",
    );
}

#[test]
fn array_length_coercion_can_freeze_same_value_or_change_current_length() {
    check(
        r#"
        function frozen(mode) {
          'use strict';
          var a = [1,2,3], calls = 0;
          var v = {valueOf() {
            calls++;
            Object.defineProperty(a, 'length', {writable:false});
            return 3;
          }};
          var ok;
          if (mode === 0) ok = ((a.length = v) === v);
          else if (mode === 1) ok = Reflect.set(a, 'length', v);
          else ok = Reflect.defineProperty(a, 'length', {value:v});
          return ok + ':' + calls + ':' + a.length + ':' +
            Object.getOwnPropertyDescriptor(a, 'length').writable;
        }
        var a = [0,1,2], calls = 0;
        var v = {valueOf() {
          calls++;
          if (calls === 1) { a[8] = 8; return 2; }
          a.length = 7; a[6] = 6; return 2;
        }};
        Object.defineProperty(a, 'length', {value:v});
        [frozen(0), frozen(1), frozen(2), calls + ':' + a.length + ':' + (6 in a) + ':' + (8 in a)].join('|')
        "#,
        "true:2:3:false|true:2:3:false|true:2:3:false|2:2:false:false",
    );
}

#[test]
fn array_length_receiver_descriptor_rejects_before_conversion() {
    check(
        r#"
        var a = [1], calls = 0, setters = 0;
        Object.defineProperty(a, 'length', {writable:false});
        var poison = {valueOf(){calls++; throw 'conversion must not run';}};
        var direct = Reflect.set(a, 'length', poison);
        var foreign = Reflect.set({length:0}, 'length', poison, a);
        Object.defineProperty(a, '0', {get(){throw 'getter must not run';}, set(v){setters++;}});
        var element = Reflect.set({'0':0}, '0', 7, a);
        [direct, foreign, element, calls, setters, a.length].join(':')
        "#,
        "false:false:false:0:0:1",
    );
}

#[test]
fn array_length_foreign_array_receiver_defines_without_replaying_setters() {
    check(
        r#"
        var calls = 0, a = [];
        var proto = {set 0(v){calls++;}, set named(v){calls++;}};
        Object.setPrototypeOf(a, proto);
        var index = Reflect.set({'0':1}, '0', 7, a);
        var named = Reflect.set({named:1}, 'named', 8, a);
        var desc = Object.getOwnPropertyDescriptor(a, '0');
        [index, named, calls, a.length, a[0], a.named,
          desc.writable, desc.enumerable, desc.configurable].join(':')
        "#,
        "true:true:0:1:7:8:true:true:true",
    );
}

#[test]
fn array_length_proxy_receiver_observes_get_descriptor_define_then_conversions() {
    check(
        r#"
        var a = [0,1,2], log = [];
        var receiver = new Proxy(a, {
          getOwnPropertyDescriptor(target, key) {
            log.push('get:' + key); return Reflect.getOwnPropertyDescriptor(target,key);
          },
          defineProperty(target, key, descriptor) {
            log.push('define:' + Object.keys(descriptor).join(','));
            return Reflect.defineProperty(target,key,descriptor);
          }
        });
        var value = {valueOf(){log.push('convert'); return 1;}};
        var ok = Reflect.set({length:0}, 'length', value, receiver);
        [ok, a.length, 1 in a, log.join('|')].join(':')
        "#,
        "true:1:false:get:length|define:value|convert|convert",
    );
}

#[test]
fn array_length_blocked_shrink_deletes_only_above_highest_blocker() {
    check(
        r#"
        var out = [];
        for (var mode = 0; mode < 3; mode++) {
          var a = [0,1,2,3,4,5,6,7,8];
          Object.defineProperty(a, '2', {configurable:false});
          Object.defineProperty(a, '5', {configurable:false});
          Object.defineProperty(a, '08', {value:88, configurable:false});
          var symbol = Symbol(); a[symbol] = 99;
          var ok;
          if (mode === 0) ok = Reflect.set(a, 'length', 1);
          else ok = Reflect.defineProperty(a, 'length', {value:1, writable:mode === 1});
          out.push([ok, a.length, 2 in a, 3 in a, 4 in a, 5 in a,
            6 in a, 7 in a, 8 in a, a['08'], a[symbol],
            Object.getOwnPropertyDescriptor(a, 'length').writable].join(':'));
        }
        out.join('|')
        "#,
        "false:6:true:true:true:true:false:false:false:88:99:true|false:6:true:true:true:true:false:false:false:88:99:true|false:6:true:true:true:true:false:false:false:88:99:false",
    );
}

#[test]
fn array_length_sparse_maximum_index_and_nonextensible_array() {
    check(
        r#"
        var a = [];
        a['4294967295'] = 7;
        Object.defineProperty(a, '4294967294', {value:8, configurable:false});
        var old = a.length;
        var ok = Reflect.defineProperty(a, 'length', {value:0, writable:false});
        var b = [0,1,2]; Object.preventExtensions(b);
        var shrink = Reflect.defineProperty(b, 'length', {value:1});
        var grow = Reflect.defineProperty(b, 'length', {value:8});
        [old, ok, a.length, a['4294967295'], a['4294967294'],
          Object.getOwnPropertyDescriptor(a, 'length').writable,
          shrink, grow, b.length, 1 in b].join(':')
        "#,
        "4294967295:false:4294967295:7:8:false:true:true:8:false",
    );
}

#[test]
fn array_length_error_order_and_abrupt_second_conversion() {
    check(
        r#"
        var a = [1,2,3], log = [], calls = 0;
        var v = {valueOf(){log.push(++calls); if(calls === 2) throw 'second'; return 0;}};
        try { a.length = v; } catch(e) { log.push(e); }
        log.push(a.length);
        for (var value of [-1, 1.5, NaN, Infinity, 4294967296, 1n, Symbol()]) {
          try { Reflect.defineProperty(a, 'length', {value, enumerable:true}); }
          catch(e) { log.push(e.name); }
        }
        log.push(Reflect.defineProperty(a, 'length', {value:1, enumerable:true}));
        log.push(a.length);
        log.join(':')
        "#,
        "1:2:second:3:RangeError:RangeError:RangeError:RangeError:RangeError:TypeError:TypeError:false:3",
    );
}

#[test]
fn array_length_normalization_assignment_result_and_strict_failure() {
    check(
        r#"
        var a = [1,2], minus = (a.length = -0), text = (a.length = '2');
        var zero = Object.is(a.length = -0, -0) && !Object.is(a.length, -0);
        a[2] = 2; Object.defineProperty(a, '2', {configurable:false});
        var error = '';
        try { (function(){'use strict'; a.length = 0;})(); }
        catch(e) { error = e.name; }
        [Object.is(minus,-0), typeof text, text, zero, error, a.length].join(':')
        "#,
        "true:string:2:true:TypeError:3",
    );
}

#[test]
fn array_length_bulk_shrink_never_calls_index_accessors_and_keeps_holes() {
    check(
        r#"
        var a = [], gets = 0;
        for (var k = 0; k < 2000; k++) if (k % 3) a[k] = {k};
        Object.defineProperty(a, '1800', {get(){gets++; throw 'getter';}, configurable:true});
        Object.defineProperty(a, 'length', {value:17, writable:false});
        var sum = 0;
        for (var k = 0; k < a.length; k++) if (k in a) sum += a[k].k;
        [gets, a.length, 0 in a, 1 in a, 18 in a, 1800 in a, sum,
          Object.getOwnPropertyDescriptor(a, 'length').writable].join(':')
        "#,
        "0:17:false:true:false:false:91:false",
    );
}
