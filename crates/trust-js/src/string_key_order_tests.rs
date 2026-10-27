//! ECMA-262 editor's draft e28783d5, spec.html: CanonicalNumericIndexString 5787,
//! IsLessThan 6085, StringGetOwnProperty 14889, StringToCodePoints 16749, SV 17829,
//! String.raw 36175, CompareArrayElements 41667.

use crate::bytecode::Tier;
use crate::value::Value;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1+2", false), Ok(Completion::Value(v)) if v == "3"));
    }
}

#[test]
fn only_canonical_in_range_indices_are_a_strings_own_properties() {
    // "01", "+1", "1.0" and " 1" are not canonical numeric strings, and an index past the end
    // names no own property: each read continues to String.prototype, and a String object's
    // ordinary own property under such a key shadows nothing of the string's.
    check(
        r#"
        function run() {
            var keys = ['1', '01', '+1', '1.0', ' 1', '-0', '3', '4294967295'], out = [];
            String.prototype['01'] = 'p01'; String.prototype['+1'] = 'pplus';
            String.prototype['3'] = 'p3'; String.prototype['-0'] = 'pm0';
            try {
                for (var k of keys) {
                    var w = Object('abc');
                    w[k] = 'own';
                    var d = Object.getOwnPropertyDescriptor(Object('abc'), k);
                    out.push(k + ':' + 'abc'[k] + ',' + Object('abc')[k] + ',' + w[k] + ','
                        + (k in Object('abc')) + ',' + Object.prototype.hasOwnProperty.call('abc', k)
                        + ',' + (d ? d.value : '-'));
                }
                var plain = Object('ab'); plain['+0'] = 1; plain['5'] = 2;
                out.push(Object.keys(plain).join(' '), '+0' in Object('ab'), '00' in Object('ab'));
                var a = 'xyz', n = 0;
                for (var i = 0; i < 5; i++) n += a[i] === undefined ? 1 : 0;
                out.push(n, a[3], a['3']);
            } finally {
                delete String.prototype['01']; delete String.prototype['+1'];
                delete String.prototype['3']; delete String.prototype['-0'];
            }
            return out.join('|');
        }
        run()
        "#,
        "1:b,b,b,true,true,b\
         |01:p01,p01,own,true,false,-\
         |+1:pplus,pplus,own,true,false,-\
         |1.0:undefined,undefined,own,false,false,-\
         | 1:undefined,undefined,own,false,false,-\
         |-0:pm0,pm0,own,true,false,-\
         |3:p3,p3,own,true,false,-\
         |4294967295:undefined,undefined,own,false,false,-\
         |0 1 5 +0|false|false|1|p3|p3",
    );
}

#[test]
fn strings_sort_and_compare_by_utf16_code_units() {
    // U+1F600 is the surrogate pair D83D DE00, below U+FF61 and U+E000 in code units though
    // above them in code points; a lone surrogate sorts by its own unit.
    check(
        r#"
        function hex(list) {
            return list.map(function (s) {
                var out = [];
                for (var i = 0; i < s.length; i++) out.push(s.charCodeAt(i).toString(16));
                return out.join(' ');
            }).join(',');
        }
        function run() {
            var words = ['｡', '\u{1F600}', '', '\uD800', '\u{10000}', 'a', '\uDC00x',
                         'a\u{1F600}', 'a￿', 'é', 'z'];
            var sorted = words.slice().sort();
            var ok = true;
            for (var i = 1; i < sorted.length; i++) ok = ok && !(sorted[i] < sorted[i - 1]);
            var objects = words.map(function (w) { return {toString: function () { return w; }}; });
            return [hex(sorted), ok, hex(objects.sort().map(String)), hex(words.toSorted()),
                    '\u{1F600}' < '｡', 'a\u{1F600}' < 'a￿', '\uDBFF' < '\u{10FFFF}'
                   ].join('|');
        }
        run()
        "#,
        "61,61 d83d de00,61 ffff,7a,e9,d800,d800 dc00,d83d de00,dc00 78,e000,ff61\
         |true\
         |61,61 d83d de00,61 ffff,7a,e9,d800,d800 dc00,d83d de00,dc00 78,e000,ff61\
         |61,61 d83d de00,61 ffff,7a,e9,d800,d800 dc00,d83d de00,dc00 78,e000,ff61\
         |true|true|true",
    );
}

#[test]
fn unit_comparison_matches_a_unit_by_unit_reference() {
    use crate::jstr::{from_units, units};
    use crate::lstr::LStr;
    let samples: Vec<Vec<u16>> = vec![
        vec![],
        vec![0x61],
        vec![0x61, 0x62],
        vec![0xE9],
        vec![0x4E2D],
        vec![0xE000],
        vec![0xFF61],
        vec![0xFFFF],
        vec![0xD83D, 0xDE00],
        vec![0xD83D],
        vec![0xD83D, 0x61],
        vec![0xDE00],
        vec![0xD800, 0xDC00],
        vec![0xDBFF, 0xDFFF],
        vec![0x61, 0xD83D, 0xDE00],
        vec![0x61, 0xFFFF],
        vec![0x61, 0xE9, 0x62],
        vec![0x61, 0xE9, 0x63],
    ];
    for a in &samples {
        for b in &samples {
            let (sa, sb) = (from_units(a), from_units(b));
            assert_eq!(units(&sa), *a);
            assert_eq!(
                LStr::from(sa.as_str()).cmp_units(&LStr::from(sb.as_str())),
                a.cmp(b),
                "{a:x?} vs {b:x?}"
            );
        }
    }
}

#[test]
fn an_array_data_property_under_a_non_canonical_key_keeps_the_length() {
    let mut engine = Engine::new();
    assert!(matches!(
        engine.eval("var array = [1];", false),
        Ok(Completion::Value(_))
    ));
    let global = Value::Obj(engine.interp.global);
    let array = engine
        .interp
        .get_member_key(&global, "array")
        .ok()
        .expect("reads");
    for key in ["01", "+1", "1.0"] {
        let defined =
            crate::builtins::cdp_or_throw(&mut engine.interp, &array, key, Value::Num(7.0));
        assert!(defined.is_ok(), "{key}");
    }
    let defined = crate::builtins::cdp_or_throw(&mut engine.interp, &array, "4", Value::Num(8.0));
    assert!(defined.is_ok());
    let length = engine.interp.get_member_key(&array, "length").ok();
    assert_eq!(length.and_then(|length| length.as_num_opt()), Some(5.0));
}

#[test]
fn object_spread_and_rest_copy_a_strings_code_units() {
    // CopyDataProperties of a String copies its StringGetOwnProperty indices: one a code unit,
    // so a surrogate pair gives two properties.
    check(
        r#"
        function run() {
            function units(o) {
                return Object.keys(o).map(function (k) {
                    return k + '=' + o[k].charCodeAt(0).toString(16);
                }).join(' ');
            }
            var spread = {...'a\u{1F600}b'};
            var {0: first, ...rest} = 'x\u{1F600}';
            var assigned = Object.assign({}, 'a\u{1F600}');
            return [units(spread), first, units(rest), units(assigned),
                    Object.keys({...'\uD800\u00e9'}).length].join('|');
        }
        run()
        "#,
        "0=61 1=d83d 2=de00 3=62|x|1=d83d 2=de00|0=61 1=d83d 2=de00|2",
    );
}

#[test]
fn literal_keys_rejoin_surrogate_pairs_split_in_source() {
    // eval's source is the string's code points, so a raw high surrogate before a `\u` low one
    // (or a `\u` high before a raw low) is a pair in the literal's SV: the key is the
    // character, and reads, writes and computed keys of it reach the same property.
    check(
        r#"
        function run(){
            var out=[];
            var o=eval('({"\ud83d\\ude00": 1, "\\ud83d\ude01": 2})'),k=Object.keys(o);
            out.push(k[0]==='\u{1f600}',k[1]==='\u{1f601}',o[k[0]],o[k[1]],k[0] in o,o['\u{1f601}']);
            var g=Function('return {"\ud83d\\ude00": 3}')();
            out.push(g['\u{1f600}'],Object.keys(g)[0].length);
            var c=eval('(class { "\ud83d\\ude00"() { return 7; } })');
            out.push(new c()['\u{1f600}']());
            var two=eval('({"\ud83d\\ude00": 1, ["\ud83d\ude00"]: 2})');
            out.push(Object.keys(two).length,two['\u{1f600}']);
            var o3={};o3['\u{1f600}']=5;
            out.push(eval('o3["\ud83d\\ude00"]'),eval('"\ud83d\\ude00" in o3'));
            eval('o3["\ud83d\\ude00"] = 6');
            out.push(o3['\u{1f600}'],Object.keys(o3).length);
            var {"\ud83d\ude00": d}=eval('({"\ud83d\\ude00": 8})');
            out.push(d);
            var lone=eval('({"\ud83d\\u0041": 1})'),lk=Object.keys(lone)[0];
            out.push(lk.length,lk.charCodeAt(0),lone[lk]);
            var big=eval('({"\udbff\\udfff": 9})'),bk=Object.keys(big)[0];
            out.push(bk==='\u{10ffff}',big['\u{10ffff}']);
            return out.join('|');
        }run()
        "#,
        "true|true|1|2|true|2|3|2|7|1|2|5|true|6|1|8|2|55357|1|true|9",
    );
}

#[test]
fn string_raw_converts_raw_with_to_object() {
    // A string `raw` is boxed: its code units are the segments, a pair's halves apart.
    check(
        r#"
        var raw=[];
        for(var r of ['abc','a\ud83d\ude00b','',7,true]){
            raw.push(JSON.stringify(String.raw({raw:r},'-','+')));
        }
        try{String.raw({raw:undefined});}catch(e){raw.push(e.constructor.name);}
        try{String.raw();}catch(e){raw.push(e.constructor.name);}
        raw.join('|')
        "#,
        r#""a-b+c"|"a-\ud83d+\ude00b"|""|""|""|TypeError|TypeError"#,
    );
}
