//! ECMA-262 snapshot e28783d5, String.prototype.slice/substring/substr
//! (spec.html:36827, 36924, 54429). Substring boundaries remain UTF-16 code-unit boundaries.

use crate::bytecode::Tier;
use crate::value::Value;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("substring fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1+2", false), Ok(Completion::Value(value)) if value == "3"));
    }
}

#[test]
fn unicode_substrings_preserve_every_code_unit_boundary() {
    check(
        r#"
        function run(){
            var base='x'.repeat(64)+'é中😀\ud800!\udfff';
            var sources=[base+'z',base+String.fromCodePoint(0x10ffff)+'z'],s;
            function reference(a,b){
                var out='';
                for(var k=a;k<b;k++)out+=String.fromCharCode(s.charCodeAt(k));
                return out;
            }
            for(var source of sources){s=source;
            for(var a=60;a<=s.length;a++)for(var b=a;b<=s.length;b++){
                var expected=reference(a,b);
                if(s.slice(a,b)!==expected||s.substring(a,b)!==expected||
                   s.substring(b,a)!==expected||s.substr(a,b-a)!==expected)return a+':'+b;
            }}
            var whole=s.slice(0), alias=s; s+='!';
            return [whole===alias,whole.length,s.length,
                whole.slice(-1),whole.substring(Infinity),whole.substr(-0.5,1)].join('|');
        }run()
        "#,
        "true|74|75|z||x",
    );
}

#[test]
fn substring_coercions_keep_original_receiver_and_abrupt_order() {
    check(
        r#"
        function run(){
            var source='x'.repeat(64)+'é😀z',log=[];
            var receiver={toString(){log.push('receiver');return source}};
            var start={valueOf(){log.push('start');
                for(var k=0;k<80;k++)('different'+k+'é'.repeat(64)).substring(1,3);
                source+='changed';return 65}};
            var end={valueOf(){log.push('end');return 67}};
            var a=String.prototype.substring.call(receiver,start,end);
            if(a!=='😀')throw 99;
            var b=String.prototype.slice.call({toString(){log.push('r2');return source}},
                {valueOf(){log.push('throw');throw 7}}, {valueOf(){log.push('unreached');return 2}});
            return b;
        }
        var caught;
        try{run()}catch(e){caught=e}
        // A separate trace keeps the thrown fixture's log observable.
        var log=[],s='x'.repeat(64)+'é😀z';
        var result=String.prototype.substring.call({toString(){log.push('receiver');return s}},
            {valueOf(){log.push('start');s+='changed';return 65}},
            {valueOf(){log.push('end');return 67}});
        var bad={valueOf(){log.push('bad');throw 9}};
        try{s.slice(bad,{valueOf(){log.push('unreached');return 2}})}catch(e){log.push(e)}
        [caught,result,log.join(','),s.length].join('|')
        "#,
        "7|😀|receiver,start,end,bad,9|75",
    );
}

#[test]
fn sparse_byte_index_matches_utf16_reconstruction_and_declines_splits() {
    for tail in ["é中😀z", "\u{10f800}x\u{10ffff}", "😀😀é", "éa中b"] {
        let source: crate::lstr::LStr = format!("{}{}", "x".repeat(64), tail).into();
        let units = crate::jstr::units(&source);
        let index = crate::jstr::Utf8Index::new(&source, usize::MAX);
        let mut copied = 0;
        for start in 60..=units.len() {
            for end in start..=units.len() {
                if let Some(bytes) = index.range(&source, start, end) {
                    assert_eq!(bytes, crate::jstr::from_units(&units[start..end]));
                    copied += 1;
                }
            }
        }
        assert!(copied > 0);
    }
    let source: crate::lstr::LStr = format!("{}😀z", "x".repeat(64)).into();
    let index = crate::jstr::Utf8Index::new(&source, usize::MAX);
    assert!(index.range(&source, 64, 65).is_none());
    assert!(index.range(&source, 65, 66).is_none());
    assert_eq!(index.range(&source, 64, 66), Some("😀"));
    let noncanonical: crate::lstr::LStr = format!(
        "{}{}{}",
        "x".repeat(64),
        crate::jstr::smuggle(0xd83d),
        crate::jstr::smuggle(0xde00)
    )
    .into();
    let index = crate::jstr::Utf8Index::new(&noncanonical, usize::MAX);
    assert!(index.range(&noncanonical, 64, 66).is_none());
    let dense: crate::lstr::LStr = "é".repeat(128).into();
    let declined = crate::jstr::Utf8Index::new(&dense, 24);
    assert!(declined.range(&dense, 0, 10).is_none());
    assert_eq!(declined.heap_bytes(), std::mem::size_of_val(&declined));
}

#[test]
fn unicode_slice_index_stays_in_existing_cache_budget_and_pins_identity() {
    let mut engine = Engine::new();
    let original: crate::lstr::LStr = format!("{}é😀z", "x".repeat(128)).into();
    let units = engine.interp.units_full(&original);
    let before = engine.interp.str_units.stats();
    let whole = engine.interp.slice_units(&original, &units, 0, units.len());
    assert!(crate::lstr::LStr::ptr_eq(&original, &whole));
    let after = engine.interp.str_units.stats();
    assert_eq!(before.0, after.0);
    assert!(
        after.1 > before.1,
        "the sparse index is charged to the existing cache"
    );
    let high = engine.interp.slice_units(&original, &units, 129, 130);
    assert_eq!(crate::jstr::units(&high), [0xd83d]);
    for n in 0..80 {
        let source: crate::lstr::LStr = format!("{n}{}é", "x".repeat(128)).into();
        let view = engine.interp.units_full(&source);
        engine.interp.slice_units(&source, &view, 1, view.len());
    }
    let (entries, bytes) = engine.interp.str_units.stats();
    assert!(entries <= 64 && bytes <= 16 << 20);
    let copied = engine
        .interp
        .slice_units(&original, &units, 128, units.len());
    assert!(matches!(Value::Str(copied), Value::Str(s) if s.as_str() == "é😀z"));
}

#[test]
fn large_views_keep_native_reads_and_observers_correct() {
    check(
        r#"
        function run(){
            var source='abcdef'.repeat(200)+'é😀',alias=source;
            var view=source.substring(60,source.length-1);
            source+='changed';
            var ascii=alias.slice(6,1100),other=ascii.substring(100,1000);
            var equal='abcdef'.repeat(200).slice(6,1100);
            var code=0;
            for(var k=0;k<300;k++)code+=ascii.charCodeAt(k%ascii.length);
            var obj={};obj[other]=7;
            var out=[ascii===equal,ascii.length,ascii.charAt(2),ascii[3],code,
                obj[other],other.startsWith('efab'),/abcdef/.exec(ascii)[0],
                ascii.replace(/f/g,'F').slice(0,12),view.charCodeAt(view.length-1),
                JSON.parse(JSON.stringify([other]))[0]===other,alias.length,source.length];
            return out.join('|');
        }run()
        "#,
        "true|1094|c|d|29850|7|true|abcdef|abcdeFabcdeF|55357|true|1203|1210",
    );
}

#[test]
fn prepared_regex_windows_preserve_slice_relative_anchors_and_indices() {
    check(
        r#"
        function run(){
            var source='prefix'+('abcé😀\ud800z').repeat(150)+'trailer';
            var input=source.substring(6,source.length-7);
            var re=/^(?<word>abcé😀)/dug, match=re.exec(input);
            var tail=/z$/d.exec(input);
            var sticky=/😀/duy;sticky.lastIndex=4;
            var hit=sticky.exec(input);
            var lookbehind=/(?<=prefix)a/du.exec(input);
            var other=input.slice(8),begin=/^abcé😀/du.exec(other);
            var empty=/^$/u.exec(other);
            var replaced=input.replace(/^abcé😀/u,'!');
            var legacy=RegExp['$&'];
            return [match.index,match[0],match.groups.word,match.indices[0].join(','),
                re.lastIndex,tail.index,tail.indices[0].join(','),hit.index,
                sticky.lastIndex,lookbehind===null,begin.index,empty===null,
                replaced.charCodeAt(1),legacy].join('|');
        }run()
        "#,
        "0|abcé😀|abcé😀|0,6|6|1199|1199,1200|4|6|true|0|true|55296|abcé😀",
    );
}

#[test]
fn prepared_windows_pin_shared_tables_and_map_astral_offsets() {
    use crate::regex::ReText;
    use std::rc::Rc;
    for unicode in [false, true] {
        let source: crate::lstr::LStr = format!("é{}😀z", "x".repeat(1024)).into();
        let mut engine = Engine::new();
        let cached_parent = engine.interp.re_text(unicode, &source);
        let parent = Rc::new(ReText::new_rc(unicode, &source));
        let input = source.slice_bytes(102, source.len());
        let cached_window = engine.interp.re_text(unicode, &input);
        assert!(
            Rc::strong_count(&cached_parent) >= 3,
            "window shares the cached table"
        );
        assert_eq!(cached_window.slice(0, cached_window.len()), input.as_str());
        let start = crate::jstr::unit_len(&source[..102]);
        let window = ReText::window(parent.clone(), &input, start, 1028).unwrap();
        assert_eq!(
            window.unit_index(window.len()),
            crate::jstr::unit_len(&input)
        );
        let reference = ReText::new_rc(unicode, &input);
        for u in 0..=window.unit_index(window.len()) + 1 {
            assert_eq!(window.elem_at_unit(u), reference.elem_at_unit(u));
        }
        for e in 0..=window.len() + 1 {
            assert_eq!(window.unit_index(e), reference.unit_index(e));
        }
        assert_eq!(
            window.slice(0, window.len()),
            reference.slice(0, reference.len())
        );
        drop(parent);
        drop(source);
        assert_eq!(window.slice(window.len() - 1, window.len()), "z");
        let window = Rc::new(window);
        assert!(ReText::window(window, &input, 0, 10).is_none());
    }
}

#[test]
fn regexp_no_match_replace_keeps_coercion_and_exec_effects() {
    check(
        r#"
        function run(){
            var original='a'.repeat(96)+'é😀\ud800',source=original,log=[];
            var receiver={toString(){log.push('source');return source}};
            var replacement={toString(){log.push('replace');source+='changed';return 'X'}};
            var re={get flags(){log.push('flags');return {toString(){log.push('flags-string');return 'g'}}},
                set lastIndex(v){log.push('last:'+v)},
                get exec(){log.push('exec');return function(s){log.push('call:'+s.length);return null}}};
            var result=RegExp.prototype[Symbol.replace].call(re,receiver,replacement);
            return [result===original,result.length,result.charCodeAt(99),log.join(','),source.length].join('|');
        }run()
        "#,
        "true|100|55296|source,replace,flags,flags-string,last:0,exec,call:100|107",
    );
}

#[test]
fn regexp_no_match_replace_preserves_abrupt_state_and_static_observers() {
    check(
        r#"
        function run(){
            /é/u.exec('é');
            var input='x'.repeat(300)+'é😀',called=0,re=/not-found/gu;
            re.lastIndex=900;
            var result=input.replace(re,function(){called++;throw 99});
            var staticMatch=RegExp['$&'];
            var readonly=/absent/g;
            Object.defineProperty(readonly,'lastIndex',{writable:false});
            var error;
            try{input.replace(readonly,'!')}catch(e){error=e.name}
            var log=[];
            try{RegExp.prototype[Symbol.replace].call({
                get flags(){log.push('flags');throw 7},
                get exec(){log.push('unreached');return ()=>null}},
                {toString(){log.push('source');return input}},
                {toString(){log.push('replace');return '!'}})}catch(e){log.push(e)}
            var alias=result;result+='!';
            return [alias===input,called,re.lastIndex,staticMatch,error,log.join(','),
                alias.length,result.length].join('|');
        }run()
        "#,
        "true|0|0|é|TypeError|source,replace,flags,7|303|304",
    );
}
