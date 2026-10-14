//! Literal String replacement and empty-separator split, ECMA-262 snapshot e28783d5.
//! Clauses sec-string.prototype.replace, sec-string.prototype.replaceall and
//! sec-string.prototype.split (local spec.html:36649, 36763, 36846).
//! Expected values were verified with Node.js on the same scripts.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("string fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1+2", false), Ok(Completion::Value(value)) if value == "3"));
    }
}

#[test]
fn literal_callback_offsets_and_empty_matches_follow_code_units() {
    check(
        r#"
function run(){
    function codeUnits(text){var out=[];for(var k=0;k<text.length;k++)out.push(text.charCodeAt(k));return out.join(',')}
    function reference(input,search,all){
        var trace=[],result='',start=0,position=0;
        while(position<=input.length){
            var matches=position+search.length<=input.length;
            for(var k=0;matches&&k<search.length;k++)matches=input.charCodeAt(position+k)===search.charCodeAt(k);
            if(!matches){position++;continue}
            for(var k=start;k<position;k++)result+=String.fromCharCode(input.charCodeAt(k));
            result+='-';trace.push(codeUnits(search)+':'+position+':true:true');
            start=position+search.length;
            if(!all)break;
            position+=Math.max(search.length,1);
        }
        for(var k=start;k<input.length;k++)result+=String.fromCharCode(input.charCodeAt(k));
        return codeUnits(result)+'/'+trace.join(';');
    }
    var checks=0;
    for(var input of ['aba','é😀a\ud800a\udfffa',String.fromCodePoint(0x10ffff)+'a']){
        for(var search of ['a','','\ud83d','\ude00','\ud800','\udfff','missing']){
            for(var all of [false,true]){
                var trace=[];
                var replacer=function(match,offset,whole){'use strict';trace.push(codeUnits(match)+':'+offset+':'+(whole===input)+':'+(this===undefined));return '-'};
                var result=all?input.replaceAll(search,replacer):input.replace(search,replacer);
                var actual=codeUnits(result)+'/'+trace.join(';');
                if(actual!==reference(input,search,all))return checks+':'+actual;
                checks++;
            }
        }
    }
    return checks;
}run()
        "#,
        "42",
    );
}

#[test]
fn literal_callback_keeps_input_live_during_alias_mutation() {
    check(
        r#"
function run(){
    var source='a'.repeat(128)+'é😀a',original=source,calls=0,sum=0,last;
    var result=source.replaceAll('a',function(match,offset,whole){
        if(whole!==original||match!=='a')throw 99;
        if(!calls){source+='changed';for(var n=0;n<80;n++)('other'+n+'é'.repeat(64)).charCodeAt(65)}
        calls++;sum+=offset;last=offset;
        return {toString(){return 'b'}};
    });
    return [result.slice(125),result.length,source.length,calls,sum,last,original.length].join('|');
}run()
        "#,
        "bbbé😀b|132|139|129|8259|131|132",
    );
}

#[test]
fn empty_separator_split_respects_unit_limit_and_coercion_order() {
    check(
        r#"
function run(){
    var checks=0,limits=[0,1,2,3,undefined,-1,4294967297];
    var ascii='abc'.repeat(120),asciiView=(ascii+'!').slice(0,300);
    var unicodeOwner='é'+ascii+'😀',asciiFromUnicode=unicodeOwner.slice(1,301);
    for(var source of ['','abc','é😀\ud800\udfff',String.fromCodePoint(0x10ffff)+'z','😀'.repeat(200),asciiView,asciiFromUnicode]){
        for(var limit of limits){
            var result=source.split('',limit),count=limit===undefined?source.length:Math.min(source.length,limit>>>0);
            if(result.length!==count)throw result.length;
            for(var k=0;k<count;k++)if(result[k].length!==1||result[k].charCodeAt(0)!==source.charCodeAt(k))throw k;
            checks++;
        }
    }
    var log=[],source={toString(){log.push('receiver');return '😀x'}},separator={toString(){log.push('separator');return ''}};
    var limit={valueOf(){log.push('limit');return 1}};
    var first=String.prototype.split.call(source,separator,limit);
    var zero=String.prototype.split.call(source,separator,{valueOf(){log.push('zero');return 0}});
    return [checks,first[0].charCodeAt(0),zero.length,log.join(',')].join('|');
}run()
        "#,
        "49|55357|0|receiver,limit,separator,receiver,zero,separator",
    );
}

#[test]
fn literal_callback_abrupt_completion_stops_later_calls() {
    check(
        r#"
function run(){
    var log=[],receiver={toString(){log.push('receiver');return 'aaa'}},search={toString(){log.push('search');return 'a'}};
    var replacer=function(match,position,whole){'use strict';log.push([match,position,whole,this===undefined].join(':'));if(position===1)throw 'stop';return {toString(){log.push('result');return 'b'}}};
    try{String.prototype.replaceAll.call(receiver,search,replacer)}catch(error){log.push(error)}
    var calls=0;
    'é😀'.replaceAll('absent',function(){calls++;throw 9});
    'é😀'.replace('absent',function(){calls++;throw 9});
    return calls+'|'+log.join(',');
}run()
        "#,
        "0|receiver,search,a:0:aaa:true,result,a:1:aaa:true,stop",
    );
}

#[test]
fn sliced_ascii_search_strings_are_not_declined_by_conservative_hints() {
    check(
        r#"
        function run() {
            var short = 'a'.repeat(300), long = 'a'.repeat(600), out = [];
            var searches = [(short + '!').slice(0, 256), ('é' + short).slice(1, 257)];
            for (var search of searches) {
                var offsets = [];
                function replace(match, offset, whole) {
                    if (match !== search || whole !== long) throw 99;
                    offsets.push(offset);
                    return 'b';
                }
                out.push(short.replace(search, 'b').length, short.replaceAll(search, 'b').length,
                    long.replace(search, 'b').length, long.replaceAll(search, 'b').length,
                    long.replace(search, replace).length, long.replaceAll(search, replace).length,
                    offsets.join(','), 'é'.repeat(300).replace(search, 'b').length);
            }
            out.push('😀'.replace('', '-'), 'é'.replace('', "$'"));
            return out.join('|');
        }
        run()
        "#,
        "45|45|345|90|345|90|0,0,256|300|45|45|345|90|345|90|0,0,256|300|-😀|éé",
    );
}
