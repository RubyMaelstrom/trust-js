// The native RegExp tier's instruction corpus, maintained by hand: its first version was generated
// once, for the one-byte tier the native tier replaced, by a generator that was never committed.
// Each case names the instructions of its program between `Save(0)` and `Save(1), Match`, and
// runs through the native code of both widths against the matcher, from every start of every
// subject.

/// The corpus's name for an instruction.
fn instruction_kind(inst: &super::super::Inst) -> String {
    use super::super::{Inst, Rep};
    match inst {
        Inst::Char(_) => "char".into(),
        Inst::Any => "any".into(),
        Inst::Class(_) => "class".into(),
        Inst::Save(_) => "save".into(),
        Inst::Split(..) => "split".into(),
        Inst::Jmp(_) => "jmp".into(),
        Inst::AssertStart => "start".into(),
        Inst::AssertEnd => "end".into(),
        Inst::WordBoundary(true) => "boundary".into(),
        Inst::WordBoundary(false) => "not_boundary".into(),
        Inst::SetMark(_) => "set_mark".into(),
        Inst::CheckProgress(_) => "check_progress".into(),
        Inst::ClearCaps(..) => "clear_caps".into(),
        Inst::Many {
            rep,
            greedy,
            atomic,
            ..
        } => {
            let element = match rep {
                Rep::Char(_) => "char",
                Rep::Any => "any",
                Rep::Class(_) => "class",
            };
            let kind = match (greedy, atomic) {
                (false, _) => "_lazy",
                (true, true) => "_atomic",
                (true, false) => "",
            };
            format!("many_{element}{kind}")
        }
        _ => "other".into(),
    }
}

#[test]
fn instruction_corpus_matches_as_the_matcher_does() {
    let cases: &[(&str, &str, &[&str])] = &[
        ("a.c", "", &["char", "any", "char"]),
        ("a.c", "s", &["char", "any", "char"]),
        ("é.Ω", "", &["char", "any", "char"]),
        ("[a-z]\\d", "", &["class", "class"]),
        ("[^\\n\\u2028][é-ü]", "", &["class", "class"]),
        ("\\w\\s.", "", &["class", "class", "any"]),
        ("^a.", "", &["start", "char", "any"]),
        ("^a.", "m", &["start", "char", "any"]),
        ("a.$", "", &["char", "any", "end"]),
        ("a.$", "m", &["char", "any", "end"]),
        ("a^b", "m", &["char", "start", "char"]),
        ("x$\\ny", "m", &["char", "end", "char", "char"]),
        ("a.^", "", &["char", "any", "start"]),
        ("\\bx.\\B", "", &["boundary", "char", "any", "not_boundary"]),
        ("(a).", "", &["save", "char", "save", "any"]),
        ("a|b.", "", &["split", "char", "jmp", "char", "any"]),
        ("ab|ac|.d", "", &["split", "char", "char", "jmp", "split", "char", "char", "jmp", "any", "char"]),
        ("a*", "", &["many_char_atomic"]),
        ("a*b", "", &["many_char_atomic", "char"]),
        ("a*a", "", &["many_char", "char"]),
        (".*x", "", &["many_any", "char"]),
        (".*x", "s", &["many_any", "char"]),
        ("[ab]+?c", "", &["many_class_lazy", "char"]),
        ("a{2,3}?a", "", &["many_char_lazy", "char"]),
        ("a{2,3}a", "", &["many_char", "char"]),
        ("\\s+$", "", &["many_class_atomic", "end"]),
        ("\\s*$", "m", &["many_class", "end"]),
        ("\\b\\w+\\b", "", &["boundary", "many_class_atomic", "boundary"]),
        ("(\\d*)(\\D*)", "", &["save", "many_class_atomic", "save", "save", "many_class_atomic", "save"]),
        ("(?:ab)*c", "", &["split", "set_mark", "char", "char", "check_progress", "jmp", "char"]),
        ("(?:ab)*?c", "", &["split", "set_mark", "char", "char", "check_progress", "jmp", "char"]),
        (
            "(a|b)*c",
            "",
            &[
                "split", "set_mark", "clear_caps", "save", "split", "char", "jmp", "char", "save",
                "check_progress", "jmp", "char",
            ],
        ),
        (
            "(a.){2}",
            "",
            &["clear_caps", "save", "char", "any", "save", "clear_caps", "save", "char", "any", "save"],
        ),
        (
            "(?:x|(y))?z",
            "",
            &["split", "set_mark", "clear_caps", "split", "char", "jmp", "save", "char", "save", "check_progress", "char"],
        ),
    ];
    let texts = [
        "", "a", "ab", "abc", "aXc", "xxaXcyy", "a\nc", "a\r\nb", "ab ab", "aab", "aaab", "abababc",
        "x\ny", "xy z", "12ab", "  \n", "\u{e9}x\u{3a9}", "abcabc d", "yz xz z", "a\u{2028}c",
    ];
    for &(pattern, flags, kinds) in cases {
        let re = super::super::Regex::new(pattern, flags).unwrap();
        let actual: Vec<String> = re.prog[1..re.prog.len() - 2]
            .iter()
            .map(instruction_kind)
            .collect();
        assert_eq!(actual, kinds, "/{pattern}/{flags}");
        for text in texts {
            for two in [false, true] {
                if !two && text.chars().any(|c| c as u32 > 0xff) {
                    continue;
                }
                let subject = Subject::of(text, two);
                if let Some((native, _)) = pair(pattern, flags, subject.width()) {
                    assert!(
                        compiled(&native, subject.width()) || !BACKEND,
                        "/{pattern}/{flags} is in the tier's instruction set"
                    );
                }
                check(pattern, flags, &subject);
            }
        }
    }
}
