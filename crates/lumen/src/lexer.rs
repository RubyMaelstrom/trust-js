//! Hand-written tokenizer. Produces the full `Vec<Token>` up front (scripts are small enough that
//! streaming buys nothing) and resolves the classic `/`-is-it-a-regex-or-division ambiguity by
//! tracking whether the previously emitted token can end an expression.
//!
//! The scanner reads the UTF-8 source in place. Positions are byte offsets that always fall on a
//! code point boundary; runs of ASCII whitespace, comment text, identifier characters and plain
//! string characters are consumed a byte (or word) at a time and copied as slices.

use crate::token::{Tok, Token, TplPart};
use std::rc::Rc;

pub struct LexError {
    pub message: String,
    pub line: u32,
    /// The lexer ran out of input mid-construct (unterminated template/comment/...). A REPL uses
    /// this to keep reading lines instead of reporting a SyntaxError.
    pub at_eof: bool,
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    /// Byte offset of the next code point.
    pos: usize,
    line: u32,
    out: Vec<Token>,
    nl_pending: bool,
    /// Script goal: Annex B HTML-like comments are recognised. Off for modules.
    html_comments: bool,
    /// Set while reading a string that contained a legacy octal / `\8` / `\9` escape.
    pending_legacy: bool,
    /// Set while reading a string that contained a lone (unpaired) surrogate `\u` escape.
    pending_lone_surrogate: bool,
    /// One entry per open `{`: `true` if it opened a block/function body (statement position),
    /// `false` if an object literal (expression position). Used to disambiguate a `/` after `}`.
    brace_stack: Vec<bool>,
    /// Byte offset where the token currently being scanned began.
    tok_start: usize,
    /// Classification of the most recently closed `}` (`true` = a block). A `/` after a block-closing
    /// `}` begins a regex; after an object-literal-closing `}` it is division.
    last_close_block: bool,
    /// One entry per `function` keyword whose body has not opened yet: `true` if the function is an
    /// *expression* (so its body `}` is followed by division), `false` if a *declaration* (its `}`
    /// ends a statement, so a `/` after it is a regex).
    pending_fn: Vec<bool>,
    /// Set when a `class` keyword is pushed: the next `{` is the class body, classified by
    /// whether the class was a declaration (statement position) or an expression.
    pending_class: Option<bool>,
    /// One entry per open `(`: `true` when it begins the head of a control statement whose
    /// matching `)` is followed by a Statement. ECMA-262 selects the `InputElementRegExp` lexical
    /// goal there, unlike an ordinary grouping/call close where `/` is division.
    control_paren_stack: Vec<bool>,
    /// Classification of the most recently closed `)`, used until the next significant token.
    last_close_control: bool,
    /// A postfix update ends an expression; a prefix update still expects an operand.
    last_update_postfix: bool,
}

/// Tokenize `src`. A lex error is reported as a SyntaxError by the caller.
pub fn tokenize(src: &str) -> Result<Vec<Token>, LexError> {
    tokenize_goal(src, true)
}

/// Tokenize with an explicit goal: `html_comments` is true for Scripts (Annex B `<!--`/`-->`
/// comments apply) and false for Modules (where they are ordinary punctuation, i.e. errors).
pub fn tokenize_goal(src: &str, html_comments: bool) -> Result<Vec<Token>, LexError> {
    lex(src, html_comments)
}

/// Token stream plus the source it indexes. Token `start`/`end` are byte offsets into `source`,
/// so the parser slices exact `[[SourceText]]` without a second scan.
pub(crate) struct LexedSource {
    pub(crate) tokens: Vec<Token>,
    pub(crate) source: Rc<str>,
}

pub(crate) fn tokenize_with_source(src: &str) -> Result<LexedSource, LexError> {
    tokenize_goal_with_source(src, true)
}

pub(crate) fn tokenize_goal_with_source(
    src: &str,
    html_comments: bool,
) -> Result<LexedSource, LexError> {
    let tokens = lex(src, html_comments)?;
    Ok(LexedSource {
        tokens,
        source: Rc::from(src),
    })
}

fn lex(src: &str, html_comments: bool) -> Result<Vec<Token>, LexError> {
    // Diagnostic-only stage timing; the disabled path takes no timestamp and allocates nothing.
    let perf_started = crate::jit::perf_stage_start();
    let mut lx = Lexer {
        src,
        bytes: src.as_bytes(),
        pos: 0,
        line: 1,
        out: Vec::new(),
        nl_pending: false,
        html_comments,
        pending_legacy: false,
        pending_lone_surrogate: false,
        brace_stack: Vec::new(),
        tok_start: 0,
        last_close_block: false,
        pending_fn: Vec::new(),
        pending_class: None,
        control_paren_stack: Vec::new(),
        last_close_control: false,
        last_update_postfix: false,
    };
    let result = lx.run();
    crate::jit::perf_lex_end(perf_started, result.is_ok());
    result.map(|()| lx.out)
}

impl Lexer<'_> {
    /// The code point starting at byte offset `at` (a code point boundary).
    #[inline]
    fn char_at(&self, at: usize) -> Option<char> {
        match self.bytes.get(at) {
            Some(&b) if b < 0x80 => Some(b as char),
            Some(_) => self.src[at..].chars().next(),
            None => None,
        }
    }
    #[inline]
    fn peek(&self) -> Option<char> {
        self.char_at(self.pos)
    }
    #[inline]
    fn peek2(&self) -> Option<char> {
        let c = self.peek()?;
        self.char_at(self.pos + c.len_utf8())
    }
    fn peek_at(&self, ahead: usize) -> Option<char> {
        let mut at = self.pos;
        for _ in 0..ahead {
            at += self.char_at(at)?.len_utf8();
        }
        self.char_at(at)
    }
    #[inline]
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
        }
        Some(c)
    }
    fn err(&self, message: impl Into<String>) -> LexError {
        LexError {
            message: message.into(),
            line: self.line,
            at_eof: self.pos >= self.bytes.len(),
        }
    }

    /// Whether the previously emitted token permits a regex literal to follow (i.e. we are at the
    /// start of an expression). After a value-producing token (`)`, `]`, identifier, number, etc.)
    /// a `/` is division; otherwise it begins a regex.
    fn regex_allowed(&self) -> bool {
        match self.out.last().map(|t| &t.kind) {
            None => true,
            Some(
                Tok::Num(_) | Tok::BigInt(_) | Tok::Str(_) | Tok::Template(_) | Tok::Regex { .. },
            ) => false,
            // `await`/`yield` are contextual: when they are keywords (module top level, async or
            // generator bodies) they prefix an expression, so a following `/` starts a regex. They
            // are `Ident` tokens here since the lexer lacks that context; allow the regex form —
            // division right after a bare `await`/`yield` *identifier* is vanishingly rare.
            Some(Tok::Ident(w)) => matches!(w.as_str(), "await" | "yield"),
            Some(Tok::Keyword(k)) => !matches!(*k, "this" | "super" | "true" | "false" | "null"),
            // A `/` after an ordinary grouping/call `)` or a `]` is division. A control-statement
            // head `)` is followed by a Statement and therefore admits a regex literal. After `}`
            // it depends on whether the brace closed a block or an object literal.
            Some(Tok::Punct(p)) => match *p {
                ")" => self.last_close_control,
                "]" => false,
                "}" => self.last_close_block,
                "++" | "--" => !self.last_update_postfix,
                _ => true,
            },
            Some(Tok::Eof) => false,
        }
    }

    /// Classify an opening `{` (about to be pushed): does the matching `}` end at statement position
    /// (a block, control body, or function *declaration* body → a following `/` is a regex) or at
    /// value position (an object literal or function *expression* body → division)?
    fn brace_ends_statement(&mut self) -> bool {
        let prev_is_paren = matches!(self.out.last().map(|t| &t.kind), Some(Tok::Punct(")")));
        if prev_is_paren {
            // A body after `)` — either a function body (classified by `pending_fn`) or a control
            // block / method (statement position).
            return match self.pending_fn.pop() {
                Some(is_expr) => !is_expr,
                None => true,
            };
        }
        match self.out.last().map(|t| &t.kind) {
            None => true,
            Some(Tok::Punct(p)) => matches!(*p, ";" | "{" | "}" | "=>"),
            // `catch {` is a Catch without a CatchParameter (ECMA-262 TryStatement): its Block
            // ends the TryStatement, so a following `/` begins a regex literal.
            Some(Tok::Keyword(k)) => matches!(*k, "else" | "do" | "try" | "catch" | "finally"),
            _ => false,
        }
    }

    /// Whether a `class` keyword (about to be pushed) sits in statement position (a declaration —
    /// so its body's closing `}` ends a statement and a following `/` starts a regex).
    fn class_is_declaration(&self) -> bool {
        match self.out.last().map(|t| &t.kind) {
            None => true,
            Some(Tok::Punct(p)) => match *p {
                ";" | "{" => true,
                "}" => self.last_close_block,
                _ => false,
            },
            Some(Tok::Keyword(k)) => {
                matches!(*k, "else" | "do" | "try" | "finally" | "export" | "default")
            }
            _ => false,
        }
    }

    /// Whether a `function` keyword (about to be pushed) sits in statement position (a declaration)
    /// rather than expression position.
    fn function_is_declaration(&self) -> bool {
        let mut before = self.out.as_slice();
        if let Some(token) = before.last() {
            if matches!(&token.kind, Tok::Ident(word) if word == "async") {
                // ECMA-262 AsyncFunction/AsyncGenerator syntax includes the unescaped `async`
                // prefix in the declaration or expression. Classify its starting position,
                // not the prefix as an ordinary value; otherwise `/` after the body is mislexed.
                // A line terminator instead separates the ordinary identifier expression from
                // a new FunctionDeclaration (ASI); the parser still validates the full context.
                if self.nl_pending {
                    return true;
                }
                if !token.escaped {
                    before = &before[..before.len() - 1];
                }
            }
        }
        match before.last().map(|t| &t.kind) {
            None => true,
            Some(Tok::Punct(p)) => match *p {
                ";" | "{" => true,
                "}" => self.last_close_block,
                _ => false,
            },
            Some(Tok::Keyword(k)) => {
                matches!(*k, "else" | "do" | "try" | "finally" | "export" | "default")
            }
            _ => false,
        }
    }

    fn push(&mut self, kind: Tok) {
        // ECMA-262 UpdateExpression: only the postfix forms select InputElementDiv,
        // and they cannot have a LineTerminator before the update operator. In particular,
        // `counter++/denominator` is division, but `++/x/.lastIndex` starts a regexp.
        let update_postfix =
            matches!(&kind, Tok::Punct("++" | "--")) && !self.nl_pending && !self.regex_allowed();
        let closes_control = matches!(&kind, Tok::Punct(")"))
            && self.control_paren_stack.last().copied().unwrap_or(false);
        match &kind {
            Tok::Keyword(k) if *k == "function" => {
                let is_expr = !self.function_is_declaration();
                self.pending_fn.push(is_expr);
            }
            Tok::Keyword(k) if *k == "class" => {
                self.pending_class = Some(self.class_is_declaration());
            }
            Tok::Punct(p) if *p == "{" => {
                // A pending `class` head claims this `{` as its body: the closing `}` sits at
                // statement position exactly when the class is a declaration.
                let ends_stmt = match self.pending_class.take() {
                    Some(is_decl) => is_decl,
                    None => self.brace_ends_statement(),
                };
                self.brace_stack.push(ends_stmt);
            }
            Tok::Punct(p) if *p == "}" => {
                self.last_close_block = self.brace_stack.pop().unwrap_or(false);
            }
            Tok::Punct(p) if *p == "(" => {
                let before_keyword = self
                    .out
                    .get(self.out.len().saturating_sub(2))
                    .map(|token| &token.kind);
                // Reserved words are valid property names (`promise.catch(...)`). Such a call
                // closes in value position and must not be mistaken for a control head.
                let keyword_is_member = matches!(before_keyword, Some(Tok::Punct("." | "?.")));
                let starts_control = match self.out.last().map(|token| &token.kind) {
                    Some(Tok::Keyword(keyword)) => {
                        (!keyword_is_member
                            && matches!(
                                *keyword,
                                "if" | "while" | "for" | "with" | "switch" | "catch"
                            ))
                            || (*keyword == "await"
                                && matches!(before_keyword, Some(Tok::Keyword("for"))))
                    }
                    _ => false,
                };
                self.control_paren_stack.push(starts_control);
            }
            Tok::Punct(p) if *p == ")" => {
                let _ = self.control_paren_stack.pop();
            }
            _ => {}
        }
        self.last_close_control = closes_control;
        self.last_update_postfix = update_postfix;
        let nl = self.nl_pending;
        self.nl_pending = false;
        self.out.push(Token {
            kind,
            line: self.line,
            start: self.tok_start as u32,
            end: self.pos as u32,
            nl_before: nl,
            legacy_octal: false,
            escaped: false,
            lone_surrogate: false,
        });
    }
    /// Flag the most recently pushed token as having contained a `\u` escape.
    fn mark_escaped(&mut self) {
        if let Some(t) = self.out.last_mut() {
            t.escaped = true;
        }
    }
    /// Flag the most recently pushed token as containing a lone surrogate.
    fn mark_lone_surrogate(&mut self) {
        if let Some(t) = self.out.last_mut() {
            t.lone_surrogate = true;
        }
    }
    /// Flag the most recently pushed token as a legacy-octal construct.
    fn mark_legacy_octal(&mut self) {
        if let Some(t) = self.out.last_mut() {
            t.legacy_octal = true;
        }
    }

    fn run(&mut self) -> Result<(), LexError> {
        // Hashbang comment: `#!...` only at the very start of the source.
        if self.bytes.starts_with(b"#!") {
            self.skip_line_comment();
        }
        while let Some(&b) = self.bytes.get(self.pos) {
            self.tok_start = self.pos;
            match b {
                b' ' | b'\t' | 0x0B | 0x0C => {
                    self.skip_ascii_whitespace();
                    continue;
                }
                b'\n' => {
                    self.nl_pending = true;
                    self.line += 1;
                    self.pos += 1;
                    continue;
                }
                b'\r' => {
                    self.nl_pending = true;
                    self.pos += 1;
                    continue;
                }
                _ => {}
            }
            let c = self.peek().expect("in bounds");
            if is_line_terminator(c) {
                self.nl_pending = true;
                self.bump();
            } else if crate::regex::js_whitespace(c) {
                // WhiteSpace (ECMA-262 #sec-white-space): ZWNBSP (U+FEFF) anywhere in the source,
                // not just as a BOM, and every Zs code point, but not U+0085 (NEL).
                self.bump();
            } else if c == '/' && self.peek2() == Some('/') {
                self.skip_line_comment();
            } else if c == '/' && self.peek2() == Some('*') {
                self.skip_block_comment()?;
            } else if c == '<' && self.html_comments && self.bytes[self.pos..].starts_with(b"<!--")
            {
                // Annex B HTML-like comment: `<!--` opens a single-line comment.
                self.skip_line_comment();
            } else if c == '-'
                && self.html_comments
                && self.bytes[self.pos..].starts_with(b"-->")
                && (self.nl_pending || self.out.is_empty())
            {
                // Annex B: `-->` at the start of a line (or of the source) is a comment to EOL.
                self.skip_line_comment();
            } else if c == '/' && self.regex_allowed() {
                self.read_regex()?;
            } else if c == '"' || c == '\'' {
                self.read_string(c)?;
            } else if c == '`' {
                self.read_template()?;
            } else if c.is_ascii_digit()
                || (c == '.' && self.peek2().is_some_and(|d| d.is_ascii_digit()))
            {
                self.read_number()?;
            } else if is_ident_start(c) || c == '#' || (c == '\\' && self.peek2() == Some('u')) {
                self.read_ident()?;
            } else {
                self.read_punct()?;
            }
        }
        self.tok_start = self.pos;
        self.push(Tok::Eof);
        Ok(())
    }

    /// Consume a run of ASCII WhiteSpace (SP, TAB, VT, FF). Long runs of spaces, as in indented
    /// or padded source, are compared eight bytes at a time.
    fn skip_ascii_whitespace(&mut self) {
        const SPACES: u64 = u64::from_ne_bytes([b' '; 8]);
        let bytes = self.bytes;
        let mut i = self.pos;
        while let Some(word) = bytes.get(i..i + 8) {
            if u64::from_ne_bytes(word.try_into().expect("eight bytes")) != SPACES {
                break;
            }
            i += 8;
        }
        while matches!(bytes.get(i), Some(b' ' | b'\t' | 0x0B | 0x0C)) {
            i += 1;
        }
        self.pos = i;
    }

    /// Whether the code point at byte offset `at` (whose first byte is `0xE2`) is U+2028 or
    /// U+2029, the only non-ASCII LineTerminators.
    #[inline]
    fn is_unicode_line_terminator_at(&self, at: usize) -> bool {
        matches!(self.bytes.get(at..at + 3), Some([0xE2, 0x80, 0xA8 | 0xA9]))
    }

    /// Skip to (not past) the next LineTerminator. A comment holds no `\n`, so the line count is
    /// unchanged; the scan stops only on ASCII or `0xE2` lead bytes, i.e. on code point boundaries.
    fn skip_line_comment(&mut self) {
        let bytes = self.bytes;
        let mut i = self.pos;
        while let Some(&b) = bytes.get(i) {
            if b == b'\n' || b == b'\r' || (b == 0xE2 && self.is_unicode_line_terminator_at(i)) {
                break;
            }
            i += 1;
        }
        self.pos = i;
    }

    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        let bytes = self.bytes;
        let mut i = self.pos + 2;
        loop {
            match bytes.get(i) {
                None => {
                    self.pos = i;
                    return Err(self.err("unterminated block comment"));
                }
                Some(b'*') if bytes.get(i + 1) == Some(&b'/') => {
                    self.pos = i + 2;
                    return Ok(());
                }
                Some(b'\n') => {
                    self.line += 1;
                    self.nl_pending = true;
                }
                Some(b'\r') => self.nl_pending = true,
                Some(0xE2) if self.is_unicode_line_terminator_at(i) => self.nl_pending = true,
                _ => {}
            }
            i += 1;
        }
    }

    /// Push an identifier or reserved word spelled `word`.
    fn push_word(&mut self, word: &str, had_escape: bool) {
        // A reserved word is always a keyword — even spelled with a `\u` escape. An escaped reserved
        // word can't be an Identifier (the parser rejects a keyword there), but it still works as a
        // property name (keywords are accepted in those positions).
        match crate::token::keyword(word) {
            Some(kw) => self.push(Tok::Keyword(kw)),
            None => self.push(Tok::Ident(word.to_owned())),
        }
        if had_escape {
            self.mark_escaped();
        }
    }

    fn read_ident(&mut self) -> Result<(), LexError> {
        // Fast path: an all-ASCII name without escapes is a slice of the source.
        let bytes = self.bytes;
        let start = self.pos;
        let mut i = start + usize::from(bytes[start] == b'#');
        if bytes.get(i).is_some_and(|&b| is_ascii_ident_start(b)) {
            i += 1;
            while bytes.get(i).is_some_and(|&b| is_ascii_ident_part(b)) {
                i += 1;
            }
            if bytes.get(i).is_none_or(|&b| b != b'\\' && b < 0x80) {
                self.pos = i;
                let src = self.src;
                self.push_word(&src[start..i], false);
                return Ok(());
            }
        }
        let mut s = String::new();
        // A leading `#` (private name) is part of the identifier but not an ident-continue char.
        if self.peek() == Some('#') {
            s.push('#');
            self.bump();
        }
        // `\uXXXX` / `\u{...}` escapes may appear in an identifier; track that so an escaped reserved
        // word stays an Identifier (a keyword written with an escape is not the keyword).
        let mut had_escape = false;
        // The first code point must be IdentifierStart; the rest IdentifierPart. This holds for an
        // escaped code point too — so `#x` (escaped `#`) and a leading combining mark are errors.
        let mut first = true;
        loop {
            match self.peek() {
                Some('\\') if self.peek2() == Some('u') => {
                    self.bump();
                    self.bump();
                    match self.read_unicode_escape_char() {
                        Some(ch) => {
                            let ok = if first {
                                is_ident_start(ch)
                            } else {
                                is_ident_part(ch)
                            };
                            if !ok {
                                return Err(self.err("invalid character in escaped identifier"));
                            }
                            had_escape = true;
                            s.push(ch);
                            first = false;
                        }
                        None => return Err(self.err("invalid unicode escape in identifier")),
                    }
                }
                Some(c) if (first && is_ident_start(c)) || (!first && is_ident_part(c)) => {
                    s.push(c);
                    self.bump();
                    first = false;
                }
                _ => break,
            }
        }
        self.push_word(&s, had_escape);
        Ok(())
    }

    /// Read the body of a `\u` identifier/string escape (already consumed `\u`): either `{HEX+}` or
    /// exactly four hex digits, yielding the code point as a `char`.
    fn read_unicode_escape_char(&mut self) -> Option<char> {
        let mut hex = String::new();
        if self.peek() == Some('{') {
            self.bump();
            while let Some(c) = self.peek() {
                if c == '}' {
                    self.bump();
                    break;
                } else if c.is_ascii_hexdigit() {
                    hex.push(c);
                    self.bump();
                } else {
                    return None;
                }
            }
        } else {
            for _ in 0..4 {
                match self.peek() {
                    Some(c) if c.is_ascii_hexdigit() => {
                        hex.push(c);
                        self.bump();
                    }
                    _ => return None,
                }
            }
        }
        u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
    }

    fn read_string(&mut self, quote: char) -> Result<(), LexError> {
        self.bump();
        let mut s = String::new();
        let mut had_escape = false;
        self.pending_legacy = false;
        self.pending_lone_surrogate = false;
        let quote_byte = quote as u8;
        loop {
            // Copy the run of characters that need no interpretation. U+2028/U+2029 may appear
            // literally in a string (json-superset); only CR/LF end it. The run stops only on
            // ASCII bytes, so both ends are code point boundaries.
            let run_start = self.pos;
            let mut i = run_start;
            while let Some(&b) = self.bytes.get(i) {
                if b == quote_byte || b == b'\\' || b == b'\n' || b == b'\r' {
                    break;
                }
                i += 1;
            }
            s.push_str(&self.src[run_start..i]);
            self.pos = i;
            match self.bump() {
                None => return Err(self.err("unterminated string literal")),
                Some(c) if c == quote => break,
                Some('\\') => {
                    had_escape = true;
                    self.read_escape(&mut s)?
                }
                Some(_) => return Err(self.err("unterminated string literal")),
            }
        }
        self.push(Tok::Str(s));
        // A directive prologue string that used escapes never matches "use strict".
        if had_escape {
            self.mark_escaped();
        }
        if self.pending_legacy {
            self.mark_legacy_octal();
            self.pending_legacy = false;
        }
        if self.pending_lone_surrogate {
            self.mark_lone_surrogate();
            self.pending_lone_surrogate = false;
        }
        Ok(())
    }

    fn read_template(&mut self) -> Result<(), LexError> {
        self.bump(); // opening backtick
        let mut parts: Vec<TplPart> = Vec::new();
        let mut cooked = String::new();
        // An invalid escape poisons the chunk's cooked value (None) instead of erroring: the
        // parser rejects it later unless the template is tagged.
        let mut invalid = false;
        let mut raw_start = self.pos; // raw source of the current chunk starts here
        loop {
            // The template's raw value normalizes line terminators: <CR><LF> and <CR> → <LF>.
            let raw_of = |text: &str| -> String {
                if !text.contains('\r') {
                    return text.to_owned();
                }
                let mut out = String::with_capacity(text.len());
                let mut chars = text.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '\r' {
                        out.push('\n');
                        chars.next_if_eq(&'\n');
                    } else {
                        out.push(c);
                    }
                }
                out
            };
            match self.peek() {
                None => return Err(self.err("unterminated template literal")),
                Some('`') => {
                    let raw = raw_of(&self.src[raw_start..self.pos]);
                    self.bump();
                    parts.push(TplPart::Str {
                        cooked: (!std::mem::take(&mut invalid))
                            .then(|| std::mem::take(&mut cooked)),
                        raw,
                    });
                    break;
                }
                Some('$') if self.bytes.get(self.pos + 1) == Some(&b'{') => {
                    let raw = raw_of(&self.src[raw_start..self.pos]);
                    parts.push(TplPart::Str {
                        cooked: (!std::mem::take(&mut invalid))
                            .then(|| std::mem::take(&mut cooked)),
                        raw,
                    });
                    cooked.clear();
                    self.bump(); // '$'
                    self.bump(); // '{'
                    let start = self.pos;
                    let sub = self.read_template_sub()?;
                    // The hole is copied verbatim, so offsets within it map back one-to-one.
                    debug_assert_eq!(self.src.get(start..start + sub.len()), Some(sub.as_str()));
                    parts.push(TplPart::Sub(sub, start as u32));
                    raw_start = self.pos;
                }
                Some('\\') => {
                    // Octal / `\8` / `\9` / malformed hex-unicode escapes poison the chunk's
                    // cooked value; a tagged template accepts them (cooked = undefined).
                    if !self.template_escape_ok() {
                        invalid = true;
                        self.bump(); // the backslash
                        self.bump(); // the character after it
                        continue;
                    }
                    self.pending_legacy = false;
                    self.bump(); // consume the backslash
                    self.read_escape(&mut cooked)?;
                    if self.pending_legacy {
                        invalid = true;
                    }
                }
                Some('\r') => {
                    // The cooked value normalizes <CR><LF> and lone <CR> to <LF> (like raw).
                    self.bump();
                    if self.peek() == Some('\n') {
                        self.bump();
                    }
                    cooked.push('\n');
                }
                Some(c) => {
                    self.bump();
                    cooked.push(c);
                }
            }
        }
        self.push(Tok::Template(parts));
        Ok(())
    }

    /// Read the raw source inside a `${ ... }` hole, returning it verbatim for the parser to
    /// sub-parse. Tracks `{}` nesting and skips over string/template literals so their braces and
    /// backticks don't confuse the matching.
    fn read_template_sub(&mut self) -> Result<String, LexError> {
        let mut src = String::new();
        let mut depth = 0i32;
        // Last non-whitespace char emitted, to disambiguate `/` (regex vs division) the same way the
        // main lexer does — so quotes/braces inside a regex literal don't confuse the brace scan.
        let mut last_sig: Option<char> = None;
        let mut newline = false;
        loop {
            // Comments: copy verbatim (their `'"{}` are inert).
            if self.peek() == Some('/') && self.peek2() == Some('/') {
                while let Some(c) = self.peek() {
                    if is_line_terminator(c) {
                        break;
                    }
                    src.push(c);
                    self.bump();
                }
                continue;
            }
            if self.peek() == Some('/') && self.peek2() == Some('*') {
                src.push_str("/*");
                self.bump();
                self.bump();
                loop {
                    match self.bump() {
                        None => return Err(self.err("unterminated comment in template")),
                        Some('*') if self.peek() == Some('/') => {
                            src.push_str("*/");
                            self.bump();
                            break;
                        }
                        Some(c) => {
                            newline |= is_line_terminator(c);
                            src.push(c);
                        }
                    }
                }
                continue;
            }
            // Regex literal: copy verbatim (its `'"{}` are inert).
            if self.peek() == Some('/') && regex_allowed_after(last_sig) {
                self.copy_regex(&mut src)?;
                last_sig = Some(')'); // a regex is a value: a following `/` is division
                newline = false;
                continue;
            }
            if matches!(self.peek(), Some('+' | '-')) && self.peek() == self.peek2() {
                let op = self.bump().unwrap();
                self.bump();
                src.push(op);
                src.push(op);
                let postfix = !newline && !regex_allowed_after(last_sig);
                last_sig = Some(if postfix { ')' } else { op });
                newline = false;
                continue;
            }
            match self.bump() {
                None => return Err(self.err("unterminated template substitution")),
                Some('}') if depth == 0 => return Ok(src),
                Some('}') => {
                    depth -= 1;
                    src.push('}');
                    last_sig = Some('}');
                    newline = false;
                }
                Some('{') => {
                    depth += 1;
                    src.push('{');
                    last_sig = Some('{');
                    newline = false;
                }
                Some(q @ ('"' | '\'')) => {
                    self.copy_quoted(q, &mut src)?;
                    last_sig = Some(')'); // string is a value
                    newline = false;
                }
                Some('`') => {
                    src.push('`');
                    self.copy_template_tail(&mut src)?;
                    last_sig = Some(')');
                    newline = false;
                }
                Some(c) => {
                    src.push(c);
                    newline |= is_line_terminator(c);
                    if !crate::regex::js_whitespace(c) {
                        last_sig = Some(c);
                        newline = false;
                    }
                }
            }
        }
    }

    /// Copy the remainder of a nested template literal after its opening backtick.
    ///
    /// ECMA-262 §12.9.6 switches from TemplateCharacters to ordinary expression tokens at `${`,
    /// then uses the TemplateSubstitutionTail lexical goal after the matching `}`. Recursing
    /// through [`read_template_sub`](Self::read_template_sub) preserves that goal transition for
    /// arbitrarily nested templates instead of mistaking an inner template's backtick for the end
    /// of its parent.
    fn copy_template_tail(&mut self, out: &mut String) -> Result<(), LexError> {
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated nested template")),
                Some('\\') => {
                    out.push('\\');
                    let c = self
                        .bump()
                        .ok_or_else(|| self.err("unterminated nested template"))?;
                    out.push(c);
                }
                Some('$') if self.peek() == Some('{') => {
                    out.push_str("${");
                    self.bump(); // '{'
                    out.push_str(&self.read_template_sub()?);
                    out.push('}');
                }
                Some('`') => {
                    out.push('`');
                    return Ok(());
                }
                Some(c) => out.push(c),
            }
        }
    }

    /// Copy a string literal (already past the opening `quote` is NOT consumed — we push it) into
    /// `out`, respecting escapes, up to and including the matching quote.
    fn copy_quoted(&mut self, quote: char, out: &mut String) -> Result<(), LexError> {
        out.push(quote);
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string in template")),
                Some('\\') => {
                    out.push('\\');
                    if let Some(c) = self.bump() {
                        out.push(c);
                    }
                }
                Some(c) if c == quote => {
                    out.push(c);
                    return Ok(());
                }
                Some(c) => out.push(c),
            }
        }
    }

    /// Copy a regex literal `/body/flags` verbatim into `out` (the leading `/` is at the cursor).
    fn copy_regex(&mut self, out: &mut String) -> Result<(), LexError> {
        out.push('/');
        self.bump(); // opening /
        let mut in_class = false;
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated regex in template")),
                Some('\\') => {
                    out.push('\\');
                    if let Some(c) = self.bump() {
                        out.push(c);
                    }
                }
                Some('[') => {
                    in_class = true;
                    out.push('[');
                }
                Some(']') => {
                    in_class = false;
                    out.push(']');
                }
                Some('/') if !in_class => {
                    out.push('/');
                    break;
                }
                Some(c) => out.push(c),
            }
        }
        while let Some(c) = self.peek() {
            if is_ident_part(c) {
                out.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Whether the escape starting at the current `\` is valid in a template's cooked string:
    /// no octal / `\8` / `\9`, `\x` needs two hex digits, `\u` four (or a braced code point).
    fn template_escape_ok(&self) -> bool {
        let at = |k: usize| self.peek_at(k);
        let hex = |c: Option<char>| c.is_some_and(|c| c.is_ascii_hexdigit());
        match at(1) {
            Some('0') => !matches!(at(2), Some(c) if c.is_ascii_digit()),
            Some(c @ '1'..='9') => {
                let _ = c;
                false
            }
            Some('x') => hex(at(2)) && hex(at(3)),
            Some('u') => {
                if at(2) == Some('{') {
                    let mut k = 3;
                    let mut digits = 0u32;
                    let mut v: u64 = 0;
                    loop {
                        match at(k) {
                            Some('}') => return digits > 0 && v <= 0x10FFFF,
                            Some(c) if c.is_ascii_hexdigit() => {
                                digits += 1;
                                v = (v.saturating_mul(16)) + c.to_digit(16).unwrap() as u64;
                                if digits > 8 {
                                    return false;
                                }
                            }
                            _ => return false,
                        }
                        k += 1;
                    }
                } else {
                    hex(at(2)) && hex(at(3)) && hex(at(4)) && hex(at(5))
                }
            }
            None => false,
            _ => true,
        }
    }

    fn read_escape(&mut self, out: &mut String) -> Result<(), LexError> {
        match self.bump() {
            None => Err(self.err("unterminated escape")),
            Some('n') => {
                out.push('\n');
                Ok(())
            }
            Some('t') => {
                out.push('\t');
                Ok(())
            }
            Some('r') => {
                out.push('\r');
                Ok(())
            }
            Some('b') => {
                out.push('\u{0008}');
                Ok(())
            }
            Some('f') => {
                out.push('\u{000C}');
                Ok(())
            }
            Some('v') => {
                out.push('\u{000B}');
                Ok(())
            }
            Some('0') if !self.peek().is_some_and(|c| c.is_ascii_digit()) => {
                out.push('\0');
                Ok(())
            }
            // Legacy octal escape: 1-3 octal digits (first three only if value <= 0o377).
            Some(c @ '0'..='7') => {
                let mut val = c.to_digit(8).unwrap();
                let max = if c <= '3' { 2 } else { 1 };
                let mut taken = 0;
                while taken < max && self.peek().is_some_and(|d| ('0'..='7').contains(&d)) {
                    val = val * 8 + self.bump().unwrap().to_digit(8).unwrap();
                    taken += 1;
                }
                out.push(char::from_u32(val).unwrap_or('\u{FFFD}'));
                self.pending_legacy = true;
                Ok(())
            }
            // `\8` / `\9` (NonOctalDecimalEscape): the digit itself, but still legacy.
            Some(c @ ('8' | '9')) => {
                out.push(c);
                self.pending_legacy = true;
                Ok(())
            }
            Some('x') => {
                let hi = self.bump().ok_or_else(|| self.err("bad \\x escape"))?;
                let lo = self.bump().ok_or_else(|| self.err("bad \\x escape"))?;
                let n = u32::from_str_radix(&format!("{hi}{lo}"), 16)
                    .map_err(|_| self.err("bad \\x escape"))?;
                out.push(char::from_u32(n).unwrap_or('\u{FFFD}'));
                Ok(())
            }
            Some('u') => self.read_unicode_escape(out),
            Some('\r') => {
                // A CRLF pair is a single LineTerminatorSequence for a line continuation.
                if self.peek() == Some('\n') {
                    self.bump();
                }
                Ok(())
            }
            Some(c) if is_line_terminator(c) => Ok(()), // line continuation
            Some(c) => {
                out.push(c);
                Ok(())
            }
        }
    }

    fn read_unicode_escape(&mut self, out: &mut String) -> Result<(), LexError> {
        let mut hex = String::new();
        if self.peek() == Some('{') {
            self.bump();
            while let Some(c) = self.peek() {
                if c == '}' {
                    break;
                }
                hex.push(c);
                self.bump();
            }
            if self.bump() != Some('}') {
                return Err(self.err("unterminated \\u{...} escape"));
            }
        } else {
            for _ in 0..4 {
                hex.push(self.bump().ok_or_else(|| self.err("bad \\u escape"))?);
            }
        }
        let n = u32::from_str_radix(&hex, 16).map_err(|_| self.err("bad \\u escape"))?;
        if n > 0x10FFFF {
            return Err(self.err("undefined Unicode code-point"));
        }
        if n >= crate::jstr::SMUGGLE_BASE {
            // A smuggle-range character is canonically its smuggled surrogate pair.
            let v = n - 0x10000;
            out.push(crate::jstr::smuggle(0xD800 + (v >> 10) as u16));
            out.push(crate::jstr::smuggle(0xDC00 + (v & 0x3FF) as u16));
            return Ok(());
        }
        if let Some(c) = char::from_u32(n) {
            out.push(c);
            return Ok(());
        }
        // `n` is a surrogate code point. A high surrogate followed by a `\u` low surrogate combines
        // into a single astral code point; anything else is a lone surrogate (representable as a JS
        // string but flagged so it can be rejected as a ModuleExportName).
        if (0xD800..=0xDBFF).contains(&n) && self.peek() == Some('\\') && self.peek2() == Some('u')
        {
            let save = self.pos;
            self.bump(); // '\'
            self.bump(); // 'u'
            let mut lo = String::new();
            let mut ok = true;
            if self.peek() == Some('{') {
                self.bump();
                while let Some(c) = self.peek() {
                    if c == '}' {
                        break;
                    }
                    lo.push(c);
                    self.bump();
                }
                if self.bump() != Some('}') {
                    ok = false;
                }
            } else {
                for _ in 0..4 {
                    match self.peek() {
                        Some(c) if c.is_ascii_hexdigit() => {
                            lo.push(c);
                            self.bump();
                        }
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            let low = if ok {
                u32::from_str_radix(&lo, 16).ok()
            } else {
                None
            };
            if let Some(low) = low.filter(|l| (0xDC00..=0xDFFF).contains(l)) {
                let cp = 0x10000 + ((n - 0xD800) << 10) + (low - 0xDC00);
                if cp < crate::jstr::SMUGGLE_BASE {
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                } else {
                    // A smuggle-range character is canonically its smuggled pair.
                    out.push(crate::jstr::smuggle(n as u16));
                    out.push(crate::jstr::smuggle(low as u16));
                }
                return Ok(());
            }
            // Not a valid low surrogate: rewind and treat the high surrogate as lone.
            self.pos = save;
        }
        self.pending_lone_surrogate = true;
        out.push(crate::jstr::smuggle(n as u16));
        Ok(())
    }

    /// A numeric separator `_` is only legal immediately between two digits of the given radix.
    fn validate_seps(&self, lo: usize, hi: usize, radix: u32) -> Result<(), LexError> {
        // Numeric literal text is ASCII.
        let s = &self.bytes[lo..hi];
        for (i, &c) in s.iter().enumerate() {
            if c == b'_' {
                let prev = i.checked_sub(1).and_then(|j| s.get(j));
                let next = s.get(i + 1);
                let ok = prev.is_some_and(|&p| (p as char).is_digit(radix))
                    && next.is_some_and(|&n| (n as char).is_digit(radix));
                if !ok {
                    return Err(self.err("invalid use of numeric separator"));
                }
            }
        }
        Ok(())
    }

    fn read_number(&mut self) -> Result<(), LexError> {
        self.read_number_inner()?;
        // The SourceCharacter following a NumericLiteral must not be an IdentifierStart or digit.
        if self
            .peek()
            .is_some_and(|c| is_ident_start(c) || c.is_ascii_digit() || c == '\\')
        {
            return Err(self.err("identifier starts immediately after numeric literal"));
        }
        Ok(())
    }

    fn read_number_inner(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        let mut radix = 10u32;
        if self.peek() == Some('0') {
            match self.peek2() {
                Some('x' | 'X') => radix = 16,
                Some('o' | 'O') => radix = 8,
                Some('b' | 'B') => radix = 2,
                _ => {}
            }
        }
        if radix != 10 {
            self.bump();
            self.bump();
            let digits_start = self.pos;
            while let Some(c) = self.peek() {
                if c == '_' || c.is_digit(radix) {
                    self.bump();
                } else {
                    break;
                }
            }
            self.validate_seps(digits_start, self.pos, radix)?;
            let digits: String = self.src[digits_start..self.pos]
                .chars()
                .filter(|c| *c != '_')
                .collect();
            if self.peek() == Some('n') {
                self.bump();
                let n = crate::bigint::JsBigInt::parse_radix(&digits, radix)
                    .ok_or_else(|| self.err("invalid BigInt literal"))?;
                self.push(Tok::BigInt(n));
                return Ok(());
            }
            // ECMA-262 §12.9.3.2/3 defines a non-decimal literal's MV as an
            // unbounded mathematical integer and only then applies the Number
            // conversion. Do not impose the host machine word width here.
            let n = crate::bigint::JsBigInt::parse_radix(&digits, radix)
                .ok_or_else(|| self.err("invalid numeric literal"))?;
            self.push(Tok::Num(n.to_f64()));
            return Ok(());
        }
        // Decimal: integer . fraction e exponent
        while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '_') {
            self.bump();
        }
        // Legacy octal (`010`) / non-octal-decimal (`08`): a leading-zero integer with no fraction,
        // exponent, or `n` suffix. Octal value in sloppy mode; the parser rejects it in strict.
        if self.bytes[start] == b'0'
            && self.pos - start > 1
            && !matches!(self.peek(), Some('.' | 'e' | 'E' | 'n' | '_'))
        {
            // A leading-zero integer (legacy octal / non-octal decimal) admits no separators.
            if self.bytes[start..self.pos].contains(&b'_') {
                return Err(self.err("numeric separator not allowed in legacy literal"));
            }
            let text = &self.src[start..self.pos];
            if text.chars().all(|c| ('0'..='7').contains(&c)) {
                let n = i64::from_str_radix(text, 8).unwrap_or(0);
                self.push(Tok::Num(n as f64));
                self.mark_legacy_octal();
                return Ok(());
            } else if text.chars().all(|c| c.is_ascii_digit()) {
                let n: f64 = text.parse().unwrap_or(0.0);
                self.push(Tok::Num(n));
                self.mark_legacy_octal();
                return Ok(());
            }
        }
        // A BigInt literal is an integer immediately followed by `n` (no fraction/exponent).
        if self.peek() == Some('n') {
            // A leading-zero integer (legacy octal / non-octal decimal) admits no BigInt suffix.
            if self.bytes[start] == b'0' && self.pos - start > 1 {
                return Err(self.err("invalid BigInt literal (leading zero)"));
            }
            self.validate_seps(start, self.pos, 10)?;
            let text: String = self.src[start..self.pos]
                .chars()
                .filter(|c| *c != '_')
                .collect();
            self.bump(); // n
            let n = crate::bigint::JsBigInt::parse_radix(&text, 10)
                .ok_or_else(|| self.err("invalid BigInt literal"))?;
            self.push(Tok::BigInt(n));
            return Ok(());
        }
        if self.peek() == Some('.') {
            self.bump();
            while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '_') {
                self.bump();
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '_') {
                self.bump();
            }
        }
        self.validate_seps(start, self.pos, 10)?;
        let digits = &self.src[start..self.pos];
        let text: std::borrow::Cow<'_, str> = if digits.contains('_') {
            digits
                .chars()
                .filter(|c| *c != '_')
                .collect::<String>()
                .into()
        } else {
            digits.into()
        };
        let n: f64 = text
            .parse()
            .map_err(|_| self.err("invalid numeric literal"))?;
        self.push(Tok::Num(n));
        Ok(())
    }

    fn read_regex(&mut self) -> Result<(), LexError> {
        self.bump(); // opening /
        let mut body = String::new();
        let mut in_class = false;
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated regular expression")),
                Some(c) if is_line_terminator(c) => {
                    return Err(self.err("unterminated regular expression"));
                }
                Some('\\') => {
                    body.push('\\');
                    // A backslash sequence can't contain a line terminator either.
                    match self.bump() {
                        Some(c) if is_line_terminator(c) => {
                            return Err(self.err("unterminated regular expression"));
                        }
                        Some(c) => body.push(c),
                        None => return Err(self.err("unterminated regular expression")),
                    }
                }
                Some('[') => {
                    in_class = true;
                    body.push('[');
                }
                Some(']') => {
                    in_class = false;
                    body.push(']');
                }
                Some('/') if !in_class => break,
                Some(c) => body.push(c),
            }
        }
        let mut flags = String::new();
        while let Some(c) = self.peek() {
            if is_ident_part(c) {
                flags.push(c);
                self.bump();
            } else {
                break;
            }
        }
        self.push(Tok::Regex { body, flags });
        Ok(())
    }

    fn read_punct(&mut self) -> Result<(), LexError> {
        // Punctuators are ASCII and at most four long: decide on a stack copy of the ASCII
        // lookahead.
        let mut buffer = [0u8; 4];
        for (slot, &b) in buffer.iter_mut().zip(&self.bytes[self.pos..]) {
            if !b.is_ascii() {
                break;
            }
            *slot = b;
        }
        // `?.` followed by a digit is `?` then `.` (a conditional like `x ? .5 : .3`), not optional
        // chaining.
        let p = if buffer.starts_with(b"?.") && buffer[2].is_ascii_digit() {
            Some("?")
        } else {
            match_punct(buffer)
        };
        if let Some(p) = p {
            self.pos += p.len();
            self.push(Tok::Punct(p));
            return Ok(());
        }
        Err(self.err(format!(
            "unexpected character {:?}",
            self.peek().unwrap_or('\0')
        )))
    }
}

/// The longest punctuator (an entry of [`crate::token::PUNCTUATORS`]) that `rest` starts with.
/// `rest` holds the ASCII lookahead, zero-padded (NUL begins no punctuator).
fn match_punct(rest: [u8; 4]) -> Option<&'static str> {
    let [b0, b1, b2, b3] = rest;
    Some(match b0 {
        b'{' => "{",
        b'}' => "}",
        b'(' => "(",
        b')' => ")",
        b'[' => "[",
        b']' => "]",
        b';' => ";",
        b',' => ",",
        b'~' => "~",
        b':' => ":",
        b'@' => "@",
        b'.' if b1 == b'.' && b2 == b'.' => "...",
        b'.' => ".",
        b'<' => match (b1, b2) {
            (b'<', b'=') => "<<=",
            (b'<', _) => "<<",
            (b'=', _) => "<=",
            _ => "<",
        },
        b'>' => match (b1, b2, b3) {
            (b'>', b'>', b'=') => ">>>=",
            (b'>', b'>', _) => ">>>",
            (b'>', b'=', _) => ">>=",
            (b'>', _, _) => ">>",
            (b'=', _, _) => ">=",
            _ => ">",
        },
        b'=' => match (b1, b2) {
            (b'=', b'=') => "===",
            (b'=', _) => "==",
            (b'>', _) => "=>",
            _ => "=",
        },
        b'!' => match (b1, b2) {
            (b'=', b'=') => "!==",
            (b'=', _) => "!=",
            _ => "!",
        },
        b'*' => match (b1, b2) {
            (b'*', b'=') => "**=",
            (b'*', _) => "**",
            (b'=', _) => "*=",
            _ => "*",
        },
        b'&' => match (b1, b2) {
            (b'&', b'=') => "&&=",
            (b'&', _) => "&&",
            (b'=', _) => "&=",
            _ => "&",
        },
        b'|' => match (b1, b2) {
            (b'|', b'=') => "||=",
            (b'|', _) => "||",
            (b'=', _) => "|=",
            _ => "|",
        },
        b'?' => match (b1, b2) {
            (b'?', b'=') => "??=",
            (b'?', _) => "??",
            (b'.', _) => "?.",
            _ => "?",
        },
        b'+' => match b1 {
            b'+' => "++",
            b'=' => "+=",
            _ => "+",
        },
        b'-' => match b1 {
            b'-' => "--",
            b'=' => "-=",
            _ => "-",
        },
        b'/' if b1 == b'=' => "/=",
        b'/' => "/",
        b'%' if b1 == b'=' => "%=",
        b'%' => "%",
        b'^' if b1 == b'=' => "^=",
        b'^' => "^",
        _ => return None,
    })
}

/// Whether a `/` following `last` (the previous significant char) starts a regex rather than being
/// a division operator. A value-terminator (identifier char, `)`, `]`, `}`) means division.
fn regex_allowed_after(last: Option<char>) -> bool {
    match last {
        None => true,
        Some(c) => !(c.is_alphanumeric() || matches!(c, '_' | '$' | ')' | ']' | '}')),
    }
}

fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}
/// Whether code point `c` is in the Unicode property `name` (an ASCII-free path; the bundled UCD
/// tables give the exact ID_Start/ID_Continue sets).
fn prop_has(name: &str, c: char) -> bool {
    let u = c as u32;
    crate::unicode_props::lookup(name, None)
        .map(|r| {
            r.binary_search_by(|&(lo, hi)| {
                if u < lo {
                    std::cmp::Ordering::Greater
                } else if u > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
        })
        .unwrap_or(false)
}
fn is_ascii_ident_start(b: u8) -> bool {
    b == b'_' || b == b'$' || b.is_ascii_alphabetic()
}
fn is_ascii_ident_part(b: u8) -> bool {
    b == b'_' || b == b'$' || b.is_ascii_alphanumeric()
}
fn is_ident_start(c: char) -> bool {
    // IdentifierStart = ID_Start ∪ {$, _} (plus `\u` escapes, handled by the caller).
    if c.is_ascii() {
        return c == '_' || c == '$' || c.is_ascii_alphabetic();
    }
    prop_has("ID_Start", c)
}
fn is_ident_part(c: char) -> bool {
    // IdentifierPart = ID_Continue ∪ {$, _, ZWNJ, ZWJ}.
    if c.is_ascii() {
        return c == '_' || c == '$' || c.is_ascii_alphanumeric();
    }
    c == '\u{200C}' || c == '\u{200D}' || prop_has("ID_Continue", c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table-driven maximal munch the punctuator matcher replaces.
    fn munch(rest: &[u8]) -> Option<&'static str> {
        crate::token::PUNCTUATORS
            .iter()
            .copied()
            .find(|p| rest.starts_with(p.as_bytes()))
    }

    #[test]
    fn punctuator_matcher_is_maximal_munch_over_the_punctuator_table() {
        let alphabet: Vec<u8> = b"{}()[].;,<>+-*/%&|^!~?:=@a0 ".to_vec();
        for &a in &alphabet {
            for &b in &alphabet {
                for &c in &alphabet {
                    for &d in &alphabet {
                        let rest = [a, b, c, d];
                        assert_eq!(match_punct(rest), munch(&rest), "{:?}", rest);
                    }
                }
            }
        }
        assert_eq!(match_punct([b'>', b'>', b'>', 0]), Some(">>>"));
        assert_eq!(match_punct([b'.', b'.', 0, 0]), Some("."));
    }

    #[test]
    fn token_offsets_are_utf8_byte_offsets() {
        let src = "é /*π*/ 𝒳ab \"ü\" `t` x";
        let tokens = tokenize(src).unwrap_or_else(|e| panic!("{}", e.message));
        let spans: Vec<&str> = tokens
            .iter()
            .filter(|t| !matches!(t.kind, Tok::Eof))
            .map(|t| &src[t.start as usize..t.end as usize])
            .collect();
        assert_eq!(spans, ["é", "𝒳ab", "\"ü\"", "`t`", "x"]);
    }

    #[test]
    fn whitespace_runs_and_comments_track_lines_and_line_terminators() {
        let src = format!("a{}\n\t b /* x\n y */ c // z\u{2028}d\r\ne", " ".repeat(37));
        let tokens = tokenize(&src).unwrap_or_else(|e| panic!("{}", e.message));
        let lines: Vec<(u32, bool)> = tokens.iter().map(|t| (t.line, t.nl_before)).collect();
        assert_eq!(
            lines,
            [
                (1, false),
                (2, true),
                (3, true),
                (3, true),
                (4, true),
                (4, false)
            ]
        );
        let error = tokenize("a /* never closed\n").expect_err("unterminated comment");
        assert!(error.at_eof);
        assert_eq!(error.line, 2);
    }
}
