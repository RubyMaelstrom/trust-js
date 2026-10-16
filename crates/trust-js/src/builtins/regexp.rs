//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;

/// A RegExp prototype getter: a flag boolean (`Some(char)`), or the special source/flags string.
fn re_flag_get(i: &Interp, this: &Value, flag: Option<char>) -> Result<Value, Value> {
    if let Some(ptr) = map_ptr(this) {
        if let Some(re) = i.regexps.get(&ptr) {
            return Ok(match flag {
                Some(c) => Value::Bool(re.flags.contains(c)),
                None => Value::lstr(re.flags.clone()),
            });
        }
        // The %RegExp.prototype% object itself has default values rather than throwing.
        if i.extra_protos.get("RegExp").map(|p| Gc::as_ptr(p) as usize) == Some(ptr) {
            return Ok(match flag {
                Some(_) => Value::Undefined,
                None => Value::lstr(""),
            });
        }
    }
    Err(i.make_error(
        "TypeError",
        "RegExp.prototype getter called on a non-RegExp",
    ))
}
fn re_source_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    if let Some(ptr) = map_ptr(&this) {
        if let Some(re) = i.regexps.get(&ptr) {
            if re.source.is_empty() {
                return Ok(Value::lstr("(?:)"));
            }
            // EscapeRegExpPattern only needs to alter solidus and line terminators for a
            // non-empty source. When none occur, the original source is already a valid literal
            // body and can be retained directly (ECMA-262 §22.2.6.13.1).
            if re
                .source
                .chars()
                .all(|c| !matches!(c, '/' | '\n' | '\r' | '\u{2028}' | '\u{2029}'))
            {
                return Ok(Value::Str(crate::lstr::LStr::from(re.source.as_str())));
            }
            // EscapeRegExpPattern: "/" and line terminators are escaped so "/"+S+"/"+F re-parses.
            let mut out = String::new();
            let mut escaped = false;
            let mut in_class = false;
            for c in re.source.chars() {
                // A line terminator renders as its escape sequence; directly after a backslash the
                // pair `\<LF>` collapses to `\n` (both match the terminator), reusing that slash.
                let terminator = match c {
                    '\n' => Some("n"),
                    '\r' => Some("r"),
                    '\u{2028}' => Some("u2028"),
                    '\u{2029}' => Some("u2029"),
                    _ => None,
                };
                if let Some(esc) = terminator {
                    if !escaped {
                        out.push('\\');
                    }
                    out.push_str(esc);
                    escaped = false;
                    continue;
                }
                if escaped {
                    out.push(c);
                    escaped = false;
                    continue;
                }
                match c {
                    '\\' => {
                        out.push(c);
                        escaped = true;
                    }
                    '[' => {
                        in_class = true;
                        out.push(c);
                    }
                    ']' => {
                        in_class = false;
                        out.push(c);
                    }
                    // Inside a character class a solidus can't terminate the literal.
                    '/' if !in_class => out.push_str("\\/"),
                    c => out.push(c),
                }
            }
            return Ok(Value::lstr(out));
        }
        if i.extra_protos.get("RegExp").map(|p| Gc::as_ptr(p) as usize) == Some(ptr) {
            return Ok(Value::lstr("(?:)"));
        }
    }
    Err(i.make_error(
        "TypeError",
        "RegExp.prototype.source called on a non-RegExp",
    ))
}

fn re_flags_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    if !matches!(this, Value::Obj(_)) {
        return Err(i.make_error(
            "TypeError",
            "RegExp.prototype.flags getter called on non-object",
        ));
    }
    let mut out = String::new();
    for (prop, ch) in [
        ("hasIndices", 'd'),
        ("global", 'g'),
        ("ignoreCase", 'i'),
        ("multiline", 'm'),
        ("dotAll", 's'),
        ("unicode", 'u'),
        ("unicodeSets", 'v'),
        ("sticky", 'y'),
    ] {
        let v = ab(i.get_member_key(&this, prop))?;
        if i.to_boolean(&v) {
            out.push(ch);
        }
    }
    Ok(Value::lstr(out))
}

macro_rules! regexp_flag_getter {
    ($name:ident, $flag:literal) => {
        fn $name(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
            re_flag_get(i, &this, Some($flag))
        }
    };
}
regexp_flag_getter!(re_has_indices_get, 'd');
regexp_flag_getter!(re_global_get, 'g');
regexp_flag_getter!(re_ignore_case_get, 'i');
regexp_flag_getter!(re_multiline_get, 'm');
regexp_flag_getter!(re_dot_all_get, 's');
regexp_flag_getter!(re_unicode_get, 'u');
regexp_flag_getter!(re_unicode_sets_get, 'v');
regexp_flag_getter!(re_sticky_get, 'y');

/// The native symbol methods may read `flags`, every component flag getter, and `exec`.
/// A wrapper-eliding specialization must prove those live inherited dependencies, not merely
/// the outer symbol-method identity.
pub(super) fn literal_match_dependencies_canonical(i: &Interp) -> bool {
    let Some(proto) = i.extra_protos.get("RegExp") else {
        return false;
    };
    let p = proto.borrow();
    let names = [
        "flags",
        "hasIndices",
        "global",
        "ignoreCase",
        "multiline",
        "dotAll",
        "unicode",
        "unicodeSets",
        "sticky",
        "exec",
    ];
    let proto_ptr = Gc::as_ptr(proto) as usize;
    let shape = p.props.shape();
    let cached = i.regexp_dependency_cache.get();
    let slots = if cached.proto == proto_ptr && cached.shape == shape {
        cached.slots
    } else {
        let mut slots = [u32::MAX; 10];
        for (index, name) in names.iter().enumerate() {
            let Some(slot) = p.props.slot_of(name) else {
                return false;
            };
            slots[index] = slot as u32;
        }
        if crate::value::is_cacheable_shape(shape) {
            i.regexp_dependency_cache
                .set(crate::interpreter::RegexpDependencyCache {
                    proto: proto_ptr,
                    shape,
                    slots,
                });
        }
        slots
    };
    let property_at = |index: usize| {
        p.props
            .entry_at(slots[index] as usize)
            .filter(|(key, _)| &***key == names[index])
            .map(|(_, property)| property)
    };
    let getter_is = |index: usize, native: NativeFn| {
        property_at(index).is_some_and(|property| {
            matches!(
                property.getter(),
                Some(Value::Obj(function))
                    if matches!(
                        function.borrow().call,
                        Callable::Native(found)
                            if found as usize == native as *const () as usize
                    )
            )
        })
    };
    getter_is(0, re_flags_get)
        && getter_is(1, re_has_indices_get)
        && getter_is(2, re_global_get)
        && getter_is(3, re_ignore_case_get)
        && getter_is(4, re_multiline_get)
        && getter_is(5, re_dot_all_get)
        && getter_is(6, re_unicode_get)
        && getter_is(7, re_unicode_sets_get)
        && getter_is(8, re_sticky_get)
        && property_at(9)
            .filter(|property| !property.accessor())
            .map(Property::value)
            .is_some_and(|value| {
                matches!(
                    value,
                    Value::Obj(ref function)
                        if matches!(
                            function.borrow().call,
                            Callable::Native(native)
                                if native as usize == regexp_exec as *const () as usize
                        )
                )
            })
}

/// Flag names `RegExp.prototype.flags` observes on its receiver (ECMA-262 §22.2.6.6). An own
/// property with any of these names makes the observable flags string depend on that property,
/// so the generic `Get` path must run instead of a direct engine-side read.
const FLAG_PROP_NAMES: [&str; 9] = [
    "hasIndices",
    "global",
    "ignoreCase",
    "multiline",
    "dotAll",
    "unicode",
    "unicodeSets",
    "sticky",
    "flags",
];

/// Read the engine-owned flags of a RegExp without dispatching the `flags` getter, when the
/// observable `RegExp.prototype.flags` string is provably canonical: `this` is an ordinary
/// direct-RegExp instance with no own flag-named property overrides, and the live
/// %RegExp.prototype% still carries the canonical `flags` and component-flag getters. Returns
/// the compiled Regex for a direct read; `None` keeps the generic getter with its observable
/// Gets (a user-overridden `flags`, a subclass prototype, or an own flag property all force it).
fn direct_reg_exp_flags(i: &Interp, this: &Value) -> Option<std::rc::Rc<crate::regex::Regex>> {
    let Value::Obj(obj) = this else {
        return None;
    };
    let ptr = Gc::as_ptr(obj) as usize;
    let re = i.regexps.get(&ptr).cloned()?;
    let b = obj.borrow();
    if !matches!(b.exotic, Exotic::None)
        || !b
            .proto
            .as_ref()
            .zip(i.extra_protos.get("RegExp"))
            .is_some_and(|(a, b)| Gc::ptr_eq(a, b))
        || FLAG_PROP_NAMES.iter().any(|name| b.props.contains(name))
    {
        return None;
    }
    drop(b);
    literal_match_dependencies_canonical(i).then_some(re)
}

/// The object and matcher of `this` when RegExpExec(this, S) is RegExpBuiltinExec without
/// observable lookups (ECMA-262 §22.2.7.1): a direct, ordinary RegExp instance with no own `exec`
/// or flag properties whose prototype carries the intrinsic `exec` and flag getters, and whose own
/// `lastIndex` is a writable non-negative integral Number, so neither reading it (ToLength) nor
/// writing it runs user code or throws.
fn direct_regexp_exec(i: &Interp, this: &Value) -> Option<(Gc, std::rc::Rc<crate::regex::Regex>)> {
    let Value::Obj(obj) = this else {
        return None;
    };
    let re = direct_reg_exp_flags(i, this)?;
    let b = obj.borrow();
    let last = b.props.get("lastIndex")?;
    let plain_last = !last.accessor()
        && last.writable()
        && matches!(last.value(), Value::Num(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0);
    if !plain_last || b.props.contains("exec") {
        return None;
    }
    drop(b);
    Some((obj.clone(), re))
}

/// The second (flags) argument to RegExp / RegExp.prototype.compile: undefined → "", else ToString.
fn regexp_flags_arg(i: &mut Interp, a: &[Value]) -> Result<String, Value> {
    match arg(a, 1) {
        Value::Undefined => Ok(String::new()),
        v => Ok(ab(i.to_string(&v))?.to_string()),
    }
}

pub(super) fn install_regexp(it: &mut Interp) {
    let proto = Object::new(Some(it.object_proto.clone()));
    Rc::make_mut(&mut it.extra_protos).insert("RegExp", proto.clone());
    // source/flags/global/... accessor getters (computed from the matcher).
    let add_getter = |it: &mut Interp, proto: &Gc, name: &str, f: NativeFn| {
        let g = it.make_native(&format!("get {name}"), 0, f);
        proto.borrow_mut().props.insert(
            name,
            Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
        );
    };
    add_getter(it, &proto, "source", re_source_get);
    // `get flags` is generic: it reads each component flag accessor via [[Get]] on the receiver.
    add_getter(it, &proto, "flags", re_flags_get);
    add_getter(it, &proto, "global", re_global_get);
    add_getter(it, &proto, "ignoreCase", re_ignore_case_get);
    add_getter(it, &proto, "multiline", re_multiline_get);
    add_getter(it, &proto, "dotAll", re_dot_all_get);
    add_getter(it, &proto, "sticky", re_sticky_get);
    add_getter(it, &proto, "unicode", re_unicode_get);
    add_getter(it, &proto, "hasIndices", re_has_indices_get);
    add_getter(it, &proto, "unicodeSets", re_unicode_sets_get);
    // Annex B B.2.5 RegExp.prototype.compile(pattern, flags): recompile this regex in place.
    it.def_method(&proto, "compile", 2, |i, this, a| {
        let ptr = map_ptr(&this).filter(|p| i.regexps.contains_key(p));
        let ptr = ptr.ok_or_else(|| i.make_error("TypeError", "compile called on non-RegExp"))?;
        // The legacy-regexp proposal: compile requires a DIRECT same-realm RegExp instance (a
        // subclass or cross-realm instance is a TypeError).
        if let (Value::Obj(o), Some(rp)) = (&this, i.extra_protos.get("RegExp")) {
            let direct = o
                .borrow()
                .proto
                .as_ref()
                .map(|p| Gc::ptr_eq(p, rp))
                .unwrap_or(false);
            if !direct {
                return Err(i.make_error(
                    "TypeError",
                    "RegExp.prototype.compile requires a direct RegExp instance",
                ));
            }
        }
        let (source, flags) = match arg(a, 0) {
            Value::Obj(o) if i.regexps.contains_key(&(Gc::as_ptr(&o) as usize)) => {
                // A RegExp pattern copies its source/flags; a second flags argument is then an error.
                if !matches!(arg(a, 1), Value::Undefined) {
                    return Err(
                        i.make_error("TypeError", "cannot supply flags when compiling a RegExp")
                    );
                }
                let re = i.regexps[&(Gc::as_ptr(&o) as usize)].clone();
                (re.source.clone(), re.flags.clone())
            }
            Value::Undefined => (String::new(), regexp_flags_arg(i, a)?),
            v => {
                let src = ab(i.to_string(&v))?.to_string();
                (src, regexp_flags_arg(i, a)?)
            }
        };
        let re = crate::regex::Regex::new(&source, &flags)
            .map_err(|e| i.make_error("SyntaxError", &e))?;
        if let Value::Obj(o) = &this {
            i.gc_pin(o);
        }
        i.regexps.insert(ptr, Rc::new(re));
        set_throw(i, &this, "lastIndex", Value::Num(0.0))?;
        ab(i.set_member_key(&this, "lastIndex", Value::Num(0.0)))?;
        Ok(this)
    });
    it.def_method(&proto, "exec", 1, regexp_exec);
    it.def_method(&proto, "test", 1, |i, this, a| {
        // RegExpExec: an own/inherited callable `exec` takes precedence over the built-in matcher.
        if !matches!(this, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "RegExp.prototype.test requires an object"));
        }
        let s = ab(i.to_string(&arg(a, 0)))?;
        // With the intrinsic `exec`, run RegExpBuiltinExec's matcher, `lastIndex` and legacy
        // static effects without building the match array `test` would discard.
        if let Some((obj, re)) = direct_regexp_exec(i, &this) {
            return regexp_exec_discard_direct(i, &obj, &re, &s).map(Value::Bool);
        }
        Ok(Value::Bool(!matches!(
            regexp_exec_abstract(i, &this, s)?,
            Value::Null
        )))
    });
    it.def_method(&proto, "toString", 0, |i, this, _| {
        if !matches!(this, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "RegExp.prototype.toString requires an object"));
        }
        let src_v = ab(i.get_member_key(&this, "source"))?;
        let src = ab(i.to_string(&src_v))?;
        let flags_v = ab(i.get_member_key(&this, "flags"))?;
        let flags = ab(i.to_string(&flags_v))?;
        // RegExp.prototype.toString performs the two observable Gets/ToStrings above, then
        // concatenates the four pieces directly (ECMA-262 §22.2.6.17).
        Ok(Value::Str(crate::lstr::LStr::concat4(
            "/", &src, "/", &flags,
        )))
    });
    let ctor = it.make_native("RegExp", 2, |i, _t, a| {
        let pattern = arg(a, 0);
        let flags_arg = arg(a, 1);
        // IsRegExp: an Object whose truthy @@match (when defined) or [[RegExpMatcher]] slot.
        let has_slots = matches!(&pattern, Value::Obj(o)
            if i.regexps.contains_key(&(Gc::as_ptr(o) as usize)));
        let pattern_is_regexp = if matches!(pattern, Value::Obj(_)) {
            let m = match well_known_key(i, "match") {
                Some(key) => ab(i.get_member_key(&pattern, &key))?,
                None => Value::Undefined,
            };
            if !matches!(m, Value::Undefined) {
                i.to_boolean(&m)
            } else {
                has_slots
            }
        } else {
            false
        };
        // Called as a function on a regexp(-like) whose .constructor is this same
        // constructor: return the argument unchanged.
        if !i.constructing && pattern_is_regexp && matches!(flags_arg, Value::Undefined) {
            let c = ab(i.get_member_key(&pattern, "constructor"))?;
            let same = match (&c, i.extra_protos.get("%RegExpCtor%")) {
                (Value::Obj(co), Some(rc)) => Gc::ptr_eq(co, rc),
                _ => false,
            };
            if same {
                return Ok(pattern);
            }
        }
        // Observable step order: the pattern's internal slots (or `source`/`flags` Gets) are
        // read first, then RegExpAlloc fetches newTarget's `prototype`, and only then does
        // RegExpInitialize run the ToString coercions.
        let (src_v, fl_v) = if has_slots {
            let re = match &pattern {
                Value::Obj(o) => i.regexps[&(Gc::as_ptr(o) as usize)].clone(),
                _ => unreachable!(),
            };
            let fl = match flags_arg {
                Value::Undefined => Value::lstr(re.flags.clone()),
                v => v,
            };
            (Value::lstr(re.source.clone()), fl)
        } else if pattern_is_regexp {
            // A regexp-like object: read its `source` and `flags` properties.
            let src = ab(i.get_member_key(&pattern, "source"))?;
            let fl = match flags_arg {
                Value::Undefined => ab(i.get_member_key(&pattern, "flags"))?,
                v => v,
            };
            (src, fl)
        } else {
            (pattern, flags_arg)
        };
        let alloc_proto = ab(i.regexp_alloc_proto())?;
        let source = match src_v {
            Value::Undefined => String::new(),
            v => ab(i.to_string(&v))?.to_string(),
        };
        let flags = match fl_v {
            Value::Undefined => String::new(),
            v => ab(i.to_string(&v))?.to_string(),
        };
        ab(i.make_regexp_with_proto(&source, &flags, alloc_proto))
    });
    Rc::make_mut(&mut it.extra_protos).insert("%RegExpCtor%", ctor.clone());
    install_regexp_legacy_statics(it, &ctor);
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto.clone()), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
    install_species(it, &ctor);
    // The split paths prove the splitter's species lookup inert by this getter's identity: its
    // trivial native body may share an address with other functions, so it cannot be recognized
    // by function pointer.
    let species_getter = well_known_key(it, "species").and_then(|key| {
        match ctor.borrow().props.get(&key).and_then(Property::getter) {
            Some(Value::Obj(getter)) => Some(getter.clone()),
            _ => None,
        }
    });
    if let Some(getter) = species_getter {
        Rc::make_mut(&mut it.extra_protos).insert("%RegExpSpeciesGetter%", getter);
    }

    // The @@match/@@replace/@@search/@@split/@@matchAll methods on RegExp.prototype (what
    // String.prototype.{match,replace,...} dispatch to).
    let methods: [(&str, usize, NativeFn); 5] = [
        ("match", 1, re_sym_match),
        ("replace", 2, re_sym_replace),
        ("search", 1, re_sym_search),
        ("split", 2, re_sym_split),
        ("matchAll", 1, re_sym_matchall),
    ];
    for (sym, len, f) in methods {
        if let Some(key) = well_known_key(it, sym) {
            let m = it.make_native(&format!("[Symbol.{sym}]"), len, f);
            proto
                .borrow_mut()
                .props
                .insert(key, Property::builtin(Value::Obj(m)));
        }
    }
    it.def_method(&ctor, "escape", 1, |i, _t, a| {
        let s = match arg(a, 0) {
            Value::Str(s) => s,
            _ => return Err(i.make_error("TypeError", "RegExp.escape requires a string")),
        };
        let mut out = String::new();
        for (idx, cp) in crate::jstr::code_points(&s).into_iter().enumerate() {
            out.push_str(&regexp_escape_cp(cp, idx == 0));
        }
        Ok(Value::lstr(out))
    });
    set_builtin(&it.global, "RegExp", Value::Obj(ctor));
}

/// EncodeForRegExpEscape: escape one code point for `RegExp.escape` (`cp` may be a lone
/// surrogate, which always hex-escapes).
fn regexp_escape_cp(cp: u32, first: bool) -> String {
    if (0xD800..0xE000).contains(&cp) {
        return format!("\\u{cp:04x}");
    }
    if cp >= crate::jstr::SMUGGLE_BASE {
        // Private-use characters are returned unescaped, in the engine's pair form.
        return crate::jstr::from_code_point(cp);
    }
    let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
    // The first character, if alphanumeric, is hex-escaped so the result can't start an identifier.
    if first && c.is_ascii_alphanumeric() {
        return format!("\\x{cp:02x}");
    }
    if "^$\\.*+?()[]{}|/".contains(c) {
        return format!("\\{c}");
    }
    match c {
        '\t' => "\\t".into(),
        '\n' => "\\n".into(),
        '\u{0b}' => "\\v".into(),
        '\u{0c}' => "\\f".into(),
        '\r' => "\\r".into(),
        // EncodeForRegExpEscape step 5: the other punctuators, WhiteSpace and LineTerminator
        // (not U+0085 or other control characters, which are returned as themselves).
        _ if ",-=<>#&!%:;@~'`\"".contains(c) || crate::regex::js_whitespace(c) => {
            if cp <= 0xff {
                format!("\\x{cp:02x}")
            } else if cp <= 0xffff {
                format!("\\u{cp:04x}")
            } else {
                format!("\\u{{{cp:x}}}")
            }
        }
        _ => c.to_string(),
    }
}

/// The legacy RegExp static accessors (`RegExp.input`/`$_`, `lastMatch`/`$&`, `lastParen`/`$+`,
/// `leftContext`/`` $` ``, `rightContext`/`$'`, `$1`..`$9`): getters (and the input setter)
/// brand-check that the receiver IS this realm's %RegExp% constructor.
fn regexp_legacy_brand(i: &mut Interp, this: &Value) -> Result<Gc, Value> {
    let ctor = i.extra_protos.get("%RegExpCtor%").cloned();
    match (this, ctor) {
        (Value::Obj(o), Some(c)) if Gc::ptr_eq(o, &c) => Ok(c),
        _ => Err(i.make_error(
            "TypeError",
            "RegExp legacy static accessor called on an incompatible receiver",
        )),
    }
}

fn regexp_legacy_get(i: &mut Interp, this: &Value, slot: &str) -> Result<Value, Value> {
    let c = regexp_legacy_brand(i, this)?;
    // Materialize the deferred last-match state (if any) before reading.
    super::flush_regexp_legacy(i);
    let v = c.borrow().internal_slots().get(slot).map(|p| p.value());
    Ok(v.unwrap_or_else(|| Value::lstr("")))
}

macro_rules! legacy_getter {
    ($fname:ident, $slot:literal) => {
        fn $fname(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
            regexp_legacy_get(i, &this, $slot)
        }
    };
}
legacy_getter!(lg_input, "#\u{0}legacy_input");
legacy_getter!(lg_last_match, "#\u{0}legacy_lastMatch");
legacy_getter!(lg_last_paren, "#\u{0}legacy_lastParen");
legacy_getter!(lg_left, "#\u{0}legacy_leftContext");
legacy_getter!(lg_right, "#\u{0}legacy_rightContext");
legacy_getter!(lg_d1, "#\u{0}legacy_$1");
legacy_getter!(lg_d2, "#\u{0}legacy_$2");
legacy_getter!(lg_d3, "#\u{0}legacy_$3");
legacy_getter!(lg_d4, "#\u{0}legacy_$4");
legacy_getter!(lg_d5, "#\u{0}legacy_$5");
legacy_getter!(lg_d6, "#\u{0}legacy_$6");
legacy_getter!(lg_d7, "#\u{0}legacy_$7");
legacy_getter!(lg_d8, "#\u{0}legacy_$8");
legacy_getter!(lg_d9, "#\u{0}legacy_$9");

fn lg_set_input(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    regexp_legacy_brand(i, &this)?;
    let v = ab(i.to_string(&arg(a, 0)))?;
    let c = regexp_legacy_brand(i, &this)?;
    // Materialize any deferred match first so this write isn't later clobbered by its flush.
    super::flush_regexp_legacy(i);
    c.borrow_mut().internal_slots_mut().insert(
        "#\u{0}legacy_input",
        Property::data(Value::Str(v), true, false, false),
    );
    Ok(Value::Undefined)
}

pub(super) fn install_regexp_legacy_statics(it: &mut Interp, ctor: &Gc) {
    let entries: [(&[&str], NativeFn, bool); 14] = [
        (&["input", "$_"], lg_input, true),
        (&["lastMatch", "$&"], lg_last_match, false),
        (&["lastParen", "$+"], lg_last_paren, false),
        (&["leftContext", "$`"], lg_left, false),
        (&["rightContext", "$'"], lg_right, false),
        (&["$1"], lg_d1, false),
        (&["$2"], lg_d2, false),
        (&["$3"], lg_d3, false),
        (&["$4"], lg_d4, false),
        (&["$5"], lg_d5, false),
        (&["$6"], lg_d6, false),
        (&["$7"], lg_d7, false),
        (&["$8"], lg_d8, false),
        (&["$9"], lg_d9, false),
    ];
    for (names, get, with_set) in entries {
        for name in names {
            let g = it.make_native("get", 0, get);
            let set = if with_set {
                Some(Value::Obj(it.make_native("set", 1, lg_set_input)))
            } else {
                None
            };
            ctor.borrow_mut().props.insert(
                *name,
                Property::accessor_prop(Some(Value::Obj(g)), set, false, true),
            );
        }
    }
}

/// `this` must be an Object for the generic `RegExp.prototype[@@…]` methods (they read flags and
/// call `exec` through ordinary property access rather than the internal slots).
fn require_regexp_this(i: &mut Interp, this: &Value, name: &str) -> Result<(), Value> {
    match this {
        Value::Obj(_) => Ok(()),
        _ => Err(i.make_error("TypeError", format!("{name} called on a non-object"))),
    }
}

pub(super) fn re_sym_match(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.match]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    let (flags, unicode) = match direct_reg_exp_flags(i, &this) {
        Some(re) => {
            let flags = re.flags.clone();
            let unicode = flags.contains('u') || flags.contains('v');
            (flags, unicode)
        }
        None => {
            let flags_v = ab(i.get_member_key(&this, "flags"))?;
            let flags = ab(i.to_string(&flags_v))?.to_string();
            let unicode = flags.contains('u') || flags.contains('v');
            (flags, unicode)
        }
    };
    if !flags.contains('g') {
        return regexp_exec_abstract(i, &this, s);
    }
    set_throw(i, &this, "lastIndex", Value::Num(0.0))?;
    // Rooted across later exec calls (I1).
    let mut results = crate::rooting::RootedVec::new();
    loop {
        let result = regexp_exec_abstract(i, &this, s.clone())?;
        if matches!(result, Value::Null) {
            break;
        }
        let m0 = ab(i.get_member_key(&result, "0"))?;
        let match_str = ab(i.to_string(&m0))?;
        results.push(Value::Str(match_str.clone()));
        if match_str.is_empty() {
            let li = ab(i.get_member_key(&this, "lastIndex"))?;
            let li = to_length_val(i, &li)?;
            let next = advance_string_index(li, &s, unicode);
            set_throw(i, &this, "lastIndex", Value::Num(next as f64))?;
        }
    }
    if results.is_empty() {
        Ok(Value::Null)
    } else {
        Ok(i.make_array(results.into_vec()))
    }
}

/// Execute a dead-result `String.prototype.match` for the fully canonical direct-RegExp case.
/// Every observable lookup is proven by the caller to resolve to the realm intrinsics, so the
/// matcher runs (updating `lastIndex` and the legacy statics exactly like live match) while the
/// result array or `null` is never materialized. `global`/`unicode` come from the engine-owned
/// flags under the same canonicality proof used by the replacement discard.
pub(super) fn re_sym_match_discard_direct(
    i: &mut Interp,
    obj: &Gc,
    re: &Rc<crate::regex::Regex>,
    input: &crate::lstr::LStr,
) -> Result<Value, Value> {
    let global = re.global;
    let unicode = re.unicode;
    if global {
        obj.borrow_mut()
            .props
            .set_value("lastIndex", Value::Num(0.0));
    }
    loop {
        let matched = regexp_exec_discard_direct(i, obj, re, input)?;
        if !matched || !global {
            break;
        }
        let empty = i
            .regexp_last
            .as_ref()
            .and_then(|last| last.caps.first().copied().flatten())
            .is_some_and(|(start, end)| start == end);
        if empty {
            let li = match obj.borrow().props.get("lastIndex").map(Property::value) {
                Some(Value::Num(n)) => n as usize,
                _ => unreachable!("direct match lastIndex guard"),
            };
            let next = advance_string_index(li, input, unicode);
            obj.borrow_mut()
                .props
                .set_value("lastIndex", Value::Num(next as f64));
        }
    }
    Ok(Value::Undefined)
}
fn re_sym_search(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.search]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    let prev = ab(i.get_member_key(&this, "lastIndex"))?;
    if !same_value(&prev, &Value::Num(0.0)) {
        set_throw(i, &this, "lastIndex", Value::Num(0.0))?;
    }
    let result = regexp_exec_abstract(i, &this, s)?;
    let cur = ab(i.get_member_key(&this, "lastIndex"))?;
    if !same_value(&cur, &prev) {
        set_throw(i, &this, "lastIndex", prev)?;
    }
    match result {
        Value::Null => Ok(Value::Num(-1.0)),
        _ => ab(i.get_member_key(&result, "index")),
    }
}
pub(super) fn re_sym_replace(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    re_sym_replace_impl(i, this, a, false)
}

/// Committed dead-result `@@replace` for a direct RegExp whose symbol method, flag accessors, and
/// exec method have all been proven canonical by the caller. The skipped wrapper operations are
/// therefore side-effect-free; matcher state, legacy statics, `lastIndex`, and empty-match
/// advancement remain identical to the builtin algorithm.
pub(super) fn re_sym_replace_discard_direct(
    i: &mut Interp,
    obj: &Gc,
    re: &Rc<crate::regex::Regex>,
    input: &crate::lstr::LStr,
) -> Result<Value, Value> {
    let global = re.global;
    let unicode = re.unicode;
    if global {
        obj.borrow_mut()
            .props
            .set_value("lastIndex", Value::Num(0.0));
    }
    loop {
        let matched = regexp_exec_discard_direct(i, obj, re, input)?;
        if !matched || !global {
            break;
        }
        let empty = i
            .regexp_last
            .as_ref()
            .and_then(|last| last.caps.first().copied().flatten())
            .is_some_and(|(start, end)| start == end);
        if empty {
            let li = match obj.borrow().props.get("lastIndex").map(Property::value) {
                Some(Value::Num(n)) => n as usize,
                _ => unreachable!("direct replace lastIndex guard"),
            };
            let next = advance_string_index(li, input, unicode);
            obj.borrow_mut()
                .props
                .set_value("lastIndex", Value::Num(next as f64));
        }
    }
    Ok(Value::Undefined)
}

fn re_sym_replace_impl(
    i: &mut Interp,
    this: Value,
    a: &[Value],
    discard_result: bool,
) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.replace]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    let repl = arg(a, 1);
    let functional = repl.is_callable();
    let repl_str = if functional {
        crate::lstr::LStr::from("")
    } else {
        ab(i.to_string(&repl))?
    };
    // With the intrinsic `exec` (which implies the intrinsic flag getters) every remaining step
    // reads engine state only, so the matches come straight from the matcher.
    if !discard_result {
        if let Some((obj, re)) = direct_regexp_exec(i, &this) {
            let replacer = functional.then_some(&repl);
            return re_sym_replace_direct(i, &obj, &re, &s, replacer, &repl_str);
        }
    }
    // Globalness/unicodeness come from the `flags` string. The hot path reads the engine-owned
    // flags directly when the observable `flags` getter result is provably canonical, avoiding
    // the getter's eight observable [[Get]]s and per-flag native dispatches on every call.
    let (flags, global, unicode) = match direct_reg_exp_flags(i, &this) {
        Some(re) => {
            let flags = re.flags.clone();
            let global = flags.contains('g');
            let unicode = flags.contains('u') || flags.contains('v');
            (flags, global, unicode)
        }
        None => {
            let flags_v = ab(i.get_member_key(&this, "flags"))?;
            let flags = ab(i.to_string(&flags_v))?.to_string();
            let global = flags.contains('g');
            let unicode = flags.contains('u') || flags.contains('v');
            (flags, global, unicode)
        }
    };
    if global {
        set_throw(i, &this, "lastIndex", Value::Num(0.0))?;
    }
    // When String#replace's result is dead, a direct ordinary RegExp can execute without
    // materializing match arrays or the replacement string. All observable flag Gets and
    // replacement coercion above have already happened. The live exec identity/prototype and
    // internal flags are checked before entering the allocation-free matcher.
    if discard_result && !functional {
        if let Value::Obj(obj) = &this {
            let ptr = Gc::as_ptr(obj) as usize;
            let re = i.regexps.get(&ptr).cloned();
            let direct_exec = re.as_ref().is_some_and(|re| {
                let b = obj.borrow();
                let direct_proto = b
                    .proto
                    .as_ref()
                    .zip(i.extra_protos.get("RegExp"))
                    .is_some_and(|(a, b)| Gc::ptr_eq(a, b));
                if !matches!(b.exotic, Exotic::None)
                    || b.props.contains("exec")
                    || !direct_proto
                    || re.flags != flags.as_ref()
                {
                    return false;
                }
                drop(b);
                i.extra_protos
                    .get("RegExp")
                    .and_then(|proto| {
                        let p = proto.borrow();
                        p.props
                            .get("exec")
                            .filter(|prop| !prop.accessor())
                            .map(Property::value)
                    })
                    .is_some_and(|value| {
                        let Value::Obj(exec) = value else {
                            return false;
                        };
                        let canonical = matches!(
                            exec.borrow().call,
                            Callable::Native(nf)
                                if nf as usize == regexp_exec as *const () as usize
                        );
                        canonical
                    })
            });
            if direct_exec {
                loop {
                    let matched = regexp_exec_discard_fast(i, &this, &s)
                        .expect("direct regexp discard guards")?;
                    if !matched || !global {
                        break;
                    }
                    let empty = i
                        .regexp_last
                        .as_ref()
                        .and_then(|last| last.caps.first().copied().flatten())
                        .is_some_and(|(start, end)| start == end);
                    if empty {
                        let li = match obj.borrow().props.get("lastIndex").map(Property::value) {
                            Some(Value::Num(n)) => n as usize,
                            _ => unreachable!("discard matcher guarded lastIndex"),
                        };
                        let next = advance_string_index(li, &s, unicode);
                        obj.borrow_mut()
                            .props
                            .set_value("lastIndex", Value::Num(next as f64));
                    }
                }
                return Ok(Value::Undefined);
            }
        }
    }
    // Collect every match (RegExpExec advances lastIndex; empty matches step forward manually).
    // The results stay rooted through later exec calls and the replacement pass (I1).
    let mut results = crate::rooting::RootedVec::new();
    loop {
        let result = regexp_exec_abstract(i, &this, s.clone())?;
        if matches!(result, Value::Null) {
            break;
        }
        results.push(result.clone());
        if !global {
            break;
        }
        let m0_v = ab(i.get_member_key(&result, "0"))?;
        let m0 = ab(i.to_string(&m0_v))?;
        if m0.is_empty() {
            let li_v = ab(i.get_member_key(&this, "lastIndex"))?;
            let li = to_length_val(i, &li_v)?;
            let next = advance_string_index(li, &s, unicode);
            set_throw(i, &this, "lastIndex", Value::Num(next as f64))?;
        }
    }
    // The immutable input's length/units have no observable effects. Defer materialization
    // until matching succeeds; all replacement coercions, flag/exec Gets and lastIndex writes
    // above still run on the no-match path (ECMA-262 RegExp.prototype [ %Symbol.replace% ]).
    if results.is_empty() {
        return Ok(Value::Str(s));
    }
    let sunits = i.units_full(&s);
    let size = sunits.len();
    let mut accumulated = String::new();
    let mut next_pos = 0usize;
    for result in &results {
        let length_v = ab(i.get_member_key(result, "length"))?;
        let length = to_length_val(i, &length_v)?;
        let ncaptures = length.saturating_sub(1);
        let matched_v = ab(i.get_member_key(result, "0"))?;
        let matched = ab(i.to_string(&matched_v))?;
        let match_len = crate::jstr::unit_len(&matched);
        let index_v = ab(i.get_member_key(result, "index"))?;
        let pos_raw = ab(i.to_number(&index_v))?;
        let position = if pos_raw.is_nan() || pos_raw < 0.0 {
            0
        } else {
            (pos_raw as usize).min(size)
        };
        // Rooted (gc-heap): each later capture's Get and ToString may run user code.
        let mut captures = crate::rooting::RootedVec::new();
        for n in 1..=ncaptures {
            let cap = ab(i.get_member_key(result, &n.to_string()))?;
            captures.push(if matches!(cap, Value::Undefined) {
                Value::Undefined
            } else {
                Value::Str(ab(i.to_string(&cap))?)
            });
        }
        let mut named = ab(i.get_member_key(result, "groups"))?;
        // Non-functional replace ToObjects a present `groups` (null → TypeError).
        if !functional && !matches!(named, Value::Undefined) && !matches!(named, Value::Obj(_)) {
            match named {
                Value::Null => {
                    return Err(i.make_error("TypeError", "cannot convert null to object"));
                }
                other => named = box_primitive(i, other),
            }
        }
        let replacement = if functional {
            let mut cbargs = vec![Value::Str(matched.clone())];
            cbargs.extend(captures.iter().cloned());
            cbargs.push(Value::Num(position as f64));
            cbargs.push(Value::Str(s.clone()));
            if !matches!(named, Value::Undefined) {
                cbargs.push(named.clone());
            }
            let r = ab(i.call(repl.clone(), Value::Undefined, &cbargs))?;
            ab(i.to_string(&r))?.to_string()
        } else {
            get_substitution(i, &matched, &sunits, position, &captures, &named, &repl_str)?
        };
        if position >= next_pos {
            accumulated.push_str(&crate::jstr::from_units(&sunits[next_pos..position]));
            accumulated.push_str(&replacement);
            // Pathological expansion ($1 of a huge match, repeated) dies as a RangeError.
            if accumulated.len() > MAX_STR_LEN {
                return Err(i.make_error("RangeError", "Invalid string length"));
            }
            next_pos = position + match_len;
        }
    }
    if next_pos < size {
        accumulated.push_str(&crate::jstr::from_units(&sunits[next_pos..]));
    }
    Ok(Value::lstr(
        crate::jstr::canonicalize(&accumulated).unwrap_or(accumulated),
    ))
}

/// RegExp.prototype[@@replace] (ECMA-262 §22.2.6.11) for a regexp whose RegExpExec is the
/// intrinsic (see [`direct_regexp_exec`]), after the observable coercions of the subject,
/// replacement and flags. Matches are collected straight from the matcher with RegExpBuiltinExec's
/// `lastIndex` writes and legacy-static updates, but without match arrays or the generic reads
/// back out of them. A replacement function then runs once per match, after every match was
/// found, as in the generic algorithm; a template without `$` is inserted as is.
fn re_sym_replace_direct(
    i: &mut Interp,
    obj: &Gc,
    re: &Rc<crate::regex::Regex>,
    s: &crate::lstr::LStr,
    replacer: Option<&Value>,
    template: &crate::lstr::LStr,
) -> Result<Value, Value> {
    let uses_last = re.global || re.sticky;
    let read_last = |obj: &Gc| match obj.borrow().props.get("lastIndex").map(Property::value) {
        Some(Value::Num(n)) => n as usize,
        _ => unreachable!("direct replace lastIndex guard"),
    };
    let set_last = |obj: &Gc, n: usize| {
        obj.borrow_mut()
            .props
            .set_value("lastIndex", Value::Num(n as f64));
    };
    let text = i.re_text(re.unicode, s);
    let size = text.unit_index(text.len());
    if re.global {
        set_last(obj, 0);
    }
    let mut matches: Vec<crate::regex::Captures> = Vec::new();
    loop {
        if let Err(error) = regexp_interrupt_tick(i) {
            return Err(regexp_match_error_value(i, error));
        }
        let last = if uses_last { read_last(obj) } else { 0 };
        if last > size {
            set_last(obj, 0);
            break;
        }
        let matched = regexp_exec_text_shared(i, re, &text, text.elem_at_unit(last));
        let Some(caps) = regexp_match_result(i, matched)? else {
            if uses_last {
                set_last(obj, 0);
            }
            break;
        };
        let (start, end) = caps[0].expect("successful regexp match has a span");
        if uses_last {
            set_last(obj, text.unit_index(end));
        }
        update_regexp_legacy_statics(i, re, &caps, &text, s);
        matches.push(caps);
        if !re.global {
            break;
        }
        if start == end {
            // AdvanceStringIndex: one code unit, or one code point in Unicode mode.
            let next = if end < text.len() {
                text.unit_index(end + 1)
            } else {
                size + 1
            };
            set_last(obj, next);
        }
    }
    if matches.is_empty() {
        return Ok(Value::Str(s.clone()));
    }
    let substitutes = replacer.is_none() && template.contains('$');
    let sunits = if substitutes {
        i.units_full(s)
    } else {
        Rc::from(Vec::new())
    };
    let mut out = String::with_capacity(s.len());
    let mut next = 0usize;
    for caps in &matches {
        let (start, end) = caps[0].expect("successful regexp match has a span");
        let position = text.unit_index(start);
        let captures = || {
            (1..=re.ngroups)
                .map(|group| match caps[group] {
                    Some((a, b)) => Value::Str(text.slice_lstr(a, b)),
                    None => Value::Undefined,
                })
                .collect::<Vec<_>>()
        };
        let replacement: crate::lstr::LStr = if let Some(function) = replacer {
            let mut args = Vec::with_capacity(re.ngroups + 4);
            args.push(Value::Str(text.slice_lstr(start, end)));
            args.extend(captures());
            args.push(Value::Num(position as f64));
            args.push(Value::Str(s.clone()));
            let groups = regexp_groups(i, re, caps, &text);
            if !matches!(groups, Value::Undefined) {
                args.push(groups);
            }
            let result = ab(i.call(function.clone(), Value::Undefined, &args))?;
            ab(i.to_string(&result))?
        } else if substitutes {
            let captures = captures();
            let named = regexp_groups(i, re, caps, &text);
            let matched = text.slice(start, end);
            get_substitution(i, &matched, &sunits, position, &captures, &named, template)?.into()
        } else {
            template.clone()
        };
        if start >= next {
            text.push_slice(&mut out, next, start);
            out.push_str(&replacement);
            // Pathological expansion ($1 of a huge match, repeated) dies as a RangeError.
            if out.len() > MAX_STR_LEN {
                return Err(i.make_error("RangeError", "Invalid string length"));
            }
            next = end;
        }
    }
    text.push_slice(&mut out, next, text.len());
    Ok(Value::lstr(crate::jstr::canonicalize(&out).unwrap_or(out)))
}

/// GetSubstitution: expand a `$`-template against a match. `captures` are already `Value::Str` /
/// `Value::Undefined`; `named` is the match's `groups` object (or `undefined`).
fn get_substitution(
    i: &mut Interp,
    matched: &str,
    sunits: &[u16],
    position: usize,
    captures: &[Value],
    named: &Value,
    template: &str,
) -> Result<String, Value> {
    let tail = (position + crate::jstr::unit_len(matched)).min(sunits.len());
    let push_cap = |out: &mut String, cap: &Value| {
        if let Value::Str(s) = cap {
            out.push_str(s);
        }
    };
    let t: Vec<char> = template.chars().collect();
    let mut out = String::new();
    let mut k = 0;
    while k < t.len() {
        // A pathological template ($1 of a megabyte match, tens of thousands of times) must die
        // as a RangeError before it exhausts memory.
        if out.len() > MAX_STR_LEN {
            return Err(i.make_error("RangeError", "Invalid string length"));
        }
        if t[k] != '$' || k + 1 >= t.len() {
            out.push(t[k]);
            k += 1;
            continue;
        }
        match t[k + 1] {
            '$' => {
                out.push('$');
                k += 2;
            }
            '&' => {
                out.push_str(matched);
                k += 2;
            }
            '`' => {
                out.push_str(&crate::jstr::from_units(
                    &sunits[..position.min(sunits.len())],
                ));
                k += 2;
            }
            '\'' => {
                out.push_str(&crate::jstr::from_units(&sunits[tail..]));
                k += 2;
            }
            '<' if !matches!(named, Value::Undefined) => {
                if let Some(rel) = t[k + 2..].iter().position(|&c| c == '>') {
                    let name: String = t[k + 2..k + 2 + rel].iter().collect();
                    let v = ab(i.get_member_key(named, &name))?;
                    if !matches!(v, Value::Undefined) {
                        out.push_str(&ab(i.to_string(&v))?);
                    }
                    k = k + 2 + rel + 1;
                } else {
                    out.push('$');
                    k += 1;
                }
            }
            d if d.is_ascii_digit() => {
                let one = d.to_digit(10).unwrap() as usize;
                let two = if k + 2 < t.len() && t[k + 2].is_ascii_digit() {
                    Some(one * 10 + t[k + 2].to_digit(10).unwrap() as usize)
                } else {
                    None
                };
                // Prefer a two-digit group reference when it is in range.
                if let Some(tw) = two {
                    if tw >= 1 && tw <= captures.len() {
                        push_cap(&mut out, &captures[tw - 1]);
                        k += 3;
                        continue;
                    }
                }
                if one >= 1 && one <= captures.len() {
                    push_cap(&mut out, &captures[one - 1]);
                    k += 2;
                } else {
                    out.push('$');
                    k += 1;
                }
            }
            _ => {
                out.push('$');
                k += 1;
            }
        }
    }
    Ok(out)
}
/// The running realm's %RegExp%: the default constructor that RegExp.prototype[@@split] and
/// [@@matchAll] pass to SpeciesConstructor (ECMA-262 §22.2.6.14 step 4, §22.2.6.9 step 4). It is
/// the intrinsic, not whatever the global `RegExp` binding currently holds.
fn regexp_intrinsic(i: &Interp) -> Value {
    i.extra_protos
        .get("%RegExpCtor%")
        .cloned()
        .map_or(Value::Undefined, Value::Obj)
}

fn re_sym_matchall(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.matchAll]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    let default_ctor = regexp_intrinsic(i);
    let c = species_constructor(i, &this, &default_ctor)?;
    let flags_v = ab(i.get_member_key(&this, "flags"))?;
    let flags = ab(i.to_string(&flags_v))?;
    let matcher = ab(i.construct(c, &[this.clone(), Value::Str(flags.clone())]))?;
    let li_v = ab(i.get_member_key(&this, "lastIndex"))?;
    let last = to_length_val(i, &li_v)?;
    ab(i.set_member_key(&matcher, "lastIndex", Value::Num(last as f64)))?;
    let global = flags.contains('g');
    let unicode = flags.contains('u') || flags.contains('v');
    Ok(create_regexp_string_iterator(
        i,
        matcher,
        s.into(),
        global,
        unicode,
    ))
}
/// The matcher of `rx` when every observable step of RegExp.prototype[@@split] before its
/// matching loop resolves to the realm's intrinsics (ECMA-262 §22.2.6.14 steps 4-10):
/// SpeciesConstructor reads the intrinsic `constructor` and @@species, `flags` reads the
/// intrinsic getters, and constructing the splitter reads `rx`'s @@match (IsRegExp) and
/// %RegExp%.prototype without running user code. The splitter's own RegExpExec then finds the
/// intrinsic `exec`. `None` when any of them could run user code.
fn split_species_canonical(i: &Interp, obj: &Gc) -> Option<Rc<crate::regex::Regex>> {
    let ptr = Gc::as_ptr(obj) as usize;
    let re = i.regexps.get(&ptr)?.clone();
    let proto = i.extra_protos.get("RegExp")?.clone();
    let ctor = i.extra_protos.get("%RegExpCtor%")?.clone();
    let species_getter = i.extra_protos.get("%RegExpSpeciesGetter%")?.clone();
    let species_key = well_known_key(i, "species")?;
    let match_key = well_known_key(i, "match")?;
    {
        let b = obj.borrow();
        if !matches!(b.exotic, Exotic::None)
            || b.proto.as_ref().is_none_or(|p| !Gc::ptr_eq(p, &proto))
            || b.props.contains(&match_key)
            || [
                "constructor",
                "flags",
                "hasIndices",
                "global",
                "ignoreCase",
                "multiline",
                "dotAll",
                "unicode",
                "unicodeSets",
                "sticky",
            ]
            .iter()
            .any(|name| b.props.contains(name))
        {
            return None;
        }
    }
    if !literal_match_dependencies_canonical(i) {
        return None;
    }
    {
        let p = proto.borrow();
        let constructor_ok = p
            .props
            .get("constructor")
            .filter(|property| !property.accessor())
            .map(Property::value)
            .is_some_and(
                |value| matches!(value, Value::Obj(ref found) if Gc::ptr_eq(found, &ctor)),
            );
        let match_ok = p
            .props
            .get(&match_key)
            .is_some_and(|property| !property.accessor());
        if !constructor_ok || !match_ok {
            return None;
        }
    }
    {
        let c = ctor.borrow();
        let species_ok = c.props.get(&species_key).is_some_and(|property| {
            matches!(
                property.getter(),
                Some(Value::Obj(function)) if Gc::ptr_eq(function, &species_getter)
            ) && property.setter().is_none()
        });
        let prototype_ok = c
            .props
            .get("prototype")
            .filter(|property| !property.accessor())
            .map(Property::value)
            .is_some_and(
                |value| matches!(value, Value::Obj(ref found) if Gc::ptr_eq(found, &proto)),
            );
        if !species_ok || !prototype_ok {
            return None;
        }
    }
    Some(re)
}

/// The non-sticky form of the splitter's pattern. A sticky attempt at each position until one
/// succeeds finds the same match as one forward search from the first of them, which is what
/// the split loops below run.
fn split_search_regexp(
    i: &mut Interp,
    re: Rc<crate::regex::Regex>,
) -> Result<Rc<crate::regex::Regex>, Value> {
    if !re.sticky {
        return Ok(re);
    }
    let flags: String = re.flags.chars().filter(|&flag| flag != 'y').collect();
    match i.compiled_regexp(&re.source, &flags) {
        Ok(search) => Ok(search),
        Err(Abrupt::Throw(value)) => Err(value),
        Err(_) => unreachable!("regexp compilation only throws"),
    }
}

/// RegExp.prototype[@@split] past its observable setup (see [`split_species_canonical`]), with
/// a Number or undefined `limit` whose conversion runs no user code. The splitter's matches come
/// straight from the matcher (see [`split_search_regexp`]), so neither the splitter object nor
/// match arrays are created; pieces, captures and legacy statics are those of the generic loop.
fn re_sym_split_direct(
    i: &mut Interp,
    rx: &Value,
    s: &crate::lstr::LStr,
    limit: &Value,
) -> Option<Result<Value, Value>> {
    let Value::Obj(obj) = rx else {
        return None;
    };
    let lim = match limit {
        Value::Undefined => u32::MAX,
        Value::Num(n) => to_uint32(*n),
        _ => return None,
    } as usize;
    let re = split_species_canonical(i, obj)?;
    Some(split_direct(i, re, s, lim))
}

fn split_direct(
    i: &mut Interp,
    re: Rc<crate::regex::Regex>,
    s: &crate::lstr::LStr,
    lim: usize,
) -> Result<Value, Value> {
    let mut parts = Vec::new();
    if lim == 0 {
        return Ok(i.make_array(parts));
    }
    let search = split_search_regexp(i, re)?;
    let text = i.re_text(search.unicode, s);
    let len = text.len();
    if len == 0 {
        let matched = regexp_exec_text_shared(i, &search, &text, 0);
        if let Some(caps) = regexp_match_result(i, matched)? {
            update_regexp_legacy_statics(i, &search, &caps, &text, s);
        } else {
            parts.push(Value::Str(s.clone()));
        }
        return Ok(i.make_array(parts));
    }
    // Element positions: `p` ends the last piece, `q` is where the next attempt starts.
    let (mut p, mut q) = (0usize, 0usize);
    while q < len {
        if let Err(error) = regexp_interrupt_tick(i) {
            return Err(regexp_match_error_value(i, error));
        }
        let matched = regexp_exec_text_shared(i, &search, &text, q);
        let Some(caps) = regexp_match_result(i, matched)? else {
            break;
        };
        let (a, e) = caps[0].expect("successful regexp match has a span");
        // The generic loop stops before attempting a match at the end of the subject.
        if a >= len {
            break;
        }
        update_regexp_legacy_statics(i, &search, &caps, &text, s);
        if e == p {
            // AdvanceStringIndex past an empty match at the last split point.
            q = a + 1;
            continue;
        }
        parts.push(Value::Str(text.slice_lstr(p, a)));
        if parts.len() == lim {
            return Ok(i.make_array(parts));
        }
        p = e;
        for group in 1..=search.ngroups {
            parts.push(match caps[group] {
                Some((x, y)) => Value::Str(text.slice_lstr(x, y)),
                None => Value::Undefined,
            });
            if parts.len() == lim {
                return Ok(i.make_array(parts));
            }
        }
        q = p;
    }
    parts.push(Value::Str(text.slice_lstr(p, len)));
    Ok(i.make_array(parts))
}

/// Execute a dead-result split for the fully canonical direct-RegExp case.
///
/// Every potentially observable lookup performed by String#split/RegExp@@split is first proven
/// to resolve to the realm intrinsics. The matcher still runs and updates the legacy RegExp
/// statics; only the unobservable sticky clone, match arrays, substrings, and final result array
/// are omitted. Returning `None` means no state was touched and the caller must use the builtin.
pub(super) fn re_sym_split_discard_fast(
    i: &mut Interp,
    input: &Value,
    separator: &Value,
    limit: &Value,
) -> Option<Result<Value, Value>> {
    let (Value::Str(input), Value::Obj(obj), Value::Undefined) = (input, separator, limit) else {
        return None;
    };
    let re = split_species_canonical(i, obj)?;
    // String#split's own @@split lookup must find the intrinsic method too.
    let split_key = well_known_key(i, "split")?;
    if obj.borrow().props.contains(&split_key) {
        return None;
    }
    let split_ok = i
        .extra_protos
        .get("RegExp")?
        .borrow()
        .props
        .get(&split_key)
        .filter(|property| !property.accessor())
        .map(Property::value)
        .is_some_and(|value| {
            matches!(
                value,
                Value::Obj(ref function)
                    if matches!(
                        function.borrow().call,
                        Callable::Native(native)
                            if native as usize == re_sym_split as *const () as usize
                    )
            )
        });
    if !split_ok {
        return None;
    }
    // A sticky split loop that advances one unit after every miss is equivalent to one ordinary
    // forward search for its next success. Removing `y` performs that search in the matcher while
    // preserving the successful captures and their legacy-static side effects. The AArch64 class
    // machine mapping is deliberately declined by the regex tier, so this remains on the checked
    // matcher for the class-heavy cases that previously exposed the machine-code issue.
    let search_re = match split_search_regexp(i, re) {
        Ok(search) => search,
        Err(error) => return Some(Err(error)),
    };
    let text = i.re_text(search_re.unicode, input);
    let size = text.unit_index(text.len());
    if size == 0 {
        let matched = regexp_find_text_shared(i, &search_re, &text, 0);
        let found = match regexp_match_result(i, matched) {
            Ok(found) => found,
            Err(error) => return Some(Err(error)),
        };
        if let Some(whole) = found {
            update_regexp_legacy_statics_lazy(i, &search_re, whole, 0, &text, input);
        }
        return Some(Ok(Value::Undefined));
    }

    let unicode = search_re.unicode;
    let mut p = 0usize;
    let mut q = 0usize;
    while q < size {
        if let Err(error) = regexp_interrupt_tick(i) {
            return Some(Err(regexp_match_error_value(i, error)));
        }
        let search_start = text.elem_at_unit(q);
        let matched = regexp_find_text_shared(i, &search_re, &text, search_start);
        let found = match regexp_match_result(i, matched) {
            Ok(found) => found,
            Err(error) => return Some(Err(error)),
        };
        let Some(whole @ (a, e)) = found else {
            break;
        };
        let a = text.unit_index(a);
        let e = text.unit_index(e);
        // RegExp@@split's sticky loop stops before attempting a match at the final position.
        if a >= size {
            break;
        }
        update_regexp_legacy_statics_lazy(i, &search_re, whole, search_start, &text, input);
        if e == p {
            q = advance_string_index(a, input, unicode);
        } else {
            p = e;
            q = p;
        }
    }
    Some(Ok(Value::Undefined))
}

pub(super) fn re_sym_split(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.split]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    if let Some(result) = re_sym_split_direct(i, &this, &s, &arg(a, 1)) {
        return result;
    }
    // All positions are UTF-16 unit offsets.
    let sunits: Vec<u16> = crate::jstr::units(&s);
    let size = sunits.len();
    // SpeciesConstructor(rx, %RegExp%), then a sticky-flagged splitter constructed from it.
    let default_ctor = regexp_intrinsic(i);
    let c = species_constructor(i, &this, &default_ctor)?;
    let flags_v = ab(i.get_member_key(&this, "flags"))?;
    let flags = ab(i.to_string(&flags_v))?;
    let unicode = flags.contains('u') || flags.contains('v');
    let new_flags = if flags.contains('y') {
        flags.to_string()
    } else {
        format!("{flags}y")
    };
    let splitter = ab(i.construct(c, &[this.clone(), Value::lstr(new_flags)]))?;
    // ToUint32 wraps modulo 2^32 over the whole Number range: -1 is 2^32 - 1, 2^32 is 0.
    let limit = match arg(a, 1) {
        Value::Undefined => u32::MAX as usize,
        v => to_uint32(ab(i.to_number(&v))?) as usize,
    };
    // Rooted across the exec calls and captures' getters (I1).
    let mut out = crate::rooting::RootedVec::new();
    if limit == 0 {
        return Ok(i.make_array(out.into_vec()));
    }
    if size == 0 {
        let z = regexp_exec_abstract(i, &splitter, s.clone())?;
        if !matches!(z, Value::Null) {
            return Ok(i.make_array(out.into_vec()));
        }
        out.push(Value::Str(s));
        return Ok(i.make_array(out.into_vec()));
    }
    let mut p = 0usize;
    let mut q = 0usize;
    while q < size {
        ab(i.set_member_key(&splitter, "lastIndex", Value::Num(q as f64)))?;
        let z = regexp_exec_abstract(i, &splitter, s.clone())?;
        if matches!(z, Value::Null) {
            q = advance_string_index(q, &s, unicode);
            continue;
        }
        let li_v = ab(i.get_member_key(&splitter, "lastIndex"))?;
        let e = to_length_val(i, &li_v)?.min(size);
        if e == p {
            q = advance_string_index(q, &s, unicode);
            continue;
        }
        out.push(Value::lstr(crate::jstr::from_units(&sunits[p..q])));
        if out.len() == limit {
            return Ok(i.make_array(out.into_vec()));
        }
        p = e;
        let len_v = ab(i.get_member_key(&z, "length"))?;
        let ncaptures = to_length_val(i, &len_v)?.saturating_sub(1);
        for n in 1..=ncaptures {
            let cap = ab(i.get_member_key(&z, &n.to_string()))?;
            out.push(cap);
            if out.len() == limit {
                return Ok(i.make_array(out.into_vec()));
            }
        }
        q = p;
    }
    out.push(Value::lstr(crate::jstr::from_units(&sunits[p..size])));
    Ok(i.make_array(out.into_vec()))
}
