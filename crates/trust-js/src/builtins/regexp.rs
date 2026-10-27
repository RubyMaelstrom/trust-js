//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

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
    if !matches!((this).view(), ValueView::Obj(_)) {
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

/// A data property of %RegExp.prototype% that the RegExp algorithms read: `exec` (RegExpExec), a
/// symbol method (String.prototype's dispatch to it) or `constructor` (SpeciesConstructor).
#[derive(Clone, Copy)]
pub(super) enum RegexpMember {
    Exec,
    Match,
    MatchAll,
    Replace,
    Search,
    Split,
    Constructor,
}

const REGEXP_MEMBERS: usize = 7;

/// The getters RegExp.prototype.flags reads on its receiver (ECMA-262 §22.2.6.4), and its own.
const REGEXP_FLAG_GETTERS: [(&str, NativeFn); 9] = [
    ("flags", re_flags_get),
    ("hasIndices", re_has_indices_get),
    ("global", re_global_get),
    ("ignoreCase", re_ignore_case_get),
    ("multiline", re_multiline_get),
    ("dotAll", re_dot_all_get),
    ("unicode", re_unicode_get),
    ("unicodeSets", re_unicode_sets_get),
    ("sticky", re_sticky_get),
];

/// [`Interp::regexp_protector`]: while `epoch` matches the live [`crate::value::proto_epoch`],
/// the current Realm's RegExp algorithms read only intrinsics. %RegExp.prototype% has the
/// intrinsic `flags` and flag getters, and a data property for each [`RegexpMember`] at `slots`
/// that held the verified object (%RegExp% for `constructor`); %RegExp% has the intrinsic
/// @@species getter. Both are marked prototypes: a defineProperty, a structural change or a
/// [[SetPrototypeOf]] bumps the epoch. An assignment can still replace a member, so each use
/// compares its slot with the verified object, which a weak pin keeps from being reused (as
/// [`super::ArrayIterationProof`] does). A RegExp of the shape RegExpAlloc gives it
/// (`instance_shape`) with %RegExp.prototype% as its prototype has no own property that these
/// algorithms read but `lastIndex`.
pub(crate) struct RegexpProof {
    epoch: u32,
    /// Noted as RegExps are made (see [`RegexpProof::note_instance_shape`]) and kept when the
    /// proof is invalidated: shapes belong to the Agent, not to a Realm.
    instance_shape: u32,
    slots: [u32; REGEXP_MEMBERS],
    /// `None` when the proof failed at `epoch`.
    pins: Option<Box<RegexpPins>>,
}

/// %RegExp.prototype% and the objects its [`RegexpMember`]s held when [`RegexpProof`] was
/// established, weakly pinned.
struct RegexpPins {
    prototype: crate::value::WeakGc,
    members: [crate::value::WeakGc; REGEXP_MEMBERS],
}

impl Default for RegexpProof {
    fn default() -> Self {
        Self {
            epoch: 0,
            instance_shape: crate::value::SHAPE_UNCACHEABLE,
            slots: [0; REGEXP_MEMBERS],
            pins: None,
        }
    }
}

impl RegexpProof {
    /// Record the shape a freshly allocated RegExp has.
    pub(crate) fn note_instance_shape(&mut self, shape: u32) {
        if crate::value::is_cacheable_shape(shape) {
            self.instance_shape = shape;
        }
    }

    /// Forget the proof, as when another Realm becomes current; epoch 0 is never live.
    pub(crate) fn invalidate(&mut self) {
        self.epoch = 0;
        self.pins = None;
    }

    /// Whether `prototype` (%RegExp.prototype%) still holds the verified object at `member`.
    fn holds(&self, pins: &RegexpPins, prototype: &Gc, member: RegexpMember) -> bool {
        let member = member as usize;
        prototype
            .borrow()
            .props
            .property_at(self.slots[member] as usize)
            .is_some_and(|property| property.holds_object(pins.members[member].as_ptr()))
    }
}

/// Establish [`RegexpProof`]'s slots and pins from scratch, marking %RegExp.prototype% and
/// %RegExp% as prototypes. `None` when the current Realm's RegExp machinery is not the
/// intrinsic one.
fn verify_regexp_proof(i: &Interp) -> Option<([u32; REGEXP_MEMBERS], Box<RegexpPins>)> {
    let prototype = *i.extra_protos.get("RegExp")?;
    let constructor = *i.extra_protos.get("%RegExpCtor%")?;
    let species_getter = *i.extra_protos.get("%RegExpSpeciesGetter%")?;
    if !i.ordinary_get_ptr(Gc::as_ptr(&prototype) as usize)
        || !i.ordinary_get_ptr(Gc::as_ptr(&constructor) as usize)
    {
        return None;
    }
    let in_this_realm = |value: &Value, behaviour: NativeFn| {
        is_native_function(value, behaviour)
            && value
                .as_obj()
                .is_some_and(|function| i.native_call_in_current_realm(&function))
    };
    let symbol = |name| well_known_key(i, name);
    let keys: [Rc<str>; REGEXP_MEMBERS] = [
        Rc::from("exec"),
        symbol("match")?,
        symbol("matchAll")?,
        symbol("replace")?,
        symbol("search")?,
        symbol("split")?,
        Rc::from("constructor"),
    ];
    let behaviours: [NativeFn; REGEXP_MEMBERS - 1] = [
        regexp_exec,
        re_sym_match,
        re_sym_matchall,
        re_sym_replace,
        re_sym_search,
        re_sym_split,
    ];
    let mut slots = [0u32; REGEXP_MEMBERS];
    let mut members: [crate::value::WeakGc; REGEXP_MEMBERS] = Default::default();
    {
        let body = prototype.borrow();
        for (name, behaviour) in REGEXP_FLAG_GETTERS {
            let property = body.props.get(name)?;
            if !property.accessor()
                || !property
                    .getter()
                    .is_some_and(|getter| in_this_realm(getter, behaviour))
            {
                return None;
            }
        }
        for (member, key) in keys.iter().enumerate() {
            let slot = body.props.slot_of(key)?;
            let property = body.props.property_at(slot)?;
            let value = property.value();
            let object = value.as_obj()?;
            let intrinsic = match behaviours.get(member) {
                Some(&behaviour) => in_this_realm(&value, behaviour),
                None => Gc::ptr_eq(&object, &constructor),
            };
            if property.accessor() || !intrinsic {
                return None;
            }
            slots[member] = u32::try_from(slot).ok()?;
            members[member] = Gc::downgrade(&object);
        }
    }
    {
        // SpeciesConstructor finds the intrinsic getter, which returns %RegExp%; constructing
        // with %RegExp% reads its non-writable, non-configurable `prototype`.
        let body = constructor.borrow();
        let species = body.props.get(&symbol("species")?)?;
        let species_intrinsic = species.setter().is_none()
            && matches!(
                species.getter().map(|getter| getter.view()),
                Some(ValueView::Obj(getter)) if Gc::ptr_eq(&getter, &species_getter)
            );
        let prototype_intrinsic = body
            .props
            .get("prototype")
            .is_some_and(|property| property.holds_object(Gc::as_ptr(&prototype)));
        if !species_intrinsic || !prototype_intrinsic {
            return None;
        }
    }
    prototype.borrow().props.mark_proto();
    constructor.borrow().props.mark_proto();
    let prototype = Gc::downgrade(&prototype);
    Some((slots, Box::new(RegexpPins { prototype, members })))
}

/// Run `f` with the current [`RegexpProof`], its pins and %RegExp.prototype%, re-establishing
/// the proof after an epoch change. `None` when the intrinsic machinery does not hold, and
/// always under `TRUST_JS_REGEXP_NO_PROTECTOR=1`, which keeps every RegExp operation on its
/// observable steps.
fn with_regexp_proof<R>(
    i: &Interp,
    f: impl FnOnce(&RegexpProof, &RegexpPins, &Gc) -> Option<R>,
) -> Option<R> {
    let epoch = crate::value::proto_epoch();
    // u32::MAX: permanently invalidated, never trust a proof.
    if epoch == u32::MAX || env_flag!("TRUST_JS_REGEXP_NO_PROTECTOR") {
        return None;
    }
    if i.regexp_protector.borrow().epoch != epoch {
        let verified = verify_regexp_proof(i);
        let mut proof = i.regexp_protector.borrow_mut();
        proof.epoch = epoch;
        (proof.slots, proof.pins) = match verified {
            Some((slots, pins)) => (slots, Some(pins)),
            None => ([0; REGEXP_MEMBERS], None),
        };
    }
    let proof = i.regexp_protector.borrow();
    let pins = proof.pins.as_deref()?;
    let prototype = pins.prototype.upgrade()?;
    f(&proof, pins, &prototype)
}

/// The object and matcher of `value` when the RegExp algorithms read nothing but intrinsics
/// through it: a RegExp of the running Realm with the shape RegExpAlloc gave it and
/// %RegExp.prototype% as its prototype, under the RegExp protector, with `members` of
/// %RegExp.prototype% still holding their intrinsics (see [`RegexpProof`]).
pub(super) fn intrinsic_regexp(
    i: &Interp,
    value: &Value,
    members: &[RegexpMember],
) -> Option<(Gc, Rc<crate::regex::Regex>)> {
    let object = value.as_obj()?;
    with_regexp_proof(i, |proof, pins, prototype| {
        if !intrinsic_receiver(proof, pins, prototype, &object, members) {
            return None;
        }
        let re = i.regexps.get(&(Gc::as_ptr(&object) as usize))?.clone();
        Some((object, re))
    })
}

/// Whether GetMethod of `member` on `value` finds %RegExp.prototype%'s intrinsic without
/// running code, under the RegExp protector: `value` has the shape RegExpAlloc gives a RegExp
/// (no own property but `lastIndex`) and %RegExp.prototype% as its prototype. It need not be a
/// RegExp: the method then checks what it reads, as it would after the lookup.
pub(super) fn dispatches_to_intrinsic(i: &Interp, value: &Value, member: RegexpMember) -> bool {
    value.as_obj().is_some_and(|object| {
        with_regexp_proof(i, |proof, pins, prototype| {
            intrinsic_receiver(proof, pins, prototype, &object, &[member]).then_some(())
        })
        .is_some()
    })
}

/// Whether `object` has RegExpAlloc's shape and `prototype` (%RegExp.prototype%) as its
/// prototype, and `members` still hold their intrinsics.
fn intrinsic_receiver(
    proof: &RegexpProof,
    pins: &RegexpPins,
    prototype: &Gc,
    object: &Gc,
    members: &[RegexpMember],
) -> bool {
    {
        let body = object.borrow();
        if body.props.shape() != proof.instance_shape
            || !crate::value::is_cacheable_shape(proof.instance_shape)
            || !body
                .proto
                .as_ref()
                .is_some_and(|p| Gc::ptr_eq(p, prototype))
        {
            return false;
        }
    }
    members
        .iter()
        .all(|&member| proof.holds(pins, prototype, member))
}

/// The slot of `lastIndex` in a RegExp of the shape RegExpAlloc gives it: its only own property.
const LAST_INDEX_SLOT: usize = 0;

/// The shape RegExpAlloc gives every RegExp, whose only own property is `lastIndex`, a writable
/// data property, at [`LAST_INDEX_SLOT`]; `None` before one is known, or under
/// `TRUST_JS_REGEXP_NO_PROTECTOR=1` (see [`with_regexp_proof`]).
fn alloc_shape(i: &Interp) -> Option<u32> {
    let shape = i.regexp_protector.borrow().instance_shape;
    (crate::value::is_cacheable_shape(shape) && !env_flag!("TRUST_JS_REGEXP_NO_PROTECTOR"))
        .then_some(shape)
}

/// Whether `object` has the shape RegExpAlloc gives every RegExp ([`alloc_shape`]).
pub(super) fn alloc_shaped(i: &Interp, object: &Gc) -> bool {
    alloc_shape(i).is_some_and(|shape| object.borrow().props.shape() == shape)
}

/// `lastIndex` of a RegExp of RegExpAlloc's shape ([`alloc_shaped`]) when ToLength of it runs no
/// code and writing it cannot throw: a writable data property holding a non-negative integral
/// Number.
pub(super) fn plain_last_index(object: &Gc) -> Option<f64> {
    plain_number(object.borrow().props.property_at(LAST_INDEX_SLOT)?)
}

/// [`plain_last_index`] of `object` when it is [`alloc_shaped`], checked under one borrow.
pub(super) fn slot_last_index(i: &Interp, object: &Gc) -> Option<f64> {
    let shape = alloc_shape(i)?;
    let body = object.borrow();
    if body.props.shape() != shape {
        return None;
    }
    plain_number(body.props.property_at(LAST_INDEX_SLOT)?)
}

fn plain_number(last: &Property) -> Option<f64> {
    if !last.writable() {
        return None;
    }
    last.number_value()
        .filter(|n| n.is_finite() && *n >= 0.0 && n.fract() == 0.0)
}

/// Write `lastIndex` of a RegExp of RegExpAlloc's shape at its slot: OrdinarySet of a writable
/// own data property, which runs no code and cannot fail.
pub(super) fn set_last_index_at_slot(object: &Gc, n: f64) {
    let mut body = object.borrow_mut();
    debug_assert!(body
        .props
        .property_at(LAST_INDEX_SLOT)
        .is_some_and(|last| last.writable() && !last.accessor()));
    body.props.set_value_at(LAST_INDEX_SLOT, Value::Num(n));
}

/// The running Realm's %RegExp%, whose legacy static properties a successful match updates.
/// The RegExp protector's pins, whatever the epoch, are the running Realm's (another Realm's
/// entry invalidates them), and its `constructor` member was %RegExp%. Without pins the proof is
/// established, and where it fails the intrinsic is looked up.
pub(super) fn realm_regexp_constructor(i: &Interp) -> Option<Gc> {
    let constructor =
        |pins: &RegexpPins| pins.members[RegexpMember::Constructor as usize].upgrade();
    let pinned = i
        .regexp_protector
        .borrow()
        .pins
        .as_deref()
        .and_then(constructor);
    pinned
        .or_else(|| with_regexp_proof(i, |_, pins, _| constructor(pins)))
        .or_else(|| i.extra_protos.get("%RegExpCtor%").copied())
}

/// Read the engine-owned flags of a RegExp without dispatching the `flags` getter, when the
/// observable `RegExp.prototype.flags` string is provably canonical (see [`intrinsic_regexp`]).
/// `None` keeps the generic getter with its observable Gets (a user-overridden `flags`, a
/// subclass prototype, or an own flag property all force it).
fn direct_reg_exp_flags(i: &Interp, this: &Value) -> Option<std::rc::Rc<crate::regex::Regex>> {
    intrinsic_regexp(i, this, &[]).map(|(_, re)| re)
}

/// The object and matcher of `this` when RegExpExec(this, S) is RegExpBuiltinExec without
/// observable lookups (ECMA-262 §22.2.7.1): an intrinsic RegExp ([`intrinsic_regexp`]) whose
/// `exec` is the intrinsic, with a plain `lastIndex` ([`plain_last_index`]), so neither reading
/// it (ToLength) nor writing it runs user code or throws.
fn direct_regexp_exec(i: &Interp, this: &Value) -> Option<(Gc, std::rc::Rc<crate::regex::Regex>)> {
    let (object, re) = intrinsic_regexp(i, this, &[RegexpMember::Exec])?;
    plain_last_index(&object)?;
    Some((object, re))
}

/// Whether the current Realm's RegExp protector holds with the intrinsic `exec` (for tests).
#[cfg(test)]
pub(super) fn regexp_protector_holds(i: &Interp) -> bool {
    with_regexp_proof(i, |proof, pins, prototype| {
        proof
            .holds(pins, prototype, RegexpMember::Exec)
            .then_some(())
    })
    .is_some()
}

/// The shape the RegExp protector expects of a RegExp RegExpAlloc made (for tests).
#[cfg(test)]
pub(super) fn regexp_instance_shape(i: &Interp) -> u32 {
    i.regexp_protector.borrow().instance_shape
}

/// The second (flags) argument to RegExp / RegExp.prototype.compile: undefined → "", else ToString.
fn regexp_flags_arg(i: &mut Interp, a: &[Value]) -> Result<String, Value> {
    {
        let __value_match_original = arg(a, 1);
        match (__value_match_original).view() {
            ValueView::Undefined => Ok(String::new()),
            _ => {
                let v = __value_match_original;
                Ok(ab(i.to_string(&v))?.to_string())
            }
        }
    }
}

pub(super) fn install_regexp(it: &mut Interp) {
    let proto = Object::new(Some(it.object_proto));
    Rc::make_mut(&mut it.extra_protos).insert("RegExp", proto);
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
        if let (ValueView::Obj(o), Some(rp)) = (&(this).view(), i.extra_protos.get("RegExp")) {
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
        let (source, flags) = {
            let __value_match_original = arg(a, 0);
            match (__value_match_original).view() {
                ValueView::Obj(o) if i.regexps.contains_key(&(Gc::as_ptr(&o) as usize)) => {
                    // A RegExp pattern copies its source/flags; a second flags argument is then an error.
                    if !matches!((arg(a, 1)).view(), ValueView::Undefined) {
                        return Err(i.make_error(
                            "TypeError",
                            "cannot supply flags when compiling a RegExp",
                        ));
                    }
                    let re = i.regexps[&(Gc::as_ptr(&o) as usize)].clone();
                    (re.source.clone(), re.flags.clone())
                }
                ValueView::Undefined => (String::new(), regexp_flags_arg(i, a)?),
                _ => {
                    let v = __value_match_original;
                    {
                        let src = ab(i.to_string(&v))?.to_string();
                        (src, regexp_flags_arg(i, a)?)
                    }
                }
            }
        };
        let re = crate::regex::Regex::new(&source, &flags)
            .map_err(|e| i.make_error("SyntaxError", &e))?;
        if let ValueView::Obj(o) = &(this).view() {
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
        if !matches!((this).view(), ValueView::Obj(_)) {
            return Err(i.make_error("TypeError", "RegExp.prototype.test requires an object"));
        }
        let s = ab(i.to_string(&arg(a, 0)))?;
        // With the intrinsic `exec`, run RegExpBuiltinExec's matcher, `lastIndex` and legacy
        // static effects without building the match array `test` would discard.
        if let Some((obj, re)) = direct_regexp_exec(i, &this) {
            return regexp_exec_discard_direct(i, &obj, &re, &s).map(Value::Bool);
        }
        Ok(Value::Bool(!matches!(
            (regexp_exec_abstract(i, &this, s)?).view(),
            ValueView::Null
        )))
    });
    it.def_method(&proto, "toString", 0, |i, this, _| {
        if !matches!((this).view(), ValueView::Obj(_)) {
            return Err(i.make_error("TypeError", "RegExp.prototype.toString requires an object"));
        }
        let src_v = ab(i.get_member_key(&this, "source"))?;
        let src = ab(i.to_string(&src_v))?;
        let flags_v = ab(i.get_member_key(&this, "flags"))?;
        let flags = ab(i.to_string(&flags_v))?;
        // RegExp.prototype.toString performs the two observable Gets/ToStrings above, then
        // concatenates the four pieces directly (ECMA-262 §22.2.6.17).
        Ok(Value::Str(crate::lstr::LStr::concat4(
            "/",
            &src.text(),
            "/",
            &flags.text(),
        )))
    });
    let ctor = it.make_native("RegExp", 2, |i, _t, a| {
        let pattern = arg(a, 0);
        let flags_arg = arg(a, 1);
        // IsRegExp: an Object whose truthy @@match (when defined) or [[RegExpMatcher]] slot.
        let has_slots = matches!(&(pattern).view(), ValueView::Obj(o)
            if i.regexps.contains_key(&(Gc::as_ptr(o) as usize)));
        let pattern_is_regexp = if matches!((pattern).view(), ValueView::Obj(_)) {
            let m = match well_known_key(i, "match") {
                Some(key) => ab(i.get_member_key(&pattern, &key))?,
                None => Value::Undefined,
            };
            if !matches!((m).view(), ValueView::Undefined) {
                i.to_boolean(&m)
            } else {
                has_slots
            }
        } else {
            false
        };
        // Called as a function on a regexp(-like) whose .constructor is this same
        // constructor: return the argument unchanged.
        if !i.constructing
            && pattern_is_regexp
            && matches!((flags_arg).view(), ValueView::Undefined)
        {
            let c = ab(i.get_member_key(&pattern, "constructor"))?;
            let same = match (&(c).view(), i.extra_protos.get("%RegExpCtor%")) {
                (ValueView::Obj(co), Some(rc)) => Gc::ptr_eq(co, rc),
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
            let re = match &(pattern).view() {
                ValueView::Obj(o) => i.regexps[&(Gc::as_ptr(o) as usize)].clone(),
                _ => unreachable!(),
            };
            let fl = {
                let __value_match_original = flags_arg;
                match (__value_match_original).view() {
                    ValueView::Undefined => Value::lstr(re.flags.clone()),
                    _ => __value_match_original,
                }
            };
            (Value::lstr(re.source.clone()), fl)
        } else if pattern_is_regexp {
            // A regexp-like object: read its `source` and `flags` properties.
            let src = ab(i.get_member_key(&pattern, "source"))?;
            let fl = {
                let __value_match_original = flags_arg;
                match (__value_match_original).view() {
                    ValueView::Undefined => ab(i.get_member_key(&pattern, "flags"))?,
                    _ => __value_match_original,
                }
            };
            (src, fl)
        } else {
            (pattern, flags_arg)
        };
        let alloc_proto = ab(i.regexp_alloc_proto())?;
        let source = match src_v {
            v if matches!(v.view(), ValueView::Undefined) => String::new(),
            v => ab(i.to_string(&v))?.to_string(),
        };
        let flags = match fl_v {
            v if matches!(v.view(), ValueView::Undefined) => String::new(),
            v => ab(i.to_string(&v))?.to_string(),
        };
        ab(i.make_regexp_with_proto(&source, &flags, alloc_proto))
    });
    Rc::make_mut(&mut it.extra_protos).insert("%RegExpCtor%", ctor);
    install_regexp_legacy_statics(it, &ctor);
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor)));
    install_species(it, &ctor);
    // The split paths prove the splitter's species lookup inert by this getter's identity: its
    // trivial native body may share an address with other functions, so it cannot be recognized
    // by function pointer.
    let species_getter = well_known_key(it, "species").and_then(|key| {
        match (ctor.borrow().props.get(&key).and_then(Property::getter))
            .map(|__value| (__value).view())
        {
            Some(ValueView::Obj(getter)) => Some(getter),
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
        let s = match (arg(a, 0)).view() {
            ValueView::Str(s) => s,
            _ => return Err(i.make_error("TypeError", "RegExp.escape requires a string")),
        };
        let mut out = String::new();
        for (idx, cp) in crate::jstr::code_points(&s.text()).into_iter().enumerate() {
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
    match (&(this).view(), ctor) {
        (ValueView::Obj(o), Some(c)) if Gc::ptr_eq(o, &c) => Ok(c),
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
    match &(this).view() {
        ValueView::Obj(_) => Ok(()),
        _ => Err(i.make_error("TypeError", format!("{name} called on a non-object"))),
    }
}

pub(super) fn re_sym_match(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.match]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    // An intrinsic RegExp's `flags` are its own and RegExpExec is RegExpBuiltinExec: without `g`
    // the match is one exec, by the matcher just read.
    if let Some((_, re)) = intrinsic_regexp(i, &this, &[RegexpMember::Exec]) {
        if !re.global {
            return regexp_builtin_exec(i, this, s, Some(re));
        }
    }
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
        let result = regexp_exec_abstract(i, &this, s)?;
        if matches!((result).view(), ValueView::Null) {
            break;
        }
        let m0 = ab(i.get_member_key(&result, "0"))?;
        let match_str = ab(i.to_string(&m0))?;
        results.push(Value::Str(match_str));
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
        set_last_index_at_slot(obj, 0.0);
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
            let li = plain_last_index(obj).expect("direct match lastIndex guard") as usize;
            let next = advance_string_index(li, input, unicode);
            set_last_index_at_slot(obj, next as f64);
        }
    }
    Ok(Value::Undefined)
}
pub(super) fn re_sym_search(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
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
    match (result).view() {
        ValueView::Null => Ok(Value::Num(-1.0)),
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
        set_last_index_at_slot(obj, 0.0);
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
            let li = plain_last_index(obj).expect("direct replace lastIndex guard") as usize;
            let next = advance_string_index(li, input, unicode);
            set_last_index_at_slot(obj, next as f64);
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
        if let ValueView::Obj(obj) = &(this).view() {
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
                        let ValueView::Obj(exec) = (value).view() else {
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
                        let li = match (obj.borrow().props.get("lastIndex").map(Property::value))
                            .map(|__value| (__value).view())
                        {
                            Some(ValueView::Num(n)) => n as usize,
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
        let result = regexp_exec_abstract(i, &this, s)?;
        if matches!((result).view(), ValueView::Null) {
            break;
        }
        results.push(result);
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
        let match_len = matched.unit_len();
        let index_v = ab(i.get_member_key(result, "index"))?;
        let pos_raw = ab(i.to_number(&index_v))?;
        let position = if pos_raw.is_nan() || pos_raw < 0.0 {
            0
        } else {
            (pos_raw as usize).min(size)
        };
        // Rooted: each later capture's Get and ToString may run user code.
        let mut captures = crate::rooting::RootedVec::new();
        for n in 1..=ncaptures {
            let cap = ab(i.get_member_key(result, &n.to_string()))?;
            captures.push(if matches!((cap).view(), ValueView::Undefined) {
                Value::Undefined
            } else {
                Value::Str(ab(i.to_string(&cap))?)
            });
        }
        let mut named = ab(i.get_member_key(result, "groups"))?;
        // Non-functional replace ToObjects a present `groups` (null → TypeError).
        if !functional
            && !matches!((named).view(), ValueView::Undefined)
            && !matches!((named).view(), ValueView::Obj(_))
        {
            match named.view() {
                ValueView::Null => {
                    return Err(i.make_error("TypeError", "cannot convert null to object"));
                }
                _ => named = box_primitive(i, named),
            }
        }
        let replacement = if functional {
            let mut cbargs = vec![Value::Str(matched)];
            cbargs.extend(captures.iter().cloned());
            cbargs.push(Value::Num(position as f64));
            cbargs.push(Value::Str(s));
            if !matches!((named).view(), ValueView::Undefined) {
                cbargs.push(named);
            }
            let r = ab(i.call(repl, Value::Undefined, &cbargs))?;
            ab(i.to_string(&r))?.to_string()
        } else {
            get_substitution(
                i,
                &matched.text(),
                &sunits,
                position,
                &captures,
                &named,
                &repl_str.text(),
            )?
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
    Ok(Value::lstr(accumulated))
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
    // `lastIndex` stays at its slot: no code runs between the guard and the last write.
    let read_last =
        |obj: &Gc| plain_last_index(obj).expect("direct replace lastIndex guard") as usize;
    let set_last = |obj: &Gc, n: usize| set_last_index_at_slot(obj, n as f64);
    let size = s.unit_len();
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
        let matched = regexp_exec_text_shared(i, re, s, last);
        let Some(caps) = regexp_match_result(i, matched)? else {
            if uses_last {
                set_last(obj, 0);
            }
            break;
        };
        let (start, end) = caps[0].expect("successful regexp match has a span");
        if uses_last {
            set_last(obj, end);
        }
        update_regexp_legacy_statics(i, re, &caps, s);
        matches.push(caps);
        if !re.global {
            break;
        }
        if start == end {
            set_last(obj, advance_string_index(end, s, re.unicode));
        }
    }
    if matches.is_empty() {
        return Ok(Value::Str(*s));
    }
    let substitutes = replacer.is_none() && template.text().contains('$');
    let sunits = if substitutes {
        i.units_full(s)
    } else {
        std::borrow::Cow::Borrowed(&[][..])
    };
    let mut out = crate::lstr::Builder::with_capacity(s.unit_len());
    let mut next = 0usize;
    for caps in &matches {
        let (start, end) = caps[0].expect("successful regexp match has a span");
        let position = start;
        let captures = || {
            (1..=re.ngroups)
                .map(|group| match caps[group] {
                    Some((a, b)) => Value::Str(s.slice_units(a, b)),
                    None => Value::Undefined,
                })
                .collect::<Vec<_>>()
        };
        let replacement: crate::lstr::LStr = if let Some(function) = replacer {
            let mut args = Vec::with_capacity(re.ngroups + 4);
            args.push(Value::Str(s.slice_units(start, end)));
            args.extend(captures());
            args.push(Value::Num(position as f64));
            args.push(Value::Str(*s));
            let groups = regexp_groups(i, re, caps, s);
            if !matches!((groups).view(), ValueView::Undefined) {
                args.push(groups);
            }
            let result = ab(i.call(*function, Value::Undefined, &args))?;
            ab(i.to_string(&result))?
        } else if substitutes {
            let captures = captures();
            let named = regexp_groups(i, re, caps, s);
            let matched = subject_text(s, start, end);
            get_substitution(
                i,
                &matched,
                &sunits,
                position,
                &captures,
                &named,
                &template.text(),
            )?
            .into()
        } else {
            *template
        };
        if start >= next {
            out.push_units(s.units().slice(next, start));
            out.push_units(replacement.units());
            // Pathological expansion ($1 of a huge match, repeated) dies as a RangeError.
            if out.len() > MAX_STR_LEN {
                return Err(i.make_error("RangeError", "Invalid string length"));
            }
            next = end;
        }
    }
    out.push_units(s.units().slice(next, s.unit_len()));
    Ok(Value::Str(out.finish()))
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
        if let ValueView::Str(s) = (cap).view() {
            out.push_str(&s.text());
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
            '<' if !matches!(&(named).view(), ValueView::Undefined) => {
                if let Some(rel) = t[k + 2..].iter().position(|&c| c == '>') {
                    let name: String = t[k + 2..k + 2 + rel].iter().collect();
                    let v = ab(i.get_member_key(named, &name))?;
                    if !matches!((v).view(), ValueView::Undefined) {
                        out.push_str(&ab(i.to_string(&v))?.text());
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

pub(super) fn re_sym_matchall(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_regexp_this(i, &this, "[Symbol.matchAll]")?;
    let s = ab(i.to_string(&arg(a, 0)))?;
    let default_ctor = regexp_intrinsic(i);
    let c = species_constructor(i, &this, &default_ctor)?;
    let flags_v = ab(i.get_member_key(&this, "flags"))?;
    let flags = ab(i.to_string(&flags_v))?;
    let matcher = ab(i.construct(c, &[this, Value::Str(flags)]))?;
    let li_v = ab(i.get_member_key(&this, "lastIndex"))?;
    let last = to_length_val(i, &li_v)?;
    ab(i.set_member_key(&matcher, "lastIndex", Value::Num(last as f64)))?;
    let global = flags.text().contains('g');
    let unicode = flags.text().contains('u') || flags.text().contains('v');
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
/// intrinsic getters, and constructing the splitter reads `rx`'s @@match (IsRegExp; a data
/// property under the protector) and %RegExp%.prototype without running user code. The
/// splitter's own RegExpExec then finds the intrinsic `exec`. `None` when any of them could run
/// user code.
fn split_species_canonical(i: &Interp, obj: &Gc) -> Option<Rc<crate::regex::Regex>> {
    let members = [RegexpMember::Constructor, RegexpMember::Exec];
    intrinsic_regexp(i, &Value::Obj(*obj), &members).map(|(_, re)| re)
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
    let ValueView::Obj(obj) = &(rx).view() else {
        return None;
    };
    let lim = match &(limit).view() {
        ValueView::Undefined => u32::MAX,
        ValueView::Num(n) => to_uint32(*n),
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
    let len = s.unit_len();
    if len == 0 {
        let matched = regexp_exec_text_shared(i, &search, s, 0);
        if let Some(caps) = regexp_match_result(i, matched)? {
            update_regexp_legacy_statics(i, &search, &caps, s);
        } else {
            parts.push(Value::Str(*s));
        }
        return Ok(i.make_array(parts));
    }
    // Unit positions: `p` ends the last piece, `q` is where the next attempt starts.
    let (mut p, mut q) = (0usize, 0usize);
    while q < len {
        if let Err(error) = regexp_interrupt_tick(i) {
            return Err(regexp_match_error_value(i, error));
        }
        let matched = regexp_exec_text_shared(i, &search, s, q);
        let Some(caps) = regexp_match_result(i, matched)? else {
            break;
        };
        let (a, e) = caps[0].expect("successful regexp match has a span");
        // The generic loop stops before attempting a match at the end of the subject.
        if a >= len {
            break;
        }
        update_regexp_legacy_statics(i, &search, &caps, s);
        if e == p {
            // AdvanceStringIndex past an empty match at the last split point.
            q = advance_string_index(a, s, search.unicode);
            continue;
        }
        parts.push(Value::Str(s.slice_units(p, a)));
        if parts.len() == lim {
            return Ok(i.make_array(parts));
        }
        p = e;
        for group in 1..=search.ngroups {
            parts.push(match caps[group] {
                Some((x, y)) => Value::Str(s.slice_units(x, y)),
                None => Value::Undefined,
            });
            if parts.len() == lim {
                return Ok(i.make_array(parts));
            }
        }
        q = p;
    }
    parts.push(Value::Str(s.slice_units(p, len)));
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
    let (ValueView::Str(input), ValueView::Obj(obj), ValueView::Undefined) =
        (&(input).view(), &(separator).view(), &(limit).view())
    else {
        return None;
    };
    // String#split's own @@split lookup must find the intrinsic method too.
    let members = [
        RegexpMember::Split,
        RegexpMember::Constructor,
        RegexpMember::Exec,
    ];
    let (_, re) = intrinsic_regexp(i, &Value::Obj(*obj), &members)?;
    // A sticky split loop that advances one unit after every miss is equivalent to one ordinary
    // forward search for its next success. Removing `y` performs that search in the matcher while
    // preserving the successful captures and their legacy-static side effects. The AArch64 class
    // machine mapping is deliberately declined by the regex tier, so this remains on the checked
    // matcher for the class-heavy cases that previously exposed the machine-code issue.
    let search_re = match split_search_regexp(i, re) {
        Ok(search) => search,
        Err(error) => return Some(Err(error)),
    };
    let size = input.unit_len();
    if size == 0 {
        let matched = regexp_find_text_shared(i, &search_re, input, 0);
        let found = match regexp_match_result(i, matched) {
            Ok(found) => found,
            Err(error) => return Some(Err(error)),
        };
        if let Some(whole) = found {
            update_regexp_legacy_statics_lazy(i, &search_re, whole, 0, input);
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
        let matched = regexp_find_text_shared(i, &search_re, input, q);
        let found = match regexp_match_result(i, matched) {
            Ok(found) => found,
            Err(error) => return Some(Err(error)),
        };
        let Some(whole @ (a, e)) = found else {
            break;
        };
        // RegExp@@split's sticky loop stops before attempting a match at the final position.
        if a >= size {
            break;
        }
        update_regexp_legacy_statics_lazy(i, &search_re, whole, q, input);
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
    let sunits: Vec<u16> = s.units().to_utf16().into_owned();
    let size = sunits.len();
    // SpeciesConstructor(rx, %RegExp%), then a sticky-flagged splitter constructed from it.
    let default_ctor = regexp_intrinsic(i);
    let c = species_constructor(i, &this, &default_ctor)?;
    let flags_v = ab(i.get_member_key(&this, "flags"))?;
    let flags = ab(i.to_string(&flags_v))?;
    let unicode = flags.text().contains('u') || flags.text().contains('v');
    let new_flags = if flags.text().contains('y') {
        flags.to_string()
    } else {
        format!("{flags}y")
    };
    let splitter = ab(i.construct(c, &[this, Value::lstr(new_flags)]))?;
    // ToUint32 wraps modulo 2^32 over the whole Number range: -1 is 2^32 - 1, 2^32 is 0.
    let limit = {
        let __value_match_original = arg(a, 1);
        match (__value_match_original).view() {
            ValueView::Undefined => u32::MAX as usize,
            _ => {
                let v = __value_match_original;
                to_uint32(ab(i.to_number(&v))?) as usize
            }
        }
    };
    // Rooted across the exec calls and captures' getters (I1).
    let mut out = crate::rooting::RootedVec::new();
    if limit == 0 {
        return Ok(i.make_array(out.into_vec()));
    }
    if size == 0 {
        let z = regexp_exec_abstract(i, &splitter, s)?;
        if !matches!((z).view(), ValueView::Null) {
            return Ok(i.make_array(out.into_vec()));
        }
        out.push(Value::Str(s));
        return Ok(i.make_array(out.into_vec()));
    }
    let mut p = 0usize;
    let mut q = 0usize;
    while q < size {
        ab(i.set_member_key(&splitter, "lastIndex", Value::Num(q as f64)))?;
        let z = regexp_exec_abstract(i, &splitter, s)?;
        if matches!((z).view(), ValueView::Null) {
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
