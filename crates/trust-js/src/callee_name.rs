//! Names for the targets of calls and `new` in TypeError messages.
//!
//! ECMA-262 only requires a TypeError when a call target is not callable (Call, step 2) or a
//! `new` target is not a constructor (EvaluateNew, step 7); the message is implementation-defined.
//! Like V8, describe the target by the shape of its source expression (`x.foo`, `foo`,
//! `obj.method(...).then`, `a[i]`) so a page's error says which call failed. The tree-walker
//! names its callee expression directly; compiled code keeps these names in a side table that is
//! consulted only when such an error is thrown (see `bytecode::Chunk::call_name`).

use crate::ast::Expr;

/// Longest name kept; longer chains fall back to a shorter form.
const MAX_LEN: usize = 96;
/// Nesting kept before an inner expression is shown as `(intermediate value)`.
const MAX_DEPTH: usize = 8;
const INTERMEDIATE: &str = "(intermediate value)";

/// The TypeError message for calling (`construct` false) or constructing the value of `callee`.
pub(crate) fn not_callable_message(callee: &Expr, construct: bool) -> String {
    named_message(&callee_name(callee), construct)
}

/// `"{name} is not a function"` or `"{name} is not a constructor"`.
pub(crate) fn named_message(name: &str, construct: bool) -> String {
    if construct {
        format!("{name} is not a constructor")
    } else {
        format!("{name} is not a function")
    }
}

/// Whether `message` is one of the unnamed messages call dispatch gives a target that is not
/// callable (or, for `construct`, not a constructor).
pub(crate) fn is_unnamed_message(message: &str, construct: bool) -> bool {
    let suffix = if construct {
        " is not a constructor"
    } else {
        " is not a function"
    };
    message.strip_suffix(suffix).is_some_and(|subject| {
        matches!(
            subject,
            "undefined"
                | "null"
                | "boolean"
                | "number"
                | "bigint"
                | "string"
                | "symbol"
                | "object"
                | "value"
        ) || (construct && matches!(subject, "function" | "this function"))
    })
}

/// A short description of `callee`, such as `x.foo`, `foo` or `obj.method(...).then`.
pub(crate) fn callee_name(callee: &Expr) -> String {
    let mut out = String::new();
    if !write(callee, &mut out, 0) || out.len() > MAX_LEN {
        out.clear();
        out.push_str(INTERMEDIATE);
        // Keep the final property of an overlong chain: `(intermediate value).then`.
        if let Some(last) = last_member(callee) {
            out.push_str(&last);
        }
    }
    out
}

/// The trailing `.name` or `[key]` of a member chain.
fn last_member(callee: &Expr) -> Option<String> {
    match callee {
        Expr::Paren(inner) | Expr::OptionalChain(inner) => last_member(inner),
        Expr::Member { prop, optional, .. } if prop.len() <= MAX_LEN / 2 => {
            Some(format!("{}{prop}", if *optional { "?." } else { "." }))
        }
        _ => None,
    }
}

/// Append the description of `expr`; false when it has no short description.
fn write(expr: &Expr, out: &mut String, depth: usize) -> bool {
    if depth > MAX_DEPTH || out.len() > MAX_LEN {
        return false;
    }
    match expr {
        Expr::Paren(inner) | Expr::OptionalChain(inner) => write(inner, out, depth),
        Expr::Ident(name, _) => {
            out.push_str(name);
            true
        }
        Expr::This => {
            out.push_str("this");
            true
        }
        Expr::Super => {
            out.push_str("super");
            true
        }
        Expr::Member {
            obj,
            prop,
            optional,
            ..
        } => {
            write_base(obj, out, depth);
            out.push_str(if *optional { "?." } else { "." });
            out.push_str(prop);
            true
        }
        Expr::Index {
            obj,
            index,
            optional,
            ..
        } => {
            write_base(obj, out, depth);
            match &**index {
                // `o["name"]` reads like `o.name`, as V8 prints it.
                Expr::Str(key) if is_identifier_name(key) => {
                    out.push_str(if *optional { "?." } else { "." });
                    out.push_str(key);
                }
                _ => {
                    out.push_str(if *optional { "?.[" } else { "[" });
                    write_key(index, out, depth);
                    out.push(']');
                }
            }
            true
        }
        Expr::Call { callee, .. } => {
            write_base(callee, out, depth);
            out.push_str("(...)");
            true
        }
        _ => false,
    }
}

/// The object of a member access or the callee of a call: an expression without a short
/// description is `(intermediate value)`, as in `(intermediate value).x`.
fn write_base(expr: &Expr, out: &mut String, depth: usize) {
    let start = out.len();
    if !write(expr, out, depth + 1) {
        out.truncate(start);
        out.push_str(INTERMEDIATE);
    }
}

/// An ASCII IdentifierName short enough to show.
fn is_identifier_name(key: &str) -> bool {
    key.len() <= MAX_LEN / 2
        && key
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// A computed key: identifiers, member chains (`Symbol.iterator`), integers and short strings
/// are shown; anything else is `...`.
fn write_key(key: &Expr, out: &mut String, depth: usize) {
    let start = out.len();
    let shown = match key {
        Expr::Num(n) if n.fract() == 0.0 && n.abs() < 1e15 => {
            out.push_str(&format!("{}", *n as i64));
            true
        }
        Expr::Str(text)
            if text.chars().count() <= 24
                && !text
                    .chars()
                    .any(|c| c == '"' || c == '\\' || c.is_control()) =>
        {
            out.push('"');
            out.push_str(text);
            out.push('"');
            true
        }
        Expr::Ident(_, _) | Expr::This | Expr::Member { .. } | Expr::Index { .. } => {
            write(key, out, depth + 1)
        }
        _ => false,
    };
    if !shown {
        out.truncate(start);
        out.push_str("...");
    }
}

#[cfg(test)]
mod tests {
    use super::callee_name;

    fn name_of(source: &str) -> String {
        let Ok(body) = crate::parser::parse_script(source, false) else {
            panic!("{source}: parses");
        };
        let Some(crate::ast::Stmt::Expr(expr)) = body.first() else {
            panic!("{source}: expression statement");
        };
        let callee = match expr {
            crate::ast::Expr::Call { callee, .. } | crate::ast::Expr::New { callee, .. } => callee,
            crate::ast::Expr::OptionalChain(inner) => match &**inner {
                crate::ast::Expr::Call { callee, .. } => callee,
                _ => panic!("{source}: optional call"),
            },
            _ => panic!("{source}: call or new"),
        };
        callee_name(callee)
    }

    #[test]
    fn names_follow_the_callee_expression() {
        for (source, expected) in [
            ("foo()", "foo"),
            ("x.foo()", "x.foo"),
            ("obj.method(1, 2).then()", "obj.method(...).then"),
            ("a[0]()", "a[0]"),
            ("a[i]()", "a[i]"),
            ("o[Symbol.iterator]()", "o[Symbol.iterator]"),
            ("o['key']()", "o.key"),
            ("o['two words']()", "o[\"two words\"]"),
            ("o[f()]()", "o[...]"),
            ("this.x()", "this.x"),
            ("(o.f)()", "o.f"),
            ("o?.x()", "o?.x"),
            ("o.a?.()", "o.a"),
            ("x.y(1)(2)", "x.y(...)"),
            ("(0, f)()", "(intermediate value)"),
            ("(function () {})()()", "(intermediate value)(...)"),
            ("(a || b).c()", "(intermediate value).c"),
            ("new o.C()", "o.C"),
            ("new C()", "C"),
            ("arr[0].f()", "arr[0].f"),
        ] {
            assert_eq!(name_of(source), expected, "{source}");
        }
        let long = format!("{}.then()", "abcdefghij.".repeat(12) + "z");
        assert_eq!(name_of(&long), "(intermediate value).then");
    }
}
