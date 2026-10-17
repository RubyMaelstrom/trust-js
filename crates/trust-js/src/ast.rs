//! The abstract syntax tree. Deliberately small: one `Stmt` enum and one `Expr` enum, with shared
//! sub-structures for functions and patterns. The interpreter walks this tree directly.

use std::rc::Rc;

pub type P<T> = Box<T>;

/// Optimization identity owned by one immutable loop syntax node. Cold sites allocate nothing.
/// A deep AST clone is a new site; Rc<Function> sharing naturally preserves the original site.
/// Snapshot codecs deliberately omit this execution-only state.
#[derive(Debug, Default)]
pub struct LoopSite {
    token: std::cell::OnceCell<Rc<LoopSiteToken>>,
}

#[derive(Debug, Default)]
pub(crate) struct LoopSiteToken {
    _identity: u8,
}

impl Clone for LoopSite {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl LoopSite {
    pub(crate) fn token(&self) -> &Rc<LoopSiteToken> {
        self.token.get_or_init(|| Rc::new(LoopSiteToken::default()))
    }

    pub(crate) fn retained_token(&self) -> Option<&Rc<LoopSiteToken>> {
        self.token.get()
    }
}

/// A loop already owns a boxed body. Store its lazy optimization cell in that allocation,
/// rather than enlarging every Stmt (including non-loop statements) by one pointer.
#[derive(Debug, Clone)]
pub struct LoopBody {
    pub site: LoopSite,
    pub statement: Stmt,
}

impl LoopBody {
    pub fn new(statement: Stmt) -> Self {
        Self {
            site: LoopSite::default(),
            statement,
        }
    }
}

impl std::ops::Deref for LoopBody {
    type Target = Stmt;
    fn deref(&self) -> &Stmt {
        &self.statement
    }
}

impl std::ops::DerefMut for LoopBody {
    fn deref_mut(&mut self) -> &mut Stmt {
        &mut self.statement
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Expr(Expr),
    /// `var` / `let` / `const` declaration: kind + (target, optional initializer) pairs.
    VarDecl {
        kind: DeclKind,
        decls: Vec<(Pattern, Option<Expr>)>,
    },
    FuncDecl(Rc<Function>),
    Return(Option<Expr>),
    If {
        test: Expr,
        cons: P<Stmt>,
        alt: Option<P<Stmt>>,
    },
    Block(Vec<Stmt>),
    While {
        test: Expr,
        body: P<LoopBody>,
    },
    DoWhile {
        body: P<LoopBody>,
        test: Expr,
    },
    /// C-style `for (init; test; update) body`.
    For {
        init: Option<P<ForInit>>,
        test: Option<Expr>,
        update: Option<Expr>,
        body: P<LoopBody>,
    },
    /// `for (left in right) body` / `for (left of right) body` (`is_await` for `for await … of`).
    ForInOf {
        decl: Option<DeclKind>,
        left: Pattern,
        right: Expr,
        of: bool,
        is_await: bool,
        body: P<LoopBody>,
    },
    Break(Option<String>),
    Continue(Option<String>),
    Throw(Expr),
    Try {
        block: Vec<Stmt>,
        handler: Option<(Option<Pattern>, Vec<Stmt>)>,
        finalizer: Option<Vec<Stmt>>,
    },
    Switch {
        disc: Expr,
        cases: Vec<SwitchCase>,
    },
    Labeled {
        label: String,
        body: P<Stmt>,
    },
    /// `with (obj) body` — resolves identifiers against `obj` first (forbidden in strict mode).
    With {
        obj: Expr,
        body: P<Stmt>,
    },
    ClassDecl(Rc<Class>),
    Empty,
    Debugger,
    /// `import …from "spec"` (or a bare `import "spec"`).
    Import(ImportDecl),
    /// `export { a, b as c }` or `export { a } from "spec"`.
    ExportNamed {
        specs: Vec<ExportSpec>,
        source: Option<Rc<str>>,
    },
    /// `export const/let/var/function/class …` — the inner declaration plus its exported names.
    ExportDecl(P<Stmt>),
    /// `export default …` (expression, function, or class).
    ExportDefault(P<Stmt>),
    /// `export * from "spec"` or `export * as ns from "spec"`.
    ExportAll {
        source: Rc<str>,
        exported: Option<String>,
    },
}

impl Stmt {
    pub(crate) fn loop_body(&self) -> Option<&LoopBody> {
        match self {
            Self::While { body, .. }
            | Self::DoWhile { body, .. }
            | Self::For { body, .. }
            | Self::ForInOf { body, .. } => Some(body),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImportDecl {
    pub source: Rc<str>,
    pub specs: Vec<ImportSpec>,
    /// The `with { type: "..." }` import attribute (json/text/bytes), if present.
    pub attr_type: Option<String>,
}
#[derive(Debug, Clone)]
pub enum ImportSpec {
    /// `import x from "…"`
    Default(String),
    /// `import * as ns from "…"`
    Namespace(String),
    /// `import defer * as ns from "…"` — evaluation deferred until the namespace is accessed.
    DeferNamespace(String),
    /// `import source x from "…"` — a source-phase import binding the module's ModuleSource.
    Source(String),
    /// `import { imported as local } from "…"`
    Named { imported: String, local: String },
}
#[derive(Debug, Clone)]
pub struct ExportSpec {
    pub local: String,
    pub exported: String,
}

#[derive(Debug, Clone)]
pub struct Class {
    pub name: Option<String>,
    pub superclass: Option<P<Expr>>,
    pub members: Vec<ClassMember>,
    /// `@dec` decorators applied to the whole class (outermost last).
    pub decorators: Vec<Expr>,
    /// The class's source text (what the constructor's `toString` returns).
    pub source: Option<Rc<str>>,
    /// The source the class was parsed from and the byte offset of its first token, which a
    /// synthesized default constructor's stack frames report (see [`Function::script`]).
    pub(crate) script: Option<Rc<crate::stack_trace::ScriptSource>>,
    pub(crate) start: u32,
}

#[derive(Debug, Clone)]
pub struct ClassMember {
    pub key: PropKey,
    pub kind: MemberKind,
    pub is_static: bool,
    /// For methods/accessors/constructor.
    pub func: Option<Rc<Function>>,
    /// For fields (`x = init` / `x`).
    pub value: Option<Expr>,
    /// `@dec` decorators applied to this element.
    pub decorators: Vec<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberKind {
    Constructor,
    Method,
    Get,
    Set,
    Field,
    /// `accessor x = init` — an auto-accessor: a private backing field plus a getter/setter pair.
    Accessor,
    /// `static { ... }` — runs once at class definition with `this` = the class.
    StaticBlock,
}

#[derive(Debug, Clone)]
pub enum ForInit {
    VarDecl {
        kind: DeclKind,
        decls: Vec<(Pattern, Option<Expr>)>,
    },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct SwitchCase {
    /// `None` is the `default:` clause.
    pub test: Option<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKind {
    Var,
    Let,
    Const,
    /// `using x = expr;` — a block-scoped binding disposed (`[Symbol.dispose]()`) at scope exit.
    Using,
    /// `await using x = expr;` — disposed via `[Symbol.asyncDispose]()` (awaited) at scope exit.
    AwaitUsing,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Ident(String),
    /// `[a, b = 1, ...rest]` — elements may be holes, may carry defaults, and the last may be a rest.
    Array(Vec<ArrayPatElem>),
    /// `{ a, b: x = 1, ...rest }`.
    Object(ObjectPat),
    /// A member-expression assignment target (`o.p`, `o[k]`) — only valid in assignment-style
    /// destructuring / `for (o.p of …)`, never in a declaration.
    Member(Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum ArrayPatElem {
    Hole,
    Elem {
        pattern: Pattern,
        default: Option<Expr>,
    },
    Rest(Pattern),
}

#[derive(Debug, Clone)]
pub struct ObjectPat {
    pub props: Vec<ObjPatProp>,
    /// `...rest` — a plain identifier collecting the remaining own enumerable keys.
    pub rest: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ObjPatProp {
    pub key: PropKey,
    pub value: Pattern,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // some node fields (regex body/flags) are parsed before they are interpreted
pub enum Expr {
    /// A parenthesized array/object literal or assignment — recorded so destructuring
    /// reinterpretation can reject it (parens block the pattern refinement); evaluates
    /// transparently.
    Paren(P<Expr>),
    Num(f64),
    BigInt(crate::bigint::BigIntLiteral),
    Str(Rc<str>),
    /// A template-literal substitution: evaluate the inner expression and apply ToString (which uses
    /// the `string` hint — toString before valueOf — unlike `+` which uses the `default` hint).
    ToStr(Box<Expr>),
    Bool(bool),
    Null,
    Undefined,
    /// An identifier reference and the byte offset of its token (V8's position for a
    /// ReferenceError); see `crate::stack_trace`.
    Ident(String, u32),
    This,
    Regex {
        body: Rc<str>,
        flags: Rc<str>,
    },
    Array(Vec<ArrayElem>),
    Object(Vec<PropDef>),
    Func(Rc<Function>),
    Class(Rc<Class>),
    /// `yield expr` / `yield* expr` (only inside a generator).
    Yield {
        delegate: bool,
        arg: Option<P<Expr>>,
    },
    /// `await expr` (only inside an async function).
    Await(P<Expr>),
    /// The bare `super` keyword (only valid as `super(...)` or `super.x` / `super[x]`).
    Super,
    Unary {
        op: &'static str,
        arg: P<Expr>,
    },
    Update {
        op: &'static str,
        prefix: bool,
        arg: P<Expr>,
    },
    Binary {
        op: &'static str,
        left: P<Expr>,
        right: P<Expr>,
    },
    Logical {
        op: &'static str,
        left: P<Expr>,
        right: P<Expr>,
    },
    Assign {
        op: &'static str,
        target: P<Expr>,
        value: P<Expr>,
        /// Byte offset of the operator (V8's position for a setter call or a failed store).
        pos: u32,
    },
    Cond {
        test: P<Expr>,
        cons: P<Expr>,
        alt: P<Expr>,
    },
    Call {
        callee: P<Expr>,
        args: Vec<ArrayElem>,
        optional: bool,
        /// Byte offset V8 reports for the call: the callee's last token when that is an
        /// identifier (`f()`, `o.method()`, `super()`), else the `(`.
        pos: u32,
    },
    New {
        callee: P<Expr>,
        args: Vec<ArrayElem>,
        /// Byte offset of the `new` keyword.
        pos: u32,
    },
    Member {
        obj: P<Expr>,
        prop: String,
        optional: bool,
        /// Byte offset of the property name.
        pos: u32,
    },
    Index {
        obj: P<Expr>,
        index: P<Expr>,
        optional: bool,
        /// Byte offset of the `[`.
        pos: u32,
    },
    Seq(Vec<Expr>),
    /// `tag\`a${x}b\`` — `quasis` are (cooked, raw) chunks (one more than `subs`).
    TaggedTemplate {
        tag: P<Expr>,
        /// Stable Parse Node identity for ECMA-262's per-Realm [[TemplateMap]].
        site: u64,
        quasis: Vec<(Option<String>, String)>,
        subs: Box<[Expr]>,
        /// Byte offset of the template, where the tag is called.
        pos: u32,
    },
    /// An optional chain (`a?.b.c`): evaluates the inner LHS, short-circuiting to `undefined` if any
    /// `?.` link sees a nullish base.
    OptionalChain(P<Expr>),
    /// Ergonomic brand check `#field in obj`: whether `obj` carries the private field.
    PrivateIn {
        name: String,
        obj: P<Expr>,
    },
    /// Dynamic `import(specifier)` / `import.source(...)` / `import.defer(...)` — returns a
    /// promise. The phase selects the import semantics.
    ImportCall {
        spec: P<Expr>,
        phase: ImportPhase,
        /// The optional second argument (`import(spec, { with: { type: "json" } })`).
        options: Option<P<Expr>>,
    },
    /// `import.meta`.
    ImportMeta,
    /// `new.target`.
    NewTarget,
}

impl Expr {
    /// The call below `self` when `self` continues a call chain: a non-optional call of the
    /// value of another plain call (`f()()`), or a method call on it (`f().m()`, `f()[k]()`).
    /// The compiler and the tree-walker run such chains in a loop rather than one native frame
    /// per link.
    pub(crate) fn call_chain_link(&self) -> Option<&Expr> {
        let Expr::Call {
            callee,
            optional: false,
            ..
        } = self
        else {
            return None;
        };
        let below = match &**callee {
            below @ Expr::Call { .. } => below,
            Expr::Member {
                obj,
                optional: false,
                ..
            }
            | Expr::Index {
                obj,
                optional: false,
                ..
            } => obj,
            _ => return None,
        };
        match below {
            Expr::Call {
                callee,
                optional: false,
                ..
            } if !matches!(&**callee, Expr::Super)
                && !matches!(&**callee, Expr::Ident(name, _) if name == "eval") =>
            {
                Some(below)
            }
            _ => None,
        }
    }

    /// Detach the operand a left-nested chain continues through (`a + b` → `a`, `a.b` → `a`,
    /// `f()` → `f`, …), leaving a leaf in its place.
    fn take_chain_link(&mut self) -> Option<Expr> {
        let link = match self {
            Expr::Binary { left, .. } | Expr::Logical { left, .. } => left,
            Expr::Member { obj, .. } | Expr::Index { obj, .. } => obj,
            Expr::Call { callee, .. } | Expr::New { callee, .. } => callee,
            Expr::TaggedTemplate { tag, .. } => tag,
            Expr::OptionalChain(inner) | Expr::Paren(inner) | Expr::ToStr(inner) => inner,
            _ => return None,
        };
        match **link {
            Expr::Null => None,
            _ => Some(std::mem::replace(&mut **link, Expr::Null)),
        }
    }
}

impl Drop for Expr {
    /// A chain is as deep as it is long, and chains are not bounded by `MAX_PARSE_DEPTH` (see
    /// `parser::MAX_CHAIN_LINKS`): release it link by link instead of one native frame per link.
    /// Other children drop recursively, through nesting the parser bounds, on a fresh stack
    /// segment when this one runs low (an AST can be dropped wherever its last owner goes).
    fn drop(&mut self) {
        if crate::interpreter::ast_stack_low() {
            let node = std::mem::replace(self, Expr::Null);
            return crate::interpreter::with_ast_stack(move || drop(node));
        }
        let mut next = self.take_chain_link();
        while let Some(mut link) = next {
            next = link.take_chain_link();
        }
    }
}

/// The phase of a dynamic `import()` call: plain evaluation, `import.source(...)` (source-phase),
/// or `import.defer(...)` (deferred-evaluation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportPhase {
    Evaluation,
    Source,
    Defer,
}

/// An array element or call argument: a value, a spread (`...x`), or a hole (`[1,,3]`).
#[derive(Debug, Clone)]
pub enum ArrayElem {
    Item(Expr),
    Spread(Expr),
    Hole,
}

#[derive(Debug, Clone)]
pub enum PropDef {
    /// `key: value` or shorthand `{ x }`.
    KeyValue {
        key: PropKey,
        value: Expr,
    },
    /// CoverInitializedName (`{ x = default }`): only valid when the literal is reinterpreted as
    /// a destructuring pattern; the parser rejects it anywhere else. `value` is the
    /// `x = default` assignment.
    Cover {
        key: PropKey,
        value: Expr,
    },
    /// Concise method `key() {}` (incl. generator/async). Carries a [[HomeObject]] so `super`
    /// inside the body resolves against the literal's prototype.
    Method {
        key: PropKey,
        func: Rc<Function>,
    },
    /// `get key() {}` / `set key(v) {}`.
    Getter {
        key: PropKey,
        func: Rc<Function>,
    },
    Setter {
        key: PropKey,
        func: Rc<Function>,
    },
    Spread(Expr),
    /// The colon-form `__proto__: value` in an object literal — sets `[[Prototype]]` (when the
    /// value is an Object or Null) rather than creating a property. Only the non-computed,
    /// non-shorthand, non-method form. As a destructuring pattern it degrades to a normal
    /// `__proto__` keyed target.
    Proto(Expr),
}

#[derive(Debug, Clone)]
pub enum PropKey {
    Ident(String),
    Str(Rc<str>),
    Num(f64),
    Computed(Expr),
}

/// See [`Function::fn_maps`].
#[derive(Debug, Clone)]
pub struct FunctionMaps {
    /// The function object's own properties: `length`, `name` and, for prototype-bearing kinds,
    /// a `prototype` placeholder patched per closure.
    pub function: crate::value::Props,
    /// The fresh `.prototype` object's properties, for prototype-bearing kinds.
    pub prototype: Option<crate::value::Props>,
    /// `function` with the `name` NamedEvaluation gave the first named closure of this
    /// anonymous function (`var f = () => {}`): a creation site's inferred name is fixed, so
    /// later closures clone it instead of running SetFunctionName per instance.
    pub named: std::cell::RefCell<Option<(Rc<str>, crate::value::Props)>>,
    /// The Agent heap whose strings the templates hold (their `name` values; see
    /// `value::active_heap_id`), which keeps them for its lifetime (`value::pin_template_value`).
    /// gc-heap strings are cells of one heap, and parsed code can outlive an engine or serve
    /// several: another heap's closures take their own name strings.
    #[cfg_attr(not(feature = "gc-heap"), allow(dead_code))]
    pub(crate) heap: u64,
}

/// The FunctionBody of a [`Function`]: the statements that ECMA-262 OrdinaryFunctionCreate
/// stores as the function object's [[ECMAScriptCode]] (local ECMA-262 snapshot e28783d5).
///
/// A parsed function owns its statements. A function decoded from a static host snapshot
/// (`crate::snapshot::decode_static`) instead keeps the encoded bytes of its body, which every
/// Agent of the process shares, and decodes them when the function first needs them. A browser
/// prelude defines thousands of platform methods that a page never calls; their Parse Nodes
/// would otherwise be the largest part of each Agent's memory. Nothing observable depends on
/// when an implementation materializes Parse Nodes: the source was fully parsed, with its early
/// errors, when the snapshot was compiled, and the snapshot was fully validated when this
/// process first decoded it.
///
/// Reading the statements (`Deref`) materializes them once. They then belong to the Function
/// like parsed ones, so their node identity (template objects, loop sites) is stable and the
/// caches built from them stay valid. [`FunctionBody::with`] serves analyses of a function that
/// is not about to run, such as an enclosing function's capture scan, from a temporary decode
/// that is dropped afterwards. Memory accounting reads only [`FunctionBody::materialized`].
#[derive(Clone)]
pub struct FunctionBody {
    stmts: std::cell::OnceCell<Vec<Stmt>>,
    /// The statements' snapshot encoding, while they may still be unmaterialized.
    encoded: Option<&'static [u8]>,
}

impl Drop for FunctionBody {
    /// Dropping a body recurses through its nesting, including the functions nested in it, so
    /// it runs where the deepest tree the parser accepts fits (see `interpreter::drop_ast`).
    fn drop(&mut self) {
        if let Some(stmts) = self.stmts.take() {
            crate::interpreter::drop_ast(stmts);
        }
    }
}

impl FunctionBody {
    /// A body whose statements are already in memory (parsed or eagerly decoded).
    pub fn new(stmts: Vec<Stmt>) -> Self {
        Self {
            stmts: std::cell::OnceCell::from(stmts),
            encoded: None,
        }
    }

    /// A body left encoded in a validated static snapshot (see `crate::snapshot`).
    pub(crate) fn encoded(encoded: &'static [u8]) -> Self {
        Self {
            stmts: std::cell::OnceCell::new(),
            encoded: Some(encoded),
        }
    }

    /// The statements, if they are in memory.
    #[inline]
    pub(crate) fn materialized(&self) -> Option<&Vec<Stmt>> {
        self.stmts.get()
    }

    /// Materialize the statements of a function that is about to run or be compiled, before
    /// any analysis of it would otherwise decode them for itself.
    #[inline]
    pub(crate) fn prepare_to_run(&self) {
        if self.stmts.get().is_none() {
            self.materialize();
        }
    }

    /// Run `f` over the statements without materializing them: a body that is still encoded is
    /// decoded for this call only. For syntax-directed analyses of functions that are not about
    /// to run; running and compiling a function read the materialized statements.
    pub(crate) fn with<T>(&self, f: impl FnOnce(&[Stmt]) -> T) -> T {
        match self.stmts.get() {
            Some(stmts) => f(stmts),
            None => f(&self.decode()),
        }
    }

    /// The snapshot bytes of a body that is still encoded: an identity that every decode of
    /// the same function shares.
    #[inline]
    pub(crate) fn encoding(&self) -> Option<&'static [u8]> {
        match self.stmts.get() {
            Some(_) => None,
            None => self.encoded,
        }
    }

    #[cold]
    #[inline(never)]
    fn materialize(&self) -> &Vec<Stmt> {
        self.stmts.get_or_init(|| self.decode())
    }

    fn decode(&self) -> Vec<Stmt> {
        let encoded = self
            .encoded
            .expect("an unmaterialized function body keeps its encoding");
        // The whole snapshot, this body included, was decoded with every budget and check of
        // an eager decode before any of its bodies was left encoded, and its bytes are
        // immutable: decoding them again cannot fail.
        crate::snapshot::decode_function_body(encoded)
            .unwrap_or_else(|error| panic!("validated snapshot function body: {error}"))
    }
}

impl From<Vec<Stmt>> for FunctionBody {
    fn from(stmts: Vec<Stmt>) -> Self {
        Self::new(stmts)
    }
}

impl std::ops::Deref for FunctionBody {
    type Target = Vec<Stmt>;
    #[inline]
    fn deref(&self) -> &Vec<Stmt> {
        match self.stmts.get() {
            Some(stmts) => stmts,
            None => self.materialize(),
        }
    }
}

impl<'a> IntoIterator for &'a FunctionBody {
    type Item = &'a Stmt;
    type IntoIter = std::slice::Iter<'a, Stmt>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::DerefMut for FunctionBody {
    fn deref_mut(&mut self) -> &mut Vec<Stmt> {
        if self.stmts.get().is_none() {
            let stmts = self.decode();
            let _ = self.stmts.set(stmts);
        }
        // Edited statements no longer match the encoding.
        self.encoded = None;
        self.stmts.get_mut().expect("materialized above")
    }
}

impl std::fmt::Debug for FunctionBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with(|stmts| stmts.fmt(f))
    }
}

impl Drop for Function {
    /// Like a body (see `FunctionBody`'s drop), parameters can nest further functions in their
    /// defaults; release both where the deepest tree the parser accepts fits.
    fn drop(&mut self) {
        let params = std::mem::take(&mut self.params);
        let stmts = self.body.stmts.take();
        if !params.is_empty() || stmts.is_some() {
            crate::interpreter::drop_ast((params, stmts));
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // expr_body is recorded for a future `toString`/source-fidelity pass
pub struct Function {
    pub name: Option<String>,
    pub params: Vec<Param>,
    pub body: FunctionBody,
    pub is_arrow: bool,
    pub is_strict: bool,
    /// Arrow with an expression body (`x => x+1`): the single statement is a synthetic `return`.
    pub expr_body: bool,
    pub is_generator: bool,
    pub is_async: bool,
    /// A concise method / getter / setter: has no own `prototype` and is not a constructor (the
    /// class `constructor` member is re-flagged false once identified).
    pub is_method: bool,
    /// A function *expression* (`(function f(){})`): its own name binds immutably inside the
    /// function. A declaration's name binds (mutably) in the enclosing scope instead.
    pub is_fn_expr: bool,
    /// The synthesized default constructor of a derived class (ECMA-262
    /// ClassDefinitionEvaluation step 14.a). `[[Construct]]` runs it as the built-in closure it
    /// is (`Interp::construct_default_derived`), forwarding the raw argument list without the
    /// observable `%Array.prototype%` iteration of a written `super(...args)`; the synthesized
    /// body is never evaluated. Always false for parsed source.
    pub default_ctor: bool,
    /// Part of the engine's self-hosted built-in source (see `crate::self_hosted`): compiled with
    /// that source's intrinsic operations, and rendered as a NativeFunction (it has no `source`).
    /// Never set for author code.
    pub self_hosted: bool,
    /// Run-once bootstrap code of a static host snapshot: a function expression that the
    /// snapshot's top-level code, or the body of another such function, calls immediately and
    /// outside any loop or class (`(function () { … })()`), so it runs once in each Realm that
    /// evaluates the snapshot. Its bytecode runs on the VM and never gets native code (see
    /// `Interp::run_compiled_chunk`). Compilation waits for call-count or executed-backedge
    /// hotness, as with ordinary functions. Never set for parsed source or
    /// other snapshots.
    pub(crate) run_once: bool,
    /// The source text this function was parsed from, for `Function.prototype.toString`.
    pub source: Option<Rc<str>>,
    /// The script, module, eval or `Function` source the function was parsed from, which its
    /// AST positions index (see `crate::stack_trace`). None for self-hosted and snapshot code.
    pub(crate) script: Option<Rc<crate::stack_trace::ScriptSource>>,
    /// Byte offset of the function's first token in `script`: the position its stack frames
    /// report when the current operation has none.
    pub start: u32,
    /// Lazily-computed body facts (see [`Function::scan_flags`]): bit 0 = scanned, bit 1 =
    /// references `arguments`, bit 2 = references `new.target`, bit 3 = references `this`.
    /// A direct `eval` sets all three (it can reach any of them dynamically).
    pub scan: std::cell::Cell<u8>,
    /// Lazily-computed hoisting plan for the body (the strict flag it was computed under plus the
    /// ops), so calls replay a flat list instead of re-walking the AST (see
    /// `interpreter::collect_hoist_ops`).
    pub hoist: std::cell::OnceCell<(bool, Rc<Vec<HoistOp>>)>,
    /// Bytecode-tier state: call count until tier-up, and the compile result once attempted
    /// (`None` = uses constructs outside the bytecode subset; runs in the tree-walker forever).
    pub calls: std::cell::Cell<u32>,
    pub code: std::cell::OnceCell<Option<Rc<crate::bytecode::Chunk>>>,
    /// Second-stage compile: the body re-compiled with hot monomorphic callees inlined (see
    /// `bytecode::plan_inlines`). Preferred over `code` wherever a chunk is fetched; `code`
    /// stays alive because filled call ICs hold raw pointers into it.
    pub code2: std::cell::OnceCell<Option<Rc<crate::bytecode::Chunk>>>,
    /// Lazily-built object-map templates for closures of this function (see
    /// `Interp::make_function`): the function object's `Props` (length/name and a `prototype`
    /// placeholder) and, for prototype-bearing kinds, the fresh `.prototype`'s `Props` — cloned
    /// per closure instance instead of rebuilt insert by insert, so key hashing and shape
    /// transitions are paid once per FUNCTION rather than once per closure. Boxed: the AST
    /// keeps a slot for every parsed function, most of which are instantiated rarely or never.
    pub fn_maps: std::cell::OnceCell<Box<FunctionMaps>>,
    /// Actual parameter/body key layouts observed before tier-up. Each cache is
    /// bounded to one layout; eval-dependent alternatives keep dynamic storage.
    pub(crate) env_layouts:
        std::cell::OnceCell<Box<[std::cell::OnceCell<Rc<crate::interpreter::BindingLayout>>; 2]>>,
}

/// One pre-scanned hoisting action for a statement list, replayed against a scope at function
/// call / script entry (see `interpreter::collect_hoist_ops` — order matters: `var`s in
/// traversal order, then function declarations in source order, then Annex B promotions).
#[derive(Debug, Clone)]
pub enum HoistOp {
    /// Declare a `var` binding (undefined) unless the name is already bound.
    Var(String),
    /// Bind a hoisted function declaration (`*default*` names as "default").
    Fn(String, Rc<Function>),
    /// Annex B.3.3: promote a sloppy block function to an (if-absent) var binding and register
    /// it for declaration-time sync.
    AnnexB(String, Rc<Function>),
}

struct RetainedAst<'a> {
    visitor: &'a mut crate::memory::Visitor,
    bytes: usize,
}

impl RetainedAst<'_> {
    fn add(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_add(bytes);
    }

    fn vec<T>(&mut self, values: &Vec<T>) {
        self.add(values.capacity().saturating_mul(std::mem::size_of::<T>()));
    }

    fn string(&mut self, value: &String) {
        self.add(value.capacity());
    }

    fn rc_str(&mut self, value: &Rc<str>) {
        self.visitor.rc_str(value);
    }

    fn boxed_expr(&mut self, value: &Expr) {
        self.add(std::mem::size_of::<Expr>());
        self.expr(value);
    }

    fn boxed_stmt(&mut self, value: &Stmt) {
        self.add(std::mem::size_of::<Stmt>());
        self.stmt(value);
    }

    fn stmt_vec(&mut self, values: &Vec<Stmt>) {
        self.vec(values);
        for value in values {
            self.stmt(value);
        }
    }

    fn decls(&mut self, decls: &Vec<(Pattern, Option<Expr>)>) {
        self.vec(decls);
        for (pattern, initializer) in decls {
            self.pattern(pattern);
            if let Some(initializer) = initializer {
                self.expr(initializer);
            }
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        if crate::interpreter::ast_stack_low() {
            return crate::interpreter::with_ast_stack(|| self.stmt(stmt));
        }
        if let Some(body) = stmt.loop_body() {
            self.add(std::mem::size_of::<LoopBody>() - std::mem::size_of::<Stmt>());
            if let Some(token) = body.site.retained_token() {
                self.visitor.loop_site_token(token);
            }
        }
        match stmt {
            Stmt::Expr(expr) | Stmt::Throw(expr) => self.expr(expr),
            Stmt::VarDecl { kind: _, decls } => self.decls(decls),
            Stmt::FuncDecl(function) => self.visitor.function(function),
            Stmt::Return(expr) => {
                if let Some(expr) = expr {
                    self.expr(expr);
                }
            }
            Stmt::If { test, cons, alt } => {
                self.expr(test);
                self.boxed_stmt(cons);
                if let Some(alt) = alt {
                    self.boxed_stmt(alt);
                }
            }
            Stmt::Block(body) => self.stmt_vec(body),
            Stmt::While { test, body, .. } => {
                self.expr(test);
                self.boxed_stmt(body);
            }
            Stmt::DoWhile { body, test, .. } => {
                self.boxed_stmt(body);
                self.expr(test);
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => {
                if let Some(init) = init {
                    self.add(std::mem::size_of::<ForInit>());
                    self.for_init(init);
                }
                if let Some(test) = test {
                    self.expr(test);
                }
                if let Some(update) = update {
                    self.expr(update);
                }
                self.boxed_stmt(body);
            }
            Stmt::ForInOf {
                decl: _,
                left,
                right,
                of: _,
                is_await: _,
                body,
                ..
            } => {
                self.pattern(left);
                self.expr(right);
                self.boxed_stmt(body);
            }
            Stmt::Break(label) | Stmt::Continue(label) => {
                if let Some(label) = label {
                    self.string(label);
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => {
                self.stmt_vec(block);
                if let Some((pattern, body)) = handler {
                    if let Some(pattern) = pattern {
                        self.pattern(pattern);
                    }
                    self.stmt_vec(body);
                }
                if let Some(finalizer) = finalizer {
                    self.stmt_vec(finalizer);
                }
            }
            Stmt::Switch { disc, cases } => {
                self.expr(disc);
                self.vec(cases);
                for case in cases {
                    if let Some(test) = &case.test {
                        self.expr(test);
                    }
                    self.stmt_vec(&case.body);
                }
            }
            Stmt::Labeled { label, body } => {
                self.string(label);
                self.boxed_stmt(body);
            }
            Stmt::With { obj, body } => {
                self.expr(obj);
                self.boxed_stmt(body);
            }
            Stmt::ClassDecl(class) => self.visitor.class(class),
            Stmt::Empty | Stmt::Debugger => {}
            Stmt::Import(import) => self.import(import),
            Stmt::ExportNamed { specs, source } => {
                self.vec(specs);
                for spec in specs {
                    self.string(&spec.local);
                    self.string(&spec.exported);
                }
                if let Some(source) = source {
                    self.rc_str(source);
                }
            }
            Stmt::ExportDecl(inner) | Stmt::ExportDefault(inner) => self.boxed_stmt(inner),
            Stmt::ExportAll { source, exported } => {
                self.rc_str(source);
                if let Some(exported) = exported {
                    self.string(exported);
                }
            }
        }
    }

    fn import(&mut self, import: &ImportDecl) {
        self.rc_str(&import.source);
        self.vec(&import.specs);
        for spec in &import.specs {
            match spec {
                ImportSpec::Default(local)
                | ImportSpec::Namespace(local)
                | ImportSpec::DeferNamespace(local)
                | ImportSpec::Source(local) => self.string(local),
                ImportSpec::Named { imported, local } => {
                    self.string(imported);
                    self.string(local);
                }
            }
        }
        if let Some(attr_type) = &import.attr_type {
            self.string(attr_type);
        }
    }

    fn for_init(&mut self, init: &ForInit) {
        match init {
            ForInit::VarDecl { kind: _, decls } => self.decls(decls),
            ForInit::Expr(expr) => self.expr(expr),
        }
    }

    fn pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Ident(name) => self.string(name),
            Pattern::Array(elements) => {
                self.vec(elements);
                for element in elements {
                    match element {
                        ArrayPatElem::Hole => {}
                        ArrayPatElem::Elem { pattern, default } => {
                            self.pattern(pattern);
                            if let Some(default) = default {
                                self.expr(default);
                            }
                        }
                        ArrayPatElem::Rest(pattern) => self.pattern(pattern),
                    }
                }
            }
            Pattern::Object(object) => {
                self.vec(&object.props);
                for property in &object.props {
                    self.prop_key(&property.key);
                    self.pattern(&property.value);
                    if let Some(default) = &property.default {
                        self.expr(default);
                    }
                }
                if let Some(rest) = &object.rest {
                    self.string(rest);
                }
            }
            Pattern::Member(expr) => self.boxed_expr(expr),
        }
    }

    fn expr(&mut self, expr: &Expr) {
        if crate::interpreter::ast_stack_low() {
            return crate::interpreter::with_ast_stack(|| self.expr(expr));
        }
        match expr {
            Expr::Paren(inner)
            | Expr::ToStr(inner)
            | Expr::Await(inner)
            | Expr::OptionalChain(inner) => self.boxed_expr(inner),
            Expr::Num(_)
            | Expr::Bool(_)
            | Expr::Null
            | Expr::Undefined
            | Expr::This
            | Expr::Super
            | Expr::ImportMeta
            | Expr::NewTarget => {}
            Expr::BigInt(value) => self.visitor.bigint_literal(value),
            Expr::Str(value) => self.rc_str(value),
            Expr::Ident(name, _) => self.string(name),
            Expr::Regex { body, flags } => {
                self.rc_str(body);
                self.rc_str(flags);
            }
            Expr::Array(elements) => {
                self.vec(elements);
                for element in elements {
                    self.array_elem(element);
                }
            }
            Expr::Object(properties) => {
                self.vec(properties);
                for property in properties {
                    self.prop_def(property);
                }
            }
            Expr::Func(function) => self.visitor.function(function),
            Expr::Class(class) => self.visitor.class(class),
            Expr::Yield { delegate: _, arg } => {
                if let Some(arg) = arg {
                    self.boxed_expr(arg);
                }
            }
            Expr::Unary { op: _, arg }
            | Expr::Update {
                op: _,
                prefix: _,
                arg,
            } => self.boxed_expr(arg),
            Expr::Binary { op: _, left, right }
            | Expr::Logical { op: _, left, right }
            | Expr::Assign {
                op: _,
                target: left,
                value: right,
                ..
            } => {
                self.boxed_expr(left);
                self.boxed_expr(right);
            }
            Expr::Cond { test, cons, alt } => {
                self.boxed_expr(test);
                self.boxed_expr(cons);
                self.boxed_expr(alt);
            }
            Expr::Call {
                callee,
                args,
                optional: _,
                pos: _,
            }
            | Expr::New {
                callee,
                args,
                pos: _,
            } => {
                self.boxed_expr(callee);
                self.vec(args);
                for argument in args {
                    self.array_elem(argument);
                }
            }
            Expr::Member {
                obj,
                prop,
                optional: _,
                pos: _,
            } => {
                self.boxed_expr(obj);
                self.string(prop);
            }
            Expr::Index {
                obj,
                index,
                optional: _,
                pos: _,
            } => {
                self.boxed_expr(obj);
                self.boxed_expr(index);
            }
            Expr::Seq(expressions) => {
                self.vec(expressions);
                for expression in expressions {
                    self.expr(expression);
                }
            }
            Expr::TaggedTemplate {
                tag,
                site: _,
                quasis,
                subs,
                ..
            } => {
                self.boxed_expr(tag);
                self.vec(quasis);
                for (cooked, raw) in quasis {
                    if let Some(cooked) = cooked {
                        self.string(cooked);
                    }
                    self.string(raw);
                }
                self.add(std::mem::size_of_val::<[Expr]>(subs));
                for substitution in subs.iter() {
                    self.expr(substitution);
                }
            }
            Expr::PrivateIn { name, obj } => {
                self.string(name);
                self.boxed_expr(obj);
            }
            Expr::ImportCall {
                spec,
                phase: _,
                options,
            } => {
                self.boxed_expr(spec);
                if let Some(options) = options {
                    self.boxed_expr(options);
                }
            }
        }
    }

    fn array_elem(&mut self, element: &ArrayElem) {
        match element {
            ArrayElem::Item(expr) | ArrayElem::Spread(expr) => self.expr(expr),
            ArrayElem::Hole => {}
        }
    }

    fn prop_def(&mut self, property: &PropDef) {
        match property {
            PropDef::KeyValue { key, value } | PropDef::Cover { key, value } => {
                self.prop_key(key);
                self.expr(value);
            }
            PropDef::Method { key, func }
            | PropDef::Getter { key, func }
            | PropDef::Setter { key, func } => {
                self.prop_key(key);
                self.visitor.function(func);
            }
            PropDef::Spread(expr) | PropDef::Proto(expr) => self.expr(expr),
        }
    }

    fn prop_key(&mut self, key: &PropKey) {
        match key {
            PropKey::Ident(name) => self.string(name),
            PropKey::Str(value) => self.rc_str(value),
            PropKey::Num(_) => {}
            PropKey::Computed(expr) => self.expr(expr),
        }
    }

    fn class(&mut self, class: &Class) {
        if let Some(name) = &class.name {
            self.string(name);
        }
        if let Some(superclass) = &class.superclass {
            self.boxed_expr(superclass);
        }
        self.vec(&class.members);
        for member in &class.members {
            self.prop_key(&member.key);
            if let Some(function) = &member.func {
                self.visitor.function(function);
            }
            if let Some(value) = &member.value {
                self.expr(value);
            }
            self.vec(&member.decorators);
            for decorator in &member.decorators {
                self.expr(decorator);
            }
        }
        self.vec(&class.decorators);
        for decorator in &class.decorators {
            self.expr(decorator);
        }
        if let Some(source) = &class.source {
            self.rc_str(source);
        }
    }
}

/// Scan one identity-unique Function allocation and every non-shared AST allocation it owns.
/// Rc-backed Functions, Classes, strings, and BigInts route through the global allocation-family
/// visitor so shared subgraphs are never attributed according to discovery order.
pub(crate) fn scan_function_retained_memory(
    function: &Function,
    visitor: &mut crate::memory::Visitor,
) -> usize {
    let mut scan = RetainedAst {
        visitor,
        bytes: std::mem::size_of::<Function>(),
    };
    if let Some(name) = &function.name {
        scan.string(name);
    }
    scan.vec(&function.params);
    for param in &function.params {
        scan.pattern(&param.pattern);
        if let Some(default) = &param.default {
            scan.expr(default);
        }
    }
    // A body still encoded in a static snapshot owns no allocation: the bytes are the process's.
    if let Some(body) = function.body.materialized() {
        scan.stmt_vec(body);
    }
    if let Some(source) = &function.source {
        scan.rc_str(source);
    }
    if let Some(script) = &function.script {
        scan.visitor.script_source(script);
    }
    scan.bytes
}

/// Scan one identity-unique Class allocation and its recursively-owned AST storage.
pub(crate) fn scan_class_retained_memory(
    class: &Class,
    visitor: &mut crate::memory::Visitor,
) -> usize {
    let mut scan = RetainedAst {
        visitor,
        bytes: std::mem::size_of::<Class>(),
    };
    scan.class(class);
    scan.bytes
}

/// Scan allocations recursively owned by an Expr whose enum body is stored inline by its caller.
pub(crate) fn scan_expr_retained_memory(
    expr: &Expr,
    visitor: &mut crate::memory::Visitor,
) -> usize {
    let mut scan = RetainedAst { visitor, bytes: 0 };
    scan.expr(expr);
    scan.bytes
}

/// Scan an Rc-owned statement-list allocation whose Vec header and element buffer are not part of
/// a Function body allocation.
pub(crate) fn scan_stmt_body_retained_memory(
    body: &Vec<Stmt>,
    visitor: &mut crate::memory::Visitor,
) -> usize {
    let mut scan = RetainedAst {
        visitor,
        bytes: std::mem::size_of::<Vec<Stmt>>(),
    };
    scan.stmt_vec(body);
    scan.bytes
}

pub const SCAN_DONE: u8 = 1;
pub const SCAN_ARGUMENTS: u8 = 2;
pub const SCAN_NEW_TARGET: u8 = 4;
pub const SCAN_THIS: u8 = 8;
/// The body itself contains a loop statement (not counting nested functions). Native admission
/// can use this fact after call-count or executed-backedge hotness admits the bytecode body.
pub const SCAN_HAS_LOOP: u8 = 16;
const SCAN_SELF_DONE: u8 = 32;
const SCAN_SELF_NEEDED: u8 = 64;
/// The body (seen through arrows, like the flags above) can observe this function's
/// [[HomeObject]]: it contains a `super` reference, or a direct `eval` whose code may contain
/// one. ECMA-262 reads [[HomeObject]] only through HasSuperBinding (MakeSuperPropertyReference,
/// PerformEval's `inMethod`) and GetSuperBase, and both are reached only from `super` in the
/// function's own code or its direct eval code.
pub const SCAN_HOME: u8 = 128;

/// Does this statement list itself own a loop? Nested functions and class static
/// blocks execute through separate entries and must not make a cold Script hot.
/// Borrow existing nodes; flat bundles allocate nothing and nested controls use
/// an explicit worklist rather than adding recursive native-stack consumption.
#[cfg(test)]
pub(crate) fn statement_list_has_own_loop(body: &[Stmt]) -> bool {
    let mut current = body;
    let mut pending: Vec<&[Stmt]> = Vec::new();
    loop {
        for statement in current {
            match statement {
                Stmt::While { .. }
                | Stmt::DoWhile { .. }
                | Stmt::For { .. }
                | Stmt::ForInOf { .. } => return true,
                Stmt::Block(body) => pending.push(body),
                Stmt::If { cons, alt, .. } => {
                    pending.push(std::slice::from_ref(cons.as_ref()));
                    if let Some(alt) = alt {
                        pending.push(std::slice::from_ref(alt.as_ref()));
                    }
                }
                Stmt::Try {
                    block,
                    handler,
                    finalizer,
                } => {
                    pending.push(block);
                    if let Some((_, body)) = handler {
                        pending.push(body);
                    }
                    if let Some(body) = finalizer {
                        pending.push(body);
                    }
                }
                Stmt::Switch { cases, .. } => {
                    for case in cases {
                        pending.push(&case.body);
                    }
                }
                Stmt::Labeled { body, .. }
                | Stmt::With { body, .. }
                | Stmt::ExportDecl(body)
                | Stmt::ExportDefault(body) => {
                    pending.push(std::slice::from_ref(body.as_ref()));
                }
                Stmt::Expr(_)
                | Stmt::VarDecl { .. }
                | Stmt::FuncDecl(_)
                | Stmt::Return(_)
                | Stmt::Break(_)
                | Stmt::Continue(_)
                | Stmt::Throw(_)
                | Stmt::ClassDecl(_)
                | Stmt::Empty
                | Stmt::Debugger
                | Stmt::Import(_)
                | Stmt::ExportNamed { .. }
                | Stmt::ExportAll { .. } => {}
            }
        }
        let Some(next) = pending.pop() else {
            return false;
        };
        current = next;
    }
}

impl Expr {
    /// The source position of a node that records one (see `crate::stack_trace`).
    pub(crate) fn position(&self) -> Option<u32> {
        match self {
            Expr::Call { pos, .. }
            | Expr::New { pos, .. }
            | Expr::Member { pos, .. }
            | Expr::Index { pos, .. }
            | Expr::TaggedTemplate { pos, .. }
            | Expr::Assign { pos, .. }
            | Expr::Ident(_, pos) => Some(*pos),
            _ => None,
        }
    }
}

impl Function {
    /// Select code for a new invocation; live activations retain their own chunk.
    /// ECMA-262 e28783d5, Execution Contexts / PrepareForOrdinaryCall: changing a
    /// later entry must not replace an existing invocation's evaluation state.
    #[inline]
    #[cfg(all(test, not(feature = "gc-heap")))]
    pub(crate) fn execution_code(&self) -> Option<&Option<Rc<crate::bytecode::Chunk>>> {
        self.code2.get().or_else(|| self.code.get())
    }

    /// Shared snapshot ASTs have one second-stage slot for every Realm. A body planned
    /// exclusively for another Realm cannot win any inline guard, so a new invocation
    /// retains its warmed primary body. Mixed/unrestricted chunks remain eligible.
    /// Active frames and code leases keep their already selected immutable chunk.
    #[inline]
    pub(crate) fn execution_code_for(
        &self,
        genv: usize,
    ) -> Option<&Option<Rc<crate::bytecode::Chunk>>> {
        match self.code2.get() {
            Some(Some(chunk)) if !chunk.inline_realm_compatible(genv) => self.code.get(),
            Some(_) => self.code2.get(),
            None => self.code.get(),
        }
    }

    /// What this function's own activation must provide: whether the body (or a nested arrow, or a
    /// possible direct `eval`) can observe `arguments`, `new.target`, `this`, or the function's
    /// [[HomeObject]] ([`SCAN_HOME`]). Ordinary nested functions are opaque (they get their own);
    /// arrows are transparent. Conservative on the safe side: a false positive only costs an
    /// unused binding.
    pub fn scan_flags(&self) -> u8 {
        let cached = self.scan.get();
        if cached & SCAN_DONE != 0 {
            return cached;
        }
        let mut flags = SCAN_DONE;
        for p in &self.params {
            scan_pattern(&p.pattern, &mut flags);
            if let Some(d) = &p.default {
                scan_expr(d, &mut flags);
            }
        }
        // Closure creation asks this of methods (MakeMethod's [[HomeObject]]) and enclosing
        // scans ask it of nested arrows; neither runs the body, so an encoded one is not kept.
        self.body.with(|body| scan_stmts(body, &mut flags));
        self.scan
            .set(flags | (cached & (SCAN_SELF_DONE | SCAN_SELF_NEEDED)));
        flags
    }

    /// Whether a method closure of this function must carry its [[HomeObject]]. Method
    /// definitions (ECMA-262 DefineMethod, MethodDefinitionEvaluation) always perform MakeMethod,
    /// but the slot is observable only through `super` references and direct `eval` in the
    /// method's own code (see [`SCAN_HOME`]). Leaving it unset otherwise avoids retaining the home
    /// object from each of its methods: an object literal and its methods would form a
    /// reference cycle that only the cycle collector could reclaim.
    pub(crate) fn observes_home_object(&self) -> bool {
        self.scan_flags() & SCAN_HOME != 0
    }

    /// Cache the conservative self-binding query on shared source, never per
    /// closure. The analysis has a finite syntax budget and retains on doubt.
    pub(crate) fn needs_self_environment(&self) -> bool {
        if !self.is_fn_expr || self.name.is_none() {
            return false;
        }
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var("TRUST_JS_ELIDE_SELF_ENV").as_deref() != Ok("0"))
        {
            return true;
        }
        let cached = self.scan.get();
        if cached & SCAN_SELF_DONE != 0 {
            return cached & SCAN_SELF_NEEDED != 0;
        }
        let needed = crate::bytecode::named_self_binding_may_be_observed(self);
        self.scan
            .set(cached | SCAN_SELF_DONE | if needed { SCAN_SELF_NEEDED } else { 0 });
        needed
    }
}

const SCAN_ALL: u8 =
    SCAN_DONE | SCAN_ARGUMENTS | SCAN_NEW_TARGET | SCAN_THIS | SCAN_HAS_LOOP | SCAN_HOME;

fn scan_stmts(body: &[Stmt], flags: &mut u8) {
    for s in body {
        scan_stmt(s, flags);
        if *flags == SCAN_ALL {
            return;
        }
    }
}

fn scan_stmt(s: &Stmt, flags: &mut u8) {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| scan_stmt(s, flags));
    }
    match s {
        Stmt::Expr(e) | Stmt::Throw(e) => scan_expr(e, flags),
        Stmt::VarDecl { kind: _, decls } => {
            for (pat, init) in decls {
                scan_pattern(pat, flags);
                if let Some(e) = init {
                    scan_expr(e, flags);
                }
            }
        }
        // A nested (non-arrow) function has its own arguments/new.target/this.
        Stmt::FuncDecl(_) => {}
        Stmt::Return(e) => {
            if let Some(e) = e {
                scan_expr(e, flags);
            }
        }
        Stmt::If { test, cons, alt } => {
            scan_expr(test, flags);
            scan_stmt(cons, flags);
            if let Some(a) = alt {
                scan_stmt(a, flags);
            }
        }
        Stmt::Block(b) => scan_stmts(b, flags),
        Stmt::While { test, body, .. } | Stmt::DoWhile { body, test, .. } => {
            *flags |= SCAN_HAS_LOOP;
            scan_expr(test, flags);
            scan_stmt(body, flags);
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            *flags |= SCAN_HAS_LOOP;
            match init.as_deref() {
                Some(ForInit::VarDecl { kind: _, decls }) => {
                    for (pat, e) in decls {
                        scan_pattern(pat, flags);
                        if let Some(e) = e {
                            scan_expr(e, flags);
                        }
                    }
                }
                Some(ForInit::Expr(e)) => scan_expr(e, flags),
                None => {}
            }
            if let Some(e) = test {
                scan_expr(e, flags);
            }
            if let Some(e) = update {
                scan_expr(e, flags);
            }
            scan_stmt(body, flags);
        }
        Stmt::ForInOf {
            decl: _,
            left,
            right,
            of: _,
            is_await: _,
            body,
            ..
        } => {
            *flags |= SCAN_HAS_LOOP;
            scan_pattern(left, flags);
            scan_expr(right, flags);
            scan_stmt(body, flags);
        }
        Stmt::Break(_) | Stmt::Continue(_) | Stmt::Empty | Stmt::Debugger => {}
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            scan_stmts(block, flags);
            if let Some((param, hbody)) = handler {
                if let Some(p) = param {
                    scan_pattern(p, flags);
                }
                scan_stmts(hbody, flags);
            }
            if let Some(f) = finalizer {
                scan_stmts(f, flags);
            }
        }
        Stmt::Switch { disc, cases } => {
            scan_expr(disc, flags);
            for c in cases {
                if let Some(t) = &c.test {
                    scan_expr(t, flags);
                }
                scan_stmts(&c.body, flags);
            }
        }
        Stmt::Labeled { label: _, body } => scan_stmt(body, flags),
        Stmt::With { obj, body } => {
            scan_expr(obj, flags);
            scan_stmt(body, flags);
        }
        Stmt::ClassDecl(c) => scan_class(c, flags),
        Stmt::Import(_) | Stmt::ExportNamed { .. } | Stmt::ExportAll { .. } => {}
        Stmt::ExportDecl(inner) | Stmt::ExportDefault(inner) => scan_stmt(inner, flags),
    }
}

/// Activation observations (`arguments`, `new.target`, `this`) of one expression evaluated in
/// the enclosing function scope, seen through arrows exactly as [`Function::scan_flags`] does.
pub(crate) fn expr_scan_flags(e: &Expr) -> u8 {
    let mut flags = 0;
    scan_expr(e, &mut flags);
    flags
}

fn scan_expr(e: &Expr, flags: &mut u8) {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| scan_expr(e, flags));
    }
    match e {
        Expr::Ident(n, _) => {
            if n == "arguments" {
                *flags |= SCAN_ARGUMENTS;
            }
        }
        Expr::This => *flags |= SCAN_THIS,
        Expr::NewTarget => *flags |= SCAN_NEW_TARGET,
        Expr::Num(_)
        | Expr::BigInt(_)
        | Expr::Str(_)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::Undefined
        | Expr::Regex { .. }
        | Expr::ImportMeta => {}
        // `super.x` resolves its receiver through the `this` binding and its base through the
        // [[HomeObject]]; `super()` initializes the `this` binding.
        Expr::Super => *flags |= SCAN_THIS | SCAN_HOME,
        Expr::Paren(inner)
        | Expr::ToStr(inner)
        | Expr::Await(inner)
        | Expr::OptionalChain(inner) => scan_expr(inner, flags),
        Expr::Array(elems) => {
            for el in elems {
                match el {
                    ArrayElem::Item(e) | ArrayElem::Spread(e) => scan_expr(e, flags),
                    ArrayElem::Hole => {}
                }
            }
        }
        Expr::Object(props) => {
            for p in props {
                match p {
                    PropDef::KeyValue { key, value } | PropDef::Cover { key, value } => {
                        scan_prop_key(key, flags);
                        scan_expr(value, flags);
                    }
                    // A concise method/accessor body is its own function scope; only its
                    // (computed) key evaluates here.
                    PropDef::Method { key, func: _ }
                    | PropDef::Getter { key, func: _ }
                    | PropDef::Setter { key, func: _ } => scan_prop_key(key, flags),
                    PropDef::Spread(e) | PropDef::Proto(e) => scan_expr(e, flags),
                }
            }
        }
        // An arrow is transparent (it closes over the enclosing activation); an ordinary
        // function expression is opaque.
        Expr::Func(f) => {
            if f.is_arrow {
                let inner = f.scan_flags();
                *flags |= inner & (SCAN_ARGUMENTS | SCAN_NEW_TARGET | SCAN_THIS | SCAN_HOME);
            }
        }
        Expr::Class(c) => scan_class(c, flags),
        Expr::Yield { delegate: _, arg } => {
            if let Some(a) = arg {
                scan_expr(a, flags);
            }
        }
        Expr::Unary { op: _, arg }
        | Expr::Update {
            op: _,
            prefix: _,
            arg,
        } => scan_expr(arg, flags),
        Expr::Binary { op: _, left, right } | Expr::Logical { op: _, left, right } => {
            scan_expr(left, flags);
            scan_expr(right, flags);
        }
        Expr::Assign {
            op: _,
            target,
            value,
            ..
        } => {
            scan_expr(target, flags);
            scan_expr(value, flags);
        }
        Expr::Cond { test, cons, alt } => {
            scan_expr(test, flags);
            scan_expr(cons, flags);
            scan_expr(alt, flags);
        }
        Expr::Call {
            callee,
            args,
            optional: _,
            pos: _,
        } => {
            // A direct `eval` can name any of the three dynamically, and its code may contain a
            // SuperProperty when the caller is method code (PerformEval's `inMethod`).
            if matches!(&**callee, Expr::Ident(n, _) if n == "eval") {
                *flags |= SCAN_ARGUMENTS | SCAN_NEW_TARGET | SCAN_THIS | SCAN_HOME;
            }
            // SuperCall begins with GetNewTarget. An arrow is transparent to that lookup, so a
            // constructor containing an async arrow with `super(...)` must retain its own
            // new.target binding after the constructor invocation returns.
            if matches!(&**callee, Expr::Super) {
                *flags |= SCAN_NEW_TARGET;
            }
            scan_expr(callee, flags);
            for a in args {
                match a {
                    ArrayElem::Item(e) | ArrayElem::Spread(e) => scan_expr(e, flags),
                    ArrayElem::Hole => {}
                }
            }
        }
        Expr::New {
            callee,
            args,
            pos: _,
        } => {
            scan_expr(callee, flags);
            for a in args {
                match a {
                    ArrayElem::Item(e) | ArrayElem::Spread(e) => scan_expr(e, flags),
                    ArrayElem::Hole => {}
                }
            }
        }
        Expr::Member {
            obj,
            prop: _,
            optional: _,
            pos: _,
        } => scan_expr(obj, flags),
        Expr::Index {
            obj,
            index,
            optional: _,
            pos: _,
        } => {
            scan_expr(obj, flags);
            scan_expr(index, flags);
        }
        Expr::Seq(exprs) => {
            for e in exprs {
                scan_expr(e, flags);
            }
        }
        Expr::TaggedTemplate {
            tag,
            quasis: _,
            subs,
            ..
        } => {
            scan_expr(tag, flags);
            for e in subs {
                scan_expr(e, flags);
            }
        }
        Expr::PrivateIn { name: _, obj } => scan_expr(obj, flags),
        Expr::ImportCall {
            spec,
            phase: _,
            options,
        } => {
            scan_expr(spec, flags);
            if let Some(o) = options {
                scan_expr(o, flags);
            }
        }
    }
}

fn scan_class(c: &Class, flags: &mut u8) {
    // Heritage, decorators, and computed keys evaluate in the enclosing scope. Member bodies are
    // their own function scopes; field/accessor initializers and static blocks can't legally name
    // `arguments`, but walking them costs only a possible false positive.
    if let Some(sc) = &c.superclass {
        scan_expr(sc, flags);
    }
    for d in &c.decorators {
        scan_expr(d, flags);
    }
    for m in &c.members {
        scan_prop_key(&m.key, flags);
        for d in &m.decorators {
            scan_expr(d, flags);
        }
        if let Some(v) = &m.value {
            scan_expr(v, flags);
        }
    }
}

fn scan_prop_key(k: &PropKey, flags: &mut u8) {
    match k {
        PropKey::Ident(_) | PropKey::Str(_) | PropKey::Num(_) => {}
        PropKey::Computed(e) => scan_expr(e, flags),
    }
}

fn scan_pattern(p: &Pattern, flags: &mut u8) {
    match p {
        Pattern::Ident(n) => {
            if n == "arguments" {
                *flags |= SCAN_ARGUMENTS;
            }
        }
        Pattern::Array(elems) => {
            for el in elems {
                match el {
                    ArrayPatElem::Hole => {}
                    ArrayPatElem::Elem { pattern, default } => {
                        scan_pattern(pattern, flags);
                        if let Some(d) = default {
                            scan_expr(d, flags);
                        }
                    }
                    ArrayPatElem::Rest(pat) => scan_pattern(pat, flags),
                }
            }
        }
        Pattern::Object(op) => {
            for prop in &op.props {
                scan_prop_key(&prop.key, flags);
                scan_pattern(&prop.value, flags);
                if let Some(d) = &prop.default {
                    scan_expr(d, flags);
                }
            }
            if op.rest.as_deref() == Some("arguments") {
                *flags |= SCAN_ARGUMENTS;
            }
        }
        Pattern::Member(e) => scan_expr(e, flags),
    }
}

#[derive(Debug, Clone)]
pub struct Param {
    pub pattern: Pattern,
    pub default: Option<Expr>,
    pub rest: bool,
}
