//! The optimizing tier's IR (P3 stage 1; the plan's "Stage 1" record and Design point 4).
//!
//! A control-flow graph of basic blocks, each a list of SSA nodes ending in one terminator.
//! Values carry a machine representation; checks are explicit nodes, each with the frame state
//! its eager exit rebuilds; effectful nodes run an operation's baseline template on the canonical
//! frame. The builder (build.rs) produces a graph from a chunk and its snapshot, the verifier
//! (verify.rs) checks it, and the back end (regalloc.rs, a64.rs) lowers it. This module is the
//! contract between them and holds no policy.

/// A block's index in [`Graph::blocks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct BlockIx(pub(crate) u32);

/// A node's index in [`Graph::nodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NodeIx(pub(crate) u32);

/// A frame state's index in [`Graph::states`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StateIx(pub(crate) u32);

impl BlockIx {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

impl NodeIx {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

impl StateIx {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// A value's machine representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Rep {
    /// A NaN-boxed word (value.rs: a Number is its f64 bits, other kinds are tags in the NaN
    /// space), in an x register.
    Tagged,
    /// An unboxed double, in a d register.
    F64,
    /// An unboxed int32, in a w register.
    I32,
    /// 0 or 1, in a w register.
    Bool,
    /// An object's payload pointer, proven by a check, in an x register.
    Cell,
    /// An untagged address inside an object's storage (P3 stage 2 step 2): an element block's
    /// first word, a typed array's view record or its data, in an x register. Only element nodes
    /// take one, each with its owner's cell, which keeps the storage alive while the address is
    /// used (Warp's keep-alive: a storage block a frame names keeps its memory, not its owner or
    /// the values it holds, gc_storage.rs). No frame state, phi or exit recipe holds one. Element
    /// words and a typed array's data are heap facts, which end at the next effect, so neither
    /// address crosses a safepoint; a view record is a value fact, fixed at install in a payload
    /// cell that lives as long as its typed array, so its address may (the collector moves
    /// nothing).
    Ptr,
    /// An Environment Record's address (`Env::as_ptr`, stage 2 step 4), in an x register. Records
    /// never move and the running activation's chain keeps every record it reaches alive, so an
    /// environment lives across effects within an activation (a value fact, unlike step 2's
    /// addresses inside an object's storage); it is no language value, so no frame state, phi or
    /// exit recipe holds one.
    Env,
    /// No value: stores, flushes, terminators.
    None,
}

/// A comparison's condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Cond {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// What an element check proves of an object (P3 stage 2 step 2): its kind tag and IC_PLAIN flag.
/// No write changes either once JavaScript can see the object, so the proof is a value fact, kept
/// across effects as a cell form is. The contract it relies on: every kind is set on a fresh
/// object (install_typed_array and install_array_buffer assert it; errors, arguments, RegExps,
/// collections, proxies and callables are made with theirs), and IC_PLAIN is only ever cleared,
/// at a creation or a registration on a fresh object (interpreter.rs's object_look). The one path
/// that moves a kind onto an existing object, run_constructor_on's native arm (eval.rs, with
/// move_collection), is unreachable: its callers pass user constructors, and built-in subclass
/// instances are made by the native constructors (construct_native_dispatch). A host's indexed
/// properties need a fresh object by their API's contract (install_readonly_indexed_properties),
/// which the code checks only in part; the cold paths that read undefined for an index the words
/// lack test IC_PLAIN again, so that a breach exits there rather than reading undefined
/// (`emit_no_inherited_elements`), the one read such an object's empty words could answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ElemKind {
    /// Element words: an ordinary kind or an Array (its tag at most KIND_ARRAY_TAG), IC_PLAIN set
    /// (the element templates' receiver test, jit.rs emit_get_elem_inline).
    Words,
    /// An Array, IC_PLAIN set.
    Array,
    /// A typed array (its tag; its IC_PLAIN is clear, which keeps the words' paths off it), of
    /// the element kind given, its view's kind byte (Maglev's map check, a map per elements
    /// kind): the view is fixed at install as well (native_typed_array.rs install_typed_array).
    Typed(Option<TypedKind>),
}

impl ElemKind {
    /// Whether a proof of `self` proves `other` too.
    pub(crate) fn proves(self, other: ElemKind) -> bool {
        self == other
            || matches!(
                (self, other),
                (ElemKind::Array, ElemKind::Words)
                    | (ElemKind::Typed(Some(_)), ElemKind::Typed(None))
            )
    }
}

/// A typed array's element kind, as its native view numbers it (jit_typed_array.rs; feedback.rs
/// ELEM_TYPED_KINDS).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TypedKind {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    F64 = 7,
    U8Clamped = 8,
}

impl TypedKind {
    pub(crate) const ALL: [TypedKind; 9] = [
        TypedKind::U8,
        TypedKind::I8,
        TypedKind::U16,
        TypedKind::I16,
        TypedKind::U32,
        TypedKind::I32,
        TypedKind::F32,
        TypedKind::F64,
        TypedKind::U8Clamped,
    ];

    /// The representation of a load's element and a store's value: an int32 for the integer
    /// kinds (a load of Uint32 excepted, whose element can pass the int32 range), an f64 for
    /// the floats.
    pub(crate) fn rep(self, store: bool) -> Rep {
        match self {
            TypedKind::F32 | TypedKind::F64 => Rep::F64,
            TypedKind::U32 if !store => Rep::F64,
            _ => Rep::I32,
        }
    }
}

/// What an element load does with a hole word (Maglev's CheckNotHole and ConvertHoleToUndefined,
/// maglev-graph-builder.cc:5233-5268).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Holes {
    /// Exits [Hole]: the record saw no hole.
    Exit,
    /// Reads undefined where the cold path proves that no element can come from the prototype
    /// chain (CanTreatHoleAsUndefined, maglev-graph-builder.cc:3993-4008), and exits [Hole]
    /// otherwise.
    Undefined,
}

/// What an element store knows of the element it writes (P3 stage 2 step 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Store {
    /// Below the used length and no hole, as a load or store of the epoch proved.
    Present,
    /// An existing element: an index at or past the used length exits [OutOfBounds], a hole
    /// [Hole].
    Existing,
    /// An existing element, or an Array's hole or next word, which the cold path fills or
    /// appends as the template's in-line set does (jit.rs emit_word_create): an extensible Array
    /// that is no marked prototype, with no index among its named entries [WrongElements], the
    /// realm's Array.prototype and the elements protector [WrongElements], a writable `length`
    /// where the index reaches it [WrongElements], room made by a pure call where there is none
    /// [CouldNotGrow] (Warp's addDenseElementPure, Maglev's MaybeGrowFastElements), the used
    /// length and `length` raised (Maglev's UpdateJSArrayLength); a gap past the next word
    /// [OutOfBounds].
    Create,
}

/// Why a check fails or an exit is taken, for the log, the exit profile (stage 2; keyed by the
/// reason's discriminant, `opt::profile`) and tests.
// Only the back end builds the reasons of exits on Booleans and of the forced-exit gate: where
// the tier has none (x86-64 until the plan's stage 6), nothing builds those two.
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    expect(dead_code, reason = "the back end builds NotBoolean and Forced")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ExitReason {
    NotNumber,
    NotInt32,
    Overflow,
    // A check's -0 exits with its node's reason (NotInt32, Overflow), one reason per check, as
    // Maglev's CheckedTruncateFloat64ToInt32 and Int32MultiplyWithOverflow give theirs.
    #[expect(
        dead_code,
        reason = "-0 exits give their check's reason, as V8's Maglev does"
    )]
    MinusZero,
    NotObject,
    WrongShape,
    NotBoolean,
    /// The inline guard of a spliced callee failed (the callee or the Realm changed).
    WrongCallee,
    /// A block whose named and call sites never ran.
    NeverRan,
    /// An operation the tier does not lower at this site (a TailCall that runs, a return inside an
    /// iterator region): the VM performs it.
    Unsupported,
    /// A throw at a pc with a handler: the VM runs the handler.
    Throw,
    /// TRUST_JS_OPT_EXIT_EVERY.
    Forced,
    // The element accesses' reasons (P3 stage 2 step 2), as Maglev's deopts name them.
    /// An element access's receiver is not of the elements kind its check expected (another
    /// object kind, or IC_PLAIN clear), or a write meets frozen elements or a receiver the
    /// in-line append does not take (V8's kWrongMap).
    WrongElements,
    /// An element index at or past the used words, or a negative one (V8's kOutOfBounds).
    OutOfBounds,
    /// A hole where the record saw none, or one the cold path cannot prove reads undefined
    /// (V8's kHole).
    Hole,
    /// A typed array whose buffer is detached, shrunk below the view or borrowed by Rust (V8's
    /// kArrayBufferWasDetached).
    Detached,
    /// An append whose growth failed (V8's kCouldNotGrowElements).
    CouldNotGrow,
    // The free names' reasons (P3 stage 2 step 4).
    /// A free name's proof failed: the realm's global environment or object is not the
    /// compile's, the global property epoch moved, or the global declarative record changed.
    /// Kind-bound, as WrongShape is: an IC's status.
    WrongName,
    /// A lexical binding read or written in its temporal dead zone: the VM runs the operation
    /// again and throws its ReferenceError (SpiderMonkey's LexicalCheck bailout, JSC's
    /// CheckNotEmpty exit).
    Tdz,
    // The last reason: a reason's discriminant is its bit in the snapshot's declined mask (a
    // u32), which snapshot.rs asserts this one's fits; a reason added after it moves the assert.
}

/// What a node does. Inputs are in [`Node::inputs`] in the order each variant names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NodeOp {
    // Values.
    /// A tagged constant word (a chunk constant, undefined, a Boolean).
    ConstTagged(u64),
    /// An f64 constant, by its bits.
    ConstF64(u64),
    ConstI32(i32),
    ConstBool(bool),
    /// The canonical frame's slot as the function was entered (Tagged).
    EntrySlot(u16),
    /// A slot reloaded from the frame after an effect (Tagged).
    FrameSlot(u16),
    /// The frame's `this` (ctx.this_word; Tagged).
    This,
    /// One input per predecessor of its block, in predecessor order.
    Phi,
    /// The i-th result of the effect that is this node's only input (Tagged).
    Projection(u8),

    // Checks: each node has a frame state (its eager exit's) and a reason.
    /// Tagged -> F64.
    CheckNumber,
    /// F64 -> I32: an exact int32 and not -0.
    CheckInt32,
    /// Tagged -> Cell.
    CheckObject,
    /// Tagged -> Cell: an object whose shape is one of these (at most four).
    CheckShape(Vec<u32>),
    /// Tagged -> Bool.
    // Stage 1's builder branches through ToBoolean and builds no CheckBoolean; the verifier and
    // the back end take one, as the tests do.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "stage 1's builder builds no CheckBoolean")
    )]
    CheckBoolean,
    /// A spliced callee's guard (InlineGuard): inputs the callee word (Tagged), then, when the
    /// InlineTarget's `check_this` is set, the receiver word (Tagged), which must be an object;
    /// the expected callee cell, its closure environment and the planning Realm's global
    /// environment are the chunk's InlineTarget.
    CheckCallee {
        target: u32,
    },
    /// The test of a guard whose WrongCallee exits the profile flagged (stage 2): inputs as
    /// CheckCallee's; true where the callee, its environment, the Realm and the receiver are as
    /// the guard expects, without an exit (-> Bool), so that the code branches to the splice or
    /// to the guard's mismatch, the generic call.
    CalleeMatches {
        target: u32,
    },

    // Conversions (no exit).
    I32ToF64,
    /// F64 -> I32: ECMAScript ToInt32 (§7.1.6), modulo 2^32, 0 for NaN and the infinities: the
    /// operands of the bitwise operations (Maglev's TruncateFloat64ToInt32).
    TruncateToInt32,
    /// NaN canonicalized, as emit_exec_number_store does.
    F64ToTagged,
    I32ToTagged,
    BoolToTagged,
    /// Tagged -> Bool: ECMAScript ToBoolean (§7.1.2) of the word, false for undefined, null,
    /// false, +0, -0, NaN, the empty String, 0n and the [[IsHTMLDDA]] object and true for every
    /// other value. It runs no JavaScript, does not collect and cannot fail: the builder branches
    /// on it, and lowers `!` through it, where the value is not proven a Boolean (CheckBoolean
    /// would exit on every other value, as `if (object)` has one).
    ToBoolean,

    // Arithmetic. The I32 forms with a state exit on overflow (Mul also on a -0 result).
    I32Add,
    I32Sub,
    I32Mul,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32Sar,
    /// `>>>`: exits when the result does not fit an int32.
    I32UShr,
    /// `>>>` -> F64: the uint32 result as a Number, where it may not fit an int32 (Maglev's
    /// Int32ShiftRightLogical "returns a Uint32").
    I32UShrToF64,
    I32Not,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    /// A call to the runtime's fmod.
    F64Mod,
    F64Neg,
    /// -> Bool.
    I32Cmp(Cond),
    /// -> Bool; false for NaN operands except under Ne.
    F64Cmp(Cond),
    /// Tagged === Tagged on words whose kinds make identity the answer (both proven Booleans,
    /// both proven objects, a proven constant of undefined or null) -> Bool.
    TaggedSame,
    BoolNot,

    // Memory.
    /// Cell -> Tagged: the property at the way's location and slot (IcState::loc). The slot holds
    /// a data property's word, never a property-only one: a read keyed `prototype`, whose slot
    /// can hold a function's deferred prototype (value.rs's PACK_LAZY_PROTO), stays an effect.
    LoadField {
        location: u16,
        slot: u32,
    },
    /// (Cell, Tagged): a store to a writable data property's slot, with a card mark when the
    /// value may be a reference.
    StoreField {
        location: u16,
        slot: u32,
    },
    /// (Tagged): writes the slot to the frame (a flush the builder places before effects and
    /// before exits that need the frame current).
    StoreSlot(u16),

    // Elements (P3 stage 2 step 2): Maglev's element access (TryBuildElementAccess,
    // maglev-graph-builder.cc:5496-5624) on TRust's element words, which are SpiderMonkey's
    // ObjectElements (word_block.rs): bounds on the used (initialized) length, a hole test per
    // element, each word checked at its use, as Warp loads dense elements.
    /// (Cell): the object's elements are of the kind [WrongElements] (the templates' kind-tag and
    /// IC_PLAIN test; Maglev's CheckMaps). Its proof is a value fact (ElemKind).
    CheckElements(ElemKind),
    /// Cell -> Ptr: the object's element words, its map's pointer, which is never null (an
    /// object without elements points past the static empty block, whose used length is 0).
    LoadElements,
    /// Ptr -> I32: the words' used length (the header's first u32).
    LoadElementsUsed,
    /// F64 -> I32: an element index, an exact integer in the int32 range with -0 taken as 0
    /// (Warp's GuardToInt32Index: "ToPropertyKey(-0) is "0""), [NotInt32] otherwise.
    CheckIndex,
    /// (I32 index, I32 length): index < length, unsigned, so that a negative index fails too
    /// [OutOfBounds] (Maglev's CheckInt32Condition(kUnsignedLessThan)).
    CheckBounds,
    /// (Cell, Ptr, I32 index) -> Tagged: the word at an index below the used length, its hole as
    /// `Holes` says [Hole]. The cell keeps the words alive while their address is used.
    LoadElement(Holes),
    /// (Cell, Ptr, I32 index, I32 used) -> Tagged: a read whose record saw indices past the used
    /// length (Maglev's Select under kHandleOOB, maglev-graph-builder.cc:5274-5286; Warp's
    /// LoadDenseElementHoleResult): below used, LoadElement(Undefined); at or past it, undefined
    /// under the same proof, a negative index exiting [OutOfBounds] (Warp's negative-index bail).
    LoadElementOrUndefined,
    /// (Cell, Ptr, I32 index, value I32 | F64 | Tagged) -> Ptr: a store to an existing element
    /// (Maglev's StoreFixedArrayElement and StoreFixedDoubleArrayElement), which keeps the kind
    /// bits exact in place where V8 transitions the map: an int32 is SCVTF and STR D, kind bits
    /// untouched while INT32 holds; an f64 its NaN canonicalized (any other NaN reads as a tag)
    /// and INT32 kept only for an exact int32; any other word clears NUMBERS and INT32 and marks
    /// the owner's card, and below a large block's traced length its dirty bit by a pure call
    /// (Warp's post barrier for large arrays). A copy-on-write block is copied by a pure call
    /// first (Maglev's EnsureWritableFastElements), frozen elements exit [WrongElements]. Its
    /// output is the object's words after the store: a copy's, a grown block's, or the input's.
    StoreElement(Store),
    /// Cell -> Tagged: an Array's `length`, its first slot's word, a Number (ArrayCreate defines
    /// it first and nothing removes it: jit_storage.rs first_slot_addr), under an Array check, so
    /// that `i < a.length` loops keep their bodies free of effects (Maglev reads a JSArray's
    /// length as a known property, maglev-reducer-inl.h:1038-1059).
    LoadArrayLength,
    /// Cell -> Ptr: a typed array's view record, fixed at install: a value fact.
    TypedView,
    /// Ptr -> I32: the view's element count, 0 for a view the code declines (BigInt or Float16
    /// elements, a shared or length-tracking buffer), so that every index fails its bounds check:
    /// a value fact (Maglev's LoadTypedArrayLength of a fixed-length array). At most 2^31, which
    /// only the unsigned bounds check reads.
    TypedLength,
    /// (Cell, Ptr view) -> Ptr: the elements' address, the buffer's bytes plus the view's offset,
    /// with the buffer attached, writable for a store, unborrowed by Rust and reaching the view's
    /// end [Detached] (Maglev's CheckTypedArrayValid, TRust having no detaching protector; the
    /// template's checks, jit_typed_array.rs emit_access): a heap fact of the effect epoch, as
    /// only Rust detaches, resizes or borrows a buffer. The element kind's check precedes it,
    /// which a declined view, whose buffer record is null, fails (its kind byte is no kind's).
    TypedData {
        write: bool,
    },
    /// (Cell, Ptr data, I32 index) -> I32 | F64: the element of `kind` at an index below the
    /// length (Maglev's Load{SignedInt,UnsignedInt,Double}TypedArrayElement): an int32, or an
    /// f64 for Uint32 and the floats, an f64's NaN as it is (its boxing canonicalizes it).
    LoadTyped(TypedKind),
    /// (Cell, Ptr data, I32 index, I32 length) -> Tagged: a read whose site exited past the
    /// view's length (Maglev's Select on BranchIfTypedArrayBounds under kHandleOOB): below the
    /// length, LoadTyped's element as a Number; at or past it, or at a negative index, undefined,
    /// as a typed array reads every index it lacks (ECMA-262 TypedArrayGetElement), with no
    /// prototype to consult.
    LoadTypedOrUndefined(TypedKind),
    /// (Cell, Ptr view, Ptr data, I32 index, value I32 | F64): the store of `kind` (Maglev's
    /// Store{Int,Double}TypedArrayElement, the value converted first): an integer kind's low bits
    /// of an int32 (a ToInt32 truncation's, or ClampToUint8's), a float's f64 (FCVT to Float32);
    /// then the buffer's dirty range and write count, kept for embedders as the template keeps
    /// them (jit_typed_array.rs emit_dirty).
    StoreTyped(TypedKind),
    /// F64 -> I32: ToUint8Clamp (ECMA-262 7.1.12): round half to even and clamp to [0, 255], NaN
    /// to 0 (FCVTNS saturates and takes NaN to 0), as the template stores Uint8Clamped.
    ClampToUint8,
    /// (Cell, I32 index[, F64 value]) -> Tagged (a load) | None (a store): an element of a typed
    /// array whose kind the snapshot does not know, through the template's TypedElement stub, a
    /// call on its register contract (jit_typed_array.rs; jit.rs SharedStub::TypedElement):
    /// [OutOfBounds] where the stub declines, whose one status does not say why.
    TypedElement {
        write: bool,
    },
    /// Cell -> Bool: whether the object is a typed array, by its kind tag, on which an element
    /// site whose record holds typed arrays and other receivers, or none, branches between the
    /// stub and the words, as the template dispatches (jit.rs emit_get_elem_inline).
    IsTypedArray,

    // Free names (stage 2, step 4): Decision 21's static addresses, as V8's context slots,
    // SpiderMonkey's aliased variables and JSC's closure variables are read.
    /// -> Env: the running environment (`ctx.env_raw`). Maglev's GetContext.
    Env,
    /// Env -> Env: a record's outer record, one load of its parent field. Maglev's PREVIOUS
    /// load, Warp's MEnclosingEnvironment, JSC's SkipScope.
    EnvParent,
    /// -> Env: a record known at the compile (a splice's holder, folded from its pinned
    /// [[Environment]]), alive while the splice's guard holds.
    ConstEnv(u64),
    /// Env -> Tagged: the binding at `slot` of the record, in place after its header (planned
    /// records never spill); the empty word while a lexical binding is in its TDZ. Maglev's
    /// LoadContextSlot, Warp's MLoadFixedSlot, JSC's GetClosureVar.
    LoadBinding {
        slot: u16,
    },
    /// (Tagged): exits [Tdz] when the word is the empty one, a binding in its TDZ. Warp's
    /// MLexicalCheck, JSC's CheckNotEmpty.
    CheckInitialized,
    /// (Env, Tagged): the word stored into the binding at `slot` of the record, a mutable one or
    /// one being initialized, with the record's card marked unless the word is no reference
    /// (the templates' `emit_cap_store`). Maglev's StoreContextSlot, Warp's SetAliasedVar, JSC's
    /// PutClosureVar.
    StoreBinding {
        slot: u16,
    },
    /// -> Cell: the running realm's global object (`ctx.global_body`, as a cell), which every
    /// global read of the code loads its property from.
    GlobalObject,
    /// (Env): the search's environment is the realm's global environment the compile resolved in
    /// [WrongName] (the templates' global-mode identity test). `proof` is the site's proof's byte
    /// (snapshot.rs's NameView::proof), which a WrongName exit keeps in the exit profile.
    CheckEnvIs {
        env: u64,
        proof: u8,
    },
    /// (Cell): the global object is the compile's [WrongName], so that its epoch and its slots
    /// are the realm's whose global environment the search checked.
    CheckCellIs {
        cell: u64,
        proof: u8,
    },
    /// (Cell): the global object's slot epoch (`Props::slot_epoch`) is the compile's
    /// [WrongName]: no property was removed, none turned accessor or back, and no lexical
    /// declaration shadows one since. SpiderMonkey's GuardGlobalGeneration.
    CheckGlobalEpoch {
        epoch: u16,
        proof: u8,
    },
    /// (Env): the global environment's record has the compile's generation [WrongName]: no
    /// binding was added, removed or moved since, so the cache's binding address is a global
    /// lexical's still (the plan's Q4: the record does not only append).
    CheckEnvGeneration {
        generation: u32,
        proof: u8,
    },
    /// Env -> Tagged: the global lexical binding at the address the cache's proof names, which
    /// the record's generation check keeps (its input, the record, orders it after the check).
    LoadGlobalLexical {
        binding: u64,
    },

    // Safepoints.
    /// A loop header's interrupt and collection poll, the first node of every loop header's
    /// body (after its phis), where the first stage polls at a backward branch's target. Its
    /// slow path may collect (the back end keeps the live values where the collector scans
    /// them); a throw from it (the live-object limit's RangeError) exits with its state when
    /// the state has a handler, and otherwise leaves the function throwing. Not effectful: it
    /// writes nothing a later node reads.
    Poll,

    // Effects.
    /// The baseline template of the operation at `pc`, on the canonical frame: inputs are the
    /// operation's popped operands (Tagged), its results are the Projections that follow it, its
    /// state is the frame state before it (the operation's own pc; used when it throws at a pc
    /// with a handler, and by the forced-exit gate).
    Effect {
        pc: u32,
        pops: u8,
        pushes: u8,
    },

    // Terminators (the last node of every block, and only there).
    Jump(BlockIx),
    /// (Bool).
    Branch {
        then_block: BlockIx,
        else_block: BlockIx,
    },
    /// (Tagged).
    Return,
    /// An unconditional exit with this node's state.
    Exit(ExitReason),
}

impl NodeOp {
    /// Whether this node ends a block.
    pub(crate) fn is_terminator(&self) -> bool {
        matches!(
            self,
            NodeOp::Jump(_) | NodeOp::Branch { .. } | NodeOp::Return | NodeOp::Exit(_)
        )
    }

    /// Whether this node can exit (and so needs a frame state).
    pub(crate) fn can_exit(&self) -> bool {
        matches!(
            self,
            NodeOp::CheckNumber
                | NodeOp::CheckInt32
                | NodeOp::CheckObject
                | NodeOp::CheckShape(_)
                | NodeOp::CheckBoolean
                | NodeOp::CheckCallee { .. }
                | NodeOp::CheckInitialized
                | NodeOp::CheckEnvIs { .. }
                | NodeOp::CheckCellIs { .. }
                | NodeOp::CheckGlobalEpoch { .. }
                | NodeOp::CheckEnvGeneration { .. }
                | NodeOp::I32Add
                | NodeOp::I32Sub
                | NodeOp::I32Mul
                | NodeOp::I32UShr
                | NodeOp::Poll
                | NodeOp::Effect { .. }
                | NodeOp::Exit(_)
                | NodeOp::CheckElements(_)
                | NodeOp::CheckIndex
                | NodeOp::CheckBounds
                | NodeOp::LoadElement(_)
                | NodeOp::LoadElementOrUndefined
                | NodeOp::StoreElement(_)
                | NodeOp::TypedData { .. }
                | NodeOp::TypedElement { .. }
        )
    }

    /// Whether running this node may run JavaScript, collect, or write anything a later node reads
    /// (heap facts die across it; values live across it need a callee-saved home).
    pub(crate) fn is_effectful(&self) -> bool {
        matches!(self, NodeOp::Effect { .. } | NodeOp::F64Mod)
    }
}

/// One SSA node.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Node {
    pub(crate) op: NodeOp,
    pub(crate) inputs: Vec<NodeIx>,
    pub(crate) rep: Rep,
    /// The frame state an exit from this node rebuilds (checks, effects, exits).
    pub(crate) state: Option<StateIx>,
    /// The exit's reason, for nodes that can exit (Exit carries its own).
    pub(crate) reason: Option<ExitReason>,
    /// The bytecode pc this node was built for.
    pub(crate) pc: u32,
    pub(crate) block: BlockIx,
}

/// Where a slot's value is when an exit rebuilds the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotSource {
    /// The frame already holds it (never written since entry or since the last reload, or
    /// flushed).
    Frame,
    /// A node holds it; the exit writes it.
    Value(NodeIx),
    /// The slot is dead at the pc (stage 2 step 6, `opt::liveness`): the VM writes it before any
    /// read, so the exit leaves the frame's word as it is.
    Dead,
}

/// The bytecode state at a pc, as an exit rebuilds it (one canonical frame; stage 3 adds the
/// parent of an inlined frame).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameState {
    /// The pc execution resumes at: the failing operation's own pc for a check before it, the next
    /// operation's after an effect.
    pub(crate) pc: u32,
    pub(crate) slots: Box<[SlotSource]>,
    /// Every live operand-stack position, bottom first (all Tagged-representable nodes).
    pub(crate) stack: Box<[NodeIx]>,
    /// Handler records active at the pc, innermost last: (push pc, operand depth at the push).
    pub(crate) handlers: Box<[(u32, u32)]>,
}

/// One basic block.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BlockData {
    /// The bytecode pc the block starts at.
    pub(crate) first_pc: u32,
    pub(crate) preds: Vec<BlockIx>,
    pub(crate) succs: Vec<BlockIx>,
    pub(crate) phis: Vec<NodeIx>,
    /// The block's nodes in order, its terminator last.
    pub(crate) body: Vec<NodeIx>,
    pub(crate) loop_header: bool,
}

/// Why the tier declines a chunk (the log names the reason and the pc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub(crate) pc: Option<u32>,
    pub(crate) reason: &'static str,
}

impl Refusal {
    pub(crate) fn at(pc: usize, reason: &'static str) -> Self {
        Refusal {
            pc: Some(pc as u32),
            reason,
        }
    }

    pub(crate) fn whole(reason: &'static str) -> Self {
        Refusal { pc: None, reason }
    }
}

/// A speculation the builder declined at a site the exit profile flagged (stage 2): the pc, the
/// flag's reason, and what the builder built instead, for the log and tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Decline {
    pub(crate) pc: u32,
    pub(crate) reason: ExitReason,
    pub(crate) form: &'static str,
}

/// A function's graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Graph {
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) states: Vec<FrameState>,
    pub(crate) n_slots: usize,
    /// The entry block (blocks[0] when built by the builder).
    pub(crate) entry: Option<BlockIx>,
    /// The speculations the builder declined, in the order it built their sites.
    pub(crate) declines: Vec<Decline>,
}

impl Graph {
    pub(crate) fn new(n_slots: usize) -> Self {
        Graph {
            n_slots,
            ..Graph::default()
        }
    }

    pub(crate) fn add_block(&mut self, first_pc: u32, loop_header: bool) -> BlockIx {
        let ix = BlockIx(self.blocks.len() as u32);
        self.blocks.push(BlockData {
            first_pc,
            preds: Vec::new(),
            succs: Vec::new(),
            phis: Vec::new(),
            body: Vec::new(),
            loop_header,
        });
        ix
    }

    /// Appends a node to `block`'s body (a phi goes to its phis).
    pub(crate) fn add_node(
        &mut self,
        block: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
    ) -> NodeIx {
        let ix = NodeIx(self.nodes.len() as u32);
        let is_phi = op == NodeOp::Phi;
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            rep,
            state: None,
            reason: None,
            pc,
            block,
        });
        let data = &mut self.blocks[block.index()];
        if is_phi {
            data.phis.push(ix);
        } else {
            data.body.push(ix);
        }
        ix
    }

    pub(crate) fn add_state(&mut self, state: FrameState) -> StateIx {
        let ix = StateIx(self.states.len() as u32);
        self.states.push(state);
        ix
    }

    /// Records an edge (predecessor lists are in the order edges are added).
    pub(crate) fn add_edge(&mut self, from: BlockIx, to: BlockIx) {
        self.blocks[from.index()].succs.push(to);
        self.blocks[to.index()].preds.push(from);
    }

    pub(crate) fn node(&self, ix: NodeIx) -> &Node {
        &self.nodes[ix.index()]
    }

    pub(crate) fn node_mut(&mut self, ix: NodeIx) -> &mut Node {
        &mut self.nodes[ix.index()]
    }

    pub(crate) fn block(&self, ix: BlockIx) -> &BlockData {
        &self.blocks[ix.index()]
    }

    pub(crate) fn state(&self, ix: StateIx) -> &FrameState {
        &self.states[ix.index()]
    }
}
