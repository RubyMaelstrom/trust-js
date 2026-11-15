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

/// A frame of the graph (P3 stage 3, step 1): [`FrameIx::COMPILED`], the compiled function's, or
/// an inlined call's, `Graph::frames[k - 1]` for `FrameIx(k)`. Frames are opened outermost first,
/// so that a frame's parent precedes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct FrameIx(pub(crate) u32);

impl FrameIx {
    /// The compiled function's frame.
    pub(crate) const COMPILED: FrameIx = FrameIx(0);

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn is_compiled(self) -> bool {
        self == FrameIx::COMPILED
    }
}

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
    /// words and a typed array's data are heap facts, which end at the next effect; element
    /// words never cross a safepoint, and a typed array's data crosses only a loop header's poll
    /// in a loop that keeps heap facts (P3 stage 3 step 0c: the bytes are a Vec in the Rust heap,
    /// reachable from the cell, which the poll's collection neither moves nor frees, and its slow
    /// path runs no code that detaches, resizes or borrows a buffer); a view record is a value
    /// fact, fixed at install in a payload
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
// the tier has none (x86-64 until the plan's stage 6), nothing but tests builds those two.
#[cfg_attr(
    all(
        not(test),
        not(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        ))
    ),
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
    // The untagged phis' reasons (P3 stage 2 step 6): a loop's entry edge's speculation, which
    // exits at the loop header's pc and which a flag declines at that header.
    /// A word entering a loop's untagged phi is no Number (Maglev's hoisted untagging's
    /// CheckedNumberToFloat64).
    PhiNotNumber,
    /// A Number entering a loop's I32 phi is no exact int32 (CheckedFloat64ToInt32 there).
    PhiNotInt32,
    /// A check of a loop invariant moved to the loop's entry edge failed there (P3 stage 3 step
    /// 0; JSC's HoistingFailed exit kind, Ion's LICM bailout kind), or one of the checks of a
    /// typed array an effect-free loop accesses, made at its entry edge (step 0c): a flag at the
    /// header leaves the invariants' and the typed arrays' checks in the loop. A moved check whose
    /// replaced checks all run whenever the loop is entered flags their own sites instead (step
    /// L1c, `MovedCheck`).
    HoistingFailed,
    /// An equality whose record saw two Strings met a word that is no String (P3 stage 3 step
    /// L1; V8's kNotAString of CheckString): the flag at its pc leaves the equality the generic
    /// operation. Not kind-bound, as NotNumber is not: a value site's speculation, at any inline
    /// kind (snapshot.rs `kind_bound`).
    NotString,
    /// An equality whose record saw objects and nullish values met a word that is neither (P3
    /// stage 3 step L1b; Maglev's CheckJSReceiverOrNullOrUndefined and its
    /// kNotAJavaScriptObjectOrNullOrUndefined, maglev-graph-builder.cc:3706-3714):
    /// the flag at its pc leaves the equality the generic operation. Its own reason, not
    /// NotObject's: an Objects speculation that met null exits NotObject, and its recompile, whose
    /// record then holds both, takes this pair. Not kind-bound, as NotString is not.
    NotObjectOrNullish,
    /// A store's TypedData met a buffer an embedder observed (`NativeBuffer::observed`; P3 stage
    /// 3 step L4): the flag at its pc (the access's, or a loop header's for the entry check step
    /// 0c made there) makes the TypedData there `book`, its stores keeping the buffer's book.
    /// Its own exit, not the moved check's: the flag keeps the loop's entry checks moved. Not
    /// kind-bound: a buffer's state, as Detached is.
    ObservedBuffer,
    /// No exit: the second-level flag (P3 stage 3 step L5). A discard adds it at a site one of
    /// whose exits had a reason its code's own snapshot already held flagged there with the same
    /// data (`ExitRecord::held`): the form that flag left at the site (the one it built, or the
    /// one it left as it was) exits for the reason the flag was to end. The builder then takes
    /// the operation's generic form at the site before any reason's decline, whatever the
    /// lowering (`Builder::generic_declined`), as JSC's ArrayMode::refine ends its ladder at
    /// Generic (dfg/DFGArrayMode.cpp:202-222, 262-269, 299-327) and V8's and SpiderMonkey's ICs
    /// end megamorphic or generic, which their compilers lower with no guard to fail. Kind-bound
    /// (snapshot.rs `kind_bound`): the context's whose code kept exiting.
    Generic,
    // The last reason: a reason's discriminant is its bit in the snapshot's declined mask (a
    // u32), which snapshot.rs asserts this one's fits; a reason added after it moves the assert.
}

/// The reasons whose held exits make no site Generic (`ExitReason::generic_eligible`), as a
/// declined mask's bits: a snapshot without any other reason flagged keeps no flags by frame state
/// (`Snapshot::may_hold`).
pub(crate) const GENERIC_INELIGIBLE: u32 = 1 << ExitReason::Throw as u32
    | 1 << ExitReason::Unsupported as u32
    | 1 << ExitReason::Forced as u32
    | 1 << ExitReason::NeverRan as u32
    | 1 << ExitReason::HoistingFailed as u32
    | 1 << ExitReason::PhiNotNumber as u32
    | 1 << ExitReason::PhiNotInt32 as u32
    | 1 << ExitReason::ObservedBuffer as u32
    | 1 << ExitReason::WrongCallee as u32
    | 1 << ExitReason::Generic as u32;

// The discard's and the back end's (driver.rs, a64.rs): where the tier has none (x86-64 until the
// plan's stage 6), nothing reads them.
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    expect(dead_code, reason = "the discard's and the back end's")
)]
impl ExitReason {
    /// Whether an exit of this reason whose flag its code already held makes its site Generic
    /// (P3 stage 3 step L5): an operation's own exits. Not throws and unsupported operations
    /// (never flagged), forced exits, a never-ran block's (its flag builds the block, which then
    /// raises it no more), the exits keyed at a loop header (HoistingFailed, PhiNotNumber,
    /// PhiNotInt32, ObservedBuffer: the header's operation is not theirs, and each flag ends what
    /// raised it: the checks stay in the loop, the phis go tagged or f64, the protector makes
    /// the stores test their own buffers), and WrongCallee, whose guards' ways the policy keeps
    /// (step 4p's rule 5: a reoptimization takes the first compile's ways).
    pub(crate) fn generic_eligible(self) -> bool {
        GENERIC_INELIGIBLE & (1 << self as u32) == 0
    }

    /// Whether `data`, an exit's profile byte (`driver::flag`'s: a shape check's ways, a free
    /// name's proof, a truncation's mark), leaves the flag `held` (the byte the snapshot read at
    /// the site) as it is: a flag whose data the exit would change did change the build (more
    /// shapes, another proof, the truncation's decline). Every other reason's flag carries none.
    pub(crate) fn data_unchanged(self, held: u8, data: u8) -> bool {
        match self {
            ExitReason::WrongShape => data <= held,
            ExitReason::WrongName => data == held,
            ExitReason::NotNumber => data & !held == 0,
            _ => true,
        }
    }
}

/// The exit profile's byte for a NotNumber exit that a TruncateNumberOrOddball took (a String,
/// a Symbol, a BigInt or an object where its record saw Numbers and oddballs), where a Number
/// check's NotNumber exit keeps 0: the builder declines a bitwise operation whose record holds
/// oddballs to the generic one only after its truncation exited (`Rules::bitwise_declined`).
pub(crate) const TRUNCATION_EXIT: u8 = 1;

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
    /// The i-th result of the effect or generic node that is this node's only input (Tagged).
    Projection(u8),
    /// The word the generic node that is this node's only input left in slot `s`, which its
    /// operation writes (Tagged; P3 stage 3 step 3a): the frame holds it after the node.
    SlotResult(u16),

    // Checks: each node has a frame state (its eager exit's) and a reason.
    /// Tagged -> F64.
    CheckNumber,
    /// F64 -> I32: an exact int32 and not -0.
    CheckInt32,
    /// Tagged -> I32: ECMAScript ToInt32 (§7.1.6) of a Number, undefined, null or a Boolean
    /// (ToNumber gives NaN, 0, 0 and 1 for the last three), exiting NotNumber for any other
    /// word: a bitwise operand whose record saw those oddballs besides Numbers (Maglev's
    /// TruncateCheckedNumberOrOddballToInt32, JSC's ValueToInt32 of a NotCellNorBigIntUse).
    TruncateNumberOrOddball,
    /// Tagged -> Cell.
    CheckObject,
    /// Tagged -> Cell: an object whose shape is one of these (at most four). `ways` is the
    /// access's receiver shapes, which its exit keeps in the exit profile, where `shapes` may be
    /// fewer: those of them the receiver's facts allow (P3 stage 3 step 3f, rule K), so that the
    /// flag's data says what the site's ways were, not what the check narrowed them to.
    CheckShape {
        shapes: Vec<u32>,
        ways: u8,
    },
    /// Cell -> Cell: the input object's [[Prototype]] (its `proto` field, which no shape
    /// implies), where it is an object of an ordinary kind with IC_PLAIN set and one of
    /// `shapes` (at most four), as the templates' hop checks it (jit.rs emit_baked); else an
    /// exit [WrongShape], a null prototype included (P3 stage 3 step L2: a named load through
    /// the chain, each level reached from the one below, as SpiderMonkey's LoadProto and
    /// GuardShape and JSC's poly-proto chain walk it). `ways` is the access's receiver shapes,
    /// which its exit keeps in the exit profile as the receiver's check would.
    CheckProto {
        shapes: Vec<u32>,
        ways: u8,
    },
    /// Tagged -> Cell: the realm's String.prototype (`Interp::string_proto`, read through
    /// ctx.interp as the templates read it: jit.rs emit_load_active_proto), where the word is a
    /// String and that prototype has IC_PLAIN set and one of `shapes` (at most four); else an
    /// exit [WrongShape] (P3 stage 3 step L2: a string's method load, Maglev's CheckString and
    /// then its map's prototype, maglev-graph-builder.cc:4720-4724, compiler/access-info.cc:
    /// 1093-1100). `ways` as CheckProto's.
    CheckStringProto {
        shapes: Vec<u32>,
        ways: u8,
    },
    /// Cell -> None: a creation's receiver may take a new property (P3 stage 3 step L3), as the
    /// template's creation probe tests it (jit.rs emit_prop_create_probe): it is extensible
    /// (FLAG_EXTENSIBLE, which no shape implies) and no marked prototype (FLAG_PROTO, whose
    /// structural changes must move the creation epoch, which the VM's insert does); else an exit
    /// [WrongShape]. `ways` as CheckProto's.
    CheckCreatable {
        ways: u8,
    },
    /// Cell -> None: the object's [[Prototype]] is null, else an exit [WrongShape]: a creation's
    /// chain ends there (P3 stage 3 step L3; SpiderMonkey's ShapeGuardProtoChain ends at the
    /// last prototype, whose shape implies its null prototype). `ways` as CheckProto's.
    CheckNoProto {
        ways: u8,
    },
    /// Tagged -> Bool [NotBoolean]: an equality's operand where its record saw two Booleans (P3
    /// stage 3 step L1); the builder branches through ToBoolean.
    CheckBoolean,
    /// Cell -> Bool: whether the object's shape is one of these (at most four), with no exit: a
    /// polymorphic named load's test of one arm's receiver shapes after its CheckShape of them
    /// all (P3 stage 3 step L2; Maglev's BuildCompareMaps, maglev-graph-builder.cc:3750-).
    ShapeIs(Vec<u32>),
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
    /// F64 -> Bool: ToBoolean of a Number (ECMA-262 §7.1.2), false for +0, -0 and NaN (Maglev's
    /// Float64ToBoolean; stage 2 step 6).
    F64ToBoolean,

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
    /// Tagged -> Bool: `x == null` (ECMA-262 IsLooselyEqual against undefined or null, P3 stage
    /// 3 step L1): true for undefined and null (`tag | 2` is null's tag for those two alone, as
    /// the templates test it; the TDZ marker's, 0x7ffa, stays 0x7ffa) and for the [[IsHTMLDDA]]
    /// object (Annex B.3.6), false for
    /// every other value. An object whose ic_plain is set is no [[IsHTMLDDA]] (its makers clear
    /// it, and it is only ever cleared); one whose ic_plain is clear takes a cold call of the
    /// interpreter's test, which runs no JavaScript, allocates nothing and does not collect:
    /// SpiderMonkey's IsNullOrLikeUndefined, V8's TestUndetectable, JSC's CompareEq on KnownOther.
    NullOrLikeUndefined,
    /// Tagged -> Bool: `typeof x === "<test>"` (P3 stage 3 step L1), the interpreter's
    /// typeof_matches exactly: a primitive by its tag, an object by its kind tag and ic_plain,
    /// and a proxy, or an object that may be [[IsHTMLDDA]], by a cold call of typeof_matches
    /// (no JavaScript, no allocation, no collection), as SpiderMonkey's MTypeOfIs calls
    /// TypeOfObject out of line. Never TypeofTest::Never, which the builder folds.
    TypeofIs(crate::value::TypeofTest),
    /// (Tagged): the word is a String's [NotString] (P3 stage 3 step L1; Maglev's CheckString,
    /// maglev-graph-builder.cc:2502-2506).
    CheckString,
    /// (Tagged): the word is an object, undefined or null (`tag | 2` is null's tag for those two
    /// alone, NullOrLikeUndefined's test) [NotObjectOrNullish] (P3 stage 3 step L1b; Maglev's
    /// CheckJSReceiverOrNullOrUndefined, maglev-graph-builder.cc:3706-, JSC's ObjectOrOtherUse).
    /// The template's class test of a pair of two tags (feedback.rs EQ_CLASS_OBJECT_OR_NULLISH)
    /// also takes the TDZ marker, 0x7ffa, which this check exits for: harmless, as the marker is no
    /// equality's operand (a read of a binding in its dead zone throws first, an array's hole
    /// reads as undefined); were it one, the record would name the pair and this check exit once
    /// there and flag it.
    CheckObjectOrNullish,
    /// (Tagged, Tagged) -> Bool: IsLooselyEqual of two words each an object or nullish (P3 stage
    /// 3 step L1b), TurboFan's form (compiler/js-typed-lowering.cc:1009-1060): the same word is
    /// equal; else a nullish word equals the other exactly where that one is nullish or an
    /// [[IsHTMLDDA]] object (NullOrLikeUndefined's test, its cold call for an object of clear
    /// ic_plain), and two objects that differ are unequal.
    ObjectOrNullishEqual,
    /// (Tagged, Tagged) -> Bool: IsStrictlyEqual of two proven Strings (P3 stage 3 step L1): the
    /// same cell is equal, different lengths unequal, and equal lengths a cold call of the
    /// contents' compare, which runs no JavaScript, allocates nothing and does not collect
    /// (Maglev's StringEqual: identity, the lengths, then its builtin, maglev-ir.cc:6586-6626).
    StringEqual,
    /// Tagged -> F64: the word's bits as a double, unchecked (P3 stage 3 step L1). A Number's
    /// word is its f64 and every other word is a NaN, so an F64Cmp against a Number is `===`
    /// exactly: unordered, so unequal, for every word that is no Number (the template's constant
    /// form, jit.rs emit_eq_constant). No form of the value: it is a Number only when it is one.
    WordAsF64,

    // Memory.
    /// Cell -> Tagged: the property at the way's location and slot (IcState::loc), of the
    /// receiver's cell or of a holder's a walk reached (CheckProto). The slot holds
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
    /// (Cell, Tagged): the store that creates a plain data property of key word `key` (P3 stage 3
    /// step L3) on an object whose shape, chain and flags its checks proved (CheckShape, a
    /// CheckProto per level and CheckNoProto, CheckCreatable), as the template's creation commit
    /// makes it (jit.rs emit_prop_create_probe,
    /// emit_set_prop_inline's commit): where the object keeps no key index of its own, has fewer
    /// than 8 named properties and has the key layout for its next slot (its own prediction
    /// holding `key` with plain attributes, or the cached layout of `shape`, the landing shape),
    /// else an exit [WrongShape] with nothing written; then, where the record lacks room for the
    /// slot at `location`, the record grown by a pure call (`value::jit_grow_named_record`: it
    /// allocates and neither collects nor runs JavaScript); then the layout, the value at
    /// `location`, the card mark unless the value is no reference, `shape` and the count, with
    /// no call between. `ways` as CheckProto's.
    StoreNew {
        location: u16,
        shape: u32,
        key: u64,
        ways: u8,
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
    /// Tagged -> I32: a String's `length` (P3 stage 3 step L3): the word a String, else an exit
    /// [WrongShape], then its header's length, the count of its UTF-16 code units for every
    /// string, a flat buffer of either width or a view (lstr.rs unit_len), at most 2^30. A
    /// String's own `length` is non-writable and non-configurable (ECMA-262 10.4.3.5): no code
    /// can intervene. Maglev's StringLength, an int32 (maglev-reducer-inl.h:5198-5205); Warp's
    /// MStringLength; JSC's GetArrayLength on a String.
    StringLength,
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
    /// only Rust detaches, resizes or borrows a buffer, which a loop without effects keeps through
    /// its header and makes at its one entry edge where no entry holds it (P3 stage 3 step 0c;
    /// its exit there HoistingFailed). The element kind's check precedes it,
    /// which a declined view, whose buffer record is null, fails (its kind byte is no kind's).
    /// For a store and not `book` it tests too that no embedder observed the buffer
    /// (`NativeBuffer::observed`), a further exit of its own reason [ObservedBuffer], and the
    /// stores it serves keep no book (P3 stage 3 step L4): only a host function can set the flag,
    /// through the embed API, so the test holds for the epoch as the buffer's attachment does,
    /// as V8's typed stores keep no detach or immutability test while its ArrayBufferDetaching
    /// and ArrayBufferMutable protectors hold (js-native-context-specialization.cc:4093-4105).
    /// `book`, once an embedder observed some buffer of the Agent (the snapshot's protector) or
    /// where that exit's flag declined the test at the node's pc: no test here, and each store it
    /// serves tests its buffer's flag itself and keeps the book where it is set, as V8's code
    /// carries the detach test at each access once its protector is invalid.
    TypedData {
        write: bool,
        book: bool,
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
    /// then, where its data's TypedData is `book` (or is no TypedData), a test of the buffer's
    /// `observed` byte and, where it is set, the buffer's dirty range and write count out of line,
    /// kept for embedders as the template keeps them (jit_typed_array.rs emit_dirty); where its
    /// TypedData tested that no embedder observed the buffer, nothing.
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
    /// with a handler, and by the forced-exit gate). The operations that need the whole frame
    /// (P3 stage 3 step 3a): those that may read any slot (`liveness::SlotUse::Any`) and those
    /// whose helper runs the VM on a copy of the operand stack (`bytecode::jit_runs_on_vm_copy`).
    Effect {
        pc: u32,
        pops: u8,
        pushes: u8,
    },
    /// Any other operation the tier does not lower, as the tier's own node (P3 stage 3 step 3a;
    /// the owner's D6, explicit operands): inputs are its popped operands, then the values of the
    /// slots it names (`liveness::SlotUse`) whose frame words are stale (the slots its state holds
    /// as values), in their order, each a word or a value the code boxes into one; its results are
    /// the Projections that follow it, the slots it writes the SlotResults after them; its state
    /// is the frame state before it, as an effect's. It reads and writes no other slot, so a slot
    /// it does not name keeps its value and its place across it. Its lowering runs the operation's
    /// baseline template on the canonical operand words at its pc's depth, the named slots'
    /// words written first.
    Generic {
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
                | NodeOp::TruncateNumberOrOddball
                | NodeOp::CheckObject
                | NodeOp::CheckShape { .. }
                | NodeOp::CheckProto { .. }
                | NodeOp::CheckStringProto { .. }
                | NodeOp::CheckCreatable { .. }
                | NodeOp::CheckNoProto { .. }
                | NodeOp::StoreNew { .. }
                | NodeOp::StringLength
                | NodeOp::CheckBoolean
                | NodeOp::CheckCallee { .. }
                | NodeOp::CheckInitialized
                | NodeOp::CheckString
                | NodeOp::CheckObjectOrNullish
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
                | NodeOp::Generic { .. }
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
        matches!(
            self,
            NodeOp::Effect { .. } | NodeOp::Generic { .. } | NodeOp::F64Mod
        )
    }

    /// Whether this node reads the compiled function's own context (stage 3): its canonical
    /// slots and operand words, `this`, running environment, chunk (an effect's template, a
    /// splice's guard data) or its return. A node of an inlined frame reads none of it: the
    /// frame's values are explicit (its states, its descriptor's callee and receiver). A generic
    /// node of an inlined frame, its projections and slot results read their own frame's: its
    /// lowering writes its explicit inputs where its template reads them, the frame's chunk,
    /// position, `this`, environment and named slots among them, and the compiled frame's back
    /// after it (P3 stage 3 step 3c).
    pub(crate) fn reads_compiled_frame(&self) -> bool {
        matches!(
            self,
            NodeOp::EntrySlot(_)
                | NodeOp::FrameSlot(_)
                | NodeOp::StoreSlot(_)
                | NodeOp::This
                | NodeOp::Env
                | NodeOp::Effect { .. }
                | NodeOp::CheckCallee { .. }
                | NodeOp::CalleeMatches { .. }
                | NodeOp::Return
        )
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
    /// The bytecode pc this node was built for, a pc of its frame's chunk.
    pub(crate) pc: u32,
    pub(crate) block: BlockIx,
    /// The frame whose operation the node was built for (P3 stage 3, step 1): with `pc`, the
    /// node's origin, as JSC's DFG node carries its CodeOrigin, a bytecode index and the inline
    /// call frame it belongs to (bytecode/CodeOrigin.h). A node of an inlined frame reads only
    /// values: the compiled frame's slots, `this`, environment and templates are not its own.
    pub(crate) frame: FrameIx,
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

/// The bytecode state of one frame at a pc, as an exit rebuilds it. A state of an inlined frame
/// chains its frame's parent state ([`InlinedFrame::parent`]), and so on to the compiled
/// function's: an exit rebuilds every frame of the chain (P3 stage 3, step 1).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameState {
    /// The pc execution resumes at: the failing operation's own pc for a check before it, the next
    /// operation's after an effect; a pc of the frame's chunk.
    pub(crate) pc: u32,
    /// One source per slot of the frame's chunk. An inlined frame has no frame of its own while
    /// the code runs (the owner's "explicit operands", plan:3538-3545): its slots are values or
    /// Dead, never `Frame`.
    pub(crate) slots: Box<[SlotSource]>,
    /// Every live operand-stack position, bottom first (all Tagged-representable nodes).
    pub(crate) stack: Box<[NodeIx]>,
    /// Handler records active at the pc, innermost last: (push pc, operand depth at the push).
    pub(crate) handlers: Box<[(u32, u32)]>,
    /// The frame the state describes.
    pub(crate) frame: FrameIx,
}

/// An inlined call's frame (P3 stage 3, step 1): Maglev's DeoptFrame with its parent
/// (maglev/maglev-ir.h:1455-1537), JSC's InlineCallFrame (bytecode/InlineCallFrame.h). The
/// callee runs from its own first-stage chunk; its states name this frame, and an exit through it
/// rebuilds the caller from `parent`, one record per frame (`opt::exit`). Shapes beyond the
/// splice's callee set come with step 9: where the actual arguments' count differs from the
/// callee's parameters, the frame keeps them here too, as Maglev's InlinedArgumentsDeoptFrame
/// sits between a call and its callee's frame (maglev-graph-builder.cc:1025-1040).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InlinedFrame {
    /// The callee's first-stage chunk: its index among the compile's inlined chunks (the job's
    /// views of them, the code's held chunks), shared by the frames of one callee's calls.
    pub(crate) chunk: u32,
    /// The caller's state at the call, one per inlined call and shared by every state of this
    /// frame (Maglev's GetCallerDeoptFrame, maglev-graph-builder.cc:1063-1066): its frame (an
    /// earlier one), resuming at `call_pc + 1` without the call's operands, where an exit pushes
    /// the call's result, as Maglev's caller frame "resumes after the call" with its accumulator
    /// dead (maglev-graph-builder.cc:1042-1062); or at `call_pc` with the callee's throw.
    pub(crate) parent: StateIx,
    /// The call's pc in the parent's chunk: an Op::Call or an Op::CallWithThis.
    pub(crate) call_pc: u32,
    /// The callee as the call met it (Tagged): a node, or a constant under its identity guard.
    pub(crate) callee: NodeIx,
    /// The receiver the call passes (Tagged): a CallWithThis's operand, undefined for a Call. An
    /// exit binds it as the call would (OrdinaryCallBindThis); the frame's generic nodes whose
    /// kind reads `this` store it in `JitCtx::this_word` while they run (P3 stage 3 step 3c).
    pub(crate) receiver: NodeIx,
    /// The frame's inline site (`stack_trace::InlineSites`, registered by the oracle on the main
    /// thread: the callee, no position of its own, the call's position in the parent), which
    /// the frame's generic nodes publish in `Interp::call_site` while they run, as a splice's
    /// checked operations publish theirs (P3 stage 3 step 3c; JSC's code origin stored before
    /// each call out, dfg/DFGJITCompiler.h:133-143); NO_POSITION where none was registered.
    pub(crate) position: u32,
    /// The callee's running environment (`Env::as_ptr` of the closure's [[Environment]], a
    /// constant under the guard), which the frame's generic nodes whose kind reads the
    /// environment store in `JitCtx::env_raw` while they run (step 3c); 0 where none is known.
    pub(crate) env: usize,
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
    /// The frame whose chunk `pc` is of (P3 stage 3, step 2: an inlined callee's site).
    pub(crate) frame: FrameIx,
}

/// An access whose receiver's facts kept `kept` of its `of` receiver shapes, in its check
/// (`arms` false) or in a polymorphic load's arms (`arms` true) (P3 stage 3 step 3f, rule K; 0
/// kept: the facts allowed none, and the check of them all always exits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Narrowed {
    pub(crate) pc: u32,
    pub(crate) frame: FrameIx,
    pub(crate) arms: bool,
    pub(crate) kept: u8,
    pub(crate) of: u8,
}

/// A check moved to a loop's one entry edge, exiting at the header's pc [HoistingFailed] (P3
/// stage 3 steps 0, 0c and 3a), and the checks in the loop it stands for (step L1c): each one's
/// operation, whose ordinary exit the moved check's exit is where every replaced check runs
/// whenever the loop is entered (JSC's control equivalence: `phis::runs_whenever_entered`), as
/// JSC's hoisted node keeps its semantic origin and is marked wasHoisted only where its hoist adds
/// a blind speculation (dfg/DFGLICMPhase.cpp:252-269, dfg/DFGOSRExitBase.cpp:38-60). The back end
/// keeps the sites in the exit table (`ExitTable::moved`), which the discard reads
/// (`driver::flag_moved`); nothing of them is in the generated code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MovedCheck {
    pub(crate) node: NodeIx,
    pub(crate) sites: Vec<MovedSite>,
}

/// A check a moved check stands for: its operation's pc and frame as that check's own exit record
/// would have them (its frame state's pc and frame, from which a64.rs `exit` makes the record; the
/// builder makes a check's state at the check's own pc, which a debug assertion holds), its
/// reason, and whether its block runs whenever the loop is entered (`anyway`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MovedSite {
    pub(crate) pc: u32,
    pub(crate) reason: ExitReason,
    pub(crate) frame: FrameIx,
    pub(crate) anyway: bool,
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
    /// The accesses whose receiver's facts narrowed their shapes or arms (P3 stage 3 step 3f,
    /// rule K), in the order it built them: the log's.
    pub(crate) narrowed: Vec<Narrowed>,
    /// The inlined frames, outermost first: `FrameIx(k)` is `frames[k - 1]` (stage 3).
    pub(crate) frames: Vec<InlinedFrame>,
    /// The checks moved to loops' entry edges and the checks each stands for (P3 stage 3 step
    /// L1c), by node, ascending.
    pub(crate) moved: Vec<MovedCheck>,
    /// The exit profile's flags the snapshots held at each frame state's site (P3 stage 3 step
    /// L5): (frame, pc) -> (reasons, data byte), from the frame's snapshot as the builder read it,
    /// so that an exit record knows whether its own reason was already flagged there.
    pub(crate) held: crate::fasthash::FastMap<(FrameIx, u32), (u32, u8)>,
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

    /// Appends a node of the compiled function's frame to `block`'s body (a phi goes to its phis).
    pub(crate) fn add_node(
        &mut self,
        block: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
    ) -> NodeIx {
        self.add_node_in(FrameIx::COMPILED, block, op, inputs, rep, pc)
    }

    /// Appends a node of `frame`, built for its chunk's operation at `pc`, to `block`'s body (a
    /// phi goes to its phis).
    pub(crate) fn add_node_in(
        &mut self,
        frame: FrameIx,
        block: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
    ) -> NodeIx {
        let is_phi = op == NodeOp::Phi;
        let ix = self.new_node(block, op, inputs, rep, pc);
        self.nodes[ix.index()].frame = frame;
        let data = &mut self.blocks[block.index()];
        if is_phi {
            data.phis.push(ix);
        } else {
            data.body.push(ix);
        }
        ix
    }

    /// A node of `block` that its lists do not hold yet: the caller places it (a pass that
    /// inserts nodes amid a block's, the phi selector's).
    pub(crate) fn new_node(
        &mut self,
        block: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
    ) -> NodeIx {
        let ix = NodeIx(self.nodes.len() as u32);
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            rep,
            state: None,
            reason: None,
            pc,
            block,
            frame: FrameIx::COMPILED,
        });
        ix
    }

    pub(crate) fn add_state(&mut self, state: FrameState) -> StateIx {
        let ix = StateIx(self.states.len() as u32);
        self.states.push(state);
        ix
    }

    /// Opens an inlined call's frame (stage 3): its parent state is the caller's (`frame`'s
    /// `parent`, added first), and the callee's nodes and states name the frame returned.
    pub(crate) fn open_frame(&mut self, frame: InlinedFrame) -> FrameIx {
        self.frames.push(frame);
        FrameIx(self.frames.len() as u32)
    }

    /// Inlined frame `frame`'s descriptor; None for the compiled function's frame.
    pub(crate) fn inlined(&self, frame: FrameIx) -> Option<&InlinedFrame> {
        frame.0.checked_sub(1).map(|k| &self.frames[k as usize])
    }

    /// The values an exit of a state of `frame` reads besides its state's own: those of every
    /// inlined frame from `frame` out, its callee, its receiver and its parent state's values
    /// (stage 3). Empty for the compiled function's frame.
    pub(crate) fn chain_values(&self, mut frame: FrameIx) -> Vec<NodeIx> {
        let mut values = Vec::new();
        while let Some(inlined) = self.inlined(frame) {
            values.push(inlined.callee);
            values.push(inlined.receiver);
            let parent = self.state(inlined.parent);
            values.extend(parent.stack.iter().copied());
            values.extend(parent.slots.iter().filter_map(|source| match source {
                SlotSource::Value(value) => Some(*value),
                SlotSource::Frame | SlotSource::Dead => None,
            }));
            frame = parent.frame;
        }
        values
    }

    /// Each value an exit with `state` reads, in order: the state's operand stack, its slots'
    /// values, then its frame's chain's (as `chain_values` lists them, without a vector). The
    /// register allocator keeps each homed at the exiting node (`regalloc`'s reads), and a call on
    /// the node's own path keeps each where its exit after the call finds it (`a64`'s
    /// `save_live_clobbered`).
    #[cfg_attr(
        not(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        )),
        expect(dead_code, reason = "the back end's")
    )]
    pub(crate) fn for_each_exit_value(&self, state: StateIx, mut f: impl FnMut(NodeIx)) {
        let mut state = self.state(state);
        loop {
            state.stack.iter().copied().for_each(&mut f);
            for source in state.slots.iter() {
                if let SlotSource::Value(value) = *source {
                    f(value);
                }
            }
            let Some(inlined) = self.inlined(state.frame) else {
                return;
            };
            f(inlined.callee);
            f(inlined.receiver);
            state = self.state(inlined.parent);
        }
    }

    /// Whether a handler record is active in `state`'s frame or any frame of its chain, so that a
    /// throw there exits to the VM rather than leaving the function (stage 3).
    #[cfg_attr(
        not(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        )),
        expect(dead_code, reason = "the back end and its tests use it")
    )]
    pub(crate) fn chain_has_handlers(&self, state: StateIx) -> bool {
        let mut state = self.state(state);
        loop {
            if !state.handlers.is_empty() {
                return true;
            }
            match self.inlined(state.frame) {
                Some(inlined) => state = self.state(inlined.parent),
                None => return false,
            }
        }
    }

    /// What the throw exit of generic node `node` rebuilds from the homes of its values, where
    /// its state has a handler (P3 stage 3 step 3a, throws by recipe): the state's slots that
    /// hold values but those the node's operation writes (its SlotResults: the frame holds what
    /// its template left there, written or not), and for each operand-stack position of the
    /// state its value below the node's operands and the innermost handler's depth, which the VM
    /// keeps, or None above them, where the frame's word stays as the template left it (the VM
    /// keeps the stack to the smaller of the state's depth and the handler's, as it did when the
    /// code wrote the whole state to the frame before the node). The exit reads every value here
    /// after the node's call, which the allocator keeps them across. Empty for any other node.
    pub(crate) fn throw_recipe(&self, node: NodeIx) -> (Vec<(u16, NodeIx)>, Vec<Option<NodeIx>>) {
        let data = self.node(node);
        let (NodeOp::Generic { pops, .. }, Some(state)) = (&data.op, data.state) else {
            return (Vec::new(), Vec::new());
        };
        let state = self.state(state);
        let Some(&(_, handler_depth)) = state.handlers.last() else {
            return (Vec::new(), Vec::new());
        };
        let written: Vec<u16> = self
            .block(data.block)
            .body
            .iter()
            .filter_map(|&n| match self.node(n).op {
                NodeOp::SlotResult(slot) if self.node(n).inputs[..] == [node] => Some(slot),
                _ => None,
            })
            .collect();
        let slots = state
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, source)| match *source {
                SlotSource::Value(value) if !written.contains(&(slot as u16)) => {
                    Some((slot as u16, value))
                }
                _ => None,
            })
            .collect();
        let kept = (state.stack.len() - usize::from(*pops)).min(handler_depth as usize);
        let stack = state
            .stack
            .iter()
            .enumerate()
            .map(|(k, &value)| (k < kept).then_some(value))
            .collect();
        (slots, stack)
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
