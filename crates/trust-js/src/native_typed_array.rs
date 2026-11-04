//! The state of typed arrays, ArrayBuffers and DataViews, in their objects' payloads (W2g step
//! 2, as V8, SpiderMonkey and JSC keep a view's buffer, offset and length and a buffer's backing
//! store in the object), and the guarded native views of ordinary ArrayBuffer storage that
//! generated code reads. The Data Block stays authoritative: native accesses reload the live Vec
//! header and never retain its data pointer. ECMA-262 TypedArrayGetElement/TypedArraySetElement
//! and NumericToRawBytes, local official snapshot e28783d5fc9d. Shared, immutable writes,
//! coercing and unsupported element types continue through the checked interpreter operations.

use crate::interpreter::{ArrayBufferBytes, BufferData, Interp};
use crate::value::{Gc, ObjectKind, TaInfo, TaKind};
use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::rc::Rc;

#[repr(C)]
pub(crate) struct NativeBuffer {
    pub storage_ptr: *const RefCell<Vec<u8>>,
    pub attached: Cell<u8>,
    pub writable: u8,
    pub dirty_start: Cell<usize>,
    pub dirty_end: Cell<usize>,
    pub writes: Cell<u64>,
    // Own the RefCell, rather than just its current data allocation. Detachment must not clear
    // an embedder-owned Data Block (notably Wasm memory) that shares the same Rc.
    pub storage: RefCell<Option<ArrayBufferBytes>>,
}

/// A typed array's state, its object's `ObjectKind::TypedArray` payload: the native view
/// generated code reads (the fields before `native`, at the offsets its templates use), then the
/// view's [[ViewedArrayBuffer]], [[ByteOffset]], [[ArrayLength]] and [[TypedArrayName]]
/// (`info`). A view generated code must decline (a BigInt or Float16 element type, a
/// SharedArrayBuffer's or a detached buffer's, one too long for the native range) has no guard
/// record and an element count of zero, which every access compares before it reads the record.
#[repr(C)]
pub(crate) struct NativeTypedArray {
    pub buffer_ptr: *const NativeBuffer,
    pub offset: usize,
    pub end: usize,
    pub limit: usize,
    pub kind: u8,
    pub shift: u8,
    /// The guard record `buffer_ptr` names, shared with its buffer's payload.
    pub native: Option<Rc<NativeBuffer>>,
    pub info: TaInfo,
}

/// An ArrayBuffer's or SharedArrayBuffer's state, its object's `ObjectKind::ArrayBuffer`
/// payload: the former `array_buffers`, `native_buffers`, `array_buffer_versions`,
/// `array_buffer_dirty_ranges`, `immutable_buffers` and `host_keyed_buffers` tables' entries.
/// The mutation generation and the dirty ranges change under a shared borrow of the object (they
/// hold no edge).
pub(crate) struct ArrayBufferData {
    /// [[ArrayBufferData]]: an ArrayBuffer's own bytes or a SharedArrayBuffer's Shared Data
    /// Block; `None` once the ArrayBuffer is detached (a SharedArrayBuffer never is).
    pub(crate) data: Option<BufferData>,
    /// The guard record of the buffer's native views, made with the first of them (an attached
    /// ArrayBuffer's own bytes only). Detaching clears it and its views' `attached` byte.
    pub(crate) native: Option<Rc<NativeBuffer>>,
    /// The mutation generation host embedders mirror the bytes by
    /// (`Interp::array_buffer_version`).
    pub(crate) version: Cell<u64>,
    /// Bounded, conservative byte ranges covering writes since the last host synchronization.
    pub(crate) dirty: RefCell<Vec<Range<usize>>>,
    /// Made by transferToImmutable or sliceToImmutable: its bytes can be read but never written,
    /// resized, detached or transferred.
    pub(crate) immutable: bool,
    /// Carries an embedder detach key: ECMAScript's transfer operations, which do not know that
    /// key, cannot detach it; the embedder still can (`Interp::detach_array_buffer`).
    pub(crate) host_keyed: bool,
    /// What this buffer counts in its Agent's live buffer data.
    ledger: Rc<BufferLedger>,
    /// The bytes of its own Data Block this buffer counts in `ledger`.
    counted: Cell<usize>,
}

impl ArrayBufferData {
    /// The state of a buffer over `data`, counted in `ledger`.
    pub(crate) fn new(data: BufferData, ledger: &Rc<BufferLedger>) -> Self {
        let counted = match &data {
            BufferData::Local(bytes) => bytes.borrow().len(),
            BufferData::Shared(block) => {
                *ledger.shared.borrow_mut().entry(block.id()).or_default() += 1;
                0
            }
        };
        ledger.local.set(ledger.local.get().saturating_add(counted));
        Self {
            data: Some(data),
            native: None,
            version: Cell::new(0),
            dirty: RefCell::new(Vec::new()),
            immutable: false,
            host_keyed: false,
            ledger: ledger.clone(),
            counted: Cell::new(counted),
        }
    }

    /// The bytes of an attached ArrayBuffer; `None` for a SharedArrayBuffer or a detached one.
    #[inline]
    pub(crate) fn local(&self) -> Option<&ArrayBufferBytes> {
        match &self.data {
            Some(BufferData::Local(bytes)) => Some(bytes),
            _ => None,
        }
    }

    /// The Shared Data Block of a SharedArrayBuffer.
    #[inline]
    pub(crate) fn shared(&self) -> Option<&crate::interpreter::SharedMem> {
        match &self.data {
            Some(BufferData::Shared(block)) => Some(block),
            _ => None,
        }
    }

    /// Count an ArrayBuffer's own Data Block again after its length changed (a resize).
    pub(crate) fn recount(&self) {
        let len = self.local().map_or(0, |bytes| bytes.borrow().len());
        let ledger = &self.ledger.local;
        ledger.set(
            ledger
                .get()
                .saturating_sub(self.counted.replace(len))
                .saturating_add(len),
        );
    }

    /// DetachArrayBuffer: the Data Block goes (its own bytes stay with any embedder sharing
    /// them), and so do its native guard record, generation, dirty ranges and detach key. Old
    /// native views keep the record, cleared, so generated code declines them.
    pub(crate) fn detach(&mut self) -> Option<BufferData> {
        if self.shared().is_some() {
            return None; // DetachArrayBuffer asserts IsSharedArrayBuffer(arrayBuffer) is false
        }
        if let Some(buffer) = self.native.take() {
            buffer.attached.set(0);
            // DetachArrayBuffer clears [[ArrayBufferData]]. Keep the guard metadata alive for
            // old views, but release its Data Block reference without modifying shared bytes.
            buffer.storage.borrow_mut().take();
        }
        let ledger = &self.ledger.local;
        ledger.set(ledger.get().saturating_sub(self.counted.replace(0)));
        self.version.set(0);
        self.dirty.get_mut().clear();
        self.host_keyed = false;
        self.data.take()
    }

    /// Fold the native views' writes since the last fold into the generation and the dirty
    /// ranges.
    pub(crate) fn flush_native_writes(&self) {
        let Some(buffer) = &self.native else {
            return;
        };
        if buffer.dirty_start.get() == usize::MAX {
            return;
        }
        let writes = buffer.writes.replace(0);
        let len = buffer
            .storage
            .borrow()
            .as_ref()
            .map_or(0, |storage| storage.borrow().len());
        let start = buffer.dirty_start.replace(usize::MAX).min(len);
        let end = buffer.dirty_end.replace(0).min(len);
        if start < end {
            crate::interpreter::record_buffer_write(&mut self.dirty.borrow_mut(), start..end, len);
        }
        self.version.set(self.version.get().wrapping_add(writes));
    }

    /// Record a write of `len` bytes from `start` into an ArrayBuffer's own bytes: one more
    /// generation and a dirty range. A Shared Data Block, which other agents write too, has no
    /// generation or dirty ranges; nor has an empty or out-of-range write.
    pub(crate) fn mark_dirty_range(&self, start: usize, len: usize) {
        let Some(buffer_len) = self.local().map(|bytes| bytes.borrow().len()) else {
            return;
        };
        let Some(end) = start.checked_add(len) else {
            return;
        };
        if len == 0 || end > buffer_len {
            return;
        }
        crate::interpreter::record_buffer_write(
            &mut self.dirty.borrow_mut(),
            start..end,
            buffer_len,
        );
        self.version.set(self.version.get().wrapping_add(1));
    }
}

impl Drop for ArrayBufferData {
    fn drop(&mut self) {
        let ledger = &self.ledger.local;
        ledger.set(ledger.get().saturating_sub(self.counted.get()));
        if let Some(BufferData::Shared(block)) = &self.data {
            let mut shared = self.ledger.shared.borrow_mut();
            if let std::collections::hash_map::Entry::Occupied(mut entry) = shared.entry(block.id())
            {
                *entry.get_mut() -= 1;
                if *entry.get() == 0 {
                    entry.remove();
                }
            }
        }
    }
}

/// The buffer data an Agent's live ArrayBuffers and SharedArrayBuffers hold, which the next
/// buffer-pressure trigger grows with (`Interp::reset_buffer_pressure`): counted as a buffer is
/// made, resized and detached and as its payload dies, as SpiderMonkey counts a zone's malloc
/// memory at a cell's creation and finalization, so no collection walks the buffers.
#[derive(Default)]
pub(crate) struct BufferLedger {
    /// The bytes of the live ArrayBuffers' own Data Blocks.
    local: Cell<usize>,
    /// The Shared Data Blocks of the live SharedArrayBuffers, each with its number of buffers:
    /// a block counts once however many buffers alias it.
    shared: RefCell<crate::fasthash::FastMap<u64, usize>>,
}

impl BufferLedger {
    /// The live buffers' data: their own Data Blocks' counted bytes and each Shared Data Block's
    /// current length.
    pub(crate) fn retained_bytes(&self) -> usize {
        let ids: Vec<u64> = self.shared.borrow().keys().copied().collect();
        let shared: usize = ids
            .into_iter()
            .filter_map(crate::interpreter::shared_mem_get)
            .map(|block| block.lock().unwrap().len())
            .sum();
        self.local.get().saturating_add(shared)
    }
}

/// A DataView's [[ViewedArrayBuffer]], [[ByteOffset]] and [[ByteLength]] (for a length-tracking
/// view, `len` is the length at its creation), its object's `ObjectKind::DataView` payload: the
/// former `data_views` table's entry.
#[derive(Clone, Copy)]
pub(crate) struct DataViewData {
    pub(crate) buffer: Gc,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) track: bool,
}

/// The ArrayBuffer state of `buffer` (an ArrayBuffer or SharedArrayBuffer object), under its
/// shared borrow.
#[inline]
pub(crate) fn with_array_buffer<R>(
    buffer: &Gc,
    f: impl FnOnce(&ArrayBufferData) -> R,
) -> Option<R> {
    match &buffer.borrow().kind {
        ObjectKind::ArrayBuffer(data) => Some(f(data)),
        _ => None,
    }
}

/// The Data Block of `buffer`, under its shared borrow: `None` for a detached ArrayBuffer or
/// any other object.
#[inline]
pub(crate) fn with_buffer_data<R>(buffer: &Gc, f: impl FnOnce(&BufferData) -> R) -> Option<R> {
    match &buffer.borrow().kind {
        ObjectKind::ArrayBuffer(data) => data.data.as_ref().map(f),
        _ => None,
    }
}

/// The view `info` describes: a native view over an attached ArrayBuffer's own bytes for the
/// element types generated code accesses, otherwise one it declines.
fn native_view(info: TaInfo) -> NativeTypedArray {
    let size = info.kind.elsize();
    let declined = NativeTypedArray {
        buffer_ptr: std::ptr::null(),
        offset: info.offset,
        end: usize::MAX,
        limit: 0,
        kind: u8::MAX,
        shift: size.trailing_zeros() as u8,
        native: None,
        info,
    };
    let kind = match info.kind {
        TaKind::U8 => 0,
        TaKind::I8 => 1,
        TaKind::U16 => 2,
        TaKind::I16 => 3,
        TaKind::U32 => 4,
        TaKind::I32 => 5,
        TaKind::F32 => 6,
        TaKind::F64 => 7,
        TaKind::U8Clamped => 8,
        _ => return declined,
    };
    // A length-tracking view has no fixed element count: it registers none, with an end no
    // buffer reaches, so generated code always takes the checked operation for it.
    let end = if info.track {
        usize::MAX
    } else {
        let Some(end) = info
            .len
            .checked_mul(size)
            .and_then(|n| info.offset.checked_add(n))
        else {
            return declined;
        };
        end
    };
    let mut object = info.buffer.borrow_mut();
    let ObjectKind::ArrayBuffer(data) = &mut object.kind else {
        return declined;
    };
    // Only an ArrayBuffer's own bytes: SharedArrayBuffer elements stay on the checked
    // operations, which lock the Shared Data Block.
    let Some(BufferData::Local(storage)) = &data.data else {
        return declined;
    };
    let writable = u8::from(!data.immutable);
    let storage = storage.clone();
    let buffer = data
        .native
        .get_or_insert_with(|| {
            Rc::new(NativeBuffer {
                storage_ptr: Rc::as_ptr(&storage),
                attached: Cell::new(1),
                writable,
                dirty_start: Cell::new(usize::MAX),
                dirty_end: Cell::new(0),
                writes: Cell::new(0),
                storage: RefCell::new(Some(storage)),
            })
        })
        .clone();
    NativeTypedArray {
        buffer_ptr: Rc::as_ptr(&buffer),
        offset: info.offset,
        end,
        // Native element counts stay within the signed 32-bit range: generated code may then
        // reject a negative key by its unsigned magnitude alone.
        limit: if info.track { 0 } else { info.len.min(1 << 31) },
        kind,
        shift: size.trailing_zeros() as u8,
        native: Some(buffer),
        info,
    }
}

impl Interp {
    /// Make `object` (a fresh object with no other kind) the typed array `info` describes.
    pub(crate) fn install_typed_array(&mut self, object: &Gc, info: TaInfo) {
        let view = native_view(info);
        object.borrow().set_ic_plain(false);
        let mut object = object.borrow_mut();
        debug_assert!(
            matches!(object.kind, ObjectKind::Plain),
            "a typed array's view goes on an object with no other kind"
        );
        object.kind = ObjectKind::TypedArray(crate::gc_storage::PayloadBox::new(view));
    }

    /// Make `object` (a fresh object with no other kind) an ArrayBuffer or SharedArrayBuffer
    /// over `data`, counted in this Agent's buffer data.
    pub(crate) fn install_array_buffer(&mut self, object: &Gc, data: BufferData) {
        let state = ArrayBufferData::new(data, &self.gc_heap.agent.buffer_ledger);
        let mut object = object.borrow_mut();
        debug_assert!(
            matches!(object.kind, ObjectKind::Plain),
            "a buffer's state goes on an object with no other kind"
        );
        object.kind = ObjectKind::ArrayBuffer(crate::gc_storage::PayloadBox::new(state));
    }
}

/// Layout of std types used by native templates. No RcBox offsets are assumed here: metadata
/// stores Rc::as_ptr addresses, and its owning Rcs keep those addresses alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub object_view: usize,
    pub refcell_value: usize,
    pub vec_ptr: usize,
    pub vec_len: usize,
}

pub(crate) fn probe_layout(object_view: usize, sample: &Gc) -> Option<Layout> {
    if !cfg!(target_endian = "little") || std::mem::size_of::<usize>() != 8 {
        return None;
    }
    let mut vec = Vec::<u8>::with_capacity(19);
    vec.resize(7, 0);
    if std::mem::size_of::<Vec<u8>>() != 3 * std::mem::size_of::<usize>() {
        return None;
    }
    // Vec consists of three initialized words; distinguish pointer, length and capacity by
    // their live values, rather than relying on Rust's unspecified field ordering.
    let words: [usize; 3] = unsafe { std::mem::transmute_copy(&vec) };
    let vec_ptr = words
        .iter()
        .position(|&word| word == vec.as_ptr() as usize)?
        * 8;
    let vec_len = words.iter().position(|&word| word == vec.len())? * 8;
    let vec_cap = words.iter().position(|&word| word == vec.capacity())? * 8;
    if vec_ptr == vec_len || vec_ptr == vec_cap || vec_len == vec_cap {
        return None;
    }
    let cell = RefCell::new(vec);
    let base = &cell as *const _ as usize;
    let refcell_value = cell.as_ptr() as usize - base;
    // Only admit the wrapper with one initialized borrow-flag word before its Vec. Verify both
    // shared and exclusive borrows, so a changed std representation fails closed.
    if refcell_value != 8 || std::mem::size_of_val(&cell) != refcell_value + 24 {
        return None;
    }
    let flag = base as *const isize;
    if unsafe { *flag } != 0 {
        return None;
    }
    {
        let one = cell.borrow();
        if unsafe { *flag } != 1 {
            return None;
        }
        let two = cell.borrow();
        if unsafe { *flag } != 2 {
            return None;
        }
        drop((one, two));
    }
    {
        let exclusive = cell.borrow_mut();
        if unsafe { *flag } != -1 {
            return None;
        }
        drop(exclusive);
    }
    let storage = Rc::new(cell);
    let buffer = Rc::new(NativeBuffer {
        storage_ptr: Rc::as_ptr(&storage),
        attached: Cell::new(1),
        writable: 1,
        dirty_start: Cell::new(usize::MAX),
        dirty_end: Cell::new(0),
        writes: Cell::new(0),
        storage: RefCell::new(Some(storage)),
    });
    // The view's buffer object is only a word here: the probe never reads through it.
    let view = Some(Box::new(NativeTypedArray {
        buffer_ptr: Rc::as_ptr(&buffer),
        offset: 0,
        end: 7,
        limit: 7,
        kind: 0,
        shift: 0,
        native: Some(buffer),
        info: TaInfo {
            buffer: *sample,
            offset: 0,
            len: 7,
            kind: TaKind::U8,
            track: false,
        },
    }));
    let none: Option<Box<NativeTypedArray>> = None;
    if std::mem::size_of_val(&view) != 8
        || unsafe { std::mem::transmute_copy::<_, usize>(&none) } != 0
        || unsafe { std::mem::transmute_copy::<_, usize>(&view) }
            != &**view.as_ref()? as *const NativeTypedArray as usize
    {
        return None;
    }
    Some(Layout {
        object_view,
        refcell_value,
        vec_ptr,
        vec_len,
    })
}

#[cfg(all(test, feature = "embed"))]
mod tests {
    use super::*;
    use crate::{bytecode::Tier, Completion, Engine};

    fn eval(engine: &mut Engine, source: &str) {
        match engine.eval(source, false).unwrap() {
            Completion::Value(_) => {}
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    /// The engine value of `source` (unbranded: the crate's own form), for internal checks.
    fn engine_value(engine: &mut Engine, source: &str) -> crate::value::Value {
        engine.scope(|cx| {
            cx.eval_value(source)
                .unwrap()
                .unwrap_or_else(|_| panic!("read {source}"))
                .into_internal()
        })
    }

    #[test]
    fn native_typed_array_writes_publish_versions_ranges_and_survive_throws() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            var buffer = new ArrayBuffer(64), heap = new Uint8Array(buffer, 4, 16);
            function captured(n) {for (var k = 0; k < n; ++k) heap[k & 15] = k & 255;}
            function local(a, n) {for (var k = 0; k < n; ++k) a[k & 15] = k & 255;}
            captured(1000); local(heap, 1000);
        "#,
        );
        let buffer = engine_value(&mut engine, "buffer");
        let object = buffer.as_obj().unwrap();
        #[cfg(all(
            target_arch = "aarch64",
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        assert!(
            with_array_buffer(&object, |data| data.native.as_ref().unwrap().writes.get()).unwrap()
                > 0,
            "the regression must execute native stores, not just checked helpers"
        );
        let version = engine.interp.array_buffer_version(&buffer).unwrap();
        assert!(version >= 2000);
        let ranges = engine
            .interp
            .take_array_buffer_dirty_ranges(&buffer)
            .unwrap();
        assert!(ranges
            .iter()
            .any(|range| range.start <= 4 && range.end >= 20));
        assert_eq!(engine.interp.array_buffer_version(&buffer), Some(version));
        assert!(engine
            .interp
            .take_array_buffer_dirty_ranges(&buffer)
            .unwrap()
            .is_empty());
        eval(
            &mut engine,
            "function beforeThrow() {heap[2] = 91; throw 17;} try {beforeThrow();} catch(e) {} ",
        );
        assert_eq!(engine.interp.local_buffer(&object).unwrap().borrow()[6], 91);
        assert_eq!(
            engine.interp.array_buffer_version(&buffer),
            Some(version + 1)
        );
        let ranges = engine
            .interp
            .take_array_buffer_dirty_ranges(&buffer)
            .unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 6..7);
        let storage = engine.interp.local_buffer(&object).unwrap();
        assert!(engine.interp.detach_array_buffer(&buffer));
        eval(
            &mut engine,
            "local(heap, 100); captured(100); if (heap[2] !== undefined) throw 1;",
        );
        assert_eq!(
            storage.borrow()[6],
            91,
            "detachment cannot modify the external Data Block"
        );
        assert_eq!(engine.interp.array_buffer_version(&buffer), None);
    }

    #[test]
    fn native_typed_array_subclasses_aliases_nan_and_key_updates() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                function check(ok, why) {if (!ok) throw new Error(why);}
                class Bytes extends Uint8Array {}
                var buffer = new ArrayBuffer(32), a = new Bytes(buffer, 4, 8);
                var b = new Uint16Array(buffer, 4, 4);
                function aliases(a, b) {
                    for (var n = 0; n < 100; ++n) {
                        a[0] = 255; a[1] = 127;
                        check(b[0] === 32767, 'read through alias');
                        b[0] = 258;
                        check(a[0] === 2 && a[1] === 1, 'write through alias');
                    }
                }
                aliases(a, b);
                function keys(a) {var k = -1; return [a[++k], a[++k], k];}
                check(keys(a).join() === '2,1,1', 'deferred key update');
                var raw = new BigUint64Array(buffer), f = new Float64Array(buffer);
                function read(a, k) {return a[k];}
                for (var n = 0; n < 100; ++n) read(f, 0);
                for (var bits of [0xfff8000000000001n, 0xfff9000000000000n,
                                  0xfffa000000000001n, 0xfffb000000000001n,
                                  0x7ff8000000000001n]) {
                    raw[0] = bits;
                    check(Number.isNaN(read(f, 0)), 'NaN cannot become a packed reference');
                }
                var moved = buffer.transfer(), newer = new Uint8Array(moved);
                check(read(a, 0) === undefined, 'old view detached after ownership transfer');
                newer[4] = 23;
                a[0] = 99;
                check(newer[4] === 23, 'detached native metadata cannot write transferred bytes');
            "#,
            );
        }
    }

    #[test]
    fn native_typed_array_metadata_is_reclaimed_with_its_view_graph() {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            "var buffer = new ArrayBuffer(64); var a = new Uint8Array(buffer); a.self = a;",
        );
        // The buffer and its address live only in this helper frame (the collector scans the
        // stack).
        let (record, storage) = crate::gc_test::in_frame(|| {
            let buffer = engine_value(&mut engine, "buffer");
            let object = buffer.as_obj().unwrap();
            let record =
                with_array_buffer(&object, |data| Rc::downgrade(data.native.as_ref().unwrap()));
            let storage = Rc::downgrade(&engine.interp.local_buffer(&object).unwrap());
            (record.unwrap(), storage)
        });
        eval(&mut engine, "a = null; buffer = null;");
        engine.interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&record),
            "the guard record dies with its buffer and views"
        );
        assert!(
            !crate::gc_test::alive(&storage),
            "native metadata must not root dead byte storage"
        );
    }

    /// W2g step 2: every typed array, of every element type and over either kind of buffer, an
    /// ArrayBuffer, a SharedArrayBuffer and a DataView keep their state in their own objects,
    /// subclass instances included; none is pinned, and a view's buffer is its edge, which
    /// keeps the buffer and its bytes alive with the view alone.
    #[test]
    fn views_and_buffers_keep_their_state_in_their_objects() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                var buffer = new ArrayBuffer(64), shared = new SharedArrayBuffer(64);
                var kinds = [Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array,
                    Int32Array, Uint32Array, Float16Array, Float32Array, Float64Array,
                    BigInt64Array, BigUint64Array];
                var views = kinds.flatMap(C => [new C(buffer, 8, 4), new C(shared, 8, 4)]);
                class Bytes extends Uint8Array {}
                class Buffer extends ArrayBuffer {}
                class View extends DataView {}
                var owned = new Buffer(16), data = new View(owned, 2, 8), sub = new Bytes(owned);
                var dv = new DataView(buffer, 4, 8);
                views.push(sub);
            "#,
            );
            let views = engine_value(&mut engine, "views");
            let count = engine.interp.array_length(&views.as_obj().unwrap());
            assert_eq!(count, 25);
            for index in 0..count {
                let view = engine_value(&mut engine, &format!("views[{index}]"));
                let object = view.as_obj().unwrap();
                let info = engine
                    .interp
                    .typed_array_of(&object)
                    .expect("a typed array");
                assert!(!engine
                    .interp
                    .gc_pins
                    .contains_key(&(Gc::as_ptr(&object) as usize)));
                let kind = object.borrow();
                let state = kind.kind.typed_array().unwrap();
                // Generated code accesses an ArrayBuffer's own bytes of the element types it
                // handles; it declines every other view by its element count.
                let native = !engine.interp.is_shared_buffer(&info.buffer)
                    && !matches!(info.kind, TaKind::F16 | TaKind::I64 | TaKind::U64);
                assert_eq!(state.native.is_some(), native, "{tier:?}: view {index}");
                assert_eq!(state.limit, if native { info.len } else { 0 });
            }
            for (name, len) in [("buffer", 64), ("shared", 64), ("owned", 16)] {
                let object = engine_value(&mut engine, name).as_obj().unwrap();
                assert!(object.borrow().kind.array_buffer().is_some());
                assert!(!engine
                    .interp
                    .gc_pins
                    .contains_key(&(Gc::as_ptr(&object) as usize)));
                assert_eq!(engine.interp.array_buffer_byte_length(&object), Some(len));
            }
            for name in ["dv", "data"] {
                let object = engine_value(&mut engine, name).as_obj().unwrap();
                assert!(object.borrow().kind.data_view().is_some());
            }
            // The views alone keep their buffers.
            eval(
                &mut engine,
                r#"
                views[2][1] = 7; views[3][1] = 9; data.setUint8(0, 5);
                buffer = shared = owned = undefined;
            "#,
            );
            engine.interp.gc_collect();
            eval(
                &mut engine,
                r#"
                if (views[2][1] !== 7 || views[3][1] !== 9) throw new Error('lost view bytes');
                if (data.getUint8(0) !== 5 || sub[2] !== 5) throw new Error('lost subclass bytes');
                if (!(data instanceof View) || !(data.buffer instanceof Buffer)) throw 1;
                if (dv.buffer !== views[0].buffer || sub.buffer !== data.buffer) throw 2;
            "#,
            );
        }
    }

    /// W2g step 2: the Agent's buffer ledger, which the buffer-pressure trigger grows with,
    /// counts a buffer's own bytes as it is made, resized and transferred (its source detached)
    /// and drops them as its payload dies; a Shared Data Block counts while a
    /// SharedArrayBuffer holds it.
    #[test]
    fn the_buffer_ledger_follows_buffers_through_their_lives() {
        let mut engine = Engine::new();
        let ledger = engine.interp.gc_heap.agent.buffer_ledger.clone();
        engine.interp.gc_collect();
        let base = ledger.retained_bytes();
        eval(
            &mut engine,
            "var a = new ArrayBuffer(1000, {maxByteLength: 5000});",
        );
        assert_eq!(ledger.retained_bytes(), base + 1000);
        eval(&mut engine, "a.resize(3000);");
        assert_eq!(ledger.retained_bytes(), base + 3000);
        eval(&mut engine, "var b = a.transfer(2000);");
        assert_eq!(ledger.retained_bytes(), base + 2000);
        eval(&mut engine, "var s = new SharedArrayBuffer(700);");
        assert_eq!(ledger.retained_bytes(), base + 2700);
        eval(&mut engine, "b = undefined; s = undefined;");
        engine.interp.gc_collect();
        assert_eq!(ledger.retained_bytes(), base);
    }

    #[test]
    fn native_typed_array_detachment_releases_storage_while_old_views_remain() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            "var buffer = new ArrayBuffer(64), a = new Uint8Array(buffer);",
        );
        let buffer = engine_value(&mut engine, "buffer");
        let storage = Rc::downgrade(
            &engine
                .interp
                .local_buffer(&buffer.as_obj().unwrap())
                .unwrap(),
        );
        assert!(engine.interp.detach_array_buffer(&buffer));
        assert!(
            storage.upgrade().is_none(),
            "a detached view must not retain the Data Block"
        );
        eval(&mut engine, "function oldView() { for (var n = 0; n < 100; ++n) { a[0] = n; if (a[0] !== undefined) throw 1; } } oldView();");
    }
}
