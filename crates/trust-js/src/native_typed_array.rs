//! Guarded native views of ordinary ArrayBuffer storage. The side tables and Vec remain
//! authoritative; native accesses reload the live Vec header and never retain its data pointer.
//! ECMA-262 TypedArrayGetElement/TypedArraySetElement and NumericToRawBytes, local official
//! snapshot e28783d5fc9d. Shared, immutable writes, coercing and unsupported element types
//! continue through the checked interpreter operations.

use crate::interpreter::{ArrayBufferBytes, BufferData, Interp};
use crate::value::{Gc, TaInfo, TaKind};
use std::cell::{Cell, RefCell};
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

#[repr(C)]
pub(crate) struct NativeTypedArray {
    pub buffer_ptr: *const NativeBuffer,
    pub offset: usize,
    pub end: usize,
    pub limit: usize,
    pub kind: u8,
    pub shift: u8,
    pub buffer: Rc<NativeBuffer>,
}

impl Interp {
    pub(crate) fn register_native_typed_array(&mut self, object: &Gc, info: TaInfo) {
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
            _ => return,
        };
        // Only an ArrayBuffer's own bytes: SharedArrayBuffer elements stay on the checked
        // operations, which lock the Shared Data Block.
        let Some(BufferData::Local(storage)) = self.array_buffers.get(&info.buffer) else {
            return;
        };
        let size = info.kind.elsize();
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
                return;
            };
            end
        };
        let writable = u8::from(!self.immutable_buffers.contains(&info.buffer));
        let buffer = self
            .native_buffers
            .entry(info.buffer)
            .or_insert_with(|| {
                Rc::new(NativeBuffer {
                    storage_ptr: Rc::as_ptr(storage),
                    attached: Cell::new(1),
                    writable,
                    dirty_start: Cell::new(usize::MAX),
                    dirty_end: Cell::new(0),
                    writes: Cell::new(0),
                    storage: RefCell::new(Some(storage.clone())),
                })
            })
            .clone();
        object.borrow_mut().native_typed_array = Some(Box::new(NativeTypedArray {
            buffer_ptr: Rc::as_ptr(&buffer),
            offset: info.offset,
            end,
            // Native element counts stay within the signed 32-bit range: generated code may
            // then reject a negative key by its unsigned magnitude alone.
            limit: if info.track { 0 } else { info.len.min(1 << 31) },
            kind,
            shift: size.trailing_zeros() as u8,
            buffer,
        }));
    }

    pub(crate) fn invalidate_native_buffer(&mut self, ptr: usize) {
        if let Some(buffer) = self.native_buffers.remove(&ptr) {
            buffer.attached.set(0);
            // DetachArrayBuffer clears [[ArrayBufferData]]. Keep the guard metadata alive for
            // old views, but release its Data Block reference without modifying shared bytes.
            buffer.storage.borrow_mut().take();
        }
    }

    pub(crate) fn flush_native_buffer_writes(&mut self, ptr: usize) {
        let Some(buffer) = self.native_buffers.get(&ptr) else {
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
            super::interpreter::record_buffer_write(
                self.array_buffer_dirty_ranges.entry(ptr).or_default(),
                start..end,
                len,
            );
        }
        let version = self.array_buffer_versions.entry(ptr).or_default();
        *version = version.wrapping_add(writes);
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

pub(crate) fn probe_layout(object_view: usize) -> Option<Layout> {
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
    let view = Some(Box::new(NativeTypedArray {
        buffer_ptr: Rc::as_ptr(&buffer),
        offset: 0,
        end: 7,
        limit: 7,
        kind: 0,
        shift: 0,
        buffer,
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
        let ptr = Gc::as_ptr(&buffer.as_obj().unwrap()) as usize;
        #[cfg(all(
            target_arch = "aarch64",
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        assert!(
            engine.interp.native_buffers[&ptr].writes.get() > 0,
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
        assert_eq!(engine.interp.local_buffer(ptr).unwrap().borrow()[6], 91);
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
        let storage = engine.interp.local_buffer(ptr).unwrap().clone();
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
        let (ptr, storage) = crate::gc_test::in_frame(|| {
            let buffer = engine_value(&mut engine, "buffer");
            let ptr = crate::gc_test::HiddenAddress::of(&buffer);
            let storage = Rc::downgrade(engine.interp.local_buffer(ptr.get()).unwrap());
            (ptr, storage)
        });
        eval(&mut engine, "a = null; buffer = null;");
        engine.interp.gc_collect();
        assert!(!ptr.in_map(&engine.interp.native_buffers));
        assert!(
            !crate::gc_test::alive(&storage),
            "native metadata must not root dead byte storage"
        );
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
        let ptr = Gc::as_ptr(&buffer.as_obj().unwrap()) as usize;
        let storage = Rc::downgrade(engine.interp.local_buffer(ptr).unwrap());
        assert!(engine.interp.detach_array_buffer(&buffer));
        assert!(
            storage.upgrade().is_none(),
            "a detached view must not retain the Data Block"
        );
        eval(&mut engine, "function oldView() { for (var n = 0; n < 100; ++n) { a[0] = n; if (a[0] !== undefined) throw 1; } } oldView();");
    }
}
