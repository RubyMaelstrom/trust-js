//! Native ops backing the `WebAssembly.*` JS API (see `js/wasm.js`). All wasm entities live in a
//! single shared [`Store`] in OpState; the JS handles (Instance/Memory/Table/Global) carry integer
//! *store addresses*, so entities can be imported and shared across instances (cross-module
//! linking). Imported JS functions are called back through [`CtxHost`].

use std::collections::HashMap;
use std::rc::Rc;

use trust_js_host::{
    ArrayBufferBytes, Ctx, RetainedExternalAllocation, RetainedExternalMemory, Value, WeakValue,
};

use crate::wasm;
use crate::wasm::exec::{Host, Imports, Store, StoreRoot, Val};
use crate::wasm::parse::{Module, ValType};

#[derive(Default)]
pub(crate) struct WasmStore {
    next_module: u32,
    next_root: u32,
    modules: HashMap<u32, Rc<Module>>,
    roots: HashMap<u32, WasmRoot>,
    store: Store,
    /// Imported JS callbacks, indexed by the host id stored in `FuncEntity::Host`.
    host_funcs: Vec<WeakValue>,
    /// One current fixed-length JS buffer per exposed memory address. The storage is retained
    /// separately so a `memory.grow` executed while `Store` is temporarily moved out for a call
    /// can detach the old buffer and identify a replacement with the grown Data Block.
    memory_buffers: HashMap<usize, MemoryBuffer>,
}

/// Each memory's JavaScript ArrayBuffer, kept for as long as the store maps it. Host
/// functions are held weakly.
impl trust_js_host::HostTrace for WasmStore {
    fn trace_gc(&self, visitor: &mut dyn trust_js_host::HostGcVisitor) {
        for memory in self.memory_buffers.values() {
            visitor.root(&memory.buffer);
        }
    }
}

#[derive(Clone, Copy)]
enum WasmRoot {
    Module(u32),
    Store(StoreRoot),
}

struct MemoryBuffer {
    storage: ArrayBufferBytes,
    buffer: Value,
}

impl RetainedExternalMemory for WasmStore {
    fn retained_external_memory(&self, visit: &mut dyn FnMut(RetainedExternalAllocation)) {
        for memory in self.store.memories.iter().flatten() {
            visit(RetainedExternalAllocation::wasm_array_buffer(&memory.bytes));
        }
    }
}

#[cfg(test)]
impl WasmStore {
    pub(crate) fn test_stats(&self) -> [usize; 9] {
        let counts = self.store.live_entity_counts();
        [
            self.modules.len(),
            self.roots.len(),
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            self.host_funcs
                .iter()
                .filter(|callback| callback.upgrade().is_some())
                .count(),
            self.memory_buffers.len(),
        ]
    }
}

// ---- value conversion -------------------------------------------------------------------------

fn val_to_js(v: Val) -> Value {
    match v {
        Val::I32(x) => Value::Num(x as f64),
        Val::I64(x) => Value::bigint_from_i64(x),
        Val::F32(x) => Value::Num(x as f64),
        Val::F64(x) => Value::Num(x),
        Val::Ref(_) => Value::Null,
    }
}

/// WebAssembly JS API § ToWebAssemblyValue. Conversion failures must propagate to JavaScript;
/// silently substituting zero changes observable calls and can turn a user exception into memory
/// corruption inside the guest.
fn js_to_val(ctx: &mut Ctx, v: &Value, ty: ValType) -> Result<Val, Value> {
    match ty {
        ValType::I32 => {
            let number = ctx.coerce_number(v)?;
            let integer = if !number.is_finite() || number == 0.0 {
                0.0
            } else {
                number.trunc().rem_euclid(4_294_967_296.0)
            };
            let signed = if integer >= 2_147_483_648.0 {
                integer - 4_294_967_296.0
            } else {
                integer
            };
            Ok(Val::I32(signed as i32))
        }
        ValType::I64 => Ok(Val::I64(ctx.coerce_bigint_i64(v)?)),
        ValType::F32 => Ok(Val::F32(ctx.coerce_number(v)? as f32)),
        ValType::F64 => Ok(Val::F64(ctx.coerce_number(v)?)),
        ValType::FuncRef | ValType::ExternRef => Ok(Val::Ref(None)),
    }
}

fn valtype_of(s: &str) -> Option<ValType> {
    match s {
        "i32" => Some(ValType::I32),
        "i64" => Some(ValType::I64),
        "f32" => Some(ValType::F32),
        "f64" => Some(ValType::F64),
        _ => None,
    }
}

fn valtype_name(ty: ValType) -> &'static str {
    match ty {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        ValType::FuncRef => "anyfunc",
        ValType::ExternRef => "externref",
    }
}

fn num(a: &[Value], i: usize) -> f64 {
    a.get(i).and_then(Value::as_num_opt).unwrap_or(0.0)
}

fn u32_arg(ctx: &mut Ctx, args: &[Value], index: usize, what: &str) -> Result<u32, Value> {
    match args.get(index).and_then(Value::as_num_opt) {
        Some(number)
            if number.is_finite()
                && number >= 0.0
                && number <= u32::MAX as f64
                && number.fract() == 0.0 =>
        {
            Ok(number as u32)
        }
        _ => Err(ctx.make_error(
            "TypeError",
            format!("WebAssembly {what} must be an unsigned 32-bit integer"),
        )),
    }
}

fn optional_u32_arg(
    ctx: &mut Ctx,
    args: &[Value],
    index: usize,
    what: &str,
) -> Result<Option<u32>, Value> {
    if matches!(args.get(index), None | Some(Value::Undefined)) {
        Ok(None)
    } else {
        u32_arg(ctx, args, index, what).map(Some)
    }
}

fn sweep_wasm_store(ctx: &mut Ctx) {
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    let store_roots = ws.roots.values().filter_map(|root| match root {
        WasmRoot::Store(root) => Some(*root),
        WasmRoot::Module(_) => None,
    });
    let swept = ws.store.sweep(store_roots);
    for address in swept.dead_memories {
        ws.memory_buffers.remove(&address);
    }
    let host_len = swept
        .live_host_funcs
        .iter()
        .max()
        .map_or(0, |address| address + 1);
    ws.host_funcs.truncate(host_len);
    if ws.host_funcs.capacity() > ws.host_funcs.len().saturating_mul(2).saturating_add(64) {
        ws.host_funcs.shrink_to_fit();
    }
    let rooted_modules: std::collections::HashSet<u32> = ws
        .roots
        .values()
        .filter_map(|root| match root {
            WasmRoot::Module(id) => Some(*id),
            WasmRoot::Store(_) => None,
        })
        .collect();
    ws.modules.retain(|id, _| rooted_modules.contains(id));
    ws.modules.shrink_to_fit();
    ws.roots.shrink_to_fit();
    ws.memory_buffers.shrink_to_fit();
}

// ---- host bridge ------------------------------------------------------------------------------

struct CtxHost<'a> {
    ctx: &'a mut Ctx,
    host_funcs: &'a [WeakValue],
    error: Option<Value>,
}

impl Host for CtxHost<'_> {
    fn call_host(
        &mut self,
        id: usize,
        args: &[Val],
        results: &[ValType],
    ) -> Result<Vec<Val>, String> {
        let callback = self
            .host_funcs
            .get(id)
            .and_then(WeakValue::upgrade)
            .ok_or("wasm: bad import id")?;
        let js_args: Vec<Value> = args.iter().map(|v| val_to_js(*v)).collect();
        let ret = match self.ctx.invoke(callback, Value::Undefined, &js_args) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return Err("wasm: imported function threw".into());
            }
        };
        match results.len() {
            0 => Ok(vec![]),
            1 => match js_to_val(self.ctx, &ret, results[0]) {
                Ok(value) => Ok(vec![value]),
                Err(error) => {
                    self.error = Some(error);
                    Err("wasm: imported function result conversion threw".into())
                }
            },
            _ => {
                let mut out = Vec::with_capacity(results.len());
                for (i, &ty) in results.iter().enumerate() {
                    let el = match self.ctx.member_get(&ret, &i.to_string()) {
                        Ok(value) => value,
                        Err(error) => {
                            self.error = Some(error);
                            return Err("wasm: imported function result access threw".into());
                        }
                    };
                    match js_to_val(self.ctx, &el, ty) {
                        Ok(value) => out.push(value),
                        Err(error) => {
                            self.error = Some(error);
                            return Err("wasm: imported function result conversion threw".into());
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    fn memory_grew(&mut self, memory_addr: usize) -> Result<(), String> {
        match refresh_memory_buffer(self.ctx, memory_addr) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.error = Some(error);
                Err("wasm: could not refresh grown Memory buffer".into())
            }
        }
    }
}

/// WebAssembly JS API §4.1 and §5.3 identify a memory instance with one JavaScript Data Block.
/// A successful grow detaches the old fixed buffer and creates a new fixed buffer over the same,
/// now-grown block. This is also called synchronously from the `memory.grow` instruction before a
/// later imported JavaScript function can observe `Memory.buffer`.
fn refresh_memory_buffer(ctx: &mut Ctx, memory_addr: usize) -> Result<(), Value> {
    let Some((storage, old_buffer)) = ctx
        .host_mut::<WasmStore>()
        .and_then(|ws| ws.memory_buffers.get(&memory_addr))
        .map(|entry| (Rc::clone(&entry.storage), entry.buffer.clone()))
    else {
        return Ok(());
    };
    if !ctx.detach_array_buffer(&old_buffer) {
        return Err(ctx.make_error("TypeError", "wasm: current Memory buffer is detached"));
    }
    let buffer = ctx.make_host_keyed_array_buffer_from_storage(Rc::clone(&storage))?;
    ctx.host_mut::<WasmStore>()
        .expect("wasm store")
        .memory_buffers
        .insert(memory_addr, MemoryBuffer { storage, buffer });
    Ok(())
}

/// Run store function `func_addr`, moving the store (and host callbacks) out of OpState so the
/// interpreter can borrow them mutably while `CtxHost` re-enters JS for imports; then move back.
fn run_func(ctx: &mut Ctx, func_addr: usize, args: Vec<Val>) -> Result<Vec<Val>, Value> {
    let (mut store, host_funcs) = {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        (
            std::mem::take(&mut ws.store),
            std::mem::take(&mut ws.host_funcs),
        )
    };
    let (result, host_err) = {
        let mut host = CtxHost {
            ctx: &mut *ctx,
            host_funcs: &host_funcs,
            error: None,
        };
        let r = store.invoke(func_addr, args, &mut host, 0);
        (r, host.error)
    };
    {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        ws.store = store;
        ws.host_funcs = host_funcs;
    }
    result.map_err(|msg| {
        host_err.unwrap_or_else(|| ctx.make_error("Error", format!("RuntimeError: {msg}")))
    })
}

// ---- ops --------------------------------------------------------------------------------------

fn arg_bytes(ctx: &mut Ctx, args: &[Value]) -> Result<Vec<u8>, Value> {
    ctx.typed_array_bytes(args.first().unwrap_or(&Value::Undefined))
        .ok_or_else(|| ctx.make_error("TypeError", "WebAssembly: expected a BufferSource"))
}

pub(crate) fn op_validate(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let bytes = arg_bytes(ctx, a)?;
    Ok(Value::Bool(wasm::validate(&bytes)))
}

pub(crate) fn op_compile(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let bytes = arg_bytes(ctx, a)?;
    match wasm::decode(&bytes) {
        Ok(module) => {
            let exhausted = ctx.make_error("RangeError", "wasm: module handle space exhausted");
            let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
            let id = ws.next_module;
            ws.next_module = ws.next_module.checked_add(1).ok_or(exhausted)?;
            ws.modules.insert(id, module);
            Ok(Value::Num(id as f64))
        }
        Err(e) => Err(ctx.make_error("Error", format!("CompileError: {e}"))),
    }
}

/// Register one JavaScript wrapper as a store root. FinalizationRegistry releases the opaque
/// token; the address itself is not a reference count because instances and tables form graphs.
pub(crate) fn op_retain(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let kind = u32_arg(ctx, a, 0, "root kind")?;
    let address = u32_arg(ctx, a, 1, "root address")?;
    let exhausted = ctx.make_error("RangeError", "wasm: root token space exhausted");
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    let root = match kind {
        0 if ws.modules.contains_key(&address) => WasmRoot::Module(address),
        1 if ws
            .store
            .instances
            .get(address as usize)
            .is_some_and(Option::is_some) =>
        {
            WasmRoot::Store(StoreRoot::Instance(address as usize))
        }
        2 if ws
            .store
            .funcs
            .get(address as usize)
            .is_some_and(Option::is_some) =>
        {
            WasmRoot::Store(StoreRoot::Func(address as usize))
        }
        3 if ws
            .store
            .memories
            .get(address as usize)
            .is_some_and(Option::is_some) =>
        {
            WasmRoot::Store(StoreRoot::Memory(address as usize))
        }
        4 if ws
            .store
            .tables
            .get(address as usize)
            .is_some_and(Option::is_some) =>
        {
            WasmRoot::Store(StoreRoot::Table(address as usize))
        }
        5 if ws
            .store
            .globals
            .get(address as usize)
            .is_some_and(Option::is_some) =>
        {
            WasmRoot::Store(StoreRoot::Global(address as usize))
        }
        _ => return Err(ctx.make_error("TypeError", "wasm: cannot retain a dead handle")),
    };
    let token = ws.next_root;
    ws.next_root = ws.next_root.checked_add(1).ok_or(exhausted)?;
    ws.roots.insert(token, root);
    Ok(Value::Num(token as f64))
}

pub(crate) fn op_release(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let token = u32_arg(ctx, a, 0, "root token")?;
    let removed = ctx
        .host_mut::<WasmStore>()
        .expect("wasm store")
        .roots
        .remove(&token)
        .is_some();
    if removed {
        sweep_wasm_store(ctx);
    }
    Ok(Value::Undefined)
}

fn module_of(ctx: &mut Ctx, id: u32) -> Result<Rc<Module>, Value> {
    ctx.host_mut::<WasmStore>()
        .and_then(|s| s.modules.get(&id).cloned())
        .ok_or_else(|| ctx.make_error("Error", "wasm: unknown module"))
}

pub(crate) fn op_module_exports(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let module = module_of(ctx, num(a, 0) as u32)?;
    let items: Vec<Value> = wasm::export_descriptors(&module)
        .into_iter()
        .map(|(name, kind)| {
            let o = Value::Obj(ctx.new_object());
            let _ = ctx.set_member(&o, "name", Value::from_string(name));
            let _ = ctx.set_member(&o, "kind", Value::str(kind));
            o
        })
        .collect();
    Ok(ctx.make_array(items))
}

pub(crate) fn op_module_imports(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let module = module_of(ctx, num(a, 0) as u32)?;
    let items: Vec<Value> = wasm::import_descriptors(&module)
        .into_iter()
        .map(|(m, name, kind)| {
            let o = Value::Obj(ctx.new_object());
            let _ = ctx.set_member(&o, "module", Value::from_string(m));
            let _ = ctx.set_member(&o, "name", Value::from_string(name));
            let _ = ctx.set_member(&o, "kind", Value::str(kind));
            o
        })
        .collect();
    Ok(ctx.make_array(items))
}

/// Private metadata used to convert primitive global imports according to their declared wasm
/// value type. `WebAssembly.Module.imports()` itself remains the standard three-field descriptor.
pub(crate) fn op_module_import_types(
    ctx: &mut Ctx,
    _t: Value,
    a: &[Value],
) -> Result<Value, Value> {
    let module = module_of(ctx, num(a, 0) as u32)?;
    let items: Vec<Value> = module
        .imports
        .iter()
        .map(|import| {
            let object = Value::Obj(ctx.new_object());
            if let wasm::ImportKind::Global(global) = import.kind {
                let _ = ctx.set_member(&object, "value", Value::str(valtype_name(global.val)));
                let _ = ctx.set_member(&object, "mutable", Value::Bool(global.mutable));
            }
            object
        })
        .collect();
    Ok(ctx.make_array(items))
}

// Standalone entity allocation (for `new WebAssembly.Memory/Table/Global`).
pub(crate) fn op_alloc_memory(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let min = u32_arg(ctx, a, 0, "memory initial")? as usize;
    let max = optional_u32_arg(ctx, a, 1, "memory maximum")?;
    let result = ctx
        .host_mut::<WasmStore>()
        .expect("wasm store")
        .store
        .alloc_memory(min, max);
    result
        .map(|address| Value::Num(address as f64))
        .map_err(|error| ctx.make_error("RangeError", error))
}
pub(crate) fn op_alloc_table(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let min = u32_arg(ctx, a, 0, "table initial")? as usize;
    let max = optional_u32_arg(ctx, a, 1, "table maximum")?;
    let result = ctx
        .host_mut::<WasmStore>()
        .expect("wasm store")
        .store
        .alloc_table(min, max);
    result
        .map(|address| Value::Num(address as f64))
        .map_err(|error| ctx.make_error("RangeError", error))
}
pub(crate) fn op_alloc_global(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let type_name = ctx.coerce_string(a.get(2).unwrap_or(&Value::Undefined))?;
    let ty = valtype_of(&type_name)
        .ok_or_else(|| ctx.make_error("TypeError", "unsupported WebAssembly global type"))?;
    let val = js_to_val(ctx, a.first().unwrap_or(&Value::Undefined), ty)?;
    let mutable = matches!(a.get(1), Some(Value::Bool(true)));
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    Ok(Value::Num(ws.store.alloc_global(val, mutable, ty) as f64))
}

/// `(moduleId, resolvedImports) -> { inst, exports: [{name, kind, addr}] }`. `resolvedImports` is a
/// flat array in module-import order: `{fn}` | `{memAddr}` | `{tableAddr}` | `{globalAddr}`.
pub(crate) fn op_instantiate(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let module = module_of(ctx, num(a, 0) as u32)?;
    let resolved = a.get(1).cloned().unwrap_or(Value::Undefined);

    // Read the JS import descriptors first (borrows ctx only).
    enum Parsed {
        Func(WeakValue, wasm::parse::FuncType),
        Mem(usize),
        Table(usize),
        Global(usize),
    }
    let mut parsed = Vec::new();
    for (i, imp) in module.imports.iter().enumerate() {
        let entry = ctx
            .get_member(&resolved, &i.to_string())
            .unwrap_or(Value::Undefined);
        match &imp.kind {
            wasm::ImportKind::Func(tyidx) => {
                let f = ctx.get_member(&entry, "fn").unwrap_or(Value::Undefined);
                if !f.is_callable() {
                    return Err(ctx.make_error(
                        "Error",
                        format!(
                            "LinkError: import {}.{} is not a function",
                            imp.module, imp.name
                        ),
                    ));
                }
                let callback = ctx
                    .downgrade_object_value(&f)
                    .expect("a callable WebAssembly import is an object");
                parsed.push(Parsed::Func(
                    callback,
                    module.types[*tyidx as usize].clone(),
                ));
            }
            wasm::ImportKind::Memory(_) => {
                parsed.push(Parsed::Mem(
                    ctx.get_member(&entry, "memAddr")
                        .ok()
                        .and_then(|v| v.as_num_opt())
                        .unwrap_or(0.0) as usize,
                ));
            }
            wasm::ImportKind::Table(_) => {
                parsed.push(Parsed::Table(
                    ctx.get_member(&entry, "tableAddr")
                        .ok()
                        .and_then(|v| v.as_num_opt())
                        .unwrap_or(0.0) as usize,
                ));
            }
            wasm::ImportKind::Global(_) => {
                parsed.push(Parsed::Global(
                    ctx.get_member(&entry, "globalAddr")
                        .ok()
                        .and_then(|v| v.as_num_opt())
                        .unwrap_or(0.0) as usize,
                ));
            }
        }
    }

    // Build Imports + register host callbacks, then link.
    let inst_result = {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        let mut imports = Imports::default();
        for p in parsed {
            match p {
                Parsed::Func(f, ty) => {
                    let id = ws.host_funcs.len();
                    ws.host_funcs.push(f);
                    imports.funcs.push((id, ty));
                }
                Parsed::Mem(addr) => imports.mem_addr = Some(addr),
                Parsed::Table(addr) => imports.table_addr = Some(addr),
                Parsed::Global(addr) => imports.global_addrs.push(addr),
            }
        }
        ws.store.instantiate(Rc::clone(&module), imports)
    };
    let inst_idx = match inst_result {
        Ok(instance) => instance,
        Err(error) => {
            sweep_wasm_store(ctx);
            return Err(ctx.make_error("Error", format!("LinkError: {error}")));
        }
    };

    // Run the start function, if any.
    if let Some(start) = module.start {
        let start_addr = {
            let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
            ws.store.instances[inst_idx]
                .as_ref()
                .expect("new instance remains live")
                .func_addrs[start as usize]
        };
        if let Err(error) = run_func(ctx, start_addr, Vec::new()) {
            sweep_wasm_store(ctx);
            return Err(error);
        }
    }

    // Export metadata: name, kind, and store address.
    let names: Vec<(String, wasm::ExportKind, usize)> = {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        module
            .exports
            .iter()
            .filter_map(|e| {
                ws.store
                    .export_addr(inst_idx, &e.name)
                    .map(|(k, addr)| (e.name.clone(), k, addr))
            })
            .collect()
    };
    let exports: Vec<Value> = names
        .into_iter()
        .map(|(name, kind, addr)| {
            let kind = match kind {
                wasm::ExportKind::Func => "function",
                wasm::ExportKind::Memory => "memory",
                wasm::ExportKind::Global => "global",
                wasm::ExportKind::Table => "table",
            };
            let o = Value::Obj(ctx.new_object());
            let _ = ctx.set_member(&o, "name", Value::from_string(name));
            let _ = ctx.set_member(&o, "kind", Value::str(kind));
            let _ = ctx.set_member(&o, "addr", Value::Num(addr as f64));
            o
        })
        .collect();
    let exports = ctx.make_array(exports);

    let result = Value::Obj(ctx.new_object());
    let _ = ctx.set_member(&result, "inst", Value::Num(inst_idx as f64));
    let _ = ctx.set_member(&result, "exports", exports);
    Ok(result)
}

/// `(funcAddr, argsArray) -> resultsArray`.
pub(crate) fn op_call(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let func_addr = num(a, 0) as usize;
    let args_arr = a.get(1).cloned().unwrap_or(Value::Undefined);

    let params = {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        match ws.store.funcs.get(func_addr).and_then(Option::as_ref) {
            Some(f) => f.ty().params.clone(),
            None => return Err(ctx.make_error("Error", "wasm: bad function address")),
        }
    };
    let mut args = Vec::with_capacity(params.len());
    for (i, &ty) in params.iter().enumerate() {
        let v = ctx
            .get_member(&args_arr, &i.to_string())
            .unwrap_or(Value::Undefined);
        args.push(js_to_val(ctx, &v, ty)?);
    }

    let results = run_func(ctx, func_addr, args)?;
    let js: Vec<Value> = results.into_iter().map(val_to_js).collect();
    Ok(ctx.make_array(js))
}

// ---- memory / table / global accessors (by store address) -------------------------------------

pub(crate) fn op_mem_buffer(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    if let Some(buffer) = ctx
        .host_mut::<WasmStore>()
        .and_then(|ws| ws.memory_buffers.get(&addr))
        .map(|entry| entry.buffer.clone())
    {
        return Ok(buffer);
    }
    let storage = ctx
        .host_mut::<WasmStore>()
        .and_then(|ws| ws.store.memories.get(addr))
        .and_then(Option::as_ref)
        .map(|memory| Rc::clone(&memory.bytes))
        .ok_or_else(|| ctx.make_error("Error", "wasm: bad memory address"))?;
    let buffer = ctx.make_host_keyed_array_buffer_from_storage(Rc::clone(&storage))?;
    ctx.host_mut::<WasmStore>()
        .expect("wasm store")
        .memory_buffers
        .insert(
            addr,
            MemoryBuffer {
                storage,
                buffer: buffer.clone(),
            },
        );
    Ok(buffer)
}

pub(crate) fn op_mem_grow(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let delta = num(a, 1) as i32;
    let previous = {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        if addr >= ws.store.memories.len() {
            return Err(ctx.make_error("Error", "wasm: bad memory address"));
        }
        ws.store.mem_grow(addr, delta)
    };
    if previous >= 0 {
        refresh_memory_buffer(ctx, addr)?;
    }
    Ok(Value::Num(previous as f64))
}

pub(crate) fn op_table_get(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let i = num(a, 1) as usize;
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    let slot = ws
        .store
        .tables
        .get(addr)
        .and_then(Option::as_ref)
        .and_then(|t| t.elems.get(i))
        .copied();
    match slot {
        Some(Some(faddr)) => Ok(Value::Num(faddr as f64)),
        Some(None) => Ok(Value::Num(-1.0)),
        None => Err(ctx.make_error("RangeError", "table index out of bounds")),
    }
}

pub(crate) fn op_table_set(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let i = num(a, 1) as usize;
    let faddr = a.get(2).and_then(Value::as_num_opt);
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    match ws
        .store
        .tables
        .get_mut(addr)
        .and_then(Option::as_mut)
        .and_then(|t| t.elems.get_mut(i))
    {
        Some(slot) => {
            *slot = faddr.filter(|n| *n >= 0.0).map(|n| n as usize);
            Ok(Value::Undefined)
        }
        None => Err(ctx.make_error("RangeError", "table index out of bounds")),
    }
}

pub(crate) fn op_table_size(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    Ok(Value::Num(
        ws.store
            .tables
            .get(addr)
            .and_then(Option::as_ref)
            .map(|t| t.elems.len())
            .unwrap_or(0) as f64,
    ))
}

pub(crate) fn op_global_get(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let v = ctx
        .host_mut::<WasmStore>()
        .and_then(|ws| ws.store.globals.get(addr))
        .and_then(Option::as_ref)
        .map(|g| g.val)
        .ok_or_else(|| ctx.make_error("Error", "wasm: bad global address"))?;
    Ok(val_to_js(v))
}

pub(crate) fn op_global_info(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let (ty, mutable) = ctx
        .host_mut::<WasmStore>()
        .and_then(|ws| ws.store.globals.get(addr))
        .and_then(Option::as_ref)
        .map(|global| (global.ty, global.mutable))
        .ok_or_else(|| ctx.make_error("Error", "wasm: bad global address"))?;
    let info = Value::Obj(ctx.new_object());
    let _ = ctx.set_member(&info, "value", Value::str(valtype_name(ty)));
    let _ = ctx.set_member(&info, "mutable", Value::Bool(mutable));
    Ok(info)
}

pub(crate) fn op_global_set(ctx: &mut Ctx, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let addr = num(a, 0) as usize;
    let raw = a.get(1).cloned().unwrap_or(Value::Undefined);
    // Coerce to the global's existing value type.
    let ty = {
        let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
        match ws.store.globals.get(addr).and_then(Option::as_ref) {
            Some(g) => match g.val {
                Val::I64(_) => ValType::I64,
                Val::F32(_) => ValType::F32,
                Val::F64(_) => ValType::F64,
                _ => ValType::I32,
            },
            None => return Err(ctx.make_error("Error", "wasm: bad global address")),
        }
    };
    let val = js_to_val(ctx, &raw, ty)?;
    let ws = ctx.host_mut::<WasmStore>().expect("wasm store");
    if let Some(g) = ws.store.globals.get_mut(addr).and_then(Option::as_mut) {
        if !g.mutable {
            return Err(ctx.make_error("TypeError", "cannot set an immutable global"));
        }
        g.val = val;
    }
    Ok(Value::Undefined)
}

#[cfg(test)]
mod retained_memory_tests {
    use super::*;

    #[test]
    fn wasm_store_reports_each_live_memory_backing_once() {
        let mut wasm = WasmStore::default();
        wasm.store
            .alloc_memory(1, Some(2))
            .expect("first memory allocation");
        wasm.store
            .alloc_memory(2, Some(3))
            .expect("second memory allocation");

        let mut allocations = Vec::new();
        wasm.retained_external_memory(&mut |allocation| allocations.push(allocation));
        assert_eq!(allocations.len(), 2);
    }
}
