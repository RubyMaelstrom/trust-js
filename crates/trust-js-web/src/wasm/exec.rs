//! A bytecode interpreter for the decoded [`Module`]. Stack machine with structured control flow
//! (block/loop/if resolved to jump targets by a one-time label scan). Covers the MVP numeric,
//! comparison, conversion, variable, memory, and call opcodes, plus sign-extension, saturating
//! truncation, and bulk-memory (0xFC) ops. Traps (OOB, divide-by-zero, unreachable, indirect-call
//! type mismatch) surface as `Err(String)` → a JS `WebAssembly.RuntimeError`.

use std::collections::HashMap;
use std::rc::Rc;

use trust_js_host::ArrayBufferBytes;

use super::parse::{FuncType, Limits, Module, ValType};

pub const PAGE_SIZE: usize = 65536;
/// Agent-store resource budgets. WebAssembly Core Appendix "Implementation Limitations" permits
/// bounded memory/table instances; allocation failure must be reported, never panic or abort.
pub const MAX_STORE_MEMORY_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_STORE_TABLE_ELEMENTS: usize = 4_000_000;
// Core allows implementations to report call-stack exhaustion. Keep the recursive baseline
// interpreter below the smallest native test-thread stack; 1024 Rust frames could overflow before
// the explicit check was reached in debug and embedded builds.
const MAX_CALL_DEPTH: usize = 256;

#[derive(Clone, Copy, Debug)]
pub enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Ref(Option<u32>), // funcref/externref: a function index or null
}

impl Val {
    pub fn i32(self) -> i32 {
        match self {
            Val::I32(v) => v,
            _ => 0,
        }
    }
    pub fn i64(self) -> i64 {
        match self {
            Val::I64(v) => v,
            _ => 0,
        }
    }
    pub fn f32(self) -> f32 {
        match self {
            Val::F32(v) => v,
            _ => 0.0,
        }
    }
    pub fn f64(self) -> f64 {
        match self {
            Val::F64(v) => v,
            _ => 0.0,
        }
    }
    pub fn default_for(t: ValType) -> Val {
        match t {
            ValType::I32 => Val::I32(0),
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(0.0),
            ValType::F64 => Val::F64(0.0),
            ValType::FuncRef | ValType::ExternRef => Val::Ref(None),
        }
    }
}

/// A callable in the store: a wasm function (with the index of its defining instance, for context)
/// or an imported host function (by id).
pub enum FuncEntity {
    Wasm {
        compiled: Rc<Compiled>,
        instance: usize,
    },
    Host {
        id: usize,
        ty: FuncType,
    },
}

impl FuncEntity {
    pub fn ty(&self) -> FuncType {
        match self {
            FuncEntity::Wasm { compiled, .. } => compiled.ty.clone(),
            FuncEntity::Host { ty, .. } => ty.clone(),
        }
    }
}

pub struct Compiled {
    pub ty: FuncType,
    pub locals: Vec<ValType>,
    pub code: Vec<u8>,
    pub labels: HashMap<usize, Label>,
}

#[derive(Clone, Copy)]
pub struct Label {
    pub else_ip: Option<usize>,
    pub end_ip: usize,
}

/// A table entity: funcref slots holding store function addresses (so a table can reference
/// functions from any instance — the basis for imported/shared tables).
pub struct TableEntity {
    pub elems: Vec<Option<usize>>,
    pub max: Option<u32>,
    pub elem_type: ValType,
}

pub struct MemEntity {
    pub bytes: ArrayBufferBytes,
    pub max: Option<u32>,
}

pub struct GlobalEntity {
    pub val: Val,
    pub mutable: bool,
    pub ty: ValType,
}

/// Import values resolved by the op layer, as store addresses (for memory/table/globals — whether
/// they came from another instance or a standalone `new WebAssembly.Memory`) and host ids (for
/// imported JS functions). In module-import order per kind.
#[derive(Default)]
pub struct Imports {
    pub funcs: Vec<(usize, FuncType)>,
    /// Existing store addresses for imported WebAssembly functions. A `None` entry means the
    /// corresponding entry in `funcs` is a host function. Keeping the address preserves function
    /// identity when one instance re-exports a function to another instance.
    pub func_addrs: Vec<Option<usize>>,
    pub mem_addr: Option<usize>,
    /// Table imports in table-index order. `table_addr` remains as the single-table convenience
    /// used by the JavaScript embedding.
    pub table_addrs: Vec<usize>,
    pub table_addr: Option<usize>,
    pub global_addrs: Vec<usize>,
}

/// One instantiated module: index-spaces mapping its local indices to store addresses. Entities
/// themselves live in the shared [`Store`], so instances can import and share them.
pub struct Instance {
    pub module: Rc<Module>,
    pub func_addrs: Vec<usize>,
    pub table_addrs: Vec<usize>,
    pub mem_addrs: Vec<usize>,
    pub global_addrs: Vec<usize>,
    /// Passive data segments remain in the module; this per-instance bitset records whether each
    /// segment is still available to `memory.init`. Active segments start dropped.
    pub data_live: std::cell::RefCell<Vec<bool>>,
}

/// The shared store: flat address spaces for every instance's functions, tables, memories, and
/// globals (WebAssembly's Store). Execution and linking go through it.
#[derive(Default)]
pub struct Store {
    pub funcs: Vec<Option<FuncEntity>>,
    pub tables: Vec<Option<TableEntity>>,
    pub memories: Vec<Option<MemEntity>>,
    pub globals: Vec<Option<GlobalEntity>>,
    pub instances: Vec<Option<Rc<Instance>>>,
    memory_bytes: usize,
    table_elements: usize,
}

#[derive(Clone, Copy)]
struct StoreCheckpoint {
    funcs: usize,
    tables: usize,
    memories: usize,
    globals: usize,
    instances: usize,
    memory_bytes: usize,
    table_elements: usize,
}

fn trim_tombstones<T>(slots: &mut Vec<Option<T>>) {
    while slots.last().is_some_and(Option::is_none) {
        slots.pop();
    }
    if slots.capacity() > slots.len().saturating_mul(2).saturating_add(64) {
        slots.shrink_to_fit();
    }
}

/// The host bridge: the op layer implements this to call imported JS functions. `results` is the
/// import's declared result types, so the host can coerce the JS return value(s) to wasm values.
pub trait Host {
    fn call_host(
        &mut self,
        id: usize,
        args: &[Val],
        results: &[ValType],
    ) -> Result<Vec<Val>, String>;

    /// WebAssembly JS API §5.3 requires the corresponding fixed ArrayBuffer to be refreshed
    /// immediately after a successful `memory.grow`, including before a later host callback.
    fn memory_grew(&mut self, _memory_addr: usize) -> Result<(), String> {
        Ok(())
    }
}

/// Roots held by live JavaScript WebAssembly wrappers. Store addresses are stable tombstone slots;
/// releasing a wrapper triggers a graph trace through instances, tables, and defining functions.
#[derive(Clone, Copy, Debug)]
pub enum StoreRoot {
    Instance(usize),
    Func(usize),
    Table(usize),
    Memory(usize),
    Global(usize),
}

#[derive(Default)]
pub struct SweepResult {
    pub dead_memories: Vec<usize>,
    pub live_host_funcs: std::collections::HashSet<usize>,
}

// ---- label pre-scan ---------------------------------------------------------------------------

/// Pre-compute the matching `else`/`end` position for every block/loop/if in a function body, so
/// branches are O(1). Requires skipping each instruction's immediates.
pub fn scan_labels(code: &[u8]) -> Result<HashMap<usize, Label>, String> {
    let mut labels = HashMap::new();
    let mut open: Vec<usize> = Vec::new();
    let mut ip = 0;
    while ip < code.len() {
        let op = code[ip];
        let start = ip;
        ip += 1;
        match op {
            0x02 | 0x03 | 0x04 => {
                // block / loop / if — blocktype immediate
                skip_blocktype(code, &mut ip)?;
                open.push(start);
                labels.insert(
                    start,
                    Label {
                        else_ip: None,
                        end_ip: 0,
                    },
                );
            }
            0x05 => {
                // else — belongs to the innermost open `if`
                let &pos = open.last().ok_or("wasm: else without open block")?;
                let l = labels.get_mut(&pos).ok_or("wasm: else without label")?;
                if l.else_ip.replace(ip).is_some() {
                    return Err("wasm: duplicate else".into());
                }
            }
            0x0b => {
                // end
                let pos = open.pop().ok_or("wasm: end without open block")?;
                labels
                    .get_mut(&pos)
                    .ok_or("wasm: end without label")?
                    .end_ip = ip;
            }
            _ => skip_immediates(code, op, &mut ip)?,
        }
    }
    if !open.is_empty() {
        return Err("wasm: unterminated structured control instruction".into());
    }
    Ok(labels)
}

fn skip_blocktype(code: &[u8], ip: &mut usize) -> Result<(), String> {
    let b = *code.get(*ip).ok_or("wasm: truncated blocktype")?;
    if b == 0x40 || matches!(b, 0x7f | 0x7e | 0x7d | 0x7c | 0x70 | 0x6f) {
        *ip += 1;
    } else {
        // s33 type index (signed LEB)
        read_sleb_bits(code, ip, 33)?;
    }
    Ok(())
}

fn read_uleb(code: &[u8], ip: &mut usize) -> Result<u64, String> {
    read_uleb_bits(code, ip, 32)
}

fn read_uleb_bits(code: &[u8], ip: &mut usize, bits: u32) -> Result<u64, String> {
    let max_bytes = bits.div_ceil(7) as usize;
    let mut result = 0u64;
    for byte_index in 0..max_bytes {
        let b = *code.get(*ip).ok_or("wasm: truncated LEB")?;
        *ip += 1;
        let shift = byte_index * 7;
        let payload = b & 0x7f;
        let remaining = bits.saturating_sub(shift as u32);
        if remaining < 7 && (payload as u64) >= (1u64 << remaining) {
            return Err("wasm: unsigned LEB128 overflow".into());
        }
        result |= (payload as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
    }
    Err("wasm: unsigned LEB128 too long".into())
}

fn read_sleb(code: &[u8], ip: &mut usize) -> Result<i64, String> {
    read_sleb_bits(code, ip, 64)
}

fn read_sleb_bits(code: &[u8], ip: &mut usize, bits: u32) -> Result<i64, String> {
    let max_bytes = bits.div_ceil(7) as usize;
    let mut result = 0i64;
    for byte_index in 0..max_bytes {
        let b = *code.get(*ip).ok_or("wasm: truncated SLEB")?;
        *ip += 1;
        let shift = byte_index * 7;
        let remaining = bits.saturating_sub(shift as u32);
        let payload = b & 0x7f;
        if remaining < 7 {
            let value_mask = (1u8 << remaining) - 1;
            let unused = payload & !value_mask;
            let sign = payload & (1u8 << (remaining - 1)) != 0;
            if (!sign && unused != 0) || (sign && unused != (!value_mask & 0x7f)) {
                return Err("wasm: signed LEB128 overflow".into());
            }
        }
        result |= (payload as i64) << shift;
        if b & 0x80 == 0 {
            let consumed = ((byte_index + 1) * 7) as u32;
            if consumed < bits && b & 0x40 != 0 {
                result |= -1i64 << consumed;
            }
            return Ok(result);
        }
    }
    Err("wasm: signed LEB128 too long".into())
}

fn skip_bytes(code: &[u8], ip: &mut usize, len: usize) -> Result<(), String> {
    let end = ip
        .checked_add(len)
        .ok_or("wasm: immediate length overflow")?;
    if end > code.len() {
        return Err("wasm: truncated instruction immediate".into());
    }
    *ip = end;
    Ok(())
}

fn read_fixed<const N: usize>(code: &[u8], ip: &mut usize) -> Result<[u8; N], String> {
    let end = ip.checked_add(N).ok_or("wasm: immediate length overflow")?;
    let bytes = code
        .get(*ip..end)
        .ok_or("wasm: truncated instruction immediate")?;
    *ip = end;
    bytes
        .try_into()
        .map_err(|_| "wasm: malformed fixed-width immediate".into())
}

/// Advance `ip` past the immediate operands of the instruction with opcode `op`.
fn skip_immediates(code: &[u8], op: u8, ip: &mut usize) -> Result<(), String> {
    match op {
        // no immediates
        0x00 | 0x01 | 0x0f | 0x1a | 0x1b | 0xd1 => Ok(()),
        // single LEB immediate (branch depth, call, local/global idx, ref.func)
        0x0c | 0x0d | 0x10 | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0xd2 => {
            read_uleb(code, ip)?;
            Ok(())
        }
        0x0e => {
            // br_table: vec of labels + default
            let n = read_uleb(code, ip)?;
            for _ in 0..=n {
                read_uleb(code, ip)?;
            }
            Ok(())
        }
        0x11 => {
            // call_indirect: type idx + table idx
            read_uleb(code, ip)?;
            read_uleb(code, ip)?;
            Ok(())
        }
        0x1c => {
            let count = read_uleb(code, ip)?;
            skip_bytes(code, ip, count as usize)
        }
        0xd0 => {
            skip_bytes(code, ip, 1) // ref.null t
        }
        0x41 => {
            read_sleb(code, ip)?; // i32.const
            Ok(())
        }
        0x42 => {
            read_sleb(code, ip)?; // i64.const
            Ok(())
        }
        0x43 => {
            skip_bytes(code, ip, 4) // f32.const
        }
        0x44 => {
            skip_bytes(code, ip, 8) // f64.const
        }
        // memory load/store: memarg (align + offset)
        0x28..=0x3e => {
            read_uleb(code, ip)?;
            read_uleb(code, ip)?;
            Ok(())
        }
        0x3f | 0x40 => {
            skip_bytes(code, ip, 1) // memory.size / memory.grow (reserved byte)
        }
        0xfc => {
            let sub = read_uleb(code, ip)?;
            match sub {
                8 => {
                    read_uleb(code, ip)?;
                    skip_bytes(code, ip, 1)?;
                } // memory.init: dataidx, mem
                9 => {
                    read_uleb(code, ip)?;
                } // data.drop
                10 => {
                    skip_bytes(code, ip, 2)?;
                } // memory.copy: two mem indices
                11 => {
                    skip_bytes(code, ip, 1)?;
                } // memory.fill: mem
                _ => {} // trunc_sat variants: no immediate
            }
            Ok(())
        }
        // everything else (numeric/comparison/etc.) has no immediate
        _ => Ok(()),
    }
}

// ---- interpreter ------------------------------------------------------------------------------

macro_rules! binop {
    ($stack:expr, $t:ident, $variant:ident, $f:expr) => {{
        let b = $stack.pop().unwrap().$t();
        let a = $stack.pop().unwrap().$t();
        $stack.push(Val::$variant($f(a, b)));
    }};
}
macro_rules! cmpop {
    ($stack:expr, $t:ident, $f:expr) => {{
        let b = $stack.pop().unwrap().$t();
        let a = $stack.pop().unwrap().$t();
        $stack.push(Val::I32(if $f(a, b) { 1 } else { 0 }));
    }};
}

struct Ctrl {
    is_loop: bool,
    end_ip: usize,
    start_ip: usize, // for loops: branch target
    stack_height: usize,
    arity: usize,
}

impl Store {
    /// Trace store entities reachable from live JavaScript wrappers, then tombstone everything
    /// else. Addresses are never renumbered, so a collection cannot invalidate another live
    /// wrapper or a table slot. The byte/element budgets are released with the entities.
    pub fn sweep(&mut self, roots: impl IntoIterator<Item = StoreRoot>) -> SweepResult {
        let mut funcs = vec![false; self.funcs.len()];
        let mut tables = vec![false; self.tables.len()];
        let mut memories = vec![false; self.memories.len()];
        let mut globals = vec![false; self.globals.len()];
        let mut instances = vec![false; self.instances.len()];
        let mut pending: Vec<StoreRoot> = roots.into_iter().collect();

        while let Some(root) = pending.pop() {
            match root {
                StoreRoot::Instance(address) => {
                    if address >= instances.len()
                        || instances[address]
                        || self.instances[address].is_none()
                    {
                        continue;
                    }
                    instances[address] = true;
                    let instance = self.instances[address].as_ref().expect("checked above");
                    pending.extend(instance.func_addrs.iter().copied().map(StoreRoot::Func));
                    pending.extend(instance.table_addrs.iter().copied().map(StoreRoot::Table));
                    pending.extend(instance.mem_addrs.iter().copied().map(StoreRoot::Memory));
                    pending.extend(instance.global_addrs.iter().copied().map(StoreRoot::Global));
                }
                StoreRoot::Func(address) => {
                    if address >= funcs.len() || funcs[address] || self.funcs[address].is_none() {
                        continue;
                    }
                    funcs[address] = true;
                    if let Some(FuncEntity::Wasm { instance, .. }) = &self.funcs[address] {
                        pending.push(StoreRoot::Instance(*instance));
                    }
                }
                StoreRoot::Table(address) => {
                    if address >= tables.len() || tables[address] || self.tables[address].is_none()
                    {
                        continue;
                    }
                    tables[address] = true;
                    pending.extend(
                        self.tables[address]
                            .as_ref()
                            .expect("checked above")
                            .elems
                            .iter()
                            .flatten()
                            .copied()
                            .map(StoreRoot::Func),
                    );
                }
                StoreRoot::Memory(address) => {
                    if address < memories.len() && self.memories[address].is_some() {
                        memories[address] = true;
                    }
                }
                StoreRoot::Global(address) => {
                    if address < globals.len() && self.globals[address].is_some() {
                        globals[address] = true;
                    }
                }
            }
        }

        let mut result = SweepResult::default();
        for (address, entity) in self.funcs.iter_mut().enumerate() {
            if funcs[address] {
                if let Some(FuncEntity::Host { id, .. }) = entity {
                    result.live_host_funcs.insert(*id);
                }
            } else {
                *entity = None;
            }
        }
        for (address, entity) in self.tables.iter_mut().enumerate() {
            if !tables[address] {
                if let Some(table) = entity.take() {
                    self.table_elements = self.table_elements.saturating_sub(table.elems.len());
                }
            }
        }
        for (address, entity) in self.memories.iter_mut().enumerate() {
            if !memories[address] {
                if let Some(memory) = entity.take() {
                    self.memory_bytes = self
                        .memory_bytes
                        .saturating_sub(memory.bytes.borrow().len());
                    result.dead_memories.push(address);
                }
            }
        }
        for (address, entity) in self.globals.iter_mut().enumerate() {
            if !globals[address] {
                *entity = None;
            }
        }
        for (address, entity) in self.instances.iter_mut().enumerate() {
            if !instances[address] {
                *entity = None;
            }
        }
        trim_tombstones(&mut self.funcs);
        trim_tombstones(&mut self.tables);
        trim_tombstones(&mut self.memories);
        trim_tombstones(&mut self.globals);
        trim_tombstones(&mut self.instances);
        result
    }

    pub fn live_entity_counts(&self) -> [usize; 5] {
        [
            self.instances.iter().flatten().count(),
            self.funcs.iter().flatten().count(),
            self.tables.iter().flatten().count(),
            self.memories.iter().flatten().count(),
            self.globals.iter().flatten().count(),
        ]
    }

    /// Call the store function at `func_addr` with `args`; returns its results.
    pub fn invoke(
        &mut self,
        func_addr: usize,
        args: Vec<Val>,
        host: &mut dyn Host,
        depth: usize,
    ) -> Result<Vec<Val>, String> {
        if depth >= MAX_CALL_DEPTH {
            return Err("wasm: call stack exhausted".into());
        }
        // Host functions run immediately; wasm functions run below, in their defining instance's
        // context. `inst`/`compiled` are cloned Rcs so the loop can borrow `self` (the store)
        // mutably for entity access without conflicting.
        let (compiled, inst) = match self
            .funcs
            .get(func_addr)
            .and_then(Option::as_ref)
            .ok_or("wasm: bad function index")?
        {
            FuncEntity::Host { id, ty } => {
                let (id, nres) = (*id, ty.results.len());
                let results = ty.results.clone();
                let r = host.call_host(id, &args, &results)?;
                if r.len() != nres {
                    return Err("wasm: host function returned wrong arity".into());
                }
                return Ok(r);
            }
            FuncEntity::Wasm { compiled, instance } => (
                compiled.clone(),
                self.instances
                    .get(*instance)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or("wasm: dead function instance")?,
            ),
        };
        let mem_addr = inst.mem_addrs.first().copied();

        // Locals = params (from args) then declared locals (zeroed).
        let mut locals: Vec<Val> = args;
        for &lt in &compiled.locals {
            locals.push(Val::default_for(lt));
        }

        let mut stack: Vec<Val> = Vec::new();
        // The function body has an implicit outer control label. Besides matching the validation
        // model, this makes a top-level `br 0` a return instead of an out-of-range branch.
        let mut ctrl = vec![Ctrl {
            is_loop: false,
            end_ip: compiled.code.len(),
            start_ip: 0,
            stack_height: 0,
            arity: compiled.ty.results.len(),
        }];
        let code = &compiled.code;
        let labels = &compiled.labels;
        let mut ip = 0;

        loop {
            if ip >= code.len() {
                break;
            }
            let op = code[ip];
            let op_start = ip;
            ip += 1;
            match op {
                0x00 => return Err("wasm: unreachable executed".into()),
                0x01 => {} // nop
                0x02 | 0x03 | 0x04 => {
                    // block / loop / if
                    let (params, results) = block_arity(&inst.module, code, &mut ip)?;
                    let label = labels[&op_start];
                    if op == 0x04 {
                        let cond = stack.pop().unwrap().i32();
                        if cond == 0 {
                            // jump to else (if any) or past end
                            match label.else_ip {
                                Some(e) => ip = e,
                                None => {
                                    ip = label.end_ip;
                                    continue;
                                }
                            }
                        }
                    }
                    ctrl.push(Ctrl {
                        is_loop: op == 0x03,
                        end_ip: label.end_ip,
                        start_ip: ip, // body start (loop branch target)
                        stack_height: stack.len().saturating_sub(params),
                        arity: if op == 0x03 { params } else { results },
                    });
                }
                0x05 => {
                    // else: reached only by falling out of the `then` arm → jump to end
                    if let Some(c) = ctrl.pop() {
                        ip = c.end_ip;
                    }
                }
                0x0b => {
                    // end
                    if ctrl.len() <= 1 {
                        return Err("wasm: unexpected function-level end".into());
                    }
                    ctrl.pop();
                }
                0x0c => {
                    let l = read_uleb(code, &mut ip)? as usize;
                    do_branch(&mut ctrl, &mut stack, &mut ip, l)?;
                }
                0x0d => {
                    let l = read_uleb(code, &mut ip)? as usize;
                    let cond = stack.pop().unwrap().i32();
                    if cond != 0 {
                        do_branch(&mut ctrl, &mut stack, &mut ip, l)?;
                    }
                }
                0x0e => {
                    let n = read_uleb(code, &mut ip)?;
                    let mut targets = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        targets.push(read_uleb(code, &mut ip)? as usize);
                    }
                    let default = read_uleb(code, &mut ip)? as usize;
                    let idx = stack.pop().unwrap().i32();
                    let l = if (idx as usize) < targets.len() {
                        targets[idx as usize]
                    } else {
                        default
                    };
                    do_branch(&mut ctrl, &mut stack, &mut ip, l)?;
                }
                0x0f => {
                    // return
                    break;
                }
                0x10 => {
                    let f = read_uleb(code, &mut ip)? as usize;
                    let addr = *inst.func_addrs.get(f).ok_or("wasm: bad function index")?;
                    let ty = self.funcs[addr]
                        .as_ref()
                        .ok_or("wasm: dead function address")?
                        .ty();
                    let mut args = Vec::with_capacity(ty.params.len());
                    for _ in 0..ty.params.len() {
                        args.push(stack.pop().ok_or("wasm: stack underflow on call")?);
                    }
                    args.reverse();
                    let results = self.invoke(addr, args, host, depth + 1)?;
                    stack.extend(results);
                }
                0x11 => {
                    let type_idx = read_uleb(code, &mut ip)? as usize;
                    let table_idx = read_uleb(code, &mut ip)? as usize;
                    let elem = stack.pop().unwrap().i32();
                    let table_addr = *inst
                        .table_addrs
                        .get(table_idx)
                        .ok_or("wasm: bad table index")?;
                    // Table slots hold store function addresses, so an imported/shared table can
                    // dispatch to functions from any instance.
                    let addr = self.tables[table_addr]
                        .as_ref()
                        .ok_or("wasm: dead table address")?
                        .elems
                        .get(elem as usize)
                        .copied()
                        .ok_or("wasm: undefined element (indirect call)")?
                        .ok_or("wasm: uninitialized table element")?;
                    let expected = inst
                        .module
                        .types
                        .get(type_idx)
                        .ok_or("wasm: bad type index")?;
                    let actual = self.funcs[addr]
                        .as_ref()
                        .ok_or("wasm: dead indirect function")?
                        .ty();
                    if !same_type(expected, &actual) {
                        return Err("wasm: indirect call type mismatch".into());
                    }
                    let mut args = Vec::with_capacity(actual.params.len());
                    for _ in 0..actual.params.len() {
                        args.push(stack.pop().ok_or("wasm: stack underflow")?);
                    }
                    args.reverse();
                    let results = self.invoke(addr, args, host, depth + 1)?;
                    stack.extend(results);
                }
                0x1a => {
                    stack.pop();
                } // drop
                0x1b => {
                    // select
                    let c = stack.pop().unwrap().i32();
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(if c != 0 { a } else { b });
                }
                0x1c => {
                    // Typed select carries a singleton result-type vector.
                    let count = read_uleb(code, &mut ip)? as usize;
                    skip_bytes(code, &mut ip, count)?;
                    let c = stack.pop().unwrap().i32();
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(if c != 0 { a } else { b });
                }
                0x20 => {
                    let i = read_uleb(code, &mut ip)? as usize;
                    stack.push(*locals.get(i).ok_or("wasm: bad local index")?);
                }
                0x21 => {
                    let i = read_uleb(code, &mut ip)? as usize;
                    let v = stack.pop().ok_or("wasm: stack underflow")?;
                    *locals.get_mut(i).ok_or("wasm: bad local index")? = v;
                }
                0x22 => {
                    let i = read_uleb(code, &mut ip)? as usize;
                    let v = *stack.last().ok_or("wasm: stack underflow")?;
                    *locals.get_mut(i).ok_or("wasm: bad local index")? = v;
                }
                0x23 => {
                    let i = read_uleb(code, &mut ip)? as usize;
                    let addr = *inst.global_addrs.get(i).ok_or("wasm: bad global index")?;
                    stack.push(
                        self.globals[addr]
                            .as_ref()
                            .ok_or("wasm: dead global address")?
                            .val,
                    );
                }
                0x24 => {
                    let i = read_uleb(code, &mut ip)? as usize;
                    let addr = *inst.global_addrs.get(i).ok_or("wasm: bad global index")?;
                    let v = stack.pop().ok_or("wasm: stack underflow")?;
                    self.globals[addr]
                        .as_mut()
                        .ok_or("wasm: dead global address")?
                        .val = v;
                }
                // memory loads/stores
                0x28..=0x3e => {
                    let ma = mem_addr.ok_or("wasm: no memory")?;
                    self.mem_op(ma, op, code, &mut ip, &mut stack)?;
                }
                0x3f => {
                    skip_bytes(code, &mut ip, 1)?; // reserved
                    let ma = mem_addr.ok_or("wasm: no memory")?;
                    stack.push(Val::I32(
                        (self.memories[ma]
                            .as_ref()
                            .ok_or("wasm: dead memory address")?
                            .bytes
                            .borrow()
                            .len()
                            / PAGE_SIZE) as i32,
                    ));
                }
                0x40 => {
                    skip_bytes(code, &mut ip, 1)?;
                    let ma = mem_addr.ok_or("wasm: no memory")?;
                    let delta = stack.pop().unwrap().i32();
                    let previous = self.mem_grow(ma, delta);
                    if previous >= 0 {
                        host.memory_grew(ma)?;
                    }
                    stack.push(Val::I32(previous));
                }
                0x41 => stack.push(Val::I32(read_sleb_bits(code, &mut ip, 32)? as i32)),
                0x42 => stack.push(Val::I64(read_sleb(code, &mut ip)?)),
                0x43 => {
                    let bytes = read_fixed::<4>(code, &mut ip)?;
                    stack.push(Val::F32(f32::from_le_bytes(bytes)));
                }
                0x44 => {
                    let bytes = read_fixed::<8>(code, &mut ip)?;
                    stack.push(Val::F64(f64::from_le_bytes(bytes)));
                }
                0xd0 => {
                    skip_bytes(code, &mut ip, 1)?;
                    stack.push(Val::Ref(None));
                }
                0xd1 => {
                    let value = stack.pop().unwrap();
                    stack.push(Val::I32(matches!(value, Val::Ref(None)) as i32));
                }
                0xfc => {
                    self.op_fc(&inst, mem_addr, code, &mut ip, &mut stack)?;
                }
                _ => numeric(op, &mut stack)?,
            }
        }

        // Return the top `results` values.
        let nres = compiled.ty.results.len();
        let start = stack.len().saturating_sub(nres);
        Ok(stack.split_off(start))
    }

    /// Grow the memory at store address `mem_addr` by `delta` pages; returns the previous page
    /// count, or -1 on failure.
    pub fn mem_grow(&mut self, mem_addr: usize, delta: i32) -> i32 {
        if delta < 0 {
            return -1;
        }
        let Some(mem) = self.memories.get(mem_addr).and_then(Option::as_ref) else {
            return -1;
        };
        let old_len = mem.bytes.borrow().len();
        let old_pages = (old_len / PAGE_SIZE) as u32;
        let Some(new_pages) = old_pages.checked_add(delta as u32) else {
            return -1;
        };
        if let Some(max) = mem.max {
            if new_pages > max {
                return -1;
            }
        }
        if new_pages > 65536 {
            return -1;
        }
        let Some(new_bytes) = (new_pages as usize).checked_mul(PAGE_SIZE) else {
            return -1;
        };
        let additional = new_bytes - old_len;
        let Some(new_total) = self.memory_bytes.checked_add(additional) else {
            return -1;
        };
        if new_total > MAX_STORE_MEMORY_BYTES {
            return -1;
        }
        let mut bytes = self.memories[mem_addr]
            .as_ref()
            .expect("checked live memory above")
            .bytes
            .borrow_mut();
        if bytes.try_reserve_exact(additional).is_err() {
            return -1;
        }
        bytes.resize(new_bytes, 0);
        self.memory_bytes = new_total;
        old_pages as i32
    }

    pub fn alloc_memory(&mut self, min_pages: usize, max: Option<u32>) -> Result<usize, String> {
        if min_pages > 65_536
            || max.is_some_and(|maximum| maximum > 65_536 || min_pages > maximum as usize)
        {
            return Err("wasm: invalid memory limits".into());
        }
        let bytes_len = min_pages
            .checked_mul(PAGE_SIZE)
            .ok_or("wasm: memory size overflow")?;
        let new_total = self
            .memory_bytes
            .checked_add(bytes_len)
            .ok_or("wasm: store memory budget overflow")?;
        if new_total > MAX_STORE_MEMORY_BYTES {
            return Err("wasm: store memory budget exceeded".into());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(bytes_len)
            .map_err(|_| "wasm: memory allocation failed")?;
        bytes.resize(bytes_len, 0);
        self.memories.push(Some(MemEntity {
            bytes: Rc::new(std::cell::RefCell::new(bytes)),
            max,
        }));
        self.memory_bytes = new_total;
        Ok(self.memories.len() - 1)
    }
    pub fn alloc_table(&mut self, min: usize, max: Option<u32>) -> Result<usize, String> {
        self.alloc_table_typed(min, max, ValType::FuncRef)
    }

    fn alloc_table_typed(
        &mut self,
        min: usize,
        max: Option<u32>,
        elem_type: ValType,
    ) -> Result<usize, String> {
        if min > u32::MAX as usize || max.is_some_and(|maximum| min > maximum as usize) {
            return Err("wasm: invalid table limits".into());
        }
        if elem_type != ValType::FuncRef {
            return Err("wasm: unsupported table element type".into());
        }
        let new_total = self
            .table_elements
            .checked_add(min)
            .ok_or("wasm: store table budget overflow")?;
        if new_total > MAX_STORE_TABLE_ELEMENTS {
            return Err("wasm: store table budget exceeded".into());
        }
        let mut elems = Vec::new();
        elems
            .try_reserve_exact(min)
            .map_err(|_| "wasm: table allocation failed")?;
        elems.resize(min, None);
        self.tables.push(Some(TableEntity {
            elems,
            max,
            elem_type,
        }));
        self.table_elements = new_total;
        Ok(self.tables.len() - 1)
    }
    pub fn alloc_global(&mut self, val: Val, mutable: bool, ty: ValType) -> usize {
        self.globals.push(Some(GlobalEntity { val, mutable, ty }));
        self.globals.len() - 1
    }

    /// Link `module` against resolved `imports` into a new instance; returns its index in
    /// `self.instances`. Allocates the module's defined functions/memory/tables/globals into the
    /// store, references imported entities by address, and runs element/data segments. Per the
    /// Core instantiation algorithm, store mutations made before a later trap are not rolled back.
    pub fn instantiate(&mut self, module: Rc<Module>, imports: Imports) -> Result<usize, String> {
        let checkpoint = StoreCheckpoint {
            funcs: self.funcs.len(),
            tables: self.tables.len(),
            memories: self.memories.len(),
            globals: self.globals.len(),
            instances: self.instances.len(),
            memory_bytes: self.memory_bytes,
            table_elements: self.table_elements,
        };
        let result = self.instantiate_inner(module, imports);
        // Import matching and allocation finish before the instance enters the store. Failures in
        // those phases expose no store mutation. Once initialization starts, however, earlier
        // segment writes and allocated entities remain observable after a later trap.
        if result.is_err() && self.instances.len() == checkpoint.instances {
            self.funcs.truncate(checkpoint.funcs);
            self.tables.truncate(checkpoint.tables);
            self.memories.truncate(checkpoint.memories);
            self.globals.truncate(checkpoint.globals);
            self.memory_bytes = checkpoint.memory_bytes;
            self.table_elements = checkpoint.table_elements;
        }
        result
    }

    fn instantiate_inner(&mut self, module: Rc<Module>, imports: Imports) -> Result<usize, String> {
        let inst_idx = self.instances.len();
        let mut func_addrs = Vec::new();
        let mut table_addrs = Vec::new();
        let mut mem_addrs = Vec::new();
        let mut global_addrs = Vec::new();

        // Imports first (they occupy the low indices of each space).
        let mut host_funcs = imports.funcs.into_iter();
        let mut imported_func_addrs = imports.func_addrs.into_iter();
        let mut imported_table_addrs = imports.table_addrs.into_iter();
        let mut imported_table_fallback = imports.table_addr;
        let mut imp_globals = imports.global_addrs.into_iter();
        for imp in &module.imports {
            match &imp.kind {
                crate::wasm::ImportKind::Func(type_index) => {
                    let (id, ty) = host_funcs.next().ok_or("wasm: missing function import")?;
                    let expected = module
                        .types
                        .get(*type_index as usize)
                        .ok_or("wasm: function import type index out of range")?;
                    if &ty != expected {
                        return Err("wasm: imported function type does not match".into());
                    }
                    if let Some(address) = imported_func_addrs.next().flatten() {
                        let actual = self
                            .funcs
                            .get(address)
                            .and_then(Option::as_ref)
                            .ok_or("wasm: imported function address out of range")?;
                        if actual.ty() != ty {
                            return Err("wasm: imported function type does not match".into());
                        }
                        func_addrs.push(address);
                    } else {
                        self.funcs.push(Some(FuncEntity::Host { id, ty }));
                        func_addrs.push(self.funcs.len() - 1);
                    }
                }
                crate::wasm::ImportKind::Table(expected) => {
                    let addr = imported_table_addrs
                        .next()
                        .or_else(|| imported_table_fallback.take())
                        .ok_or("wasm: missing table import")?;
                    let actual = self
                        .tables
                        .get(addr)
                        .and_then(Option::as_ref)
                        .ok_or("wasm: imported table address out of range")?;
                    if actual.elem_type != expected.elem
                        || !limits_match(actual.elems.len() as u32, actual.max, expected.limits)
                    {
                        return Err("wasm: imported table type does not match".into());
                    }
                    table_addrs.push(addr);
                }
                crate::wasm::ImportKind::Memory(expected) => {
                    let addr = imports.mem_addr.ok_or("wasm: missing memory import")?;
                    let actual = self
                        .memories
                        .get(addr)
                        .and_then(Option::as_ref)
                        .ok_or("wasm: imported memory address out of range")?;
                    if !limits_match(
                        (actual.bytes.borrow().len() / PAGE_SIZE) as u32,
                        actual.max,
                        *expected,
                    ) {
                        return Err("wasm: imported memory type does not match".into());
                    }
                    mem_addrs.push(addr);
                }
                crate::wasm::ImportKind::Global(expected) => {
                    let addr = imp_globals.next().ok_or("wasm: missing global import")?;
                    let actual = self
                        .globals
                        .get(addr)
                        .and_then(Option::as_ref)
                        .ok_or("wasm: imported global address out of range")?;
                    if actual.ty != expected.val || actual.mutable != expected.mutable {
                        return Err("wasm: imported global type does not match".into());
                    }
                    global_addrs.push(addr);
                }
            }
        }

        // Defined functions.
        for (i, &type_idx) in module.func_types.iter().enumerate() {
            let ty = module.types[type_idx as usize].clone();
            let body = &module.code[i];
            let labels = scan_labels(&body.code)?;
            self.funcs.push(Some(FuncEntity::Wasm {
                compiled: Rc::new(Compiled {
                    ty,
                    locals: body.locals.clone(),
                    code: body.code.clone(),
                    labels,
                }),
                instance: inst_idx,
            }));
            func_addrs.push(self.funcs.len() - 1);
        }
        // Defined memory (single, MVP) and tables.
        if let Some(l) = module.memories.first() {
            let a = self.alloc_memory(l.min as usize, l.max)?;
            mem_addrs.push(a);
        }
        for t in &module.tables {
            let a = self.alloc_table_typed(t.limits.min as usize, t.limits.max, t.elem)?;
            table_addrs.push(a);
        }
        // Defined globals: each init expr sees the globals resolved so far (by value).
        for g in &module.globals {
            let seen: Vec<Val> = global_addrs
                .iter()
                .map(|&a| self.globals[a].as_ref().expect("live global").val)
                .collect();
            let v = eval_const_expr(&g.init, &seen)?;
            let a = self.alloc_global(v, g.ty.mutable, g.ty.val);
            global_addrs.push(a);
        }

        let inst = Rc::new(Instance {
            module: Rc::clone(&module),
            func_addrs,
            table_addrs,
            mem_addrs,
            global_addrs,
            data_live: std::cell::RefCell::new(
                module
                    .data
                    .iter()
                    .map(|segment| segment.active.is_none())
                    .collect(),
            ),
        });
        let seen_globals: Vec<Val> = inst
            .global_addrs
            .iter()
            .map(|&a| self.globals[a].as_ref().expect("live global").val)
            .collect();

        // Allocation precedes initialization in the normative algorithm. Keep the instance in the
        // store even if a later segment traps: an earlier segment may already have exposed one of
        // its functions through an imported table.
        self.instances.push(Some(Rc::clone(&inst)));
        for seg in &module.elems {
            let offset = eval_const_expr(&seg.offset, &seen_globals)?.i32() as u32 as usize;
            let table_addr = *inst
                .table_addrs
                .get(seg.table as usize)
                .ok_or("wasm: element segment references missing table")?;
            let end = offset
                .checked_add(seg.func_indices.len())
                .ok_or("wasm: element segment offset overflow")?;
            if end
                > self.tables[table_addr]
                    .as_ref()
                    .expect("live table")
                    .elems
                    .len()
            {
                return Err("wasm: element segment out of table bounds".into());
            }
            if seg
                .func_indices
                .iter()
                .any(|&f| f as usize >= inst.func_addrs.len())
            {
                return Err("wasm: elem func index out of range".into());
            }
            for (slot, &func_index) in seg.func_indices.iter().enumerate() {
                self.tables[table_addr].as_mut().expect("live table").elems[offset + slot] =
                    Some(inst.func_addrs[func_index as usize]);
            }
        }
        for seg in &module.data {
            if let Some((_memory, offset_expr)) = &seg.active {
                let offset = eval_const_expr(offset_expr, &seen_globals)?.i32() as u32 as usize;
                let memory_addr = *inst
                    .mem_addrs
                    .first()
                    .ok_or("wasm: data segment but no memory")?;
                let end = offset
                    .checked_add(seg.bytes.len())
                    .ok_or("wasm: data offset overflow")?;
                if end
                    > self.memories[memory_addr]
                        .as_ref()
                        .expect("live memory")
                        .bytes
                        .borrow()
                        .len()
                {
                    return Err("wasm: data segment out of memory bounds".into());
                }
                self.memories[memory_addr]
                    .as_ref()
                    .expect("live memory")
                    .bytes
                    .borrow_mut()[offset..end]
                    .copy_from_slice(&seg.bytes);
            }
        }
        Ok(inst_idx)
    }

    /// The store address of instance `inst_idx`'s export named `name`, plus its kind.
    pub fn export_addr(
        &self,
        inst_idx: usize,
        name: &str,
    ) -> Option<(crate::wasm::ExportKind, usize)> {
        let inst = self.instances.get(inst_idx)?.as_ref()?;
        let e = inst.module.exports.iter().find(|e| e.name == name)?;
        let addr = match e.kind {
            crate::wasm::ExportKind::Func => *inst.func_addrs.get(e.index as usize)?,
            crate::wasm::ExportKind::Memory => *inst.mem_addrs.get(e.index as usize)?,
            crate::wasm::ExportKind::Table => *inst.table_addrs.get(e.index as usize)?,
            crate::wasm::ExportKind::Global => *inst.global_addrs.get(e.index as usize)?,
        };
        Some((e.kind, addr))
    }

    fn mem_effective_addr(
        mem_len: usize,
        code: &[u8],
        ip: &mut usize,
        stack: &mut Vec<Val>,
        size: usize,
    ) -> Result<usize, String> {
        let _align = read_uleb(code, ip)?;
        let offset = read_uleb(code, ip)? as usize;
        let base = stack.pop().ok_or("wasm: stack underflow")?.i32() as u32 as usize;
        let addr = base
            .checked_add(offset)
            .ok_or("wasm: memory address overflow")?;
        if addr.checked_add(size).is_none_or(|end| end > mem_len) {
            return Err("wasm: out of bounds memory access".into());
        }
        Ok(addr)
    }

    fn mem_op(
        &mut self,
        mem_addr: usize,
        op: u8,
        code: &[u8],
        ip: &mut usize,
        stack: &mut Vec<Val>,
    ) -> Result<(), String> {
        let storage = Rc::clone(
            &self.memories[mem_addr]
                .as_ref()
                .ok_or("wasm: dead memory address")?
                .bytes,
        );
        let mut mem = storage.borrow_mut();
        macro_rules! load {
            ($size:expr, $conv:expr) => {{
                let a = Self::mem_effective_addr(mem.len(), code, ip, stack, $size)?;
                let bytes = &mem[a..a + $size];
                stack.push($conv(bytes));
            }};
        }
        macro_rules! store {
            ($t:ident, $size:expr, $bytes:expr) => {{
                let v = stack.pop().ok_or("wasm: stack underflow")?.$t();
                let a = Self::mem_effective_addr(mem.len(), code, ip, stack, $size)?;
                let b = $bytes(v);
                mem[a..a + $size].copy_from_slice(&b);
            }};
        }
        match op {
            0x28 => load!(4, |b: &[u8]| Val::I32(i32::from_le_bytes(
                b.try_into().unwrap()
            ))),
            0x29 => load!(8, |b: &[u8]| Val::I64(i64::from_le_bytes(
                b.try_into().unwrap()
            ))),
            0x2a => load!(4, |b: &[u8]| Val::F32(f32::from_le_bytes(
                b.try_into().unwrap()
            ))),
            0x2b => load!(8, |b: &[u8]| Val::F64(f64::from_le_bytes(
                b.try_into().unwrap()
            ))),
            0x2c => load!(1, |b: &[u8]| Val::I32(b[0] as i8 as i32)),
            0x2d => load!(1, |b: &[u8]| Val::I32(b[0] as i32)),
            0x2e => load!(2, |b: &[u8]| Val::I32(
                i16::from_le_bytes(b.try_into().unwrap()) as i32
            )),
            0x2f => load!(2, |b: &[u8]| Val::I32(
                u16::from_le_bytes(b.try_into().unwrap()) as i32
            )),
            0x30 => load!(1, |b: &[u8]| Val::I64(b[0] as i8 as i64)),
            0x31 => load!(1, |b: &[u8]| Val::I64(b[0] as i64)),
            0x32 => load!(2, |b: &[u8]| Val::I64(
                i16::from_le_bytes(b.try_into().unwrap()) as i64
            )),
            0x33 => load!(2, |b: &[u8]| Val::I64(
                u16::from_le_bytes(b.try_into().unwrap()) as i64
            )),
            0x34 => load!(4, |b: &[u8]| Val::I64(
                i32::from_le_bytes(b.try_into().unwrap()) as i64
            )),
            0x35 => load!(4, |b: &[u8]| Val::I64(
                u32::from_le_bytes(b.try_into().unwrap()) as i64
            )),
            0x36 => store!(i32, 4, |v: i32| v.to_le_bytes()),
            0x37 => store!(i64, 8, |v: i64| v.to_le_bytes()),
            0x38 => store!(f32, 4, |v: f32| v.to_le_bytes()),
            0x39 => store!(f64, 8, |v: f64| v.to_le_bytes()),
            0x3a => store!(i32, 1, |v: i32| [(v as u8)]),
            0x3b => store!(i32, 2, |v: i32| (v as u16).to_le_bytes()),
            0x3c => store!(i64, 1, |v: i64| [(v as u8)]),
            0x3d => store!(i64, 2, |v: i64| (v as u16).to_le_bytes()),
            0x3e => store!(i64, 4, |v: i64| (v as u32).to_le_bytes()),
            _ => return Err("wasm: bad memory opcode".into()),
        }
        Ok(())
    }

    fn op_fc(
        &mut self,
        inst: &Instance,
        mem_addr: Option<usize>,
        code: &[u8],
        ip: &mut usize,
        stack: &mut Vec<Val>,
    ) -> Result<(), String> {
        let sub = read_uleb(code, ip)?;
        match sub {
            // saturating truncation
            0 => {
                let v = stack.pop().unwrap().f32();
                stack.push(Val::I32(sat_i32(v as f64)));
            }
            1 => {
                let v = stack.pop().unwrap().f32();
                stack.push(Val::I32(sat_u32(v as f64) as i32));
            }
            2 => {
                let v = stack.pop().unwrap().f64();
                stack.push(Val::I32(sat_i32(v)));
            }
            3 => {
                let v = stack.pop().unwrap().f64();
                stack.push(Val::I32(sat_u32(v) as i32));
            }
            4 => {
                let v = stack.pop().unwrap().f32();
                stack.push(Val::I64(sat_i64(v as f64)));
            }
            5 => {
                let v = stack.pop().unwrap().f32();
                stack.push(Val::I64(sat_u64(v as f64) as i64));
            }
            6 => {
                let v = stack.pop().unwrap().f64();
                stack.push(Val::I64(sat_i64(v)));
            }
            7 => {
                let v = stack.pop().unwrap().f64();
                stack.push(Val::I64(sat_u64(v) as i64));
            }
            8 => {
                // memory.init dataidx memidx
                let data_index = read_uleb(code, ip)? as usize;
                if read_uleb(code, ip)? != 0 {
                    return Err("wasm: unsupported memory index".into());
                }
                let n = stack.pop().unwrap().i32() as u32 as usize;
                let src = stack.pop().unwrap().i32() as u32 as usize;
                let dst = stack.pop().unwrap().i32() as u32 as usize;
                let segment = inst
                    .module
                    .data
                    .get(data_index)
                    .ok_or("wasm: data segment index out of range")?;
                let live = *inst
                    .data_live
                    .borrow()
                    .get(data_index)
                    .ok_or("wasm: data segment index out of range")?;
                let source: &[u8] = if live { &segment.bytes } else { &[] };
                if src.checked_add(n).is_none_or(|end| end > source.len()) {
                    return Err("wasm: out of bounds memory.init source".into());
                }
                let storage = Rc::clone(
                    &self.memories[mem_addr.ok_or("wasm: no memory")?]
                        .as_ref()
                        .ok_or("wasm: dead memory address")?
                        .bytes,
                );
                let mut memory = storage.borrow_mut();
                if dst.checked_add(n).is_none_or(|end| end > memory.len()) {
                    return Err("wasm: out of bounds memory.init destination".into());
                }
                memory[dst..dst + n].copy_from_slice(&source[src..src + n]);
            }
            9 => {
                // data.drop dataidx
                let data_index = read_uleb(code, ip)? as usize;
                let mut live = inst.data_live.borrow_mut();
                *live
                    .get_mut(data_index)
                    .ok_or("wasm: data segment index out of range")? = false;
            }
            10 => {
                // memory.copy
                skip_bytes(code, ip, 2)?;
                let storage = Rc::clone(
                    &self.memories[mem_addr.ok_or("wasm: no memory")?]
                        .as_ref()
                        .ok_or("wasm: dead memory address")?
                        .bytes,
                );
                let mut mem = storage.borrow_mut();
                let n = stack.pop().unwrap().i32() as usize;
                let src = stack.pop().unwrap().i32() as u32 as usize;
                let dst = stack.pop().unwrap().i32() as u32 as usize;
                if src.checked_add(n).is_none_or(|end| end > mem.len())
                    || dst.checked_add(n).is_none_or(|end| end > mem.len())
                {
                    return Err("wasm: out of bounds memory.copy".into());
                }
                mem.copy_within(src..src + n, dst);
            }
            11 => {
                // memory.fill
                skip_bytes(code, ip, 1)?;
                let storage = Rc::clone(
                    &self.memories[mem_addr.ok_or("wasm: no memory")?]
                        .as_ref()
                        .ok_or("wasm: dead memory address")?
                        .bytes,
                );
                let mut mem = storage.borrow_mut();
                let n = stack.pop().unwrap().i32() as usize;
                let val = stack.pop().unwrap().i32() as u8;
                let dst = stack.pop().unwrap().i32() as u32 as usize;
                if dst.checked_add(n).is_none_or(|end| end > mem.len()) {
                    return Err("wasm: out of bounds memory.fill".into());
                }
                for b in &mut mem[dst..dst + n] {
                    *b = val;
                }
            }
            other => return Err(format!("wasm: unsupported 0xfc op {other}")),
        }
        Ok(())
    }
}

fn do_branch(
    ctrl: &mut Vec<Ctrl>,
    stack: &mut Vec<Val>,
    ip: &mut usize,
    l: usize,
) -> Result<(), String> {
    if l >= ctrl.len() {
        return Err("wasm: branch depth out of range".into());
    }
    let target_idx = ctrl.len() - 1 - l;
    let arity = ctrl[target_idx].arity;
    let height = ctrl[target_idx].stack_height;
    let is_loop = ctrl[target_idx].is_loop;
    let target_ip = if is_loop {
        ctrl[target_idx].start_ip
    } else {
        ctrl[target_idx].end_ip
    };
    // Keep the top `arity` values, drop the rest down to the label's base height.
    let keep_start = stack
        .len()
        .checked_sub(arity)
        .ok_or("wasm: branch stack underflow")?;
    let kept: Vec<Val> = stack.split_off(keep_start);
    stack.truncate(height);
    stack.extend(kept);
    // Popping to (and including) the target frame; a loop keeps its own frame.
    ctrl.truncate(if is_loop { target_idx + 1 } else { target_idx });
    *ip = target_ip;
    Ok(())
}

/// Parse a block's type immediate and return (param_count, result_count).
fn block_arity(module: &Module, code: &[u8], ip: &mut usize) -> Result<(usize, usize), String> {
    let b = *code.get(*ip).ok_or("wasm: truncated block type")?;
    if b == 0x40 {
        *ip += 1;
        Ok((0, 0))
    } else if matches!(b, 0x7f | 0x7e | 0x7d | 0x7c | 0x70 | 0x6f) {
        *ip += 1;
        Ok((0, 1))
    } else {
        let idx = read_sleb_bits(code, ip, 33)? as usize;
        let ty = module.types.get(idx).ok_or("wasm: bad block type index")?;
        Ok((ty.params.len(), ty.results.len()))
    }
}

fn same_type(a: &FuncType, b: &FuncType) -> bool {
    a.params == b.params && a.results == b.results
}

/// WebAssembly Core § Import Matching: an actual limit must provide at least the expected minimum;
/// when the expected type has a maximum, the actual type must also have one no greater than it.
fn limits_match(actual_min: u32, actual_max: Option<u32>, expected: Limits) -> bool {
    actual_min >= expected.min
        && expected
            .max
            .is_none_or(|expected_max| actual_max.is_some_and(|max| max <= expected_max))
}

fn sat_i32(v: f64) -> i32 {
    if v.is_nan() {
        0
    } else if v <= i32::MIN as f64 {
        i32::MIN
    } else if v >= i32::MAX as f64 {
        i32::MAX
    } else {
        v as i32
    }
}
fn sat_u32(v: f64) -> u32 {
    if v.is_nan() || v <= 0.0 {
        0
    } else if v >= u32::MAX as f64 {
        u32::MAX
    } else {
        v as u32
    }
}
fn sat_i64(v: f64) -> i64 {
    if v.is_nan() {
        0
    } else if v <= i64::MIN as f64 {
        i64::MIN
    } else if v >= i64::MAX as f64 {
        i64::MAX
    } else {
        v as i64
    }
}
fn sat_u64(v: f64) -> u64 {
    if v.is_nan() || v <= 0.0 {
        0
    } else if v >= u64::MAX as f64 {
        u64::MAX
    } else {
        v as u64
    }
}

// WebAssembly Core 2.0, Numeric Instructions: the non-saturating truncation operators trap when
// the truncated integer is outside the destination range. Rust float-to-int casts deliberately
// saturate, so using `as` without these checks changes observable Wasm behavior.
fn trunc_i32(v: f64) -> Result<i32, String> {
    let v = v.trunc();
    if !v.is_finite() || !(-((1u64 << 31) as f64)..(1u64 << 31) as f64).contains(&v) {
        Err("wasm: invalid conversion to integer".into())
    } else {
        Ok(v as i32)
    }
}

fn trunc_u32(v: f64) -> Result<u32, String> {
    let v = v.trunc();
    if !v.is_finite() || !(0.0..(1u64 << 32) as f64).contains(&v) {
        Err("wasm: invalid conversion to integer".into())
    } else {
        Ok(v as u32)
    }
}

fn trunc_i64(v: f64) -> Result<i64, String> {
    let v = v.trunc();
    if !v.is_finite() || !(-((1u128 << 63) as f64)..(1u128 << 63) as f64).contains(&v) {
        Err("wasm: invalid conversion to integer".into())
    } else {
        Ok(v as i64)
    }
}

fn trunc_u64(v: f64) -> Result<u64, String> {
    let v = v.trunc();
    if !v.is_finite() || !(0.0..(1u128 << 64) as f64).contains(&v) {
        Err("wasm: invalid conversion to integer".into())
    } else {
        Ok(v as u64)
    }
}

// WebAssembly propagates NaNs for min/max, unlike Rust's `f32::min`/`max`, and specifies the sign
// selected when the operands are equal zeros.
fn wasm_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        f32::from_bits(a.to_bits() | b.to_bits())
    } else if a < b {
        a
    } else {
        b
    }
}

fn wasm_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        f32::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

fn wasm_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        f64::from_bits(a.to_bits() | b.to_bits())
    } else if a < b {
        a
    } else {
        b
    }
}

fn wasm_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        f64::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

/// Numeric, comparison, and conversion opcodes (no immediates, no memory/control).
fn numeric(op: u8, stack: &mut Vec<Val>) -> Result<(), String> {
    match op {
        // i32 comparisons
        0x45 => {
            let a = stack.pop().unwrap().i32();
            stack.push(Val::I32((a == 0) as i32));
        }
        0x46 => cmpop!(stack, i32, |a, b| a == b),
        0x47 => cmpop!(stack, i32, |a, b| a != b),
        0x48 => cmpop!(stack, i32, |a, b| a < b),
        0x49 => cmpop!(stack, i32, |a: i32, b: i32| (a as u32) < (b as u32)),
        0x4a => cmpop!(stack, i32, |a, b| a > b),
        0x4b => cmpop!(stack, i32, |a: i32, b: i32| (a as u32) > (b as u32)),
        0x4c => cmpop!(stack, i32, |a, b| a <= b),
        0x4d => cmpop!(stack, i32, |a: i32, b: i32| (a as u32) <= (b as u32)),
        0x4e => cmpop!(stack, i32, |a, b| a >= b),
        0x4f => cmpop!(stack, i32, |a: i32, b: i32| (a as u32) >= (b as u32)),
        // i64 comparisons
        0x50 => {
            let a = stack.pop().unwrap().i64();
            stack.push(Val::I32((a == 0) as i32));
        }
        0x51 => cmpop!(stack, i64, |a, b| a == b),
        0x52 => cmpop!(stack, i64, |a, b| a != b),
        0x53 => cmpop!(stack, i64, |a, b| a < b),
        0x54 => cmpop!(stack, i64, |a: i64, b: i64| (a as u64) < (b as u64)),
        0x55 => cmpop!(stack, i64, |a, b| a > b),
        0x56 => cmpop!(stack, i64, |a: i64, b: i64| (a as u64) > (b as u64)),
        0x57 => cmpop!(stack, i64, |a, b| a <= b),
        0x58 => cmpop!(stack, i64, |a: i64, b: i64| (a as u64) <= (b as u64)),
        0x59 => cmpop!(stack, i64, |a, b| a >= b),
        0x5a => cmpop!(stack, i64, |a: i64, b: i64| (a as u64) >= (b as u64)),
        // f32 comparisons
        0x5b => cmpop!(stack, f32, |a, b| a == b),
        0x5c => cmpop!(stack, f32, |a, b| a != b),
        0x5d => cmpop!(stack, f32, |a, b| a < b),
        0x5e => cmpop!(stack, f32, |a, b| a > b),
        0x5f => cmpop!(stack, f32, |a, b| a <= b),
        0x60 => cmpop!(stack, f32, |a, b| a >= b),
        // f64 comparisons
        0x61 => cmpop!(stack, f64, |a, b| a == b),
        0x62 => cmpop!(stack, f64, |a, b| a != b),
        0x63 => cmpop!(stack, f64, |a, b| a < b),
        0x64 => cmpop!(stack, f64, |a, b| a > b),
        0x65 => cmpop!(stack, f64, |a, b| a <= b),
        0x66 => cmpop!(stack, f64, |a, b| a >= b),
        // i32 arithmetic
        0x67 => un(stack, |a: i32| a.leading_zeros() as i32),
        0x68 => un(stack, |a: i32| a.trailing_zeros() as i32),
        0x69 => un(stack, |a: i32| a.count_ones() as i32),
        0x6a => binop!(stack, i32, I32, |a: i32, b: i32| a.wrapping_add(b)),
        0x6b => binop!(stack, i32, I32, |a: i32, b: i32| a.wrapping_sub(b)),
        0x6c => binop!(stack, i32, I32, |a: i32, b: i32| a.wrapping_mul(b)),
        0x6d => return idiv_i32(stack, true),
        0x6e => return idiv_i32(stack, false),
        0x6f => return irem_i32(stack, true),
        0x70 => return irem_i32(stack, false),
        0x71 => binop!(stack, i32, I32, |a: i32, b: i32| a & b),
        0x72 => binop!(stack, i32, I32, |a: i32, b: i32| a | b),
        0x73 => binop!(stack, i32, I32, |a: i32, b: i32| a ^ b),
        0x74 => binop!(stack, i32, I32, |a: i32, b: i32| a.wrapping_shl(b as u32)),
        0x75 => binop!(stack, i32, I32, |a: i32, b: i32| a.wrapping_shr(b as u32)),
        0x76 => binop!(
            stack,
            i32,
            I32,
            |a: i32, b: i32| ((a as u32).wrapping_shr(b as u32)) as i32
        ),
        0x77 => binop!(stack, i32, I32, |a: i32, b: i32| a.rotate_left(b as u32)),
        0x78 => binop!(stack, i32, I32, |a: i32, b: i32| a.rotate_right(b as u32)),
        // i64 arithmetic
        0x79 => un64(stack, |a: i64| a.leading_zeros() as i64),
        0x7a => un64(stack, |a: i64| a.trailing_zeros() as i64),
        0x7b => un64(stack, |a: i64| a.count_ones() as i64),
        0x7c => binop!(stack, i64, I64, |a: i64, b: i64| a.wrapping_add(b)),
        0x7d => binop!(stack, i64, I64, |a: i64, b: i64| a.wrapping_sub(b)),
        0x7e => binop!(stack, i64, I64, |a: i64, b: i64| a.wrapping_mul(b)),
        0x7f => return idiv_i64(stack, true),
        0x80 => return idiv_i64(stack, false),
        0x81 => return irem_i64(stack, true),
        0x82 => return irem_i64(stack, false),
        0x83 => binop!(stack, i64, I64, |a: i64, b: i64| a & b),
        0x84 => binop!(stack, i64, I64, |a: i64, b: i64| a | b),
        0x85 => binop!(stack, i64, I64, |a: i64, b: i64| a ^ b),
        0x86 => binop!(stack, i64, I64, |a: i64, b: i64| a.wrapping_shl(b as u32)),
        0x87 => binop!(stack, i64, I64, |a: i64, b: i64| a.wrapping_shr(b as u32)),
        0x88 => binop!(
            stack,
            i64,
            I64,
            |a: i64, b: i64| ((a as u64).wrapping_shr(b as u32)) as i64
        ),
        0x89 => binop!(stack, i64, I64, |a: i64, b: i64| a.rotate_left(b as u32)),
        0x8a => binop!(stack, i64, I64, |a: i64, b: i64| a.rotate_right(b as u32)),
        // f32 arithmetic
        0x8b => unf32(stack, |a: f32| a.abs()),
        0x8c => unf32(stack, |a: f32| -a),
        0x8d => unf32(stack, |a: f32| a.ceil()),
        0x8e => unf32(stack, |a: f32| a.floor()),
        0x8f => unf32(stack, |a: f32| a.trunc()),
        0x90 => unf32(stack, round_ties_even_f32),
        0x91 => unf32(stack, |a: f32| a.sqrt()),
        0x92 => binop!(stack, f32, F32, |a: f32, b: f32| a + b),
        0x93 => binop!(stack, f32, F32, |a: f32, b: f32| a - b),
        0x94 => binop!(stack, f32, F32, |a: f32, b: f32| a * b),
        0x95 => binop!(stack, f32, F32, |a: f32, b: f32| a / b),
        0x96 => binop!(stack, f32, F32, wasm_min_f32),
        0x97 => binop!(stack, f32, F32, wasm_max_f32),
        0x98 => binop!(stack, f32, F32, |a: f32, b: f32| a.copysign(b)),
        // f64 arithmetic
        0x99 => unf64(stack, |a: f64| a.abs()),
        0x9a => unf64(stack, |a: f64| -a),
        0x9b => unf64(stack, |a: f64| a.ceil()),
        0x9c => unf64(stack, |a: f64| a.floor()),
        0x9d => unf64(stack, |a: f64| a.trunc()),
        0x9e => unf64(stack, round_ties_even_f64),
        0x9f => unf64(stack, |a: f64| a.sqrt()),
        0xa0 => binop!(stack, f64, F64, |a: f64, b: f64| a + b),
        0xa1 => binop!(stack, f64, F64, |a: f64, b: f64| a - b),
        0xa2 => binop!(stack, f64, F64, |a: f64, b: f64| a * b),
        0xa3 => binop!(stack, f64, F64, |a: f64, b: f64| a / b),
        0xa4 => binop!(stack, f64, F64, wasm_min_f64),
        0xa5 => binop!(stack, f64, F64, wasm_max_f64),
        0xa6 => binop!(stack, f64, F64, |a: f64, b: f64| a.copysign(b)),
        // conversions
        0xa7 => conv(stack, |v: Val| Val::I32(v.i64() as i32)), // i32.wrap_i64
        0xa8 => {
            let value = trunc_i32(stack.pop().unwrap().f32() as f64)?;
            stack.push(Val::I32(value));
        }
        0xa9 => {
            let value = trunc_u32(stack.pop().unwrap().f32() as f64)?;
            stack.push(Val::I32(value as i32));
        }
        0xaa => {
            let value = trunc_i32(stack.pop().unwrap().f64())?;
            stack.push(Val::I32(value));
        }
        0xab => {
            let value = trunc_u32(stack.pop().unwrap().f64())?;
            stack.push(Val::I32(value as i32));
        }
        0xac => conv(stack, |v: Val| Val::I64(v.i32() as i64)), // i64.extend_i32_s
        0xad => conv(stack, |v: Val| Val::I64(v.i32() as u32 as i64)), // i64.extend_i32_u
        0xae => {
            let value = trunc_i64(stack.pop().unwrap().f32() as f64)?;
            stack.push(Val::I64(value));
        }
        0xaf => {
            let value = trunc_u64(stack.pop().unwrap().f32() as f64)?;
            stack.push(Val::I64(value as i64));
        }
        0xb0 => {
            let value = trunc_i64(stack.pop().unwrap().f64())?;
            stack.push(Val::I64(value));
        }
        0xb1 => {
            let value = trunc_u64(stack.pop().unwrap().f64())?;
            stack.push(Val::I64(value as i64));
        }
        0xb2 => conv(stack, |v: Val| Val::F32(v.i32() as f32)),
        0xb3 => conv(stack, |v: Val| Val::F32(v.i32() as u32 as f32)),
        0xb4 => conv(stack, |v: Val| Val::F32(v.i64() as f32)),
        0xb5 => conv(stack, |v: Val| Val::F32(v.i64() as u64 as f32)),
        0xb6 => conv(stack, |v: Val| Val::F32(v.f64() as f32)),
        0xb7 => conv(stack, |v: Val| Val::F64(v.i32() as f64)),
        0xb8 => conv(stack, |v: Val| Val::F64(v.i32() as u32 as f64)),
        0xb9 => conv(stack, |v: Val| Val::F64(v.i64() as f64)),
        0xba => conv(stack, |v: Val| Val::F64(v.i64() as u64 as f64)),
        0xbb => conv(stack, |v: Val| Val::F64(v.f32() as f64)),
        0xbc => conv(stack, |v: Val| Val::I32(v.f32().to_bits() as i32)),
        0xbd => conv(stack, |v: Val| Val::I64(v.f64().to_bits() as i64)),
        0xbe => conv(stack, |v: Val| Val::F32(f32::from_bits(v.i32() as u32))),
        0xbf => conv(stack, |v: Val| Val::F64(f64::from_bits(v.i64() as u64))),
        // sign extension
        0xc0 => conv(stack, |v: Val| Val::I32(v.i32() as i8 as i32)),
        0xc1 => conv(stack, |v: Val| Val::I32(v.i32() as i16 as i32)),
        0xc2 => conv(stack, |v: Val| Val::I64(v.i64() as i8 as i64)),
        0xc3 => conv(stack, |v: Val| Val::I64(v.i64() as i16 as i64)),
        0xc4 => conv(stack, |v: Val| Val::I64(v.i64() as i32 as i64)),
        other => return Err(format!("wasm: unsupported opcode 0x{other:x}")),
    }
    Ok(())
}

fn un(stack: &mut Vec<Val>, f: impl Fn(i32) -> i32) {
    let a = stack.pop().unwrap().i32();
    stack.push(Val::I32(f(a)));
}
fn un64(stack: &mut Vec<Val>, f: impl Fn(i64) -> i64) {
    let a = stack.pop().unwrap().i64();
    stack.push(Val::I64(f(a)));
}
fn unf32(stack: &mut Vec<Val>, f: impl Fn(f32) -> f32) {
    let a = stack.pop().unwrap().f32();
    stack.push(Val::F32(f(a)));
}
fn unf64(stack: &mut Vec<Val>, f: impl Fn(f64) -> f64) {
    let a = stack.pop().unwrap().f64();
    stack.push(Val::F64(f(a)));
}
fn conv(stack: &mut Vec<Val>, f: impl Fn(Val) -> Val) {
    let a = stack.pop().unwrap();
    stack.push(f(a));
}

fn round_ties_even_f32(a: f32) -> f32 {
    a.round_ties_even()
}
fn round_ties_even_f64(a: f64) -> f64 {
    a.round_ties_even()
}

fn idiv_i32(stack: &mut Vec<Val>, signed: bool) -> Result<(), String> {
    let b = stack.pop().unwrap().i32();
    let a = stack.pop().unwrap().i32();
    if b == 0 {
        return Err("wasm: integer divide by zero".into());
    }
    let r = if signed {
        if a == i32::MIN && b == -1 {
            return Err("wasm: integer overflow".into());
        }
        a.wrapping_div(b)
    } else {
        ((a as u32) / (b as u32)) as i32
    };
    stack.push(Val::I32(r));
    Ok(())
}
fn irem_i32(stack: &mut Vec<Val>, signed: bool) -> Result<(), String> {
    let b = stack.pop().unwrap().i32();
    let a = stack.pop().unwrap().i32();
    if b == 0 {
        return Err("wasm: integer divide by zero".into());
    }
    let r = if signed {
        a.wrapping_rem(b)
    } else {
        ((a as u32) % (b as u32)) as i32
    };
    stack.push(Val::I32(r));
    Ok(())
}
fn idiv_i64(stack: &mut Vec<Val>, signed: bool) -> Result<(), String> {
    let b = stack.pop().unwrap().i64();
    let a = stack.pop().unwrap().i64();
    if b == 0 {
        return Err("wasm: integer divide by zero".into());
    }
    let r = if signed {
        if a == i64::MIN && b == -1 {
            return Err("wasm: integer overflow".into());
        }
        a.wrapping_div(b)
    } else {
        ((a as u64) / (b as u64)) as i64
    };
    stack.push(Val::I64(r));
    Ok(())
}
fn irem_i64(stack: &mut Vec<Val>, signed: bool) -> Result<(), String> {
    let b = stack.pop().unwrap().i64();
    let a = stack.pop().unwrap().i64();
    if b == 0 {
        return Err("wasm: integer divide by zero".into());
    }
    let r = if signed {
        a.wrapping_rem(b)
    } else {
        ((a as u64) % (b as u64)) as i64
    };
    stack.push(Val::I64(r));
    Ok(())
}

/// Evaluate a constant init expression (used for globals, data/elem offsets). Only const opcodes
/// and `global.get` of an already-initialized global.
pub fn eval_const_expr(code: &[u8], globals: &[Val]) -> Result<Val, String> {
    let mut ip = 0;
    let mut result = Val::I32(0);
    while ip < code.len() {
        let op = code[ip];
        ip += 1;
        match op {
            0x41 => result = Val::I32(read_sleb_bits(code, &mut ip, 32)? as i32),
            0x42 => result = Val::I64(read_sleb(code, &mut ip)?),
            0x43 => {
                result = Val::F32(f32::from_le_bytes(read_fixed::<4>(code, &mut ip)?));
            }
            0x44 => {
                result = Val::F64(f64::from_le_bytes(read_fixed::<8>(code, &mut ip)?));
            }
            0x23 => {
                let g = read_uleb(code, &mut ip)? as usize;
                result = *globals
                    .get(g)
                    .ok_or("wasm: const expr global out of range")?;
            }
            0xd0 => {
                skip_bytes(code, &mut ip, 1)?;
                result = Val::Ref(None);
            }
            0xd2 => {
                result = Val::Ref(Some(read_uleb(code, &mut ip)? as u32));
            }
            0x0b => break,
            other => return Err(format!("wasm: bad const expr opcode 0x{other:x}")),
        }
    }
    Ok(result)
}
