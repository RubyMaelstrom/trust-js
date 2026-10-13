//! WebAssembly binary-format decoder (the MVP + a few post-MVP proposals: multi-value results,
//! sign-extension, non-trapping conversions, bulk memory). Produces a [`Module`] the interpreter in
//! `exec.rs` runs. Decoding enforces the WebAssembly binary format and implementation resource
//! limits; [`super::validate`] then applies the normative module/instruction validation algorithm
//! before a module can be compiled or instantiated.

use std::rc::Rc;

// WebAssembly Core, Appendix "Implementation Limitations" permits embedders to impose reasonably
// large syntactic/binary limits. These bounds keep a tiny malicious module from expanding counts
// into unbounded host allocations while remaining well above ordinary web workloads.
pub const MAX_MODULE_BYTES: usize = 64 * 1024 * 1024;
const MAX_SECTION_ENTRIES: usize = 1_000_000;
const MAX_FUNCTIONS: usize = 100_000;
const MAX_LOCALS_PER_FUNCTION: usize = 100_000;
const MAX_FUNCTION_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_NAME_BYTES: usize = 1024 * 1024;
const MAX_CONST_EXPR_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
    FuncRef,
    ExternRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

#[derive(Debug, Clone)]
pub enum ImportKind {
    Func(u32), // type index
    Table(TableType),
    Memory(Limits),
    Global(GlobalType),
}

#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub kind: ImportKind,
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct TableType {
    pub elem: ValType,
    pub limits: Limits,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalType {
    pub val: ValType,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct Global {
    pub ty: GlobalType,
    pub init: Vec<u8>, // a constant init expression (raw bytes, ending in `end`)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportKind {
    Func,
    Table,
    Memory,
    Global,
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
    pub kind: ExportKind,
    pub index: u32,
}

#[derive(Debug, Clone)]
pub struct FuncBody {
    pub locals: Vec<ValType>, // flattened (each declared local, expanded from run-length groups)
    pub code: Vec<u8>,        // raw instruction bytes, up to and excluding the final `end`
}

#[derive(Debug, Clone)]
pub struct DataSegment {
    pub active: Option<(u32, Vec<u8>)>, // (memory index, offset init expr) for active segments
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ElemSegment {
    pub table: u32,
    pub offset: Vec<u8>, // offset init expr
    pub func_indices: Vec<u32>,
}

#[derive(Debug, Default)]
pub struct Module {
    pub types: Vec<FuncType>,
    pub imports: Vec<Import>,
    /// Type index for each *defined* function (imports excluded).
    pub func_types: Vec<u32>,
    pub tables: Vec<TableType>,
    pub memories: Vec<Limits>,
    pub globals: Vec<Global>,
    pub exports: Vec<Export>,
    pub start: Option<u32>,
    pub elems: Vec<ElemSegment>,
    pub code: Vec<FuncBody>,
    pub data: Vec<DataSegment>,
    /// The optional DataCount section value. In the binary format this section precedes Code even
    /// though its numeric id is 12; validation checks it against the actual data segment count.
    pub data_count: Option<u32>,
    /// Number of imported functions (defined funcs are indexed after these).
    pub imported_func_count: u32,
    pub imported_table_count: u32,
    pub imported_mem_count: u32,
    pub imported_global_count: u32,
}

// ---- byte reader with LEB128 ------------------------------------------------------------------

pub struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

type R<T> = Result<T, String>;

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }
    pub fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }
    pub fn byte(&mut self) -> R<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or("wasm: unexpected end of input")?;
        self.pos += 1;
        Ok(b)
    }
    pub fn bytes(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or("wasm: length overflow")?;
        let s = self
            .data
            .get(self.pos..end)
            .ok_or("wasm: unexpected end of input")?;
        self.pos = end;
        Ok(s)
    }
    pub fn u32(&mut self) -> R<u32> {
        Ok(self.unsigned_leb(32)? as u32)
    }
    /// Unsigned LEB128.
    pub fn u64_leb(&mut self) -> R<u64> {
        self.unsigned_leb(64)
    }
    pub(crate) fn unsigned_leb(&mut self, bits: u32) -> R<u64> {
        let max_bytes = bits.div_ceil(7) as usize;
        let mut result = 0u64;
        for byte_index in 0..max_bytes {
            let b = self.byte()?;
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
    /// Signed LEB128.
    pub fn i64_leb(&mut self) -> R<i64> {
        self.signed_leb(64)
    }
    pub(crate) fn signed_leb(&mut self, bits: u32) -> R<i64> {
        let max_bytes = bits.div_ceil(7) as usize;
        let mut result = 0i64;
        for byte_index in 0..max_bytes {
            let b = self.byte()?;
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
                if consumed < bits && (b & 0x40) != 0 {
                    result |= -1i64 << consumed;
                }
                return Ok(result);
            }
        }
        Err("wasm: signed LEB128 too long".into())
    }
    pub fn i32(&mut self) -> R<i32> {
        Ok(self.signed_leb(32)? as i32)
    }
    pub fn f32(&mut self) -> R<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    pub fn f64(&mut self) -> R<f64> {
        Ok(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    pub fn name(&mut self) -> R<String> {
        let len = self.u32()? as usize;
        if len > MAX_NAME_BYTES {
            return Err("wasm: name exceeds implementation limit".into());
        }
        let bytes = self.bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| "wasm: invalid utf-8 in name".into())
    }

    fn count(&mut self, limit: usize, what: &str) -> R<usize> {
        let count = self.u32()? as usize;
        if count > limit {
            return Err(format!("wasm: {what} count exceeds implementation limit"));
        }
        Ok(count)
    }
}

fn reserve<T>(items: &mut Vec<T>, additional: usize, what: &str) -> R<()> {
    items
        .try_reserve(additional)
        .map_err(|_| format!("wasm: cannot allocate {what}"))
}

fn val_type(b: u8) -> Result<ValType, String> {
    match b {
        0x7f => Ok(ValType::I32),
        0x7e => Ok(ValType::I64),
        0x7d => Ok(ValType::F32),
        0x7c => Ok(ValType::F64),
        0x70 => Ok(ValType::FuncRef),
        0x6f => Ok(ValType::ExternRef),
        other => Err(format!("wasm: unknown value type 0x{other:x}")),
    }
}

fn limits(r: &mut Reader, upper: u32, what: &str) -> Result<Limits, String> {
    let flag = r.byte()?;
    if !matches!(flag, 0x00 | 0x01) {
        return Err(format!("wasm: unsupported {what} limits flags 0x{flag:x}"));
    }
    let min = r.u32()?;
    let max = if flag & 1 != 0 { Some(r.u32()?) } else { None };
    if min > upper || max.is_some_and(|max| max > upper || min > max) {
        return Err(format!("wasm: invalid {what} limits"));
    }
    Ok(Limits { min, max })
}

fn table_type(r: &mut Reader) -> Result<TableType, String> {
    let elem = val_type(r.byte()?)?;
    if !matches!(elem, ValType::FuncRef | ValType::ExternRef) {
        return Err("wasm: table element type must be a reference type".into());
    }
    Ok(TableType {
        elem,
        limits: limits(r, u32::MAX, "table")?,
    })
}

fn global_type(r: &mut Reader) -> Result<GlobalType, String> {
    let val = val_type(r.byte()?)?;
    let mutable = match r.byte()? {
        0 => false,
        1 => true,
        other => return Err(format!("wasm: invalid global mutability {other}")),
    };
    Ok(GlobalType { val, mutable })
}

/// Read a constant/offset init expression (raw bytes) up to and including the terminating `end`.
fn read_const_expr(r: &mut Reader) -> Result<Vec<u8>, String> {
    let start = r.pos;
    let mut instructions = 0usize;
    loop {
        if r.pos.saturating_sub(start) > MAX_CONST_EXPR_BYTES {
            return Err("wasm: constant expression exceeds implementation limit".into());
        }
        let op = r.byte()?;
        match op {
            0x41 => {
                r.i32()?;
            } // i32.const
            0x42 => {
                r.i64_leb()?;
            } // i64.const
            0x43 => {
                r.bytes(4)?;
            } // f32.const
            0x44 => {
                r.bytes(8)?;
            } // f64.const
            0x23 => {
                r.u32()?;
            } // global.get
            0xd0 => {
                val_type(r.byte()?)?;
            } // ref.null t
            0xd2 => {
                r.u32()?;
            } // ref.func
            0x0b => {
                break;
            }
            _ => return Err(format!("wasm: unsupported opcode 0x{op:x} in const expr")),
        }
        instructions += 1;
        if instructions > 1 {
            // TRust JS currently implements the MVP constant-expression subset. The validator also
            // checks the instruction's context and result type.
            return Err("wasm: extended constant expressions are not supported".into());
        }
    }
    Ok(r.data[start..r.pos].to_vec())
}

/// Skip past an instruction's immediate operands during a section scan (unused here but kept for
/// clarity of the const-expr reader's structure).
pub fn decode(data: &[u8]) -> Result<Rc<Module>, String> {
    if data.len() > MAX_MODULE_BYTES {
        return Err("wasm: module exceeds implementation byte limit".into());
    }
    let mut r = Reader::new(data);
    if r.bytes(4)? != b"\0asm" {
        return Err("wasm: bad magic".into());
    }
    if r.bytes(4)? != 1u32.to_le_bytes() {
        return Err("wasm: unsupported version".into());
    }

    let mut m = Module::default();
    let mut last_section_order = 0u8;
    while !r.eof() {
        let id = r.byte()?;
        let size = r.u32()? as usize;
        let section_bytes = r.bytes(size)?;
        let mut section = Reader::new(section_bytes);
        // Binary section order is not numeric: DataCount (id 12) precedes Code (id 10).
        if id != 0 {
            let order = section_order(id)?;
            if order <= last_section_order {
                return Err("wasm: sections out of order".into());
            }
            last_section_order = order;
        }
        match id {
            0 => {
                // Core binary format §5.5.3: a custom section always begins with a valid name.
                // Its payload is uninterpreted, but skipping the whole section would accept a
                // missing name or an overlong/overflowing name-length LEB.
                section.name()?;
                section.pos = section.data.len();
            }
            1 => decode_types(&mut section, &mut m)?,
            2 => decode_imports(&mut section, &mut m)?,
            3 => decode_functions(&mut section, &mut m)?,
            4 => {
                let count = section.count(MAX_SECTION_ENTRIES, "table")?;
                reserve(&mut m.tables, count, "tables")?;
                for _ in 0..count {
                    m.tables.push(table_type(&mut section)?);
                }
            }
            5 => {
                let count = section.count(MAX_SECTION_ENTRIES, "memory")?;
                reserve(&mut m.memories, count, "memories")?;
                for _ in 0..count {
                    m.memories.push(limits(&mut section, 65_536, "memory")?);
                }
            }
            6 => decode_globals(&mut section, &mut m)?,
            7 => decode_exports(&mut section, &mut m)?,
            8 => m.start = Some(section.u32()?),
            9 => decode_elems(&mut section, &mut m)?,
            10 => decode_code(&mut section, &mut m)?,
            11 => decode_data(&mut section, &mut m)?,
            12 => {
                m.data_count = Some(section.u32()?);
            }
            _ => unreachable!("section_order rejected unknown section"),
        }
        if !section.eof() {
            return Err(format!("wasm: section {id} size mismatch"));
        }
    }
    super::validate::validate_module(&m)?;
    Ok(Rc::new(m))
}

fn section_order(id: u8) -> R<u8> {
    match id {
        1..=9 => Ok(id),
        12 => Ok(10), // DataCount
        10 => Ok(11), // Code
        11 => Ok(12), // Data
        other => Err(format!("wasm: unknown section id {other}")),
    }
}

fn decode_types(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_SECTION_ENTRIES, "type")?;
    reserve(&mut m.types, count, "types")?;
    for _ in 0..count {
        if r.byte()? != 0x60 {
            return Err("wasm: expected func type (0x60)".into());
        }
        let param_count = r.count(MAX_SECTION_ENTRIES, "function parameter")?;
        let mut params = Vec::new();
        reserve(&mut params, param_count, "function parameters")?;
        for _ in 0..param_count {
            params.push(val_type(r.byte()?)?);
        }
        let result_count = r.count(MAX_SECTION_ENTRIES, "function result")?;
        let mut results = Vec::new();
        reserve(&mut results, result_count, "function results")?;
        for _ in 0..result_count {
            results.push(val_type(r.byte()?)?);
        }
        m.types.push(FuncType { params, results });
    }
    Ok(())
}

fn decode_imports(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_SECTION_ENTRIES, "import")?;
    reserve(&mut m.imports, count, "imports")?;
    for _ in 0..count {
        let module = r.name()?;
        let name = r.name()?;
        let kind = match r.byte()? {
            0x00 => {
                let t = r.u32()?;
                if t as usize >= m.types.len() {
                    return Err("wasm: imported function type index out of range".into());
                }
                m.imported_func_count = m
                    .imported_func_count
                    .checked_add(1)
                    .ok_or("wasm: too many imported functions")?;
                ImportKind::Func(t)
            }
            0x01 => {
                m.imported_table_count = m
                    .imported_table_count
                    .checked_add(1)
                    .ok_or("wasm: too many imported tables")?;
                ImportKind::Table(table_type(r)?)
            }
            0x02 => {
                m.imported_mem_count = m
                    .imported_mem_count
                    .checked_add(1)
                    .ok_or("wasm: too many imported memories")?;
                ImportKind::Memory(limits(r, 65_536, "memory")?)
            }
            0x03 => {
                m.imported_global_count = m
                    .imported_global_count
                    .checked_add(1)
                    .ok_or("wasm: too many imported globals")?;
                ImportKind::Global(global_type(r)?)
            }
            other => return Err(format!("wasm: unknown import kind {other}")),
        };
        m.imports.push(Import { module, name, kind });
    }
    Ok(())
}

fn decode_functions(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_FUNCTIONS, "function")?;
    reserve(&mut m.func_types, count, "functions")?;
    for _ in 0..count {
        let t = r.u32()?;
        if t as usize >= m.types.len() {
            return Err("wasm: function type index out of range".into());
        }
        m.func_types.push(t);
    }
    Ok(())
}

fn decode_globals(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_SECTION_ENTRIES, "global")?;
    reserve(&mut m.globals, count, "globals")?;
    for _ in 0..count {
        let ty = global_type(r)?;
        let init = read_const_expr(r)?;
        m.globals.push(Global { ty, init });
    }
    Ok(())
}

fn decode_exports(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_SECTION_ENTRIES, "export")?;
    reserve(&mut m.exports, count, "exports")?;
    for _ in 0..count {
        let name = r.name()?;
        let kind = match r.byte()? {
            0x00 => ExportKind::Func,
            0x01 => ExportKind::Table,
            0x02 => ExportKind::Memory,
            0x03 => ExportKind::Global,
            other => return Err(format!("wasm: unknown export kind {other}")),
        };
        let index = r.u32()?;
        m.exports.push(Export { name, kind, index });
    }
    Ok(())
}

fn decode_elems(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_SECTION_ENTRIES, "element segment")?;
    reserve(&mut m.elems, count, "element segments")?;
    for _ in 0..count {
        let flags = r.u32()?;
        // Support the common active-segment forms (flags 0 and 2); others are rejected.
        match flags {
            0 => {
                let offset = read_const_expr(r)?;
                let item_count = r.count(MAX_SECTION_ENTRIES, "element")?;
                let mut func_indices = Vec::new();
                reserve(&mut func_indices, item_count, "element indices")?;
                for _ in 0..item_count {
                    func_indices.push(r.u32()?);
                }
                m.elems.push(ElemSegment {
                    table: 0,
                    offset,
                    func_indices,
                });
            }
            2 => {
                let table = r.u32()?;
                let offset = read_const_expr(r)?;
                if r.byte()? != 0x00 {
                    return Err("wasm: element kind must be funcref".into());
                }
                let item_count = r.count(MAX_SECTION_ENTRIES, "element")?;
                let mut func_indices = Vec::new();
                reserve(&mut func_indices, item_count, "element indices")?;
                for _ in 0..item_count {
                    func_indices.push(r.u32()?);
                }
                m.elems.push(ElemSegment {
                    table,
                    offset,
                    func_indices,
                });
            }
            other => return Err(format!("wasm: unsupported element segment kind {other}")),
        }
    }
    Ok(())
}

fn decode_code(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_FUNCTIONS, "code body")?;
    reserve(&mut m.code, count, "code bodies")?;
    for _ in 0..count {
        let size = r.u32()? as usize;
        if size > MAX_FUNCTION_BODY_BYTES {
            return Err("wasm: function body exceeds implementation byte limit".into());
        }
        let body_bytes = r.bytes(size)?;
        let mut body = Reader::new(body_bytes);
        let mut locals = Vec::new();
        let group_count = body.count(MAX_LOCALS_PER_FUNCTION, "local declaration group")?;
        for _ in 0..group_count {
            let count = body.u32()? as usize;
            let ty = val_type(body.byte()?)?;
            let new_len = locals
                .len()
                .checked_add(count)
                .ok_or("wasm: local count overflow")?;
            if new_len > MAX_LOCALS_PER_FUNCTION {
                return Err("wasm: local count exceeds implementation limit".into());
            }
            reserve(&mut locals, count, "locals")?;
            locals.resize(new_len, ty);
        }
        // Remaining bytes (minus the trailing `end`) are the instruction stream.
        let remaining = body.data.len().saturating_sub(body.pos);
        if remaining == 0 || body.data.last() != Some(&0x0b) {
            return Err("wasm: function body not terminated by end".into());
        }
        let code = body.data[body.pos..body.data.len() - 1].to_vec();
        body.pos = body.data.len();
        m.code.push(FuncBody { locals, code });
    }
    if m.code.len() != m.func_types.len() {
        return Err("wasm: code/function count mismatch".into());
    }
    Ok(())
}

fn decode_data(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let count = r.count(MAX_SECTION_ENTRIES, "data segment")?;
    reserve(&mut m.data, count, "data segments")?;
    for _ in 0..count {
        let flags = r.u32()?;
        match flags {
            0 => {
                let offset = read_const_expr(r)?;
                let len = r.u32()? as usize;
                let bytes = r.bytes(len)?.to_vec();
                m.data.push(DataSegment {
                    active: Some((0, offset)),
                    bytes,
                });
            }
            1 => {
                let len = r.u32()? as usize;
                let bytes = r.bytes(len)?.to_vec();
                m.data.push(DataSegment {
                    active: None,
                    bytes,
                });
            }
            2 => {
                let memidx = r.u32()?;
                let offset = read_const_expr(r)?;
                let len = r.u32()? as usize;
                let bytes = r.bytes(len)?.to_vec();
                m.data.push(DataSegment {
                    active: Some((memidx, offset)),
                    bytes,
                });
            }
            other => return Err(format!("wasm: unsupported data segment kind {other}")),
        }
    }
    Ok(())
}

/// The JS `WebAssembly.validate` operation: binary decoding and normative module validation must
/// both succeed. See WebAssembly JS API § WebAssembly.validate.
pub fn validate(data: &[u8]) -> bool {
    decode(data).is_ok()
}
