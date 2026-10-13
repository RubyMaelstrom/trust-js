//! WebAssembly Core validation for the instruction subset executed by `exec.rs`.
//!
//! This follows the normative stack-machine algorithm from WebAssembly Core,
//! Appendix "Validation Algorithm": a value stack, a control stack containing
//! each label's start/end types and height, and stack-polymorphic `Bot` values
//! after unconditional control transfer. Modules reach the executor only after
//! this pass succeeds.

use std::collections::HashSet;

use super::parse::{
    ExportKind, FuncBody, FuncType, GlobalType, ImportKind, Module, Reader, TableType, ValType,
};

type R<T> = Result<T, String>;

const MAX_VALIDATION_STACK: usize = 100_000;
const MAX_CONTROL_DEPTH: usize = 10_000;
const MAX_BR_TABLE_LABELS: usize = 100_000;
const MAX_INSTRUCTIONS_PER_FUNCTION: usize = 5_000_000;

pub(crate) fn validate_module(module: &Module) -> R<()> {
    let mut funcs: Vec<&FuncType> = Vec::new();
    let mut tables: Vec<TableType> = Vec::new();
    let mut memories = Vec::new();
    let mut globals: Vec<GlobalType> = Vec::new();
    funcs
        .try_reserve(module.imported_func_count as usize + module.func_types.len())
        .map_err(|_| "wasm: cannot allocate function validation context")?;

    for import in &module.imports {
        match import.kind {
            ImportKind::Func(type_index) => funcs.push(
                module
                    .types
                    .get(type_index as usize)
                    .ok_or("wasm: imported function type index out of range")?,
            ),
            ImportKind::Table(table) => {
                tables.push(table);
            }
            ImportKind::Memory(memory) => memories.push(memory),
            ImportKind::Global(global) => globals.push(global),
        }
    }
    // The current executor implements the single-memory model. Reject an unsupported index-space
    // shape rather than silently aliasing two memory imports.
    if module.imported_mem_count > 1 {
        return Err("wasm: multiple imported memories are not supported".into());
    }
    for &type_index in &module.func_types {
        funcs.push(
            module
                .types
                .get(type_index as usize)
                .ok_or("wasm: function type index out of range")?,
        );
    }
    tables.extend(module.tables.iter().copied());
    memories.extend(module.memories.iter().copied());
    globals.extend(module.globals.iter().map(|global| global.ty));

    // TRust JS currently implements the 32-bit, single-memory execution model. Multiple memories are
    // a distinct proposal and must not be silently collapsed to `memories[0]` by the executor.
    if memories.len() > 1 {
        return Err("wasm: multiple memories are not supported".into());
    }
    if tables.iter().any(|table| table.elem != ValType::FuncRef) {
        return Err("wasm: externref tables are not supported".into());
    }

    validate_exports(
        module,
        funcs.len(),
        tables.len(),
        memories.len(),
        globals.len(),
    )?;
    if let Some(start) = module.start {
        let ty = funcs
            .get(start as usize)
            .ok_or("wasm: start function index out of range")?;
        if !ty.params.is_empty() || !ty.results.is_empty() {
            return Err("wasm: start function must have type [] -> []".into());
        }
    }
    if let Some(count) = module.data_count {
        if count as usize != module.data.len() {
            return Err("wasm: data count does not match data section".into());
        }
    }

    let imported_globals = module.imported_global_count as usize;
    for global in &module.globals {
        validate_const_expr(
            &global.init,
            global.ty.val,
            &globals,
            imported_globals,
            funcs.len(),
        )?;
    }
    for segment in &module.elems {
        let table = tables
            .get(segment.table as usize)
            .ok_or("wasm: element segment table index out of range")?;
        if table.elem != ValType::FuncRef {
            return Err("wasm: function-index element segment requires funcref table".into());
        }
        validate_const_expr(
            &segment.offset,
            ValType::I32,
            &globals,
            imported_globals,
            funcs.len(),
        )?;
        if segment
            .func_indices
            .iter()
            .any(|&index| index as usize >= funcs.len())
        {
            return Err("wasm: element function index out of range".into());
        }
    }
    for segment in &module.data {
        if let Some((memory, offset)) = &segment.active {
            if *memory as usize >= memories.len() {
                return Err("wasm: data segment memory index out of range".into());
            }
            validate_const_expr(
                offset,
                ValType::I32,
                &globals,
                imported_globals,
                funcs.len(),
            )?;
        }
    }

    if module.code.len() != module.func_types.len() {
        return Err("wasm: code/function count mismatch".into());
    }
    let context = ModuleContext {
        types: &module.types,
        funcs: &funcs,
        tables: &tables,
        memories: memories.len(),
        globals: &globals,
        data_count: module.data_count.map(|count| count as usize),
    };
    for (defined_index, body) in module.code.iter().enumerate() {
        let type_index = module.func_types[defined_index] as usize;
        let ty = module
            .types
            .get(type_index)
            .ok_or("wasm: function type index out of range")?;
        FunctionValidator::new(&context, ty, body)?.validate()?;
    }
    Ok(())
}

fn validate_exports(
    module: &Module,
    functions: usize,
    tables: usize,
    memories: usize,
    globals: usize,
) -> R<()> {
    let mut names = HashSet::new();
    for export in &module.exports {
        if !names.insert(export.name.as_str()) {
            return Err("wasm: duplicate export name".into());
        }
        let limit = match export.kind {
            ExportKind::Func => functions,
            ExportKind::Table => tables,
            ExportKind::Memory => memories,
            ExportKind::Global => globals,
        };
        if export.index as usize >= limit {
            return Err("wasm: export index out of range".into());
        }
    }
    Ok(())
}

fn validate_const_expr(
    code: &[u8],
    expected: ValType,
    globals: &[GlobalType],
    imported_globals: usize,
    functions: usize,
) -> R<()> {
    let mut reader = Reader::new(code);
    let actual = match reader.byte()? {
        0x41 => {
            reader.i32()?;
            ValType::I32
        }
        0x42 => {
            reader.i64_leb()?;
            ValType::I64
        }
        0x43 => {
            reader.bytes(4)?;
            ValType::F32
        }
        0x44 => {
            reader.bytes(8)?;
            ValType::F64
        }
        0x23 => {
            let index = reader.u32()? as usize;
            let global = globals
                .get(index)
                .ok_or("wasm: constant global index out of range")?;
            if index >= imported_globals || global.mutable {
                return Err(
                    "wasm: constant global.get must reference an immutable imported global".into(),
                );
            }
            global.val
        }
        0xd0 => match reader.byte()? {
            0x70 => ValType::FuncRef,
            0x6f => ValType::ExternRef,
            _ => return Err("wasm: invalid ref.null heap type".into()),
        },
        0xd2 => {
            if reader.u32()? as usize >= functions {
                return Err("wasm: ref.func index out of range".into());
            }
            ValType::FuncRef
        }
        opcode => {
            return Err(format!(
                "wasm: invalid constant expression opcode 0x{opcode:x}"
            ))
        }
    };
    if actual != expected {
        return Err("wasm: constant expression result type mismatch".into());
    }
    if reader.byte()? != 0x0b || !reader.eof() {
        return Err("wasm: malformed constant expression".into());
    }
    Ok(())
}

struct ModuleContext<'a> {
    types: &'a [FuncType],
    funcs: &'a [&'a FuncType],
    tables: &'a [TableType],
    memories: usize,
    globals: &'a [GlobalType],
    data_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StackType {
    Known(ValType),
    Bot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlKind {
    Function,
    Block,
    Loop,
    If,
    Else,
}

#[derive(Clone)]
struct ControlFrame {
    kind: ControlKind,
    start_types: Vec<ValType>,
    end_types: Vec<ValType>,
    value_height: usize,
    unreachable: bool,
}

struct FunctionValidator<'a> {
    context: &'a ModuleContext<'a>,
    reader: Reader<'a>,
    locals: Vec<ValType>,
    returns: &'a [ValType],
    values: Vec<StackType>,
    controls: Vec<ControlFrame>,
    instruction_count: usize,
}

impl<'a> FunctionValidator<'a> {
    fn new(context: &'a ModuleContext<'a>, ty: &'a FuncType, body: &'a FuncBody) -> R<Self> {
        let local_count = ty
            .params
            .len()
            .checked_add(body.locals.len())
            .ok_or("wasm: local count overflow")?;
        if local_count > MAX_VALIDATION_STACK {
            return Err("wasm: local count exceeds validation limit".into());
        }
        let mut locals = Vec::new();
        locals
            .try_reserve(local_count)
            .map_err(|_| "wasm: cannot allocate local validation context")?;
        locals.extend_from_slice(&ty.params);
        locals.extend_from_slice(&body.locals);
        let mut validator = FunctionValidator {
            context,
            reader: Reader::new(&body.code),
            locals,
            returns: &ty.results,
            values: Vec::new(),
            controls: Vec::new(),
            instruction_count: 0,
        };
        validator.push_control(ControlKind::Function, &[], &ty.results)?;
        Ok(validator)
    }

    fn validate(mut self) -> R<()> {
        while !self.reader.eof() {
            self.instruction_count += 1;
            if self.instruction_count > MAX_INSTRUCTIONS_PER_FUNCTION {
                return Err("wasm: instruction count exceeds implementation limit".into());
            }
            let opcode = self.reader.byte()?;
            self.validate_opcode(opcode)?;
        }
        if self.controls.len() != 1 {
            return Err("wasm: unterminated structured control instruction".into());
        }
        self.pop_control()?;
        if !self.values.is_empty() {
            return Err("wasm: values remain after function validation".into());
        }
        Ok(())
    }

    fn validate_opcode(&mut self, opcode: u8) -> R<()> {
        match opcode {
            0x00 => self.mark_unreachable(),
            0x01 => Ok(()),
            0x02 | 0x03 | 0x04 => {
                let (params, results) = self.read_block_type()?;
                if opcode == 0x04 {
                    self.pop_expect(ValType::I32)?;
                }
                self.pop_types(&params)?;
                let kind = match opcode {
                    0x02 => ControlKind::Block,
                    0x03 => ControlKind::Loop,
                    _ => ControlKind::If,
                };
                self.push_control(kind, &params, &results)
            }
            0x05 => {
                let frame = self.pop_control()?;
                if frame.kind != ControlKind::If {
                    return Err("wasm: else without matching if".into());
                }
                self.push_control(ControlKind::Else, &frame.start_types, &frame.end_types)
            }
            0x0b => {
                let frame = self.pop_control()?;
                if frame.kind == ControlKind::Function {
                    return Err("wasm: function end appears before end of body".into());
                }
                if frame.kind == ControlKind::If && frame.start_types != frame.end_types {
                    return Err("wasm: if without else has incompatible input/result types".into());
                }
                self.push_types(&frame.end_types)
            }
            0x0c => {
                let depth = self.reader.u32()? as usize;
                let label = self.label_types(depth)?;
                self.pop_types(&label)?;
                self.mark_unreachable()
            }
            0x0d => {
                let depth = self.reader.u32()? as usize;
                self.pop_expect(ValType::I32)?;
                let label = self.label_types(depth)?;
                self.pop_types(&label)?;
                self.push_types(&label)
            }
            0x0e => self.validate_br_table(),
            0x0f => {
                let returns = self.returns.to_vec();
                self.pop_types(&returns)?;
                self.mark_unreachable()
            }
            0x10 => {
                let function = self.reader.u32()? as usize;
                let ty = *self
                    .context
                    .funcs
                    .get(function)
                    .ok_or("wasm: call function index out of range")?;
                self.pop_types(&ty.params)?;
                self.push_types(&ty.results)
            }
            0x11 => {
                let type_index = self.reader.u32()? as usize;
                let table_index = self.reader.u32()? as usize;
                let ty = self
                    .context
                    .types
                    .get(type_index)
                    .ok_or("wasm: call_indirect type index out of range")?;
                let table = self
                    .context
                    .tables
                    .get(table_index)
                    .ok_or("wasm: call_indirect table index out of range")?;
                if table.elem != ValType::FuncRef {
                    return Err("wasm: call_indirect requires a funcref table".into());
                }
                self.pop_expect(ValType::I32)?;
                self.pop_types(&ty.params)?;
                self.push_types(&ty.results)
            }
            0x1a => {
                self.pop_any()?;
                Ok(())
            }
            0x1b => self.validate_select(),
            0x1c => self.validate_typed_select(),
            0x20..=0x22 => self.validate_local(opcode),
            0x23 | 0x24 => self.validate_global(opcode),
            0x28..=0x3e => self.validate_memory_op(opcode),
            0x3f => {
                self.require_memory()?;
                self.require_zero_byte("memory.size memory index")?;
                self.push(ValType::I32)
            }
            0x40 => {
                self.require_memory()?;
                self.require_zero_byte("memory.grow memory index")?;
                self.pop_expect(ValType::I32)?;
                self.push(ValType::I32)
            }
            0x41 => {
                self.reader.i32()?;
                self.push(ValType::I32)
            }
            0x42 => {
                self.reader.i64_leb()?;
                self.push(ValType::I64)
            }
            0x43 => {
                self.reader.bytes(4)?;
                self.push(ValType::F32)
            }
            0x44 => {
                self.reader.bytes(8)?;
                self.push(ValType::F64)
            }
            0xd0 => {
                let ty = self.read_reference_type()?;
                self.push(ty)
            }
            0xd1 => {
                match self.pop_any()? {
                    StackType::Known(ValType::FuncRef | ValType::ExternRef) | StackType::Bot => {}
                    StackType::Known(_) => {
                        return Err("wasm: ref.is_null operand must be a reference".into())
                    }
                }
                self.push(ValType::I32)
            }
            0x45..=0xc4 => self.validate_numeric(opcode),
            0xfc => self.validate_fc(),
            other => Err(format!("wasm: unsupported opcode 0x{other:x}")),
        }
    }

    fn validate_br_table(&mut self) -> R<()> {
        let count = self.reader.u32()? as usize;
        if count > MAX_BR_TABLE_LABELS {
            return Err("wasm: br_table exceeds implementation label limit".into());
        }
        let mut common: Option<Vec<ValType>> = None;
        for _ in 0..count {
            let depth = self.reader.u32()? as usize;
            let label = self.label_types(depth)?;
            if common.as_ref().is_some_and(|types| *types != label) {
                return Err("wasm: br_table label type mismatch".into());
            }
            common.get_or_insert(label);
        }
        let default_depth = self.reader.u32()? as usize;
        let default = self.label_types(default_depth)?;
        if common.as_ref().is_some_and(|types| *types != default) {
            return Err("wasm: br_table default label type mismatch".into());
        }
        self.pop_expect(ValType::I32)?;
        self.pop_types(&default)?;
        self.mark_unreachable()
    }

    fn validate_select(&mut self) -> R<()> {
        self.pop_expect(ValType::I32)?;
        let second = self.pop_any()?;
        let first = self.pop_any()?;
        let selected = match (first, second) {
            (StackType::Known(a), StackType::Known(b)) if a == b && is_numeric(a) => a,
            (StackType::Known(a), StackType::Bot) | (StackType::Bot, StackType::Known(a))
                if is_numeric(a) =>
            {
                a
            }
            (StackType::Bot, StackType::Bot) => {
                self.push_stack(StackType::Bot)?;
                return Ok(());
            }
            _ => return Err("wasm: select operands must have the same numeric type".into()),
        };
        self.push(selected)
    }

    fn validate_typed_select(&mut self) -> R<()> {
        if self.reader.u32()? != 1 {
            return Err("wasm: typed select must have one result type".into());
        }
        let ty = self.read_value_type()?;
        self.pop_expect(ValType::I32)?;
        self.pop_expect(ty)?;
        self.pop_expect(ty)?;
        self.push(ty)
    }

    fn read_value_type(&mut self) -> R<ValType> {
        match self.reader.byte()? {
            0x7f => Ok(ValType::I32),
            0x7e => Ok(ValType::I64),
            0x7d => Ok(ValType::F32),
            0x7c => Ok(ValType::F64),
            0x70 => Ok(ValType::FuncRef),
            0x6f => Ok(ValType::ExternRef),
            byte => Err(format!("wasm: unknown value type 0x{byte:x}")),
        }
    }

    fn read_reference_type(&mut self) -> R<ValType> {
        let ty = self.read_value_type()?;
        if matches!(ty, ValType::FuncRef | ValType::ExternRef) {
            Ok(ty)
        } else {
            Err("wasm: invalid reference type".into())
        }
    }

    fn validate_local(&mut self, opcode: u8) -> R<()> {
        let index = self.reader.u32()? as usize;
        let ty = *self
            .locals
            .get(index)
            .ok_or("wasm: local index out of range")?;
        match opcode {
            0x20 => self.push(ty),
            0x21 => self.pop_expect(ty).map(|_| ()),
            0x22 => {
                self.pop_expect(ty)?;
                self.push(ty)
            }
            _ => unreachable!(),
        }
    }

    fn validate_global(&mut self, opcode: u8) -> R<()> {
        let index = self.reader.u32()? as usize;
        let global = *self
            .context
            .globals
            .get(index)
            .ok_or("wasm: global index out of range")?;
        if opcode == 0x23 {
            self.push(global.val)
        } else {
            if !global.mutable {
                return Err("wasm: global.set targets immutable global".into());
            }
            self.pop_expect(global.val).map(|_| ())
        }
    }

    fn validate_memory_op(&mut self, opcode: u8) -> R<()> {
        self.require_memory()?;
        let (value, natural_alignment, store) = match opcode {
            0x28 => (ValType::I32, 2, false),
            0x29 => (ValType::I64, 3, false),
            0x2a => (ValType::F32, 2, false),
            0x2b => (ValType::F64, 3, false),
            0x2c | 0x2d => (ValType::I32, 0, false),
            0x2e | 0x2f => (ValType::I32, 1, false),
            0x30 | 0x31 => (ValType::I64, 0, false),
            0x32 | 0x33 => (ValType::I64, 1, false),
            0x34 | 0x35 => (ValType::I64, 2, false),
            0x36 => (ValType::I32, 2, true),
            0x37 => (ValType::I64, 3, true),
            0x38 => (ValType::F32, 2, true),
            0x39 => (ValType::F64, 3, true),
            0x3a => (ValType::I32, 0, true),
            0x3b => (ValType::I32, 1, true),
            0x3c => (ValType::I64, 0, true),
            0x3d => (ValType::I64, 1, true),
            0x3e => (ValType::I64, 2, true),
            _ => unreachable!(),
        };
        let alignment = self.reader.u32()?;
        self.reader.u32()?; // offset
        if alignment > natural_alignment {
            return Err("wasm: memory alignment exceeds natural alignment".into());
        }
        if store {
            self.pop_expect(value)?;
            self.pop_expect(ValType::I32).map(|_| ())
        } else {
            self.pop_expect(ValType::I32)?;
            self.push(value)
        }
    }

    fn validate_numeric(&mut self, opcode: u8) -> R<()> {
        match opcode {
            0x45 => self.unary(ValType::I32, ValType::I32),
            0x46..=0x4f => self.binary(ValType::I32, ValType::I32),
            0x50 => self.unary(ValType::I64, ValType::I32),
            0x51..=0x5a => self.binary(ValType::I64, ValType::I32),
            0x5b..=0x60 => self.binary(ValType::F32, ValType::I32),
            0x61..=0x66 => self.binary(ValType::F64, ValType::I32),
            0x67..=0x69 => self.unary(ValType::I32, ValType::I32),
            0x6a..=0x78 => self.binary(ValType::I32, ValType::I32),
            0x79..=0x7b => self.unary(ValType::I64, ValType::I64),
            0x7c..=0x8a => self.binary(ValType::I64, ValType::I64),
            0x8b..=0x91 => self.unary(ValType::F32, ValType::F32),
            0x92..=0x98 => self.binary(ValType::F32, ValType::F32),
            0x99..=0x9f => self.unary(ValType::F64, ValType::F64),
            0xa0..=0xa6 => self.binary(ValType::F64, ValType::F64),
            0xa7 => self.unary(ValType::I64, ValType::I32),
            0xa8..=0xa9 => self.unary(ValType::F32, ValType::I32),
            0xaa..=0xab => self.unary(ValType::F64, ValType::I32),
            0xac..=0xad => self.unary(ValType::I32, ValType::I64),
            0xae..=0xaf => self.unary(ValType::F32, ValType::I64),
            0xb0..=0xb1 => self.unary(ValType::F64, ValType::I64),
            0xb2..=0xb3 => self.unary(ValType::I32, ValType::F32),
            0xb4..=0xb5 => self.unary(ValType::I64, ValType::F32),
            0xb6 => self.unary(ValType::F64, ValType::F32),
            0xb7..=0xb8 => self.unary(ValType::I32, ValType::F64),
            0xb9..=0xba => self.unary(ValType::I64, ValType::F64),
            0xbb => self.unary(ValType::F32, ValType::F64),
            0xbc => self.unary(ValType::F32, ValType::I32),
            0xbd => self.unary(ValType::F64, ValType::I64),
            0xbe => self.unary(ValType::I32, ValType::F32),
            0xbf => self.unary(ValType::I64, ValType::F64),
            0xc0..=0xc1 => self.unary(ValType::I32, ValType::I32),
            0xc2..=0xc4 => self.unary(ValType::I64, ValType::I64),
            _ => Err(format!("wasm: unsupported numeric opcode 0x{opcode:x}")),
        }
    }

    fn validate_fc(&mut self) -> R<()> {
        match self.reader.u32()? {
            0 | 1 => self.unary(ValType::F32, ValType::I32),
            2 | 3 => self.unary(ValType::F64, ValType::I32),
            4 | 5 => self.unary(ValType::F32, ValType::I64),
            6 | 7 => self.unary(ValType::F64, ValType::I64),
            8 => {
                self.require_memory()?;
                let data_index = self.reader.u32()? as usize;
                let count = self
                    .context
                    .data_count
                    .ok_or("wasm: memory.init requires a data count section")?;
                if data_index >= count {
                    return Err("wasm: data segment index out of range".into());
                }
                self.require_zero_byte("memory.init memory index")?;
                self.pop_expect(ValType::I32)?;
                self.pop_expect(ValType::I32)?;
                self.pop_expect(ValType::I32).map(|_| ())
            }
            9 => {
                let data_index = self.reader.u32()? as usize;
                let count = self
                    .context
                    .data_count
                    .ok_or("wasm: data.drop requires a data count section")?;
                if data_index >= count {
                    Err("wasm: data segment index out of range".into())
                } else {
                    Ok(())
                }
            }
            10 => {
                self.require_memory()?;
                self.require_zero_byte("memory.copy destination memory index")?;
                self.require_zero_byte("memory.copy source memory index")?;
                self.pop_expect(ValType::I32)?;
                self.pop_expect(ValType::I32)?;
                self.pop_expect(ValType::I32).map(|_| ())
            }
            11 => {
                self.require_memory()?;
                self.require_zero_byte("memory.fill memory index")?;
                self.pop_expect(ValType::I32)?;
                self.pop_expect(ValType::I32)?;
                self.pop_expect(ValType::I32).map(|_| ())
            }
            other => Err(format!("wasm: unsupported 0xfc opcode {other}")),
        }
    }

    fn read_block_type(&mut self) -> R<(Vec<ValType>, Vec<ValType>)> {
        let start = self.reader.pos;
        let first = self.reader.byte()?;
        match first {
            0x40 => Ok((Vec::new(), Vec::new())),
            0x7f => Ok((Vec::new(), vec![ValType::I32])),
            0x7e => Ok((Vec::new(), vec![ValType::I64])),
            0x7d => Ok((Vec::new(), vec![ValType::F32])),
            0x7c => Ok((Vec::new(), vec![ValType::F64])),
            0x70 => Ok((Vec::new(), vec![ValType::FuncRef])),
            0x6f => Ok((Vec::new(), vec![ValType::ExternRef])),
            _ => {
                self.reader.pos = start;
                let index = self.reader.signed_leb(33)?;
                if index < 0 {
                    return Err("wasm: invalid block type".into());
                }
                let ty = self
                    .context
                    .types
                    .get(index as usize)
                    .ok_or("wasm: block type index out of range")?;
                Ok((ty.params.clone(), ty.results.clone()))
            }
        }
    }

    fn unary(&mut self, input: ValType, output: ValType) -> R<()> {
        self.pop_expect(input)?;
        self.push(output)
    }

    fn binary(&mut self, input: ValType, output: ValType) -> R<()> {
        self.pop_expect(input)?;
        self.pop_expect(input)?;
        self.push(output)
    }

    fn require_memory(&self) -> R<()> {
        if self.context.memories == 0 {
            Err("wasm: memory instruction requires a memory".into())
        } else {
            Ok(())
        }
    }

    fn require_zero_byte(&mut self, what: &str) -> R<()> {
        if self.reader.byte()? == 0 {
            Ok(())
        } else {
            Err(format!("wasm: {what} must be zero"))
        }
    }

    fn label_types(&self, depth: usize) -> R<Vec<ValType>> {
        let index = self
            .controls
            .len()
            .checked_sub(depth + 1)
            .ok_or("wasm: branch depth out of range")?;
        let frame = &self.controls[index];
        Ok(if frame.kind == ControlKind::Loop {
            frame.start_types.clone()
        } else {
            frame.end_types.clone()
        })
    }

    fn push_control(
        &mut self,
        kind: ControlKind,
        start_types: &[ValType],
        end_types: &[ValType],
    ) -> R<()> {
        if self.controls.len() >= MAX_CONTROL_DEPTH {
            return Err("wasm: control nesting exceeds implementation limit".into());
        }
        let frame = ControlFrame {
            kind,
            start_types: start_types.to_vec(),
            end_types: end_types.to_vec(),
            value_height: self.values.len(),
            unreachable: false,
        };
        self.controls.push(frame);
        self.push_types(start_types)
    }

    fn pop_control(&mut self) -> R<ControlFrame> {
        let frame = self
            .controls
            .last()
            .cloned()
            .ok_or("wasm: unexpected end/else")?;
        self.pop_types(&frame.end_types)?;
        if self.values.len() != frame.value_height {
            return Err("wasm: control frame leaves unexpected values".into());
        }
        self.controls.pop();
        Ok(frame)
    }

    fn mark_unreachable(&mut self) -> R<()> {
        let frame = self
            .controls
            .last_mut()
            .ok_or("wasm: unreachable outside control frame")?;
        self.values.truncate(frame.value_height);
        frame.unreachable = true;
        Ok(())
    }

    fn pop_any(&mut self) -> R<StackType> {
        let frame = self
            .controls
            .last()
            .ok_or("wasm: operand outside control frame")?;
        if self.values.len() == frame.value_height && frame.unreachable {
            return Ok(StackType::Bot);
        }
        if self.values.len() <= frame.value_height {
            return Err("wasm: operand stack underflow".into());
        }
        self.values
            .pop()
            .ok_or("wasm: operand stack underflow".into())
    }

    fn pop_expect(&mut self, expected: ValType) -> R<StackType> {
        let actual = self.pop_any()?;
        if matches!(actual, StackType::Known(actual) if actual != expected) {
            return Err(format!(
                "wasm: operand type mismatch (expected {expected:?}, got {actual:?})"
            ));
        }
        Ok(actual)
    }

    fn pop_types(&mut self, types: &[ValType]) -> R<()> {
        for &ty in types.iter().rev() {
            self.pop_expect(ty)?;
        }
        Ok(())
    }

    fn push(&mut self, ty: ValType) -> R<()> {
        self.push_stack(StackType::Known(ty))
    }

    fn push_stack(&mut self, ty: StackType) -> R<()> {
        if self.values.len() >= MAX_VALIDATION_STACK {
            return Err("wasm: operand stack exceeds implementation limit".into());
        }
        self.values.push(ty);
        Ok(())
    }

    fn push_types(&mut self, types: &[ValType]) -> R<()> {
        for &ty in types {
            self.push(ty)?;
        }
        Ok(())
    }
}

fn is_numeric(ty: ValType) -> bool {
    matches!(
        ty,
        ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64
    )
}
