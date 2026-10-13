//! WebAssembly: a from-scratch, std-only engine — binary decoder (`parse`), a bytecode interpreter
//! (`exec`), and the JS `WebAssembly.*` API assembled over native ops in `lib.rs`/`js/wasm.js`.
//! Supports the MVP instruction set plus common post-MVP ops (multi-value, sign-extension,
//! saturating conversions, bulk memory). Not supported: SIMD, threads/atomics, exceptions, GC.

pub mod exec;
pub mod parse;
mod validate;

pub use parse::{ExportKind, ImportKind, Module};

/// Describe a module's exports as `(name, kind)` for `WebAssembly.Module.exports()`.
pub fn export_descriptors(m: &Module) -> Vec<(String, &'static str)> {
    m.exports
        .iter()
        .map(|e| (e.name.clone(), kind_str(e.kind)))
        .collect()
}

/// Describe a module's imports as `(module, name, kind)` for `WebAssembly.Module.imports()`.
pub fn import_descriptors(m: &Module) -> Vec<(String, String, &'static str)> {
    m.imports
        .iter()
        .map(|i| {
            let kind = match i.kind {
                ImportKind::Func(_) => "function",
                ImportKind::Table(_) => "table",
                ImportKind::Memory(_) => "memory",
                ImportKind::Global(_) => "global",
            };
            (i.module.clone(), i.name.clone(), kind)
        })
        .collect()
}

fn kind_str(kind: ExportKind) -> &'static str {
    match kind {
        ExportKind::Func => "function",
        ExportKind::Table => "table",
        ExportKind::Memory => "memory",
        ExportKind::Global => "global",
    }
}

pub use parse::{decode, validate};

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::exec::{Host, Imports, Store, Val};
    use super::*;

    struct NoHost;
    impl Host for NoHost {
        fn call_host(
            &mut self,
            _id: usize,
            _a: &[Val],
            _r: &[parse::ValType],
        ) -> Result<Vec<Val>, String> {
            Err("no imports".into())
        }
    }

    fn instance_of(bytes: &[u8]) -> (Store, usize) {
        let module = decode(bytes).expect("decode");
        let mut store = Store::default();
        let idx = store
            .instantiate(module, Imports::default())
            .expect("instantiate");
        (store, idx)
    }

    fn call(store: &mut Store, inst: usize, name: &str, args: Vec<Val>) -> Vec<Val> {
        let (_, addr) = store.export_addr(inst, name).expect("export");
        store.invoke(addr, args, &mut NoHost, 0).expect("invoke")
    }

    fn valtype_byte(ty: parse::ValType) -> u8 {
        match ty {
            parse::ValType::I32 => 0x7f,
            parse::ValType::I64 => 0x7e,
            parse::ValType::F32 => 0x7d,
            parse::ValType::F64 => 0x7c,
            parse::ValType::FuncRef => 0x70,
            parse::ValType::ExternRef => 0x6f,
        }
    }

    fn enc_u32(out: &mut Vec<u8>, mut value: u32) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    fn section(module: &mut Vec<u8>, id: u8, payload: &[u8]) {
        module.push(id);
        enc_u32(module, payload.len() as u32);
        module.extend_from_slice(payload);
    }

    fn single_function_module(
        params: &[parse::ValType],
        results: &[parse::ValType],
        local_groups: &[(u32, parse::ValType)],
        code_without_function_end: &[u8],
    ) -> Vec<u8> {
        let mut module = b"\0asm\x01\0\0\0".to_vec();

        let mut types = vec![1, 0x60];
        enc_u32(&mut types, params.len() as u32);
        types.extend(params.iter().copied().map(valtype_byte));
        enc_u32(&mut types, results.len() as u32);
        types.extend(results.iter().copied().map(valtype_byte));
        section(&mut module, 1, &types);
        section(&mut module, 3, &[1, 0]);
        section(&mut module, 7, &[1, 1, b'f', 0, 0]);

        let mut body = Vec::new();
        enc_u32(&mut body, local_groups.len() as u32);
        for &(count, ty) in local_groups {
            enc_u32(&mut body, count);
            body.push(valtype_byte(ty));
        }
        body.extend_from_slice(code_without_function_end);
        body.push(0x0b);
        let mut code = vec![1];
        enc_u32(&mut code, body.len() as u32);
        code.extend(body);
        section(&mut module, 10, &code);
        module
    }

    // (module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))
    const ADD: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // header
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // type
        0x03, 0x02, 0x01, 0x00, // func
        0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00, // export "add"
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b, // code
    ];

    // (module (func (export "fac") (param i64) (result i64)
    //   (if (result i64) (i64.eqz (local.get 0))
    //     (then (i64.const 1))
    //     (else (i64.mul (local.get 0) (call 0 (i64.sub (local.get 0) (i64.const 1))))))))
    const FAC: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, //
        0x01, 0x06, 0x01, 0x60, 0x01, 0x7e, 0x01, 0x7e, // type: (i64)->i64
        0x03, 0x02, 0x01, 0x00, //
        0x07, 0x07, 0x01, 0x03, b'f', b'a', b'c', 0x00, 0x00, //
        0x0a, 0x17, 0x01, 0x15, 0x00, // code section, body size 0x15, 0 locals
        0x20, 0x00, 0x50, // local.get 0; i64.eqz
        0x04, 0x7e, // if (result i64)
        0x42, 0x01, // i64.const 1
        0x05, // else
        0x20, 0x00, // local.get 0
        0x20, 0x00, 0x42, 0x01, 0x7d, // local.get 0; i64.const 1; i64.sub
        0x10, 0x00, // call 0
        0x7e, // i64.mul
        0x0b, // end if
        0x0b, // end func
    ];

    #[test]
    fn runs_add() {
        let (mut store, inst) = instance_of(ADD);
        let r = call(&mut store, inst, "add", vec![Val::I32(2), Val::I32(3)]);
        assert_eq!(r[0].i32(), 5);
        let r = call(&mut store, inst, "add", vec![Val::I32(-10), Val::I32(4)]);
        assert_eq!(r[0].i32(), -6);
    }

    #[test]
    fn runs_recursion_and_control_flow() {
        let (mut store, inst) = instance_of(FAC);
        let r = call(&mut store, inst, "fac", vec![Val::I64(5)]);
        assert_eq!(r[0].i64(), 120);
        let r = call(&mut store, inst, "fac", vec![Val::I64(10)]);
        assert_eq!(r[0].i64(), 3628800);
    }

    #[test]
    fn two_instances_share_one_store() {
        // The same module instantiated twice in one store yields independent instances with
        // distinct function addresses, both callable.
        let module = decode(ADD).expect("decode");
        let mut store = Store::default();
        let a = store
            .instantiate(Rc::clone(&module), Imports::default())
            .unwrap();
        let b = store.instantiate(module, Imports::default()).unwrap();
        assert_ne!(a, b);
        let (_, addr_a) = store.export_addr(a, "add").unwrap();
        let (_, addr_b) = store.export_addr(b, "add").unwrap();
        assert_ne!(addr_a, addr_b);
        assert_eq!(
            store
                .invoke(addr_a, vec![Val::I32(1), Val::I32(2)], &mut NoHost, 0)
                .unwrap()[0]
                .i32(),
            3
        );
        assert_eq!(
            store
                .invoke(addr_b, vec![Val::I32(40), Val::I32(2)], &mut NoHost, 0)
                .unwrap()[0]
                .i32(),
            42
        );
    }

    #[test]
    fn validate_accepts_and_rejects() {
        assert!(validate(ADD));
        assert!(!validate(b"not wasm"));
        assert!(!validate(&[0x00, 0x61, 0x73, 0x6d, 0x02, 0, 0, 0])); // bad version
    }

    #[test]
    fn validation_enforces_operand_types_and_stack_polymorphism() {
        use parse::ValType::{I32, I64};

        // `unreachable` makes the current frame stack-polymorphic, so i32.add may consume Bots and
        // synthesize the required result (Core validation algorithm, `unreachable`).
        assert!(validate(&single_function_module(
            &[],
            &[I32],
            &[],
            &[0x00, 0x6a]
        )));

        // A concrete i64 above the polymorphic base still has to satisfy i32.add's operand type.
        assert!(!validate(&single_function_module(
            &[],
            &[I32],
            &[],
            &[0x00, 0x42, 0x00, 0x6a],
        )));
        assert!(!validate(&single_function_module(&[], &[], &[], &[0x1a]))); // drop underflow
        assert!(!validate(&single_function_module(
            &[],
            &[I32],
            &[],
            &[0x42, 0x00, 0x45], // i64.const; i32.eqz
        )));
        assert!(!validate(&single_function_module(
            &[],
            &[I64],
            &[],
            &[0x41, 0x00],
        )));
    }

    #[test]
    fn validation_enforces_control_labels_and_if_result_paths() {
        use parse::ValType::I32;

        assert!(!validate(&single_function_module(
            &[],
            &[],
            &[],
            &[0x0c, 0x01], // only the implicit function label (depth 0) exists
        )));
        assert!(!validate(&single_function_module(
            &[],
            &[I32],
            &[],
            &[
                0x41, 0x01, // condition
                0x04, 0x7f, // if (result i32)
                0x41, 0x07, // then result; no else result
                0x0b,
            ],
        )));
    }

    #[test]
    fn top_level_branch_targets_the_implicit_function_label() {
        use parse::ValType::I32;

        let module = single_function_module(
            &[],
            &[I32],
            &[],
            &[
                0x41, 0x07, // branch result
                0x0c, 0x00, // br 0 => implicit function label
                0x41, 0x09, // unreachable alternative
            ],
        );
        let (mut store, instance) = instance_of(&module);
        assert_eq!(call(&mut store, instance, "f", vec![])[0].i32(), 7);
    }

    #[test]
    fn decoding_rejects_oversized_counts_and_malformed_leb_without_allocating() {
        use parse::ValType::I32;

        let too_many_locals = single_function_module(&[], &[], &[(100_001, I32)], &[]);
        assert!(!validate(&too_many_locals));

        let mut malformed = b"\0asm\x01\0\0\0".to_vec();
        malformed.extend_from_slice(&[1, 0x80, 0x80, 0x80, 0x80, 0x10]);
        assert!(!validate(&malformed));

        // memory min 2, max 1
        let mut bad_limits = b"\0asm\x01\0\0\0".to_vec();
        section(&mut bad_limits, 5, &[1, 1, 2, 1]);
        assert!(!validate(&bad_limits));
    }

    #[test]
    fn data_count_uses_its_normative_pre_code_section_order() {
        let mut module = b"\0asm\x01\0\0\0".to_vec();
        section(&mut module, 1, &[1, 0x60, 0, 0]);
        section(&mut module, 3, &[1, 0]);
        section(&mut module, 12, &[0]);
        section(&mut module, 10, &[1, 2, 0, 0x0b]);
        section(&mut module, 11, &[0]);
        assert!(validate(&module));

        let mut mismatch = b"\0asm\x01\0\0\0".to_vec();
        section(&mut mismatch, 1, &[1, 0x60, 0, 0]);
        section(&mut mismatch, 3, &[1, 0]);
        section(&mut mismatch, 12, &[0]);
        section(&mut mismatch, 10, &[1, 2, 0, 0x0b]);
        // One valid passive, empty data segment while DataCount declares zero.
        section(&mut mismatch, 11, &[1, 1, 0]);
        assert!(!validate(&mismatch));
    }

    #[test]
    fn store_allocation_limits_fail_without_panicking() {
        use exec::{MAX_STORE_MEMORY_BYTES, MAX_STORE_TABLE_ELEMENTS, PAGE_SIZE};

        let mut store = Store::default();
        assert!(store
            .alloc_memory(MAX_STORE_MEMORY_BYTES / PAGE_SIZE + 1, None)
            .is_err());
        assert!(store
            .alloc_table(MAX_STORE_TABLE_ELEMENTS + 1, None)
            .is_err());
        assert!(store.alloc_memory(1, Some(0)).is_err());
        assert!(store.alloc_table(1, Some(0)).is_err());
    }

    #[test]
    fn instantiation_matches_import_types_and_preserves_store_changes_before_a_trap() {
        use parse::{DataSegment, GlobalType, Import, Limits, Module, ValType};

        let mut store = Store::default();
        let memory = store.alloc_memory(1, Some(10)).unwrap();
        let global = store.alloc_global(Val::I32(0), false, ValType::I32);

        let mut bad_memory_import = Module::default();
        bad_memory_import.imports.push(Import {
            module: "m".into(),
            name: "memory".into(),
            kind: ImportKind::Memory(Limits {
                min: 2,
                max: Some(5),
            }),
        });
        bad_memory_import.imported_mem_count = 1;
        assert!(store
            .instantiate(
                Rc::new(bad_memory_import),
                Imports {
                    mem_addr: Some(memory),
                    ..Imports::default()
                }
            )
            .is_err());

        let mut bad_global_import = Module::default();
        bad_global_import.imports.push(Import {
            module: "m".into(),
            name: "global".into(),
            kind: ImportKind::Global(GlobalType {
                val: ValType::F64,
                mutable: false,
            }),
        });
        bad_global_import.imported_global_count = 1;
        assert!(store
            .instantiate(
                Rc::new(bad_global_import),
                Imports {
                    global_addrs: vec![global],
                    ..Imports::default()
                }
            )
            .is_err());

        let memories_before = store.memories.len();
        let instances_before = store.instances.len();
        let mut out_of_bounds_data = Module::default();
        out_of_bounds_data.memories.push(Limits {
            min: 1,
            max: Some(1),
        });
        out_of_bounds_data.data.push(DataSegment {
            active: Some((0, vec![0x41, 0x80, 0x80, 0x04, 0x0b])), // i32.const 65536
            bytes: vec![1],
        });
        assert!(store
            .instantiate(Rc::new(out_of_bounds_data), Imports::default())
            .is_err());
        // WebAssembly Core 2.0 §4.5.10: allocation precedes segment initialization and a trap does
        // not roll the store back. This is observable when earlier segments mutate imports.
        assert_eq!(store.memories.len(), memories_before + 1);
        assert_eq!(store.instances.len(), instances_before + 1);
    }
}
