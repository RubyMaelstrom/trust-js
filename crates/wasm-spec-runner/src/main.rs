//! Runner for the official WebAssembly core `*.wast` suite and a deterministic malformed-binary
//! corpus. It intentionally drives TRust JS's own decoder, validator, linker, and interpreter rather
//! than comparing against another engine.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use trust_js_web::wasm::exec::{Host, Imports, Store, Val};
use trust_js_web::wasm::parse::{FuncType, ImportKind, Module, ValType};
use trust_js_web::wasm::{self, ExportKind};
use wast::core::{NanPattern, WastArgCore, WastRetCore};
use wast::parser::{self, ParseBuffer};
use wast::{QuoteWat, Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet, Wat};

#[derive(Default)]
struct SpecHost;

impl Host for SpecHost {
    fn call_host(
        &mut self,
        _id: usize,
        _args: &[Val],
        results: &[ValType],
    ) -> Result<Vec<Val>, String> {
        Ok(results.iter().copied().map(Val::default_for).collect())
    }
}

struct Spectest {
    memory: usize,
    table: usize,
    globals: HashMap<&'static str, usize>,
}

struct SpecRunner {
    store: Store,
    host: SpecHost,
    spectest: Spectest,
    last: Option<usize>,
    named: HashMap<String, usize>,
    registered: HashMap<String, usize>,
    definitions: HashMap<String, Rc<Module>>,
    last_definition: Option<Rc<Module>>,
}

impl SpecRunner {
    fn new() -> Result<Self, String> {
        let mut store = Store::default();
        let memory = store.alloc_memory(1, Some(2))?;
        let table = store.alloc_table(10, Some(20))?;
        let mut globals = HashMap::new();
        globals.insert(
            "global_i32",
            store.alloc_global(Val::I32(666), false, ValType::I32),
        );
        globals.insert(
            "global_i64",
            store.alloc_global(Val::I64(666), false, ValType::I64),
        );
        globals.insert(
            "global_f32",
            store.alloc_global(Val::F32(666.6), false, ValType::F32),
        );
        globals.insert(
            "global_f64",
            store.alloc_global(Val::F64(666.6), false, ValType::F64),
        );
        Ok(Self {
            store,
            host: SpecHost,
            spectest: Spectest {
                memory,
                table,
                globals,
            },
            last: None,
            named: HashMap::new(),
            registered: HashMap::new(),
            definitions: HashMap::new(),
            last_definition: None,
        })
    }

    fn spectest_function(name: &str) -> Option<FuncType> {
        let params = match name {
            "print" => Vec::new(),
            "print_i32" => vec![ValType::I32],
            "print_i64" => vec![ValType::I64],
            "print_f32" => vec![ValType::F32],
            "print_f64" => vec![ValType::F64],
            "print_i32_f32" => vec![ValType::I32, ValType::F32],
            "print_f64_f64" => vec![ValType::F64, ValType::F64],
            _ => return None,
        };
        Some(FuncType {
            params,
            results: Vec::new(),
        })
    }

    fn imports_for(&self, module: &Module) -> Result<Imports, String> {
        let mut imports = Imports::default();
        for import in &module.imports {
            if import.module == "spectest" {
                match &import.kind {
                    ImportKind::Func(_) => {
                        let ty = Self::spectest_function(&import.name)
                            .ok_or_else(|| format!("unknown spectest function {}", import.name))?;
                        imports.funcs.push((imports.funcs.len(), ty));
                        imports.func_addrs.push(None);
                    }
                    ImportKind::Memory(_) if import.name == "memory" => {
                        imports.mem_addr = Some(self.spectest.memory);
                    }
                    ImportKind::Table(_) if import.name == "table" => {
                        imports.table_addrs.push(self.spectest.table);
                    }
                    ImportKind::Global(_) => imports.global_addrs.push(
                        *self
                            .spectest
                            .globals
                            .get(import.name.as_str())
                            .ok_or_else(|| format!("unknown spectest global {}", import.name))?,
                    ),
                    _ => return Err(format!("unknown spectest import {}", import.name)),
                }
                continue;
            }

            let instance = *self
                .registered
                .get(&import.module)
                .ok_or_else(|| format!("unknown import module {}", import.module))?;
            let (kind, address) = self
                .store
                .export_addr(instance, &import.name)
                .ok_or_else(|| format!("unknown import {}.{}", import.module, import.name))?;
            match (&import.kind, kind) {
                (ImportKind::Memory(_), ExportKind::Memory) => imports.mem_addr = Some(address),
                (ImportKind::Table(_), ExportKind::Table) => imports.table_addrs.push(address),
                (ImportKind::Global(_), ExportKind::Global) => imports.global_addrs.push(address),
                (ImportKind::Func(_), ExportKind::Func) => {
                    let ty = self
                        .store
                        .funcs
                        .get(address)
                        .and_then(Option::as_ref)
                        .ok_or("dead function export")?
                        .ty();
                    imports.funcs.push((0, ty));
                    imports.func_addrs.push(Some(address));
                }
                _ => {
                    return Err(format!(
                        "import kind mismatch for {}.{}",
                        import.module, import.name
                    ))
                }
            }
        }
        Ok(imports)
    }

    fn instantiate(&mut self, module: Rc<Module>) -> Result<usize, String> {
        let imports = self.imports_for(&module)?;
        let start = module.start;
        let instance = self.store.instantiate(module, imports)?;
        if let Some(index) = start {
            let address = *self.store.instances[instance]
                .as_ref()
                .and_then(|instance| instance.func_addrs.get(index as usize))
                .ok_or("start function index out of range")?;
            self.store.invoke(address, Vec::new(), &mut self.host, 0)?;
        }
        Ok(instance)
    }

    fn define_quote(&mut self, module: &mut QuoteWat<'_>) -> Result<Rc<Module>, String> {
        let bytes = module.encode().map_err(|error| error.to_string())?;
        wasm::decode(&bytes)
    }

    fn define_wat(&mut self, module: &mut Wat<'_>) -> Result<Rc<Module>, String> {
        let bytes = module.encode().map_err(|error| error.to_string())?;
        wasm::decode(&bytes)
    }

    fn module_instance(&self, module: Option<wast::token::Id<'_>>) -> Result<usize, String> {
        match module {
            Some(id) => self
                .named
                .get(id.name())
                .copied()
                .ok_or_else(|| format!("unknown module ${}", id.name())),
            None => self
                .last
                .ok_or_else(|| "no previous module instance".into()),
        }
    }

    fn invoke(&mut self, invoke: &WastInvoke<'_>) -> Result<Vec<Val>, String> {
        let instance = self.module_instance(invoke.module)?;
        let (kind, address) = self
            .store
            .export_addr(instance, invoke.name)
            .ok_or_else(|| format!("unknown export {}", invoke.name))?;
        if kind != ExportKind::Func {
            return Err(format!("export {} is not a function", invoke.name));
        }
        let args = invoke
            .args
            .iter()
            .map(argument_value)
            .collect::<Result<Vec<_>, _>>()?;
        self.store.invoke(address, args, &mut self.host, 0)
    }

    fn execute(&mut self, exec: &mut WastExecute<'_>) -> Result<Vec<Val>, String> {
        match exec {
            WastExecute::Invoke(invoke) => self.invoke(invoke),
            WastExecute::Get { module, global, .. } => {
                let instance = self.module_instance(*module)?;
                let (kind, address) = self
                    .store
                    .export_addr(instance, global)
                    .ok_or_else(|| format!("unknown export {global}"))?;
                if kind != ExportKind::Global {
                    return Err(format!("export {global} is not a global"));
                }
                let value = self
                    .store
                    .globals
                    .get(address)
                    .and_then(Option::as_ref)
                    .ok_or("dead global export")?
                    .val;
                Ok(vec![value])
            }
            WastExecute::Wat(module) => {
                let module = self.define_wat(module)?;
                self.instantiate(module)?;
                Ok(Vec::new())
            }
        }
    }

    fn run_directive(&mut self, directive: &mut WastDirective<'_>) -> Result<(), String> {
        match directive {
            WastDirective::Module(module) => {
                let name = module.name().map(|id| id.name().to_string());
                let module = self.define_quote(module)?;
                let instance = self.instantiate(module)?;
                self.last = Some(instance);
                if let Some(name) = name {
                    self.named.insert(name, instance);
                }
                Ok(())
            }
            WastDirective::ModuleDefinition(module) => {
                let name = module.name().map(|id| id.name().to_string());
                let module = self.define_quote(module)?;
                if let Some(name) = name {
                    self.definitions.insert(name, Rc::clone(&module));
                }
                self.last_definition = Some(module);
                Ok(())
            }
            WastDirective::ModuleInstance {
                instance, module, ..
            } => {
                let definition = match module {
                    Some(id) => self
                        .definitions
                        .get(id.name())
                        .cloned()
                        .ok_or_else(|| format!("unknown module definition ${}", id.name()))?,
                    None => self
                        .last_definition
                        .clone()
                        .ok_or("no previous module definition")?,
                };
                let address = self.instantiate(definition)?;
                self.last = Some(address);
                if let Some(id) = instance {
                    self.named.insert(id.name().to_string(), address);
                }
                Ok(())
            }
            WastDirective::AssertMalformed { module, .. }
            | WastDirective::AssertInvalid { module, .. } => match module.encode() {
                Err(_) => Ok(()),
                Ok(bytes) if !wasm::validate(&bytes) => Ok(()),
                Ok(_) => Err("module unexpectedly decoded and validated".into()),
            },
            // Core custom sections never affect decoding or validation, so a module whose only
            // fault is in a custom section must still compile (WebAssembly/spec
            // test/harness/sync_index.js `assert_invalid_custom`). Text the encoder rejects
            // leaves nothing for the engine to check.
            WastDirective::AssertMalformedCustom { module, .. }
            | WastDirective::AssertInvalidCustom { module, .. } => match module.encode() {
                Err(_) => Ok(()),
                Ok(bytes) if wasm::validate(&bytes) => Ok(()),
                Ok(_) => Err("custom-section error made the module fail to compile".into()),
            },
            WastDirective::Register { name, module, .. } => {
                let instance = self.module_instance(*module)?;
                self.registered.insert((*name).to_string(), instance);
                Ok(())
            }
            WastDirective::Invoke(invoke) => self.invoke(invoke).map(|_| ()),
            WastDirective::AssertTrap { exec, .. } => match self.execute(exec) {
                Err(_) => Ok(()),
                Ok(_) => Err("execution unexpectedly completed without a trap".into()),
            },
            WastDirective::AssertReturn { exec, results, .. } => {
                let actual = self.execute(exec)?;
                compare_results(&actual, results)
            }
            WastDirective::AssertExhaustion { call, .. } => match self.invoke(call) {
                Err(_) => Ok(()),
                Ok(_) => Err("execution unexpectedly completed without exhaustion".into()),
            },
            WastDirective::AssertUnlinkable { module, .. } => {
                let module = self.define_wat(module)?;
                match self.instantiate(module) {
                    Err(_) => Ok(()),
                    Ok(_) => Err("module unexpectedly linked".into()),
                }
            }
            WastDirective::AssertException { .. }
            | WastDirective::AssertSuspension { .. }
            | WastDirective::Thread(_)
            | WastDirective::Wait { .. } => {
                Err("directive requires an unsupported proposal".into())
            }
        }
    }
}

fn argument_value(arg: &WastArg<'_>) -> Result<Val, String> {
    match arg {
        WastArg::Core(WastArgCore::I32(value)) => Ok(Val::I32(*value)),
        WastArg::Core(WastArgCore::I64(value)) => Ok(Val::I64(*value)),
        WastArg::Core(WastArgCore::F32(value)) => Ok(Val::F32(f32::from_bits(value.bits))),
        WastArg::Core(WastArgCore::F64(value)) => Ok(Val::F64(f64::from_bits(value.bits))),
        WastArg::Core(WastArgCore::RefNull(_)) => Ok(Val::Ref(None)),
        WastArg::Core(WastArgCore::RefExtern(value) | WastArgCore::RefHost(value)) => {
            Ok(Val::Ref(Some(*value)))
        }
        _ => Err("unsupported WebAssembly argument value".into()),
    }
}

fn compare_results(actual: &[Val], expected: &[WastRet<'_>]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "result arity mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if !result_matches(*actual, expected) {
            return Err(format!(
                "result {index} mismatch: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    Ok(())
}

fn result_matches(actual: Val, expected: &WastRet<'_>) -> bool {
    match expected {
        WastRet::Core(expected) => core_result_matches(actual, expected),
        _ => false,
    }
}

fn core_result_matches(actual: Val, expected: &WastRetCore<'_>) -> bool {
    match expected {
        WastRetCore::I32(expected) => {
            matches!(actual, Val::I32(actual) if actual == *expected)
        }
        WastRetCore::I64(expected) => {
            matches!(actual, Val::I64(actual) if actual == *expected)
        }
        WastRetCore::F32(pattern) => match actual {
            Val::F32(actual) => float32_matches(actual.to_bits(), pattern),
            _ => false,
        },
        WastRetCore::F64(pattern) => match actual {
            Val::F64(actual) => float64_matches(actual.to_bits(), pattern),
            _ => false,
        },
        WastRetCore::RefNull(_) => matches!(actual, Val::Ref(None)),
        WastRetCore::RefExtern(expected) => {
            matches!(actual, Val::Ref(Some(actual)) if expected.is_none_or(|expected| actual == expected))
        }
        WastRetCore::RefFunc(_) => matches!(actual, Val::Ref(Some(_))),
        WastRetCore::Either(options) => options
            .iter()
            .any(|option| core_result_matches(actual, option)),
        _ => false,
    }
}

fn float32_matches(actual: u32, expected: &NanPattern<wast::token::F32>) -> bool {
    match expected {
        NanPattern::Value(expected) => actual == expected.bits,
        NanPattern::CanonicalNan => actual & 0x7fff_ffff == 0x7fc0_0000,
        NanPattern::ArithmeticNan => actual & 0x7fc0_0000 == 0x7fc0_0000,
    }
}

fn float64_matches(actual: u64, expected: &NanPattern<wast::token::F64>) -> bool {
    match expected {
        NanPattern::Value(expected) => actual == expected.bits,
        NanPattern::CanonicalNan => actual & 0x7fff_ffff_ffff_ffff == 0x7ff8_0000_0000_0000,
        NanPattern::ArithmeticNan => actual & 0x7ff8_0000_0000_0000 == 0x7ff8_0000_0000_0000,
    }
}

fn run_file(path: &Path) -> Result<(usize, Vec<String>), String> {
    let source = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let buffer = ParseBuffer::new(&source).map_err(|error| error.to_string())?;
    let mut wast = parser::parse::<Wast<'_>>(&buffer).map_err(|error| error.to_string())?;
    let mut runner = SpecRunner::new()?;
    let mut passed = 0usize;
    let mut failures = Vec::new();
    for directive in &mut wast.directives {
        let span = directive.span();
        match runner.run_directive(directive) {
            Ok(()) => passed += 1,
            Err(error) => {
                let (line, column) = span.linecol_in(&source);
                failures.push(format!("{}:{}: {error}", line + 1, column + 1));
            }
        }
    }
    Ok((passed, failures))
}

fn focus_paths() -> Result<Vec<PathBuf>, String> {
    let manifest = fs::read_to_string("wasm-spec-focus.txt").map_err(|error| error.to_string())?;
    Ok(manifest
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect())
}

fn fuzz(iterations: usize) -> Result<(), String> {
    let seed = [
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f,
        0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01,
        0x6a, 0x0b,
    ];
    let mut state = 0x6c75_6d65_6e57_4153u64;
    for iteration in 0..iterations {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let mut bytes = if iteration % 3 == 0 {
            seed.to_vec()
        } else {
            vec![0; (state as usize) & 0x3ff]
        };
        for byte in &mut bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if iteration % 3 != 0 || state & 7 == 0 {
                *byte ^= state as u8;
            }
        }
        if iteration % 11 == 0 {
            bytes.truncate((state as usize) % (bytes.len() + 1));
        }
        let result = std::panic::catch_unwind(|| wasm::decode(&bytes));
        if result.is_err() {
            return Err(format!(
                "decoder panicked on deterministic case {iteration}"
            ));
        }
    }
    println!("PASS malformed-binary fuzz corpus ({iterations} cases)");
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("--fuzz") {
        let iterations = args
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(100_000);
        if let Err(error) = fuzz(iterations) {
            eprintln!("FAIL {error}");
            std::process::exit(1);
        }
        return;
    }

    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let paths = if paths.is_empty() {
        focus_paths().unwrap_or_else(|error| {
            eprintln!("FAIL cannot read wasm-spec-focus.txt: {error}");
            std::process::exit(2);
        })
    } else {
        paths
    };
    let mut passed = 0usize;
    let mut failed = 0usize;
    for path in &paths {
        match run_file(path) {
            Ok((file_passed, failures)) => {
                passed += file_passed;
                failed += failures.len();
                if failures.is_empty() {
                    println!("PASS {} ({file_passed} directives)", path.display());
                } else {
                    println!(
                        "FAIL {} ({file_passed} passed, {} failed)",
                        path.display(),
                        failures.len()
                    );
                    for failure in failures.iter().take(20) {
                        println!("  {failure}");
                    }
                    if failures.len() > 20 {
                        println!("  ... {} more", failures.len() - 20);
                    }
                }
            }
            Err(error) => {
                failed += 1;
                println!("FAIL {}: {error}", path.display());
            }
        }
    }
    println!(
        "WebAssembly core spec: {passed}/{} directives passed across {} files",
        passed + failed,
        paths.len()
    );
    if failed != 0 {
        std::process::exit(1);
    }
}
