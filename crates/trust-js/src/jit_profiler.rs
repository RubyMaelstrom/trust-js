//! Optional generated-code attribution for GNU gprofng. This is metadata at
//! compile/drop time, not instrumentation of generated instructions or JS calls.
//! The ABI is the public GNU Binutils `collectorAPI.h` interface. No collector
//! is linked or loaded by TRust JS; both symbols must already exist in the process.
//! Non-GNU-Linux targets retain a no-op implementation without a libdl dependency.
//! Parameter contracts (including nullable alias/source/line table):
//! <https://docs.oracle.com/cd/E18659_01/mkdl_gen_files/pdf/821-1379.pdf>, pp. 48–49.

use crate::bytecode::Chunk;

pub(super) fn register(chunk: &Chunk, memory: *mut u8, length: usize) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    native::register(chunk, memory, length);
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let _ = (chunk, memory, length);
}

pub(super) fn unregister(memory: *mut u8) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    native::unregister(memory);
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let _ = memory;
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod native {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::sync::{Mutex, OnceLock};

    // `lntable` is null because this registration supplies a chunk symbol, not
    // invented source line numbers. Names explicitly say "locals": the current
    // Chunk representation does not retain a source function name.
    type Load = unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        *mut c_void,
        c_int,
        c_int,
        *mut c_void,
    );
    type Unload = unsafe extern "C" fn(*mut c_void);

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    struct Collector {
        load: Load,
        unload: Unload,
        // Keep names alive through unload, even if an external collector stores
        // their pointers. Entries are removed before the mapping can be reused;
        // the table is bounded by live JIT mappings, not compilation history.
        names: Mutex<HashMap<usize, CString>>,
    }

    fn collector() -> Option<&'static Collector> {
        static COLLECTOR: OnceLock<Option<Collector>> = OnceLock::new();
        COLLECTOR
            .get_or_init(|| {
                if std::env::var("TRUST_JS_JIT_GPROFNG").as_deref() != Ok("1") {
                    return None;
                }
                // On GNU/Linux RTLD_DEFAULT is null. Only resolve the already-loaded
                // collector; never dlopen a library or guess a local installation path.
                let load = unsafe { dlsym(std::ptr::null_mut(), c"collector_func_load".as_ptr()) };
                let unload =
                    unsafe { dlsym(std::ptr::null_mut(), c"collector_func_unload".as_ptr()) };
                if load.is_null() || unload.is_null() {
                    return None;
                }
                Some(Collector {
                    // SAFETY: these are the public C functions with the signatures
                    // declared in GNU Binutils collectorAPI.h, not callable JS data.
                    load: unsafe { std::mem::transmute::<*mut c_void, Load>(load) },
                    unload: unsafe { std::mem::transmute::<*mut c_void, Unload>(unload) },
                    names: Mutex::new(HashMap::new()),
                })
            })
            .as_ref()
    }

    fn label(chunk: &Chunk, memory: usize) -> CString {
        let names: Vec<String> = chunk
            .jit_slot_names()
            .iter()
            .take(4)
            .map(|name| {
                name.chars()
                    .take(128)
                    .map(|ch| {
                        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '$' {
                            ch
                        } else {
                            '_'
                        }
                    })
                    .collect()
            })
            .collect();
        // A mapping address disambiguates anonymous bodies with identical locals.
        CString::new(format!(
            "TrustJsJIT::locals({})@{memory:x}",
            names.join(",")
        ))
        .expect("escaped local identifiers have no interior NUL")
    }

    pub(super) fn register(chunk: &Chunk, memory: *mut u8, length: usize) {
        if collector().is_none() {
            return;
        }
        register_label(label(chunk, memory as usize), memory, length);
    }

    fn register_label(name: CString, memory: *mut u8, length: usize) {
        let Some(collector) = collector() else { return };
        let Ok(length) = c_int::try_from(length) else {
            return;
        };
        if memory.is_null() || length <= 0 {
            return;
        }
        // The code is already published RX, and this synchronous registration
        // finishes before the code can execute or its owning JitCode can drop.
        unsafe {
            (collector.load)(
                name.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                memory.cast(),
                length,
                0,
                std::ptr::null_mut(),
            )
        };
        collector
            .names
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(memory as usize, name);
    }

    pub(super) fn unregister(memory: *mut u8) {
        let Some(collector) = collector() else { return };
        let name = collector
            .names
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&(memory as usize));
        if let Some(_name) = name {
            // JitCode's Drop runs before ExecutableBuffer's Drop, while the
            // memory and label remain valid. No registry lock crosses FFI.
            unsafe { (collector.unload)(memory.cast()) };
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn labels_are_bounded_and_explicit_about_locals_not_function_identity() {
            let body = crate::parser::parse_script("function demo(value){return value;}", false)
                .ok()
                .expect("fixture parses");
            let crate::ast::Stmt::FuncDecl(function) = &body[0] else {
                panic!("function")
            };
            let chunk = crate::bytecode::compile(function).expect("fixture compiles");
            let label = label(&chunk, 0x1234);
            assert!(label
                .to_str()
                .unwrap()
                .starts_with("TrustJsJIT::locals(value"));
            assert!(label.to_str().unwrap().ends_with("@1234"));
            assert!(label.as_bytes().len() < 4096);
        }
    }
}
