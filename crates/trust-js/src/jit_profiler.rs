//! Generated-code attribution for external profilers. This is metadata at
//! compile/drop time, not instrumentation of generated instructions or JS calls.
//!
//! `TRUST_JS_PERF_MAP` lists every piece of executable code the engine generates in perf's JIT
//! map, `/tmp/perf-<pid>.map`: compiled chunks of either backend (registered here from
//! `compile_profiled_with`), the process-wide shared stubs and the native RegExp tier's
//! programs. Each is one `START SIZE name` line, hexadecimal without `0x`, which `perf report`
//! reads for samples in anonymous executable memory (Linux
//! tools/perf/Documentation/jit-interface.txt). V8's `--perf-basic-prof` writes the same file,
//! one line-buffered line per code object (`PerfBasicLogger`, src/logging/log.cc, as of V8
//! f581b4e0, 2026-10-27), and a chunk is named as V8 names baseline code
//! (`CodeEventLogger::CodeCreateEvent`): `JS:^<name> <url>:<line>:<column>`, `^` being V8's
//! marker for Sparkplug code (`CodeKindToMarker`, src/objects/code-kind.cc). A shared stub is
//! `Stub:<kind>` and a RegExp program `RegExp:/<source>/<flags>` (V8 now writes
//! `RegExp.< src: '<source>' flags: '<flags>'`, a form pprof's demangler needs). Freed code is
//! not unlisted, as the format has no unload record (V8's map has the same limit; a jitdump for
//! `perf inject --jit` would lift it if address reuse confused a profile).
//!
//! `TRUST_JS_JIT_GPROFNG` registers live chunks with GNU gprofng. The ABI is the public GNU
//! Binutils `collectorAPI.h` interface. No collector is linked or loaded by TRust JS; both
//! symbols must already exist in the process. Non-GNU-Linux targets retain a no-op
//! implementation without a libdl dependency. Parameter contracts (including nullable
//! alias/source/line table):
//! <https://docs.oracle.com/cd/E18659_01/mkdl_gen_files/pdf/821-1379.pdf>, pp. 48–49.

use crate::bytecode::Chunk;
use std::rc::Rc;

pub(super) fn register(chunk: &Chunk, memory: *mut u8, length: usize) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    native::register(chunk, memory, length);
    perf_map_code(memory, length, || match &chunk.perf_map_label {
        Some(label) => label.to_string(),
        // Compiled while this thread's map was off: only a test turns a map on mid-process.
        None => chunk_label(None, chunk.jit_slot_names()).into(),
    });
}

pub(super) fn unregister(memory: *mut u8) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    native::unregister(memory);
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let _ = memory;
}

/// Whether the code this thread generates is listed in a perf map (`TRUST_JS_PERF_MAP`). Names
/// are built only when it is, so the default path pays this check per piece of code.
pub(crate) fn perf_map_enabled() -> bool {
    with_perf_map(|_| ()).is_some()
}

/// List `length` bytes of generated code at `memory` in the perf map, under the name `name`
/// builds, if the map is on.
pub(crate) fn perf_map_code(memory: *const u8, length: usize, name: impl FnOnce() -> String) {
    with_perf_map(|map| map.write(&perf_map_line(memory as usize, length, &name())));
}

/// The perf map name of a chunk compiled from `function` (none for a loop fragment), whose
/// locals are `slot_names`: V8's name for baseline code, `JS:^<name> <url>:<line>:<column>`,
/// positioned at the function's first token as its stack frames are. A function without a
/// name is `<anonymous>`. Code without a script (Script and module bodies, loop fragments,
/// self-hosted and snapshot code) names its first locals instead, as the gprofng label does:
/// `JS:^<name> locals(a,b)`.
pub(crate) fn chunk_label(
    function: Option<&crate::ast::Function>,
    slot_names: &[Rc<str>],
) -> Box<str> {
    use std::fmt::Write as _;
    let name = function
        .and_then(|function| function.name.as_deref())
        .filter(|name| !name.is_empty())
        .unwrap_or("<anonymous>");
    let mut label = format!("JS:^{} ", crate::jstr::to_text(name));
    match function.and_then(|function| Some((function.script.as_deref()?, function.start))) {
        Some((script, start)) => {
            let (line, column) = script.line_column(start);
            let url = crate::jstr::to_text(script.url());
            let _ = write!(label, "{url}:{line}:{column}");
        }
        None => {
            label.push_str("locals(");
            for (index, local) in slot_names.iter().take(4).enumerate() {
                if index > 0 {
                    label.push(',');
                }
                label.extend(crate::jstr::to_text(local).chars().take(128));
            }
            label.push(')');
        }
    }
    label.into_boxed_str()
}

/// One perf map line: start and size in lowercase hexadecimal without `0x`, as perf reads them,
/// then the name, which runs to the end of the line; its control characters, which could end
/// the line early, are escaped.
fn perf_map_line(memory: usize, length: usize, name: &str) -> String {
    let mut line = format!("{memory:x} {length:x} ");
    for ch in name.chars() {
        if ch.is_control() {
            line.extend(ch.escape_default());
        } else {
            line.push(ch);
        }
    }
    line.push('\n');
    line
}

/// One perf map file. It opens on its first line and is appended to, never truncated.
struct PerfMap {
    path: std::path::PathBuf,
    /// One write per line under this lock: Agents on other threads compile too.
    file: std::sync::Mutex<PerfMapFile>,
}

enum PerfMapFile {
    Unopened,
    Open(std::fs::File),
    /// The file could not be opened. That is reported once, and later lines are dropped.
    Failed,
}

impl PerfMap {
    fn new(path: std::path::PathBuf) -> Self {
        PerfMap {
            path,
            file: std::sync::Mutex::new(PerfMapFile::Unopened),
        }
    }

    fn write(&self, line: &str) {
        use std::io::Write as _;
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if matches!(*file, PerfMapFile::Unopened) {
            let opened = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path);
            *file = match opened {
                Ok(opened) => PerfMapFile::Open(opened),
                Err(error) => {
                    eprintln!(
                        "TRust JS: TRUST_JS_PERF_MAP: cannot open {}: {error}",
                        self.path.display()
                    );
                    PerfMapFile::Failed
                }
            };
        }
        if let PerfMapFile::Open(opened) = &mut *file {
            // A diagnostic: a failed write loses the line, never the program.
            let _ = opened.write_all(line.as_bytes());
        }
    }
}

/// The process's perf map, at perf's fixed path, when `TRUST_JS_PERF_MAP` is set to anything
/// but `0` or nothing. The switch is read once.
fn process_perf_map() -> Option<&'static PerfMap> {
    static MAP: std::sync::OnceLock<Option<PerfMap>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let value = std::env::var_os("TRUST_JS_PERF_MAP")?;
        if value.is_empty() || value == "0" {
            return None;
        }
        let path = format!("/tmp/perf-{}.map", std::process::id());
        Some(PerfMap::new(path.into()))
    })
    .as_ref()
}

#[cfg(test)]
thread_local! {
    /// A test's own map (see [`PerfMapCapture`]), which this thread's code is listed in instead
    /// of the process's.
    static TEST_PERF_MAP: std::cell::RefCell<Option<Rc<PerfMap>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` on the map this thread's code is listed in, if any.
fn with_perf_map<R>(f: impl FnOnce(&PerfMap) -> R) -> Option<R> {
    #[cfg(test)]
    if let Some(map) = TEST_PERF_MAP
        .try_with(|map| map.borrow().clone())
        .ok()
        .flatten()
    {
        return Some(f(&map));
    }
    process_perf_map().map(f)
}

/// A test's own perf map: while it is on, the code this thread generates is listed in a file of
/// its own, whatever `TRUST_JS_PERF_MAP` says, so tests neither depend on the process's
/// environment nor write to its map. The file is removed when the capture drops.
#[cfg(test)]
pub(crate) struct PerfMapCapture {
    path: std::path::PathBuf,
}

#[cfg(test)]
impl PerfMapCapture {
    pub(crate) fn start(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "trust-js-perf-map-{}-{serial}-{tag}.map",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let map = Rc::new(PerfMap::new(path.clone()));
        TEST_PERF_MAP.with(|current| *current.borrow_mut() = Some(map));
        PerfMapCapture { path }
    }

    /// Stop listing this thread's code; the lines written so far stay readable.
    pub(crate) fn stop(&self) {
        TEST_PERF_MAP.with(|current| *current.borrow_mut() = None);
    }

    /// The lines written so far as (start, size, name), each checked to parse as perf parses
    /// it (tools/perf/util/symbol.c, `dso__load_perf_map`).
    pub(crate) fn lines(&self) -> Vec<(usize, usize, String)> {
        let text = std::fs::read_to_string(&self.path).unwrap_or_default();
        assert!(
            text.is_empty() || text.ends_with('\n'),
            "every line is complete"
        );
        text.lines()
            .map(|line| {
                let parsed = line.split_once(' ').and_then(|(start, rest)| {
                    let (size, name) = rest.split_once(' ')?;
                    let hex = |field: &str| {
                        let digits = !field.is_empty()
                            && field
                                .bytes()
                                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
                        if digits {
                            usize::from_str_radix(field, 16).ok()
                        } else {
                            None
                        }
                    };
                    // perf skips a line whose name is shorter than three bytes.
                    (name.len() >= 3).then_some((hex(start)?, hex(size)?, name.to_string()))
                });
                parsed.unwrap_or_else(|| panic!("perf cannot parse map line {line:?}"))
            })
            .collect()
    }
}

#[cfg(test)]
impl Drop for PerfMapCapture {
    fn drop(&mut self) {
        self.stop();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod native {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::sync::{Mutex, OnceLock};

    // `lntable` is null because this registration supplies a chunk symbol, not
    // invented source line numbers. Names explicitly say "locals": a Chunk
    // retains its source function's name only while the perf map is on.
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

#[cfg(test)]
#[path = "jit_perf_map_tests.rs"]
mod perf_map_tests;
