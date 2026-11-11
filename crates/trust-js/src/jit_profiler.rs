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
//! not unlisted, as the format has no unload record (V8's map has the same limit).
//!
//! `TRUST_JS_PERF_JITDUMP=<directory>` writes perf's jitdump, `<directory>/jit-<pid>.dump`
//! (Linux tools/perf/Documentation/jitdump-specification.txt), as V8's `--perf-prof` does
//! (`LinuxPerfJitLogger`, src/diagnostics/perf-jit.cc): one JIT_CODE_LOAD record per piece of code
//! the perf map lists, timestamped with CLOCK_MONOTONIC, so that a sample at an address a freed
//! code's memory held before another code took it is attributed to the code that ran then, where
//! the perf map's last line at the address takes every sample. A record names its code as the
//! perf map does, and with the optimizing tier on, a second stage's template code adds its
//! class, ` (second stage)` or ` (fallback)` (`opt::template_class`).
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
    list_code(
        memory,
        length,
        crate::opt::template_class(chunk),
        || match &chunk.perf_map_label {
            Some(label) => label.to_string(),
            // Compiled while this thread's map was off: only a test turns a map on mid-process.
            None => chunk_label(None, chunk.jit_slot_names()).into(),
        },
    );
}

/// Register the optimizing tier's code for `chunk` (P3 stage 1, `opt::a64`): named as V8 names
/// Maglev's code, `JS:+<name> <url>:<line>:<column>` (`+` is V8's marker for Maglev,
/// `CodeKindToMarker`), the first stage's name with its marker replaced.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(super) fn register_optimized(chunk: &Chunk, memory: *mut u8, length: usize) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    native::register(chunk, memory, length);
    perf_map_code(memory, length, || match &chunk.perf_map_label {
        Some(label) => optimized_label(label),
        None => optimized_label(&chunk_label(None, chunk.jit_slot_names())),
    });
}

/// A chunk's perf map name for the optimizing tier's code: `JS:+` in place of the first stage's
/// `JS:^`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) fn optimized_label(label: &str) -> String {
    match label.strip_prefix("JS:^") {
        Some(rest) => format!("JS:+{rest}"),
        None => format!("JS:+{label}"),
    }
}

pub(super) fn unregister(memory: *mut u8) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    native::unregister(memory);
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let _ = memory;
}

/// Whether the code this thread generates is listed in a perf map (`TRUST_JS_PERF_MAP`) or a
/// jitdump (`TRUST_JS_PERF_JITDUMP`). Names are built only when it is, so the default path pays
/// this check per piece of code.
pub(crate) fn perf_map_enabled() -> bool {
    with_perf_map(|_| ()).is_some() || jitdump::process().is_some()
}

/// List `length` bytes of generated code at `memory` in the perf map and the jitdump, whichever
/// is on, under the name `name` builds.
pub(crate) fn perf_map_code(memory: *const u8, length: usize, name: impl FnOnce() -> String) {
    list_code(memory, length, None, name);
}

/// [`perf_map_code`], the jitdump's name adding `class`, the optimizing tier's class of a second
/// stage's template code, in parentheses.
fn list_code(memory: *const u8, length: usize, class: Option<&str>, name: impl FnOnce() -> String) {
    let Some(dump) = jitdump::process() else {
        with_perf_map(|map| map.write(&perf_map_line(memory as usize, length, &name())));
        return;
    };
    let name = name();
    with_perf_map(|map| map.write(&perf_map_line(memory as usize, length, &name)));
    match class {
        Some(class) => dump.load(memory, length, &format!("{name} ({class})")),
        None => dump.load(memory, length, &name),
    }
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
    /// A test's capture ended (`PerfMapCapture`): later lines, from a job that outlived the
    /// test, are dropped.
    #[cfg(test)]
    Closed,
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
    /// of the process's; on a thread that compiles a job, its main thread's (`TestPerfMap`).
    static TEST_PERF_MAP: std::cell::RefCell<Option<std::sync::Arc<PerfMap>>> =
        const { std::cell::RefCell::new(None) };
}

/// The perf map capture of a test's main thread, which a job carries to the thread that compiles
/// it, so that the stubs the compile makes are listed where the test reads them (P3 stage 2,
/// step 17; as `jit::TestAddresses` carries its counters).
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Default)]
pub(crate) struct TestPerfMap(Option<std::sync::Arc<PerfMap>>);

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
impl TestPerfMap {
    /// This thread's capture, if one is on.
    pub(crate) fn capture() -> Self {
        TestPerfMap(TEST_PERF_MAP.with(|map| map.borrow().clone()))
    }

    /// List this thread's code in the capture while the guard lives.
    pub(crate) fn install(self) -> TestPerfMapGuard {
        TestPerfMapGuard(TEST_PERF_MAP.with(|map| map.replace(self.0)))
    }
}

/// [`TestPerfMap::install`]'s guard, which restores the previous capture.
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) struct TestPerfMapGuard(Option<std::sync::Arc<PerfMap>>);

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
impl Drop for TestPerfMapGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        let _ = TEST_PERF_MAP.try_with(|map| *map.borrow_mut() = previous);
    }
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
    map: std::sync::Arc<PerfMap>,
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
        let map = std::sync::Arc::new(PerfMap::new(path.clone()));
        TEST_PERF_MAP.with(|current| *current.borrow_mut() = Some(map.clone()));
        PerfMapCapture { path, map }
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
        *self
            .map
            .file
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = PerfMapFile::Closed;
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

/// perf's jitdump (see the module's doc): the header, the executable mapping of the file's first
/// page that perf record logs and `perf inject --jit` finds the dump by, and one load record per
/// piece of code.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod jitdump {
    use std::io::Write as _;
    use std::os::unix::io::AsRawFd as _;
    use std::sync::{Mutex, OnceLock};

    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }

    extern "C" {
        fn clock_gettime(clock: i32, time: *mut Timespec) -> i32;
        fn gettid() -> i32;
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn sysconf(name: i32) -> i64;
    }

    const CLOCK_MONOTONIC: i32 = 1;
    const PROT_READ: i32 = 1;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE: i32 = 2;
    const SC_PAGESIZE: i32 = 30;
    /// "JiTD" (jitdump-specification.txt, struct jitheader).
    const MAGIC: u32 = 0x4A69_5444;
    const VERSION: u32 = 1;
    const HEADER_SIZE: u32 = 40;
    const JIT_CODE_LOAD: u32 = 0;
    /// A load record's fixed part: the prefix (id, size, timestamp), then pid, tid, vma, code
    /// address, code size and code index.
    const LOAD_SIZE: usize = 16 + 40;
    #[cfg(target_arch = "aarch64")]
    const ELF_MACH: u32 = 183; // EM_AARCH64
    #[cfg(target_arch = "x86_64")]
    const ELF_MACH: u32 = 62; // EM_X86_64
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    const ELF_MACH: u32 = 0;

    /// CLOCK_MONOTONIC in nanoseconds: `perf record -k mono` stamps its samples with it.
    fn now() -> u64 {
        let mut time = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: a valid out pointer; CLOCK_MONOTONIC cannot fail on Linux.
        unsafe { clock_gettime(CLOCK_MONOTONIC, &mut time) };
        (time.tv_sec as u64)
            .wrapping_mul(1_000_000_000)
            .wrapping_add(time.tv_nsec as u64)
    }

    pub(super) struct JitDump {
        /// The file and the next record's code index; one write per record under this lock.
        file: Mutex<(std::fs::File, u64)>,
    }

    impl JitDump {
        /// A new dump at `path`, its header written and its first page mapped executable.
        pub(super) fn create(path: &std::path::Path) -> std::io::Result<JitDump> {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            let mut header = Vec::with_capacity(HEADER_SIZE as usize);
            header.extend_from_slice(&MAGIC.to_ne_bytes());
            header.extend_from_slice(&VERSION.to_ne_bytes());
            header.extend_from_slice(&HEADER_SIZE.to_ne_bytes());
            header.extend_from_slice(&ELF_MACH.to_ne_bytes());
            header.extend_from_slice(&0u32.to_ne_bytes());
            header.extend_from_slice(&std::process::id().to_ne_bytes());
            header.extend_from_slice(&now().to_ne_bytes());
            header.extend_from_slice(&0u64.to_ne_bytes());
            file.write_all(&header)?;
            // The marker: perf record logs only executable mappings, and `perf inject --jit`
            // finds the dump by this one's file name (V8's OpenMarkerFile). It stays mapped.
            // SAFETY: a fresh private mapping of an open file, which nothing reads or writes.
            let page = unsafe { sysconf(SC_PAGESIZE) }.max(4096) as usize;
            let marker = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    page,
                    PROT_READ | PROT_EXEC,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if marker as isize == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(JitDump {
                file: Mutex::new((file, 0)),
            })
        }

        /// A JIT_CODE_LOAD record for the `length` bytes of code at `memory`, named `name` (its
        /// control characters escaped, as the perf map's are, so that no NUL ends it early).
        pub(super) fn load(&self, memory: *const u8, length: usize, name: &str) {
            let timestamp = now();
            let mut escaped = String::with_capacity(name.len());
            for ch in name.chars() {
                if ch.is_control() {
                    escaped.extend(ch.escape_default());
                } else {
                    escaped.push(ch);
                }
            }
            let total = LOAD_SIZE + escaped.len() + 1 + length;
            let Ok(total_u32) = u32::try_from(total) else {
                return;
            };
            // SAFETY: the code was just written and published readable (its registration
            // precedes any free of it).
            let code = unsafe { std::slice::from_raw_parts(memory, length) };
            // SAFETY: gettid cannot fail.
            let tid = unsafe { gettid() } as u32;
            let mut file = self
                .file
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let index = file.1;
            file.1 += 1;
            let mut record = Vec::with_capacity(total);
            record.extend_from_slice(&JIT_CODE_LOAD.to_ne_bytes());
            record.extend_from_slice(&total_u32.to_ne_bytes());
            record.extend_from_slice(&timestamp.to_ne_bytes());
            record.extend_from_slice(&std::process::id().to_ne_bytes());
            record.extend_from_slice(&tid.to_ne_bytes());
            record.extend_from_slice(&(memory as u64).to_ne_bytes());
            record.extend_from_slice(&(memory as u64).to_ne_bytes());
            record.extend_from_slice(&(length as u64).to_ne_bytes());
            record.extend_from_slice(&index.to_ne_bytes());
            record.extend_from_slice(escaped.as_bytes());
            record.push(0);
            record.extend_from_slice(code);
            // A diagnostic: a failed write loses the record, never the program.
            let _ = file.0.write_all(&record);
        }
    }

    /// The process's dump, when `TRUST_JS_PERF_JITDUMP` names a directory (not empty, not `0`).
    /// The switch is read once; a dump that cannot be made is reported once, and nothing is
    /// written.
    pub(super) fn process() -> Option<&'static JitDump> {
        static DUMP: OnceLock<Option<JitDump>> = OnceLock::new();
        DUMP.get_or_init(|| {
            let directory = std::env::var_os("TRUST_JS_PERF_JITDUMP")?;
            if directory.is_empty() || directory == "0" {
                return None;
            }
            let path =
                std::path::Path::new(&directory).join(format!("jit-{}.dump", std::process::id()));
            match JitDump::create(&path) {
                Ok(dump) => Some(dump),
                Err(error) => {
                    eprintln!(
                        "TRust JS: TRUST_JS_PERF_JITDUMP: cannot make {}: {error}",
                        path.display()
                    );
                    None
                }
            }
        })
        .as_ref()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn u32_at(bytes: &[u8], at: usize) -> u32 {
            u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap())
        }

        fn u64_at(bytes: &[u8], at: usize) -> u64 {
            u64::from_ne_bytes(bytes[at..at + 8].try_into().unwrap())
        }

        #[test]
        fn a_dump_holds_perfs_header_and_a_load_record_per_code() {
            let path = std::env::temp_dir().join(format!(
                "trust-js-jitdump-{}-{:?}.dump",
                std::process::id(),
                std::thread::current().id()
            ));
            let dump = JitDump::create(&path).expect("dump");
            let code = [0xd503_201fu32; 4]; // four NOPs
            let before = now();
            // A computed function name can hold a NUL, which would end the record's name early.
            dump.load(code.as_ptr().cast(), 16, "JS:^de\0mo (second stage)");
            dump.load(code.as_ptr().cast(), 8, "Stub:a\nb");
            let bytes = std::fs::read(&path).expect("read");
            let _ = std::fs::remove_file(&path);
            assert_eq!(u32_at(&bytes, 0), MAGIC);
            assert_eq!(u32_at(&bytes, 4), VERSION);
            assert_eq!(u32_at(&bytes, 8), HEADER_SIZE);
            assert_eq!(u32_at(&bytes, 12), ELF_MACH);
            assert_eq!(u32_at(&bytes, 20), std::process::id());
            let mut at = HEADER_SIZE as usize;
            let mut names = Vec::new();
            for (index, size) in [(0u64, 16usize), (1, 8)] {
                assert_eq!(u32_at(&bytes, at), JIT_CODE_LOAD);
                let total = u32_at(&bytes, at + 4) as usize;
                assert!(u64_at(&bytes, at + 8) >= before, "a monotonic timestamp");
                assert_eq!(u32_at(&bytes, at + 16), std::process::id());
                assert_eq!(u64_at(&bytes, at + 24), code.as_ptr() as u64);
                assert_eq!(u64_at(&bytes, at + 32), code.as_ptr() as u64);
                assert_eq!(u64_at(&bytes, at + 40), size as u64);
                assert_eq!(u64_at(&bytes, at + 48), index);
                let name_end = at
                    + LOAD_SIZE
                    + bytes[at + LOAD_SIZE..]
                        .iter()
                        .position(|&byte| byte == 0)
                        .expect("a NUL-terminated name");
                names.push(String::from_utf8(bytes[at + LOAD_SIZE..name_end].to_vec()).unwrap());
                assert_eq!(total, name_end + 1 + size - at);
                assert_eq!(
                    &bytes[name_end + 1..name_end + 1 + size],
                    &code
                        .iter()
                        .flat_map(|word| word.to_ne_bytes())
                        .collect::<Vec<_>>()[..size]
                );
                at += total;
            }
            assert_eq!(at, bytes.len());
            assert_eq!(names, ["JS:^de\\u{0}mo (second stage)", "Stub:a\\nb"]);
        }
    }
}

/// No jitdump where perf's interface is not GNU/Linux's.
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
mod jitdump {
    pub(super) struct JitDump;

    impl JitDump {
        pub(super) fn load(&self, _: *const u8, _: usize, _: &str) {}
    }

    pub(super) fn process() -> Option<&'static JitDump> {
        None
    }
}

#[cfg(test)]
#[path = "jit_perf_map_tests.rs"]
mod perf_map_tests;
