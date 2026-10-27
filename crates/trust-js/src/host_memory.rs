//! Physical-memory probes used to size engine-wide resource ceilings and the heap's maximum.
//!
//! These are sizing heuristics only: they never allocate engine memory and failures simply return
//! `None`, leaving callers on a fixed fallback.

/// Memory available to this process: physical RAM, further limited by a cgroup v2/v1 memory
/// limit on Linux when one is configured.
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    platform::physical_memory_bytes().filter(|&bytes| bytes > 0)
}

/// The machine's physical memory as V8 reads it (`base::SysInfo::AmountOfPhysicalMemory`,
/// src/base/sys-info.cc), and as Chrome reads it to size V8's heap (Chromium's
/// `base::SysInfo::AmountOfTotalPhysicalMemory`): `sysconf(_SC_PHYS_PAGES)` pages of
/// `sysconf(_SC_PAGESIZE)` bytes on Linux, `hw.memsize` on macOS and `GlobalMemoryStatusEx`'s
/// total on Windows, with no cgroup's limit, unlike [`physical_memory_bytes`]. `None` where the
/// system does not say (V8 then reads 0).
pub(crate) fn amount_of_physical_memory() -> Option<u64> {
    platform::amount_of_physical_memory().filter(|&bytes| bytes > 0)
}

#[cfg(target_os = "linux")]
mod platform {
    unsafe extern "C" {
        fn sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
    }
    /// glibc's and musl's `_SC_PAGESIZE` and `_SC_PHYS_PAGES`.
    const SC_PAGESIZE: std::ffi::c_int = 30;
    const SC_PHYS_PAGES: std::ffi::c_int = 85;

    pub(super) fn amount_of_physical_memory() -> Option<u64> {
        // SAFETY: sysconf has no preconditions.
        let (pages, page_size) = unsafe { (sysconf(SC_PHYS_PAGES), sysconf(SC_PAGESIZE)) };
        let pages = u64::try_from(pages).ok()?;
        let page_size = u64::try_from(page_size).ok()?;
        Some(pages.saturating_mul(page_size))
    }

    pub(super) fn physical_memory_bytes() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let total_kib = meminfo
            .lines()
            .find(|line| line.starts_with("MemTotal:"))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?;
        let total = total_kib.saturating_mul(1024);
        // "max" (unlimited) and unreadable files leave the physical total in effect.
        let cgroup = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()
            .or_else(|| std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes").ok())
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|&limit| limit > 0);
        Some(cgroup.map_or(total, |limit| limit.min(total)))
    }
}

#[cfg(target_os = "macos")]
mod platform {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> std::ffi::c_int;
    }

    pub(super) fn physical_memory_bytes() -> Option<u64> {
        let mut bytes: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `hw.memsize` is a 64-bit integer; the output buffer and its length describe
        // exactly that storage, and no new value is written.
        let status = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && len == std::mem::size_of::<u64>()).then_some(bytes)
    }

    /// V8's reading too (`sysctl` of `CTL_HW`, `HW_MEMSIZE`).
    pub(super) fn amount_of_physical_memory() -> Option<u64> {
        physical_memory_bytes()
    }
}

#[cfg(windows)]
mod platform {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub(super) fn physical_memory_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: the structure matches MEMORYSTATUSEX and its length field is initialized as
        // the API requires.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.total_phys)
    }

    /// V8's reading too (`ullTotalPhys`).
    pub(super) fn amount_of_physical_memory() -> Option<u64> {
        physical_memory_bytes()
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod platform {
    pub(super) fn physical_memory_bytes() -> Option<u64> {
        None
    }

    pub(super) fn amount_of_physical_memory() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn physical_memory_probe_is_plausible_when_available() {
        if let Some(bytes) = super::physical_memory_bytes() {
            // Any machine running the test suite has more than 64 MiB and less than 1 PiB.
            assert!(bytes > 64 << 20, "{bytes}");
            assert!(bytes < 1 << 50, "{bytes}");
        }
    }

    #[test]
    fn v8s_physical_memory_reading_is_plausible_and_covers_what_is_available() {
        let total = super::amount_of_physical_memory();
        if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            let bytes = total.expect("V8 reads physical memory here");
            assert!(bytes > 64 << 20, "{bytes}");
            assert!(bytes < 1 << 50, "{bytes}");
        }
        // The memory available to this process is at most all of it.
        if let (Some(total), Some(available)) = (total, super::physical_memory_bytes()) {
            assert!(available <= total, "{available} of {total}");
        }
    }
}
