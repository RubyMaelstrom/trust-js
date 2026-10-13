//! Physical-memory probe used to size engine-wide resource ceilings.
//!
//! This is a sizing heuristic only: it never allocates engine memory and failures simply return
//! `None`, leaving callers on a fixed fallback.

/// Memory available to this process: physical RAM, further limited by a cgroup v2/v1 memory
/// limit on Linux when one is configured.
pub(crate) fn physical_memory_bytes() -> Option<u64> {
    platform::physical_memory_bytes().filter(|&bytes| bytes > 0)
}

#[cfg(target_os = "linux")]
mod platform {
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
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod platform {
    pub(super) fn physical_memory_bytes() -> Option<u64> {
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
}
