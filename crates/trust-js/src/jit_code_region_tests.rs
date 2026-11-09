//! The code region (P3 stage 2 step 9): placement, the cursor's wrap, the fallback to a mapping of
//! its own, threads, and each object's protection.

use super::super::ExecutableBuffer;
use super::*;

/// Code that returns `value`: `movz w0, #value; ret` on ARM64, `mov eax, value; ret` on x86-64.
fn returning(value: u16) -> Vec<u8> {
    #[cfg(target_arch = "aarch64")]
    {
        [0x5280_0000 | (u32::from(value) << 5), 0xd65f_03c0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mut code = vec![0xb8];
        code.extend_from_slice(&u32::from(value).to_le_bytes());
        code.push(0xc3);
        code
    }
}

/// `returning(value)` followed by zeros, `len` bytes in all, `len` at least the function's.
fn returning_in(value: u16, len: usize) -> Vec<u8> {
    let mut code = returning(value);
    assert!(len >= code.len());
    code.resize(len, 0);
    code
}

/// Run the `returning` code at the buffer's start.
fn run(buffer: &ExecutableBuffer) -> u32 {
    // SAFETY: the buffer starts with a `returning` function, published executable.
    let entry: extern "C" fn() -> u32 = unsafe { std::mem::transmute(buffer.as_ptr()) };
    entry()
}

/// A region of `pages` pages of its own, kept for the process (its buffers hold it as
/// `&'static`).
fn small_region(pages: usize) -> &'static CodeRegion {
    Box::leak(Box::new(
        CodeRegion::reserve(pages * sys::page_size()).expect("a small code region"),
    ))
}

impl CodeRegion {
    fn end(&self) -> usize {
        self.base + self.page_count * self.page_size
    }

    fn contains(&self, address: *const u8) -> bool {
        (self.base..self.end()).contains(&(address as usize))
    }

    fn taken_pages(&self) -> usize {
        self.map()
            .taken
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }
}

fn page_map(page_count: usize) -> PageMap {
    PageMap {
        taken: vec![0; page_count.div_ceil(64)].into_boxed_slice(),
        cursor: 0,
    }
}

#[test]
fn the_cursor_moves_past_each_run_and_never_back() {
    let mut map = page_map(8);
    assert_eq!(map.take(1, 8), Some(0));
    map.release(0, 1);
    // The freed page waits behind the cursor.
    assert_eq!(map.take(1, 8), Some(1));
    assert_eq!(map.take(3, 8), Some(2));
    assert_eq!(map.cursor, 5);
    map.release(2, 3);
    assert_eq!(map.take(3, 8), Some(5));
    // The run ended at the region's end: the search wraps to the start, where page 0 is free
    // but page 1 is taken, so a run of two takes pages 2 and 3.
    assert_eq!(map.cursor, 0);
    assert_eq!(map.take(2, 8), Some(2));
    assert_eq!(map.take(1, 8), Some(4));
    assert_eq!(map.take(1, 8), Some(0));
    assert_eq!(map.take(1, 8), None, "every page taken");
}

#[test]
fn runs_cross_words_skip_taken_pages_and_never_cross_the_regions_end() {
    let mut map = page_map(200);
    assert_eq!(map.take(60, 200), Some(0));
    assert_eq!(map.take(10, 200), Some(60), "a run across a word's end");
    assert_eq!(map.take(130, 200), Some(70), "a run to the region's end");
    assert_eq!(map.cursor, 0);
    map.release(0, 60);
    map.release(65, 1);
    assert_eq!(map.take(61, 200), None, "no 61 free pages in a row");
    assert_eq!(map.take(60, 200), Some(0));
    assert_eq!(map.take(1, 200), Some(65));

    // Free pages at the end and at the start make no run across the end.
    let mut map = page_map(8);
    assert_eq!(map.take(2, 8), Some(0));
    assert_eq!(map.take(4, 8), Some(2));
    assert_eq!(map.take(2, 8), Some(6));
    map.release(0, 2);
    map.release(6, 2);
    assert_eq!(map.take(4, 8), None);
    assert_eq!(map.take(2, 8), Some(0));
}

#[test]
fn an_allocation_after_a_free_of_the_same_size_takes_other_pages() {
    // In the process's region, as the tiers allocate: the freed pages are not taken again
    // while the region has others, whatever other tests allocate meanwhile.
    let code = returning(7);
    let first = ExecutableBuffer::from_bytes(&code).unwrap();
    assert!(
        first.region.is_some(),
        "the process's region holds the code"
    );
    let freed = first.as_ptr();
    drop(first);
    let second = ExecutableBuffer::from_bytes(&code).unwrap();
    assert!(second.region.is_some());
    assert_ne!(second.as_ptr(), freed);
    assert_eq!(run(&second), 7);

    // In a region of its own: the next page.
    let region = small_region(4);
    let first = ExecutableBuffer::from_bytes_in(&code, Some(region)).unwrap();
    let freed = first.as_ptr() as usize;
    drop(first);
    let second = ExecutableBuffer::from_bytes_in(&code, Some(region)).unwrap();
    assert_eq!(second.as_ptr() as usize, freed + region.page_size);
    assert_eq!(region.taken_pages(), 1);
}

#[test]
fn the_cursor_wraps_and_reuses_freed_pages() {
    let region = small_region(4);
    let first = ExecutableBuffer::from_bytes_in(&returning(1), Some(region)).unwrap();
    let start = first.as_ptr() as usize;
    assert_eq!(start, region.base);
    drop(first);
    let rest: Vec<ExecutableBuffer> = (2..5)
        .map(|value| ExecutableBuffer::from_bytes_in(&returning(value), Some(region)).unwrap())
        .collect();
    for (page, buffer) in (1..).zip(&rest) {
        assert_eq!(buffer.as_ptr() as usize, start + page * region.page_size);
    }
    // Past the region's end the cursor wraps, and the first page, freed, holds new code.
    let again = ExecutableBuffer::from_bytes_in(&returning(9), Some(region)).unwrap();
    assert_eq!(again.as_ptr() as usize, start);
    assert_eq!(run(&again), 9);
    for (value, buffer) in (2..).zip(&rest) {
        assert_eq!(run(buffer), value);
    }
    assert_eq!(region.taken_pages(), 4);
}

#[test]
fn a_region_without_a_free_run_long_enough_falls_back_to_a_mapping_of_its_own() {
    let region = small_region(4);
    let page = region.page_size;
    let mut held: Vec<ExecutableBuffer> = (0..4)
        .map(|value| ExecutableBuffer::from_bytes_in(&returning(value), Some(region)).unwrap())
        .collect();
    // Pages 1 and 3 free: two pages, no run of two.
    drop(held.remove(3));
    drop(held.remove(1));
    assert_eq!(region.taken_pages(), 2);
    let own = ExecutableBuffer::from_bytes_in(&returning_in(5, page + 1), Some(region)).unwrap();
    assert!(own.region.is_none());
    assert!(!region.contains(own.as_ptr()));
    assert_eq!(run(&own), 5);
    // A one-page code still takes a free page of the region.
    let short = ExecutableBuffer::from_bytes_in(&returning(6), Some(region)).unwrap();
    assert!(short.region.is_some() && region.contains(short.as_ptr()));
    assert_eq!(run(&short), 6);
    drop(own);
    assert_eq!(region.taken_pages(), 3);
    // Without a region (one the system refused) every allocation maps its own.
    let unplaced = ExecutableBuffer::from_bytes_in(&returning(8), None).unwrap();
    assert!(unplaced.region.is_none());
    assert_eq!(run(&unplaced), 8);
    for (value, buffer) in [0, 2].into_iter().zip(&held) {
        assert_eq!(run(buffer), value);
    }
}

#[test]
fn allocations_from_several_threads_take_disjoint_pages() {
    // Eight threads keep up to four codes of one to three pages each live in 24 pages: the
    // cursor wraps many times, allocations fall back while the threads' codes fill the region,
    // and every live code's pages are its own (each runs its own value after every allocation
    // of its thread).
    let region = small_region(24);
    let page = region.page_size;
    let live = Mutex::new(Vec::<(usize, usize)>::new());
    let pages_of = |buffer: &ExecutableBuffer| {
        let start = buffer.as_ptr() as usize;
        (start, start + buffer.len().div_ceil(page) * page)
    };
    let forget = |buffer: &ExecutableBuffer| {
        if buffer.region.is_some() {
            let range = pages_of(buffer);
            let mut live = live.lock().unwrap();
            let at = live.iter().position(|&held| held == range).unwrap();
            live.swap_remove(at);
        }
    };
    let placed = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for thread in 0..8u16 {
            let (live, placed) = (&live, &placed);
            scope.spawn(move || {
                let mut held = std::collections::VecDeque::new();
                for round in 0..200u16 {
                    let value = thread * 1000 + round;
                    let len = returning(value).len() + usize::from(round % 3) * page;
                    let buffer =
                        ExecutableBuffer::from_bytes_in(&returning_in(value, len), Some(region))
                            .unwrap();
                    if buffer.region.is_some() {
                        let (start, end) = pages_of(&buffer);
                        assert!(region.contains(start as *const u8) && end <= region.end());
                        let mut live = live.lock().unwrap();
                        assert!(
                            live.iter().all(|&(s, e)| end <= s || e <= start),
                            "two live codes share a page"
                        );
                        live.push((start, end));
                        placed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    held.push_back((value, buffer));
                    for (value, buffer) in &held {
                        assert_eq!(run(buffer), u32::from(*value));
                    }
                    if held.len() == 4 {
                        let (_, oldest) = held.pop_front().unwrap();
                        forget(&oldest);
                        drop(oldest);
                    }
                }
                for (_, buffer) in held {
                    forget(&buffer);
                    drop(buffer);
                }
            });
        }
    });
    assert!(live.lock().unwrap().is_empty());
    assert_eq!(region.taken_pages(), 0, "every page freed");
    assert!(placed.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

/// A page's access, as the system reports it.
#[derive(Debug, PartialEq)]
enum Access {
    ReadExecute,
    ReadWrite,
    /// Reserved, with no access and no memory behind it.
    Reserved,
    Other(String),
}

#[cfg(target_os = "linux")]
fn mappings() -> Vec<(usize, usize, String)> {
    std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let (start, end) = fields.next()?.split_once('-')?;
            Some((
                usize::from_str_radix(start, 16).ok()?,
                usize::from_str_radix(end, 16).ok()?,
                fields.next()?.to_owned(),
            ))
        })
        .collect()
}

/// The access of the page at `address` (/proc/self/maps); `None` where nothing is mapped.
#[cfg(target_os = "linux")]
fn access(address: usize) -> Option<Access> {
    let (_, _, perms) = mappings()
        .into_iter()
        .find(|&(start, end, _)| (start..end).contains(&address))?;
    Some(match perms.as_str() {
        "r-xp" => Access::ReadExecute,
        "rw-p" => Access::ReadWrite,
        "---p" => Access::Reserved,
        _ => Access::Other(perms),
    })
}

/// Whether a page of `region` is writable and executable at once.
#[cfg(target_os = "linux")]
fn writable_and_executable(region: &CodeRegion) -> bool {
    mappings().iter().any(|(start, end, perms)| {
        *start < region.end() && region.base < *end && perms.contains('w') && perms.contains('x')
    })
}

/// MEMORY_BASIC_INFORMATION (64-bit).
#[cfg(target_os = "windows")]
#[repr(C)]
struct MemoryBasicInformation {
    base_address: *mut u8,
    _allocation_base: *mut u8,
    _allocation_protect: u32,
    _partition_id: u16,
    region_size: usize,
    state: u32,
    protect: u32,
    _kind: u32,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn VirtualQuery(address: *const u8, info: *mut MemoryBasicInformation, len: usize) -> usize;
}

#[cfg(target_os = "windows")]
fn query(address: usize) -> MemoryBasicInformation {
    // SAFETY: MEMORY_BASIC_INFORMATION is plain data, which VirtualQuery fills.
    unsafe {
        let mut info = std::mem::zeroed::<MemoryBasicInformation>();
        let written = VirtualQuery(
            address as *const u8,
            &mut info,
            std::mem::size_of::<MemoryBasicInformation>(),
        );
        assert_ne!(written, 0, "VirtualQuery({address:#x})");
        info
    }
}

/// The access of the page at `address` (VirtualQuery); `None` where nothing is reserved.
#[cfg(target_os = "windows")]
fn access(address: usize) -> Option<Access> {
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_FREE: u32 = 0x10000;
    const PAGE_READWRITE: u32 = 0x04;
    const PAGE_EXECUTE_READ: u32 = 0x20;
    let info = query(address);
    Some(match (info.state, info.protect) {
        (MEM_FREE, _) => return None,
        (MEM_COMMIT, PAGE_EXECUTE_READ) => Access::ReadExecute,
        (MEM_COMMIT, PAGE_READWRITE) => Access::ReadWrite,
        (MEM_RESERVE, _) => Access::Reserved,
        (state, protect) => Access::Other(format!("state {state:#x}, protect {protect:#x}")),
    })
}

/// Whether a page of `region` is writable and executable at once.
#[cfg(target_os = "windows")]
fn writable_and_executable(region: &CodeRegion) -> bool {
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;
    const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;
    let mut address = region.base;
    while address < region.end() {
        let info = query(address);
        if info.protect & (PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0 {
            return true;
        }
        address = info.base_address as usize + info.region_size;
    }
    false
}

#[test]
fn region_code_runs_and_each_objects_pages_carry_its_own_protection() {
    let region = process_region().expect("the process's code region");
    let page = region.page_size;
    let one = ExecutableBuffer::from_bytes(&returning(11)).unwrap();
    let two = ExecutableBuffer::from_bytes(&returning_in(12, 2 * page)).unwrap();
    assert_eq!(run(&one), 11);
    assert_eq!(run(&two), 12);
    for buffer in [&one, &two] {
        assert!(region.contains(buffer.as_ptr()));
        for offset in (0..buffer.len()).step_by(page) {
            let address = buffer.as_ptr() as usize + offset;
            assert_eq!(access(address), Some(Access::ReadExecute), "{address:#x}");
        }
    }
    // A freed object's pages keep their addresses, reserved with no access, beside live code.
    let freed = one.as_ptr() as usize;
    drop(one);
    assert_eq!(access(freed), Some(Access::Reserved));
    assert_eq!(run(&two), 12);
    assert_eq!(access(two.as_ptr() as usize), Some(Access::ReadExecute));
    // The region's pages the cursor has not reached are reserved too.
    assert_eq!(access(region.end() - page), Some(Access::Reserved));
    assert!(!writable_and_executable(region), "W^X");
}

#[test]
fn the_process_region_is_a_gib_or_twice_a_larger_code_budget() {
    assert_eq!(region_bytes(128 << 20), 1 << 30);
    assert_eq!(region_bytes(512 << 20), 1 << 30);
    assert_eq!(region_bytes(513 << 20), 1026 << 20);
    assert_eq!(region_bytes(1024 << 20), 2 << 30);
    let region = process_region().expect("the process's code region");
    assert_eq!(
        region.page_count * region.page_size,
        region_bytes(super::super::executable_code_budget().limit)
    );
    assert_eq!(region.page_size, sys::page_size());
}
