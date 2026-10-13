//! DEFLATE (RFC 1951) with zlib (RFC 1950) and gzip (RFC 1952) framing — a from-scratch, std-only
//! codec shared by the web `CompressionStream`/`DecompressionStream` and `node:zlib`. No external
//! crates: inflate handles stored/fixed/dynamic Huffman blocks; deflate emits fixed-Huffman blocks
//! with greedy LZ77 matching (real compression, not stored-only). Checksums: Adler-32 (zlib) and
//! CRC-32 (gzip).

// ---- checksums --------------------------------------------------------------------------------

pub fn adler32(data: &[u8]) -> u32 {
    adler32_from(1, data)
}

fn adler32_from(seed: u32, data: &[u8]) -> u32 {
    let (mut a, mut b) = (seed & 0xffff, seed >> 16);
    // zlib's NMAX bound keeps both accumulators within u32 and reduces division from twice per
    // byte to twice per 5,552-byte batch.
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb88320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

const CRC32_TABLE: [u32; 256] = crc32_table();

pub fn crc32(data: &[u8]) -> u32 {
    crc32_from(0, data)
}

/// CRC-32 continued from a prior checksum `seed` (0 to start fresh) — the form `node:zlib.crc32`
/// exposes so callers can chain checksums across chunks.
pub fn crc32_from(seed: u32, data: &[u8]) -> u32 {
    let mut crc = seed ^ 0xffff_ffff;
    for &byte in data {
        crc = CRC32_TABLE[((crc ^ byte as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

// ---- bit reader (LSB-first, per DEFLATE) ------------------------------------------------------

#[derive(Clone, Copy)]
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }
    fn bit(&mut self) -> Result<u32, String> {
        if self.bit_cnt == 0 {
            if self.pos >= self.data.len() {
                return Err("inflate: unexpected end of input".into());
            }
            self.bit_buf = self.data[self.pos] as u32;
            self.pos += 1;
            self.bit_cnt = 8;
        }
        let b = self.bit_buf & 1;
        self.bit_buf >>= 1;
        self.bit_cnt -= 1;
        Ok(b)
    }
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        let mut v = 0;
        for i in 0..n {
            v |= self.bit()? << i;
        }
        Ok(v)
    }
    fn align_to_byte(&mut self) {
        self.bit_buf = 0;
        self.bit_cnt = 0;
    }
}

// ---- Huffman decoding -------------------------------------------------------------------------

/// Canonical Huffman decode table built from per-symbol code lengths.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut counts = [0u16; 16];
        for &len in lengths {
            counts[len as usize] += 1;
        }
        counts[0] = 0;
        let mut offsets = [0u16; 16];
        for i in 1..16 {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &len) in lengths.iter().enumerate() {
            if len != 0 {
                symbols[offsets[len as usize] as usize] = sym as u16;
                offsets[len as usize] += 1;
            }
        }
        Huffman { counts, symbols }
    }
    fn decode(&self, reader: &mut BitReader) -> Result<u16, String> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= reader.bit()? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("inflate: invalid Huffman code".into())
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn fixed_huffman() -> (Huffman, Huffman) {
    let mut lit_lengths = [0u8; 288];
    for (i, len) in lit_lengths.iter_mut().enumerate() {
        *len = if i < 144 {
            8
        } else if i < 256 {
            9
        } else if i < 280 {
            7
        } else {
            8
        };
    }
    let dist_lengths = [5u8; 30];
    (Huffman::new(&lit_lengths), Huffman::new(&dist_lengths))
}

fn inflate_block(
    reader: &mut BitReader,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
    limit: usize,
) -> Result<(), String> {
    loop {
        let sym = lit.decode(reader)?;
        match sym {
            0..=255 => {
                crate::checked_decompressed_len(out.len(), 1, limit, "inflate")?;
                out.push(sym as u8);
            }
            256 => return Ok(()), // end of block
            257..=285 => {
                let i = (sym - 257) as usize;
                let length =
                    LENGTH_BASE[i] as usize + reader.bits(LENGTH_EXTRA[i] as u32)? as usize;
                let dsym = dist.decode(reader)? as usize;
                if dsym >= 30 {
                    return Err("inflate: invalid distance symbol".into());
                }
                let distance =
                    DIST_BASE[dsym] as usize + reader.bits(DIST_EXTRA[dsym] as u32)? as usize;
                if distance > out.len() {
                    return Err("inflate: distance too far back".into());
                }
                crate::checked_decompressed_len(out.len(), length, limit, "inflate")?;
                let start = out.len() - distance;
                for k in 0..length {
                    out.push(out[start + k]);
                }
            }
            _ => return Err("inflate: invalid literal/length symbol".into()),
        }
    }
}

// ---- incremental inflater --------------------------------------------------------------------

const STREAM_OUTPUT_CHUNK: usize = 64 * 1024;
const DEFLATE_WINDOW: usize = 32 * 1024;

struct StreamOutput {
    window: std::collections::VecDeque<u8>,
    pending: Vec<u8>,
    total: usize,
    limit: usize,
}

impl StreamOutput {
    fn new(limit: usize) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(DEFLATE_WINDOW),
            pending: Vec::with_capacity(STREAM_OUTPUT_CHUNK),
            total: 0,
            limit,
        }
    }

    fn reserve(&self, additional: usize) -> Result<(), String> {
        crate::checked_decompressed_len(self.total, additional, self.limit, "inflate").map(|_| ())
    }

    fn push(&mut self, byte: u8) {
        if self.window.len() == DEFLATE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(byte);
        self.pending.push(byte);
        self.total += 1;
    }

    fn extend(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.reserve(bytes.len())?;
        for &byte in bytes {
            self.push(byte);
        }
        Ok(())
    }

    fn copy(&mut self, distance: usize, length: usize) -> Result<(), String> {
        if distance == 0 || distance > self.window.len() {
            return Err("inflate: distance too far back".into());
        }
        self.reserve(length)?;
        for _ in 0..length {
            let byte = self.window[self.window.len() - distance];
            self.push(byte);
        }
        Ok(())
    }

    fn take(&mut self) -> Vec<u8> {
        std::mem::replace(&mut self.pending, Vec::with_capacity(STREAM_OUTPUT_CHUNK))
    }
}

enum InflateStreamState {
    BlockHeader,
    StoredHeader {
        final_block: bool,
    },
    Stored {
        final_block: bool,
        remaining: usize,
    },
    DynamicHeader {
        final_block: bool,
    },
    Compressed {
        final_block: bool,
        lit: Huffman,
        dist: Huffman,
    },
    Done,
}

/// One incremental raw-DEFLATE step. Output is split near 64 KiB so callers can enqueue it under
/// backpressure instead of materializing the full regenerated stream.
#[derive(Debug)]
pub struct InflateStep {
    pub output: Vec<u8>,
    pub done: bool,
    pub needs_input: bool,
}

/// Incremental RFC 1951 decoder with a 32 KiB history window and a total regenerated-byte limit.
pub struct InflateStream {
    input: Vec<u8>,
    bit_buf: u32,
    bit_cnt: u32,
    state: InflateStreamState,
    output: StreamOutput,
    finishing: bool,
}

impl InflateStream {
    pub fn new(limit: usize) -> Self {
        Self {
            input: Vec::new(),
            bit_buf: 0,
            bit_cnt: 0,
            state: InflateStreamState::BlockHeader,
            output: StreamOutput::new(limit),
            finishing: false,
        }
    }

    /// Consume another compressed chunk. Once `finish` is true no more input may be supplied;
    /// callers keep invoking this with an empty chunk until `done` drains all pending output.
    pub fn push(&mut self, input: &[u8], finish: bool) -> Result<InflateStep, String> {
        if self.finishing && !input.is_empty() {
            return Err("inflate: input supplied after finish".into());
        }
        if matches!(self.state, InflateStreamState::Done) && !input.is_empty() {
            return Err("inflate: trailing bytes after final block".into());
        }
        if self.input.len().saturating_add(input.len()) > crate::MAX_DECOMPRESSED_BYTES {
            return Err("inflate: compressed input exceeds byte limit".into());
        }
        self.input.extend_from_slice(input);
        self.finishing |= finish;

        let mut state = std::mem::replace(&mut self.state, InflateStreamState::Done);
        let mut reader = BitReader {
            data: &self.input,
            pos: 0,
            bit_buf: self.bit_buf,
            bit_cnt: self.bit_cnt,
        };
        let mut needs_input = false;

        'decode: while self.output.pending.len() < STREAM_OUTPUT_CHUNK {
            match &mut state {
                InflateStreamState::BlockHeader => {
                    let checkpoint = reader;
                    let header = (|| -> Result<(bool, u32), String> {
                        Ok((reader.bit()? != 0, reader.bits(2)?))
                    })();
                    let (final_block, block_type) = match header {
                        Ok(header) => header,
                        Err(error) if error == "inflate: unexpected end of input" => {
                            reader = checkpoint;
                            needs_input = true;
                            break 'decode;
                        }
                        Err(error) => return Err(error),
                    };
                    match block_type {
                        0 => {
                            reader.align_to_byte();
                            state = InflateStreamState::StoredHeader { final_block };
                        }
                        1 => {
                            let (lit, dist) = fixed_huffman();
                            state = InflateStreamState::Compressed {
                                final_block,
                                lit,
                                dist,
                            };
                        }
                        2 => {
                            state = InflateStreamState::DynamicHeader { final_block };
                        }
                        _ => return Err("inflate: reserved block type".into()),
                    }
                }
                InflateStreamState::StoredHeader { final_block } => {
                    let checkpoint = reader;
                    let lengths = (|| -> Result<(usize, usize), String> {
                        Ok((reader.bits(16)? as usize, reader.bits(16)? as usize))
                    })();
                    let (len, nlen) = match lengths {
                        Ok(lengths) => lengths,
                        Err(error) if error == "inflate: unexpected end of input" => {
                            reader = checkpoint;
                            needs_input = true;
                            break 'decode;
                        }
                        Err(error) => return Err(error),
                    };
                    if len ^ nlen != 0xffff {
                        return Err("inflate: invalid stored block length".into());
                    }
                    state = InflateStreamState::Stored {
                        final_block: *final_block,
                        remaining: len,
                    };
                }
                InflateStreamState::Stored {
                    final_block,
                    remaining,
                } => {
                    if *remaining == 0 {
                        state = if *final_block {
                            InflateStreamState::Done
                        } else {
                            InflateStreamState::BlockHeader
                        };
                        continue;
                    }
                    debug_assert_eq!(reader.bit_cnt, 0);
                    let available = reader.data.len().saturating_sub(reader.pos);
                    let room = STREAM_OUTPUT_CHUNK - self.output.pending.len();
                    let count = (*remaining).min(available).min(room);
                    if count == 0 {
                        needs_input = available == 0;
                        break 'decode;
                    }
                    self.output
                        .extend(&reader.data[reader.pos..reader.pos + count])?;
                    reader.pos += count;
                    *remaining -= count;
                }
                InflateStreamState::DynamicHeader { final_block } => {
                    let checkpoint = reader;
                    match read_dynamic_tables(&mut reader) {
                        Ok((lit, dist)) => {
                            state = InflateStreamState::Compressed {
                                final_block: *final_block,
                                lit,
                                dist,
                            };
                        }
                        Err(error) if error == "inflate: unexpected end of input" => {
                            reader = checkpoint;
                            needs_input = true;
                            break 'decode;
                        }
                        Err(error) => return Err(error),
                    }
                }
                InflateStreamState::Compressed {
                    final_block,
                    lit,
                    dist,
                } => {
                    let checkpoint = reader;
                    let symbol = match lit.decode(&mut reader) {
                        Ok(symbol) => symbol,
                        Err(error) if error == "inflate: unexpected end of input" => {
                            reader = checkpoint;
                            needs_input = true;
                            break 'decode;
                        }
                        Err(error) => return Err(error),
                    };
                    match symbol {
                        0..=255 => {
                            self.output.reserve(1)?;
                            self.output.push(symbol as u8);
                        }
                        256 => {
                            state = if *final_block {
                                InflateStreamState::Done
                            } else {
                                InflateStreamState::BlockHeader
                            };
                        }
                        257..=285 => {
                            let parsed = (|| -> Result<(usize, usize), String> {
                                let index = (symbol - 257) as usize;
                                let length = LENGTH_BASE[index] as usize
                                    + reader.bits(LENGTH_EXTRA[index] as u32)? as usize;
                                let distance_symbol = dist.decode(&mut reader)? as usize;
                                if distance_symbol >= 30 {
                                    return Err("inflate: invalid distance symbol".into());
                                }
                                let distance = DIST_BASE[distance_symbol] as usize
                                    + reader.bits(DIST_EXTRA[distance_symbol] as u32)? as usize;
                                Ok((distance, length))
                            })();
                            let (distance, length) = match parsed {
                                Ok(values) => values,
                                Err(error) if error == "inflate: unexpected end of input" => {
                                    reader = checkpoint;
                                    needs_input = true;
                                    break 'decode;
                                }
                                Err(error) => return Err(error),
                            };
                            self.output.copy(distance, length)?;
                        }
                        _ => return Err("inflate: invalid literal/length symbol".into()),
                    }
                }
                InflateStreamState::Done => {
                    if reader.pos != reader.data.len() {
                        return Err("inflate: trailing bytes after final block".into());
                    }
                    break 'decode;
                }
            }
        }

        let consumed = reader.pos;
        self.bit_buf = reader.bit_buf;
        self.bit_cnt = reader.bit_cnt;
        self.input.drain(..consumed);
        self.state = state;

        let done = matches!(self.state, InflateStreamState::Done) && self.input.is_empty();
        if self.finishing && needs_input && !done {
            return Err("inflate: unexpected end of input".into());
        }
        Ok(InflateStep {
            output: self.output.take(),
            done,
            needs_input,
        })
    }
}

/// Framing understood by [`DeflateDecoder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeflateFormat {
    Raw,
    Zlib,
    Gzip,
}

enum ExpectedTrailer {
    None,
    Zlib(u32),
    Gzip { crc: u32, size: u32 },
}

/// Incremental raw/zlib/gzip decoder. Wrapper trailers are held back while compressed bytes feed
/// [`InflateStream`], so checksum validation does not require buffering the encoded body.
pub struct DeflateDecoder {
    format: DeflateFormat,
    inflater: InflateStream,
    buffer: Vec<u8>,
    header_done: bool,
    finishing: bool,
    expected: Option<ExpectedTrailer>,
    adler: u32,
    crc: u32,
    size: u32,
    done: bool,
}

impl DeflateDecoder {
    pub fn new(format: DeflateFormat, limit: usize) -> Self {
        Self {
            format,
            inflater: InflateStream::new(limit),
            buffer: Vec::new(),
            header_done: format == DeflateFormat::Raw,
            finishing: false,
            expected: None,
            adler: 1,
            crc: 0,
            size: 0,
            done: false,
        }
    }

    pub fn push(&mut self, input: &[u8], finish: bool) -> Result<InflateStep, String> {
        if self.done {
            if input.is_empty() {
                return Ok(InflateStep {
                    output: Vec::new(),
                    done: true,
                    needs_input: false,
                });
            }
            return Err("decompression: input supplied after end of stream".into());
        }
        if self.finishing && !input.is_empty() {
            return Err("decompression: input supplied after finish".into());
        }
        if self.buffer.len().saturating_add(input.len()) > crate::MAX_DECOMPRESSED_BYTES {
            return Err("decompression: compressed input exceeds byte limit".into());
        }
        self.buffer.extend_from_slice(input);

        if !self.header_done {
            let header_len = match self.format {
                DeflateFormat::Raw => 0,
                DeflateFormat::Zlib => match parse_zlib_header(&self.buffer)? {
                    Some(length) => length,
                    None if finish => return Err("zlib: truncated header".into()),
                    None => {
                        return Ok(InflateStep {
                            output: Vec::new(),
                            done: false,
                            needs_input: true,
                        });
                    }
                },
                DeflateFormat::Gzip => match parse_gzip_header(&self.buffer)? {
                    Some(length) => length,
                    None if finish => return Err("gzip: truncated header".into()),
                    None => {
                        return Ok(InflateStep {
                            output: Vec::new(),
                            done: false,
                            needs_input: true,
                        });
                    }
                },
            };
            self.buffer.drain(..header_len);
            self.header_done = true;
        }

        let trailer_len = match self.format {
            DeflateFormat::Raw => 0,
            DeflateFormat::Zlib => 4,
            DeflateFormat::Gzip => 8,
        };
        if finish && !self.finishing {
            if self.buffer.len() < trailer_len {
                return Err(match self.format {
                    DeflateFormat::Raw => "inflate: unexpected end of input",
                    DeflateFormat::Zlib => "zlib: truncated trailer",
                    DeflateFormat::Gzip => "gzip: truncated trailer",
                }
                .into());
            }
            let trailer_at = self.buffer.len() - trailer_len;
            self.expected = Some(match self.format {
                DeflateFormat::Raw => ExpectedTrailer::None,
                DeflateFormat::Zlib => ExpectedTrailer::Zlib(u32::from_be_bytes(
                    self.buffer[trailer_at..].try_into().unwrap(),
                )),
                DeflateFormat::Gzip => ExpectedTrailer::Gzip {
                    crc: u32::from_le_bytes(
                        self.buffer[trailer_at..trailer_at + 4].try_into().unwrap(),
                    ),
                    size: u32::from_le_bytes(self.buffer[trailer_at + 4..].try_into().unwrap()),
                },
            });
            self.buffer.truncate(trailer_at);
            self.finishing = true;
        }

        let feed_len = if self.finishing {
            self.buffer.len()
        } else {
            self.buffer.len().saturating_sub(trailer_len)
        };
        let mut step = self
            .inflater
            .push(&self.buffer[..feed_len], self.finishing)?;
        self.buffer.drain(..feed_len);
        self.adler = adler32_from(self.adler, &step.output);
        self.crc = crc32_from(self.crc, &step.output);
        self.size = self.size.wrapping_add(step.output.len() as u32);

        let inflater_done = step.done;
        if inflater_done && self.finishing {
            match self.expected.take().unwrap_or(ExpectedTrailer::None) {
                ExpectedTrailer::None => {}
                ExpectedTrailer::Zlib(expected) if expected != self.adler => {
                    return Err("zlib: Adler-32 checksum mismatch".into())
                }
                ExpectedTrailer::Gzip { crc, .. } if crc != self.crc => {
                    return Err("gzip: CRC-32 checksum mismatch".into())
                }
                ExpectedTrailer::Gzip { size, .. } if size != self.size => {
                    return Err("gzip: uncompressed size mismatch".into())
                }
                _ => {}
            }
            self.done = true;
            step.done = true;
        } else {
            step.done = false;
            if inflater_done {
                step.needs_input = true;
            }
        }
        Ok(step)
    }
}

fn parse_zlib_header(data: &[u8]) -> Result<Option<usize>, String> {
    let Some((&cmf, rest)) = data.split_first() else {
        return Ok(None);
    };
    let Some(&flg) = rest.first() else {
        return Ok(None);
    };
    if cmf & 0x0f != 8 {
        return Err("zlib: unsupported compression method".into());
    }
    if cmf >> 4 > 7 {
        return Err("zlib: invalid window size".into());
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err("zlib: invalid header check bits".into());
    }
    if flg & 0x20 != 0 {
        return Err("zlib: preset dictionaries are not supported".into());
    }
    Ok(Some(2))
}

fn parse_gzip_header(data: &[u8]) -> Result<Option<usize>, String> {
    if data.len() < 10 {
        return Ok(None);
    }
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err("gzip: bad magic".into());
    }
    if data[2] != 8 {
        return Err("gzip: unsupported compression method".into());
    }
    let flags = data[3];
    if flags & 0xe0 != 0 {
        return Err("gzip: reserved flags are set".into());
    }
    let mut pos = 10usize;
    if flags & 0x04 != 0 {
        if data.len() < pos + 2 {
            return Ok(None);
        }
        let length = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos = pos
            .checked_add(2 + length)
            .ok_or("gzip: header length overflow")?;
        if data.len() < pos {
            return Ok(None);
        }
    }
    for flag in [0x08, 0x10] {
        if flags & flag != 0 {
            let Some(end) = data[pos..].iter().position(|&byte| byte == 0) else {
                return Ok(None);
            };
            pos += end + 1;
        }
    }
    if flags & 0x02 != 0 {
        if data.len() < pos + 2 {
            return Ok(None);
        }
        let expected = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        if crc32(&data[..pos]) as u16 != expected {
            return Err("gzip: header checksum mismatch".into());
        }
        pos += 2;
    }
    Ok(Some(pos))
}

/// Decode raw DEFLATE (no zlib/gzip wrapper).
pub fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    inflate_with_limit(data, crate::MAX_DECOMPRESSED_BYTES)
}

/// Decode raw DEFLATE while rejecting output beyond `limit` before allocating or copying it.
pub fn inflate_with_limit(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let (out, consumed) = inflate_inner(data, limit)?;
    // Compression Standard §3 requires an error for bytes after the BFINAL block.
    if consumed != data.len() {
        return Err("inflate: trailing bytes after final block".into());
    }
    Ok(out)
}

fn inflate_inner(data: &[u8], limit: usize) -> Result<(Vec<u8>, usize), String> {
    let mut reader = BitReader::new(data);
    let mut out = Vec::new();
    loop {
        let final_block = reader.bit()?;
        let btype = reader.bits(2)?;
        match btype {
            0 => {
                reader.align_to_byte();
                if reader.pos + 4 > data.len() {
                    return Err("inflate: truncated stored block".into());
                }
                let len = data[reader.pos] as usize | ((data[reader.pos + 1] as usize) << 8);
                let nlen = data[reader.pos + 2] as usize | ((data[reader.pos + 3] as usize) << 8);
                if len ^ nlen != 0xffff {
                    return Err("inflate: invalid stored block length".into());
                }
                reader.pos += 4; // LEN + NLEN
                if reader.pos + len > data.len() {
                    return Err("inflate: stored block overruns input".into());
                }
                crate::checked_decompressed_len(out.len(), len, limit, "inflate")?;
                out.extend_from_slice(&data[reader.pos..reader.pos + len]);
                reader.pos += len;
            }
            1 => {
                let (lit, dist) = fixed_huffman();
                inflate_block(&mut reader, &mut out, &lit, &dist, limit)?;
            }
            2 => {
                let (lit, dist) = read_dynamic_tables(&mut reader)?;
                inflate_block(&mut reader, &mut out, &lit, &dist, limit)?;
            }
            _ => return Err("inflate: reserved block type".into()),
        }
        if final_block == 1 {
            return Ok((out, reader.pos));
        }
    }
}

const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn read_dynamic_tables(reader: &mut BitReader) -> Result<(Huffman, Huffman), String> {
    let hlit = reader.bits(5)? as usize + 257;
    let hdist = reader.bits(5)? as usize + 1;
    let hclen = reader.bits(4)? as usize + 4;

    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[CODE_LENGTH_ORDER[i]] = reader.bits(3)? as u8;
    }
    let cl_huffman = Huffman::new(&cl_lengths);

    let mut lengths = Vec::with_capacity(hlit + hdist);
    while lengths.len() < hlit + hdist {
        let sym = cl_huffman.decode(reader)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths
                    .last()
                    .ok_or("inflate: repeat with no previous length")?;
                for _ in 0..(reader.bits(2)? + 3) {
                    lengths.push(prev);
                }
            }
            17 => {
                let n = reader.bits(3)? as usize + 3;
                lengths.resize(lengths.len() + n, 0);
            }
            18 => {
                let n = reader.bits(7)? as usize + 11;
                lengths.resize(lengths.len() + n, 0);
            }
            _ => return Err("inflate: invalid code-length symbol".into()),
        }
    }
    if lengths.len() > hlit + hdist {
        return Err("inflate: code-length overrun".into());
    }
    let (lit_lengths, dist_lengths) = lengths.split_at(hlit);
    Ok((Huffman::new(lit_lengths), Huffman::new(dist_lengths)))
}

// ---- deflate encoder (fixed Huffman + greedy LZ77) --------------------------------------------

struct BitWriter {
    out: Vec<u8>,
    bit_buf: u32,
    bit_cnt: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            bit_buf: 0,
            bit_cnt: 0,
        }
    }
    fn write(&mut self, value: u32, n: u32) {
        self.bit_buf |= value << self.bit_cnt;
        self.bit_cnt += n;
        while self.bit_cnt >= 8 {
            self.out.push((self.bit_buf & 0xff) as u8);
            self.bit_buf >>= 8;
            self.bit_cnt -= 8;
        }
    }
    /// Huffman codes are written MSB-first (bit-reversed relative to the LSB bit order).
    fn write_code(&mut self, code: u32, n: u32) {
        let mut reversed = 0;
        for i in 0..n {
            reversed |= ((code >> i) & 1) << (n - 1 - i);
        }
        self.write(reversed, n);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bit_cnt > 0 {
            self.out.push((self.bit_buf & 0xff) as u8);
        }
        self.out
    }
    fn take_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.out)
    }
}

/// Fixed-Huffman literal/length code for a symbol (0..=287) — code value and bit length.
fn fixed_lit_code(sym: u16) -> (u32, u32) {
    match sym {
        0..=143 => (0x30 + sym as u32, 8),
        144..=255 => (0x190 + (sym as u32 - 144), 9),
        256..=279 => (sym as u32 - 256, 7),
        _ => (0xc0 + (sym as u32 - 280), 8),
    }
}

fn length_symbol(length: usize) -> (u16, u32, u32) {
    for i in (0..29).rev() {
        if length >= LENGTH_BASE[i] as usize {
            let extra = length - LENGTH_BASE[i] as usize;
            return (257 + i as u16, extra as u32, LENGTH_EXTRA[i] as u32);
        }
    }
    (257, 0, 0)
}

fn dist_symbol(distance: usize) -> (u16, u32, u32) {
    for i in (0..30).rev() {
        if distance >= DIST_BASE[i] as usize {
            let extra = distance - DIST_BASE[i] as usize;
            return (i as u16, extra as u32, DIST_EXTRA[i] as u32);
        }
    }
    (0, 0, 0)
}

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;

fn hash3(data: &[u8], i: usize) -> usize {
    let v = (data[i] as usize) << 16 | (data[i + 1] as usize) << 8 | data[i + 2] as usize;
    (v.wrapping_mul(2654435761)) >> (32 - HASH_BITS) & (HASH_SIZE - 1)
}

/// Encode raw DEFLATE: one fixed-Huffman block with greedy LZ77 (hash-chain match finder).
pub fn deflate(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    encode_fixed_block(&mut w, data, true);
    w.finish()
}

fn encode_fixed_block(w: &mut BitWriter, data: &[u8], final_block: bool) {
    w.write(u32::from(final_block), 1);
    w.write(1, 2); // BTYPE = 01 (fixed Huffman)

    let emit_literal = |w: &mut BitWriter, byte: u8| {
        let (code, len) = fixed_lit_code(byte as u16);
        w.write_code(code, len);
    };

    let n = data.len();
    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; n.max(1)];
    let mut i = 0;
    while i < n {
        let mut best_len = 0;
        let mut best_dist = 0;
        if i + MIN_MATCH <= n {
            let h = hash3(data, i);
            let mut cand = head[h];
            let mut chain = 0;
            while cand != usize::MAX && chain < 128 {
                let max_len = (n - i).min(MAX_MATCH);
                let mut len = 0;
                while len < max_len && data[cand + len] == data[i + len] {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_dist = i - cand;
                    if len >= max_len {
                        break;
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
            prev[i] = head[h];
            head[h] = i;
        }

        if best_len >= MIN_MATCH {
            let (lsym, lextra, lbits) = length_symbol(best_len);
            let (lcode, lcodelen) = fixed_lit_code(lsym);
            w.write_code(lcode, lcodelen);
            if lbits > 0 {
                w.write(lextra, lbits);
            }
            let (dsym, dextra, dbits) = dist_symbol(best_dist);
            w.write_code(dsym as u32, 5);
            if dbits > 0 {
                w.write(dextra, dbits);
            }
            // Insert hash entries for the bytes the match covers (skip the first, already inserted).
            let end = i + best_len;
            let mut j = i + 1;
            while j < end && j + MIN_MATCH <= n {
                let h = hash3(data, j);
                prev[j] = head[h];
                head[h] = j;
                j += 1;
            }
            i = end;
        } else {
            emit_literal(w, data[i]);
            i += 1;
        }
    }
    // End-of-block symbol (256).
    let (code, len) = fixed_lit_code(256);
    w.write_code(code, len);
}

/// Incremental compressor for the web CompressionStream formats. Each input chunk is one
/// non-final fixed-Huffman block; this preserves streaming latency and within-chunk LZ77 speed
/// without retaining author input. The flush adds a final empty block and the wrapper checksum.
pub struct DeflateEncoder {
    format: DeflateFormat,
    writer: Option<BitWriter>,
    pending: Vec<u8>,
    adler: u32,
    crc: u32,
    size: u32,
    done: bool,
}

impl DeflateEncoder {
    pub fn new(format: DeflateFormat) -> Self {
        let pending = match format {
            DeflateFormat::Raw => Vec::new(),
            DeflateFormat::Zlib => vec![0x78, 0x9c],
            DeflateFormat::Gzip => vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff],
        };
        Self {
            format,
            writer: Some(BitWriter::new()),
            pending,
            adler: 1,
            crc: 0,
            size: 0,
            done: false,
        }
    }

    pub fn push(&mut self, input: &[u8], finish: bool) -> Result<Vec<u8>, String> {
        if self.done {
            return if input.is_empty() {
                Ok(Vec::new())
            } else {
                Err("deflate: input supplied after finish".into())
            };
        }
        if !input.is_empty() {
            encode_fixed_block(self.writer.as_mut().unwrap(), input, false);
            self.adler = adler32_from(self.adler, input);
            self.crc = crc32_from(self.crc, input);
            self.size = self.size.wrapping_add(input.len() as u32);
        }
        self.pending
            .extend(self.writer.as_mut().unwrap().take_output());
        if finish {
            let mut writer = self.writer.take().unwrap();
            encode_fixed_block(&mut writer, &[], true);
            self.pending.extend(writer.finish());
            match self.format {
                DeflateFormat::Raw => {}
                DeflateFormat::Zlib => self.pending.extend_from_slice(&self.adler.to_be_bytes()),
                DeflateFormat::Gzip => {
                    self.pending.extend_from_slice(&self.crc.to_le_bytes());
                    self.pending.extend_from_slice(&self.size.to_le_bytes());
                }
            }
            self.done = true;
        }
        Ok(std::mem::take(&mut self.pending))
    }
}

// ---- zlib / gzip framing ----------------------------------------------------------------------

pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x9c]; // CMF/FLG (deflate, default window, default level)
    out.extend_from_slice(&deflate(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    zlib_decompress_with_limit(data, crate::MAX_DECOMPRESSED_BYTES)
}

pub fn zlib_decompress_with_limit(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    if data.len() < 6 {
        return Err("zlib: input too short".into());
    }
    let cmf = data[0];
    let flg = data[1];
    // RFC 1950 §2.2 plus Compression Standard §3: CM=8, CINFO<=7, a valid FCHECK, and no
    // preset dictionary for this API.
    if cmf & 0x0f != 8 {
        return Err("zlib: unsupported compression method".into());
    }
    if cmf >> 4 > 7 {
        return Err("zlib: invalid window size".into());
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err("zlib: invalid header check bits".into());
    }
    if flg & 0x20 != 0 {
        return Err("zlib: preset dictionaries are not supported".into());
    }
    let compressed = &data[2..data.len() - 4];
    let (out, consumed) = inflate_inner(compressed, limit)?;
    if consumed != compressed.len() {
        return Err("zlib: trailing bytes after compressed data".into());
    }
    let expected = u32::from_be_bytes(data[data.len() - 4..].try_into().unwrap());
    if adler32(&out) != expected {
        return Err("zlib: Adler-32 checksum mismatch".into());
    }
    Ok(out)
}

pub fn gzip_compress(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff]; // magic, method, flags, mtime, xfl, os
    out.extend_from_slice(&deflate(data));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

pub fn gzip_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    gzip_decompress_with_limit(data, crate::MAX_DECOMPRESSED_BYTES)
}

pub fn gzip_decompress_with_limit(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b {
        return Err("gzip: bad magic".into());
    }
    if data[2] != 8 {
        return Err("gzip: unsupported compression method".into());
    }
    let flags = data[3];
    if flags & 0xe0 != 0 {
        return Err("gzip: reserved flags are set".into());
    }
    let mut pos = 10;
    if flags & 0x04 != 0 {
        // FEXTRA
        if pos + 2 > data.len() {
            return Err("gzip: truncated extra field".into());
        }
        let xlen = data[pos] as usize | ((data[pos + 1] as usize) << 8);
        pos = pos
            .checked_add(2)
            .and_then(|pos| pos.checked_add(xlen))
            .filter(|&pos| pos <= data.len())
            .ok_or("gzip: truncated extra field")?;
    }
    if flags & 0x08 != 0 {
        // FNAME (NUL-terminated)
        pos += data[pos..]
            .iter()
            .position(|&byte| byte == 0)
            .ok_or("gzip: unterminated file name")?
            + 1;
    }
    if flags & 0x10 != 0 {
        // FCOMMENT
        pos += data[pos..]
            .iter()
            .position(|&byte| byte == 0)
            .ok_or("gzip: unterminated comment")?
            + 1;
    }
    if flags & 0x02 != 0 {
        let end = pos.checked_add(2).ok_or("gzip: header overflow")?;
        let expected = u16::from_le_bytes(
            data.get(pos..end)
                .ok_or("gzip: truncated header checksum")?
                .try_into()
                .unwrap(),
        );
        if crc32(&data[..pos]) as u16 != expected {
            return Err("gzip: header checksum mismatch".into());
        }
        pos = end;
    }
    if pos + 8 > data.len() {
        return Err("gzip: truncated".into());
    }
    let compressed = &data[pos..data.len() - 8];
    let (out, consumed) = inflate_inner(compressed, limit)?;
    if consumed != compressed.len() {
        return Err("gzip: trailing bytes after compressed data".into());
    }
    let trailer = &data[data.len() - 8..];
    let expected_crc = u32::from_le_bytes(trailer[..4].try_into().unwrap());
    let expected_size = u32::from_le_bytes(trailer[4..].try_into().unwrap());
    if crc32(&out) != expected_crc {
        return Err("gzip: CRC-32 checksum mismatch".into());
    }
    if out.len() as u32 != expected_size {
        return Err("gzip: uncompressed size mismatch".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(input: &str) -> Vec<u8> {
        input
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let digit = |byte| match byte {
                    b'0'..=b'9' => byte - b'0',
                    b'a'..=b'f' => byte - b'a' + 10,
                    _ => panic!("invalid hex"),
                };
                digit(pair[0]) << 4 | digit(pair[1])
            })
            .collect()
    }

    fn stream_one_byte_at_a_time(encoded: &[u8], limit: usize) -> Result<Vec<u8>, String> {
        let mut stream = InflateStream::new(limit);
        let mut output = Vec::new();
        for byte in encoded {
            let mut step = stream.push(std::slice::from_ref(byte), false)?;
            loop {
                assert!(step.output.len() <= STREAM_OUTPUT_CHUNK + 258);
                output.extend_from_slice(&step.output);
                if step.done || step.needs_input {
                    break;
                }
                step = stream.push(&[], false)?;
            }
        }
        loop {
            let step = stream.push(&[], true)?;
            output.extend_from_slice(&step.output);
            if step.done {
                return Ok(output);
            }
        }
    }

    fn stream_wrapper_one_byte_at_a_time(
        encoded: &[u8],
        format: DeflateFormat,
        limit: usize,
    ) -> Result<Vec<u8>, String> {
        let mut stream = DeflateDecoder::new(format, limit);
        let mut output = Vec::new();
        for byte in encoded {
            let mut step = stream.push(std::slice::from_ref(byte), false)?;
            loop {
                output.extend_from_slice(&step.output);
                if step.done || step.needs_input {
                    break;
                }
                step = stream.push(&[], false)?;
            }
        }
        loop {
            let step = stream.push(&[], true)?;
            output.extend_from_slice(&step.output);
            if step.done {
                return Ok(output);
            }
        }
    }

    fn roundtrip(data: &[u8]) {
        assert_eq!(inflate(&deflate(data)).unwrap(), data, "raw deflate");
        assert_eq!(zlib_decompress(&zlib_compress(data)).unwrap(), data, "zlib");
        assert_eq!(gzip_decompress(&gzip_compress(data)).unwrap(), data, "gzip");
    }

    #[test]
    fn roundtrips() {
        roundtrip(b"");
        roundtrip(b"a");
        roundtrip(b"hello, hello, hello world!");
        roundtrip(&[0u8; 1000]); // long run — exercises back-references
        let repetitive: Vec<u8> = (0..5000).map(|i| (i % 7) as u8).collect();
        roundtrip(&repetitive);
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        roundtrip(text.as_bytes());
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = "abcabcabcabc".repeat(100);
        let compressed = deflate(data.as_bytes());
        assert!(
            compressed.len() < data.len() / 2,
            "expected real compression"
        );
    }

    #[test]
    fn checksums_match_known_values() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn decompression_limits_apply_before_expansion() {
        let input = b"highly compressible highly compressible";
        assert!(inflate_with_limit(&deflate(input), input.len() - 1)
            .unwrap_err()
            .contains("byte limit"));
        assert!(
            zlib_decompress_with_limit(&zlib_compress(input), input.len() - 1)
                .unwrap_err()
                .contains("byte limit")
        );
        assert!(
            gzip_decompress_with_limit(&gzip_compress(input), input.len() - 1)
                .unwrap_err()
                .contains("byte limit")
        );
    }

    #[test]
    fn wrappers_validate_normative_framing_and_checksums() {
        let mut zlib = zlib_compress(b"payload");
        *zlib.last_mut().unwrap() ^= 1;
        assert!(zlib_decompress(&zlib)
            .unwrap_err()
            .contains("checksum mismatch"));

        let mut gzip = gzip_compress(b"payload");
        let crc_at = gzip.len() - 8;
        gzip[crc_at] ^= 1;
        assert!(gzip_decompress(&gzip)
            .unwrap_err()
            .contains("checksum mismatch"));

        let mut with_header_crc = gzip_compress(b"header CRC");
        with_header_crc[3] |= 0x02;
        let checksum = (crc32(&with_header_crc[..10]) as u16).to_le_bytes();
        with_header_crc.splice(10..10, checksum);
        assert_eq!(gzip_decompress(&with_header_crc).unwrap(), b"header CRC");
        with_header_crc[10] ^= 1;
        assert!(gzip_decompress(&with_header_crc)
            .unwrap_err()
            .contains("header checksum"));

        let mut raw = deflate(b"payload");
        raw.push(0);
        assert!(inflate(&raw).unwrap_err().contains("trailing bytes"));

        // One final stored block containing one byte, but LEN and NLEN are not complements.
        assert!(inflate(&[1, 1, 0, 1, 0, b'x'])
            .unwrap_err()
            .contains("stored block length"));
    }

    #[test]
    fn incremental_inflater_preserves_state_across_every_byte_boundary() {
        let expected = b"abc123".repeat(1000);
        let dynamic = unhex("edc4310100000400b04c4884fe1d84f06ec77a36b2dab66ddbb66ddbb66d3f3e");
        assert_eq!(
            stream_one_byte_at_a_time(&dynamic, expected.len()).unwrap(),
            expected
        );

        let stored = [1, 3, 0, 0xfc, 0xff, b'a', b'b', b'c'];
        assert_eq!(stream_one_byte_at_a_time(&stored, 3).unwrap(), b"abc");
    }

    #[test]
    fn incremental_inflater_bounds_output_and_finish_state() {
        let encoded = deflate(&b"x".repeat(1000));
        assert!(stream_one_byte_at_a_time(&encoded, 999)
            .unwrap_err()
            .contains("byte limit"));

        let mut stream = InflateStream::new(1024);
        assert!(stream.push(&encoded[..encoded.len() - 1], true).is_err());

        let mut stream = InflateStream::new(1024);
        let mut with_trailing = encoded;
        with_trailing.push(0);
        assert!(stream
            .push(&with_trailing, true)
            .unwrap_err()
            .contains("trailing bytes"));
    }

    #[test]
    fn incremental_wrappers_validate_across_every_byte_boundary() {
        let input = b"wrapped incremental checksums".repeat(1000);
        assert_eq!(
            stream_wrapper_one_byte_at_a_time(
                &zlib_compress(&input),
                DeflateFormat::Zlib,
                input.len(),
            )
            .unwrap(),
            input
        );

        let mut gzip = gzip_compress(&input);
        gzip[3] |= 0x02;
        let header_crc = (crc32(&gzip[..10]) as u16).to_le_bytes();
        gzip.splice(10..10, header_crc);
        assert_eq!(
            stream_wrapper_one_byte_at_a_time(&gzip, DeflateFormat::Gzip, input.len()).unwrap(),
            input
        );

        let mut random = 0x1234_5678u32;
        let incompressible: Vec<u8> = (0..200_000)
            .map(|_| {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                (random >> 24) as u8
            })
            .collect();
        let encoded = gzip_compress(&incompressible);
        let mut decoder = DeflateDecoder::new(DeflateFormat::Gzip, incompressible.len());
        let first = decoder.push(&encoded[..encoded.len() / 4], false).unwrap();
        assert!(
            !first.output.is_empty(),
            "large chunk produced no early output"
        );
    }

    #[test]
    fn incremental_wrappers_reject_checksums_and_limits() {
        let input = b"bounded wrapper".repeat(100);
        let mut zlib = zlib_compress(&input);
        *zlib.last_mut().unwrap() ^= 1;
        assert!(
            stream_wrapper_one_byte_at_a_time(&zlib, DeflateFormat::Zlib, input.len())
                .unwrap_err()
                .contains("checksum mismatch")
        );

        assert!(stream_wrapper_one_byte_at_a_time(
            &gzip_compress(&input),
            DeflateFormat::Gzip,
            input.len() - 1,
        )
        .unwrap_err()
        .contains("byte limit"));
    }

    #[test]
    fn incremental_encoder_emits_valid_blocks_without_retaining_input() {
        let input = b"chunk-local compression still streams".repeat(100);
        for format in [DeflateFormat::Raw, DeflateFormat::Zlib, DeflateFormat::Gzip] {
            let mut encoder = DeflateEncoder::new(format);
            let mut encoded = Vec::new();
            for chunk in input.chunks(17) {
                encoded.extend(encoder.push(chunk, false).unwrap());
            }
            encoded.extend(encoder.push(&[], true).unwrap());
            let decoded = match format {
                DeflateFormat::Raw => inflate(&encoded),
                DeflateFormat::Zlib => zlib_decompress(&encoded),
                DeflateFormat::Gzip => gzip_decompress(&encoded),
            }
            .unwrap();
            assert_eq!(decoded, input);
        }
    }
}
