use alloc::vec::Vec;

pub const HEADER: usize = 64;
pub const DIR_ENTRY: usize = 32;
pub const ALIGN: u64 = 64;
pub const I8: u8 = 0;
pub const F32: u8 = 1;

#[derive(Debug)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub ctx_len: u32,
    pub vocab: u32,
    pub temp: f32,
    pub n_tensors: u32,
    pub crc32: u32,
    // maybe extra stuff later ig
}

#[derive(Debug, PartialEq, Eq)]
pub enum LoadError {
    TooShort,
    BadMagic,
    BadVersion,
    DModel,
    NLayers,
    CtxSize,
    Temp,
    Vocab,
    DModelNHeads,
    NTensors,
    RecordOutOfBounds,
    BadDtype,
    BadNDims,
    BadDims,
    BadScaleCount,
    BadAlignment,
    DataOutOfBounds,
    ScalesOutOfBounds,
    BadTensorShape,
    BadTotalLength,
    BadCrc,
    BadScaleValue,
}

#[derive(Debug)]
pub struct TensorRecord {
    pub dtype: u8,
    pub n_dims: u8,
    pub dim0: u32,
    pub dim1: u32,
    pub scale_count: u32,
    pub data_offset: u64,
    pub scales_offset: u64,
}

impl TensorRecord {
    // dim1 == 0 means 1-D, so treat it as a single column for the size math
    pub fn cols(&self) -> u64 {
        if self.dim1 == 0 { 1 } else { self.dim1 as u64 }
    }

    pub fn element_size(&self) -> u64 {
        if self.dtype == I8 { 1 } else { 4 }
    }

    pub fn data_len(&self) -> u64 {
        self.dim0 as u64 * self.cols() * self.element_size()
    }

    pub fn scales_len(&self) -> u64 {
        self.scale_count as u64 * 4
    }

    fn load(buf: &[u8], offset: usize, dir_end: u64) -> Result<TensorRecord, LoadError> {
        if offset + DIR_ENTRY > buf.len() {
            return Err(LoadError::RecordOutOfBounds);
        }

        let dtype = buf[offset];
        if dtype != I8 && dtype != F32 {
            return Err(LoadError::BadDtype);
        }

        let n_dims = buf[offset + 1];
        if n_dims != 1 && n_dims != 2 {
            return Err(LoadError::BadNDims);
        }

        let dim0 = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap());
        let dim1 = u32::from_le_bytes(buf[offset + 8..offset + 12].try_into().unwrap());
        let scale_count = u32::from_le_bytes(buf[offset + 12..offset + 16].try_into().unwrap());
        let data_offset = u64::from_le_bytes(buf[offset + 16..offset + 24].try_into().unwrap());
        let scales_offset = u64::from_le_bytes(buf[offset + 24..offset + 32].try_into().unwrap());

        if dim0 == 0 || (n_dims == 1 && dim1 != 0) || (n_dims == 2 && dim1 == 0) {
            return Err(LoadError::BadDims);
        }
        if scale_count != 0 && scale_count != dim0 {
            return Err(LoadError::BadScaleCount);
        }
        if data_offset % ALIGN != 0 {
            return Err(LoadError::BadAlignment);
        }

        let record = TensorRecord {
            dtype,
            n_dims,
            dim0,
            dim1,
            scale_count,
            data_offset,
            scales_offset,
        };

        let data_end = data_offset
            .checked_add(record.data_len())
            .ok_or(LoadError::DataOutOfBounds)?;
        if data_offset < dir_end || data_end > buf.len() as u64 {
            return Err(LoadError::DataOutOfBounds);
        }

        if scale_count > 0 {
            let scales_end = scales_offset
                .checked_add(record.scales_len())
                .ok_or(LoadError::ScalesOutOfBounds)?;
            if scales_offset < dir_end || scales_end > buf.len() as u64 {
                return Err(LoadError::ScalesOutOfBounds);
            }
        }

        Ok(record)
    }
}

// expected (dtype, n_dims, dim0, dim1, scale_count) for tensor directory position `pos`,
// per the frozen order: tok_emb, pos_emb, then 8 tensors per layer, then lnf.g
fn expected_shape(pos: usize, header: &Header) -> (u8, u8, u32, u32, u32) {
    let d = header.d_model;

    if pos == 0 {
        return (I8, 2, 256, d, 256);
    }
    if pos == 1 {
        return (F32, 2, header.ctx_len, d, 0);
    }

    let last = 2 + header.n_layers as usize * 8;
    if pos == last {
        return (F32, 1, d, 0, 0);
    }

    match (pos - 2) % 8 {
        0 => (F32, 1, d, 0, 0),        // ln1.g
        1 => (I8, 2, d, d, d),         // wq
        2 => (I8, 2, d, d, d),         // wk
        3 => (I8, 2, d, d, d),         // wv
        4 => (I8, 2, d, d, d),         // wo
        5 => (F32, 1, d, 0, 0),        // ln2.g
        6 => (I8, 2, 4 * d, d, 4 * d), // w_up
        _ => (I8, 2, d, 4 * d, d),     // w_down
    }
}

fn validate_tensor_order(records: &[TensorRecord], header: &Header) -> Result<(), LoadError> {
    for (pos, rec) in records.iter().enumerate() {
        let (dtype, n_dims, dim0, dim1, scale_count) = expected_shape(pos, header);
        if rec.dtype != dtype
            || rec.n_dims != n_dims
            || rec.dim0 != dim0
            || rec.dim1 != dim1
            || rec.scale_count != scale_count
        {
            return Err(LoadError::BadTensorShape);
        }
    }
    Ok(())
}

pub fn parse_directory(buf: &[u8], header: &Header) -> Result<Vec<TensorRecord>, LoadError> {
    let dir_end = (HEADER + DIR_ENTRY * header.n_tensors as usize) as u64;
    if (buf.len() as u64) < dir_end {
        return Err(LoadError::TooShort);
    }

    let mut records = Vec::with_capacity(header.n_tensors as usize);
    for i in 0..header.n_tensors as usize {
        records.push(TensorRecord::load(buf, HEADER + i * DIR_ENTRY, dir_end)?);
    }

    validate_tensor_order(&records, header)?;

    // every byte after the directory must belong to some tensor's data or scales,
    // and nothing may hang off the end unaccounted for
    let mut max_extent: u64 = 0;
    for rec in &records {
        max_extent = max_extent.max(rec.data_offset + rec.data_len());
        if rec.scale_count > 0 {
            max_extent = max_extent.max(rec.scales_offset + rec.scales_len());
        }
    }
    if max_extent != buf.len() as u64 {
        return Err(LoadError::BadTotalLength);
    }

    Ok(records)
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// zlib crc32 over everything past the header. libm only, so no crate for this.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = CRC_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

impl Header {
    pub fn load(buf: &[u8]) -> Result<Header, LoadError> {
        if buf.len() < HEADER {
            return Err(LoadError::TooShort);
        }

        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if &magic != b"NNSH" {
            return Err(LoadError::BadMagic);
        }

        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != 1 {
            return Err(LoadError::BadVersion);
        }

        let d_model = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let n_layers = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let n_heads = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let ctx_len = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let vocab = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let temp = f32::from_le_bytes(buf[28..32].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let crc = u32::from_le_bytes(buf[36..40].try_into().unwrap());

        if d_model == 0 || d_model > 1024 {
            return Err(LoadError::DModel);
        }
        if n_layers == 0 || n_layers > 16 {
            return Err(LoadError::NLayers);
        }
        if ctx_len == 0 || ctx_len > 2048 {
            return Err(LoadError::CtxSize);
        }
        if !(temp > 0.0 && temp < 100.0) {
            return Err(LoadError::Temp);
        }
        if vocab != 256 {
            return Err(LoadError::Vocab);
        }
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(LoadError::DModelNHeads);
        }
        if n_tensors != 3 + 8 * n_layers {
            return Err(LoadError::NTensors);
        }
        if crc != 0 && crc32(&buf[HEADER..]) != crc {
            return Err(LoadError::BadCrc);
        }

        Ok(Header {
            magic,
            version,
            d_model,
            n_layers,
            n_heads,
            ctx_len,
            vocab,
            temp,
            n_tensors,
            crc32: crc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn sh_bin() -> Vec<u8> {
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../nn/nn/sh.bin")).expect("sh.bin")
    }

    fn gen_valid_header_bytes() -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0..4].copy_from_slice(b"NNSH");
        buf[4..8].copy_from_slice(&u32::to_le_bytes(1));
        buf[8..12].copy_from_slice(&u32::to_le_bytes(320));
        buf[12..16].copy_from_slice(&u32::to_le_bytes(10));
        buf[16..20].copy_from_slice(&u32::to_le_bytes(10));
        buf[20..24].copy_from_slice(&u32::to_le_bytes(512));
        buf[24..28].copy_from_slice(&u32::to_le_bytes(256));
        buf[28..32].copy_from_slice(&f32::to_le_bytes(1.4));
        buf[32..36].copy_from_slice(&u32::to_le_bytes(83));
        // crc stays 0, which means unchecked
        buf
    }

    #[test]
    fn check_gud_header() {
        let Ok(header) = Header::load(&gen_valid_header_bytes()) else {
            panic!("header load failed, ") // give debug if this torques itsefl ig
        };
        assert_eq!(header.d_model, 320);
        assert_eq!(header.n_heads, 10);
        assert_eq!(header.n_tensors, 83);
    }

    #[test]
    fn check_file() {
        let f = sh_bin();
        let Ok(header) = Header::load(&f) else {
            panic!("header load failed, ") // give debug if this torques itsefl ig
        };
        assert_eq!(header.d_model, 320);
        assert_eq!(header.n_heads, 10);
        assert_eq!(header.n_tensors, 83);
        assert_eq!(header.ctx_len, 512);
        assert!((header.temp - 1.4).abs() < 1e-6);
        assert_ne!(header.crc32, 0);
    }

    #[test]
    fn check_tensor_records() {
        let f = sh_bin();
        let header = Header::load(&f).unwrap();
        let recs = parse_directory(&f, &header).unwrap();

        // record 0: tok_emb [256, 320] i8, per-row scales
        assert_eq!(recs[0].dtype, I8);
        assert_eq!(recs[0].dim0, 256);
        assert_eq!(recs[0].dim1, 320);
        assert_eq!(recs[0].scale_count, 256);

        // record 1: pos_emb [512, 320] f32, no scales
        assert_eq!(recs[1].dtype, F32);
        assert_eq!(recs[1].dim0, 512);
        assert_eq!(recs[1].dim1, 320);
        assert_eq!(recs[1].scale_count, 0);
    }

    #[test]
    fn check_full_directory() {
        let f = sh_bin();
        let header = Header::load(&f).unwrap();
        let records = parse_directory(&f, &header).unwrap();

        assert_eq!(records.len(), 83);
        let lnf_g = records.last().unwrap();
        assert_eq!(lnf_g.dtype, F32);
        assert_eq!(lnf_g.n_dims, 1);
        assert_eq!(lnf_g.dim0, 320);
        assert_eq!(lnf_g.scale_count, 0);
    }

    fn patched(f: impl Fn(&mut [u8; 64])) -> LoadError {
        let mut buf = gen_valid_header_bytes();
        f(&mut buf);
        Header::load(&buf).expect_err("should have been rejected")
    }

    #[test]
    fn rejects_bad_headers() {
        assert_eq!(Header::load(&[0u8; 40]).unwrap_err(), LoadError::TooShort);
        assert_eq!(patched(|b| b[0] = b'X'), LoadError::BadMagic);
        assert_eq!(
            patched(|b| b[4..8].copy_from_slice(&2u32.to_le_bytes())),
            LoadError::BadVersion
        );
        assert_eq!(
            patched(|b| b[8..12].copy_from_slice(&2000u32.to_le_bytes())),
            LoadError::DModel
        );
        assert_eq!(
            patched(|b| b[12..16].copy_from_slice(&17u32.to_le_bytes())),
            LoadError::NLayers
        );
        assert_eq!(
            patched(|b| b[20..24].copy_from_slice(&4096u32.to_le_bytes())),
            LoadError::CtxSize
        );
        assert_eq!(
            patched(|b| b[28..32].copy_from_slice(&0.0f32.to_le_bytes())),
            LoadError::Temp
        );
        assert_eq!(
            patched(|b| b[24..28].copy_from_slice(&512u32.to_le_bytes())),
            LoadError::Vocab
        );
        assert_eq!(
            patched(|b| b[16..20].copy_from_slice(&7u32.to_le_bytes())),
            LoadError::DModelNHeads
        );
        assert_eq!(
            patched(|b| b[32..36].copy_from_slice(&84u32.to_le_bytes())),
            LoadError::NTensors
        );
    }

    #[test]
    fn rejects_truncated_directory() {
        let f = sh_bin();
        let header = Header::load(&f).unwrap();
        let short = &f[..HEADER + DIR_ENTRY * 4];
        assert_eq!(
            parse_directory(short, &header).unwrap_err(),
            LoadError::TooShort
        );
    }

    #[test]
    fn crc_matches_real_file() {
        let f = sh_bin();
        let crc = u32::from_le_bytes(f[36..40].try_into().unwrap());
        assert_eq!(crc32(&f[HEADER..]), crc);
    }

    // xorshift, because no rand crate and Math.random is not a thing here
    fn next_rand(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn fuzz_load() {
        let orig = sh_bin();
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let (mut rejected, mut accepted) = (0, 0);

        for _ in 0..500 {
            let mut b = orig.clone();
            let n = 1 + next_rand(&mut state) % 5;
            for _ in 0..n {
                let at = (next_rand(&mut state) as usize) % b.len();
                b[at] = (next_rand(&mut state) % 256) as u8;
            }
            match Header::load(&b).and_then(|h| parse_directory(&b, &h)) {
                Ok(_) => accepted += 1,
                Err(_) => rejected += 1,
            }
        }

        // no panics is the actual assertion here; crc catches essentially everything
        assert_eq!(rejected + accepted, 500);
        assert!(rejected > 400, "only {rejected} of 500 rejected");
    }
}
