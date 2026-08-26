use alloc::vec::Vec;

struct Header {
    magic: [u8; 4],
    version: u32,
    d_model: u32,
    n_layers: u32,
    n_heads: u32,
    ctx_len: u32,
    vocab: u32,
    temp: f32,
    n_tensors: u32,
    crc32: u32,
    // maybe extra stuff later ig
}

#[derive(Debug)]
enum LoadError {
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
}

struct TensorRecord {
    dtype: u8,
    n_dims: u8,
    dim0: u32,
    dim1: u32,
    scale_count: u32,
    data_offset: u64,
    scales_offset: u64,
}

impl TensorRecord {
    // dim1 == 0 means 1-D, so treat it as a single column for the size math
    fn cols(&self) -> u64 {
        if self.dim1 == 0 { 1 } else { self.dim1 as u64 }
    }

    fn element_size(&self) -> u64 {
        if self.dtype == 0 { 1 } else { 4 }
    }

    fn data_len(&self) -> u64 {
        self.dim0 as u64 * self.cols() * self.element_size()
    }

    fn scales_len(&self) -> u64 {
        self.scale_count as u64 * 4
    }

    fn load(buf: &[u8], offset: usize) -> Result<TensorRecord, LoadError> {
        if offset + 32 > buf.len() {
            return Err(LoadError::RecordOutOfBounds);
        }

        let dtype = buf[offset];
        if dtype > 1 {
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
        if data_offset % 64 != 0 {
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
        if data_end > buf.len() as u64 {
            return Err(LoadError::DataOutOfBounds);
        }

        if scale_count > 0 {
            let scales_end = scales_offset
                .checked_add(record.scales_len())
                .ok_or(LoadError::ScalesOutOfBounds)?;
            if scales_end > buf.len() as u64 {
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
        return (0, 2, 256, d, 256);
    }
    if pos == 1 {
        return (1, 2, header.ctx_len, d, 0);
    }

    let last = 2 + header.n_layers as usize * 8;
    if pos == last {
        return (1, 1, d, 0, 0);
    }

    match (pos - 2) % 8 {
        0 => (1, 1, d, 0, 0),         // ln1.g
        1 => (0, 2, d, d, d),         // wq
        2 => (0, 2, d, d, d),         // wk
        3 => (0, 2, d, d, d),         // wv
        4 => (0, 2, d, d, d),         // wo
        5 => (1, 1, d, 0, 0),         // ln2.g
        6 => (0, 2, 4 * d, d, 4 * d), // w_up
        _ => (0, 2, d, 4 * d, d),     // w_down (slot 7)
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

fn parse_directory(buf: &[u8], header: &Header) -> Result<Vec<TensorRecord>, LoadError> {
    let mut records = Vec::with_capacity(header.n_tensors as usize);
    for i in 0..header.n_tensors as usize {
        records.push(TensorRecord::load(buf, 64 + i * 32)?);
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

impl Header {
    fn load(buf: &[u8]) -> Result<Header, LoadError> {
        if buf.len() < 64 {
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
        let crc32 = u32::from_le_bytes(buf[36..40].try_into().unwrap());

        if d_model == 0 || d_model > 1024 {
            return Err(LoadError::DModel);
        }
        if n_layers > 16 {
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
        // TODO: check crc ig, need crate...

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
            crc32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // TODO: crc, its 0 for now
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
        let f = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../nn/nn/sh.bin"))
            .expect("file not found");
        let Ok(header) = Header::load(&f) else {
            panic!("header load failed, ") // give debug if this torques itsefl ig
        };
        assert_eq!(header.d_model, 320);
        assert_eq!(header.n_heads, 10);
        assert_eq!(header.n_tensors, 83);
    }

    #[test]
    fn check_tensor_records() {
        let f = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../nn/nn/sh.bin"))
            .expect("file not found");

        // record 0: tok_emb [256, 320] i8, per-row scales
        let tok_emb = TensorRecord::load(&f, 64).expect("tok_emb record");
        assert_eq!(tok_emb.dtype, 0);
        assert_eq!(tok_emb.n_dims, 2);
        assert_eq!(tok_emb.dim0, 256);
        assert_eq!(tok_emb.dim1, 320);
        assert_eq!(tok_emb.scale_count, 256);

        // record 1: pos_emb [512, 320] f32, no scales
        let pos_emb = TensorRecord::load(&f, 64 + 32).expect("pos_emb record");
        assert_eq!(pos_emb.dtype, 1);
        assert_eq!(pos_emb.n_dims, 2);
        assert_eq!(pos_emb.dim0, 512);
        assert_eq!(pos_emb.dim1, 320);
        assert_eq!(pos_emb.scale_count, 0);
    }

    #[test]
    fn check_full_directory() {
        let f = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../nn/nn/sh.bin"))
            .expect("file not found");
        let header = Header::load(&f).expect("header");
        let records = parse_directory(&f, &header).expect("directory");

        assert_eq!(records.len(), 83);
        // last tensor is lnf.g: [320] f32, no scales
        let lnf_g = records.last().unwrap();
        assert_eq!(lnf_g.dtype, 1);
        assert_eq!(lnf_g.n_dims, 1);
        assert_eq!(lnf_g.dim0, 320);
        assert_eq!(lnf_g.scale_count, 0);
    }
}
