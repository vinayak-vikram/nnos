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
}
