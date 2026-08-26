use alloc::vec::Vec;

use crate::format::{I8, LoadError, TensorRecord};

/// i8 tensors stay as offsets into the file buffer; f32 tensors get copied out because
/// a Vec<u8> is only byte-aligned and we would be reading f32 straight out of it.
pub enum Data {
    I8 { offset: usize, scales: Vec<f32> },
    F32(Vec<f32>),
}

pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Data,
}

fn read_f32s(buf: &[u8], offset: u64, count: usize) -> Vec<f32> {
    let at = offset as usize;
    buf[at..at + count * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl Tensor {
    pub fn build(buf: &[u8], rec: &TensorRecord) -> Result<Tensor, LoadError> {
        let rows = rec.dim0 as usize;
        let cols = rec.cols() as usize;

        let data = if rec.dtype == I8 {
            let scales = read_f32s(buf, rec.scales_offset, rec.scale_count as usize);
            if scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
                return Err(LoadError::BadScaleValue);
            }
            Data::I8 {
                offset: rec.data_offset as usize,
                scales,
            }
        } else {
            Data::F32(read_f32s(buf, rec.data_offset, rows * cols))
        };

        Ok(Tensor { rows, cols, data })
    }

    pub fn scales(&self) -> &[f32] {
        match &self.data {
            Data::I8 { scales, .. } => scales,
            Data::F32(_) => &[],
        }
    }

    /// f32 row, panics on i8 tensors. only called on pos_emb and the norm gains.
    pub fn f32_row(&self, r: usize) -> &[f32] {
        match &self.data {
            Data::F32(v) => &v[r * self.cols..(r + 1) * self.cols],
            Data::I8 { .. } => panic!("f32_row on an i8 tensor"),
        }
    }

    pub fn f32_all(&self) -> &[f32] {
        match &self.data {
            Data::F32(v) => v,
            Data::I8 { .. } => panic!("f32_all on an i8 tensor"),
        }
    }

    /// raw quantized bytes, row-major [rows, cols]. reinterpreted as i8 by the kernels.
    pub fn i8_data<'a>(&self, buf: &'a [u8]) -> &'a [i8] {
        match &self.data {
            Data::I8 { offset, .. } => {
                let raw = &buf[*offset..*offset + self.rows * self.cols];
                // u8 -> i8 is a pure reinterpretation, same layout and alignment
                unsafe { core::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) }
            }
            Data::F32(_) => panic!("i8_data on an f32 tensor"),
        }
    }
}
