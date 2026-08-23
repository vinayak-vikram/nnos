#[derive(Clone, Copy)]
pub struct Ramdisk {
    pub ptr: *mut u8,
    pub len: usize,
}

#[derive(Debug)]
pub enum RamdiskError {
    OutOfBounds,
}

impl core::fmt::Display for RamdiskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfBounds => write!(f, "r/w out of bounds"),
        }
    }
}

impl core::error::Error for RamdiskError {}
