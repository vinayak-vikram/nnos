use super::ramdisk::{Ramdisk, RamdiskError};
use alloc::boxed::Box;
use async_trait::async_trait;
use ext4plus::{Ext4Read, Ext4Write};

unsafe impl Sync for Ramdisk {}

#[async_trait(?Send)]
impl Ext4Read for Ramdisk {
    async fn read(
        &self,
        start_byte: u64,
        dst: &mut [u8],
    ) -> Result<(), Box<dyn core::error::Error + Send + Sync + 'static>> {
        let start = start_byte as usize;
        let end = start + dst.len();
        if end > self.len {
            return Err(Box::new(RamdiskError::OutOfBounds));
        }
        unsafe {
            core::ptr::copy_nonoverlapping(self.ptr.add(start), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl Ext4Write for Ramdisk {
    async fn write(
        &self,
        start_byte: u64,
        src: &[u8],
    ) -> Result<(), Box<dyn core::error::Error + Send + Sync + 'static>> {
        let start = start_byte as usize;
        let end = start + src.len();
        if end > self.len {
            return Err(Box::new(RamdiskError::OutOfBounds));
        }
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(start), src.len());
        }
        Ok(())
    }
}
