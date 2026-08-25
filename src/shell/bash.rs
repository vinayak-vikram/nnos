use alloc::format;
use ext4plus::Ext4;

use super::{CommandBuffer, ShellProfile};
use crate::kernel::syscall::{Intent, Syscall};

pub struct BashProfile;

impl ShellProfile for BashProfile {
    async fn init(&mut self, _fs: &Ext4) -> Result<(), ()> {
        Ok(())
    }

    async fn infer(&self, cmd: CommandBuffer, _fs: &Ext4) -> Option<Intent> {
        let s = core::str::from_utf8(&cmd.buf[..cmd.len]).ok()?;
        Some(Intent {
            sc: Syscall::Print {
                message: format!("\r\ngot: {}", s),
            },
            confidence: 1.0,
        })
    }
}
