use ext4plus::Ext4;

use super::{CommandBuffer, ShellProfile};
use crate::kernel::syscall::Intent;

pub struct NNProfile;

impl ShellProfile for NNProfile {
    async fn init(&mut self, _fs: &Ext4) -> Result<(), ()> {
        Err(())
    }

    async fn infer(&self, _cmd: CommandBuffer, _fs: &Ext4) -> Option<Intent> {
        todo!()
    }
}
