pub mod bash;
pub mod nn;

use ext4plus::Ext4;

use crate::helpers::stdio::*;
use crate::kernel::syscall::{Intent, exec_syscall};

pub use bash::BashProfile;
pub use nn::NNProfile;

#[derive(Clone, Copy)]
pub struct CommandBuffer {
    pub buf: [u8; 256],
    pub len: usize,
}

pub async fn shell_task(mut profile: impl ShellProfile, fs: Ext4) {
    loop {
        print("> ");
        let cmd = readln().await;
        let Some(intent) = profile.infer(cmd, &fs).await else {
            println("intent not discernible, skipping");
            continue;
        };
        match intent.process() {
            Ok(sc) => {
                if exec_syscall(sc, &fs).await.is_err() {
                    println("syscall failed");
                }
            }
            Err(_) => println("confidence below threshold"),
        }
    }
}

pub trait ShellProfile {
    async fn init(&mut self, fs: &Ext4) -> Result<(), ()>;
    async fn infer(&mut self, cmd: CommandBuffer, fs: &Ext4) -> Option<Intent>;
}
