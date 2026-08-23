use crate::helpers::stdio::*;

#[derive(Clone, Copy)]
pub struct CommandBuffer {
    pub buf: [u8; 256],
    pub len: usize,
}

pub async fn shell_task() {
    loop {
        print("> ");
        let cmd = readln().await;
        print("\r\ngot: ");
        printb(&cmd);
    }
}
