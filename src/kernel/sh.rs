use crate::driver::serial::Serial;

#[derive(Clone, Copy)]
pub struct CommandBuffer {
    pub buf: [u8; 256],
    pub len: usize,
}

pub async fn shell_task(cmd: CommandBuffer, console: Serial) {
    console.print("\r\ngot: ");
    console.printb(&cmd.buf, cmd.len);
    console.print("\r\n");
}
