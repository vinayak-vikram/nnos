use alloc::vec::Vec;

use crate::driver::serial::{CONSOLE, poll_rb};
use crate::shell::CommandBuffer;

#[inline(always)]
pub fn print(s: &str) {
    CONSOLE.print(s);
}

#[inline(always)]
pub fn println(s: &str) {
    print(s);
    print("\r\n");
}

#[inline(always)]
pub fn printb(cmd: &CommandBuffer) {
    CONSOLE.printb(&cmd.buf, cmd.len);
}

#[inline(always)]
pub fn printv(v: &Vec<u8>) {
    CONSOLE.printv(v);
}

pub async fn readln() -> CommandBuffer {
    let mut cmd = CommandBuffer {
        buf: [0; 256],
        len: 0,
    };
    loop {
        let b = poll_rb().await;
        CONSOLE.wb(b);
        match b {
            b'\r' | b'\n' => {
                return cmd;
            }
            _ if cmd.len < cmd.buf.len() => {
                cmd.buf[cmd.len] = b;
                cmd.len += 1;
            }
            _ => {
                println("serial buffer overflowed");
                return cmd;
            }
        }
    }
}
