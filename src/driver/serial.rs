use crate::asyncrt::{GlobalWaker, spawn};
use crate::helpers::ringbuf::RingBuffer;
use crate::kernel::sh::{CommandBuffer, shell_task};
use core::future::poll_fn;
use core::ptr::{read_volatile, write_volatile};
use core::task::Poll;

/// For transmit
/// Data register
pub const UART0_DR: *mut u32 = 0x0900_0000 as *mut u32;
/// Flag register
pub const UART0_FR: *mut u32 = 0x0900_0018 as *mut u32;
/// Bit that indicates if TX buf is full or not
pub const UART_FR_TXFF: u32 = 1 << 5;
/// For recv
/// Interrupt mask register
const UART0_IMSC: *mut u32 = (0x0900_0038) as *mut u32;
/// Interrupt clear register
const UART0_ICR: *mut u32 = (0x0900_0044) as *mut u32;
/// UART interrupt GIC id
pub const UART_GIC: u8 = 33;

static RX: RingBuffer = RingBuffer::new();
static RX_WAKER: GlobalWaker = GlobalWaker::new();

#[derive(Clone, Copy)]
pub struct Serial {
    dr: *mut u32,
    fr: *const u32,
    fr_txff: u32,
    imscr: *mut u32,
    icr: *mut u32,
    rx: &'static RingBuffer,
}

impl Serial {
    pub fn new() -> Self {
        unsafe {
            let current_mask = read_volatile(UART0_IMSC);
            write_volatile(UART0_IMSC, current_mask | (1 << 4));
        }
        Self {
            dr: UART0_DR,
            fr: UART0_FR,
            fr_txff: UART_FR_TXFF,
            imscr: UART0_IMSC,
            icr: UART0_ICR,
            rx: &RX,
        }
    }
    pub fn wb(&self, b: u8) {
        unsafe {
            while read_volatile(self.fr) & self.fr_txff != 0 {}
            write_volatile(self.dr, b as u32);
        }
    }
    pub fn print(&self, s: &str) {
        for b in s.bytes() {
            self.wb(b);
        }
    }
    pub fn printb(&self, s: &[u8; 256], len: usize) {
        for &b in &s[..len] {
            self.wb(b);
        }
    }
    pub fn rb(&self) -> Option<u8> {
        self.rx.pop()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn handle_uart_irq() {
    unsafe {
        let received_byte = (read_volatile(UART0_DR) & 0xFF) as u8;
        RX.push(received_byte);
        write_volatile(UART0_ICR, 1 << 4);
    }
    RX_WAKER.wake();
}

async fn poll_rb(serial: &Serial) -> u8 {
    poll_fn(|cx| {
        RX_WAKER.arm(cx);
        match serial.rb() {
            Some(b) => Poll::Ready(b),
            None => Poll::Pending,
        }
    })
    .await
}

pub async fn serial_task(serial: Serial) {
    let mut cmd = CommandBuffer {
        buf: [0; 256],
        len: 0,
    };
    loop {
        let b = poll_rb(&serial).await;
        serial.wb(b);
        match b {
            b'\r' | b'\n' => {
                spawn(shell_task(cmd, serial));
                cmd.len = 0;
            }
            _ if cmd.len < cmd.buf.len() => {
                cmd.buf[cmd.len] = b;
                cmd.len += 1;
            }
            _ => {}
        }
    }
}
