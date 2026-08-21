use core::ptr::{read_volatile, write_volatile};

pub const UART0_DR: *mut u32 = 0x0900_0000 as *mut u32;
pub const UART0_FR: *mut u32 = 0x0900_0018 as *mut u32;
pub const UART_FR_TXFF: u32 = 1 << 5;

pub struct Serial {
    dr: *mut u32,
    fr: *const u32,
    fr_txff: u32,
}

impl Serial {
    pub fn new(dr: *mut u32, fr: *const u32, fr_txff: u32) -> Self {
        Self { dr, fr, fr_txff }
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
}
