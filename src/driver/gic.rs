use core::ptr::{read_volatile, write_volatile};

const GICD_BASE: usize = 0x0800_0000;
const GICD_CTLR: *mut u32 = GICD_BASE as *mut u32;
/// Enable register for SPIs (Serial Peripheral Interrupts, not like that spi) #32-63
const GICD_ISENABLER1: *mut u32 = (GICD_BASE + 0x104) as *mut u32;

const GICC_BASE: usize = 0x0801_0000;
const GICC_CTLR: *mut u32 = GICC_BASE as *mut u32;
const GICC_PMR: *mut u32 = (GICC_BASE + 0x04) as *mut u32;
const GICC_IAR: *const u32 = (GICC_BASE + 0x0C) as *const u32;
const GICC_EOIR: *mut u32 = (GICC_BASE + 0x10) as *mut u32;

/// ID returned by ack() when no pending interrupt
const SPURIOUS: u32 = 1023;

pub unsafe fn init() {
    unsafe {
        write_volatile(GICD_CTLR, 1);
        write_volatile(GICC_PMR, 0xFF);
        write_volatile(GICC_CTLR, 1);
    }
}

// TODO: extend
pub unsafe fn enable_interrupt(n: u8) {
    unsafe {
        write_volatile(GICD_ISENABLER1, 1 << (n - 32));
    }
}

/// Acknowledge the highest priority pending interrupt
/// and rretun its ID
pub unsafe fn ack() -> u32 {
    unsafe { read_volatile(GICC_IAR) & 0x3FF }
}

pub unsafe fn eoi(id: u32) {
    if id != SPURIOUS {
        unsafe {
            write_volatile(GICC_EOIR, id);
        }
    }
}
