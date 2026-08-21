#![no_std]
#![no_main]

mod driver;
mod helpers;
mod interrupts;

use core::panic::PanicInfo;

use driver::gic;
use driver::serial::{Serial, UART_GIC};

#[unsafe(no_mangle)]
pub extern "C" fn main() -> ! {
    let console = Serial::new();
    unsafe {
        gic::init();
        gic::enable_interrupt(UART_GIC);
        interrupts::unmask_irqs();
    }
    console.print("hello gee\n");
    loop {
        if let Some(b) = console.rb() {
            console.wb(b);
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
