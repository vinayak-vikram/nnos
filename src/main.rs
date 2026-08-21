#![no_std]
#![no_main]

mod driver;

use core::panic::PanicInfo;

use driver::serial::*;

#[unsafe(no_mangle)]
pub extern "C" fn main() -> ! {
    let console = Serial::new(UART0_DR, UART0_FR, UART_FR_TXFF);
    console.print("hello gee");
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
