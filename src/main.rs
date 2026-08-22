#![no_std]
#![no_main]

extern crate alloc;

mod driver;
mod helpers;
mod interrupts;

use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;

use driver::gic;
use driver::serial::{Serial, UART_GIC};

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[unsafe(no_mangle)]
pub extern "C" fn main() -> ! {
    unsafe {
        embedded_alloc::init!(HEAP, 1024);
    }
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
