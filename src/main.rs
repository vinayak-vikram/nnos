#![no_std]
#![no_main]

extern crate alloc;

mod asyncrt;
mod driver;
mod fs;
mod helpers;
mod interrupts;
mod kernel;

use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;

use asyncrt::{Executor, spawn};
use driver::gic;
use driver::serial::{Serial, UART_GIC};

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[unsafe(no_mangle)]
pub extern "C" fn main(dtb_ptr: *const u8) -> ! {
    unsafe {
        embedded_alloc::init!(HEAP, 4096);
    }
    let console = Serial::new();
    unsafe {
        gic::init();
        gic::enable_interrupt(UART_GIC);
        interrupts::unmask_irqs();
    }
    console.print("booted\r\n");
    console.print("interrupt vector table initialized\r\n");
    console.print("heap allocator initalized\r\n");
    console.print("serial console initialized\r\n");
    let mut rt = Executor::new();
    spawn(kernel::init_task(console, dtb_ptr));
    rt.run();
}
