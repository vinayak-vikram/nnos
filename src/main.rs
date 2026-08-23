#![no_std]
#![no_main]

extern crate alloc;

mod asyncrt;
mod driver;
mod helpers;
mod interrupts;
mod kernel;

use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;

use asyncrt::{Executor, spawn};
use driver::gic;
use driver::serial::{Serial, UART_GIC, serial_task};

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
    console.print("booted\r\n");
    console.print("interrupt vector table initialized\r\n");
    console.print("serial console initialized\r\n");
    console.print("heap allocator initalized\r\n");
    let mut rt = Executor::new();
    console.print("async executor initialized\r\n");
    console.print("scheduling tasks...\r\n");
    console
        .print("https://github.com/vinayak-vikram/nnos 0.0.1 kernel initialization complete\r\n");
    spawn(serial_task(console)); // we dont need it outside of this task, if i do everything properly
    rt.run();
}
