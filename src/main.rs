#![no_std]
#![no_main]

extern crate alloc;

mod asyncrt;
mod driver;
mod helpers;
mod interrupts;

use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;

use asyncrt::Executor;
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
    console.print("booted\n");
    console.print("interrupt vector table initialized\n");
    console.print("serial console initialized\n");
    console.print("heap allocator initalized\n");
    let mut rt = Executor::new();
    console.print("async executor initialized\n");
    console.print("scheduling tasks...\n");
    console.print("https://github.com/vinayak-vikram/nnos 0.0.1 kernel initialization complete\n");
    rt.spawn(serial_task(console)); // we dont need it outside of this task, if i do everything properly
    rt.run();
}
