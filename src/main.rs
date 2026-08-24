#![no_std]
#![no_main]

extern crate alloc;

mod asyncrt;
mod driver;
mod fs;
mod helpers;
mod interrupts;
mod kernel;
mod shell;

use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;

use asyncrt::{Executor, spawn};
use driver::gic;
use driver::serial::{CONSOLE, UART_GIC};
use helpers::stdio::println;

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[unsafe(no_mangle)]
pub extern "C" fn main(dtb_ptr: *const u8) -> ! {
    unsafe {
        embedded_alloc::init!(HEAP, 1024 * 1024);
    }
    CONSOLE.enable_rx_irq();
    unsafe {
        gic::init();
        gic::enable_interrupt(UART_GIC);
        interrupts::unmask_irqs();
    }
    println("booted");
    println("interrupt vector table initialized");
    println("heap allocator initalized");
    println("serial console initialized");
    let mut rt = Executor::new();
    spawn(kernel::init_task(dtb_ptr));
    rt.run();
}
