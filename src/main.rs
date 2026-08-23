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
use driver::serial::{Serial, UART_GIC, serial_task};
use fs::dtb;

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[unsafe(no_mangle)]
pub extern "C" fn main(dtb_ptr: *const u8) -> ! {
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
    console.print("heap allocator initalized\r\n");
    console.print("serial console initialized\r\n");
    let mut rt = Executor::new();
    console.print("async executor initialized\r\n");
    let Some(dtbt) = (unsafe { dtb::get_dtb(dtb_ptr) }) else {
        console.print("dtb loading failed\r\n");
        panic!();
    };
    let rd = dtb::locate_initrd(dtbt).expect("initrd not found\r\n");
    let img: &'static [u8] =
        unsafe { core::slice::from_raw_parts(rd.start as *const u8, rd.end - rd.start) };
    console.print("initrd located at 0x");
    console.printh(rd.start as u64);
    console.print("..0x");
    console.printh(rd.end as u64);
    console.print(" (0x");
    console.printh(img.len() as u64);
    console.print(" bytes)\r\n");
    console
        .print("https://github.com/vinayak-vikram/nnos 0.0.1 kernel initialization complete\r\n");
    spawn(serial_task(console)); // we dont need it outside of this task, if i do everything properly
    rt.run();
}
