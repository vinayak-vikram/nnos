#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// Instead of a standard boootloader, ARM Cortex-M chips load the vector table into memory
/// and search for the entry point
/// By cnoventiono, this entry poinot is called `Reset`
/// (note difference fromo _start on typical x86_64 targets)
#[unsafe(no_mangle)]
pub extern "C" fn Reset() -> ! {
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
