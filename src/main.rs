#![no_main]
#![no_std]

use core::panic::PanicInfo;
use cortex_m_semihosting::{debug, hprintln};

#[rtic::app(device = lm3s6965)]
mod app {
    use super::*;

    #[shared]
    struct Shared {}

    #[local]
    struct Local {}

    #[init]
    fn init(cx: init::Context) -> (Shared, Local) {
        hprintln!("hello gee");
        debug::exit(debug::EXIT_SUCCESS);
        (Shared {}, Local {})
    }
}

#[panic_handler]
fn panic_handler(_: &PanicInfo) -> ! {
    debug::exit(debug::EXIT_FAILURE);
    loop {}
}
