use alloc::boxed::Box;
use ext4plus::Ext4;

use crate::asyncrt::spawn;
use crate::driver::serial::{Serial, serial_task};
use crate::fs::dtb;

pub async fn init_task(console: Serial, dtb_ptr: *const u8) {
    console.print("async executor initialized\r\n");
    let Some(dtbt) = (unsafe { dtb::get_dtb(dtb_ptr) }) else {
        console.print("dtb loading failed\r\n");
        panic!();
    };
    let rd = dtb::locate_initrd(dtbt).expect("initrd not found\r\n");
    console.print("initrd located at 0x");
    console.printh(rd.ptr as u64);
    console.print(" (0x");
    console.printh(rd.len as u64);
    console.print(" bytes)\r\n");
    // TODO: why tf is the executor torquing itself on this future even when not in wfi mode...
    // let Ok(fs) = Ext4::load(Box::new(rd)).await else {
    //     console.print("fs loading failed\r\n");
    //     panic!();
    // };
    console.print("successfully loaoded ext4 filesystem\r\n");
    console
        .print("https://github.com/vinayak-vikram/nnos 0.0.1 kernel initialization complete\r\n");
    spawn(serial_task(console)); // we dont need it outside of this task, if i do everything properly
}
