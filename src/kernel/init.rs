use alloc::boxed::Box;
use ext4plus::Ext4;

use crate::asyncrt::spawn;
use crate::driver::serial::CONSOLE;
use crate::fs::dtb;
use crate::helpers::stdio::{print, println};
use crate::shell::{BashProfile, NNProfile, ShellProfile, shell_task};

pub async fn init_task(dtb_ptr: *const u8) {
    println("async executor initialized");
    let Some(dtbt) = (unsafe { dtb::get_dtb(dtb_ptr) }) else {
        println("dtb loading failed");
        panic!();
    };
    let rd = dtb::locate_initrd(dtbt).expect("initrd not found");
    print("initrd located at 0x");
    CONSOLE.printh(rd.ptr as u64);
    print(" (0x");
    CONSOLE.printh(rd.len as u64);
    println(" bytes)");
    let Ok(fs) = Ext4::load_with_writer(Box::new(rd), Some(Box::new(rd))).await else {
        println("fs loading failed");
        panic!();
    };
    println("successfully loaoded ext4 filesystem");
    println("https://github.com/vinayak-vikram/nnos 0.0.1 kernel initialization complete\r\n");
    let mut nn = NNProfile;
    if nn.init(&fs).await.is_ok() {
        spawn(shell_task(nn, fs));
    } else {
        println("nn shell profile failed to load, falling back to bash");
        let mut bash = BashProfile;
        let _ = bash.init(&fs).await;
        spawn(shell_task(bash, fs));
    }
}
