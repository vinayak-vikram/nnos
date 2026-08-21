//! On Cortex-M chips, the vector table is loaded directly into memory on init,
//! and the entry point is taken fromt here (canonically `Reset`).
//! Unfortunately, not all ARM chips are Cortex-M so we have to do this bullshit.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let boot_o = out_dir.join("boot.o");

    let status = Command::new("clang")
        .args(["-target", "aarch64-unknown-none", "-c", "src/boot.s", "-o"])
        .arg(&boot_o)
        .status()
        .expect("error when assembling bootloadeer");
    assert!(status.success(), "did noto assemble botloader, exiting");

    println!("cargo:rustc-link-arg={}", boot_o.display());
    println!("cargo:rerun-if-changed=src/boot.s");
}
