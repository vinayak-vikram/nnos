//! On Cortex-M chips, the vector table is loaded directly into memory on init,
//! and the entry point is taken fromt here (canonically `Reset`).
//! Unfortunately, not all ARM chips are Cortex-M so we have to do this bullshit.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    for src in ["src/boot.s", "src/exceptions.s"] {
        let obj = out_dir.join(PathBuf::from(src).with_extension("o").file_name().unwrap());

        let status = Command::new("clang")
            .args(["-target", "aarch64-unknown-none", "-c", src, "-o"])
            .arg(&obj)
            .status()
            .expect("error when assembling");
        assert!(status.success(), "did noto assemble successfully, exiting");

        println!("cargo:rustc-link-arg={}", obj.display());
        println!("cargo:rerun-if-changed={}", src);
    }
}
