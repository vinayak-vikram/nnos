#!/bin/sh
# need to force linux boot protocol foor QEMU to pass the DTB
set -e
OBJCOPY="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-objcopy"
"$OBJCOPY" -O binary "$1" "$1.bin"
exec qemu-system-aarch64 -M virt -cpu cortex-a53 -m 256M -display cocoa -serial vc \
    -initrd initrd.img -kernel "$1.bin"
