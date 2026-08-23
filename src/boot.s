.section .text._start, "ax"
.global _start
_start:
    // stash DTB address that QEMU gives us
    mov x19, x0

    // Enable FP and SIMD
    mrs x1, cpacr_el1
    orr x1, x1, #(0x3 << 20)
    msr cpacr_el1, x1
    isb

    // use SP_EL1, so exceptions taken at EL1 land in the "current EL, SPx" vectors
    msr spsel, #1

    // set up stack
    // stack_top is from link.x
    adrp x1, stack_top
    add x1, x1, :lo12:stack_top
    mov sp, x1

    // point VBAR_EL1 at exceptin vt
    adrp x1, vector_table
    add x1, x1, :lo12:vector_table
    msr vbar_el1, x1
    isb

    // main(dtb: *const u8)
    mov x0, x19
    bl main
1:
    // spin forever in case main returns
    wfe
    b 1b
