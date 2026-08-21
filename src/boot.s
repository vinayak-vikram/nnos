.global _start
_start:
    // Enable FP and SIMD
    mrs x0, cpacr_el1
    orr x0, x0, #(0x3 << 20)
    msr cpacr_el1, x0
    isb

    // use SP_EL1, so exceptions taken at EL1 land in the "current EL, SPx" vectors
    msr spsel, #1

    // set up stack
    // stack_top is from link.x
    adrp x0, stack_top
    add x0, x0, :lo12:stack_top
    mov sp, x0

    // point VBAR_EL1 at exceptin vt
    adrp x0, vector_table
    add x0, x0, :lo12:vector_table
    msr vbar_el1, x0
    isb

    bl main
1:
    // spin forever in case main returns
    wfe
    b 1b
