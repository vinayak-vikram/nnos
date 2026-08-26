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

    bl enable_mmu

    // main(dtb: *const u8)
    mov x0, x19
    bl main
1:
    // spin forever in case main returns
    wfe
    b 1b

// Need the MMU for atomic RMWs.
enable_mmu:
    adrp x0, pgtbl
    add x0, x0, :lo12:pgtbl

    mov x1, #0
    mov x2, #512
1:
    str xzr, [x0, x1, lsl #3]
    add x1, x1, #1
    cmp x1, x2
    b.lo 1b

    movz x1, #0x0401
    movk x1, #0x0060, lsl #48
    str x1, [x0, #0]

    movz x1, #0x0705
    movk x1, #0x4000, lsl #16
    str x1, [x0, #8]

    movz x1, #0xff00
    msr mair_el1, x1

    movz x1, #0x3519
    movk x1, #0x0080, lsl #16
    mrs x2, id_aa64mmfr0_el1
    and x2, x2, #0xf
    orr x1, x1, x2, lsl #32
    msr tcr_el1, x1

    msr ttbr0_el1, x0
    isb

    dsb ishst
    tlbi vmalle1
    dsb ish
    isb

    mrs x1, sctlr_el1
    orr x1, x1, #(1 << 0)
    orr x1, x1, #(1 << 2)
    orr x1, x1, #(1 << 12)
    msr sctlr_el1, x1
    isb

    ret

.section .data.pgtbl, "aw"
.align 12
pgtbl:
    .space 4096
