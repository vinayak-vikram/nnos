MEMORY
{
  /* kernel image is loaded at loader_start + 0x80000 */
  RAM : ORIGIN = 0x40080000, LENGTH = 256M - 512K
}

ENTRY(_start);

SECTIONS
{
  . = ORIGIN(RAM);

  .text :
  {
    *(.text._start)
    . = ALIGN(0x800);
    *(.text.vectors)
    *(.text .text.*);
  } > RAM

  .rodata : { *(.rodata .rodata.*); } > RAM
  .data : { *(.data .data.*); } > RAM
  .bss : { *(.bss .bss.*); } > RAM

  . = ALIGN(16);
  . = . + 0x40000; /* stack */
  stack_top = .;
}
