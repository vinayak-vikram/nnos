MEMORY
{
  RAM : ORIGIN = 0x40000000, LENGTH = 128M
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
  . = . + 0x4000; /* stack */
  stack_top = .;
}
