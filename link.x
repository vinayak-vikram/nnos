MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM   : ORIGIN = 0x20000000, LENGTH = 128K /* Belw this is the vector table */
}

ENTRY(Reset);

SECTIONS
{
  .vector_table :
  {
    LONG(ORIGIN(RAM) + LENGTH(RAM)); /* SP */
    LONG(Reset); /* Reset */
  } > FLASH

  .text :
  {
    *(.text .text.*);
  } > FLASH
}
