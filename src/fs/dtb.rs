use fdt::Fdt;

const FDT_MAGIC: u32 = 0xD00D_FEED;

pub unsafe fn get_dtb(raw_ptr: *const u8) -> Option<&'static [u8]> {
    let header = raw_ptr as *const u32;
    if u32::from_be(header.read_volatile()) != FDT_MAGIC {
        return None;
    }
    let len = u32::from_be(header.add(1).read_volatile()) as usize;
    Some(core::slice::from_raw_parts(raw_ptr, len))
}

pub struct Ramdisk {
    pub start: usize,
    pub end: usize,
}

pub fn locate_initrd(dtb_data: &[u8]) -> Option<Ramdisk> {
    let fdt = Fdt::new(dtb_data).ok()?;
    let chosen_node = fdt.find_node("/chosen")?;

    let initrd_start = chosen_node.property("linux,initrd-start")?.as_usize()?;
    let initrd_end = chosen_node.property("linux,initrd-end")?.as_usize()?;

    Some(Ramdisk {
        start: initrd_start,
        end: initrd_end,
    })
}
