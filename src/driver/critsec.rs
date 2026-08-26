use critical_section::RawRestoreState;

struct AArch64CriticalSection;
critical_section::set_impl!(AArch64CriticalSection);

unsafe impl critical_section::Impl for AArch64CriticalSection {
    unsafe fn acquire() -> RawRestoreState {
        let original_daif: u64;

        // setting bits [7:6] disables interrupts
        core::arch::asm!(
            "mrs {0}, DAIF",
            "msr DAIFSet, #3",
            out(reg) original_daif,
            options(nostack) // nomem torques this for some reason
        );

        original_daif
    }

    unsafe fn release(restore_state: RawRestoreState) {
        // Restore the processor flags back to what they were before acquiring
        core::arch::asm!(
            "msr DAIF, {0}",
            in(reg) restore_state,
            options(nostack)
        );
    }
}
