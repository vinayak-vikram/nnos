use crate::driver::{gic, serial};
use core::arch::asm;

/// Called from the IRQ vector
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irq_handler() {
    unsafe {
        let id = gic::ack();
        if id == serial::UART_GIC as u32 {
            serial::handle_uart_irq();
        }
        gic::eoi(id);
    }
}

/// This unmasks IRQs on the CPU itself
/// allowing it to lanch the routinse
pub unsafe fn unmask_irqs() {
    asm!("msr daifclr, #2");
}
