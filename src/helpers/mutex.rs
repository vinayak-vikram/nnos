use core::cell::UnsafeCell;

/// A sane critical section mutex that allows for mutable access to the inner value.
///
/// # Safety
/// Only usable in a single-core environment (i.e. RTIC)
/// Multiple simultaneous accesses across different cores/enviorments would cause undefined behavior.
pub struct Mutex<T> {
    inner: UnsafeCell<T>,
}

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            inner: UnsafeCell::new(value),
        }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        critical_section::with(|_cs| f(unsafe { &mut *self.inner.get() }))
    }
}

unsafe impl<T> Send for Mutex<T> {}
unsafe impl<T> Sync for Mutex<T> {}
