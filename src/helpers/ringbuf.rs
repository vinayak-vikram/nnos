use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

pub struct RingBuffer {
    buf: UnsafeCell<[u8; 256]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    pub const fn new() -> Self {
        Self {
            buf: UnsafeCell::new([0; 256]),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, b: u8) {
        let head = self.head.load(Ordering::Relaxed);
        let next = (head + 1) % 256;
        if next == self.tail.load(Ordering::Acquire) {
            return;
        }
        unsafe {
            (*self.buf.get())[head] = b;
        }
        self.head.store(next, Ordering::Release);
    }

    pub fn pop(&self) -> Option<u8> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let b = unsafe { (*self.buf.get())[tail] };
        self.tail.store((tail + 1) % 256, Ordering::Release);
        Some(b)
    }
}
