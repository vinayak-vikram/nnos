use super::executor::wake_task;
use super::task::TaskInner;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::task::{Context, RawWaker, RawWakerVTable, Waker};
use critical_section::Mutex;

static VTABLE: RawWakerVTable = RawWakerVTable::new(
    // Clone
    |p| unsafe {
        let rc = Rc::from_raw(p as *const RefCell<TaskInner>);
        let cloned = rc.clone();
        core::mem::forget(rc);
        RawWaker::new(Rc::into_raw(cloned) as *const (), &VTABLE)
    },
    // Wake
    |p| unsafe { wake_task(Rc::from_raw(p as *const RefCell<TaskInner>)) },
    // Wake by reference
    |p| unsafe {
        let rc = Rc::from_raw(p as *const RefCell<TaskInner>);
        let cloned = rc.clone();
        core::mem::forget(rc);
        wake_task(cloned);
    },
    // Drop
    |p| unsafe {
        let _ = Rc::from_raw(p as *const RefCell<TaskInner>);
    },
);

pub fn task_waker(task: Rc<RefCell<TaskInner>>) -> Waker {
    let raw = Rc::into_raw(task) as *const ();
    unsafe { Waker::from_raw(RawWaker::new(raw, &VTABLE)) }
}

pub struct GlobalWaker(Mutex<RefCell<Option<Waker>>>);
impl GlobalWaker {
    pub const fn new() -> GlobalWaker {
        GlobalWaker(Mutex::new(RefCell::new(None)))
    }
    pub fn arm(&self, cx: &mut Context<'_>) {
        critical_section::with(|cs| *self.0.borrow(cs).borrow_mut() = Some(cx.waker().clone()));
    }
    pub fn wake(&self) {
        let waker = critical_section::with(|cs| self.0.borrow(cs).borrow_mut().take());
        if let Some(w) = waker {
            w.wake();
        }
    }
}
