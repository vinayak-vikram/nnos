use super::executor::wake_task;
use super::task::TaskInner;
use crate::helpers::Mutex;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::task::{Context, RawWaker, RawWakerVTable, Waker};

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

pub struct GlobalWaker(Mutex<Option<Waker>>);
impl GlobalWaker {
    pub const fn new() -> GlobalWaker {
        GlobalWaker(Mutex::new(None))
    }
    pub fn arm(&self, cx: &mut Context<'_>) {
        self.0.with(|w| *w = Some(cx.waker().clone()));
    }
    pub fn wake(&self) {
        let waker = self.0.with(|w| w.take());
        if let Some(w) = waker {
            w.wake();
        }
    }
}
