use alloc::boxed::Box;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::pin::Pin;

pub type TaskInner = Pin<Box<dyn Future<Output = ()>>>;
pub struct Task(pub Rc<RefCell<TaskInner>>);
impl Task {
    pub fn new(t: TaskInner) -> Task {
        Task(Rc::new(RefCell::new(t)))
    }
}
unsafe impl Send for Task {}
