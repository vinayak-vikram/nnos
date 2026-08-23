use super::Task;
use super::task::TaskInner;
use super::waker;
use crate::helpers::Mutex;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::task::Context;

static TASKS: Mutex<VecDeque<Task>> = Mutex::new(VecDeque::new());

pub(crate) fn wake_task(task: Rc<RefCell<TaskInner>>) {
    TASKS.with(|tasks| tasks.push_back(Task(task)));
}

pub struct Executor {}

impl Executor {
    pub fn new() -> Executor {
        Executor {}
    }
    /// Spawn a future onto the executoor instance.
    pub fn spawn<F: Future<Output = ()> + 'static>(&mut self, future: F) {
        TASKS.with(|tasks| tasks.push_back(Task::new(Box::pin(future))));
    }
    /// Run the async executor
    ///
    /// A task is only polled when its in the ready queue.
    /// It is added to the ready queue first on spawn
    /// and whenever woken by its deisgnated waker.
    /// Tasks are popped out of the queue instantly on poll.
    pub fn run(&mut self) -> ! {
        loop {
            while let Some(task) = TASKS.with(|tasks| tasks.pop_front()) {
                let w = waker::task_waker(task.0.clone());
                let mut cx = Context::from_waker(&w);
                let _ = task.0.borrow_mut().as_mut().poll(&mut cx);
            }
            // TODO: figure out tasks that dont have interrupt on ready to poll... idk
            TASKS.with(|tasks| {
                if tasks.is_empty() {
                    unsafe { core::arch::asm!("wfi", options(nomem, nostack)) };
                }
            });
        }
    }
}
