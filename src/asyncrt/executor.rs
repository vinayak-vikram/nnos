use super::Task;
use super::task::TaskInner;
use super::waker;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::task::Context;
use critical_section::Mutex;

static TASKS: Mutex<RefCell<VecDeque<Task>>> = Mutex::new(RefCell::new(VecDeque::new()));

pub(crate) fn wake_task(task: Rc<RefCell<TaskInner>>) {
    critical_section::with(|cs| {
        TASKS.borrow(cs).borrow_mut().push_back(Task(task));
    });
}

pub struct Executor {}

impl Executor {
    pub fn new() -> Executor {
        Executor {}
    }
    /// Spawn a future onto the executoor instance.
    pub fn spawn<F: Future<Output = ()> + 'static>(&mut self, future: F) {
        critical_section::with(|cs| {
            TASKS
                .borrow(cs)
                .borrow_mut()
                .push_back(Task::new(Box::pin(future)));
        });
    }
    /// Run the async executor
    ///
    /// A task is only polled when its in the ready queue.
    /// It is added to the ready queue first on spawn
    /// and whenever woken by its deisgnated waker.
    /// Tasks are popped out of the queue instantly on poll.
    pub fn run(&mut self) -> ! {
        loop {
            while let Some(task) =
                critical_section::with(|cs| TASKS.borrow(cs).borrow_mut().pop_front())
            {
                let w = waker::task_waker(task.0.clone());
                let mut cx = Context::from_waker(&w);
                let _ = task.0.borrow_mut().as_mut().poll(&mut cx);
            }
        }
    }
}
