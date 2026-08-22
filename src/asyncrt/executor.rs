use super::Task;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use core::task::Context;
use futures::task;

pub struct Executor {
    tasks: VecDeque<Task>,
}

impl Executor {
    pub fn new() -> Executor {
        Executor {
            tasks: VecDeque::new(),
        }
    }
    /// Spawn a future onto the executoor instance.
    pub fn spawn<F: Future<Output = ()> + 'static>(&mut self, future: F) {
        self.tasks.push_back(Box::pin(future));
    }
    /// Run the async executor
    ///
    /// Currently just looping and polling bc gee
    pub fn run(&mut self) {
        let waker = task::noop_waker();
        let mut cx = Context::from_waker(&waker);

        while let Some(mut task) = self.tasks.pop_front() {
            if task.as_mut().poll(&mut cx).is_pending() {
                self.tasks.push_back(task);
            }
        }
    }
}
