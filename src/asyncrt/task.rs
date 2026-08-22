use alloc::boxed::Box;
use core::pin::Pin;

pub type Task = Pin<Box<dyn Future<Output = ()>>>;
