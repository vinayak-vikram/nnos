#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod decode;
pub mod format;
pub mod intent;
pub mod kernels;
pub mod model;
pub mod shell;
pub mod tensor;

pub use decode::{Candidate, MAX_GEN};
pub use format::LoadError;
pub use intent::Intent;
pub use model::{GenError, Model};
pub use shell::{Outcome, Shell};
