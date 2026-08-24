use alloc::vec::Vec;
use core::arch::asm;
use core::ptr::read_volatile;
use core::time::Duration;
use ext4plus::prelude::*;

use crate::helpers::stdio::{println, printv};

const RTC_DR: *const u32 = 0x0901_0000 as *const u32; //seconds since epoch
const RS_REG: u64 = 0x8400_0009;

pub enum Syscall<'a> {
    Print { message: &'a str },
    Read { path: &'a str },
    // Write { path: &'a str, data: Vec<u8> }, // TODO: diffs? :)
    Create { path: &'a str },
    Delete { path: &'a str },
    List { path: &'a str },
    Time,
    Reboot,
}

pub struct Intent<'a> {
    pub sc: Syscall<'a>,
    pub confidence: f64,
}

impl<'a> Intent<'a> {
    pub fn process(self) -> Result<Syscall<'a>, f64> {
        // TODO: figure out exact feasible numbers later ig
        let th: f64 = match self.sc {
            Syscall::Print { .. } => 0.5,
            Syscall::Read { .. } => 0.6,
            // Syscall::Write { .. } => 0.8,
            Syscall::Create { .. } => 0.7,
            Syscall::Delete { .. } => 0.95,
            Syscall::List { .. } => 0.6,
            Syscall::Time => 0.5,
            Syscall::Reboot => 0.97,
        };
        if self.confidence > th {
            Ok(self.sc)
        } else {
            Err(th)
        }
    }
}

pub async fn exec_syscall(sc: Syscall<'_>, fs: &Ext4) -> Result<(), Ext4Error> {
    match sc {
        Syscall::Print { message } => {
            println(message);
            Ok(())
        }
        Syscall::Read { path } => {
            printv(&fs.read(path).await?);
            Ok(())
        }
        // Syscall::Write { path, data } => (),
        Syscall::Create { path } => touch(fs, path).await,
        Syscall::Delete { path } => delete(fs, path).await,
        Syscall::List { path } => listdir(fs, path).await,
        Syscall::Time => {
            get_curr_time();
            Ok(())
        }
        Syscall::Reboot => {
            reboot();
            Ok(())
        }
    }
}

fn get_path(path: &str) -> Result<DirEntryName<'_>, Ext4Error> {
    DirEntryName::try_from(path.trim_start_matches('/')).map_err(|_| Ext4Error::MalformedPath)
}

async fn touch(fs: &Ext4, path: &str) -> Result<(), Ext4Error> {
    let name = get_path(path)?;
    let mut dir = Dir::open_inode(fs, fs.read_root_inode().await?)?;
    let mut inode = fs
        .create_inode(InodeCreationOptions {
            file_type: FileType::Regular,
            mode: InodeMode::S_IFREG | InodeMode::S_IRUSR | InodeMode::S_IWUSR,
            uid: 0,
            gid: 0,
            time: get_curr_time(),
            flags: InodeFlags::empty(),
        })
        .await?;
    dir.link(name, &mut inode).await
}

async fn delete(fs: &Ext4, path: &str) -> Result<(), Ext4Error> {
    let name = get_path(path)?;
    let mut dir = Dir::open_inode(fs, fs.read_root_inode().await?)?;
    let inode = dir.get_entry(name).await?;
    dir.unlink(name, inode).await?;
    Ok(())
}

async fn listdir(fs: &Ext4, path: &str) -> Result<(), Ext4Error> {
    let mut entries = fs.read_dir(path).await?;
    while let Some(entry) = entries.next().await {
        let entry = entry?;
        match entry.file_name().as_str() {
            Ok(name) => println(name),
            Err(_) => println("<error: name not utf8-encodable>"),
        }
    }
    Ok(())
}

pub fn get_curr_time() -> Duration {
    Duration::from_secs(u64::from(unsafe { read_volatile(RTC_DR) }))
}

pub fn reboot() {
    unsafe {
        asm!("msr daifset, #2", options(nomem, nostack));
        asm!(
            "hvc #0",
            inout("x0") RS_REG => _,
            out("x1") _,
            out("x2") _,
            out("x3") _,
            options(nomem, nostack),
        );
    }
    // if this returns () it failed ig
}
