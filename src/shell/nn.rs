use alloc::string::String;
use alloc::vec::Vec;
use ext4plus::Ext4;
use ext4plus::iters::AsyncIterator;

use super::{CommandBuffer, ShellProfile};
use crate::helpers::stdio::*;
use crate::kernel::syscall::{Intent, Syscall};

pub struct NNProfile(Option<nn_runner::Shell>);

impl NNProfile {
    pub fn new() -> Self {
        NNProfile(None)
    }
}

impl ShellProfile for NNProfile {
    async fn init(&mut self, fs: &Ext4) -> Result<(), ()> {
        let Ok(shbin) = fs.read("/sh.bin").await else {
            return Err(());
        };
        self.0 = Some(nn_runner::Shell::load(shbin, 1).map_err(|_| ())?);
        Ok(())
    }

    async fn infer(&mut self, cmd: CommandBuffer, fs: &Ext4) -> Option<Intent> {
        let Ok(mut entries) = fs.read_dir("/").await else {
            return None;
        };
        let mut dirl: Vec<String> = Vec::new();
        while let Some(entry) = entries.next().await {
            let Ok(entry) = entry else { continue };
            if let Ok(name) = entry.file_name().as_str()
                && name != "."
                && name != ".."
            {
                dirl.push(String::from(name));
            }
        }
        let refs: Vec<&str> = dirl.iter().map(String::as_str).collect();

        println("\r\n[nn] inference");
        let shell = self.0.as_mut()?;
        let Ok(out) = shell.run(&refs, &cmd.buf[..cmd.len]) else {
            println("inference failed");
            return None;
        };

        if out.output == b"NONE" {
            println("no actionable intent");
            return None;
        }
        let Some(sc) = parse(&out.output) else {
            println("syscall rejected (intent unparseable)");
            printv(&out.output);
            println("");
            return None;
        };
        Some(Intent {
            sc,
            confidence: out.min_token_prob as f64, //turns out that actual confidence is kinda geed
        })
    }
}

fn quoted(s: &str) -> bool {
    !s.contains('"') && !s.contains('\\')
}
fn path(s: &str) -> bool {
    s.starts_with('/') && quoted(s)
}

fn parse(out: &[u8]) -> Option<Syscall> {
    let s = core::str::from_utf8(out).ok()?;

    match s {
        "TIME" => return Some(Syscall::Time),
        "REBOOT" => return Some(Syscall::Reboot),
        "LIST(\"/\")" => {
            return Some(Syscall::List {
                path: String::from("/"),
            });
        } //TODO: actually have nested dirs, fml
        _ => {}
    }

    let open = s.find("(\"")?;
    if !s.ends_with("\")") || s.len() < open + 4 {
        return None;
    }
    let arg = &s[open + 2..s.len() - 2];

    match &s[..open] {
        "PRINT" => quoted(arg).then(|| Syscall::Print {
            message: String::from(arg),
        }),
        "READ" => path(arg).then(|| Syscall::Read {
            path: String::from(arg),
        }),
        "CREATE" => path(arg).then(|| Syscall::Create {
            path: String::from(arg),
        }),
        "DELETE" => path(arg).then(|| Syscall::Delete {
            path: String::from(arg),
        }),
        "WRITE" => {
            let mid = arg.find("\", \"")?;
            let (p, text) = (&arg[..mid], &arg[mid + 4..]);
            (path(p) && quoted(text)).then(|| Syscall::Write {
                path: String::from(p),
                data: text.as_bytes().to_vec(),
            })
        }
        _ => None,
    }
}
