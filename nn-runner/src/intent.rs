//! The frozen prompt and output contract. Change anything here and the model has to be
//! retrained, so the truncation rule mirrors ds/datagen.py::truncate_listing exactly.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const PAD: u8 = 0;
pub const BOS: u8 = 1;
pub const SEP: u8 = 2;
pub const SOO: u8 = 3;
pub const EOS: u8 = 4;

pub const LISTING_CAP: usize = 180;
pub const OUTPUT_RESERVE: usize = 96;

#[derive(Debug, PartialEq, Eq)]
pub enum Intent {
    Print(String),
    Read(String),
    Write(String, String),
    Create(String),
    Delete(String),
    List,
    Time,
    Reboot,
    /// trained output for chitchat and garbage, not a parse failure
    None,
}

/// Keep whole names in directory order until the budget is spent. The budget shrinks
/// with the command length so a long line cannot push the sequence past ctx.
pub fn truncate_listing<'a>(names: &[&'a str], line_len: usize, ctx: usize) -> Vec<&'a str> {
    let budget = LISTING_CAP.min(ctx.saturating_sub(4 + line_len + OUTPUT_RESERVE));
    let mut kept = Vec::new();
    let mut used = 0;
    for n in names {
        let add = n.len() + if kept.is_empty() { 0 } else { 1 };
        if used + add > budget {
            break;
        }
        kept.push(*n);
        used += add;
    }
    kept
}

/// BOS + listing + SEP + line + SOO. `line` is the raw typed bytes minus the newline;
/// no lowercasing or trimming, the model was trained on noisy input.
pub fn build_prompt(listing: &[&str], line: &[u8], ctx: usize) -> Vec<u8> {
    // bytes 0..=4 are the special tokens, so they cannot appear in user input
    let clean: Vec<u8> = line.iter().map(|b| if *b <= EOS { b'?' } else { *b }).collect();
    let kept = truncate_listing(listing, clean.len(), ctx);

    let mut p = Vec::with_capacity(clean.len() + LISTING_CAP + 3);
    p.push(BOS);
    for (i, n) in kept.iter().enumerate() {
        if i > 0 {
            p.push(b' ');
        }
        p.extend_from_slice(n.as_bytes());
    }
    p.push(SEP);
    p.extend_from_slice(&clean);
    p.push(SOO);
    p
}

fn ok_text(s: &str) -> bool {
    !s.contains('"') && !s.contains('\\')
}

fn ok_path(s: &str) -> bool {
    s.starts_with('/') && s.len() > 1 && ok_text(s)
}

/// Deliberately dumb: match the verb up to `("`, then scan for `", "` and `")`.
/// Anything that does not parse is an unparseable intent, which is the caller's problem.
pub fn parse(out: &[u8]) -> Option<Intent> {
    let end = out.iter().position(|b| *b <= EOS).unwrap_or(out.len());
    let s = core::str::from_utf8(&out[..end]).ok()?;

    match s {
        "TIME" => return Some(Intent::Time),
        "REBOOT" => return Some(Intent::Reboot),
        "NONE" => return Some(Intent::None),
        // LIST is root-only, the ramdisk is flat
        "LIST(\"/\")" => return Some(Intent::List),
        _ => {}
    }

    let open = s.find("(\"")?;
    if !s.ends_with("\")") || s.len() < open + 4 {
        return None;
    }
    let verb = &s[..open];
    let inner = &s[open + 2..s.len() - 2];

    match verb {
        "PRINT" => ok_text(inner).then(|| Intent::Print(inner.to_string())),
        "READ" => ok_path(inner).then(|| Intent::Read(inner.to_string())),
        "CREATE" => ok_path(inner).then(|| Intent::Create(inner.to_string())),
        "DELETE" => ok_path(inner).then(|| Intent::Delete(inner.to_string())),
        "WRITE" => {
            let mid = inner.find("\", \"")?;
            let path = &inner[..mid];
            let text = &inner[mid + 4..];
            (ok_path(path) && ok_text(text))
                .then(|| Intent::Write(path.to_string(), text.to_string()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn parses_every_verb() {
        assert_eq!(parse(b"TIME"), Some(Intent::Time));
        assert_eq!(parse(b"REBOOT"), Some(Intent::Reboot));
        assert_eq!(parse(b"NONE"), Some(Intent::None));
        assert_eq!(parse(b"LIST(\"/\")"), Some(Intent::List));
        assert_eq!(
            parse(b"READ(\"/hello.txt\")"),
            Some(Intent::Read("/hello.txt".to_string()))
        );
        assert_eq!(
            parse(b"DELETE(\"/a.txt\")"),
            Some(Intent::Delete("/a.txt".to_string()))
        );
        assert_eq!(
            parse(b"CREATE(\"/new\")"),
            Some(Intent::Create("/new".to_string()))
        );
        assert_eq!(
            parse(b"PRINT(\"hi there\")"),
            Some(Intent::Print("hi there".to_string()))
        );
        assert_eq!(
            parse(b"WRITE(\"/a.txt\", \"some text\")"),
            Some(Intent::Write("/a.txt".to_string(), "some text".to_string()))
        );
    }

    #[test]
    fn stops_at_eos() {
        let mut v = b"TIME".to_vec();
        v.push(EOS);
        v.extend_from_slice(b"garbage");
        assert_eq!(parse(&v), Some(Intent::Time));
    }

    #[test]
    fn rejects_junk() {
        assert_eq!(parse(b""), None);
        assert_eq!(parse(b"DELET(\"/a\")"), None);
        assert_eq!(parse(b"DELETE(\"/a\""), None);
        assert_eq!(parse(b"DELETE(\"a.txt\")"), None); // no leading slash
        assert_eq!(parse(b"DELETE(\"/\")"), None); // root is not a file
        assert_eq!(parse(b"LIST(\"/sub\")"), None); // root-only
        assert_eq!(parse(b"WRITE(\"/a\")"), None); // missing text arg
    }

    #[test]
    fn truncation_matches_the_python_rule() {
        let names = ["aaa", "bbb", "ccc"];
        // 3 + 1 + 3 + 1 + 3 = 11 fits in the cap
        assert_eq!(truncate_listing(&names, 10, 512).len(), 3);
        // budget 4 keeps "aaa" only, since adding "bbb" costs the space too
        assert_eq!(truncate_listing(&names, 512 - 4 - 4 - 96, 512), vec!["aaa"]);
        // a long line squeezes the listing out entirely
        assert!(truncate_listing(&names, 512, 512).is_empty());
    }

    #[test]
    fn long_command_shrinks_the_listing() {
        let names = ["a.txt", "b.txt", "c.txt", "d.txt"];
        let short = build_prompt(&names, b"hi", 512);
        let long = build_prompt(&names, &[b'x'; 256], 512);
        assert!(long.len() <= 512);
        assert!(short.len() < long.len());
    }

    #[test]
    fn sanitizes_special_bytes() {
        let p = build_prompt(&["a.txt"], &[b'h', 0, b'i', 4], 512);
        assert_eq!(p[0], BOS);
        assert_eq!(*p.last().unwrap(), SOO);
        // exactly one SEP, and no stray specials from the user line
        assert_eq!(p.iter().filter(|b| **b == SEP).count(), 1);
        assert_eq!(p.iter().filter(|b| **b == PAD).count(), 0);
        assert_eq!(p.iter().filter(|b| **b == EOS).count(), 0);
    }

    #[test]
    fn prompt_matches_golden_encoding() {
        let p = build_prompt(
            &["birthday.txt", "hello.txt", "secret.txt"],
            b"delete hello.txt",
            512,
        );
        let want: Vec<u8> = {
            let mut v = vec![BOS];
            v.extend_from_slice(b"birthday.txt hello.txt secret.txt");
            v.push(SEP);
            v.extend_from_slice(b"delete hello.txt");
            v.push(SOO);
            v
        };
        assert_eq!(p, want);
        assert_eq!(p.len(), 52);
    }
}
