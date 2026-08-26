use alloc::vec;
use alloc::vec::Vec;
use libm::logf;

use crate::kernels::softmax;
use crate::model::{Cache, GenError, Model, Session};

pub const EOS: u8 = 4;
pub const MAX_GEN: usize = 64;

pub struct Candidate {
    pub bytes: Vec<u8>,
    pub mean_logprob: f32,
    pub confidence: f32,
    /// weakest token in the sequence. usually the verb or the first char of a path,
    /// i.e. the point where the model actually decided something
    pub min_token_prob: f32,
}

impl Candidate {
    fn new(bytes: Vec<u8>, sum_lp: f32, min_p: f32) -> Candidate {
        let n = bytes.len().max(1) as f32;
        let mean = sum_lp / n;
        Candidate {
            bytes,
            mean_logprob: mean,
            confidence: libm::expf(mean),
            min_token_prob: min_p,
        }
    }
}

/// logits -> probabilities at the model's calibrated temperature
fn probs_into(out: &mut [f32], logits: &[f32], temp: f32) {
    for (o, l) in out.iter_mut().zip(logits) {
        *o = l / temp;
    }
    softmax(out);
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

/// probability floor so a zeroed softmax entry cannot produce -inf
fn safe_ln(p: f32) -> f32 {
    logf(p.max(1e-30))
}

pub fn greedy(
    model: &Model,
    sess: &mut Session,
    prompt: &[u8],
    max_tokens: usize,
) -> Result<Candidate, GenError> {
    if max_tokens == 0 || prompt.len() + max_tokens > model.cfg.ctx {
        return Err(GenError::NoRoomToGenerate);
    }
    model.prefill(sess, prompt)?;

    let plen = sess.prompt_len;
    let mut ext = Cache::new(&model.cfg, max_tokens);
    let mut probs = vec![0.0f32; model.cfg.vocab];
    let mut out = Vec::with_capacity(max_tokens);
    let mut sum_lp = 0.0f32;
    let mut min_p = 1.0f32;

    loop {
        probs_into(&mut probs, &sess.scratch.logits, model.temp);
        let tok = argmax(&probs);
        let p = probs[tok];

        sum_lp += safe_ln(p);
        min_p = min_p.min(p);
        out.push(tok as u8);

        if tok as u8 == EOS || out.len() >= max_tokens {
            break;
        }

        let pos = plen + out.len() - 1;
        model.forward(
            &mut sess.scratch,
            &mut sess.prefill,
            Some(&mut ext),
            plen,
            pos,
            tok as u8,
        );
    }

    Ok(Candidate::new(out, sum_lp, min_p))
}

struct Beam {
    tokens: Vec<u8>,
    sum_lp: f32,
    min_p: f32,
    logits: Vec<f32>,
}

impl Beam {
    fn new(cfg_vocab: usize, max_tokens: usize) -> Beam {
        Beam {
            tokens: Vec::with_capacity(max_tokens),
            sum_lp: 0.0,
            min_p: 1.0,
            logits: vec![0.0; cfg_vocab],
        }
    }

    fn copy_from(&mut self, other: &Beam) {
        self.tokens.clear();
        self.tokens.extend_from_slice(&other.tokens);
        self.sum_lp = other.sum_lp;
        self.min_p = other.min_p;
        self.logits.copy_from_slice(&other.logits);
    }
}

/// everything beam search needs, allocated once. two sets of caches and beams because a
/// single parent can spawn several children, so the step cannot be done in place.
pub struct Beams {
    width: usize,
    max_tokens: usize,
    caches: Vec<Cache>,
    next_caches: Vec<Cache>,
    live: Vec<Beam>,
    next_live: Vec<Beam>,
    cands: Vec<(f32, usize, u8, f32)>,
    probs: Vec<f32>,
}

impl Beams {
    pub fn new(model: &Model, width: usize, max_tokens: usize) -> Beams {
        let cfg = &model.cfg;
        Beams {
            width,
            max_tokens,
            caches: (0..width).map(|_| Cache::new(cfg, max_tokens)).collect(),
            next_caches: (0..width).map(|_| Cache::new(cfg, max_tokens)).collect(),
            live: (0..width).map(|_| Beam::new(cfg.vocab, max_tokens)).collect(),
            next_live: (0..width).map(|_| Beam::new(cfg.vocab, max_tokens)).collect(),
            cands: Vec::with_capacity(width * cfg.vocab),
            probs: vec![0.0; cfg.vocab],
        }
    }
}

/// Width-N beam search. A beam that emits EOS is retired to the finished list but does
/// NOT stop the search: short wrong hypotheses hit EOS several tokens before the right
/// longer one, and stopping early loses to greedy.
pub fn beam(
    model: &Model,
    sess: &mut Session,
    bs: &mut Beams,
    prompt: &[u8],
) -> Result<Vec<Candidate>, GenError> {
    if bs.max_tokens == 0 || prompt.len() + bs.max_tokens > model.cfg.ctx {
        return Err(GenError::NoRoomToGenerate);
    }
    model.prefill(sess, prompt)?;

    let plen = sess.prompt_len;
    let vocab = model.cfg.vocab;
    let mut finished: Vec<Candidate> = Vec::new();

    probs_into(&mut bs.probs, &sess.scratch.logits, model.temp);

    // seed: the top `width` first tokens
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_unstable_by(|a, b| bs.probs[*b].total_cmp(&bs.probs[*a]));

    let mut n_live = 0;
    for &tok in order.iter().take(bs.width) {
        let p = bs.probs[tok];
        let lp = safe_ln(p);

        if tok as u8 == EOS {
            finished.push(Candidate::new(vec![EOS], lp, p));
            continue;
        }

        model.forward(
            &mut sess.scratch,
            &mut sess.prefill,
            Some(&mut bs.caches[n_live]),
            plen,
            plen,
            tok as u8,
        );

        let b = &mut bs.live[n_live];
        b.tokens.clear();
        b.tokens.push(tok as u8);
        b.sum_lp = lp;
        b.min_p = p;
        b.logits.copy_from_slice(&sess.scratch.logits);
        n_live += 1;
    }

    while n_live > 0 {
        let gen_len = bs.live[0].tokens.len();
        if gen_len >= bs.max_tokens {
            // cap hit; retire whatever is still running
            for b in bs.live[..n_live].iter() {
                finished.push(Candidate::new(b.tokens.clone(), b.sum_lp, b.min_p));
            }
            break;
        }

        bs.cands.clear();
        for bi in 0..n_live {
            probs_into(&mut bs.probs, &bs.live[bi].logits, model.temp);
            let base = bs.live[bi].sum_lp;
            let n = (bs.live[bi].tokens.len() + 1) as f32;
            for t in 0..vocab {
                let p = bs.probs[t];
                // length-normalized, else DELETE("/a") always outscores the long path
                bs.cands.push(((base + safe_ln(p)) / n, bi, t as u8, p));
            }
        }
        bs.cands
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let mut n_next = 0;
        for &(_, bi, tok, p) in bs.cands.iter().take(bs.width) {
            let parent = &bs.live[bi];
            let sum_lp = parent.sum_lp + safe_ln(p);
            let min_p = parent.min_p.min(p);

            if tok == EOS {
                let mut bytes = parent.tokens.clone();
                bytes.push(EOS);
                finished.push(Candidate::new(bytes, sum_lp, min_p));
                continue;
            }

            bs.next_caches[n_next].copy_from(&bs.caches[bi]);
            bs.next_live[n_next].copy_from(parent);

            let b = &mut bs.next_live[n_next];
            b.tokens.push(tok);
            b.sum_lp = sum_lp;
            b.min_p = min_p;

            let pos = plen + b.tokens.len() - 1;
            model.forward(
                &mut sess.scratch,
                &mut sess.prefill,
                Some(&mut bs.next_caches[n_next]),
                plen,
                pos,
                tok,
            );
            bs.next_live[n_next]
                .logits
                .copy_from_slice(&sess.scratch.logits);
            n_next += 1;
        }

        core::mem::swap(&mut bs.caches, &mut bs.next_caches);
        core::mem::swap(&mut bs.live, &mut bs.next_live);
        n_live = n_next;
    }

    finished.sort_unstable_by(|a, b| b.mean_logprob.total_cmp(&a.mean_logprob));
    Ok(finished)
}
