use alloc::vec;
use alloc::vec::Vec;
use libm::sqrtf;

use crate::format::{self, Header, LoadError};
use crate::kernels::{dot, gelu, matvec_f32, matvec_i8, quantize, rmsnorm, softmax};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ctx: usize,
    pub vocab: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    /// int8 weights and activations, what the kernel runs
    I8,
    /// dequantized weights, f32 activations. slower, and what golden.json was made with
    F32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum GenError {
    EmptyPrompt,
    PromptTooLong,
    NoRoomToGenerate,
}

// per-layer tensor slots within the frozen order
const LN1: usize = 0;
const WQ: usize = 1;
const WK: usize = 2;
const WV: usize = 3;
const WO: usize = 4;
const LN2: usize = 5;
const W_UP: usize = 6;
const W_DOWN: usize = 7;

pub struct Model {
    pub cfg: Config,
    pub temp: f32,
    pub precision: Precision,
    buf: Vec<u8>,
    tensors: Vec<Tensor>,
}

pub struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
    cap: usize,
    d: usize,
}

impl Cache {
    pub fn new(cfg: &Config, cap: usize) -> Cache {
        let n = cfg.n_layers * cap * cfg.d_model;
        Cache {
            k: vec![0.0; n],
            v: vec![0.0; n],
            cap,
            d: cfg.d_model,
        }
    }

    fn at(&self, layer: usize, pos: usize) -> usize {
        (layer * self.cap + pos) * self.d
    }

    fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let o = self.at(layer, pos);
        self.k[o..o + self.d].copy_from_slice(k);
        self.v[o..o + self.d].copy_from_slice(v);
    }

    fn k_at(&self, layer: usize, pos: usize) -> &[f32] {
        let o = self.at(layer, pos);
        &self.k[o..o + self.d]
    }

    fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        let o = self.at(layer, pos);
        &self.v[o..o + self.d]
    }
}

pub struct Scratch {
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    kt: Vec<f32>,
    vt: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    hidden: Vec<f32>,
    scores: Vec<f32>,
    xq: Vec<i8>,
    pub logits: Vec<f32>,
    /// when set, forward stashes the residual stream after the embedding and after
    /// every block. bisects a bad layer in one run instead of an evening.
    pub tracing: bool,
    pub trace: Vec<f32>,
}

impl Scratch {
    pub fn new(cfg: &Config) -> Scratch {
        let d = cfg.d_model;
        Scratch {
            x: vec![0.0; d],
            xn: vec![0.0; d],
            q: vec![0.0; d],
            kt: vec![0.0; d],
            vt: vec![0.0; d],
            att: vec![0.0; d],
            proj: vec![0.0; d],
            hidden: vec![0.0; 4 * d],
            scores: vec![0.0; cfg.ctx],
            xq: vec![0; 4 * d],
            logits: vec![0.0; cfg.vocab],
            tracing: false,
            trace: vec![0.0; (cfg.n_layers + 1) * d],
        }
    }

    /// slot 0 is the embedding, slot i+1 is the output of block i
    pub fn trace_at(&self, slot: usize, d: usize) -> &[f32] {
        &self.trace[slot * d..(slot + 1) * d]
    }
}

/// owns the prompt KV cache and every scratch buffer, so Model stays immutable
pub struct Session {
    pub prefill: Cache,
    pub scratch: Scratch,
    pub prompt_len: usize,
}

impl Session {
    pub fn new(cfg: &Config) -> Session {
        Session {
            prefill: Cache::new(cfg, cfg.ctx),
            scratch: Scratch::new(cfg),
            prompt_len: 0,
        }
    }
}

impl Model {
    pub fn load(buf: Vec<u8>) -> Result<Model, LoadError> {
        let h = Header::load(&buf)?;
        let recs = format::parse_directory(&buf, &h)?;

        let mut tensors = Vec::with_capacity(recs.len());
        for r in &recs {
            tensors.push(Tensor::build(&buf, r)?);
        }

        let cfg = Config {
            d_model: h.d_model as usize,
            n_layers: h.n_layers as usize,
            n_heads: h.n_heads as usize,
            head_dim: (h.d_model / h.n_heads) as usize,
            ctx: h.ctx_len as usize,
            vocab: h.vocab as usize,
        };

        Ok(Model {
            cfg,
            temp: h.temp,
            precision: Precision::I8,
            buf,
            tensors,
        })
    }

    pub fn new_session(&self) -> Session {
        Session::new(&self.cfg)
    }

    pub fn n_params(&self) -> usize {
        let d = self.cfg.d_model;
        12 * self.cfg.n_layers * d * d + 256 * d + self.cfg.ctx * d
    }

    fn tok_emb(&self) -> &Tensor {
        &self.tensors[0]
    }

    fn pos_emb(&self) -> &Tensor {
        &self.tensors[1]
    }

    fn layer(&self, i: usize) -> &[Tensor] {
        &self.tensors[2 + i * 8..2 + i * 8 + 8]
    }

    fn lnf(&self) -> &Tensor {
        &self.tensors[2 + self.cfg.n_layers * 8]
    }

    fn prep(&self, xq: &mut [i8], x: &[f32]) -> f32 {
        match self.precision {
            Precision::I8 => quantize(&mut xq[..x.len()], x),
            Precision::F32 => 1.0,
        }
    }

    fn mv(&self, out: &mut [f32], w: &Tensor, x: &[f32], xq: &[i8], sx: f32) {
        let data = w.i8_data(&self.buf);
        match self.precision {
            Precision::I8 => matvec_i8(out, data, w.scales(), xq, sx, w.cols),
            Precision::F32 => matvec_f32(out, data, w.scales(), x, w.cols),
        }
    }

    /// one position. writes K/V for `pos` into ext if given, else into prefill.
    /// `prompt_len` splits absolute positions between the two caches; pass usize::MAX
    /// during prefill so every key lands in the prefill cache.
    pub fn forward(
        &self,
        sc: &mut Scratch,
        prefill: &mut Cache,
        mut ext: Option<&mut Cache>,
        prompt_len: usize,
        pos: usize,
        tok: u8,
    ) {
        let d = self.cfg.d_model;
        let hd = self.cfg.head_dim;
        let n_keys = pos + 1;
        let ascale = 1.0 / sqrtf(hd as f32);

        let Scratch {
            x,
            xn,
            q,
            kt,
            vt,
            att,
            proj,
            hidden,
            scores,
            xq,
            logits,
            tracing,
            trace,
        } = sc;
        let tracing = *tracing;

        // embed: dequantized token row plus the learned absolute position
        let te = self.tok_emb();
        let te_data = te.i8_data(&self.buf);
        let ts = te.scales()[tok as usize];
        let trow = &te_data[tok as usize * d..(tok as usize + 1) * d];
        let prow = self.pos_emb().f32_row(pos);
        for c in 0..d {
            x[c] = trow[c] as f32 * ts + prow[c];
        }
        if tracing {
            trace[..d].copy_from_slice(x);
        }

        for l in 0..self.cfg.n_layers {
            let lt = self.layer(l);

            rmsnorm(xn, x, lt[LN1].f32_all());
            let sx = self.prep(xq, xn);
            self.mv(q, &lt[WQ], xn, xq, sx);
            self.mv(kt, &lt[WK], xn, xq, sx);
            self.mv(vt, &lt[WV], xn, xq, sx);

            match ext.as_deref_mut() {
                Some(e) => e.write(l, pos - prompt_len, kt, vt),
                None => prefill.write(l, pos, kt, vt),
            }

            let pf: &Cache = prefill;
            let ex: Option<&Cache> = ext.as_deref();

            for h in 0..self.cfg.n_heads {
                let off = h * hd;
                let qh = &q[off..off + hd];

                for (j, s) in scores[..n_keys].iter_mut().enumerate() {
                    let kj = if j < prompt_len {
                        pf.k_at(l, j)
                    } else {
                        ex.unwrap().k_at(l, j - prompt_len)
                    };
                    *s = dot(qh, &kj[off..off + hd]) * ascale;
                }
                softmax(&mut scores[..n_keys]);

                let dst = &mut att[off..off + hd];
                dst.fill(0.0);
                for (j, &w) in scores[..n_keys].iter().enumerate() {
                    let vj = if j < prompt_len {
                        pf.v_at(l, j)
                    } else {
                        ex.unwrap().v_at(l, j - prompt_len)
                    };
                    for (o, vv) in dst.iter_mut().zip(&vj[off..off + hd]) {
                        *o += w * vv;
                    }
                }
            }

            let sx = self.prep(xq, att);
            self.mv(proj, &lt[WO], att, xq, sx);
            for (xv, pv) in x.iter_mut().zip(proj.iter()) {
                *xv += pv;
            }

            rmsnorm(xn, x, lt[LN2].f32_all());
            let sx = self.prep(xq, xn);
            self.mv(hidden, &lt[W_UP], xn, xq, sx);
            gelu(hidden);
            let sh = self.prep(xq, hidden);
            self.mv(proj, &lt[W_DOWN], hidden, xq, sh);
            for (xv, pv) in x.iter_mut().zip(proj.iter()) {
                *xv += pv;
            }

            if tracing {
                trace[(l + 1) * d..(l + 2) * d].copy_from_slice(x);
            }
        }

        rmsnorm(xn, x, self.lnf().f32_all());
        let sx = self.prep(xq, xn);
        self.mv(logits, self.tok_emb(), xn, xq, sx);
    }

    /// run the prompt through, leaving K/V for every prompt position in sess.prefill
    pub fn prefill(&self, sess: &mut Session, prompt: &[u8]) -> Result<(), GenError> {
        if prompt.is_empty() {
            return Err(GenError::EmptyPrompt);
        }
        if prompt.len() > self.cfg.ctx {
            return Err(GenError::PromptTooLong);
        }

        sess.prompt_len = prompt.len();
        for (i, &t) in prompt.iter().enumerate() {
            self.forward(
                &mut sess.scratch,
                &mut sess.prefill,
                None,
                usize::MAX,
                i,
                t,
            );
        }
        Ok(())
    }
}
