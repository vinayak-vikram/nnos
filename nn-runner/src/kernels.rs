use libm::{expf, fabsf, roundf, sqrtf, tanhf};

pub const RMS_EPS: f32 = 1e-5;

pub fn rmsnorm(out: &mut [f32], x: &[f32], g: &[f32]) {
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / sqrtf(ss / x.len() as f32 + RMS_EPS);
    for ((o, xv), gv) in out.iter_mut().zip(x).zip(g) {
        *o = gv * xv * inv;
    }
}

/// per-vector symmetric quantize, returns the activation scale
pub fn quantize(out: &mut [i8], x: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    for v in x {
        let a = fabsf(*v);
        if a > amax {
            amax = a;
        }
    }
    if amax == 0.0 {
        out[..x.len()].fill(0);
        return 1.0;
    }
    let s = amax / 127.0;
    let inv = 1.0 / s;
    for (o, xv) in out.iter_mut().zip(x) {
        let q = roundf(xv * inv);
        *o = if q > 127.0 {
            127
        } else if q < -127.0 {
            -127
        } else {
            q as i8
        };
    }
    s
}

/// y[o] = sum_c q[o,c] * xq[c] * scale[o] * sx. worst case 1280 * 127 * 127 fits i32 easily.
pub fn matvec_i8(out: &mut [f32], w: &[i8], scales: &[f32], xq: &[i8], sx: f32, cols: usize) {
    for (o, dst) in out.iter_mut().enumerate() {
        let row = &w[o * cols..(o + 1) * cols];
        let (mut a0, mut a1, mut a2, mut a3) = (0i32, 0i32, 0i32, 0i32);
        let mut wc = row.chunks_exact(4);
        let mut xc = xq[..cols].chunks_exact(4);
        for (wv, xv) in wc.by_ref().zip(xc.by_ref()) {
            a0 += wv[0] as i32 * xv[0] as i32;
            a1 += wv[1] as i32 * xv[1] as i32;
            a2 += wv[2] as i32 * xv[2] as i32;
            a3 += wv[3] as i32 * xv[3] as i32;
        }
        let mut acc = a0 + a1 + a2 + a3;
        for (wv, xv) in wc.remainder().iter().zip(xc.remainder()) {
            acc += *wv as i32 * *xv as i32;
        }
        *dst = acc as f32 * scales[o] * sx;
    }
}

/// same weights, no activation quantization. the row scale factors out of the dot, so
/// this is exact dequantized math and is what golden.json was generated against.
pub fn matvec_f32(out: &mut [f32], w: &[i8], scales: &[f32], x: &[f32], cols: usize) {
    for (o, dst) in out.iter_mut().enumerate() {
        let row = &w[o * cols..(o + 1) * cols];
        let mut acc = 0.0f32;
        for (wv, xv) in row.iter().zip(&x[..cols]) {
            acc += *wv as f32 * *xv;
        }
        *dst = acc * scales[o];
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for v in x.iter() {
        if *v > m {
            m = *v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - m);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// tanh approximation, has to match torch's approximate="tanh"
pub fn gelu(x: &mut [f32]) {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + tanhf(C * (u + 0.044715 * u * u * u)));
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn rmsnorm_matches_reference() {
        let x = [1.0f32, -2.0, 3.0, 0.5];
        let g = [1.0f32, 2.0, 0.5, 1.5];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &g);

        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        for i in 0..4 {
            let want = g[i] * x[i] / (ms + RMS_EPS).sqrt();
            assert!((out[i] - want).abs() < 1e-6, "{} vs {}", out[i], want);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, -50.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0] && x[0] > x[3]);
    }

    #[test]
    fn softmax_survives_big_logits() {
        let mut x = [1000.0f32, 1001.0, 999.0];
        softmax(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_matches_tanh_form() {
        let mut x = [-3.0f32, -0.5, 0.0, 0.5, 3.0];
        let orig = x;
        gelu(&mut x);
        for i in 0..5 {
            let u = orig[i];
            let want =
                0.5 * u * (1.0 + ((2.0f32 / core::f32::consts::PI).sqrt() * (u + 0.044715 * u * u * u)).tanh());
            assert!((x[i] - want).abs() < 1e-6, "{} vs {}", x[i], want);
        }
    }

    #[test]
    fn quantize_handles_all_zero() {
        let x = [0.0f32; 8];
        let mut q = [0i8; 8];
        let s = quantize(&mut q, &x);
        assert!(s > 0.0 && s.is_finite());
        assert!(q.iter().all(|v| *v == 0));
    }

    #[test]
    fn matvec_i8_close_to_f32() {
        // 3 outputs, 8 inputs
        let w: [i8; 24] = [
            100, -50, 25, 12, -7, 3, -1, 64, //
            -100, 50, -25, -12, 7, -3, 1, -64, //
            10, 20, 30, 40, 50, 60, 70, 80,
        ];
        let scales = [0.01f32, 0.02, 0.005];
        let x = [1.0f32, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];

        let mut want = vec![0.0f32; 3];
        matvec_f32(&mut want, &w, &scales, &x, 8);

        let mut xq = [0i8; 8];
        let sx = quantize(&mut xq, &x);
        let mut got = vec![0.0f32; 3];
        matvec_i8(&mut got, &w, &scales, &xq, sx, 8);

        for i in 0..3 {
            let rel = (got[i] - want[i]).abs() / want[i].abs().max(1e-6);
            assert!(rel < 1e-2, "row {i}: {} vs {}", got[i], want[i]);
        }
    }
}
