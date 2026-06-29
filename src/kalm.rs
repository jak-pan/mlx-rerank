// In-process MLX reranker for KaLM-Reranker-V1 (t5gemma2 encoder-decoder).
//
// Faithful Rust/mlx-rs port of the pure-MLX reference
// `../kalm-reranker-v1-small-mlx/modeling_kalm_mlx.py`. ONE config-generalized
// implementation serves every size in the family (Nano, Small, ...): all dims,
// head counts, layer counts/types, RoPE params and rms_norm_eps are read from the
// model's own `config.json` (`encoder.text_config` + `decoder`).
//
// Pipeline (FBNL / fast-but-not-late-interaction):
//   * ENCODER pre-encodes each passage once (bidirectional Gemma2 layers) then
//     chunk-pools (4-token masked mean) -> one compressed vector sequence/passage.
//   * DECODER embeds the Gemma chat prompt (yes/no judgment template) and runs
//     Gemma2 layers with a MERGED self+cross attention over the cached passage
//     encodings (self k/v from decoder states + RoPE; cross k/v from the pooled
//     passage, k_norm but NO rope), one SDPA over the [self || cross] union.
//   * score = softmax([logit_yes, logit_no])[yes] from the decoder's last position.
//
// Everything below mirrors `modeling_kalm_mlx.py` line-for-line; comments cite the
// Python line numbers. bf16 throughout (matches the reference / upstream weights).

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, StreamOrDevice};
use tokenizers::Tokenizer;

const NEG: f32 = -1e30; // matches Python `NEG = mx.array(-1e30, mx.float32)`
const MAX_LEN: usize = 512; // tok(..., max_length=512)
const CHUNK: i32 = 4; // chunk-pool size (l.117 CS=4)

/// Default encode sub-batch (Python ref SUBB default = 25).
pub const SUBB_DEFAULT: usize = 25;

/// Config read from `config.json` (`encoder.text_config` + `decoder`), generalizing
/// across the whole KaLM/t5gemma2 family.
pub struct KalmConfig {
    pub h: i32,           // hidden_size
    pub hd: i32,          // head_dim
    pub nh: i32,          // num_attention_heads
    pub nkv: i32,         // num_key_value_heads
    pub eps: f32,         // rms_norm_eps
    pub enl: usize,       // encoder num_hidden_layers
    pub dnl: usize,       // decoder num_hidden_layers
    pub elt: Vec<String>, // encoder layer_types
    pub dlt: Vec<String>, // decoder layer_types
    pub scale: f32,       // query_pre_attn_scalar ** -0.5
    pub esc: f32,         // embedding scale = sqrt(H)
    // per-side inverse-frequency tables, keyed by attention type.
    pub enc_invf: HashMap<String, Array>,
    pub dec_invf: HashMap<String, Array>,
}

fn cfg_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow!("config missing numeric {key}"))
}
fn cfg_i64(v: &serde_json::Value, key: &str) -> Result<i64> {
    v.get(key)
        .and_then(|x| x.as_i64())
        .ok_or_else(|| anyhow!("config missing int {key}"))
}

/// invfreq = (1 / theta^(arange(0,HD,2)/HD)) / factor   (Python `_invfreq`, l.90-91)
fn invfreq(theta: f64, factor: f64, hd: i32, stream: &StreamOrDevice) -> Result<Array> {
    let j = mlx_rs::ops::arange_device::<f32, f32>(0.0, hd as f32, 2.0, stream)?; // arange(0,HD,2)
    let exp = j.divide(Array::from_f32(hd as f32))?; // j / HD
    let base = Array::from_f32(theta as f32).power(&exp)?; // theta^(j/HD)
    let inv = base.reciprocal()?; // 1 / ...
    inv.divide(Array::from_f32(factor as f32)).map_err(Into::into) // / factor
}

/// Build {full_attention, sliding_attention} invfreq tables for one side (l.92-96).
fn rope_table(
    side_cfg: &serde_json::Value,
    hd: i32,
    stream: &StreamOrDevice,
) -> Result<HashMap<String, Array>> {
    let rp = side_cfg
        .get("rope_parameters")
        .ok_or_else(|| anyhow!("missing rope_parameters"))?;
    let mut out = HashMap::new();
    for t in ["full_attention", "sliding_attention"] {
        let p = rp.get(t).ok_or_else(|| anyhow!("missing rope_parameters.{t}"))?;
        let theta = cfg_f64(p, "rope_theta")?;
        let factor = p.get("factor").and_then(|x| x.as_f64()).unwrap_or(1.0);
        out.insert(t.to_string(), invfreq(theta, factor, hd, stream)?);
    }
    Ok(out)
}

impl KalmConfig {
    pub fn from_json(cfg: &serde_json::Value, stream: &StreamOrDevice) -> Result<Self> {
        let ec = cfg
            .get("encoder")
            .and_then(|e| e.get("text_config"))
            .ok_or_else(|| anyhow!("config missing encoder.text_config"))?;
        let dc = cfg.get("decoder").ok_or_else(|| anyhow!("config missing decoder"))?;
        let h = cfg_i64(ec, "hidden_size")? as i32;
        let hd = cfg_i64(ec, "head_dim")? as i32;
        let nh = cfg_i64(ec, "num_attention_heads")? as i32;
        let nkv = cfg_i64(ec, "num_key_value_heads")? as i32;
        let eps = cfg_f64(ec, "rms_norm_eps")? as f32;
        let enl = cfg_i64(ec, "num_hidden_layers")? as usize;
        let dnl = cfg_i64(dc, "num_hidden_layers")? as usize;
        let qpas = cfg_f64(ec, "query_pre_attn_scalar")?;
        let elt: Vec<String> = ec
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("missing encoder layer_types"))?
            .iter()
            .map(|v| v.as_str().unwrap_or("full_attention").to_string())
            .collect();
        let dlt: Vec<String> = dc
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("missing decoder layer_types"))?
            .iter()
            .map(|v| v.as_str().unwrap_or("full_attention").to_string())
            .collect();
        let enc_invf = rope_table(ec, hd, stream)?;
        let dec_invf = rope_table(dc, hd, stream)?;
        Ok(Self {
            h,
            hd,
            nh,
            nkv,
            eps,
            enl,
            dnl,
            elt,
            dlt,
            scale: (qpas as f32).powf(-0.5), // query_pre_attn_scalar ** -0.5 (l.74)
            esc: (h as f32).sqrt(),           // ESC = H ** 0.5 (l.74)
            enc_invf,
            dec_invf,
        })
    }
}

pub struct KalmModel {
    w: HashMap<String, Array>,
    cfg: KalmConfig,
    emb: Array, // model.encoder.embed_tokens.weight  (l.86 EMB)
    yes_id: u32,
    no_id: u32,
    stream: StreamOrDevice,
}

const DT: Dtype = Dtype::Bfloat16;

impl KalmModel {
    pub fn load(dir: &std::path::Path, tok: &Tokenizer) -> Result<Self> {
        let stream = StreamOrDevice::gpu();
        let cpu = StreamOrDevice::cpu();
        let cfg_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let cfg = KalmConfig::from_json(&cfg_json, &stream)?;

        // Load all safetensors shard(s); keep model.encoder.*/model.decoder.*, skip
        // vision_tower/multi_modal; cast bf16. (Python l.82-86)
        let mut w: HashMap<String, Array> = HashMap::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = entry?.path();
            if p.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let raw = Array::load_safetensors_device(p.to_str().unwrap(), &cpu)
                .map_err(|e| anyhow!("load_safetensors {}: {e}", p.display()))?;
            for (k, v) in raw {
                if k.contains("vision_tower") || k.contains("multi_modal") {
                    continue;
                }
                if k.starts_with("model.encoder.") || k.starts_with("model.decoder.") {
                    w.insert(k, v.as_dtype_device(DT, &stream)?);
                }
            }
        }
        if w.is_empty() {
            return Err(anyhow!("no model.encoder.*/model.decoder.* weights found"));
        }
        mlx_rs::transforms::eval(w.values())?;
        let emb = w
            .get("model.encoder.embed_tokens.weight")
            .ok_or_else(|| anyhow!("missing model.encoder.embed_tokens.weight"))?
            .clone();

        // yes/no token ids: add_special_tokens=False, last id (Python l.89)
        let yes_id = *tok
            .encode("yes", false)
            .map_err(|e| anyhow!("tok yes: {e}"))?
            .get_ids()
            .last()
            .ok_or_else(|| anyhow!("empty yes encoding"))?;
        let no_id = *tok
            .encode("no", false)
            .map_err(|e| anyhow!("tok no: {e}"))?
            .get_ids()
            .last()
            .ok_or_else(|| anyhow!("empty no encoding"))?;

        Ok(Self {
            w,
            cfg,
            emb,
            yes_id,
            no_id,
            stream,
        })
    }

    fn get(&self, k: &str) -> Result<&Array> {
        self.w.get(k).ok_or_else(|| anyhow!("missing weight {k}"))
    }
    fn ge(&self, n: &str) -> Result<&Array> {
        self.get(&format!("model.encoder.{n}"))
    }
    fn gd(&self, n: &str) -> Result<&Array> {
        self.get(&format!("model.decoder.{n}"))
    }

    /// RMSNorm with the Gemma (1.0 + w) convention, fp32 math, cast back (l.99-100).
    fn rms(&self, x: &Array, w: &Array) -> Result<Array> {
        let xf = x.as_dtype(Dtype::Float32)?;
        let ms = xf
            .multiply(&xf)?
            .mean_axis_device(-1, true, &self.stream)?; // mean(x*x, -1, keepdims)
        let inv = ms.add(Array::from_f32(self.cfg.eps))?.rsqrt()?;
        let o = xf.multiply(&inv)?;
        let wf = w.as_dtype(Dtype::Float32)?;
        let scale = Array::from_f32(1.0).add(&wf)?; // (1.0 + w)
        o.multiply(&scale)?.as_dtype(x.dtype()).map_err(Into::into)
    }
    fn rms_k(&self, x: &Array, wkey: &str, enc: bool) -> Result<Array> {
        let w = if enc { self.ge(wkey)? } else { self.gd(wkey)? }.clone();
        self.rms(x, &w)
    }

    /// gelu-tanh (l.101).
    fn gelu(&self, x: &Array) -> Result<Array> {
        // 0.5*x*(1 + tanh(0.7978845608028654*(x + 0.044715*x^3)))
        let x3 = x.multiply(x)?.multiply(x)?;
        let inner = x.add(&x3.multiply(Array::from_f32(0.044715))?)?;
        let t = mlx_rs::ops::tanh_device(
            inner.multiply(Array::from_f32(0.7978845608028654))?,
            &self.stream,
        )?;
        let g = t.add(Array::from_f32(1.0))?;
        x.multiply(Array::from_f32(0.5))?.multiply(&g).map_err(Into::into)
    }

    /// cos/sin tables for a side+layer_type at length T (l.102-104).
    /// Returns (cos[1,1,T,HD], sin[1,1,T,HD]) bf16.
    fn cs(&self, enc: bool, lt: &str, t: i32) -> Result<(Array, Array)> {
        let invf = if enc { &self.cfg.enc_invf } else { &self.cfg.dec_invf };
        let inv = invf.get(lt).ok_or_else(|| anyhow!("no invf for {lt}"))?;
        let pos = mlx_rs::ops::arange_device::<f32, f32>(0.0, t as f32, 1.0, &self.stream)?; // [T]
        // fr = pos[:,None] * invf[None,:]  -> [T, HD/2]
        let fr = pos
            .reshape(&[t, 1])?
            .multiply(&inv.reshape(&[1, -1])?)?;
        let emb = mlx_rs::ops::concatenate_axis_device(&[&fr, &fr], -1, &self.stream)?; // [T, HD]
        let emb4 = emb.reshape(&[1, 1, t, -1])?;
        let cos = emb4.cos()?.as_dtype(DT)?;
        let sin = emb4.sin()?.as_dtype(DT)?;
        Ok((cos, sin))
    }

    /// rope(x,c,s): x1=x[..,:HD/2], x2=x[..,HD/2:]; x*c + concat(-x2,x1)*s (l.105).
    fn rope(&self, x: &Array, c: &Array, s: &Array) -> Result<Array> {
        // x: [B, n_heads, T, HD]. Ranges don't reduce dims, so tuple-range indexing
        // here is correct (unlike integer-index, which mis-selects in this mlx-rs build).
        let half = self.cfg.hd / 2;
        let x1 = x.index_device((.., .., .., 0..half), &self.stream); // [..,:half]
        let x2 = x.index_device((.., .., .., half..self.cfg.hd), &self.stream); // [..,half:]
        let neg_x2 = x2.multiply(Array::from_f32(-1.0))?;
        let rot = mlx_rs::ops::concatenate_axis_device(&[&neg_x2, &x1], -1, &self.stream)?;
        x.multiply(c)?.add(&rot.multiply(s)?).map_err(Into::into)
    }

    /// prj(x,w,n,B,T): (x @ w.T).reshape(B,T,n,HD).transpose(0,2,1,3) (l.106).
    fn prj(&self, x: &Array, w: &Array, n: i32, b: i32, t: i32) -> Result<Array> {
        let y = x.matmul_device(&w.t(), &self.stream)?;
        y.reshape(&[b, t, n, self.cfg.hd])?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Into::into)
    }

    /// Repeat KV heads to NH (GQA): mx.repeat(x, NH, axis=1) repeats EACH head NH
    /// times. The reference does mx.repeat(k, NH, 1) where k has NKV heads; with
    /// NKV=1 this yields NH copies. We mirror mx.repeat exactly (per-element repeat).
    fn repeat_kv(&self, x: &Array) -> Result<Array> {
        // mx.repeat(x, count, axis=1): interleaved per-element repeat along axis 1.
        mlx_rs::ops::repeat_axis_device::<f32>(x.clone(), self.cfg.nh, 1, &self.stream)
            .map_err(Into::into)
    }

    /// One Gemma2 transformer block (shared by encoder & decoder self-path), given a
    /// precomputed (q,k,v) and additive mask. Returns the post-block hidden state.
    /// This factors l.111-116 / l.126-133 (the sandwich-norm + gated-MLP residual).
    #[allow(clippy::too_many_arguments)]
    fn block_tail(
        &self,
        h: &Array,         // residual input [B,T,H]
        attn_out: &Array,  // SDPA output [B,NH,T,HD]
        b: i32,
        t: i32,
        o_key: &str,
        post_attn_key: &str,
        pre_ff_key: &str,
        post_ff_key: &str,
        gate_key: &str,
        up_key: &str,
        down_key: &str,
        enc: bool,
    ) -> Result<Array> {
        // o = rms(attn.transpose.reshape @ o_proj.T, post_self_attn_layernorm); h = r + o
        let merged = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, t, self.cfg.nh * self.cfg.hd])?;
        let ow = if enc { self.ge(o_key)? } else { self.gd(o_key)? };
        let o = merged.matmul_device(&ow.t(), &self.stream)?;
        let o = self.rms_k(&o, post_attn_key, enc)?;
        let h = h.add(&o)?;
        // feedforward (sandwich-normed gated MLP)
        let x = self.rms_k(&h, pre_ff_key, enc)?;
        let gw = if enc { self.ge(gate_key)? } else { self.gd(gate_key)? };
        let uw = if enc { self.ge(up_key)? } else { self.gd(up_key)? };
        let dw = if enc { self.ge(down_key)? } else { self.gd(down_key)? };
        let gate = self.gelu(&x.matmul_device(&gw.t(), &self.stream)?)?;
        let up = x.matmul_device(&uw.t(), &self.stream)?;
        let act = gate.multiply(&up)?;
        let down = act.matmul_device(&dw.t(), &self.stream)?;
        let down = self.rms_k(&down, post_ff_key, enc)?;
        h.add(&down).map_err(Into::into)
    }

    /// Encoder (_encode, l.107-122): bidirectional layers -> norm -> chunk-pool.
    /// ids/am: [B,Te]. Returns (pooled [B,nc,H], pmask [B,nc]).
    fn encode(&self, ids: &Array, am: &Array) -> Result<(Array, Array)> {
        let b = ids.dim(0);
        let te = ids.dim(1);
        // h = EMB[ids] * ESC
        let flat = ids.reshape(&[-1])?;
        let h0 = self.emb.take_axis_device(&flat, 0, &self.stream)?;
        let mut h = h0.reshape(&[b, te, self.cfg.h])?.multiply(Array::from_f32(self.cfg.esc).as_dtype(DT)?)?;
        // kp = where(am[:,None,None,:]==0, NEG, 0.0).astype(DT)  (padding mask, l.108)
        let am4 = am.reshape(&[b, 1, 1, te])?;
        let is_pad = am4.eq(&Array::from_f32(0.0))?;
        let neg = Array::from_f32(NEG);
        let zero = Array::from_f32(0.0);
        let kp = mlx_rs::ops::which_device(&is_pad, &neg, &zero, &self.stream)?.as_dtype(DT)?;
        // CSm[t] for both attention types at length Te
        let cs_full = self.cs(true, "full_attention", te)?;
        let cs_slide = self.cs(true, "sliding_attention", te)?;

        for i in 0..self.cfg.enl {
            let p = format!("layers.{i}.");
            let r = h.clone();
            let x = self.rms_k(&h, &format!("{p}pre_self_attn_layernorm.weight"), true)?;
            let q = self.prj(&x, self.ge(&format!("{p}self_attn.q_proj.weight"))?, self.cfg.nh, b, te)?;
            let k = self.prj(&x, self.ge(&format!("{p}self_attn.k_proj.weight"))?, self.cfg.nkv, b, te)?;
            let v = self.prj(&x, self.ge(&format!("{p}self_attn.v_proj.weight"))?, self.cfg.nkv, b, te)?;
            let q = self.rms_k(&q, &format!("{p}self_attn.q_norm.weight"), true)?;
            let k = self.rms_k(&k, &format!("{p}self_attn.k_norm.weight"), true)?;
            let (c, s) = if self.cfg.elt[i] == "full_attention" { &cs_full } else { &cs_slide };
            let q = self.rope(&q, c, s)?;
            let k = self.rope(&k, c, s)?;
            let kr = self.repeat_kv(&k)?;
            let vr = self.repeat_kv(&v)?;
            let o = mlx_rs::fast::scaled_dot_product_attention_device(
                &q,
                &kr,
                &vr,
                self.cfg.scale,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(&kp),
                &self.stream,
            )?;
            h = self.block_tail(
                &r,
                &o,
                b,
                te,
                &format!("{p}self_attn.o_proj.weight"),
                &format!("{p}post_self_attn_layernorm.weight"),
                &format!("{p}pre_feedforward_layernorm.weight"),
                &format!("{p}post_feedforward_layernorm.weight"),
                &format!("{p}mlp.gate_proj.weight"),
                &format!("{p}mlp.up_proj.weight"),
                &format!("{p}mlp.down_proj.weight"),
                true,
            )?;
        }
        let h = self.rms_k(&h, "norm.weight", true)?; // final encoder norm (l.117)

        // chunk-pool: group seq into CHUNK-token chunks, masked-mean (l.117-122)
        let te2 = h.dim(1);
        let nc = (te2 + CHUNK - 1) / CHUNK;
        let pad = nc * CHUNK - te2;
        let (h, am) = if pad > 0 {
            let hz = mlx_rs::ops::zeros_dtype_device(&[b, pad, self.cfg.h], h.dtype(), &self.stream)?;
            let hp = mlx_rs::ops::concatenate_axis_device(&[&h, &hz], 1, &self.stream)?;
            let mz = mlx_rs::ops::zeros_dtype_device(&[b, pad], am.dtype(), &self.stream)?;
            let mp = mlx_rs::ops::concatenate_axis_device(&[am, &mz], 1, &self.stream)?;
            (hp, mp)
        } else {
            (h, am.clone())
        };
        let hr = h.reshape(&[b, nc, CHUNK, self.cfg.h])?;
        let m = am.reshape(&[b, nc, CHUNK, 1])?.as_dtype(Dtype::Float32)?;
        // pooled = (h.f32 * m).sum(2) / maximum(m.sum(2), 1.0)
        let hf = hr.as_dtype(Dtype::Float32)?;
        let num = hf.multiply(&m)?.sum_axis_device(2, false, &self.stream)?; // [B,nc,H]
        let den = m.sum_axis_device(2, false, &self.stream)?; // [B,nc,1]
        let den = mlx_rs::ops::maximum_device(&den, &Array::from_f32(1.0), &self.stream)?;
        let pooled = num.divide(&den)?.as_dtype(DT)?;
        // pmask = (m.reshape(B,nc,CHUNK).sum(2) > 0).f32
        let msum = m
            .reshape(&[b, nc, CHUNK])?
            .sum_axis_device(2, false, &self.stream)?;
        let pmask = msum.gt(&Array::from_f32(0.0))?.as_dtype(Dtype::Float32)?;
        Ok((pooled, pmask))
    }

    /// Decoder (_decode, l.123-134): merged self+cross attention, final norm.
    /// dids: [B,Td]; pooled: [B,Ne,H]; MM additive mask [B,1,Td,Td+Ne].
    fn decode(&self, dids: &Array, pooled: &Array, mm: &Array) -> Result<Array> {
        let b = dids.dim(0);
        let td = dids.dim(1);
        let ne = pooled.dim(1);
        let flat = dids.reshape(&[-1])?;
        let h0 = self.emb.take_axis_device(&flat, 0, &self.stream)?;
        let mut h = h0.reshape(&[b, td, self.cfg.h])?.multiply(Array::from_f32(self.cfg.esc).as_dtype(DT)?)?;
        let cs_full = self.cs(false, "full_attention", td)?;
        let cs_slide = self.cs(false, "sliding_attention", td)?;

        for i in 0..self.cfg.dnl {
            let p = format!("layers.{i}.");
            let r = h.clone();
            let x = self.rms_k(&h, &format!("{p}pre_self_attn_layernorm.weight"), false)?;
            // self q/k/v
            let q = self.prj(&x, self.gd(&format!("{p}self_attn.q_proj.weight"))?, self.cfg.nh, b, td)?;
            let sk = self.prj(&x, self.gd(&format!("{p}self_attn.k_proj.weight"))?, self.cfg.nkv, b, td)?;
            let sv = self.prj(&x, self.gd(&format!("{p}self_attn.v_proj.weight"))?, self.cfg.nkv, b, td)?;
            let q = self.rms_k(&q, &format!("{p}self_attn.q_norm.weight"), false)?;
            let sk = self.rms_k(&sk, &format!("{p}self_attn.k_norm.weight"), false)?;
            let (c, s) = if self.cfg.dlt[i] == "full_attention" { &cs_full } else { &cs_slide };
            let q = self.rope(&q, c, s)?;
            let sk = self.rope(&sk, c, s)?;
            // cross k/v from pooled (reuse self_attn k/v proj weights); k gets k_norm, NO rope
            let ck = self.prj(pooled, self.gd(&format!("{p}self_attn.k_proj.weight"))?, self.cfg.nkv, b, ne)?;
            let cv = self.prj(pooled, self.gd(&format!("{p}self_attn.v_proj.weight"))?, self.cfg.nkv, b, ne)?;
            let ck = self.rms_k(&ck, &format!("{p}self_attn.k_norm.weight"), false)?;
            // key = repeat(concat([sk,ck],2), NH, 1); val = repeat(concat([sv,cv],2), NH, 1)
            let key = mlx_rs::ops::concatenate_axis_device(&[&sk, &ck], 2, &self.stream)?;
            let val = mlx_rs::ops::concatenate_axis_device(&[&sv, &cv], 2, &self.stream)?;
            let key = self.repeat_kv(&key)?;
            let val = self.repeat_kv(&val)?;
            let o = mlx_rs::fast::scaled_dot_product_attention_device(
                &q,
                &key,
                &val,
                self.cfg.scale,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(mm),
                &self.stream,
            )?;
            h = self.block_tail(
                &r,
                &o,
                b,
                td,
                &format!("{p}self_attn.o_proj.weight"),
                &format!("{p}post_self_attn_layernorm.weight"),
                &format!("{p}pre_feedforward_layernorm.weight"),
                &format!("{p}post_feedforward_layernorm.weight"),
                &format!("{p}mlp.gate_proj.weight"),
                &format!("{p}mlp.up_proj.weight"),
                &format!("{p}mlp.down_proj.weight"),
                false,
            )?;
        }
        self.rms_k(&h, "norm.weight", false) // final decoder norm (l.134)
    }

    /// Build the Gemma chat decoder prompt for the query (dtext, l.138-140).
    fn dtext(&self, tok: &Tokenizer, query: &str) -> Result<String> {
        const INSTR: &str = "Given a search query, retrieve relevant documents that answer the query";
        // qids = tok(q, add_special_tokens=False, truncation, max_length=512); q = decode(qids)
        let enc = tok
            .encode(query, false)
            .map_err(|e| anyhow!("tok query: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().iter().take(MAX_LEN).copied().collect();
        let q = tok
            .decode(&ids, false) // skip_special_tokens=False
            .map_err(|e| anyhow!("decode query: {e}"))?;
        Ok(format!(
            "<bos><start_of_turn>user\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\n\n<Instruct>: {INSTR}\n<Query>: {q}<end_of_turn>\n<start_of_turn>model\n\n\n\n"
        ))
    }

    /// Score all docs for a query, returning relevance in ORIGINAL doc order.
    /// Mirrors rerank() l.142-170 (length-sort, sub-batch encode, one big decode, unsort).
    pub fn score(&self, tok: &Tokenizer, query: &str, docs: &[String], subb: usize) -> Result<Vec<f32>> {
        let n = docs.len();
        if n == 0 {
            return Ok(vec![]);
        }
        // length-SORT by token length of "<Document>: "+doc (l.144)
        let prefixed: Vec<String> = docs.iter().map(|d| format!("<Document>: {d}")).collect();
        let doc_encs = tok
            .encode_batch(prefixed.clone(), false)
            .map_err(|e| anyhow!("tok docs: {e}"))?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| doc_encs[i].get_ids().len());

        // decoder prompt ids (same for every doc) (l.145)
        let dtext = self.dtext(tok, query)?;
        let denc = tok
            .encode(dtext, false)
            .map_err(|e| anyhow!("tok dtext: {e}"))?;
        let dprompt: Vec<i32> = denc.get_ids().iter().map(|&x| x as i32).collect();
        let td = dprompt.len() as i32;

        // SORTED sub-batch encode -> pooled/masks list (l.148-151)
        let mut pools: Vec<Array> = Vec::new();
        let mut masks: Vec<Array> = Vec::new();
        let mut j = 0usize;
        while j < n {
            let end = (j + subb).min(n);
            let sub: Vec<&tokenizers::Encoding> = (j..end).map(|k| &doc_encs[order[k]]).collect();
            // pad to longest in sub-batch, cap MAX_LEN
            let maxlen = sub
                .iter()
                .map(|e| e.get_ids().len().min(MAX_LEN))
                .max()
                .unwrap_or(1)
                .max(1);
            let bsz = sub.len();
            let mut ids = vec![0i32; bsz * maxlen];
            let mut am = vec![0f32; bsz * maxlen];
            for (bi, e) in sub.iter().enumerate() {
                let eids = e.get_ids();
                let len = eids.len().min(MAX_LEN);
                for t in 0..len {
                    ids[bi * maxlen + t] = eids[t] as i32;
                    am[bi * maxlen + t] = 1.0;
                }
            }
            let ids_a = Array::from_slice(&ids, &[bsz as i32, maxlen as i32]);
            let am_a = Array::from_slice(&am, &[bsz as i32, maxlen as i32]);
            let (pl, pm) = self.encode(&ids_a, &am_a)?;
            pools.push(pl);
            masks.push(pm);
            j = end;
        }
        // force encode completion (l.152)
        {
            let mut all: Vec<&Array> = pools.iter().collect();
            all.extend(masks.iter());
            mlx_rs::transforms::eval(all)?;
        }

        // pad pooled to common Ne -> ONE big decode (l.153-159)
        let max_ne = pools.iter().map(|p| p.dim(1)).max().unwrap();
        let mut pp: Vec<Array> = Vec::new();
        let mut pm: Vec<Array> = Vec::new();
        for (p, m) in pools.iter().zip(masks.iter()) {
            let d = max_ne - p.dim(1);
            if d > 0 {
                let pz = mlx_rs::ops::zeros_dtype_device(&[p.dim(0), d, self.cfg.h], p.dtype(), &self.stream)?;
                let pcat = mlx_rs::ops::concatenate_axis_device(&[p, &pz], 1, &self.stream)?;
                let mz = mlx_rs::ops::zeros_dtype_device(&[m.dim(0), d], m.dtype(), &self.stream)?;
                let mcat = mlx_rs::ops::concatenate_axis_device(&[m, &mz], 1, &self.stream)?;
                pp.push(pcat);
                pm.push(mcat);
            } else {
                pp.push(p.clone());
                pm.push(m.clone());
            }
        }
        let pooled = mlx_rs::ops::concatenate_axis_device(&pp.iter().collect::<Vec<_>>(), 0, &self.stream)?;
        let pmask = mlx_rs::ops::concatenate_axis_device(&pm.iter().collect::<Vec<_>>(), 0, &self.stream)?;

        // dids = repeat decoder prompt across n docs (sorted order) (l.160)
        let nn = n as i32;
        let d1 = Array::from_slice(&dprompt, &[1, td]);
        let dids = mlx_rs::ops::repeat_axis_device::<i32>(d1, nn, 0, &self.stream)?;

        // MM = [causal Td×Td ‖ cross-pad Td×Ne]  (l.161-163)
        let ones = mlx_rs::ops::ones_dtype_device(&[td, td], Dtype::Float32, &self.stream)?;
        let triu = mlx_rs::ops::triu_device(&ones, 1, &self.stream)?; // strict upper
        let is_future = triu.gt(&Array::from_f32(0.0))?;
        let causal = mlx_rs::ops::which_device(&is_future, &Array::from_f32(NEG), &Array::from_f32(0.0), &self.stream)?
            .reshape(&[1, 1, td, td])?;
        let causal = mlx_rs::ops::broadcast_to_device(&causal, &[nn, 1, td, td], &self.stream)?;
        // cross = where(pmask[:,None,None,:]==0, NEG, 0.0) broadcast to [n,1,Td,Ne]
        let pm4 = pmask.reshape(&[nn, 1, 1, max_ne])?;
        let is_pad = pm4.eq(&Array::from_f32(0.0))?;
        let cross = mlx_rs::ops::which_device(&is_pad, &Array::from_f32(NEG), &Array::from_f32(0.0), &self.stream)?;
        let cross = mlx_rs::ops::broadcast_to_device(&cross, &[nn, 1, td, max_ne], &self.stream)?;
        let mm = mlx_rs::ops::concatenate_axis_device(&[&causal, &cross], -1, &self.stream)?.as_dtype(DT)?;

        // decode -> logits at last pos -> softmax([yes,no])[yes] (l.164-165)
        let hd = self.decode(&dids, &pooled, &mm)?;
        // h[:, Td-1]  -> [n,H]  (take_axis on the seq axis, then drop it)
        let last_idx = Array::from_int(td - 1);
        let last = hd
            .take_axis_device(&last_idx, 1, &self.stream)?
            .reshape(&[nn, self.cfg.h])?;
        let lg = last.matmul_device(&self.emb.t(), &self.stream)?; // [n, V]
        // gather yes/no columns
        let yn = Array::from_slice(&[self.yes_id as i32, self.no_id as i32], &[2]);
        let pair = lg.take_axis_device(&yn, 1, &self.stream)?; // [n,2] in (yes,no) order
        let sm = mlx_rs::ops::softmax_axis_device(&pair, 1, None, &self.stream)?;
        // P(yes) = column 0 of the softmax. Gather via take_axis (NOT integer-index,
        // which mis-selects on a [n,2] array in this mlx-rs build).
        let zero_idx = Array::from_int(0);
        let sc = sm.take_axis_device(&zero_idx, 1, &self.stream)?.reshape(&[nn])?; // [n] = P(yes)
        mlx_rs::transforms::eval([&sc])?;
        let host = sc.as_dtype(Dtype::Float32)?;
        let slice = host.as_slice::<f32>();

        // unsort -> original order (l.167-168)
        let mut scores = vec![0f32; n];
        for k in 0..n {
            scores[order[k]] = slice[k];
        }
        Ok(scores)
    }
}
