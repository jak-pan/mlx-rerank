// In-process MLX reranker for nvidia/llama-nemotron-rerank-1b-v2.
// Ported from the Python MLX reference (/tmp/nemo_opt.py + llama_bidirectional_model.py).
//
// Standard Llama-3.2-1B arch but BIDIRECTIONAL (padding mask, not causal),
// masked-MEAN pooling, and a `score` linear head [1, 2048] instead of an lm_head.
//
// Stages:
//   default          -> Stage A/C: score payload-0 (commute query, 100 docs), print top-3 + timing.
//   --dump <out.json> -> dump per-doc scores for the first N docs (Stage B reference diff).
//   --ndocs <N>       -> limit number of docs (default 100; with --dump default 3).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use mlx_rs::{Array, Device, Dtype, StreamOrDevice};
use tokenizers::Tokenizer;

mod kalm;

/// Default model when RERANK_MODEL is unset (back-compat with the original binary).
const DEFAULT_MODEL: &str = "nvidia/llama-nemotron-rerank-1b-v2";

const HIDDEN: i32 = 2048;
const N_LAYERS: usize = 16;
const N_HEADS: i32 = 32;
const N_KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 64;
const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 500000.0;
const SDPA_SCALE: f32 = 0.125; // 1/sqrt(64)
const MAX_LEN: usize = 512;
const SUB_BATCH: usize = 10; // swept optimum: length-sorted chunks at the padding-vs-launch knee
const NEG: f32 = -6e4;

/// llama3 rope_scaling params (from config.json).
const ROPE_FACTOR: f32 = 32.0;
const ROPE_LOW_FREQ: f32 = 1.0;
const ROPE_HIGH_FREQ: f32 = 4.0;
const ROPE_ORIG_CTX: f32 = 8192.0;

/// Resolve `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*` for any HF
/// repo id (e.g. "nvidia/llama-nemotron-rerank-1b-v2", "KaLM-Embedding/KaLM-Reranker-V1-Small").
fn model_dir(repo_id: &str) -> Result<std::path::PathBuf> {
    let slug = repo_id.replace('/', "--");
    let base = dirs_home().join(format!(".cache/huggingface/hub/models--{slug}/snapshots"));
    let snap = std::fs::read_dir(&base)
        .with_context(|| format!("reading {}", base.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .ok_or_else(|| anyhow!("no snapshot dir under {}", base.display()))?;
    Ok(snap)
}

fn dirs_home() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
}

/// Precompute llama3-scaled rope frequencies, matching mlx-lm `Llama3Rope::new`.
fn llama3_freqs(stream: &StreamOrDevice) -> Result<Array> {
    let half = HEAD_DIM / 2;
    // freqs = base ** (arange(0,dim,2)/dim) == base^(2i/dim)
    let indices = mlx_rs::ops::arange_device::<f32, f32>(0.0, half as f32, 1.0, stream)?;
    let exponents = indices.multiply(Array::from_f32(2.0 / HEAD_DIM as f32))?;
    let freqs = Array::from_f32(ROPE_THETA).power(&exponents)?;

    let low_freq_wavelen = ROPE_ORIG_CTX / ROPE_LOW_FREQ;
    let high_freq_wavelen = ROPE_ORIG_CTX / ROPE_HIGH_FREQ;

    let two_pi = Array::from_f32(2.0 * std::f32::consts::PI);
    let wavelens = freqs.multiply(&two_pi)?;

    // low frequencies (long wavelengths) scaled by factor
    let is_low = wavelens.gt(Array::from_f32(low_freq_wavelen))?;
    let freqs = mlx_rs::ops::which(&is_low, &freqs.multiply(Array::from_f32(ROPE_FACTOR))?, &freqs)?;

    // medium frequencies: smooth interpolation
    let is_medium = wavelens
        .gt(Array::from_f32(high_freq_wavelen))?
        .logical_and(&wavelens.lt(Array::from_f32(low_freq_wavelen))?)?;
    let smooth_factors = wavelens
        .reciprocal()?
        .multiply(Array::from_f32(ROPE_ORIG_CTX))?
        .subtract(Array::from_f32(ROPE_LOW_FREQ))?
        .divide(Array::from_f32(ROPE_HIGH_FREQ - ROPE_LOW_FREQ))?;
    let one_minus = Array::from_f32(1.0).subtract(&smooth_factors)?;
    let denom = one_minus
        .divide(Array::from_f32(ROPE_FACTOR))?
        .add(&smooth_factors)?;
    let smooth_freqs = freqs.divide(&denom)?;
    let freqs = mlx_rs::ops::which(&is_medium, &smooth_freqs, &freqs)?;
    Ok(freqs)
}

struct Weights {
    w: HashMap<String, Array>,
}

impl Weights {
    fn get(&self, k: &str) -> Result<&Array> {
        self.w.get(k).ok_or_else(|| anyhow!("missing weight {k}"))
    }
}

struct Model {
    w: Weights,
    freqs: Array,
    stream: StreamOrDevice,
}

impl Model {
    fn load(dir: &std::path::Path) -> Result<Self> {
        let stream = StreamOrDevice::gpu();
        let cpu = StreamOrDevice::cpu();
        let path = dir.join("model.safetensors");
        // safetensors load must run on a CPU stream (it's an I/O op).
        let raw = Array::load_safetensors_device(path.to_str().unwrap(), &cpu)
            .map_err(|e| anyhow!("load_safetensors: {e}"))?;
        // cast all weights to f16 (the Python ref loads weights as float16), on GPU.
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let v16 = v.as_dtype_device(Dtype::Float16, &stream)?;
            w.insert(k, v16);
        }
        mlx_rs::transforms::eval(w.values())?;
        let freqs = llama3_freqs(&stream)?;
        mlx_rs::transforms::eval([&freqs])?;
        Ok(Self {
            w: Weights { w },
            freqs,
            stream,
        })
    }

    /// Linear: x @ W.T  (W is [out, in], stored row-major as in PyTorch).
    fn linear(&self, x: &Array, wkey: &str) -> Result<Array> {
        let wt = self.w.get(wkey)?.t();
        x.matmul_device(&wt, &self.stream).map_err(Into::into)
    }

    fn rms_norm(&self, x: &Array, wkey: &str) -> Result<Array> {
        let w = self.w.get(wkey)?;
        mlx_rs::fast::rms_norm_device(x, w, RMS_EPS, &self.stream).map_err(Into::into)
    }

    fn rope(&self, x: &Array) -> Result<Array> {
        // x: [B, H, S, head_dim]; flatten leading dims like mlx-lm does.
        let shape = x.shape().to_vec();
        let x3 = x.reshape(&[-1, x.dim(-2), x.dim(-1)])?;
        let r = mlx_rs::fast::rope_device(
            &x3,
            HEAD_DIM,
            false,            // not traditional
            None::<f32>,      // base ignored when freqs supplied
            1.0,              // scale
            0,                // offset
            Some(&self.freqs),
            &self.stream,
        )?;
        r.reshape(&shape).map_err(Into::into)
    }

    /// One bidirectional Llama decoder layer. `mask` is additive [B,1,1,S].
    fn layer(&self, h: &Array, mask: &Array, i: usize) -> Result<Array> {
        let p = format!("model.layers.{i}");
        let b = h.dim(0);
        let s = h.dim(1);

        // --- attention ---
        let x = self.rms_norm(h, &format!("{p}.input_layernorm.weight"))?;
        let q = self.linear(&x, &format!("{p}.self_attn.q_proj.weight"))?;
        let k = self.linear(&x, &format!("{p}.self_attn.k_proj.weight"))?;
        let v = self.linear(&x, &format!("{p}.self_attn.v_proj.weight"))?;

        let q = q
            .reshape(&[b, s, N_HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, s, N_KV_HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, s, N_KV_HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.rope(&q)?;
        let k = self.rope(&k)?;

        let attn = mlx_rs::fast::scaled_dot_product_attention_device(
            &q,
            &k,
            &v,
            SDPA_SCALE,
            mlx_rs::fast::ScaledDotProductAttentionMask::Array(mask),
            &self.stream,
        )?;
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
        let attn = self.linear(&attn, &format!("{p}.self_attn.o_proj.weight"))?;
        let h = h.add(&attn)?;

        // --- mlp ---
        let x = self.rms_norm(&h, &format!("{p}.post_attention_layernorm.weight"))?;
        let gate = self.linear(&x, &format!("{p}.mlp.gate_proj.weight"))?;
        let up = self.linear(&x, &format!("{p}.mlp.up_proj.weight"))?;
        let act = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
        let down = self.linear(&act, &format!("{p}.mlp.down_proj.weight"))?;
        h.add(&down).map_err(Into::into)
    }

    /// Forward a padded sub-batch -> [B] relevance scores.
    /// input_ids: [B,S] i32, attention_mask: [B,S] i32 (1=keep,0=pad).
    fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let b = input_ids.dim(0);
        let s = input_ids.dim(1);

        // embed: gather rows of embed_tokens.weight [V,2048]
        let flat = input_ids.reshape(&[-1])?;
        let emb_w = self.w.get("model.embed_tokens.weight")?;
        let h = emb_w.take_axis_device(&flat, 0, &self.stream)?;
        let mut h = h.reshape(&[b, s, HIDDEN])?;

        // additive padding mask [B,1,1,S]: 0.0 where keep, NEG where pad.
        let am_f = attention_mask.as_dtype(Dtype::Float16)?;
        let am4 = am_f.reshape(&[b, 1, 1, s])?;
        let neg = Array::from_f32(NEG).as_dtype(Dtype::Float16)?;
        let zero = Array::from_f32(0.0).as_dtype(Dtype::Float16)?;
        // where(am==0, NEG, 0.0)
        let is_pad = am4.eq(&zero)?;
        let mask = mlx_rs::ops::which(&is_pad, &neg, &zero)?.as_dtype(Dtype::Float16)?;

        for i in 0..N_LAYERS {
            h = self.layer(&h, &mask, i)?;
        }
        let h = self.rms_norm(&h, "model.norm.weight")?;

        // masked-MEAN pool over seq: (h * amf).sum(1) / amf.sum(1)
        let amf = am_f.reshape(&[b, s, 1])?; // [B,S,1]
        let masked = h.multiply(&amf)?;
        let summed = masked.sum_axis_device(1, false, &self.stream)?; // [B,2048]
        let denom = amf.sum_axis_device(1, false, &self.stream)?; // [B,1]
        let pooled = summed.divide(&denom)?; // [B,2048]

        // score head: pooled @ score.weight.T  -> [B,1]
        let score = self.linear(&pooled, "score.weight")?;
        score.reshape(&[b]).map_err(Into::into)
    }
}

fn prompt(q: &str, p: &str) -> String {
    format!("question:{q} \n \n passage:{p}")
}

/// Which backend architecture a snapshot's config.json selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    /// nvidia llama-nemotron-rerank (model_type "llama_bidirec").
    Nemotron,
    /// KaLM-Reranker-V1 t5gemma2 encoder-decoder.
    Kalm,
}

/// Detect the architecture from a snapshot's config.json.
fn detect_arch(dir: &std::path::Path) -> Result<Arch> {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let mt = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let arch0 = cfg
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if mt == "llama_bidirec" || arch0.starts_with("LlamaBidirectional") {
        Ok(Arch::Nemotron)
    } else if mt == "t5gemma2" || arch0.starts_with("T5Gemma2") {
        Ok(Arch::Kalm)
    } else {
        Err(anyhow!(
            "unrecognized architecture: model_type={mt:?} architectures[0]={arch0:?}"
        ))
    }
}

/// Loaded reranker engine: one process loads exactly one model.
enum Engine {
    Nemotron(Model),
    Kalm(kalm::KalmModel),
}

impl Engine {
    /// Score docs for a query, returning relevance in ORIGINAL doc order.
    fn score(&self, tok: &Tokenizer, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        match self {
            // Nemotron: length-sorted deferred-eval path (default sub-batch 10).
            Engine::Nemotron(m) => run(m, tok, query, docs, SUB_BATCH),
            // KaLM: encoder-decoder path (default sub-batch 25, matching the Python ref).
            Engine::Kalm(m) => m.score(tok, query, docs, kalm::SUBB_DEFAULT),
        }
    }
}

/// Tokenize a batch of strings (one parallel `encode_batch`); propagate `tokenizers::Error`.
fn tokenize_batch(
    tok: &Tokenizer,
    texts: Vec<String>,
) -> Result<Vec<tokenizers::Encoding>, tokenizers::Error> {
    tok.encode_batch(texts, true)
}

/// Build padded [B,S] (ids, mask) i32 arrays from a slice of pre-computed encodings,
/// padding to the longest encoding in the slice (capped at MAX_LEN). Buffers reusable.
fn pad_chunk(
    encs: &[&tokenizers::Encoding],
    ids: &mut Vec<i32>,
    mask: &mut Vec<i32>,
) -> (Array, Array) {
    let b = encs.len();
    let maxlen = encs
        .iter()
        .map(|e| e.get_ids().len().min(MAX_LEN))
        .max()
        .unwrap_or(1)
        .max(1);
    ids.clear();
    ids.resize(b * maxlen, 0);
    mask.clear();
    mask.resize(b * maxlen, 0);
    for (bi, e) in encs.iter().enumerate() {
        let eids = e.get_ids();
        let n = eids.len().min(MAX_LEN);
        for j in 0..n {
            ids[bi * maxlen + j] = eids[j] as i32;
            mask[bi * maxlen + j] = 1;
        }
    }
    (
        Array::from_slice(ids, &[b as i32, maxlen as i32]),
        Array::from_slice(mask, &[b as i32, maxlen as i32]),
    )
}

/// Score all docs: length-sorted sub-batches of `sub_batch`, masked-mean, score head.
///
/// Deferred eval: build ALL sub-batch score arrays lazily first (no per-chunk
/// GPU barrier), then `eval` over all of them ONCE so MLX can pipeline the chunks,
/// then read them back.
fn run(model: &Model, tok: &Tokenizer, q: &str, docs: &[String], sub_batch: usize) -> Result<Vec<f32>> {
    let mut scores = vec![0f32; docs.len()];

    // Tokenize ALL docs in one parallel encode_batch (max rayon parallelism), in
    // original order, then sort by REAL token length for tight per-chunk padding.
    let texts: Vec<String> = docs.iter().map(|d| prompt(q, d)).collect();
    let encs = tokenize_batch(tok, texts).map_err(|e| anyhow!("tokenize: {e}"))?;
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_unstable_by_key(|&i| encs[i].get_ids().len().min(MAX_LEN));

    // Build every chunk's forward lazily; collect (idx-slice, score-array).
    let mut chunks: Vec<(Vec<usize>, Array)> = Vec::new();
    let mut ids_buf: Vec<i32> = Vec::with_capacity(sub_batch * MAX_LEN);
    let mut mask_buf: Vec<i32> = Vec::with_capacity(sub_batch * MAX_LEN);
    let mut i = 0;
    while i < order.len() {
        let idx: Vec<usize> = order[i..(i + sub_batch).min(order.len())].to_vec();
        let chunk_encs: Vec<&tokenizers::Encoding> = idx.iter().map(|&j| &encs[j]).collect();
        let (ids, mask) = pad_chunk(&chunk_encs, &mut ids_buf, &mut mask_buf);
        let sc = model.forward(&ids, &mask)?; // lazy, not eval'd
        chunks.push((idx, sc));
        i += sub_batch;
    }

    // One barrier over all chunks -> MLX pipelines them.
    mlx_rs::transforms::eval(chunks.iter().map(|(_, sc)| sc))?;

    for (idx, sc) in &chunks {
        let host = sc.as_dtype(Dtype::Float32)?;
        let slice = host.as_slice::<f32>(); // read directly, no extra Vec
        for (kk, &j) in idx.iter().enumerate() {
            scores[j] = slice[kk];
        }
    }
    Ok(scores)
}

/// Old behavior (baseline): eval after EVERY chunk — a GPU barrier per sub-batch.
/// Kept for A/B comparison against deferred-eval `run`.
fn run_perchunk(model: &Model, tok: &Tokenizer, q: &str, docs: &[String], sub_batch: usize) -> Result<Vec<f32>> {
    let mut scores = vec![0f32; docs.len()];
    let texts: Vec<String> = docs.iter().map(|d| prompt(q, d)).collect();
    let encs = tokenize_batch(tok, texts).map_err(|e| anyhow!("tokenize: {e}"))?;
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_unstable_by_key(|&i| encs[i].get_ids().len().min(MAX_LEN));

    let mut ids_buf: Vec<i32> = Vec::with_capacity(sub_batch * MAX_LEN);
    let mut mask_buf: Vec<i32> = Vec::with_capacity(sub_batch * MAX_LEN);
    let mut i = 0;
    while i < order.len() {
        let idx: Vec<usize> = order[i..(i + sub_batch).min(order.len())].to_vec();
        let chunk_encs: Vec<&tokenizers::Encoding> = idx.iter().map(|&j| &encs[j]).collect();
        let (ids, mask) = pad_chunk(&chunk_encs, &mut ids_buf, &mut mask_buf);
        let sc = model.forward(&ids, &mask)?;
        mlx_rs::transforms::eval([&sc])?; // per-chunk barrier
        let host = sc.as_dtype(Dtype::Float32)?;
        let slice = host.as_slice::<f32>();
        for (kk, &j) in idx.iter().enumerate() {
            scores[j] = slice[kk];
        }
        i += sub_batch;
    }
    Ok(scores)
}

/// Variant: tokenize all docs upfront, then overlap per-chunk array-building (`pad_chunk`)
/// on a worker thread with GPU compute of the prior chunk (still deferred-eval at the end).
fn run_overlap(
    model: &Model,
    tok: &Tokenizer,
    q: &str,
    docs: &[String],
    sub_batch: usize,
) -> Result<Vec<f32>> {
    use std::sync::mpsc;
    let mut scores = vec![0f32; docs.len()];

    let texts: Vec<String> = docs.iter().map(|d| prompt(q, d)).collect();
    let encs = std::sync::Arc::new(tokenize_batch(tok, texts).map_err(|e| anyhow!("tokenize: {e}"))?);
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_unstable_by_key(|&i| encs[i].get_ids().len().min(MAX_LEN));

    // Pre-slice chunk index lists (cheap), then build padded arrays on a worker.
    let mut chunk_idx: Vec<Vec<usize>> = Vec::new();
    let mut i = 0;
    while i < order.len() {
        chunk_idx.push(order[i..(i + sub_batch).min(order.len())].to_vec());
        i += sub_batch;
    }

    let (tx, rx) = mpsc::sync_channel::<(usize, Array, Array)>(2);
    let encs_w = encs.clone();
    let chunk_idx_w = chunk_idx.clone();
    let worker = std::thread::spawn(move || {
        let mut ids_buf: Vec<i32> = Vec::with_capacity(sub_batch * MAX_LEN);
        let mut mask_buf: Vec<i32> = Vec::with_capacity(sub_batch * MAX_LEN);
        for (ci, idx) in chunk_idx_w.into_iter().enumerate() {
            let chunk_encs: Vec<&tokenizers::Encoding> = idx.iter().map(|&j| &encs_w[j]).collect();
            let (ids, mask) = pad_chunk(&chunk_encs, &mut ids_buf, &mut mask_buf);
            if tx.send((ci, ids, mask)).is_err() {
                break;
            }
        }
    });

    let n_chunks = chunk_idx.len();
    let mut sc_by_chunk: Vec<Option<Array>> = (0..n_chunks).map(|_| None).collect();
    for _ in 0..n_chunks {
        let (ci, ids, mask) = rx.recv().expect("worker channel closed");
        let sc = model.forward(&ids, &mask)?; // lazy
        sc_by_chunk[ci] = Some(sc);
    }
    worker.join().ok();

    let all: Vec<&Array> = sc_by_chunk.iter().map(|s| s.as_ref().unwrap()).collect();
    mlx_rs::transforms::eval(all.into_iter())?;

    for (ci, idx) in chunk_idx.iter().enumerate() {
        let sc = sc_by_chunk[ci].as_ref().unwrap();
        let host = sc.as_dtype(Dtype::Float32)?;
        let slice = host.as_slice::<f32>();
        for (kk, &j) in idx.iter().enumerate() {
            scores[j] = slice[kk];
        }
    }
    Ok(scores)
}

fn top3(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    idx.into_iter().take(3).collect()
}

// ---- HTTP /rerank server (Cohere/OpenRouter-compatible) ----

#[derive(serde::Deserialize)]
struct RerankRequest {
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<usize>,
}

#[derive(serde::Serialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

#[derive(serde::Serialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

/// Score documents for one rerank request, reusing the model forward.
/// Returns results sorted by relevance DESC, truncated to top_n (all if None).
fn rerank(
    engine: &Engine,
    tok: &Tokenizer,
    req: &RerankRequest,
) -> Result<RerankResponse> {
    if req.documents.is_empty() {
        return Ok(RerankResponse { results: vec![] });
    }
    let scores = engine.score(tok, &req.query, &req.documents)?;
    let mut results: Vec<RerankResult> = scores
        .iter()
        .enumerate()
        .map(|(index, &relevance_score)| RerankResult {
            index,
            relevance_score,
        })
        .collect();
    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(n) = req.top_n {
        results.truncate(n);
    }
    Ok(RerankResponse { results })
}

/// Synchronous, single-threaded blocking server: one request at a time (matches
/// the single GPU and the pipeline's cap-1 queue). Loads nothing here; model+tok
/// are already loaded by the caller.
fn serve_http(engine: &Engine, tok: &Tokenizer, port: u16) -> Result<()> {
    let addr = format!("127.0.0.1:{port}");
    let server = tiny_http::Server::http(&addr)
        .map_err(|e| anyhow!("failed to bind {addr}: {e}"))?;
    // Readiness gate line — exact format other tooling waits on.
    eprintln!("rerank server ready on http://127.0.0.1:{port}");

    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        let is_rerank = method == tiny_http::Method::Post && {
            // strip any query string
            let path = url.split('?').next().unwrap_or(&url);
            path == "/rerank"
        };

        if !is_rerank {
            let resp = tiny_http::Response::from_string("not found").with_status_code(404);
            let _ = request.respond(resp);
            continue;
        }

        let mut body = String::new();
        if std::io::Read::read_to_string(request.as_reader(), &mut body).is_err() {
            let resp = tiny_http::Response::from_string("bad request").with_status_code(400);
            let _ = request.respond(resp);
            continue;
        }

        let req: RerankRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                let resp = tiny_http::Response::from_string(format!("bad request: {e}"))
                    .with_status_code(400);
                let _ = request.respond(resp);
                continue;
            }
        };

        let response = match rerank(engine, tok, &req) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rerank error: {e}");
                let resp = tiny_http::Response::from_string(format!("internal error: {e}"))
                    .with_status_code(500);
                let _ = request.respond(resp);
                continue;
            }
        };

        let body = serde_json::to_string(&response)?;
        let header = tiny_http::Header::from_bytes(
            &b"Content-Type"[..],
            &b"application/json"[..],
        )
        .unwrap();
        let resp = tiny_http::Response::from_string(body).with_header(header);
        let _ = request.respond(resp);
    }
    Ok(())
}

fn main() -> Result<()> {
    // Pin MLX's default device to the Metal GPU so default-device ops use it too.
    Device::set_default(&Device::gpu());

    let args: Vec<String> = std::env::args().collect();
    let mut dump: Option<String> = None;
    let mut ndocs: Option<usize> = None;
    let mut sub_batch: usize = SUB_BATCH;
    let mut sweep = false;
    let mut overlap = false;
    let mut perchunk = false;
    let mut serve = false;
    let mut port: u16 = 8088;
    let mut a = 1;
    while a < args.len() {
        match args[a].as_str() {
            "--dump" => {
                dump = Some(
                    args.get(a + 1)
                        .ok_or_else(|| anyhow!("--dump needs a value"))?
                        .clone(),
                );
                a += 2;
            }
            "--ndocs" => {
                ndocs = Some(
                    args.get(a + 1)
                        .ok_or_else(|| anyhow!("--ndocs needs a value"))?
                        .parse()?,
                );
                a += 2;
            }
            "--subbatch" => {
                sub_batch = args
                    .get(a + 1)
                    .ok_or_else(|| anyhow!("--subbatch needs a value"))?
                    .parse()?;
                if sub_batch == 0 {
                    return Err(anyhow!("--subbatch must be >= 1"));
                }
                a += 2;
            }
            "--sweep" => {
                sweep = true;
                a += 1;
            }
            "--overlap" => {
                overlap = true;
                a += 1;
            }
            "--perchunk" => {
                perchunk = true;
                a += 1;
            }
            "--serve" => {
                serve = true;
                a += 1;
            }
            "--port" => {
                port = args
                    .get(a + 1)
                    .ok_or_else(|| anyhow!("--port needs a value"))?
                    .parse()?;
                a += 2;
            }
            _ => a += 1,
        }
    }

    // Model selection: RERANK_MODEL=<hf-repo-id> (default = Nemotron, back-compat).
    let repo_id = std::env::var("RERANK_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let dir = model_dir(&repo_id)?;
    let arch = detect_arch(&dir)?;
    eprintln!("model: {repo_id} ({arch:?})");
    eprintln!("model dir: {}", dir.display());
    let load_t = Instant::now();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let engine = match arch {
        Arch::Nemotron => Engine::Nemotron(Model::load(&dir)?),
        Arch::Kalm => Engine::Kalm(kalm::KalmModel::load(&dir, &tok)?),
    };
    eprintln!("loaded in {:?}", load_t.elapsed());

    if serve {
        return serve_http(&engine, &tok, port);
    }

    // The remaining CLI paths (dump/sweep/bench) read a fixed payload file and exercise
    // Nemotron-specific run variants. For KaLM, route everything through Engine::score.
    let model = match &engine {
        Engine::Nemotron(m) => Some(m),
        Engine::Kalm(_) => None,
    };

    // payload 0
    let line = std::fs::read_to_string("/tmp/rerank_payloads.jsonl")?
        .lines()
        .next()
        .ok_or_else(|| anyhow!("empty payloads"))?
        .to_string();
    let payload: serde_json::Value = serde_json::from_str(&line)?;
    let q = payload["query"].as_str().unwrap().to_string();
    let all_docs: Vec<String> = payload["documents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_str().unwrap().to_string())
        .collect();

    let n = ndocs.unwrap_or(if dump.is_some() { 3 } else { 100 });
    let docs: Vec<String> = all_docs.into_iter().take(n).collect();
    eprintln!("query: {q}");
    eprintln!("scoring {} docs", docs.len());

    if let Some(out) = dump {
        // Per-doc scores in ORIGINAL order (works for either backend via Engine::score).
        let scores = match model {
            Some(m) => run(m, &tok, &q, &docs, sub_batch)?,
            None => engine.score(&tok, &q, &docs)?,
        };
        let mut obj = serde_json::Map::new();
        obj.insert("query".into(), serde_json::json!(q));
        obj.insert("scores".into(), serde_json::json!(scores));
        std::fs::write(&out, serde_json::to_string_pretty(&obj)?)?;
        eprintln!("wrote {out}");
        for (i, s) in scores.iter().enumerate() {
            println!("doc[{i}] = {s:.6}");
        }
        return Ok(());
    }

    // Timing helper: warm once, then 8 timed runs of `f`. Returns (min,median,mean,top3).
    let bench = |label: &str, f: &dyn Fn() -> Result<Vec<f32>>| -> Result<(f64, f64, f64, Vec<usize>)> {
        let _ = f()?; // warm / trace
        let mut times = vec![];
        for _ in 0..8 {
            let t = Instant::now();
            let _ = f()?;
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let scores = f()?;
        let t3 = top3(&scores);
        let min = times[0];
        let median = times[times.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!(
            "  {label:<22} min={min:>6.0}ms median={median:>6.0}ms mean={mean:>6.0}ms  top3={t3:?}"
        );
        Ok((min, median, mean, t3))
    };

    // KaLM has a single forward path; run a plain bench through Engine::score.
    let model = match model {
        Some(m) => m,
        None => {
            println!("{repo_id} mlx-rs, payload 0, {} docs:", docs.len());
            let _ = bench("kalm", &|| engine.score(&tok, &q, &docs))?;
            return Ok(());
        }
    };

    if sweep {
        println!("Nemotron-1B mlx-rs sweep, payload 0, {} docs (deferred-eval):", docs.len());
        for &sb in &[100usize, 50, 25, 10] {
            let (_, _, _, _) = bench(&format!("subbatch={sb}"), &|| run(model, &tok, &q, &docs, sb))?;
        }
        // Bonus: deferred-eval + tokenization overlap at the default sub-batch.
        println!("  --- tokenization overlap (subbatch={sub_batch}) ---");
        let _ = bench("overlap", &|| run_overlap(model, &tok, &q, &docs, sub_batch))?;
        return Ok(());
    }

    let mode = if perchunk { ", per-chunk-eval" } else if overlap { ", overlap" } else { "" };
    println!(
        "Nemotron-1B mlx-rs, payload 0, {} docs (subbatch={sub_batch}{mode}):",
        docs.len()
    );
    if perchunk {
        let _ = bench("per-chunk-eval", &|| run_perchunk(model, &tok, &q, &docs, sub_batch))?;
    } else if overlap {
        let _ = bench("overlap", &|| run_overlap(model, &tok, &q, &docs, sub_batch))?;
    } else {
        let _ = bench("deferred-eval", &|| run(model, &tok, &q, &docs, sub_batch))?;
    }
    Ok(())
}
