# mlx-rerank

A fast, in-process **MLX cross-encoder reranker server** for Apple Silicon. No
PyTorch in the process — weights, tokenizer, and the full forward pass are
hand-built in Rust on top of [`mlx-rs`](https://crates.io/crates/mlx-rs).

It runs strong rerankers natively on Metal behind one HTTP
`/rerank` server**, picked at startup via `RERANK_MODEL` (see *Model selection*
below):

- **NVIDIA [`llama-nemotron-rerank-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2)** — a Llama-3.2 bidirectional cross-encoder.
- **[`KaLM-Reranker-V1`](https://huggingface.co/KaLM-Embedding)** (Nano + Small) — a t5gemma2 encoder-decoder reranker (FBNL).

Give it a query and candidate docs; it scores each `(query, doc)` pair and returns
them reordered best-first. Ships as a CLI (benchmarking / one-shot scoring) and the
server.

> Companion MLX model cards:
> [`llama-nemotron-rerank-1b-v2-mlx`](https://huggingface.co/jak-pan/llama-nemotron-rerank-1b-v2-mlx) ·
> [`kalm-reranker-v1-nano-mlx`](https://huggingface.co/jak-pan/kalm-reranker-v1-nano-mlx) ·
> [`kalm-reranker-v1-small-mlx`](https://huggingface.co/jak-pan/kalm-reranker-v1-small-mlx)

---

## Why this exists

Reranking is one of the highest-leverage stages in a retrieval pipeline: a strong
cross-encoder rescores the first-stage candidates so the most relevant evidence
lands at the top. To be worth running on *every* query it has to be both
**accurate** and **cheap**.

The catch is running a genuinely strong ~1B cross-encoder *locally*. There is no
off-the-shelf runtime that loads this model on a Mac with any speed. The only
options were PyTorch-MPS (works, but ~5.5s end-to-end for Nemotron — far too slow)
or building the inference path by hand in MLX. This is that hand-built path.

What you get:

- **~5x faster than PyTorch.** ~1.43s / 100 docs here vs ~5.5s for Nemotron under
  PyTorch-MPS, on the identical payload. Full numbers and the optimization story
  are in [BENCHMARKS.md](BENCHMARKS.md).
- **Bit-faithful to the original weights.** Validated against the PyTorch
  reference: single-doc P(yes) = 0.14934 (MLX) vs 0.15001 (PyTorch), diff 0.0007;
  batched 100-doc top-3 identical (`[2, 26, 0]`) in both. The gap is only
  floating-point accumulation order — it is the *same model*.

---

## What it is (architecture)

Nemotron-1B is a standard **Llama-3.2-1B** backbone (16 layers, hidden 2048, 32
attention heads, 8 KV heads, head_dim 64, RoPE θ=500000 with llama3 scaling) run
as a **bidirectional** sequence-classification model:

- attention is **non-causal** — the causal mask is replaced by a padding-only
  bidirectional mask, so every token attends to every other token (encoder-style);
- pooling is **masked mean** over all non-pad tokens → one vector per
  `(query, doc)` pair;
- a **linear score head** `[1, 2048]` projects that vector to a single relevance
  logit.

The prompt format is `question:{q} \n \n passage:{p}`. Weights are loaded once
(cast to f16, matching the Python reference), the tokenizer comes straight from
the HF snapshot, and everything runs on the Metal GPU.

The model is read from your local HuggingFace cache:
`~/.cache/huggingface/hub/models--nvidia--llama-nemotron-rerank-1b-v2/snapshots/<hash>/`
(weights, `tokenizer.json`). Make sure the model is downloaded before running.

---

## Model selection (`RERANK_MODEL`)

The same binary can run a second architecture: **KaLM-Reranker-V1** (a `t5gemma2`
encoder-decoder reranker, sizes Nano and Small). Select the checkpoint at startup
with the `RERANK_MODEL` environment variable (an HF repo id); it defaults to
Nemotron for back-compatibility:

```sh
# Nemotron (default — unchanged behavior)
target/release/rerank --serve

# KaLM-Reranker-V1-Small / Nano
RERANK_MODEL=KaLM-Embedding/KaLM-Reranker-V1-Small target/release/rerank --serve
RERANK_MODEL=KaLM-Embedding/KaLM-Reranker-V1-Nano  target/release/rerank --serve
```

One process loads exactly one model. The architecture is auto-detected from the
snapshot's `config.json` (`model_type: llama_bidirec` → Nemotron;
`t5gemma2`/`T5Gemma2ForConditionalGeneration` → KaLM), and `model_dir` resolves
`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*` for any repo id. The
CLI surface (`--serve`, `--dump`, etc.) is identical for whichever model is
selected.

The KaLM path is a torch-free Rust/mlx-rs port of the reference
`modeling_kalm_mlx.py` (Gemma2 primitives — QK-norm, GQA, gelu-tanh gated MLP,
sandwich norms, per-layer full/sliding RoPE — plus chunk-pool encoding and a
merged self+cross-attention decoder). It is config-generalized, so the one
implementation serves every size in the family. Validated bit-faithful against the
reference: top-5 ranking identical and per-doc P(yes) within bf16 tolerance
(max abs diff 0.008 / Small, 0.002 / Nano on a 15-doc set).

---

## Build

```sh
cargo build --release
```

Produces a single binary, `target/release/rerank`. The only system requirement is
Apple Silicon + Metal (this is an MLX program); see the footer for the verified
toolchain.

---

## CLI

The default invocation scores a fixed sample payload (the first record of
`/tmp/rerank_payloads.jsonl`) and prints the top-3 plus timing — it's the
benchmarking entry point.

```sh
# Score 100 docs of the sample payload, print top-3 + min/median/mean timing.
target/release/rerank

# Limit the number of docs scored (default 100).
target/release/rerank --ndocs 50

# Override the length-sorted sub-batch size (default 10 — the swept optimum).
target/release/rerank --subbatch 25

# Sweep sub-batch sizes (100, 50, 25, 10) and report timing for each,
# plus the tokenization-overlap variant. Reproduces BENCHMARKS.md.
target/release/rerank --sweep

# Dump per-doc scores in ORIGINAL order to a JSON file (numerical-validation /
# reference-diff path). Default --ndocs is 3 in this mode.
target/release/rerank --dump scores.json --ndocs 100
```

| Flag | Default | Effect |
|---|---|---|
| `--ndocs <N>` | 100 (3 with `--dump`) | Number of docs to score from the sample payload. |
| `--subbatch <N>` | 10 | Length-sorted sub-batch size fed to each forward pass. |
| `--sweep` | off | Sweep sub-batch sizes + overlap variant; print a timing table. |
| `--dump <out.json>` | off | Write per-doc scores (original order) for reference diffing. |
| `--serve` | off | Start the HTTP `/rerank` server instead of running the bench. |
| `--port <P>` | 8088 | Port for `--serve`. |
| `--overlap` / `--perchunk` | off | A/B variants of the scoring loop (see BENCHMARKS.md). |

> `--overlap` (tokenization/array-build overlapped on a worker thread) and
> `--perchunk` (a GPU barrier after every sub-batch) exist to demonstrate that
> those strategies are **no-ops or regressions** — this forward is compute-bound.
> The default deferred-eval path is the fast one. Details in
> [BENCHMARKS.md](BENCHMARKS.md).

---

## The `--serve` HTTP server

```sh
target/release/rerank --serve            # binds 127.0.0.1:8088
target/release/rerank --serve --port 9000
```

On startup it loads the model once, then prints a readiness line that downstream
tooling waits on:

```
rerank server ready on http://127.0.0.1:8088
```

It exposes a **Cohere-compatible** `POST /rerank` endpoint. Request:

```json
{
  "query": "what did I order for the commute setup?",
  "documents": ["...doc 0...", "...doc 1...", "..."],
  "top_n": 10
}
```

(`model` is accepted and ignored; `top_n` is optional — omit it to get all docs
back.) Response — results sorted by `relevance_score` descending, truncated to
`top_n`:

```json
{
  "results": [
    { "index": 2,  "relevance_score": 10.31 },
    { "index": 26, "relevance_score":  9.84 },
    { "index": 0,  "relevance_score":  8.12 }
  ]
}
```

```sh
curl -s http://127.0.0.1:8088/rerank \
  -H 'content-type: application/json' \
  -d '{"query":"...","documents":["a","b","c"],"top_n":3}'
```

**Single-threaded by design.** The server handles one request at a time. There is
exactly one Metal GPU, and the reranker forward is compute-bound on it — running
requests concurrently would just contend for the same device and add no
throughput (concurrency measured at 0.87x). One request at a time matches the
single GPU and the pipeline's cap-1 queue. Any non-`POST /rerank` route returns
404; malformed bodies return 400; scoring errors return 500.

---

## Pointing a pipeline at it

The server speaks the same `/rerank` shape as Cohere/OpenRouter, so any pipeline
that talks to a Cohere-style reranker base URL can use it unchanged — just point
that base URL at the local server:

```sh
export RERANK_BASE_URL=http://127.0.0.1:8088
```

The pipeline then POSTs `{query, documents, top_n}` to `${RERANK_BASE_URL}/rerank`
and reads back `{results: [{index, relevance_score}, ...]}` — no code changes,
no API key, no per-query cost, fully offline. Start the server first (wait for the
`rerank server ready` line), then run the pipeline.

---

## See also

- [BENCHMARKS.md](BENCHMARKS.md) — the optimization story and all the numbers
  (sub-batch sweep, the padding-vs-launch knee, the dead levers, PyTorch/Cohere
  comparison, numerical validation).
- [`llama-nemotron-rerank-1b-v2-mlx`](https://huggingface.co/jak-pan/llama-nemotron-rerank-1b-v2-mlx) —
  the model card / weights-side documentation for the MLX port.

---

## Environment (verified)

- **Machine:** MacBook Pro (MacBookPro18,4), Apple M1 Max, 10 CPU cores
  (8P + 2E), 64 GB unified memory, macOS 26.5.1 (25F80). Apple Silicon
  unified-memory GPU driven via Metal; MLX runs natively on it.
- **Toolchain:** rustc 1.93.0 (254b59607 2026-01-19); `mlx-rs` 0.25;
  `tokenizers` 0.20 (`onig`); `serde` 1 / `serde_json` 1 / `anyhow` 1 /
  `tiny_http` 0.12; release `opt-level = 3`.

All speed figures above were measured on this machine.

---

## License & credits

- **This engine** (the Rust MLX forward pass + server) is released under
  **Apache-2.0** (see [`LICENSE`](LICENSE)).
- **The model weights are NVIDIA's** —
  [`nvidia/llama-nemotron-rerank-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2),
  loaded from your local Hugging Face cache at runtime. **No weights are
  redistributed in this repo**; they are governed by NVIDIA's license for that
  model — verify the terms on the model page before relying on it.
- Built on [`mlx-rs`](https://crates.io/crates/mlx-rs) and
  [`tokenizers`](https://crates.io/crates/tokenizers); thanks to the
  [MLX](https://github.com/ml-explore/mlx) team at Apple.
- Companion model card:
  [`jak-pan/llama-nemotron-rerank-1b-v2-mlx`](https://huggingface.co/jak-pan/llama-nemotron-rerank-1b-v2-mlx).
