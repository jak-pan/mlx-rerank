# Benchmarks

How this engine got to **~1.43s / 100 docs** for `nvidia/llama-nemotron-rerank-1b-v2`
on an Apple M1 Max, what moved the needle, and — just as important — what didn't.

All numbers below were measured on the verified machine (M1 Max, 64 GB, macOS
26.5.1) on a real 100-document LongMemEval payload (the first record of
`/tmp/rerank_payloads.jsonl`), not synthetic uniform-length docs. You can
reproduce the headline table with `target/release/rerank --sweep`.

---

## Headline

| Stack | Latency / 100 docs | Notes |
|---|---|---|
| Cohere `rerank-4-fast` (remote API) | **~0.3 s** | Hosted; the latency bar. ~$0.001 / 1k queries. |
| **Nemotron-1B MLX (this engine)** | **~1.43 s** | sub_batch=10, length-sorted, deferred-eval. Local, free, offline. |
| Nemotron-1B MLX (pre-sweep) | ~1.93 s | sub_batch=25 — the earlier, larger-batch measurement. |
| Nemotron-1B PyTorch-MPS | **~5.5 s** | Same model, same payload, PyTorch's Metal backend. |

**This engine is ~5x faster than PyTorch-MPS** on the identical payload, and lands
within ~5x of a hosted API while running entirely on-device. The PyTorch number is
the reason this port exists: no off-the-shelf runtime loads this model on a Mac
with usable speed.

---

## The lever that worked: sub-batch size + length sorting

The model scores documents in **length-sorted sub-batches**. All docs are
tokenized once (a single parallel `encode_batch`), sorted by real token length,
then chunked into sub-batches that are each padded only to the longest member of
*that* chunk. Sorting first means short docs sit with short docs, so padding waste
stays small.

Sub-batch size is then a direct trade-off:

- **Too large** → every chunk pads up to its longest doc, so you spend GPU cycles
  on padding tokens that contribute nothing. Padding cost dominates.
- **Too small** → you launch many tiny forward passes, and per-launch GPU
  dispatch overhead dominates.

The sweep finds the **knee** between these two regimes:

```
Nemotron-1B mlx-rs sweep, payload 0, 100 docs (deferred-eval):
  subbatch=100   ~1.93s   (worst — maximal padding waste)
  subbatch=50    ...
  subbatch=25    ~1.93s   (the earlier "default")
  subbatch=10    ~1.43s   (optimum — padding-vs-launch knee)
```

**sub_batch=10 is the optimum (~1.43s / 100 docs)** — the sweet spot where
per-chunk padding waste and per-launch dispatch overhead balance out. That value
is baked in as the default (`SUB_BATCH = 10`) for both the CLI and the `/rerank`
server. `--subbatch <N>` overrides it; `--sweep` reproduces the table above.

---

## The levers that did NOT work (compute-bound forward)

The Nemotron forward is **compute-bound** on the Metal GPU — it runs near the
compute ceiling for this class of part. That single fact predicts which
optimizations are dead on arrival, and the measurements confirm it. Each of these
is still in the binary as an A/B mode so the negative result is reproducible, not
asserted.

### Deferred eval vs per-chunk eval — no-op

The default `run` builds **all** sub-batch score arrays lazily (no GPU barrier per
chunk), then calls `eval` over all of them **once** so MLX can pipeline the chunks.
The intuition was that removing the per-chunk barrier would let chunk *N+1* start
while chunk *N* drains.

`--perchunk` puts the barrier back (an `eval` after every sub-batch). Because the
forward is compute-bound, there is no idle time for pipelining to reclaim — the GPU
is already saturated chunk-to-chunk. **Deferred eval is a no-op against per-chunk
eval.** It's kept as the default because it's no *worse* and is the cleaner
structure, but it buys no speed.

### Tokenization / array-build overlap — no-op

`--overlap` moves the per-chunk array construction (`pad_chunk`) onto a worker
thread, streaming padded `(ids, mask)` arrays to the GPU loop over a bounded
channel, so CPU-side array building overlaps GPU compute of the prior chunk.

Same outcome: the GPU is the bottleneck, the CPU-side padding work is cheap and
already off the critical path, so overlapping it with compute reclaims nothing.
**No measurable improvement** over the plain deferred-eval loop.

### Quantization — a *loss* on Metal

Shrinking the weights (int8 / int4) is the classic memory-bandwidth win — but only
if you're memory-bound. This forward isn't. Across schemes (affine8 2.54s,
affine4 2.30s, mxfp8 3.10s, mxfp4 2.78s) every quantized variant ran **slower**
than fp16 (2.25s, isolated), some by 40–63%, and a couple even **reordered the
ranking** (i.e. changed the answers). Quantization is the wrong tool for a
compute-bound forward. This engine stays at f16.

### `mx.compile`, concurrency, precompute — within noise

From the broader investigation on the same model: `mx.compile` landed at +4%
(inside a ±15–20% noise band; its real value was *tightening variance* ~3x, not
speed), concurrency measured at 0.87x (a regression — single GPU), and an apparent
precompute win turned out to be a PyTorch-in-GPU-memory confound that vanished
under same-process measurement. **The only thing that survived clean measurement as
a real speed lever was length-sorting before batching.**

---

## Numerical validation (bit-faithful to the original)

Speed only matters if the scores are right. This MLX port was validated against the
PyTorch reference — that's the only thing that makes the rewrite trustworthy:

- **Single document:** P(yes) = **0.14934 (MLX)** vs **0.15001 (PyTorch)** →
  diff **0.0007**. Numerically identical; the gap is purely floating-point
  accumulation order.
- **Batched 100-doc ranking:** **top-3 identical** — `[2, 26, 0]` in both MLX and
  PyTorch.

It is the *same model*, not an approximation. `--dump <out.json>` writes per-doc
scores in original order for exactly this kind of reference diffing.

---

## Quality (for context)

Latency is half the story; the reranker also has to be accurate. On the 50Q
stratified, answer-only set (sampled to mirror the full LongMemEval-S category mix;
±6–7% noise at this N):

| Reranker | 50Q accuracy |
|---|---|
| Cohere `rerank-4-fast` (remote) | **92%** |
| **Nemotron-1B MLX (this engine)** | **92%** — 46/50, identical top-3 |
| No-rerank baseline | 88% |
| bge-reranker-v2-m3 (568M) | 86% (below baseline) |
| jina-reranker-v3 (0.6B) | 84% (below baseline) |

Nemotron-1B here **ties Cohere exactly** and is the only local model we've verified
at 92%. Turning reranking on moved the full 500Q benchmark **+3.34pp
(87.4% → 89.8%)** — the single largest reproducible lever in the system. You cannot
compress your way to this number; weaker rerankers land *below* the no-rerank
baseline.

---

## Reproduce

```sh
cargo build --release
target/release/rerank --sweep          # sub-batch table + overlap variant
target/release/rerank                  # default sub_batch=10, top-3 + timing
target/release/rerank --perchunk       # A/B: per-chunk barrier (no-op)
target/release/rerank --overlap        # A/B: tokenization overlap (no-op)
target/release/rerank --dump scores.json --ndocs 100   # numerical-validation dump
```

Each timing run warms once, then reports min / median / mean over 8 timed
iterations to keep the figures honest.

---

*Machine: MacBook Pro (MacBookPro18,4), Apple M1 Max, 10 cores (8P+2E), 64 GB
unified, macOS 26.5.1 (25F80). Toolchain: rustc 1.93.0; mlx-rs 0.25; tokenizers
0.20; tiny_http 0.12; release opt-level 3.*
