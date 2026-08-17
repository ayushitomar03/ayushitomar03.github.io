---
title: "Quantization & Serving Benchmark Suite"
excerpt: "Built and benchmarked. FP16 vs INT8 RTN vs GPTQ INT4 vs AWQ INT4 on Qwen2.5-7B — five accuracy tasks, four concurrency levels. 2.7× throughput for 0.84 points of MMLU, and one variant whose tail latency quietly gets worse under load."
collection: projects
order: 4
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">Qwen2.5-7B-Instruct</span>
  <span class="pill">L40S 46GB</span>
  <span class="pill">vLLM · GPTQ · AWQ</span>
</p>

Three quantized variants of `Qwen2.5-7B-Instruct` benchmarked end to end against an fp16
baseline — not one accuracy number and one latency number, but five eval tasks across four
concurrency levels, because both halves of the tradeoff change shape with load.

## What was built

- **Three quantization pipelines** — INT8 round-to-nearest (W8A16, no calibration, as a floor),
  GPTQ INT4 (W4A16), and AWQ INT4 (W4A16 asymmetric).
- **An accuracy suite spanning task types** — MMLU, ARC-Easy, and HellaSwag (multiple choice),
  GSM8K (multi-step generation), and wikitext perplexity. The spread is deliberate:
  multiple-choice benchmarks can hide damage that perplexity catches, since a model can still
  rank the right answer first while its underlying distribution has gotten noticeably noisier.
- **A serving sweep at concurrency 1/4/16/32** via vLLM, tracking TTFT, TPOT, and throughput —
  measuring at one load level would hide how each scheme actually behaves.

## Results

| | fp16 | INT8 RTN | GPTQ INT4 | AWQ INT4 |
|---|---|---|---|---|
| Size | 15.24 GB | 8.82 GB | 5.56 GB | 5.58 GB |
| MMLU | 71.76 | 71.66 | 70.92 | 70.94 |
| Wikitext PPL ↓ | 9.408 | 9.409 | 10.028 | 10.026 |
| Throughput @ c=32 | 1,163 tok/s | 2,075 tok/s | **3,126 tok/s** | 3,102 tok/s |
| Quantization time | — | — | 17.2 min | **9.5 min** |

INT4 buys **2.7× throughput for 0.84 points of MMLU**. GPTQ and AWQ are statistically
indistinguishable on quality — so the reason to pick AWQ is that it reaches the same place in
**half the wall clock**, not accuracy or serving speed.

The result I'd have missed by reporting throughput alone: **INT8 RTN's TTFT p99 at concurrency
32 is worse than fp16's** (613 ms vs 373 ms) despite 1.8× the throughput. If tail latency under
load matters, it's a weak pick.

[Read the full writeup →](/blog/quantization-deep-dive/)

## Limitations

One 7B model on one L40S. Larger-scale behaviour is estimated directionally, not measured, and
the calibration set size (512 samples) was never tested for sensitivity.
