---
title: "Quantization & Serving Benchmark Suite"
excerpt: "**Status: Planned.** A rigorous benchmark of FP16 vs INT8 vs INT4 (GPTQ/AWQ) across latency, throughput, and accuracy for open LLMs. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Every frontier lab has to decide how to serve models cheaply without wrecking quality — quantization is the lever they pull hardest. Being able to reason precisely about "how much accuracy do we lose for how much latency/cost gain" is a concrete, high-value skill, not a theoretical one.

## What to build
- Take one open model family (e.g. Llama 3 8B/70B or Qwen 2.5) and quantize it with:
  - Weight-only INT8
  - GPTQ (INT4)
  - AWQ (INT4)
- Benchmark each variant on:
  - Latency (TTFT, per-token latency) and throughput (tokens/sec at various batch sizes)
  - Memory footprint
  - Accuracy on a standard eval (e.g. MMLU subset, or perplexity on a held-out set) to quantify the actual quality cost
- Produce a single Pareto chart: accuracy vs. latency vs. memory, so the tradeoff is visual and immediate

## Key concepts to demonstrate
- Why weight-only quantization is easier than full activation quantization, and where each breaks down
- The mechanics of GPTQ (layer-wise reconstruction error minimization) vs. AWQ (activation-aware scaling) — not just "which library to call"
- How quantization error compounds differently across model sizes

## Suggested stack
`bitsandbytes`, `AutoGPTQ`, `AutoAWQ`; `lm-eval-harness` for accuracy evals; vLLM or TensorRT-LLM for serving each variant under identical load.

## Definition of done
A repo with reproducible benchmark scripts, a results table/chart for all variants, and a short write-up recommending which quantization scheme to use for which latency/accuracy budget.
