---
title: 'Sizing KV Caches and Choosing Inference Sharding, From First Principles'
date: 2026-07-12
permalink: /blog/sizing-kv-caches-for-inference/
tags:
  - inference
  - llm-serving
excerpt: "Writeup pending. The sizing calculator is built — KV bytes per token, max batch size at a given sequence length, and the memory-bound decode latency floor, each checked against measured numbers."
published: false
---

<p class="pill-row">
  <span class="pill pill--built">Tool built</span>
  <span class="pill pill--progress">Writeup pending</span>
</p>

The [calculator](/projects/inference-roofline-kv-cache-calculator/) answers the sizing questions
that come *before* you can decide how to shard a model for serving. Sample output for
Qwen2.5-7B on one L40S:

- KV cache per token (int8): **28,672 bytes**
- Max batch size at seq_len 2,048: **481** · at 8,192: **120** · at 32,768: **30**
- Memory-bound decode-step latency floor (bf16 weights): **17.63 ms**

The measured steady-state decode step on the same box came in at ~17–25 ms/token with
batching — the same order of magnitude as the theoretical floor, which indicates the system is
memory-bound rather than losing time somewhere unaccounted for.

The full walkthrough, including the MoE variant and the prefill-vs-decode sharding tradeoff with
ICI communication cost, is still to write.
