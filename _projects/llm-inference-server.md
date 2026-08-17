---
title: "High-Throughput LLM Inference Server"
excerpt: "Partially built. The engine core, batched decode, and a paged block-based KV cache exist inside the disaggregated inference project — what's missing is the CUDA kernel that would make paging actually pay off."
collection: projects
order: 8
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Core built</span>
  <span class="pill pill--progress">Standalone server pending</span>
</p>

A from-scratch serving engine, benchmarked against vLLM on identical hardware.

## What exists

Most of the engine core was built inside the
[disaggregated inference project](/projects/disaggregated-inference-system/): a request queue,
batched decode that merges concurrent requests into a single forward call per tick, manual
KV-cache management, and a paged/block-based cache with per-request block tables — validated
token-for-token including multi-block-boundary stress cases.

It was benchmarked against vLLM at TP=2 on the same hardware and the same workload. The honest
headline: vLLM's inter-token p99 was **~30× better** (15.6 ms vs 1,671.6 ms). That gap is the
most useful number the project produced — same architectural ideas, orders-of-magnitude
different constant factor, because vLLM's batching is a fused kernel and mine was a Python loop
over tensor slices.

## What's missing

The paged cache, once correct, was *slower* than the naive version it replaced — reconstructing
a request's cache from block storage costs `seq_len / block_size` slice-and-concat operations
per layer, each paying fixed Python and launch overhead. Making paging pay off requires writing
the CUDA kernel that operates directly on block memory, which is the actual work here and
hasn't been done.

Also pending: a standalone HTTP/streaming serving layer rather than in-process calls.
