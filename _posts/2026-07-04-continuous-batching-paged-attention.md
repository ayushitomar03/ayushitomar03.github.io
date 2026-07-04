---
title: 'Continuous Batching and PagedAttention: How Modern LLM Servers Hit High Throughput'
date: 2026-07-04
permalink: /blog/continuous-batching-paged-attention/
tags:
  - inference
  - llm-serving
---

**Status:** 🔜 Planned

## Why write this
This is the single most-asked-about idea in LLM serving interviews, and most people can gesture at "vLLM is fast" without being able to explain the actual mechanism. Writing this well — in your own words, ideally alongside your [inference server project](/projects/llm-inference-server/) — proves real understanding.

## Outline
1. The naive baseline: static batching and why it wastes GPU cycles (padding to the longest sequence, batch can't start until all requests arrive)
2. Continuous batching: adding/removing requests from a running batch at every decode step
3. The KV-cache memory problem this creates — variable-length caches, fragmentation
4. PagedAttention: block-based KV-cache allocation borrowed from OS virtual memory paging
5. Put numbers on it: throughput improvement continuous batching + paged KV-cache gives over naive batching (cite vLLM paper numbers, ideally back with your own benchmark)

## Key points to nail
- Explain *why* padding to the longest sequence in a batch is wasteful, with a concrete example
- Be precise about what "page" means here (fixed-size KV-cache blocks, not virtual memory pages) — this is the detail that separates a real understanding from a surface one
- Tie it back to a metric that matters: throughput (tokens/sec) and GPU memory utilization

## Reference reading
- vLLM paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Orca paper (continuous batching origin)
