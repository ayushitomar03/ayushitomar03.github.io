---
title: 'Disaggregated Prefill/Decode Serving: Why Splitting Inference Stages Improves Throughput'
date: 2026-07-09
permalink: /blog/disaggregated-prefill-decode/
tags:
  - inference
  - llm-serving
---

**Status:** 🔜 Planned

## Why write this
This is a more advanced serving idea than continuous batching, and fewer people can speak to it fluently — which makes it a good differentiator in interviews once you've built the [disaggregated inference project](/projects/disaggregated-inference-system/). It also shows you're tracking current production techniques, not just textbook material.

## Outline
1. Recap: prefill (processing the prompt, compute-bound, highly parallel) vs. decode (generating tokens one at a time, memory-bandwidth-bound)
2. The interference problem: on a shared GPU pool, a long prefill request blocks decode steps for other requests, spiking tail latency
3. The disaggregation idea: separate GPU pools for prefill and decode, with KV-cache handed off between them
4. What handing off KV-cache actually costs (network transfer, and why it's still worth it)
5. Your own benchmark: tail latency and throughput, disaggregated vs. colocated, same total GPU count

## Key points to nail
- Be precise about *why* prefill and decode have different hardware utilization profiles (compute-bound vs. memory-bandwidth-bound) — this is the crux of the whole argument
- Don't hand-wave the KV-cache transfer cost; quantify it
- Present your benchmark results even if they're modest — a real (if small) measured result beats a purely theoretical explanation

## Reference reading
- DistServe paper (disaggregated prefill/decode serving)
- Splitwise paper (Microsoft's take on the same idea)
