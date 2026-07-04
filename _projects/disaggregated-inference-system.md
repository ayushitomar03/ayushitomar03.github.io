---
title: "Disaggregated Prefill/Decode Inference System"
excerpt: "**Status: Planned.** A serving architecture that splits prefill and decode onto separate GPU pools to improve throughput, modeled after production frontier-lab serving stacks. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Disaggregated prefill/decode serving is one of the more advanced ideas frontier labs use in production (it's the core idea behind systems like DistServe and Splitwise, and reportedly used internally at the largest labs). Understanding — and better, implementing — why this split helps is a strong signal of systems depth beyond "I called the vLLM API."

## What to build
- A two-tier serving setup: a "prefill pool" of GPUs that handles the compute-bound prompt-processing phase, and a "decode pool" that handles the memory-bandwidth-bound token-generation phase
- A scheduler that routes incoming requests to prefill first, then hands off KV-cache state to a decode worker
- Benchmark against a baseline single-pool (non-disaggregated) setup on the same total GPU count, measuring throughput and per-request latency (especially tail latency / TTFT)

## Key concepts to demonstrate
- Why prefill (compute-bound, parallel over prompt tokens) and decode (memory-bandwidth-bound, sequential) have fundamentally different hardware utilization profiles
- Why colocating them on the same GPUs causes interference (long prefills blocking decode steps, hurting tail latency)
- How KV-cache has to be transferred or made accessible across the prefill/decode boundary, and what that costs

## Suggested stack
Build on vLLM or a custom minimal engine; NCCL or a simple network transfer for KV-cache handoff between pools; a load generator with mixed short/long prompts to expose the interference problem in the baseline.

## Definition of done
A working disaggregated setup with a benchmark showing improved tail latency and/or throughput vs. a non-disaggregated baseline on identical hardware, with a write-up explaining why.
