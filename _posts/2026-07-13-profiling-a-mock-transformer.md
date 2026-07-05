---
title: 'What a Profiler Trace Actually Tells You: Optimizing a Mock Transformer Layer'
date: 2026-07-13
permalink: /blog/profiling-a-mock-transformer/
tags:
  - performance
  - debugging
---

**Status:** 🔜 Planned

## Why write this
This is the "show, don't tell" post — instead of explaining rooflines and sharding in the abstract, walk through an actual profiler trace and diagnose a real performance bug live, the way you'd do it on the job. Pairs with the [TPU/XLA profiling project](/projects/tpu-profiling-mock-transformer/), based on the [profiling chapter](https://jax-ml.github.io/scaling-book/profiling/) of *How to Scale Your Model*.

## Outline
1. Set the challenge: given only a profiler trace (no code), figure out the matrix shapes, the sharding strategy, and what fraction of time goes to attention vs. MLP
2. Walk through your own reasoning process on the "unknown computation" exercise — what clues in the trace gave away the 8-way model-parallel sharding
3. The mock transformer benchmark: your baseline latency per layer, and the specific sharding misconfiguration you found
4. The fix, and the before/after numbers (aim to show your own version of the chapter's 184ms → 80-90ms improvement)
5. Close the loop: compare your optimized latency against the roofline bound from your [roofline toolkit](/projects/roofline-modeling-toolkit/) — how close to theoretically optimal did you get, and what's left on the table

## Key points to nail
- Include actual trace screenshots/exports, not just prose descriptions — this post's credibility rests on showing the real artifact
- Be honest about wrong turns in your diagnosis before you found the real bug; that's more instructive than a clean narrative
- End with a concrete number: measured MXU/FLOPs utilization vs. the roofline-predicted ceiling

## Reference reading
- *How to Scale Your Model*, Chapter 9: ["How to Profile TPU Programs"](https://jax-ml.github.io/scaling-book/profiling/) (Google DeepMind)
