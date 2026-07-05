---
title: 'Sizing KV Caches and Choosing Inference Sharding, Worked From First Principles'
date: 2026-07-12
permalink: /blog/sizing-kv-caches-for-inference/
tags:
  - inference
  - llm-serving
---

**Status:** 🔜 Planned

## Why write this
"How big will the KV cache be, and how many requests can I actually batch?" is a question every inference infra team has to answer before writing any serving code — and most people either hand-wave it or get it wrong by an order of magnitude. This pairs with the [inference roofline & KV cache calculator project](/projects/inference-roofline-kv-cache-calculator/), based on the [inference chapter](https://jax-ml.github.io/scaling-book/inference/) of *How to Scale Your Model*.

## Outline
1. The KV cache size formula: per-token cost as a function of layers, heads, head dimension, and precision — work it out for a real model size (Q1)
2. From cache size to max batch size at a given sequence length and chip memory budget (Q2)
3. Parameter loading time as a latency floor: why "time to load weights from HBM" sets a hard lower bound on per-step latency regardless of how fast the compute is (Q3)
4. Designing prefill vs. decode sharding separately, and why the ICI (interconnect) bandwidth shows up directly in the answer (Q4)
5. The Mixture-of-Experts twist: how total vs. activated parameters change the FLOPs-bound batch size and expert-sharding weight-loading time (Q5/Q6)
6. When 2D weight-stationary sharding beats standard 3D model sharding, and why (Q7)

## Key points to nail
- Use one consistent worked example (a specific model size and chip) all the way through, so the numbers build on each other
- Make the MoE section a real comparison against the dense case, not a separate disconnected example
- If you validated any of this against a real served model, include the actual vs. predicted numbers

## Reference reading
- *How to Scale Your Model*, Chapter 7: ["All About Transformer Inference"](https://jax-ml.github.io/scaling-book/inference/) (Google DeepMind)
