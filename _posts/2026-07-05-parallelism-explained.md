---
title: 'Anatomy of a Distributed Training Run: Data, Tensor, and Pipeline Parallelism'
date: 2026-07-05
permalink: /blog/parallelism-explained/
tags:
  - training-infra
  - distributed-systems
---

**Status:** 🔜 Planned

## Why write this
Frontier lab infra interviews probe hard on "how would you train a model that doesn't fit on one GPU." Most candidates know the names of the parallelism strategies but can't explain the actual communication pattern each one requires — that gap is exactly what this post should close, for you and for readers.

## Outline
1. The core problem: a model (and its optimizer states) can be bigger than one GPU's memory, or training would just be too slow on one GPU
2. Data parallelism: replicate the model, split the batch, all-reduce gradients — simplest, but doesn't help if the model itself doesn't fit
3. Tensor parallelism: split individual layers/matrices across GPUs — needed for models that don't fit even one layer's worth of activations comfortably, at the cost of heavy communication per layer
4. Pipeline parallelism: split the model by layer across GPUs, pass activations forward — cheaper communication, but the "bubble" (idle time) problem
5. How real systems combine all three (3D parallelism) and why the combination is necessary at frontier scale, not just theoretically interesting

## Key points to nail
- Draw (literally, with a diagram) the communication pattern for each strategy — this is what separates "I've heard of it" from "I understand it"
- Explain the pipeline bubble problem concretely and how micro-batching (GPipe-style) mitigates it
- Connect this to your [training orchestrator project](/projects/distributed-training-orchestrator/) with real numbers if you have them

## Reference reading
- Megatron-LM paper (tensor + pipeline parallelism)
- GPipe paper (pipeline parallelism, bubble problem)
- ZeRO paper (memory-efficient data parallelism)
