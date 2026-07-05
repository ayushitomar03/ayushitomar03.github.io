---
title: "Roofline Modeling & Benchmarking Toolkit"
excerpt: "**Status: Planned.** Solve and code up the Roofline chapter's exercises from 'How to Scale Your Model,' then validate the predictions against real GPU benchmarks. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Roofline analysis (compute-bound vs. memory-bandwidth-bound) is the single mental model that underlies every serving and training optimization decision at a frontier lab. Being able to predict *before you run anything* whether a kernel will be compute- or memory-bound — and being right — is a concrete signal of systems maturity.

## What to build
Work through the exercises in ["A Brief Intro to Roofline Analysis"](https://jax-ml.github.io/scaling-book/roofline/) from Google DeepMind's *How to Scale Your Model*, then turn the math into reusable code:
- Q1/Q4: a calculator for bytes loaded/written, total FLOPs, arithmetic intensity, and predicted runtime for int8 matmuls of arbitrary shape
- Q2: derive (and code) the batch-size threshold where a mixed int8-weight/bf16-activation matmul flips from memory-bound to compute-bound
- Q3: generate the actual roofline plot (peak FLOPs/s vs. batch size) for `F = D = 4096` and `F = D = 1024`
- Q5: repeat the memory-roofline analysis for an H100 GPU (using its published specs, including structured sparsity claims) instead of a TPU
- Validate: benchmark real matmuls at several shapes/batch sizes on whatever GPU you have access to, and check how well your predicted crossover point matches the measured one

## Key concepts to demonstrate
- Arithmetic intensity as the single number that determines compute- vs. memory-bound
- Why the "roofline" is two lines (a memory-bandwidth line and a peak-FLOPs line) and the crossover point is the batch size that matters
- Where real hardware deviates from the idealized roofline (kernel launch overhead, imperfect utilization)

## Suggested stack
Plain Python/NumPy for the calculator; PyTorch or JAX for the validation benchmarks; matplotlib for the roofline plots.

## Definition of done
A small library/notebook that reproduces all five exercises with code (not just algebra), plus a validation section showing measured vs. predicted crossover batch size on real hardware.
