---
title: "JAX/XLA Profiling: Optimizing a Mock Transformer Layer"
excerpt: "**Status: Planned.** Reproduce the profiling chapter's exercise from 'How to Scale Your Model': profile a mock transformer, find the sharding bug, and close the gap to the roofline. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Knowing the theory (rooflines, sharding, MFU) is table stakes; being able to open an actual profiler trace and diagnose *why* a real kernel is slow is the skill that infra teams actually hire for day to day. The [profiling chapter](https://jax-ml.github.io/scaling-book/profiling/) of *How to Scale Your Model* poses exactly this as a hands-on exercise.

## What to build
- Implement the chapter's mock transformer layer benchmark in JAX and capture a profile with the JAX/XLA profiler
- Diagnose, from the profile alone (before looking at the code, per the chapter's own instructions):
  - What the true shapes of each matrix are and how they're sharded
  - How much time is spent in attention vs. MLP
  - What sharding strategy is actually being used
- Compare against the chapter's reported baseline (~184ms/layer) and see how close you can get to their optimized number (~80-90ms/layer) by fixing the sharding
- Separately, analyze the provided "unknown computation" profile (Question 1) and write up how you reverse-engineered the two matmuls and their 8-way model-parallel sharding from the trace alone

## Key concepts to demonstrate
- How to read a JAX/XLA trace: identifying compute ops vs. communication ops vs. idle/waiting time
- Connecting a profile back to the roofline model: what fraction of time *should* be spent on each op, and where the actual trace diverges
- The specific, common bug class this exercise is built around: a sharding misconfiguration that silently kills MXU utilization

## Suggested stack
JAX + `jax.profiler`; TensorBoard or Perfetto for trace visualization; a single-host multi-device setup (real or simulated) is enough to reproduce the exercise.

## Definition of done
A write-up (with annotated profile screenshots) showing the before/after per-layer latency, the sharding bug found and fixed, and the achieved MXU/FLOPs utilization relative to the roofline bound.
