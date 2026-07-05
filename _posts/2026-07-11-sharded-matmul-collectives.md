---
title: 'Sharded Matrix Multiplication: AllGather, AllReduce, and AllToAll Compared'
date: 2026-07-11
permalink: /blog/sharded-matmul-collectives/
tags:
  - distributed-systems
  - hardware
---

**Status:** 🔜 Planned

## Why write this
"Just all-reduce the gradients" hides a surprising amount of design space — which collective you use, and how you shard your inputs, changes communication cost by multiples. This is the write-up for the [sharded matmul & collectives benchmark project](/projects/sharded-matmul-collectives-benchmark/), grounded in the [sharding chapter](https://jax-ml.github.io/scaling-book/sharding/) of *How to Scale Your Model*.

## Outline
1. Setup: what it means for a matrix to be "sharded" across a device mesh, and the notation the scaling book uses
2. AllGather: when you need the full array on every device, and its latency- vs. bandwidth-bound regimes (Q2/Q3)
3. Two ways to do a sharded matmul — gather-then-multiply vs. multiply-then-reduce — and why the "right" one depends on shape (Q4/Q9)
4. AllToAll vs. AllGather: derive (and then measure) the ~4x cost advantage AllToAll can have, and when it applies (Q10)
5. Your own benchmark numbers across all four core collectives, measured vs. theoretical

## Key points to nail
- Make the mesh/sharding notation concrete with a worked example, not just abstract symbols
- Be honest about where your measured numbers diverged from the theoretical prediction, and hypothesize why (network topology, framework overhead)
- Tie this back to a real system: why does this matter for training a model that doesn't fit on one device?

## Reference reading
- *How to Scale Your Model*, Chapter 3: ["Sharded Matrices and How to Multiply Them"](https://jax-ml.github.io/scaling-book/sharding/) (Google DeepMind)
