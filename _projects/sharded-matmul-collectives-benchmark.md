---
title: "Sharded Matmul & Communication Collectives Benchmark"
excerpt: "**Status: Planned.** Implement and benchmark AllGather, AllReduce, and AllToAll in JAX, reproducing the communication-cost exercises from 'How to Scale Your Model.' <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Every distributed training or serving system is bottlenecked by communication as much as compute. The chapter on [sharded matrices](https://jax-ml.github.io/scaling-book/sharding/) in *How to Scale Your Model* is the clearest existing treatment of *why* — reproducing it in code (not just reading it) is what turns "I know AllReduce exists" into "I can predict its cost and pick the right collective for the job."

## What to build
Using JAX's `shard_map` or `pmap` across whatever devices you have (multi-GPU, or simulated multi-device via `XLA_FLAGS=--xla_force_host_platform_device_count=N` on CPU):
- Q2/Q3: measure AllGather latency across different sharding patterns and array sizes, including the small-array latency-bound regime
- Q4/Q9: implement two different sharded matmul strategies (AllGather-then-multiply vs. multiply-then-AllReduce/outer-product) and compare their measured FLOPs and communication cost
- Q8 [challenge]: benchmark all four core communication primitives (AllGather, AllReduce, ReduceScatter, AllToAll) end-to-end
- Q10: measure AllToAll vs. AllGather cost directly and check it against the theoretical ~4x advantage the chapter derives

## Key concepts to demonstrate
- Why the "right" sharded matmul strategy depends on array shape and mesh topology, not a fixed rule
- The difference between bandwidth-bound and latency-bound communication regimes
- How to read a communication cost formula and turn it into a testable prediction

## Suggested stack
JAX with `shard_map`; a multi-GPU box or simulated multi-device CPU setup if that's what's available; simple timing harness with warmup + multiple trials.

## Definition of done
A repo with working implementations of all four collectives, measured timings across several array sizes/device counts, and a comparison table of measured vs. theoretical cost for each.
