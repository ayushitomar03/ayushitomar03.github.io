---
title: "Sharded Matmul & Communication Collectives Benchmark"
excerpt: "Planned. Measuring what each collective actually costs across sharding patterns and array sizes — starting from the one all-reduce measurement that's already load-bearing elsewhere."
collection: projects
order: 9
status: Upcoming
---

<p class="pill-row"><span class="pill pill--upcoming">Upcoming</span></p>

Every distributed training or serving system is bottlenecked by communication as much as
compute, and which collective you use — plus how you shard the inputs — changes the cost by
multiples.

## Where it stands

One measurement exists so far: a **200 MB NCCL all-reduce at ~12.3 ms steady state** on a PCIe
box, taken during [the observability work](/projects/gpu-observability-dashboard/). That number
turned out to be load-bearing — it's where the
[roofline calculator](/projects/roofline-modeling-toolkit/) gets its interconnect bandwidth,
instead of a spec-sheet figure.

## Plan

- AllGather latency across sharding patterns and array sizes, including the small-array
  latency-bound regime
- Two sharded matmul strategies compared — gather-then-multiply vs multiply-then-reduce — since
  the right one depends on shape, not a fixed rule
- All four core primitives (AllGather, AllReduce, ReduceScatter, AllToAll) measured end to end,
  against theoretical cost
