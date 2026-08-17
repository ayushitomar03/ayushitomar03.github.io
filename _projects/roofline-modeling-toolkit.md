---
title: "Roofline Modeling & Benchmarking Toolkit"
excerpt: "Built. Arithmetic-intensity and roofline analysis turned into reusable code, with the crossover between memory-bound and compute-bound derived rather than asserted. Shares its validated core with the inference sizing calculator."
collection: projects
order: 7
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Tool built</span>
  <span class="pill pill--progress">Writeup pending</span>
  <span class="pill">Python</span>
</p>

Roofline reasoning — will this kernel be compute-bound or memory-bandwidth-bound, and at what
batch size does it flip — is the mental model underneath every serving and training
optimisation decision. This turns the math into code you can re-run against different model and
chip configs instead of re-deriving it each time.

It shares its core with the
[inference sizing calculator](/projects/inference-roofline-kv-cache-calculator/): the same
`DenseModelConfig` / `ChipConfig` dataclasses, with `num_experts=1` recovering the dense case
exactly so one code path covers both dense and MoE.

## What it computes

- Bytes loaded/written, total FLOPs, arithmetic intensity, and predicted runtime for matmuls of
  arbitrary shape
- The batch-size threshold where a mixed int8-weight / bf16-activation matmul flips from
  memory-bound to compute-bound
- Parameter-loading time from HBM as a hard decode-step latency floor
- Prefill vs decode sharding across N chips, including interconnect communication cost

## Validation

Both hardware constants it leans on are anchored to measurements from other projects rather
than spec sheets — the parameter formula checks against a real model load to **0.0044%**, and
the interconnect bandwidth is backed out from a measured NCCL all-reduce. Details in
[the sizing calculator](/projects/inference-roofline-kv-cache-calculator/).

## Pending

The writeup that shows the derivation — where the roofline knee sits and why decode lands on
the memory-bound side while prefill doesn't — plus a validation sweep benchmarking real matmuls
at several shapes to check predicted vs measured crossover.
