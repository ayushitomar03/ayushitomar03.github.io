---
title: 'Anatomy of a Distributed Training Run: Data, Tensor, and Pipeline Parallelism'
date: 2026-07-05
permalink: /blog/parallelism-explained/
tags:
  - training-infra
  - distributed-systems
excerpt: "In progress. What each parallelism strategy actually moves across the wire, and the point at which the communication cost stops being worth it."
published: false
---

<p class="pill-row"><span class="pill pill--progress">Writeup in progress</span></p>

The parts of this I've actually measured are written up elsewhere: FSDP2 full sharding on
2× A100s, including the finding that per-step communication came out to **+244 ms — nearly
equal to the compute itself** — on a PCIe box with no NVLink. That's in
[the MFU and observability post](/blog/model-flops-utilization/), with the fault-tolerance
implications in [the recovery post](/blog/fault-tolerant-training/).

What this post still owes: a proper side-by-side of data, tensor, and pipeline parallelism —
what each one shards, what each sends across the interconnect per step, and how to reason about
which combination fits a given model size and topology *before* running anything.
