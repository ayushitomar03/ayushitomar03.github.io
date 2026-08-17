---
title: 'Rooflines: When Does a Matmul Become Compute-Bound?'
date: 2026-07-10
permalink: /blog/rooflines-compute-vs-memory-bound/
tags:
  - hardware
  - performance
excerpt: "Writeup pending. The calculator is built and validated against real measurements — parameter counts to within 0.004% of a measured model load, and interconnect bandwidth backed out from a real NCCL all-reduce."
published: false
---

<p class="pill-row">
  <span class="pill pill--built">Tool built</span>
  <span class="pill pill--progress">Writeup pending</span>
</p>

The [roofline and KV-cache calculator](/projects/roofline-modeling-toolkit/) is built and
validated. Two validation points I care about more than the arithmetic itself:

- Its parameter-count formula gives 7,615,283,200 for Qwen2.5-7B-Instruct against a
  **measured** 7,615,616,512 from actually loading the model — 0.0044% off, fully accounted for
  by RMSNorm scale parameters the formula omits as negligible.
- The interconnect bandwidth it uses (16.26 GB/s) isn't a spec-sheet figure. It's backed out
  from a real 200 MB NCCL all-reduce measured on the same box during
  [the observability work](/blog/model-flops-utilization/).

The post that explains the reasoning — arithmetic intensity, where the roofline knee sits, and
why decode lands on the memory-bound side of it while prefill doesn't — is still to write.
