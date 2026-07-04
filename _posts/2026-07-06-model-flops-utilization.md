---
title: 'Model FLOPs Utilization: The Metric Frontier Labs Obsess Over'
date: 2026-07-06
permalink: /blog/model-flops-utilization/
tags:
  - training-infra
  - performance
---

**Status:** 🔜 Planned

## Why write this
MFU is the metric that infra teams at frontier labs report internally as a north star, and it rarely gets explained clearly outside of paper appendices. A crisp explainer, backed by your own measurement from the [observability dashboard project](/projects/gpu-observability-dashboard/), is a strong, specific signal.

## Outline
1. Define MFU precisely: achieved FLOPs/sec ÷ theoretical peak FLOPs/sec of the hardware
2. Why "GPU utilization: 99%" (from `nvidia-smi`) is nearly meaningless and can be high while MFU is low
3. Where FLOPs get wasted in practice: communication stalls, data loading stalls, small batch sizes, activation recomputation overhead
4. What "good" MFU looks like in practice (well-known published numbers from large training runs, roughly 30-50%) and why 100% is essentially unreachable
5. How to actually measure it in your own training loop

## Key points to nail
- Show the actual formula and walk through a worked numeric example, not just the definition
- Be specific about *why* the gap between utilization and MFU exists — this is the part people get vague about
- If possible, include a real measurement from your own training run showing MFU over time and what changed it

## Reference reading
- PaLM paper (reports MFU explicitly and discusses it)
- Megatron-LM paper (efficient large-scale training, discusses achieved vs peak FLOPs)
