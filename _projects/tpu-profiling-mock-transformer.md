---
title: "Profiling: Optimizing a Mock Transformer Layer"
excerpt: "Planned. Diagnosing a performance bug from a profiler trace alone, then closing the gap to the roofline bound — the per-kernel layer beneath the step-level attribution already built."
collection: projects
order: 10
status: Upcoming
---

<p class="pill-row"><span class="pill pill--upcoming">Upcoming</span></p>

Knowing the theory is table stakes; opening a profiler trace and diagnosing *why* a real kernel
is slow is the day-to-day skill.

## Where it stands

The step-level version of this already exists: the
[observability instrumentation](/projects/gpu-observability-dashboard/) attributes each step's
wall clock to compute vs NCCL communication using nested CUDA events, which was enough to
diagnose FSDP's +244 ms/step communication overhead on a PCIe box.

What it can't do is tell you *which* kernel or *which* collective. Per-kernel attribution via
`torch.profiler`, wired into the same metrics pipeline, is the missing layer — and it's the
natural next step for the observability project as much as a standalone one.

## Plan

- Capture a profile of a mock transformer layer and diagnose shapes, sharding, and the
  attention-vs-MLP time split from the trace alone
- Find and fix the sharding misconfiguration, with before/after per-layer latency
- Compare the optimised result against the bound from the
  [roofline toolkit](/projects/roofline-modeling-toolkit/) — how close to theoretically optimal,
  and what's left on the table
