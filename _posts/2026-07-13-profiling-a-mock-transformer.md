---
title: 'What a Profiler Trace Actually Tells You'
date: 2026-07-13
permalink: /blog/profiling-a-mock-transformer/
tags:
  - performance
  - debugging
excerpt: "Planned. Diagnosing a real performance bug from a profiler trace alone — the show-don't-tell companion to the roofline and observability work."
published: false
---

<p class="pill-row"><span class="pill pill--progress">Planned</span></p>

The show-don't-tell version of the performance posts: given only a profiler trace, work out the
matrix shapes, the sharding strategy, and where the time is actually going — then fix it and
compare the result against the roofline bound.

The closest thing I have to this today is the step-time decomposition in
[the MFU and observability post](/blog/model-flops-utilization/), which attributes cost to
communication vs compute at the step level. Per-kernel attribution via `torch.profiler` is the
missing piece, and it's the next thing I want to wire into that same metrics pipeline.
