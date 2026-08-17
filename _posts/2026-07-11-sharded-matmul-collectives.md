---
title: 'Sharded Matrix Multiplication: AllGather, AllReduce, and AllToAll Compared'
date: 2026-07-11
permalink: /blog/sharded-matmul-collectives/
tags:
  - distributed-systems
  - hardware
excerpt: "Planned. Which collective you pick, and how you shard the inputs, changes communication cost by multiples — this post will measure that rather than assert it."
published: false
---

<p class="pill-row"><span class="pill pill--progress">Planned</span></p>

"Just all-reduce the gradients" hides a real design space: which collective you use, and how
you shard the inputs, changes communication cost by multiples.

The one measurement I have so far is a 200 MB NCCL all-reduce on a PCIe box, which came in at
~12.3 ms steady state — a number that ended up load-bearing in
[the roofline calculator](/projects/roofline-modeling-toolkit/), since it's where its
interconnect-bandwidth figure comes from rather than a spec sheet.

The post itself — sharded matmul as gather-then-multiply vs multiply-then-reduce, when AllToAll
beats AllGather, and measured-vs-theoretical cost across the four core collectives — is still
ahead of me.
