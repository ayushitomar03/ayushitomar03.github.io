---
title: 'Continuous Batching and PagedAttention: Why It Has to Be a Kernel'
date: 2026-07-04
permalink: /blog/continuous-batching-paged-attention/
tags:
  - inference
  - llm-serving
excerpt: "In progress. I built a paged block-based KV cache and it came out slower than the naive version — the reason why is the argument for PagedAttention being a CUDA kernel rather than a data structure."
published: false
---

<p class="pill-row"><span class="pill pill--progress">Writeup in progress</span></p>

I built a paged, block-based KV cache — fixed-size blocks, per-request block tables — as part
of the [disaggregated inference project](/blog/disaggregated-prefill-decode/), validated it
token-for-token including multi-block-boundary stress cases, and found two real defects doing
so: block allocation happening once per layer instead of once per token, and blocks never
being freed on request completion.

Then it came out **slower** than the naive batch-capped version it was meant to replace. That
result deserves its own post: reconstructing a request's cache from
block storage every tick costs `seq_len / block_size` slice-and-concat operations per layer,
each paying fixed Python and kernel-launch overhead regardless of how much data it moves. Which
is precisely why production PagedAttention is a CUDA kernel operating directly on block memory,
not a Python-level gather.

The measurements are already in the
[disaggregation writeup](/blog/disaggregated-prefill-decode/). This post will cover continuous
batching and the block-table mechanism properly on their own terms.
