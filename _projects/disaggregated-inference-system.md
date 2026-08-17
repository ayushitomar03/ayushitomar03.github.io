---
title: "Disaggregated Prefill/Decode Inference System"
excerpt: "Built and benchmarked. A hand-built disaggregated engine, vLLM at TP=2, and real NixlConnector disaggregation on Qwen2.5-32B — all measured against colocated baselines at matched GPU count. 2.4× TTFT improvement at 7B, and a tail-latency failure traced to interconnect saturation."
collection: projects
order: 1
status: Published
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">Qwen2.5-7B / 32B</span>
  <span class="pill">2× A100-80GB PCIe</span>
  <span class="pill">vLLM · NIXL · PyTorch</span>
</p>

Prefill is compute-bound; decode is memory-bandwidth-bound. Running them on the same GPU means
each phase competes for the resource the other isn't using. This project implements the split
three different ways and measures all of them against colocated baselines at matched GPU count.

## What was built

- **A custom engine** calling HF `transformers` directly, with manual KV-cache management and
  separate prefill/decode OS processes (threads were evaluated and rejected — the GIL
  serialises CPU-side scheduling and made tail latency *worse* than the baseline).
- **A paged/block-based KV cache** with per-request block tables, validated token-for-token
  including multi-block-boundary stress cases.
- **A real disaggregated vLLM deployment** on Qwen2.5-32B: separate prefill and decode servers
  with `--kv_role=kv_producer`/`kv_consumer`, plus a proxy relaying `kv_transfer_params`, with
  KV blocks moving over NIXL (RDMA-style, UCX transport).
- **One workload generator** driving every configuration identically: 50 requests, ~80% short /
  20% long, Poisson arrivals at 150 ms mean interarrival, fixed seed.

## Results

At 7B, disaggregation improved TTFT p99 by **2.4×** (795 ms → 332 ms). At 32B it improved TTFT
p50 by **2.8×**. But TTFT p99 under load hit **15.1 s** — and vLLM's own KV Transfer metrics
showed why: average transfer payload climbed from 4 MB to 424 MB as long prompts entered
rotation, saturating a PCIe interconnect with no NVLink.

Three implementations, three proximate causes (GPU contention, batch-size admission queueing,
interconnect saturation), one shared root cause: **a fixed 1:1 prefill:decode ratio
under-provisions decode once demand exceeds one GPU's capacity.**

Two correctness defects worth noting: `sdpa` attention silently corrupts output for left-padded
rows combined with an explicit `Cache` object (caught by token-for-token diff, not by any
exception), and the paged cache initially allocated blocks once per *layer* instead of once per
*token* while never freeing them on completion.

[Read the full writeup →](/blog/disaggregated-prefill-decode/)

## Open

Right-sizing the prefill:decode ratio with multiple replicas per role behind the proxy — vLLM's
routing already supports it; I haven't run it.
