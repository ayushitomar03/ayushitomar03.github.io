---
title: 'Disaggregated Prefill/Decode Serving, Measured Three Ways'
date: 2026-08-01
permalink: /blog/disaggregated-prefill-decode/
tags:
  - inference
  - llm-serving
  - gpu
excerpt: "Prefill and decode are bound by different hardware resources, so splitting them across GPUs should help. I built that split three ways — a custom engine, vLLM at TP=2, and real NIXL disaggregation on Qwen2.5-32B — and measured where the idea pays and where it stops paying."
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">Qwen2.5-7B / 32B</span>
  <span class="pill">2× A100-80GB PCIe</span>
  <span class="pill">vLLM · NIXL · PyTorch</span>
</p>

Serving an LLM request is two workloads with different bottlenecks. **Prefill** processes the whole
prompt in parallel: it saturates the tensor cores and is compute-bound. **Decode** generates one
token at a time, re-streaming the model weights out of HBM for every single token: it is
memory-bandwidth-bound and leaves most of the compute idle.

Run both on the same GPU and each phase spends its time contending for the resource the other one
isn't using. Disaggregation puts them on separate GPUs and ships the KV cache across the boundary.

I built that three ways and benchmarked all of them against colocated baselines at matched GPU
count. The short version: **the TTFT benefit is real and scales with prefill cost, but all three
implementations hit the same ceiling — and it isn't the transfer mechanism, it's the ratio of
prefill workers to decode workers.**

## The idea

<figure class="diagram">
<svg viewBox="0 0 760 250" role="img" aria-labelledby="disagg-title">
  <title id="disagg-title">Colocated versus disaggregated prefill/decode topology</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="12" fill="currentColor">

    <text x="0" y="14" font-size="12.5" font-weight="600">Colocated — 2 replicas</text>
    <text x="0" y="31" font-size="11" fill="var(--d-muted)">each GPU runs prefill and decode for its own requests</text>

    <rect x="0" y="46" width="150" height="72" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="12" y="66" font-size="11" font-weight="600">GPU 0</text>
    <rect x="12" y="76" width="60" height="28" rx="5" fill="none" stroke="var(--d-accent)"/>
    <text x="42" y="94" font-size="10" text-anchor="middle" fill="var(--d-accent)">prefill</text>
    <rect x="78" y="76" width="60" height="28" rx="5" fill="none" stroke="var(--d-muted)"/>
    <text x="108" y="94" font-size="10" text-anchor="middle" fill="var(--d-muted)">decode</text>

    <rect x="0" y="130" width="150" height="72" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="12" y="150" font-size="11" font-weight="600">GPU 1</text>
    <rect x="12" y="160" width="60" height="28" rx="5" fill="none" stroke="var(--d-accent)"/>
    <text x="42" y="178" font-size="10" text-anchor="middle" fill="var(--d-accent)">prefill</text>
    <rect x="78" y="160" width="60" height="28" rx="5" fill="none" stroke="var(--d-muted)"/>
    <text x="108" y="178" font-size="10" text-anchor="middle" fill="var(--d-muted)">decode</text>

    <text x="0" y="228" font-size="10.5" fill="var(--d-warn)">a long prefill blocks that GPU's decode steps → TTFT tail spikes</text>

    <line x1="330" y1="40" x2="330" y2="215" stroke="var(--d-line)" stroke-dasharray="3 4"/>

    <text x="380" y="14" font-size="12.5" font-weight="600">Disaggregated — 1 prefill, 1 decode (1P1D)</text>
    <text x="380" y="31" font-size="11" fill="var(--d-muted)">roles are split; the KV cache crosses the boundary</text>

    <rect x="380" y="46" width="140" height="72" rx="8" fill="var(--d-fill)" stroke="var(--d-accent)" stroke-width="1.5"/>
    <text x="392" y="66" font-size="11" font-weight="600">GPU 0</text>
    <text x="392" y="83" font-size="10.5" fill="var(--d-accent)">prefill only</text>
    <text x="392" y="99" font-size="9.5" fill="var(--d-muted)">compute-bound</text>
    <text x="392" y="112" font-size="9.5" fill="var(--d-muted)">kv_producer</text>

    <rect x="620" y="46" width="140" height="72" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="632" y="66" font-size="11" font-weight="600">GPU 1</text>
    <text x="632" y="83" font-size="10.5">decode only</text>
    <text x="632" y="99" font-size="9.5" fill="var(--d-muted)">bandwidth-bound</text>
    <text x="632" y="112" font-size="9.5" fill="var(--d-muted)">kv_consumer</text>

    <line x1="524" y1="82" x2="612" y2="82" stroke="var(--d-warn)" stroke-width="1.5"/>
    <polygon points="620,82 610,78 610,86" fill="var(--d-warn)"/>
    <text x="568" y="74" font-size="9.5" text-anchor="middle" fill="var(--d-warn)">KV blocks</text>

    <rect x="470" y="160" width="200" height="42" rx="8" fill="var(--d-fill)" stroke="var(--d-line)" stroke-dasharray="4 3"/>
    <text x="570" y="177" font-size="10.5" text-anchor="middle" font-weight="600">proxy :8192</text>
    <text x="570" y="192" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">relays kv_transfer_params</text>
    <line x1="470" y1="181" x2="450" y2="181" stroke="var(--d-line)"/>
    <line x1="450" y1="181" x2="450" y2="118" stroke="var(--d-line)"/>
    <line x1="670" y1="181" x2="690" y2="181" stroke="var(--d-line)"/>
    <line x1="690" y1="181" x2="690" y2="118" stroke="var(--d-line)"/>

    <text x="380" y="228" font-size="10.5" fill="var(--d-good)">prefill is never queued behind another request's decode work</text>
  </g>
</svg>
<figcaption>Both topologies use the same total GPU count. The difference is whether a GPU is
specialised by role or by request.</figcaption>
</figure>

## Benchmark design

One workload generator drove every configuration, so the comparisons are like-for-like:

- **50 requests**, roughly 80% short prompts and 20% long (up to ~1,800–2,000 tokens)
- **Poisson arrivals** at 150 ms mean interarrival, about 6.7 requests/second
- **Fixed seed**, so the identical request set and arrival schedule is replayed per configuration
- **Matched GPU count** — two GPUs everywhere, differing only in how they are used

Three metrics: **TTFT** (prefill-bound), **inter-token latency** (decode-bound), and aggregate
throughput. I tracked p50 and p99 rather than means, because the tail behaviour turned out to be
the main finding and a mean barely moves while the tail dominates.

The custom engine and the single-process vLLM configurations were driven by in-process calls. Real
disaggregation has no in-process entry point — it is HTTP across independent server processes by
construction — so that configuration was driven with streaming `httpx` requests against the proxy.

## Implementation 1: a custom engine

The first version calls Hugging Face `transformers` directly with manual KV-cache management, in
two configurations on the same two GPUs: a baseline of two independent replicas, and a
disaggregated split with a dedicated prefill process and a dedicated decode process, the KV cache
crossing the boundary through host memory (GPU → CPU → GPU).

**Processes, not threads.** I tried threads first. The GIL serialises CPU-side scheduling, so
prefill's Python work starves the decode thread's scheduling loop and produces *worse* tail latency
than the colocated baseline. Separate OS processes are also the path you need for true multi-node
disaggregation without direct peer-to-peer.

**Batched decode.** Concurrent requests merge into a single forward call per tick rather than one
call per request, since decode's fixed weight-read cost amortises across the batch. That is where
the throughput comes from, and it introduced the one correctness problem worth flagging: the default
`sdpa` attention backend silently corrupts output for left-padded rows combined with an explicit
`Cache` object — no exception, no warning, just wrong tokens. A token-for-token diff against the
unpadded sequential path caught it; forcing `attn_implementation="eager"` fixed it.

### Qwen2.5-7B, baseline vs disaggregated

| Metric | Baseline (2× colocated) | Disaggregated (1P1D) |
|---|---|---|
| Throughput | 105.9 tok/s | 135.5 tok/s |
| TTFT p99 | 795.1 ms | **332.2 ms (2.4×)** |
| Inter-token p99 | 500.5 ms | 1,671.6 ms |
| Inter-token max | 4,573.6 ms | 3,501.8 ms |

TTFT improves 2.4×, which is the predicted effect: a dedicated prefill process is never queued
behind another request's decode work.

Inter-token latency regresses about 3.3×, and the cause is a decode batch-size cap of 16 I added to
fix an OOM — unbounded left-padding across 35 concurrent requests with lengths from 4 to 1,700+
tokens exceeded 48 GB. The cap removes the OOM by converting the problem into admission queueing:
all decode work is funnelled onto one GPU while the baseline spreads the same load across two.

That is the first appearance of the finding: **a fixed 1:1 prefill:decode ratio under-provisions
decode once aggregate demand exceeds one GPU's batch capacity.**

### The paged KV cache

To remove the batch cap at its source rather than bounding it, I built a paged block-based KV cache
with per-request block tables, validated token-for-token including multi-block-boundary cases.

Once correct, it was **slower** than the batch-capped version. Reconstructing a request's cache from
block storage every tick costs `seq_len / block_size` slice-and-concat operations per layer —
roughly 3,000 operations per tick for a 1,700-token request at `block_size=16` — each paying fixed
Python and kernel-launch overhead regardless of how much data it moves.

That result is the argument for why production PagedAttention is a CUDA kernel operating directly on
block memory rather than a Python-level gather. The data structure isn't the win; the kernel is.
Writing that kernel was a separate project, so the comparison moved to an implementation that
already had one.

## Implementation 2: vLLM at TP=2

Same workload and model, vLLM's `AsyncLLM` with `tensor_parallel_size=2` — FlashAttention-2,
production PagedAttention, CUDA graphs, `torch.compile`:

| Metric | Custom (disaggregated) | vLLM (TP=2) |
|---|---|---|
| Throughput | 135.5 tok/s | **198.5 tok/s** |
| TTFT p99 | 332.2 ms | 314.5 ms |
| Inter-token p99 | 1,671.6 ms | **15.6 ms** |

TP=2 is sharding, not disaggregation — one instance with the model split across GPUs, not two
instances with distinct roles. It is here as a reference point for what a well-optimised serving
stack achieves on this hardware, and the inter-token gap measures the distance between a
Python-level batching loop and a fused-kernel implementation of the same idea.

## Implementation 3: real disaggregation via NIXL, at 32B

Testing genuine disaggregation requires a genuine multi-process deployment, so I ran it on
Qwen2.5-32B-Instruct — the scale where prefill is most expensive and disaggregation's advantage
should be largest.

[NIXL](https://github.com/ai-dynamo/nixl) is an RDMA-style async transfer layer (UCX transport) for
moving KV blocks between independent GPU processes using registered memory descriptors — the
production analogue of the custom engine's GPU→host→GPU round trip.

```
prefill server (GPU0:8100)  --kv_role=kv_producer
decode server  (GPU1:8200)  --kv_role=kv_consumer
proxy          (:8192)      routes and relays kv_transfer_params
```

The proxy issues each request to the prefill server first with `max_tokens=1` to force a
prefill-only pass, extracts the returned `kv_transfer_params` (block IDs, engine ID, host/port),
attaches them to the original request, and forwards to the decode server — which RDMA-pulls the
specified blocks before streaming generation. Both servers hold a full 32B copy at
`--gpu-memory-utilization 0.92`, so per-GPU memory footprint matches two independent replicas and
only the role differs.

### Every configuration, same workload

| Metric | Custom baseline | Custom disagg | vLLM TP=2 | vLLM colocated | vLLM NIXL (1P1D) |
|---|---|---|---|---|---|
| Throughput | 25.7 tok/s | 25.4 tok/s | 175.9 tok/s | 160.3 tok/s | 89.8 tok/s |
| TTFT p50 | 8,165 ms | 5,034 ms | 70.7 ms | 125.8 ms | 929 ms |
| TTFT p99 | 22,790 ms | 8,060 ms | 572 ms | 642 ms | **15,081 ms** |
| Inter-token p99 | 2,491 ms | 2,261 ms | 29.9 ms | 50.2 ms | 87.6 ms |
| Inter-token max | 2,631 ms | 3,776 ms | 277 ms | 466 ms | 99.9 ms |

Read honestly: **on this hardware and this workload, a single prefill/decode pair loses to a
well-tuned TP=2 instance** — roughly half the throughput and a far worse TTFT tail. It substantially
beats the custom engine's version of the same idea, and its inter-token latency is in the same order
of magnitude as TP=2, so the NIXL transfer path itself is fast under light concurrency. But 15.1 s
TTFT p99 from a production connector needs an explanation.

## Where the TTFT tail comes from

vLLM emits a periodic `KV Transfer metrics` log, which attributes it directly rather than leaving it
to inference:

| Window | Concurrent transfers | Avg transfer time | Avg payload | Descriptors |
|---|---|---|---|---|
| 1 | 1 | 27 ms | 4.0 MB | 64 |
| 2 | 24 | 436 ms | 4.0 MB | 64 |
| 3 | 18 | 1,673 ms | 50.7 MB | 1,564 |
| 4 | 8 | **12,398 ms** | 424 MB | 13,568 |

<figure class="diagram">
<svg viewBox="0 0 700 246" role="img" aria-labelledby="nixl-title">
  <title id="nixl-title">Average KV transfer time per logging window against concurrency and payload size</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Average KV transfer time per window</text>
    <text x="0" y="28" font-size="9.5" fill="var(--d-muted)">log scale, 150px per decade — from vLLM's own KV Transfer metrics log during the 32B 1P1D run</text>

    <g stroke="var(--d-line)" stroke-dasharray="2 4">
      <line x1="90" y1="42" x2="90" y2="186"/>
      <line x1="240" y1="42" x2="240" y2="186"/>
      <line x1="390" y1="42" x2="390" y2="186"/>
      <line x1="540" y1="42" x2="540" y2="186"/>
    </g>
    <g font-size="9.5" fill="var(--d-muted)" text-anchor="middle">
      <text x="90" y="200">10 ms</text>
      <text x="240" y="200">100 ms</text>
      <text x="390" y="200">1 s</text>
      <text x="540" y="200">10 s</text>
    </g>
    <line x1="90" y1="186" x2="660" y2="186" stroke="var(--d-line)"/>

    <rect x="90" y="160" width="65" height="16" rx="3" fill="var(--d-accent)" opacity="0.75"/>
    <text x="163" y="172" font-size="10">27 ms</text>
    <text x="90" y="153" font-size="9.5" fill="var(--d-muted)">window 1 · 1 concurrent · 4.0 MB</text>

    <rect x="90" y="122" width="246" height="16" rx="3" fill="var(--d-accent)" opacity="0.8"/>
    <text x="344" y="134" font-size="10">436 ms</text>
    <text x="90" y="115" font-size="9.5" fill="var(--d-muted)">window 2 · 24 concurrent · 4.0 MB</text>

    <rect x="90" y="84" width="333" height="16" rx="3" fill="var(--d-warn)" opacity="0.8"/>
    <text x="431" y="96" font-size="10">1,673 ms</text>
    <text x="90" y="77" font-size="9.5" fill="var(--d-muted)">window 3 · 18 concurrent · 50.7 MB</text>

    <rect x="90" y="46" width="464" height="16" rx="3" fill="var(--d-warn)"/>
    <text x="562" y="58" font-size="10" font-weight="600">12,398 ms</text>
    <text x="90" y="39" font-size="9.5" fill="var(--d-muted)">window 4 · 8 concurrent · 424 MB · 13,568 descriptors</text>

    <text x="0" y="226" font-size="10.5" fill="var(--d-muted)">Window 2 shows 24 simultaneous transfers completing in 436 ms — concurrency alone is handled fine.</text>
    <text x="0" y="242" font-size="10.5" fill="var(--d-muted)">Window 4 is a payload effect: long prompts push average payload ~100×, saturating a PCIe link with no NVLink.</text>
  </g>
</svg>
<figcaption>Transfer time against concurrency and payload size. The jump from window 2 to window 4
tracks payload, not the number of concurrent transfers.</figcaption>
</figure>

Window 2 is the informative one: 24 simultaneous transfers at 4 MB each still complete in 436 ms, so
the connector handles concurrency. What breaks it is window 4, where long prompts enter rotation and
average payload climbs to 424 MB — several concurrent multi-hundred-megabyte transfers saturate a
PCIe link with no NVLink.

So the bottleneck is not `NixlConnector`. It is **the capacity of a single prefill / single decode
deployment under rising load** — the same limitation the custom engine hit, in a third form. Three
implementations, three different proximate causes (raw GPU contention, batch-size admission
queueing, interconnect-bound transfer saturation), one shared root: a fixed 1:1 prefill:decode
ratio.

## What I'd conclude

Disaggregation's TTFT benefit is real and scales with prefill cost — 2.4× at 7B, 2.8× at 32B on p50.
But on two PCIe-connected GPUs, a 1P1D deployment is the wrong shape for this workload, and a
well-tuned TP=2 instance beats it on every metric except TTFT under light load.

The lever that matters is topology, not the transfer mechanism. Getting the prefill:decode ratio
right for the actual traffic mix — and provisioning enough decode capacity for the aggregate token
rate — determines whether the idea pays at all. The production answer is more replicas per role
behind the proxy, which vLLM's routing already supports and which I haven't yet run. That's the next
experiment, and the one that would turn this from a measurement into a recommendation.
