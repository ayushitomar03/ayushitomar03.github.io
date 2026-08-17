---
title: "Inference Roofline & KV Cache Sizing Calculator"
excerpt: "Built and validated. A calculator for parameter counts, KV cache memory, memory-bound latency floors, and tensor-parallel sharding tradeoffs — checked against a real model load to 0.0044% and a measured NCCL all-reduce, not spec sheets."
collection: projects
order: 6
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Tool built &amp; validated</span>
  <span class="pill pill--progress">Writeup pending</span>
  <span class="pill">Python</span>
</p>

The sizing questions that come *before* you can decide how to shard a model for inference:
parameter counts, KV cache memory, memory-bound latency floors, tensor-parallel tradeoffs, and
the MoE-specific version of each. It extends the
[disaggregated inference project](/projects/disaggregated-inference-system/) directly — the
prefill:decode ratio that writeup flags as unresolved is exactly a sharding-sizing question.

## Validation, which is the interesting part

A calculator that's only internally consistent proves nothing. Two anchors to real
measurements:

- **Parameter count** — the formula gives 7,615,283,200 for Qwen2.5-7B-Instruct against a
  measured 7,615,616,512 from actually loading the model. **0.0044% off**, fully accounted for
  by RMSNorm scale parameters the formula omits as negligible.
- **Interconnect bandwidth** — the 16.26 GB/s figure isn't from a spec sheet. It's backed out
  from a real 200 MB NCCL all-reduce (~12.3 ms steady state) measured on the same box during
  [the observability work](/projects/gpu-observability-dashboard/).

## Sample output

Qwen2.5-7B on one L40S:

- KV cache per token (int8): **28,672 bytes**
- Max batch size — seq_len 2,048: **481** · 8,192: **120** · 32,768: **30**
- Memory-bound decode-step latency floor (bf16): **17.63 ms**, against a measured steady-state
  decode step of ~17–25 ms/token with batching — same order of magnitude, which is what
  confirms the system is genuinely memory-bound

## Honest caveat

The MoE model (Mixtral-8x7B) was never independently loaded, so its numbers are checked only
against widely-known public figures (~46.7B total / ~12.9B active) — one notch less verified
than the Qwen numbers, and flagged as such in the repo.
