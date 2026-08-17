---
title: "Fault-Tolerant Distributed Training Orchestrator"
excerpt: "Built and benchmarked. An FSDP2 training job that survives an unannounced SIGKILL, recovers in 6.72s, and resumes bit-exactly — with a failure-injection harness that caught three real bugs the happy path never would have."
collection: projects
order: 2
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">FSDP2 · torchrun elastic</span>
  <span class="pill">2× A100-80GB PCIe</span>
</p>

A training job that survives an unannounced worker death mid-run, recovers automatically, and
finishes correctly — with the cost of that failure measured rather than asserted.

## What was built

- **A hand-built GPT-2-style decoder** (12 layers, 768 dim, 85,219,584 params) rather than HF
  `transformers`, so parameter count and FLOPs are exactly known for MFU purposes.
- **FSDP2 full sharding** (`fully_shard`) across both GPUs, with deterministically seeded
  per-rank data batches — which is what makes the correctness checks below possible.
- **Two checkpoint formats**: `full` (gathered to rank 0) and `sharded`
  (`torch.distributed.checkpoint`, each rank writing its own shard in parallel — the format you
  need once rank 0 would OOM gathering the state).
- **A failure-injection harness** that tails the JSONL step log and, at a target step, sends
  `SIGKILL` to a specific worker's OS process — plus a `hang` mode using `SIGSTOP` and a
  watchdog, to simulate a worker that's stuck rather than dead.
- **A control-vs-treatment comparator** that diffs an uninterrupted run against a
  failure-injected run at every one of 400 steps.

## Results

Killed at step 150 with the last checkpoint at step 125: **6.72 s** to restart, **7.43 s** full
outage, ~12.9 s of redone compute — **~20.3 s total, about 8% of wall clock.** MFU was
statistically identical before and after (2.67% → 2.66%), so recovery leaves no lingering
regression.

All 25 replayed steps matched the pre-kill pass to **six decimal places**, proving the restored
optimizer state was numerically identical rather than merely structurally present.

**Three real bugs the injection harness caught**, none of which the happy path would surface: a
redundant NCCL broadcast that only failed on cold restarts; non-atomic checkpoint writes that
made a mid-write crash *permanently* unrecoverable; and unseeded model initialisation, caught
only by the full-curve control comparison.

[Read the full writeup →](/blog/fault-tolerant-training/)

## Scope, stated honestly

"Elastic" here means restart-based recovery via `torchrun`'s c10d rendezvous and
`--max-restarts`, not live degrade-and-continue — with 2 GPUs on one node there's no genuine
"continue on the survivor" scenario. Only one worker (rank 1) was killed; rank 0 also runs the
checkpoint writer and wasn't tested separately. Silent data corruption and true multi-node loss
were not tested.
