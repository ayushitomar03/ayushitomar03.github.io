---
title: 'Killing a Training Job on Purpose: What Fault Tolerance Actually Costs'
date: 2026-07-30
permalink: /blog/fault-tolerant-training/
tags:
  - training-infra
  - distributed-systems
  - reliability
excerpt: "A distributed training run that survives an unannounced SIGKILL mid-run, recovers automatically, and finishes correctly — with the recovery cost measured rather than asserted, and three real bugs the failure injection caught."
published: false
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">FSDP2 · torchrun elastic</span>
  <span class="pill">2× A100-80GB PCIe</span>
  <span class="pill">85M-param GPT</span>
</p>

Everyone's training script has a `save_checkpoint()` call. Far fewer have ever had a worker
killed out from under them and watched what happens next. The gap between those two states is
where the bugs live — and I found three of them by closing it.

The setup: a training job on 2× A100s that survives an unannounced `SIGKILL` to a worker,
recovers automatically with no human involvement, and finishes to the target step. Every
number below comes from a real run, not from theory.

<ul class="keyresults">
  <li><span class="keyresults__value">6.72s</span><span class="keyresults__label">kill → elastic agent confirms restart</span></li>
  <li><span class="keyresults__value">~20.3s</span><span class="keyresults__label">total cost of the failure, ~8% of wall clock</span></li>
  <li><span class="keyresults__value">0.000000</span><span class="keyresults__label">loss diff across the restart boundary, all 25 replayed steps</span></li>
  <li><span class="keyresults__value">2.67%</span><span class="keyresults__label">MFU — an order of magnitude under production, by design on an 85M model</span></li>
</ul>

## Scope, stated up front

Two places where the honest scope is narrower than the words might suggest:

**"Tensor/pipeline parallel" here means FSDP2 full sharding**, not Megatron-style tensor
parallelism. FSDP shards parameters, gradients, and optimizer state across both GPUs
(ZeRO-3-equivalent) — a different mechanism from splitting individual matmuls across devices,
but the same underlying property: one model's state distributed across more than one GPU,
requiring cross-GPU communication to reconstruct what a single device needs.

**"Elastic" here means restart-based recovery, not live degrade-and-continue.** With two GPUs
on one physical node there is no genuine "continue on the survivor while the other
reprovisions" scenario. What's implemented and tested is `torchrun`'s c10d rendezvous plus
`--max-restarts`: a worker dying tears down the whole worker group, torchrun's elastic agent
relaunches it, and the new processes resume from the latest checkpoint with no manual
intervention. Real, working, automatic fault tolerance — a different flavour from seamless
dynamic scaling.

## What's in the box

- **Model** — a hand-built GPT-2-style decoder (not HF `transformers`), so parameter count and
  FLOPs are exactly known for MFU purposes: 12 layers, 768 embedding dim, 12 heads,
  85,219,584 params. Character-level tokenisation over Tiny Shakespeare, no external tokenizer
  dependency.
- **Parallelism** — FSDP2 (`torch.distributed.fsdp.fully_shard`) applied per transformer block
  and to the whole model, plus real data parallelism: each rank draws a different,
  deterministically-seeded microbatch per step
  (`seed = global_seed + rank*1_000_003 + step`). Determinism via explicit seeding, not
  RNG-state checkpointing — which is what makes the correctness checks below possible.
- **Checkpointing** — every 25 steps, full model + optimizer state gathered to rank 0 via
  `torch.distributed.checkpoint.state_dict` with `cpu_offload=True`. On startup, every rank
  independently scans for the highest step number and resumes from it.
- **Failure injection** (`inject_failure.py`) — launches the elastic job, tails its JSONL step
  log, and at a target step sends `SIGKILL` directly to one worker's OS process (read from a
  PID file), *not* to the torchrun launcher. Then watches for the restart-and-resume event and
  computes recovery time and before/after MFU automatically.

## The run: 400 steps, checkpoint every 25, killed at step 150

<figure class="diagram">
<svg viewBox="0 0 720 210" role="img" aria-labelledby="ft-title">
  <title id="ft-title">Timeline of the failure, recovery, and resumed training</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">

    <text x="0" y="12" font-size="12" font-weight="600">Failure and recovery timeline</text>

    <!-- main track -->
    <rect x="0" y="40" width="300" height="20" rx="4" fill="var(--d-accent)" opacity="0.28"/>
    <rect x="300" y="40" width="60" height="20" rx="4" fill="var(--d-warn)" opacity="0.45"/>
    <rect x="360" y="40" width="360" height="20" rx="4" fill="var(--d-accent)" opacity="0.28"/>

    <!-- checkpoint ticks -->
    <g stroke="var(--d-line)">
      <line x1="50" y1="36" x2="50" y2="64"/>
      <line x1="100" y1="36" x2="100" y2="64"/>
      <line x1="150" y1="36" x2="150" y2="64"/>
      <line x1="200" y1="36" x2="200" y2="64"/>
    </g>
    <line x1="250" y1="32" x2="250" y2="68" stroke="var(--d-good)" stroke-width="2"/>
    <text x="250" y="26" font-size="10" text-anchor="middle" fill="var(--d-good)">ckpt @ step 125</text>

    <!-- kill marker -->
    <line x1="300" y1="32" x2="300" y2="68" stroke="var(--d-warn)" stroke-width="2"/>
    <text x="304" y="26" font-size="10" fill="var(--d-warn)">SIGKILL @ step 150</text>

    <text x="0" y="80" font-size="9.5" fill="var(--d-muted)">step 0</text>
    <text x="712" y="80" font-size="9.5" text-anchor="end" fill="var(--d-muted)">step 400</text>
    <text x="330" y="80" font-size="9.5" text-anchor="middle" fill="var(--d-warn)">outage</text>

    <!-- wasted region bracket -->
    <line x1="250" y1="96" x2="300" y2="96" stroke="var(--d-warn)"/>
    <line x1="250" y1="92" x2="250" y2="100" stroke="var(--d-warn)"/>
    <line x1="300" y1="92" x2="300" y2="100" stroke="var(--d-warn)"/>
    <text x="275" y="112" font-size="10" text-anchor="middle" fill="var(--d-warn)">25 steps redone</text>
    <text x="275" y="125" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">~12.9s wasted compute</text>

    <!-- resume arrow -->
    <path d="M 360 68 C 360 90, 250 90, 250 70" fill="none" stroke="var(--d-good)" stroke-dasharray="3 3"/>
    <polygon points="250,64 246,74 254,74" fill="var(--d-good)"/>
    <text x="380" y="98" font-size="10" fill="var(--d-good)">RESUMED reports start_step: 125</text>

    <!-- cost breakdown -->
    <line x1="0" y1="146" x2="720" y2="146" stroke="var(--d-line)"/>
    <text x="0" y="166" font-size="10.5">Recovery (kill → agent confirms restart)</text>
    <text x="300" y="166" font-size="10.5" font-weight="600">6.72s</text>
    <text x="360" y="166" font-size="10.5">Full outage (kill → first new step)</text>
    <text x="660" y="166" font-size="10.5" font-weight="600">7.43s</text>
    <text x="0" y="184" font-size="10.5">Wasted compute (25 steps × 515ms)</text>
    <text x="300" y="184" font-size="10.5" font-weight="600">~12.9s</text>
    <text x="360" y="184" font-size="10.5">Total cost of the failure</text>
    <text x="660" y="184" font-size="10.5" font-weight="600" fill="var(--d-warn)">~20.3s</text>
    <text x="0" y="202" font-size="10" fill="var(--d-muted)">MFU pre-kill 2.67% → post-resume 2.66% · no lingering throughput regression</text>
  </g>
</svg>
<figcaption>The cost of a worker death decomposes into two independent parts: the outage
(fixed, set by the restart mechanism) and the wasted compute (variable, set by checkpoint
interval).</figcaption>
</figure>

| Metric | Value |
|---|---|
| Total wall clock, start to finish including the failure | ~247 s |
| Kill triggered at step | 150 |
| Last checkpoint before kill | step 125 |
| Wasted steps (redone after resume) | 25 |
| Wasted compute (25 steps × measured per-step time) | ~12.9 s |
| Recovery time (kill → elastic agent confirms restart) | 6.72 s |
| Full outage (kill → first new training step executes) | 7.43 s |
| **Total cost of the failure** | **~20.3 s (~8% of total wall clock)** |
| MFU, steady state pre-kill | 2.67% |
| MFU, steady state post-resume | 2.66% |
| Completed to target step | Yes (400/400) |
| Params per GPU (of 85,219,584, FSDP2-sharded) | rank 0: 42,610,176 / rank 1: 42,609,408 |

Reproduced twice with consistent numbers — recovery 6.2–6.7 s, MFU within 0.1 percentage
points both times.

## Proving the resume was actually correct

Checking that `RESUMED` reports `start_step: 125` proves the run picked the right checkpoint.
It does **not** prove the state inside that checkpoint was restored correctly. A subtly broken
restore — Adam's momentum buffers silently reset instead of loaded, say — would report the
right step number and still be wrong.

Because batches are seeded by `(rank, step)` alone, steps 126–150 get trained twice: once in
the doomed first attempt, and again after resuming from step 125. Diffing both passes:

| Step | 1st pass loss (pre-kill) | 2nd pass loss (post-resume) | Diff |
|---|---|---|---|
| 126 | 2.610977 | 2.610977 | 0.000000 |
| 130 | 2.588378 | 2.588378 | 0.000000 |
| 140 | 2.564350 | 2.564350 | 0.000000 |
| 150 | 2.569039 | 2.569039 | 0.000000 |

All 25 steps match to six decimal places. Bit-exact reproduction across the restart boundary
is direct evidence the restored model and optimizer state are numerically identical to what
was lost — not merely structurally present.

## The three bugs

This is the part that justifies building the injection harness at all.

### 1. An NCCL broadcast that only failed on cold restarts

The first full run appeared to succeed. Two subsequent reproduction attempts failed
*deterministically*: every restart attempt crashed with an NCCL `DistBackendError` during
`set_model_state_dict`'s checkpoint-loading broadcast.

Root cause: `train.py` already has every rank independently `torch.load()` the checkpoint from
the shared filesystem — no rank-0 guard. But `StateDictOptions` still passed
`broadcast_from_rank0=True`, telling PyTorch that only rank 0 has valid data and the others
need it delivered over NCCL. That redundant broadcast, run against a freshly-restarted process
group, raced against NCCL communicator setup on a cold start. Fixed by setting
`broadcast_from_rank0=False`, since every rank already had identical data from its own disk
read.

A first success followed by two identical-config failures is exactly what "the recovery path
works... in theory" is meant to catch — and would have gone unnoticed without actually
re-running it.

### 2. Checkpoint writes weren't atomic

Adding a sharded checkpoint format (`torch.distributed.checkpoint`, each rank writing its own
shard in parallel with no gather step — the format you need once rank 0 would OOM gathering a
100B-param model) exposed a bug the happy path never could.

A kill landing exactly mid-write left a directory with mismatched shard files and a missing
`step.json`. Every restart attempt then picked that broken directory up as "the latest
checkpoint," failed identically, and exhausted all retries without ever recovering. A crash
during checkpointing had made the job *permanently* unrecoverable.

Fixed by writing to a `.tmp`-suffixed name and only `os.rename`-ing to the final name once the
write completes — atomic on POSIX. Applied to both formats, since the `full` format had the
same theoretical exposure even though it hadn't been observed yet.

### 3. Unseeded model initialisation, caught by a full-curve comparison

Every correctness check so far compared two passes *within the same run's lineage* — both
descended from the same checkpoint. That structurally cannot catch a bug in how a run
*starts*.

`compare_control_treatment.py` closes the gap: since batches are seeded by `(rank, step)`
alone, an uninterrupted control run and a failure-injected treatment run should produce
identical loss at *every* step from 1 to 400.

They didn't. They mismatched starting at **step 1** — before any failure was injected — which
is only possible if two independent launches started from different weights. `global_seed` was
wired into the data sampler but never passed to `torch.manual_seed()` before model
construction, so each launch built its initial weights from PyTorch's unseeded global RNG.

After adding `torch.manual_seed()` / `torch.cuda.manual_seed_all()` before model construction:

| Step range | Behaviour |
|---|---|
| 1–5 | Exact match, diff = 0.0 |
| 6–124 | Match to ~1e-6 — floating-point noise |
| 125–151 (recovery boundary) | diff 5e-5 to 3e-4 — no jump relative to surrounding trend |
| 231 (worst case) | diff = 0.057 |
| 400 (final) | diff = 0.033 |

What remains is a small, smoothly-growing divergence with **no discontinuity at the recovery
boundary** — ordinary CUDA/cuBLAS floating-point non-associativity between two independently
launched processes. So the defensible claim is: *recovery introduces no additional divergence
beyond ordinary run-to-run GPU non-determinism.* Narrower than an artificial bit-exact
guarantee, and true.

## The checkpoint-frequency tradeoff, with real numbers

"Checkpoint frequency trades wasted compute against I/O overhead" is the standard hand-wave.
Here's what each side actually cost on this system:

- **Wasted compute** — at `ckpt_every=25` and ~515 ms/step, a failure costs up to
  25 × 0.515 s ≈ **12.9 s** of redone work in the worst case. Halving to `ckpt_every=12` roughly
  halves that.
- **I/O overhead** — each checkpoint is **1.02 GB** (85M params × 4 bytes × 3: weights plus
  AdamW's two fp32 moment buffers; measured file size 1,022,952,359 bytes matches the
  arithmetic). Each save added a measurable pause between the "step" and "checkpoint" events.
  Twice as often roughly doubles aggregate time spent not training.

The right interval is a genuine minimum of (expected wasted compute + I/O overhead) as a
function of MTBF. For a workload this small and reliable it barely matters in absolute
seconds — the tradeoff only dominates at the scale where checkpoints take minutes to write.
The **mechanism** generalises; these specific seconds do not.

## A second failure mode: the worker that's alive but stuck

`SIGKILL` simulates a clean crash. It doesn't simulate a worker hung inside an NCCL
collective, which looks completely different to its peers: no dead connection, just a
collective that never returns.

`--fault-mode hang` reproduces this with `SIGSTOP` (freezes the target without killing it)
plus a watchdog that declares a hang once the step log has gone quiet longer than
`--hang-timeout-s`, then force-kills the frozen process — converging onto the same
restart-and-resume path already validated for the kill case.

## MFU is 2.6%, and that's expected

Production training MFU runs 30–55%. This model is deliberately tiny (85M params, so a full
failure-injection run finishes in ~4 minutes instead of hours), and at that scale
Python/kernel-launch overhead and cross-GPU communication dominate a genuinely small amount of
real compute.

One measurement from this run is covered separately: a **single-GPU** run of the
identical config averages 271.5 ms/step, while the 2-GPU FSDP run averages 515.1 ms/step —
roughly **+244 ms of pure communication overhead, almost equal to the compute time itself.**
This box has no NVLink (PHB topology, PCIe through the CPU host bridge), so FSDP's per-block
all-gather/reduce-scatter dominates. That's not a bug; it's model size vs interconnect
bandwidth, and an 85M-param model is on the wrong side of the crossover. I dug into how to
measure that properly in [the MFU and observability post](/blog/model-flops-utilization/).

## What I didn't test

Being specific about this matters more than claiming completeness:

- **Only one failure type was injected** — a hard `SIGKILL` to a worker process (three times
  total), plus the `SIGSTOP` hang mode. Not tested: silent data corruption (bit-flips producing
  wrong-but-plausible gradients — a real frontier-scale failure mode that needs redundant
  computation or checksumming to catch), and true multi-node loss.
- **Only rank 1 was killed.** Rank 0 also runs the step logger and checkpoint writer, so
  killing it specifically might behave differently. Worth a dedicated test.
- **Hang mode has only been validated against the `full` format; kill mode only against
  `sharded`.** Cross-covering both isn't done.
- **True dynamic elastic scaling** (`--nnodes=min:max`, continuing with fewer workers without a
  full-group restart) needs a real multi-node cluster.

"Fault-tolerant" in this writeup means one specific thing: tolerant of one worker's clean
death, verified.
