---
title: "Fault-Tolerant Distributed Training Orchestrator"
excerpt: "**Status: Planned.** A multi-node training harness with elastic scaling, automatic checkpoint/resume, and failure injection testing. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
At frontier-lab scale, training jobs run on thousands of GPUs for weeks, and hardware fails constantly (NCCL hangs, node drops, silent data corruption). The infra that keeps a training run alive through these failures — not the model architecture — is what separates a research prototype from a production training system.

## What to build
- A multi-node training job (data parallel + at least one of tensor/pipeline parallel) using PyTorch DDP/FSDP or DeepSpeed
- Automatic checkpointing on a schedule, with fast resume from the last good checkpoint
- Elastic scaling: the job survives a worker being killed and continues with the remaining workers (or scales back up when one rejoins)
- A failure injection harness that kills random workers mid-training to prove the recovery path actually works, not just in theory

## Key concepts to demonstrate
- The difference between data, tensor, and pipeline parallelism, and when each is used
- Why checkpoint frequency is a tradeoff between wasted compute on failure and I/O overhead
- How elastic training (e.g. `torchrun` elastic, Ray Train) differs from fixed-worker-count training
- Real MFU (Model FLOPs Utilization) measurement before/after a simulated failure and recovery

## Suggested stack
PyTorch FSDP or DeepSpeed ZeRO; `torchrun --elastic` or Ray Train; a multi-node setup (can be simulated with multiple GPUs/processes if a real cluster isn't available); Prometheus for tracking MFU and recovery time.

## Definition of done
A repo demonstrating a training run that survives at least one injected node failure with automatic recovery, plus a write-up of recovery time and MFU impact.
