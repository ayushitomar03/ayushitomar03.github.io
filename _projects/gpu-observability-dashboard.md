---
title: "GPU Cluster Observability & MFU Dashboard"
excerpt: "**Status: Planned.** A real-time dashboard tracking GPU utilization, NCCL communication overhead, and Model FLOPs Utilization across a training cluster. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
"Is my cluster actually being used efficiently?" is one of the most expensive questions in ML infra — at frontier-lab scale, a 10% MFU improvement is worth millions of dollars in compute. Teams that build the observability to even answer that question are core to every frontier lab's infra org.

## What to build
- Custom exporters (Prometheus) that scrape `nvidia-smi`/DCGM metrics: GPU utilization, memory usage, power draw, temperature
- Instrumentation inside a training loop to compute and export **MFU** (achieved FLOPs / peak hardware FLOPs) in real time
- NCCL communication time tracking — how much time is spent in all-reduce/all-gather vs. actual compute
- A Grafana dashboard tying it together: per-node GPU health, cluster-wide MFU over time, communication-vs-compute breakdown

## Key concepts to demonstrate
- Why raw "GPU utilization %" is a misleading metric and MFU is the one that actually matters
- How to attribute wasted cycles to specific causes (communication stalls, data loading stalls, small batch size)
- Reading a training run's health from metrics alone, without staring at logs

## Suggested stack
DCGM + Prometheus node exporter; a custom PyTorch profiler hook or `torch.profiler` integration for MFU; Grafana for visualization.

## Definition of done
A working dashboard (screenshot/recording) showing a real training run's GPU utilization and MFU over time, plus a short analysis of where the compute is actually going.
