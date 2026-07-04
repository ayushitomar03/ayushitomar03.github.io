---
title: "High-Throughput LLM Inference Server"
excerpt: "**Status: Planned.** A from-scratch inference server implementing continuous batching and paged KV-cache, benchmarked against vLLM. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Every frontier lab runs enormous fleets of GPUs just to serve inference. The engineers who own that layer (vLLM, TGI, TensorRT-LLM, in-house serving stacks) are some of the highest-leverage hires at these labs. Building a miniature version of this system end-to-end is the single most direct way to prove you understand the problem, not just the buzzwords.

## What to build
- A serving engine for a small open model (e.g. Llama 3 8B or Qwen 2.5 7B) that supports:
  - Continuous batching (dynamically add/remove requests mid-flight instead of static batches)
  - Paged KV-cache (block-based cache allocation, no cache fragmentation)
  - Streaming token output over HTTP/gRPC
- A load generator that simulates realistic request patterns (varying prompt/output lengths, Poisson arrivals)
- A benchmark harness comparing your engine's throughput/latency (TTFT, TPOT, p50/p99) against vLLM on the same hardware

## Key concepts to demonstrate
- Why naive batching wastes GPU cycles on padding
- How paged attention eliminates KV-cache fragmentation
- The throughput/latency tradeoff curve as batch size grows
- Where the bottleneck actually is: memory bandwidth vs. compute-bound regimes

## Suggested stack
PyTorch + CUDA graphs, or build on top of `flash-attn` kernels; FastAPI/gRPC for the serving layer; Prometheus for metrics.

## Definition of done
A public repo with a working server, a reproducible benchmark script, and a results write-up (README or linked blog post) with real throughput/latency numbers vs. vLLM on identical hardware.
