---
title: "Inference Roofline & KV Cache Sizing Calculator"
excerpt: "**Status: Planned.** A calculator that reproduces the inference-chapter exercises from 'How to Scale Your Model' — KV cache sizing, batch size limits, and prefill/decode sharding, including a Mixture-of-Experts variant. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
This directly extends the [disaggregated inference project](/projects/disaggregated-inference-system/): before you can decide *how* to shard prefill/decode, you need to be able to calculate KV cache size, parameter loading time, and compute-bound batch size for a given model and chip. The [inference chapter](https://jax-ml.github.io/scaling-book/inference/) of *How to Scale Your Model* works through exactly this, including a Mixture-of-Experts case.

## What to build
Turn the chapter's worked problems into a reusable calculator:
- Q1/Q2: given model config (layers, heads, dims), compute total parameters and per-token KV cache size in int8, then the max batch size that fits on a given chip slice at a given sequence length
- Q3/Q4: compute parameter-loading time from HBM as a latency lower bound, and design a sharding strategy for prefill vs. decode separately, including ICI (chip-to-chip interconnect) bandwidth constraints
- Q5/Q6: extend the calculator to a Mixture-of-Experts model (E experts, k active) — total vs. activated parameters, FLOPs-bound batch size, and expert-sharding weight-loading time
- Q7: implement the 2D weight-stationary sharding math and find the crossover point where it beats standard 3D model sharding

## Key concepts to demonstrate
- Why prefill and decode need *different* sharding strategies, derived from the numbers, not intuition
- How MoE changes the KV-cache/parameter-count math versus a dense model
- Reading hardware specs (HBM bandwidth, ICI bandwidth, peak FLOPs) as direct inputs to a design decision, not background trivia

## Suggested stack
Python calculator (dataclasses for model/chip configs); validate a subset against a real served model if you have access to one (e.g. an open MoE model like Mixtral).

## Definition of done
A calculator/notebook covering all seven exercises with your own worked answers, plus a short comparison of a dense vs. MoE model's inference sizing on the same hardware.
