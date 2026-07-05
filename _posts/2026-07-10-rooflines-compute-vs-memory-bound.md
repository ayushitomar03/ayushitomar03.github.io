---
title: 'Rooflines: When Does a Matmul Become Compute-Bound?'
date: 2026-07-10
permalink: /blog/rooflines-compute-vs-memory-bound/
tags:
  - hardware
  - performance
---

**Status:** 🔜 Planned

## Why write this
Roofline reasoning is the mental model every good infra engineer uses instinctively, but it's rarely explained from first principles in a way that lets someone else redo the math themselves. This should be the write-up companion to the [roofline modeling toolkit project](/projects/roofline-modeling-toolkit/), working the actual exercises from Google DeepMind's [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/roofline/).

## Outline
1. Two numbers that determine everything: peak compute (FLOPs/s) and peak memory bandwidth (bytes/s) — their ratio is the hardware's "critical arithmetic intensity"
2. Arithmetic intensity of an operation (FLOPs ÷ bytes moved) and why comparing it to the critical ratio tells you compute- or memory-bound
3. Work through the int8 matmul exercise (Q1) end to end with real numbers
4. The mixed-precision case (Q2): why using int8 weights with bf16 activations shifts the compute-bound threshold, and derive the batch size where it flips
5. Plot it (Q3): a real roofline chart, not just a verbal description
6. Do it again for a GPU (Q5, H100 specs) to show the model generalizes across hardware, not just TPUs

## Key points to nail
- Show the actual arithmetic, not just the conclusion — this is a post where the derivation *is* the value
- Be explicit that the "roofline" is a plot of two lines and the interesting content is the crossover point
- If you ran the validation benchmark from the project, lead with how close measured vs. predicted crossover actually was

## Reference reading
- *How to Scale Your Model*, Chapter 1: ["A Brief Intro to Roofline Analysis"](https://jax-ml.github.io/scaling-book/roofline/) (Google DeepMind)
