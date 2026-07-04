---
title: 'Quantization Deep Dive: INT4, GPTQ, AWQ, and the Accuracy-Latency Tradeoff'
date: 2026-07-07
permalink: /blog/quantization-deep-dive/
tags:
  - inference
  - model-optimization
---

**Status:** 🔜 Planned

## Why write this
"We quantized the model to make it faster" is a one-liner everyone says; almost no one can explain what GPTQ or AWQ actually do differently, or why one might beat the other on a given model. This post — paired with your [quantization benchmark project](/projects/quantization-benchmark-suite/) — should make that difference concrete.

## Outline
1. Why quantization works at all: LLM weights are heavily over-parameterized relative to the precision needed at inference time
2. INT8 weight-only quantization: the simplest case, minimal accuracy loss, moderate speedup
3. GPTQ: layer-by-layer quantization that minimizes reconstruction error using calibration data, one-shot post-training
4. AWQ: the insight that a small fraction of "salient" weight channels (identified via activation magnitude) matter disproportionately, and protecting them preserves accuracy better than naive quantization
5. Your own benchmark numbers: accuracy vs. latency vs. memory across all three, presented as a single Pareto chart

## Key points to nail
- Explain GPTQ's use of the Hessian/second-order information at an intuitive level, not just "it's an algorithm that quantizes layers"
- Explain precisely what "activation-aware" means in AWQ and why it beats naive round-to-nearest quantization
- Be honest about where each method breaks down (very low bit-widths, certain model architectures)

## Reference reading
- GPTQ paper: "Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- AWQ paper: "Activation-aware Weight Quantization for LLM Compression and Acceleration"
