---
title: 'Quantization Deep Dive: INT8, GPTQ, AWQ, and What the Benchmark Actually Said'
date: 2026-07-28
permalink: /blog/quantization-deep-dive/
tags:
  - inference
  - model-optimization
  - llm-serving
excerpt: "Three quantized variants of Qwen2.5-7B-Instruct benchmarked end to end — five accuracy tasks, four concurrency levels, real serving numbers. GPTQ and AWQ land in the same place on accuracy; the reason to pick one is somewhere else entirely."
published: false
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">Qwen2.5-7B-Instruct</span>
  <span class="pill">L40S 46GB</span>
  <span class="pill">vLLM · GPTQ · AWQ</span>
</p>

"We quantized the model to make it faster" is a sentence everyone says and almost nobody can
unpack. What does GPTQ actually do differently from AWQ? When does INT8 beat INT4? Does the
accuracy loss show up where you're measuring it?

I quantized `Qwen2.5-7B-Instruct` three ways and benchmarked all four variants across five
accuracy tasks and four concurrency levels. The headline result is that the two INT4 methods
are statistically indistinguishable on quality — so the decision between them comes down to
something the accuracy tables don't show.

<ul class="keyresults">
  <li><span class="keyresults__value">2.74×</span><span class="keyresults__label">compression, INT4 (15.24 → 5.56 GB)</span></li>
  <li><span class="keyresults__value">2.7×</span><span class="keyresults__label">throughput vs fp16 at concurrency 32</span></li>
  <li><span class="keyresults__value">0.84 pt</span><span class="keyresults__label">MMLU cost of INT4</span></li>
  <li><span class="keyresults__value">1.8×</span><span class="keyresults__label">GPTQ quantization time vs AWQ, same result</span></li>
</ul>

## What got quantized, and why only these

Baseline: `Qwen2.5-7B-Instruct`, 7.62B params, fp16, 15.24 GB. Three variants against it:

| Variant | Scheme | Size | Compression |
|---|---|---|---|
| INT8 RTN | W8A16 | 8.82 GB | 1.73× |
| GPTQ INT4 | W4A16 | 5.56 GB | 2.74× |
| AWQ INT4 | W4A16 (asym) | 5.58 GB | 2.73× |

All three are **weight-only** — activations stay FP16 in every case.

**Why weight-only over activation quantization.** Weights live in HBM for the whole serving
session, but the compute units can't hold them between steps — the on-chip cache is far
smaller than the model. So every single token you generate re-pulls the weights from HBM into
the compute path, and *that transfer*, not the arithmetic, is what decode waits on. Shrink the
weights and the transfer gets 2–4× faster, so generation speeds up directly.

Activations are different: they depend on the input and change at every layer and every token,
so you can't calibrate them once and be done — you'd have to handle them at runtime. Weights
are fixed the moment training ends. Quantize once, offline, never think about it again.

**Why stop at INT4.** Below INT4 the accuracy cost gets real, and vLLM's fast fused kernel
(Marlin) is built around INT4 — there's no serving speed left to gain by going lower on this
stack. And since decode is memory-bound rather than compute-bound, quantizing activations too
(W8A8/FP8) buys compute savings we don't need, at the cost of dealing with activation
outliers.

## How GPTQ and AWQ differ

<figure class="diagram">
<svg viewBox="0 0 740 250" role="img" aria-labelledby="quant-title">
  <title id="quant-title">GPTQ error correction versus AWQ channel protection</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">

    <!-- GPTQ -->
    <text x="0" y="14" font-size="12.5" font-weight="600">GPTQ — correct the error after making it</text>
    <text x="0" y="31" font-size="10.5" fill="var(--d-muted)">quantize one column, then adjust the remaining columns to cancel what you just introduced</text>

    <g>
      <rect x="0" y="46" width="26" height="90" rx="3" fill="var(--d-accent)" opacity="0.75"/>
      <rect x="30" y="46" width="26" height="90" rx="3" fill="var(--d-accent)" opacity="0.4"/>
      <rect x="60" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <rect x="90" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <rect x="120" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <rect x="150" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
    </g>
    <text x="13" y="150" font-size="9" text-anchor="middle" fill="var(--d-accent)">done</text>
    <text x="43" y="150" font-size="9" text-anchor="middle" fill="var(--d-accent)">now</text>
    <text x="118" y="150" font-size="9" text-anchor="middle" fill="var(--d-muted)">pending</text>

    <path d="M 43 40 C 70 20, 130 20, 148 40" fill="none" stroke="var(--d-warn)" stroke-width="1.3"/>
    <polygon points="150,44 142,36 148,34" fill="var(--d-warn)"/>
    <text x="95" y="18" font-size="9.5" text-anchor="middle" fill="var(--d-warn)">error pushed forward</text>

    <text x="0" y="176" font-size="10.5">Uses a Hessian from calibration data to know how</text>
    <text x="0" y="191" font-size="10.5">each weight's error affects the layer output.</text>
    <text x="0" y="214" font-size="10.5" font-weight="600">Cost: Hessian solve + per-column correction</text>
    <text x="0" y="232" font-size="13" font-weight="650" fill="var(--d-warn)">17.2 min</text>

    <!-- divider -->
    <line x1="350" y1="40" x2="350" y2="225" stroke="var(--d-line)" stroke-dasharray="3 4"/>

    <!-- AWQ -->
    <text x="400" y="14" font-size="12.5" font-weight="600">AWQ — avoid the error in the first place</text>
    <text x="400" y="31" font-size="10.5" fill="var(--d-muted)">find the channels that see the largest activations, scale them up before rounding</text>

    <g>
      <rect x="400" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <rect x="430" y="46" width="26" height="90" rx="3" fill="var(--d-good)" opacity="0.75"/>
      <rect x="460" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <rect x="490" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <rect x="520" y="46" width="26" height="90" rx="3" fill="var(--d-good)" opacity="0.75"/>
      <rect x="550" y="46" width="26" height="90" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
    </g>
    <text x="443" y="150" font-size="9" text-anchor="middle" fill="var(--d-good)">salient</text>
    <text x="533" y="150" font-size="9" text-anchor="middle" fill="var(--d-good)">salient</text>

    <line x1="443" y1="40" x2="443" y2="26" stroke="var(--d-good)"/>
    <line x1="533" y1="40" x2="533" y2="26" stroke="var(--d-good)"/>
    <text x="488" y="20" font-size="9.5" text-anchor="middle" fill="var(--d-good)">scaled up, then rounded</text>

    <text x="400" y="176" font-size="10.5">Scales the matching activations down to cancel it,</text>
    <text x="400" y="191" font-size="10.5">so rounding lands with less relative error where it matters.</text>
    <text x="400" y="214" font-size="10.5" font-weight="600">Cost: a scaling search, no Hessian</text>
    <text x="400" y="232" font-size="13" font-weight="650" fill="var(--d-good)">9.5 min</text>
  </g>
</svg>
<figcaption>Same target bit-width, two philosophies: GPTQ accounts for error after introducing
it, AWQ arranges for less error to be introduced.</figcaption>
</figure>

**GPTQ** works layer by layer, column by column. For each linear layer it estimates a Hessian
from calibration data — essentially, how sensitive that layer's output is to error in each
weight. It quantizes one column at a time, and after each column adjusts the still-unquantized
columns to cancel the error it just introduced. By the end of a layer, *total output error* is
minimised, not just each weight's individual rounding error. That accounting is why it's slow.

**AWQ** skips the correction and tries to avoid the error. It looks at calibration activations,
finds which weight channels routinely see the largest activation values — those are where
rounding error hurts the output most — and scales those channels up before quantizing, scaling
the matching activations down to cancel it mathematically. No Hessian, no column-by-column
correction, just a scaling search. Roughly half the cost.

## Accuracy: five tasks, not one number

I picked a spread of task types deliberately. Multiple-choice benchmarks can hide damage —
a model can still rank the right answer highest even after its underlying probability
distribution has gotten noticeably noisier. Perplexity catches that; MMLU doesn't.

| Metric | fp16 | INT8 RTN | GPTQ INT4 | AWQ INT4 |
|---|---|---|---|---|
| MMLU | 71.76 | 71.66 | 70.92 | 70.94 |
| HellaSwag | 80.54 | 80.50 | 79.77 | 79.77 |
| ARC-Easy | 81.52 | 81.86 | 81.82 | 82.03 |
| GSM8K (n=200) | 52.0 | 52.0 | 56.0 | 60.0 |
| Wikitext PPL ↓ | 9.408 | 9.409 | 10.028 | 10.026 |

INT8 RTN is essentially lossless — perplexity matches fp16 to three decimal places with no
calibration data at all. That's the floor: this is what you get for doing nothing clever.

INT4 costs about **0.84 points of MMLU and 0.6 points of perplexity**. Both INT4 methods land
in the same place — MMLU 70.92 vs 70.94, identical HellaSwag, perplexity differing in the
third decimal. The GSM8K spread (56 vs 60) is on n=200; that's noise, not an edge for AWQ.

**So the real difference between GPTQ and AWQ here is quantization cost: 17.2 min vs 9.5 min
for the same outcome.** That's the reason to pick AWQ on this model — not accuracy, not
serving speed.

## Serving: sweep the concurrency, don't report one point

Measuring at a single load level hides how each scheme behaves. At concurrency 1 you're purely
memory-bound and see raw per-token latency; as concurrency climbs, batching kicks in and
throughput becomes the story.

<figure class="diagram">
<svg viewBox="0 0 700 240" role="img" aria-labelledby="tput-title">
  <title id="tput-title">Output throughput by variant across concurrency levels</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Output throughput (tok/s)</text>

    <line x1="60" y1="34" x2="60" y2="180" stroke="var(--d-line)"/>
    <line x1="60" y1="180" x2="680" y2="180" stroke="var(--d-line)"/>

    <!-- scale: 3200 tok/s over 146px -> 0.0456 px per tok -->
    <text x="54" y="184" font-size="9" text-anchor="end" fill="var(--d-muted)">0</text>
    <line x1="60" y1="112" x2="680" y2="112" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="54" y="116" font-size="9" text-anchor="end" fill="var(--d-muted)">1500</text>
    <line x1="60" y1="43" x2="680" y2="43" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="54" y="47" font-size="9" text-anchor="end" fill="var(--d-muted)">3000</text>

    <!-- groups at c=1, 4, 16, 32 -->
    <!-- c=1: fp16 49, int8 85, gptq 134, awq 133 -->
    <g>
      <rect x="80" y="178" width="16" height="2" fill="var(--d-muted)"/>
      <rect x="98" y="176" width="16" height="4" fill="var(--d-muted)" opacity="0.6"/>
      <rect x="116" y="174" width="16" height="6" fill="var(--d-accent)"/>
      <rect x="134" y="174" width="16" height="6" fill="var(--d-accent)" opacity="0.6"/>
      <text x="115" y="196" font-size="10" text-anchor="middle">c = 1</text>
      <text x="115" y="209" font-size="9" text-anchor="middle" fill="var(--d-muted)">49 → 134</text>
    </g>
    <!-- c=4: 185, 326, 507, 504 -->
    <g>
      <rect x="230" y="172" width="16" height="8" fill="var(--d-muted)"/>
      <rect x="248" y="165" width="16" height="15" fill="var(--d-muted)" opacity="0.6"/>
      <rect x="266" y="157" width="16" height="23" fill="var(--d-accent)"/>
      <rect x="284" y="157" width="16" height="23" fill="var(--d-accent)" opacity="0.6"/>
      <text x="265" y="196" font-size="10" text-anchor="middle">c = 4</text>
      <text x="265" y="209" font-size="9" text-anchor="middle" fill="var(--d-muted)">185 → 507</text>
    </g>
    <!-- c=16: 619, 947, 1288, 1290 -->
    <g>
      <rect x="380" y="152" width="16" height="28" fill="var(--d-muted)"/>
      <rect x="398" y="137" width="16" height="43" fill="var(--d-muted)" opacity="0.6"/>
      <rect x="416" y="121" width="16" height="59" fill="var(--d-accent)"/>
      <rect x="434" y="121" width="16" height="59" fill="var(--d-accent)" opacity="0.6"/>
      <text x="415" y="196" font-size="10" text-anchor="middle">c = 16</text>
      <text x="415" y="209" font-size="9" text-anchor="middle" fill="var(--d-muted)">619 → 1,290</text>
    </g>
    <!-- c=32: 1163, 2075, 3126, 3102 -->
    <g>
      <rect x="530" y="127" width="16" height="53" fill="var(--d-muted)"/>
      <rect x="548" y="85" width="16" height="95" fill="var(--d-muted)" opacity="0.6"/>
      <rect x="566" y="37" width="16" height="143" fill="var(--d-accent)"/>
      <rect x="584" y="38" width="16" height="142" fill="var(--d-accent)" opacity="0.6"/>
      <text x="565" y="196" font-size="10" text-anchor="middle">c = 32</text>
      <text x="565" y="209" font-size="9" text-anchor="middle" fill="var(--d-muted)">1,163 → 3,126</text>
    </g>

    <!-- legend -->
    <g transform="translate(60,224)">
      <rect x="0" y="-8" width="10" height="10" fill="var(--d-muted)"/><text x="15" y="1" font-size="9.5">fp16</text>
      <rect x="60" y="-8" width="10" height="10" fill="var(--d-muted)" opacity="0.6"/><text x="75" y="1" font-size="9.5">INT8 RTN</text>
      <rect x="150" y="-8" width="10" height="10" fill="var(--d-accent)"/><text x="165" y="1" font-size="9.5">GPTQ INT4</text>
      <rect x="245" y="-8" width="10" height="10" fill="var(--d-accent)" opacity="0.6"/><text x="260" y="1" font-size="9.5">AWQ INT4</text>
    </g>
  </g>
</svg>
<figcaption>INT4's advantage widens with load — 2.7× fp16 at concurrency 32. The two INT4
methods are indistinguishable at every point.</figcaption>
</figure>

### Concurrency 1 — one user waiting

| Variant | Mean TTFT | Mean time-per-output-token | Throughput |
|---|---|---|---|
| fp16 | 54.3 ms | 20.18 ms | 49.2 tok/s |
| INT8 RTN | 58.5 ms | 11.63 ms | 84.6 tok/s |
| GPTQ INT4 | 44.0 ms | 7.34 ms | 133.6 tok/s |
| AWQ INT4 | 43.7 ms | 7.38 ms | 132.9 tok/s |

INT4 wins outright — ~44 ms TTFT and ~7.3 ms per token against fp16's 54 ms and 20 ms. INT8
RTN sits in between: TTFT similar to fp16, but roughly half the per-token latency. A real win,
just not close to INT4.

### Concurrency 32 — throughput regime

| Variant | Throughput | TTFT p99 | Mean TPOT |
|---|---|---|---|
| fp16 | 1,162.7 tok/s | 373.5 ms | 25.25 ms |
| INT8 RTN | 2,074.9 tok/s | 613.5 ms | 13.64 ms |
| GPTQ INT4 | 3,125.6 tok/s | 314.0 ms | 9.21 ms |
| AWQ INT4 | 3,101.5 tok/s | 318.4 ms | 9.25 ms |

**INT8 RTN's tail latency is the interesting failure here.** It gets 1.8× fp16 throughput, but
its TTFT p99 at this load is *worse* than fp16 — 613 ms vs 373 ms. If tail latency under load
matters and not just aggregate tokens per second, INT8 RTN is a weak pick despite the
respectable throughput number. Reporting only throughput would have made it look fine.

One measurement I don't want to over-claim: at concurrency 16, all three quantized variants
show a TTFT p99 in the 4.6–4.9 s range while fp16 sits at 311 ms. That's out of line with both
the c=4 and c=32 results on either side of it, and I read it as a measurement artifact in that
sweep rather than a real property of the schemes — but I haven't isolated the cause, so I'm
flagging it rather than quietly dropping it.

## The lever that isn't about speed

Smaller weights leave more VRAM free for KV cache. On this card INT4 fits roughly **140×**
concurrent requests' worth of context against fp16's **99×**. When the real constraint is how
many concurrent users you can hold rather than how fast any single token is, that's the number
that decides the deployment.

## Bottom line

GPTQ/AWQ INT4 wins on every axis measured here — single-request latency, throughput under
load, and concurrency headroom. The only reason to step back is if the ~0.8 pt accuracy
regression is unacceptable for the task — and in that case the fallback is **fp16, not INT8
RTN**. INT8's accuracy is closest to fp16, but its serving numbers are the weakest of all four
once concurrency rises.

Between GPTQ and AWQ: take AWQ, purely because it gets to the same place in half the wall
clock.

## Limitations

This is one 7B model on one L40S 46GB. Things I'd expect to change at larger scale —
directional estimates based on how quantization behaves elsewhere, **not** measured here:

- **INT4 accuracy loss should shrink, not grow.** Larger models have more redundant capacity to
  absorb rounding error. The ~0.8 pt MMLU drop might be closer to 0.3–0.5 pt on a 70B-class
  model.
- **Compression ratio holds.** 2.7× is mostly a function of bit-width and group size, not
  parameter count.
- **Quantization wall clock gets much worse, especially for GPTQ.** The Hessian solve scales
  with layer width, so a 70B model could take hours where this took 17 minutes, while AWQ's
  scaling search should scale up more gently. The 1.8× gap likely widens — which makes AWQ's
  speed advantage matter more, not less.
- **fp16 stops being a single-GPU baseline.** A 70B model at fp16 is ~140 GB and needs
  multi-GPU tensor parallelism regardless, bringing cross-GPU communication into the latency
  picture. An INT4 version might still fit on one GPU — so at scale, part of "quantized is
  faster" becomes "quantized avoids cross-GPU overhead entirely," which is a bigger effect than
  the kernel-level win measured here.
- **Activation outliers get more prominent past ~6–7B.** Doesn't affect these weight-only
  results directly, but it's why W8A8/FP8 gets harder rather than easier at scale — the case
  for staying weight-only holds more strongly, not less.
- **Calibration set size (512 samples) was never tested for sensitivity.** Whether that's still
  enough for good Hessian/per-channel estimates on much larger weight matrices is an assumption
  carried over unchanged, not something this benchmark validated.
