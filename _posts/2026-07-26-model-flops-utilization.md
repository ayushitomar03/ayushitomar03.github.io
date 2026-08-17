---
title: 'GPU Utilization Is Lying to You: Instrumenting MFU on a Real Cluster'
date: 2026-07-26
permalink: /blog/model-flops-utilization/
tags:
  - training-infra
  - performance
  - observability
excerpt: "nvidia-smi says 100%. Your MFU is 2.6%. Building the instrumentation to tell those apart — CUDA-event step timing, NCCL comm fraction, memory-bandwidth utilization — and the finding it produced: communication costing more than the compute itself."
published: false
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; deployed</span>
  <span class="pill">DCGM · Prometheus · Grafana</span>
  <span class="pill">PyTorch instrumentation</span>
  <span class="pill">L40S / A100</span>
</p>

`nvidia-smi` reports 100% GPU utilization. What that actually means is "at least one kernel was
resident on the device during the sampling window." It says nothing about whether that kernel
was doing useful arithmetic, sitting in an all-reduce, or waiting on a memory read.

You can hit 100% utilization while doing almost no useful work. The metric that doesn't lie is
**Model FLOPs Utilization** — achieved FLOPs divided by the hardware's peak — and nothing
reports it for you. You have to instrument it.

<ul class="keyresults">
  <li><span class="keyresults__value">+244 ms</span><span class="keyresults__label">FSDP communication overhead per step, nearly equal to the compute</span></li>
  <li><span class="keyresults__value">271.5 → 515.1</span><span class="keyresults__label">ms per step, 1 GPU → 2 GPUs on the identical config</span></li>
  <li><span class="keyresults__value">4.95% → 2.67%</span><span class="keyresults__label">MFU across that same change</span></li>
  <li><span class="keyresults__value">30–55%</span><span class="keyresults__label">typical production MFU — this demo sits far below it</span></li>
</ul>

## The metric

For a dense transformer, FLOPs per token follow a standard approximation:

```
train   (forward + backward):  ≈ 6 × N × tokens
prefill (forward only):        ≈ 2 × N × tokens
```

where `N` is parameter count. Divide achieved FLOPs-per-second by the hardware's dense
BF16 peak and you have MFU:

```python
def count_dense_transformer_flops(num_params, num_tokens, mode="train"):
    multiplier = 6 if mode == "train" else 2
    return multiplier * num_params * num_tokens

# achieved / peak
mfu = (num_flops / step_seconds) / peak_flops_bf16
```

Simple arithmetic. The engineering is entirely in getting `step_seconds` and `peak_flops_bf16`
right — and both have a trap.

### Trap 1: wall-clock time is the wrong clock

CUDA kernel launches are asynchronous. Timing a step with `time.time()` measures how long it
took Python to *queue* the work, not how long the GPU spent doing it. You get numbers that are
smooth, plausible, and wrong.

The fix is CUDA events, which are recorded in the stream itself:

```python
class StepTimer:
    def __enter__(self):
        self._comm_time_ms = 0.0
        self._start_evt.record()
        return self

    def __exit__(self, *exc):
        self._end_evt.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self._start_evt.elapsed_time(self._end_evt)
```

### Trap 2: the peak-FLOPS constant is per-GPU-variant, and getting it wrong fails silently

This one bit me. The instrumentation was originally written for an L40S box:

```python
L40S_PEAK_FLOPS_BF16      = 181e12   # FLOPs/sec, BF16 tensor core, dense
L40S_PEAK_MEM_BW          = 864e9    # bytes/sec, GDDR6
```

When I reused it on an A100 box, MFU kept reporting numbers that looked fine — because the
formula still ran, just against the wrong denominator. Worse, A100 PCIe and A100 SXM4 share
the same compute die (312 TFLOPS) but *differ in memory bandwidth* — 1,935 vs 2,039 GB/s. Use
the SXM constant on a PCIe card and your bandwidth utilization is quietly 5% optimistic
forever.

```python
A100_PCIE_PEAK_FLOPS_BF16 = 312e12
A100_PCIE_PEAK_MEM_BW     = 1935e9   # HBM2e, PCIe variant
A100_SXM4_PEAK_MEM_BW     = 2039e9   # HBM2e, SXM4 variant
```

So `StepTimer` now takes them explicitly rather than defaulting silently, and the right move is
confirming the variant via `nvidia-smi --query-gpu=name` rather than assuming. **A metric that
is silently wrong is worse than no metric** — it produces confident decisions built on nothing.

## Three metrics, not one

MFU alone can't tell you *why* a step was slow. The instrumentation exports three:

| Gauge | What it answers |
|---|---|
| `gpu_mfu` | Are compute-bound phases actually using the tensor cores? |
| `gpu_mem_bw_util` | Are memory-bound phases actually saturating HBM? |
| `gpu_comm_fraction` | What fraction of step wall clock went into NCCL collectives? |
| `gpu_step_time_ms` | The denominator everything else is measured against |

The comm fraction comes from nesting a second CUDA-event timer inside the step timer, so
collectives are attributed rather than inferred:

```python
with StepTimer(rank, A100_PCIE_PEAK_FLOPS_BF16, A100_PCIE_PEAK_MEM_BW) as timer:
    loss = model(batch).backward()
    with NCCLTimer(timer):          # adds its elapsed time to the step's comm total
        dist.all_reduce(grads)
timer.report_mfu(count_dense_transformer_flops(n_params, tokens))
```

A step that's slow with high comm fraction is an interconnect problem. A step that's slow with
low MFU *and* low bandwidth utilization is a stall — data loading, kernel launch overhead,
small batch. Those need different fixes, and one number can't distinguish them.

## The stack

<figure class="diagram">
<svg viewBox="0 0 740 250" role="img" aria-labelledby="obs-title">
  <title id="obs-title">Metrics pipeline from GPU and training processes through Prometheus to Grafana</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">

    <text x="0" y="12" font-size="12" font-weight="600">Two independent metric sources, one dashboard</text>

    <!-- hardware layer -->
    <text x="0" y="42" font-size="9.5" font-weight="600" fill="var(--d-muted)">HARDWARE</text>
    <rect x="0" y="52" width="150" height="46" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="12" y="70" font-size="10.5" font-weight="600">DCGM exporter</text>
    <text x="12" y="86" font-size="9.5" fill="var(--d-muted)">:9400 · util, power, XID</text>

    <rect x="0" y="106" width="150" height="46" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="12" y="124" font-size="10.5" font-weight="600">node exporter</text>
    <text x="12" y="140" font-size="9.5" fill="var(--d-muted)">:9100 · CPU, host mem</text>

    <!-- app layer -->
    <text x="0" y="182" font-size="9.5" font-weight="600" fill="var(--d-muted)">APPLICATION</text>
    <rect x="0" y="192" width="150" height="46" rx="7" fill="var(--d-fill)" stroke="var(--d-accent)" stroke-width="1.5"/>
    <text x="12" y="210" font-size="10.5" font-weight="600" fill="var(--d-accent)">gpu_metrics.py</text>
    <text x="12" y="226" font-size="9.5" fill="var(--d-muted)">:9401+rank · one per rank</text>

    <!-- arrows in -->
    <line x1="150" y1="75" x2="255" y2="120" stroke="var(--d-line)"/>
    <line x1="150" y1="129" x2="255" y2="132" stroke="var(--d-line)"/>
    <line x1="150" y1="215" x2="255" y2="152" stroke="var(--d-accent)"/>

    <!-- prometheus -->
    <rect x="260" y="100" width="170" height="66" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="345" y="122" font-size="11" font-weight="600" text-anchor="middle">Prometheus</text>
    <text x="345" y="139" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">scrape_interval: 5s</text>
    <text x="345" y="154" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">one target per rank</text>

    <line x1="430" y1="133" x2="545" y2="133" stroke="var(--d-line)"/>
    <polygon points="553,133 543,129 543,137" fill="var(--d-line)"/>

    <!-- grafana -->
    <rect x="555" y="60" width="185" height="146" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="647" y="80" font-size="11" font-weight="600" text-anchor="middle">Grafana</text>
    <g font-size="9.5" fill="var(--d-muted)">
      <text x="570" y="100">GPU utilization / memory</text>
      <text x="570" y="116">Power draw · temperature</text>
      <text x="570" y="132">NVLink bandwidth</text>
      <text x="570" y="148" fill="var(--d-warn)">XID errors (hardware faults)</text>
      <text x="570" y="164" fill="var(--d-accent)">MFU / step time</text>
      <text x="570" y="180" fill="var(--d-accent)">Request queue length</text>
      <text x="570" y="196">Host CPU / memory</text>
    </g>

    <text x="0" y="250" font-size="10" fill="var(--d-muted)">Per-rank HTTP servers instead of prometheus_client's multiprocess mode — simpler, and Prometheus just scrapes one target per worker.</text>
  </g>
</svg>
<figcaption>DCGM answers "is the hardware healthy?"; the in-process exporter answers "is the
model using it?" Neither is sufficient alone.</figcaption>
</figure>

Each training process runs its own metrics HTTP server on `METRICS_BASE_PORT + rank` rather
than using `prometheus_client`'s multiprocess mode — simpler, and Prometheus just scrapes one
target per worker:

```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'dcgm'
    static_configs: [{ targets: ['localhost:9400'] }]
  - job_name: 'node'
    static_configs: [{ targets: ['localhost:9100'] }]
  - job_name: 'app_metrics'
    static_configs: [{ targets: ['localhost:9401', 'localhost:9402'] }]
```

The Grafana dashboard deliberately mixes both layers. **XID errors** sit next to MFU on
purpose: when throughput drops, the first question is whether the model got slower or the
hardware got sick, and having those on one screen turns a multi-hour investigation into a
glance.

## Validating the instrumentation before trusting it

Instrumentation that has never been checked against a known answer is just a number generator.
Two toy jobs establish that each hook reads what it claims to.

**Compute + communication** — a 2-process job doing a large matmul (known FLOP count:
`2 × 8192³`) followed by a 200 MB all-reduce, so both the MFU path and the comm-fraction path
have a ground truth to be checked against:

```python
with StepTimer(RANK) as timer:
    c = torch.matmul(a, b)
    with NCCLTimer(timer):
        dist.all_reduce(comm_tensor)
```

**Pure memory bandwidth** — an elementwise multiply over 400M bf16 elements, which does
essentially zero arithmetic and reads + writes a known byte count. If `report_mem_bw_util`
doesn't approach the card's rated bandwidth here, it's wrong:

```python
y = x * 2.0                                   # read N×2 bytes + write N×2 bytes
num_bytes = x.numel() * x.element_size() * 2
util = timer.report_mem_bw_util(num_bytes)
```

## What the instrumentation actually found

The payoff came when I wired this into a real FSDP2 training run (the same one from
[the fault-tolerance post](/blog/fault-tolerant-training/)) — an 85M-param GPT on 2× A100-80GB
PCIe.

<figure class="diagram">
<svg viewBox="0 0 700 200" role="img" aria-labelledby="mfu-title">
  <title id="mfu-title">Step time on one GPU versus two GPUs with FSDP sharding</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Step time, identical model and batch config</text>

    <!-- scale: 550ms over 480px -> 0.873 px/ms -->
    <text x="0" y="52" font-size="10.5" font-weight="600">1 GPU</text>
    <text x="0" y="67" font-size="9.5" fill="var(--d-muted)">no sharding</text>
    <rect x="110" y="38" width="237" height="26" rx="4" fill="var(--d-accent)" opacity="0.75"/>
    <text x="357" y="56" font-size="11" font-weight="600">271.5 ms</text>
    <text x="440" y="56" font-size="10" fill="var(--d-muted)">MFU 4.95%</text>

    <text x="0" y="112" font-size="10.5" font-weight="600">2 GPUs</text>
    <text x="0" y="127" font-size="9.5" fill="var(--d-muted)">FSDP2 full shard</text>
    <rect x="110" y="98" width="237" height="26" rx="4" fill="var(--d-accent)" opacity="0.75"/>
    <rect x="347" y="98" width="213" height="26" rx="4" fill="var(--d-warn)" opacity="0.7"/>
    <text x="570" y="116" font-size="11" font-weight="600">515.1 ms</text>
    <text x="440" y="146" font-size="10" fill="var(--d-muted)">MFU 2.67%</text>

    <text x="228" y="88" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">compute</text>
    <text x="453" y="88" font-size="9.5" text-anchor="middle" fill="var(--d-warn)">+244 ms all-gather / reduce-scatter</text>

    <line x1="347" y1="132" x2="560" y2="132" stroke="var(--d-warn)"/>
    <line x1="347" y1="128" x2="347" y2="136" stroke="var(--d-warn)"/>
    <line x1="560" y1="128" x2="560" y2="136" stroke="var(--d-warn)"/>

    <text x="0" y="176" font-size="10.5" fill="var(--d-muted)">No NVLink on this box (PHB topology — PCIe through the CPU host bridge).</text>
    <text x="0" y="192" font-size="10.5" fill="var(--d-muted)">Communication cost is fixed per step; it only pays for itself once compute-per-step is large enough to amortize it.</text>
  </g>
</svg>
<figcaption>Adding a second GPU nearly doubled step time. Without comm-fraction instrumentation
this reads as "distributed training is slow"; with it, the cause is unambiguous.</figcaption>
</figure>

A single-GPU run of the identical model and batch config — no sharding, no cross-GPU traffic —
averages **271.5 ms/step at 4.95% MFU**. The 2-GPU FSDP run averages **515.1 ms/step at 2.67%
MFU**. That's roughly **+244 ms of overhead per step, almost exactly equal to the compute time
itself.**

The box has no NVLink — PHB topology, PCIe through the CPU host bridge, confirmed with
`nvidia-smi topo -m`. So FSDP's per-block all-gather and reduce-scatter dominate a small
model's actual compute.

This isn't a bug, and it isn't an argument against FSDP. It's a direct consequence of model
size versus interconnect bandwidth: FSDP's benefit (fitting bigger models, scaling across more
devices) trades against a **fixed per-step communication cost** that only pays for itself once
compute-per-step is large enough to amortise it. An 85M-param model on a PCIe box is on the
wrong side of that crossover.

The point is that the instrumentation made this a five-minute diagnosis instead of a guess.
Without comm-fraction attribution, "adding a GPU made it slower" is a mystery. With it, the
answer is on the dashboard.

## Why 2.67% MFU isn't the scandal it looks like

Production training MFU runs 30–55%. This is an order of magnitude below that, and the reason
is disclosed rather than defended: the model is deliberately tiny (85M params, so a full
failure-injection run completes in ~4 minutes rather than hours). At that scale, Python and
kernel-launch overhead plus communication dominate a genuinely small amount of real compute.

MFU here was used as a **diagnostic** — a before/after comparison to prove that recovering
from a worker death introduces no lingering throughput regression (2.67% pre-kill, 2.66%
post-resume) — not as something to maximise. `torch.compile`, CUDA graphs, and larger batch
sizes are the levers for the latter, and they're a different project.

## What I'd add next

- **Per-kernel attribution.** Comm fraction tells you communication is expensive; it doesn't
  tell you which collective. Wiring `torch.profiler` traces into the same pipeline would close
  that.
- **Alerting on MFU regression**, not just visualisation. A run whose MFU quietly drops 20% at
  hour 6 is exactly the thing dashboards are bad at catching and alerts are good at.
- **Data-loader stall attribution.** Currently a stalled step shows up as low MFU with low
  bandwidth utilization — correct, but it doesn't distinguish "waiting on disk" from "waiting
  on the host."
