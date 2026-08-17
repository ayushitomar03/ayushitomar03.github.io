---
title: "GPU Cluster Observability & MFU Dashboard"
excerpt: "Built and deployed. Prometheus + Grafana over DCGM, plus in-process PyTorch instrumentation exporting MFU, memory-bandwidth utilization, and NCCL comm fraction — which turned 'why did adding a GPU make it slower' into a five-minute diagnosis."
collection: projects
order: 3
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Built &amp; deployed</span>
  <span class="pill">DCGM · Prometheus · Grafana</span>
  <span class="pill">PyTorch instrumentation</span>
</p>

`nvidia-smi` reporting 100% utilization only means a kernel was resident on the device. It says
nothing about whether that kernel was doing useful arithmetic, sitting in an all-reduce, or
waiting on memory. This project builds the instrumentation that can tell those apart.

## What was built

- **`gpu_metrics.py`** — reusable in-process instrumentation exporting four Prometheus gauges:
  `gpu_mfu`, `gpu_mem_bw_util`, `gpu_comm_fraction`, and `gpu_step_time_ms`. Step timing uses
  CUDA events, not wall clock, since asynchronous kernel launches make `time.time()` measure
  how long Python took to *queue* work rather than how long the GPU spent doing it.
- **`NCCLTimer`** — a nested CUDA-event timer that attributes collective time to the enclosing
  step, so communication cost is measured rather than inferred.
- **Per-GPU-variant peak constants** for L40S, A100 PCIe, and A100 SXM4 — passed explicitly
  rather than defaulted, because A100 PCIe and SXM4 share a compute die but differ in memory
  bandwidth (1,935 vs 2,039 GB/s), and using the wrong one makes every utilization number
  silently optimistic forever.
- **A Grafana dashboard** deliberately mixing hardware and application layers — XID hardware-
  fault counters sit next to MFU, so "did the model get slower or did the hardware get sick"
  is one glance rather than an investigation.
- **Two validation jobs** with known ground truth: a matmul + 200 MB all-reduce (known FLOP
  count, exercises both the MFU and comm-fraction paths) and an elementwise multiply over 400M
  bf16 elements (near-zero compute, known byte count, validates the bandwidth hook).

## What it found

Wired into a real FSDP2 training run: a single-GPU run of the identical config averaged
**271.5 ms/step at 4.95% MFU**, while the 2-GPU FSDP run averaged **515.1 ms/step at 2.67%
MFU** — roughly **+244 ms of communication overhead per step, nearly equal to the compute time
itself**, on a box with no NVLink (PHB topology, PCIe through the CPU host bridge).

Without comm-fraction attribution, "adding a GPU made it slower" is a mystery. With it, the
cause is on the dashboard.

[Read the full writeup →](/blog/model-flops-utilization/)

## Open

Per-kernel attribution via `torch.profiler`, alerting on MFU regression rather than just
visualising it, and distinguishing data-loader stalls from host-side stalls.
