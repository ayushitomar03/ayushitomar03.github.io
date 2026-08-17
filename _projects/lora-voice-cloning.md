---
title: "LoRA Voice Cloning — Training a Codec-LM TTS Model on One Speaker"
excerpt: "Built and evaluated. A 93 MB LoRA adapter on Orpheus 3B that a listener could not distinguish from the real speaker in a blind test (4/12, p=0.388). Three bugs invisible to the loss curve cost two training runs."
collection: projects
order: 1
status: Published
---

<p class="pill-row">
  <span class="pill pill--built">Trained &amp; evaluated</span>
  <span class="pill">Orpheus 3B · SNAC codec</span>
  <span class="pill">LoRA r=16 · 93 MB adapter</span>
  <span class="pill">1× L40S · ~5h · ~$6</span>
</p>

Fine-tuning a codec-LM TTS model on a single target speaker so synthesis carries that speaker's
**prosody as well as timbre** — the part zero-shot cloning reliably gets wrong.

<ul class="keyresults">
  <li><span class="keyresults__value">4 / 12</span><span class="keyresults__label">blind A/B — listener scored below chance (p = 0.388)</span></li>
  <li><span class="keyresults__value">0.166 → 0.808</span><span class="keyresults__label">speaker similarity, base model vs adapter</span></li>
  <li><span class="keyresults__value">1.1 Hz</span><span class="keyresults__label">off the speaker's mean pitch; zero-shot is 10.2 Hz off</span></li>
  <li><span class="keyresults__value">0.73%</span><span class="keyresults__label">of model params trained (24.3M)</span></li>
</ul>

## Result

| Condition | SECS (ECAPA) | SECS (WavLM) | WER | Prosody dist. |
|---|---|---|---|---|
| **LoRA r=16** | **0.808** | 0.981 | 1.8% | **0.103** |
| Real speaker | 0.794 | 0.975 | 1.0% | — |
| Zero-shot (Chatterbox) | 0.790 | 0.984 | 1.8% | 0.187 |
| Base Orpheus (no adapter) | 0.166 | 0.657 | 1.8% | 0.258 |
| *ceiling (real vs real)* | *0.784* | *0.974* | | |

**The gate I set failed, and the honest reading is that the gate was wrong.** I'd required LoRA
to beat zero-shot by +0.050 SECS; it managed +0.018. But both systems score *above the
real-vs-real ceiling of 0.784* — once two systems exceed the ceiling the metric has no headroom
and a 0.05 margin is unreachable in principle, not narrowly missed.

Scoring above real audio doesn't mean "better than the speaker." It means **more prototypical**:
the model converges on the average of her voice while real recordings carry natural
session-to-session variation. The blind test supports this independently — the listener scored
*below* chance, tending to pick the synthetic clip as the real one.

**Where LoRA genuinely wins is prosody**, which is what the whole project was about:

| Feature | Real | LoRA | Zero-shot |
|---|---|---|---|
| F0 mean (Hz) | 224.5 | **223.4** | 234.7 |
| F0 std (Hz) | 52.2 | **49.4** | 66.0 |
| Mean pause (s) | 0.18 | **0.26** | 0.06 |
| Syllables/sec | 4.35 | 3.87 | **4.19** |

Zero-shot is markedly over-expressive in pitch and barely pauses at all. LoRA's one clear loss is
speech rate — it runs slow.

The **base-model control is the most important number here**: 0.166 → 0.808. Without the adapter
Orpheus scores *below* typical cross-speaker similarity, so none of the achieved similarity is the
base model happening to sound like her.

## What was built

~5,500 lines: an importable, unit-tested library plus 12 CLI stages and 113 tests.

- **Stub backends behind every heavy model** — `--backend stub` runs the entire pipeline,
  training loop included, on a laptop with no weights. 113 tests in 2 seconds; the same code runs
  on the GPU box by flipping a flag.
- **A 7-group preflight data gate** that `make train` depends on, so no run can skip it.
- **Checkpoint selection on validation minimum**, which fired on all three runs — in the final
  run the best *training* loss was epoch 7 and its eval was the worst of the last four, so
  "keep the last epoch" would have silently shipped a worse model.
- **Separate venvs** for the zero-shot baseline, which pins torch 2.6 against training's 2.13.

## Data

Hi-Fi TTS speaker 92, CC BY 4.0. Chosen over LJSpeech because it's 44.1 kHz and downsamples
cleanly to SNAC's 24 kHz — LJSpeech's 22.05 kHz would need *upsampling*, inventing no information
and leaving a spectral cliff at 11 kHz that the model would learn as "this voice has no highs."

**Split by book, not by clip.** A book is the real recording session — fixed mic, room, register.
Splitting finer leaks acoustics between train and eval. Result: 75% of eval word-bigrams never
appear in training.

**A corpus reality check worth repeating:** 27.3 advertised hours yielded **5.3 usable hours** in
a 5–15 s window. The median clip is 2.42 s and 65% are under 3 s, because read-speech corpora
segment per sentence rather than per breath group. Don't plan around an advertised duration.

## The three bugs

All invisible to the training loss. Full account in [the writeup](/blog/lora-voice-cloning/).

1. **Double text normalisation** — the corpus ships pre-normalised text; running our normaliser
   over it again turned ALL-CAPS emphasis into spelled-out initialisms (`"WHAT"` → `"W H A T"`)
   in 6.6% of clips. The Whisper ground-truth check *passed* on this data.
2. **SNAC frame interleaving** — we grouped the codec's three hierarchical layers as
   `L1 L2 L2 L3 L3 L3 L3`; Orpheus interleaves them `L1 L2 L3 L3 L2 L3 L3`. Correlation with real
   audio: 0.838 correct vs 0.139 grouped.
3. **Incomplete prompt sequence** — two missing Llama-3 special tokens produced fluent speech with
   the *wrong words*, at 116% WER. Adding them took it to 0.0%.

The verification script proved the encoder and decoder agreed **with each other**, and I treated
that as proof they agreed with the *model*. Different claims. The missing check took two minutes
to write: make the unmodified base model speak a sentence before fine-tuning it. It's now a hard
gate.

## Training

r=16, α=32, attention + MLP, bf16, batch 4 × accum 4, lr 1e-4 cosine. **51 minutes, peak 17.6 GiB,
93 MB adapter.** Nearly all adaptation happened in the first two epochs — epochs 3–4 contributed
0.003 combined and 5–7 actively hurt.

**My VRAM estimate was wrong in an instructive way.** I'd planned for 24 GB reasoning from
attention activations. For a codec-LM the **LM-head logits dominate**, because the vocabulary is
156,940: `batch 8 × seq 1286 × vocab 156,940 × 4 bytes = 6.5 GB` for one logits tensor. Batch 8
without checkpointing OOMs on 44 GiB. A 24 GB card would not have run this.

## Known weaknesses

- **Short utterances** — the shortest eval clip ran **1.65× too long**. 92% of training data was
  5–15 s, so the model has little experience terminating quickly. A data-distribution issue,
  distinct from the bugs.
- **SECS is strongly length-dependent** — ECAPA needs ~3 s to estimate identity, so scores on
  sub-2 s clips are largely measurement noise.
- **The blind test is n=12 with one listener.** It's evidence, not a listening panel.
- **The data-volume ablation hasn't run** — which is the single most valuable remaining
  experiment, since it answers "how much audio do I need for the next speaker."
