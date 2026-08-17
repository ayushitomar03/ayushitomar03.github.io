---
title: 'Cloning a Voice with LoRA'
date: 2026-08-02
permalink: /blog/lora-voice-cloning/
tags:
  - speech
  - tts
  - fine-tuning
excerpt: "Fine-tuning a codec-LM TTS model on a single speaker so synthesis carries her rhythm, not just her timbre — the codec token layout that has to be exactly right, why training loss can't see any of it, and the evaluation design needed to tell whether it worked."
---

Zero-shot voice cloning can capture what a speaker sounds like, but not always how they speak.
Given a reference clip, it can reproduce the speaker's timbre — the characteristics that make the
voice sound like the same person. But it often misses prosody: where they pause, how their pitch
moves, and how quickly they speak. Those patterns are a major part of what makes a voice feel like
that person.

So the goal was narrow and measurable: fine-tune a codec-LM TTS model on a single speaker and test
whether it could reproduce her prosody better than a zero-shot system using the same speaker
reference.

We used Orpheus 3B with SNAC and applied LoRA (r=16) to the attention and MLP layers, training
24.3M parameters — just 0.73% of the full model.

## What "did it work" required

The hard part of this project was not training. It was building an evaluation that could
distinguish *sounds like her* from *is more prototypically her than her own recordings* — which
turn out to be different things that most metrics score identically.

Four measurements, each catching something the others can't:

| Measurement | What it catches |
|---|---|
| **SECS** (ECAPA + WavLM speaker embeddings) | Timbre and identity similarity — but it saturates, see below |
| **WER** (Whisper transcription) | Whether speech is still intelligible after fine-tuning |
| **Prosody distance** (F0 mean/std, pause length, speech rate) | The actual target: rhythm and intonation |
| **Blind A/B listening** | Whether any of the above corresponds to human judgement |

And three controls, which matter more than the metrics: the **real speaker** as an upper reference,
**zero-shot on the same speaker**, and the **unmodified base model** as a floor.

## The result

<figure class="diagram">
<svg viewBox="0 0 720 250" role="img" aria-labelledby="secs-title">
  <title id="secs-title">Speaker similarity scores against the real-vs-real ceiling</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Speaker similarity (ECAPA cosine — higher = more similar to the target speaker)</text>
    <text x="0" y="28" font-size="9.5" fill="var(--d-muted)">measured on a frozen 48-utterance eval subset, held out by book</text>

    <line x1="150" y1="46" x2="150" y2="196" stroke="var(--d-line)"/>
    <g font-size="9" fill="var(--d-muted)" text-anchor="middle">
      <text x="150" y="212">0.0</text>
      <text x="323" y="212">0.3</text>
      <text x="497" y="212">0.6</text>
      <text x="670" y="212">0.9</text>
    </g>

    <rect x="603" y="46" width="67" height="150" fill="var(--d-warn)" opacity="0.12"/>
    <line x1="603" y1="46" x2="603" y2="196" stroke="var(--d-warn)" stroke-dasharray="4 3"/>
    <text x="597" y="60" font-size="9.5" text-anchor="end" fill="var(--d-warn)">0.784 — real vs real ceiling</text>

    <text x="140" y="82" font-size="10.5" text-anchor="end" font-weight="600">LoRA r=16</text>
    <rect x="150" y="68" width="467" height="18" rx="3" fill="var(--d-accent)"/>
    <text x="625" y="82" font-size="10" font-weight="600">0.808</text>

    <text x="140" y="112" font-size="10.5" text-anchor="end">Real speaker</text>
    <rect x="150" y="98" width="459" height="18" rx="3" fill="var(--d-muted)" opacity="0.45"/>
    <text x="617" y="112" font-size="10">0.794</text>

    <text x="140" y="142" font-size="10.5" text-anchor="end">Zero-shot</text>
    <rect x="150" y="128" width="457" height="18" rx="3" fill="var(--d-muted)" opacity="0.3"/>
    <text x="615" y="142" font-size="10">0.790</text>

    <text x="140" y="172" font-size="10.5" text-anchor="end">Base model</text>
    <rect x="150" y="158" width="96" height="18" rx="3" fill="var(--d-warn)" opacity="0.6"/>
    <text x="254" y="172" font-size="10">0.166</text>

    <text x="0" y="234" font-size="10" fill="var(--d-muted)">Everything but the base model sits ABOVE the ceiling — no headroom left, so margins measured here are noise.</text>
    <text x="0" y="248" font-size="10" fill="var(--d-muted)">The base-model bar carries the result: without the adapter, this is a different person's voice.</text>
  </g>
</svg>
<figcaption>The ceiling is real audio from this speaker scored against other real audio from her —
session-to-session variation means even she doesn't score 1.0 against herself.</figcaption>
</figure>

Speaker similarity is the metric everyone reaches for first, and on this task it is **saturated**.
The adapter scores 0.808 — above the **real-vs-real ceiling of 0.784**, the score real recordings
of this speaker get against other real recordings of her. Once a system is above that line the
metric has no headroom left, so any margin measured there is noise. WavLM behaves identically
(0.975 / 0.981 / 0.984 across real / LoRA / zero-shot, indistinguishable).

Scoring above real audio does not mean the clone is better than the speaker. It means it is **more
prototypical**: the model converges on an average of her voice, while her real recordings carry
natural variation between sessions.

The one thing this metric does establish is that the adapter is doing the work: the base model
sits at **0.166**, below typical *cross-speaker* similarity, meaning it sounds like a different
person entirely. That rules out the base model simply happening to resemble her.

The practical lesson is to measure the ceiling before designing around a metric. A target like
"beat the baseline by +0.05 on SECS" is unreachable in principle here, and there is no way to
know that without scoring real audio against itself first.

### Where the adapter wins

<figure class="diagram">
<svg viewBox="0 0 720 250" role="img" aria-labelledby="prosody-title">
  <title id="prosody-title">Prosodic features of the clone and the zero-shot baseline against the real speaker</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Four prosodic features — the vertical line is the real speaker's value</text>

    <g font-size="10">
      <text x="0" y="50" font-weight="600">Pitch, mean</text>
      <text x="0" y="64" font-size="9" fill="var(--d-muted)">real: 224.5 Hz</text>
      <line x1="200" y1="32" x2="200" y2="72" stroke="var(--d-line)" stroke-width="1.5"/>
      <circle cx="206" cy="52" r="5" fill="var(--d-accent)"/>
      <text x="218" y="56" font-size="9.5" fill="var(--d-accent)">LoRA 223.4 — off by 1.1 Hz</text>
      <circle cx="352" cy="52" r="5" fill="var(--d-warn)" opacity="0.85"/>
      <text x="364" y="56" font-size="9.5" fill="var(--d-warn)">zero-shot 234.7 — off by 10.2 Hz</text>

      <text x="0" y="110" font-weight="600">Pitch, variation</text>
      <text x="0" y="124" font-size="9" fill="var(--d-muted)">real: 52.2 Hz</text>
      <line x1="200" y1="92" x2="200" y2="132" stroke="var(--d-line)" stroke-width="1.5"/>
      <circle cx="188" cy="112" r="5" fill="var(--d-accent)"/>
      <text x="120" y="116" font-size="9.5" text-anchor="end" fill="var(--d-accent)">LoRA 49.4</text>
      <circle cx="340" cy="112" r="5" fill="var(--d-warn)" opacity="0.85"/>
      <text x="352" y="116" font-size="9.5" fill="var(--d-warn)">zero-shot 66.0 — markedly over-expressive</text>

      <text x="0" y="170" font-weight="600">Mean pause</text>
      <text x="0" y="184" font-size="9" fill="var(--d-muted)">real: 0.18 s</text>
      <line x1="200" y1="152" x2="200" y2="192" stroke="var(--d-line)" stroke-width="1.5"/>
      <circle cx="232" cy="172" r="5" fill="var(--d-accent)"/>
      <text x="244" y="176" font-size="9.5" fill="var(--d-accent)">LoRA 0.26 s</text>
      <circle cx="128" cy="172" r="5" fill="var(--d-warn)" opacity="0.85"/>
      <text x="120" y="176" font-size="9.5" text-anchor="end" fill="var(--d-warn)">zero-shot 0.06 s — barely pauses</text>

      <text x="0" y="230" font-weight="600">Speech rate</text>
      <text x="0" y="244" font-size="9" fill="var(--d-muted)">real: 4.35 syll/s</text>
      <line x1="200" y1="212" x2="200" y2="252" stroke="var(--d-line)" stroke-width="1.5"/>
      <circle cx="128" cy="232" r="5" fill="var(--d-warn)" opacity="0.85"/>
      <text x="120" y="236" font-size="9.5" text-anchor="end" fill="var(--d-warn)">LoRA 3.87 — runs slow</text>
      <circle cx="172" cy="232" r="5" fill="var(--d-muted)"/>
      <text x="244" y="236" font-size="9.5" fill="var(--d-muted)">zero-shot 4.19 — closer here</text>
    </g>
  </g>
</svg>
<figcaption>Overall prosodic distance: 0.103 for the adapter against 0.187 for zero-shot — 45%
closer to the speaker on the dimension the project set out to improve.</figcaption>
</figure>

This is the result the project was about, and the one place the two systems clearly separate.
Zero-shot is over-expressive in pitch and barely pauses; the adapter tracks her mean pitch to
1.1 Hz. Its one clear loss is speech rate, where it runs slow.

**Blind A/B:** a listener scored 4 correct out of 12 (p = 0.388) — below chance, consistent with
the "more prototypical" reading above, since the listener tended to pick the synthetic clip as the
real one. But 12 trials with one listener is suggestive, not a listening study.

## Data decisions

**44.1 kHz, not 22.05 kHz.** I chose Hi-Fi TTS speaker 92 over the more common LJSpeech because it
downsamples cleanly to SNAC's 24 kHz. LJSpeech at 22.05 kHz would need *up*sampling, which invents
no information and leaves a spectral cliff at 11 kHz — and the model would faithfully learn that
cliff as "this voice has no high frequencies."

**Split by book, not by clip.** A book is the real recording session: fixed microphone, room and
register. Splitting any finer leaks acoustics between train and eval, so the eval would measure
memorisation rather than generalisation. Splitting by book gives a genuinely held-out set — 75% of
eval word-bigrams never appear in training.

**The corpus is much smaller than advertised.** 27.3 hours across 7 books yielded **5.3 usable
hours** in a 5–15 second window: 2,915 clips after an audit for clipping, SNR and DC offset, split
2,425 / 490. The median clip is 2.42 seconds because read-speech corpora segment per sentence
rather than per breath group — worth knowing before planning around a stated duration.

## Getting the model interface right

A codec-LM TTS model has a specific contract — a codec token layout and a prompt structure. Get
either wrong and the output is confidently, fluently wrong rather than obviously broken.

<figure class="diagram">
<svg viewBox="0 0 720 226" role="img" aria-labelledby="tokens-title">
  <title id="tokens-title">SNAC frame layout and prompt structure, assumed versus actual</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Two interface details, each of which silently changes the output</text>

    <text x="0" y="40" font-size="10.5" font-weight="600">SNAC frame — 3 hierarchical layers at 1× / 2× / 4× frame rate, flattened to 7 tokens</text>
    <g font-family="JetBrains Mono, monospace" font-size="10">
      <text x="0" y="62" fill="var(--d-muted)">assumed</text>
      <text x="100" y="62" fill="var(--d-warn)">L1 L2 L2 L3 L3 L3 L3</text>
      <text x="300" y="62" font-size="9.5" fill="var(--d-muted)">grouped by layer</text>
      <text x="0" y="80" fill="var(--d-muted)">actual</text>
      <text x="100" y="80" fill="var(--d-good)">L1 L2 L3 L3 L2 L3 L3</text>
      <text x="300" y="80" font-size="9.5" fill="var(--d-muted)">interleaved</text>
    </g>

    <text x="0" y="106" font-size="9.5" fill="var(--d-muted)">Correlation of the decoded waveform with real audio, by ordering:</text>
    <g font-size="9.5">
      <text x="0" y="126">correct</text>
      <rect x="150" y="116" width="240" height="12" rx="2" fill="var(--d-good)"/><text x="398" y="126" font-weight="600">+0.838</text>
      <text x="0" y="144">grouped (assumed)</text>
      <rect x="150" y="134" width="40" height="12" rx="2" fill="var(--d-warn)"/><text x="198" y="144" font-weight="600">+0.139</text>
      <text x="0" y="162">shifted one frame</text>
      <rect x="150" y="152" width="2" height="12" fill="var(--d-muted)"/><text x="198" y="162" fill="var(--d-muted)">−0.012</text>
      <text x="0" y="180">random codes</text>
      <rect x="150" y="170" width="2" height="12" fill="var(--d-muted)"/><text x="198" y="180" fill="var(--d-muted)">−0.005</text>
    </g>

    <line x1="0" y1="192" x2="720" y2="192" stroke="var(--d-line)"/>
    <text x="0" y="208" font-size="10.5" font-weight="600">Prompt sequence</text>
    <g font-family="JetBrains Mono, monospace" font-size="9.5">
      <text x="130" y="208" fill="var(--d-warn)">[SOH] text [EOH][SOAI][SOS]</text>
      <text x="350" y="208" font-family="Inter, sans-serif" fill="var(--d-muted)">→ fluent speech, wrong words · 116% WER</text>
      <text x="130" y="222" fill="var(--d-good)">[SOH] [128000] text [128009] [EOH][SOAI][SOS]</text>
      <text x="440" y="222" font-family="Inter, sans-serif" fill="var(--d-muted)">→ 0.0% WER</text>
    </g>
  </g>
</svg>
<figcaption>+0.139 is recognisable speech in a distorted voice — the coarsest layer sits at position
0 in both orderings and survives, while everything finer scrambles.</figcaption>
</figure>

The two missing tokens in the prompt are Llama-3's `<|begin_of_text|>` and `<|eot_id|>`. The
reference implementation gets them from `add_special_tokens=True`, which is easy to miss when you
assemble the prompt yourself.

The check that catches both is now a hard gate the training target depends on, and it takes two
minutes: **make the unmodified base model speak a sentence through your harness before fine-tuning
anything.** I had a verification script, but it only proved my encoder and decoder agreed *with
each other* — a different claim from agreeing with the model. A self-consistent but wrong
convention round-trips cleanly and trains to a textbook loss curve.

Which is the broader lesson: **training loss is blind to almost everything that matters here.** Two
runs differing only in 6.6% corrupted training text scored 5.7224 and 5.7226, while their synthesis
WER differed by 33.6% versus 24.9%. A run with a scrambled codec plateaued at 5.72 for eight epochs
looking converged; fixing the layout put *epoch 0* at 4.16. Loss tells you optimisation is
proceeding, not what is being optimised.

## Training

`r=16, α=32`, attention + MLP, bf16, batch 4 × grad-accum 4, lr 1e-4 cosine, gradient checkpointing
on. Checkpoint selection on validation minimum, which mattered: epoch 7 had the best *training* loss
and among the worst eval, so a default "keep the last epoch" would have shipped a worse model.

```
ep 0  train 4.4418  eval 4.1585  <- best
ep 4  train 3.8574  eval 4.0610  <- best   SELECTED
ep 7  train 3.7713  eval 4.0848
early stop: no improvement for 3 epochs
```

Nearly all adaptation happened in the first two epochs; epochs 3–4 contributed 0.003 combined and
5–7 actively hurt. `epochs: 4` would halve the cost of any future sweep.

**My VRAM planning was wrong in a way worth repeating.** I budgeted from attention activations and
concluded 24 GB would be comfortable. For a codec-LM the **LM-head logits dominate**, because the
vocabulary is 156,940 (7 × 4096 audio tokens plus text):

```
batch 8 × seq 1286 × vocab 156,940 × 4 bytes = 6.5 GB   for ONE logits tensor
```

Measured: batch 8 without checkpointing OOMs on 44 GiB; batch 4 with checkpointing sits at
17.6 GiB. A 24 GB card would not have run this. For any codec-LM, budget `batch × seq × vocab`
first and attention second.

## Engineering that made iteration affordable

**Stub backends behind every heavy model.** `--backend stub` runs the entire pipeline — including
the training loop — on a laptop with no weights. 113 unit tests execute in about 2 seconds, and the
same code runs on the GPU box by flipping a flag. Every structural bug that could be caught without
a GPU was caught without one.

**A preflight data gate that `make train` depends on**, so no run can skip it. Combined with the
base-model speech check, the expensive path cannot start until the cheap checks pass.
