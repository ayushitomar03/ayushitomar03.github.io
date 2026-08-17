---
title: 'Cloning a Voice with LoRA'
date: 2026-08-02
permalink: /blog/lora-voice-cloning/
tags:
  - speech
  - tts
  - fine-tuning
excerpt: "Fine-tuning a codec-LM TTS model on one speaker cut prosodic distance by 45%. Getting there exposed several less obvious problems in codec-LM training: token ordering, prompt formatting, misleading loss curves, and a very different VRAM bottleneck from standard text LMs."
---

Zero-shot voice cloning is good at reproducing **what a speaker sounds like**. It is less reliable
at reproducing **how they speak**.

A short reference clip can provide enough information to reproduce timbre — the characteristics that
make a voice recognizable as the same person. But it does not necessarily capture prosody: how pitch
changes, where the speaker pauses, and how quickly they speak.

This project tests whether fine-tuning on a single speaker can recover that second part. I
fine-tuned **Orpheus 3B with SNAC** using LoRA (`r=16`) on the attention and MLP layers, training
**24.3M parameters, or 0.73% of the model**.

The result was a **45% reduction in prosodic distance** compared with zero-shot cloning. Getting
there exposed several less obvious problems in codec-LM training: token ordering, prompt
formatting, misleading loss curves, and a very different VRAM bottleneck from standard text LMs.

## How the model works

Orpheus is a **codec language model**: instead of directly generating a waveform, it generates
discrete audio tokens.

<figure class="diagram">
<svg viewBox="0 0 700 84" role="img" aria-labelledby="pipe-t">
  <title id="pipe-t">How a codec language model turns text into a waveform</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <g font-size="11" text-anchor="middle">
      <rect x="0" y="22" width="112" height="42" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="56" y="48">text</text>

      <rect x="146" y="22" width="150" height="42" rx="8" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/>
      <text x="221" y="42" font-weight="600">Orpheus 3B</text>
      <text x="221" y="56" font-size="9" fill="var(--d-muted)">transformer</text>

      <rect x="330" y="22" width="150" height="42" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="405" y="42" font-weight="600">SNAC tokens</text>
      <text x="405" y="56" font-size="9" fill="var(--d-muted)">discrete audio codes</text>

      <rect x="514" y="22" width="186" height="42" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="607" y="42" font-weight="600">SNAC decoder</text>
      <text x="607" y="56" font-size="9" fill="var(--d-muted)">→ waveform</text>
    </g>
    <g stroke="var(--d-line)">
      <line x1="112" y1="43" x2="128" y2="43"/><polygon points="146,43 130,37 130,49" fill="var(--d-line)"/>
      <line x1="296" y1="43" x2="312" y2="43"/><polygon points="330,43 314,37 314,49" fill="var(--d-line)"/>
      <line x1="480" y1="43" x2="496" y2="43"/><polygon points="514,43 498,37 498,49" fill="var(--d-line)"/>
    </g>
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">predicted autoregressively, one audio token at a time</text>
  </g>
</svg>
</figure>

SNAC is the neural audio codec. It compresses speech into discrete tokens representing the audio at
different levels of detail. Orpheus learns to predict those tokens autoregressively, much like a
language model predicts the next text token.

This architecture has two consequences that matter later:

1. audio-token ordering is part of the model interface and must match exactly;
2. the vocabulary is unusually large because it contains both text and audio tokens.

## Data: split by recording session, not clip

I used **Hi-Fi TTS speaker 92**, starting from 44.1 kHz audio and downsampling to SNAC's 24 kHz
input rate.

This was preferable to starting with a 22.05 kHz corpus such as LJSpeech. Upsampling 22.05 → 24 kHz
changes the sample rate but cannot restore frequencies above the original ~11 kHz Nyquist limit. For
voice adaptation, I wanted the source recording to contain the full frequency range available to the
codec.

The second decision was more important: **split by book, not by individual clip**.

A book roughly corresponds to a recording session: the same microphone, room, and vocal register.
Randomly distributing clips from one book across train and evaluation sets would leak those acoustic
characteristics into evaluation.

Splitting by book gives a harder generalization test. As an additional check, **75% of word bigrams
in the evaluation set never occur in training**.

The nominal **27.3-hour** corpus became **5.3 usable hours / 2,915 clips** after restricting samples
to 5–15 seconds and filtering for clipping and noise:

<figure class="diagram">
<svg viewBox="0 0 700 92" role="img" aria-labelledby="split-t">
  <title id="split-t">From nominal corpus size to the usable train and eval split</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">what survived filtering</text>
    <g font-size="11" text-anchor="middle">
      <rect x="0" y="24" width="150" height="46" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="75" y="45" font-size="15" font-weight="700" fill="var(--d-muted)">27.3 h</text>
      <text x="75" y="60" font-size="9" fill="var(--d-muted)">nominal corpus</text>

      <rect x="196" y="24" width="150" height="46" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="271" y="45" font-size="15" font-weight="700">5.3 h</text>
      <text x="271" y="60" font-size="9" fill="var(--d-muted)">2,915 usable clips</text>

      <rect x="416" y="24" width="132" height="46" rx="8" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/>
      <text x="482" y="45" font-size="15" font-weight="700">2,425</text>
      <text x="482" y="60" font-size="9" fill="var(--d-muted)">train</text>

      <rect x="568" y="24" width="132" height="46" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="634" y="45" font-size="15" font-weight="700">490</text>
      <text x="634" y="60" font-size="9" fill="var(--d-muted)">eval</text>
    </g>
    <g stroke="var(--d-line)">
      <line x1="150" y1="47" x2="178" y2="47"/><polygon points="196,47 178,41 178,53" fill="var(--d-line)"/>
      <line x1="346" y1="47" x2="398" y2="47"/><polygon points="416,47 398,41 398,53" fill="var(--d-line)"/>
    </g>
    <text x="372" y="86" font-size="9" fill="var(--d-muted)">split by book, not by clip</text>
  </g>
</svg>
</figure>

## Two bugs that produced perfectly plausible speech

The most expensive problems were not training problems, they were interface problems. A codec-LM
expects both its audio tokens and text prompt in a specific format, and getting either wrong does
not necessarily produce an exception or obvious noise — the model keeps generating fluent speech,
just the wrong speech.

### 1. SNAC token ordering

SNAC represents each audio frame hierarchically: a coarse layer captures broad acoustic structure
while finer layers add detail. Those layers produce seven tokens per frame, which have to be
flattened into the exact ordering Orpheus expects. 

<figure class="diagram">
<svg viewBox="0 0 700 150" role="img" aria-labelledby="tok-t">
  <title id="tok-t">Assumed versus actual SNAC token ordering within one frame</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">seven tokens per frame, flattened for the language model</text>

    <text x="0" y="46" font-size="10.5" font-weight="600" fill="var(--d-warn)">assumed</text>
    <text x="0" y="59" font-size="8.5" fill="var(--d-muted)">grouped by layer</text>
    <g font-family="JetBrains Mono, monospace" font-size="12" text-anchor="middle">
      <rect x="112" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="147" y="52">L1</text>
      <rect x="190" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="225" y="52">L2</text>
      <rect x="268" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-warn)"/><text x="303" y="52" fill="var(--d-warn)">L2</text>
      <rect x="346" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-warn)"/><text x="381" y="52" fill="var(--d-warn)">L3</text>
      <rect x="424" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-warn)"/><text x="459" y="52" fill="var(--d-warn)">L3</text>
      <rect x="502" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="537" y="52">L3</text>
      <rect x="580" y="30" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="615" y="52">L3</text>
    </g>

    <text x="0" y="112" font-size="10.5" font-weight="600" fill="var(--d-good)">actual</text>
    <text x="0" y="125" font-size="8.5" fill="var(--d-muted)">interleaved</text>
    <g font-family="JetBrains Mono, monospace" font-size="12" text-anchor="middle">
      <rect x="112" y="96" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="147" y="118">L1</text>
      <rect x="190" y="96" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="225" y="118">L2</text>
      <rect x="268" y="96" width="70" height="34" rx="6" fill="var(--d-good-soft)" stroke="var(--d-good)"/><text x="303" y="118" fill="var(--d-good)">L3</text>
      <rect x="346" y="96" width="70" height="34" rx="6" fill="var(--d-good-soft)" stroke="var(--d-good)"/><text x="381" y="118" fill="var(--d-good)">L3</text>
      <rect x="424" y="96" width="70" height="34" rx="6" fill="var(--d-good-soft)" stroke="var(--d-good)"/><text x="459" y="118" fill="var(--d-good)">L2</text>
      <rect x="502" y="96" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="537" y="118">L3</text>
      <rect x="580" y="96" width="70" height="34" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="615" y="118">L3</text>
    </g>

    <text x="112" y="146" font-size="9" fill="var(--d-muted)">three positions differ — each coarse token is followed by the finer tokens belonging to it</text>
  </g>
</svg>
</figure>

The difference looks small. In audio space, it is not.

<figure class="diagram">
<svg viewBox="0 0 700 196" role="img" aria-labelledby="corr-t">
  <title id="corr-t">Decoding one clip under four token orderings and correlating each against the original</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="12" font-size="10.5" font-weight="600">The test: encode a real clip, decode it back under each ordering, compare to the original</text>
    <text x="0" y="27" font-size="9.5" fill="var(--d-muted)">waveform correlation — 1.0 is identical to the original recording, 0 is unrelated</text>

    <line x1="216" y1="40" x2="216" y2="164" stroke="var(--d-line)"/>
    <text x="216" y="178" font-size="8.5" text-anchor="middle" fill="var(--d-muted)">0</text>
    <text x="656" y="178" font-size="8.5" text-anchor="middle" fill="var(--d-muted)">1.0</text>

    <text x="206" y="58" font-size="10.5" text-anchor="end" font-weight="600">interleaved</text>
    <text x="206" y="70" font-size="8.5" text-anchor="end" fill="var(--d-muted)">what Orpheus expects</text>
    <rect x="216" y="46" width="369" height="22" rx="3" fill="var(--d-good)"/>
    <text x="595" y="62" font-size="11.5" font-weight="700" fill="var(--d-good)">+0.838</text>

    <text x="206" y="96" font-size="10.5" text-anchor="end" font-weight="600">grouped by layer</text>
    <text x="206" y="108" font-size="8.5" text-anchor="end" fill="var(--d-warn)">what I assumed</text>
    <rect x="216" y="84" width="61" height="22" rx="3" fill="var(--d-warn)"/>
    <text x="287" y="100" font-size="11.5" font-weight="700" fill="var(--d-warn)">+0.139</text>
    <text x="350" y="100" font-size="9.5" fill="var(--d-muted)">still audibly speech — which is why it survived review</text>

    <text x="206" y="134" font-size="10.5" text-anchor="end" fill="var(--d-muted)">shifted one frame</text>
    <text x="206" y="146" font-size="8.5" text-anchor="end" fill="var(--d-muted)">control</text>
    <rect x="210" y="122" width="6" height="22" rx="2" fill="var(--d-muted)"/>
    <text x="226" y="138" font-size="10.5" fill="var(--d-muted)">−0.012</text>

    <text x="206" y="164" font-size="10.5" text-anchor="end" fill="var(--d-muted)">random codes</text>
    <rect x="211" y="152" width="5" height="18" rx="2" fill="var(--d-muted)"/>
    <text x="226" y="166" font-size="10.5" fill="var(--d-muted)">−0.005</text>
    <text x="300" y="152" font-size="9.5" fill="var(--d-muted)">the two controls show what no relationship looks like</text>
  </g>
</svg>
<figcaption>The controls sit at zero, so the grouped ordering's +0.139 is real signal rather than
noise: the coarsest layer occupies position 0 in both layouts and survives, while everything finer
is scrambled.</figcaption>
</figure>

The result to note is `+0.139`: the incorrect ordering still produces recognizable speech, because
the coarse layer remains partially intact while finer information is scrambled. Listening for "does
this sound like speech?" is not enough to validate a codec pipeline.

### 2. Two missing prompt tokens

The text side had a similar failure. 

<figure class="diagram">
<svg viewBox="0 0 700 128" role="img" aria-labelledby="prm-t">
  <title id="prm-t">The prompt sequence, with and without the two Llama 3 special tokens</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="14" font-size="10.5" font-weight="600" fill="var(--d-warn)">mine</text>
    <g font-family="JetBrains Mono, monospace" font-size="11" text-anchor="middle">
      <rect x="70" y="24" width="72" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="106" y="44">[SOH]</text>
      <rect x="150" y="24" width="86" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="193" y="44">text</text>
      <rect x="244" y="24" width="72" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="280" y="44">[EOH]</text>
      <rect x="324" y="24" width="78" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="363" y="44">[SOAI]</text>
      <rect x="410" y="24" width="72" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="446" y="44">[SOS]</text>
    </g>
    <text x="500" y="44" font-size="11" font-weight="700" fill="var(--d-warn)">116% WER</text>

    <text x="0" y="80" font-size="10.5" font-weight="600" fill="var(--d-good)">correct</text>
    <g font-family="JetBrains Mono, monospace" font-size="11" text-anchor="middle">
      <rect x="70" y="90" width="60" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="100" y="110">[SOH]</text>
      <rect x="136" y="90" width="82" height="30" rx="6" fill="var(--d-good-soft)" stroke="var(--d-good)"/><text x="177" y="110" fill="var(--d-good)">128000</text>
      <rect x="224" y="90" width="70" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="259" y="110">text</text>
      <rect x="300" y="90" width="82" height="30" rx="6" fill="var(--d-good-soft)" stroke="var(--d-good)"/><text x="341" y="110" fill="var(--d-good)">128009</text>
      <rect x="388" y="90" width="60" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="418" y="110">[EOH]</text>
      <rect x="454" y="90" width="66" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="487" y="110">[SOAI]</text>
      <rect x="526" y="90" width="60" height="30" rx="6" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="556" y="110">[SOS]</text>
    </g>
    <text x="600" y="110" font-size="11" font-weight="700" fill="var(--d-good)">0.0% WER</text>
  </g>
</svg>
<figcaption>The two highlighted IDs are Llama 3's <code>&lt;|begin_of_text|&gt;</code> and
<code>&lt;|eot_id|&gt;</code>.</figcaption>
</figure>

The missing IDs correspond to Llama 3's `<|begin_of_text|>` and `<|eot_id|>` tokens, which are easy
to miss because the reference implementation inserts them implicitly through
`add_special_tokens=True`.

Without them the model still produced fluent speech, but synthesis reached **116% WER**. Word error
rate counts substitutions, deletions and insertions against the expected transcript, so it can
exceed 100% when the model inserts enough additional incorrect words. Adding the two tokens reduced
it to **0.0%**.

## The preflight test that catches both

My original validation encoded audio and decoded it again, and it passed. The problem is that a
round trip only proves the encoder and decoder agree **with each other** — if both implement the
same wrong convention, it stays internally consistent while being wrong.

The better test is to run the unmodified base model end to end through the exact harness that will
be used for training. If the base model cannot synthesize a known sentence correctly, there is no
reason to start fine-tuning. That check is now mandatory:

<figure class="diagram">
<svg viewBox="0 0 700 78" role="img" aria-labelledby="pre-t">
  <title id="pre-t">The mandatory preflight sequence before a training run</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <g font-size="11" text-anchor="middle">
      <rect x="0" y="20" width="150" height="38" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="75" y="44">validate data</text>
      <rect x="176" y="20" width="170" height="38" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="261" y="44">verify codec + prompt</text>
      <rect x="372" y="20" width="190" height="38" rx="8" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="467" y="44">base-model synthesis</text>
      <rect x="588" y="20" width="112" height="38" rx="8" fill="var(--d-good-soft)" stroke="var(--d-good)"/>
      <text x="644" y="44" font-weight="600">train</text>
    </g>
    <g stroke="var(--d-line)">
      <line x1="150" y1="39" x2="164" y2="39"/><polygon points="176,39 164,34 164,44" fill="var(--d-line)"/>
      <line x1="346" y1="39" x2="360" y2="39"/><polygon points="372,39 360,34 360,44" fill="var(--d-line)"/>
      <line x1="562" y1="39" x2="576" y2="39"/><polygon points="588,39 576,34 576,44" fill="var(--d-line)"/>
    </g>
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">every step must pass before the GPU run starts</text>
  </g>
</svg>
</figure>

## Why training loss didn't catch the bugs

Neither interface bug looked obviously wrong in the loss curve. Two runs differing by **6.6%
corrupted training text** converged to almost identical loss and very different speech:

<figure class="diagram">
<svg viewBox="0 0 700 150" role="img" aria-labelledby="loss-t">
  <title id="loss-t">Two runs with near-identical loss and very different word error rate</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="14" font-size="10" font-weight="700" letter-spacing="0.06em" fill="var(--d-muted)">FINAL TRAINING LOSS</text>
    <text x="380" y="14" font-size="10" font-weight="700" letter-spacing="0.06em" fill="var(--d-muted)">SYNTHESIS WER</text>

    <rect x="0" y="26" width="320" height="42" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="18" y="52" font-size="11">clean text</text>
    <text x="300" y="52" font-size="13" text-anchor="end" font-family="JetBrains Mono, monospace">5.7226</text>

    <rect x="0" y="76" width="320" height="42" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="18" y="102" font-size="11">6.6% corrupted</text>
    <text x="300" y="102" font-size="13" text-anchor="end" font-family="JetBrains Mono, monospace">5.7224</text>

    <rect x="380" y="26" width="320" height="42" rx="7" fill="var(--d-good-soft)" stroke="var(--d-good)"/>
    <text x="398" y="52" font-size="11">clean text</text>
    <text x="680" y="52" font-size="13" text-anchor="end" font-weight="700" fill="var(--d-good)">24.9%</text>

    <rect x="380" y="76" width="320" height="42" rx="7" fill="var(--d-fill)" stroke="var(--d-warn)"/>
    <text x="398" y="102" font-size="11">6.6% corrupted</text>
    <text x="680" y="102" font-size="13" text-anchor="end" font-weight="700" fill="var(--d-warn)">33.6%</text>

    <text x="0" y="140" font-size="10" fill="var(--d-muted)">a 0.0002 gap in loss · an 8.7 point gap in word error rate</text>
  </g>
</svg>
</figure>

The incorrect codec layout similarly plateaued around **5.72 for eight epochs** and appeared to
converge normally; after fixing the layout, epoch 0 started at **4.16**.

The distinction matters. **Loss tells you whether the model is learning the objective you supplied,
not whether you constructed the correct objective** — which is why behavioral evaluation has to sit
alongside the training curve for generative audio systems.

## Training

The final configuration was:

| Setting | Value |
|---|---|
| LoRA rank / alpha | `r=16`, `α=32` |
| Target modules | attention + MLP |
| Precision | bf16 |
| Batch | 4, with gradient accumulation 4 |
| Learning rate | 1e-4, cosine schedule |
| Gradient checkpointing | on |

LoRA keeps the base model frozen and trains small low-rank matrices inside selected layers. Here
that reduced the trainable portion of Orpheus 3B to **0.73%**.

Rank 16 was chosen as a starting point: r=8 is often sufficient for adapting to a single speaker,
while r=32 mostly buys capacity to overfit at this data size — 5.3 hours is small enough that extra
rank is more likely to memorise recordings than to generalise.

Validation loss selected the checkpoint:

<figure class="diagram">
<svg viewBox="0 0 700 160" role="img" aria-labelledby="ckpt-t">
  <title id="ckpt-t">Training and validation loss across epochs, with the selected checkpoint</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">loss by epoch — the checkpoint is chosen on validation, not training</text>

    <line x1="60" y1="130" x2="660" y2="130" stroke="var(--d-line)"/>
    <g font-size="9" fill="var(--d-muted)" text-anchor="middle">
      <text x="120" y="146">epoch 0</text><text x="380" y="146">epoch 4</text><text x="600" y="146">epoch 7</text>
    </g>

    <polyline points="120,31 380,106 600,117" fill="none" stroke="var(--d-muted)" stroke-width="1.6"/>
    <polyline points="120,68 380,80 600,77" fill="none" stroke="var(--d-accent)" stroke-width="1.8"/>

    <circle cx="120" cy="31" r="4" fill="var(--d-muted)"/><circle cx="380" cy="106" r="4" fill="var(--d-muted)"/><circle cx="600" cy="117" r="4" fill="var(--d-muted)"/>
    <circle cx="120" cy="68" r="4" fill="var(--d-accent)"/><circle cx="600" cy="77" r="4" fill="var(--d-accent)"/>
    <circle cx="380" cy="80" r="6.5" fill="var(--d-good)"/>

    <text x="120" y="24" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">4.4418</text>
    <text x="600" y="130" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">3.7713</text>
    <text x="640" y="70" font-size="9.5" fill="var(--d-accent)">4.0848</text>
    <text x="380" y="70" font-size="9.5" text-anchor="middle" font-weight="700" fill="var(--d-good)">4.0610</text>
    <text x="380" y="98" font-size="9" text-anchor="middle" fill="var(--d-good)">selected</text>

    <text x="60" y="112" font-size="9.5" fill="var(--d-muted)">training loss</text>
    <text x="60" y="58" font-size="9.5" fill="var(--d-accent)">validation loss</text>
  </g>
</svg>
<figcaption>Training loss keeps falling after epoch 4 while validation turns upward — the lowest
training loss is not the best checkpoint.</figcaption>
</figure>

Training loss continued improving after epoch 4 while validation performance degraded. Most useful
adaptation happened early, so future sweeps can likely stop around four epochs.

## Codec-LM memory is dominated by the vocabulary

I initially estimated VRAM primarily from transformer attention activations. But Orpheus has a
vocabulary of **156,940 tokens**, because the output space contains codec tokens as well as text,
and that produces a very large logits tensor during training. At batch 8 and sequence length 1,286:

<figure class="diagram">
<svg viewBox="0 0 700 118" role="img" aria-labelledby="vram-t">
  <title id="vram-t">The logits tensor dominates codec-LM memory</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">one logits tensor, at batch 8 and sequence length 1,286</text>
    <g font-size="12" text-anchor="middle" font-family="JetBrains Mono, monospace">
      <rect x="0" y="26" width="104" height="40" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="52" y="45">8</text>
      <text x="52" y="60" font-size="8" font-family="Inter, sans-serif" fill="var(--d-muted)">batch</text>
      <text x="118" y="50" font-size="13">×</text>
      <rect x="134" y="26" width="104" height="40" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="186" y="45">1,286</text>
      <text x="186" y="60" font-size="8" font-family="Inter, sans-serif" fill="var(--d-muted)">sequence</text>
      <text x="252" y="50" font-size="13">×</text>
      <rect x="268" y="26" width="126" height="40" rx="7" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/><text x="331" y="45">156,940</text>
      <text x="331" y="60" font-size="8" font-family="Inter, sans-serif" fill="var(--d-muted)">vocabulary</text>
      <text x="408" y="50" font-size="13">×</text>
      <rect x="424" y="26" width="104" height="40" rx="7" fill="var(--d-fill)" stroke="var(--d-line)"/><text x="476" y="45">4 B</text>
      <text x="476" y="60" font-size="8" font-family="Inter, sans-serif" fill="var(--d-muted)">fp32</text>
      <text x="546" y="50" font-size="13">=</text>
      <rect x="566" y="26" width="134" height="40" rx="7" fill="var(--d-good-soft)" stroke="var(--d-good)"/>
      <text x="633" y="51" font-size="16" font-weight="700" fill="var(--d-good)">6.5 GB</text>
    </g>
    <text x="0" y="92" font-size="10">batch 8, no checkpointing</text><text x="300" y="92" font-size="10" fill="var(--d-warn)">OOM on a 44 GiB GPU</text>
    <text x="0" y="110" font-size="10">batch 4, checkpointing on</text><text x="300" y="110" font-size="10" fill="var(--d-good)">17.6 GiB</text>
  </g>
</svg>
</figure>

In practice:

- batch 8 without checkpointing OOMed on a **44 GiB GPU**;
- batch 4 with checkpointing used **17.6 GiB**.

For codec-LMs, a useful first-order memory estimate is therefore **`batch × sequence length ×
vocabulary size`** before optimizing attention memory.

## Evaluating the clone

The experiment was designed to separate two questions:

1. **Does it sound like the same speaker?**
2. **Does it speak like that speaker?**

I measured four dimensions:

| Measurement | What it tells us |
|---|---|
| **ECAPA / WavLM similarity** | Is the speaker identity preserved? |
| **WER** | Is the generated speech intelligible and correct? |
| **Prosody distance** | Does pitch, timing, and pause behavior match? |
| **Blind A/B** | Do human judgments agree with the metrics? |

The controls were the real speaker, zero-shot cloning using the same speaker reference, and the
unmodified Orpheus base model.

## Speaker similarity had already saturated

On a frozen 48-utterance evaluation set:

| System | ECAPA similarity |
|---|---:|
| LoRA | **0.808** |
| Real speaker | 0.794 |
| Zero-shot | 0.790 |
| Real-vs-real | 0.784 |
| Base model | 0.166 |

ECAPA maps speech into an embedding representing speaker identity, and cosine similarity measures
how close two embeddings are.

**Real-vs-real** is the control that makes the rest of the table readable: two genuine recordings of
the same speaker, from different books, scored against each other. It comes out at 0.784 rather than
1.0, because the same person recorded on different days — different microphone position, energy,
room — does not produce identical embeddings. That number is the practical ceiling of the metric on
this speaker.

The interesting result is that **LoRA, zero-shot and real speech all occupy approximately the same
range**, at or above that ceiling.

LoRA scoring above real-vs-real does not mean the generated voice is "more real". Real recordings
naturally vary between sessions, while a model converges toward a more standardized version of the
speaker. WavLM showed the same saturation:

| System | WavLM similarity |
|---|---:|
| real | 0.975 |
| LoRA | 0.981 |
| zero-shot | 0.984 |

Once synthetic speech reaches the real-vs-real range, small differences in speaker similarity stop
being useful evidence. The base model at **0.166** is the one number that still carries information:
without adaptation or a speaker reference, the voice is clearly a different person.

## Prosody is where LoRA helped

Prosody separated the systems much more clearly.

| Feature | Real | LoRA | Zero-shot |
|---|---:|---:|---:|
| Mean pitch | 224.5 Hz | **223.4 Hz** | 234.7 Hz |
| Pitch variation | 52.2 Hz | **49.4 Hz** | 66.0 Hz |
| Mean pause | 0.18 s | 0.26 s | 0.06 s |
| Speech rate | 4.35 syll/s | 3.87 | **4.19** |

The adapter tracks the speaker's mean pitch within **1.1 Hz**, compared with 10.2 Hz for zero-shot,
and it better reproduces pitch variation and pause behavior. Zero-shot performs better on speech
rate, so adaptation is not uniformly better. Across the combined prosody metric:

<figure class="diagram">
<svg viewBox="0 0 700 116" role="img" aria-labelledby="pros-t">
  <title id="pros-t">Combined prosodic distance from the real speaker</title>
  <g font-family="Inter, system-ui, sans-serif" fill="currentColor">
    <text x="0" y="12" font-size="10" fill="var(--d-muted)">combined prosodic distance from the real speaker — lower is closer</text>

    <text x="100" y="46" font-size="11.5" text-anchor="end">zero-shot</text>
    <rect x="112" y="32" width="470" height="20" rx="4" fill="var(--d-warn)" opacity="0.75"/>
    <text x="594" y="47" font-size="12" font-weight="700" fill="var(--d-warn)">0.187</text>

    <text x="100" y="82" font-size="11.5" text-anchor="end" font-weight="600">LoRA</text>
    <rect x="112" y="68" width="259" height="20" rx="4" fill="var(--d-good)"/>
    <text x="383" y="83" font-size="12" font-weight="700" fill="var(--d-good)">0.103</text>

    <line x1="371" y1="30" x2="371" y2="96" stroke="var(--d-muted)" stroke-dasharray="3 3"/>
    <text x="380" y="108" font-size="10" fill="var(--d-muted)">45% closer to the target speaker</text>
  </g>
</svg>
</figure>

That is a **45% reduction in prosodic distance**, and it is the main result of the experiment.
Zero-shot cloning had already captured enough timbre to saturate speaker-identity metrics;
fine-tuning primarily improved **how the speaker delivers speech**.

A small blind A/B test produced 4 correct identifications out of 12 (`p = 0.388`). With one listener
and 12 trials, that is too small to support a strong perceptual claim, so I treat it as directional
evidence only.

## Takeaways

The main result was not that LoRA made the voice sound dramatically more like the target speaker —
zero-shot cloning had already largely solved that. Instead, training **0.73% of Orpheus 3B** reduced
prosodic distance by **45%**, bringing pitch and pause behavior substantially closer to the target
speaker.

The engineering lessons were equally useful:

- Calibrate similarity metrics against real-vs-real performance before optimizing them.
- Split speech data by recording session rather than random clips.
- Validate the codec and prompt contract with the unmodified model before training.
- Do not assume a healthy loss curve means a healthy generation pipeline.
- For codec-LMs, include vocabulary-sized logits when estimating VRAM.
- Move every cheap failure into preflight so expensive experiments test the model rather than the
  harness.

Zero-shot cloning captured **who the speaker sounds like**. Fine-tuning helped capture **how she
speaks**.
