---
title: 'Pretraining Data Pipelines: Dedup, Filtering, and the Bugs That Silently Delete Your Corpus'
date: 2026-07-24
permalink: /blog/pretraining-data-pipelines/
tags:
  - data-engineering
  - llm-training
  - distributed-systems
excerpt: "A FineWeb slice through MinHash/LSH near-dup detection, Gopher-style quality filters, and perplexity scoring — with per-stage lineage manifests recording what got removed and why. Which turned out to be the whole point."
published: false
---

<p class="pill-row">
  <span class="pill pill--built">Built</span>
  <span class="pill">Ray Data · MinHash/LSH</span>
  <span class="pill">FineWeb sample-10BT</span>
  <span class="pill">tiktoken · fastText · GPT-2 PPL</span>
</p>

Data-pipeline bugs don't crash. They delete 63% of your corpus and hand you a smaller file that
looks completely normal.

That is the difficulty of pretraining data engineering. A training bug gives you a
stack trace or a diverging loss curve. A filtering bug gives you a plausible-looking dataset,
and you find out three weeks and several thousand GPU-hours later — if you find out at all.

So the design decision that mattered most in this pipeline wasn't which filters to use. It was
making **every stage record what it removed and why** into a versioned lineage manifest.

<ul class="keyresults">
  <li><span class="keyresults__value">100%</span><span class="keyresults__label">of documents silently rejected by a language-code mismatch</span></li>
  <li><span class="keyresults__value">400</span><span class="keyresults__label">documents in the reference run — a smoke test, not a scale run</span></li>
  <li><span class="keyresults__value">252 vs 1</span><span class="keyresults__label">heuristic-filter removals across two runs with the same config hash, unexplained</span></li>
  <li><span class="keyresults__value">2.4×</span><span class="keyresults__label">token reduction, filtered vs unfiltered — no training result yet</span></li>
</ul>

## Why near-duplicate detection, not exact-match

Hashing raw documents catches byte-identical pages. It misses the dominant case on the web:
two pages sharing >90% of their content but differing in a timestamp, an ad slot, or a
templated sidebar. Those hash completely differently despite being near-identical, and they'll
happily teach your model the same boilerplate a thousand times.

**MinHash** approximates Jaccard similarity over shingle sets cheaply — instead of comparing
documents directly, you compare short signatures. **LSH banding** then turns "find all pairs
above a similarity threshold" from an O(n²) comparison into a hash-bucket lookup. Documents
that collide in any band become candidates; everything else is never compared at all.

Configuration used here: 128 permutations, 5-token shingles, Jaccard threshold 0.8, 32 bands.

There's an honest architectural split worth naming. MinHash signature computation — the
expensive, per-document part — is distributed via Ray Data. LSH candidate generation and
connected-components clustering run on the driver, because they need global knowledge of every
bucket. That's the same split production near-dup pipelines use. It doesn't scale to
arbitrarily many nodes, but at hundreds of thousands of documents, driver memory at
`num_perm × 8` bytes/doc isn't the bottleneck.

## The pipeline

<figure class="diagram">
<svg viewBox="0 0 720 300" role="img" aria-labelledby="funnel-title">
  <title id="funnel-title">Document funnel through each pipeline stage</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Document funnel — 400-doc reference run</text>

    <!-- rows: width proportional to docs -->
    <!-- 400 -> 420px -->
    <g>
      <rect x="150" y="30" width="420" height="26" rx="4" fill="var(--d-accent)" opacity="0.7"/>
      <text x="140" y="48" font-size="10.5" text-anchor="end" font-weight="600">00 ingest</text>
      <text x="580" y="48" font-size="10.5">400</text>
      <text x="640" y="48" font-size="9.5" fill="var(--d-muted)">FineWeb</text>
    </g>
    <g>
      <rect x="150" y="66" width="420" height="26" rx="4" fill="var(--d-accent)" opacity="0.7"/>
      <text x="140" y="84" font-size="10.5" text-anchor="end" font-weight="600">01 MinHash/LSH</text>
      <text x="580" y="84" font-size="10.5">400</text>
      <text x="640" y="84" font-size="9.5" fill="var(--d-muted)">0 near-dups</text>
    </g>
    <g>
      <rect x="150" y="102" width="155" height="26" rx="4" fill="var(--d-accent)" opacity="0.7"/>
      <rect x="305" y="102" width="265" height="26" rx="4" fill="var(--d-warn)" opacity="0.28"/>
      <text x="140" y="120" font-size="10.5" text-anchor="end" font-weight="600">02a heuristics</text>
      <text x="580" y="120" font-size="10.5">148</text>
      <text x="640" y="120" font-size="9.5" fill="var(--d-warn)">−252</text>
    </g>
    <g>
      <rect x="150" y="138" width="155" height="26" rx="4" fill="var(--d-accent)" opacity="0.7"/>
      <text x="140" y="156" font-size="10.5" text-anchor="end" font-weight="600">02b language ID</text>
      <text x="580" y="156" font-size="10.5">148</text>
      <text x="640" y="156" font-size="9.5" fill="var(--d-muted)">0 removed</text>
    </g>
    <g>
      <rect x="150" y="174" width="140" height="26" rx="4" fill="var(--d-accent)" opacity="0.7"/>
      <rect x="290" y="174" width="15" height="26" rx="2" fill="var(--d-warn)" opacity="0.28"/>
      <text x="140" y="192" font-size="10.5" text-anchor="end" font-weight="600">02c perplexity</text>
      <text x="580" y="192" font-size="10.5">133</text>
      <text x="640" y="192" font-size="9.5" fill="var(--d-warn)">−15</text>
    </g>

    <line x1="150" y1="216" x2="700" y2="216" stroke="var(--d-line)"/>

    <g>
      <rect x="150" y="232" width="173" height="24" rx="4" fill="var(--d-good)" opacity="0.6"/>
      <text x="140" y="248" font-size="10.5" text-anchor="end" font-weight="600">03 packed (filtered)</text>
      <text x="333" y="248" font-size="10">109,146 tokens · 212 seqs</text>
    </g>
    <g>
      <rect x="150" y="264" width="420" height="24" rx="4" fill="var(--d-muted)" opacity="0.35"/>
      <text x="140" y="280" font-size="10.5" text-anchor="end" font-weight="600">03 packed (baseline)</text>
      <text x="580" y="280" font-size="10">265,403 tokens · 517 seqs</text>
    </g>
  </g>
</svg>
<figcaption>Every bar here is a recorded manifest entry, not a reconstruction. The
removal reason for each of those 252 documents is in the manifest too.</figcaption>
</figure>

| Stage | What it does |
|---|---|
| 0. Ingest | Streams a FineWeb slice (no full-corpus download), writes sharded JSONL |
| 1. Dedup | Distributed MinHash signatures via Ray Data + LSH near-dup clustering |
| 2a. Heuristic filter | Gopher/FineWeb document-statistics rules — length, symbol ratio, bullet-line fraction, stopword presence |
| 2b. Language ID | fastText, keeps high-confidence English |
| 2c. Perplexity filter | GPT-2 perplexity scoring on GPU, drops the highest-perplexity tail |
| 3. Tokenize + pack | tiktoken BPE, packed into fixed-length `block_size+1` sequences |
| — | `lineage.py` — every stage above logs docs in/out and removal reasons to a versioned manifest |

The filters run cheapest-first on purpose: heuristics are pure string statistics, language ID
is a small fastText model, and the perplexity filter needs a GPU forward pass per document. By
the time GPT-2 scoring runs, the corpus has already shrunk from 400 documents to 148 — which
is the difference between a filter you can afford and one you can't.

## The lineage manifest

Every run writes `manifests/run-<timestamp>-<config_hash>.json` containing the exact
`PipelineConfig` (hashed, so two runs with the same hash provably used identical thresholds),
the git commit, and a per-stage record of docs in/out and *why* documents were removed:

```json
{
  "name": "02a_heuristic_quality_filter",
  "docs_in": 400,
  "docs_out": 148,
  "docs_removed": 252,
  "removed_fraction": 0.63,
  "reasons": { "low_alpha_fraction": 252, "too_few_words": 2 }
}
```

```json
{
  "name": "02c_perplexity_filter",
  "docs_in": 148, "docs_out": 133,
  "reasons": { "perplexity_above_threshold": 15 },
  "extra": { "ppl_threshold": 66.799, "model": "gpt2" }
}
```

That `low_alpha_fraction: 252` line is why the manifest exists. Without the
manifest, stage 02a is a black box that turned 400 documents into 148. With it, you know
exactly which rule fired and can go argue about whether the threshold is right — which is a
tractable conversation. "Where did my corpus go" is not.

## Four bugs, three of which were silent

Smoke-testing at 60 and 400 documents before any scale run caught four real defects. Three of
them produced no error at all.

**1. A language code that rejected everything.** The filter was written against the classic
`lid.176` fastText model, whose labels are two-letter codes — `"en"`. The model actually loaded
was Meta's newer `fasttext-language-identification`, which uses FLORES-200 codes —
`"eng_Latn"`. Every document was compared against a label that could never match, so the stage
**silently rejected 100% of documents**. No exception. Just an empty dataset, flowing
downstream.

```python
# facebook/fasttext-language-identification uses FLORES-200 codes
# (e.g. "eng_Latn"), not the classic lid.176 "en" — mismatching this
# silently rejects every document, so it's worth calling out explicitly.
lang_keep: str = "eng_Latn"
```

**2. Ray Data's `write_json` appends rather than replaces.** Re-running any stage — exactly
what the "smoke test small, then scale up" workflow does constantly — left the previous run's
part-files in the output directory. Downstream stages then read a silent mixture of current and
stale documents. Fixed by having each stage clear its `out_dir` before writing. This one is
nasty precisely because it makes results *slightly* wrong in a way that changes between runs.

**3. The perplexity filter crashed on empty input.** `np.quantile` on a zero-length array —
which is what it received once bug #1 emptied the corpus upstream. This is the one bug that
*did* announce itself, and only because a different silent bug fed it.

**4. `.view()` on a non-contiguous tensor** in the tiny-GPT loss, from column-sliced numpy
batches. Runtime error; fixed with `.reshape()`.

**The loud bug was downstream of the silent one.** If the
perplexity filter had handled empty input gracefully, the language-ID bug would have propagated
all the way to a training run on an empty corpus.

## A discrepancy the manifest surfaced, still unexplained

Two runs recorded the **same config hash** (`4ff043d439fc`) and produced very different
heuristic-filter results:

| Run | Stage 02a docs in → out | `low_alpha_fraction` removals |
|---|---|---|
| `run-1784568560` | 400 → 148 | 252 |
| `run-1784575489` | 400 → 397 | 1 |

Identical thresholds, 63% removal versus 0.75%. The two runs record different git commits, so
something in the filtering path changed between them — but I haven't isolated which change, and
I'm not going to guess at it in public.

What I'll claim is narrower and, I think, more useful: **this discrepancy is only visible
because both runs wrote manifests.** Without per-stage removal-reason logging, two runs of "the
same pipeline" would have quietly produced corpora differing by a factor of 2.7× in document
count, and the only symptom would have been a training run that underperformed for no apparent
reason. Tracking it down is the next task; the manifest is what turned it into a task at all.

## Filtered vs unfiltered: the harness, and what hasn't run

The pipeline also tokenizes and packs the **raw, unfiltered** ingested documents as a baseline
corpus. `train_compare/run_comparison.py` trains two identical tiny GPTs (~20–30M params) —
same seed, optimizer, LR schedule, step count, and **token budget** — one on each corpus, and
compares validation loss. Holding everything but the data fixed is what isolates the effect of
filtering.

**Being precise: that comparison hasn't been run at a scale that would mean anything, and there
is no result to report.** The reference run above produced 109,146 filtered tokens against
265,403 unfiltered — a 2.4× reduction, but from 400 documents. That's a smoke test that proves
the plumbing works end to end, not an experiment. The training loop is CPU-heavy and belongs on
a GPU; the harness is written and the corpora are packed, but the number that would justify the
whole pipeline isn't in hand yet.

I'd rather say that than show a loss curve from 212 sequences and let it imply something.

## Honest limitations

- **"Distributed" here means Ray Data's local-mode multi-core execution** by default. The code
  is unchanged if pointed at a real Ray cluster — `ray.init()` is the only thing that moves —
  but this repo doesn't provision one, and no multi-node run has happened.
- **The reference run is 400 documents.** The pipeline targets 150k and the README documents
  that path, but every number in this post comes from a smoke run. Near-dup detection finding
  zero duplicates in 400 FineWeb documents is expected and tells you nothing about how MinHash
  behaves at scale — that's the measurement I most want next.
- **The perplexity filter uses GPT-2**, not a domain-specific small LM. Good enough to catch
  garbled or non-natural-language text; not a precise quality signal on its own. That's exactly
  why it runs last, after cheaper filters have already shrunk the input.
- **The heuristic thresholds are inherited from the Gopher/FineWeb papers**, not tuned or
  ablated here. Given the 02a discrepancy above, that's the first thing I'd want to validate.
