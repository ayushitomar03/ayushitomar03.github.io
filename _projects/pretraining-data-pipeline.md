---
title: "Pretraining Data Pipeline at Scale"
excerpt: "Built. A FineWeb slice through distributed MinHash/LSH near-dup detection, Gopher-style quality filters, and GPT-2 perplexity scoring — with per-stage lineage manifests recording what got removed and why, which is how four silent bugs got caught."
collection: projects
order: 5
status: Upcoming
---

<p class="pill-row">
  <span class="pill pill--upcoming">Writeup upcoming</span>
  <span class="pill pill--built">Pipeline built</span>
  <span class="pill pill--progress">Scale run pending</span>
  <span class="pill">Ray Data · MinHash/LSH</span>
  <span class="pill">tiktoken · fastText</span>
</p>

Data-pipeline bugs don't crash — they delete part of your corpus and hand you a smaller file
that looks completely normal. So the design decision that mattered most here wasn't which
filters to use; it was making every stage record what it removed and why.

## What was built

Six stages, each writing its own lineage record:

| Stage | What it does |
|---|---|
| Ingest | Streams a FineWeb slice (no full-corpus download), writes sharded JSONL |
| Dedup | Distributed MinHash signatures via Ray Data + LSH near-dup clustering (128 perms, 5-token shingles, Jaccard 0.8, 32 bands) |
| Heuristic filter | Gopher/FineWeb document-statistics rules — length, symbol ratio, bullet-line fraction, stopword presence |
| Language ID | fastText, high-confidence English only |
| Perplexity filter | GPT-2 perplexity scoring on GPU, drops the highest-perplexity tail |
| Tokenize + pack | tiktoken BPE, packed into fixed-length sequences |

Filters run cheapest-first by design: by the time GPU perplexity scoring runs, the corpus has
already shrunk substantially — the difference between a filter you can afford and one you
can't.

Every run writes `manifests/run-<timestamp>-<config_hash>.json`: the exact config (hashed), the
git commit, and per-stage docs in/out plus removal reasons
(`{"low_alpha_fraction": 252, "too_few_words": 2}`).

## What it caught

Smoke-testing at 60 and 400 documents found four real defects, **three of them silent**:

- The language filter compared `lid.176`-style `"en"` codes against a model emitting FLORES-200
  `"eng_Latn"` — **silently rejecting 100% of documents**, with no exception.
- Ray Data's `write_json` appends part-files rather than replacing the directory, so re-running
  any stage silently mixed stale documents from prior runs into downstream stages.
- The perplexity filter crashed on empty input — the one loud bug, and only because a silent bug
  upstream fed it an empty corpus.

The manifests also surfaced a discrepancy I haven't resolved: two runs with the same config hash
removed 252 vs 1 documents at the heuristic stage. Without per-stage logging that would have
been an invisible 2.7× difference in corpus size.

[Read the full writeup →](/blog/pretraining-data-pipelines/)

## Not done yet

The filtered-vs-unfiltered training comparison harness is written and both corpora are packed,
but **it hasn't been run at a scale that would mean anything, and there is no result to
report.** The reference run is 400 documents — a smoke test proving the plumbing works, not an
experiment. "Distributed" currently means Ray Data local-mode multi-core; no multi-node run has
happened.
