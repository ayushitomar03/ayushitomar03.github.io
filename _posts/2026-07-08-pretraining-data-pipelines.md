---
title: 'Building Pretraining Data Pipelines: Dedup, Filtering, and Quality at Trillion-Token Scale'
date: 2026-07-08
permalink: /blog/pretraining-data-pipelines/
tags:
  - data-engineering
  - llm-training
---

**Status:** 🔜 Planned

## Why write this
Data engineering for pretraining is under-discussed relative to how much it determines model quality, which makes it a good area to differentiate yourself in — especially with your data engineering background. This should be the write-up companion to your [pretraining data pipeline project](/projects/pretraining-data-pipeline/).

## Outline
1. Why data quality dominates at scale: more compute can't fully compensate for bad data (cite Chinchilla-adjacent findings on data quality vs. quantity)
2. Deduplication: exact-match vs. near-duplicate (MinHash/LSH) — how much a real web corpus actually deduplicates by each method
3. Quality filtering: language ID, heuristic rules (repetition, boilerplate detection), perplexity-based filtering with a small reference model
4. The tokenization and packing step: why sequence packing matters for training efficiency, and the subtle bugs that show up at the document-boundary edges
5. Making it reproducible: dataset manifests, versioning, and being able to answer "what exactly was this model trained on"

## Key points to nail
- Use real numbers from your own pipeline run (e.g. "X% of documents removed by exact dedup, Y% more by near-dup, Z% by quality filters")
- Explain the actual mechanics of MinHash/LSH at a level someone could reimplement from your description
- If you ran the small-model perplexity experiment (filtered vs. unfiltered data), lead with those results

## Reference reading
- FineWeb technical report (data curation methodology, ablations on filtering impact)
- Gopher paper (data quality heuristics)
- "Deduplicating Training Data Makes Language Models Better" (dedup impact on memorization and quality)
