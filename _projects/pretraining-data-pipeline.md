---
title: "Pretraining Data Pipeline at Scale"
excerpt: "**Status: Planned.** A distributed pipeline for deduplication, quality filtering, and tokenization of a multi-billion-token web corpus. <br/><img src='/images/500x300.png'>"
collection: projects
---

**Status:** 🔜 Planned

## Why this matters
Data engineering is the unglamorous half of frontier model training, and it's chronically understaffed relative to how much it matters — garbage in, garbage model out. Showing you can build a pipeline that operates at real scale (not a toy 10MB dataset) is exactly the signal data/ML infra teams at frontier labs screen for.

## What to build
- Ingest a public web-scale corpus (e.g. a slice of Common Crawl / FineWeb / C4)
- Implement, at minimum:
  - MinHash/LSH-based near-duplicate detection at scale (not just exact-match dedup)
  - Quality filtering (language ID, perplexity filtering, heuristic rules à la Gopher/FineWeb)
  - Tokenization with a production tokenizer (tiktoken/ SentencePiece) and packing into fixed-length sequences
- Make it distributed — Spark, Ray Data, or Dask — not a single-process script
- Track dataset lineage/versioning (what filters removed what fraction, reproducible manifests)

## Key concepts to demonstrate
- Why exact-match dedup misses most real duplicates (near-duplicate boilerplate, templated pages)
- The measurable effect of data quality filtering on downstream loss (even at small scale, e.g. training a tiny model on filtered vs. unfiltered data)
- How to make a data pipeline reproducible and auditable, not just "a script that ran once"

## Suggested stack
Ray Data or Spark for distributed processing; `datasketch` or a custom MinHash implementation for dedup; Hugging Face `datasets` for storage; DuckDB for fast local analysis of pipeline stats.

## Definition of done
A repo with the pipeline code, a report on how much data was removed at each stage and why, and ideally a small experiment showing filtered data improves downstream perplexity/loss vs. unfiltered.
