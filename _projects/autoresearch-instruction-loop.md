---
title: "Self-Improving Loop for Cross-App Agentic Instructions"
excerpt: "Built and benchmarked. Our baseline scored 33 out of 100 on cross-app phone tasks. A loop that reads its own failures, rewrites the agentic instructions it puts in front of each task, and keeps the rewrite only if a judge scores it higher — with a knowledge graph underneath that holds what a per-app store cannot: the failure that lives between two apps. 25% to 45.2% across eight iterations."
collection: projects
order: 0
status: Published
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">100 cross-app tasks</span>
  <span class="pill">17 apps</span>
  <span class="pill">LLM-as-judge · knowledge graph</span>
</p>

Cross-app phone tasks — read a price in one app and text it from another, change a named alarm
and confirm it by message, compare listings across two shopping apps — fail at the *handoff*
far more than inside either leg. The baseline on a 100-task suite, graded from the screen
rather than from the agent's own report, was **33 out of 100**.

Every instruction set we had was scoped to a single app, and a single-app set has nothing to say
about carrying state across a switch. This is a loop that writes those instructions itself, over
a store built to hold the part that spans two apps.

## What was built

- **A knowledge graph keyed on the app pair.** `edge_key(app_a, app_b)` canonicalises an
  unordered pair into `edges/clock__text-messages.json`. **Node facts** are true of one app
  whatever you are running; **edge facts** belong to a handoff and to neither app alone — which
  is exactly what a flat, app-keyed store has nowhere to put. 14 app nodes, 17 recorded
  handoffs, 50 node facts. The lineage is Graph RAG: entity-level retrieval misses the
  relationships, and the relationships are where the failures live.
- **A champion-gated optimisation loop.** Propose a rewrite of one class's instructions from
  graph evidence, dispatch, judge every episode frame-by-frame against per-task sub-goals, and
  promote only if it beat the reigning champion *on that class's own tasks*. Without the gate each
  round proposes from its parent, and prompt text drifts downhill.
- **A screenshot-grounded judge** with a failure taxonomy, plus an independent manual
  cross-check by a second model deliberately never wired into the loop, so the trend is not
  graded by the same family writing the instructions.
- **A holdout set** dispatched each round and excluded from the gate, as the only read on
  overfitting the visible tasks.

## Results

| | value |
|---|---|
| Baseline, no instructions | 33.0% |
| Flat knowledge-base loop | 25.0% |
| Champion-gated graph loop, iteration 1 → 8 | 25.0% → **45.2%** |
| Net gain | **+20.2 points** |
| Largest remaining fixable failure | instruction-following, 57% |

The curve is not monotonic — iterations four and five flatten then fall to 33% before
recovering. The gate is per class against that class's own floor, so a class can be reverted
while the aggregate still dips. A curve that only ever rose would mean the floor was set where
nothing could fail it.

## What it cost to learn

Four bugs, none of them in the loop design, each in a component working exactly as specified on
a path where the specification was wrong: a length cap that truncated proposals and
deleted the send-and-confirm steps that were the fix; an evidence budget that gave graph facts
84% of the proposer's page and real failures 5%; fact eviction ranked by confirmation count,
which promotes the behaviours that never break; and a pause-handler regex that matched a bare
dollar sign, terminated tasks whose objective was to read a price, and wrote the result to the
graph as an instruction problem.

None of them crashed. All produced plausible output. The only symptom was a score that would
not move — which is also what a genuinely hard problem looks like.

And the caveat that gates all of it: `insufficient_visual_evidence` appears in **83%** of judged
loop failures. Until trace capture improves, every number here is measured through a lens that
frequently cannot see whether the task was done.

[Read the full writeup →](/blog/autoresearch-instruction-loop/)
