---
title: 'A Self-Improving Loop for Cross-App Agentic Instructions'
date: 2026-09-08
permalink: /blog/autoresearch-instruction-loop/
tags:
  - agents
  - evaluation
  - knowledge-graph
excerpt: "Most of what we do on a phone spans several apps. Those tasks are where agents break, and they break at the handoff between apps rather than inside any one of them. This is a loop that writes its own agentic instructions, and a knowledge graph built to hold the thing a per-app store cannot: what goes wrong in between."
---

<p class="pill-row">
  <span class="pill pill--built">Built &amp; benchmarked</span>
  <span class="pill">Cross-app task suite</span>
  <span class="pill">LLM-as-judge</span>
  <span class="pill">Knowledge graph</span>
</p>

A friend mentions a song, so you open Spotify, find it, copy the link, switch to WhatsApp and
send it. You take a photo, open the gallery, pick the right one, open Instagram and post it with
a caption.

Neither of these lives inside a single app. You carry something in your head — a link, a photo —
across a switch, and you finish somewhere different from where you started.

For a phone agent, this is much harder than operating any one app. Three things stack up:

**The horizon is long.** A single-app task is a handful of taps. A cross-app task runs many
times longer, and every extra step is another chance to drift.

**State has to survive the switch.** The agent has to read a real value in one app, hold it
unchanged, and use it in another. Not a remembered value, not a plausible-looking one. The one
actually on screen.

**No app can tell you it worked.** Each app only knows about its own half. Nothing in Spotify
knows whether the link ever got sent, and nothing in WhatsApp knows whether it was the right
link. Success only exists across the boundary.

And that is where these tasks fail. Not on the taps. Watch enough failed runs and the pattern is
almost always the same: the agent does the first half competently, then fumbles the handoff. It
reads the price and texts a different number. It picks the photo and never posts it. It finishes
one leg and reports the whole task done.

Someone could just write those instructions by hand, and people do. You sit down with a stack of
failed runs, work out what keeps going wrong, and write the lines that address it. It works. It
also doesn't scale — every new app and every new kind of task needs somebody starting over.

But look at the shape of that work: read the failures, write a better version, measure it, keep
it if it won. That is a loop. And every step in it is something a model can do — as long as the
step that decides whether a version actually won belongs to something other than the model doing
the writing.

## The loop

<figure class="diagram">
<svg viewBox="0 0 700 292" role="img" aria-labelledby="loop-t">
  <title id="loop-t">One round of the loop, showing the knowledge graph the judge writes to and the proposer reads from</title>
  <defs>
    <marker id="lp-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--d-muted)"/>
    </marker>
    <marker id="lp-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--d-accent)"/>
    </marker>
    <marker id="lp-g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--d-good)"/>
    </marker>
  </defs>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">One round. The graph is what carries a lesson from this round into the next one.</text>

    <rect x="8" y="42" width="124" height="52" rx="5" fill="var(--d-good-soft)" stroke="var(--d-good)"/>
    <text x="70" y="64" font-size="10.5" text-anchor="middle" font-weight="600">best set so far</text>
    <text x="70" y="79" font-size="9" text-anchor="middle" fill="var(--d-muted)">instructions for this class</text>

    <rect x="148" y="42" width="124" height="52" rx="5" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="210" y="64" font-size="10.5" text-anchor="middle" font-weight="600">1 &#183; propose</text>
    <text x="210" y="79" font-size="9" text-anchor="middle" fill="var(--d-muted)">write a new version</text>

    <rect x="288" y="42" width="124" height="52" rx="5" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="350" y="64" font-size="10.5" text-anchor="middle" font-weight="600">2 &#183; run</text>
    <text x="350" y="79" font-size="9" text-anchor="middle" fill="var(--d-muted)">on the training tasks</text>

    <rect x="428" y="42" width="124" height="52" rx="5" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="490" y="64" font-size="10.5" text-anchor="middle" font-weight="600">3 &#183; judge</text>
    <text x="490" y="79" font-size="9" text-anchor="middle" fill="var(--d-muted)">grade from screenshots</text>

    <rect x="568" y="42" width="124" height="52" rx="5" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/>
    <text x="630" y="64" font-size="10.5" text-anchor="middle" font-weight="600">4 &#183; compare</text>
    <text x="630" y="79" font-size="9" text-anchor="middle" fill="var(--d-muted)">beat the best set?</text>

    <line x1="132" y1="68" x2="146" y2="68" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>
    <line x1="272" y1="68" x2="286" y2="68" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>
    <line x1="412" y1="68" x2="426" y2="68" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>
    <line x1="552" y1="68" x2="566" y2="68" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>

    <rect x="176" y="128" width="348" height="46" rx="6" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/>
    <text x="350" y="147" font-size="10.5" text-anchor="middle" font-weight="600">knowledge graph</text>
    <text x="350" y="162" font-size="9" text-anchor="middle" fill="var(--d-muted)">app nodes &#183; handoff edges between them</text>

    <line x1="490" y1="96" x2="490" y2="126" stroke="var(--d-accent)" stroke-width="1.6" marker-end="url(#lp-b)"/>
    <text x="500" y="116" font-size="9" fill="var(--d-accent)" font-weight="600">writes the failures</text>

    <line x1="210" y1="126" x2="210" y2="96" stroke="var(--d-accent)" stroke-width="1.6" marker-end="url(#lp-b)"/>
    <text x="108" y="111" font-size="9" fill="var(--d-accent)" font-weight="600">reads facts for</text>
    <text x="108" y="122" font-size="9" fill="var(--d-accent)" font-weight="600">these apps</text>

    <line x1="630" y1="96" x2="630" y2="196" stroke="var(--d-muted)"/>
    <line x1="242" y1="196" x2="630" y2="196" stroke="var(--d-muted)"/>
    <line x1="242" y1="196" x2="242" y2="206" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>
    <line x1="382" y1="196" x2="382" y2="206" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>
    <line x1="522" y1="196" x2="522" y2="206" stroke="var(--d-muted)" marker-end="url(#lp-a)"/>

    <rect x="182" y="208" width="120" height="42" rx="5" fill="var(--d-good-soft)" stroke="var(--d-good)"/>
    <text x="242" y="226" font-size="10.5" text-anchor="middle" font-weight="600" fill="var(--d-good)">keep it</text>
    <text x="242" y="240" font-size="9" text-anchor="middle" fill="var(--d-muted)">it becomes the best</text>

    <rect x="322" y="208" width="120" height="42" rx="5" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="382" y="226" font-size="10.5" text-anchor="middle" font-weight="600">throw it out</text>
    <text x="382" y="240" font-size="9" text-anchor="middle" fill="var(--d-muted)">old set stays</text>

    <rect x="462" y="208" width="120" height="42" rx="5" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="522" y="226" font-size="10.5" text-anchor="middle" font-weight="600">too close to call</text>
    <text x="522" y="240" font-size="9" text-anchor="middle" fill="var(--d-muted)">no change</text>

    <path d="M182,229 L70,229 L70,96" fill="none" stroke="var(--d-good)" stroke-width="1.6" marker-end="url(#lp-g)"/>
    <text x="14" y="190" font-size="9" fill="var(--d-good)" font-weight="600">5 &#183; promote</text>

    <text x="0" y="278" font-size="9.5" fill="var(--d-muted)">A holdout set runs every round as well, but never affects the decision &#8212; it is there to catch the loop memorising the tasks it can see.</text>
  </g>
</svg>
<figcaption>Steps 4 and 5 are what matter. Without it, each round builds on whatever the last round produced, so one unlucky round becomes the starting point for the next.</figcaption>
</figure>

One round, five steps:

1. **Propose.** A model writes a new instruction set for one class of task.
2. **Run.** Those instructions go out with the training tasks on real devices.
3. **Judge.** Another model grades every run from its screenshots, against the sub-goals the task
   was supposed to hit. Not from the agent's own report — agents are unreliable narrators of
   their own success.
4. **Compare.** Score the new instructions against the best set we have for that class, on the
   same tasks.
5. **Keep or throw away.** Higher score, it becomes the new best. Lower, we throw it out and the
   old set stays.

Steps four and five are what matter. Without them, every round builds on whatever the last round
produced, so a single bad round becomes the starting point for the next one. And rounds go bad
for boring reasons: a flaky device, a judge mistake, an edit that helps one class and hurts two.
There are far more ways to write a worse set of instructions than a better one, so a loop with no
gate drifts downward.

A holdout set runs each round but never affects the decision. It is there to catch the loop
memorising the tasks it can see.

## Why a graph and not a list

The judge produces failures. Those failures have to be stored somewhere the proposer can read
them next round. The simple version is a flat knowledge base: a list of facts, each filed under
the app it concerns.

Here is the run that broke it.

The task: change the "Morning Run" alarm to 6:30 AM, then text someone the new time. The agent
opened Clock, failed to find the alarm, switched to messages anyway, and sent *your "Morning Run"
alarm is now set for 6:30 AM.* The text delivered fine. It was false.

There are two lessons in that run, and they are different kinds of thing.

<figure class="diagram">
<svg viewBox="0 0 700 330" role="img" aria-labelledby="store-t">
  <title id="store-t">A flat app-keyed store against a graph keyed on the app pair, for one task</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">One task, two stores: where does the lesson go?</text>
    <text x="0" y="28" font-size="9.5" fill="var(--d-muted)">&#8220;Change the time of your &#8216;Morning Run&#8217; alarm in Clock to 6:30 AM and text a contact the new time.&#8221;</text>

    <rect x="0" y="44" width="336" height="176" rx="6" fill="var(--d-panel)" stroke="var(--d-line)"/>
    <text x="16" y="66" font-size="10.5" font-weight="600">FLAT KNOWLEDGE BASE</text>
    <text x="16" y="80" font-size="9.5" fill="var(--d-muted)">one key: the app</text>
    <rect x="16" y="90" width="304" height="26" rx="4" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="28" y="107" font-size="9.5" font-family="ui-monospace, SFMono-Regular, monospace">app: com.&#8230;.clockpackage</text>
    <text x="16" y="136" font-size="9.5" fill="var(--d-muted)">Fine here &#8212; this is true of Clock on its own:</text>
    <text x="16" y="154" font-size="9.5">&#8220;Alarm rows show a generic label, not the alarm</text>
    <text x="16" y="167" font-size="9.5">name. Open the row and read its name field</text>
    <text x="16" y="180" font-size="9.5">before changing it.&#8221;</text>
    <text x="16" y="202" font-size="9.5" fill="var(--d-warn)" font-weight="600">But the part that failed was the handoff.</text>

    <rect x="364" y="44" width="336" height="176" rx="6" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/>
    <text x="380" y="66" font-size="10.5" font-weight="600">KNOWLEDGE GRAPH</text>
    <text x="380" y="80" font-size="9.5" fill="var(--d-muted)">two keys: the unordered app pair</text>
    <rect x="380" y="90" width="304" height="26" rx="4" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="392" y="107" font-size="9.5" font-family="ui-monospace, SFMono-Regular, monospace">edges/clock__text-messages.json</text>
    <text x="380" y="136" font-size="9.5" fill="var(--d-muted)">Holds what belongs to neither app:</text>
    <text x="380" y="154" font-size="9.5">&#8220;Do not compose a confirmation text asserting an</text>
    <text x="380" y="167" font-size="9.5">alarm&#8217;s name and new time unless that exact named</text>
    <text x="380" y="180" font-size="9.5">alarm was actually located and edited in Clock.&#8221;</text>
    <text x="380" y="202" font-size="9.5" fill="var(--d-accent)" font-weight="600">The text can send perfectly and still be false.</text>

    <line x1="0" y1="240" x2="700" y2="240" stroke="var(--d-line)"/>

    <text x="0" y="262" font-size="10.5" font-weight="600">Two kinds of fact, two places to keep them</text>

    <rect x="0" y="272" width="336" height="46" rx="5" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="16" y="290" font-size="10" font-weight="600">NODE FACT &#8212; about one app</text>
    <text x="16" y="306" font-size="9.5" fill="var(--d-muted)">True of that app no matter what you are doing.</text>

    <rect x="364" y="272" width="336" height="46" rx="5" fill="var(--d-accent-soft)" stroke="var(--d-accent)"/>
    <text x="380" y="290" font-size="10" font-weight="600">EDGE FACT &#8212; about a handoff between two apps</text>
    <text x="380" y="306" font-size="9.5" fill="var(--d-muted)">Belongs to neither app. A one-app store cannot file it.</text>
  </g>
</svg>
<figcaption>This split is the whole design. Instructions used to cover one app at a time, so a failure that only happens <em>between</em> two apps had nowhere to go &mdash; and that is where these tasks break most often.</figcaption>
</figure>

**The first is about Clock.** Alarm rows show a generic label instead of the alarm's name, so you
have to open the row to check which one you're editing. True of Clock whatever you're doing. A
flat store files this under Clock and everything works.

**The second is the one that cost us the task.** Don't send a confirmation about something you
never verified. Now try to file that under an app.

File it under Clock, and the messaging app never sees it. File it under messaging, and it reads
as nonsense, because messaging did its job perfectly. It isn't a fact about either app. It's a
fact about moving between them — and a store with one slot per app has nowhere to put it.

So we key the store on the *pair* of apps as well as on single apps:

```python
def edge_key(app_a, app_b, action=None):
    """Canonical key for a transfer between two apps."""
```

Apps become **nodes**. Handoffs between apps become **edges**. Two app names go in, one filename
comes out — `edges/clock__text-messages.json` — and now the second lesson has somewhere to live.

### What's actually in them

A **node** holds facts about one app on its own. On the eBay node:

> "Search results can show a wide price range across listings ($139.99 to $270.74) for the same
> query. The agent must decide which listing is the relevant one rather than re-searching."

An **edge** holds facts about a handoff. On the Amazon ↔ eBay edge:

> "After reading prices from both Amazon and eBay, an agent can loop back and forth re-running the
> same searches dozens of times instead of comparing the values it already has."

Neither app owns that second one. It only exists because the task asked for both.

### How this reaches the instructions

When the loop proposes a new instruction set for a task class, it looks up the apps it touches and
pulls both kinds of fact — the nodes for those apps, and the edges between them. Those go into
the proposer's prompt as evidence, and the proposer turns them into instructions.

The alarm edge fact becomes an instruction roughly like:

> Before sending any confirmation message, go back to the app you changed and confirm the change
> is visible on screen. If you cannot see it, say so instead of confirming it.

That line is useless in Clock's own instructions and useless in messaging's. It only makes sense
as advice about the pair.

And because it's stored on the edge rather than on an app, it transfers. A lesson learned from
*share a link, then confirm it arrived* applies to *share a photo, then confirm it arrived* —
different apps, same shape of handoff. The graph knows they're the same shape, so one observed
failure covers both.

<figure class="diagram">
<svg viewBox="0 0 700 356" role="img" aria-labelledby="kg-t">
  <title id="kg-t">The knowledge graph: 14 app nodes, 17 recorded handoffs, with one task's retrieval highlighted</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">14 app nodes, 17 recorded handoffs</text>
    <text x="0" y="28" font-size="9.5" fill="var(--d-muted)">Highlighted: what one task looks up &#8212; three apps, and the handoffs between them</text>

    <g stroke="var(--d-line)" stroke-width="1">
      <line x1="120" y1="120" x2="238" y2="90"/>
      <line x1="120" y1="120" x2="150" y2="238"/>
      <line x1="330" y1="80" x2="420" y2="150"/>
      <line x1="238" y1="120" x2="300" y2="212"/>
      <line x1="90"  y1="185" x2="150" y2="238"/>
      <line x1="70"  y1="272" x2="150" y2="252"/>
      <line x1="300" y1="228" x2="404" y2="244"/>
      <line x1="245" y1="272" x2="300" y2="228"/>
      <line x1="180" y1="300" x2="150" y2="262"/>
      <line x1="352" y1="300" x2="404" y2="258"/>
      <line x1="330" y1="92"  x2="300" y2="200"/>
      <line x1="120" y1="132" x2="245" y2="266"/>
    </g>

    <g stroke="var(--d-accent)" stroke-width="2.2">
      <line x1="238" y1="76" x2="404" y2="150"/>
      <line x1="238" y1="90" x2="160" y2="232"/>
      <line x1="404" y1="164" x2="186" y2="238"/>
    </g>

    <g font-size="9.5" fill="var(--d-muted)">
      <rect x="72" y="108" width="96" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="120" y="124" text-anchor="middle">WhatsApp &#183; 6</text>
      <rect x="286" y="68" width="88" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="330" y="84" text-anchor="middle">Camera &#183; 5</text>
      <rect x="46" y="173" width="88" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="90" y="189" text-anchor="middle">Clock &#183; 3</text>
      <rect x="26" y="260" width="88" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="70" y="276" text-anchor="middle">Uber &#183; 3</text>
      <rect x="256" y="200" width="88" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="300" y="216" text-anchor="middle">Netflix &#183; 4</text>
      <rect x="200" y="260" width="90" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="245" y="276" text-anchor="middle">Instagram &#183; 5</text>
      <rect x="136" y="288" width="88" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="180" y="304" text-anchor="middle">Settings &#183; 2</text>
      <rect x="308" y="288" width="88" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="352" y="304" text-anchor="middle">Lyft &#183; 1</text>
      <rect x="348" y="232" width="112" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="404" y="248" text-anchor="middle">Google Calendar &#183; 9</text>
      <rect x="452" y="176" width="94" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="499" y="192" text-anchor="middle">Uber Eats &#183; 2</text>
      <rect x="452" y="108" width="90" height="24" rx="12" fill="var(--d-fill)" stroke="var(--d-line)"/>
      <text x="497" y="124" text-anchor="middle">YouTube &#183; 1</text>
      <line x1="452" y1="120" x2="374" y2="86" stroke="var(--d-line)" stroke-width="1"/>
      <line x1="452" y1="188" x2="440" y2="164" stroke="var(--d-line)" stroke-width="1"/>
    </g>

    <rect x="190" y="64" width="96" height="26" rx="13" fill="var(--d-accent)" stroke="var(--d-accent)"/>
    <text x="238" y="81" font-size="10" text-anchor="middle" fill="var(--d-panel)" font-weight="600">Amazon &#183; 4</text>

    <rect x="404" y="138" width="80" height="26" rx="13" fill="var(--d-accent)" stroke="var(--d-accent)"/>
    <text x="444" y="155" font-size="10" text-anchor="middle" fill="var(--d-panel)" font-weight="600">eBay &#183; 1</text>

    <rect x="108" y="232" width="122" height="26" rx="13" fill="var(--d-accent)" stroke="var(--d-accent)"/>
    <text x="169" y="249" font-size="10" text-anchor="middle" fill="var(--d-panel)" font-weight="600">Text Messages &#183; 4</text>

    <line x1="0" y1="326" x2="700" y2="326" stroke="var(--d-line)"/>
    <text x="0" y="344" font-size="9.5" fill="var(--d-muted)">A node&#8217;s number is how many facts are true of that app alone. The edges hold the rest. A task pulls both.</text>
  </g>
</svg>
<figcaption>The graph records what actually happened, not what we expected to happen. Which is why the counts are lopsided: the apps that collect the most facts are the ones with the most ways to go wrong.</figcaption>
</figure>

Some apps collect far more facts than others:

<figure class="diagram">
<svg viewBox="0 0 700 346" role="img" aria-labelledby="nodes-t">
  <title id="nodes-t">Number of learned facts held on each app node in the knowledge graph</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Facts held on each app node</text>
    <text x="0" y="28" font-size="9.5" fill="var(--d-muted)">50 node facts across 14 apps &#8212; and separately, 17 recorded handoffs carrying facts of their own</text>
    <text x="162" y="58" font-size="10" text-anchor="end">Google Calendar</text>
    <rect x="172" y="46" width="396.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="576.0" y="58" font-size="10" font-weight="600" fill="var(--d-muted)">9</text>
    <text x="162" y="79" font-size="10" text-anchor="end">WhatsApp</text>
    <rect x="172" y="67" width="264.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="444.0" y="79" font-size="10" font-weight="600" fill="var(--d-muted)">6</text>
    <text x="162" y="100" font-size="10" text-anchor="end">Camera</text>
    <rect x="172" y="88" width="220.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="400.0" y="100" font-size="10" font-weight="600" fill="var(--d-muted)">5</text>
    <text x="162" y="121" font-size="10" text-anchor="end">Instagram</text>
    <rect x="172" y="109" width="220.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="400.0" y="121" font-size="10" font-weight="600" fill="var(--d-muted)">5</text>
    <text x="162" y="142" font-size="10" text-anchor="end">Amazon</text>
    <rect x="172" y="130" width="176.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="356.0" y="142" font-size="10" font-weight="600" fill="var(--d-muted)">4</text>
    <text x="162" y="163" font-size="10" text-anchor="end">Netflix</text>
    <rect x="172" y="151" width="176.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="356.0" y="163" font-size="10" font-weight="600" fill="var(--d-muted)">4</text>
    <text x="162" y="184" font-size="10" text-anchor="end">Text Messages</text>
    <rect x="172" y="172" width="176.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="356.0" y="184" font-size="10" font-weight="600" fill="var(--d-muted)">4</text>
    <text x="162" y="205" font-size="10" text-anchor="end">Clock</text>
    <rect x="172" y="193" width="132.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="312.0" y="205" font-size="10" font-weight="600" fill="var(--d-muted)">3</text>
    <text x="162" y="226" font-size="10" text-anchor="end">Uber</text>
    <rect x="172" y="214" width="132.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="312.0" y="226" font-size="10" font-weight="600" fill="var(--d-muted)">3</text>
    <text x="162" y="247" font-size="10" text-anchor="end">Settings</text>
    <rect x="172" y="235" width="88.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="268.0" y="247" font-size="10" font-weight="600" fill="var(--d-muted)">2</text>
    <text x="162" y="268" font-size="10" text-anchor="end">Uber Eats</text>
    <rect x="172" y="256" width="88.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="268.0" y="268" font-size="10" font-weight="600" fill="var(--d-muted)">2</text>
    <text x="162" y="289" font-size="10" text-anchor="end">eBay</text>
    <rect x="172" y="277" width="44.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="224.0" y="289" font-size="10" font-weight="600" fill="var(--d-muted)">1</text>
    <text x="162" y="310" font-size="10" text-anchor="end">Lyft</text>
    <rect x="172" y="298" width="44.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="224.0" y="310" font-size="10" font-weight="600" fill="var(--d-muted)">1</text>
    <text x="162" y="331" font-size="10" text-anchor="end">YouTube</text>
    <rect x="172" y="319" width="44.0" height="15" rx="3" fill="var(--d-accent)"/>
    <text x="224.0" y="331" font-size="10" font-weight="600" fill="var(--d-muted)">1</text>
    <line x1="172" y1="40" x2="172" y2="336" stroke="var(--d-line)"/>
  </g>
</svg>
<figcaption>Google Calendar leads because vague requests land on scheduling, and there are more ways to get those wrong. eBay has one fact because it appears in one task.</figcaption>
</figure>

Google Calendar leads because vague requests land on scheduling. *"Let's plan a movie night, find
something and get it on my calendar"* names no title, no time, and no app — there are simply more
ways to get it wrong. eBay has one fact because it appears in one task.

We know the graph is doing the work, because we measured the version without it. The earlier loop
used the flat, one-app-per-fact store. It scored **25%** — eight points *worse* than running with
no instructions at all.

## Results

<figure class="diagram">
<svg viewBox="0 0 700 250" role="img" aria-labelledby="curve-t">
  <title id="curve-t">Judged pass rate across eight iterations of the champion-gated loop</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">Judged pass rate per iteration, champion-gated loop</text>
    <text x="0" y="27" font-size="9.5" fill="var(--d-muted)">the loop is the only condition that moves &#8212; the reference points are each one measurement</text>
    <line x1="74" y1="34" x2="74" y2="208" stroke="var(--d-line)"/>
    <line x1="74" y1="208" x2="656" y2="208" stroke="var(--d-line)"/>
    <line x1="74" y1="208.0" x2="656" y2="208.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="211.0" font-size="9" text-anchor="end" fill="var(--d-muted)">20%</text>
    <line x1="74" y1="180.0" x2="656" y2="180.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="183.0" font-size="9" text-anchor="end" fill="var(--d-muted)">25%</text>
    <line x1="74" y1="152.0" x2="656" y2="152.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="155.0" font-size="9" text-anchor="end" fill="var(--d-muted)">30%</text>
    <line x1="74" y1="124.0" x2="656" y2="124.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="127.0" font-size="9" text-anchor="end" fill="var(--d-muted)">35%</text>
    <line x1="74" y1="96.0" x2="656" y2="96.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="99.0" font-size="9" text-anchor="end" fill="var(--d-muted)">40%</text>
    <line x1="74" y1="68.0" x2="656" y2="68.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="71.0" font-size="9" text-anchor="end" fill="var(--d-muted)">45%</text>
    <line x1="74" y1="40.0" x2="656" y2="40.0" stroke="var(--d-line)" stroke-dasharray="2 4"/>
    <text x="66" y="43.0" font-size="9" text-anchor="end" fill="var(--d-muted)">50%</text>
    <line x1="74" y1="135.2" x2="656" y2="135.2" stroke="var(--d-muted)" stroke-width="1.5"/>
    <text x="624" y="129.2" font-size="9.5" text-anchor="end" fill="var(--d-muted)" font-weight="600">baseline, no instructions &#8212; 33.0%</text>
    <line x1="74" y1="180.0" x2="656" y2="180.0" stroke="var(--d-muted)" stroke-width="1.5"/>
    <text x="624" y="174.0" font-size="9.5" text-anchor="end" fill="var(--d-muted)" font-weight="600">flat knowledge base loop &#8212; 25.0%</text>
    <polyline points="74.0,180.0 153.7,168.8 233.4,124.0 313.1,124.0 392.9,135.2 472.6,112.8 552.3,112.8 632.0,66.9" fill="none" stroke="var(--d-accent)" stroke-width="2.4"/>
    <circle cx="74.0" cy="180.0" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="74.0" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 1</text>
    <circle cx="153.7" cy="168.8" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="153.7" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 2</text>
    <circle cx="233.4" cy="124.0" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="233.4" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 3</text>
    <circle cx="313.1" cy="124.0" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="313.1" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 4</text>
    <circle cx="392.9" cy="135.2" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="392.9" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 5</text>
    <circle cx="472.6" cy="112.8" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="472.6" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 6</text>
    <circle cx="552.3" cy="112.8" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="552.3" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 7</text>
    <circle cx="632.0" cy="66.9" r="4" fill="var(--d-accent)" stroke="var(--d-panel)" stroke-width="1.6"/>
    <text x="632.0" y="224.0" font-size="10" text-anchor="middle" fill="var(--d-muted)">it 8</text>
    <text x="392.9" y="151.2" font-size="10" text-anchor="middle" fill="var(--d-accent)" font-weight="600">33.0%</text>
    <text x="632.0" y="56.9" font-size="10" text-anchor="end" fill="var(--d-accent)" font-weight="600">45.2%</text>
    <line x1="646.0" y1="180.0" x2="646.0" y2="66.9" stroke="var(--d-warn)" stroke-dasharray="2 3"/>
    <text x="652.0" y="126.4" font-size="10" fill="var(--d-warn)" font-weight="600">+20.2</text>
    <text x="392.9" y="165.2" font-size="9.5" text-anchor="middle" fill="var(--d-muted)">gate reverts a round below its floor</text>
  </g>
</svg>
<figcaption>Eight rounds, against two fixed reference points. The dip at rounds 4&ndash;5 is the gate rejecting a set that scored below its floor. A line that only went up would mean the floor was too low to reject anything.</figcaption>
</figure>

The baseline, with no instructions, passes **33 out of 100**. The loop starts at 25% and finishes at
**45.2%** after eight rounds — a gain of **20.2 points**, from eight points below the baseline to
twelve above it.

The line dips at rounds four and five. That is the gate working: it checks each task class
against that class's own floor, so one class can be rolled back while the overall average falls.
A line that only went up would mean the floor was set too low to reject anything.

Two caveats on that number. The loop's figures come from the 90 training tasks, not all 100. And
the two reference lines are single measurements, so read them as fixed landmarks, not trends.

## What's left

<figure class="diagram">
<svg viewBox="0 0 700 204" role="img" aria-labelledby="fail-t">
  <title id="fail-t">Distribution of instruction-fixable failure categories in the most recent judged round</title>
  <g font-family="Inter, system-ui, sans-serif" font-size="11" fill="currentColor">
    <text x="0" y="12" font-size="12" font-weight="600">What the instructions could still fix, most recent judged round</text>
    <text x="0" y="28" font-size="9.5" fill="var(--d-muted)">58 fixable occurrences across 53 judged failures &#8212; measured, not assigned</text>
    <text x="206" y="59" font-size="10.5" text-anchor="end" font-weight="600">Instruction-following</text>
    <text x="206" y="71" font-size="9" text-anchor="end" fill="var(--d-muted)">33 of 58</text>
    <rect x="216" y="48" width="372.0" height="20" rx="3" fill="var(--d-accent)"/>
    <text x="596.0" y="62" font-size="10.5" font-weight="600" fill="var(--d-accent)">56.9%</text>
    <text x="206" y="93" font-size="10.5" text-anchor="end" font-weight="400">Value &amp; artifact transfer</text>
    <text x="206" y="105" font-size="9" text-anchor="end" fill="var(--d-muted)">10 of 58</text>
    <rect x="216" y="82" width="112.7" height="20" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="336.7" y="96" font-size="10.5" font-weight="600" fill="var(--d-muted)">17.2%</text>
    <text x="206" y="127" font-size="10.5" text-anchor="end" font-weight="400">Empty-state honesty</text>
    <text x="206" y="139" font-size="9" text-anchor="end" fill="var(--d-muted)">8 of 58</text>
    <rect x="216" y="116" width="90.2" height="20" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="314.2" y="130" font-size="10.5" font-weight="600" fill="var(--d-muted)">13.8%</text>
    <text x="206" y="161" font-size="10.5" text-anchor="end" font-weight="400">Entity / target selection</text>
    <text x="206" y="173" font-size="9" text-anchor="end" fill="var(--d-muted)">7 of 58</text>
    <rect x="216" y="150" width="78.9" height="20" rx="3" fill="var(--d-fill)" stroke="var(--d-line)"/>
    <text x="302.9" y="164" font-size="10.5" font-weight="600" fill="var(--d-muted)">12.1%</text>
    <line x1="216" y1="42" x2="216" y2="176" stroke="var(--d-line)"/>
    <text x="0" y="194" font-size="9.5" fill="var(--d-muted)">Dropped sub-goals, wrong app, incomplete round-trips: over half of what remains is still the shape of the task, not the mechanics of any one app.</text>
  </g>
</svg>
<figcaption>After eight rounds of rewriting the instructions, the biggest remaining problem they could fix is the agent not following them.</figcaption>
</figure>

Over half of the remaining fixable failures are instruction-following — skipping a sub-goal,
using the wrong app, doing half a round-trip.

That's an uncomfortable result for a project about writing better instructions. After eight
rounds of improving the instructions, the biggest thing they could still fix is the agent not
following them. It's also why they ended up looking like checklists rather than documentation: list the sub-goals, carry the real value across, finish both halves, then go back
and show each one was done.

## One caveat that affects everything

**83% of judged failures are tagged `insufficient_visual_evidence`** — the screenshots can't show
whether the agent succeeded, because the run ended on a loading screen or the app closed before
the sent message rendered.

So every number here is read through a lens that often can't see the outcome. Fixing screenshot
capture would do more than any change to the loop.

## What would make this solid

**Compile the facts, don't just collect them.** A fact enters the graph as a raw observation from
one judged failure and stays in that form. They pile up, overlap, and say the same thing three
different ways. A reasoning model should pass over each node and edge and consolidate them —
merge the duplicates, generalise the ones that share a shape, drop what a later fact supersedes —
so a node holds a short list of distinct lessons instead of an append-only log.

**Compile them per task, too.** The same reasoning pass should run again at retrieval: take the
facts for these apps, rank them against the task in hand, and rewrite the survivors down to
whatever the character budget allows. Today retrieval pulls every fact that matches the apps and
a truncation step cuts the overflow — so what survives is decided by its position in the file
rather than by how much it matters.

**Start from one thorough instruction set** instead of growing it edit by edit, so the loop begins with
broad coverage rather than rediscovering basics.

**Write rubrics for task shapes, not single tasks**, so one judged failure teaches us about every
task with the same shape. Same argument the graph makes, one level up.

**Check each stage, not just the final score.** Compare what the model was handed against what
actually got sent, every round. A component can do exactly what it was told while what it was
told is wrong for the data going through it — nothing crashes, and the only symptom is a score
that will not move.

**Vary the models on both sides.** One model writes and another grades; a panel on each side
would make this far less sensitive to a single model's blind spots. There is no human-labelled
ground truth for the grading yet, and everything downstream inherits whatever the judge gets
wrong.

<p class="post-note">Benchmark, task suite and infrastructure details are internal and omitted;
everything reported here is from my own runs against them.</p>
