# Kindred — 3-Minute Demo Script

Rehearsal doc. Read the stage directions silently, say only the spoken paragraphs out loud.

---

**0:00-0:20 — The problem**

"LinkedIn shows you who you already know. But the people who'd actually change your life are usually zero hops away in meaning and infinite hops away in who they know — you think alike, you'd never meet. Social graphs measure who's connected. Nobody's measuring who's *aligned*. That's what Kindred does."

*Landing on the Kindred graph page, empty state, before any context is entered.*

---

**0:20-0:45 — Drop context, the graph blooms**

"Watch. I drop in my context — a few lines about what I'm building and where I've been." *Type/paste a short bio into the input, hit submit.* "Kindred splits me into six dimensions, not one blob: domain, focus, trajectory, seeking, collab style, expertise. The graph blooms live." *Nodes animate in around the center.* "Let's click one." *Click a node.* "Here's the actual reasoning — not a black box." *Read the reasoning text on the node's panel aloud, verbatim, one line.*

---

**0:45-1:15 — Connect, the Introducer drafts via BAND**

"Say I want to reach this one." *Click "Connect" on the selected node.* "The Introducer just drafted an opener — through BAND — and it's not 'we're both in finance.' It leads with the thing that actually matters: same pivot, same ask, same async style." *Read the drafted opener aloud, one or two lines.* "That's a real intro thread, not a template."

---

**1:15-2:00 — The money shot: hit "learn"**

"Now the part that makes this a self-evolving agent and not a static matcher." *Click "Learn."* "The Evaluator just looked at which introductions actually landed, and found something we didn't expect: domain and topic — the thing LinkedIn optimizes for — was a *weak* predictor. Trajectory and intent predicted real landings far better." *Point at the connection-rate readout.* "Watch the connection rate: 41 percent... climbing... 84 percent." *Let the number animate up on screen.* "And the graph is re-clustering in real time — the topic clumps are dissolving, and it's regrouping around shared trajectory and shared ask instead." *Point at nodes visibly reorganizing.*

---

**2:00-2:25 — Pivot to the village page**

"Here's how that decision actually got made." *Switch tab/window to `/village`.* "Every agent in the loop is a villager, arguing this out in the town square." *Let a few lines of dialogue play — Matcher defending domain, Evaluator pushing back with the 41% number, Pioneer's held-out score landing, consensus meter climbing to the gate passing.* "This is the autonomy piece — nobody hand-tuned those weights, the agents reached consensus and the system promoted itself. And the actual intro thread you saw a second ago runs through the same BAND channel these agents are debating over."

---

**2:25-2:50 — The sponsor tour, in one breath**

"Quick tour of who's doing what: Actian holds and retrieves the six-dimension vectors — that's the semantic search underneath the graph. Gemini does the profiling and the reasoning you read on that node. Pioneer is the fine-tuned scorer — it's the one that actually got measurably better, 41 to 84. And BAND is both the intro thread and the village consensus channel you just watched. Four sponsors, four distinct jobs, nothing overlapping."

---

**2:50-3:00 — The close**

"LinkedIn shows you who you already know. Kindred finds who you should."

*Cut back to the graph, fully re-clustered, sitting on the final state.*

---

## If the live demo breaks

Kindred ships a deterministic replay: run with `--replay` and it plays back a recorded `run.json` — same graph, same reasoning, same 41→84% climb, same village dialogue, byte-for-byte. Mention this once at rehearsal so whoever's driving knows the flag exists. If something glitches on stage, just switch to replay and keep talking — don't apologize, don't explain, the audience can't tell the difference.

## Honesty reminder

This is **one self-improving loop over a match-scoring function** — the Evaluator refits weights against held-out outcomes and a promotion gate only ships them if they improve. It is not eight independently-evolving agents; the agents are the deliberation and interface layer around that one loop. Say it that way: confident about what the loop actually does, no overclaiming a swarm of separately-learning agents.
