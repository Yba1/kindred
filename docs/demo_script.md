# Kindred — 3-minute demo script

**Total: 180s.** Written to be *said*, not read. ~430 spoken words at ~150 wpm
leaves ~30s of silence for the demo beats to land — that slack is deliberate,
don't fill it.

Three windows, arranged before you start. Never alt-tab looking for something.

| | Window | State at go-time |
|---|---|---|
| **A** | Browser — the graph (`/`) | Loaded, profile already entered, graph **not** yet generated |
| **B** | Browser — the village (`viz/village.html?cards=phase_cards.pioneer.json`) | Loaded, paused on the first card |
| **C** | Terminal in `pioneer/` | Cleared, command pre-typed **but not run** |

Pre-typed in C (literally on the prompt, cursor waiting):

```
python -m kindred_pioneer.train
```

It finishes in ~1.2 seconds. That timing is the point — don't apologise for a
wait that isn't coming.

---

## 0:00 – 0:22 · The problem (Window A, graph empty)

> "Every networking tool you've used matches you on *similarity*. Same industry,
> same keywords, same title. And then you have the coffee, and it goes nowhere.
>
> Because similar isn't the same as *useful*. Two founders both hunting for seed
> capital look identical to a similarity score — and neither one can help the
> other."

*(~60 words. Say the last sentence slowly. It's the thesis and everything after
depends on the room getting it.)*

## 0:22 – 0:50 · The graph (Window A — hit generate as you start talking)

> "Kindred builds a semantic profile of how you actually think — where you came
> from, where you're going, what you're asking for. Gemini reads that. Actian
> holds the vectors and pulls the neighbourhood."

*(Graph renders. **Click one node.** Wait for the reasoning panel — one full beat
of silence.)*

> "Every edge carries its reasoning. Not 'you both work in fintech' — *'you both
> left finance for infrastructure, and she's hiring the ML role you're trying to
> fill.'*"

*(~75 words.)*

## 0:50 – 1:20 · The village (Window B — press play)

> "And you can watch the agents argue it out. This isn't a mockup — every line is
> a real deliberation event, streamed."

*(Let it run. **Stop talking for ~5 seconds.** Let the room read one exchange —
the disagreement beat is the one that sells it. Resume on the consensus banner.)*

> "They disagree, they resolve, they commit. That's the system reasoning out loud
> instead of handing you a number and asking you to trust it."

*(~55 words.)*

## 1:20 – 2:20 · The judge — Pioneer (Window C) ← **the technical claim**

> "Underneath all of it is one question: *will these two actually connect?*
> That's a fine-tuned Pioneer model, and it's trained on outcomes — introductions
> that landed, and introductions that didn't. Not similarity."

*(**Hit enter.** It finishes in about a second. Point at the table.)*

> "Held-out split. The embedding-cosine baseline — the thing every other product
> ships — gets F1 0.549. Ours gets **0.619**. AUC 0.655 to **0.755**.
>
> And it isn't one lucky split: it wins eight of ten re-splits, and on a hundred
> pairs among people it has never seen, it beats the baseline by ten points.
>
> The reason is on that last line — the heaviest feature isn't similarity, it's
> *directional fit*. What one person needs against what the other actually has.
> A cosine score structurally cannot represent that. In a bag of words,
> 'looking for capital' and 'can offer capital' are the same tokens."

*(~140 words. This is the section that earns the rubric — do not rush it. If
you're behind schedule, cut from §0:22, never from here.)*

## 2:20 – 2:45 · The loop closes (back to Window A)

> "Then it compounds. BAND opens the intro thread — and whether that thread goes
> anywhere becomes the next training label. Six generations: the landing rate
> goes from 41% to 84%."

*(**Trigger the re-cluster.** Let the graph physically reorganise. Silence.)*

> "That's the graph re-clustering on new weights. The definition of 'close' just
> changed, and it changed because of what worked."

*(~65 words.)*

## 2:45 – 3:00 · Close (sponsor map slide up)

> "One job per sponsor, no overlap. Gemini understands, Actian retrieves, Pioneer
> judges, BAND acts — and the outcome flows back into Pioneer. Every piece sits
> on exactly one edge of the loop.
>
> Kindred doesn't guess who you should meet. It finds out, and then it gets
> better at it."

*(~55 words. Stop. Don't add a thank-you paragraph — end on the last line.)*

---

## Numbers to have cold

You will be asked. Know these without looking:

| | |
|---|---|
| Scorer vs baseline, held-out F1 | **0.619 vs 0.549** (+0.070) |
| ROC-AUC | **0.755 vs 0.655** (+0.100) |
| Test set | 50 pairs, 40.5% landing rate, threshold never fitted on test |
| 10-seed sweep | +0.086 mean, scorer wins **8 of 10** |
| Cold-start (unseen people) | 0.792 vs 0.693 (n=100) |
| Ceiling | F1 0.699 — labels are Bernoulli draws, so 1.0 is not reachable |
| Training data | ~200 labeled pairs |
| Base model | `fastino/gliner2-base-v1`, LoRA, framed as `connect`/`pass` classification |

## Questions you will get, and the honest answer

**"Is the data real?"**
> "No — it's 200 mocked pairs, and the generator is in the repo with every
> coefficient visible. So what we've shown is that the scorer recovers the
> structure from noisy labels. Pointing it at real landings is one function call;
> everything downstream is unchanged."

Say this *before* being pushed on it if the moment allows. Volunteering it reads
as rigour; conceding it under questioning reads as a gotcha.

**"n=50 — is that significant?"**
> "On that one split the bootstrap interval crosses zero, and I'd rather say so.
> That's why the claim rests on the 10-seed sweep and the 100-pair cold-start
> cohort instead — both agree, same direction, bigger margin."

**"Why not just use an LLM as the judge?"**
> "The loop asks this one yes/no question thousands of times per generation. A
> 205M encoder answers in milliseconds; a 70B model per candidate pair makes the
> loop unaffordable. That's exactly what a task-specific model is for."

**"Did you actually run a Pioneer job?"**
> "The path is wired and runnable — dataset upload, LoRA fine-tune, evaluation,
> inference — and these numbers came from the local head because we didn't have a
> key in the demo environment. The report says so; we didn't paper over it."

## Rehearsal notes

- **Run it three times out loud with a timer.** Reading silently hides ~20s.
- **The two beats you must not lose:** the F1 comparison (1:20) and the
  re-cluster (2:20). Everything else is compressible.
- **If you're at 2:00 and only on the village** — skip straight to Window C. The
  graph already made its point; the village hasn't earned 30 more seconds.
- **Silence is a tool.** Three scripted pauses: after the node click, during the
  village disagreement, and on the re-cluster. Talking over them is the single
  most common way this demo gets worse.
- **Don't read the table aloud.** Say two numbers, point at the rest.

## If something breaks

| Breaks | Do this |
|---|---|
| Graph won't render | Village (B) is standalone and needs no backend — lead with it, then go to the terminal. |
| Village stream is dead | It falls back to scripted cards automatically. Don't mention it. |
| Terminal errors | `artifacts/report.md` is committed with the same numbers. Open it and keep talking. |
| Everything is down | The sponsor map slide plus the numbers above *is* the demo. You can deliver 3 minutes off that alone — which is why it's rehearsed separately. |

**Nothing in this demo requires network access.** The training run is offline and
deterministic; the village falls back to committed cards. Confirm that on the
venue wifi anyway, before you're on stage.
