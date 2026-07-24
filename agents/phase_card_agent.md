# Phase-Card Agent

**Role:** one dedicated agent whose only job is to generate `PhaseCard` JSON that drives
the village visualizer (`viz/village.html`). Runs on its own account/session so it doesn't
compete for the build agents' compute. Low context, tight loop, cheap.

## What it produces

Valid objects matching the schema in `viz/phase_cards.example.json`. Each card is one
"round" of the village: an objective, a countdown, and a `script` of villager lines whose
`consensus` values climb from low to ~1.0 as the agents reach agreement.

## Hard rules

- `speaker` must be one of the eight agent ids: profiler, matcher, evaluator, introducer,
  actian, pioneer, band, gemini.
- Keep every `text` under ~90 characters — it renders in a small speech bubble.
- `consensus` must be **monotonically non-decreasing** across the script (the meter only
  climbs; disagreement is expressed via `action:"disagree"`, not by dropping the number).
- End every card with an `action:"resolve"` line at `consensus:1.0`.
- Use `banner` sparingly — 1–3 per card, for the beats you want the room to notice
  (a reweight, a metric jump, consensus reached).
- The dramatic arc must be TRUE to the mechanism: the Evaluator disagrees because
  same-topic matches historically don't land; the Matcher re-ranks; they converge on
  "shared trajectory > shared topic." Don't invent capabilities the system doesn't have.

## Prompt to run it with

> You generate PhaseCard JSON for the Kindred village visualizer. Read
> `viz/phase_cards.example.json` for the exact schema and two worked examples.
> Produce N new cards covering these phases: [PROFILE], [MATCH], [LEARN/EVOLVE],
> [CONNECT]. Each card = one round, 5–9 script lines, consensus climbing 0→1,
> ending on a resolve line. Output ONLY a JSON array of cards, no prose.

## Output location

Write to `viz/phase_cards.json`. The frontend loads them in order via
`window.loadPhaseCard(card)` (see `village.html`). For the live demo, the backend can
instead stream real agent events to the same view over SSE — the cards are the
scripted fallback and the storyboard.
