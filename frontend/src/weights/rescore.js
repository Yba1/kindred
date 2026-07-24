/* Pure scoring. No DOM, no D3 — this is the piece the re-cluster animation
   leans on, so it's kept testable in isolation.

   The Evaluator (workstream D) emits an ordered weight vector `w` over the
   feature dims below. An edge carries its per-dim `features`; the match score
   is the weighted mean of those features.

   Edges WITHOUT a `features` array keep their payload weight unchanged — so a
   backend that only emits `{source,target,weight}` still renders fine, it just
   won't re-cluster. (Workstream A: emit `features` per edge to light this up.) */

export const FEATURE_NAMES = ['topic', 'trajectory', 'seeking', 'stage']

const clamp01 = (n) => (Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0)

/** Non-negative weights summing to 1. Falls back to uniform on garbage input. */
export function normalizeWeights(w) {
  if (!Array.isArray(w) || w.length === 0) return null
  const clean = w.map((x) => (Number.isFinite(+x) ? Math.max(0, +x) : 0))
  const sum = clean.reduce((a, b) => a + b, 0)
  if (sum <= 0) return clean.map(() => 1 / clean.length)
  return clean.map((x) => x / sum)
}

/** Weighted mean of an edge's features under normalized weights `wn`. */
export function scoreEdge(edge, wn) {
  const f = edge?.features
  if (!wn || !Array.isArray(f) || f.length === 0) return clamp01(edge?.weight ?? 0)
  const n = Math.min(f.length, wn.length)
  let acc = 0
  let mass = 0
  for (let i = 0; i < n; i += 1) {
    acc += wn[i] * clamp01(+f[i])
    mass += wn[i]
  }
  return mass > 0 ? clamp01(acc / mass) : 0
}

/** Index-aligned scores for a whole edge list. */
export function scoreEdges(edges, w) {
  const wn = normalizeWeights(w)
  return edges.map((e) => scoreEdge(e, wn))
}

/** Which feature dim actually drove this score — the "why it moved" chip. */
export function driver(edge, w) {
  const wn = normalizeWeights(w)
  const f = edge?.features
  if (!wn || !Array.isArray(f) || f.length === 0) return null
  let best = -1
  let bestVal = -Infinity
  let total = 0
  for (let i = 0; i < Math.min(f.length, wn.length); i += 1) {
    const contribution = wn[i] * clamp01(+f[i])
    total += contribution
    if (contribution > bestVal) {
      bestVal = contribution
      best = i
    }
  }
  if (best < 0 || total <= 0) return null
  return {
    index: best,
    name: FEATURE_NAMES[best] ?? `dim ${best}`,
    share: bestVal / total,
  }
}

/**
 * Node score = strength of its tie to the graph center, which is what the
 * ranked match list reads. Nodes with no tie to center fall back to their
 * strongest edge so nothing silently drops to zero.
 */
export function scoreNodes(nodes, edges, edgeScores, centerId) {
  const toCenter = new Map()
  const strongest = new Map()
  edges.forEach((e, i) => {
    const s = edgeScores[i]
    const src = idOf(e.source)
    const tgt = idOf(e.target)
    if (src === centerId) toCenter.set(tgt, s)
    else if (tgt === centerId) toCenter.set(src, s)
    strongest.set(src, Math.max(strongest.get(src) ?? 0, s))
    strongest.set(tgt, Math.max(strongest.get(tgt) ?? 0, s))
  })
  const out = new Map()
  for (const n of nodes) {
    if (n.id === centerId) out.set(n.id, 1)
    else out.set(n.id, toCenter.get(n.id) ?? strongest.get(n.id) ?? 0)
  }
  return out
}

/** d3-force rewrites `source`/`target` into node objects once the sim runs. */
export function idOf(endpoint) {
  return typeof endpoint === 'object' && endpoint !== null ? endpoint.id : endpoint
}
