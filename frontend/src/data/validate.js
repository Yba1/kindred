/* Normalizes whatever /graph hands us into something the renderer can trust.
   Bad rows get dropped with a warning rather than blanking the surface — at a
   5-hour event a graph missing two edges beats a white screen. */

import { idOf } from '../weights/rescore.js'

export function validateGraph(payload) {
  const warnings = []

  if (!payload || typeof payload !== 'object') {
    return { ok: false, graph: null, warnings: ['payload is not an object'] }
  }
  if (!Array.isArray(payload.nodes) || payload.nodes.length === 0) {
    return { ok: false, graph: null, warnings: ['payload.nodes is missing or empty'] }
  }

  const seen = new Set()
  const nodes = []
  for (const raw of payload.nodes) {
    const id = raw?.id
    if (typeof id !== 'string' || id === '') {
      warnings.push('dropped a node with no id')
      continue
    }
    if (seen.has(id)) {
      warnings.push(`dropped duplicate node "${id}"`)
      continue
    }
    seen.add(id)
    nodes.push({
      ...raw,
      id,
      name: typeof raw.name === 'string' && raw.name ? raw.name : id,
      score: num(raw.score, 0),
      // x/y ship in the payload so there's an honest first paint before the
      // sim settles; d3 takes them as the starting positions.
      x: Number.isFinite(+raw.x) ? +raw.x : undefined,
      y: Number.isFinite(+raw.y) ? +raw.y : undefined,
    })
  }

  const center = seen.has(payload.center) ? payload.center : nodes[0].id
  if (payload.center && !seen.has(payload.center)) {
    warnings.push(`center "${payload.center}" is not in nodes; using "${center}"`)
  }

  const edges = []
  for (const raw of Array.isArray(payload.edges) ? payload.edges : []) {
    const source = idOf(raw?.source)
    const target = idOf(raw?.target)
    if (!seen.has(source) || !seen.has(target)) {
      warnings.push(`dropped edge ${source} → ${target} (unknown endpoint)`)
      continue
    }
    if (source === target) {
      warnings.push(`dropped self-edge on "${source}"`)
      continue
    }
    edges.push({
      ...raw,
      source,
      target,
      weight: clamp01(num(raw.weight, 0)),
      features: Array.isArray(raw.features) ? raw.features.map((f) => clamp01(num(f, 0))) : undefined,
      baseWeight: clamp01(num(raw.weight, 0)),
    })
  }

  const reasons = {}
  const rawReasons = payload.reasons && typeof payload.reasons === 'object' ? payload.reasons : {}
  for (const [id, list] of Object.entries(rawReasons)) {
    if (!seen.has(id)) continue
    const strings = (Array.isArray(list) ? list : [list]).filter((s) => typeof s === 'string' && s.trim())
    if (strings.length) reasons[id] = strings
  }

  return {
    ok: true,
    graph: { center, nodes, edges, reasons, meta: payload.meta ?? {} },
    warnings,
  }
}

const num = (v, fallback) => (Number.isFinite(+v) ? +v : fallback)
const clamp01 = (n) => Math.min(1, Math.max(0, n))
