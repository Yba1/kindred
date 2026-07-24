/* The force layout, and the re-cluster tween that is the demo's money shot.

   Edge weight drives BOTH link distance and link strength, so when a new
   weight vector lands the nodes physically flow into a new configuration
   instead of snapping — clumps visibly dissolve and re-form. */

import * as d3 from 'd3'
import { scoreEdges, scoreNodes, idOf } from '../weights/rescore.js'

const MIN_DISTANCE = 38
const DISTANCE_SPAN = 280
const RECLUSTER_MS = 2000

export function createSimulation({ graph, width, height, onTick, onRescore }) {
  // d3 mutates what it's given; keep the validated payload clean
  // `center` is derived from the payload's center id, not trusted off the node
  const nodes = graph.nodes.map((n) => ({ ...n, center: n.id === graph.center }))
  const edges = graph.edges.map((e) => ({ ...e }))
  const byId = new Map(nodes.map((n) => [n.id, n]))

  // degree is fixed for the life of the graph — precompute so the per-frame
  // link-force refresh during a re-cluster stays cheap
  const degree = new Map()
  for (const e of edges) {
    bump(degree, idOf(e.source))
    bump(degree, idOf(e.target))
  }

  const linkDistance = (e) => MIN_DISTANCE + (1 - e.weight) ** 1.6 * DISTANCE_SPAN
  // degree-normalized, the way d3's default does it, or hub nodes tear the
  // layout apart at 400+ edges
  const linkStrength = (e) => {
    const min = Math.min(degree.get(idOf(e.source)) ?? 1, degree.get(idOf(e.target)) ?? 1)
    return Math.max(0.02, e.weight) / Math.max(1, min)
  }

  const linkForce = d3
    .forceLink(edges)
    .id((d) => d.id)
    .distance(linkDistance)
    .strength(linkStrength)

  const sim = d3
    .forceSimulation(nodes)
    .force('link', linkForce)
    .force('charge', d3.forceManyBody().strength(-150).distanceMax(520))
    .force('collide', d3.forceCollide().radius((d) => radiusFor(d) + 3))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .velocityDecay(0.35)
    .on('tick', () => onTick?.())

  // Containment pulled harder on the short axis, so the node cloud settles into
  // roughly the viewport's aspect instead of a tall column that wastes half a
  // widescreen projector.
  applyAspectForces(width, height)
  pinCenter(width, height)

  let tween = null

  /** Re-score every edge under `w` and let the layout re-settle into it. */
  function applyWeights(w, { duration = RECLUSTER_MS, onDone } = {}) {
    const from = edges.map((e) => e.weight)
    const to = scoreEdges(edges, w)

    tween?.stop()
    sim.alphaTarget(0.32).restart()

    tween = d3.timer((elapsed) => {
      const k = Math.min(1, elapsed / duration)
      const eased = d3.easeCubicInOut(k)
      for (let i = 0; i < edges.length; i += 1) {
        edges[i].weight = from[i] + (to[i] - from[i]) * eased
      }
      // re-setting the accessors re-initializes d3's cached distance/strength
      linkForce.distance(linkDistance).strength(linkStrength)
      publishScores()

      if (k >= 1) {
        tween.stop()
        tween = null
        sim.alphaTarget(0).alpha(0.45).restart()
        onDone?.()
      }
    })
  }

  /** Node score follows its tie to center — that's what the ranked list reads. */
  function publishScores() {
    const scores = scoreNodes(nodes, edges, edges.map((e) => e.weight), graph.center)
    for (const n of nodes) n.score = scores.get(n.id) ?? 0
    onRescore?.({ nodes, edges })
  }

  function applyAspectForces(w, h) {
    const aspect = w > 0 && h > 0 ? w / h : 1
    sim
      .force('x', d3.forceX(w / 2).strength(0.012))
      .force('y', d3.forceY(h / 2).strength(0.012 * Math.max(1, aspect)))
  }

  function pinCenter(w, h) {
    const center = byId.get(graph.center)
    if (!center) return
    center.fx = w / 2
    center.fy = h / 2
  }

  function resize(w, h) {
    sim.force('center', d3.forceCenter(w / 2, h / 2))
    applyAspectForces(w, h)
    pinCenter(w, h)
    sim.alpha(0.3).restart()
  }

  function reheat(alpha = 0.6) {
    sim.alpha(alpha).restart()
  }

  function stop() {
    tween?.stop()
    sim.stop()
  }

  return { sim, nodes, edges, byId, applyWeights, publishScores, resize, reheat, stop }
}

/** Shared by the sim (collision) and the renderer (circle r) so they agree. */
export function radiusFor(node) {
  if (node.center) return 17
  return 4.5 + Math.min(1, Math.max(0, node.score ?? 0)) * 12
}

function bump(map, key) {
  map.set(key, (map.get(key) ?? 0) + 1)
}
