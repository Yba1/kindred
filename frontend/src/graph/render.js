/* D3 owns this SVG. React never re-renders on a tick — it hands the renderer a
   graph once and then talks to it through the small imperative API below. */

import * as d3 from 'd3'
import { radiusFor } from './simulation.js'
import { driver, idOf } from '../weights/rescore.js'

// Categorical fill by problem space, drawn from the village's palette so the
// two surfaces read as one product. Positions move on re-cluster; colors don't
// — stable color is what makes the motion legible.
const DOMAIN_COLORS = {
  'agent-infra': '#7c6ad0',
  climate: '#5aa860',
  devtools: '#3c9c9c',
  'health-ai': '#c06a9c',
  robotics: '#c0913c',
  fintech: '#4a6ab0',
}
const FALLBACK_COLORS = ['#7c6ad0', '#5aa860', '#3c9c9c', '#c06a9c', '#c0913c', '#4a6ab0', '#b05a5a', '#8b6bb0']
const CENTER_COLOR = '#f4c542'
const LABEL_COUNT = 14

export function createRenderer(svgEl, tooltipEl, { onSelect, onHoverNode } = {}) {
  const svg = d3.select(svgEl)
  svg.selectAll('*').remove()

  const root = svg.append('g').attr('class', 'zoom-root')
  const edgeLayer = root.append('g').attr('class', 'edges')
  const nodeLayer = root.append('g').attr('class', 'nodes')
  const labelLayer = root.append('g').attr('class', 'labels')
  const tooltip = d3.select(tooltipEl)

  const zoom = d3
    .zoom()
    .scaleExtent([0.35, 4])
    .on('zoom', (event) => root.attr('transform', event.transform))
  svg.call(zoom).on('dblclick.zoom', null)

  let state = null // { nodes, edges, byId, reasons, center, weights, selected, hovered }
  let edgeSel = null
  let nodeSel = null
  let labelSel = null

  function bind({ nodes, edges, byId, sim, reasons, center, weights }) {
    state = { nodes, edges, byId, reasons, center, weights, selected: null, hovered: null }

    const colorOf = makeColorScale(nodes)

    edgeSel = edgeLayer
      .selectAll('line')
      .data(edges, (d) => `${idOf(d.source)}~${idOf(d.target)}`)
      .join('line')
      .attr('class', 'edge')
      .on('mousemove', (event, d) => showTooltip(event, edgeTooltip(d)))
      .on('mouseleave', hideTooltip)

    nodeSel = nodeLayer
      .selectAll('circle')
      .data(nodes, (d) => d.id)
      .join('circle')
      .attr('class', (d) => (d.center ? 'node node-center' : 'node'))
      .attr('fill', (d) => (d.center ? CENTER_COLOR : colorOf(d)))
      .on('click', (event, d) => {
        event.stopPropagation()
        onSelect?.(d.id)
      })
      .on('mouseenter', (event, d) => {
        state.hovered = d.id
        showTooltip(event, nodeTooltip(d))
        onHoverNode?.(d.id)
        paint()
      })
      .on('mousemove', (event) => moveTooltip(event))
      .on('mouseleave', () => {
        state.hovered = null
        hideTooltip()
        onHoverNode?.(null)
        paint()
      })
      .call(dragBehavior(sim))

    labelSel = labelLayer
      .selectAll('text')
      .data(nodes, (d) => d.id)
      .join('text')
      .attr('class', (d) => (d.center ? 'node-label node-label-center' : 'node-label'))
      .attr('text-anchor', 'middle')
      .text((d) => d.name)

    svg.on('click', () => onSelect?.(null))
    paint()
    positions()
  }

  /** Called on every sim tick — cheap attribute writes only. */
  function positions() {
    if (!edgeSel) return
    edgeSel
      .attr('x1', (d) => d.source.x)
      .attr('y1', (d) => d.source.y)
      .attr('x2', (d) => d.target.x)
      .attr('y2', (d) => d.target.y)
    nodeSel.attr('cx', (d) => d.x).attr('cy', (d) => d.y)
    labelSel.attr('x', (d) => d.x).attr('y', (d) => d.y - radiusFor(d) - 6)
    declutter()
  }

  /** Called when weights change — sizes, thicknesses and which labels show. */
  function paint() {
    if (!state) return
    const { selected, hovered, center } = state
    const focus = selected ?? hovered
    const neighbors = focus ? neighborsOf(focus) : null

    edgeSel
      .attr('stroke-width', (d) => 0.5 + d.weight * 3.6)
      .attr('stroke', (d) => (touches(d, center) ? '#b9a6ff' : '#6f6a9e'))
      .attr('stroke-opacity', (d) => {
        const base = 0.1 + d.weight * 0.55
        if (!focus) return base
        return touches(d, focus) ? Math.min(0.95, base + 0.35) : base * 0.15
      })

    nodeSel
      .attr('r', (d) => radiusFor(d))
      .attr('opacity', (d) => (!focus || neighbors.has(d.id) ? 1 : 0.25))
      .classed('is-selected', (d) => d.id === selected)

    // sized to survive fit-to-view shrinking it on a projector
    labelSel.attr('font-size', (d) => (d.center ? 13 : 11))
    declutter(true)
  }

  /**
   * Labels collide badly once a cluster tightens, so only the strongest label
   * in any overlapping run survives. Positions move every tick; recomputing
   * this at ~4Hz is imperceptible and keeps the tick cheap.
   */
  let lastDeclutter = 0
  function declutter(force = false) {
    if (!state || !labelSel) return
    const now = performance.now()
    if (!force && now - lastDeclutter < 260) return
    lastDeclutter = now

    const focus = state.selected ?? state.hovered
    const neighbors = focus ? neighborsOf(focus) : null
    const cutoff = labelCutoff()

    const rank = (n) => (n.center ? 100 : 0) + (n.id === focus ? 50 : 0) + n.score
    const candidates = state.nodes
      .filter((n) => {
        if (n.center || n.id === focus) return true
        if (focus) return neighbors.has(n.id) && n.score >= cutoff * 0.6
        return n.score >= cutoff
      })
      .sort((a, b) => rank(b) - rank(a))

    const kept = []
    const visible = new Set()
    for (const n of candidates) {
      const halfW = n.name.length * 3.5 + 4 // 11px mono, roughly
      const clash = kept.some((k) => Math.abs(k.x - n.x) < halfW + k.halfW && Math.abs(k.y - n.y) < 17)
      if (clash) continue
      kept.push({ x: n.x, y: n.y, halfW })
      visible.add(n.id)
    }

    labelSel.attr('opacity', (d) => {
      if (!visible.has(d.id)) return 0
      return d.center || d.id === focus ? 1 : 0.85
    })
  }

  /**
   * Frame the whole graph. A learned weight vector lengthens weak links, so
   * after a re-cluster the layout can spill past the viewport — this pulls it
   * back without the demo having to touch the mouse.
   */
  function fitToView({ padding = 70, duration = 700 } = {}) {
    if (!state || state.nodes.length === 0) return
    const xs = state.nodes.map((n) => n.x).filter(Number.isFinite)
    const ys = state.nodes.map((n) => n.y).filter(Number.isFinite)
    if (!xs.length) return

    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const { width, height } = svgEl.getBoundingClientRect()
    if (!width || !height) return

    const scale = Math.min(
      2.5,
      Math.max(0.35, 0.95 * Math.min((width - padding * 2) / Math.max(1, maxX - minX), (height - padding * 2) / Math.max(1, maxY - minY))),
    )
    const transform = d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(scale)
      .translate(-(minX + maxX) / 2, -(minY + maxY) / 2)

    svg.transition().duration(duration).call(zoom.transform, transform)
  }

  function setWeights(w) {
    if (state) state.weights = w
  }

  function setSelected(id) {
    if (!state) return
    state.selected = id
    paint()
  }

  /** Center the view on a node — used when the panel's match list is clicked. */
  function focusNode(id) {
    const node = state?.byId.get(id)
    if (!node) return
    const { width, height } = svgEl.getBoundingClientRect()
    svg
      .transition()
      .duration(600)
      .call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(1.4).translate(-node.x, -node.y))
  }

  function resetZoom() {
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity)
  }

  function destroy() {
    hideTooltip()
    svg.on('.zoom', null).on('click', null)
    svg.selectAll('*').remove()
    state = null
  }

  // ---- helpers ------------------------------------------------------------

  function labelCutoff() {
    const scores = state.nodes
      .filter((n) => !n.center)
      .map((n) => n.score)
      .sort((a, b) => b - a)
    return scores[Math.min(LABEL_COUNT, scores.length - 1)] ?? 0
  }

  function neighborsOf(id) {
    const set = new Set([id])
    for (const e of state.edges) {
      const s = idOf(e.source)
      const t = idOf(e.target)
      if (s === id) set.add(t)
      else if (t === id) set.add(s)
    }
    return set
  }

  function touches(edge, id) {
    return idOf(edge.source) === id || idOf(edge.target) === id
  }

  function edgeTooltip(d) {
    const a = state.byId.get(idOf(d.source))
    const b = state.byId.get(idOf(d.target))
    const other = idOf(d.source) === state.center ? idOf(d.target) : idOf(d.source)
    const reason = touches(d, state.center) ? state.reasons?.[other]?.[0] : null
    const drove = driver(d, state.weights)
    return [
      `<b>${esc(a?.name)} ↔ ${esc(b?.name)}</b>`,
      `<span class="tt-score">${d.weight.toFixed(2)}</span>${drove ? ` · ${esc(drove.name)}` : ''}`,
      reason ? `<span class="tt-reason">${esc(reason)}</span>` : '',
    ].filter(Boolean).join('<br>')
  }

  function nodeTooltip(d) {
    if (d.center) return `<b>${esc(d.name)}</b><br><span class="tt-reason">you — the graph centre</span>`
    const first = state.reasons?.[d.id]?.[0]
    return [
      `<b>${esc(d.name)}</b> <span class="tt-score">${d.score.toFixed(2)}</span>`,
      first ? `<span class="tt-reason">${esc(first)}</span>` : '',
      '<span class="tt-hint">click for the full reasoning</span>',
    ].filter(Boolean).join('<br>')
  }

  function showTooltip(event, html) {
    tooltip.html(html).classed('show', true)
    moveTooltip(event)
  }

  function moveTooltip(event) {
    const bounds = svgEl.getBoundingClientRect()
    tooltip.style('left', `${event.clientX - bounds.left + 14}px`).style('top', `${event.clientY - bounds.top + 14}px`)
  }

  function hideTooltip() {
    tooltip.classed('show', false)
  }

  function dragBehavior(sim) {
    return d3
      .drag()
      .on('start', (event, d) => {
        if (!event.active) sim.alphaTarget(0.25).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on('drag', (event, d) => {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', (event, d) => {
        if (!event.active) sim.alphaTarget(0)
        if (d.center) return // the centre stays pinned
        d.fx = null
        d.fy = null
      })
  }

  return { bind, positions, paint, setSelected, setWeights, focusNode, fitToView, resetZoom, destroy }
}

function makeColorScale(nodes) {
  const domains = [...new Set(nodes.map((n) => n.to).filter(Boolean))]
  const ordinal = d3.scaleOrdinal(domains, FALLBACK_COLORS)
  // payloads that carry `to` get the product's domain colors; anything else
  // still gets a stable categorical color
  return (d) => DOMAIN_COLORS[d.to] ?? (d.to ? ordinal(d.to) : '#6f6a9e')
}

export function domainColor(to) {
  return DOMAIN_COLORS[to] ?? '#6f6a9e'
}

const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
