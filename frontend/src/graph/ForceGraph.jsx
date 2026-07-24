import { useEffect, useRef } from 'react'
import { createSimulation } from './simulation.js'
import { createRenderer } from './render.js'

/**
 * Thin React shell around the D3 layer. The graph is built once per payload;
 * everything after that (selection, weights, zoom) goes through the imperative
 * controller handed back via onReady — React state never drives a tick.
 */
export default function ForceGraph({ graph, weights, selectedId, onSelect, onReady, onScores }) {
  const svgRef = useRef(null)
  const tooltipRef = useRef(null)
  const wrapRef = useRef(null)
  const controllerRef = useRef(null)

  // live refs so the build effect below depends only on the payload
  const onSelectRef = useRef(onSelect)
  const onReadyRef = useRef(onReady)
  const onScoresRef = useRef(onScores)
  const weightsRef = useRef(weights)
  onSelectRef.current = onSelect
  onReadyRef.current = onReady
  onScoresRef.current = onScores
  weightsRef.current = weights

  useEffect(() => {
    if (!graph) return undefined
    const wrap = wrapRef.current
    const box = wrap.getBoundingClientRect()

    const renderer = createRenderer(svgRef.current, tooltipRef.current, {
      onSelect: (id) => onSelectRef.current?.(id),
    })

    const simulation = createSimulation({
      graph,
      width: box.width || 1180,
      height: box.height || 720,
      onTick: () => renderer.positions(),
      onRescore: ({ nodes }) => {
        renderer.paint()
        onScoresRef.current?.(nodes.map((n) => ({ id: n.id, score: n.score })))
      },
    })

    renderer.bind({
      nodes: simulation.nodes,
      edges: simulation.edges,
      byId: simulation.byId,
      sim: simulation.sim,
      reasons: graph.reasons,
      center: graph.center,
      weights: weightsRef.current,
    })

    // frame the graph once it first settles, then leave the view alone —
    // re-fitting after every drag would fight the user
    let fitted = false
    simulation.sim.on('end', () => {
      if (fitted) return
      fitted = true
      renderer.fitToView()
    })

    const controller = {
      applyWeights(w, opts) {
        renderer.setWeights(w)
        simulation.applyWeights(w, opts)
      },
      focusNode: renderer.focusNode,
      fitToView: renderer.fitToView,
      resetZoom: renderer.resetZoom,
      reheat: simulation.reheat,
      nodes: simulation.nodes,
    }
    controllerRef.current = { renderer, simulation }
    onReadyRef.current?.(controller)

    const onResize = () => {
      const next = wrap.getBoundingClientRect()
      simulation.resize(next.width, next.height)
    }
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      simulation.stop()
      renderer.destroy()
      controllerRef.current = null
    }
  }, [graph])

  // selection is a cheap repaint, not a rebuild
  useEffect(() => {
    controllerRef.current?.renderer.setSelected(selectedId ?? null)
  }, [selectedId])

  return (
    <div className="graph-wrap" ref={wrapRef}>
      <svg className="graph" ref={svgRef} />
      <div className="graph-tooltip" ref={tooltipRef} />
    </div>
  )
}
