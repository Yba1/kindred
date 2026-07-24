import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph from './graph/ForceGraph.jsx'
import ReasoningPanel from './panel/ReasoningPanel.jsx'
import { loadGraph } from './data/loadGraph.js'
import { GENERATIONS, DEFAULT_GENERATION } from './weights/vectors.js'
import { FEATURE_NAMES, normalizeWeights } from './weights/rescore.js'
import { domainColor } from './graph/render.js'

export default function App() {
  const [graph, setGraph] = useState(null)
  const [source, setSource] = useState('loading')
  const [error, setError] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [weights, setWeights] = useState(GENERATIONS[DEFAULT_GENERATION].w)
  const [caption, setCaption] = useState(null)
  const [scores, setScores] = useState(() => new Map())

  const controllerRef = useRef(null)
  const lastScorePush = useRef(0)

  useEffect(() => {
    let cancelled = false
    loadGraph()
      .then(({ graph: g, warnings, source: src }) => {
        if (cancelled) return
        if (warnings.length) console.warn('[kindred] graph payload warnings:', warnings)
        setGraph(g)
        setSource(src)
        setScores(new Map(g.nodes.map((n) => [n.id, n.score])))
      })
      .catch((err) => !cancelled && setError(err.message))
    return () => { cancelled = true }
  }, [])

  /** The money shot. Re-scores every edge and lets the layout flow into it. */
  const applyWeights = useCallback((w, meta = {}) => {
    const next = normalizeWeights(w)
    if (!next) {
      console.warn('[kindred] applyWeights needs a numeric vector, got:', w)
      return
    }
    setWeights(next)
    setCaption(meta.caption ?? null)
    controllerRef.current?.applyWeights(next, {
      onDone: () => {
        // a learned vector lengthens weak links — re-frame so nothing walks
        // off the edge of the demo screen
        controllerRef.current?.fitToView()
        setTimeout(() => setCaption(null), 2600)
      },
    })
  }, [])

  // window.applyWeights(w) — how workstream D drives the re-cluster.
  useEffect(() => {
    window.applyWeights = (w, meta) => applyWeights(w, meta ?? {})
    window.KINDRED_GENERATIONS = GENERATIONS
    return () => { delete window.applyWeights; delete window.KINDRED_GENERATIONS }
  }, [applyWeights])

  const onGeneration = (index) => {
    const g = GENERATIONS[index]
    applyWeights(g.w, { caption: g.caption })
  }

  // the sim emits scores every frame during a re-cluster; the panel doesn't
  // need 60fps, so throttle before it hits React
  const onScores = useCallback((next) => {
    const now = performance.now()
    if (now - lastScorePush.current < 120) return
    lastScorePush.current = now
    setScores(new Map(next.map((n) => [n.id, n.score])))
  }, [])

  const onReady = useCallback((controller) => {
    controllerRef.current = controller
  }, [])

  const wn = useMemo(() => normalizeWeights(weights) ?? [], [weights])
  // derived, not stored: window.applyWeights can hand us any vector, and the
  // chip must not keep claiming GEN 1 after workstream D drives it elsewhere
  const genIndex = useMemo(
    () => GENERATIONS.findIndex((g) => sameVector(normalizeWeights(g.w), wn)),
    [wn],
  )
  const genLabel = genIndex >= 0 ? GENERATIONS[genIndex].label : 'CUSTOM w'

  if (error) {
    return (
      <div className="fatal">
        <h1>graph unavailable</h1>
        <p>{error}</p>
        <p className="panel-note">the stub lives at <code>public/sample_graph.json</code>.</p>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">KINDRED</span>
        <div className="topbar-text">
          <h1>SEMANTIC MATCH GRAPH</h1>
          <p>{caption ?? 'the people closest to you in meaning — click a node for the reasoning'}</p>
        </div>

        <div className="weights" title="the weight vector the graph is laid out under">
          {FEATURE_NAMES.map((name, i) => (
            <div className="weight" key={name} title={`${name}: ${(wn[i] ?? 0).toFixed(2)}`}>
              <div className="weight-bar"><i style={{ height: `${Math.round((wn[i] ?? 0) * 100)}%` }} /></div>
              <span>{SHORT_FEATURE[name] ?? name}</span>
            </div>
          ))}
        </div>

        <div className="gens">
          {GENERATIONS.map((g, i) => (
            <button
              key={g.gen}
              className={`gen ${i === genIndex ? 'on' : ''}`}
              onClick={() => onGeneration(i)}
              title={g.caption}
            >
              {g.label}
              <small>{Math.round(g.accuracy * 100)}%</small>
            </button>
          ))}
        </div>

        <a className="village-link" href="/village">VILLAGE →</a>
      </header>

      <main className="stage">
        <div className="graph-col">
          <Legend graph={graph} />
          <ForceGraph
          graph={graph}
          weights={weights}
          selectedId={selectedId}
          onSelect={setSelectedId}
            onReady={onReady}
            onScores={onScores}
          />
        </div>
        <ReasoningPanel
          graph={graph}
          scores={scores}
          selectedId={selectedId}
          weights={weights}
          onSelect={setSelectedId}
          onFocus={(id) => controllerRef.current?.focusNode(id)}
        />
      </main>

      <footer className="statusbar">
        <span className={`chip chip-${source}`}>
          {source === 'stub' ? 'STUB DATA · sample_graph.json' : source === 'live' ? 'LIVE · /graph' : 'LOADING…'}
        </span>
        <span className="status-text">
          edge thickness = match score · node size = tie to you · drag to rearrange, scroll to zoom
        </span>
        <span className="spacer" />
        <span className="status-gen">{genLabel} · w = [{wn.map((x) => x.toFixed(2)).join(', ')}]</span>
      </footer>
    </div>
  )
}

/** Colour is problem space, and it stays put while positions move. */
function Legend({ graph }) {
  const vocab = graph?.meta?.vocab?.to
  if (!vocab) return null
  return (
    <div className="legend">
      <span className="legend-title">problem space</span>
      {Object.entries(vocab).map(([id, label]) => (
        <span className="legend-item" key={id}>
          <i style={{ background: domainColor(id) }} />
          {label}
        </span>
      ))}
    </div>
  )
}

const SHORT_FEATURE = { topic: 'TOPIC', trajectory: 'TRAJ', seeking: 'SEEK', stage: 'STAGE' }

const sameVector = (a, b) =>
  Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) < 1e-6)
