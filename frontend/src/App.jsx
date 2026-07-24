import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph from './graph/ForceGraph.jsx'
import ReasoningPanel from './panel/ReasoningPanel.jsx'
import IntakePanel from './intake/IntakePanel.jsx'
import { loadGraph, INTAKE_TIMEOUT_MS } from './data/loadGraph.js'
import { GENERATIONS, DEFAULT_GENERATION } from './weights/vectors.js'
import { FEATURE_NAMES, normalizeWeights } from './weights/rescore.js'
import { domainColor } from './graph/render.js'

const STUB_NOTICE =
  "couldn't reach the matcher at POST /graph — this is stub data from sample_graph.json, not your real matches."

export default function App() {
  const [graph, setGraph] = useState(null)
  const [source, setSource] = useState('loading')
  const [error, setError] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [weights, setWeights] = useState(GENERATIONS[DEFAULT_GENERATION].w)
  const [caption, setCaption] = useState(null)
  const [scores, setScores] = useState(() => new Map())

  // the intake surface: open on first load, dismissible, re-openable from the bar
  const [intakeOpen, setIntakeOpen] = useState(true)
  const [intakePending, setIntakePending] = useState(false)
  const [intakeError, setIntakeError] = useState(null)
  const [context, setContext] = useState('')
  const [notice, setNotice] = useState(null)

  const controllerRef = useRef(null)
  const lastScorePush = useRef(0)

  /** One landing point for a graph payload, wherever it came from. */
  const acceptGraph = useCallback(({ graph: g, warnings, source: src }) => {
    if (warnings.length) console.warn('[kindred] graph payload warnings:', warnings)
    setGraph(g)
    setSource(src)
    setScores(new Map(g.nodes.map((n) => [n.id, n.score])))
    setSelectedId(null)
  }, [])

  useEffect(() => {
    let cancelled = false
    loadGraph()
      .then((result) => {
        if (cancelled) return
        acceptGraph(result)
        if (result.source === 'stub') setNotice(STUB_NOTICE)
      })
      .catch((err) => !cancelled && setError(err.message))
    return () => { cancelled = true }
  }, [acceptGraph])

  /**
   * Intake. Re-requests the graph around this person's own context. If the
   * backend is unreachable loadGraph still hands back the stub, so the screen
   * is never blank — we just say plainly that it isn't theirs.
   */
  const submitContext = useCallback(async (text) => {
    const next = text.trim()
    if (!next) return
    setIntakePending(true)
    setIntakeError(null)
    try {
      const result = await loadGraph({ context: next, timeoutMs: INTAKE_TIMEOUT_MS })
      acceptGraph(result)
      setContext(next)
      setNotice(result.source === 'stub' ? STUB_NOTICE : null)
      setIntakeOpen(false)
    } catch (err) {
      // even the stub failed — hold the surface open and say why rather than
      // dropping the user onto a dead screen
      setIntakeError(`${err.message} — nothing to render for that context yet.`)
    } finally {
      setIntakePending(false)
    }
  }, [acceptGraph])

  const closeIntake = useCallback(() => setIntakeOpen(false), [])

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
          <p title={context || undefined}>{caption ?? subtitle(context, source)}</p>
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

        <button
          className="intake-open"
          onClick={() => { setIntakeError(null); setIntakeOpen(true) }}
          title={context ? `your context: ${context}` : 'drop your context — the graph rebuilds around you'}
        >
          {context ? 'EDIT CONTEXT' : 'YOUR CONTEXT'}
        </button>

        <a className="village-link" href="/village">VILLAGE →</a>
      </header>

      {notice && (
        <div className="notice" role="status">
          <span className="notice-tag">STUB</span>
          <span className="notice-text">{notice}</span>
          <button className="notice-x" onClick={() => setNotice(null)} aria-label="dismiss notice">×</button>
        </div>
      )}

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
        <span className={`chip chip-${source}`}>{sourceLabel(source, context)}</span>
        <span className="status-text">
          edge thickness = match score · node size = tie to you · drag to rearrange, scroll to zoom
        </span>
        <span className="spacer" />
        <span className="status-gen">{genLabel} · w = [{wn.map((x) => x.toFixed(2)).join(', ')}]</span>
      </footer>

      <IntakePanel
        open={intakeOpen}
        pending={intakePending}
        error={intakeError}
        submitted={Boolean(context)}
        onSubmit={submitContext}
        onClose={closeIntake}
      />
    </div>
  )
}

/** Says whose graph this is — and, on stub, says plainly that it isn't yours. */
function sourceLabel(source, context) {
  if (source === 'live') return context ? 'LIVE · YOUR GRAPH' : 'LIVE · /graph'
  if (source === 'stub') return context ? 'STUB DATA · NOT YOUR MATCHES' : 'STUB DATA · sample_graph.json'
  return 'LOADING…'
}

/** Never claims a match the backend didn't actually make. */
const subtitle = (context, source) => {
  if (!context) return 'the people closest to you in meaning — click a node for the reasoning'
  const verb = source === 'live' ? 'matched on your context' : 'your context'
  return `${verb}: “${truncate(context, 92)}”`
}

const truncate = (s, n) => (s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s)

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
