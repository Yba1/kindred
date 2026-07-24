import { driver } from '../weights/rescore.js'
import { domainColor } from '../graph/render.js'

/**
 * The "why", not just the "who". With a node selected it shows that person's
 * profile and every reason the matcher gave; with nothing selected it shows the
 * ranked match list, which visibly reorders as the weights learn.
 */
export default function ReasoningPanel({ graph, scores, selectedId, weights, onSelect, onFocus }) {
  if (!graph) return <aside className="panel panel-empty">loading…</aside>

  const node = selectedId ? graph.nodes.find((n) => n.id === selectedId) : null
  if (!node || node.id === graph.center) {
    return <MatchList graph={graph} scores={scores} weights={weights} onSelect={onSelect} onFocus={onFocus} />
  }

  const reasons = graph.reasons?.[node.id] ?? []
  const edge = graph.edges.find((e) => endpointsOf(e).includes(node.id) && endpointsOf(e).includes(graph.center))
  const drove = edge ? driver(edge, weights) : null
  const score = scores.get(node.id) ?? node.score

  return (
    <aside className="panel">
      <button className="panel-back" onClick={() => onSelect(null)}>← all matches</button>

      <header className="panel-head">
        <span className="panel-dot" style={{ background: domainColor(node.to) }} />
        <h2>{node.name}</h2>
        <span className="panel-score">{score.toFixed(2)}</span>
      </header>

      <dl className="profile">
        {node.from && node.to && (
          <>
            <dt>path</dt>
            <dd>{label(graph, 'from', node.from)} → {label(graph, 'to', node.to)}</dd>
          </>
        )}
        {node.seeking && (
          <>
            <dt>the ask</dt>
            <dd>looking for {label(graph, 'seeking', node.seeking)}</dd>
          </>
        )}
        {node.stage && (
          <>
            <dt>stage</dt>
            <dd>{node.stage}</dd>
          </>
        )}
      </dl>

      <h3 className="panel-sub">
        why this match
        {drove && <span className="driver-chip">driven by {drove.name}</span>}
      </h3>

      {reasons.length ? (
        <ul className="reasons">
          {reasons.map((r) => <li key={r}>{r}</li>)}
        </ul>
      ) : (
        <p className="panel-note">no reasoning returned for this node.</p>
      )}

      <div className="panel-actions">
        <button className="btn" onClick={() => onFocus?.(node.id)}>centre on graph</button>
        <button className="btn btn-primary" onClick={() => alert(`Intro thread with ${node.name} — wired to the Introducer (workstream A).`)}>
          connect
        </button>
      </div>
    </aside>
  )
}

function MatchList({ graph, scores, weights, onSelect, onFocus }) {
  const ranked = graph.nodes
    .filter((n) => n.id !== graph.center)
    .map((n) => ({ node: n, score: scores.get(n.id) ?? n.score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 12)

  return (
    <aside className="panel">
      <header className="panel-head">
        <h2>top matches</h2>
        <span className="panel-score">{graph.nodes.length - 1}</span>
      </header>
      <p className="panel-note">
        ranked by the current weight vector — this list reorders when the weights learn.
      </p>
      <ol className="match-list">
        {ranked.map(({ node, score }) => {
          const edge = graph.edges.find((e) => endpointsOf(e).includes(node.id) && endpointsOf(e).includes(graph.center))
          const drove = edge ? driver(edge, weights) : null
          return (
            <li key={node.id}>
              <button className="match-row" onClick={() => { onSelect(node.id); onFocus?.(node.id) }}>
                <span className="match-dot" style={{ background: domainColor(node.to) }} />
                <span className="match-name">{node.name}</span>
                {drove && <span className="match-driver">{drove.name}</span>}
                <span className="match-score">{score.toFixed(2)}</span>
                <span className="match-bar"><i style={{ width: `${Math.round(score * 100)}%` }} /></span>
              </button>
              <p className="match-reason">{graph.reasons?.[node.id]?.[0] ?? ''}</p>
            </li>
          )
        })}
      </ol>
    </aside>
  )
}

const endpointsOf = (e) => [
  typeof e.source === 'object' ? e.source.id : e.source,
  typeof e.target === 'object' ? e.target.id : e.target,
]

const label = (graph, kind, id) => graph.meta?.vocab?.[kind]?.[id] ?? id
