import { useState } from 'react'

/**
 * The front door: "drop in who you are" before the graph blooms around you.
 * Submits {name, context} straight to loadGraph(), which POSTs it to the
 * real backend's /graph and falls back to the stub payload on any failure —
 * so this is never a dead end, even with the backend down.
 */
export default function IntakeForm({ onSubmit, onSkip, busy, error }) {
  const [name, setName] = useState('')
  const [context, setContext] = useState('')

  const canSubmit = context.trim().length >= 12 && !busy

  const submit = (e) => {
    e.preventDefault()
    if (!canSubmit) return
    onSubmit({ name: name.trim() || undefined, context: context.trim(), top_k: 8 })
  }

  return (
    <div className="intake">
      <form className="intake-card" onSubmit={submit}>
        <span className="intake-badge">NEW PROFILE</span>
        <h1>WHO ARE YOU?</h1>
        <p className="intake-sub">
          A few sentences on your background, what you're building, and what you're
          looking for. The graph blooms around this — the matches, and why they matched,
          come straight from what you write here.
        </p>

        <label className="intake-label" htmlFor="intake-name">NAME (optional)</label>
        <input
          id="intake-name"
          className="intake-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Karthik"
          maxLength={60}
          disabled={busy}
        />

        <label className="intake-label" htmlFor="intake-context">YOUR CONTEXT</label>
        <textarea
          id="intake-context"
          className="intake-textarea"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          placeholder="Ex-quant from a derivatives desk, now building an evaluation harness for tool-using agents. Looking for a technical cofounder."
          rows={5}
          disabled={busy}
        />
        <div className="intake-hint">{context.trim().length}/12 min characters</div>

        {error && <div className="intake-error">{error}</div>}

        <div className="intake-actions">
          <button type="submit" className="intake-submit" disabled={!canSubmit}>
            {busy ? 'BUILDING YOUR PROFILE…' : 'FIND MY MATCHES'}
          </button>
          <button type="button" className="intake-skip" onClick={onSkip} disabled={busy}>
            skip — view the demo graph
          </button>
        </div>
      </form>
    </div>
  )
}
