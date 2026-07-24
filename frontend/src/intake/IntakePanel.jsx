import { useEffect, useRef, useState } from 'react'

/**
 * The front door: "drop your context" from the README, made real.
 *
 * One surface, no wizard. The draft lives here rather than in App so typing
 * never re-renders the graph shell — and because this component stays mounted
 * while closed (it just renders nothing), whatever you typed is still there
 * when the demo re-opens it on stage.
 */
export default function IntakePanel({ open, pending, error, submitted, onSubmit, onClose }) {
  const [draft, setDraft] = useState('')
  const areaRef = useRef(null)

  // focus the textarea whenever the surface opens, and let esc dismiss it
  useEffect(() => {
    if (!open) return undefined
    const id = requestAnimationFrame(() => areaRef.current?.focus())
    const onKey = (e) => { if (e.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', onKey)
    return () => {
      cancelAnimationFrame(id)
      window.removeEventListener('keydown', onKey)
    }
  }, [open, onClose])

  if (!open) return null

  const ready = draft.trim().length > 0 && !pending
  const submit = (e) => {
    e?.preventDefault?.()
    if (ready) onSubmit(draft)
  }

  return (
    <div
      className="intake-scrim"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose?.() }}
    >
      <form
        className="intake"
        role="dialog"
        aria-modal="true"
        aria-labelledby="intake-title"
        onSubmit={submit}
      >
        <header className="intake-head">
          <span className="intake-step">YOU</span>
          <h2 id="intake-title">DROP YOUR CONTEXT</h2>
          <button
            type="button"
            className="intake-close"
            onClick={() => onClose?.()}
            aria-label="dismiss intake"
            title="dismiss (esc)"
          >
            ×
          </button>
        </header>

        <p className="intake-lede">
          Where you&apos;ve been, what you&apos;re building, what you&apos;re looking for. The
          profiler reads it, the matcher scores everyone against it, and the graph rebuilds
          with you at the centre.
        </p>

        <textarea
          ref={areaRef}
          className="intake-area"
          value={draft}
          disabled={pending}
          spellCheck="false"
          placeholder={PLACEHOLDER}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit(e) }}
          aria-label="your context"
        />

        <div className="intake-row">
          <button
            type="button"
            className="intake-example"
            disabled={pending}
            onClick={() => { setDraft(EXAMPLE); areaRef.current?.focus() }}
          >
            paste an example
          </button>
          <span className="intake-count">{draft.trim().length} chars</span>
        </div>

        {error && <p className="intake-error">{error}</p>}

        <div className="intake-foot">
          <span className="intake-hint">
            {pending ? 'profiling you, then scoring every candidate…' : '⌘/ctrl + enter to build · esc to dismiss'}
          </span>
          <button type="button" className="btn intake-btn" onClick={() => onClose?.()}>
            {submitted ? 'CANCEL' : 'SKIP'}
          </button>
          <button type="submit" className="btn btn-primary intake-btn" disabled={!ready}>
            {pending ? 'MATCHING…' : 'BUILD MY GRAPH'}
          </button>
        </div>
      </form>
    </div>
  )
}

const PLACEHOLDER =
  'e.g. Six years in fintech risk infra. Left in March to build an agent that reads regulation ' +
  'and writes the controls. Solo, pre-seed, looking for a technical cofounder who has shipped ' +
  'eval pipelines.'

const EXAMPLE = `Six years in fintech risk infra — I built the pipelines that decide whether a transaction gets blocked. Left in March to build an agent that reads incoming regulation and writes the controls for it, so compliance teams stop hand-translating PDFs into rules. Solo and pre-seed, hunting for a technical cofounder who has actually shipped eval pipelines for LLM systems, and for anyone who has sold into a compliance buyer before.`
