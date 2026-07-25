/* Graph source with a stub fallback, so this surface is never blocked on
   workstream A. POSTs the user's own context to /graph so the graph comes back
   built around THEM; on any failure it serves public/sample_graph.json and
   flags the app as running on stub data. */

import { validateGraph } from './validate.js'

const GRAPH_ENDPOINT = '/graph'
const STUB = '/sample_graph.json'
const TIMEOUT_MS = 2500
// a real intake is worth waiting on — the profiler runs an LLM before the
// matcher can answer, which the empty first-paint request never does
export const INTAKE_TIMEOUT_MS = 15000

/**
 * @param {{context?: string, timeoutMs?: number}} profile
 *   `context` is the raw paste from the intake surface — the same field
 *   `POST /profile` and `POST /graph` take. Empty on first paint, which is the
 *   generic graph; filled after intake, which is the user's own.
 * @returns {Promise<{graph: object, warnings: string[], source: 'live'|'stub'}>}
 */
export async function loadGraph({ context = '', timeoutMs = TIMEOUT_MS } = {}) {
  const body = { context: typeof context === 'string' ? context.trim() : '' }

  const live = await tryLive(body, timeoutMs)
  if (live) return live

  // the 500 you'll see in the console above is the proxied POST /graph with no
  // backend behind it — expected until workstream A is up
  console.info('[kindred] /graph unavailable, rendering the stub payload')
  const res = await fetch(STUB)
  if (!res.ok) throw new Error(`stub graph unavailable (${res.status})`)
  const { ok, graph, warnings } = validateGraph(await res.json())
  if (!ok) throw new Error(`stub graph is malformed: ${warnings.join('; ')}`)
  return { graph, warnings, source: 'stub' }
}

async function tryLive(body, timeoutMs) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(GRAPH_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!res.ok) return null
    const { ok, graph, warnings } = validateGraph(await res.json())
    if (!ok) {
      console.warn('[kindred] /graph returned a malformed payload, using stub:', warnings)
      return null
    }
    return { graph, warnings, source: 'live' }
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}
