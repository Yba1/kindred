/* Graph source with a stub fallback, so this surface is never blocked on
   workstream A. Tries POST /graph; on any failure serves public/sample_graph.json
   and flags the app as running on stub data. */

import { validateGraph } from './validate.js'

const GRAPH_ENDPOINT = '/graph'
const STUB = '/sample_graph.json'
const TIMEOUT_MS = 2500

export async function loadGraph(profile = {}) {
  const live = await tryLive(profile)
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

async function tryLive(profile) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    const res = await fetch(GRAPH_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile),
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
