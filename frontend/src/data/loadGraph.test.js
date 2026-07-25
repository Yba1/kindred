import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadGraph } from './loadGraph.js'

const LIVE = {
  center: 'user',
  nodes: [{ id: 'user', name: 'You', score: 1 }, { id: 'p1', name: 'Ada', score: 0.8 }],
  edges: [{ source: 'user', target: 'p1', weight: 0.8 }],
  reasons: { p1: ['same trajectory'] },
}
const STUB = {
  center: 'user',
  nodes: [{ id: 'user', name: 'You', score: 1 }, { id: 's1', name: 'Stub Person', score: 0.4 }],
  edges: [{ source: 'user', target: 's1', weight: 0.4 }],
  reasons: {},
}

const ok = (body) => ({ ok: true, status: 200, json: async () => body })
const dead = { ok: false, status: 500, json: async () => ({}) }

/** Routes /graph and /sample_graph.json independently, and records the calls. */
function mockFetch({ graph = ok(LIVE), stub = ok(STUB) } = {}) {
  const calls = []
  const fn = vi.fn(async (url, init) => {
    calls.push({ url, init })
    if (url === '/graph') return typeof graph === 'function' ? graph(init) : graph
    return stub
  })
  vi.stubGlobal('fetch', fn)
  return calls
}

afterEach(() => vi.unstubAllGlobals())

describe('loadGraph', () => {
  it('POSTs the user context to /graph', async () => {
    const calls = mockFetch()
    const res = await loadGraph({ context: '  building an agent, seeking a cofounder  ' })

    expect(res.source).toBe('live')
    expect(calls[0].url).toBe('/graph')
    expect(calls[0].init.method).toBe('POST')
    expect(JSON.parse(calls[0].init.body)).toEqual({ context: 'building an agent, seeking a cofounder' })
  })

  it('still sends a well-formed body when nothing has been entered yet', async () => {
    const calls = mockFetch()
    await loadGraph()
    expect(JSON.parse(calls[0].init.body)).toEqual({ context: '' })
  })

  it('falls back to the stub when /graph is unreachable', async () => {
    const calls = mockFetch({ graph: () => { throw new Error('ECONNREFUSED') } })
    const res = await loadGraph({ context: 'anything' })

    expect(res.source).toBe('stub')
    expect(res.graph.nodes.map((n) => n.id)).toEqual(['user', 's1'])
    expect(calls[1].url).toBe('/sample_graph.json')
  })

  it('falls back to the stub when /graph errors', async () => {
    mockFetch({ graph: dead })
    expect((await loadGraph({ context: 'anything' })).source).toBe('stub')
  })

  it('falls back to the stub when /graph returns a malformed payload', async () => {
    mockFetch({ graph: ok({ nodes: [] }) })
    expect((await loadGraph({ context: 'anything' })).source).toBe('stub')
  })

  it('throws only when the stub is gone too, so the caller can say why', async () => {
    mockFetch({ graph: dead, stub: dead })
    await expect(loadGraph({ context: 'anything' })).rejects.toThrow(/stub graph unavailable/)
  })
})
