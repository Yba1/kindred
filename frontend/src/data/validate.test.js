import { describe, it, expect } from 'vitest'
import { validateGraph } from './validate.js'

const good = {
  center: 'user',
  nodes: [
    { id: 'user', name: 'You', score: 1, x: 0, y: 0 },
    { id: 'p1', name: 'Maya Chen', score: 0.87, x: 10, y: 20 },
  ],
  edges: [{ source: 'user', target: 'p1', weight: 0.87 }],
  reasons: { p1: ['same trajectory: finance → agent infra'] },
}

describe('validateGraph', () => {
  it('accepts the A→B contract payload as-is', () => {
    const { ok, graph, warnings } = validateGraph(good)
    expect(ok).toBe(true)
    expect(warnings).toEqual([])
    expect(graph.center).toBe('user')
    expect(graph.nodes).toHaveLength(2)
    expect(graph.edges).toHaveLength(1)
    expect(graph.reasons.p1).toHaveLength(1)
  })

  it('rejects a payload with no nodes instead of rendering nothing', () => {
    expect(validateGraph({ nodes: [] }).ok).toBe(false)
    expect(validateGraph(null).ok).toBe(false)
  })

  it('drops edges pointing at unknown nodes and says so', () => {
    const { graph, warnings } = validateGraph({
      ...good,
      edges: [...good.edges, { source: 'user', target: 'ghost', weight: 0.5 }],
    })
    expect(graph.edges).toHaveLength(1)
    expect(warnings.join()).toMatch(/ghost/)
  })

  it('drops self-edges', () => {
    const { graph } = validateGraph({ ...good, edges: [{ source: 'p1', target: 'p1', weight: 1 }] })
    expect(graph.edges).toHaveLength(0)
  })

  it('drops duplicate node ids', () => {
    const { graph, warnings } = validateGraph({ ...good, nodes: [...good.nodes, { id: 'p1', name: 'Dupe' }] })
    expect(graph.nodes).toHaveLength(2)
    expect(warnings.join()).toMatch(/duplicate/)
  })

  it('falls back to the first node when center is missing', () => {
    const { graph, warnings } = validateGraph({ ...good, center: 'nobody' })
    expect(graph.center).toBe('user')
    expect(warnings.join()).toMatch(/center/)
  })

  it('preserves edge features and records the payload weight as baseWeight', () => {
    const { graph } = validateGraph({
      ...good,
      edges: [{ source: 'user', target: 'p1', weight: 0.87, features: [1, 0.5, 0, 0] }],
    })
    expect(graph.edges[0].features).toEqual([1, 0.5, 0, 0])
    expect(graph.edges[0].baseWeight).toBe(0.87)
  })

  it('clamps weights and coerces junk scores', () => {
    const { graph } = validateGraph({
      ...good,
      nodes: [{ id: 'user' }, { id: 'p1', score: 'not a number' }],
      edges: [{ source: 'user', target: 'p1', weight: 4 }],
    })
    expect(graph.edges[0].weight).toBe(1)
    expect(graph.nodes[1].score).toBe(0)
  })

  it('names a node after its id when the payload omits a name', () => {
    const { graph } = validateGraph({ ...good, nodes: [{ id: 'user' }, { id: 'p1' }] })
    expect(graph.nodes[1].name).toBe('p1')
  })

  it('ignores reasons for nodes that are not in the graph', () => {
    const { graph } = validateGraph({ ...good, reasons: { ghost: ['x'], p1: ['y'] } })
    expect(Object.keys(graph.reasons)).toEqual(['p1'])
  })

  it('survives a payload with no edges or reasons at all', () => {
    const { ok, graph } = validateGraph({ center: 'user', nodes: [{ id: 'user' }] })
    expect(ok).toBe(true)
    expect(graph.edges).toEqual([])
    expect(graph.reasons).toEqual({})
  })
})
