import { describe, it, expect } from 'vitest'
import { normalizeWeights, scoreEdge, scoreEdges, driver, scoreNodes } from './rescore.js'

describe('normalizeWeights', () => {
  it('scales to sum 1', () => {
    expect(normalizeWeights([1, 1, 2, 0])).toEqual([0.25, 0.25, 0.5, 0])
  })
  it('falls back to uniform when everything is zero', () => {
    expect(normalizeWeights([0, 0])).toEqual([0.5, 0.5])
  })
  it('clamps negatives away rather than inverting the score', () => {
    expect(normalizeWeights([-1, 1])).toEqual([0, 1])
  })
  it('returns null for non-vectors', () => {
    expect(normalizeWeights(null)).toBeNull()
    expect(normalizeWeights([])).toBeNull()
  })
})

describe('scoreEdge', () => {
  const edge = { weight: 0.5, features: [1, 0, 0, 0] }

  it('follows the weight vector', () => {
    expect(scoreEdge(edge, normalizeWeights([1, 0, 0, 0]))).toBe(1)
    expect(scoreEdge(edge, normalizeWeights([0, 1, 0, 0]))).toBe(0)
  })

  it('is a weighted mean, so it stays in 0..1', () => {
    const s = scoreEdge({ features: [1, 1, 1, 1] }, normalizeWeights([0.7, 0.1, 0.1, 0.1]))
    expect(s).toBe(1)
  })

  it('keeps the payload weight when the edge has no features', () => {
    // a backend emitting the bare {source,target,weight} contract still renders
    expect(scoreEdge({ weight: 0.42 }, normalizeWeights([1, 0, 0, 0]))).toBe(0.42)
  })

  it('tolerates a weight vector longer than the feature vector', () => {
    expect(scoreEdge({ features: [1, 0] }, normalizeWeights([1, 0, 0, 0]))).toBe(1)
  })

  it('clamps out-of-range features', () => {
    expect(scoreEdge({ features: [5] }, normalizeWeights([1]))).toBe(1)
    expect(scoreEdge({ features: [-5] }, normalizeWeights([1]))).toBe(0)
  })
})

describe('scoreEdges', () => {
  it('stays index-aligned with the input', () => {
    const edges = [{ features: [1, 0] }, { features: [0, 1] }, { weight: 0.3 }]
    expect(scoreEdges(edges, [1, 0])).toEqual([1, 0, 0.3])
  })
})

describe('driver', () => {
  it('names the dim that contributed most', () => {
    const d = driver({ features: [0.2, 1, 0, 0] }, [0.1, 0.6, 0.25, 0.05])
    expect(d.name).toBe('trajectory')
  })

  it('flips as the weights learn', () => {
    const edge = { features: [1, 0.8, 0, 0] }
    expect(driver(edge, [0.7, 0.1, 0.1, 0.1]).name).toBe('topic')
    expect(driver(edge, [0.1, 0.6, 0.25, 0.05]).name).toBe('trajectory')
  })

  it('is null when there is nothing to attribute', () => {
    expect(driver({ weight: 0.5 }, [1, 0])).toBeNull()
    expect(driver({ features: [0, 0] }, [1, 1])).toBeNull()
  })
})

describe('scoreNodes', () => {
  const nodes = [{ id: 'user' }, { id: 'p1' }, { id: 'p2' }, { id: 'orphan' }]
  const edges = [
    { source: 'user', target: 'p1' },
    { source: 'p1', target: 'p2' },
  ]

  it('reads a node score off its tie to center', () => {
    const scores = scoreNodes(nodes, edges, [0.9, 0.4], 'user')
    expect(scores.get('p1')).toBe(0.9)
  })

  it('pins the center at 1', () => {
    expect(scoreNodes(nodes, edges, [0.9, 0.4], 'user').get('user')).toBe(1)
  })

  it('falls back to the strongest edge when there is no tie to center', () => {
    expect(scoreNodes(nodes, edges, [0.9, 0.4], 'user').get('p2')).toBe(0.4)
  })

  it('gives disconnected nodes zero instead of undefined', () => {
    expect(scoreNodes(nodes, edges, [0.9, 0.4], 'user').get('orphan')).toBe(0)
  })

  it('works after d3 has rewritten endpoints into node objects', () => {
    const live = [{ source: { id: 'user' }, target: { id: 'p1' } }]
    expect(scoreNodes(nodes, live, [0.77], 'user').get('p1')).toBe(0.77)
  })
})
