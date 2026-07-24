/* Hardcoded weight vectors standing in for workstream D's evolution loop.
   Ordered over FEATURE_NAMES: [topic, trajectory, seeking, stage].

   When D is wired, replace this list with the `w` vectors coming off the
   Evaluator — nothing else in the frontend needs to change; the graph already
   re-clusters on whatever lands in window.applyWeights(). */

export const GENERATIONS = [
  {
    gen: 1,
    label: 'GEN 1',
    w: [0.7, 0.1, 0.1, 0.1],
    accuracy: 0.41,
    caption: 'Baseline: topic dominates. People clump by what they work on.',
  },
  {
    gen: 3,
    label: 'GEN 3',
    w: [0.35, 0.35, 0.2, 0.1],
    accuracy: 0.63,
    caption: 'Trajectory starts to count as much as topic. Clumps loosen.',
  },
  {
    gen: 6,
    label: 'GEN 6',
    w: [0.1, 0.6, 0.25, 0.05],
    accuracy: 0.84,
    caption: 'Shared trajectory beats shared topic. Clumps melt into arcs.',
  },
]

export const DEFAULT_GENERATION = 0
