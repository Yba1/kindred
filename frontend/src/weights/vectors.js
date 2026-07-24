/* REAL weight vectors from the evolution loop's run.json, not stand-ins.

   loop/contracts.py's FEATURES (6 dims): [domain_sim, focus_sim, trajectory_sim,
   seeking_match, collab_sim, expertise_fit]. Mapped down to this frontend's
   FEATURE_NAMES (4 dims: [topic, trajectory, seeking, stage]) by taking the
   closest counterpart per dim — domain_sim->topic, trajectory_sim->trajectory,
   seeking_match->seeking, expertise_fit->stage — and dropping focus_sim/
   collab_sim (no frontend slot for them). Negative components (the loop
   demoting a dimension) come through as-is; normalizeWeights() clamps them to
   0 and renormalizes, which is exactly the "this dimension stopped mattering"
   story. Regenerate this file with `python -m loop.run` + re-copy run.json's
   `generations[].weights` if the loop is re-tuned. */

export const GENERATIONS = [
  {
    gen: 0,
    label: 'GEN 0',
    w: [1.0, 0.0, 0.0, 0.0],
    accuracy: 0.3846,
    caption: 'Baseline: topic dominates. People clump by what they work on.',
  },
  {
    gen: 3,
    label: 'GEN 3',
    w: [-0.0786, 0.1665, 0.4127, 0.0003],
    accuracy: 0.6923,
    caption: 'Domain demoted, seeking and trajectory rising. Clumps loosen.',
  },
  {
    gen: 5,
    label: 'GEN 5',
    w: [-0.2614, 0.2474, 0.6072, 0.1186],
    accuracy: 0.8462,
    caption: 'Shared trajectory and intent beat shared topic. Clumps melt into arcs.',
  },
]

export const DEFAULT_GENERATION = 0
