# `fix/translation-init` — coverage-invariant translation seeding

This branch adds one option to the registration method and changes nothing by
default. It is written for a reviewer who has not followed the project.

## The problem it addresses

The task is to align a SLAM reconstruction of an indoor space to a BIM model of
the same building. The BIM covers **less** than the scan: it models one or two
rooms, while the scan also contains a corridor and a neighbouring room. Coverage
of the source by the reference ranges from about 30% to 100% across our data.

The method seeds its translation two ways, and both are statistics of the
observed range:

- `_struct_centroid` — the centroid of the structural points
- `_axis_span` — the midpoint of the 2nd–98th percentile along each axis

Neither is invariant to coverage. If the reference is one room and the source is
that room plus a corridor, the two centroids sit several metres apart, and the
seed starts with that gap built in.

Measured on our 10 real scans (30 configurations where the rotation converged to
within 5°): the translation error correlates with the centroid gap at r = +0.75.
Where the gap is 1.4 m the final translation error is 0.10 m; where it is 9.1 m
the error is 20.1 m.

A plane's position along its own normal does **not** move when you see only part
of the plane. That is the invariant this branch uses.

## What was added

`Registration/regbim/plane_match.py`

- `offset_candidates(src_proj, dst_proj, ...)` — projects wall points onto one
  axis, builds a 1-D histogram for each side and cross-correlates them. Returns
  the top-K offsets, kept apart by `min_separation_m`.
- `plane_offset(src_proj, dst_proj)` — mode difference, used for the vertical
  axis where floors give a single sharp peak.

Cross-correlation rather than "align the strongest peak", because under partial
coverage the two sides do not have the same set of walls: the source sees
corridor walls the reference does not contain, so aligning single peaks can align
two different walls. Correlating the whole profile lets the remaining walls
outvote a missing one.

It returns **candidates, not an answer**, because the ambiguity is real. For
walls at 0/3/6/9/12 m matched against a reference containing only 0 and 3, four
offsets (−6, −3, 0, −9) score within 2.3% of each other, and the correct one is
not rank 1. `tests/test_plane_match.py` asserts exactly that.

`Registration/regbim/methods/proposed.py`

`proposed.translation_init` selects the behaviour:

| value | behaviour |
|---|---|
| `centroid` (default) | one seed per Manhattan yaw candidate, full ICP on each, then pick the best — unchanged |
| `plane_match` | adds translation seeds from the correlation, ranks all seeds at the seed pose, runs ICP on the best `max_icp_candidates` |

## Default behaviour is unchanged, and this is checked

`Registration/scripts/regress_scale_init.sh <base-commit>` loads the point clouds
once, saves them, and then calls `register()` under both the base commit (in a
detached git worktree) and the working tree, comparing the returned 4×4 Sim(3)
element-wise.

Against `3e6e4b3` (the commit this branch starts from): **max difference
0.000e+00**.

The clouds have to be fixed for this comparison because the pipeline samples
200k points from the mesh at load time, so two runs of identical code otherwise
disagree. `register()` itself is deterministic (the RANSAC generator is seeded).

## Result: it does not help, and the reason is more interesting than the fix

Measured against a ground truth built independently of the method (manual
alignment in CloudCompare, refined by plain geometric ICP — the proposed method
is not involved):

| scene | mode | selection score | rotation error | translation error |
|---|---|---:|---:|---:|
| m3_cor_c E2 | reference transform | **0.305** | — | — |
| m3_cor_c E2 | `centroid` | 0.136 | 90.6° | 20.1 m |
| m3_cor_c E2 | `plane_match` | **0.339** | 179.4° | 13.8 m |
| m3_cor_a E3 | reference transform | **0.489** | — | — |
| m3_cor_a E3 | `centroid` | 0.220 | 0.1° | 10.6 m |
| m3_cor_a E3 | `plane_match` | 0.346 | 90.1° | 37.8 m |

On `m3_cor_c` the new seeds found a pose that **scores higher than the
independently built reference transform** (0.339 vs 0.305) while being 179° wrong.

The score being maximised is the class-constrained inlier ratio, the same
criterion the method already used to pick among yaw candidates. What this
establishes is a counterexample:

> a badly wrong placement exists whose score exceeds the reference transform's.

That is enough to say **adding the correct transform to the candidate set does not
by itself remove the inversion**. It does *not* show that the wrong pose is the
global maximum of the score, nor that every pose within the success tolerance of
the reference scores below it. Those are stronger statements and are not tested
here.

Under the conditions run, disabling the pre-ranking (`max_icp_candidates` 8 → 64,
178 s instead of 52 s) returned the identical pose, so the pruning does not
account for this instance.

### Scale is not the whole story

The error metric reports `|s_method/s_ref - 1|`, which cannot distinguish shrinking
from growing. The signed ratios do:

| scene | mode | scale ratio | rotation error | score |
|---|---|---:|---:|---:|
| m3_cor_c E2 | `plane_match` | **0.644** (36% smaller) | 179.1° | 0.342 |
| m3_cor_a E3 | `plane_match` | **1.007** (0.7% off) | 90.3° | 0.356 |
| m3_cor_a E3 | `centroid` | 0.831 | 0.3° | 0.219 |

Shrinking the source packs more of it against a small reference, which raises the
inlier count, and that fits `m3_cor_c`. It does not fit `m3_cor_a`: there the scale
is correct to 0.7% and the pose still outscores the `centroid` result while being
90° wrong. **A free scale is therefore not necessary for the inversion.**

The objective has no term requiring the reference to be explained, and no prior on
scale even though the input is metric (ARKit + LiDAR). Which of those matters, if
either, is not yet established.

## Status

The implementation is complete and the default is verified unchanged. Whether to
adopt `plane_match` is not decided here: on this evidence the binding constraint
is the selection criterion, not the seeding.
