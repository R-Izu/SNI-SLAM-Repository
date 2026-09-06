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

## Result: it does not help, and my earlier explanation of why was wrong

**Retraction.** An earlier revision of this document reported that `plane_match`
found a pose scoring **above** the independently built reference transform
(0.339 vs 0.305), and concluded from that the selection objective was at fault.
**That figure does not reproduce.** Re-measuring at the exact commit it came from
(`50c34f2`) gives:

| scene | mode | selection score | rotation error | translation error |
|---|---|---:|---:|---:|
| m3_cor_c E2 | reference transform | **0.306** | — | — |
| m3_cor_c E2 | `centroid` | 0.135 | 90.4° | 20.1 m |
| m3_cor_c E2 | `plane_match` | **0.092** | 179.6° | 15.8 m |

The returned pose is the same one as before (179.6°, scale ratio 0.62); only the
score differs. So the 0.339 was a measurement error on my side, not a property of
the objective. Repeating with five different sampling seeds gives 0.073–0.092
every time, and the reference transform scores highest in all five.

**The objective is not shown to be broken.** What these numbers show is that the
method's search does not reach the correct pose: the reference transform scores
3.3× the best pose the method returns.

Reference against which this is measured: manual alignment in CloudCompare,
refined by plain geometric ICP. The proposed method is not involved.

### What the scale sweep does show

Holding the rotation and the image of the source centroid fixed and varying only
the scale (`Registration/scripts/diag_scale_sweep.py`):

- for the reference rotation the score peaks at scale ratio **1.005** (0.307), so
  the objective does prefer the correct scale once the rotation is right;
- the wrong pose peaks at **0.176** (scale ratio 0.725), still well below 0.307.

The identity `N_C · Δρ = N_enter − N_leave` holds exactly in both sweeps — the
denominator does not depend on the transform — which checks the instrumentation.

### Reference-side coverage separates the two cleanly

Fraction of reference points having a same-class source point within the gate:

| scene | reference transform | wrong pose |
|---|---:|---:|
| m3_cor_c E2 | **0.664** | 0.071 |
| m3_cor_a E3 | **0.690** | 0.500 |

The wrong poses leave most of the reference unexplained, and the current score
only looks from the source side, so it cannot see this. A reference-side term
would separate these cases. Whether it would help the *search* reach the right
pose is a different question, and is not tested here.

### Signed scale, for the record

The error metric reports `|s_method/s_ref − 1|`, which cannot distinguish
shrinking from growing:

| scene | mode | scale ratio | rotation error |
|---|---|---:|---:|
| m3_cor_c E2 | `plane_match` | 0.621 (38% smaller) | 179.6° |
| m3_cor_a E3 | `plane_match` | 1.007 (0.7% off) | 90.3° |
| m3_cor_a E3 | `centroid` | 0.831 | 0.3° |

The two failures differ in scale by a lot, so no single scale-based story covers
both.

## Status

The implementation is complete and the default is verified unchanged. Whether to
adopt `plane_match` is not decided here: on this evidence the binding constraint
is the search reaching the correct pose, not the objective ranking it. The reference
transform outscores everything the method returns.
