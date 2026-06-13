# Registration — Semantic SLAM → BIM-surrogate point-cloud alignment

Aligns a SLAM-reconstructed semantic point cloud to a reference point cloud using
**physical constraints from semantic labels** (walls vertical, floor/ceiling
horizontal) plus a **multi-class semantic ICP**. The Replica GT mesh is used here
as a convenience stand-in for the final target (an **IFC → point cloud**); the
pipeline is source-agnostic so swapping in IFC is a loader change only.

## Idea (novelty)
1. **Physical-constraint rotation (6→3 DOF).** Floor/ceiling normals fix gravity
   (roll/pitch); wall normals fix the Manhattan yaw. The *label* makes the plane
   assignment certain (cf. Hübner 2021, which infers it from geometry).
2. **Multi-class semantic ICP.** Correspondences are formed **only within the same
   class**, and only on classes present in *both* clouds — so the reference's lack
   of furniture (like IFC) and the SLAM cloud's background simply drop out.

## Layout
```
regbim/        labels (LabeledCloud), io_utils (adapters), preprocess,
               rotation, scale, semantic_icp, metrics, config
regbim/methods/  base + registry; proposed.py, baseline_open3d.py (committed);
                 experimental/ (gitignored — try variants here)
scripts/       extract_labels, establish_gt, run_registration, benchmark
configs/       replica_room0.yaml  (all tunables)
tests/         pytest (synthetic, deterministic)
```

## Run (conda env `sni-slam`, from repo root)
```bash
python Registration/scripts/extract_labels.py  --config Registration/configs/replica_room0.yaml
python -m pytest Registration/tests/
python Registration/scripts/establish_gt.py    --config Registration/configs/replica_room0.yaml
python Registration/scripts/run_registration.py --config Registration/configs/replica_room0.yaml --method proposed
python Registration/scripts/benchmark.py        --config Registration/configs/replica_room0.yaml --trials 100
```
Outputs (gitignored) land in `Registration/output/`: labelled clouds, the frozen
`T_gt.json`, a GT overlay, and `eval/results.csv` + `comparison.png`.

## Adding a method
Drop a module in `regbim/methods/experimental/` that defines
`@register_method("name")` on a `BaseRegistration` subclass with
`register(self, src, dst, cfg) -> 4x4 Sim3`. It is auto-discovered and benchmarked.
When one wins, promote it to `regbim/methods/proposed.py`; the rest stay local
(gitignored).

## Results so far (Replica room_0, SLAM → structural-only reference, 20 trials)
| method | recovery success | median rot err | direct chamfer | time/run |
|---|---|---|---|---|
| proposed | 0.55 | 0.27° | 0.53 m | ~15 s |
| baseline_open3d | 0.0 | 95° | 0.65 m | ~1.5 s |

(Proposed's misses are mostly translation just over the 0.1 m gate — a tuning
target for experimental variants, not a failure of the rotation novelty, whose
median error is sub-degree.)

Recovery = does the method return to its own alignment after a known Sim3
perturbation. The geometric baseline collapses under large global Sim3; the
label/physical constraints recover it. (Robustness uses each method's own
unperturbed result as reference; `T_gt` is only used for the direct error metric.)
```
