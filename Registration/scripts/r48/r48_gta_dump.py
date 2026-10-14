"""R48 段階1：GT-A の 8 シーンの参照・source・正解・Ω・摂動を書き出す（sni-slam の環境）。**登録はしない。**

- 参照：`gt_a_plan_a/<scene>.yaml`（R45 と同じ、seed 0、200,000 点、構造 5 クラス）→ `refs/GTA_<scene>.npz`
- source：同じ config の source（GT-A の再投影、`points_ply`）→ `inputs_sni/GTA_<scene>__inf.npz`
- 正解 G：R45 の trial_matrices の T_gt。Ω：`build_omega(gt_a/<scene>.yaml, seed=0)`（R45・R46 と同じ）
- 摂動：R45 の plan_a の trial 0〜19 の P が、`r48_common.perturbations()` と一致することを確かめる（一致しなければ止める）
出力：`~/rizu/r48_data/gta.npz`（G・Ω）、上の npz
"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r48"))
os.chdir(REPO)
from regbim import io_utils                                      # noqa: E402
from criterion_verdict import build_omega                        # noqa: E402
import r48_common as C                                           # noqa: E402


def main():
    for d in ("refs", "inputs_sni"):
        os.makedirs(os.path.join(C.D, d), exist_ok=True)
    Ps = C.perturbations(); out = {}
    for s in C.GTA:
        cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % s))
        for meth in ("plan_a", "centroid"):
            tm = json.load(open("output/Registration/r45/%s/%s/trial_matrices.json" % (meth, s)))
            rows = {t["trial"]: t for t in tm["trials"] if t["trial"] >= 0}
            dmax = max(np.abs(np.asarray(rows[k]["P"]) - Ps[k]).max() for k in range(C.N_TRIALS))
            if dmax > 1e-9:
                raise SystemExit("%s %s: R45 の P と r48_common.perturbations() が一致しない（%.3g）" % (s, meth, dmax))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        ref = io_utils.load_reference_cloud(cfg); src = io_utils.load_source_cloud(cfg)
        np.savez_compressed(os.path.join(C.D, "refs", "GTA_%s.npz" % s), points=ref.points.astype(np.float32),
                            labels=ref.labels.astype(np.int8), normals=ref.normals.astype(np.float32))
        np.savez_compressed(os.path.join(C.D, "inputs_sni", "GTA_%s__inf.npz" % s), points=src.points.astype(np.float32),
                            labels=src.labels.astype(np.int8), normals=src.normals.astype(np.float32))
        out["G_" + s] = G; out["omega_" + s] = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s)), seed=0)
        print(s, "ref", len(ref.points), "src", len(src.points), "P match (plan_a, centroid) ok", flush=True)
    np.savez_compressed(os.path.join(C.D, "gta.npz"), **out)


if __name__ == "__main__":
    main()
