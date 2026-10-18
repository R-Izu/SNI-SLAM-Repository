"""R50 段階1 B：位置合わせの後の床の傾き（sni-slam の環境）。**位置合わせは回さない。回す前に commit する。** 指示書 §4-2。

対象：R49 の完全な方式（1 本ずつの実行、`~/rizu/r49_data/results/<data>/full/`）のうち、大きな失敗でない試行。
変換：source → 参照 = T_est · P。実データは正解 G1（`real_criteria`）も（参照ごとに 1 つ）。

指示書に書かれていない細部（ここで決める）
- BIM の床の点：参照（`load_reference_cloud`、R49 と同じ config）のうち floor のラベルの点
- 床の外形：BIM の床の点の xy を 0.1 m の格子に落とし、1 格子分ふさいで（closing）、内側の穴を埋めた格子の集まり。
  実データの 410・411 の外形は、IFC のキャッシュ `Registration/output/ifc/m3_ifc_410.npz`・`m3_ifc_411.npz` の、
  is_inner かつ floor の点から同じ作り方で作る
- BIM の床の面：BIM の床の点に z = a x + b y + c を最小二乗で当てたもの。法線 (−a, −b, 1) の正規化
- 床の点：変換をかけた source の点のうち、floor のラベル・外形の内側・BIM の床の面から鉛直に（z の差で）0.10 m 以内、をすべて満たすもの
- (B-1) 床の点に z = a x + b y + c を最小二乗で当て、その法線と BIM の床の法線との角 [°]。点の数。残差の RMS（当てた平面からの直交距離）
- (B-2) 外形の xy の最小を 1 m に切り下げた所を起点に、1 m × 1 m の格子に分ける。外形の格子を含む 1 m の格子だけを数える。
  床の点が 200 以上の格子ごとに (B-1) と同じ角。200 未満の格子の数
出力：試行ごとの値 `~/rizu/r50_data/tilt/<data>.json`、まとめ `Registration/output/diag/r50/r50_tilt.json`
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics                            # noqa: E402
from regbim.labels import NAME_TO_ID                            # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial, real_criteria                # noqa: E402
import r49_run as RR                                            # noqa: E402
from r49_score import truth, big                                # noqa: E402

FLOOR = NAME_TO_ID["floor"]; CELL = 0.1; GRID = 1.0; NMIN = 200; BAND = 0.10
OUT = os.path.expanduser("~/rizu/r50_data/tilt")


class Outline:
    def __init__(self, xy):
        self.o = np.floor(xy.min(0) / CELL) * CELL - CELL
        ij = np.floor((xy - self.o) / CELL).astype(int)
        shp = ij.max(0) + 3
        m = np.zeros(shp, bool); m[ij[:, 0], ij[:, 1]] = True
        self.m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=1))

    def inside(self, xy):
        ij = np.floor((xy - self.o) / CELL).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.m.shape[0]) & (ij[:, 1] < self.m.shape[1])
        out = np.zeros(len(xy), bool); out[ok] = self.m[ij[ok, 0], ij[ok, 1]]
        return out

    def cells_xy(self):
        return self.o + (np.argwhere(self.m) + 0.5) * CELL


def fit(P):
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    (a, b, c), *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
    rms = float(np.sqrt(np.mean(((P[:, 2] - A @ np.array([a, b, c])) * n[2]) ** 2)))
    return n, (a, b, c), rms


def ang(n1, n2):
    return float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1.0, 1.0))))


def measure(Q, outline, nb, plane):
    a, b, c = plane
    m = outline.inside(Q[:, :2]) & (np.abs(Q[:, 2] - (a * Q[:, 0] + b * Q[:, 1] + c)) <= BAND)
    F = Q[m]
    r = {"n_points": int(len(F))}
    if len(F) >= 3:
        n, _, rms = fit(F); r.update(angle_deg=ang(n, nb), rms_m=rms)
    o = np.floor(outline.cells_xy().min(0) / GRID) * GRID
    cells = {tuple(x) for x in np.floor((outline.cells_xy() - o) / GRID).astype(int)}
    ij = np.floor((F[:, :2] - o) / GRID).astype(int) if len(F) else np.zeros((0, 2), int)
    grid = []; few = 0
    for cc in sorted(cells):
        sel = (ij[:, 0] == cc[0]) & (ij[:, 1] == cc[1]) if len(F) else np.zeros(0, bool)
        k = int(sel.sum())
        if k >= NMIN:
            n, _, _ = fit(F[sel]); grid.append({"cell": [int(cc[0]), int(cc[1])], "n": k, "angle_deg": ang(n, nb)})
        else:
            few += 1
    r.update(n_cells=len(cells), n_cells_few=few, grid=grid)
    return r


def setup(data, target):
    cfg = RR.config(data, target, "full")
    src = io_utils.load_source_cloud(cfg); ref = io_utils.load_reference_cloud(cfg)
    S = src.points[src.labels == FLOOR]
    B = ref.points[ref.labels == FLOOR]
    nb, plane, _ = fit(B)
    outlines = {"all": Outline(B[:, :2])}
    if data == "real":
        sp = ["411"] if target.endswith("E2") else ["410", "411"]
        for s in sp:
            z = np.load("Registration/output/ifc/m3_ifc_%s.npz" % s)
            outlines[s] = Outline(z["points"][(z["labels"] == FLOOR) & z["is_inner"], :2])
    return S, nb, plane, outlines


def q(v):
    v = [x for x in v if x is not None]
    return None if not v else {"n": len(v), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                               "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}


def main():
    os.makedirs(OUT, exist_ok=True)
    summ = {"provenance": provenance()}
    for data in ("synth", "bim", "real"):
        rows = []
        for target in RR.targets(data):
            Gs, om, _ = truth(data, target)
            S, nb, plane, outlines = setup(data, target)
            res_rows = json.load(open(os.path.join(os.path.expanduser("~/rizu/r49_data/results"), data, "full", target + ".json")))["rows"]
            trans = []
            for k in range(RR.n_trials(data)):
                r = res_rows[str(k)]
                if big(per_trial(r["T_est"], r["P"], Gs["1"], om, None)):
                    continue
                trans.append(("T_est", k, np.asarray(r["T_est"]) @ np.asarray(r["P"])))
            if data == "real":
                trans.append(("G1", None, Gs["1"]))
            for kind, k, T in trans:
                Q = metrics.apply_sim3(T, S)
                rows.append({"target": target, "transform": kind, "trial": k,
                             "by_outline": {name: measure(Q, ol, nb, plane) for name, ol in outlines.items()}})
            print(data, target, "transforms", len(trans), flush=True)
        json.dump({"provenance": provenance(), "rows": rows}, open(os.path.join(OUT, data + ".json"), "w"))
        ent = {}
        for kind in ("T_est", "G1"):
            for name in ("all", "410", "411"):
                rs = [r["by_outline"][name] for r in rows if r["transform"] == kind and name in r["by_outline"]]
                if not rs:
                    continue
                g = [c["angle_deg"] for r in rs for c in r["grid"]]
                ent["%s|%s" % (kind, name)] = {
                    "n_transforms": len(rs), "B1_angle_deg": q([r.get("angle_deg") for r in rs]), "B1_n_points": q([r["n_points"] for r in rs]),
                    "B1_rms_m": q([r.get("rms_m") for r in rs]), "B2_angle_deg": q(g), "B2_n_cells_measured": len(g),
                    "B2_n_cells_total": int(sum(r["n_cells"] for r in rs)), "B2_n_cells_few": int(sum(r["n_cells_few"] for r in rs))}
        summ[data] = ent
        print(data, json.dumps(ent), flush=True)
    os.makedirs("Registration/output/diag/r50", exist_ok=True)
    json.dump(summ, open("Registration/output/diag/r50/r50_tilt.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
