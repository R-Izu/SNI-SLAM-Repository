"""R46：STSL の前処理と登録を、入力の集合ごとに回す（libim の環境で回す）。**STSL のコードは変えない。**

入力：`$ROOT/inputs/<set>/NNNNNN.npy`（点）と `spec.json`（各試行の T_true：入力 → BIM 座標。正解の無い場合は省略可）
設定（§8-3）：A = 公式 `2f_office_02`（30 m）、B = 公式 `1f_office_03`（30 m）。平面分割は `configs/pointsam/submap3d/30m/...`、
登録と 2D 化・線分は `configs/interval/30m/...`。**変えるのは seq と floor（データの置き場所）だけ**。
乱数（§8-5）：平面分割は seed を固定できないので、K 回（既定 5）回す。k ごとに別の seq にする。

前処理（公式の `make_benchmarks` の 'full' の経路と同じ処理を、1 つの点群 = 1 つの submap として行う）
1. 点を `$ROOT/<seq>/submap3d/30m/points/NNNNNN.pcd` に書く
2. 平面分割（`r46_plane_seg`：公式の `plane_seg.cpp` の process() の写し）→ walls_points・ground_points
3. 公式の Python：`PCD2dManager(...).generate_pcd2d_list(num=1, start=i)` → occupied = og_walls_cropped（z=0 に落として 0.05 m で間引く）、
   free = og_ground（0.3 m で間引き、z < 0.01）、線分 = `LineSegmentManager(cfg).submap_image_based_extract(pcd2d)`
4. 正解：T_gt = Shift(−z_floor) · T_true · T_rp^{-1} を (x, y, yaw) にする（公式の `generate_gt` と同じ形。ICP の精緻化はしない）
5. 登録：`r46_reg_dump`（公式の reg_bm の benchmark と同じ処理）→ `$ROOT/results/<set>/<setting>_k<k>_<floor>.jsonl`
   T_rp は `$ROOT/processed/<seq>/30m/T_rp.json` に残す（変換を BIM 座標に戻すため：T_bim = Shift(+z_floor) · tf · T_rp）

使い方：python r46_stsl_pipeline.py <set> <setting A|B> <K> <floor> [<floor> ...]
"""
import copy, json, os, shutil, subprocess, sys
import numpy as np
import yaml

STSL = os.path.expanduser("~/rizu/LiDAR2BIM-Registration")
BUILD = os.path.expanduser("~/rizu/r46_stsl_build")
ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data")).rstrip("/") + "/"
os.environ["LiBIM_UST_ROOT"] = ROOT
sys.path.insert(0, STSL); os.chdir(STSL)
import open3d as o3d                                                          # noqa: E402
from preprocess.config import load_cfg                                       # noqa: E402
from preprocess.datasets.pcd2d_manager import PCD2dManager                   # noqa: E402
from preprocess.datasets.lineseg_manager import LineSegmentManager           # noqa: E402
from preprocess.datasets.dir import DirManager                               # noqa: E402
from preprocess.utils import se32xyyaw                                       # noqa: E402

SETTINGS = {"A": ("configs/pointsam/submap3d/30m/2f_office_02.yaml", "configs/interval/30m/2F/2f_office_02.yaml"),
            "B": ("configs/pointsam/submap3d/30m/1F/1f_office_03.yaml", "configs/interval/30m/1F/1f_office_03.yaml")}


def write_yaml(src, dst, edits):
    y = yaml.safe_load(open(os.path.join(STSL, src)))
    for k, v in edits.items():
        cur = y
        for kk in k.split(".")[:-1]:
            cur = cur.setdefault(kk, {})
        cur[k.split(".")[-1]] = v
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    yaml.safe_dump(y, open(dst, "w"), sort_keys=False)
    return dst


def prep(setname, setting, k):
    seq = "%s_%s_k%d" % (setname, setting, k)
    inp = os.path.join(ROOT, "inputs", setname)
    files = sorted(f for f in os.listdir(inp) if f.endswith(".npy"))
    sub = os.path.join(ROOT, seq, "submap3d", "30m")
    for d in (os.path.join(ROOT, seq), os.path.join(ROOT, "processed", seq)):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(sub, "points"))
    for f in files:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.load(os.path.join(inp, f)).astype(np.float64)))
        o3d.io.write_point_cloud(os.path.join(sub, "points", f.replace(".npy", ".pcd")), pc)
    ps_cfg = write_yaml(SETTINGS[setting][0], os.path.join(ROOT, "cfg", seq + "_pointsam.yaml"),
                        {"benchmarks.seq": seq, "benchmarks.interval": "30m", "benchmarks.maptype": "submap3d"})
    r = subprocess.run([os.path.join(BUILD, "r46_plane_seg"), ps_cfg], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    # 公式の Python の前処理（cfg は EasyDict の大域変数なので、毎回読み直す）
    iv_cfg = write_yaml(SETTINGS[setting][1], os.path.join(ROOT, "cfg", seq + "_interval_prep.yaml"), {"seq": seq})
    cfg = load_cfg(iv_cfg)
    dirs = DirManager(cfg)
    for d in (dirs.occupied_pcd_flatten_dir, dirs.free_pcd_dir, dirs.lineseg_dir):
        os.makedirs(d, exist_ok=True)
    spec_p = os.path.join(inp, "spec.json")
    spec = json.load(open(spec_p)) if os.path.exists(spec_p) else None
    xyyaw, trp = [], {}
    for i, f in enumerate(files):
        pcd2d = PCD2dManager(dirs.submap3d_wall_pcd_dir, dirs.submap3d_ground_pcd_dir, cfg).generate_pcd2d_list(num=1, start=i)
        if not pcd2d:
            raise RuntimeError("pcd2d failed for %s %d" % (seq, i))
        pcd2d = pcd2d[0]
        occ = pcd2d.og_walls_cropped
        pts = np.asarray(occ.points).copy(); pts[:, 2] = 0
        flat = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.05)
        free = pcd2d.og_ground.voxel_down_sample(voxel_size=0.3)
        fp = np.asarray(free.points); free = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fp[fp[:, 2] < 0.01]))
        lines = LineSegmentManager(cfg).submap_image_based_extract(pcd2d)
        o3d.io.write_point_cloud(os.path.join(dirs.occupied_pcd_flatten_dir, "%06d.pcd" % i), flat)
        o3d.io.write_point_cloud(os.path.join(dirs.free_pcd_dir, "%06d.pcd" % i), free)
        lines.save_to_file(os.path.join(dirs.lineseg_dir, "%06d.txt" % i))
        T_rp = np.asarray(pcd2d.T_rp, dtype=np.float64); trp[i] = T_rp.tolist()
        if spec is not None:
            T_true = np.asarray(spec["trials"][i]["T_true"], dtype=np.float64)
            xyyaw.append(T_true)          # z_floor は floor ごとに違いうるので、ここでは T_true を残し、下で floor ごとに書く
    json.dump(trp, open(os.path.join(dirs.seq_dir, "T_rp.json"), "w"))
    return seq, dirs.seq_dir, xyyaw, trp


def register(setname, setting, k, seq, seq_dir, T_trues, trp, floor):
    bim_meta = json.load(open(os.path.join(ROOT, "BIM", floor, "meta.json")))
    zf = float(bim_meta["z_floor"])
    floor_s = "%s_%s" % (floor, setting)                 # ハッシュ表は設定ごとに作られるので、BIM の置き場所を設定ごとに分ける
    dst = os.path.join(ROOT, "BIM", floor_s)
    if not os.path.exists(dst):
        os.makedirs(dst)
        for f in ("bimLineSeg.txt", "bim.pcd", "bim2d.pcd", "meta.json"):
            shutil.copy(os.path.join(ROOT, "BIM", floor, f), dst)
    S = np.eye(4); S[2, 3] = -zf
    with open(os.path.join(seq_dir, "pose.txt"), "w") as f:
        f.write("x y yaw\n")
        for i in range(len(trp)):
            if T_trues:
                x, y, yaw = se32xyyaw(S @ T_trues[i] @ np.linalg.inv(np.asarray(trp[i])), degrees=True)
            else:
                x, y, yaw = 0.0, 0.0, 0.0                      # 正解が無い入力（成功判定は使わない）
            f.write("%r %r %r\n" % (float(x), float(y), float(yaw)))
    # 登録の config は STSL の configs/interval/ の下に置く（InitGLOG がその位置を前提にしている。STSL の .git/info/exclude で追跡から外す）
    rc = write_yaml(SETTINGS[setting][1], os.path.join(STSL, "configs", "interval", "r46", "%s_%s.yaml" % (seq, floor)),
                    {"seq": seq, "floor": floor_s})
    out = os.path.join(ROOT, "results", setname); os.makedirs(out, exist_ok=True)
    out = os.path.join(out, "%s_k%d_%s.jsonl" % (setting, k, floor))
    r = subprocess.run([os.path.join(BUILD, "r46_reg_dump"), rc, out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return out


def main():
    setname, setting, K, floors = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    for k in range(1, K + 1):
        seq, seq_dir, T_trues, trp = prep(setname, setting, k)
        for fl in floors:
            print("done", register(setname, setting, k, seq, seq_dir, T_trues, trp, fl), flush=True)


if __name__ == "__main__":
    main()
