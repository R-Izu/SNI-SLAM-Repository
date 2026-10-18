"""R50 §3-1・§3-2：スイッチを 3 つ足す（既定は off）。完全一致の置き換えだけで行い、改行（CRLF）を保つ。
- ablation.icp_single_class：ICP の対応づけ（semantic_icp）だけでクラスをまとめる（single_class と同じまとめ方）
- ablation.score_single_class：最終の採点 ρ（src_score・dst_score）だけでクラスをまとめる（single_class と同じまとめ方）
- reference.geometric_classes：参照のクラスを、法線と +z の角 θ と高さから作る（θ ≤ 20° か ≥ 160° が水平。z が参照の点の
  z の最小と最大の中点より下なら床、上なら天井。70° ≤ θ ≤ 110° が壁。それ以外は background。扉・窓は使わない）
"""
import os, sys

CR, LF = chr(13), chr(10)
R = os.path.expanduser("~/rizu/SNI-SLAM/Registration/regbim")


def patch(path, reps):
    raw = open(path, encoding="utf-8", newline="").read()
    crlf = CR + LF in raw
    s = raw.replace(CR + LF, LF)
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit("not unique/absent in %s: %r" % (path, old[:80]))
        s = s.replace(old, new)
    if crlf:
        s = s.replace(LF, CR + LF)
    open(path, "w", encoding="utf-8", newline="").write(s)


patch(os.path.join(R, "semantic_icp.py"), [
    ("    if (cfg.get(\"ablation\") or {}).get(\"single_class\"):\n",
     "    _abl = cfg.get(\"ablation\") or {}\n"
     "    # R50 §3-1：`icp_single_class` は ICP の対応づけだけでクラスをまとめる（`single_class` と同じまとめ方）\n"
     "    if _abl.get(\"single_class\") or _abl.get(\"icp_single_class\"):\n"),
])
patch(os.path.join(R, "methods", "proposed.py"), [
    ("        if abl.get(\"single_class\"):\n            match_ids = ",
     "        # R50 §3-1：`score_single_class` は最終の採点 ρ だけでクラスをまとめる（`single_class` と同じまとめ方）\n"
     "        if abl.get(\"single_class\") or abl.get(\"score_single_class\"):\n            match_ids = "),
])
patch(os.path.join(R, "io_utils.py"), [
    ("    # Partial-coverage study (R3 T3). Absent in every existing config, and a\n",
     "    # R50 §3-2：参照のクラスを幾何から作る（既定 off）\n"
     "    if spec.get(\"geometric_classes\"):\n"
     "        cloud = geometric_reference_classes(cloud)\n"
     "\n"
     "    # Partial-coverage study (R3 T3). Absent in every existing config, and a\n"),
    ("def load_reference_cloud(cfg: Dict) -> LabeledCloud:\n",
     "def geometric_reference_classes(cloud: LabeledCloud) -> LabeledCloud:\n"
     "    \"\"\"R50 §3-2：参照の点のクラスを、法線と +z の角 θ と高さだけから作り直す。\n"
     "\n"
     "    θ ≤ 20° または θ ≥ 160° を水平とし、z が参照の点の z の最小と最大の中点より下なら床、上なら天井。\n"
     "    70° ≤ θ ≤ 110° を壁。それ以外は background。扉・窓のクラスは使わない。元のクラスは meta に残す。\n"
     "    \"\"\"\n"
     "    if cloud.normals is None:\n"
     "        raise ValueError(\"reference.geometric_classes needs reference normals\")\n"
     "    n = np.asarray(cloud.normals, dtype=np.float64)\n"
     "    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)\n"
     "    theta = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))\n"
     "    z = np.asarray(cloud.points)[:, 2]\n"
     "    zmid = 0.5 * (float(z.min()) + float(z.max()))\n"
     "    horiz = (theta <= 20.0) | (theta >= 160.0)\n"
     "    lab = np.zeros(len(z), dtype=np.asarray(cloud.labels).dtype)\n"
     "    lab[horiz & (z < zmid)] = NAME_TO_ID[\"floor\"]\n"
     "    lab[horiz & (z >= zmid)] = NAME_TO_ID[\"ceiling\"]\n"
     "    lab[(theta >= 70.0) & (theta <= 110.0)] = NAME_TO_ID[\"wall\"]\n"
     "    return LabeledCloud(points=cloud.points, labels=lab, normals=cloud.normals,\n"
     "                        meta=dict(cloud.meta, geometric_classes=True, original_labels=np.asarray(cloud.labels).copy()))\n"
     "\n"
     "\n"
     "def load_reference_cloud(cfg: Dict) -> LabeledCloud:\n"),
])
print("patched")
