"""R49 段階1 §8-2-1：方式「壁の向きからのヨーの候補なし」（proposed_no_wall_yaw）と、勝った候補の番号の記録を足す。
完全一致の置き換えだけで行い、改行（CRLF）を保つ。"""
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


patch(os.path.join(R, "methods", "proposed.py"), [
    ("        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(\n            src_p, dst_p, cfg)\n"
     "        _ins_rot.__exit__(None, None, None)\n",
     "        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(\n            src_p, dst_p, cfg)\n"
     "        # R49 §8-2-1 アブレーション `yaw_grid_deg`＝**壁の向きからのヨーの候補を外す**（重力は残す）。\n"
     "        # 重力で上を揃えた座標（水平の基底は壁に合わせず `_horizontal_basis` の固定の向き）で、\n"
     "        # 0° から yaw_grid_deg おきの向きを候補にする。相関・短い精緻化・採点・選択は完全な方式と同じ。\n"
     "        if abl.get(\"yaw_grid_deg\"):\n"
     "            def _grav_frame(cloud):\n"
     "                up = rotation.estimate_gravity_axis(cloud, cfg)\n"
     "                e1, e2 = rotation._horizontal_basis(up)\n"
     "                return np.stack([e1, e2, up], axis=0)\n"
     "            Rs_g, Rr_g = _grav_frame(src_p), _grav_frame(dst_p)\n"
     "            candidates = [Rr_g.T @ rotation._rot_z(np.deg2rad(a)) @ Rs_g\n"
     "                          for a in np.arange(0.0, 360.0, float(abl[\"yaw_grid_deg\"]))]\n"
     "        _ins_rot.__exit__(None, None, None)\n"),
    ("            if score > best_score:\n                best_score = score\n",
     "            if score > best_score:\n"
     "                if _ins.ACTIVE is not None:      # R49：勝った候補の番号（通常の精緻化の何回目か）\n"
     "                    _ins.ACTIVE[\"winner_index\"] = len(cand_R) - 1\n"
     "                best_score = score\n"),
])
patch(os.path.join(R, "methods", "ablations.py"), [
    ("@register_method(\"proposed_fixed_scale\")\n",
     "@register_method(\"proposed_no_wall_yaw\")\n"
     "class ProposedNoWallYaw(Proposed):\n"
     "    \"\"\"R49 §8-2-1：壁の向きからのヨーの候補なし（重力は残し、0° から 10° おきの 36 向き）。\"\"\"\n"
     "    name = \"proposed_no_wall_yaw\"\n"
     "\n"
     "    def register(self, src: LabeledCloud, dst: LabeledCloud, cfg: Dict,\n"
     "                 tracer: Optional[Tracer] = None) -> np.ndarray:\n"
     "        return super().register(src, dst, _with_ablation(cfg, yaw_grid_deg=10.0),\n"
     "                                tracer)\n"
     "\n"
     "\n"
     "@register_method(\"proposed_fixed_scale\")\n"),
])
print("patched")
