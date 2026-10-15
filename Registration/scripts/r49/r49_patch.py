"""R49：計測のための追加（オプトイン）を semantic_icp.py と proposed.py に入れる。置き換えは完全一致だけで行い、見つからなければ止める"""
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


ICP_REC = '''        if _rec:
            _Ro, _to, _so = decompose_sim3(T); _Rn, _tn, _sn = decompose_sim3(T_new)
            _c = float(np.clip((np.trace(_Rn @ _Ro.T) - 1.0) / 2.0, -1.0, 1.0))
            _deltas.append({"frob": delta, "rot_deg": float(np.degrees(np.arccos(_c))),
                            "trans": float(np.linalg.norm(_tn - _to)), "scale_rel": float(abs(_sn / _so - 1.0))})
            _ncorr.append(int(len(src_corr))); _resid.append(float(np.concatenate(_dl).mean()))
'''

patch(os.path.join(R, "semantic_icp.py"), [
    ("from typing import Dict, Optional\n", "import time\nfrom typing import Dict, Optional\n"),
    ("from .trace import Tracer\n", "from .trace import Tracer\nfrom . import instrument as _ins\n"),
    ("    # Per-class KD-trees over the destination (built once).\n",
     "    _rec = _ins.ACTIVE is not None          # R49：計測（オプトイン。計算には触れない）\n"
     "    if _rec:\n"
     "        _t0 = time.perf_counter()\n"
     "    # Per-class KD-trees over the destination (built once).\n"),
    ("    T = np.array(init_T, dtype=np.float64)\n    R_fixed, _, _ = decompose_sim3(T)\n",
     "    if _rec:\n"
     "        _t_tree = time.perf_counter() - _t0\n"
     "        _deltas, _ncorr, _resid, _stop = [], [], [], \"max\"\n"
     "    T = np.array(init_T, dtype=np.float64)\n    R_fixed, _, _ = decompose_sim3(T)\n"),
    ("        s_list, d_list, w_list = [], [], []\n",
     "        s_list, d_list, w_list = [], [], []\n        _dl = []\n"),
    ("            w_list.append(_tukey_weights(dist[keep], tukey_c))\n        if not s_list:\n            break\n",
     "            w_list.append(_tukey_weights(dist[keep], tukey_c))\n"
     "            if _rec:\n"
     "                _dl.append(dist[keep])\n"
     "        if not s_list:\n"
     "            if _rec:\n"
     "                _stop = \"no_corr\"\n"
     "            break\n"),
    ("        if len(src_corr) < 3 or w.sum() < 1e-9:\n            break\n",
     "        if len(src_corr) < 3 or w.sum() < 1e-9:\n"
     "            if _rec:\n"
     "                _stop = \"few_corr\"\n"
     "            break\n"),
    ("        delta = float(np.linalg.norm(T_new - T))\n",
     "        delta = float(np.linalg.norm(T_new - T))\n" + ICP_REC),
    ("        if delta < conv:\n            break\n",
     "        if delta < conv:\n"
     "            if _rec:\n"
     "                _stop = \"conv\"\n"
     "            break\n"),
    ("        tracer.record(last_it, T, force=True)   # guarantee the final pose is logged\n    return T\n",
     "        tracer.record(last_it, T, force=True)   # guarantee the final pose is logged\n"
     "    if _rec:\n"
     "        _ins.ACTIVE[\"icp\"].append({\"tag\": _ins.TAG, \"rotation_fixed\": bool(rotation_fixed), \"max_iter\": max_iter,\n"
     "                                    \"iters\": last_it, \"stop\": _stop, \"deltas\": _deltas, \"n_corr\": _ncorr,\n"
     "                                    \"resid\": _resid, \"t_tree\": _t_tree, \"t_total\": time.perf_counter() - _t0})\n"
     "    return T\n"),
])
print("semantic_icp patched")

patch(os.path.join(R, "methods", "proposed.py"), [
    ("from .. import plan_correlate as _plan_corr\n",
     "from .. import plan_correlate as _plan_corr\nfrom .. import instrument as _ins     # R49：段の時間（オプトイン）\n"),
    # 案A の種：相関の探索・短い精緻化・短い精緻化の後の採点
    ("    cands, diag = _plan_corr.generate_candidates(\n"
     "        src_p.points, src_p.labels, src_p.normals,\n"
     "        dst_p.points, dst_p.labels, dst_p.normals,\n"
     "        rotations, NAME_TO_ID, pcfg.get(\"plan_correlate\") or {})\n",
     "    with _ins.stage(\"correlate\"):\n"
     "        cands, diag = _plan_corr.generate_candidates(\n"
     "            src_p.points, src_p.labels, src_p.normals,\n"
     "            dst_p.points, dst_p.labels, dst_p.normals,\n"
     "            rotations, NAME_TO_ID, pcfg.get(\"plan_correlate\") or {})\n"),
    ("            T = _sicp(src_p, dst_p, T, c2, rotation_fixed=True)\n"
     "        refined.append({**c, \"T_short\": T,\n"
     "                        \"score_short\": float(class_inlier_ratio(src_p, dst_p, T,\n"
     "                                                                thresh))})\n",
     "            with _ins.stage(\"icp\", tag=\"icp_short\"):\n"
     "                T = _sicp(src_p, dst_p, T, c2, rotation_fixed=True)\n"
     "        with _ins.stage(\"select\"):\n"
     "            _ss = float(class_inlier_ratio(src_p, dst_p, T, thresh))\n"
     "        refined.append({**c, \"T_short\": T, \"score_short\": _ss})\n"),
    ("        src_p = preprocess.prepare(src, cfg)\n        dst_p = preprocess.prepare(dst, cfg)\n",
     "        with _ins.stage(\"prep\"):\n"
     "            src_p = preprocess.prepare(src, cfg)\n"
     "            dst_p = preprocess.prepare(dst, cfg)\n"),
    ("            return semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=False,\n"
     "                                tracer=tracer)\n",
     "            with _ins.stage(\"icp\", tag=\"icp_free\"):\n"
     "                return semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=False,\n"
     "                                    tracer=tracer)\n"),
    ("        axes = canonical_axes(dst_p, cfg)\n",
     "        _ins_rot = _ins.stage(\"rotation\"); _ins_rot.__enter__()\n"
     "        axes = canonical_axes(dst_p, cfg)\n"),
    ("        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(\n            src_p, dst_p, cfg)\n",
     "        candidates: List[np.ndarray] = rotation.relative_rotation_candidates(\n            src_p, dst_p, cfg)\n"
     "        _ins_rot.__exit__(None, None, None)\n"),
    ("            T = semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=True,\n"
     "                             tracer=cand_tracer)\n"
     "            score = class_inlier_ratio(src_score, dst_score, T, thresh)\n",
     "            with _ins.stage(\"icp\", tag=\"icp_normal\"):\n"
     "                T = semantic_icp(src_p, dst_p, init_T, cfg, rotation_fixed=True,\n"
     "                                 tracer=cand_tracer)\n"
     "            with _ins.stage(\"select\"):\n"
     "                score = class_inlier_ratio(src_score, dst_score, T, thresh)\n"),
    ("        refine_T = semantic_icp(src_p, dst_p, refine_T0, cfg, rotation_fixed=True,\n"
     "                                tracer=refine_trace)\n",
     "        with _ins.stage(\"icp\", tag=\"icp_refine\"):\n"
     "            refine_T = semantic_icp(src_p, dst_p, refine_T0, cfg, rotation_fixed=True,\n"
     "                                    tracer=refine_trace)\n"),
    ("        cand = [(_chamfer(plane_T, src_p, dst_p), plane_T, plane_trace),\n"
     "                (_chamfer(refine_T, src_p, dst_p), refine_T, refine_trace)]\n",
     "        with _ins.stage(\"select\"):\n"
     "            cand = [(_chamfer(plane_T, src_p, dst_p), plane_T, plane_trace),\n"
     "                    (_chamfer(refine_T, src_p, dst_p), refine_T, refine_trace)]\n"),
])
print("proposed patched")
