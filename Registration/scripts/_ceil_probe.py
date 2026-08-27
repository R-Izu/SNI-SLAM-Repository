"""410 の天井が 3.3%・窓が 0 になる件を実測で確かめる。

面積からの推測（天井 303.1 m2 ≒ 2 x 411の床 146.6 m2 なので 410 を覆っていないのでは）は
**推測でしかない**ので、天井要素を実際にサンプリングして 410 の真上にあるかを数える。

決着すべきこと:
  - `IfcCovering` 16842（z=3.100・階 4FL）は 410 の上を覆っているか
  - 窓9枚は 410 の壁にあるか（410 が外壁に面していない内室なら 0 で正しい）
"""
from __future__ import annotations

import numpy as np
import ifcopenshell
import ifcopenshell.geom

f = ifcopenshell.open("BIM_IFC_Extraction/input/m3-411.ifc")
s = ifcopenshell.geom.settings()
try:
    s.set("use-world-coords", True)
except Exception:
    s.set(s.USE_WORLD_COORDS, True)


def sample(el, n=100000):
    sh = ifcopenshell.geom.create_shape(s, el)
    v = np.asarray(sh.geometry.verts, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(sh.geometry.faces, dtype=np.int64).reshape(-1, 3)
    v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    a = 0.5 * np.linalg.norm(cr, axis=1)
    rng = np.random.default_rng(0)
    fi = rng.choice(len(t), n, p=a / a.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return ((1 - r1)[:, None] * v0[fi] + (r1 * (1 - r2))[:, None] * v1[fi]
            + (r1 * r2)[:, None] * v2[fi])


CELL = 0.1
cells = {}
for sp in f.by_type("IfcSpace"):
    p = sample(sp, 200000)
    k = np.unique(np.floor(p[:, :2] / CELL).astype(np.int64), axis=0)
    cells[sp.id()] = set(map(tuple, k.tolist()))
    print("IfcSpace id=%s Name=%r  床 %.1f m2" % (sp.id(), sp.Name, len(k) * CELL * CELL))

alias = {"411": 413, "410": 166}
print("\n--- 天井 IfcCovering が、どちらの室の真上にあるか（占有セルで判定）---")
for el in f.by_type("IfcCovering"):
    p = sample(el)
    k = np.floor(p[:, :2] / CELL).astype(np.int64)
    tup = [tuple(x) for x in k.tolist()]
    frac = {nm: float(np.mean([t in cells[i] for t in tup])) for nm, i in alias.items()}
    print("  id=%-6s z=%.3f  411の真上 %.1f%% / 410の真上 %.1f%% / どちらでもない %.1f%%"
          % (el.id(), p[:, 2].min(), 100 * frac["411"], 100 * frac["410"],
             100 * (1 - frac["411"] - frac["410"])))

print("\n--- 窓の位置（410 の bbox は x[-13.66,-7.07] y[-0.48,5.77]）---")
for el in f.by_type("IfcWindow"):
    p = sample(el, 5000)
    lo, hi = p.min(axis=0), p.max(axis=0)
    c = 0.5 * (lo + hi)
    tup = tuple(np.floor(c[:2] / CELL).astype(np.int64).tolist())
    where = [nm for nm, i in alias.items() if tup in cells[i]] or ["外周"]
    print("  id=%-6s 中心(%.2f, %.2f, %.2f) z[%.2f,%.2f]  室=%s"
          % (el.id(), c[0], c[1], c[2], lo[2], hi[2], "/".join(where)))
