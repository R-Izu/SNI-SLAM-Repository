"""R30 §2-2 の材料がどこにあるかを確かめる（構造の確認のみ）。"""

import json
import os

os.chdir("/home/student/rizu/SNI-SLAM")

d = json.load(open("output/Registration/gt_a/room_0/trial_matrices.json"))
print("=== GT-A trial_matrices ===")
print("keys:", list(d.keys()))
tr = d["trials"]
print("n trials:", len(tr))
print("trial[0] keys:", list(tr[0].keys()))
print("methods:", sorted({t["method"] for t in tr}))
print("trial 番号の範囲:", min(t["trial"] for t in tr), "..", max(t["trial"] for t in tr))

print("\n=== realdata_direct_v2 ===")
r = json.load(open("Registration/output/diag/realdata_direct_v2.json"))
print("keys:", list(r.keys()))
res = r["results"]
print("results type:", type(res), "len:", len(res))
if isinstance(res, dict):
    k = sorted(res)[:3]
    print("keys(先頭3):", k)
    v = res[k[0]]
    print("要素:", type(v), (list(v.keys()) if isinstance(v, dict) else len(v)))
    if isinstance(v, list) and v:
        print("  要素[0] keys:", list(v[0].keys()))
else:
    print("results[0] keys:", list(res[0].keys()))

print("\n=== criterion_verdict ===")
c = json.load(open("Registration/output/diag/criterion_verdict.json"))
print("keys:", list(c.keys()))
for k in c:
    v = c[k]
    if isinstance(v, list) and v and isinstance(v[0], dict):
        print("  %s: %d 件 / keys %s" % (k, len(v), list(v[0].keys())))
