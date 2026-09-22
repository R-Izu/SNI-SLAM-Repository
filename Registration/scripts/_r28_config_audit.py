"""R28 §2-3 — **「同じ config」は検証されているか。**

R27 §4 で「4 run の ATE が 10.69〜66.15 cm に散るのは run 間ばらつきである」と
書いたが、**それは run 名から推測したのか、実効設定を突き合わせたのかを書いていなかった。**

**設定が違っていたなら、それは run 間ばらつきではなく設定差である。**
**6.2 倍という数字の意味が、まるごと変わる。**

**⚠ これは R23 §4-1（出力に来歴を持たせる）が無かった時期の run である。**
**辿れない項目は「辿れない」と書く。推測で埋めない。**
"""

import glob
import json
import os
import sys

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
os.chdir("/home/student/rizu/SNI-SLAM")

RUNS = sorted(glob.glob("output/Replica/room0_official/*/"))

print("## 各 run に残っている来歴")
print("%-18s %8s %8s %8s %10s %s"
      % ("run", "ATE", "mesh", "ckpt", "cfg 保存", "その他の手がかり"))
print("-" * 86)
info = {}
for d in RUNS:
    name = os.path.basename(d.rstrip("/"))
    ate_p = os.path.join(d, "eval_ate.json")
    ate = None
    if os.path.exists(ate_p):
        try:
            ate = json.load(open(ate_p))["absolute_translational_error.rmse"]
        except Exception:
            ate = None
    has_mesh = os.path.exists(os.path.join(d, "mesh", "final_mesh_semantic.ply"))
    ck = sorted(glob.glob(os.path.join(d, "ckpts", "*.tar")))
    # config を保存している run があるか（yaml / json）
    cfgs = [p for p in glob.glob(os.path.join(d, "*"))
            if p.endswith((".yaml", ".yml")) or os.path.basename(p) == "config.json"]
    extra = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*"))
                   if os.path.isfile(p))
    info[name] = {"ate": ate, "mesh": has_mesh, "n_ckpt": len(ck),
                  "cfg_saved": [os.path.basename(c) for c in cfgs],
                  "files": extra}
    print("%-18s %8s %8s %8d %10s %s"
          % (name, "—" if ate is None else "%.2f" % ate,
             "あり" if has_mesh else "—", len(ck),
             ",".join(os.path.basename(c) for c in cfgs) or "**無し**",
             ",".join(extra[:4]) or "—"))

print("\n## 判定")
saved = [k for k, v in info.items() if v["cfg_saved"]]
print("  実効 config を保存している run: %s"
      % (", ".join(saved) if saved else "**1 つも無い**"))
print()
if not saved:
    print("  **→ 4 run の実効設定を突き合わせることはできない。**")
    print("  **→ R27 §4 の「同じ config を回し直したときのばらつき」は、**")
    print("     **run 名とディレクトリの慣習から推測したものであり、検証されていない。**")
    print("  **→ 「6.2 倍は run 間ばらつきである」と断定できない。**")
    print("     設定差の可能性を排除できないことを、報告に明記する必要がある。")

print("\n## 現在の config が何を指しているか（**当時のものとは限らない**）")
import yaml
p = "configs/Replica/room0_official.yaml"
c = yaml.safe_load(open(p))
print("  %s" % p)
print("    data.output = %s" % (c.get("data") or {}).get("output"))
print("    inherit_from = %s" % c.get("inherit_from"))
print("  **この config の output は 1 つの run だけを指している。**")
print("  **他の run は、この行を書き換えて回したはずだが、その記録は残っていない。**")
