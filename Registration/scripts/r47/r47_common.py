"""R47 の条件の定義（回す前に固定）。

B-1：シーン 4 × 入力の範囲 {in411 = 411 の床から切る（R46 の E2 の入力）、in410_411 = 410＋411 の床から切る（R46 の E3 の入力）}
     × r {0.5, 3 m} × 参照 {ref411（R46 の E2 の参照）、ref410_411（E3 の参照）}、＋ 切らない × 参照 2 → 10 組 × 4 = 40 条件
     名前：<scene>__<in>__<ref>_r<r>（切らない は in = inall、r = inf）
B-2：シーン 4 × {in410_411 の r = 0.5 m、切らない} = 8 入力 × 向き 0〜350°（10° ごと）、参照 ref410_411
R46 の再利用：入力の範囲と参照が同じ床の組（in411×ref411、in410_411×ref410_411）と 切らない は、R46 の結果をそのまま使う
（同じ入力・同じ手法のコード。R46 の SNI-SLAM 側は c13446d、以後 regbim は変えていない）。新しく回すのは交差の組（16 条件）と B-2
"""
SCENES = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]
IN_E = {"in411": "E2", "in410_411": "E3"}          # 入力の範囲 → R46 の入力（target の E）
REF_E = {"ref411": "E2", "ref410_411": "E3"}       # 参照 → R46 の参照
FLOOR = {"ref411": "M3_411", "ref410_411": "M3_410_411"}
CUTS = ["0p5", "3"]
YAWS = list(range(0, 360, 10))
B2_INPUTS = [(s, c) for s in SCENES for c in ("0p5", "inf")]


def b1_conditions():
    out = []
    for s in SCENES:
        for inn in IN_E:
            for ref in REF_E:
                for c in CUTS:
                    out.append({"name": "%s__%s__%s_r%s" % (s, inn, ref, c), "scene": s, "in": inn, "ref": ref, "cut": c,
                                "cross": IN_E[inn] != REF_E[ref],
                                "input_set": "%s__%s_r%s" % (s, IN_E[inn], c), "ref_target": "%s__%s" % (s, REF_E[ref])})
        for ref in REF_E:
            out.append({"name": "%s__inall__%s_rinf" % (s, ref), "scene": s, "in": "inall", "ref": ref, "cut": "inf", "cross": False,
                        "input_set": "%s__%s_rinf" % (s, REF_E[ref]), "ref_target": "%s__%s" % (s, REF_E[ref])})
    return out
