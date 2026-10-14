"""R48 段階1：Replica のメッシュを三角形と 6 クラスのラベルで書き出す（sni-slam の環境）。raycast（libim の環境）の入力。

ラベルの付け方は `io_utils._load_replica_reference` と同じ（object_id → info_semantic の id_to_label → replica_to_six、既定 background）。
ただし**全クラスを残す**（source は家具も観測する）。四角形は同じ向きで 2 つの三角形に分ける。
出力：`~/rizu/r48_data/mesh/<scene>.npz`（V・F・label）
"""
import json, os, sys
import numpy as np
import yaml
from plyfile import PlyData

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim.labels import build_replica_to_six                  # noqa: E402

SCENES = ["apartment_0", "apartment_1", "apartment_2", "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
          "frl_apartment_3", "frl_apartment_4", "frl_apartment_5", "hotel_0"]
OUT = os.path.expanduser("~/rizu/r48_data/mesh")


def main():
    os.makedirs(OUT, exist_ok=True)
    rep2six = build_replica_to_six(yaml.safe_load(open("Registration/configs/gt_a/room_0.yaml"))["classes"]["replica_to_six"])
    for s in sys.argv[1:] or SCENES:
        if os.path.exists(os.path.join(OUT, s + ".npz")):
            continue
        ply = PlyData.read("data/replica/%s/habitat/mesh_semantic.ply" % s)
        v = ply["vertex"]; V = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        F = np.stack(np.asarray(ply["face"].data["vertex_indices"])).astype(np.int32)
        oid = np.asarray(ply["face"].data["object_id"]).astype(np.int64)
        id2l = np.asarray(json.load(open("data/replica/%s/habitat/info_semantic.json" % s))["id_to_label"], dtype=np.int64)
        rep = np.full(len(F), -1); ok = (oid >= 0) & (oid < len(id2l)); rep[ok] = id2l[oid[ok]]
        lab = np.zeros(len(F), np.int8)
        for rid, six in rep2six.items():
            lab[rep == rid] = six
        tris = np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]]); tl = np.concatenate([lab, lab])
        np.savez_compressed(os.path.join(OUT, s + ".npz"), V=V, F=tris, label=tl)
        print(s, "V", len(V), "tris", len(tris), "labels", np.bincount(tl, minlength=6).tolist(), flush=True)


if __name__ == "__main__":
    main()
