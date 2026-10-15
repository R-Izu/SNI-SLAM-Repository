"""R49 §1-3・§4：段ごとの時間と ICP の反復の記録（オプトイン。既定では何もしない）。

**手法の計算には触れない。** `start()` を呼んだときだけ、`time.perf_counter()` の差と、
ICP の各反復の変化量・対応点の数・残差を `ACTIVE` に書く。呼ばなければ `ACTIVE` は None のままで、
各所の `if ACTIVE is not None` を通るだけになる。

段の名前（§4）：prep（読み込み後の前処理）、rotation（重力と向きの候補）、correlate（縮尺と位置の相関の探索）、
icp（ICP。KD 木を作る時間を含む）、select（候補の採点と選択）
ICP の呼び出しの tag：icp_short（案A の短い精緻化）、icp_normal（候補ごとの通常の精緻化）、icp_refine（段2 の重心からの精緻化）、
icp_free（no_gravity の自由回転の ICP）
"""

from __future__ import annotations

import time
from contextlib import contextmanager

ACTIVE = None
TAG = None


def start():
    global ACTIVE
    ACTIVE = {"stages": {}, "icp": []}


def stop():
    global ACTIVE
    d, ACTIVE = ACTIVE, None
    return d


@contextmanager
def stage(name, tag=None):
    """段の時間を足し込む。tag を渡すと、その中の ICP の呼び出しに tag を付ける"""
    global TAG
    if ACTIVE is None:
        yield
        return
    prev = TAG
    if tag is not None:
        TAG = tag
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ACTIVE["stages"][name] = ACTIVE["stages"].get(name, 0.0) + (time.perf_counter() - t0)
        TAG = prev
