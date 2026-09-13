"""measure_spec.json の `instrument` が引用符無しで書かれていたのを直す（1回限りの補助）。

本人が現地で埋めたときに `"instrument": メジャー,` となっており、JSON として読めなかった。
**値は変えない。引用符を足すだけ。**
"""

import io
import json
import re
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "Registration/configs/measure_spec.json"
s = io.open(p, encoding="utf-8").read()


def quote(m):
    return '"instrument": "%s"%s' % (m.group(1).strip(), m.group(2))


fixed = re.sub(r'"instrument":[ \t]*([^"\s\[\{][^,\n]*?)[ \t]*(,|\n)', quote, s)
io.open(p, "w", encoding="utf-8").write(fixed)

d = json.load(io.open(p, encoding="utf-8"))
print("JSON として読める。埋まった測定値：")
for r in d["measurements"]:
    print("  %-18s measured_m=%-8s instrument=%s"
          % (r["id"], r["measured_m"], r["instrument"]))
