#!/usr/bin/env python3
"""Compare tyger-cli --frames-csv/--json output against scripts/reference.py stream output."""
import csv
import json
import sys

import numpy as np

ref = json.load(open(sys.argv[1], encoding="utf-8"))
rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
rf = np.array([[f[0]] + f[6:10] for f in ref["frames"]], dtype=np.float64)  # t, hazard, yield, bc, p_end
gf = np.array([[float(r["t"]), float(r["hazard"]), float(r["yield"]), float(r["bc"]), float(r["p_end"])] for r in rows])
n = min(len(rf), len(gf))
print(f"frames ref {len(rf)} got {len(gf)}, time aligned: {np.allclose(rf[:n, 0], gf[:n, 0])}")
for k, name in enumerate(["hazard", "yield", "bc", "p_end"], 1):
    d = np.abs(rf[:n, k] - gf[:n, k])
    print(f"  {name:7s} max abs {d.max():.4f} mean {d.mean():.5f} p99 {np.percentile(d, 99):.4f}")
if len(sys.argv) > 3:
    got = []
    for line in open(sys.argv[3], encoding="utf-8"):
        line = line.strip()
        if line.startswith("["):
            got += [e for e in json.loads(line) if e["kind"] not in ("FRAME", "PARTIAL")]
    want = ref["events"]
    print(f"events ref {len(want)} got {len(got)}")
    for i in range(max(len(want), len(got))):
        a = want[i] if i < len(want) else None
        b = got[i] if i < len(got) else None
        fa = f"{a['t']:6.2f} {a['kind']:11s} {a.get('transcript', '')}" if a else "-"
        fb = f"{b['t']:6.2f} {b['kind']:11s} {b.get('transcript', '')}" if b else "-"
        mark = " " if a and b and a["kind"] == b["kind"] and abs(a["t"] - b["t"]) < 1e-6 else "*"
        print(f" {mark} ref {fa:60s} | got {fb}")
