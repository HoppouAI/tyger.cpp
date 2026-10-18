#!/usr/bin/env python3
"""Compare a tyger-cli --encode dump (raw f32 [T][1024]) against a reference .npy."""
import sys
import numpy as np

ref = np.load(sys.argv[1])
got = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, ref.shape[1])
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
n = min(len(ref), len(got)) - skip
r, g = ref[:n], got[:n]
d = np.abs(r - g)
cos = (r * g).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(g, axis=1) + 1e-12)
print(f"frames ref {len(ref)} got {len(got)} | max abs {d.max():.3e} mean abs {d.mean():.3e} | ref scale {np.abs(r).mean():.3f} | cos min {cos.min():.6f} mean {cos.mean():.6f}")
