"""Copies results/preds/*.pt into a smaller form for transfer: keeps the fields used by the analysis
(argmaxT, bestpT, best, rhet per sentence) for the cal/test/xdom/quran/held-out splits.
usage: python3 code/cs/slim_preds.py <out_dir>
"""
import os, sys, glob
import numpy as np, torch
out = sys.argv[1]; os.makedirs(os.path.join(out, "preds"), exist_ok=True)
KEEP = ("argmaxT", "bestpT", "best", "rhet")
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "results", "preds", "*.pt"))):
    d = torch.load(f, weights_only=False)
    s = {k: v for k, v in d.items() if not isinstance(v, list)}
    for k, v in d.items():
        if isinstance(v, list) and (k in ("dev", "cal", "test", "xdom", "quran") or k.startswith("md_")):
            s[k] = [{kk: (np.asarray(p[kk]).astype(np.float32) if kk in ("bestpT", "rhet") else np.asarray(p[kk]).astype(np.int32))
                     for kk in KEEP if kk in p} for p in v]
    torch.save(s, os.path.join(out, "preds", os.path.basename(f)))
print("slim preds written to", out)
