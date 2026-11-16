"""Reads the stage-1/2 hyperparameter runs, selects for every model the setting with the highest
development-split F0.5, writes results/hp_search.csv and a queue that retrains the selected setting
with seeds 1-5 (tag suffix _hpbest).
usage: python3 code/cs/hp_select.py
"""
import os, sys, glob, re, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import *
import analysis as A

rows = []
for f in sorted(glob.glob(os.path.join(RES, "preds", "*_hp_*_s1.pt"))):
    tag = os.path.basename(f)[:-6]
    model, cfg = tag.split("_hp_")
    d = A.dump(tag, 1)
    rows.append({"model": model, "cfg": cfg, "F05_dev": A.evaluate("dev", A.hyps_tagger("dev", d))["F05"],
                 "train_min": d["train_time"] / 60})
    print(rows[-1], flush=True)
# the default setting (lr 6e-4, dropout 0.1, 3 epochs) is the seed-1 run of the main experiments
for model in sorted({r["model"] for r in rows}):
    d = A.dump(model, 1)
    if d is not None:
        rows.append({"model": model, "cfg": "lr6e-4_d0.1_e3", "F05_dev": A.evaluate("dev", A.hyps_tagger("dev", d))["F05"],
                     "train_min": d["train_time"] / 60})
with open(os.path.join(RES, "hp_search.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
best = {}
for r in rows:
    if r["model"] not in best or r["F05_dev"] > best[r["model"]]["F05_dev"]:
        best[r["model"]] = r
q = []
for m, r in best.items():
    c = r["cfg"]
    env = {"LR": "6e-4", "DROP": "0.1", "LAMBDA": "0.5", "RHW": "8"}; ep = 3
    for k, v in re.findall(r"(lr|d|e|lam|rhw)([0-9.e-]+)", c):
        if k == "lr": env["LR"] = v
        elif k == "d": env["DROP"] = v
        elif k == "e": ep = int(v)
        elif k == "lam": env["LAMBDA"] = v
        elif k == "rhw": env["RHW"] = v
    es = " ".join(f"{k}={v}" for k, v in env.items())
    for s in range(1, 6):
        tag = f"{m}_hpbest_s{s}"
        q.append(f"[ -f results/preds/{tag}.pt ] || env THREADS=1 {es} TAG_SUFFIX=_hpbest python3 code/train.py "
                 f"--model {m} --seed {s} --epochs {ep} > results/logs/{tag}.log 2>&1")
    print("selected", m, c, round(r["F05_dev"] * 100, 2))
open(os.path.join(os.path.dirname(__file__), "queue_hp_final.txt"), "w").write("\n".join(q) + "\n")
