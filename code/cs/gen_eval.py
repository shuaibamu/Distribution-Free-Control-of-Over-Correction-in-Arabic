"""Metrics and edit-level conformal risk control for generative correctors (results/gen/*.json).
Rhetoric scores for the gate come from the RACE seed-1 model.  -> results/gen_eval.csv
usage: python3 code/cs/gen_eval.py [--alpha 0.005]
"""
import os, sys, glob, json, csv, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import analysis as A
import gen_control as G

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, nargs="+", default=[0.003, 0.005, 0.01])
a = ap.parse_args()
_rh = {}


def rhet(split, n):
    if split not in _rh:
        d = A.dump("ours", 1)
        if d is not None and split in d:
            _rh[split] = [p["rhet"] for p in d[split]]
        else:
            import serve_bench as SB
            m = [SB.load_tagger("ours_s1")]
            its = A.items(split); out = []
            for i in range(0, len(its), 128):
                P, R, M = SB.tier_probs(m, its[i:i + 128])
                out += [R[b, :int(M[b].sum())].numpy() for b in range(R.size(0))]
            _rh[split] = out
    return _rh[split][:n]


def gold_of(x):
    from gec_data import apply_tags
    return apply_tags(x["noisy"][:MAXW], x["tags"][:MAXW])


rows = []
for f in sorted(glob.glob(os.path.join(RES, "gen", "*.json"))):
    g = json.load(open(f))
    if "smoke" in g["tag"]:
        continue
    rec0 = {"tag": g["tag"], "seed": g["seed"], "params_M": g["params"] / 1e6, "train_min": g["train_time"] / 60,
            "infer_sent_per_s": g["infer_sent_per_s"], "device": g["device"]}
    S = {}
    for split in [k for k in g if isinstance(g[k], list)]:
        its = A.items(split)[:len(g[split])]
        srcs = [x["noisy"][:MAXW] for x in its]
        hyps = [h["hyp"] for h in g[split]]
        conf = [h["conf"] for h in g[split]]
        R = rhet(split, len(its))
        E = [G.edit_list(s, h, c, r) for s, h, c, r in zip(srcs, hyps, conf, R)]
        golds = srcs if split == "quran" else [gold_of(x) for x in its]
        S[split] = (its, srcs, hyps, E, golds)
    for split, (its, srcs, hyps, E, golds) in S.items():
        rec = dict(rec0, split=split, mode="all-edits", n=len(its))
        if split == "quran":
            rec["Q_all"] = A.quran_ocr(hyps + [x["noisy"] for x in A.items("quran")[len(hyps):]])["all"]
        else:
            e = score_corpus(srcs, hyps, golds); rec.update(F05=e["F05"], P=e["P"], R=e["R"], OCR=e["OCR"])
        rows.append(rec)
    if "cal" not in S:
        continue
    its, srcs, hyps, E, golds = S["cal"]
    for beta in (0.0, A.BETA):
        Lc = G.loss_matrix(srcs, hyps, E, golds, A.TAU_GRID, beta)
        for al in a.alpha:
            tau = G.crc_tau(Lc, A.TAU_GRID, al)
            for split, (its2, s2, h2, E2, g2) in S.items():
                if split == "cal":
                    continue
                out = [G.apply_edits(s, h, e, tau, beta) for s, h, e in zip(s2, h2, E2)]
                risk = float(np.mean([G.false_edit_loss(s, o, gg) for s, o, gg in zip(s2, out, g2)]))
                rec = dict(rec0, split=split, mode=f"crc_a{al}_b{beta}", tau=tau, risk=risk, n=len(its2))
                if split == "quran":
                    rec["Q_all"] = A.quran_ocr(out + [x["noisy"] for x in A.items("quran")[len(out):]])["all"]
                else:
                    e = score_corpus(s2, out, g2); rec.update(F05=e["F05"], P=e["P"], R=e["R"], OCR=e["OCR"])
                rows.append(rec)
    print(g["tag"], g["seed"], flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("tag", "seed", "split", "mode"), k))
with open(os.path.join(RES, "gen_eval.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
print("rows", len(rows))
