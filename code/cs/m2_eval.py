"""Evaluation on human-annotated Arabic GEC data in M2 format (QALB-2014/2015, ZAEBUC), if the
licensed files are available.  The taggers are applied zero-shot (they were trained on synthetic
Classical Arabic errors), and the conformal threshold is recalibrated on the annotated development
file, so the risk-control claim is tested on real errors and real text.

usage: python3 code/cs/m2_eval.py --dev QALB-2014-L1-Dev.m2 --test QALB-2014-L1-Test.m2 --name qalb14
       -> data/processed/gec_real_<name>_{cal,test}.json, results/real_<name>.csv
"""
import os, sys, json, argparse, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import analysis as A
import multidomain as MD


def read_m2(path, annotator=0):
    items, src, edits = [], None, []
    for line in open(path, encoding="utf-8").read().split("\n") + [""]:
        if line.startswith("S "):
            src = line[2:].split(); edits = []
        elif line.startswith("A "):
            f = line[2:].split("|||")
            i, j = map(int, f[0].split())
            if f[1] in ("noop", "UNK") or int(f[-1]) != annotator:
                continue
            edits.append((i, j, f[2].split()))
        elif not line.strip() and src is not None:
            tgt, k = [], 0
            touched = set()
            for i, j, rep in sorted(edits):
                tgt += src[k:i]; tgt += rep; k = max(k, j)
                touched.update(range(i, j) if j > i else [max(i - 1, 0)])
            tgt += src[k:]
            tags = ["KEEP" if t not in touched else "EDIT" for t in range(len(src))]
            items.append({"src": "real", "noisy": src, "clean": tgt, "tags": tags})
            src = None
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    a = ap.parse_args()
    cal, test = read_m2(a.dev), read_m2(a.test)
    for sp, d in (("cal", cal), ("test", test)):
        json.dump(d, open(os.path.join(P, f"gec_real_{a.name}_{sp}.json"), "w"), ensure_ascii=False)
    gold = lambda its: [x["clean"] for x in its]
    rows = []
    for name, s in [("ours", s) for s in A.SEEDS] + [("pt_tagger", s) for s in A.SEEDS]:
        if not os.path.exists(os.path.join(CK, f"{name}_s{s}.pt")):
            continue
        pt = MD.tagger_preds(name, s, test); pc = MD.tagger_preds(name, s, cal)
        src = [x["noisy"][:MAXW] for x in test]
        g = [x["clean"] for x in test]
        h = MD.hyps(test, pt)
        e = score_corpus(src, h, g)
        rows.append({"model": name, "seed": s, "mode": "argmax", "F05": e["F05"], "P": e["P"], "R": e["R"], "OCR": e["OCR"]})
        for al in a.alphas:
            tau = MD.crc_tau(MD.loss_mat(cal, pc, 0.0), al)
            Lt = MD.loss_mat(test, pt, 0.0)
            i = int(np.argmin(np.abs(A.TAU_GRID - tau)))
            e = score_corpus(src, MD.hyps(test, pt, tau, 0.0), g)
            rows.append({"model": name, "seed": s, "mode": f"crc_{al}", "tau": tau, "risk": float(Lt[:, i].mean()),
                         "F05": e["F05"], "P": e["P"], "R": e["R"], "OCR": e["OCR"]})
        print(name, s, flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RES, f"real_{a.name}.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
