"""Generic evaluator for any trained tagger tag (e.g. ours_lf, pt_tagger_hp): argmax and risk-controlled
metrics on all splits, averaged over the seeds present.
usage: python3 code/cs/eval_tags.py --tags ours ours_lf abl_quranclean_lf --out results/cs_eval.csv
"""
import os, sys, argparse, glob, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import *
import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--alpha", type=float, default=A.ALPHA_MAIN)
ap.add_argument("--out", default=os.path.join(RES, "cs_eval.csv"))
a = ap.parse_args()
md = sorted(os.path.basename(f)[4:-5] for f in glob.glob(os.path.join(ROOT, "data", "processed", "gec_md_*_test.json")))
rows = []
for tag in a.tags:
    for s in range(1, 11):
        d = A.dump(tag, s)
        if d is None:
            continue
        rec = {"tag": tag, "seed": s}
        e = A.evaluate("test", A.hyps_tagger("test", d)); rec.update(F05=e["F05"], P=e["P"], R=e["R"], OCR=e["OCR"])
        rec["F05_dev"] = A.evaluate("dev", A.hyps_tagger("dev", d))["F05"]
        rec["F05_xdom"] = A.evaluate("xdom", A.hyps_tagger("xdom", d))["F05"]
        q = A.quran_ocr(A.hyps_tagger("quran", d)); rec.update(Q_all=q["all"], Q_rhet=q["rhet"])
        for k in md:
            if k in d:
                rec["F05_" + k] = A.evaluate(k, A.hyps_tagger(k, d))["F05"]
        beta = A.BETA if "rhet" in d["cal"][0] and float(np.max(d["cal"][0]["rhet"])) > 0 else 0.0
        tau, _ = A.crc_threshold(d, a.alpha, beta)
        rec["tau"] = tau
        rec["F05_crc"] = A.evaluate("test", A.hyps_tagger("test", d, tau=tau, beta=beta))["F05"]
        rec["risk_crc"] = A.realised_risk(d, "test", tau, beta)
        rec["Q_all_crc"] = A.quran_ocr(A.hyps_tagger("quran", d, tau=tau, beta=beta))["all"]
        rec["train_time"] = d.get("train_time", float("nan"))
        rows.append(rec); print(rec, flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("tag", "seed"), k))
with open(a.out, "w") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
