"""Remove from the correction-training and pretraining text every sentence that shares a
normalised five-word sequence with any Qur'anic verse, then write gec_train_lf.json and
pretrain_lf.json.  Test, calibration and Qur'anic splits are unchanged.
usage: python3 code/cs/leakfree_data.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import *
from prep_data import norm_match

DIAC = re.compile(r"[ً-ْٰـ]")


def ng(ws, k=5):
    ws = [norm_match(DIAC.sub("", w)) for w in ws]
    return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}


verses = load("quran.json")
Q = set()
for v in verses:
    Q |= ng(v["tokens"])
tr = load("gec_train.json"); pre = load("pretrain.json")
tr_lf = [x for x in tr if not (ng(x["clean"]) & Q)]
pre_lf = [s for s in pre if not (ng(s) & Q)]
json.dump(tr_lf, open(os.path.join(P, "gec_train_lf.json"), "w"), ensure_ascii=False)
json.dump(pre_lf, open(os.path.join(P, "pretrain_lf.json"), "w"), ensure_ascii=False)
stats = {"train": [len(tr), len(tr_lf)], "pretrain": [len(pre), len(pre_lf)], "quran_5grams": len(Q)}
json.dump(stats, open(os.path.join(RES, "leakfree_stats.json"), "w"), indent=1)
print(stats)
