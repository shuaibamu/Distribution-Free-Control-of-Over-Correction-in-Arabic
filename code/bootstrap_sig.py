"""Paired sentence-level bootstrap of F0.5 differences (RACE minus baseline).

For every resample, the same sentence indices are drawn for both systems and for all
seeds; F0.5 is computed per seed from summed tp/fp/fn and averaged over seeds.
This adds the sampling variance of the test sentences that seed-level tests omit.
Writes results/bootstrap_sig.csv.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import analysis as A
from common import edits, RES
from gec_data import apply_tags

B = 2000
RNG = np.random.default_rng(20261114)
SEEDS = [1, 2, 3, 4, 5]
ALPHA = 0.005


def counts(split, hyps):
    its = A.items(split)
    out = np.zeros((len(its), 3))
    for k, (x, h) in enumerate(zip(its, hyps)):
        s = x["noisy"][:A.MAXW]
        g = apply_tags(s, x["tags"][:A.MAXW])
        eh, eg = edits(s, h), edits(s, g)
        out[k] = (len(eh & eg), len(eh - eg), len(eg - eh))
    return out


def f05(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-9), 0)
    return np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0)


def model_counts(m, split, risk=False):
    res = []
    for s in SEEDS:
        d = A.dump(m, s)
        if d is None or split not in d:
            continue
        if m == "seq2seq":
            h = d[split]
        elif risk:
            tau, _ = A.crc_threshold(d, ALPHA, A.BETA if m == "ours" else 0.0, use_rhet=(m == "ours"))
            h = A.hyps_tagger(split, d, tau=tau, beta=A.BETA if m == "ours" else 0.0, use_rhet=(m == "ours"))
        else:
            h = A.hyps_tagger(split, d)
        res.append(counts(split, h))
        A._dump.pop((m, s), None)
    return np.stack(res)  # seeds x n x 3


def boot(cO, cB):
    n = cO.shape[1]
    obs = f05(cO.sum(1)).mean() - f05(cB.sum(1)).mean()
    diffs = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        diffs[b] = f05(cO[:, idx].sum(1)).mean() - f05(cB[:, idx].sum(1)).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, max(p, 1.0 / B)


def main():
    rows = []
    for setting, risk in (("argmax", False), ("crc", True)):
        for split in ("test", "xdom"):
            cO = model_counts("ours", split, risk)
            bases = A.BASE + ([] if risk else ["seq2seq"])
            for m in bases:
                cB = model_counts(m, split, risk)
                obs, lo, hi, p = boot(cO, cB)
                rows.append(dict(setting=setting, split=split, baseline=m, seeds_ours=cO.shape[0],
                                 seeds_base=cB.shape[0], diff=100 * obs, lo=100 * lo, hi=100 * hi, p=p))
                print(rows[-1], flush=True)
    with open(os.path.join(RES, "bootstrap_sig.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
