"""Evaluation on public human-annotated data (QALB-2014 test; ZAEBUC test as a shifted domain).

Run from the QALB root created by code/qalb_prep.py (its code/ directory links to this one), after
training with code/train.py in that root. Gold corrections are the human corrections of each chunk
(x["clean"]), so edits that no tag can express count as misses.

Outputs (results/ of the QALB root):
  real_main.csv     per model and seed: argmax P, R, F0.5 on QALB-2014 test and ZAEBUC test,
                    Qur'anic over-correction, oracle F0.5 of the tag vocabulary
  real_crc.csv      RACE and pretrained tagger under conformal risk control, per alpha: threshold,
                    realised false-edit rate and F0.5 on QALB-2014 test, ZAEBUC with the QALB
                    threshold, ZAEBUC after recalibration on ZAEBUC dev
  real_resplit.csv  1,000 random calibration/test resplits of QALB (calibration half of dev + test)
  real_boot.csv     paired bootstrap of F0.5 (RACE minus pretrained tagger), argmax and conformal
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from gec_data import apply_tags
import analysis as A

MODELS = ["ours", "pt_tagger"]
SEEDS = [1, 2, 3]
ALPHAS = [0.005, 0.01, 0.02, 0.05]
RNG = np.random.default_rng(20261116)


def gold(split):
    return [x["clean"] for x in A.items(split)]


def src(split):
    return [x["noisy"][:MAXW] for x in A.items(split)]


def ev(split, hyps):
    its = A.items(split)
    g = [x["clean"] if len(x["noisy"]) <= MAXW else apply_tags(x["noisy"][:MAXW], x["tags"][:MAXW]) for x in its]
    return score_corpus(src(split), hyps, g)


def lossmat(split, preds, beta, use_rhet):
    its = A.items(split)
    return np.array([[A.token_risk(p, x["tags"], t, beta, use_rhet) for t in A.TAU_GRID] for x, p in zip(its, preds)])


def crc(L, alpha):
    n = len(L)
    ok = (n / (n + 1)) * L.mean(0) + 1 / (n + 1) <= alpha
    return int(np.argmax(ok)) if ok.any() else len(A.TAU_GRID) - 1


def counts(split, hyps):
    its = A.items(split); out = np.zeros((len(its), 3))
    g = [x["clean"] if len(x["noisy"]) <= MAXW else apply_tags(x["noisy"][:MAXW], x["tags"][:MAXW]) for x in its]
    for k, (s, h, gg) in enumerate(zip(src(split), hyps, g)):
        eh, eg = edits(s, h), edits(s, gg)
        out[k] = (len(eh & eg), len(eh - eg), len(eg - eh))
    return out


def f05(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-9), 0)
    return np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0)


def main():
    main_rows, crc_rows, boot_store = [], [], {}
    tagset = set(A.TAGS)
    oracle = {}
    for sp in ("test", "xdom"):
        hy = [apply_tags(x["noisy"][:MAXW], [t if t in tagset else "KEEP" for t in x["tags"][:MAXW]]) for x in A.items(sp)]
        oracle[sp] = ev(sp, hy)["F05"]
    for m in MODELS:
        ur = (m == "ours"); beta = A.BETA if ur else 0.0
        for s in SEEDS:
            d = A.dump(m, s)
            if d is None:
                continue
            ht = A.hyps_tagger("test", d); hx = A.hyps_tagger("xdom", d); hq = A.hyps_tagger("quran", d)
            et, ex = ev("test", ht), ev("xdom", hx); q = A.quran_ocr(hq)
            main_rows.append({"model": m, "seed": s, "P": 100 * et["P"], "R": 100 * et["R"], "F05": 100 * et["F05"], "OCR": 100 * et["OCR"],
                              "F05_zaebuc": 100 * ex["F05"], "P_zaebuc": 100 * ex["P"], "R_zaebuc": 100 * ex["R"], "Q_all": 100 * q["all"],
                              "T": d["T"], "oracle_test": 100 * oracle["test"], "oracle_zaebuc": 100 * oracle["xdom"],
                              "train_time": d["train_time"], "params": d["params"]})
            boot_store[(m, s, "argmax")] = counts("test", ht)
            Lc = lossmat("cal", d["cal"], beta, ur); Lt = lossmat("test", d["test"], beta, ur)
            Lx = lossmat("xdom", d["xdom"], beta, ur); Lz = lossmat("md_ZAEBUC_cal", d["md_ZAEBUC_cal"], beta, ur)
            Lq = lossmat("quran", d["quran"], beta, ur)
            for a in ALPHAS:
                i = crc(Lc, a); tau = float(A.TAU_GRID[i]); iz = crc(Lz, a); tz = float(A.TAU_GRID[iz])
                h_t = A.hyps_tagger("test", d, tau=tau, beta=beta, use_rhet=ur)
                h_x = A.hyps_tagger("xdom", d, tau=tau, beta=beta, use_rhet=ur)
                h_z = A.hyps_tagger("xdom", d, tau=tz, beta=beta, use_rhet=ur)
                crc_rows.append({"model": m, "seed": s, "alpha": a, "tau": tau, "risk_test": 100 * Lt[:, i].mean(),
                                 "F05_test": 100 * ev("test", h_t)["F05"], "risk_zaebuc_in": 100 * Lx[:, i].mean(),
                                 "F05_zaebuc_in": 100 * ev("xdom", h_x)["F05"], "tau_rec": tz,
                                 "risk_zaebuc_rec": 100 * Lx[:, iz].mean(), "F05_zaebuc_rec": 100 * ev("xdom", h_z)["F05"],
                                 "risk_quran": 100 * Lq[:, i].mean()})
                if a == 0.005:
                    boot_store[(m, s, "crc")] = counts("test", h_t)
                    if m == "ours" and s == 1:
                        # resplits of calibration + test
                        Lall = np.vstack([Lc, Lt]); n = len(Lc); rr = []
                        for b in range(1000):
                            idx = RNG.permutation(len(Lall)); ci, ti = idx[:n], idx[n:]
                            j = crc(Lall[ci], a); rr.append(Lall[ti, j].mean())
                        rr = np.array(rr)
                        with open(os.path.join(RES, "real_resplit.csv"), "w") as f:
                            f.write("alpha,n_cal,n_test,mean_risk,sd_risk,share_above_alpha\n")
                            f.write(f"{a},{n},{len(Lall)-n},{100*rr.mean():.4f},{100*rr.std():.4f},{(rr > a).mean():.4f}\n")
            print(m, s, round(main_rows[-1]["F05"], 2), flush=True)
            A._dump.pop((m, s), None)
    A.write_csv("real_main.csv", main_rows); A.write_csv("real_crc.csv", crc_rows)
    rows = []
    for mode in ("argmax", "crc"):
        cO = np.stack([boot_store[k] for k in sorted(boot_store) if k[0] == "ours" and k[2] == mode])
        cB = np.stack([boot_store[k] for k in sorted(boot_store) if k[0] == "pt_tagger" and k[2] == mode])
        n = cO.shape[1]; obs = f05(cO.sum(1)).mean() - f05(cB.sum(1)).mean(); diffs = []
        for b in range(2000):
            idx = RNG.integers(0, n, n)
            diffs.append(f05(cO[:, idx].sum(1)).mean() - f05(cB[:, idx].sum(1)).mean())
        diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / 2000)
        rows.append({"mode": mode, "diff": 100 * obs, "lo": 100 * lo, "hi": 100 * hi, "p": p})
    A.write_csv("real_boot.csv", rows)
    print(rows)


if __name__ == "__main__":
    main()
