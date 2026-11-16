"""Additional analyses requested in review:
  resplit   - conformal risk control over 1000 random calibration/test resplits per seed
  leak      - Qur'anic verse overlap with training prose; over-correction on seen vs unseen segments
  noninf    - one-sided non-inferiority test of H1 (margin 1 F0.5 point)
  auc       - bootstrap confidence intervals for the rhetoric-head AUCs
  dedup     - exact duplicates of evaluation sentences in the pretraining corpus
usage: python3 code/analysis_extra.py [resplit leak noninf auc dedup]
"""
import os, sys, re, json, csv
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import analysis as A
from analysis import TAGS
from canonical_istiara import CANON

GRID = A.TAU_GRID
B = 1000


def sent_loss_matrix(d, split, beta):
    """n x |grid| matrix of per-sentence false-edit losses."""
    rows = []
    for x, p in zip(A.items(split), d[split]):
        n = len(p["bestpT"])
        keep = np.array([t == "KEEP" for t in x["tags"][:n]])
        v = np.sort((p["bestpT"] - beta * p["rhet"])[keep])
        rows.append((len(v) - np.searchsorted(v, GRID, side="right")) / n)
    return np.array(rows)


def resplit():
    rng = np.random.default_rng(0)
    out = []
    for variant, beta in (("ours", A.BETA), ("no-gate", 0.0)):
        for s in A.SEEDS:
            d = A.dump("ours", s)
            if d is None:
                continue
            L = np.vstack([sent_loss_matrix(d, "cal", beta), sent_loss_matrix(d, "test", beta)])
            n_cal = len(A.items("cal")); N = len(L)
            res = {a: [] for a in A.ALPHAS}
            for _ in range(B):
                perm = rng.permutation(N); c, t = perm[:n_cal], perm[n_cal:]
                Rc = L[c].mean(0); Rt = L[t].mean(0)
                for a in A.ALPHAS:
                    ok = (n_cal / (n_cal + 1)) * Rc + 1 / (n_cal + 1) <= a
                    res[a].append(Rt[np.argmax(ok)] if ok.any() else 0.0)
            for a in A.ALPHAS:
                r = np.array(res[a])
                out.append({"variant": variant, "seed": s, "alpha": a, "mean_risk": float(r.mean()),
                            "q95_risk": float(np.quantile(r, 0.95)), "max_risk": float(r.max()),
                            "viol_freq": float((r > a).mean())})
    A.write_csv("crc_resplit.csv", out)
    for variant in ("ours", "no-gate"):
        rr = [r for r in out if r["variant"] == variant]
        print(variant, "max mean_risk/alpha", max(r["mean_risk"] / r["alpha"] for r in rr),
              "mean viol", np.mean([r["viol_freq"] for r in rr]), "max viol", max(r["viol_freq"] for r in rr))


DIAC = re.compile(r"[ً-ْٰـ]")


def norm(w):
    return DIAC.sub("", w)


def ngrams(ws, k=5):
    ws = [norm(w) for w in ws]
    if len(ws) < k:
        return {tuple(ws)}
    return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}


def leak():
    train = set(); pre = set()
    for x in load("gec_train.json"):
        train |= ngrams(x["clean"])
    for x in load("pretrain.json"):
        pre |= ngrams(x)
    its = A.items("quran")
    seen_tr = np.array([bool(ngrams(x["noisy"]) & train) for x in its])
    seen_any = np.array([bool(ngrams(x["noisy"]) & (train | pre)) for x in its])
    ntok = np.array([len(x["noisy"]) for x in its])
    rows = []
    for s in A.SEEDS:
        d = A.dump("ours", s)
        if d is None:
            continue
        tau, _ = A.crc_threshold(d, A.ALPHA_MAIN, A.BETA)
        for mode, kw in (("argmax", {}), ("crc", dict(tau=tau, beta=A.BETA))):
            hyps = A.hyps_tagger("quran", d, **kw)
            touched_ = np.array([len(touched(x["noisy"], h)) for x, h in zip(its, hyps)])
            for subset, m in (("seen_train", seen_tr), ("unseen_train", ~seen_tr), ("seen_any", seen_any),
                              ("unseen_any", ~seen_any), ("all", np.ones(len(its), bool))):
                rows.append({"seed": s, "mode": mode, "subset": subset, "segments": int(m.sum()),
                             "tokens": int(ntok[m].sum()), "Q_all": float(touched_[m].sum() / ntok[m].sum())})
    A.write_csv("quran_leak.csv", rows)
    print("segments", len(its), "seen in train prose", int(seen_tr.sum()), "seen in train or pretraining", int(seen_any.sum()))
    for mode in ("argmax", "crc"):
        for sub in ("seen_any", "unseen_any"):
            v = [r["Q_all"] for r in rows if r["mode"] == mode and r["subset"] == sub]
            print(mode, sub, round(100 * np.mean(v), 3))


def noninf(margin=0.01):
    ps = {}
    for r in csv.DictReader(open(os.path.join(RES, "main_per_seed.csv"))):
        ps.setdefault(r["model"], []).append(float(r["F05"]))
    rows = []
    for m in A.BASE:
        if m not in ps:
            continue
        a = np.array(ps["ours"]); b = np.array(ps[m])
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        diff = a.mean() - b.mean()
        lo = diff - stats.t.ppf(0.95, df) * se
        p = 1 - stats.t.cdf((diff + margin) / se, df)
        rows.append({"baseline": m, "diff": float(diff), "lower95": float(lo), "p_noninf": float(p), "df": float(df)})
    A.write_csv("noninferiority.csv", rows)
    print(rows)


def auc():
    canon = {(s, a): set(ws) for s, a, ws in CANON}
    its = A.items("quran")
    preds = [A.dump("ours", s)["quran"] for s in A.SEEDS if A.dump("ours", s) is not None]
    y, yc, pr, pc = [], [], [], []
    for i, x in enumerate(its):
        r = np.mean([P[i]["rhet"] for P in preds], 0)
        y.extend(x["rhet"]); pr.extend(r)
        if (x["ref"][0], x["ref"][1]) in canon:
            for w, q in zip(x["noisy"], r):
                yc.append(int(w in canon[(x["ref"][0], x["ref"][1])])); pc.append(q)
    rng = np.random.default_rng(1)
    rows = []
    for nm, yy, pp in (("silver", np.array(y), np.array(pr)), ("canonical", np.array(yc), np.array(pc))):
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(yy), len(yy))
            if len(set(yy[idx])) > 1:
                bs.append(roc_auc_score(yy[idx], pp[idx]))
        rows.append({"set": nm, "auc": float(roc_auc_score(yy, pp)), "lo": float(np.quantile(bs, 0.025)),
                     "hi": float(np.quantile(bs, 0.975)), "pos": int(yy.sum()), "n": len(yy)})
    A.write_csv("rhetoric_auc_ci.csv", rows)
    print(rows)


def dedup():
    pre = {" ".join(x) for x in load("pretrain.json")}
    out = {}
    for s in ("cal", "test", "xdom"):
        out[s] = sum(" ".join(x["clean"]) in pre for x in A.items(s))
    json.dump(out, open(os.path.join(RES, "dedup.json"), "w"))
    print(out)


def count_mats(d, split, beta):
    """Per-sentence token counts over the tau grid: false edits, correct edits, wrong edits, gold edits, n."""
    FE, TP, WE, G, N = [], [], [], [], []
    for x, p in zip(A.items(split), d[split]):
        n = len(p["bestpT"])
        gold = x["tags"][:n]
        keep = np.array([t == "KEEP" for t in gold])
        sc = p["bestpT"] - beta * p["rhet"]
        bt = p["best"]
        corr = np.array([(not k) and (bt is not None and TAGS[int(b)] == g) for k, b, g in
                         zip(keep, bt if bt is not None else np.zeros(n), gold)])
        app = sc[None, :] > A.TAU_GRID[:, None]                      # |grid| x n
        FE.append((app & keep[None, :]).sum(1)); TP.append((app & corr[None, :]).sum(1))
        WE.append((app & ~keep[None, :] & ~corr[None, :]).sum(1)); G.append((~keep).sum()); N.append(n)
    return np.array(FE), np.array(TP), np.array(WE), np.array(G), np.array(N)


def hb_pvalue(R, n, a):
    """Hoeffding-Bentkus p-value for H0: risk > a, losses in [0, 1] (Bates et al.; Angelopoulos et al.)."""
    from scipy.stats import binom
    R = np.minimum(R, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(R > 0, R * np.log(R / a), 0.0) + (1 - R) * np.log((1 - R) / (1 - a))
    p_h = np.exp(-n * h)
    p_b = np.e * binom.cdf(np.ceil(n * R), n, a)
    return np.where(R < a, np.minimum(p_h, p_b), 1.0)


def rcbase(B=200, delta=0.1):
    """Risk-control baselines over random resplits: CRC (ours), naive empirical threshold,
    Learn-then-Test with Hoeffding-Bentkus p-values and fixed-sequence testing (high-probability bound)."""
    rng = np.random.default_rng(3)
    out = []
    for s in A.SEEDS:
        d = A.dump("ours", s)
        if d is None:
            continue
        mats = [np.concatenate([a, b]) for a, b in zip(count_mats(d, "cal", A.BETA), count_mats(d, "test", A.BETA))]
        FE, TP, WE, G, N = mats
        L = FE / N[:, None]
        n_cal = len(A.items("cal")); M = len(L)
        for _ in range(B):
            perm = rng.permutation(M); c, t = perm[:n_cal], perm[n_cal:]
            Rc = L[c].mean(0)
            for a in A.ALPHAS:
                taus = {}
                ok = (n_cal / (n_cal + 1)) * Rc + 1 / (n_cal + 1) <= a
                taus["CRC"] = int(np.argmax(ok)) if ok.any() else len(A.TAU_GRID) - 1
                ok = Rc <= a
                taus["Naive"] = int(np.argmax(ok)) if ok.any() else len(A.TAU_GRID) - 1
                pv = hb_pvalue(Rc, n_cal, a)
                j = len(A.TAU_GRID) - 1                                # fixed-sequence from tau = 1 downwards
                while j - 1 >= 0 and pv[j - 1] <= delta:
                    j -= 1
                taus["LTT"] = j
                for mth, j in taus.items():
                    tp, fp = TP[t, j].sum(), FE[t, j].sum() + WE[t, j].sum()
                    fn = G[t].sum() - tp
                    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                    f = 1.25 * pr * rc / max(0.25 * pr + rc, 1e-9)
                    out.append({"seed": s, "alpha": a, "method": mth, "tau": float(A.TAU_GRID[j]),
                                "risk": float(L[t, j].mean()), "viol": float(L[t, j].mean() > a), "tokF05": float(f)})
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        agg[(r["method"], r["alpha"])].append(r)
    rows = [{"method": m, "alpha": a, "mean_risk": float(np.mean([r["risk"] for r in v])),
             "viol_freq": float(np.mean([r["viol"] for r in v])), "tokF05": float(np.mean([r["tokF05"] for r in v])),
             "tau": float(np.mean([r["tau"] for r in v])), "n": len(v)} for (m, a), v in sorted(agg.items())]
    A.write_csv("riskcontrol_baselines.csv", rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    which = sys.argv[1:] or ["resplit", "leak", "noninf", "auc", "dedup"]
    for w in which:
        globals()[w]()
