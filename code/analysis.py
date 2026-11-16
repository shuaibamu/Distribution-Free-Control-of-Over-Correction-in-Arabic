"""All quantitative analyses from saved prediction dumps -> results/*.csv.

Decision rule for taggers (calibrated, rhetoric-gated):
    apply edit at token i  iff  p(best non-KEEP tag)_i > tau + beta * r_i
r_i is the rhetoric-head probability (0 for models without the head).
"""
import os, sys, glob, json, math, itertools, collections
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from gec_data import apply_tags
from train import quran_items
from canonical_istiara import CANON

V = Vocab()
TAGS = V.tags
SEEDS = [1, 2, 3, 4, 5]
BETA = 0.5
BASE = ["bilstm", "trans_scratch", "pt_tagger", "stack_hybrid"]
ABL = ["abl_nopre", "abl_nornn", "abl_concat", "abl_lstm", "abl_nodet", "abl_norhet", "abl_quranclean"]
NAMES = {"ours": "RACE (ours)", "bilstm": "BiLSTM tagger", "trans_scratch": "Transformer tagger",
         "pt_tagger": "PT-tagger (GECToR-style)", "stack_hybrid": "PT-BiLSTM stack", "seq2seq": "Seq2seq+copy",
         "abl_nopre": "w/o pretraining", "abl_nornn": "w/o BiGRU branch", "abl_concat": "concat fusion",
         "abl_lstm": "BiLSTM branch", "abl_nodet": "w/o detection head", "abl_norhet": "w/o rhetoric head",
         "abl_quranclean": "clean-Qur'an exposure"}

_items = {}


def items(split):
    if split not in _items:
        if split in ("quran", "quran_dev"):
            verses = load("quran.json"); sp = quran_splits(verses)
            _items[split] = quran_items(verses, "test" if split == "quran" else "dev", sp)
        else:
            _items[split] = load(f"gec_{split}.json")
    return _items[split]


_dump = {}


def dump(model, seed):
    k = (model, seed)
    if k not in _dump:
        f = os.path.join(RES, "preds", f"{model}_s{seed}.pt")
        _dump[k] = torch.load(f, weights_only=False) if os.path.exists(f) else None
    return _dump[k]


def tags_for(pred, tau=None, beta=0.0, cal=True, use_rhet=True):
    """Return predicted tag strings for one sentence."""
    if tau is None:  # plain argmax decoding
        am = pred["argmaxT"]
        return [TAGS[i] for i in am]
    sc = pred["bestpT"] if cal else pred["bestp1"]
    thr = tau + (beta * pred["rhet"] if use_rhet else 0.0)
    app = sc > thr
    return [TAGS[b] if a else "KEEP" for a, b in zip(app, pred["best"])]


def hyps_tagger(split, d, **kw):
    its = items(split)
    return [apply_tags(x["noisy"][:MAXW], tags_for(p, **kw)) for x, p in zip(its, d[split])]


def evaluate(split, hyps):
    its = items(split)
    src = [x["noisy"][:MAXW] for x in its]
    gold = [apply_tags(x["noisy"][:MAXW], x["tags"][:MAXW]) for x in its]
    return score_corpus(src, hyps, gold)


def quran_ocr(hyps, split="quran"):
    """OCR on error-free Qur'anic text, split by rhetorical / isti'ara / other tokens."""
    its = items(split)
    tot = collections.Counter(); bad = collections.Counter()
    for x, h in zip(its, hyps):
        t = touched(x["noisy"], h)
        for k in range(len(x["noisy"])):
            cls = ["all"]
            cls.append("rhet" if x["rhet"][k] else "plain")
            if x["ist"][k]:
                cls.append("ist")
            for c in cls:
                tot[c] += 1; bad[c] += k in t
    return {c: bad[c] / tot[c] for c in tot}


def token_risk(pred, gold_tags, tau, beta, use_rhet=True):
    sc = pred["bestpT"]; thr = tau + (beta * pred["rhet"] if use_rhet else 0.0)
    app = sc > thr
    g = np.array([t == "KEEP" for t in gold_tags[:len(sc)]])
    return float((app & g).sum()) / len(sc)


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / math.sqrt(len(xs))


def write_csv(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(RES, name), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")


# ------------------------------------------------------------------ CRC
TAU_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 3)  # tau = 1 means no edits, so L(1) = 0
ALPHAS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.010, 0.015]
ALPHA_MAIN = 0.005


def crc_threshold(d, alpha, beta, use_rhet=True, split="cal"):
    its = items(split)
    n = len(its)
    L = np.array([[token_risk(p, x["tags"], t, beta, use_rhet) for t in TAU_GRID]
                  for x, p in zip(its, d[split])])          # n x |grid|
    Rhat = L.mean(0)
    ok = (n / (n + 1)) * Rhat + 1.0 / (n + 1) <= alpha
    idx = np.argmax(ok) if ok.any() else len(TAU_GRID) - 1
    return float(TAU_GRID[idx]), Rhat


def realised_risk(d, split, tau, beta, use_rhet=True):
    its = items(split)
    return float(np.mean([token_risk(p, x["tags"], tau, beta, use_rhet) for x, p in zip(its, d[split])]))


def main_results():
    rows = []; per_seed = collections.defaultdict(list)
    models = ["ours"] + BASE + ["seq2seq"]
    for m in models:
        for s in SEEDS:
            d = dump(m, s)
            if d is None:
                continue
            rec = {"model": m, "seed": s}
            if m == "seq2seq":
                ht = d["test"]; hx = d["xdom"]; hq = d["quran"]
            else:
                ht = hyps_tagger("test", d); hx = hyps_tagger("xdom", d); hq = hyps_tagger("quran", d)
            r = evaluate("test", ht)
            rec.update({k: r[k] for k in ("P", "R", "F05", "OCR", "EM")})
            rec["F05_xdom"] = evaluate("xdom", hx)["F05"]
            q = quran_ocr(hq)
            rec.update({"Q_all": q["all"], "Q_rhet": q["rhet"], "Q_plain": q["plain"], "Q_ist": q.get("ist", 0.0)})
            if m != "seq2seq":
                conf1, confT, corr = [], [], []
                for x, p in zip(items("test"), d["test"]):
                    g = [V.ttoi.get(t, -1) for t in x["tags"][:len(p["pk1"])]]
                    am = p["argmaxT"]
                    m_ = am != 0   # edit-level calibration: proposed edits only
                    conf1.extend(p["bestp1"][m_]); confT.extend(p["bestpT"][m_]); corr.extend((am == np.array(g))[m_])
                rec["ECE_uncal"] = ece(conf1, corr); rec["ECE_cal"] = ece(confT, corr); rec["T"] = d["T"]
            else:
                rec["ECE_uncal"] = float("nan"); rec["ECE_cal"] = float("nan"); rec["T"] = float("nan")
            rec["params"] = d["params"]; rec["train_time"] = d["train_time"]
            rows.append(rec); per_seed[m].append(rec)
            if m != "ours":
                _dump.pop((m, s), None)          # keep memory bounded
    write_csv("main_per_seed.csv", rows)
    agg = []
    for m in models:
        rs = per_seed[m]
        if not rs:
            continue
        a = {"model": m, "name": NAMES[m], "n": len(rs)}
        for k in ("P", "R", "F05", "OCR", "EM", "F05_xdom", "Q_all", "Q_rhet", "Q_plain", "Q_ist", "ECE_uncal", "ECE_cal"):
            v = [r[k] for r in rs]
            a[k] = float(np.mean(v)); a[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            a[k + "_ci"] = ci95(v)
        a["params"] = rs[0]["params"]; a["train_time"] = float(np.mean([r["train_time"] for r in rs]))
        agg.append(a)
    write_csv("main_agg.csv", agg)
    return per_seed


def significance(per_seed):
    rows = []
    ours = per_seed["ours"]
    pvals = []
    for m in BASE + ["seq2seq"]:
        for k in ("F05", "F05_xdom", "Q_all"):
            a = np.array([r[k] for r in ours]); b = np.array([r[k] for r in per_seed[m]])
            if len(a) < 2 or len(b) < 2:
                continue
            sp = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            if sp == 0:   # identical constant samples: no evidence of a difference
                rows.append({"baseline": m, "metric": k, "ours": a.mean(), "base": b.mean(),
                             "t": 0.0, "p": 1.0, "cohen_d": 0.0}); continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({"baseline": m, "metric": k, "ours": a.mean(), "base": b.mean(),
                         "t": float(t), "p": float(p), "cohen_d": float((a.mean() - b.mean()) / sp)})
    # Holm correction within each metric
    for k in ("F05", "F05_xdom", "Q_all"):
        sub = sorted([r for r in rows if r["metric"] == k], key=lambda r: r["p"])
        m = len(sub)
        prev = 0
        for i, r in enumerate(sub):
            adj = min(1.0, max(prev, (m - i) * r["p"])); r["p_holm"] = adj; prev = adj
    write_csv("significance.csv", rows)


def crc_analysis():
    rows = []
    for s in SEEDS:
        d = dump("ours", s); dn = dump("abl_norhet", s)
        if d is None:
            continue
        for a in ALPHAS:
            for variant, dd, beta, ur in [("ours", d, BETA, True), ("no-gate", d, 0.0, False),
                                          ("w/o rhetoric head", dn, 0.0, False)]:
                if dd is None:
                    continue
                tau, _ = crc_threshold(dd, a, beta, ur)
                rt = realised_risk(dd, "test", tau, beta, ur)
                rq = realised_risk(dd, "quran", tau, beta, ur)
                ht = hyps_tagger("test", dd, tau=tau, beta=beta, use_rhet=ur)
                hq = hyps_tagger("quran", dd, tau=tau, beta=beta, use_rhet=ur)
                e = evaluate("test", ht); q = quran_ocr(hq)
                rows.append({"variant": variant, "seed": s, "alpha": a, "tau": tau, "risk_test": rt,
                             "risk_quran": rq, "F05": e["F05"], "P": e["P"], "R": e["R"],
                             "Q_rhet": q["rhet"], "Q_plain": q["plain"], "Q_all": q["all"]})
    write_csv("crc_per_seed.csv", rows)
    agg = []
    for v in ["ours", "no-gate", "w/o rhetoric head"]:
        for a in ALPHAS:
            rs = [r for r in rows if r["variant"] == v and r["alpha"] == a]
            if not rs:
                continue
            g = {"variant": v, "alpha": a}
            for k in ("tau", "risk_test", "risk_quran", "F05", "P", "R", "Q_rhet", "Q_plain", "Q_all"):
                vals = [r[k] for r in rs]; g[k] = float(np.mean(vals)); g[k + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            g["viol_rate"] = float(np.mean([r["risk_test"] > a for r in rs]))
            agg.append(g)
    write_csv("crc_agg.csv", agg)


def beta_sensitivity():
    rows = []
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        vals = collections.defaultdict(list)
        for s in SEEDS:
            d = dump("ours", s)
            if d is None:
                continue
            tau, _ = crc_threshold(d, ALPHA_MAIN, beta, True)
            e = evaluate("test", hyps_tagger("test", d, tau=tau, beta=beta))
            q = quran_ocr(hyps_tagger("quran", d, tau=tau, beta=beta))
            vals["F05"].append(e["F05"]); vals["Q_rhet"].append(q["rhet"]); vals["Q_plain"].append(q["plain"])
        rows.append({"beta": beta, **{k: float(np.mean(v)) for k, v in vals.items()},
                     **{k + "_sd": float(np.std(v, ddof=1)) for k, v in vals.items()}})
    write_csv("beta_sensitivity.csv", rows)


def reliability():
    rows = []
    d = dump("ours", 1)
    for cal in (False, True):
        conf, corr = [], []
        for x, p in zip(items("test"), d["test"]):
            g = [V.ttoi.get(t, -1) for t in x["tags"][:len(p["pk1"])]]
            am = p["argmaxT"]
            c = np.where(am == 0, p["pkT"] if cal else p["pk1"], p["bestpT"] if cal else p["bestp1"])
            m = am != 0  # reliability over proposed edits (the decisions that matter)
            conf.extend(c[m]); corr.extend((am == np.array(g))[m])
        conf = np.array(conf); corr = np.array(corr, float)
        edges = np.linspace(0, 1, 11)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.sum() >= 20:
                rows.append({"calibrated": int(cal), "bin_mid": (lo + hi) / 2, "conf": float(conf[mk].mean()),
                             "acc": float(corr[mk].mean()), "count": int(mk.sum())})
    write_csv("reliability.csv", rows)


def per_family():
    rows = []
    fams = ["ORTH", "TYPO", "SPACE", "MISS", "RED", "LEX"]
    for m in ["ours"] + BASE:
        hit = collections.Counter(); tot = collections.Counter()
        for s in SEEDS:
            d = dump(m, s)
            if d is None:
                continue
            for x, p in zip(items("test"), d["test"]):
                pt = tags_for(p)
                for k, (f, g) in enumerate(zip(x["fam"][:len(pt)], x["tags"][:len(pt)])):
                    if f != "-":
                        tot[f] += 1; hit[f] += pt[k] == g
        for f in fams:
            rows.append({"model": m, "family": f, "recall": hit[f] / max(tot[f], 1), "n": tot[f]})
    write_csv("per_family.csv", rows)


def robustness():
    rows = []
    for m in ["ours"] + BASE + ["seq2seq"]:
        for r, split in [(5, "test_r05"), (10, "test_r10"), (15, "test"), (20, "test_r20"), (25, "test_r25"), (30, "test_r30")]:
            fs = []
            for s in SEEDS:
                d = dump(m, s)
                if d is None or split not in d:
                    continue
                h = d[split] if m == "seq2seq" else hyps_tagger(split, d)
                fs.append(evaluate(split, h)["F05"])
            if fs:
                rows.append({"model": m, "rate": r, "F05": float(np.mean(fs)), "F05_sd": float(np.std(fs, ddof=1)) if len(fs) > 1 else 0.0})
    write_csv("robustness.csv", rows)


def ablation():
    # seed-matched: every row uses only the seeds available for all ablation variants
    aseeds = [s for s in SEEDS if all(os.path.exists(os.path.join(RES, "preds", f"{m}_s{s}.pt")) for m in ABL)]
    rows = []
    for m in ["ours"] + ABL:
        rs = []
        for s in aseeds:
            d = dump(m, s)
            if d is None:
                continue
            ht = hyps_tagger("test", d); e = evaluate("test", ht)
            hx = hyps_tagger("xdom", d)
            hq = hyps_tagger("quran", d); q = quran_ocr(hq)
            beta = BETA if CFGHAS(m) else 0.0
            tau, _ = crc_threshold(d, ALPHA_MAIN, beta, CFGHAS(m))
            ec = evaluate("test", hyps_tagger("test", d, tau=tau, beta=beta, use_rhet=CFGHAS(m)))
            qc = quran_ocr(hyps_tagger("quran", d, tau=tau, beta=beta, use_rhet=CFGHAS(m)))
            conf, corr = [], []
            for x, p in zip(items("test"), d["test"]):
                g = [V.ttoi.get(t, -1) for t in x["tags"][:len(p["pk1"])]]
                am = p["argmaxT"]; m_ = am != 0; conf.extend(p["bestpT"][m_]); corr.extend((am == np.array(g))[m_])
            rs.append({"F05": e["F05"], "F05_xdom": evaluate("xdom", hx)["F05"], "Q_rhet": q["rhet"],
                       "F05_crc": ec["F05"], "Q_rhet_crc": qc["rhet"], "Q_all_crc": qc["all"], "ECE": ece(conf, corr)})
            if m != "ours":
                _dump.pop((m, s), None)
        if rs:
            row = {"model": m, "name": NAMES[m], "n": len(rs)}
            for k in rs[0]:
                v = [r[k] for r in rs]; row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            rows.append(row)
    # post-hoc ablations of the decision layer on the full model
    for variant in ("w/o temperature scaling", "w/o risk control (tau=0.5)"):
        rs = []
        for s in aseeds:
            d = dump("ours", s)
            if d is None:
                continue
            if variant.startswith("w/o temperature"):
                its = items("cal")
                L = np.array([[np.mean((p["bestp1"] > (t + BETA * p["rhet"])) & np.array([g == "KEEP" for g in x["tags"][:len(p["pk1"])]]))
                               for t in TAU_GRID] for x, p in zip(its, d["cal"])])
                n = len(its); ok = (n / (n + 1)) * L.mean(0) + 1 / (n + 1) <= ALPHA_MAIN
                tau = float(TAU_GRID[np.argmax(ok)]) if ok.any() else 0.99
                kw = dict(tau=tau, beta=BETA, cal=False)
            else:
                kw = dict(tau=0.5, beta=BETA)
            ec = evaluate("test", hyps_tagger("test", d, **kw)); qc = quran_ocr(hyps_tagger("quran", d, **kw))
            rs.append({"F05_crc": ec["F05"], "Q_rhet_crc": qc["rhet"], "Q_all_crc": qc["all"],
                       "risk": float(np.mean([np.mean((((p["bestpT"] if kw.get("cal", True) else p["bestp1"])) > (kw["tau"] + BETA * p["rhet"])) &
                                                      np.array([g == "KEEP" for g in x["tags"][:len(p["pk1"])]]))
                                              for x, p in zip(items("test"), d["test"])]))})
        row = {"model": variant, "name": variant, "n": len(rs)}
        for k in rs[0]:
            v = [r[k] for r in rs]; row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v, ddof=1))
        rows.append(row)
    write_csv("ablation.csv", rows)


def CFGHAS(m):
    from train import CFG
    return CFG[m]["rhet"]


def rhetoric_eval():
    rows = []; curves = []
    canon = {(s, a): set(ws) for s, a, ws in CANON}
    for m in ["ours", "abl_nornn", "abl_concat", "abl_lstm", "abl_nopre"]:
        for s in SEEDS:
            d = dump(m, s)
            if d is None:
                continue
            y, pr, yi, yc, pc = [], [], [], [], []
            for x, p in zip(items("quran"), d["quran"]):
                y.extend(x["rhet"]); pr.extend(p["rhet"]); yi.extend(x["ist"])
                key = (x["ref"][0], x["ref"][1])
                if key in canon:
                    for w, q in zip(x["noisy"], p["rhet"]):
                        yc.append(int(w in canon[key])); pc.append(q)
            rows.append({"model": m, "seed": s, "AUC_silver": roc_auc_score(y, pr), "AP_silver": average_precision_score(y, pr),
                         "prev_silver": float(np.mean(y)), "AUC_istiara": roc_auc_score(yi, pr),
                         "AUC_canon": roc_auc_score(yc, pc) if len(set(yc)) > 1 else float("nan"),
                         "AP_canon": average_precision_score(yc, pc) if len(set(yc)) > 1 else float("nan"),
                         "prev_canon": float(np.mean(yc)) if yc else float("nan"), "n_canon": len(yc)})
            if m == "ours" and s == 1:
                for nm, yy, pp in [("silver", y, pr), ("canonical", yc, pc)]:
                    P_, R_, _ = precision_recall_curve(yy, pp)
                    sel = np.unique(np.linspace(0, len(P_) - 1, 40).astype(int))
                    for i in sel:
                        curves.append({"set": nm, "recall": float(R_[i]), "precision": float(P_[i])})
    write_csv("rhetoric_per_seed.csv", rows)
    write_csv("rhetoric_pr.csv", curves)
    # agreement of silver labels with canonical exemplars
    verses = load("quran.json"); vm = {(v["sura"], v["aya"]): v for v in verses}
    tp = fn = 0
    for (s, a), ws in canon.items():
        v = vm.get((s, a))
        if v is None:
            continue
        for k, w in enumerate(v["tokens"]):
            if w in ws:
                if k in v["rhetorical"]:
                    tp += 1
                else:
                    fn += 1
    json.dump({"canon_tokens": tp + fn, "silver_recall_on_canon": tp / max(tp + fn, 1)},
              open(os.path.join(RES, "silver_vs_canon.json"), "w"), indent=1)


def gate_analysis():
    rows = []
    for s in SEEDS:
        d = dump("ours", s)
        if d is None:
            continue
        acc = collections.defaultdict(list)
        for x, p in zip(items("test"), d["test"]):
            for k, f in enumerate(x["fam"][:len(p["gate"])]):
                acc["error:" + f if f != "-" else "clean prose"].append(p["gate"][k])
        for x, p in zip(items("quran"), d["quran"]):
            for k in range(len(p["gate"])):
                acc["Qur'an rhetorical" if x["rhet"][k] else "Qur'an plain"].append(p["gate"][k])
        for c, v in acc.items():
            v = np.array(v)
            rows.append({"seed": s, "cls": c, "mean": float(v.mean()), "q1": float(np.percentile(v, 25)),
                         "med": float(np.median(v)), "q3": float(np.percentile(v, 75)),
                         "lo": float(np.percentile(v, 5)), "hi": float(np.percentile(v, 95)), "n": len(v)})
    write_csv("gate_stats.csv", rows)


def suppression():
    """Explanation: change in edit probability on Qur'anic tokens caused by the rhetoric gate,
    binned by rhetoric-head probability."""
    rows = []
    d = dump("ours", 1)
    tau, _ = crc_threshold(d, ALPHA_MAIN, BETA, True)
    r_all, pe_all, gated_off = [], [], []
    for x, p in zip(items("quran"), d["quran"]):
        pe = p["bestpT"]
        r_all.extend(p["rhet"]); pe_all.extend(pe)
        gated_off.extend(((pe > tau) & (pe <= tau + BETA * p["rhet"])).astype(int))
    r_all = np.array(r_all); pe_all = np.array(pe_all); gated_off = np.array(gated_off)
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r_all > lo) & (r_all <= hi) if lo > 0 else (r_all <= hi)
        if m.sum() > 0:
            rows.append({"r_mid": (lo + hi) / 2, "n": int(m.sum()), "mean_pedit": float(pe_all[m].mean()),
                         "suppressed_rate": float(gated_off[m].mean())})
    write_csv("suppression.csv", rows)


def quran_edit_types():
    """Which edit types RACE applies to error-free Qur'anic text (seed-pooled)."""
    rows = []
    cnt = {"argmax": collections.Counter(), "crc": collections.Counter()}
    ntok = 0
    for s in SEEDS:
        d = dump("ours", s)
        if d is None:
            continue
        tau, _ = crc_threshold(d, ALPHA_MAIN, BETA, True)
        for x, p in zip(items("quran"), d["quran"]):
            ntok += len(x["noisy"])
            for mode, tg in (("argmax", tags_for(p)), ("crc", tags_for(p, tau=tau, beta=BETA))):
                for t in tg:
                    if t == "KEEP":
                        continue
                    if t.startswith("C:"):
                        k = "orthographic/char" if any(c in t for c in "أإآءؤئةىه") else "other char"
                    elif t.startswith("SPLIT"):
                        k = "split"
                    elif t == "MERGE":
                        k = "merge"
                    elif t == "DEL":
                        k = "delete"
                    elif t.startswith("APP_"):
                        k = "insert"
                    else:
                        k = "lexical"
                    cnt[mode][k] += 1
    for mode, c in cnt.items():
        tot = sum(c.values())
        for k, v in c.most_common():
            rows.append({"mode": mode, "type": k, "count": v, "share": v / max(tot, 1), "per_10k_tokens": v / max(ntok, 1) * 1e4})
    write_csv("quran_edit_types.csv", rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["main", "crc", "beta", "rel", "fam", "rob", "abl", "rhet", "gate", "sup", "qtypes"]
    if "main" in which:
        ps = main_results(); significance(ps)
    if "crc" in which: crc_analysis()
    if "beta" in which: beta_sensitivity()
    if "rel" in which: reliability()
    if "fam" in which: per_family()
    if "rob" in which: robustness()
    if "abl" in which: ablation()
    if "rhet" in which: rhetoric_eval()
    if "gate" in which: gate_analysis()
    if "sup" in which: suppression()
    if "qtypes" in which: quran_edit_types()
