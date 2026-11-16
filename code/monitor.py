"""Label-free shift monitor that triggers recalibration of the conformal threshold.

The deployed light model (seed 1) is applied to a stream of sentence windows. For each window,
the per-sentence routing statistic u = max_i H2(p_i) (binary entropy of the calibrated edit
probability, already computed by the router) is compared with its distribution on the calibration
split with a two-sample Kolmogorov-Smirnov test. An alarm is raised when p < P_ALARM. No labels
are needed for monitoring; labels are needed only for the recalibration sample.

Reports, per domain: share of windows that raise an alarm, realised false-edit rate with the
in-domain threshold, and with a threshold recalibrated on the domain's calibration sample.
usage: python3 code/monitor.py synthetic   (classical prose, five held-out works)
       python3 code/monitor.py real        (run from the QALB root: QALB-2014 test vs ZAEBUC)
writes results/monitor_<setting>.csv and results/monitor_<setting>_meta.json
"""
import os, sys, json, csv
import numpy as np
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import analysis as A
import multidomain as MD

P_ALARM = 0.01
LOOKS = [100, 200, 500, 1000, 2000]
SIZES = [100, 200, 300, 500, 1000]
REPS = 200
ALPHA = 0.005
SEED = 1


def h2(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ustat(preds):
    return np.array([h2(p["bestpT"]).max() if len(p["bestpT"]) else 0.0 for p in preds])


def erate(preds, tau, beta):
    """Per-sentence share of tokens the deployed rule would edit (label-free)."""
    return np.array([float((p["bestpT"] > tau + beta * p["rhet"]).mean()) if len(p["bestpT"]) else 0.0 for p in preds])


def windows(n, w):
    return [(i, i + w) for i in range(0, n - w + 1, w)]


def run(setting):
    if setting == "synthetic":
        sets = {k: (load(f"gec_md_{k}_cal.json"), load(f"gec_md_{k}_test.json")) for k in MD.WORKS}
        sets["Jurjani"] = (None, A.items("xdom"))
        W = 100
    else:
        sets = {"ZAEBUC": (load("gec_md_ZAEBUC_cal.json"), load("gec_xdom.json")),
                "Quran": (A.items("quran_dev"), A.items("quran"))}
        W = 50
    cal = load("gec_cal.json"); test = load("gec_test.json")
    pc = MD.tagger_preds("ours", SEED, cal)
    ucal = ustat(pc)
    Lcal = MD.loss_mat(cal, pc, A.BETA)
    tau = MD.crc_tau(Lcal, ALPHA)
    it = int(np.argmin(np.abs(A.TAU_GRID - tau)))
    ecal = erate(pc, tau, A.BETA)
    rows = []
    allsets = {"in-domain": (None, test)}; allsets.update(sets)
    for k, (c, t) in allsets.items():
        pt = MD.tagger_preds("ours", SEED, t)
        u = ustat(pt); Lt = MD.loss_mat(t, pt, A.BETA)
        ws = windows(len(t), W)
        e = erate(pt, tau, A.BETA)
        pv = [ks_2samp(u[a:b], ucal).pvalue for a, b in ws]
        pe = [ks_2samp(e[a:b], ecal).pvalue for a, b in ws]
        alarms = [p < P_ALARM for p in pv]
        al_e = [p < P_ALARM for p in pe]
        first = next((j for j, z in enumerate(al_e) if z), None)
        # sequential look at growing prefixes, Bonferroni over LOOKS
        seq = {}
        for nlk in LOOKS:
            if nlk <= len(t):
                seq[f"p_u_{nlk}"] = float(ks_2samp(u[:nlk], ucal).pvalue)
                seq[f"p_e_{nlk}"] = float(ks_2samp(e[:nlk], ecal).pvalue)
        det_u = next((nlk for nlk in LOOKS if seq.get(f"p_u_{nlk}", 1) < P_ALARM / len(LOOKS)), None)
        det_e = next((nlk for nlk in LOOKS if seq.get(f"p_e_{nlk}", 1) < P_ALARM / len(LOOKS)), None)
        rec = {"domain": k, "windows": len(ws), "window_size": W, "alarm_rate": float(np.mean(alarms)) if ws else float("nan"),
               "first_alarm_window": first, "median_p": float(np.median(pv)) if pv else float("nan"),
               "mean_u": float(u.mean()), "alarm_rate_edit": float(np.mean(al_e)) if ws else float("nan"),
               "median_p_edit": float(np.median(pe)) if pe else float("nan"), "edit_rate": 100 * float(e.mean()), "risk_in": 100 * float(Lt[:, it].mean()), "tau_in": tau,
               "detect_after_u": det_u, "detect_after_e": det_e, **seq}
        if c is not None:
            pcd = MD.tagger_preds("ours", SEED, c)
            td = MD.crc_tau(MD.loss_mat(c, pcd, A.BETA), ALPHA)
            idd = int(np.argmin(np.abs(A.TAU_GRID - td)))
            rec.update(tau_rec=td, risk_rec=100 * float(Lt[:, idd].mean()), n_recal=len(c))
            Lc_d = MD.loss_mat(c, pcd, A.BETA)
            err_tok = [(p["bestpT"] - A.BETA * p["rhet"])[np.array([tg != "KEEP" for tg in x["tags"][:len(p["bestpT"])]])] for x, p in zip(t, pt)]
            err_all = np.concatenate([v for v in err_tok if len(v)]) if any(len(v) for v in err_tok) else np.zeros(0)
            rng = np.random.default_rng(7)
            for nr in SIZES:
                if nr > len(c):
                    continue
                rk, tp, nod = [], [], 0
                for _ in range(REPS):
                    idx = rng.choice(len(c), nr, replace=False)
                    tr = MD.crc_tau(Lc_d[idx], ALPHA); ir = int(np.argmin(np.abs(A.TAU_GRID - tr)))
                    rk.append(Lt[:, ir].mean()); tp.append(float((err_all > tr).mean()) if len(err_all) else 0.0)
                    nod += tr >= 1.0
                rec[f"rec{nr}_risk"] = 100 * float(np.mean(rk)); rec[f"rec{nr}_viol"] = float(np.mean(np.array(rk) > ALPHA))
                rec[f"rec{nr}_errtok_edited"] = 100 * float(np.mean(tp)); rec[f"rec{nr}_noedit"] = nod / REPS
        rows.append(rec); print(rec, flush=True)
    with open(os.path.join(RES, f"monitor_{setting}.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump({"p_alarm": P_ALARM, "alpha": ALPHA, "seed": SEED, "window": W, "mean_u_cal": float(ucal.mean()), "edit_rate_cal": 100 * float(ecal.mean()),
               "n_cal": len(cal)}, open(os.path.join(RES, f"monitor_{setting}_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
