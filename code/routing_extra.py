"""Routing analysis with (i) theta chosen on the development split, (ii) realised mixture risk,
(iii) an end-to-end conformal threshold for the routed pipeline, and (iv) cost against an
ensemble-only service.  Uses the light (seed 1) and heavy (seeds 2-5) tiers of serve_bench.

usage: python3 code/routing_extra.py   -> results/routing2.csv, results/routing2_meta.json
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from gec_data import apply_tags
import analysis as A
import serve_bench as SB

GRID = A.TAU_GRID


@torch.no_grad()
def raw(models, its, bs=128):
    out = []
    for i in range(0, len(its), bs):
        P, R, M = SB.tier_probs(models, its[i:i + bs])
        PP = P.clone(); PP[..., 0] = 0
        pe, best = PP.max(-1)
        for b in range(P.size(0)):
            l = int(M[b].sum())
            out.append((pe[b, :l].numpy(), R[b, :l].numpy(), best[b, :l].numpy()))
    return out


def keepmask(its):
    return [np.array([t == "KEEP" for t in x["tags"][:MAXW]]) for x in its]


def applied(r, tau):
    pe, rh, _ = r
    return pe > tau + A.BETA * rh


def tags(r, tau):
    app = applied(r, tau)
    return [A.TAGS[int(b)] if a else "KEEP" for a, b in zip(app, r[2])]


def loss(r, k, tau):
    app = applied(r, tau)
    return float((app & k[:len(app)]).sum()) / len(app)


def crc(rs, ks, alpha):
    n = len(rs)
    for tau in GRID:
        if (n / (n + 1)) * np.mean([loss(r, k, tau) for r, k in zip(rs, ks)]) + 1 / (n + 1) <= alpha:
            return float(tau)
    return 1.0


def main():
    light = [SB.load_tagger("ours_s1")]
    heavy = [SB.load_tagger(f"ours_s{s}") for s in (2, 3, 4, 5)]
    splits = ["cal", "dev", "test", "quran", "xdom"]
    its = {s: A.items(s) for s in splits}
    Lr = {s: raw(light, its[s]) for s in splits}
    Hr = {s: raw(heavy, its[s]) for s in splits}
    K = {s: keepmask(its[s]) for s in splits}
    alpha = A.ALPHA_MAIN
    tl = crc(Lr["cal"], K["cal"], alpha); th = crc(Hr["cal"], K["cal"], alpha)
    lat = A_rd("latency_samples.csv")
    ms = {t: float(np.mean([float(x["ms"]) for x in lat if x["tier"] == t])) for t in ("light", "heavy")}
    U = {s: np.array([SB.entropy_u(r[0]) for r in Lr[s]]) for s in splits}
    rows = []
    for theta in [1.01, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        rec = {"theta": theta}
        route = {s: U[s] > theta for s in splits}
        # end-to-end CRC: one shared tau for the composite rule (routing depends only on the light
        # tier's probabilities, so the composite loss is still monotone in tau)
        comp = [h if u else l for l, h, u in zip(Lr["cal"], Hr["cal"], route["cal"])]
        tc = crc(comp, K["cal"], alpha); rec["tau_comp"] = tc
        for s in splits:
            sep = [(h, th) if u else (l, tl) for l, h, u in zip(Lr[s], Hr[s], route[s])]
            cmp_ = [(h if u else l, tc) for l, h, u in zip(Lr[s], Hr[s], route[s])]
            for nm, sel in (("sep", sep), ("comp", cmp_)):
                hyps = [apply_tags(x["noisy"][:MAXW], tags(r, t)) for x, (r, t) in zip(its[s], sel)]
                risk = float(np.mean([loss(r, k, t) for (r, t), k in zip(sel, K[s])]))
                rec[f"risk_{s}_{nm}"] = risk
                if s == "quran":
                    rec[f"Q_all_{nm}"] = A.quran_ocr(hyps)["all"]
                else:
                    rec[f"F05_{s}_{nm}"] = A.evaluate(s, hyps)["F05"]
            rec[f"routed_{s}"] = float(route[s].mean())
        rec["ms_per_sent"] = ms["light"] + rec["routed_test"] * ms["heavy"]
        rows.append(rec)
        print(theta, {k: round(v, 4) for k, v in rec.items() if isinstance(v, float)}, flush=True)
    A.write_csv("routing2.csv", rows)
    # theta rule fixed in advance, applied on the development split (disjoint from calibration and
    # test, so the conformal step still sees a fixed routing rule): the largest theta whose dev F0.5
    # is within 0.2 points of routing everything
    full = [r for r in rows if r["theta"] == 0.0][0]["F05_dev_sep"]
    cand = [r for r in rows if r["theta"] <= 1.0 and r["F05_dev_sep"] >= full - 0.002]
    sel = max(cand, key=lambda r: r["theta"])
    json.dump({"tau_light": tl, "tau_heavy": th, "ms_light": ms["light"], "ms_heavy": ms["heavy"],
               "theta_cal": sel["theta"], "theta_rule_split": "dev", "alpha": alpha}, open(os.path.join(RES, "routing2_meta.json"), "w"), indent=1)


def A_rd(name):
    import csv
    return list(csv.DictReader(open(os.path.join(RES, name))))


if __name__ == "__main__":
    main()
