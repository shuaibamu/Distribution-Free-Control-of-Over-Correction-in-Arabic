"""Risk-invariant design-space exploration of the routed serving architecture.

For every routing level theta in results/routing2.csv the pipeline uses one end-to-end
conformal threshold (alpha = 0.5%), so every configuration satisfies the same risk
bound. For each theta and target arrival rate nu, the smallest number of CPU cores
whose simulated p99 latency meets a derated SLO is found by bisection over cores.
The SLO is derated by the largest measured simulator underestimate of p99 latency
(results/load_validation.csv), so that the simulated design is conservative.
Outputs: results/design_space.csv, results/design_select.csv
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import simulate as S
from common import RES

LOADS = [250, 500, 1000, 2000]          # requests per second
EPS = 0.002                             # quality tolerance on dev F0.5 (0.2 points)


def derate():
    rows = S.read("load_validation.csv")
    err = max((float(r["meas_p99_ms"]) - float(r["sim_p99_ms"])) / float(r["meas_p99_ms"]) for r in rows)
    return S.SLO_S * (1 - max(err, 0.0)), err


def ok(nu, cores, rho, slo, seeds=(0, 1)):
    if rho <= 0:
        nl, nh, mode = cores, 1, "light"
    else:
        nl, nh = S.split_cores(cores, rho); mode = "routed"
    mean = S.SAMPLES["light"].mean() + rho * S.SAMPLES["heavy"].mean()
    if mode == "routed" and (nu * S.SAMPLES["light"].mean() >= nl or nu * rho * S.SAMPLES["heavy"].mean() >= nh):
        return False
    if nu * mean >= cores:
        return False
    return all(np.percentile(S.simulate(nu, max(nl, 1), max(nh, 1), mode, rho, seed=s, n_req=20000), 99) <= slo for s in seeds)


def min_cores(nu, rho, slo):
    lo, hi = 1, 4
    while not ok(nu, hi, rho, slo):
        lo, hi = hi, hi * 2
        if hi > 4096:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(nu, mid, rho, slo):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    slo, err = derate()
    rows = []
    for r in S.route:
        th = float(r["theta"]); rho = float(r["routed_dev"])
        for nu in LOADS:
            c = min_cores(nu, rho, slo)
            rows.append(dict(theta=th, routed_dev=rho, routed_test=float(r["routed_test"]), nu=nu, cores=c,
                             F05_dev=100 * float(r["F05_dev_comp"]), F05_test=100 * float(r["F05_test_comp"]),
                             risk_test=100 * float(r["risk_test_comp"]), risk_xdom=100 * float(r["risk_xdom_comp"])))
            print(rows[-1], flush=True)
    with open(os.path.join(RES, "design_space.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # selection rule: quality target on dev, then fewest cores; ties -> larger theta
    fmax = max(x["F05_dev"] for x in rows)
    sel = []
    for nu in LOADS:
        cand = [x for x in rows if x["nu"] == nu and x["cores"] and x["F05_dev"] >= fmax - 100 * EPS]
        best = min(cand, key=lambda x: (x["cores"], -x["theta"]))
        light = next(x for x in rows if x["nu"] == nu and x["routed_dev"] == 0)
        heavy = next(x for x in rows if x["nu"] == nu and x["routed_dev"] == 1)  # all sentences reach the ensemble
        sel.append(dict(nu=nu, theta=best["theta"], routed=best["routed_dev"], cores=best["cores"],
                        F05_test=best["F05_test"], risk_test=best["risk_test"],
                        cores_light=light["cores"], F05_test_light=light["F05_test"],
                        cores_heavy=heavy["cores"], F05_test_heavy=heavy["F05_test"]))
        print(sel[-1], flush=True)
    with open(os.path.join(RES, "design_select.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel[0])); w.writeheader(); w.writerows(sel)
    json.dump(dict(slo_ms=1000 * S.SLO_S, derated_slo_ms=1000 * slo, sim_p99_underestimate=err, eps_points=100 * EPS,
                   loads=LOADS), open(os.path.join(RES, "design_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
