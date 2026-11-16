"""Discrete-event simulation of an uncertainty-routed GEC service on a CPU cluster.

Service times are resampled from the per-request latencies measured on this
machine (results/latency_samples.csv); the routed fraction comes from the
measured cascade (results/routing.csv). Each worker is one CPU core serving one
request at a time (FCFS, shared queue per pool, i.e. M/G/c per tier). Network
cost per hop = fixed RTT/2 + payload / bandwidth.
Outputs: results/sim_load.csv, results/sim_scaling.csv, results/sim_validation.csv
"""
import os, sys, csv, json, heapq, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import RES

NET_HOP_S = 0.00025          # 0.25 ms one-way intra-cluster latency (assumption, stated in paper)
BW_BPS = 10e9 / 8            # 10 Gb/s links (assumption)
SLO_S = 0.200                # p99 latency target used for capacity
N_REQ = 40000


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


lat = read("latency_samples.csv")
SAMPLES = {t: np.array([float(r["ms"]) / 1000 for r in lat if r["tier"] == t]) for t in ("light", "heavy", "seq2seq")}
route = read("routing2.csv")   # routing analysis with theta chosen on the calibration split
PAY = json.load(open(os.path.join(RES, "payload.json")))
HOP = NET_HOP_S + (PAY["req_bytes_mean"] + PAY["resp_bytes_mean"]) / BW_BPS


def routed_frac(theta):
    r = min(route, key=lambda r: abs(float(r["theta"]) - theta))
    return float(r["routed_test"])


def simulate(lam, n_light, n_heavy, mode, rho=0.0, seed=0, n_req=N_REQ):
    """mode: 'light' | 'heavy' | 'seq2seq' | 'routed'. Returns latency array (s)."""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1 / lam, n_req))
    first = {"light": "light", "routed": "light", "heavy": "heavy", "seq2seq": "seq2seq"}[mode]
    pools = {"light": [0.0] * n_light, "heavy": [0.0] * n_heavy, "seq2seq": [0.0] * n_light}
    for p in pools.values():
        heapq.heapify(p)
    lats = np.empty(n_req)
    # events processed in arrival order per stage (FCFS with earliest-free worker)
    stage1_done = np.empty(n_req)
    for i, t in enumerate(arr):
        pool = pools[first]
        free = heapq.heappop(pool)
        start = max(t + HOP, free)
        end = start + rng.choice(SAMPLES[first])
        heapq.heappush(pool, end)
        stage1_done[i] = end
    if mode == "routed":
        go = rng.random(n_req) < rho
        order = np.argsort(stage1_done)
        done = stage1_done.copy()
        pool = pools["heavy"]
        for i in order:
            if not go[i]:
                continue
            free = heapq.heappop(pool)
            start = max(stage1_done[i] + HOP, free)
            end = start + rng.choice(SAMPLES["heavy"])
            heapq.heappush(pool, end)
            done[i] = end
        stage1_done = done
    lats = stage1_done + HOP - arr
    warm = n_req // 10
    return lats[warm:]


def split_cores(n, rho):
    """Allocate cores to the routed pipeline in proportion to expected work."""
    wl = SAMPLES["light"].mean(); wh = rho * SAMPLES["heavy"].mean()
    nh = max(1, int(round(n * wh / (wl + wh)))) if rho > 0 else 0
    nh = min(nh, n - 1)
    return n - nh, nh


def capacity(n, mode, rho=0.0):
    """Largest arrival rate (req/s) with p99 <= SLO, by bisection."""
    if mode == "routed":
        nl, nh = split_cores(n, rho)
    elif mode == "heavy":
        nl, nh = 0, n
    else:
        nl, nh = n, 0
    mean_s = {"light": SAMPLES["light"].mean(), "heavy": SAMPLES["heavy"].mean(), "seq2seq": SAMPLES["seq2seq"].mean()}
    if mode == "routed":
        hi = n / (mean_s["light"] + rho * mean_s["heavy"]) * 1.2
    else:
        hi = n / mean_s[mode] * 1.2
    lo = 0.0
    for _ in range(16):
        mid = (lo + hi) / 2
        l = simulate(mid, max(nl, 1), max(nh, 1), mode, rho, n_req=20000)
        if np.percentile(l, 99) <= SLO_S:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    theta = float(os.environ.get("THETA", json.load(open(os.path.join(RES, "routing2_meta.json")))["theta_cal"]))
    rho = routed_frac(theta)
    # 1) validation against the measured 1- and 2-worker throughput (saturation throughput)
    val = []
    meas = read("worker_scaling.csv")
    batch = read("batch_throughput.csv")
    per_req_b32 = 1.0 / float([b for b in batch if int(b["batch"]) == 32][0]["sent_per_s"])
    for m in meas:
        w = int(m["workers"])
        val.append({"workers": w, "measured_sent_per_s": float(m["sent_per_s"]),
                    "model_sent_per_s": w / per_req_b32})
    with open(os.path.join(RES, "sim_validation.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(val[0].keys())); wr.writeheader(); wr.writerows(val)
    # 2) latency vs offered load on a 16-core cluster
    rows = []
    n = 16
    nl, nh = split_cores(n, rho)
    for mode in ("light", "heavy", "seq2seq", "routed"):
        cap = n / (SAMPLES[mode].mean() if mode != "routed" else (SAMPLES["light"].mean() + rho * SAMPLES["heavy"].mean()))
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            lam = frac * cap
            if mode == "routed":
                l = simulate(lam, nl, nh, mode, rho)
            elif mode == "heavy":
                l = simulate(lam, 1, n, mode)
            else:
                l = simulate(lam, n, 1, mode)
            rows.append({"mode": mode, "load": frac, "lam": lam, "p50_ms": float(np.percentile(l, 50) * 1000),
                         "p99_ms": float(np.percentile(l, 99) * 1000), "mean_ms": float(l.mean() * 1000)})
    with open(os.path.join(RES, "sim_load.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    # 3) capacity scaling with cluster size
    sc = []
    for n in [2, 4, 8, 16, 32, 64, 128]:
        rec = {"cores": n}
        for mode in ("light", "heavy", "routed", "seq2seq"):
            rec[mode] = capacity(n, mode, rho)
        sc.append(rec)
    base = {m: sc[0][m] / sc[0]["cores"] for m in ("light", "heavy", "routed", "seq2seq")}
    for r in sc:
        for m in ("light", "heavy", "routed", "seq2seq"):
            r[m + "_eff"] = r[m] / (base[m] * r["cores"])
    with open(os.path.join(RES, "sim_scaling.csv"), "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(sc[0].keys())); wr.writeheader(); wr.writerows(sc)
    json.dump({"theta": theta, "rho": rho, "hop_s": HOP, "slo_s": SLO_S, "split16": [nl, nh]},
              open(os.path.join(RES, "sim_meta.json"), "w"), indent=1)
    print("sim done", rho)


if __name__ == "__main__":
    main()
