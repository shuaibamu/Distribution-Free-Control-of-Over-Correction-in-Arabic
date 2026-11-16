"""Measured open-loop load test on the physical 2-core node, used to validate the discrete-event
cluster model (simulate.py) on tail latency rather than on throughput alone.

A dispatcher issues requests with Poisson inter-arrival times into a shared queue; worker
processes (one thread each) serve them: 'light' = two light-tier workers; 'heavy' = two ensemble workers; 'routed' = one
light-tier worker and one heavy-tier (4-model) worker, with routing by edit entropy (theta).
End-to-end latency = completion time (taken by the worker) - scheduled arrival time.  The same arrival rates are then
fed to simulate.simulate() with zero network hop, and measured vs. simulated p50/p99 are written to
results/load_validation.csv.
"""
import os, sys, time, json, csv
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import RES

N_REQ = int(os.environ.get("N_REQ", "3000"))
THETA = float(os.environ.get("THETA", json.load(open(os.path.join(RES, "routing2_meta.json")))["theta_cal"]))


def light_worker(q_in, q_heavy, q_out, routed):
    import torch; torch.set_num_threads(1)
    import serve_bench as SB, analysis as A
    its = A.items("test"); m = [SB.load_tagger("ours_s1")]
    for i in range(64):
        SB.tier_probs(m, [its[i]])
    q_out.put(("ready", 0, 0.0))
    while True:
        job = q_in.get()
        if job is None:
            break
        i, t_arr = job
        P, R, M = SB.tier_probs(m, [its[i % len(its)]])
        if routed:
            PP = P.clone(); PP[..., 0] = 0
            pe = PP.max(-1).values[0, :int(M[0].sum())].numpy()
            if SB.entropy_u(pe) > THETA:
                q_heavy.put((i, t_arr)); continue
        q_out.put(("L", i, time.perf_counter() - t_arr))


def heavy_worker(q_heavy, q_out):
    import torch; torch.set_num_threads(1)
    import serve_bench as SB, analysis as A
    its = A.items("test"); m = [SB.load_tagger(f"ours_s{s}") for s in (2, 3, 4, 5)]
    for i in range(32):
        SB.tier_probs(m, [its[i]])
    q_out.put(("ready", 0, 0.0))
    while True:
        job = q_heavy.get()
        if job is None:
            break
        i, t_arr = job
        SB.tier_probs(m, [its[i % len(its)]])
        q_out.put(("H", i, time.perf_counter() - t_arr))


def run(mode, lam, seed=0):
    ctx = mp.get_context("fork")
    q_in, q_heavy, q_out = ctx.Queue(), ctx.Queue(), ctx.Queue()
    routed = mode == "routed"
    procs = []
    n_light = {"light": 2, "routed": 1, "heavy": 0}[mode]
    n_heavy = {"light": 0, "routed": 1, "heavy": 2}[mode]
    for _ in range(n_light):
        procs.append(ctx.Process(target=light_worker, args=(q_in, q_heavy, q_out, routed)))
    for _ in range(n_heavy):
        procs.append(ctx.Process(target=heavy_worker, args=(q_heavy, q_out)))
    q_first = q_heavy if mode == "heavy" else q_in
    for p in procs:
        p.start()
    for _ in procs:
        q_out.get()                               # wait until every worker is warm
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter() + 0.2
    arr = t0 + np.cumsum(rng.exponential(1 / lam, N_REQ))
    order = rng.permutation(N_REQ)
    for k, t in enumerate(arr):
        while True:
            dt = t - time.perf_counter()
            if dt <= 0:
                break
            time.sleep(min(dt, 0.0005))
        q_first.put((int(order[k]), float(t)))
    lats = []; nh = 0
    for _ in range(N_REQ):
        tier, i, lat = q_out.get()          # latency is measured by the worker at completion
        lats.append(lat); nh += tier == "H"
    for _ in range(n_light):
        q_in.put(None)
    for _ in range(n_heavy):
        q_heavy.put(None)
    for p in procs:
        p.join()
    lats = np.array(lats[N_REQ // 10:])
    return lats, nh / N_REQ


def main():
    import simulate as S
    S.HOP = 0.0
    mean_l = S.SAMPLES["light"].mean(); mean_h = S.SAMPLES["heavy"].mean()
    rows = []
    only = os.environ.get("MODES")
    for mode in (only.split(",") if only else ("light", "heavy", "routed")):
        for frac in (0.3, 0.5, 0.7, 0.8):       # utilisation of the workers (offered work / cores)
            if mode == "light":
                cap = 2 / mean_l
            elif mode == "heavy":
                cap = 2 / mean_h
            else:
                rho0 = S.routed_frac(THETA)
                cap = min(1 / mean_l, 1 / (rho0 * mean_h))
            lam = frac * cap
            meas, rho = run(mode, lam)
            if mode == "light":
                sim = S.simulate(lam, 2, 1, "light", n_req=40000)
            elif mode == "heavy":
                sim = S.simulate(lam, 1, 2, "heavy", n_req=40000)
            else:
                sim = S.simulate(lam, 1, 1, "routed", rho=rho, n_req=40000)
            rec = {"mode": mode, "load": frac, "lam": lam, "routed": rho,
                   "meas_p50_ms": float(np.percentile(meas, 50) * 1e3), "meas_p99_ms": float(np.percentile(meas, 99) * 1e3),
                   "sim_p50_ms": float(np.percentile(sim, 50) * 1e3), "sim_p99_ms": float(np.percentile(sim, 99) * 1e3)}
            rows.append(rec); print(rec, flush=True)
    out = os.path.join(RES, "load_validation.csv")
    if only and os.path.exists(out):                     # merge with earlier modes
        old = [r for r in csv.DictReader(open(out)) if r["mode"] not in only.split(",")]
        rows = old + rows
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
