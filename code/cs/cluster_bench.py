"""Measured multi-node serving benchmark (replaces simulation for the cluster results).

Every worker is a single-threaded process serving one tier over TCP (JSON lines).  A dispatcher
issues open-loop Poisson traffic, sends each request to the light worker with the fewest
outstanding requests and, for the routed service, forwards uncertain sentences (edit entropy > theta)
to the least-loaded heavy worker.  End-to-end latency includes the network.

  start workers on every node (one per core you want to use):
      python3 code/cs/cluster_bench.py worker --port 7001 --tier light
      python3 code/cs/cluster_bench.py worker --port 7101 --tier heavy
  list them in a hosts file, one "host:port:tier" per line, then run the dispatcher on one node:
      python3 code/cs/cluster_bench.py client --hosts hosts.txt --mode routed --tag routed_16c
  modes: light (light workers only), heavy (heavy workers only), routed (both)
  -> results/cluster_<tag>.csv  (rate, p50, p99, achieved throughput) and the SLO capacity.
"""
import os, sys, json, time, asyncio, argparse, random, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RES


def worker(port, tier, theta):
    import torch
    torch.set_num_threads(1)
    import serve_bench as SB, analysis as A
    its = A.items("test")
    models = [SB.load_tagger("ours_s1")] if tier == "light" else [SB.load_tagger(f"ours_s{s}") for s in (2, 3, 4, 5)]
    for i in range(32):
        SB.tier_probs(models, [its[i]])

    def serve(idx):
        t = time.perf_counter()
        P, R, M = SB.tier_probs(models, [its[idx % len(its)]])
        route = False
        if tier == "light":
            PP = P.clone(); PP[..., 0] = 0
            route = bool(SB.entropy_u(PP.max(-1).values[0, :int(M[0].sum())].numpy()) > theta)
        return route, (time.perf_counter() - t) * 1e3

    async def handle(reader, writer):
        loop = asyncio.get_running_loop()
        while True:
            line = await reader.readline()
            if not line:
                break
            req = json.loads(line)
            route, ms = serve(req["idx"])             # one request at a time: single-server queue
            writer.write((json.dumps({"id": req["id"], "route": route, "svc_ms": ms}) + "\n").encode())
            await writer.drain()

    async def main():
        srv = await asyncio.start_server(handle, "0.0.0.0", port)
        print(f"worker {tier} on :{port} ready", flush=True)
        async with srv:
            await srv.serve_forever()
    asyncio.run(main())


class Pool:
    def __init__(self, addrs):
        self.addrs = addrs; self.conns = []; self.out = []

    async def open(self, on_reply):
        for h, p in self.addrs:
            r, w = await asyncio.open_connection(h, p)
            k = len(self.conns); self.conns.append((r, w)); self.out.append(0)
            asyncio.create_task(self._reader(k, r, on_reply))

    async def _reader(self, k, r, on_reply):
        while True:
            line = await r.readline()
            if not line:
                return
            self.out[k] -= 1
            await on_reply(json.loads(line))

    def send(self, msg):
        k = int(np.argmin(self.out)); self.out[k] += 1
        self.conns[k][1].write((json.dumps(msg) + "\n").encode())


async def run_rate(light, heavy, mode, lam, n, seed):
    rng = np.random.default_rng(seed)
    t_arr = {}; lat = []; done = asyncio.Event(); routed = [0]
    state = {"left": n}

    async def finish(rep):
        lat.append(time.perf_counter() - t_arr[rep["id"]])
        state["left"] -= 1
        if state["left"] == 0:
            done.set()

    async def on_light(rep):
        if mode == "routed" and rep["route"]:
            routed[0] += 1; heavy.send({"id": rep["id"], "idx": rep["id"]})
        else:
            await finish(rep)

    await light.open(on_light) if light else None
    await heavy.open(finish) if heavy else None
    first = heavy if mode == "heavy" else light
    t0 = time.perf_counter(); t = t0
    for i in range(n):
        t += rng.exponential(1 / lam)
        d = t - time.perf_counter()
        if d > 0:
            await asyncio.sleep(d)
        t_arr[i] = time.perf_counter()
        first.send({"id": i, "idx": int(rng.integers(0, 4000))})
    await asyncio.wait_for(done.wait(), timeout=600)
    span = time.perf_counter() - t0
    for p in (light, heavy):
        if p:
            for r, w in p.conns:
                w.close()
    x = np.array(lat[n // 10:]) * 1e3
    return {"rate": lam, "p50_ms": float(np.percentile(x, 50)), "p99_ms": float(np.percentile(x, 99)),
            "throughput": n / span, "routed": routed[0] / n}


def client(a):
    hosts = [l.strip().split(":") for l in open(a.hosts) if l.strip() and not l.startswith("#")]
    L = [(h, int(p)) for h, p, t in hosts if t == "light"]
    H = [(h, int(p)) for h, p, t in hosts if t == "heavy"]
    rows = []; lam = a.start
    while lam <= a.max_rate:
        light = Pool(L) if a.mode in ("light", "routed") else None
        heavy = Pool(H) if a.mode in ("heavy", "routed") else None
        rec = asyncio.run(run_rate(light, heavy, a.mode, lam, a.n, a.seed))
        rec.update(mode=a.mode, n_light=len(L) if light else 0, n_heavy=len(H) if heavy else 0)
        rows.append(rec); print(rec, flush=True)
        if rec["p99_ms"] > a.slo * 1.5:
            break
        lam *= a.step
    ok = [r["rate"] for r in rows if r["p99_ms"] <= a.slo]
    with open(os.path.join(RES, f"cluster_{a.tag}.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"capacity at p99<={a.slo:.0f} ms: {max(ok) if ok else 0:.1f} req/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["worker", "client"])
    ap.add_argument("--port", type=int, default=7001)
    ap.add_argument("--tier", default="light")
    ap.add_argument("--theta", type=float, default=float(os.environ.get("THETA", "0.9")))
    ap.add_argument("--hosts"); ap.add_argument("--mode", default="routed"); ap.add_argument("--tag", default="run")
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=float, default=20.0); ap.add_argument("--step", type=float, default=1.25)
    ap.add_argument("--max_rate", type=float, default=50000.0); ap.add_argument("--slo", type=float, default=200.0)
    a = ap.parse_args()
    worker(a.port, a.tier, a.theta) if a.role == "worker" else client(a)
