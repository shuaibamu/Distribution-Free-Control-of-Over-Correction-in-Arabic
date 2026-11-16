"""Equal-footing cost comparison of every model: parameters, training time, per-request latency
(batch 1, one thread) and batched throughput (batch 64, one thread), on the same CPU and the same
400 test sentences.  Run on an idle machine.  -> results/cost_all.csv
"""
import os, sys, time, csv
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import analysis as A
import serve_bench as SB
from train import s2s_decode

torch.set_num_threads(1)
N = int(os.environ.get("N_LAT", "400"))


def lat_tp(fn, its):
    for i in range(32):
        fn(its[i:i + 1])
    lat = []
    for i in range(N):
        t = time.perf_counter(); fn(its[i:i + 1]); lat.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter()
    for i in range(0, N, 64):
        fn(its[i:i + 64])
    tp = N / (time.perf_counter() - t)
    return np.array(lat), tp


def main():
    its = A.items("test")
    agg = {r["model"]: r for r in csv.DictReader(open(os.path.join(RES, "main_agg.csv")))}
    rows = []
    for name in ("bilstm", "trans_scratch", "pt_tagger", "stack_hybrid", "ours"):
        if not os.path.exists(os.path.join(CK, f"{name}_s1.pt")):
            continue
        m = [SB.load_tagger(f"{name}_s1", name)]
        lat, tp = lat_tp(lambda b: SB.tier_probs(m, b), its)
        rows.append({"model": name, "params_M": sum(p.numel() for p in m[0][0].parameters()) / 1e6,
                     "train_min": float(agg[name]["train_time"]) / 60, "p50_ms": np.percentile(lat, 50),
                     "p99_ms": np.percentile(lat, 99), "mean_ms": lat.mean(), "tp_b64": tp})
        print(rows[-1], flush=True)
    heavy = [SB.load_tagger(f"ours_s{s}") for s in (2, 3, 4, 5)]
    lat, tp = lat_tp(lambda b: SB.tier_probs(heavy, b), its)
    rows.append({"model": "ours_ensemble4", "params_M": 4 * rows[-1]["params_M"], "train_min": 4 * rows[-1]["train_min"],
                 "p50_ms": np.percentile(lat, 50), "p99_ms": np.percentile(lat, 99), "mean_ms": lat.mean(), "tp_b64": tp})
    if not os.path.exists(os.path.join(CK, "seq2seq_s1.pt")):
        A.write_csv("cost_all.csv", rows); return
    s2s = Seq2SeqCopy(SB.V); s2s.load_state_dict(torch.load(os.path.join(CK, "seq2seq_s1.pt"), map_location="cpu")); s2s.eval()
    with torch.no_grad():
        lat, tp = lat_tp(lambda b: s2s_decode(s2s, SB.V, b, bs=64), its)
    rows.append({"model": "seq2seq", "params_M": sum(p.numel() for p in s2s.parameters()) / 1e6,
                 "train_min": float(agg["seq2seq"]["train_time"]) / 60, "p50_ms": np.percentile(lat, 50),
                 "p99_ms": np.percentile(lat, 99), "mean_ms": lat.mean(), "tp_b64": tp})
    print(rows[-1], flush=True)
    A.write_csv("cost_all.csv", rows)


if __name__ == "__main__":
    with torch.no_grad():
        main()
