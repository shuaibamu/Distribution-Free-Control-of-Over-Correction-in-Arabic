"""Measured serving behaviour of the light tier (single RACE model), the heavy
tier (4-model RACE ensemble) and the seq2seq baseline on this machine, plus the
uncertainty-routed cascade evaluated end to end.

Outputs:
  results/latency_samples.csv     per-request latency (batch 1, 1 thread)
  results/batch_throughput.csv    throughput vs batch size
  results/worker_scaling.csv      measured throughput with 1 and 2 worker processes
  results/routing.csv             cascade quality/cost vs routing threshold
  results/payload.json            request/response sizes
"""
import os, sys, time, json, math
import numpy as np
import torch
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from gec_data import apply_tags
from train import CFG, quran_items, s2s_decode
import analysis as A

torch.set_num_threads(1)
V = Vocab()


def load_tagger(tag, name="ours"):
    c = CFG[name]
    m = Tagger(V, enc=c["enc"], rnn=c["rnn"] or "gru", fusion=c["fusion"] or "gate", detect=c["detect"], rhet=c["rhet"])
    m.load_state_dict(torch.load(os.path.join(CK, tag + ".pt"), map_location="cpu")); m.eval()
    f = os.path.join(RES, "preds", tag + ".pt")
    if os.path.exists(f):
        T = torch.load(f, weights_only=False)["T"]
    else:                                   # temperatures of the released checkpoints
        T = json.load(open(os.path.join(RES, "temperatures.json")))[tag]
    return m, T


@torch.no_grad()
def tier_probs(models, batch):
    """Average calibrated tag distribution and rhetoric probability over models."""
    W, C, M, *_ = batchify(V, batch, with_tags=False)
    P = 0; R = 0
    for m, T in models:
        o = m(W, C, M)
        P = P + (o["tag"] / T).softmax(-1); R = R + (o["rhet"].softmax(-1)[..., 1] if "rhet" in o else 0.0)
    P = P / len(models); R = R / len(models)
    return P, R, M


def decode(P, R, M, tau, beta=A.BETA):
    out = []
    PP = P.clone(); PP[..., 0] = 0
    pe, best = PP.max(-1)
    for b in range(P.size(0)):
        l = int(M[b].sum())
        app = (pe[b, :l] > tau + beta * R[b, :l])
        out.append(([V.tags[int(t)] if a else "KEEP" for a, t in zip(app, best[b, :l])],
                    pe[b, :l].numpy(), R[b, :l].numpy()))
    return out


def run_tier(models, its, tau, bs=128):
    res = []
    for i in range(0, len(its), bs):
        P, R, M = tier_probs(models, its[i:i + bs])
        res.extend(decode(P, R, M, tau))
    return res


def crc_tau(models, alpha=A.ALPHA_MAIN):
    its = A.items("cal"); n = len(its)
    pes, rs, gk = [], [], []
    for i in range(0, n, 128):
        P, R, M = tier_probs(models, its[i:i + 128])
        for b in range(P.size(0)):
            l = int(M[b].sum()); PP = P[b, :l].clone(); PP[:, 0] = 0
            pes.append(PP.max(-1).values.numpy()); rs.append(R[b, :l].numpy())
    for x, pe in zip(its, pes):
        gk.append(np.array([t == "KEEP" for t in x["tags"][:len(pe)]]))
    for tau in A.TAU_GRID:
        Lm = np.mean([np.mean((pe > tau + A.BETA * r) & g) for pe, r, g in zip(pes, rs, gk)])
        if (n / (n + 1)) * Lm + 1 / (n + 1) <= alpha:
            return float(tau)
    return 1.0


def entropy_u(pe):
    p = np.clip(pe, 1e-6, 1 - 1e-6)
    return float(np.max(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def latency_samples(light, heavy, s2s, its, n=400):
    rows = []
    for x in its[:n]:
        for tier, fn in [("light", lambda b: tier_probs(light, b)), ("heavy", lambda b: tier_probs(heavy, b)),
                         ("seq2seq", lambda b: s2s_decode(s2s, V, b, bs=1))]:
            t0 = time.perf_counter(); fn([x]); dt = time.perf_counter() - t0
            rows.append({"tier": tier, "len": len(x["noisy"]), "ms": dt * 1000})
    A.write_csv("latency_samples.csv", rows)
    return rows


def batch_throughput(light, its):
    rows = []
    for bs in [1, 4, 16, 32, 64, 128]:
        sub = its[:max(256, bs * 4)]
        t0 = time.perf_counter()
        for i in range(0, len(sub), bs):
            tier_probs(light, sub[i:i + bs])
        dt = time.perf_counter() - t0
        rows.append({"batch": bs, "sent_per_s": len(sub) / dt})
    A.write_csv("batch_throughput.csv", rows)


def _worker(args):
    tag, chunk = args
    torch.set_num_threads(1)
    m, T = load_tagger(tag)
    for i in range(0, 128, 32):          # warm-up, not timed
        tier_probs([(m, T)], chunk[i:i + 32])
    reps = []
    for _ in range(3):                   # median of three timed passes
        t0 = time.perf_counter()
        for i in range(0, len(chunk), 32):
            tier_probs([(m, T)], chunk[i:i + 32])
        reps.append(time.perf_counter() - t0)
    return float(np.median(reps))


def worker_scaling(its, n=2048):
    rows = []
    sub = its[:n]
    for w in [1, 2]:
        parts = [sub[k::w] for k in range(w)]
        ctx = mp.get_context("fork")
        t0 = time.perf_counter()
        with ctx.Pool(w) as pool:
            busy = pool.map(_worker, [("ours_s1", p) for p in parts])
        wall = time.perf_counter() - t0
        rows.append({"workers": w, "sent_per_s": n / max(busy), "wall_s": wall, "max_busy_s": max(busy)})
    A.write_csv("worker_scaling.csv", rows)


def routing(light, heavy, lat):
    tl = crc_tau(light); th = crc_tau(heavy)
    out = {}
    for split in ["test", "quran", "xdom"]:
        its = A.items(split)
        out[split] = (run_tier(light, its, tl), run_tier(heavy, its, th))
    ms = {t: np.mean([r["ms"] for r in lat if r["tier"] == t]) for t in ("light", "heavy")}
    rows = []
    for theta in [1.01, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        rec = {"theta": theta}
        for split in ["test", "quran", "xdom"]:
            its = A.items(split); L, H = out[split]
            hyps = []; routed = 0
            for x, l, h in zip(its, L, H):
                use_h = entropy_u(l[1]) > theta
                routed += use_h
                hyps.append(apply_tags(x["noisy"][:MAXW], (h if use_h else l)[0]))
            if split == "quran":
                q = A.quran_ocr(hyps); rec["Q_rhet"] = q["rhet"]; rec["Q_all"] = q["all"]
            else:
                e = A.evaluate(split, hyps)
                rec[f"F05_{split}"] = e["F05"]
                if split == "test":
                    rec["routed"] = routed / len(its)
                    rec["ms_per_sent"] = ms["light"] + rec["routed"] * ms["heavy"]
        rows.append(rec)
    A.write_csv("routing.csv", rows)
    json.dump({"tau_light": tl, "tau_heavy": th, "ms_light": ms["light"], "ms_heavy": ms["heavy"]},
              open(os.path.join(RES, "routing_meta.json"), "w"), indent=1)


def payload(its):
    req = [len(json.dumps({"id": i, "text": " ".join(x["noisy"])}, ensure_ascii=False).encode()) for i, x in enumerate(its[:1000])]
    resp = [len(json.dumps({"id": i, "text": " ".join(x["clean"]), "edits": [], "conf": 0.0}, ensure_ascii=False).encode())
            for i, x in enumerate(its[:1000])]
    json.dump({"req_bytes_mean": float(np.mean(req)), "resp_bytes_mean": float(np.mean(resp))},
              open(os.path.join(RES, "payload.json"), "w"), indent=1)


if __name__ == "__main__":
    its = A.items("test")
    light = [load_tagger("ours_s1")]
    heavy = [load_tagger(f"ours_s{s}") for s in (2, 3, 4, 5)]
    s2s = Seq2SeqCopy(V); s2s.load_state_dict(torch.load(os.path.join(CK, "seq2seq_s1.pt"), map_location="cpu")); s2s.eval()
    if os.environ.get("ONLY") == "workers":
        worker_scaling(its); sys.exit(0)
    payload(its)
    worker_scaling(its)
    batch_throughput(light, its)
    lat = latency_samples(light, heavy, s2s, its)
    routing(light, heavy, lat)
    print("serve bench done")
