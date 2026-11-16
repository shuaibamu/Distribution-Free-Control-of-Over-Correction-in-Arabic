"""Evaluation on four additional held-out classical works (authors, genres and centuries not seen
in training), with and without recalibration of the conformal threshold on the target work.

  build  : extract sentences from data/openiti_extra/*, corrupt them with the same generator,
           write data/processed/md_<work>_{cal,test}.json (1,000 calibration + 2,000 test each)
  eval   : all taggers (argmax), seq2seq (greedy), and RACE under conformal risk control with
           (a) the in-domain threshold and (b) a threshold recalibrated on 1,000 target sentences
           -> results/multidomain.csv, results/multidomain_crc.csv
usage: python3 code/multidomain.py build eval
"""
import os, sys, json, random, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from gec_data import corrupt, apply_tags
import prep_data as PD
import analysis as A
from analysis import TAGS

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
EXTRA = os.path.join(ROOT, "data", "openiti_extra")
WORKS = {  # key: (file prefix, century AH, genre)
    "Jahiz": ("0255Jahiz.BayanWaTabyin", 3, "adab / rhetoric"),
    "Mubarrad": ("0285IbnYazidMubarrad.KamilFiLugha", 3, "philology"),
    "IbnAbdRabbih": ("0328IbnCabdRabbih.CiqdFarid", 4, "adab anthology"),
    "IbnKhaldun": ("0808IbnKhaldun.Tarikh", 9, "history"),
}
N_CAL, N_TEST = 1000, 2000
ALPHAS = [0.003, 0.005, 0.01]


def build():
    conf = json.load(open(os.path.join(ROOT, "data", "processed", "confusions.json")))
    seen = {" ".join(s) for s in load("pretrain.json")}
    for k, (pre, _, _) in WORKS.items():
        path = glob.glob(os.path.join(EXTRA, pre + "*"))[0]
        random.seed(11)
        sents = [s for s in PD.sentences_from(path, k, 20000) if " ".join(s["tokens"]) not in seen]
        sents = sents[:N_CAL + N_TEST]
        rng = random.Random(300 + len(k))
        data = []
        for s in sents:
            n, t, f = corrupt(s["tokens"], rng, conf, rate=0.15)
            data.append({"src": k, "clean": s["tokens"], "noisy": n, "tags": t, "fam": f})
        assert all(apply_tags(x["noisy"], x["tags"]) == x["clean"] for x in data)
        json.dump(data[:N_CAL], open(os.path.join(ROOT, "data", "processed", f"gec_md_{k}_cal.json"), "w"), ensure_ascii=False)
        json.dump(data[N_CAL:], open(os.path.join(ROOT, "data", "processed", f"gec_md_{k}_test.json"), "w"), ensure_ascii=False)
        print(k, len(data), "sentences", sum(len(x["clean"]) for x in data), "words")


V = None


def tagger_preds(name, seed, its, bs=128):
    """Per-sentence dicts in the same format as results/preds (argmaxT, bestpT, rhet)."""
    global V
    import serve_bench as SB
    m, T = SB.load_tagger(f"{name}_s{seed}", name)
    out = []
    with torch.no_grad():
        for i in range(0, len(its), bs):
            W, C, M, *_ = batchify(SB.V, its[i:i + bs], with_tags=False)
            o = m(W, C, M)
            P = (o["tag"] / T).softmax(-1)
            R = o["rhet"].softmax(-1)[..., 1] if "rhet" in o and o["rhet"] is not None else torch.zeros(P.shape[:2])
            PP = P.clone(); PP[..., 0] = 0
            pe, _ = PP.max(-1)
            am = P.argmax(-1)
            for b in range(P.size(0)):
                l = int(M[b].sum())
                out.append({"argmaxT": am[b, :l].numpy(), "bestpT": pe[b, :l].numpy(), "rhet": R[b, :l].numpy(),
                            "bestT": PP[b, :l].argmax(-1).numpy()})
    return out


def hyps(its, preds, tau=None, beta=0.0):
    res = []
    for x, p in zip(its, preds):
        if tau is None:
            tg = [TAGS[i] for i in p["argmaxT"]]
        else:
            app = p["bestpT"] > tau + beta * p["rhet"]
            tg = [TAGS[int(b)] if a else "KEEP" for a, b in zip(app, p["bestT"])]
        res.append(apply_tags(x["noisy"][:MAXW], tg))
    return res


def score(its, hy):
    src = [x["noisy"][:MAXW] for x in its]
    gold = [apply_tags(x["noisy"][:MAXW], x["tags"][:MAXW]) for x in its]
    return score_corpus(src, hy, gold)


def loss_mat(its, preds, beta):
    rows = []
    for x, p in zip(its, preds):
        n = len(p["bestpT"])
        keep = np.array([t == "KEEP" for t in x["tags"][:n]])
        v = np.sort((p["bestpT"] - beta * p["rhet"])[keep])
        rows.append((len(v) - np.searchsorted(v, A.TAU_GRID, side="right")) / n)
    return np.array(rows)


def crc_tau(L, alpha):
    n = len(L)
    ok = (n / (n + 1)) * L.mean(0) + 1 / (n + 1) <= alpha
    return float(A.TAU_GRID[np.argmax(ok)]) if ok.any() else 1.0


def evaluate():
    import serve_bench as SB
    from train import s2s_decode
    sets = {k: (load(f"gec_md_{k}_cal.json"), load(f"gec_md_{k}_test.json")) for k in WORKS}
    sets["Jurjani"] = (None, A.items("xdom"))
    sets["in-domain"] = (A.items("cal"), A.items("test"))
    rows, crows = [], []
    models = [("ours", s) for s in A.SEEDS] + [(m, s) for m in ("pt_tagger", "trans_scratch", "stack_hybrid", "bilstm")
                                                 for s in A.SEEDS]
    for name, seed in models:
        if not os.path.exists(os.path.join(CK, f"{name}_s{seed}.pt")):
            continue
        if name == "ours":
            d_in = A.dump(name, seed)
            Lin = {beta: loss_mat(A.items("cal"), d_in["cal"], beta) for beta in (0.0, A.BETA)}
            A._dump.pop((name, seed), None)
        for k, (cal, test) in sets.items():
            pt = tagger_preds(name, seed, test)
            e = score(test, hyps(test, pt))
            rows.append({"model": name, "seed": seed, "domain": k, "F05": e["F05"], "P": e["P"], "R": e["R"]})
            if name != "ours":
                continue
            pc = tagger_preds(name, seed, cal) if cal is not None else None
            for beta in (0.0, A.BETA):
                Lt = loss_mat(test, pt, beta)
                for a in ALPHAS:
                    rec = {"seed": seed, "domain": k, "beta": beta, "alpha": a}
                    t_in = crc_tau(Lin[beta], a)
                    i_in = int(np.argmin(np.abs(A.TAU_GRID - t_in)))
                    rec.update(tau_in=t_in, risk_in=float(Lt[:, i_in].mean()),
                               F05_in=score(test, hyps(test, pt, t_in, beta))["F05"])
                    if pc is not None:
                        t_d = crc_tau(loss_mat(cal, pc, beta), a)
                        i_d = int(np.argmin(np.abs(A.TAU_GRID - t_d)))
                        rec.update(tau_rec=t_d, risk_rec=float(Lt[:, i_d].mean()),
                                   F05_rec=score(test, hyps(test, pt, t_d, beta))["F05"])
                    crows.append(rec)
        print(name, seed, flush=True)
    # seq2seq: greedy decoding
    from common import Seq2SeqCopy
    for s in A.SEEDS:
        f = os.path.join(CK, f"seq2seq_s{s}.pt")
        if not os.path.exists(f):
            continue
        m = Seq2SeqCopy(SB.V); m.load_state_dict(torch.load(f, map_location="cpu")); m.eval()
        for k, (cal, test) in sets.items():
            with torch.no_grad():
                out = s2s_decode(m, SB.V, test)
            e = score(test, [o[:MAXW] for o in out])
            rows.append({"model": "seq2seq", "seed": s, "domain": k, "F05": e["F05"], "P": e["P"], "R": e["R"]})
        print("seq2seq", s, flush=True)
    A.write_csv("multidomain.csv", rows)
    A.write_csv("multidomain_crc.csv", crows)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["build", "eval"]:
        {"build": build, "eval": evaluate}[w]()
