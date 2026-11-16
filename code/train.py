"""Train one configuration with one seed and dump token-level predictions.

usage: python train.py --model ours --seed 1
"""
import os, glob, sys, time, json, argparse, random
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from gec_data import apply_tags

CFG = {
    # proposed
    "ours":           dict(enc="hybrid", rnn="gru", fusion="gate", detect=True, rhet=True, pre=True),
    # re-implemented baselines (identical data, tag set, epochs, optimiser)
    "bilstm":         dict(enc="rnn", rnn="lstm", fusion=None, detect=False, rhet=False, pre=False),
    "trans_scratch":  dict(enc="trans", rnn=None, fusion=None, detect=False, rhet=False, pre=False),
    "pt_tagger":      dict(enc="trans", rnn=None, fusion=None, detect=False, rhet=False, pre=True),
    "stack_hybrid":   dict(enc="hybrid", rnn="lstm", fusion="stack", detect=False, rhet=False, pre=True),
    # ablations
    "abl_nopre":      dict(enc="hybrid", rnn="gru", fusion="gate", detect=True, rhet=True, pre=False),
    "abl_nornn":      dict(enc="trans", rnn=None, fusion=None, detect=True, rhet=True, pre=True),
    "abl_concat":     dict(enc="hybrid", rnn="gru", fusion="concat", detect=True, rhet=True, pre=True),
    "abl_lstm":       dict(enc="hybrid", rnn="lstm", fusion="gate", detect=True, rhet=True, pre=True),
    "abl_nodet":      dict(enc="hybrid", rnn="gru", fusion="gate", detect=False, rhet=True, pre=True),
    "abl_norhet":     dict(enc="hybrid", rnn="gru", fusion="gate", detect=True, rhet=False, pre=True),
    "abl_quranclean": dict(enc="hybrid", rnn="gru", fusion="gate", detect=True, rhet=False, pre=True, qclean=True),
}

EPOCHS, BS, LR = 3, 64, 6e-4
# optional overrides (defaults reproduce the paper): VARIANT selects alternative training data
# (gec_train<VARIANT>.json, mlm<VARIANT>.pt); LR / DROP / LAMBDA / RHW support the hyperparameter search
VARIANT = os.environ.get("VARIANT", "")
LR = float(os.environ.get("LR", LR)); DROP = float(os.environ.get("DROP", 0.1))
LAMBDA = float(os.environ.get("LAMBDA", 0.5)); RHW = float(os.environ.get("RHW", 8.0))
TAG_SUFFIX = os.environ.get("TAG_SUFFIX", "")


def quran_items(verses, split, sp, chunk=30):
    out = []
    for v in verses:
        if sp[(v["sura"], v["aya"])] != split:
            continue
        t = v["tokens"]; rh = set(v["rhetorical"]); ist = set(v["istiara"])
        for s in range(0, len(t), chunk):
            seg = t[s:s + chunk]
            if len(seg) < 3:
                continue
            out.append({"noisy": seg, "clean": seg, "tags": ["KEEP"] * len(seg),
                        "rhet": [int(k in rh) for k in range(s, s + len(seg))],
                        "ist": [int(k in ist) for k in range(s, s + len(seg))],
                        "ref": [v["sura"], v["aya"], s]})
    return out


@torch.no_grad()
def predict(model, V, items, T=1.0, bs=128):
    model.eval()
    res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        W, C, M, *_ = batchify(V, chunk, with_tags=False)
        o = model(W, C, M, return_gate=True)
        z = o["tag"]
        p1 = z.softmax(-1); pT = (z / T).softmax(-1)
        zz = z.clone(); zz[..., 0] = -1e9
        best = zz.argmax(-1)
        rh = o["rhet"].softmax(-1)[..., 1] if "rhet" in o else torch.zeros_like(p1[..., 0])
        dt = o["det"].softmax(-1)[..., 1] if "det" in o else torch.zeros_like(p1[..., 0])
        g = o["gate"].mean(-1) if o.get("gate") is not None else torch.zeros_like(p1[..., 0])
        for b, x in enumerate(chunk):
            l = int(M[b].sum())
            res.append({"pk1": p1[b, :l, 0].numpy().astype(np.float32),
                        "pkT": pT[b, :l, 0].numpy().astype(np.float32),
                        "best": best[b, :l].numpy().astype(np.int32),
                        "bestpT": pT[b, :l].gather(-1, best[b, :l, None])[:, 0].numpy().astype(np.float32),
                        "bestp1": p1[b, :l].gather(-1, best[b, :l, None])[:, 0].numpy().astype(np.float32),
                        "argmaxT": pT[b, :l].argmax(-1).numpy().astype(np.int32),
                        "rhet": rh[b, :l].numpy().astype(np.float32),
                        "det": dt[b, :l].numpy().astype(np.float32),
                        "gate": g[b, :l].numpy().astype(np.float32)})
    return res


def fit_temperature(model, V, items):
    """Temperature scaling on the dev split (NLL of gold tags)."""
    model.eval(); zs, ys = [], []
    with torch.no_grad():
        for i in range(0, len(items), 128):
            W, C, M, T, *_ = batchify(V, items[i:i + 128])
            z = model(W, C, M)["tag"]; m = T != -100
            zs.append(z[m]); ys.append(T[m])
    z = torch.cat(zs); y = torch.cat(ys)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    def clo():
        opt.zero_grad(); l = F.cross_entropy(z / logT.exp(), y); l.backward(); return l
    opt.step(clo)
    return float(logT.detach().exp())


def train_tagger(name, seed, epochs=EPOCHS, out_tag=None):
    cfg = CFG[name]
    set_seed(seed)
    V = Vocab()
    model = Tagger(V, enc=cfg["enc"], rnn=cfg["rnn"] or "gru", fusion=cfg["fusion"] or "gate",
                   detect=cfg["detect"], rhet=cfg["rhet"], drop=DROP)
    if cfg["pre"]:
        model.load_pretrained(os.path.join(CK, f"mlm{VARIANT}.pt"))
    train = load(f"gec_train{VARIANT}.json"); dev = load("gec_dev.json")
    verses = load("quran.json"); sp = quran_splits(verses)
    q_tr = quran_items(verses, "train", sp)
    if os.environ.get("SMOKE"):
        train = train[:640]; dev = dev[:100]
    if cfg.get("qclean"):
        train = train + [{k: x[k] for k in ("noisy", "clean", "tags")} for x in q_tr]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * (len(train) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.08)
    ntag = len(V.tags)
    w_rh = torch.tensor([1.0, RHW])
    t0 = time.time(); step = 0; qi = 0
    rng = random.Random(seed)
    for ep in range(epochs):
        rng.shuffle(train); rng.shuffle(q_tr)
        model.train()
        for i in range(0, len(train) - BS + 1, BS):
            W, C, M, T, D, _ = batchify(V, train[i:i + BS])
            o = model(W, C, M)
            loss = F.cross_entropy(o["tag"].reshape(-1, ntag), T.reshape(-1), ignore_index=-100)
            if "det" in o:
                loss = loss + LAMBDA * F.cross_entropy(o["det"].reshape(-1, 2), D.reshape(-1), ignore_index=-100)
            if "rhet" in o and step % 2 == 0:
                qb = q_tr[qi:qi + 32]; qi = (qi + 32) % max(len(q_tr) - 32, 1)
                W2, C2, M2, _, _, R2 = batchify(V, qb, rhet=True)
                o2 = model(W2, C2, M2)
                loss = loss + LAMBDA * F.cross_entropy(o2["rhet"].reshape(-1, 2), R2.reshape(-1),
                                                    weight=w_rh, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"[{name} s{seed}] ep{ep} step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    train_time = time.time() - t0
    Tm = fit_temperature(model, V, dev)
    tag = out_tag or f"{name}{VARIANT}{TAG_SUFFIX}_s{seed}"
    torch.save(model.state_dict(), os.path.join(CK, tag + ".pt"))
    dump = {"model": name, "seed": seed, "T": Tm, "train_time": train_time,
            "params": sum(p.numel() for p in model.parameters())}
    sets = {s: load(f"gec_{s}.json") for s in ["dev", "cal", "test", "xdom"]}
    for r in ["05", "10", "20", "25", "30"]:
        sets["test_r" + r] = load(f"gec_test_r{r}.json")
    sets["quran"] = quran_items(verses, "test", sp)
    sets["quran_dev"] = quran_items(verses, "dev", sp)
    for f in sorted(glob.glob(os.path.join(P, "gec_md_*.json"))):     # extra held-out works
        sets[os.path.basename(f)[4:-5]] = json.load(open(f))
    if os.environ.get("SMOKE"):
        sets = {k: v[:50] for k, v in sets.items()}
    for s, items in sets.items():
        dump[s] = predict(model, V, items, T=Tm)
    os.makedirs(os.path.join(RES, "preds"), exist_ok=True)
    torch.save(dump, os.path.join(RES, "preds", tag + ".pt"))
    print(f"done {tag} T={Tm:.3f} train_time={train_time:.0f}s", flush=True)


# ------------------------------------------------------------------ seq2seq baseline
def ext_ids(V, toks):
    oov = {}
    ids = []
    for w in toks:
        i = V.wid(w)
        if i == UNK:
            i = len(V.itos) + oov.setdefault(w, len(oov))
        ids.append(i)
    return ids, oov


def s2s_batch(V, items, with_tgt=True):
    W, C, M, *_ = batchify(V, items, with_tags=False)
    n, L = W.shape
    src_ext = torch.zeros(n, L, dtype=torch.long); oovs = []
    for i, x in enumerate(items):
        ids, oov = ext_ids(V, x["noisy"][:MAXW]); src_ext[i, :len(ids)] = torch.tensor(ids); oovs.append(oov)
    n_ext = max(len(o) for o in oovs) if oovs else 0
    if not with_tgt:
        return W, C, M, src_ext, n_ext, oovs
    Lt = max(len(x["clean"][:MAXW + 4]) for x in items) + 1
    Yin = torch.zeros(n, Lt, dtype=torch.long); Yout = torch.full((n, Lt), -100, dtype=torch.long)
    for i, x in enumerate(items):
        tgt = []
        for w in x["clean"][:MAXW + 4]:
            j = V.wid(w)
            if j == UNK and w in oovs[i]:
                j = len(V.itos) + oovs[i][w]
            tgt.append(j)
        seq_in = [BOS] + [t if t < len(V.itos) else UNK for t in tgt]
        seq_out = tgt + [EOS]
        Yin[i, :len(seq_in)] = torch.tensor(seq_in); Yout[i, :len(seq_out)] = torch.tensor(seq_out)
    return W, C, M, src_ext, n_ext, oovs, Yin[:, :Lt], Yout[:, :Lt]


@torch.no_grad()
def s2s_decode(model, V, items, bs=64):
    model.eval(); out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        W, C, M, src_ext, n_ext, oovs = s2s_batch(V, chunk, with_tgt=False)
        mem = model.encode(W, C, M)
        n = W.size(0)
        Y = torch.full((n, 1), BOS, dtype=torch.long); res = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool)
        maxlen = W.size(1) + 4
        for t in range(maxlen):
            h, att, pgen = model.decode_step(mem, M, Y)
            pv = model.out(h[:, -1]).softmax(-1) * pgen[:, -1]
            pv = torch.cat([pv, torch.zeros(n, n_ext)], -1)
            pv = pv.scatter_add(1, src_ext, (1 - pgen[:, -1]) * att[:, -1])
            nxt = pv.argmax(-1)
            for b in range(n):
                if done[b]:
                    continue
                k = int(nxt[b])
                if k == EOS:
                    done[b] = True; continue
                if k >= len(V.itos):
                    inv = {v: w for w, v in oovs[b].items()}; res[b].append(inv.get(k - len(V.itos), "<unk>"))
                else:
                    res[b].append(V.itos[k])
            if done.all():
                break
            Y = torch.cat([Y, nxt.clamp(max=len(V.itos) - 1)[:, None]], 1)
        out.extend(res)
    return out


def train_s2s(seed, epochs=EPOCHS):
    set_seed(seed); V = Vocab()
    model = Seq2SeqCopy(V)
    # initialise encoder embeddings from the pretrained MLM (same budget as pretrained taggers)
    sd = torch.load(os.path.join(CK, f"mlm{VARIANT}.pt"), map_location="cpu")
    model.emb.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("emb.")})
    train = load(f"gec_train{VARIANT}.json")
    if os.environ.get("SMOKE"):
        train = train[:640]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * (len(train) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.08)
    rng = random.Random(seed); t0 = time.time(); step = 0
    for ep in range(epochs):
        rng.shuffle(train); model.train()
        for i in range(0, len(train) - BS + 1, BS):
            W, C, M, src_ext, n_ext, _, Yin, Yout = s2s_batch(V, train[i:i + BS])
            pv = model(W, C, M, Yin, src_ext, n_ext)
            m = Yout != -100
            loss = -torch.log(pv[m].gather(-1, Yout[m][:, None]).clamp(min=1e-9)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"[s2s s{seed}] step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    train_time = time.time() - t0
    tag = f"seq2seq{VARIANT}{TAG_SUFFIX}_s{seed}"
    torch.save(model.state_dict(), os.path.join(CK, tag + ".pt"))
    verses = load("quran.json"); sp = quran_splits(verses)
    dump = {"model": "seq2seq", "seed": seed, "train_time": train_time,
            "params": sum(p.numel() for p in model.parameters())}
    sets = {s: load(f"gec_{s}.json") for s in ["test", "xdom"]}
    for r in ["05", "10", "20", "25", "30"]:
        sets["test_r" + r] = load(f"gec_test_r{r}.json")
    sets["quran"] = quran_items(verses, "test", sp)
    if os.environ.get("SMOKE"):
        sets = {k: v[:50] for k, v in sets.items()}
    for s, items in sets.items():
        dump[s] = s2s_decode(model, V, items)
    os.makedirs(os.path.join(RES, "preds"), exist_ok=True)
    torch.save(dump, os.path.join(RES, "preds", tag + ".pt"))
    print(f"done {tag} train_time={train_time:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    if a.model == "seq2seq":
        train_s2s(a.seed, a.epochs)
    else:
        train_tagger(a.model, a.seed, a.epochs)
