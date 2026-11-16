"""Pretrained Arabic encoder (Hugging Face) fine-tuned as a GECToR-style edit tagger on the same data,
tag set and splits as RACE.  Writes results/preds/<tag>_s<seed>.pt in the same format as train.py,
so code/cs/eval_tags.py and conformal risk control apply unchanged.

Suggested encoders: CAMeL-Lab/bert-base-arabic-camelbert-ca (Classical Arabic),
                    aubmindlab/bert-base-arabertv02, UBC-NLP/MARBERTv2
usage: python3 code/cs/hf_tagger.py --hf CAMeL-Lab/bert-base-arabic-camelbert-ca --tag hf_camelbert_ca --seed 1
"""
import os, sys, time, json, glob, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import *
from train import quran_items

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

dev_ = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
V = Vocab()
tok = AutoTokenizer.from_pretrained(a.hf)


class HFTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(a.hf)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, len(V.tags))

    def forward(self, enc, first):
        H = self.enc(**enc).last_hidden_state                       # B x S x h
        idx = first.clamp(min=0)[..., None].expand(-1, -1, H.size(-1))
        Hw = H.gather(1, idx)                                         # B x W x h (first sub-word)
        return self.head(self.drop(Hw))


def encode(batch):
    words = [x["noisy"][:MAXW] for x in batch]
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=a.max_len,
              padding=True, return_tensors="pt")
    W = max(len(w) for w in words)
    first = torch.full((len(batch), W), -1, dtype=torch.long)
    for b in range(len(batch)):
        seen = set()
        for p, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen.add(wid); first[b, wid] = p
    T = torch.full((len(batch), W), -100, dtype=torch.long)
    for b, x in enumerate(batch):
        if "tags" in x:
            for i, t in enumerate(x["tags"][:MAXW]):
                if first[b, i] >= 0:
                    T[b, i] = V.ttoi.get(t, -100)
    return {k: v.to(dev_) for k, v in enc.items()}, first.to(dev_), T.to(dev_)


model = HFTagger().to(dev_)
train = load("gec_train.json"); dev = load("gec_dev.json")
if a.smoke:
    train, dev = train[:256], dev[:64]
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * (len(train) // a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler(enabled=dev_ == "cuda")
t0 = time.time(); step = 0
for ep in range(a.epochs):
    random.shuffle(train); model.train()
    for i in range(0, len(train) - a.bs + 1, a.bs):
        enc, first, T = encode(train[i:i + a.bs])
        with torch.autocast(dev_, enabled=dev_ == "cuda"):
            logit = model(enc, first)
            loss = F.cross_entropy(logit.float().reshape(-1, len(V.tags)), T.reshape(-1), ignore_index=-100)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step(); step += 1
        if step % 200 == 0:
            print(f"[{a.tag} s{a.seed}] ep{ep} {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
train_time = time.time() - t0


@torch.no_grad()
def logits_for(items, bs=128):
    model.eval(); out = []
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        enc, first, _ = encode(b)
        lg = model(enc, first).float().cpu()
        for k, x in enumerate(b):
            out.append(lg[k, :min(len(x["noisy"]), MAXW)])
    return out


# temperature scaling on the development split (same procedure as train.fit_temperature)
dl = logits_for(dev)
Z = torch.cat(dl); Y = torch.cat([torch.tensor([V.ttoi.get(t, -100) for t in x["tags"][:MAXW]]) for x in dev])
m = Y >= 0
logT = torch.zeros(1, requires_grad=True)
o = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
def closure():
    o.zero_grad(); l = F.cross_entropy(Z[m] / logT.exp(), Y[m]); l.backward(); return l
o.step(closure)
Tm = float(logT.exp())

verses = load("quran.json"); sp = quran_splits(verses)
sets = {s: load(f"gec_{s}.json") for s in ["dev", "cal", "test", "xdom"]}
sets["quran"] = quran_items(verses, "test", sp)
for f in sorted(glob.glob(os.path.join(P, "gec_md_*.json"))):
    sets[os.path.basename(f)[4:-5]] = json.load(open(f))
if a.smoke:
    sets = {k: v[:40] for k, v in sets.items()}
dump = {"model": a.tag, "seed": a.seed, "T": Tm, "train_time": train_time, "hf": a.hf,
        "params": sum(p.numel() for p in model.parameters()), "device": dev_}
t1 = time.time(); n_inf = 0
for name, its in sets.items():
    res = []
    for lg in logits_for(its):
        P1 = lg.softmax(-1); PT = (lg / Tm).softmax(-1)
        PP = PT.clone(); PP[:, 0] = 0
        pe, best = PP.max(-1)
        res.append({"argmaxT": PT.argmax(-1).numpy(), "bestpT": pe.numpy().astype(np.float32),
                    "best": best.numpy(), "bestp1": P1.clone().index_fill_(1, torch.tensor([0]), 0).max(-1).values.numpy(),
                    "rhet": np.zeros(len(pe), np.float32)})
    dump[name] = res; n_inf += len(its)
dump["infer_sent_per_s"] = n_inf / (time.time() - t1)
os.makedirs(os.path.join(RES, "preds"), exist_ok=True)
torch.save(dump, os.path.join(RES, "preds", f"{a.tag}_s{a.seed}.pt"))
print(f"done {a.tag} s{a.seed} T={Tm:.3f} train_time={train_time:.0f}s", flush=True)
