"""Fine-tunes a pretrained Arabic sequence-to-sequence model (e.g. moussaKam/AraBART,
UBC-NLP/AraT5v2-base-1024) on the same noisy->clean pairs and writes greedy hypotheses with
per-word generation confidences to results/gen/<tag>_s<seed>.json for code/cs/gen_eval.py.
usage: python3 code/cs/hf_seq2seq.py --hf moussaKam/AraBART --tag arabart --seed 1
"""
import os, sys, time, json, glob, argparse, random
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from train import quran_items
import gen_control as G

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_len", type=int, default=96)
ap.add_argument("--prefix", default="")          # e.g. "gec: " for T5-style models
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

dev_ = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.hf)
model = AutoModelForSeq2SeqLM.from_pretrained(a.hf).to(dev_)
train = load("gec_train.json")
if a.smoke:
    train = train[:256]
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * (len(train) // a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
t0 = time.time(); step = 0
for ep in range(a.epochs):
    random.shuffle(train); model.train()
    for i in range(0, len(train) - a.bs + 1, a.bs):
        b = train[i:i + a.bs]
        enc = tok([a.prefix + " ".join(x["noisy"][:MAXW]) for x in b], truncation=True, max_length=a.max_len,
                  padding=True, return_tensors="pt", return_token_type_ids=False).to(dev_)
        lab = tok(text_target=[" ".join(x["clean"][:MAXW + 4]) for x in b], truncation=True, max_length=a.max_len,
                  padding=True, return_tensors="pt").input_ids.to(dev_)
        lab[lab == tok.pad_token_id] = -100
        with torch.autocast(dev_, dtype=torch.bfloat16, enabled=dev_ == "cuda"):
            loss = model(**enc, labels=lab).loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); step += 1
        if step % 200 == 0:
            print(f"[{a.tag} s{a.seed}] ep{ep} {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
train_time = time.time() - t0


@torch.no_grad()
def generate(items, bs=64):
    model.eval(); out = []
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        enc = tok([a.prefix + " ".join(x["noisy"][:MAXW]) for x in b], truncation=True, max_length=a.max_len,
                  padding=True, return_tensors="pt", return_token_type_ids=False).to(dev_)
        g = model.generate(**enc, max_new_tokens=a.max_len, num_beams=1, do_sample=False,
                           output_scores=True, return_dict_in_generate=True)
        seq = g.sequences[:, 1:]                                     # drop decoder start token
        probs = torch.stack([s.float().softmax(-1) for s in g.scores], 1)  # B x L x V
        pt = probs.gather(2, seq[:, :probs.size(1), None]).squeeze(-1).cpu().numpy()
        for k in range(len(b)):
            ids = [int(t) for t in seq[k] if int(t) not in (tok.pad_token_id, tok.eos_token_id)]
            text = tok.decode(ids, skip_special_tokens=True).strip()
            out.append({"hyp": text.split(), "conf": G.word_confidences(tok, ids, pt[k][:len(ids)], text)})
    return out


verses = load("quran.json"); sp = quran_splits(verses)
sets = {s: load(f"gec_{s}.json") for s in ["cal", "test", "xdom"]}
sets["quran"] = quran_items(verses, "test", sp)
for f in sorted(glob.glob(os.path.join(P, "gec_md_*.json"))):
    sets[os.path.basename(f)[4:-5]] = json.load(open(f))
if a.smoke:
    sets = {k: v[:24] for k, v in sets.items()}
res = {"tag": a.tag, "seed": a.seed, "hf": a.hf, "train_time": train_time, "device": dev_,
       "params": sum(p.numel() for p in model.parameters())}
t1 = time.time(); n = 0
for k, its in sets.items():
    res[k] = generate(its); n += len(its)
res["infer_sent_per_s"] = n / (time.time() - t1)
os.makedirs(os.path.join(RES, "gen"), exist_ok=True)
json.dump(res, open(os.path.join(RES, "gen", f"{a.tag}_s{a.seed}.json"), "w"), ensure_ascii=False)
print(f"done {a.tag} s{a.seed} train_time={train_time:.0f}s", flush=True)
