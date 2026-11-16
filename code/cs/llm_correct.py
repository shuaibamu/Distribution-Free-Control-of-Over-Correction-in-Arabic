"""Few-shot correction with an open instruction-tuned LLM (no fine-tuning).  Writes greedy
hypotheses with per-word generation confidences to results/gen/<tag>_s<seed>.json, in the same
format as hf_seq2seq.py, so code/cs/gen_eval.py applies the same metrics and edit-level risk control.

Suggested models (any chat model with an Arabic-capable tokenizer works):
  Qwen/Qwen2.5-7B-Instruct, ALLaM-AI/ALLaM-7B-Instruct-preview, CohereForAI/aya-expanse-8b
usage: python3 code/cs/llm_correct.py --hf Qwen/Qwen2.5-7B-Instruct --tag qwen7b --n 1000
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
ap.add_argument("--n", type=int, default=1000, help="sentences per split (Qur'an test is always complete)")
ap.add_argument("--shots", type=int, default=5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM

dev_ = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.hf); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.bfloat16 if dev_ == "cuda" else torch.float32,
                                             device_map="auto" if dev_ == "cuda" else None)
model.eval()
INSTR = ("صحّح الأخطاء الإملائية والنحوية في الجملة العربية الآتية بأقل قدر من التعديل. "
         "لا تغيّر الكلمات الصحيحة ولا تعِد صياغة الجملة. أعد الجملة المصححة فقط.")
rng = random.Random(0)
shots = rng.sample([x for x in load("gec_train.json") if x["noisy"] != x["clean"]], a.shots)


def prompt(words):
    msgs = [{"role": "system", "content": INSTR}]
    for x in shots:
        msgs += [{"role": "user", "content": " ".join(x["noisy"])}, {"role": "assistant", "content": " ".join(x["clean"])}]
    msgs.append({"role": "user", "content": " ".join(words)})
    if tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return INSTR + "\n" + "\n".join(f"{m['role']}: {m['content']}" for m in msgs[1:]) + "\nassistant:"


@torch.no_grad()
def generate(items):
    out = []
    for i in range(0, len(items), a.bs):
        b = items[i:i + a.bs]
        enc = tok([prompt(x["noisy"][:MAXW]) for x in b], return_tensors="pt", padding=True,
                  return_token_type_ids=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=96, do_sample=False, output_scores=True,
                           return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        seq = g.sequences[:, enc.input_ids.size(1):]
        probs = torch.stack([s.float().softmax(-1) for s in g.scores], 1)
        pt = probs.gather(2, seq[:, :probs.size(1), None]).squeeze(-1).cpu().numpy()
        for k in range(len(b)):
            ids = []
            for t in seq[k].tolist():
                if t in (tok.eos_token_id, tok.pad_token_id):
                    break
                ids.append(t)
            text = tok.decode(ids, skip_special_tokens=True).strip().split("\n")[0].strip()
            out.append({"hyp": text.split(), "conf": G.word_confidences(tok, ids, pt[k][:len(ids)], text)})
        print(f"{a.tag}: {i + len(b)}/{len(items)}", flush=True)
    return out


verses = load("quran.json"); sp = quran_splits(verses)
sets = {"cal": load("gec_cal.json")[:a.n], "test": load("gec_test.json")[:a.n], "xdom": load("gec_xdom.json")[:a.n],
        "quran": quran_items(verses, "test", sp)}
for f in sorted(glob.glob(os.path.join(P, "gec_md_*_test.json"))):
    sets[os.path.basename(f)[4:-5]] = json.load(open(f))[:a.n]
if a.smoke:
    sets = {k: v[:4] for k, v in sets.items()}
res = {"tag": a.tag, "seed": a.seed, "hf": a.hf, "train_time": 0.0, "device": dev_, "n": a.n,
       "params": sum(p.numel() for p in model.parameters())}
t1 = time.time(); n = 0
for k, its in sets.items():
    res[k] = generate(its); n += len(its)
res["infer_sent_per_s"] = n / (time.time() - t1)
os.makedirs(os.path.join(RES, "gen"), exist_ok=True)
json.dump(res, open(os.path.join(RES, "gen", f"{a.tag}_s{a.seed}.json"), "w"), ensure_ascii=False)
print("done", a.tag)
