"""Shared vocabulary, batching, models and evaluation utilities."""
import json, os, math, random, collections, difflib, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
CK = os.path.join(ROOT, "checkpoints")
os.makedirs(RES, exist_ok=True); os.makedirs(CK, exist_ok=True)

PAD, UNK, MASK, BOS, EOS = 0, 1, 2, 3, 4
MAXW, MAXC = 40, 14


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load(name):
    return json.load(open(os.path.join(P, name)))


class Vocab:
    def __init__(self, path=os.path.join(P, "vocab.json")):
        if not os.path.exists(path):
            self.build(path)
        d = json.load(open(path))
        self.itos = d["words"]; self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.chars = d["chars"]; self.ctoi = {c: i for i, c in enumerate(self.chars)}
        self.tags = d["tags"]; self.ttoi = {t: i for i, t in enumerate(self.tags)}

    @staticmethod
    def build(path, nwords=30000, min_tag=3):
        pre = load("pretrain.json")
        cnt = collections.Counter(w for s in pre for w in s)
        words = ["<pad>", "<unk>", "<mask>", "<s>", "</s>"] + [w for w, _ in cnt.most_common(nwords)]
        chars = ["<pad>", "<unk>"] + sorted({c for s in pre for w in s for c in w})
        tr = load("gec_train.json")
        tc = collections.Counter(t for x in tr for t in x["tags"])
        tags = ["KEEP"] + sorted(t for t, c in tc.items() if c >= min_tag and t != "KEEP")
        json.dump({"words": words, "chars": chars, "tags": tags}, open(path, "w"), ensure_ascii=False)

    def wid(self, w):
        return self.stoi.get(w, UNK)

    def cids(self, w):
        c = [self.ctoi.get(ch, 1) for ch in w[:MAXC]]
        return c + [0] * (MAXC - len(c))


def encode_tokens(V, toks):
    toks = toks[:MAXW]
    return [V.wid(w) for w in toks], [V.cids(w) for w in toks]


def batchify(V, items, key="noisy", with_tags=True, rhet=False):
    """items: list of dicts. Returns padded tensors."""
    n = len(items); L = max(min(len(x[key]), MAXW) for x in items)
    W = torch.zeros(n, L, dtype=torch.long); C = torch.zeros(n, L, MAXC, dtype=torch.long)
    M = torch.zeros(n, L, dtype=torch.bool)
    T = torch.full((n, L), -100, dtype=torch.long); D = torch.full((n, L), -100, dtype=torch.long)
    R = torch.full((n, L), -100, dtype=torch.long)
    for i, x in enumerate(items):
        w, c = encode_tokens(V, x[key]); l = len(w)
        W[i, :l] = torch.tensor(w); C[i, :l] = torch.tensor(c); M[i, :l] = True
        if with_tags and "tags" in x:
            tg = [V.ttoi.get(t, -100) for t in x["tags"][:l]]   # unreachable tags ignored in loss
            T[i, :l] = torch.tensor(tg)
            D[i, :l] = torch.tensor([int(t != "KEEP") for t in x["tags"][:l]])
        if rhet and "rhet" in x:
            R[i, :l] = torch.tensor(x["rhet"][:l])
    return W, C, M, T, D, R


# ------------------------------------------------------------------ modules
class CharWordEmb(nn.Module):
    def __init__(self, nw, nc, d, dc=64):
        super().__init__()
        self.w = nn.Embedding(nw, d, padding_idx=0)
        self.c = nn.Embedding(nc, dc, padding_idx=0)
        self.conv = nn.Conv1d(dc, dc, 3, padding=1)
        self.proj = nn.Linear(d + dc, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, W, C):
        b, l, k = C.shape
        ce = self.c(C.view(b * l, k)).transpose(1, 2)
        ce = F.relu(self.conv(ce)).max(-1).values.view(b, l, -1)
        return self.ln(self.proj(torch.cat([self.w(W), ce], -1)))


class TransEnc(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, ff=512, drop=0.1):
        super().__init__()
        self.pos = nn.Embedding(MAXW + 2, d)
        lay = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, M):
        pos = torch.arange(x.size(1), device=x.device)
        return self.ln(self.enc(x + self.pos(pos)[None], src_key_padding_mask=~M))


class MLM(nn.Module):
    """Encoder pretraining with masked-word prediction."""
    def __init__(self, V, d=256, layers=4):
        super().__init__()
        self.emb = CharWordEmb(len(V.itos), len(V.chars), d)
        self.enc = TransEnc(d, layers)
        self.out = nn.Linear(d, len(V.itos))

    def forward(self, W, C, M, sel=None):
        h = self.enc(self.emb(W, C), M)
        return self.out(h[sel]) if sel is not None else self.out(h)


class Tagger(nn.Module):
    """Configurable sequence tagger covering the proposed model, baselines and ablations.

    enc:   'trans' | 'rnn' | 'hybrid'
    rnn:   'gru' | 'lstm'
    fusion:'gate' | 'concat' | 'stack'
    heads: tag (always), detect (multi-task), rhet (rhetoric head)
    """
    def __init__(self, V, d=256, layers=4, enc="hybrid", rnn="gru", fusion="gate",
                 detect=True, rhet=True, drop=0.1):
        super().__init__()
        self.cfg = dict(enc=enc, rnn=rnn, fusion=fusion, detect=detect, rhet=rhet)
        self.emb = CharWordEmb(len(V.itos), len(V.chars), d)
        self.trans = TransEnc(d, layers) if enc in ("trans", "hybrid") else None
        if enc in ("rnn", "hybrid"):
            R = nn.GRU if rnn == "gru" else nn.LSTM
            nl = 2 if enc == "rnn" else 1
            inp = d
            self.rnn = R(inp, d // 2, num_layers=nl, batch_first=True, bidirectional=True, dropout=drop if nl > 1 else 0)
        else:
            self.rnn = None
        if enc == "hybrid":
            if fusion == "gate":
                self.gate = nn.Linear(2 * d, d)
                hd = d
            elif fusion == "concat":
                hd = 2 * d
            else:  # stack: RNN on top of transformer
                hd = d
        else:
            hd = d
        self.hd = hd
        self.drop = nn.Dropout(drop)
        self.tag = nn.Linear(hd, len(V.tags))
        self.det = nn.Linear(hd, 2) if detect else None
        self.rh = nn.Sequential(nn.Linear(hd, hd // 2), nn.GELU(), nn.Linear(hd // 2, 2)) if rhet else None

    def load_pretrained(self, path):
        sd = torch.load(path, map_location="cpu")
        self.emb.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("emb.")})
        if self.trans is not None:
            self.trans.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("enc.")})

    def rnn_run(self, x, M):
        lens = M.sum(1).clamp(min=1).cpu()
        pk = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(pk)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out

    def forward(self, W, C, M, return_gate=False):
        e = self.emb(W, C)
        g = None
        if self.cfg["enc"] == "trans":
            h = self.trans(e, M)
        elif self.cfg["enc"] == "rnn":
            h = self.rnn_run(e, M)
        else:
            ht = self.trans(e, M)
            if self.cfg["fusion"] == "stack":
                h = self.rnn_run(ht, M)
            else:
                hr = self.rnn_run(e, M)
                if self.cfg["fusion"] == "gate":
                    g = torch.sigmoid(self.gate(torch.cat([ht, hr], -1)))
                    h = g * ht + (1 - g) * hr
                else:
                    h = torch.cat([ht, hr], -1)
        h = self.drop(h)
        out = {"tag": self.tag(h)}
        if self.det is not None:
            out["det"] = self.det(h)
        if self.rh is not None:
            out["rhet"] = self.rh(h)
        if return_gate:
            out["gate"] = g
        return out


# ------------------------------------------------------------------ seq2seq baseline
class Seq2SeqCopy(nn.Module):
    """Word-level transformer encoder-decoder with a pointer (copy) mechanism."""
    def __init__(self, V, d=256, layers=3, heads=4, ff=512, drop=0.1):
        super().__init__()
        self.V = V
        self.emb = CharWordEmb(len(V.itos), len(V.chars), d)
        self.temb = nn.Embedding(len(V.itos), d, padding_idx=0)
        self.pos = nn.Embedding(MAXW + 8, d)
        self.tf = nn.Transformer(d, heads, layers, layers, ff, drop, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, len(V.itos))
        self.pg = nn.Linear(2 * d, 1)

    def encode(self, W, C, M):
        x = self.emb(W, C) + self.pos(torch.arange(W.size(1)))[None]
        return self.tf.encoder(x, src_key_padding_mask=~M)

    def decode_step(self, mem, M, Y):
        L = Y.size(1)
        y = self.temb(Y) + self.pos(torch.arange(L))[None]
        causal = torch.triu(torch.full((L, L), float("-inf")), 1)
        h = self.tf.decoder(y, mem, tgt_mask=causal, memory_key_padding_mask=~M)
        att = torch.einsum("bld,bsd->bls", h, mem) / math.sqrt(h.size(-1))
        att = att.masked_fill(~M[:, None, :], float("-inf")).softmax(-1)
        ctx = torch.einsum("bls,bsd->bld", att, mem)
        pgen = torch.sigmoid(self.pg(torch.cat([h, ctx], -1)))
        return h, att, pgen

    def forward(self, W, C, M, Y_in, src_ext, n_ext):
        """src_ext: source ids in extended vocab (OOVs get ids >= |V|)."""
        mem = self.encode(W, C, M)
        h, att, pgen = self.decode_step(mem, M, Y_in)
        pv = self.out(h).softmax(-1) * pgen
        b, l, _ = pv.shape
        ext = torch.zeros(b, l, n_ext)
        pv = torch.cat([pv, ext], -1)
        idx = src_ext[:, None, :].expand(-1, l, -1)
        pv = pv.scatter_add(2, idx, (1 - pgen) * att)
        return pv


# ------------------------------------------------------------------ edit extraction & metrics
def edits(src, hyp):
    """Word-level edit set between source and hypothesis token lists."""
    out = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, hyp, autojunk=False).get_opcodes():
        if tag != "equal":
            out.add((i1, i2, " ".join(hyp[j1:j2])))
    return out


def touched(src, hyp):
    """Indices of source tokens modified by hyp (insertions charge the left token)."""
    t = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, hyp, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            t.update(range(i1, i2))
        else:
            t.add(max(i1 - 1, 0))
    return t


def prf(tp, fp, fn, beta=0.5):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0
    return p, r, f


def score_corpus(srcs, hyps, golds):
    tp = fp = fn = 0
    oc = ntok = 0
    exact = 0
    for s, h, g in zip(srcs, hyps, golds):
        eh, eg = edits(s, h), edits(s, g)
        tp += len(eh & eg); fp += len(eh - eg); fn += len(eg - eh)
        th, tg = touched(s, h), touched(s, g)
        oc += len(th - tg); ntok += len(s)
        exact += h == g
    p, r, f = prf(tp, fp, fn)
    return {"P": p, "R": r, "F05": f, "OCR": oc / max(ntok, 1), "EM": exact / max(len(srcs), 1),
            "tp": tp, "fp": fp, "fn": fn}


def ece(conf, correct, bins=15):
    conf = np.asarray(conf); correct = np.asarray(correct, dtype=float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def quran_splits(verses):
    """Verse-level split for rhetoric-head training (70/10/20) with canonical
    exemplars forced into test."""
    from canonical_istiara import CANON
    canon = {(s, a) for s, a, _ in CANON}
    sp = {}
    for v in verses:
        k = (v["sura"], v["aya"])
        if k in canon:
            sp[k] = "test"; continue
        h = zlib.crc32(f"{k[0]}:{k[1]}".encode()) % 10
        sp[k] = "train" if h < 7 else ("dev" if h == 7 else "test")
    return sp
