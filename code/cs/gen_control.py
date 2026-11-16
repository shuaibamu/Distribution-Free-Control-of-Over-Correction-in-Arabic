"""Edit-level risk control for generative correctors (fine-tuned seq2seq models and prompted LLMs).

A generative hypothesis is aligned to the source with difflib; every non-equal opcode is one edit.
The confidence of an edit is the minimum generation probability of the output words it introduces
(for a pure deletion, the probability of the word generated next).  An edit is applied only if
    conf > tau + beta * r,  r = max rhetoric probability of the source tokens it touches,
so the applied-edit set shrinks monotonically in tau and conformal risk control applies with the same
false-edit loss as for the taggers (fraction of source tokens that are touched but should be kept).
"""
import difflib
import numpy as np


def word_confidences(tokenizer, ids, probs, text):
    """Map per-token generation probabilities to the whitespace words of `text` (the decoded
    output). Characters added when token k is decoded are attributed to token k; a word's
    confidence is the minimum probability of the tokens that produced any of its characters."""
    owner = []
    prev = 0
    for k in range(len(ids)):
        cur = len(tokenizer.decode(ids[:k + 1], skip_special_tokens=True))
        owner.extend([k] * max(cur - prev, 0))
        prev = max(prev, cur)
    conf, pos = [], 0
    for w in text.split():
        st = text.index(w, pos); en = st + len(w); pos = en
        ks = {owner[c] for c in range(st, min(en, len(owner)))}
        conf.append(min((float(probs[k]) for k in ks), default=1.0))
    return conf


def opcodes(src, hyp):
    return [o for o in difflib.SequenceMatcher(None, src, hyp, autojunk=False).get_opcodes() if o[0] != "equal"]


def edit_list(src, hyp, wconf, rhet):
    """[(i1, i2, j1, j2, conf, r)] for every edit of hyp against src."""
    out = []
    for tag, i1, i2, j1, j2 in opcodes(src, hyp):
        if j2 > j1:
            c = min(wconf[j1:j2])
        else:
            c = wconf[j1] if j1 < len(wconf) else (wconf[-1] if wconf else 1.0)
        span = range(i1, i2) if i2 > i1 else [max(i1 - 1, 0)]
        r = max((rhet[i] for i in span if i < len(rhet)), default=0.0)
        out.append((i1, i2, j1, j2, c, r))
    return out


def apply_edits(src, hyp, edits, tau, beta):
    """Rebuild the output keeping only the edits accepted at (tau, beta)."""
    acc = {(e[0], e[1]): e for e in edits if e[4] > tau + beta * e[5]}
    out = []; i = 0
    for e in sorted(acc.values()):
        out.extend(src[i:e[0]]); out.extend(hyp[e[2]:e[3]]); i = e[1]
    out.extend(src[i:])
    return out


def touched(src, hyp):
    t = set()
    for tag, i1, i2, j1, j2 in opcodes(src, hyp):
        t.update(range(i1, i2) if i2 > i1 else [max(i1 - 1, 0)])
    return t


def false_edit_loss(src, out, gold):
    """Fraction of source tokens that out touches although gold keeps them."""
    return len(touched(src, out) - touched(src, gold)) / max(len(src), 1)


def loss_matrix(srcs, hyps, edits, golds, grid, beta):
    L = np.zeros((len(srcs), len(grid)))
    for n, (s, h, e, g) in enumerate(zip(srcs, hyps, edits, golds)):
        tg = touched(s, g)
        for k, tau in enumerate(grid):
            o = apply_edits(s, h, e, tau, beta)
            L[n, k] = len(touched(s, o) - tg) / max(len(s), 1)
    return L


def crc_tau(L, grid, alpha):
    n = len(L)
    ok = (n / (n + 1)) * L.mean(0) + 1 / (n + 1) <= alpha
    return float(grid[int(np.argmax(ok))]) if ok.any() else float(grid[-1])
