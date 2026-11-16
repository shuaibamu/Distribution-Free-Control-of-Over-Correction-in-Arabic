"""Seeded synthetic Arabic error generator and data-derived edit-tag scheme.

Each source word receives at most one edit so that every corruption is exactly
representable by one word-level tag. Tags are derived from the data
(token-pair edit scripts) rather than from hand-written language rules:
    KEEP                 source word is correct
    DEL                  spurious word, delete it
    MERGE                join this word with the next one (no space)
    SPLIT_k              insert a space after character k
    APP_<w>              append missing function word <w> after this word
    C:<script>           character-level transform (derived from edit script)
    W:<w>                whole-word replacement (lexical confusion)
Error families (used for per-type analysis):
    ORTH  hamza / taa-marbuta / alif-maqsura / dad-zaa confusions
    TYPO  insertion, deletion, transposition, substitution of one character
    SPACE merge / split errors
    MISS  missing function word
    RED   redundant function word
    LEX   lexical confusion (contextually wrong but well-formed word)
"""
import json, os, random, collections, difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")

FUNC = ["في", "من", "على", "إلى", "عن", "أن", "ما", "لا", "قد", "ثم", "الذي", "التي", "هو", "هي", "به", "له"]
ORTH_PAIRS = [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي"), ("ي", "ى"),
              ("ؤ", "و"), ("ئ", "ي"), ("ض", "ظ"), ("ظ", "ض"), ("ا", "أ"), ("ا", "إ")]
LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
FAMILIES = ["ORTH", "TYPO", "SPACE", "MISS", "RED", "LEX"]


def char_script(src, tgt):
    """Compact, position-anchored edit script turning src into tgt.
    Positions are counted from the word end when the edit is in the last two
    characters (suffix edits recur across words), otherwise from the start."""
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        pos = f"e{len(src) - i1}" if len(src) - i1 <= 2 else f"b{i1}"
        ops.append(f"{tag[0]}{pos}:{src[i1:i2]}>{tgt[j1:j2]}")
    return "|".join(ops)


def apply_script(src, script):
    """Inverse of char_script; returns None when the script does not apply."""
    w = src
    # apply from right to left in source coordinates
    parsed = []
    for op in script.split("|"):
        kind, rest = op[0], op[1:]
        pos, chg = rest.split(":", 1)
        a, b = chg.split(">", 1)
        i = len(src) - int(pos[1:]) if pos[0] == "e" else int(pos[1:])
        parsed.append((i, a, b))
    for i, a, b in sorted(parsed, reverse=True):
        if i < 0 or w[i:i + len(a)] != a:
            return None
        w = w[:i] + b + w[i + len(a):]
    return w


def build_confusions(corpus, n_pairs=2400, seed=0):
    """Lexical confusion pairs: frequent words sharing their last two letters
    and length (same morphological pattern), so the corrupted word is
    well-formed but contextually wrong."""
    rng = random.Random(seed)
    cnt = collections.Counter(w for s in corpus for w in s["tokens"])
    freq = [w for w, c in cnt.most_common(8000) if len(w) >= 4 and w not in FUNC]
    bucket = collections.defaultdict(list)
    for w in freq:
        bucket[(len(w), w[-2:])].append(w)
    pairs = {}
    keys = [k for k, v in bucket.items() if len(v) >= 2]
    rng.shuffle(keys)
    for k in keys:
        ws = bucket[k][:]
        rng.shuffle(ws)
        for a, b in zip(ws[::2], ws[1::2]):
            pairs[a] = b
            pairs[b] = a
        if len(pairs) >= n_pairs:
            break
    return pairs


def corrupt(tokens, rng, conf, rate=0.15, clean_prob=0.10):
    """Return (noisy_tokens, tags, families) aligned on noisy tokens."""
    noisy, tags, fams = [], [], []
    if rng.random() < clean_prob:
        return list(tokens), ["KEEP"] * len(tokens), ["-"] * len(tokens)
    i = 0
    n = len(tokens)
    while i < n:
        w = tokens[i]
        if rng.random() >= rate:
            noisy.append(w); tags.append("KEEP"); fams.append("-"); i += 1
            continue
        fam = rng.choices(FAMILIES, weights=[0.32, 0.22, 0.14, 0.10, 0.10, 0.12])[0]
        done = False
        if fam == "ORTH":
            cands = [(a, b) for a, b in ORTH_PAIRS if a in w]
            if cands:
                a, b = rng.choice(cands)
                idxs = [k for k in range(len(w)) if w[k] == a]
                k = rng.choice(idxs)
                bad = w[:k] + b + w[k + 1:]
                noisy.append(bad); tags.append("C:" + char_script(bad, w)); fams.append(fam); done = True
        elif fam == "TYPO" and len(w) >= 3:
            k = rng.randrange(len(w))
            op = rng.choice(["ins", "del", "sub", "swap"])
            if op == "ins":
                bad = w[:k] + rng.choice(LETTERS) + w[k:]
            elif op == "del":
                bad = w[:k] + w[k + 1:]
            elif op == "sub":
                bad = w[:k] + rng.choice(LETTERS) + w[k + 1:]
            else:
                k = min(k, len(w) - 2)
                bad = w[:k] + w[k + 1] + w[k] + w[k + 2:]
            if bad != w and len(bad) >= 2:
                noisy.append(bad); tags.append("C:" + char_script(bad, w)); fams.append(fam); done = True
        elif fam == "SPACE":
            if i + 1 < n and rng.random() < 0.5 and len(w) + len(tokens[i + 1]) <= 14:
                # merge error: two words written together -> SPLIT tag on the merged token
                noisy.append(w + tokens[i + 1]); tags.append(f"SPLIT_{len(w)}"); fams.append(fam)
                i += 2
                continue
            elif len(w) >= 4 and w[0] in "وفبل":
                # split error: proclitic detached -> MERGE tag on the proclitic
                noisy.append(w[0]); tags.append("MERGE"); fams.append(fam)
                noisy.append(w[1:]); tags.append("KEEP"); fams.append("-")
                done = True
        elif fam == "MISS" and i + 1 < n and tokens[i + 1] in FUNC:
            noisy.append(w); tags.append("APP_" + tokens[i + 1]); fams.append(fam)
            i += 2
            continue
        elif fam == "RED":
            noisy.append(w); tags.append("KEEP"); fams.append("-")
            noisy.append(rng.choice(FUNC)); tags.append("DEL"); fams.append(fam)
            done = True
        elif fam == "LEX" and w in conf:
            noisy.append(conf[w]); tags.append("W:" + w); fams.append(fam); done = True
        if not done:
            noisy.append(w); tags.append("KEEP"); fams.append("-")
        i += 1
    return noisy, tags, fams


def apply_tags(noisy, tags):
    out = []
    skip_merge = False
    for k, (w, t) in enumerate(zip(noisy, tags)):
        if skip_merge:
            out[-1] = out[-1] + w
            skip_merge = False
            continue
        if t == "KEEP":
            out.append(w)
        elif t == "DEL":
            continue
        elif t == "MERGE":
            out.append(w); skip_merge = True
        elif t.startswith("SPLIT_"):
            j = int(t[6:])
            if 0 < j < len(w):
                out.extend([w[:j], w[j:]])
            else:
                out.append(w)
        elif t.startswith("APP_"):
            out.extend([w, t[4:]])
        elif t.startswith("C:"):
            r = apply_script(w, t[2:])
            out.append(r if r else w)
        elif t.startswith("W:"):
            out.append(t[2:])
        else:
            out.append(w)
    return out


def make_split(corpus, conf, seed, rate=0.15):
    rng = random.Random(seed)
    data = []
    for s in corpus:
        n, t, f = corrupt(s["tokens"], rng, conf, rate=rate)
        data.append({"src": s["src"], "clean": s["tokens"], "noisy": n, "tags": t, "fam": f})
    return data


def main():
    corpus = json.load(open(os.path.join(P, "classical.json")))
    # held-out author for cross-domain test: al-Jurjani; Bukhari kept in-domain
    jur = [s for s in corpus if s["src"] == "Jurjani-AsrarBalagha"]
    rest = [s for s in corpus if s["src"] != "Jurjani-AsrarBalagha"]
    rng = random.Random(7)
    rng.shuffle(rest)
    splits = {
        "pretrain": rest[:60000],      # MLM pretraining text (clean only)
        "train": rest[:30000],
        "dev": rest[60000:62000],
        "cal": rest[62000:65000],
        "test": rest[65000:69000],
        "xdom": jur,
    }
    conf = build_confusions(splits["pretrain"])
    json.dump(conf, open(os.path.join(P, "confusions.json"), "w"), ensure_ascii=False)
    stats = {}
    for name, seed in [("train", 101), ("dev", 102), ("cal", 103), ("test", 104), ("xdom", 105)]:
        d = make_split(splits[name], conf, seed)
        # round-trip check: applying gold tags must recover the clean sentence
        ok = sum(apply_tags(x["noisy"], x["tags"]) == x["clean"] for x in d)
        stats[name] = {"n": len(d), "roundtrip_ok": ok,
                       "err_rate": round(sum(t != "KEEP" for x in d for t in x["tags"]) /
                                         sum(len(x["tags"]) for x in d), 4),
                       "families": dict(collections.Counter(f for x in d for f in x["fam"] if f != "-"))}
        json.dump(d, open(os.path.join(P, f"gec_{name}.json"), "w"), ensure_ascii=False)
    # noise-sweep test sets for robustness
    for r in [0.05, 0.10, 0.20, 0.25, 0.30]:
        d = make_split(splits["test"], conf, 200 + int(r * 100), rate=r)
        json.dump(d, open(os.path.join(P, f"gec_test_r{int(r*100):02d}.json"), "w"), ensure_ascii=False)
    json.dump([s["tokens"] for s in splits["pretrain"]], open(os.path.join(P, "pretrain.json"), "w"), ensure_ascii=False)
    json.dump(stats, open(os.path.join(P, "gec_stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
