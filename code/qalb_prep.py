"""Prepare the public human-annotated Arabic GEC data (QALB-2014 and ZAEBUC) for the tagger.

Source: CAMeL Lab arabic-gec repository (data/gec/modeling/<set>/wo_camelira/full/<split>.json),
json lines with keys raw (erroneous text) and cor (corrected text).

Each paragraph is aligned at word level and converted into one data-derived edit tag per source
word, with the same tag families as the synthetic data (KEEP, DEL, MERGE, SPLIT_k, APP_<w>,
C:<char script>, W:<word>). Paragraphs are cut into chunks of at most MAXW source words, at
positions inside unchanged runs so that no edit is split. The gold correction of each chunk is the
corresponding span of the human correction, so edits that one tag cannot express still count as
errors in evaluation.

usage: python3 code/qalb_prep.py RAW_DIR OUT_ROOT
  writes OUT_ROOT/data/processed/{gec_train,gec_dev,gec_cal,gec_test,gec_xdom,gec_md_ZAEBUC_cal}.json,
  vocab.json (word and character vocabulary of the synthetic setting, tags from QALB training data)
  and qalb_stats.json
"""
import os, sys, json, random, difflib, collections
sys.path.insert(0, os.path.dirname(__file__))
from gec_data import char_script, apply_script, apply_tags
from common import MAXW, P as P_SYN


def read(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            out.append((d["raw"].split(), d["cor"].split()))
    return out


def one_to_one(s, t):
    if s == t:
        return "KEEP"
    if not any(ch in s + t for ch in "|>:"):
        sc = char_script(s, t)
        try:
            if sc.count("|") <= 1 and apply_script(s, sc) == t:
                return "C:" + sc
        except ValueError:
            pass
    return "W:" + t


def derive(src, tgt):
    """Tags for src; returns tags, opcodes."""
    tags = ["KEEP"] * len(src)
    ops = difflib.SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if tag == "delete":
            for k in range(i1, i2):
                tags[k] = "DEL"
        elif tag == "insert":
            if i1 > 0 and tags[i1 - 1] == "KEEP" and j2 - j1 == 1:
                tags[i1 - 1] = "APP_" + tgt[j1]
            elif i1 > 0 and tags[i1 - 1] == "KEEP":
                tags[i1 - 1] = "APP_" + tgt[j1]          # only the first inserted word is expressible
        else:  # replace
            n, m = i2 - i1, j2 - j1
            if n == 2 and m == 1 and src[i1] + src[i1 + 1] == tgt[j1]:
                tags[i1] = "MERGE"; tags[i1 + 1] = "MERGED"
            elif n == 1 and m == 2 and src[i1] == tgt[j1] + tgt[j1 + 1]:
                tags[i1] = f"SPLIT_{len(tgt[j1])}"
            elif n == 1 and m == 2:
                tags[i1] = one_to_one(src[i1], tgt[j1])   # second word not expressible
            else:
                for k in range(n):
                    tags[i1 + k] = one_to_one(src[i1 + k], tgt[j1 + k]) if k < m else "DEL"
    return tags, ops


def chunks(src, tgt, rng_len=MAXW):
    """Cut a paragraph into pieces of <= rng_len source words at positions inside equal runs."""
    tags, ops = derive(src, tgt)
    # map: for each source cut position k, the target position, valid if k-1 and k are in one equal run
    valid = {0: 0, len(src): len(tgt)}
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i1 + 1, i2):
                valid[k] = j1 + (k - i1)
    out, a = [], 0
    while a < len(src):
        b = min(a + rng_len, len(src))
        if b < len(src):
            cand = [k for k in valid if a < k <= b]
            b = max(cand) if cand else None
            if b is None:   # no clean cut inside the window: skip to the next clean cut
                nxt = [k for k in valid if k > a]
                a = min(nxt); continue
        out.append({"noisy": src[a:b], "clean": tgt[valid[a]:valid[b]], "tags": tags[a:b]})
        a = b
    return out


def build(pairs):
    items = []
    for s, t in pairs:
        items += chunks(s, t)
    for x in items:
        x["src"] = "real"
    return items


def main():
    raw, root = sys.argv[1], sys.argv[2]
    P = os.path.join(root, "data", "processed"); os.makedirs(P, exist_ok=True)
    tr = build(read(os.path.join(raw, "qalb14_train.json")))
    dv = build(read(os.path.join(raw, "qalb14_dev.json")))
    te = build(read(os.path.join(raw, "qalb14_test.json")))
    zd = build(read(os.path.join(raw, "zaebuc_dev.json")))
    zt = build(read(os.path.join(raw, "zaebuc_test.json")))
    rng = random.Random(11); idx = list(range(len(dv))); rng.shuffle(idx)
    half = len(idx) // 2
    dev = [dv[i] for i in sorted(idx[:half])]; cal = [dv[i] for i in sorted(idx[half:])]
    sets = {"train": tr, "dev": dev, "cal": cal, "test": te, "xdom": zt, "md_ZAEBUC_cal": zd}
    for k, v in sets.items():
        json.dump(v, open(os.path.join(P, f"gec_{k}.json"), "w"), ensure_ascii=False)
    for r in ["05", "10", "20", "25", "30"]:
        json.dump([], open(os.path.join(P, f"gec_test_r{r}.json"), "w"))
    syn = json.load(open(os.path.join(P_SYN, "vocab.json")))
    tc = collections.Counter(t for x in tr for t in x["tags"] if t != "MERGED")
    tags = ["KEEP"] + sorted(t for t, c in tc.items() if c >= 3 and t != "KEEP")
    json.dump({"words": syn["words"], "chars": syn["chars"], "tags": tags},
              open(os.path.join(P, "vocab.json"), "w"), ensure_ascii=False)
    stats = {"tags": len(tags)}
    tset = set(tags)
    for k, v in sets.items():
        nw = sum(len(x["noisy"]) for x in v)
        ne = sum(t != "KEEP" for x in v for t in x["tags"])
        rt = sum(apply_tags(x["noisy"], x["tags"]) == x["clean"] for x in v)
        cov = sum(all(t in tset or t == "MERGED" for t in x["tags"]) and apply_tags(x["noisy"], x["tags"]) == x["clean"] for x in v)
        stats[k] = {"chunks": len(v), "words": nw, "edited_word_rate": round(ne / max(nw, 1), 4),
                    "exact_by_tags": round(rt / max(len(v), 1), 4), "exact_by_vocab_tags": round(cov / max(len(v), 1), 4)}
    json.dump(stats, open(os.path.join(P, "qalb_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
