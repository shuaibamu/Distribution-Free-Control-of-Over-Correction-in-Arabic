"""Corpus preparation.

Builds three resources from public OpenITI / Tanzil files:
  1. clean classical sentences (GEC source + MLM pretraining),
  2. Qur'anic verses (error-free faithfulness test set),
  3. distant-supervision ("silver") isti'ara token labels for Qur'anic verses,
     mined from two tafsir works (al-Zamakhshari's al-Kashshaf and
     Ibn 'Ashur's al-Tahrir wa al-Tanwir).
All randomness is seeded. Outputs go to data/processed/.
"""
import glob, json, os, re, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "processed")
os.makedirs(OUT, exist_ok=True)
random.seed(13)

DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
AR_WORD = re.compile(r"^[ء-ي]+$")


def strip_markup(line):
    line = re.sub(r"PageV\d+P\d+", " ", line)
    line = re.sub(r"ms\d+", " ", line)
    line = re.sub(r"^#+\s*\|?\s*\d*", " ", line)
    line = line.replace("~~", " ").replace("#", " ")
    line = re.sub(r"«\d+»|\(\d+\)|\[[^\]]*\]", " ", line)
    return DIAC.sub("", line)


def norm_match(w):
    """Aggressive normalisation used ONLY for verse matching, never for GEC."""
    w = re.sub("[أإآٱ]", "ا", w)
    w = w.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return w


def read_body(path):
    txt = open(path, encoding="utf-8").read()
    if "#META#Header#End#" in txt:
        txt = txt.split("#META#Header#End#", 1)[1]
    return txt


# ---------------------------------------------------------------- Qur'an
def load_quran():
    verses = []
    cur = None
    for line in read_body(os.path.join(D, "quran_openiti.md")).splitlines():
        m = re.match(r"^#\s*(\d+)\|(\d+)\|\s*(.*)$", line)
        if m:
            if cur:
                verses.append(cur)
            cur = {"sura": int(m.group(1)), "aya": int(m.group(2)), "text": m.group(3)}
        elif line.startswith("~~") and cur:
            cur["text"] += " " + line[2:]
    if cur:
        verses.append(cur)
    for v in verses:
        t = strip_markup(v["text"])
        v["tokens"] = [w for w in t.split() if AR_WORD.match(w)]
    return verses


# ---------------------------------------------------------------- classical prose
def sentences_from(path, src, max_n):
    body = read_body(path)
    body = "\n".join(strip_markup(l) for l in body.splitlines() if not l.startswith("#META"))
    body = re.sub(r"\s+", " ", body)
    parts = re.split(r"[.!?؟:؛\n]", body)
    out = []
    for p in parts:
        toks = [w for w in re.sub(r"[،,\"'«»()\-]", " ", p).split()]
        toks = [w for w in toks if AR_WORD.match(w)]
        if 8 <= len(toks) <= 30:
            out.append({"src": src, "tokens": toks})
    random.shuffle(out)
    return out[:max_n]


SOURCES = {
    "Bukhari-AdabMufrad": ("openiti/0275AH/**/0256Bukhari.AdabMufrad.Shamela0009647-ara1", 8000),
    "Jurjani-AsrarBalagha": ("openiti/0475AH/**/0471CabdQahirJurjani.AsrarBalagha.Shamela0008187-ara1", 8000),
    "Zamakhshari-Kashshaf": ("openiti/0550AH/**/0538JarAllahZamakhshari.Kashshaf.Shamela0023627-ara1", 30000),
    "IbnAshur-Tahrir": ("openiti/1400AH/**/1393MuhammadTahirIbnCashurTunisi.TahrirWaTanwir.Shamela0009776-ara1", 60000),
}

TERMS = re.compile(r"^(و?ب?ال?)?(استعار|استعير|مستعار|الاستعار|استعاره|استعارة|مستعارة)")


FIG = re.compile(r"^(و?ب?ال?)?(مجاز|المجاز|مجازا|مجازي|تمثيل|تمثيلية|التمثيل|كناية|الكناية)")


def mine_istiara(path, verses, ngram_index, window=6, terms=TERMS):
    """Return {(sura,aya): set(token_idx)} of verse tokens discussed near an isti'ara term."""
    body = read_body(path)
    toks = []
    for l in body.splitlines():
        toks.extend(w for w in strip_markup(l).split() if AR_WORD.match(w))
    ntoks = [norm_match(w) for w in toks]
    labels = collections.defaultdict(set)
    hits = 0
    for i, w in enumerate(toks):
        if not terms.match(w):
            continue
        hits += 1
        lo, hi = max(0, i - 40), min(len(toks), i + 25)
        # find verse with longest 3-gram overlap inside the commentary window
        score = collections.Counter()
        for j in range(lo, hi - 2):
            g = tuple(ntoks[j:j + 3])
            for vid in ngram_index.get(g, ()):
                score[vid] += 1
        if not score:
            continue
        vid, s = score.most_common(1)[0]
        if s < 1:
            continue
        v = verses[vid]
        vnorm = [norm_match(x) for x in v["tokens"]]
        near = set(ntoks[max(lo, i - window): min(hi, i + window + 1)])
        for k, vw in enumerate(vnorm):
            if vw in near and len(vw) > 2:
                labels[(v["sura"], v["aya"])].add(k)
    return labels, hits


def main():
    verses = load_quran()
    print("verses", len(verses), "tokens", sum(len(v["tokens"]) for v in verses))
    ngram_index = collections.defaultdict(set)
    for vid, v in enumerate(verses):
        nt = [norm_match(x) for x in v["tokens"]]
        for j in range(len(nt) - 2):
            ngram_index[tuple(nt[j:j + 3])].add(vid)

    lab = {}
    stats = {}
    for name, key in [("Zamakhshari-Kashshaf", "kashshaf"), ("IbnAshur-Tahrir", "tahrir")]:
        path = glob.glob(os.path.join(D, SOURCES[name][0]), recursive=True)[0]
        l, hits = mine_istiara(path, verses, ngram_index)
        lab[key] = l
        stats[key] = {"term_hits": hits, "verses_labelled": len(l),
                      "tokens_labelled": sum(len(s) for s in l.values())}
        l2, hits2 = mine_istiara(path, verses, ngram_index, terms=FIG)
        lab[key + "_fig"] = l2
        stats[key + "_fig"] = {"term_hits": hits2, "verses_labelled": len(l2),
                               "tokens_labelled": sum(len(s) for s in l2.values())}
    for v in verses:
        k = (v["sura"], v["aya"])
        a = lab["kashshaf"].get(k, set())
        b = lab["tahrir"].get(k, set())
        v["istiara_kashshaf"] = sorted(a)
        v["istiara_tahrir"] = sorted(b)
        v["istiara"] = sorted(a | b)
        f = lab["kashshaf_fig"].get(k, set()) | lab["tahrir_fig"].get(k, set())
        v["rhetorical"] = sorted(a | b | f)
    # inter-source token agreement over verses discussed by both sources
    both = [v for v in verses if v["istiara_kashshaf"] or v["istiara_tahrir"]]
    agree_tok = tot_tok = 0
    n11 = n10 = n01 = n00 = 0
    for v in verses:
        a, b = set(v["istiara_kashshaf"]), set(v["istiara_tahrir"])
        for k in range(len(v["tokens"])):
            x, y = k in a, k in b
            n11 += x and y; n10 += x and not y; n01 += y and not x; n00 += (not x) and (not y)
    N = n11 + n10 + n01 + n00
    po = (n11 + n00) / N
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / N ** 2
    stats["token_kappa_between_sources"] = round((po - pe) / (1 - pe), 4)
    stats["verses_with_any_label"] = len(both)
    stats["positive_tokens_union"] = sum(len(v["istiara"]) for v in verses)
    stats["positive_tokens_rhetorical"] = sum(len(v["rhetorical"]) for v in verses)
    stats["verses_with_rhetorical"] = sum(1 for v in verses if v["rhetorical"])
    stats["total_quran_tokens"] = sum(len(v["tokens"]) for v in verses)
    json.dump(verses, open(os.path.join(OUT, "quran.json"), "w"), ensure_ascii=False)

    corpus = []
    for name, (pat, n) in SOURCES.items():
        path = glob.glob(os.path.join(D, pat), recursive=True)[0]
        s = sentences_from(path, name, n)
        stats["sent_" + name] = len(s)
        corpus.extend(s)
    random.shuffle(corpus)
    json.dump(corpus, open(os.path.join(OUT, "classical.json"), "w"), ensure_ascii=False)
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
