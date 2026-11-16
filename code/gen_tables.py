"""Generates tab_lit.tex, tab_sota.tex, tab_ablation.tex, tab_cost.tex and
numbers.tex (LaTeX macros for every number quoted in the text) from results/*.csv.
Rows that carry a reference are sorted by their citation number (read from main.aux)."""
import os, csv, json, re, collections, statistics
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


MAP2 = {"F05_test": "F05_test_sep", "F05_xdom": "F05_xdom_sep", "routed": "routed_test", "Q_all": "Q_all_sep", "Q_rhet": "Q_all_sep"}


def rd(n):
    if n == "routing.csv":   # routing analysis with calibration-selected theta (routing_extra.py)
        rows = list(csv.DictReader(open(os.path.join(RES, "routing2.csv"))))
        for r in rows:
            for a, b in MAP2.items():
                r[a] = r[b]
            r["theta"] = {"0.0": "0"}.get(r["theta"], r["theta"])
        return rows
    with open(os.path.join(RES, n)) as f:
        return list(csv.DictReader(f))


def citenum():
    aux = os.path.join(ROOT, "main.aux")
    m = {}
    if os.path.exists(aux):
        for l in open(aux):
            r = re.match(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", l)
            if r:
                m[r.group(1)] = int(r.group(2))
    return m


CN = citenum()
Y, N, P = r"\cmark", r"\xmark", r"\pmark"

LIT = [  # key, research focus, approach, reported finding, gap relative to this study
    ("alhafni2025", "Arabic GEC as text editing", "Data-derived edit tags on AraBERTv02, cascaded tagger",
     "80.5 F$_{0.5}$ on QALB-2014, 83.1 on ZAEBUC; about six times faster than the best seq2seq model on an A100 GPU",
     "Argmax decoding; false-edit rate not stated; Modern Standard Arabic essays only"),
    ("kwon2023", "Prompted LLMs for Arabic GEC", "Chain-of-thought and expert prompting of GPT-4 and others",
     "Five-shot GPT-4 at 67.82 F$_{0.5}$ on QALB-2014, below fine-tuned AraT5v2 at 70.81",
     "No control of over-correction; no serving cost"),
    ("mubarak2026", "Arabic grammar benchmark (Nahw)", "Evaluation of commercial and Arabic-centred LLMs",
     "GPT-4o about 67\%; best Arabic-centred 7B model about 42\%",
     "Evaluates knowledge of grammar, not correction decisions"),
    ("karpo2026", "Over-correction by prompted LLMs", "Taxonomy-based instructions and batching of several sentences",
     "78.32 F$_{0.5}$ on BEA-2019 with a lower edit rate",
     "Edit rate reduced on average, with no bound stated before deployment"),
    ("omelianchuk2020", "Tag-based English GEC (GECToR)", "Token-level edit tags, iterative refinement, KEEP bias",
     "Inference up to ten times faster than a transformer seq2seq corrector",
     "Decision thresholds tuned on development data, without a guarantee"),
    ("alhafni2023", "Seq2seq Arabic GEC", "AraBART and AraT5 with error-detection features and morphological preprocessing",
     "Up to 79.5 F$_{0.5}$ on QALB-2014",
     "Free rewriting raises the risk of changing correct words"),
    ("staruch2025", "Minimal-edit correction with LLMs", "Error-rate adaptation of decoder-only models; corrected references",
     "Adapts LLMs to minimal-edit correction",
     "No stated false-edit rate; English data"),
    ("alghallabi2025", "Arabic poetry understanding (Fann or Flop)", "LLM benchmark across genres and eras",
     "Benchmarks poetry understanding across genres and eras",
     "Not connected to any correction system"),
    ("attia2026", "Figurative language and pragmatics", "Evaluation of 22 LLMs on idioms and proverbs",
     "Accuracy drops by 14.07\% from understanding to pragmatic use",
     "Benchmark only; no use for constraining edits"),
    ("campos2024", "Conformal prediction for NLP", "Survey of conformal sets and risk control",
     "Maps guarantees available for NLP tasks",
     "No application to edit decisions in GEC"),
    ("angelopoulos2024", "Conformal risk control", "Threshold on a bounded loss that is monotone in the threshold",
     "Bound on the expected loss under exchangeability",
     "Generic; does not address routed pipelines or author shift"),
    ("dai2024", "Early exits in ML serving (Apparate)", "Exit thresholds added to served models and tuned online",
     "Balances latency and throughput in production serving",
     "No bound on an error rate of the served output"),
    ("chen2024frugal", "LLM cascades (FrugalGPT)", "Scorer decides whether to escalate to a larger model",
     "Matches the best single LLM at up to 98\% lower cost",
     "Composite system carries no stated error bound"),
    ("choi2021", "English metaphor detection (MelBERT)", "Late-interaction transformer based on metaphor identification theories",
     "Improved metaphor detection on English benchmarks",
     "English only; not linked to correction"),
    ("alsabahi2024", "Multi-head tagging for English GEC", "Sequence tagger with several prediction heads",
     "Applies multi-head tagging to correction",
     "Argmax decoding; no bound on false edits"),
    ("alosaimi2024", "Hybrid transformer--recurrent model", "AraBERT features passed through an LSTM",
     "97.12\% accuracy on the Sub-AHS sentiment dataset",
     "Fixed combination of views; classification, not correction"),
    ("nazir2025", "DNN partitioning on edge clusters", "Adaptive partitioning of model layers across nodes",
     "40\% lower communication overhead",
     "Routes by layer cost, not by calibrated uncertainty"),
    ("guo2026", "Data-parallel inference on edge devices", "Leader and follower nodes share inference work",
     "Up to 53.89\% shorter inference time with four followers",
     "Generic workloads; no accuracy constraint"),
]


def tab_lit():
    rows = sorted(LIT, key=lambda r: CN.get(r[0], 999))
    body = "\n".join(r"\citep{%s} & %s & %s & %s & %s \\" % r for r in rows)
    body += ("\n\\midrule\nRACE (this work) & Faithful Classical Arabic GEC under a latency target & "
             "Hybrid edit tagger, conformal threshold on the routed pipeline, entropy routing, risk-invariant core sizing & "
             "Expected false-edit rate below $\\alpha$ for every routing level; measured latency, memory and load & "
             "Synthetic errors; physical tests on one node \\\\")
    s = r"""\begin{table*}[pos=t]
\centering
\caption{Related work: focus, approach, reported findings and open gaps.}
\label{tab:lit}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.045\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.235\textwidth}@{}}
\toprule
Ref. & Research focus & Approach & Reported finding & Gap relative to this study \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % body
    open(os.path.join(ROOT, "tab_lit.tex"), "w").write(s)


def pm(m, s, k=1, scale=100):
    return f"{float(m)*scale:.{k}f}$\\pm${float(s)*scale:.{k}f}"


NAMES = {"ours": "RACE (ours)", "bilstm": "BiLSTM tagger", "trans_scratch": "Transformer tagger",
         "pt_tagger": "Pretrained tagger", "stack_hybrid": "Pretrained + stacked BiLSTM", "seq2seq": "Seq2seq + copy"}
REFS = {"bilstm": None, "trans_scratch": None, "pt_tagger": None, "stack_hybrid": None, "seq2seq": None}


def tab_sota(mac):
    agg = {r["model"]: r for r in rd("main_agg.csv")}
    crc = {(r["variant"], r["alpha"]): r for r in rd("crc_agg.csv")}
    base = [m for m in ["bilstm", "trans_scratch", "pt_tagger", "stack_hybrid", "seq2seq"] if m in agg]
    base = sorted(base, key=lambda m: (0, 0) if REFS[m] is None else (1, CN.get(REFS[m], 999)))
    lines = []
    for m in base + ["ours"]:
        a = agg[m]
        ref = (r"\citep{%s}" % REFS[m]) if REFS.get(m) else "--"
        ece = "--" if a["ECE_cal"] in ("nan", "") else f"{float(a['ECE_cal'])*100:.2f}"
        nm = NAMES[m] if m != "ours" else r"\textbf{RACE, argmax}"
        lines.append(f"{nm} & {ref} & {int(a['params'])/1e6:.1f} & {pm(a['P'],a['P_sd'])} & {pm(a['R'],a['R_sd'])} & {pm(a['F05'],a['F05_sd'])} & {pm(a['F05_xdom'],a['F05_xdom_sd'])} & {float(a['OCR'])*100:.2f} & {float(a['Q_all'])*100:.2f} & {float(a['Q_rhet'])*100:.2f} & {ece} \\\\")
    for var, vl in (("ours", r", $\beta$=0.5"), ("no-gate", r", $\beta$=0")):
     for al, lab in (("0.003", "0.3"), ("0.005", "0.5")):
      c = crc.get((var, al))
      if c:
        lines.append(r"\textbf{RACE, CRC $\alpha$=" + lab + r"\%" + vl + r"} & -- & " + f"{int(agg['ours']['params'])/1e6:.1f} & {pm(c['P'],c['P_sd'])} & {pm(c['R'],c['R_sd'])} & {pm(c['F05'],c['F05_sd'])} & -- & -- & {float(c['Q_all'])*100:.2f} & {float(c['Q_rhet'])*100:.2f} & -- \\\\")
    rep = [  # reported values (QALB-2014 test, M2 scorer), copied from the cited papers
        ("kwon2023", "GPT-4, 5-shot CoT", "69.46", "61.96", "67.82"),
        ("kwon2023", "AraT5v2 (fine-tuned)", "73.04", "63.09", "70.81"),
        ("alhafni2025", "Text-editing tagger", "84.5", "67.7", "80.5"),
        ("alhafni2025", "4-model ensemble", "89.7", "60.2", "81.7"),
        ("alhafni2023", "AraBART+Morph+GED", "83.9", "65.7", "79.5"),
    ]
    rep = sorted(rep, key=lambda r: CN.get(r[0], 999))
    rl = [f"{n} & \\citep{{{k}}} & -- & {p} & {r} & {f} & -- & -- & -- & -- & -- \\\\" for k, n, p, r, f in rep]
    s = r"""\begin{table*}[pos=t]
\centering
\caption{Comparison with re-implemented baselines and reported results.}
\label{tab:sota}
\footnotesize
\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{llccccccccc}
\toprule
Model & Ref. & Params (M) & P (\%%) & R (\%%) & F$_{0.5}$ (\%%) & F$_{0.5}$ x-author & OCR (\%%) & Q-OCR (\%%) & Q-rhet (\%%) & ECE (\%%) \\
\midrule
\multicolumn{11}{l}{\emph{Re-implemented, synthetic classical-Arabic test (mean$\pm$sd; seeds: %s)}}\\
%s
\midrule
\multicolumn{11}{l}{\emph{Reported values on QALB-2014 test (M2 scorer); different data, not directly comparable}}\\
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (SEEDNOTE(), "\n".join(lines), "\n".join(rl))
    open(os.path.join(ROOT, "tab_sota.tex"), "w").write(s)


def tab_ablation():
    r = rd("ablation.csv")
    lines = []
    for x in r:
        nm = x["name"].replace("w/o risk control (tau=0.5)", r"fixed $\tau$=0.5")
        if x["model"] == "ours":
            nm = r"\textbf{RACE (full)}"
        def g(k, scale=100, d=1):
            v = x.get(k, "")
            if v in ("", None):
                return "--"
            return f"{float(v)*scale:.{d}f}$\\pm${float(x.get(k + '_sd', 0) or 0)*scale:.{d}f}"
        lines.append(f"{nm} & {x['n']} & {g('F05')} & {g('F05_xdom')} & {g('ECE', d=2)} & {g('F05_crc')} & {g('Q_all_crc', d=2)} \\\\")
    s = r"""\begin{table}[pos=t]
\centering
\caption{Ablation study.}
\label{tab:ablation}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
{%%
\begin{tabular}{lcccccc}
\toprule
Variant & Seeds & F$_{0.5}$ & F$_{0.5}$ x-auth. & ECE & F$_{0.5}$@$\alpha$=0.5\%% & Q-OCR@0.5\%% \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(lines)
    open(os.path.join(ROOT, "tab_ablation.tex"), "w").write(s)


def SEEDNOTE():
    a = {r["model"]: int(r["n"]) for r in rd("main_agg.csv")}
    nb = sorted({v for m, v in a.items() if m not in ("ours", "seq2seq")})
    return f"RACE {a['ours']}, baselines {'/'.join(map(str, nb))}, seq2seq {a.get('seq2seq', 0)}"


def tab_cost(mac):
    agg = {r["model"]: r for r in rd("main_agg.csv")}
    lat = rd("latency_samples.csv")
    med = {t: statistics.median(float(x["ms"]) for x in lat if x["tier"] == t) for t in ("light", "heavy", "seq2seq")}
    p99 = {}
    for t in med:
        v = sorted(float(x["ms"]) for x in lat if x["tier"] == t); p99[t] = v[int(0.99 * (len(v) - 1))]
    meta = json.load(open(os.path.join(RES, "routing2_meta.json")))
    sim = json.load(open(os.path.join(RES, "sim_meta.json")))
    route = rd("routing.csv")
    rr = min(route, key=lambda x: abs(float(x["theta"]) - sim["theta"]))
    ddp = [json.loads(l) for l in open(os.path.join(RES, "ddp_timing.jsonl")) if json.loads(l)["steps"] == 60]
    rows = [
        ("RACE light tier", f"{int(agg['ours']['params'])/1e6:.2f}", f"{float(agg['ours']['train_time'])/60:.1f}", f"{med['light']:.1f}", f"{p99['light']:.1f}", f"{meta['ms_light']:.1f}"),
        ("RACE heavy tier (4 models)", f"{4*int(agg['ours']['params'])/1e6:.2f}", f"{4*float(agg['ours']['train_time'])/60:.1f} (4$\\times$)", f"{med['heavy']:.1f}", f"{p99['heavy']:.1f}", f"{meta['ms_heavy']:.1f}"),
        (f"Routed, $\\theta$={sim['theta']}", "--", "--", "--", "--", f"{float(rr['ms_per_sent']):.1f}"),
    ]
    if "seq2seq" in agg:
        rows.append(("Seq2seq + copy", f"{int(agg['seq2seq']['params'])/1e6:.2f}", f"{float(agg['seq2seq']['train_time'])/60:.1f}", f"{med['seq2seq']:.1f}", f"{p99['seq2seq']:.1f}", f"{statistics.mean(float(x['ms']) for x in lat if x['tier']=='seq2seq'):.1f}"))
    for m in ["pt_tagger", "stack_hybrid", "bilstm"]:
        if m in agg:
            rows.append((NAMES[m], f"{int(agg[m]['params'])/1e6:.2f}", f"{float(agg[m]['train_time'])/60:.1f}", "--", "--", "--"))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    dd = f"{ddp[0]['samples_per_s']:.1f} / {ddp[1]['samples_per_s']:.1f} / {ddp[2]['samples_per_s']:.1f} (1 thr. / 2 thr. / DDP 2 proc.)"
    s = r"""\begin{table}[pos=t]
\centering
\caption{Cost and efficiency on one CPU core.}
\label{tab:cost}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
{%%
\begin{tabular}{lccccc}
\toprule
System & Params (M) & Train (min) & p50 (ms) & p99 (ms) & Mean (ms/sent.) \\
\midrule
%s
\midrule
MLM pretraining & \multicolumn{5}{l}{samples/s: %s}\\
\bottomrule
\end{tabular}}
\end{table}
""" % (body, dd)
    open(os.path.join(ROOT, "tab_cost.tex"), "w").write(s)


WORKS = [("in-domain", "In-domain", "4th--8th", "mixed"), ("Jurjani", "al-Jurj\\={a}n\\={i}", "5th", "rhetoric"),
         ("Jahiz", "al-J\\={a}\\d{h}i\\d{z}", "3rd", "adab"), ("Mubarrad", "al-Mubarrad", "3rd", "philology"),
         ("IbnAbdRabbih", "Ibn \\textquotesingle Abd Rabbih", "4th", "anthology"), ("IbnKhaldun", "Ibn Khald\\={u}n", "9th", "history")]


def tab_domains():
    r = rd("multidomain.csv")
    order = ["bilstm", "trans_scratch", "seq2seq", "stack_hybrid", "pt_tagger", "ours"]
    lines = []
    for m in order:
        cells = []
        for k, *_ in WORKS:
            v = [float(x["F05"]) * 100 for x in r if x["model"] == m and x["domain"] == k]
            cells.append(f"{statistics.mean(v):.1f}$\\pm${statistics.stdev(v):.1f}" if len(v) > 1 else "--")
        ho = [statistics.mean(float(x["F05"]) * 100 for x in r if x["model"] == m and x["domain"] == k) for k, *_ in WORKS[1:]]
        nm = r"\textbf{RACE}" if m == "ours" else NAMES[m]
        lines.append(f"{nm} & " + " & ".join(cells) + f" & {statistics.mean(ho):.1f}" + r" \\")
    hdr1 = " & ".join(w[1] for w in WORKS)
    hdr2 = " & ".join(f"{w[2]} c.\\,AH, {w[3]}" if w[0] != "in-domain" else "test split" for w in WORKS)
    s = r"""\begin{table*}[pos=t]
\centering
\caption{F$_{0.5}$ (\%%) in domain and on five held-out works (argmax decoding).}
\label{tab:domains}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{lccccccc}
\toprule
Model & %s & Held-out mean \\
 & %s & \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table*}
""" % (hdr1, "{\\scriptsize " + hdr2.replace(" & ", "} & {\\scriptsize ") + "}", "\n".join(lines))
    open(os.path.join(ROOT, "tab_domains.tex"), "w").write(s)


def tab_cost_all():
    c = {r["model"]: r for r in rd("cost_all.csv")}
    order = [("bilstm", "BiLSTM tagger"), ("trans_scratch", "Transformer (scratch)"), ("pt_tagger", "Pretrained tagger"),
             ("stack_hybrid", "Pretrained + stacked BiLSTM"), ("ours", r"\textbf{RACE} (light tier)"),
             ("ours_ensemble4", "RACE ensemble (heavy tier)"), ("seq2seq", "Seq2seq + copy")]
    agg = {r["model"]: r for r in rd("main_agg.csv")}
    rows = []
    for k, nm in order:
        if k not in c:
            continue
        x = c[k]
        f = agg["ours"]["F05"] if k.startswith("ours") else agg.get(k, {}).get("F05")
        fs = f"{float(f)*100:.1f}" if f and k != "ours_ensemble4" else "--"
        base_k = "ours" if k.startswith("ours") else k
        tm = float(agg[base_k]["train_time"]) / 60 * (4 if k == "ours_ensemble4" else 1) if base_k in agg else float(x["train_min"])
        rows.append(f"{nm} & {float(x['params_M']):.1f} & {tm:.1f} & {float(x['p50_ms']):.1f} & "
                    f"{float(x['p99_ms']):.1f} & {float(x['tp_b64']):.0f} & {fs}" + r" \\")
    s = r"""\begin{table}[pos=t]
\centering
\caption{Cost of every model on one CPU core.}
\label{tab:cost}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
{%%
\begin{tabular}{lcccccc}
\toprule
Model & Params (M) & Train (min) & p50 (ms) & p99 (ms) & Sent./s (b=64) & F$_{0.5}$ \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(rows)
    open(os.path.join(ROOT, "tab_cost.tex"), "w").write(s)


LIT2 = [  # ref, method, decision rule, guarantee, assumption or limitation for GEC serving
    ("omelianchuk2020", "GECToR edit tagging", "Edit applied when its probability passes a KEEP bias and a minimum error probability tuned on development data",
     "None", "Tuned thresholds may not transfer to new text"),
    ("desai2020", "Calibration of pretrained transformers", "Temperature scaling of output probabilities",
     "None", "Calibrated on average; a thresholded decision can still fail often"),
    ("campos2024", "Conformal prediction for NLP (survey)", "Conformal prediction sets and risk control",
     "Coverage or risk, depending on method", "Survey; no GEC decision rule"),
    ("angelopoulos2024", "Conformal risk control", "Smallest threshold whose corrected empirical risk is at most $\\alpha$",
     "Expected loss $\\le\\alpha$", "Requires exchangeable data and a loss monotone in the threshold"),
    ("chen2024frugal", "LLM cascades (FrugalGPT)", "A learned scorer decides whether to escalate a query",
     "None", "Escalation changes quality, which must be re-measured"),
    ("dai2024", "Early exits in ML serving (Apparate)", "Exit thresholds adapted online to the observed workload",
     "None", "Thresholds follow latency, not an error bound"),
    ("staruch2025", "Minimal-edit LLM correction", "Fine-tuning that adapts the error rate of the training data",
     "None", "Generative decoding gives no monotone score to threshold"),
    ("alhafni2025", "Arabic text-editing GEC", "Most probable edit tag per word (argmax)",
     "None", "Any edit that wins the argmax is applied"),
    ("karpo2026", "Context and over-correction", "Longer input context and taxonomy-based prompts",
     "None", "Lower edit rate on average only"),
    ("angelopoulos2025ltt", "Learn-then-Test", "Multiple testing over candidate thresholds",
     "Risk $\\le\\alpha$ with probability $\\ge1-\\delta$", "Conservative at strict levels"),
    ("bates2021rcps", "Risk-controlling prediction sets", "Concentration bound on the risk (Hoeffding--Bentkus)",
     "Risk $\\le\\alpha$ with high probability", "Needs larger calibration sets for the same $\\alpha$"),
]


def tab_lit2():
    rows = sorted(LIT2, key=lambda r: CN.get(r[0], 999))
    body = "\n".join(r"\citep{%s} & %s & %s & %s & %s \\" % r for r in rows)
    body += ("\n\\midrule\nRACE (this work) & Risk-controlled edit tagging with routing & Edit when the calibrated probability exceeds $\\hat\\tau+\\beta r_i$; one $\\hat\\tau$ for the whole routed pipeline & "
             "Expected false-edit rate $\\le\\alpha$ for every routing level & Exchangeability; recalibration needed on new authors \\\\")
    s = r"""\begin{table*}[pos=t]
\centering
\caption{Decision rules of related methods: how they decide to edit or escalate.}
\label{tab:lit2}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.045\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.3\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.265\textwidth}@{}}
\toprule
Ref. & Method & Decision rule & Formal guarantee & Assumption or limitation \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % body
    open(os.path.join(ROOT, "tab_lit2.tex"), "w").write(s)


def numbers_extra(M):
    put = lambda k, v: M.__setitem__(k, v)
    if os.path.exists(os.path.join(RES, "hp_eval.csv")):
        from scipy import stats as _st
        import numpy as _np
        h = rd("hp_eval.csv")
        g = lambda t, k: [float(x[k]) for x in h if x["tag"] == t]
        for nm, t in (("HpOurs", "ours_hpbest"), ("HpPt", "pt_tagger_hpbest")):
            put(nm + "F", f"{statistics.mean(g(t, 'F05'))*100:.1f}"); put(nm + "Fx", f"{statistics.mean(g(t, 'F05_xdom'))*100:.1f}")
            put(nm + "Fc", f"{statistics.mean(g(t, 'F05_crc'))*100:.1f}"); put(nm + "Qc", f"{statistics.mean(g(t, 'Q_all_crc'))*100:.2f}")
            put(nm + "Rc", f"{statistics.mean(g(t, 'risk_crc'))*100:.2f}"); put(nm + "N", len(g(t, 'F05')))
        a = _np.array(g("ours_hpbest", "F05")); b = _np.array(g("pt_tagger_hpbest", "F05"))
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = _np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); d = a.mean() - b.mean()
        put("HpDiff", f"{d*100:.1f}"); put("HpLower", f"{(d - _st.t.ppf(.95, df) * se)*100:.1f}")
        p = _st.ttest_ind(a, b, equal_var=False).pvalue
        put("HpP", f"{p:.1g}" if p >= 1e-3 else "<0.001")
    if os.path.exists(os.path.join(RES, "leakfree_eval.csv")):
        lf = rd("leakfree_eval.csv"); ref = [x for x in rd("leakfree_ref_eval.csv") if int(x["seed"]) <= 3]
        mm = lambda rows, t, k: statistics.mean(float(x[k]) for x in rows if x["tag"] == t) * 100
        for nm, rows, t in (("LfOurs", lf, "ours_lf"), ("LfQc", lf, "abl_quranclean_lf"), ("RefOurs", ref, "ours"), ("RefQc", ref, "abl_quranclean")):
            put(nm + "F", f"{mm(rows, t, 'F05'):.1f}"); put(nm + "Fx", f"{mm(rows, t, 'F05_xdom'):.1f}")
            put(nm + "Q", f"{mm(rows, t, 'Q_all'):.2f}"); put(nm + "Qc", f"{mm(rows, t, 'Q_all_crc'):.2f}")
            put(nm + "Fc", f"{mm(rows, t, 'F05_crc'):.1f}")
        st_ = json.load(open(os.path.join(RES, "leakfree_stats.json")))
        put("LfTrainRemoved", f"{(1 - st_['train'][1] / st_['train'][0]) * 100:.0f}")
        put("LfPreRemoved", f"{(1 - st_['pretrain'][1] / st_['pretrain'][0]) * 100:.0f}")
    if os.path.exists(os.path.join(RES, "multidomain.csv")):
        r = rd("multidomain.csv")
        mean = lambda m, k: statistics.mean(float(x["F05"]) * 100 for x in r if x["model"] == m and x["domain"] == k)
        diffs = [round(mean("ours", k), 1) - round(mean("pt_tagger", k), 1) for k, *_ in WORKS[2:]]   # as shown in the table
        put("MdGainMin", f"{min(diffs):.1f}"); put("MdGainMax", f"{max(diffs):.1f}")
        put("MdOursHeld", f"{statistics.mean(mean('ours', k) for k, *_ in WORKS[1:]):.1f}")
        put("MdPtHeld", f"{statistics.mean(mean('pt_tagger', k) for k, *_ in WORKS[1:]):.1f}")
        put("MdOursIn", f"{mean('ours', 'in-domain'):.1f}")
        c = rd("multidomain_crc.csv")
        sel = lambda d, b, a: [x for x in c if x["domain"] == d and float(x["beta"]) == b and float(x["alpha"]) == a]
        new = [k for k, *_ in WORKS[2:]]
        ri = [statistics.mean(float(x["risk_in"]) * 100 for x in sel(k, 0.5, 0.005)) for k in new]
        rr = [statistics.mean(float(x["risk_rec"]) * 100 for x in sel(k, 0.5, 0.005)) for k in new]
        put("ShiftRiskInMin", f"{min(ri):.2f}"); put("ShiftRiskInMax", f"{max(ri):.2f}")
        put("ShiftRiskRecMin", f"{min(rr):.2f}"); put("ShiftRiskRecMax", f"{max(rr):.2f}")
        put("ShiftJurjani", f"{statistics.mean(float(x['risk_in']) * 100 for x in sel('Jurjani', 0.5, 0.005)):.2f}")
        fi = [statistics.mean(float(x["F05_in"]) * 100 for x in sel(k, 0.5, 0.005)) for k in new]
        fr = [statistics.mean(float(x["F05_rec"]) * 100 for x in sel(k, 0.5, 0.005)) for k in new]
        put("ShiftFLoss", f"{statistics.mean(a - b for a, b in zip(fi, fr)):.1f}")
        # worst seed-level and mean at all alphas after recalibration
        worst = max(statistics.mean(float(x["risk_rec"]) for x in sel(k, b, a)) / a for k in new for b in (0.0, 0.5) for a in (0.003, 0.005, 0.01))
        put("ShiftRecWorstRatio", f"{worst:.2f}")
    if os.path.exists(os.path.join(RES, "riskcontrol_baselines.csv")):
        r = {(x["method"], float(x["alpha"])): x for x in rd("riskcontrol_baselines.csv")}
        for m in ("CRC", "Naive", "LTT"):
            put(f"Rc{m}ViolFive", f"{float(r[(m, 0.005)]['viol_freq'])*100:.0f}")
            put(f"Rc{m}RiskFive", f"{float(r[(m, 0.005)]['mean_risk'])*100:.2f}")
            put(f"Rc{m}FFive", f"{float(r[(m, 0.005)]['tokF05'])*100:.1f}")
            put(f"Rc{m}FOne", f"{float(r[(m, 0.001)]['tokF05'])*100:.1f}")
            put(f"Rc{m}ViolMax", f"{max(float(v['viol_freq']) for (mm, a), v in r.items() if mm == m)*100:.0f}")
    if os.path.exists(os.path.join(RES, "cost_all.csv")):
        c = {x["model"]: x for x in rd("cost_all.csv")}
        put("CostOursPfifty", f"{float(c['ours']['p50_ms']):.1f}"); put("CostPtPfifty", f"{float(c['pt_tagger']['p50_ms']):.1f}")
        put("CostEnsPfifty", f"{float(c['ours_ensemble4']['p50_ms']):.1f}")
        if "seq2seq" in c:
            put("CostSeqPfifty", f"{float(c['seq2seq']['p50_ms']):.1f}")
            put("CostSeqRatio", f"{float(c['seq2seq']['p50_ms'])/float(c['ours']['p50_ms']):.0f}")
        put("CostOursOverPt", f"{(float(c['ours']['p50_ms'])/float(c['pt_tagger']['p50_ms'])-1)*100:.0f}")
        rsel = [x for x in rd("routing.csv") if abs(float(x["theta"]) - json.load(open(os.path.join(RES, "routing2_meta.json")))["theta_cal"]) < 1e-9][0]
        iso = (float(c['ours']['mean_ms']) + float(rsel['routed']) * float(c['ours_ensemble4']['mean_ms'])) / float(c['ours_ensemble4']['mean_ms'])
        put("RouteCostShareIso", f"{iso*100:.0f}")
        put("CostOursTp", f"{float(c['ours']['tp_b64']):.0f}"); put("CostPtTp", f"{float(c['pt_tagger']['tp_b64']):.0f}")


BOOTNAMES = [("bilstm", "BiLSTM tagger"), ("trans_scratch", "Transformer (scratch)"), ("stack_hybrid", "Pretrained + stacked BiLSTM"),
             ("pt_tagger", "Pretrained tagger"), ("seq2seq", "Seq2seq + copy")]


def tab_boot():
    f = os.path.join(RES, "bootstrap_sig.csv")
    if not os.path.exists(f):
        return
    r = rd("bootstrap_sig.csv")
    get = lambda st, sp, m: next((x for x in r if x["setting"] == st and x["split"] == sp and x["baseline"] == m), None)
    def cell(x):
        if x is None:
            return "-- & --"
        p = float(x["p"]); ps = "$<$0.001" if p <= 0.0005 else f"{p:.3f}"
        return f"{float(x['diff']):+.1f} [{float(x['lo']):+.1f}, {float(x['hi']):+.1f}] & {ps}"
    rows = []
    for m, nm in BOOTNAMES:
        rows.append(nm + " & " + " & ".join(cell(get(st, sp, m)) for st in ("argmax", "crc") for sp in ("test", "xdom")) + r" \\")
    s = r"""\begin{table*}[pos=t]
\centering
\caption{Paired bootstrap of F$_{0.5}$ differences, RACE minus baseline (points).}
\label{tab:boot}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{Argmax decoding} & \multicolumn{4}{c}{Conformal threshold, $\alpha=0.5\%$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Baseline & In domain [95\% CI] & $p$ & Cross-author [95\% CI] & $p$ & In domain [95\% CI] & $p$ & Cross-author [95\% CI] & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
    open(os.path.join(ROOT, "tab_boot.tex"), "w").write(s)


def numbers_boot(M):
    put = lambda k, v: M.__setitem__(k, v)
    f = os.path.join(RES, "bootstrap_sig.csv")
    if os.path.exists(f):
        r = rd("bootstrap_sig.csv")
        g = lambda st, sp, m: next(x for x in r if x["setting"] == st and x["split"] == sp and x["baseline"] == m)
        for key, st, sp in (("BsPtArg", "argmax", "test"), ("BsPtArgX", "argmax", "xdom"), ("BsPtCrc", "crc", "test"), ("BsPtCrcX", "crc", "xdom")):
            x = g(st, sp, "pt_tagger")
            put(key, f"{float(x['diff']):.1f}"); put(key + "Lo", f"{float(x['lo']):.1f}"); put(key + "Hi", f"{float(x['hi']):.1f}")
            put(key + "P", f"{float(x['p']):.3f}")
        x = g("crc", "xdom", "trans_scratch"); put("BsTsCrcXP", f"{float(x['p']):.2f}")
        put("BsReps", "2,000")
    f = os.path.join(RES, "mem_profile.json")
    if os.path.exists(f):
        mp_ = json.load(open(f))
        put("MemRuntime", f"{mp_['light']['rss_runtime_mb']/1024:.2f}")
        put("MemLight", f"{mp_['light']['peak_serving_mb']/1024:.2f}"); put("MemHeavy", f"{mp_['heavy']['peak_serving_mb']/1024:.2f}")
        put("MemLightModel", f"{mp_['light']['rss_loaded_mb']-mp_['light']['rss_runtime_mb']:.0f}")
        put("MemHeavyModel", f"{mp_['heavy']['rss_loaded_mb']-mp_['heavy']['rss_runtime_mb']:.0f}")
    if "ShiftRiskInMax" in M:
        put("ShiftTvLb", f"{float(M['ShiftRiskInMax'])-0.5:.2f}")



def tab_design():
    f = os.path.join(RES, "design_select.csv")
    if not os.path.exists(f):
        return
    r = rd("design_select.csv")
    rows = []
    for x in r:
        sav = 100 * (1 - int(x["cores"]) / int(x["cores_heavy"]))
        rows.append(f"{int(x['nu']):,} & {int(x['cores_light'])} & {float(x['F05_test_light']):.1f} & {int(x['cores_heavy'])} & {float(x['F05_test_heavy']):.1f} & {float(x['theta']):.1f} & {int(x['cores'])} & {float(x['F05_test']):.1f} & {sav:.0f}" + r" \\")
    s = r"""\begin{table}[pos=t]
\centering
\caption{Cores needed for a derated p99 target of 102\,ms, $\alpha=0.5\%$.}
\label{tab:design}
\footnotesize
\setlength{\tabcolsep}{2.4pt}
{%
\begin{tabular}{rcccccccc}
\toprule
 & \multicolumn{2}{c}{Light only} & \multicolumn{2}{c}{All to ensemble} & \multicolumn{4}{c}{Selected configuration} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}
$\nu$ (req/s) & Cores & F$_{0.5}$ & Cores & F$_{0.5}$ & $\theta$ & Cores & F$_{0.5}$ & Saved (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(os.path.join(ROOT, "tab_design.tex"), "w").write(s)


def numbers_design(M):
    put = lambda k, v: M.__setitem__(k, v)
    f = os.path.join(RES, "design_select.csv")
    if not os.path.exists(f):
        return
    r = rd("design_select.csv"); meta = json.load(open(os.path.join(RES, "design_meta.json")))
    x = next(y for y in r if int(y["nu"]) == 2000)
    put("DsCoresSel", x["cores"]); put("DsCoresHeavy", x["cores_heavy"]); put("DsCoresLight", x["cores_light"])
    put("DsFSel", f"{float(x['F05_test']):.1f}"); put("DsFHeavy", f"{float(x['F05_test_heavy']):.1f}"); put("DsFLight", f"{float(x['F05_test_light']):.1f}")
    put("DsThetaSel", f"{float(x['theta']):.1f}")
    sav = [100 * (1 - int(y["cores"]) / int(y["cores_heavy"])) for y in r]
    put("DsSaveMin", f"{min(sav):.0f}"); put("DsSaveMax", f"{max(sav):.0f}")
    put("DsSlo", f"{meta['derated_slo_ms']:.0f}"); put("DsEps", f"{meta['eps_points']:.1f}")
    put("DsSimErr", f"{100*meta['sim_p99_underestimate']:.0f}")
    rs = rd("design_space.csv")
    put("DsRiskMax", f"{max(float(y['risk_test']) for y in rs):.2f}")


def fig_recal():
    f = os.path.join(RES, "monitor_synthetic.csv")
    if not os.path.exists(f):
        return
    r = [x for x in rd("monitor_synthetic.csv") if x.get("rec200_risk")]
    sizes = [100, 200, 300, 500, 1000]
    risk = [np.mean([float(x[f"rec{n}_risk"]) for x in r]) for n in sizes]
    err = [np.mean([float(x[f"rec{n}_errtok_edited"]) for x in r]) for n in sizes]
    co = lambda v: " ".join(f"({n},{y:.3f})" for n, y in zip(sizes, v))
    s = r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, width=0.42\linewidth, height=0.3\linewidth,
 xmode=log, log basis x=10, xtick={100,200,500,1000}, xticklabels={100,200,500,1000}, xlabel={Labelled sentences}, tick label style={font=\scriptsize},
 label style={font=\footnotesize}, ymajorgrids, grid style={black!8}, xmin=85, xmax=1150]
\nextgroupplot[ylabel={False-edit rate (\%%)}, ymin=0, ymax=0.62]
\addplot[c1, thick, mark=*, mark size=1.4pt] coordinates {%s};
\addplot[red!70!black, dashed, thick, domain=85:1150, samples=2] {0.5};
\node[font=\tiny, anchor=south east, text=red!70!black] at (axis cs:1100,0.5) {$\alpha$};
\nextgroupplot[ylabel={Erroneous words edited (\%%)}, ymin=0, ymax=28]
\addplot[c2, thick, mark=square*, mark size=1.4pt] coordinates {%s};
\end{groupplot}
\end{tikzpicture}
""" % (co(risk), co(err))
    open(os.path.join(ROOT, "figures", "fig_recal.tex"), "w").write(s)


def numbers_monitor(M):
    put = lambda k, v: M.__setitem__(k, v)
    f = os.path.join(RES, "monitor_synthetic.csv")
    if not os.path.exists(f):
        return
    r = rd("monitor_synthetic.csv"); meta = json.load(open(os.path.join(RES, "monitor_synthetic_meta.json")))
    sh = [x for x in r if x["domain"] not in ("in-domain",)]
    put("MonEditCal", f"{meta['edit_rate_cal']:.2f}")
    put("MonEditShiftMin", f"{min(float(x['edit_rate']) for x in sh):.2f}"); put("MonEditShiftMax", f"{max(float(x['edit_rate']) for x in sh):.2f}")
    det = sum(1 for x in sh if x["detect_after_u"] or x["detect_after_e"]); put("MonDetected", str(det)); put("MonShifted", str(len(sh)))
    ind = next(x for x in r if x["domain"] == "in-domain")
    put("MonFalseAlarm", "no" if not (ind["detect_after_u"] or ind["detect_after_e"]) else "one")
    rr = [x for x in r if x.get("rec200_risk")]
    for n, k in ((200, "Two"), (300, "Three"), (500, "Five"), (1000, "Thou")):
        put(f"Rec{k}Risk", f"{np.mean([float(x[f'rec{n}_risk']) for x in rr]):.2f}")
        put(f"Rec{k}Err", f"{np.mean([float(x[f'rec{n}_errtok_edited']) for x in rr]):.1f}")
    put("RecMaxViol", f"{100*max(float(x[f'rec{n}_viol']) for x in rr for n in (200,300,500)):.1f}")


REALD = os.path.join(RES, "real")


def _rr(name):
    with open(os.path.join(REALD, name)) as f:
        return list(csv.DictReader(f))


def numbers_real(M):
    put = lambda k, v: M.__setitem__(k, v)
    st = os.path.join(REALD, "qalb_stats.json")
    if os.path.exists(st):
        q = json.load(open(st))
        put("RealTrChunks", f"{q['train']['chunks']:,}"); put("RealTrWords", f"{q['train']['words']/1e6:.2f}\\,M")
        put("RealTrEdit", f"{100*q['train']['edited_word_rate']:.1f}"); put("RealTags", f"{q['tags']:,}")
        put("RealDevChunks", f"{q['dev']['chunks']:,}"); put("RealCalChunks", f"{q['cal']['chunks']:,}")
        put("RealTestChunks", f"{q['test']['chunks']:,}"); put("RealTestWords", f"{q['test']['words']:,}")
        put("RealZaeChunks", f"{q['xdom']['chunks']:,}"); put("RealZaeCalChunks", f"{q['md_ZAEBUC_cal']['chunks']:,}")

    fm = os.path.join(REALD, "monitor_real.csv")
    if os.path.exists(fm):
        mr = {x["domain"]: x for x in _rr("monitor_real.csv")}
        ind = mr["in-domain"]
        put("MonRealWin", ind["windows"]); put("MonRealWinSize", ind["window_size"])
        put("MonRealFalse", str(int(round(float(ind["alarm_rate"]) * int(ind["windows"])))))
        put("MonRealFalseE", str(int(round(float(ind["alarm_rate_edit"]) * int(ind["windows"])))))
        for k, nm in (("ZAEBUC", "Z"), ("Quran", "Q")):
            x = mr[k]
            dets = [int(v) for v in (x["detect_after_u"], x["detect_after_e"]) if v not in ("", "None")]
            put(f"MonReal{nm}Det", f"{min(dets):,}" if dets else "--")
            put(f"MonReal{nm}Alarm", f"{100*float(x['alarm_rate']):.0f}")
            put(f"MonReal{nm}RiskIn", f"{float(x['risk_in']):.2f}")
            put(f"MonReal{nm}RiskRec", f"{float(x['risk_rec']):.2f}")
        dd = [int(x[k]) for x in (mr["ZAEBUC"], mr["Quran"]) for k in ("detect_after_u", "detect_after_e") if x[k] not in ("", "None")]
        put("MonRealDet", f"{max(min(int(v) for v in (mr[d]['detect_after_u'], mr[d]['detect_after_e']) if v not in ('', 'None')) for d in ('ZAEBUC', 'Quran')):,}")
        put("MonRealEditCal", f"{json.load(open(os.path.join(REALD, 'monitor_real_meta.json')))['edit_rate_cal']:.1f}")
    f = os.path.join(REALD, "real_main.csv")
    if not os.path.exists(f):
        return
    r = _rr("real_main.csv")
    mean = lambda m, k: float(np.mean([float(x[k]) for x in r if x["model"] == m]))
    sd = lambda m, k: float(np.std([float(x[k]) for x in r if x["model"] == m], ddof=1))
    for m, nm in (("ours", "Ours"), ("pt_tagger", "Pt")):
        for k, kk in (("F05", "F"), ("P", "P"), ("R", "R"), ("OCR", "Ocr"), ("F05_zaebuc", "Z"), ("Q_all", "Q")):
            put(f"Real{nm}{kk}", f"{mean(m, k):.1f}")
        put(f"Real{nm}Fsd", f"{sd(m, 'F05'):.1f}")
        put(f"Real{nm}N", str(sum(1 for x in r if x["model"] == m)))
    put("RealOracle", f"{float(r[0]['oracle_test']):.1f}"); put("RealOracleZ", f"{float(r[0]['oracle_zaebuc']):.1f}")
    put("RealFDiff", f"{mean('ours','F05')-mean('pt_tagger','F05'):.1f}")
    c = _rr("real_crc.csv")
    cm = lambda m, a, k: float(np.mean([float(x[k]) for x in c if x["model"] == m and abs(float(x["alpha"]) - a) < 1e-9]))
    for a, ak in ((0.005, "Five"), (0.01, "One"), (0.02, "Two"), (0.05, "Fifty")):
        for m, nm in (("ours", "Ours"), ("pt_tagger", "Pt")):
            put(f"RealRisk{nm}{ak}", f"{cm(m, a, 'risk_test'):.2f}"); put(f"RealF{nm}{ak}", f"{cm(m, a, 'F05_test'):.1f}")
            put(f"RealZin{nm}{ak}", f"{cm(m, a, 'risk_zaebuc_in'):.2f}"); put(f"RealZrec{nm}{ak}", f"{cm(m, a, 'risk_zaebuc_rec'):.2f}")
            put(f"RealZFin{nm}{ak}", f"{cm(m, a, 'F05_zaebuc_in'):.1f}"); put(f"RealZFrec{nm}{ak}", f"{cm(m, a, 'F05_zaebuc_rec'):.1f}")
            put(f"RealQ{nm}{ak}", f"{cm(m, a, 'risk_quran'):.1f}")
    put("RealQRatio", f"{cm('ours', 0.005, 'risk_quran') / 0.5:.0f}")
    rs = _rr("real_resplit.csv")[0]
    put("RealResMean", f"{float(rs['mean_risk']):.2f}"); put("RealResViol", f"{100*float(rs['share_above_alpha']):.0f}")
    b = _rr("real_boot.csv")
    for x in b:
        k = "RealBsArg" if x["mode"] == "argmax" else "RealBsCrc"
        put(k, f"{float(x['diff']):.1f}"); put(k + "Lo", f"{float(x['lo']):.1f}"); put(k + "Hi", f"{float(x['hi']):.1f}")
        p_ = float(x["p"]); put(k + "P", "<0.001" if p_ <= 0.0005 else f"={p_:.3f}")


def tab_real():
    f = os.path.join(REALD, "real_crc.csv")
    if not os.path.exists(f):
        return
    r = _rr("real_main.csv"); c = _rr("real_crc.csv")
    mean = lambda rows, k: float(np.mean([float(x[k]) for x in rows]))
    sdv = lambda rows, k: float(np.std([float(x[k]) for x in rows], ddof=1))
    lines = []
    for m, nm in (("pt_tagger", "Pretrained tagger"), ("ours", "RACE")):
        rows = [x for x in r if x["model"] == m]
        lines.append(f"{nm} & argmax & -- & {mean(rows,'P'):.1f} & {mean(rows,'R'):.1f} & {mean(rows,'F05'):.1f}$\\pm${sdv(rows,'F05'):.1f} & {mean(rows,'OCR'):.2f} & {mean(rows,'F05_zaebuc'):.1f} & -- & {mean(rows,'Q_all'):.2f} \\\\")
        for a in (0.005, 0.01, 0.02, 0.05):
            cr = [x for x in c if x["model"] == m and abs(float(x["alpha"]) - a) < 1e-9]
            lines.append(f" & $\\alpha={100*a:g}\\%$ & {mean(cr,'risk_test'):.2f} & -- & -- & {mean(cr,'F05_test'):.1f} & -- & {mean(cr,'F05_zaebuc_in'):.1f} & {mean(cr,'risk_zaebuc_in'):.2f} / {mean(cr,'risk_zaebuc_rec'):.2f} & {mean(cr,'risk_quran'):.2f} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    s = r"""\begin{table*}[pos=t]
\centering
\caption{Public human-annotated data: QALB-2014 test and ZAEBUC test (mean of three seeds).}
\label{tab:real}
\footnotesize
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{5}{c}{QALB-2014 test} & \multicolumn{2}{c}{ZAEBUC test} & Qur'an \\
\cmidrule(lr){3-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}
Model & Decision & False edits (\%) & P & R & F$_{0.5}$ & OCR & F$_{0.5}$ & False edits (\%), QALB / own $\hat\tau$ & Words edited (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(ROOT, "tab_real.tex"), "w").write(s)

def tab_hparams():
    """Hyperparameters, search ranges and the selection made on the development split."""
    hp = rd("hp_search.csv") if os.path.exists(os.path.join(RES, "hp_search.csv")) else []
    sel = {}
    for r in hp:
        if r["model"] not in sel or float(r["F05_dev"]) > float(sel[r["model"]]["F05_dev"]):
            sel[r["model"]] = r
    def chosen(m):
        if m not in sel:
            return "--"
        c = sel[m]["cfg"]
        lr = re.search(r"lr([0-9.e-]+)", c); ep = re.search(r"_e(\d+)", c)
        lrs = {"3e-4": "$3\\times10^{-4}$", "6e-4": "$6\\times10^{-4}$", "1e-3": "$10^{-3}$"}
        return f"{lrs.get(lr.group(1), lr.group(1)) if lr else lrs['6e-4']}, {ep.group(1) if ep else '3'} epochs"
    rows = [
        ("Encoder", "$d$=256, 4 layers, 4 heads, FFN 512", "fixed (CPU budget)"),
        ("BiGRU branch", "1 layer, 128 units per direction", "fixed"),
        ("MLM pretraining", "2 epochs, 15\\% masking, batch 64", "fixed"),
        ("Peak learning rate", "$6\\times10^{-4}$, one-cycle AdamW", "default; grid \\{3, 6, 10\\}$\\times10^{-4}$ in tuning study"),
        ("Epochs / batch", "3 / 64", "default (CPU budget); grid \\{3, 5\\} in tuning study"),
        ("Dropout", "0.1", "fixed"),
        ("Auxiliary weights $\\lambda_d=\\lambda_r$; rhetoric class weight", "0.5; 8", "fixed"),
        ("Temperature $T$", "fitted per model", "NLL on dev"),
        ("Risk level $\\alpha$", "0.5\\% (main)", "operator choice; 0.1--1.5\\% reported"),
        ("Rhetoric margin $\\beta$", "0.5", "swept 0--1.5 (Fig.~\\ref{fig:beta})"),
        ("Threshold grid $\\Lambda$", "0, 0.01, \\dots, 1", "fixed"),
        ("Routing level $\\theta$", "\\RouteThetaSel", "rule on dev F$_{0.5}$"),
    ]
    body = "\n".join(f"{a} & {b} & {c}" + r" \\" for a, b, c in rows)
    if sel:
        body += "\n\\midrule\n" + "\n".join(f"Selected for {NAMES.get(m, 'RACE').lower() if m != 'ours' else 'RACE'} & {chosen(m)} & dev F$_{{0.5}}$ {float(sel[m]['F05_dev'])*100:.1f} (seed 1)" + r" \\" for m in sorted(sel))
    t = r"""\begin{table}[pos=t]
\centering
\caption{Hyperparameters and how they were chosen.}
\label{tab:hparams}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
{%%
\begin{tabular}{lll}
\toprule
Item & Value & Selection \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % body
    open(os.path.join(ROOT, "tab_hparams.tex"), "w").write(t)


def numbers():
    """Every number quoted in prose is a macro defined here from the CSVs."""
    M = {}
    def put(k, v):
        M[k] = v
    agg = {r["model"]: r for r in rd("main_agg.csv")}
    for m, a in agg.items():
        t = re.sub(r"[^A-Za-z]", "", m.title())
        for k in ("P", "R", "F05", "F05_xdom", "OCR", "Q_rhet", "Q_plain", "Q_all", "Q_ist", "ECE_uncal", "ECE_cal"):
            if a.get(k) not in (None, "", "nan"):
                kk = re.sub(r"[^A-Za-z]", "", k.replace("05", "H"))
                d = 2 if k.startswith(("Q", "OCR", "ECE")) else 1
                put(f"{t}{kk}", f"{float(a[k])*100:.{d}f}")
    sig = rd("significance.csv")
    for r in sig:
        t = re.sub(r"[^A-Za-z]", "", r["baseline"].title()) + re.sub(r"[^A-Za-z]", "", r["metric"].replace("05", "H"))
        put(f"P{t}", f"{float(r['p_holm']):.3g}"); put(f"D{t}", f"{float(r['cohen_d']):.2f}")
    for r in rd("crc_agg.csv"):
        v = {"ours": "Gate", "no-gate": "NoGate", "w/o rhetoric head": "NoHead"}[r["variant"]]
        a = {"0.001": "a", "0.002": "b", "0.003": "c", "0.004": "d", "0.005": "e", "0.0075": "f", "0.01": "g", "0.015": "h"}[r["alpha"]]
        for k in ("F05", "Q_rhet", "Q_plain", "Q_all", "risk_test", "risk_quran", "tau", "viol_rate"):
            kk = re.sub(r"[^A-Za-z]", "", k.replace("05", "H").title())
            scale = 1 if k in ("tau", "viol_rate") else 100
            d = 2 if k in ("tau",) else (2 if k != "F05" else 1)
            put(f"Crc{v}{kk}{a}", f"{float(r[k])*scale:.{d}f}")
    sv = json.load(open(os.path.join(RES, "silver_vs_canon.json")))
    put("SilverCanonRecall", f"{sv['silver_recall_on_canon']*100:.1f}"); put("CanonTokens", str(sv["canon_tokens"]))
    rh = [r for r in rd("rhetoric_per_seed.csv") if r["model"] == "ours"]
    for k in ("AUC_silver", "AP_silver", "AUC_istiara", "AUC_canon", "AP_canon", "prev_silver", "prev_canon"):
        vals = [float(r[k]) for r in rh if r[k] not in ("nan", "")]
        if vals:
            put("Rh" + re.sub(r"[^A-Za-z]", "", k.title()), f"{statistics.mean(vals)*(100 if k.startswith(('AP','prev')) else 1):.{1 if k.startswith(('AP','prev')) else 3}f}")
            put("Rh" + re.sub(r"[^A-Za-z]", "", k.title()) + "Sd", f"{statistics.stdev(vals)*(100 if k.startswith(('AP','prev')) else 1):.{1 if k.startswith(('AP','prev')) else 3}f}" if len(vals) > 1 else "0")
    for r in rd("routing.csv"):
        th = r["theta"].replace(".", "p")
        th = {"1p01": "Off", "0p0": "All"}.get(th, th)
        put(f"RouteF{th}".replace("0p", "Z"), f"{float(r['F05_test'])*100:.1f}")
        put(f"RouteMs{th}".replace("0p", "Z"), f"{float(r['ms_per_sent']):.1f}")
        put(f"RouteFrac{th}".replace("0p", "Z"), f"{float(r['routed'])*100:.1f}")
        put(f"RouteQ{th}".replace("0p", "Z"), f"{float(r['Q_rhet'])*100:.2f}")
    meta = json.load(open(os.path.join(RES, "routing2_meta.json")))
    put("MsLight", f"{meta['ms_light']:.1f}"); put("MsHeavy", f"{meta['ms_heavy']:.1f}")
    put("TauLight", f"{meta['tau_light']:.2f}"); put("TauHeavy", f"{meta['tau_heavy']:.2f}")
    ws = rd("worker_scaling.csv")
    put("WOne", f"{float(ws[0]['sent_per_s']):.1f}"); put("WTwo", f"{float(ws[1]['sent_per_s']):.1f}")
    put("WSpeedup", f"{float(ws[1]['sent_per_s'])/float(ws[0]['sent_per_s']):.2f}")
    ddp = [json.loads(l) for l in open(os.path.join(RES, "ddp_timing.jsonl")) if json.loads(l)["steps"] == 60]
    put("DdpOne", f"{ddp[0]['samples_per_s']:.1f}"); put("DdpTwoThr", f"{ddp[1]['samples_per_s']:.1f}"); put("DdpTwo", f"{ddp[2]['samples_per_s']:.1f}")
    put("DdpSpeedup", f"{ddp[2]['samples_per_s']/ddp[0]['samples_per_s']:.2f}"); put("DdpEff", f"{ddp[2]['samples_per_s']/ddp[0]['samples_per_s']/2*100:.0f}")
    sim = json.load(open(os.path.join(RES, "sim_meta.json")))
    put("SimTheta", f"{sim['theta']}"); put("SimRho", f"{sim['rho']*100:.1f}")
    sc = rd("sim_scaling.csv")
    for r in sc:
        c = r["cores"]
        for m in ("light", "heavy", "routed", "seq2seq"):
            put(f"Cap{m.title()}{ {'2':'Two','16':'Sixteen','128':'OneTwoEight','64':'SixtyFour','32':'ThirtyTwo','8':'Eight','4':'Four'}[c]}", f"{float(r[m]):.0f}")
            put(f"Eff{m.title()}{ {'2':'Two','16':'Sixteen','128':'OneTwoEight','64':'SixtyFour','32':'ThirtyTwo','8':'Eight','4':'Four'}[c]}", f"{float(r[m+'_eff'])*100:.0f}")
    for r in rd("beta_sensitivity.csv"):
        b = r["beta"].replace(".", "p")
        put(f"BetaF{b}".replace("0p", "Z").replace("1p", "One"), f"{float(r['F05'])*100:.1f}")
        put(f"BetaQ{b}".replace("0p", "Z").replace("1p", "One"), f"{float(r['Q_rhet'])*100:.2f}")
        put(f"BetaP{b}".replace("0p", "Z").replace("1p", "One"), f"{float(r['Q_plain'])*100:.2f}")
    # derived quantities quoted in the text
    rt = {r["theta"]: r for r in rd("routing.csv")}
    off, allr = rt["1.01"], rt["0"]
    sel = min(rd("routing.csv"), key=lambda x: abs(float(x["theta"]) - sim["theta"]))
    m2 = json.load(open(os.path.join(RES, "routing2_meta.json")))
    put("RouteCostShare", f"{float(sel['ms_per_sent'])/m2['ms_heavy']*100:.0f}")   # vs an ensemble-only service
    put("RouteThetaSel", f"{float(sel['theta']):g}")
    put("RouteGapF", f"{(float(allr['F05_test'])-float(sel['F05_test']))*100:.1f}")
    put("RouteFSel", f"{float(sel['F05_test'])*100:.1f}"); put("RouteMsSel", f"{float(sel['ms_per_sent']):.1f}")
    put("RouteFracSel", f"{float(sel['routed'])*100:.1f}"); put("RouteQSel", f"{float(sel['Q_all_sep'])*100:.2f}")
    put("RouteRiskSep", f"{float(sel['risk_test_sep'])*100:.2f}"); put("RouteRiskComp", f"{float(sel['risk_test_comp'])*100:.2f}")
    put("RouteCompLoss", f"{(float(sel['F05_test_sep'])-float(sel['F05_test_comp']))*100:.1f}"); put("RouteCompGap", f"{(float(allr['F05_test'])-float(sel['F05_test_comp']))*100:.1f}")
    put("RouteFComp", f"{float(sel['F05_test_comp'])*100:.1f}"); put("RouteQComp", f"{float(sel['Q_all_comp'])*100:.2f}")
    put("RouteTauComp", f"{float(sel['tau_comp']):.2f}")
    put("RouteRiskQSep", f"{float(sel['risk_quran_sep'])*100:.2f}"); put("RouteRiskQComp", f"{float(sel['risk_quran_comp'])*100:.2f}")
    put("XdomRiskLight", f"{float(off['risk_xdom_sep'])*100:.2f}"); put("XdomRiskSel", f"{float(sel['risk_xdom_sep'])*100:.2f}")
    put("XdomRiskComp", f"{float(sel['risk_xdom_comp'])*100:.2f}")
    put("RouteFCalSel", f"{float(sel['F05_cal_sep'])*100:.1f}"); put("RouteFCalAll", f"{float(allr['F05_cal_sep'])*100:.1f}")
    put("RouteGainF", f"{(float(sel['F05_test'])-float(off['F05_test']))*100:.1f}")
    put("RouteXdomSel", f"{float(sel['F05_xdom'])*100:.1f}"); put("RouteXdomOff", f"{float(off['F05_xdom'])*100:.1f}"); put("RouteXdomAll", f"{float(allr['F05_xdom'])*100:.1f}")
    cap16 = [r for r in sc if r["cores"] == "16"][0]
    put("CapRatioRoutedHeavy", f"{float(cap16['routed'])/float(cap16['heavy']):.2f}")
    put("CapRatioLightRouted", f"{float(cap16['light'])/float(cap16['routed']):.2f}")
    sv2 = rd("sim_validation.csv")
    put("SimValErr", f"{max(abs(float(r['model_sent_per_s'])-float(r['measured_sent_per_s']))/float(r['measured_sent_per_s']) for r in sv2)*100:.0f}")
    load = rd("sim_load.csv")
    def p99(mode, l): return [float(r["p99_ms"]) for r in load if r["mode"] == mode and abs(float(r["load"]) - l) < 1e-9][0]
    put("LoadRoutedEighty", f"{p99('routed',0.8):.0f}"); put("LoadHeavyEighty", f"{p99('heavy',0.8):.0f}"); put("LoadLightEighty", f"{p99('light',0.8):.1f}")
    put("LoadRoutedNinetyFive", f"{p99('routed',0.95):.0f}"); put("LoadHeavyNinetyFive", f"{p99('heavy',0.95):.0f}")
    bt = rd("batch_throughput.csv")
    put("BatchOne", f"{float(bt[0]['sent_per_s']):.0f}"); put("BatchMax", f"{max(float(b['sent_per_s']) for b in bt):.0f}")
    qe = rd("quran_edit_types.csv")
    for r in qe:
        key = {"orthographic/char": "Orth", "split": "Split", "delete": "Del"}.get(r["type"])
        if key:
            put(f"Qet{r['mode'].title()}{key}", f"{float(r['share'])*100:.0f}")
    put("QetArgN", str(sum(int(r["count"]) for r in qe if r["mode"] == "argmax"))); put("QetCrcN", str(sum(int(r["count"]) for r in qe if r["mode"] == "crc")))
    for r in rd("per_family.csv"):
        put(f"Fam{re.sub('[^A-Za-z]','',r['model'].title())}{r['family'].title()}", f"{float(r['recall'])*100:.1f}")
    for r in rd("robustness.csv"):
        put(f"Rob{re.sub('[^A-Za-z]','',r['model'].title())}{r['rate']}", f"{float(r['F05'])*100:.1f}")
    gs = collections.defaultdict(list)
    for r in rd("gate_stats.csv"):
        gs[r["cls"]].append(float(r["mean"]))
    for c, v in gs.items():
        put("Gate" + re.sub("[^A-Za-z]", "", c.title()), f"{statistics.mean(v):.3f}")
    sup = rd("suppression.csv")
    tot = sum(int(r["n"]) for r in sup)
    put("SuppRate", f"{sum(int(r['n'])*float(r['suppressed_rate']) for r in sup)/tot*100:.2f}")
    agg_ = {r["model"]: r for r in rd("main_agg.csv")}
    put("SeedsMain", agg_["ours"]["n"]); put("SeedsSeq", agg_.get("seq2seq", {"n": "0"})["n"])
    put("SeedsBase", min(int(agg_[m]["n"]) for m in agg_ if m not in ("ours", "seq2seq")))
    ab = rd("ablation.csv"); put("SeedsAbl", ab[0]["n"])
    for r in ab:
        key = re.sub("[^A-Za-z]", "", r["model"].replace("abl_", "").title())
        for k in ("F05", "F05_xdom", "ECE", "F05_crc", "Q_all_crc"):
            if r.get(k):
                put(f"Abl{key}{re.sub('[^A-Za-z]','',k.title())}", f"{float(r[k])*100:.{2 if k in ('ECE','Q_all_crc') else 1}f}")
    # ---- analyses added in revision
    fam = json.load(open(os.path.join(ROOT, "data", "processed", "gec_stats.json")))["test"]["families"]
    ft = sum(fam.values())
    for k, v in fam.items():
        put("Mix" + k.title(), f"{v/ft*100:.0f}")
    ni = {r["baseline"]: r for r in rd("noninferiority.csv")} if os.path.exists(os.path.join(RES, "noninferiority.csv")) else {}
    if "pt_tagger" in ni:
        put("NiDiff", f"{float(ni['pt_tagger']['diff'])*100:.1f}"); put("NiLower", f"{float(ni['pt_tagger']['lower95'])*100:.1f}")
        put("NiP", f"{float(ni['pt_tagger']['p_noninf']):.1g}" if float(ni['pt_tagger']['p_noninf']) >= 1e-3 else "<0.001")
    if os.path.exists(os.path.join(RES, "rhetoric_auc_ci.csv")):
        for r in rd("rhetoric_auc_ci.csv"):
            k = r["set"].title()
            put(f"AucCi{k}", f"{float(r['auc']):.3f}"); put(f"AucCi{k}Lo", f"{float(r['lo']):.3f}"); put(f"AucCi{k}Hi", f"{float(r['hi']):.3f}")
    if os.path.exists(os.path.join(RES, "dedup.json")):
        dd = json.load(open(os.path.join(RES, "dedup.json"))); put("DupTest", dd["test"]); put("DupCal", dd["cal"])
    if os.path.exists(os.path.join(RES, "crc_resplit.csv")):
        rs = [r for r in rd("crc_resplit.csv")]
        for var, tag in (("ours", "Gate"), ("no-gate", "NoGate")):
            v = [r for r in rs if r["variant"] == var]
            put(f"Rs{tag}MaxRatio", f"{max(float(r['mean_risk'])/float(r['alpha']) for r in v):.2f}")
            put(f"Rs{tag}ViolMean", f"{statistics.mean(float(r['viol_freq']) for r in v)*100:.0f}")
            put(f"Rs{tag}ViolMax", f"{max(float(r['viol_freq']) for r in v)*100:.0f}")
            put(f"Rs{tag}QNineFiveMax", f"{max(float(r['q95_risk'])/float(r['alpha']) for r in v):.2f}")
    if os.path.exists(os.path.join(RES, "quran_leak.csv")):
        lk = rd("quran_leak.csv")
        def lq(mode, sub): return statistics.mean(float(r["Q_all"]) for r in lk if r["mode"] == mode and r["subset"] == sub) * 100
        seg = {r["subset"]: (int(r["segments"]), int(r["tokens"])) for r in lk if r["seed"] == "1" and r["mode"] == "argmax"}
        put("LeakSegTrain", seg["seen_train"][0]); put("LeakSegAny", seg["seen_any"][0]); put("LeakSegAll", seg["all"][0])
        put("LeakTokUnseen", seg["unseen_any"][1])
        for mode in ("argmax", "crc"):
            for sub, nm in (("seen_any", "Seen"), ("unseen_any", "Unseen")):
                put(f"Leak{mode.title()}{nm}", f"{lq(mode, sub):.2f}")
    if os.path.exists(os.path.join(RES, "load_validation.csv")):
        lv = rd("load_validation.csv")
        errs50 = [abs(float(r["sim_p50_ms"]) - float(r["meas_p50_ms"])) / float(r["meas_p50_ms"]) for r in lv]
        errs99 = [abs(float(r["sim_p99_ms"]) - float(r["meas_p99_ms"])) / float(r["meas_p99_ms"]) for r in lv]
        put("LvErrFifty", f"{max(errs50)*100:.0f}"); put("LvErrNinetyNine", f"{max(errs99)*100:.0f}")
        put("LvMedErrNinetyNine", f"{statistics.median(errs99)*100:.0f}")
        for md in ("light", "heavy", "routed"):
            sub = [r for r in lv if r["mode"] == md]
            if sub:
                put(f"LvErr{md.title()}", f"{max(abs(float(r['sim_p99_ms']) - float(r['meas_p99_ms'])) / float(r['meas_p99_ms']) for r in sub)*100:.0f}")
                ok = [float(r["lam"]) for r in sub if float(r["meas_p99_ms"]) <= 200]
                put(f"LvMaxOk{md.title()}", f"{max(ok):.0f}" if ok else "0")
                put(f"LvMaxLam{md.title()}", f"{max(float(r['lam']) for r in sub):.0f}")
    put("DdpVsTwoThr", f"{ddp[2]['samples_per_s']/ddp[1]['samples_per_s']:.2f}")
    put("TrainMinPt", f"{float(agg_['pt_tagger']['train_time'])/60:.0f}")
    put("ParamsOurs", f"{int(agg_['ours']['params'])/1e6:.1f}"); put("TrainMinOurs", f"{float(agg_['ours']['train_time'])/60:.0f}")
    put("OverEditRatio", f"{float(agg_['seq2seq']['Q_all'])/float(agg_['ours']['Q_all']):.0f}")
    numbers_extra(M)
    numbers_boot(M)
    numbers_design(M)
    numbers_monitor(M)
    numbers_real(M)
    with open(os.path.join(ROOT, "numbers.tex"), "w") as f:
        for k, v in sorted(M.items()):
            k2 = re.sub(r"\d", lambda d: "ZOTHFIXSEN"[int(d.group())], k)
            f.write(f"\\newcommand{{\\{k2}}}{{{v}}}\n")
    return M


if __name__ == "__main__":
    tab_lit()
    try:
        M = numbers()
        tab_sota(M); tab_ablation(); tab_cost_all(); tab_domains(); tab_lit2(); tab_hparams(); tab_boot(); tab_design(); fig_recal(); tab_real()
    except FileNotFoundError as e:
        print("missing", e)
