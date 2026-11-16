"""Writes figure data (figures/data/*.csv) from results/*.csv and one pgfplots
file per results figure (figures/figN.tex). Colours follow the figures4papers
palette: blue = ours, reds = baselines, greens/teal/violet = variants."""
import os, csv, json, math, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures"); FD = os.path.join(FIG, "data")
os.makedirs(FD, exist_ok=True)


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


def wr(name, header, rows):
    with open(os.path.join(FD, name), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def tex(name, body):
    open(os.path.join(FIG, name + ".tex"), "w").write(body)


AX = r"""axis lines*=left, tick align=outside, tick style={black}, every axis plot/.append style={line width=1pt},
legend style={draw=none, fill=none, font=\scriptsize}, label style={font=\footnotesize}, tick label style={font=\scriptsize},
width=\linewidth, height=0.62\linewidth, ymajorgrids, grid style={gray!20}"""
SHORT = {"ours": "RACE", "bilstm": "BiLSTM", "trans_scratch": "Trans.", "pt_tagger": "PT-tag.",
         "stack_hybrid": "PT-BiLSTM", "seq2seq": "S2S"}
ORDER = ["bilstm", "trans_scratch", "pt_tagger", "stack_hybrid", "seq2seq", "ours"]


def f_crc():
    agg = rd("crc_agg.csv")
    rows = collections.defaultdict(dict)
    for r in agg:
        rows[float(r["alpha"])][r["variant"]] = r
    al = sorted(rows)
    wr("crc.csv", ["alpha", "f_ours", "f_nogate", "f_norhet", "q_ours", "q_nogate", "q_norhet"],
       [[a * 100, float(rows[a]["ours"]["F05"]) * 100, float(rows[a]["no-gate"]["F05"]) * 100,
         float(rows[a].get("w/o rhetoric head", rows[a]["no-gate"])["F05"]) * 100, float(rows[a]["ours"]["Q_all"]) * 100,
         float(rows[a]["no-gate"]["Q_all"]) * 100, float(rows[a].get("w/o rhetoric head", rows[a]["no-gate"])["Q_all"]) * 100] for a in al])
    tex("fig_crc", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.15cm}, %s, width=0.47\linewidth, height=0.3\linewidth,
 xlabel={Risk level $\alpha$ (\%%)}]
\nextgroupplot[ylabel={F$_{0.5}$ (\%%) $\uparrow$}, legend to name=crcleg, legend columns=3]
\addplot[c1, mark=*, mark size=1.4pt] table[x=alpha,y=f_ours,col sep=comma]{figures/data/crc.csv}; \addlegendentry{RACE (gated)}
\addplot[c3, dashed, mark=square*, mark size=1.3pt] table[x=alpha,y=f_nogate,col sep=comma]{figures/data/crc.csv}; \addlegendentry{RACE, $\beta=0$}
\addplot[c4, dotted, mark=triangle*, mark size=1.6pt] table[x=alpha,y=f_norhet,col sep=comma]{figures/data/crc.csv}; \addlegendentry{w/o rhetoric head}
\nextgroupplot[ylabel={Qur'an OCR, all tokens (\%%) $\downarrow$}]
\addplot[c1, mark=*, mark size=1.4pt] table[x=alpha,y=q_ours,col sep=comma]{figures/data/crc.csv};
\addplot[c3, dashed, mark=square*, mark size=1.3pt] table[x=alpha,y=q_nogate,col sep=comma]{figures/data/crc.csv};
\addplot[c4, dotted, mark=triangle*, mark size=1.6pt] table[x=alpha,y=q_norhet,col sep=comma]{figures/data/crc.csv};
\end{groupplot}
\node at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.05)$) {\pgfplotslegendfromname{crcleg}};
\end{tikzpicture}""" % AX)


def f_validity():
    agg = [r for r in rd("crc_agg.csv") if r["variant"] == "ours"]
    wr("valid.csv", ["alpha", "test", "test_sd", "quran", "quran_sd"],
       [[float(r["alpha"]) * 100, float(r["risk_test"]) * 100, float(r["risk_test_sd"]) * 100,
         float(r["risk_quran"]) * 100, float(r["risk_quran_sd"]) * 100] for r in agg])
    mx = max(float(r["alpha"]) for r in agg) * 100
    tex("fig_valid", r"""\begin{tikzpicture}
\begin{axis}[%s, xlabel={Target risk $\alpha$ (\%%)}, ylabel={Realised false-edit rate (\%%)}, legend pos=north west, xmin=0, ymin=0]
\addplot[gray, densely dashed, domain=0:%g, samples=2] {x}; \addlegendentry{$y=\alpha$}
\addplot[c1, mark=*, mark size=1.4pt, error bars/.cd, y dir=both, y explicit] table[x=alpha,y=test,y error=test_sd,col sep=comma]{figures/data/valid.csv}; \addlegendentry{In-domain test}
\addplot[c3, mark=diamond*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] table[x=alpha,y=quran,y error=quran_sd,col sep=comma]{figures/data/valid.csv}; \addlegendentry{Qur'an (shifted)}
\end{axis}
\end{tikzpicture}""" % (AX, mx))


def f_reliab():
    r = rd("reliability.csv")
    for cal in ("0", "1"):
        wr(f"rel{cal}.csv", ["conf", "acc"], [[x["conf"], x["acc"]] for x in r if x["calibrated"] == cal])
    tex("fig_reliab", r"""\begin{tikzpicture}
\begin{axis}[%s, xlabel={Confidence of proposed edit}, ylabel={Accuracy}, xmin=0, xmax=1, ymin=0, ymax=1, legend pos=north west]
\addplot[gray, densely dashed, domain=0:1, samples=2] {x}; \addlegendentry{Perfect calibration}
\addplot[c3, mark=square*, mark size=1.5pt] table[x=conf,y=acc,col sep=comma]{figures/data/rel0.csv}; \addlegendentry{Uncalibrated}
\addplot[c1, mark=*, mark size=1.5pt] table[x=conf,y=acc,col sep=comma]{figures/data/rel1.csv}; \addlegendentry{Temperature-scaled}
\end{axis}
\end{tikzpicture}""" % AX)


def bar_models(models, cols, fname, ylabel, legend, scale=100.0, src="main_agg.csv", ymax=None, legpos="north east"):
    agg = {r["model"]: r for r in rd(src)}
    rows = []
    for i, m in enumerate(models):
        if m in agg:
            rows.append([i, SHORT[m]] + [float(agg[m][c]) * scale for c in cols] + [float(agg[m][c + "_sd"]) * scale for c in cols])
    wr(fname + ".csv", ["x", "name"] + cols + [c + "_sd" for c in cols], rows)
    xt = ",".join(str(r[0]) for r in rows); xl = ",".join(r[1] for r in rows)
    bars = []
    styles = ["fill=c1!85, draw=black", "fill=c3!70, draw=black, postaction={pattern=north east lines}", "fill=c2, draw=black"]
    for k, c in enumerate(cols):
        bars.append(r"\addplot[%s, error bars/.cd, y dir=both, y explicit] table[x=x,y=%s,y error=%s_sd,col sep=comma]{figures/data/%s.csv}; \addlegendentry{%s}"
                    % (styles[k], c, c, fname, legend[k]))
    tex(fname, r"""\begin{tikzpicture}
\begin{axis}[%s, ybar, bar width=5pt, xtick={%s}, xticklabels={%s}, x tick label style={rotate=30, anchor=east, font=\scriptsize},
 ylabel={%s}, %s legend style={at={(0.5,1.02)}, anchor=south, legend columns=2}, enlarge x limits=0.1]
%s
\end{axis}
\end{tikzpicture}""" % (AX, xt, xl, ylabel, ("ymode=log, ymin=0.01, log origin=infty," if fname == "fig_qocr" else "ymin=0,") + (("ymax=%g," % ymax) if ymax else ""), "\n".join(bars)))


def f_beta():
    r = rd("beta_sensitivity.csv")
    wr("beta.csv", ["beta", "F05", "Q_rhet", "Q_plain"], [[x["beta"], float(x["F05"]) * 100, float(x["Q_rhet"]) * 100, float(x["Q_plain"]) * 100] for x in r])
    tex("fig_beta", r"""\begin{tikzpicture}
\begin{axis}[%s, width=0.84\linewidth, xlabel={Rhetoric margin $\beta$}, ylabel={F$_{0.5}$ (\%%) $\uparrow$}, axis y line*=left, legend style={at={(0.5,1.03)}, anchor=south, legend columns=3}]
\addplot[c1, mark=*, mark size=1.4pt] table[x=beta,y=F05,col sep=comma]{figures/data/beta.csv}; \label{pb1}\addlegendentry{F$_{0.5}$ (test)}
\addlegendimage{c3, dashed, mark=square*}\addlegendentry{Rhet. OCR}
\addlegendimage{c2, dotted, mark=triangle*}\addlegendentry{Plain OCR}
\end{axis}
\begin{axis}[%s, width=0.84\linewidth, axis y line*=right, axis x line=none, ylabel={Qur'an OCR (\%%) $\downarrow$}, ymajorgrids=false]
\addplot[c3, dashed, mark=square*, mark size=1.3pt] table[x=beta,y=Q_rhet,col sep=comma]{figures/data/beta.csv};
\addplot[c2, dotted, mark=triangle*, mark size=1.6pt] table[x=beta,y=Q_plain,col sep=comma]{figures/data/beta.csv};
\end{axis}
\end{tikzpicture}""" % (AX, AX))


def f_rhetpr():
    r = rd("rhetoric_pr.csv")
    for s in ("silver", "canonical"):
        wr(f"pr_{s}.csv", ["recall", "precision"], [[x["recall"], x["precision"]] for x in r if x["set"] == s])
    tex("fig_rhetpr", r"""\begin{tikzpicture}
\begin{axis}[%s, xlabel={Recall}, ylabel={Precision}, xmin=0, xmax=1, ymin=0, ymax=1, legend pos=north east]
\addplot[c1, mark=none, line width=1.2pt] table[x=recall,y=precision,col sep=comma]{figures/data/pr_silver.csv}; \addlegendentry{Silver labels (test)}
\addplot[c3, dashed, mark=none, line width=1.2pt] table[x=recall,y=precision,col sep=comma]{figures/data/pr_canonical.csv}; \addlegendentry{Canonical exemplars}
\end{axis}
\end{tikzpicture}""" % AX)


def f_gate():
    r = rd("gate_stats.csv")
    by = collections.defaultdict(list)
    for x in r:
        by[x["cls"]].append(x)
    order = ["clean prose", "error:ORTH", "error:TYPO", "error:SPACE", "error:MISS", "error:RED", "error:LEX", "Qur'an plain", "Qur'an rhetorical"]
    lab = {"clean prose": "clean", "error:ORTH": "ORTH", "error:TYPO": "TYPO", "error:SPACE": "SPACE", "error:MISS": "MISS",
           "error:RED": "RED", "error:LEX": "LEX", "Qur'an plain": "Q-plain", "Qur'an rhetorical": "Q-rhet"}
    plots = []; names = []
    for i, c in enumerate([o for o in order if o in by]):
        xs = [x for x in by[c] if x["seed"] == "1"] or by[c]
        x = xs[0]
        col = "c1" if c.startswith("error") else ("c3" if "rhet" in c else ("c2" if "Qur" in c else "c5"))
        plots.append(r"\addplot+[boxplot prepared={median=%s, upper quartile=%s, lower quartile=%s, upper whisker=%s, lower whisker=%s}, fill=%s!40, draw=%s] coordinates {};"
                     % (x["med"], x["q3"], x["q1"], x["hi"], x["lo"], col, col))
        names.append(lab[c])
    tex("fig_gate", r"""\begin{tikzpicture}
\begin{axis}[%s, boxplot/draw direction=y, ylabel={Mean gate value $\bar g_i$ (transformer share)}, xtick={%s}, xticklabels={%s},
 x tick label style={rotate=35, anchor=east, font=\scriptsize}, boxplot/box extend=0.5]
%s
\end{axis}
\end{tikzpicture}""" % (AX, ",".join(str(i + 1) for i in range(len(names))), ",".join(names), "\n".join(plots)))


def f_family():
    r = rd("per_family.csv")
    fams = ["ORTH", "TYPO", "SPACE", "MISS", "RED", "LEX"]
    models = [m for m in ["bilstm", "trans_scratch", "pt_tagger", "stack_hybrid", "ours"] if any(x["model"] == m for x in r)]
    rows = []
    for j, m in enumerate(models):
        for i, f in enumerate(fams):
            v = [x for x in r if x["model"] == m and x["family"] == f][0]
            rows.append([i, j, float(v["recall"]) * 100])
    wr("family.csv", ["x", "y", "v"], rows)
    nodes = "\n".join(r"\node[font=\tiny, text=%s] at (axis cs:%d,%d) {%.0f};" % ("white" if v > 55 else "black", x, y, v) for x, y, v in rows)
    tex("fig_family", r"""\begin{tikzpicture}
\begin{axis}[width=0.86\linewidth, height=0.55\linewidth, enlargelimits=false, colormap={bl}{color=(white) color=(c1)}, point meta min=0, point meta max=100,
 colorbar, colorbar style={width=0.18cm, ytick={0,50,100}, yticklabel style={font=\tiny}},
 xtick={0,...,5}, xticklabels={%s}, ytick={0,...,%d}, yticklabels={%s}, tick label style={font=\scriptsize}, xlabel={Error family}, label style={font=\footnotesize},
 xmin=-0.5, xmax=5.5, ymin=-0.5, ymax=%g]
\addplot[matrix plot*, mesh/cols=6, point meta=explicit] table[x=x, y=y, meta=v, col sep=comma]{figures/data/family.csv};
%s
\end{axis}
\end{tikzpicture}""" % (",".join(fams), len(models) - 1, ",".join(SHORT[m] for m in models), len(models) - 0.5, nodes))


def f_routing():
    r = rd("routing.csv")
    rows = sorted([[float(x["ms_per_sent"]), float(x["F05_test"]) * 100, float(x["Q_rhet"]) * 100, float(x["routed"]) * 100, x["theta"]] for x in r])
    wr("routing.csv", ["ms", "F05", "Q", "routed", "theta"], rows)
    meta = json.load(open(os.path.join(RES, "routing2_meta.json")))
    selr = [x for x in rows if abs(float(x[4]) - meta["theta_cal"]) < 1e-9][0]
    tex("fig_routing", r"""\begin{tikzpicture}
\begin{axis}[%s, xlabel={CPU time per sentence (ms) $\downarrow$}, ylabel={F$_{0.5}$ (\%%) $\uparrow$}, legend pos=south east]
\addplot[c1, mark=*, mark size=1.4pt] table[x=ms,y=F05,col sep=comma]{figures/data/routing.csv}; \addlegendentry{Routed cascade ($\theta$ sweep)}
\addplot[only marks, mark=square*, c3, mark size=2.2pt] coordinates {(%g,%g)}; \addlegendentry{Light only}
\addplot[only marks, mark=triangle*, c2, mark size=2.8pt] coordinates {(%g,%g)}; \addlegendentry{Ensemble only}
\addplot[only marks, mark=*, draw=black, fill=gold, mark size=2.4pt] coordinates {(%g,%g)}; \addlegendentry{Selected $\theta$ (dev)}
\end{axis}
\end{tikzpicture}""" % (AX, rows[0][0], rows[0][1], meta["ms_heavy"], rows[-1][1], selr[0], selr[1]))


def f_scaling():
    r = rd("sim_scaling.csv")
    wr("scaling.csv", ["cores", "light", "heavy", "routed", "seq2seq"], [[x["cores"], x["light"], x["heavy"], x["routed"], x["seq2seq"]] for x in r])
    tex("fig_scaling", r"""\begin{tikzpicture}
\begin{axis}[%s, xmode=log, ymode=log, log basis x=2, xlabel={Cluster cores}, ylabel={Capacity at p99$\leq$200 ms (req/s)}, legend pos=south east, xtick={2,4,8,16,32,64,128}, xticklabels={2,4,8,16,32,64,128}]
\addplot[c3, mark=square*, mark size=1.3pt] table[x=cores,y=light,col sep=comma]{figures/data/scaling.csv}; \addlegendentry{Light only}
\addplot[c1, mark=*, mark size=1.4pt] table[x=cores,y=routed,col sep=comma]{figures/data/scaling.csv}; \addlegendentry{Routed}
\addplot[c2, mark=triangle*, mark size=1.6pt] table[x=cores,y=heavy,col sep=comma]{figures/data/scaling.csv}; \addlegendentry{Heavy only}
\addplot[c4, dashed, mark=diamond*, mark size=1.6pt] table[x=cores,y=seq2seq,col sep=comma]{figures/data/scaling.csv}; \addlegendentry{Seq2seq}
\end{axis}
\end{tikzpicture}""" % AX)


def f_load():
    r = rd("sim_load.csv")
    for m in ("light", "heavy", "seq2seq", "routed"):
        wr(f"load_{m}.csv", ["load", "p99"], [[float(x["load"]) * 100, x["p99_ms"]] for x in r if x["mode"] == m])
    tex("fig_load", r"""\begin{tikzpicture}
\begin{axis}[%s, ymode=log, xlabel={Utilisation (\%%), 16 simulated cores}, ylabel={p99 latency (ms) $\downarrow$}, legend style={at={(0.5,1.02)}, anchor=south, legend columns=4}]
\addplot[c3, mark=square*, mark size=1.3pt] table[x=load,y=p99,col sep=comma]{figures/data/load_light.csv}; \addlegendentry{Light only}
\addplot[c1, mark=*, mark size=1.4pt] table[x=load,y=p99,col sep=comma]{figures/data/load_routed.csv}; \addlegendentry{Routed}
\addplot[c2, mark=triangle*, mark size=1.6pt] table[x=load,y=p99,col sep=comma]{figures/data/load_heavy.csv}; \addlegendentry{Heavy only}
\addplot[c4, dashed, mark=diamond*, mark size=1.6pt] table[x=load,y=p99,col sep=comma]{figures/data/load_seq2seq.csv}; \addlegendentry{Seq2seq}
\addplot[gray, densely dotted, domain=10:95, samples=2] {200};
\end{axis}
\end{tikzpicture}""" % AX)


def f_cdf():
    r = rd("latency_samples.csv")
    for t in ("light", "heavy", "seq2seq"):
        v = sorted(float(x["ms"]) for x in r if x["tier"] == t)
        n = len(v)
        pts = [[v[int(q * (n - 1))], q] for q in [i / 50 for i in range(51)]]
        wr(f"cdf_{t}.csv", ["ms", "p"], pts)
    tex("fig_cdf", r"""\begin{tikzpicture}
\begin{axis}[%s, xmode=log, xlabel={Per-request latency, batch 1 (ms)}, ylabel={Empirical CDF}, ymin=0, ymax=1, legend pos=south east]
\addplot[c1, line width=1.2pt] table[x=ms,y=p,col sep=comma]{figures/data/cdf_light.csv}; \addlegendentry{Light (RACE)}
\addplot[c2, line width=1.2pt, dashed] table[x=ms,y=p,col sep=comma]{figures/data/cdf_heavy.csv}; \addlegendentry{Heavy (4-ensemble)}
\addplot[c4, line width=1.2pt, dotted] table[x=ms,y=p,col sep=comma]{figures/data/cdf_seq2seq.csv}; \addlegendentry{Seq2seq+copy}
\end{axis}
\end{tikzpicture}""" % AX)


def f_ablation():
    r = rd("ablation.csv")
    keep = [x for x in r if x.get("F05_crc")]
    rows = []
    for i, x in enumerate(keep):
        rows.append([i, x["name"].replace("'", "").replace(",", ""), float(x["F05_crc"]) * 100, float(x["F05_crc_sd"]) * 100,
                     float(x["Q_all_crc"]) * 100, float(x["Q_all_crc_sd"]) * 100])
    wr("ablation.csv", ["y", "name", "F", "Fsd", "Q", "Qsd"], rows)
    yl = ",".join(("{%s}" % x["name"].replace("w/o risk control (tau=0.5)", r"fixed $\tau$=0.5")) for x in keep)
    tex("fig_ablation", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=0.5cm, yticklabels at=edge left}, axis lines*=left, xmajorgrids, grid style={gray!20},
 tick label style={font=\scriptsize}, label style={font=\footnotesize}, width=0.4\linewidth, height=0.95\linewidth, ytick={0,...,%d}, yticklabels={%s},
 y dir=reverse, enlarge y limits=0.05]
\nextgroupplot[xbar, bar width=4pt, xmin=50, xmax=60, xtick={50,55,60}, xlabel={F$_{0.5}$ (\%%) $\uparrow$}]
\addplot[fill=c1!85, draw=black, error bars/.cd, x dir=both, x explicit] table[x=F, y=y, x error=Fsd, col sep=comma]{figures/data/ablation.csv};
\nextgroupplot[xbar, bar width=4pt, xmin=0, xlabel={Qur'an OCR (\%%) $\downarrow$}]
\addplot[fill=c3!70, draw=black, error bars/.cd, x dir=both, x explicit] table[x=Q, y=y, x error=Qsd, col sep=comma]{figures/data/ablation.csv};
\end{groupplot}
\end{tikzpicture}""" % (len(rows) - 1, yl))


def f_robust():
    r = rd("robustness.csv")
    for m in ORDER:
        wr(f"rob_{m}.csv", ["rate", "F05"], [[x["rate"], float(x["F05"]) * 100] for x in r if x["model"] == m])
    sty = {"bilstm": "c5, mark=o", "trans_scratch": "c4, dotted, mark=square", "pt_tagger": "c3, dashed, mark=square*",
           "stack_hybrid": "c2, mark=triangle*", "seq2seq": "c6, dashdotted, mark=diamond*", "ours": "c1, mark=*, line width=1.3pt"}
    lines = "\n".join(r"\addplot[%s, mark size=1.4pt] table[x=rate,y=F05,col sep=comma]{figures/data/rob_%s.csv}; \addlegendentry{%s}" % (sty[m], m, SHORT[m]) for m in ORDER
                      if any(x["model"] == m for x in r))
    tex("fig_robust", r"""\begin{tikzpicture}
\begin{axis}[%s, xlabel={Word corruption probability (\%%)}, ylabel={F$_{0.5}$ (\%%) $\uparrow$}, legend columns=3, legend style={at={(0.5,1.02)}, anchor=south}]
%s
\end{axis}
\end{tikzpicture}""" % (AX, lines))


def f_ddp():
    rows = [json.loads(l) for l in open(os.path.join(RES, "ddp_timing.jsonl")) if json.loads(l)["steps"] == 60]
    # order: 1 proc/1 thread, 1 proc/2 threads, 2 procs (DDP)
    lab = ["1 proc, 1 thr", "1 proc, 2 thr", "DDP, 2 proc"]
    wr("ddp.csv", ["x", "name", "sps"], [[i, lab[i], round(r["samples_per_s"], 2)] for i, r in enumerate(rows[:3])])
    return rows


def f_shift():
    """Realised false-edit rate on held-out works with the in-domain threshold and after recalibration."""
    import statistics as st
    c = rd("multidomain_crc.csv")
    works = [("Jurjani", "Jurj."), ("Jahiz", "J\\={a}\\d{h}i\\d{z}"), ("Mubarrad", "Mubar."), ("IbnAbdRabbih", "I.\\,A.\\,Rab."), ("IbnKhaldun", "I.\\,Khal.")]
    rows = []
    for i, (k, lab) in enumerate(works):
        s = [x for x in c if x["domain"] == k and float(x["beta"]) == 0.5 and float(x["alpha"]) == 0.005]
        ri = [float(x["risk_in"]) * 100 for x in s]
        rr = [float(x["risk_rec"]) * 100 for x in s if x.get("risk_rec")]
        rows.append([i, lab, st.mean(ri), st.stdev(ri), st.mean(rr) if rr else "nan", st.stdev(rr) if len(rr) > 1 else 0])
    wr("fig_shift.csv", ["x", "name", "rin", "rin_sd", "rrec", "rrec_sd"], rows)
    xt = ",".join(str(r[0]) for r in rows); xl = ",".join("{%s}" % r[1] for r in rows)
    tex("fig_shift", r"""\begin{tikzpicture}
\begin{axis}[%s, ybar, bar width=6pt, xtick={%s}, xticklabels={%s}, x tick label style={font=\scriptsize},
 ylabel={Realised false-edit rate (\%%)}, ymin=0, ymax=1.15, legend style={at={(0.5,1.02)}, anchor=south, legend columns=3}, enlarge x limits=0.12]
\addplot[fill=c3!70, draw=black, postaction={pattern=north east lines}, error bars/.cd, y dir=both, y explicit] table[x=x,y=rin,y error=rin_sd,col sep=comma]{figures/data/fig_shift.csv}; \addlegendentry{In-domain $\hat\tau$}
\addplot[fill=c1!85, draw=black, error bars/.cd, y dir=both, y explicit] table[x=x,y=rrec,y error=rrec_sd,col sep=comma]{figures/data/fig_shift.csv}; \addlegendentry{Recalibrated $\hat\tau$}
\addplot[black, dashed, sharp plot, update limits=false] coordinates {(-1,0.5) (5,0.5)}; \addlegendentry{$\alpha$}
\end{axis}
\end{tikzpicture}""" % (AX, xt, xl))


def f_rcbase():
    """Risk-control methods over 200 random calibration/test resplits per seed."""
    r = rd("riskcontrol_baselines.csv")
    al = sorted({float(x["alpha"]) for x in r})
    rows = []
    for a in al:
        g = {x["method"]: x for x in r if float(x["alpha"]) == a}
        rows.append([a * 100] + [float(g[m]["viol_freq"]) * 100 for m in ("CRC", "Naive", "LTT")] + [float(g[m]["tokF05"]) * 100 for m in ("CRC", "Naive", "LTT")])
    wr("fig_rcbase.csv", ["a", "vCRC", "vNaive", "vLTT", "fCRC", "fNaive", "fLTT"], rows)
    st = {"CRC": "c1, mark=*", "Naive": "c3, mark=square*, dashed", "LTT": "c2, mark=triangle*"}
    lab = {"CRC": "CRC (ours)", "Naive": "Naive threshold", "LTT": "Learn-then-Test"}
    def plots(pfx):
        return "\n".join(r"\addplot[%s, mark size=1.5pt] table[x=a,y=%s%s,col sep=comma]{figures/data/fig_rcbase.csv};%s"
                         % (st[m], pfx, m, (r" \addlegendentry{%s}" % lab[m]) if pfx == "v" else "") for m in ("CRC", "Naive", "LTT"))
    tex("fig_rcbase", r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.0cm}, %s, width=0.56\linewidth, height=0.5\linewidth, xmode=log,
 xtick={0.1,0.3,1,2}, xticklabels={0.1,0.3,1,2}, xlabel={Risk level $\alpha$ (\%%)}]
\nextgroupplot[ylabel={Splits with risk $>\alpha$ (\%%)}, ymin=0, legend style={at={(1.1,1.03)}, anchor=south, legend columns=3}]
%s
\nextgroupplot[ylabel={Token F$_{0.5}$ (\%%)}]
%s
\end{groupplot}
\end{tikzpicture}""" % (AX, plots("v"), plots("f")))


if __name__ == "__main__":
    import sys
    fs = [f_crc, f_validity, f_reliab, f_beta, f_rhetpr, f_gate, f_family, f_routing, f_scaling, f_load, f_cdf, f_ablation, f_robust, f_ddp, f_shift, f_rcbase]
    for f in fs:
        try:
            f()
        except Exception as e:
            print("skip", f.__name__, repr(e))
    for args in [(["bilstm", "trans_scratch", "pt_tagger", "stack_hybrid", "ours"], ["ECE_uncal", "ECE_cal"], "fig_ece", "ECE (\\%) $\\downarrow$", ["Uncalibrated", "Temperature-scaled"]),
                 (ORDER, ["Q_plain", "Q_rhet"], "fig_qocr", "Qur'an OCR (\\%) $\\downarrow$", ["Plain tokens", "Rhetorical tokens"])]:
        try:
            bar_models(*args)
        except Exception as e:
            print("skip bar", args[2], repr(e))
