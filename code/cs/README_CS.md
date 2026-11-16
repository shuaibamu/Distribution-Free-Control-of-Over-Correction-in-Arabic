# Additional experiments for the revision

All commands run from the repository root. Every stage skips runs whose output already exists, so a
stage can be restarted after an interruption. Logs go to `results/logs/`.

## 0. Setup

```
pip install -r code/requirements.txt -r code/cs/requirements_cs.txt
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`data/processed/` and the pretrained encoder `checkpoints/mlm.pt` are included, so no data
preparation is needed. The GPU stages download public models from the Hugging Face Hub.

## 1. Stages

| Stage | Command | Hardware | Approx. time | Main output |
|---|---|---|---|---|
| Remaining seeds | `JOBS=32 bash code/cs/run_all.sh seeds` | CPU, 32+ cores | 2–3 h | `results/preds/*_s4.pt`, `*_s5.pt` |
| Hyperparameter search | `JOBS=64 bash code/cs/run_all.sh hp` | CPU, 64 cores | 4–6 h | `results/hp_search.csv`, `*_hpbest_s*.pt` |
| Leak-free retraining | `JOBS=16 bash code/cs/run_all.sh leakfree` | CPU, 16 cores | 3–4 h | `results/leakfree_stats.json`, `*_lf_s*.pt` |
| Pretrained Arabic models | `bash code/cs/run_all.sh hf` | 1 GPU, 16 GB | 6–10 h | `hf_*_s*.pt`, `results/gen/arabart_*.json`, `arat5v2_*.json` |
| Open LLMs (few-shot) | `bash code/cs/run_all.sh llm` | 1 GPU, 24 GB+ | 3–6 h | `results/gen/qwen25_7b_s1.json`, … |
| Cost on equal footing | `bash code/cs/run_all.sh cost` | idle CPU node | 20 min | `results/cost_all.csv` |
| Evaluation | `bash code/cs/run_all.sh eval` | CPU | 1–2 h | `results/cs_eval.csv`, `results/gen_eval.csv`, `results/multidomain*.csv` |
| Pack results | `bash code/cs/run_all.sh collect` | – | – | `cs_results.zip` |

The `cost` stage is optional: the equal-footing cost table was already measured (`results/cost_all.csv`). Re-running it needs the seed-1 baseline checkpoints from the `seeds` stage or the original package, and a machine with nothing else running.
If a model in the `llm` stage needs a licence acceptance on the Hub, log in with
`huggingface-cli login` first; models that fail to download are skipped and reported in the log.

## 2. Cluster (multi-node serving, measured)

Use 2–4 machines on the same network, with this repository at the same path on each.

1. On every machine, start one worker per core you want to use. Example for a 16-core node used for
   the routed service, with 9 light and 7 heavy workers:
   ```
   for p in $(seq 7001 7009); do nohup python3 code/cs/cluster_bench.py worker --port $p --tier light > /tmp/w$p.log 2>&1 & done
   for p in $(seq 7101 7107); do nohup python3 code/cs/cluster_bench.py worker --port $p --tier heavy > /tmp/w$p.log 2>&1 & done
   ```
2. Write `hosts.txt` on the dispatcher machine, one `host:port:tier` line per worker.
3. Run, for each configuration (light only, heavy only, routed) and each cluster size
   (1, 2 and 4 nodes):
   ```
   python3 code/cs/cluster_bench.py client --hosts hosts.txt --mode routed --tag routed_2n
   python3 code/cs/cluster_bench.py client --hosts hosts_heavy.txt --mode heavy --tag heavy_2n
   python3 code/cs/cluster_bench.py client --hosts hosts_light.txt --mode light --tag light_2n
   ```
   For a fair comparison, give each configuration the same total number of cores.
   Each run prints the capacity at p99 ≤ 200 ms and writes `results/cluster_<tag>.csv`.
4. Note the CPU model, cores per node and network (for example 10 GbE) in `results/cluster_hardware.txt`.

## 3. Optional: human-annotated data

If you hold the QALB-2014 (or ZAEBUC) M2 files:
```
python3 code/cs/m2_eval.py --dev QALB-2014-L1-Dev.m2 --test QALB-2014-L1-Test.m2 --name qalb14
```
Do not upload the licensed files themselves; `results/real_qalb14.csv` is enough.

## 4. What to send back

`cs_results.zip` from the `collect` stage (results tables, logs and compact prediction files),
plus `results/cluster_*.csv`, `results/cluster_hardware.txt` and any `results/real_*.csv`.
