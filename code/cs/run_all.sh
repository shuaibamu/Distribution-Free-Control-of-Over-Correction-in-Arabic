#!/bin/bash
# Additional experiments for the revision. Run from the repository root:
#     bash code/cs/run_all.sh <stage>
# Stages can run in any order unless noted; each writes into results/ and skips finished runs.
#   seeds      remaining seeds (baselines 4-5, seq2seq 3-5, ablations 4-5)           CPU, many cores
#   hp         identical hyperparameter grid for RACE and all baselines, then
#              5-seed retraining of each model's selected setting                     CPU, many cores
#   leakfree   remove Qur'an quotations from training/pretraining text, re-pretrain,
#              retrain RACE / pretrained tagger / clean-Qur'an variant (5 seeds)       CPU, many cores
#   hf         pretrained Arabic encoders as taggers + pretrained seq2seq models       1 GPU (>= 16 GB)
#   llm        few-shot open LLMs with edit-level risk control                         1 GPU (>= 24 GB)
#   cost       equal-footing latency / throughput of every model (idle machine)        CPU
#   eval       evaluation of everything above -> results/cs_eval.csv, results/gen_eval.csv
#   collect    pack results for return -> cs_results.zip
# Multi-node serving is run by hand; see code/cs/README_CS.md, section "Cluster".
set -e
cd "$(dirname "$0")/../.."
mkdir -p results/logs results/preds results/gen
J=${JOBS:-$(nproc)}
case "$1" in
  seeds)
    python3 code/cs/make_queues.py
    JOBS=$J bash code/cs/run_cmds.sh code/cs/queue_seeds.txt ;;
  hp)
    python3 code/cs/make_queues.py
    JOBS=$J bash code/cs/run_cmds.sh code/cs/queue_hp.txt
    python3 code/cs/hp_select.py
    JOBS=$J bash code/cs/run_cmds.sh code/cs/queue_hp_final.txt ;;
  leakfree)
    python3 code/cs/make_queues.py
    python3 code/cs/leakfree_data.py
    JOBS=1 bash code/cs/run_cmds.sh code/cs/queue_leakfree.txt
    JOBS=$J bash code/cs/run_cmds.sh code/cs/queue_leakfree_stage2.txt ;;
  hf)
    for s in 1 2 3 4 5; do
      for m in "CAMeL-Lab/bert-base-arabic-camelbert-ca hf_camelbert_ca" "aubmindlab/bert-base-arabertv02 hf_arabertv02"; do
        set -- $m
        [ -f results/preds/${2}_s${s}.pt ] || python3 code/cs/hf_tagger.py --hf $1 --tag $2 --seed $s > results/logs/${2}_s${s}.log 2>&1
      done
    done
    for s in 1 2 3; do
      [ -f results/gen/arabart_s${s}.json ] || python3 code/cs/hf_seq2seq.py --hf moussaKam/AraBART --tag arabart --seed $s > results/logs/arabart_s${s}.log 2>&1
      [ -f results/gen/arat5v2_s${s}.json ] || python3 code/cs/hf_seq2seq.py --hf UBC-NLP/AraT5v2-base-1024 --tag arat5v2 --seed $s --lr 3e-4 > results/logs/arat5v2_s${s}.log 2>&1
    done ;;
  llm)
    for m in "Qwen/Qwen2.5-7B-Instruct qwen25_7b" "CohereForAI/aya-expanse-8b aya8b" "ALLaM-AI/ALLaM-7B-Instruct-preview allam7b"; do
      set -- $m
      [ -f results/gen/${2}_s1.json ] || python3 code/cs/llm_correct.py --hf $1 --tag $2 --n 1000 > results/logs/${2}.log 2>&1 || echo "skipped $1 (download or licence)"
    done ;;
  cost)
    python3 code/cost_all.py > results/logs/cost_all.log 2>&1 ;;
  eval)
    TAGS=$(ls results/preds | sed -E 's/_s[0-9]+\.pt$//' | sort -u | grep -E '^(ours|pt_tagger|bilstm|trans_scratch|stack_hybrid|abl_|hf_)' | tr '\n' ' ')
    python3 code/cs/eval_tags.py --tags $TAGS --out results/cs_eval.csv > results/logs/cs_eval.log 2>&1
    python3 code/cs/gen_eval.py > results/logs/gen_eval.log 2>&1 || true ;;
  collect)
    zip -qr cs_results.zip results -x "results/preds/*" && \
    python3 code/cs/slim_preds.py cs_results_preds && zip -qr cs_results.zip cs_results_preds && rm -rf cs_results_preds
    ls -la cs_results.zip ;;
  *) echo "usage: bash code/cs/run_all.sh {seeds|hp|leakfree|hf|llm|cost|eval|collect}"; exit 1 ;;
esac
