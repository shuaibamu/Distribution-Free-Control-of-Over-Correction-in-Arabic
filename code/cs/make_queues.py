"""Writes the command queues used on the larger machine.
  queue_seeds.txt   : seeds 4-5 for re-implemented baselines, seeds 3-5 for seq2seq, seeds 4-5 for ablations
  queue_hp.txt      : hyperparameter grid (one seed) for RACE and every re-implemented baseline
  queue_leakfree.txt: leak-free pretraining, then RACE and the clean-Qur'an variant, seeds 1-5
"""
import os, itertools
H = os.path.dirname(os.path.abspath(__file__))


def cmd(model, seed, env="", epochs=3, suffix=""):
    tag = f"{model}{suffix}_s{seed}"
    return (f"[ -f results/preds/{tag}.pt ] || env THREADS=1 {env} python3 code/train.py --model {model} --seed {seed} "
            f"--epochs {epochs} > results/logs/{tag}.log 2>&1")


base = ["bilstm", "trans_scratch", "pt_tagger", "stack_hybrid"]
abl = ["abl_nopre", "abl_nornn", "abl_concat", "abl_lstm", "abl_nodet", "abl_norhet", "abl_quranclean"]
with open(os.path.join(H, "queue_seeds.txt"), "w") as f:
    f.write("# seq2seq is the slowest job: start it first\n")
    for s in (3, 4, 5):
        f.write(cmd("seq2seq", s) + "\n")
    for m in base + abl:
        for s in (4, 5):
            f.write(cmd(m, s) + "\n")

grid = list(itertools.product(["3e-4", "6e-4", "1e-3"], ["0.1", "0.2"], [3, 5]))
with open(os.path.join(H, "queue_hp.txt"), "w") as f:
    f.write("# stage 1: identical grid (lr x dropout x epochs) for every tagger, seed 1, selection on dev F0.5\n")
    for m in ["ours"] + base:
        for lr, dr, ep in grid:
            sfx = f"_hp_lr{lr}_d{dr}_e{ep}"
            f.write(cmd(m, 1, f"LR={lr} DROP={dr} TAG_SUFFIX={sfx}", ep, sfx) + "\n")
    f.write("# stage 2 (RACE only): auxiliary-loss weight and rhetoric class weight at lr=6e-4, dropout=0.1, 3 epochs\n")
    for lam in ["0.25", "1.0"]:
        for rhw in ["4", "8", "16"]:
            sfx = f"_hp_lam{lam}_rhw{rhw}"
            f.write(cmd("ours", 1, f"LAMBDA={lam} RHW={rhw} TAG_SUFFIX={sfx}", 3, sfx) + "\n")

with open(os.path.join(H, "queue_leakfree.txt"), "w") as f:
    f.write("# run code/cs/leakfree_data.py first, then this queue in two stages (pretraining must finish first)\n")
    f.write("[ -f checkpoints/mlm_lf.pt ] || env VARIANT=_lf THREADS=4 python3 code/pretrain.py --epochs 2 "
            "--out checkpoints/mlm_lf.pt > results/logs/mlm_lf.log 2>&1\n")
with open(os.path.join(H, "queue_leakfree_stage2.txt"), "w") as f:
    for m in ("ours", "pt_tagger", "abl_quranclean"):
        for s in range(1, 6):
            f.write(cmd(m, s, "VARIANT=_lf", 3, "_lf") + "\n")
print("queues written")
