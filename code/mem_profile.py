"""Resident memory of a serving worker (light tier: 1 model; heavy tier: 4 models).

Each configuration runs in a fresh process: RSS is read after imports, after loading
the checkpoints, and as the peak after serving 500 test sentences one at a time.
Writes results/mem_profile.json.
"""
import os, sys, json, resource
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(__file__))


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def job(tags, q):
    import torch
    torch.set_num_threads(1)
    import serve_bench as S
    base = rss_mb()
    models = [S.load_tagger(t) for t in tags]
    loaded = rss_mb()
    its = S.A.items("test")[:500]
    for x in its:
        with torch.no_grad():
            S.tier_probs(models, [x])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    q.put(dict(models=len(tags), rss_runtime_mb=base, rss_loaded_mb=loaded, peak_serving_mb=peak))


if __name__ == "__main__":
    out = {}
    for name, tags in (("light", ["ours_s1"]), ("heavy", [f"ours_s{s}" for s in (2, 3, 4, 5)])):
        q = mp.Queue(); p = mp.Process(target=job, args=(tags, q)); p.start(); r = q.get(); p.join()
        out[name] = r; print(name, r, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "mem_profile.json"), "w"), indent=1)
