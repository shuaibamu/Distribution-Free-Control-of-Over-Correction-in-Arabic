"""Masked-word pretraining of the compact Arabic encoder on clean classical text.
Supports single-process and torch.distributed (gloo) data-parallel runs so the
same script yields the measured DDP scaling numbers."""
import os, sys, time, json, argparse, random
import torch, torch.nn.functional as F
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(__file__))
from common import *


def mask_batch(W, M, V, rng, p=0.15):
    T = torch.full_like(W, -100)
    sel = (torch.rand(W.shape, generator=rng) < p) & M & (W > 4)
    T[sel] = W[sel]
    W2 = W.clone()
    r = torch.rand(W.shape, generator=rng)
    W2[sel & (r < 0.8)] = MASK
    rnd = sel & (r >= 0.8) & (r < 0.9)
    W2[rnd] = torch.randint(5, len(V.itos), (int(rnd.sum()),), generator=rng)
    return W2, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(CK, "mlm.pt"))
    ap.add_argument("--timing_only", action="store_true")
    a = ap.parse_args()
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    rank, world = 0, 1
    if ddp:
        dist.init_process_group("gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    set_seed(1 + rank)
    V = Vocab()
    data = [{"noisy": s} for s in load(f"pretrain{os.environ.get('VARIANT', '')}.json")]
    data = data[rank::world]
    model = MLM(V)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    steps_ep = len(data) // a.bs
    total = a.max_steps or int(a.epochs * steps_ep)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=total, pct_start=0.06)
    rng = torch.Generator().manual_seed(11 + rank)
    step, t0, log = 0, time.time(), []
    model.train()
    while step < total:
        random.shuffle(data)
        for i in range(0, len(data) - a.bs + 1, a.bs):
            W, C, M, *_ = batchify(V, data[i:i + a.bs], with_tags=False)
            W2, T = mask_batch(W, M, V, rng)
            sel = T != -100
            logits = model(W2, C, M, sel)
            loss = F.cross_entropy(logits, T[sel])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0 and rank == 0:
                el = time.time() - t0
                log.append({"step": step, "loss": loss.item(), "elapsed": el})
                print(f"step {step}/{total} loss {loss.item():.3f} {el:.0f}s", flush=True)
            if step >= total:
                break
    el = time.time() - t0
    if rank == 0:
        if not a.timing_only:
            sd = model.module.state_dict() if ddp else model.state_dict()
            torch.save(sd, a.out)
            json.dump(log, open(os.path.join(RES, "mlm_log.json"), "w"))
        rec = {"world": world, "steps": total, "batch_per_worker": a.bs, "seconds": el,
               "samples_per_s": total * a.bs * world / el}
        with open(os.path.join(RES, "ddp_timing.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
