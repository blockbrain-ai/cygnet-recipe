#!/usr/bin/env python3
"""Fit Cygnet's calibration temperature. Inputs are in this folder; JevBench's public set is not used.

    python3 calibration/fit.py <path to a JevBench checkout at 2fa63fa>

items.jsonl        241 items we generated (date-window, date-ordering and long-policy families; labels computed by the
                   generators and re-derived from each item's text)
letters-reads.jsonl  the frozen model's letter distribution on each item (T = 1), read with the same readout as
                   shim/cygnet_shim.py (the V14-fixed letter sums), one read per item
The temperature minimises negative log-likelihood of the gold label over T in 0.5, 0.6, ..., 5.0, applied as p^(1/T)
renormalised. Prints T and, as a check, the fit on one half with the other half held out.
"""
import json, math, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, sys.argv[1])
from jevbench.tasks import Task
from jevbench.scoring import score_task
from jevbench.metrics import ece_top_label

T = {t.id: t for t in (Task.from_dict(json.loads(l)) for l in open(here / "items.jsonl") if l.strip())}
data = []
for l in open(here / "letters-reads.jsonl"):
    r = json.loads(l); t = T[r["task_id"]]; a = r["answer"]
    p = {"yes": a["noul"], "no": 1 - a["noul"]} if t.question["type"] == "noul" else dict(a["probabilities"])
    data.append((t, p))
order = {tid: i for i, tid in enumerate(T)}
data.sort(key=lambda tp: order[tp[0].id])


def temper(p, t):
    z = {k: max(v, 1e-12) ** (1 / t) for k, v in p.items()}; s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def evaluate(d, t):
    nll, pairs = 0.0, []
    for task, p in d:
        q = temper(p, t); s = score_task(q, task)
        pairs.append((max(s["probs"].values()), bool(s["correct"])))
        gold = str(task.expected).lower() if task.question["type"] == "noul" else str(task.expected)
        nll -= math.log(max(q.get({"true": "yes", "false": "no"}.get(gold, gold), 1e-12), 1e-12))
    return nll / len(d), ece_top_label(pairs)["ece"]


grid = [round(0.5 + 0.1 * i, 2) for i in range(46)]
best = min(grid, key=lambda t: evaluate(data, t)[0])
print(f"all {len(data)} items: T = {best}; NLL {evaluate(data, 1.0)[0]:.3f} -> {evaluate(data, best)[0]:.3f}")
fit, held = data[0::2], data[1::2]
bh = min(grid, key=lambda t: evaluate(fit, t)[0])
print(f"fit on {len(fit)} (T = {bh}), held-out {len(held)}: ECE {evaluate(held, 1.0)[1]:.3f} -> {evaluate(held, bh)[1]:.3f}")
