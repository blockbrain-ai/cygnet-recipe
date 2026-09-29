#!/usr/bin/env python3
"""Items for the decision server's real-model check (README.md), seeded, from the files in data/ (SOURCES.md).

    python3 build_items.py        # writes items-b77.jsonl, items-c150.jsonl, items-b20.jsonl, items-load-*.jsonl, items-build.json

Each item: id, state (the user's message), gold (its intent), options (intent names, shown to the model as they are).
B77 and C150 offer every intent in alphabetical order; B20 offers the gold intent and 19 others in random order. The
load items are BANKING77 test messages outside B's sample, one disjoint slice per C run, so no C prompt is cached.
"""
import csv, hashlib, json, pathlib, random

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
SEED, N = 20260929, 400
LOAD = {"c1": (200, 20), "c2": (400, 20), "c3": (400, 20), "c4": (400, 20), "c5": (200, 77)}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def write(name, rows):
    with open(HERE / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sha(HERE / name)


rng = random.Random(SEED)
b77 = list(csv.DictReader(open(DATA / "banking77-test.csv", newline="", encoding="utf-8")))
b77_labels = sorted({r["category"] for r in b77})
clinc = json.load(open(DATA / "clinc150-data_full.json"))["test"]
c150_labels = sorted({label for _text, label in clinc})
assert len(b77_labels) == 77 and len(c150_labels) == 150


def with_distractors(gold, k):
    opts = rng.sample([x for x in b77_labels if x != gold], k - 1) + [gold]
    rng.shuffle(opts)
    return opts


pick = rng.sample(range(len(b77)), N)
b77_items = [{"id": f"b77-{i}", "state": b77[i]["text"], "gold": b77[i]["category"], "options": b77_labels} for i in pick]
c150_items = [{"id": f"c150-{i}", "state": clinc[i][0], "gold": clinc[i][1], "options": c150_labels}
              for i in rng.sample(range(len(clinc)), N)]
b20_items = [{**it, "id": it["id"] + "-20", "options": with_distractors(it["gold"], 20)} for it in b77_items]

rest = [i for i in range(len(b77)) if i not in set(pick)]
rng.shuffle(rest)
record = {"seed": SEED, "sources": {f: sha(DATA / f) for f in ("banking77-test.csv", "clinc150-data_full.json")},
          "files": {"items-b77.jsonl": write("items-b77.jsonl", b77_items),
                    "items-c150.jsonl": write("items-c150.jsonl", c150_items),
                    "items-b20.jsonl": write("items-b20.jsonl", b20_items)}}
at = 0
for run, (n, k) in LOAD.items():
    rows = [{"id": f"load-{run}-{i}", "state": b77[i]["text"], "gold": b77[i]["category"],
             "options": b77_labels if k == 77 else with_distractors(b77[i]["category"], k)} for i in rest[at:at + n]]
    at += n
    record["files"][f"items-load-{run}.jsonl"] = write(f"items-load-{run}.jsonl", rows)
json.dump(record, open(HERE / "items-build.json", "w"), indent=1)
print(json.dumps(record, indent=1))
