#!/usr/bin/env python3
"""Client for the decision server's real-model check (README.md, B and C). Standard library only.

    gpu_check.py read <items.jsonl> <endpoint> <out.jsonl>                  # serial; one record per item
    gpu_check.py load <items.jsonl> <endpoint> <out.json> <clients>         # every item once, <clients> at a time

Each item is sent as one Choice named "intent", the options as criteria with null descriptions.
"""
import json, statistics, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

INSTRUCTIONS = "Which of these intents does the user's message express?"


def ask(endpoint, item):
    body = {"state": item["state"], "questions": {"intent": {"type": "choice", "instructions": INSTRUCTIONS,
                                                              "criteria": {o: None for o in item["options"]}}}}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status, resp = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, resp = e.code, {"error": e.read().decode(errors="replace")[:500]}
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        status, resp = None, {"error": repr(e)[:500]}
    return status, resp, time.perf_counter() - t0


def record(item, status, resp, dt):
    a = (resp.get("answers") or {}).get("intent") or {}
    return {"id": item["id"], "gold": item["gold"], "n_options": len(item["options"]), "status": status,
            "latency_s": dt, "choice": a.get("choice"), "confidence": a.get("confidence"),
            "probabilities": a.get("probabilities"), "usage": resp.get("usage"), "error": resp.get("error")}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    mode, items_path, endpoint, out = sys.argv[1:5]
    items = [json.loads(line) for line in open(items_path) if line.strip()]
    if mode == "read":
        with open(out, "w") as f:
            for it in items:
                f.write(json.dumps(record(it, *ask(endpoint, it))) + "\n")
                f.flush()
    elif mode == "load":
        clients = int(sys.argv[5])
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            results = list(pool.map(lambda it: (it, *ask(endpoint, it)), items))
        wall = time.perf_counter() - t0
        lat = [dt for _it, st, _r, dt in results if st == 200]
        json.dump({"items": items_path, "endpoint": endpoint, "clients": clients, "requests": len(items),
                   "ok": len(lat), "non_200": len(items) - len(lat), "wall_s": wall, "requests_per_s": len(items) / wall,
                   "latency_s": {"p50": pct(lat, 0.50), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99),
                                 "mean": statistics.fmean(lat) if lat else None},
                   "accuracy": sum(r.get("answers", {}).get("intent", {}).get("choice") == it["gold"]
                                   for it, st, r, _dt in results if st == 200) / max(1, len(lat)),
                   "errors": sorted({str(r.get("error"))[:200] for _it, st, r, _dt in results if st != 200})[:5]},
                  open(out, "w"), indent=1)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
