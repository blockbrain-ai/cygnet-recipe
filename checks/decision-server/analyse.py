#!/usr/bin/env python3
"""The decision server's real-model check: every figure in README.md comes from this script.

    python3 analyse.py [out dir, default ./out] > analyse-output.txt
"""
import gzip, json, math, pathlib, sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent / "out")
T_SHIPPED = 3.4


def jl(p):
    p = pathlib.Path(p)
    f = open(p) if p.exists() else gzip.open(f"{p}.gz", "rt")      # the published copy keeps the reads gzipped
    return [json.loads(line) for line in f if line.strip()]


def temper(probs, T):
    w = {k: v ** (1.0 / T) for k, v in probs.items()}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def metrics(recs, T):
    ok = [r for r in recs if r["status"] == 200]
    acc = conf = nll = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]          # count, sum of confidence, sum of correct
    for r in ok:
        p = temper(r["probabilities"], T)
        top = max(p, key=p.get)
        c = p[top]
        hit = top == r["gold"]
        acc += hit; conf += c; nll += -math.log(max(p.get(r["gold"], 0.0), 1e-12))
        b = bins[min(9, int(c * 10))]
        b[0] += 1; b[1] += c; b[2] += hit
    n = len(ok)
    ece = sum(abs(b[1] - b[2]) for b in bins if b[0]) / n
    return {"n": n, "non_200": len(recs) - n, "accuracy": acc / n, "mean_top_p": conf / n, "ece": ece, "nll": nll / n}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


print(f"out: {OUT.name}/")
# ---- A
A = {a: {r["task_id"]: r for r in jl(OUT / a / "results.jsonl")} for a in ("A1", "A2", "A3")}
for a, rs in A.items():
    print(f"A {a}: {len(rs)} records; correct {sum(r['correct'] is True for r in rs.values())}")


def diff(x, y):
    ids = sorted(A[x])
    pred = sum(A[x][i]["predicted"] != A[y][i]["predicted"] for i in ids)
    dp = max(abs(A[x][i]["probs"][k] - A[y][i]["probs"][k]) for i in ids for k in A[y][i]["probs"])
    return pred, dp, [i for i in ids if A[x][i]["predicted"] != A[y][i]["predicted"]]


p23, d23, ids23 = diff("A2", "A3")
p13, d13, ids13 = diff("A1", "A3")
print(f"A server (A2) vs shim (A3): {p23} of 231 predictions differ {ids23}; largest probability difference {d23:.3g}")
print(f"A shim (A1) vs shim (A3), run-to-run: {p13} of 231 predictions differ {ids13}; largest probability difference {d13:.3g}")
print(f"A PASS: {p23 == 0 and d23 <= d13}   (A2 and A3 agree on all 231, and |dp|(A2, A3) <= |dp|(A1, A3))")
# ---- B
B = {b: jl(OUT / f"{b}.jsonl") for b in ("B77", "C150", "B20-single", "B20-grouped")}
for b, recs in B.items():
    lat = [r["latency_s"] for r in recs if r["status"] == 200]
    tok = [r["usage"]["input_tokens"] for r in recs if r["status"] == 200]
    print(f"B {b}: {recs[0]['n_options']} options; latency p50 {pct(lat, .5):.3f} s p95 {pct(lat, .95):.3f} s; "
          f"mean input tokens per request {sum(tok) / len(tok):.0f}")
    for T in (1.0, T_SHIPPED):
        m = metrics(recs, T)
        print(f"   T {T}: n {m['n']} (non-200 {m['non_200']}); accuracy {100 * m['accuracy']:.2f} %; mean top p "
              f"{m['mean_top_p']:.3f}; ECE {m['ece']:.3f}; NLL {m['nll']:.3f}")
s = {r["id"]: r for r in B["B20-single"]}
g = {r["id"]: r for r in B["B20-grouped"]}
both = [i for i in s if s[i]["status"] == 200 and g.get(i, {}).get("status") == 200]
hs = {i: s[i]["choice"] == s[i]["gold"] for i in both}
hg = {i: g[i]["choice"] == g[i]["gold"] for i in both}
agree = sum(s[i]["choice"] == g[i]["choice"] for i in both)
b_only = sum(hs[i] and not hg[i] for i in both)
c_only = sum(hg[i] and not hs[i] for i in both)
gap = 100 * (sum(hs.values()) - sum(hg.values())) / len(both)
print(f"B20 paired, {len(both)} items: single {sum(hs.values())} correct, grouped {sum(hg.values())}; "
      f"same top option on {agree}; right only single {b_only}, right only grouped {c_only} (exact McNemar p "
      f"{mcnemar_exact(b_only, c_only):.3g}); single minus grouped {gap:.2f} points")
print(f"B grouping fit to ship: {gap <= 3.0}   (grouped no more than 3 points below single pass)")
for b in ("B77", "C150"):
    e1, e34 = metrics(B[b], 1.0)["ece"], metrics(B[b], T_SHIPPED)["ece"]
    print(f"B {b}: ECE at T 3.4 {'below' if e34 < e1 else 'not below'} T 1 ({e34:.3f} vs {e1:.3f})")
# ---- C
for c in ("c1", "c2", "c3", "c4", "c5"):
    d = json.load(open(OUT / f"{c}.json"))
    L = d["latency_s"]
    print(f"C {c}: {d['clients']} clients, {d['requests']} requests of {'77' if c == 'c5' else '20'} options via "
          f"{d['endpoint'].rsplit(':', 1)[1]}: {d['requests_per_s']:.1f} requests/s; latency p50 {L['p50']:.3f} s, "
          f"p95 {L['p95']:.3f} s, p99 {L['p99']:.3f} s; non-200 {d['non_200']}; accuracy {100 * d['accuracy']:.1f} %")
