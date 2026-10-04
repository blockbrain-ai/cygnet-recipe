"""The README's pictures, drawn from recorded numbers: python showcase/render.py

Writes hero, board, axes and readout SVGs, each in a light and a dark version, beside this file.
Standard library only. Board and axis figures are the constants below, copied from JevBench
v1.5.4's aggregate results (https://benchmarkheaven.com/api/jevbench/v1.5.4, sha256
0cf210b76bf85084a5f3fb40fb109e9a2c2f93df42ff6628696377666e89db45, retrieved 2026-09-30). The readout
item is read from runs/l40s-pinned/results.jsonl in this repository.

GitHub shows these inside <img>: no script runs and no web font loads, so text uses the reader's
own fonts and every run leaves room to spare. Only the readout moves; it changes the bars from the
model's raw distribution to the calibrated one, which is a real transformation, and it rests on the
distribution as returned when the reader prefers reduced motion. The charts do not animate.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

RELEASE = "JevBench v1.5.4"
RANKED = 106
SOURCE_SHA = "0cf210b76bf85084a5f3fb40fb109e9a2c2f93df42ff6628696377666e89db45"
TIE_CI = (-0.46, 1.24)   # board A's paired-bootstrap marker for ranks 1 and 2: "joint leaders (statistical tie)"

# rank, name, how it is built (our reading of each row's notes), score, 95% bootstrap interval,
# then the four axes: intelligence, calibration, speed, cost
BOARD = [
    (1, "Cygnet", "stock weights", 73.7013, 72.3624, 74.4595, 71.09, 87.01, 90.97, 56.43),
    (2, "Winnow-12B Q8", "Gemma-4-12B fine-tune", 73.2331, 72.0217, 73.9947, 74.43, 84.07, 86.14, 56.56),
    (3, "Jev 1.13.0", "proprietary", 72.1329, 71.0143, 72.6148, 72.00, 88.03, 83.81, 54.73),
    (4, "JevK5 v0.3", "Qwen3.5-4B fine-tune", 71.8979, 69.3919, 72.9476, 56.27, 88.34, 93.55, 63.07),
    (5, "Plumb-4B", "Qwen3.5-4B fine-tune", 71.5607, 69.1983, 72.7280, 55.85, 87.44, 93.45, 63.07),
    (6, "Jev-Omni", "Gemma-4-12B fine-tune", 71.5005, 70.2142, 72.4014, 70.49, 82.60, 84.68, 56.05),
    (7, "decider-4b v2", "Qwen3.5-4B fine-tune", 71.2842, 69.1105, 72.3469, 55.77, 85.59, 90.86, 64.54),
    (8, "Decision 4B v1.2", "Qwen3.5-4B fine-tune", 70.8274, 68.5159, 71.9677, 53.66, 88.56, 93.50, 63.07),
    (9, "Imajev-4B", "Qwen3.5-4B fine-tune", 70.3866, 67.7997, 71.6119, 53.47, 88.12, 91.13, 63.26),
    (10, "Decision 4B v1.1", "Qwen3.5-4B fine-tune", 70.3858, 66.8584, 71.5662, 53.11, 87.31, 93.52, 63.07),
    (34, "GPT-6 Luna, low effort", "general LLM, API", 40.4820, 40.2690, 40.6639, 95.29, 94.92, 73.20, 39.06),
    (51, "Gemini 3.1 Flash-Lite", "general LLM, API", 19.5835, 19.3036, 19.8318, 77.59, 74.68, 80.05, 29.76),
    (68, "DeepSeek V4.1 Flash", "general LLM, API", 6.6455, 6.6241, 6.6635, 93.69, 96.92, 69.40, 19.09),
]
AXES_ROWS = ["Cygnet", "Jev 1.13.0", "Winnow-12B Q8", "GPT-6 Luna, low effort"]

# one public item, as the shim returned it (T = 3.4) in the L40S run
ITEM_ID = "original-extraction-01-1"
ITEM_STATE = "The final arrangement is depot pickup, replacing the earlier courier idea."
ITEM_ASK = "Extract the final confirmed delivery method. Ignore cancelled plans and hypothetical alternatives."
ITEM_OPTIONS = [("A", "courier", "Courier delivery"), ("B", "pickup", "Customer pickup"),
                ("C", "post", "Postal service"), ("D", "unknown", "No final confirmed method")]
T = 3.4

# a night sky over the swan, and a first-place gold. `gold` marks Cygnet in charts (validated
# against the slate on each chart surface); `glow` is the brighter gold for display text only
THEMES = {
    "light": dict(sky="#f2f4f9", surface="#ffffff", surface2="#e9edf4", line="#d9dee8", ink="#141b2d",
                  ink2="#46506a", ink3="#737c92", gold="#a26f00", glow="#8f6200", slate="#8590a6",
                  grid="#141b2d17", tint="#a26f0014"),
    "dark": dict(sky="#0a0f1e", surface="#121a2e", surface2="#1a2340", line="#263050", ink="#e9edf7",
                 ink2="#aeb7cc", ink3="#8a93aa", gold="#bb8a22", glow="#f0b429", slate="#74809a",
                 grid="#e9edf71c", tint="#f0b4291a"),
}
SERIF = "Georgia,'Iowan Old Style','Palatino Linotype','Book Antiqua',Palatino,serif"
SANS = "system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"


class Svg:
    def __init__(self, w: int, h: int, title: str, desc: str, t: dict, css: str = ""):
        self.w, self.h, self.t, self.parts = w, h, t, []
        colours = "".join(f".{k}{{fill:{v}}}" for k, v in t.items())
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                     f'role="img" aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
                     f"<style>.serif{{font-family:{SERIF}}}.sans{{font-family:{SANS}}}.mono{{font-family:{MONO}}}"
                     f"text{{font-variant-numeric:tabular-nums}}{colours}{css}</style>")

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size, colour, face="sans", weight=400, anchor="start", cls="", spacing=None, italic=False):
        attrs = [f'x="{x:g}"', f'y="{y:g}"', f'font-size="{size}"', f'class="{face} {colour}{" " + cls if cls else ""}"']
        if weight != 400:
            attrs.append(f'font-weight="{weight}"')
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if spacing:
            attrs.append(f'letter-spacing="{spacing}"')
        if italic:
            attrs.append('font-style="italic"')
        self.add(f"<text {' '.join(attrs)}>{escape(s)}</text>")

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, cls=""):
        s = f' stroke="{stroke}"' if stroke else ""
        c = f' class="{cls}"' if cls else ""
        self.add(f'<rect x="{x:g}" y="{y:g}" width="{max(w, 0):.2f}" height="{h:g}" rx="{rx}" fill="{fill}"{s}{c}/>')

    def line(self, x1, y1, x2, y2, stroke, width=1):
        self.add(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" stroke-width="{width}"/>')

    def done(self) -> str:
        return self.head + "".join(self.parts) + "</svg>\n"


def cygnus(svg: Svg, x: float, y: float, size: float, colour: str) -> None:
    """The mark: the Northern Cross of Cygnus, the swan - tail, heart and head on the long line,
    the wing tips across it."""
    stars = {"tail": (0.14, 0.08), "heart": (0.46, 0.40), "neck": (0.66, 0.64), "head": (0.90, 0.92),
             "wing1": (0.86, 0.14), "wing2": (0.10, 0.74)}
    p = {k: (x + a * size, y + b * size) for k, (a, b) in stars.items()}
    for a, b in (("tail", "heart"), ("heart", "neck"), ("neck", "head"), ("wing1", "heart"), ("heart", "wing2")):
        svg.line(*p[a], *p[b], colour, max(1.2, size / 30))
    for k, (cx, cy) in p.items():
        r = size * (0.085 if k in ("tail", "heart") else 0.06)
        svg.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{colour}"/>')


# ------------------------------------------------------------------------------------ hero
def hero(t: dict) -> str:
    W, H = 880, 290
    s = Svg(W, H, "Cygnet: No. 1 overall on JevBench v1.5.4",
            f"Cygnet, typed decisions from frozen Gemma-4-12B with one token per decision, ranks first of {RANKED} "
            f"systems on the official {RELEASE} score at 73.70, ahead of Winnow-12B Q8 at 73.23 (a statistical tie) "
            "and Jev 1.13.0 at 72.13.", t)
    s.rect(1, 1, W - 2, H - 2, t["sky"], 16, t["line"])
    cygnus(s, 40, 38, 46, t["glow"])
    s.text(100, 78, "Cygnet", 46, "ink", "serif")
    s.text(40, 128, "Typed decisions from frozen Gemma-4-12B,", 18, "ink2")
    s.text(40, 153, "one token per decision.", 18, "ink2")
    s.text(40, 214, f"No. 1 overall on JevBench", 30, "glow", "serif")
    s.text(40, 246, "stock weights  ·  no fine-tuning  ·  one output token", 13, "ink3", "mono")
    # the board's top three
    x0, y0, cw, ch = 520, 34, 322, 222
    s.rect(x0, y0, cw, ch, t["surface"], 12, t["line"])
    s.text(x0 + 22, y0 + 32, f"{RELEASE.upper()}  ·  OFFICIAL SCORE", 11, "ink3", weight=600, spacing="0.8")
    for i, (rank, name, _tag, score, *_rest) in enumerate(BOARD[:3]):
        yy = y0 + 76 + i * 44
        first = i == 0
        if first:
            s.rect(x0 + 10, yy - 28, cw - 20, 40, t["tint"], 8)
        s.text(x0 + 26, yy, str(rank), 26, "glow" if first else "ink3", "serif")
        s.text(x0 + 60, yy - 2, name, 16, "ink" if first else "ink2", weight=600 if first else 400)
        s.text(x0 + cw - 24, yy - 2, f"{score:.2f}", 16, "ink" if first else "ink2", "mono", 600 if first else 400, "end")
    s.text(x0 + 22, y0 + ch - 20, f"of {RANKED} ranked  ·  1 and 2 are a statistical tie", 12, "ink3")
    return s.done()


# ----------------------------------------------------------------------------------- board
def board(t: dict) -> str:
    W = 880
    top, row = 118, 34
    rows = BOARD[:10] + [None] + BOARD[10:]
    H = top + len(rows) * row + 96
    s = Svg(W, H, f"{RELEASE}: the top ten and three general LLMs",
            f"Official JevBench Score out of 100 with 95% bootstrap intervals. " +
            "; ".join(f"#{r[0]} {r[1]} ({r[2]}) {r[3]:.2f}" for r in BOARD) + ".", t)
    s.rect(1, 1, W - 2, H - 2, t["surface"], 14, t["line"])
    s.text(32, 46, f"No. 1 of {RANKED}, and the only top-ten system on stock weights", 19, "ink", weight=600)
    s.text(32, 72, "Official JevBench Score, 0 to 100: the equal-weight harmonic mean of Intelligence, Calibration,", 13.5, "ink2")
    s.text(32, 92, "Speed and Cost. Whiskers are 95% bootstrap intervals.", 13.5, "ink2")
    L, R = 392, 812
    k = (R - L) / 100
    bottom = top + len(rows) * row - 8
    for v in (0, 25, 50, 75, 100):
        x = L + v * k
        s.line(x, top - 12, x, bottom, t["ink3"] if v == 0 else t["grid"], 1.4 if v == 0 else 1)
        s.text(x, bottom + 18, str(v), 12, "ink3", anchor="middle")
    for i, r in enumerate(rows):
        y = top + i * row
        if r is None:
            s.text(64, y + 12, f"23 systems between", 12, "ink3", italic=True)
            s.line(32, y + 20, R, y + 20, t["line"])
            continue
        rank, name, tag, score, lo, hi = r[:6]
        first = rank == 1
        if first:
            s.rect(22, y - 6, R - 14, row - 2, t["tint"], 6)
        s.text(52, y + 14, str(rank), 13, "ink" if first else "ink3", "mono", 600 if first else 400, "end")
        s.text(64, y + 14, name, 14, "ink" if first else "ink2", weight=600 if first else 400)
        s.text(236, y + 14, tag, 12, "ink" if first else "ink3", weight=600 if first else 400)
        s.rect(L, y + 3, score * k, 16, t["gold"] if first else t["slate"])
        cy = y + 11
        s.line(L + lo * k, cy, L + hi * k, cy, t["ink"], 1.4)
        s.line(L + lo * k, cy - 5, L + lo * k, cy + 5, t["ink"], 1.4)
        s.line(L + hi * k, cy - 5, L + hi * k, cy + 5, t["ink"], 1.4)
        s.text(L + hi * k + 8, y + 15, f"{score:.2f}", 12.5, "ink" if first else "ink2", "mono", 600 if first else 400)
    s.text(32, H - 44, f"Ranks 1 and 2 are a statistical tie by the board's own paired bootstrap "
                       f"(difference {TIE_CI[0]:+.2f} to {TIE_CI[1]:+.2f}). How each system is built is our reading of its row.", 12, "ink3")
    s.text(32, H - 24, f"Source: {RELEASE} aggregate results, benchmarkheaven.com, sha256 {SOURCE_SHA[:12]}…, retrieved 2026-09-30.",
           12, "ink3")
    return s.done()


# ------------------------------------------------------------------------------------ axes
def axes(t: dict) -> str:
    W = 880
    names = {r[1]: r for r in BOARD}
    picked = [names[n] for n in AXES_ROWS]
    top, row = 138, 38
    H = top + len(picked) * row + 96
    s = Svg(W, H, f"{RELEASE}: the four axes behind the score",
            "Axis scores out of 100. " + "; ".join(f"{r[1]}: intelligence {r[6]}, calibration {r[7]}, speed {r[8]}, cost {r[9]}"
                                                   for r in picked) + ".", t)
    s.rect(1, 1, W - 2, H - 2, t["surface"], 14, t["line"])
    s.text(32, 46, "About a point behind Jev on Intelligence and Calibration, ahead on Speed and Cost", 19, "ink", weight=600)
    s.text(32, 72, f"{RELEASE} axis scores, 0 to 100, higher is better on each", 13.5, "ink2")
    cols = [("Intelligence", 6), ("Calibration", 7), ("Speed", 8), ("Cost", 9)]
    x0, colw, bar = 214, 158, 104
    for c, (label, _j) in enumerate(cols):
        s.text(x0 + c * colw, top - 22, label, 13, "ink", weight=600)
    for i, r in enumerate(picked):
        y = top + i * row
        first = r[1] == "Cygnet"
        if first:
            s.rect(22, y - 8, W - 44, row - 4, t["tint"], 6)
        s.text(32, y + 13, r[1], 14, "ink" if first else "ink2", weight=600 if first else 400)
        for c, (_label, j) in enumerate(cols):
            x = x0 + c * colw
            s.rect(x, y + 1, bar, 14, t["surface2"])
            s.rect(x, y + 1, bar * r[j] / 100, 14, t["gold"] if first else t["slate"])
            s.text(x + bar + 6, y + 13, f"{r[j]:.1f}", 12.5, "ink" if first else "ink2", "mono", 600 if first else 400)
    s.text(32, H - 62, "The score is a harmonic mean, so the weakest axis weighs most: GPT-6 Luna out-reasons every row here,", 12, "ink3")
    s.text(32, H - 44, "but its Cost axis of 39.1 holds its score to 40.48.", 12, "ink3")
    s.text(32, H - 22, f"Source: {RELEASE} aggregate results, benchmarkheaven.com, sha256 {SOURCE_SHA[:12]}…, retrieved 2026-09-30.",
           12, "ink3")
    return s.done()


# --------------------------------------------------------------------------------- readout
LOOP = 8.0


def pct(sec: float) -> str:
    return f"{sec / LOOP * 100:.2f}%"


def item_probs() -> tuple[dict, dict, dict]:
    rec = next(r for r in (json.loads(l) for l in (REPO / "runs/l40s-pinned/results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
               if r["task_id"] == ITEM_ID)
    returned = rec["probs"]
    raw_unnorm = {k: v ** T for k, v in returned.items()}  # the shim returns p^(1/T), renormalised; this undoes it
    z = sum(raw_unnorm.values())
    return rec, {k: v / z for k, v in raw_unnorm.items()}, returned


def fmt(p: float) -> str:
    if p < 0.0001:
        return "<0.01%"
    return f"{p * 100:.2f}%" if p < 0.1 or p >= 0.99 else f"{p * 100:.1f}%"


def readout(t: dict) -> str:
    rec, raw, cal = item_probs()
    W, H = 880, 356
    ease = "cubic-bezier(0.2,0,0,1)"
    css = [".bar{transform-box:fill-box;transform-origin:0 50%}"]
    rows = []
    for i, (_letter, key, _desc) in enumerate(ITEM_OPTIONS):
        a, b = raw[key], cal[key]  # drawn exactly: a near-zero bar draws as nothing, on its track, with its number beside it
        css.append(f"@keyframes s{i}{{0%,{pct(2.6)}{{transform:scaleX({a:.4f});animation-timing-function:{ease}}}"
                   f"{pct(3.05)},{pct(7.3)}{{transform:scaleX({b:.4f});animation-timing-function:{ease}}}"
                   f"{pct(7.6)},100%{{transform:scaleX({a:.4f})}}}}"
                   f".s{i}{{transform:scaleX({b:.4f});animation:s{i} {LOOP}s infinite}}")
        rows.append((key, raw[key], cal[key]))
    fade_out = (f"@keyframes before{{0%,{pct(2.6)}{{opacity:1}}{pct(2.9)},{pct(7.35)}{{opacity:0}}{pct(7.6)},100%{{opacity:1}}}}"
                f".before{{opacity:0;animation:before {LOOP}s infinite}}")
    fade_in = (f"@keyframes after{{0%,{pct(2.75)}{{opacity:0}}{pct(3.05)},{pct(7.3)}{{opacity:1}}{pct(7.5)},100%{{opacity:0}}}}"
               f".after{{animation:after {LOOP}s infinite}}")
    css += [fade_out, fade_in,
            "@media (prefers-reduced-motion:reduce){.bar,.before,.after{animation:none}.before{opacity:0}.after{opacity:1}}"]
    s = Svg(W, H, "How Cygnet reads a decision",
            f"JevBench public item {ITEM_ID}. The options are lettered A to D and Gemma-4-12B-it answers with one token, "
            "limited to those letters. The shim sums every token that decodes to the same letter, renormalises over the "
            "options, and applies one temperature, T = 3.4. " +
            "; ".join(f"{k}: {fmt(r)} before the temperature, {fmt(c)} as returned" for k, r, c in rows) + ".",
            t, "".join(css))
    s.rect(1, 1, W - 2, H - 2, t["surface"], 14, t["line"])

    # the item
    s.text(32, 42, "ONE JEVBENCH PUBLIC ITEM", 11, "ink3", weight=600, spacing="0.8")
    s.text(32, 68, "“The final arrangement is depot pickup,", 14, "ink", "serif", italic=True)
    s.text(32, 88, "replacing the earlier courier idea.”", 14, "ink", "serif", italic=True)
    s.text(32, 116, "Extract the final confirmed delivery method.", 12.5, "ink2")
    s.text(32, 134, "Ignore cancelled plans and hypothetical alternatives.", 12.5, "ink2")
    for i, (letter, key, desc) in enumerate(ITEM_OPTIONS):
        y = 162 + i * 30
        s.rect(32, y, 272, 24, t["surface2"], 6)
        s.text(44, y + 17, letter, 13, "gold", "mono", 700)
        s.text(64, y + 17, key, 13, "ink", "mono")
        s.text(134, y + 17, desc, 12, "ink3")

    # the pass
    x = 330
    s.rect(x, 52, 196, 236, t["sky"], 12, t["line"])
    s.text(x + 98, 84, "Gemma-4-12B-it", 16, "ink", "serif", anchor="middle")
    s.text(x + 98, 104, "frozen, on stock vLLM 0.30.0", 12, "ink3", anchor="middle")
    for j, line in enumerate(("one forward pass;", "the answer position may", "only be A, B, C or D;",
                              "20 log-probabilities back;", "tokens that decode to the", "same letter are summed")):
        s.text(x + 18, 138 + j * 20, line, 12.5, "ink2")
    s.text(x + 98, 274, f"{rec['usage']['prompt_tokens']} in · 1 out · {rec['latency_s']:.3f} s", 12, "ink3", "mono", anchor="middle")
    for ax in (x - 22, x + 206):
        s.add(f'<path d="M{ax} 170h14m-5 -5l5 5l-5 5" fill="none" stroke="{t["ink3"]}" stroke-width="1.5"/>')

    # the distribution
    x = 562
    track = 200
    s.add('<g class="before">')
    s.text(x, 42, "BEFORE THE TEMPERATURE (T = 1)", 11, "ink3", weight=600, spacing="0.8")
    s.add("</g>")
    s.add('<g class="after">')
    s.text(x, 42, "AS RETURNED (T = 3.4)", 11, "ink2", weight=600, spacing="0.8")
    s.add("</g>")
    for i, (key, r, c) in enumerate(rows):
        y = 80 + i * 50
        win = key == "pickup"
        s.text(x, y, key, 14, "ink" if win else "ink2", "mono", 600 if win else 400)
        s.rect(x, y + 10, track, 12, t["surface2"])
        s.add(f'<rect class="bar s{i}" x="{x}" y="{y + 10}" width="{track}" height="12" fill="{t["gold"] if win else t["slate"]}"/>')
        s.add('<g class="before">')
        s.text(x + track + 76, y, fmt(r), 14, "ink" if win else "ink2", "mono", 600 if win else 400, "end")
        s.add("</g>")
        s.add('<g class="after">')
        s.text(x + track + 76, y, fmt(c), 14, "ink" if win else "ink2", "mono", 600 if win else 400, "end")
        s.add("</g>")
    s.text(32, H - 34, f"Recorded in runs/l40s-pinned/results.jsonl (item {ITEM_ID}, answered correctly). The T = 1 figures undo "
                       "the temperature exactly:", 12, "ink3")
    s.text(32, H - 16, "p ∝ p_returned^3.4. T = 3.4 was fitted on 241 items of our own; JevBench's public items were never used to fit it.",
           12, "ink3")
    return s.done()


def main() -> None:
    for name, fn in (("hero", hero), ("board", board), ("axes", axes), ("readout", readout)):
        for mode, t in THEMES.items():
            out = HERE / f"{name}-{mode}.svg"
            out.write_text(fn(t), encoding="utf-8", newline="\n")
            print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    main()
