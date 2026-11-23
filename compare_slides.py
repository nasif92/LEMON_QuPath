"""Per-slide paired comparison of two configurations, averaged over seeds.

    python compare_slides.py runs/Ki67_*.log                                   # default: fusion all_derived vs qupath all_derived
    python compare_slides.py runs/Ki67_*.log --a "fusion | context_only_derived" --b "qupath | context_only_derived"

Names are the ones summarize.py prints. Prints accuracy per slide for A and B, the difference, how many slides A wins,
and a paired t-test over slides (only a rough guide with 6 slides).
"""
import argparse
import json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--a", default="fusion | all_derived")
ap.add_argument("--b", default="qupath | all_derived")
ap.add_argument("--set", choices=["annotated", "labelled"], default="annotated", help="which test set to compare")
a = ap.parse_args()

acc = defaultdict(lambda: defaultdict(list))        # config -> slide -> [acc per seed]
for path in a.logs:
    for line in open(path, errors="replace"):
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            name = r["input"] + ("" if r["feature_set"] is None else f" | {r['feature_set']}") + \
                ("" if not r["lemon_pca"] else f" | pca{r['lemon_pca']}")
            for slide, v in (r["per_slide"] if a.set == "annotated" else r.get("per_slide_labelled", {})).items():
                acc[name][slide].append(v)

for n in (a.a, a.b):
    if n not in acc:
        raise SystemExit(f"'{n}' not found. Available: {sorted(acc)}")
slides = sorted(set(acc[a.a]) & set(acc[a.b]))
A = np.array([np.mean(acc[a.a][s]) for s in slides])
B = np.array([np.mean(acc[a.b][s]) for s in slides])
d = A - B
print(f"A = {a.a}\nB = {a.b}\n")
print(f"{'slide':<32}{'A':>8}{'B':>8}{'A-B':>9}")
for s, x, y, z in sorted(zip(slides, A, B, d), key=lambda t: t[3]):
    print(f"{s:<32}{x:8.4f}{y:8.4f}{z:+9.4f}")
print(f"\nslide mean: A {A.mean():.4f}  B {B.mean():.4f}  diff {d.mean():+.4f}")
print(f"A better on {(d > 0).sum()} / {len(d)} slides, worse on {(d < 0).sum()}")
if len(d) > 2 and d.std(ddof=1) > 0:
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    try:
        from scipy import stats
        print(f"paired t = {t:.2f}, p = {2 * stats.t.sf(abs(t), len(d) - 1):.3f}  (n = {len(d)} slides; rough guide only)")
    except ImportError:
        print(f"paired t = {t:.2f}  (n = {len(d)} slides; scipy not installed for p)")