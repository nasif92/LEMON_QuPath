"""Slide-grouped k-fold cross-validation on the TRAINING slides (phases 1-3). The reference slides are never loaded.

Every configuration sees exactly the same folds, so differences between them are paired. One model per stain, same MLP
and training settings as train.py; the standardisation and median fill use only the training folds' rows.

    python cv.py --subtypes Ki67 PR ER --folds 5 --seeds 42 > cv_log.txt
    python cv.py --subtypes Ki67 --folds 5 --seeds 42 43 44 --configs qupath_all fusion_all

Writes cv_runs/cv_results.csv (one row per held-out slide, config, seed) and prints the tables.
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import lemon_data as D
import lemon_paths as P
from train import MLP

# name -> (QuPath feature set or None, use LEMON, PCA components)
CONFIGS = {
    "qupath_all":       ("all_derived", False, 0),
    "lemon":            (None, True, 0),
    "fusion_all":       ("all_derived", True, 0),
    "fusion_all_pca32": ("all_derived", True, 32),
    "qupath_context":   ("context_only_derived", False, 0),
    "fusion_context":   ("context_only_derived", True, 0),
    "qupath_nocontext": ("no_context_derived", False, 0),
}
PAIRS = [("fusion_all", "qupath_all"), ("fusion_context", "qupath_context"),
         ("fusion_context", "qupath_all"), ("lemon", "qupath_nocontext")]


def build_inputs(tr_rows, te_rows, E_tr, E_te, qcols, use_lemon, pca, seed):
    blocks_tr, blocks_te = [], []
    if qcols:
        med = tr_rows[qcols].median()
        a = tr_rows[qcols].astype(np.float64).fillna(med).to_numpy()
        b = te_rows[qcols].astype(np.float64).fillna(med).to_numpy()
        mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
        blocks_tr.append(np.clip((a - mu) / sd, -20, 20)); blocks_te.append(np.clip((b - mu) / sd, -20, 20))
    if use_lemon:
        a, b = E_tr.astype(np.float64), E_te.astype(np.float64)
        if pca:
            from sklearn.decomposition import PCA
            p = PCA(n_components=pca, random_state=seed).fit(a)
            a, b = p.transform(a), p.transform(b)
        mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
        blocks_tr.append(np.clip((a - mu) / sd, -20, 20)); blocks_te.append(np.clip((b - mu) / sd, -20, 20))
    return np.concatenate(blocks_tr, 1).astype(np.float32), np.concatenate(blocks_te, 1).astype(np.float32)


def fit_predict(Xtr, ytr, Xte, a, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t, Xte_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Xte).to(dev)
    y = torch.as_tensor(ytr, device=dev)
    model = MLP(Xtr.shape[1], 2, a.hidden, a.dropout).to(dev)
    cnt = Counter(ytr.tolist())
    w = torch.as_tensor([1.0 / cnt[int(t)] ** a.reweight_power for t in ytr.tolist()], dtype=torch.double, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    steps = max(1, len(y) // a.batch_size)
    for _ in range(a.epochs):
        model.train()
        for _ in range(steps):
            i = torch.multinomial(w, a.batch_size, replacement=True)
            opt.zero_grad()
            crit(model(Xtr_t[i]), y[i]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(Xte_t[i:i + 50000]) for i in range(0, len(Xte_t), 50000)]).argmax(1).cpu().numpy()


def phase_of(path):
    s = str(path)
    for tag in ("phase_1", "phase1", "phase_2", "phase2", "phase_3", "phase3"):
        if tag in s:
            return "phase" + tag[-1]
    return "other"


def summarize(res, subtypes, configs):
    per = res.groupby(["subtype", "config", "slide_uid"]).acc.mean().reset_index()     # mean over seeds, one row per slide
    tab = per.groupby(["config", "subtype"]).acc.mean().unstack("subtype").reindex(configs)[subtypes]
    tab["mean"] = tab.mean(1)
    print("\nSlide-mean accuracy on held-out training slides (mean over slides, then over stains)")
    print(tab.round(4).to_string())
    print("\nPaired differences over held-out slides (A - B)")
    for A, B in PAIRS:
        if A not in configs or B not in configs:
            continue
        wide = per.pivot_table(index=["subtype", "slide_uid"], columns="config", values="acc")
        d = (wide[A] - wide[B]).dropna()
        if d.empty:
            continue
        line = f"  {A} vs {B}: {d.mean() * 100:+.2f} points, A better on {(d > 0).sum()}/{len(d)} slides"
        if len(d) > 2 and d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            try:
                from scipy import stats
                line += f", t={t:.2f}, p={2 * stats.t.sf(abs(t), len(d) - 1):.3f}"
            except ImportError:
                line += f", t={t:.2f}"
        print(line)
        by = d.groupby(level="subtype").mean() * 100
        print("      by stain: " + ", ".join(f"{k} {v:+.2f}" for k, v in by.items()))
    if "phase" in res.columns:
        ph = res.groupby(["config", "phase"]).acc.mean().unstack("phase").reindex(configs)
        print("\nBy annotation phase of the held-out slide (accuracy, mean over slides)")
        print(ph.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtypes", nargs="+", default=["Ki67"])
    ap.add_argument("--classes", nargs="+", default=["Tumor", "Non Tumor"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0, help="seeds the slide-to-fold assignment (same for all configs)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="model seeds")
    ap.add_argument("--lemon", default=P.LEMON_NPZ)
    ap.add_argument("--hidden", nargs="+", type=int, default=[96, 48, 24, 12, 6])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--reweight_power", type=float, default=0.5)
    ap.add_argument("--device", default="")
    ap.add_argument("--cache_dir", default=P.CACHE_DIR)
    ap.add_argument("--out_dir", default=os.path.join(P.PROJECT_DIR, "cv_runs"))
    a = ap.parse_args()

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = list(a.classes)
    c2i = {c: i for i, c in enumerate(classes)}
    tr_all = D.label_and_filter(D.load_training(a.subtypes, a.cache_dir), classes, "train")
    idx = D.EmbeddingIndex(a.lemon) if any(CONFIGS[c][1] for c in a.configs) else None
    print(f"device={dev}  folds={a.folds}  model seeds={a.seeds}  configs={a.configs}  (reference slides not loaded)")

    rows = []
    for st in a.subtypes:
        d = tr_all[tr_all["subtype"] == st].reset_index(drop=True)
        if d.empty:
            print(f"{st}: no training nuclei")
            continue
        slides = sorted(d["slide_uid"].unique())
        order = np.random.RandomState(a.fold_seed).permutation(len(slides))
        fold_of = {slides[j]: n % a.folds for n, j in enumerate(order)}
        fold = d["slide_uid"].map(fold_of).to_numpy()
        phase = d["ann_source_dir"].map(phase_of)
        y = d["_label"].map(c2i).to_numpy()
        E = idx.lookup(d, st) if idx is not None else None
        print(f"\n{st}: {len(d):,} nuclei, {len(slides)} slides, folds of "
              f"{sorted(Counter(fold_of.values()).values())} slides; slides per phase "
              f"{dict(Counter(phase.groupby(d['slide_uid']).first()))}")
        for name in a.configs:
            fs, use_lemon, pca = CONFIGS[name]
            qcols = []
            if fs:
                wanted = D.FEATURE_SETS[fs]
                qcols = [c for c in wanted if c in d.columns and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().any()]
            for seed in a.seeds:
                for f in range(a.folds):
                    te_m = fold == f
                    Xtr, Xte = build_inputs(d[~te_m], d[te_m], E[~te_m] if E is not None else None,
                                            E[te_m] if E is not None else None, qcols, use_lemon, pca, seed)
                    pred = fit_predict(Xtr, y[~te_m], Xte, a, dev, seed)
                    ok = pd.Series(pred == y[te_m])
                    g = d.loc[te_m, "slide_uid"].reset_index(drop=True)
                    ph = phase[te_m].reset_index(drop=True)
                    for s, v in ok.groupby(g):
                        rows.append(dict(subtype=st, config=name, seed=seed, fold=f, slide_uid=s, n=int(len(v)),
                                         acc=float(v.mean()), phase=ph[g == s].iloc[0]))
                print(f"  {st} {name} seed {seed}: slide-mean "
                      f"{np.mean([r['acc'] for r in rows if r['subtype'] == st and r['config'] == name and r['seed'] == seed]):.4f}",
                      flush=True)
    res = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "cv_results.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved per-slide results -> {out}")
    summarize(res, [s for s in a.subtypes if s in set(res["subtype"])], a.configs)


if __name__ == "__main__":
    main()