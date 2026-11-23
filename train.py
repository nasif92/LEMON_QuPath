"""Step 3: does LEMON + QuPath beat QuPath alone?

One MLP per run, on one of three inputs, trained on all training slides (phases 1-3) and scored once on the
annotated reference nuclei. The nuclei, labels, MLP and training settings are identical across inputs; only
the input columns change.

    --input qupath   QuPath features from config.FEATURE_SETS[--feature_set]          (the baseline)
    --input lemon    LEMON embeddings only
    --input fusion   both, concatenated

    python train.py --input qupath --subtypes Ki67 --seed 42 --out runs/Ki67_qupath_s42.pt
    python train.py --input fusion --subtypes Ki67 --seed 42 --out runs/Ki67_fusion_s42.pt
    python train.py --input fusion --lemon_pca 32 ...     # LEMON reduced to 32 PCA components (fit on train)

The final epoch is kept (cosine schedule, no epoch selection), so the reference slides never influence the
model. --holdout_frac only prints a held-out-slide score for monitoring.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import lemon_data as D
import lemon_paths as P


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------------- reporting
def report(title, y, pred, classes, sub, slides):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    out = {}
    for st in sorted(set(sub)):
        m = sub == st
        yt, pr = y[m], pred[m]
        print(f"\n=== {title} - {st} ({int(m.sum())} nuclei, {len(set(slides[m]))} slides) ===")
        print(f"acc={np.mean(yt == pr):.4f}  macro_f1={f1_score(yt, pr, average='macro', zero_division=0):.4f}")
        print(classification_report(yt, pr, labels=list(range(len(classes))), target_names=classes, zero_division=0))
        cm = confusion_matrix(yt, pr, labels=list(range(len(classes))))
        print("Confusion (rows=true, cols=pred): " + "  ".join(classes))
        for i, row in enumerate(cm):
            print(f"{classes[i]:>10}: " + " ".join(f"{v:7d}" for v in row))
    ok = pd.Series(pred == y)
    per = pd.DataFrame({"subtype": pd.Series(sub).groupby(slides).first(),
                        "n": ok.groupby(slides).size(), "acc": ok.groupby(slides).mean()})
    print(f"\n{title} - per slide:")
    print(per.sort_values(["subtype", "acc"]).round(3).to_string())
    for st, g in per.groupby("subtype"):
        out[st] = float(g["acc"].mean())
    print("slide-mean acc by subtype: " + ", ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out, per["acc"].round(4).to_dict()


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["qupath", "lemon", "fusion"], required=True)
    ap.add_argument("--subtypes", nargs="+", default=["Ki67"])
    ap.add_argument("--classes", nargs="+", default=["Tumor", "Non Tumor"])
    ap.add_argument("--feature_set", default="all_derived", choices=sorted(D.FEATURE_SETS))
    ap.add_argument("--lemon", default=P.LEMON_NPZ, help="embeddings from extract_lemon.py")
    ap.add_argument("--lemon_pca", type=int, default=0, help="reduce LEMON to this many PCA components (0 = keep all)")
    ap.add_argument("--hidden", nargs="+", type=int, default=[96, 48, 24, 12, 6])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--reweight_power", type=float, default=0.5, help="class sampling weight 1/n_class**p")
    ap.add_argument("--holdout_frac", type=float, default=0.0, help="training slides held out, monitoring only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    ap.add_argument("--cache_dir", default=P.CACHE_DIR)
    ap.add_argument("--no_labelled", action="store_true", help="do not score the labelled-binary test set")
    ap.add_argument("--out", default=None, help="save the final model here (default: do not save)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = list(a.classes)
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"input={a.input}  subtypes={a.subtypes}  feature_set={a.feature_set}  lemon_pca={a.lemon_pca}  "
          f"seed={a.seed}  device={dev}")

    # ------------------------------------------------------------- nuclei
    tr = D.label_and_filter(D.load_training(a.subtypes, a.cache_dir), classes, "train")
    te = D.label_and_filter(D.load_reference(a.subtypes, a.cache_dir), classes, "reference")
    tl = None
    if not a.no_labelled:
        lb = D.load_reference_labelled(a.subtypes, a.cache_dir)
        if lb is not None:
            tl = D.label_and_filter(lb, classes, "labelled test")
    if tl is None or tl.empty:
        tl = te.iloc[0:0].copy()
    overlap = set(tr["wsi_name"]) & (set(te["wsi_name"]) | set(tl["wsi_name"]))
    if overlap:
        raise SystemExit(f"reference slides also in training: {sorted(overlap)[:5]}")
    print("  no slide overlap between training and reference slides")

    hold = np.zeros(len(tr), bool)
    if a.holdout_frac > 0:
        slides = sorted(tr["slide_uid"].unique())
        np.random.RandomState(a.seed).shuffle(slides)
        held = set(slides[:max(1, int(len(slides) * a.holdout_frac))])
        hold = tr["slide_uid"].isin(held).to_numpy()
        print(f"  monitoring holdout: {sorted(held)}")
    va, tr = tr[hold].reset_index(drop=True), tr[~hold].reset_index(drop=True)
    print(f"\ntraining: {len(tr)} nuclei | {tr['slide_uid'].nunique()} slides")
    print(tr.groupby("subtype").agg(nuclei=("_label", "size"), slides=("wsi_name", "nunique"),
                                    tumor_frac=("_label", lambda s: (s == "Tumor").mean())).round(3).to_string())
    print(f"reference (annotated): {len(te)} nuclei | {te['slide_uid'].nunique()} slides")
    print(f"reference (labelled-binary): {len(tl)} nuclei | {tl['slide_uid'].nunique()} slides")

    # ------------------------------------------------------------- inputs (all statistics from training rows)
    blocks = {"tr": [], "va": [], "te": [], "tl": []}
    info = {}
    if a.input in ("qupath", "fusion"):
        wanted = D.FEATURE_SETS[a.feature_set]
        cols = [c for c in wanted if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c]) and tr[c].notna().any()]
        if len(cols) < len(wanted):
            print(f"  WARNING: {len(wanted) - len(cols)} feature(s) of {a.feature_set} not in the data: "
                  f"{[c for c in wanted if c not in cols][:6]}")
        for nm, dd in (("reference (annotated)", te), ("reference (labelled-binary)", tl)):
            if len(dd):
                miss = float(dd.reindex(columns=cols).isna().to_numpy().mean())
                print(f"  {nm}: {miss:.1%} of the QuPath feature values are missing")
                if miss > 0.2:
                    raise SystemExit(f"{nm} nuclei lack most QuPath features ({miss:.0%} missing) - those files are not "
                                     f"feature exports; check the folder in config.py")
        med = tr[cols].median()
        X = {k: d.reindex(columns=cols).astype(np.float64).fillna(med).to_numpy()
             for k, d in (("tr", tr), ("va", va), ("te", te), ("tl", tl))}
        mu, sd = X["tr"].mean(0), X["tr"].std(0); sd[sd == 0] = 1
        for k in X:
            blocks[k].append(np.clip((X[k] - mu) / sd, -20, 20))
        info.update(qupath_cols=cols, qupath_medians=med.to_numpy(), qupath_mu=mu, qupath_sd=sd)
        print(f"  QuPath: {len(cols)} features ({a.feature_set})")
    if a.input in ("lemon", "fusion"):
        idx = D.EmbeddingIndex(a.lemon)
        E = {k: idx.lookup(d, k).astype(np.float64) if len(d) else np.zeros((0, idx.dim))
             for k, d in (("tr", tr), ("va", va), ("te", te), ("tl", tl))}
        if a.lemon_pca:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=a.lemon_pca, random_state=a.seed).fit(E["tr"])
            E = {k: pca.transform(v) if len(v) else np.zeros((0, a.lemon_pca)) for k, v in E.items()}
            info.update(pca_components=pca.components_, pca_mean=pca.mean_)
            print(f"  LEMON PCA {idx.dim} -> {a.lemon_pca} (explained variance {pca.explained_variance_ratio_.sum():.3f})")
        mu, sd = E["tr"].mean(0), E["tr"].std(0); sd[sd == 0] = 1
        for k in E:
            blocks[k].append(np.clip((E[k] - mu) / sd, -20, 20))
        info.update(lemon_meta=idx.meta, lemon_mu=mu, lemon_sd=sd)
        print(f"  LEMON: {E['tr'].shape[1]} dims ({idx.meta})")

    to_t = lambda b: torch.from_numpy(np.concatenate(b, 1).astype(np.float32)).to(dev)
    Xtr, Xte = to_t(blocks["tr"]), to_t(blocks["te"])
    Xtl = to_t(blocks["tl"]) if len(tl) else None
    Xva = to_t(blocks["va"]) if len(va) else None
    ytr = torch.as_tensor(tr["_label"].map(c2i).to_numpy(), device=dev)
    yte = te["_label"].map(c2i).to_numpy()
    ytl = tl["_label"].map(c2i).to_numpy()
    print(f"  model input: {Xtr.shape[1]} columns")

    # ------------------------------------------------------------- train
    model = MLP(Xtr.shape[1], len(classes), a.hidden, a.dropout).to(dev)
    print(f"\nMLP {a.hidden}: {sum(p.numel() for p in model.parameters()):,} parameters")
    cnt = Counter(ytr.tolist())
    w = torch.as_tensor([1.0 / cnt[int(t)] ** a.reweight_power for t in ytr.tolist()], dtype=torch.double, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    steps = max(1, len(ytr) // a.batch_size)

    def predict(X):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(X[i:i + 50000]) for i in range(0, len(X), 50000)])

    for ep in range(1, a.epochs + 1):
        model.train()
        for _ in range(steps):
            i = torch.multinomial(w, a.batch_size, replacement=True)
            opt.zero_grad()
            loss = crit(model(Xtr[i]), ytr[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if ep == 1 or ep % 20 == 0 or ep == a.epochs:
            lg = predict(Xtr)
            msg = (f"  epoch {ep:03d}  train_loss={crit(lg, ytr).item():.4f}  "
                   f"train_acc={(lg.argmax(1) == ytr).float().mean().item():.4f}")
            if Xva is not None:
                yv = torch.as_tensor(va["_label"].map(c2i).to_numpy(), device=dev)
                msg += f"  holdout_acc={(predict(Xva).argmax(1) == yv).float().mean().item():.4f}"
            print(msg + f"  lr={opt.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------- evaluate + save
    if Xva is not None:
        report("HOLDOUT (monitoring)", va["_label"].map(c2i).to_numpy(), predict(Xva).argmax(1).cpu().numpy(),
               classes, va["subtype"].to_numpy(), va["slide_uid"].to_numpy())
    pred = predict(Xte).argmax(1).cpu().numpy()
    res, per_slide = report("EXTERNAL TEST", yte, pred, classes, te["subtype"].to_numpy(), te["slide_uid"].to_numpy())

    res_l, per_slide_l = {}, {}
    if Xtl is not None:
        res_l, per_slide_l = report("EXTERNAL TEST (labelled-binary)", ytl, predict(Xtl).argmax(1).cpu().numpy(), classes,
                                    tl["subtype"].to_numpy(), tl["slide_uid"].to_numpy())

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save({"model_state": model.state_dict(), "args": vars(a), "classes": classes,
                    "n_inputs": Xtr.shape[1], **info}, a.out)
        print(f"\nSaved final-epoch model -> {a.out}")
    print("RESULT " + json.dumps({"input": a.input, "feature_set": a.feature_set if a.input != "lemon" else None,
                                  "lemon_pca": a.lemon_pca if a.input != "qupath" else None, "seed": a.seed,
                                  "n_inputs": int(Xtr.shape[1]), "slide_mean": res, "per_slide": per_slide,
                                  "slide_mean_labelled": res_l, "per_slide_labelled": per_slide_l,
                                  "out": a.out}))


if __name__ == "__main__":
    main()