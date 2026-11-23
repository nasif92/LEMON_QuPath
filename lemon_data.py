"""Loading nuclei (QuPath features + labels) and joining LEMON embeddings to them.

Nuclei come from your existing nuclei_io.py, labels from config.collapse_label, so the nuclei, labels and
QuPath features are the same ones train_deploy.py used. The pickle cache uses the same key and format as
MLP_ANN's ./cache, so pointing --cache_dir at /mnt/NAS/MAGEE/MLP_ANN/cache reuses pickles you already have.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

import lemon_paths as P

if P.CODE_DIR not in sys.path:
    sys.path.append(P.CODE_DIR)          # appended, so this project's own modules are found first
import config as cf                      # noqa: E402  (MLP_ANN/config.py)
from nuclei_io import LOADER_VERSION, iter_slides, load_slide_nuclei  # noqa: E402

FEATURE_SETS = cf.FEATURE_SETS
DERIVED_ASPECT = "Derived: Aspect ratio"
KEY_COLS = ["subtype", "wsi_name", "det_index"]   # det_index: position in the detection file, fixed before any
                                                  # filtering (nuclei_io says never to join on nucleus_id)


# ------------------------------------------------------------------------------------------- loading
def _load_root(feat_dir, ann_dirs, subtypes, feat_pattern, label, use_classification=False):
    """use_classification=True: no annotations, each detection's own classification is its label."""
    frames = []
    for rec in iter_slides(feat_dir, None if use_classification else ann_dirs, subtypes, feat_pattern,
                           use_classification):
        name, st = rec["slide_name"], rec["subtype"]
        if rec["skip"]:
            print(f"  {name}: {rec['skip']} - skipping")
            continue
        try:
            d, _ = load_slide_nuclei(rec["feat_path"], rec["ann_path"], name)
        except Exception as e:  # one bad file should not stop the run
            print(f"  {name}: FAILED to load ({type(e).__name__}: {e}) - skipping")
            continue
        if d is None or d.empty:
            continue
        d = d.assign(subtype=st, slide_uid=st + ":" + d["wsi_name"].astype(str),
                     ann_source_dir=rec["ann_source_dir"])
        frames.append(d)
        print(f"  {st} ({label}): {name} - {len(d)} nuclei")
    return pd.concat(frames, ignore_index=True) if frames else None


def _newest_mtime(root, suffixes):
    newest = 0.0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(suffixes):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, fn)))
                except OSError:
                    pass
    return newest


def load_dir_cached(feat_dir, ann_dirs, subtypes, feat_pattern, cache_dir, label, use_classification=False):
    """Same cache key / invalidation rule as MLP_ANN/train_deploy.py."""
    ann_dirs = [] if ann_dirs is None else [ann_dirs] if isinstance(ann_dirs, str) else list(ann_dirs)
    if not cache_dir:
        return _load_root(feat_dir, ann_dirs, subtypes, feat_pattern, label, use_classification)
    raw = "|".join([feat_dir, "|".join(sorted(ann_dirs)), ",".join(sorted(subtypes)), feat_pattern,
                    str(bool(use_classification)), f"loader{LOADER_VERSION}"])
    path = os.path.join(cache_dir, hashlib.md5(raw.encode()).hexdigest()[:16] + ".pkl")
    newest = max([_newest_mtime(feat_dir, (feat_pattern,))] +
                 [_newest_mtime(d, (".geojson", ".geojson.gz")) for d in ann_dirs])
    if os.path.exists(path) and os.path.getmtime(path) >= newest:
        print(f"  ({label}) loading from cache: {path}")
        return pd.read_pickle(path)
    df = _load_root(feat_dir, ann_dirs, subtypes, feat_pattern, label, use_classification)
    if df is not None:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_pickle(path)
        print(f"  ({label}) cached -> {path}")
    return df


def check_dirs(dirs, what):
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"{what}: folder(s) not found:\n  " + "\n  ".join(missing)
                         + "\nFix the path in config.py / lemon_paths.py.")


def load_training(subtypes, cache_dir):
    check_dirs([P.TRAIN_FEAT_DIR] + P.TRAIN_ANN_DIRS, "training")
    print("training annotation folders (first match wins):\n  " + "\n  ".join(P.TRAIN_ANN_DIRS))
    df = load_dir_cached(P.TRAIN_FEAT_DIR, P.TRAIN_ANN_DIRS, subtypes, P.TRAIN_FEAT_PATTERN, cache_dir, "train")
    if df is None:
        raise SystemExit("No training nuclei loaded - check TRAIN_FEAT_DIR / TRAIN_ANN_DIRS in lemon_paths.py")
    return df


def load_reference(subtypes, cache_dir):
    print("reference (test) folders:")
    for st in subtypes:
        if st in P.REFERENCE_SETS:
            print(f"  {st}: detections {P.REFERENCE_SETS[st]['detections']}\n        annotations {P.REFERENCE_SETS[st]['annotations']}")
    frames = []
    for st in subtypes:
        ref = P.REFERENCE_SETS.get(st)
        if not ref or not os.path.isdir(ref["detections"]) or not os.path.isdir(ref["annotations"]):
            print(f"  (reference) {st}: reference dirs not found - skipped")
            continue
        d = load_dir_cached(ref["detections"], ref["annotations"], [st], P.REF_FEAT_PATTERN, cache_dir,
                            f"reference: {st}")
        if d is not None and not d.empty:
            frames.append(d)
    if not frames:
        raise SystemExit("No reference nuclei loaded - check REFERENCE_SETS in lemon_paths.py")
    return pd.concat(frames, ignore_index=True)


def load_reference_labelled(subtypes, cache_dir):
    """Reference slides, every detection labelled by its own classification (the '-labeled-binary' folders).
    Returns None when no such folder exists for the requested stains. Rows carry src='lab' so their keys differ
    from the annotated reference nuclei (det_index counts positions in a different file)."""
    frames = []
    for st in subtypes:
        root = P.REFERENCE_LABELLED.get(st)
        if not root or not os.path.isdir(root):
            print(f"  (labelled test) {st}: no labelled-binary folder configured - skipped")
            continue
        d = None
        for pat in (".geojson.gz", ".geojson"):
            d = load_dir_cached(root, [], [st], pat, cache_dir, f"labelled test: {st}", use_classification=True)
            if d is not None and not d.empty:
                break
        if d is not None and not d.empty:
            frames.append(d.assign(src="lab"))
        else:
            print(f"  (labelled test) {st}: no nuclei loaded from {root}")
    return pd.concat(frames, ignore_index=True) if frames else None


# ------------------------------------------------------------------------------------------- labels
def label_and_filter(df, classes, what):
    """collapse_label, then keep only the requested classes. Reference files hold every detection on the
    slide (millions labelled e.g. 'MAIN'); only annotated nuclei survive this step."""
    df = df.copy()
    df["_label"] = df["label"].apply(cf.collapse_label)
    keep = df["_label"].isin(classes)
    n_drop = int((~keep).sum())
    if n_drop:
        top = df.loc[~keep, "label"].value_counts(dropna=False).head(6)
        print(f"  {what}: dropped {n_drop} nuclei not in {classes} ("
              + ", ".join(f"'{k}': {v}" for k, v in top.items()) + ")")
    df = df[keep].reset_index(drop=True)
    add_aspect_ratio(df)
    return df


def add_aspect_ratio(df):
    """Same derived feature as moe.add_derived_features (min / max diameter, 1 = round)."""
    if DERIVED_ASPECT not in df.columns and {"Min diameter µm", "Max diameter µm"} <= set(df.columns):
        mx = df["Max diameter µm"].astype(float)
        df[DERIVED_ASPECT] = (df["Min diameter µm"].astype(float) / mx.where(mx > 0)).clip(0, 1)
    return df


# ------------------------------------------------------------------------------------------- identity
def nucleus_keys(df):
    """'subtype|wsi_name|det_index' - the only link between a nucleus and its embedding (never row order).
    det_index stays the same when annotations change; nucleus_id does not."""
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"nuclei have no {missing} column - cannot link them to embeddings")
    if df[KEY_COLS].isna().any().any():
        raise ValueError("some nuclei have an empty subtype / wsi_name / det_index")
    k = df["subtype"].astype(str) + "|" + df["wsi_name"].astype(str) + "|" + df["det_index"].astype(int).astype(str)
    if "src" in df.columns:     # nuclei from the labelled-binary files: det_index refers to a different file
        k = df["src"].fillna("").astype(str).map(lambda s: s + "|" if s else "") + k
    dup = k.duplicated()
    if dup.any():
        raise ValueError(f"{int(dup.sum())} repeated det_index values within a slide, e.g. {k[dup].head(3).tolist()}")
    return k


class EmbeddingIndex:
    """LEMON embeddings saved by extract_lemon.py, looked up by nucleus key."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.features = z["features"].astype(np.float32)
            self.keys = pd.Index(z["key"].astype(str))
            self.meta = {k: str(z[k]) for k in ("arch", "crop_um", "output_px") if k in z.files}
        if self.features.ndim != 2 or len(self.keys) != len(self.features) or self.keys.has_duplicates:
            raise ValueError(f"bad embedding file {path}: shape {self.features.shape}, {len(self.keys)} keys")
        self.dim = self.features.shape[1]

    def lookup(self, df, what):
        k = nucleus_keys(df)
        pos = self.keys.get_indexer(k)
        if (pos < 0).any():
            raise ValueError(f"{what}: {int((pos < 0).sum())} of {len(pos)} nuclei have no LEMON embedding, "
                             f"e.g. {k[pos < 0].head(3).tolist()}. Rebuild the manifest and re-run extraction.")
        return self.features[pos]