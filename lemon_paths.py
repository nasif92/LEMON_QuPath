"""Every path the LEMON + QuPath project uses, in one place.

Training and reference (test) paths are read from /mnt/NAS/MAGEE/MLP_ANN/config.py; the strings in this file are
only fallbacks for names config.py does not define. Also taken from MLP_ANN: nuclei_io.py (GeoJSON + annotation
loading) and, from config.py, FEATURE_SETS and collapse_label - so QuPath features and labels are the ones used before.
"""
import os

# Folder with your existing config.py and nuclei_io.py (appended to sys.path by lemon_data.py).
CODE_DIR = "/mnt/NAS/MAGEE/MLP_ANN"

# Whole-slide images, searched recursively. Add a root if some slides (e.g. reference slides) live elsewhere.
WSI_ROOTS = ["/mnt/NAS/Abhineet/BreastCancerWSIs/OCTOBER-2024",
             "/mnt/NAS/Abhineet/BreastCancerWSIs/ODX-August-Ref",
             "/mnt/NAS/Abhineet/BreastCancerWSIs/reference"]

# ---------------------------------------------------------------- training slides (phases 1-3)
# Read from MLP_ANN/config.py, so a renamed folder there is picked up here automatically.
import sys as _sys
if CODE_DIR not in _sys.path:
    _sys.path.append(CODE_DIR)
import config as _cf  # noqa: E402

TRAIN_FEAT_DIR = _cf.all_features
TRAIN_FEAT_PATTERN = "-feat-no_cells.geojson.gz"
def _first_attr(*names):
    for n in names:
        v = getattr(_cf, n, None)
        if v:
            return v
    raise SystemExit(f"config.py has none of {names} - phase 3 annotations would be silently dropped. Fix lemon_paths.py.")


TRAIN_ANN_DIRS = [_cf.phase1, _cf.phase2,       # searched in order, first match per slide wins
                  _first_attr("ann_ki67_phase3", "ref_set_ann_ki67_phase3"),
                  _first_attr("ann_pr_phase3", "ref_set_ann_pr_phase3"),
                  _first_attr("ann_er_phase3", "ref_set_ann_er_phase3")]

# ---------------------------------------------------------------- reference (test) slides, 6 per stain
REF_FEAT_PATTERN = ".geojson.gz"


def _cfg(name, fallback):
    v = getattr(_cf, name, None)
    return v if v else fallback


# Everything below follows config.py; the strings are only fallbacks for a name config.py does not define.
_QP = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/"
REFERENCE_SETS = {
    "ER": {"detections": _cfg("ref_set_er_all_detections", _QP + "qp-6_reference_slides-no_artifacts-no_cells-er/detections-binary"),
           "annotations": _cfg("ref_set_ann_er", _QP + "qp-6_reference_slides-no_artifacts-no_cells-er/annotations-labeled")},
    "PR": {"detections": _cfg("ref_set_pr_all_detections", _QP + "qp-6_reference_slides-no_artifacts-no_cells/detections-binary"),
           "annotations": _cfg("ref_set_ann_pr", _QP + "qp-6_reference_slides-no_artifacts-no_cells/annotations-labeled")},
    "Ki67": {"detections": _cfg("ref_set_ki67_all_detections", _QP + "qp-6_reference_slides-no_artifacts-no_cells-ki67/detections-binary"),
             "annotations": _cfg("ref_set_ann_ki67", _QP + "qp-6_reference_slides-no_artifacts-no_cells-ki67/annotations-labeled")},
}

# Second test set: every detection on the reference slides labelled by its own classification (Tumor / Non Tumor),
# from the '-labeled-binary' folders (PR: its detections-binary folder). Scored in addition to the annotated nuclei.
def _first_defined(*names):
    for n in names:
        if getattr(_cf, n, None):
            return getattr(_cf, n)
    return None


# PR: its detections-binary folder already carries Tumor / Non-Tumor classifications (collapse_label maps both).
REFERENCE_LABELLED = {"ER": _first_defined("ref_set_er"),
                      "PR": _first_defined("ref_set_pr", "ref_set_pr_all_detections"),
                      "Ki67": _first_defined("ref_set_ki67")}
REFERENCE_LABELLED = {k: v for k, v in REFERENCE_LABELLED.items() if v}

# ---------------------------------------------------------------- this project's own files
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")               # pickled GeoJSON loads (same format as MLP_ANN/cache)
MANIFEST_CSV = os.path.join(PROJECT_DIR, "data", "nuclei_manifest.csv")
LEMON_SOURCE = os.path.join(PROJECT_DIR, "third_party", "LEMON")   # official LEMON code (model.py) + weights
SHARD_DIR = os.path.join(PROJECT_DIR, "embeddings", "shards")      # one .npz per slide, makes extraction resumable
LEMON_NPZ = os.path.join(PROJECT_DIR, "embeddings", "lemon_vits8.npz")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")