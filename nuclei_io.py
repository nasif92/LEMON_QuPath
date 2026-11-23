"""
Shared nucleus loading for train_deploy.py and extract_crops.py.

Both scripts MUST select the same nuclei, with the same labels and centroids,
or CNN crops would silently attach to the wrong rows of handcrafted features.
So the selection logic lives here once, instead of being copied into each.

Per slide:
  1. read the detection GeoJSON, keep only objectType == "detection"
  2. centroid = mean of the polygon's outer-ring vertices
  3. drop detections outside the bbox of all annotation regions
  4. label by point-in-annotation (extract_features.assign_labels), or by the
     detection's own classification when no annotations are used
  5. drop unlabelled nuclei

Join keys added to every row:
  det_index - position of the detection among the file's detections, recorded
              BEFORE the bbox/label filters, so it doesn't shift when
              annotations change. Stable as long as the detection GeoJSON
              itself isn't regenerated.
  det_uuid  - the QuPath Feature "id" where present, "" otherwise. Used as a
              cross-check, since not every export carries one.
(nucleus_id is kept for backward compatibility, but it's a running count
assigned AFTER filtering - never join on it.)
"""
import gzip
import json
import os

import numpy as np
import pandas as pd

# Bump when the loader's output columns or selection logic change, so
# train_deploy.py's on-disk cache is invalidated automatically.
LOADER_VERSION = "2"

ID_COLS = ("wsi_name", "nucleus_id", "det_index", "det_uuid", "cx_wsi", "cy_wsi")


def infer_subtype_from_name(slide_name, subtypes):
    names = f"_{slide_name}_".lower()
    for st in subtypes:
        if f"_{st}_".lower() in names:
            return st
    return None


def load_geojson_detections(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"] if isinstance(data, dict) and "features" in data else (
        data if isinstance(data, list) else []
    )
    # Some exports mix annotation-type objects into the detection/feature
    # file alongside real nuclei. Only real detections have genuine
    # per-nucleus measurements - filter strictly so an annotation shape
    # can never masquerade as a nucleus downstream.
    return [f for f in feats
            if (f.get("properties", {}) or {}).get("objectType") == "detection"]


def _centroid_of_ring(ring):
    return np.asarray(ring, dtype=float).mean(axis=0)


def polygon_centroid(geometry):
    """Centroid of a GeoJSON geometry dict (Point / Polygon / MultiPolygon).
    Vertex mean - kept identical to the original train_deploy.py behaviour."""
    gtype = geometry.get("type")
    coords = geometry["coordinates"]
    if gtype == "Point":
        return np.asarray(coords, dtype=float)
    if gtype == "Polygon":
        return _centroid_of_ring(coords[0])
    if gtype == "MultiPolygon":
        rings = [poly[0] for poly in coords if poly]
        pts = [_centroid_of_ring(r) for r in rings if r]
        return np.mean(pts, axis=0) if pts else np.array([np.nan, np.nan])
    return np.array([np.nan, np.nan])


def find_annotation_file(ann_dir, slide_name):
    """Recursively search ann_dir for this slide's annotations file.
    ann_dir may be a single directory or a list of directories - searched
    in order, first match wins (lets --ann_dir combine multiple sources,
    e.g. phase1_annotations + phase2_annotations, in one training run)."""
    candidates = [f"{slide_name}.geojson", f"{slide_name}.geojson.gz",
                  f"{slide_name}_annotations.geojson", f"{slide_name}_annotations.geojson.gz"]
    ann_dirs = [ann_dir] if isinstance(ann_dir, str) else list(ann_dir)
    for d in ann_dirs:
        for dirpath, _, filenames in os.walk(d):
            for fn in filenames:
                if fn in candidates:
                    return os.path.join(dirpath, fn), d
    return None, None


def labels_from_classification(cell_features):
    """Use each detection's own classification as its label."""
    labels = []
    for f in cell_features:
        cls = f["properties"].get("classification")
        name = cls.get("name") if isinstance(cls, dict) else None
        labels.append(name if name and name.lower() != "nucleus" else None)
    return labels


def iter_slides(feat_dir, ann_dir, subtypes, feat_pattern=".geojson.gz",
                use_classification=False):
    """Yield one dict per slide under feat_dir:
    {slide_name, subtype, feat_path, ann_path, ann_source_dir, skip}.
    `skip` is None, or a short reason string when the slide can't be used
    (callers print it). Slides with no recognised subtype token are not
    yielded at all, matching the original train_deploy.py behaviour."""
    for dirpath, _, filenames in os.walk(feat_dir):
        for fn in filenames:  # os.walk order, same as the original loop (keeps seeded splits identical)
            if not fn.endswith(feat_pattern):
                continue
            slide_name = fn[: -len(feat_pattern)]
            st = infer_subtype_from_name(slide_name, subtypes)
            if st is None:
                continue
            rec = {"slide_name": slide_name, "subtype": st,
                   "feat_path": os.path.join(dirpath, fn),
                   "ann_path": None, "ann_source_dir": None, "skip": None}
            if not use_classification:
                ann_path, matched = (find_annotation_file(ann_dir, slide_name)
                                     if ann_dir else (None, None))
                if ann_path is None:
                    rec["skip"] = f"no annotations found under {ann_dir}"
                rec["ann_path"], rec["ann_source_dir"] = ann_path, matched
            yield rec


def load_slide_nuclei(feat_path, ann_path, slide_name, with_geometry=False,
                      verbose=True):
    """Load one slide's labelled nuclei.

    Returns (df, geometries): df has ID_COLS + label + is_ground_truth + one
    column per measurement; geometries is the list of GeoJSON geometry dicts
    in the same row order (None unless with_geometry=True). Returns
    (None, None) when nothing survives."""
    cell_features = load_geojson_detections(feat_path)
    if not cell_features:
        return None, None

    det_index = np.arange(len(cell_features))
    centroids = np.array([polygon_centroid(f["geometry"]) for f in cell_features])

    if ann_path is not None:
        import extract_features as ef

        # Reject detections outside the union bbox of all annotation regions
        # before anything else - on a whole-slide feature file this is the
        # difference between hundreds of thousands of detections and the few
        # thousand actually inside annotated regions.
        ann_bbox = ef.annotation_bbox(ann_path)
        if ann_bbox is not None:
            x0, x1, y0, y1 = ann_bbox
            in_bbox = ((centroids[:, 0] >= x0) & (centroids[:, 0] <= x1) &
                       (centroids[:, 1] >= y0) & (centroids[:, 1] <= y1))
            n_before = len(cell_features)
            cell_features = [f for f, keep in zip(cell_features, in_bbox) if keep]
            centroids = centroids[in_bbox]
            det_index = det_index[in_bbox]
            if verbose:
                print(f"    {slide_name}: {n_before} detections -> "
                      f"{len(cell_features)} inside annotation bbox")
            if not cell_features:
                return None, None

        labels = ef.assign_labels(centroids, ann_path, verbose=False)
    else:
        labels = labels_from_classification(cell_features)

    keep = [i for i, label in enumerate(labels) if label is not None]
    cell_features = [cell_features[i] for i in keep]
    centroids = centroids[keep]
    det_index = det_index[keep]
    labels = [labels[i] for i in keep]
    if not cell_features:
        return None, None

    n = len(cell_features)
    all_names = sorted({k for f in cell_features
                        for k in f["properties"].get("measurements", {})})

    # Columnar construction (fast) instead of a list of per-nucleus dicts.
    m_arrays = {name: np.full(n, np.nan, dtype=np.float64) for name in all_names}
    for i, f in enumerate(cell_features):
        for name, val in f["properties"].get("measurements", {}).items():
            if val is not None:
                m_arrays[name][i] = val

    data = {
        "wsi_name": slide_name,
        "nucleus_id": np.arange(n),
        "det_index": det_index.astype(np.int64),
        "det_uuid": [str(f.get("id") or "") for f in cell_features],
        "cx_wsi": centroids[:, 0].astype(np.float64),
        "cy_wsi": centroids[:, 1].astype(np.float64),
        "label": labels,
        "is_ground_truth": [l is not None for l in labels],
    }
    data.update(m_arrays)
    geoms = [f["geometry"] for f in cell_features] if with_geometry else None
    return pd.DataFrame(data), geoms
