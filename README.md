# LEMON + QuPath vs QuPath alone

Question: does adding LEMON nucleus embeddings to the 108 QuPath features improve Tumor vs Non Tumor
classification on the reference slides?

Standalone project. It borrows only two things from `/mnt/NAS/MAGEE/MLP_ANN`:
`nuclei_io.py` (GeoJSON + annotation loading) and, from `config.py`, `FEATURE_SETS` and `collapse_label`.
So nuclei, labels and QuPath features are identical to the earlier MLP runs. Your `train_deploy.py` is not used.

## Files

| File | What it does |
|---|---|
| `lemon_paths.py` | Every path (training phases 1-3, reference sets, WSIs, outputs). Edit here only. |
| `lemon_data.py` | Loads nuclei, applies labels, adds `Derived: Aspect ratio`, joins embeddings by `subtype\|wsi_name\|nucleus_id` |
| `prepare_manifest.py` | Step 1: list every labelled nucleus (train + reference) with its WSI file and centroid |
| `extract_lemon.py` | Step 2: crop 10 x 10 um around each nucleus straight from the WSI, embed with LEMON (no tiling) |
| `train.py` | Step 3: one MLP on `qupath`, `lemon` or `fusion` input; scores the reference slides once |
| `run_experiments.sh` | All comparisons, 3 seeds each |
| `summarize.py` | Mean +- SD table from the logs |

## Setup

```bash
mkdir -p /mnt/NAS/MAGEE/LEMON_QuPath && cp *.py *.sh README.md /mnt/NAS/MAGEE/LEMON_QuPath/
cd /mnt/NAS/MAGEE/LEMON_QuPath
pip install openslide-python torch torchvision scikit-learn pandas pillow
# official LEMON code + weights (model.py must provide load_lemon_moco_model(arch) -> (model, dtype, transform)):
git clone https://huggingface.co/aliceblondel/LEMON third_party/LEMON
```

Check `third_party/LEMON/model.py` really has `load_lemon_moco_model` with that return value; if the
function is named differently, change `load_model()` in `extract_lemon.py`.

## Run

```bash
python prepare_manifest.py                 # ER, PR, Ki67 (or --subtypes Ki67 for a faster first try)
python extract_lemon.py --check 60         # LOOK at data/crop_check.png: one nucleus centred in each crop
python extract_lemon.py --workers 8        # full extraction; resumable (one shard per slide)
bash run_experiments.sh Ki67               # 6 configurations x 3 seeds, then the summary table
```

Optional: `--cache_dir /mnt/NAS/MAGEE/MLP_ANN/cache` on `prepare_manifest.py` and `train.py` reuses the
pickles you already have (same key and format), which skips re-reading the GeoJSONs.

## Configurations in `run_experiments.sh`

| Name | Input | Compare with |
|---|---|---|
| `qupath_all` | 108 QuPath features | baseline (0.837 on Ki67 earlier) |
| `lemon` | LEMON only (384) | shape 0.755, intensity 0.745 |
| `fusion_all` | 108 + 384 | `qupath_all` - the main question |
| `fusion_all_pca32` | 108 + LEMON reduced to 32 PCA components (fit on training) | `fusion_all` |
| `qupath_context` | 44 neighbourhood features | 0.814 earlier |
| `fusion_context` | 44 + 384 | `qupath_all`: can LEMON replace the nucleus-level features? |

Same nuclei, MLP (96-48-24-12-6, dropout 0.5), sampling, cosine schedule and 200 epochs for every
configuration. The final epoch is kept, so the reference slides never influence training.
Seed spread was 0.2-0.9 points before; treat differences under ~1 point as noise.

## What was fixed compared with the first version

- Phase 3 (Ki67 and PR) is included; the old manifest used phases 1-2 only, so training would have failed.
- The manifest keeps only labelled nuclei. Reference files hold ~2.7M unannotated detections per stain
  ('MAIN'); embedding those would have taken days for nothing.
- `Derived: Aspect ratio` is added, so `all_derived` is the same 108 features as before.
- No epoch selection on the reference slides (the old default `--val_source test` did that).
- Extraction is resumable, uses parallel workers, and has a `--check` mode to confirm coordinates first.
- Standalone: does not depend on any version of `train_deploy.py`.

## Caveats

- LEMON was trained and evaluated on H&E only; these slides are DAB IHC. Transfer is untested.
- `cx_wsi`/`cy_wsi` must be level-0 pixel coordinates. `--check` is how you confirm it.
- Slides need MPP metadata (or pass `--mpp`). At 0.25 um/px a 10 um crop is exactly 40 px.
- Check the LEMON weights' licence before publishing.
- `train.py` needs nothing on the GPU beyond ~1 GB for ~400k nuclei x 492 columns.
