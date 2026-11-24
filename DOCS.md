# LEMON + QuPath vs QuPath alone

Question: does adding frozen LEMON nucleus embeddings to the 108 QuPath features improve Tumor vs Non Tumor
classification of nuclei in ER, PR and Ki67 IHC breast slides?

Standalone project. From `/mnt/NAS/MAGEE/MLP_ANN` it uses only `nuclei_io.py` (GeoJSON and annotation loading)
and, from `config.py`, `FEATURE_SETS`, `collapse_label` and the folder names. Nuclei, labels and QuPath features
are therefore the same as in the earlier MLP runs.

## Pipeline (run in this order)

| Step | Command | Output |
|---|---|---|
| 1. List nuclei | `python prepare_manifest.py --no_labelled` | `data/nuclei_manifest.csv` |
| 2. Check crops | `python extract_lemon.py --check 60 [--crop_um 50]` | `data/crop_check.png` |
| 3. Embed | `python extract_lemon.py --workers 8 [--crop_um 50]` | `embeddings/lemon_vits8*.npz` |
| 4. Cross-validate | `python cv.py --subtypes Ki67 PR ER --folds 5 --seeds 42 43 44` | `cv_runs/cv_results.csv` + tables |
| 5. Score reference slides (once) | `python train.py ... --no_labelled` | `RESULT {...}` line in the log |
| 6. Summarise | `python summarize.py runs/*.log`, `python compare_slides.py runs/*.log` | tables |

Training slides are annotation phases 1 to 3. The reference slides are test only: never used for training,
tuning or choosing a setting. Choose settings with `cv.py`, then score the reference slides once.

## Files

### `lemon_paths.py`
Every path in one place. Training and reference folders are read from `MLP_ANN/config.py`, so a renamed folder
there is picked up here. Defines `CODE_DIR`, `WSI_ROOTS`, `TRAIN_FEAT_DIR`, `TRAIN_ANN_DIRS` (phase 1, 2, 3 for
each stain; first match per slide wins), `REFERENCE_SETS` (annotated reference nuclei), `REFERENCE_LABELLED`
(labelled-binary test files), `CACHE_DIR`, `MANIFEST_CSV`, `LEMON_SOURCE`, `SHARD_DIR`, `LEMON_NPZ`.
Helpers `_first_attr`, `_cfg`, `_first_defined` take the first config name that exists and raise if none does,
so a renamed phase folder cannot be dropped silently.

### `lemon_data.py`
Loads nuclei and joins embeddings to them.
- `load_training`, `load_reference`, `load_reference_labelled`: read QuPath features and labels through
  `nuclei_io`, with a pickle cache (rejected if any source file is newer than the pickle).
  `load_reference_labelled` uses each detection's own classification (the labelled-binary files) and tags
  those nuclei `src='lab'`.
- `check_dirs`: stops with a clear message if a configured folder is missing.
- `label_and_filter`: applies `collapse_label` and keeps only Tumor / Non Tumor (Ghost, MIXED, Normal
  Epithelium and Artifact are dropped).
- `add_aspect_ratio`: adds `Derived: Aspect ratio` (the 108th feature).
- `nucleus_keys`: the key `subtype|wsi_name|det_index` (prefixed `lab|` for labelled-binary nuclei).
  `det_index` is the position in the detection file; never join on `nucleus_id`.
- `EmbeddingIndex(path)`: loads embeddings and looks them up by key, never by row order; raises if any
  nucleus has no embedding. Accepts several files joined by commas (`a.npz,b.npz`): features are aligned on
  the first file's keys and concatenated (the two-scale input).

### `prepare_manifest.py` (step 1)
Writes one row per labelled nucleus that needs an embedding: key, slide path, level-0 centroid (`x`, `y`).
Finds each slide file by name under `WSI_ROOTS` (skips mask/thumbnail/label folders; accepts identical-size
duplicates), refuses a slide that appears in both training and reference, and stops if any nucleus has no
centroid. `--no_labelled` skips the 2.76 million labelled-binary nuclei (they are not embedded unless you want them).

### `extract_lemon.py` (step 3)
Reads a square crop of `--crop_um` (default 10) around each centroid straight from the slide with OpenSlide,
resizes to 40 x 40 and runs LEMON ViT-S/8 (frozen, 384 dimensions). No tiling.
- Slide resolution comes from the slide's MPP metadata (0.263 um/px for our 40x slides); `--mpp` overrides.
- One shard per slide (resumable); `--merge_only` rebuilds the merged file from shards.
- Non-default `--arch`, `--crop_um` or `--output_px` get their own shards and output name
  (`lemon_vits8_50um_40px.npz`), so sizes never overwrite each other.
- `--check N` saves a grid of N crops for a visual check and exits.
- Needs `model.py`, `mean_std.json`, `lemon_vits8.pth.tar` in `third_party/LEMON`
  (huggingface.co/aliceblondel/LEMON); the script changes into that folder to load them.

### `train.py` (step 5)
Trains one MLP on `--input qupath | lemon | fusion` using all training slides, scores the annotated reference
nuclei (and the labelled-binary set unless `--no_labelled`), prints per-class metrics, per-slide accuracy and a
`RESULT {...}` json line. The final epoch is kept (cosine schedule, no epoch selection), so the reference slides
never influence the model. Models are not saved unless `--out` is given.
- MLP: hidden 96-48-24-12-6, BatchNorm, dropout 0.5, lr 1e-4, weight decay 1e-3, batch 10,000, 200 epochs,
  class sampling weight 1/n^0.5.
- `--feature_set`: `all_derived` (108), `context_only_derived` (44, neighbourhood), `no_context_derived`
  (64, own-nucleus).
- `--lemon FILE[,FILE2]`: embedding file(s); two files give the two-scale input.
- `--lemon_pca K`: reduce embeddings to K components (fit on training only).
- Guard: stops if more than 20% of QuPath feature values are missing.
- Standardisation and median fill use training rows only.

### `cv.py` (step 4)
Slide-grouped k-fold cross-validation on the training slides only (reference slides are never loaded). Every
configuration sees the same folds, so differences are paired. Configurations (`CONFIGS`): `qupath_all`,
`lemon`, `fusion_all`, `fusion_all_pca32`, `qupath_context`, `fusion_context`, `qupath_nocontext`.
- Prints slide-mean accuracy per stain, paired differences (`PAIRS`: fusion_all vs qupath_all, fusion_context
  vs qupath_context, fusion_context vs qupath_all, lemon vs qupath_nocontext; with slide wins and a t-test),
  and accuracy by annotation phase of the held-out slide.
- Writes `cv_results.csv` (one row per held-out slide, config and seed) to `--out_dir`. Use a new `--out_dir`
  for each run; reusing one overwrites it.
- Options: `--lemon FILE[,FILE2]`, `--configs`, `--folds`, `--fold_seed`, `--seeds`.

### `run_experiments.sh`
Runs the seven configurations for one stain on the reference slides with seeds 42, 43, 44, writing
`runs/<stain>_<name>_s<seed>.log`; skips finished logs, then runs `summarize.py`. Uses `--no_labelled`
unless `LABELLED=1`. It trains on the default embedding file; pass the two-scale file through `train.py`
directly (the script has no `--lemon` option).

### `summarize.py`
Table of mean +- SD over seeds from the `RESULT` lines of the given logs (annotated reference nuclei, and
labelled-binary if present). Configurations are named `input | feature_set [| pcaK]`.

### `compare_slides.py`
Paired per-slide comparison of two configurations, averaged over seeds. Logs are positional arguments;
`--a` and `--b` name the configurations (defaults: `fusion | all_derived` vs `qupath | all_derived`);
`--set annotated|labelled`. Prints accuracy per slide for A and B, the difference, slides A wins, and a paired
t-test. Do not pass logs from two fusion variants together: they share a config name and would be averaged.

## Metric
Slide-mean accuracy: the mean over slides of each slide's accuracy (then over seeds). Slides from the same case
are not independent, so p-values are a rough guide.

## Things to remember
- Config names match what `summarize.py` prints, so `fusion | all_derived` covers any crop size; keep logs for
  different crop settings in separate files and compare them one pair at a time.
- A changed annotation folder changes the cache key and triggers a reload; rerun `prepare_manifest.py` and
  `extract_lemon.py` after it, because the manifest lists the labelled nuclei.
- Embeddings are tied to nuclei by key and centroid. A check on Ki67 training nuclei found a maximum centroid
  difference of 3e-11 px; PR, ER and the reference nuclei were not yet run.
