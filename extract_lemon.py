"""Step 2: LEMON embedding for every nucleus in the manifest. No tiling: each nucleus is a 10 x 10 um crop
read straight from the WSI around its centroid and resized to 40 x 40 px (LEMON's input: 40 px at 40x).

    python extract_lemon.py --check 60        # save a grid of 60 random crops to data/crop_check.png, no model
    python extract_lemon.py                   # full run, one shard per slide (resumable), then merge
    python extract_lemon.py --merge_only      # rebuild the merged file from existing shards

Needs: openslide-python, torch, timm, and these files from huggingface.co/aliceblondel/LEMON in
third_party/LEMON: model.py, mean_std.json, lemon_vits8.pth.tar (MIT licence per the model card).
"""
import argparse
from pathlib import Path
import contextlib
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

import lemon_paths as P


def slide_mpp(slide, override):
    if override:
        return override, override
    mx, my = (slide.properties.get(k) for k in ("openslide.mpp-x", "openslide.mpp-y"))
    try:
        mx, my = float(mx), float(my)
    except (TypeError, ValueError):
        mx = my = float("nan")
    if not (np.isfinite(mx) and np.isfinite(my) and mx > 0 and my > 0):
        raise ValueError("slide has no openslide.mpp-x/y; pass --mpp (um per level-0 pixel)")
    return mx, my


class Crops(Dataset):
    """Nuclei of ONE slide. The OpenSlide handle is opened lazily, so each DataLoader worker has its own."""

    def __init__(self, wsi_path, xy, crop_um, output_px, mpp=None):
        self.wsi_path, self.xy = wsi_path, xy
        self.crop_um, self.output_px, self.mpp = float(crop_um), int(output_px), mpp
        self.slide = self.size = None

    def __len__(self):
        return len(self.xy)

    def _open(self):
        import openslide
        self.slide = openslide.OpenSlide(self.wsi_path)
        mx, my = slide_mpp(self.slide, self.mpp)
        self.size = (max(1, round(self.crop_um / mx)), max(1, round(self.crop_um / my)))

    def __getitem__(self, i):
        if self.slide is None:
            self._open()
        x, y = self.xy[i]
        w, h = self.size
        im = self.slide.read_region((int(round(x - w / 2)), int(round(y - h / 2))), 0, (w, h)).convert("RGB")
        if im.size != (self.output_px, self.output_px):
            im = im.resize((self.output_px, self.output_px), Image.Resampling.BICUBIC)
        return im, i


def check_crops(df, n, a):
    rows = df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)
    tiles = []
    for wsi, g in rows.groupby("wsi_path"):
        ds = Crops(wsi, g[["x", "y"]].to_numpy(), a.crop_um, a.output_px, a.mpp)
        for j in range(len(ds)):
            tiles.append((ds[j][0], g["label"].iloc[j]))
        print(f"  {os.path.basename(wsi)}: crop {ds.size[0]}x{ds.size[1]} level-0 px -> {a.output_px}px")
    s, cols = a.output_px * 4, 10
    grid = Image.new("RGB", (cols * s, ((len(tiles) + cols - 1) // cols) * (s + 14)), "white")
    draw = ImageDraw.Draw(grid)
    for k, (im, lab) in enumerate(tiles):
        x0, y0 = (k % cols) * s, (k // cols) * (s + 14)
        grid.paste(im.resize((s, s), Image.Resampling.NEAREST), (x0, y0))
        draw.text((x0 + 2, y0 + s), str(lab), fill="black")
    out = os.path.join(os.path.dirname(P.MANIFEST_CSV), "crop_check.png")
    grid.save(out)
    print(f"Saved {len(tiles)} crops -> {out}\nEach crop should show ONE nucleus in the centre. If they show "
          f"background or are offset, cx_wsi/cy_wsi are not level-0 pixels; stop and fix that first.")


def load_model(a):
    """LEMON's model.py (huggingface.co/aliceblondel/LEMON): load_lemon_moco_model(arch) -> (model, dtype, transform).
    It reads mean_std.json and lemon_<arch>.pth.tar from the CURRENT directory, so we cd into --lemon_source."""
    src = Path(a.lemon_source).resolve()
    need = ["model.py", "mean_std.json", f"lemon_{a.arch}.pth.tar"]
    missing = [f for f in need if not (src / f).exists() or (src / f).stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing or empty in {src}: {missing}\nDownload them from https://huggingface.co/aliceblondel/LEMON")
    sys.path.insert(0, str(src))
    try:
        from model import load_lemon_moco_model
    except ImportError as e:
        raise SystemExit(f"Could not import the LEMON model.py from {src}: {e}\n(pip install timm torchvision if a module is missing)")
    cwd = os.getcwd()
    os.chdir(src)
    try:
        model, dtype, transform = load_lemon_moco_model(a.arch)
    finally:
        os.chdir(cwd)
    return model, dtype, transform


def setting_tag(a):
    """Empty for the baseline (vits8, 10 um, 40 px) so old shards and file names stay valid; otherwise names the setting."""
    if (a.arch, float(a.crop_um), int(a.output_px)) == ("vits8", 10.0, 40):
        return ""
    return f"_{a.arch}_{a.crop_um:g}um_{a.output_px}px"


def shard_path(wsi_path, keys, a):
    tag = hashlib.md5(("|".join(keys)).encode()).hexdigest()[:10]
    return os.path.join(P.SHARD_DIR, f"{os.path.splitext(os.path.basename(wsi_path))[0]}_{tag}{setting_tag(a)}.npz")


def merge(df, a):
    feats, keys = [], []
    for wsi, g in df.groupby("wsi_path", sort=False):
        sp = shard_path(wsi, g["key"].tolist(), a)
        if not os.path.exists(sp):
            raise SystemExit(f"missing shard for {wsi} - run without --merge_only first")
        with np.load(sp, allow_pickle=False) as z:
            feats.append(z["features"]); keys.append(z["key"])
    feats, keys = np.concatenate(feats), np.concatenate(keys)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, features=feats, key=keys, arch=a.arch, crop_um=a.crop_um, output_px=a.output_px)
    print(f"Saved {feats.shape[0]:,} x {feats.shape[1]} embeddings -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=P.MANIFEST_CSV)
    ap.add_argument("--out", default=None, help="default: embeddings/lemon_vits8.npz, or lemon_<arch>_<crop>um_<px>px.npz for other settings")
    ap.add_argument("--lemon_source", default=P.LEMON_SOURCE)
    ap.add_argument("--arch", default="vits8", choices=["vits8", "vitb8"])
    ap.add_argument("--crop_um", type=float, default=10.0, help="physical crop size (LEMON: 40 px at 0.25 um/px)")
    ap.add_argument("--output_px", type=int, default=40)
    ap.add_argument("--mpp", type=float, default=None, help="override um/px for slides without MPP metadata")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check", type=int, default=0, help="save N random crops for a visual check and exit")
    ap.add_argument("--merge_only", action="store_true")
    a = ap.parse_args()
    if a.out is None:
        a.out = P.LEMON_NPZ if not setting_tag(a) else P.LEMON_NPZ.replace(".npz", "") .replace("lemon_vits8", "lemon") + setting_tag(a) + ".npz"

    df = pd.read_csv(a.manifest, dtype={"key": str})
    print(f"{len(df):,} nuclei, {df['wsi_path'].nunique()} slides in {a.manifest}")
    if a.check:
        return check_crops(df, a.check, a)
    if a.merge_only:
        return merge(df, a)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    model, dtype, transform = load_model(a)
    model = model.to(device).eval()
    amp = (torch.autocast("cuda", dtype=dtype) if device.startswith("cuda") and dtype in (torch.float16, torch.bfloat16)
           else contextlib.nullcontext())

    def collate(batch):   # runs inside the workers, so the transform is parallel too
        return torch.stack([transform(im) for im, _ in batch]), np.asarray([i for _, i in batch])

    os.makedirs(P.SHARD_DIR, exist_ok=True)
    groups = list(df.groupby("wsi_path", sort=False))
    for n, (wsi, g) in enumerate(groups, 1):
        sp = shard_path(wsi, g["key"].tolist(), a)
        if os.path.exists(sp):
            print(f"[{n}/{len(groups)}] {os.path.basename(wsi)}: done, skipping")
            continue
        ds = Crops(wsi, g[["x", "y"]].to_numpy(), a.crop_um, a.output_px, a.mpp)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, collate_fn=collate)
        out = None
        with torch.inference_mode(), amp:
            for images, idx in dl:
                f = model(images.to(device)).float().cpu().numpy()
                if f.ndim != 2 or f.shape[0] != len(idx):
                    raise SystemExit(f"model returned shape {f.shape} for {len(idx)} crops - expected (N, dim)")
                if out is None:
                    out = np.empty((len(ds), f.shape[1]), np.float32)
                out[idx] = f
        np.savez(sp + ".tmp.npz", features=out, key=g["key"].to_numpy().astype(str))
        os.replace(sp + ".tmp.npz", sp)       # a crash never leaves a half-written shard
        print(f"[{n}/{len(groups)}] {os.path.basename(wsi)}: {len(ds):,} nuclei -> {out.shape[1]}-d")
    merge(df, a)


if __name__ == "__main__":
    main()