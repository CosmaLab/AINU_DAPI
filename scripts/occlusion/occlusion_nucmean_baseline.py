"""
Occlusion attribution with a
per-image nuclear-mean baseline, holding window=8x8/stride=4x4 fixed to match the
existing gray-baseline (0.5,0.5,0.5) files (`<stem>_positive_occlusion_AML_baseline.npy`)
for the same 1,662-cell MRD subset -- so the ONLY thing that differs from those files is
the baseline value, isolating baseline choice from window/stride.

Baseline = per-image mean intensity of non-black (non-padding) pixels, broadcast as a
constant-fill image of the same shape -- the exact recipe given by the user.

Usage: python3 run_nucmean_baseline_occlusion.py [--limit N] [--test]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
import albumentations as A
from captum.attr import Occlusion

ROOT = os.environ.get("AINU_ROOT", ".")
MODEL_PATH = f"{ROOT}/data/konstantina_rendering/ainu_split3_model_best.pth.tar"
AUG_PATH = f"{ROOT}/augmentation/test/PadResize128_GD_r03_u4-32_p07_grey_blackbg.json"
IMG_DIR = f"{ROOT}/data/human_data"
MRD_MDS_DIR = f"{ROOT}/results/occlusion_analysis/ainu_split3_model_best.pth.tar/ainu_split3_MRD_MDS"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 2)
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    alt_stems = sorted(re.sub(r"_positive_occlusion_AML_baseline\.npy$", "", os.path.basename(f))
                        for f in glob.glob(f"{MRD_MDS_DIR}/*_positive_occlusion_AML_baseline.npy"))
    if args.limit:
        alt_stems = alt_stems[:args.limit]
    print(f"stems to process: {len(alt_stems)}", flush=True)

    transform = A.load(AUG_PATH)
    model = load_model()
    occlusion = Occlusion(model)

    out_dir = f"{OUT_DIR}/nucmean_baseline_occlusion"
    os.makedirs(out_dir, exist_ok=True)

    done, skipped = 0, 0
    for i, name in enumerate(alt_stems):
        out_path = f"{out_dir}/{name}_nucmean_occlusion.npy"
        if os.path.exists(out_path):
            done += 1
            continue
        img_path = f"{IMG_DIR}/{name}.png"
        if not os.path.exists(img_path):
            skipped += 1
            continue
        img = Image.open(img_path).convert("RGB")
        transformed = transform(image=np.asarray(img))
        img_tensor = transformed["image"] / 255
        input_tensor = img_tensor.unsqueeze(0).float().to(device)

        # Per-image nuclear mean baseline: exclude black-padding pixels
        nuc_mean = input_tensor[input_tensor > 0.04].mean()  # 0.04 ~= 10/255; excludes border
        baseline_img = torch.full_like(input_tensor, nuc_mean)

        with torch.no_grad():
            output = F.softmax(model(input_tensor), dim=1)
        score, pred_idx = torch.topk(output, 1)
        predicted_label = "positive" if pred_idx.item() == 1 else "negative"
        print(f"  {name} -> {predicted_label} ({score.item():.4f})", flush=True)

        attributions_occ = occlusion.attribute(
            input_tensor,
            strides=(3, 4, 4),
            target=1,
            sliding_window_shapes=(3, 8, 8),
            baselines=baseline_img,
            perturbations_per_eval=64,
        )
        arr = attributions_occ.squeeze().detach().cpu().numpy()
        np.save(out_path, arr)
        done += 1

        if (i + 1) % 50 == 0:
            print(f"  progress {i+1}/{len(alt_stems)}", flush=True)

    print(f"\ndone={done} skipped(no png)={skipped}", flush=True)


if __name__ == "__main__":
    main()
