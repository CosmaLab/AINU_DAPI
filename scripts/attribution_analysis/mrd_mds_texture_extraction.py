"""
Extend the S4C texture result (r_localvar, delta_int) to the MDS and MRD cohorts, so the
patient-level panel can show all four groups: control, MDS, MRD, AML.

Group definitions (from MRD_MDS.csv, same logic as MRD_MDS_Analysis.ipynb). Group label
"Preleukemia" (not "MDS") to match occlusion_feature_analysis_AML_MDS_CT.ipynb's naming:
  Preleukemia : disease_type in {MDS, CMML, CCUS}   (MRD column is NaN -- true preleukemia,
        not "unknown MRD status"; confirmed 1:1 in the earlier notebook session)
  MRD : disease_type in {de novo, "AML _post-CMML"}  (MRD column has a value: negative /
        low-positive / positive / unspecific -- the de-novo-AML-post-treatment monitoring
        cohort)

This cohort's occlusion maps live in a different directory with a different naming
convention than the original AML/control test set used by full_attr_analysis.py /
pos_neg_split.py (`<stem>_AML.npy` here vs `<stem>_AML_occlusion.npy` there), but the array
is the same shape/semantics (channel 0 of a (3,128,128) float32 array = Captum Occlusion
attribution toward the AML logit), so the same nucleus segmentation / metric code is reused.

Output: mrd_mds_texture_metrics.csv (per-cell) in this dir, for
patient_level_texture_panel.py to merge in.
"""
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

OUT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("AINU_ROOT", ".")
MRD_MDS_DIR = (_ROOT + "/results/occlusion_analysis/"
               "ainu_split3_model_best.pth.tar/ainu_split3_MRD_MDS")
IMG_DIR = _ROOT + "/data/human_data"
SIZE = 128
MIN_MASK_PX = 300
MIN_SIGN_PX = 25
MIN_SIGN_MASS = 1e-6


def model_frame(stem):
    p = f"{IMG_DIR}/{stem}.png"
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("L")
    w, h = im.size
    s = SIZE / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    c = Image.new("L", (SIZE, SIZE), 0)
    c.paste(im, ((SIZE - im.size[0]) // 2, (SIZE - im.size[1]) // 2))
    return np.asarray(c, float) / 255.0


def attr_map(stem):
    return np.load(f"{MRD_MDS_DIR}/{stem}_AML.npy")[0]


def nucleus_mask(gray01):
    g = ndimage.gaussian_filter(gray01 * 255.0, sigma=1.5)
    b = g > g.max() * 0.2
    b = ndimage.binary_fill_holes(b)
    lab, n = ndimage.label(b)
    if n == 0:
        return b.astype(bool)
    keep = 1 + int(np.argmax(ndimage.sum(b, lab, range(1, n + 1))))
    return ndimage.binary_fill_holes(lab == keep).astype(bool)


def local_variance(g, k=5):
    m1 = ndimage.uniform_filter(g, k)
    m2 = ndimage.uniform_filter(g * g, k)
    return np.clip(m2 - m1 * m1, 0, None)


def zc(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def wmean(v, w):
    s = w.sum()
    return float((v * w).sum() / s) if s > MIN_SIGN_MASS else np.nan


def analyse(stem):
    gray = model_frame(stem)
    if gray is None:
        return None
    try:
        attr = attr_map(stem)
    except Exception:
        return None
    m = nucleus_mask(gray)
    if m.sum() < MIN_MASK_PX:
        return None

    gi, ai = gray[m], attr[m]
    vi = local_variance(gray)[m]
    r_lv = np.corrcoef(zc(vi), zc(ai))[0, 1]
    # brightness-null control (matches full_attr_analysis.py's r_int_pearson): correlation
    # of attribution with raw DAPI intensity, not texture -- tests whether the texture
    # relationship above is just brightness in disguise.
    r_int = np.corrcoef(zc(gi), zc(ai))[0, 1]

    inten_z = np.zeros_like(gray); inten_z[m] = zc(gray[m])
    lv_full = local_variance(gray); lv_z = np.zeros_like(gray); lv_z[m] = zc(lv_full[m])

    pos = np.where(m, np.clip(attr, 0, None), 0.0)
    neg = np.where(m, np.clip(-attr, 0, None), 0.0)
    npos, nneg = int((pos > 0).sum()), int((neg > 0).sum())

    d = dict(File=stem, n_mask_px=int(m.sum()), r_localvar=float(r_lv), r_int_pearson=float(r_int))
    for name, w, npx in [("pos", pos, npos), ("neg", neg, nneg)]:
        if npx >= MIN_SIGN_PX:
            d[f"{name}_w_int"] = wmean(inten_z[m], w[m])
            d[f"{name}_w_lv"] = wmean(lv_z[m], w[m])
        else:
            d[f"{name}_w_int"] = d[f"{name}_w_lv"] = np.nan
    d["delta_int"] = d["pos_w_int"] - d["neg_w_int"]
    d["delta_lv"] = d["pos_w_lv"] - d["neg_w_lv"]
    return d


def main():
    meta = pd.read_csv(f"{MRD_MDS_DIR}/MRD_MDS.csv", sep="\t", low_memory=False)
    meta = meta[["File", "patientID", "disease_type", "MRD", "condition"]].drop_duplicates("File")

    is_mds = meta["disease_type"].isin(["MDS", "CMML", "CCUS"])
    is_mrd = meta["disease_type"].isin(["de novo", "AML _post-CMML"])
    meta["group"] = np.select([is_mds, is_mrd], ["Preleukemia", "MRD"], default=None)
    meta = meta[meta["group"].notna()].reset_index(drop=True)
    print(f"metadata rows tagged Preleukemia/MRD: {len(meta)}  "
          f"(Preleukemia={int(is_mds.sum())}, MRD={int(is_mrd.sum())})", flush=True)

    stems = sorted(os.path.basename(p).replace("_AML.npy", "")
                    for p in glob.glob(f"{MRD_MDS_DIR}/*_AML.npy"))
    stems = [s for s in stems if s in set(meta["File"])]
    print(f"stems with an occlusion map and a group label: {len(stems)}", flush=True)

    rows = []
    for i, s in enumerate(stems):
        r = analyse(s)
        if r:
            rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(stems)}  kept {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df = df.merge(meta[["File", "patientID", "group"]], on="File", how="left")
    df.to_csv(f"{OUT}/mrd_mds_texture_metrics.csv", index=False)
    print(f"\nwrote mrd_mds_texture_metrics.csv: {df.shape}", flush=True)
    print(df.groupby("group")[["File", "patientID"]].nunique(), flush=True)


if __name__ == "__main__":
    main()
