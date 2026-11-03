"""
Robustness check: does r_texture survive under a different occlusion baseline?

ainu_split3_MRD_MDS/ has two occlusion variants for a 1,662-cell subset of the MRD
cohort (de novo + AML_post-CMML disease_type, 17 patients):
  <stem>_AML.npy                             -- main black-baseline occlusion (used everywhere else)
  <stem>_positive_occlusion_AML_baseline.npy -- an alternative-baseline variant

Confirmed these are NOT duplicates: whole-image channel-0 correlation between the two
ranges from -0.06 to +0.40 across a random sample of cells (i.e. related but distinct
attribution maps, consistent with a different occlusion baseline choice).

This recomputes r_texture per cell under BOTH baselines for this same 1,662-cell subset,
aggregates to patient level, and compares -- directly testing whether the texture-attribution
correlation is an artifact of the black-baseline choice.
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from mrd_mds_texture_extraction import model_frame, nucleus_mask, local_variance, zc, MRD_MDS_DIR, MIN_MASK_PX

def attr_map_alt(stem):
    return np.load(f"{MRD_MDS_DIR}/{stem}_positive_occlusion_AML_baseline.npy")[0]

def attr_map_black(stem):
    return np.load(f"{MRD_MDS_DIR}/{stem}_AML.npy")[0]

def r_texture_for(stem, attr_fn):
    gray = model_frame(stem)
    if gray is None:
        return None
    try:
        attr = attr_fn(stem)
    except Exception:
        return None
    m = nucleus_mask(gray)
    if m.sum() < MIN_MASK_PX:
        return None
    vi = local_variance(gray)[m]
    ai = attr[m]
    return float(np.corrcoef(zc(vi), zc(ai))[0, 1])

meta = pd.read_csv(f"{MRD_MDS_DIR}/MRD_MDS.csv", sep="\t", low_memory=False)
meta = meta[["File", "patientID"]].drop_duplicates("File")

alt_stems = sorted(os.path.basename(p).replace("_positive_occlusion_AML_baseline.npy", "")
                   for p in glob.glob(f"{MRD_MDS_DIR}/*_positive_occlusion_AML_baseline.npy"))
print(f"subset: {len(alt_stems)} cells", flush=True)

rows = []
for i, s in enumerate(alt_stems):
    r_black = r_texture_for(s, attr_map_black)
    r_alt = r_texture_for(s, attr_map_alt)
    if r_black is not None and r_alt is not None:
        rows.append(dict(File=s, r_texture_black=r_black, r_texture_alt=r_alt))
    if (i + 1) % 400 == 0:
        print(f"  {i+1}/{len(alt_stems)}", flush=True)

df = pd.DataFrame(rows).merge(meta, on="File", how="left")
df.to_csv("baseline_robustness_percell.csv", index=False)
print(f"kept {len(df)} cells, {df['patientID'].nunique()} patients", flush=True)

# cell-level agreement
r_cellwise = np.corrcoef(df["r_texture_black"], df["r_texture_alt"])[0, 1]
print(f"\nCell-level correlation between black-baseline and alt-baseline r_texture: {r_cellwise:+.3f}", flush=True)

# patient-level medians, paired comparison
pat = df.groupby("patientID")[["r_texture_black", "r_texture_alt"]].median()
n_cells_pp = df.groupby("patientID").size()
pat = pat[n_cells_pp >= 5]
print(f"\nPatients with >=5 cells: {len(pat)}", flush=True)
print(pat.round(3), flush=True)

w = wilcoxon(pat["r_texture_black"], pat["r_texture_alt"])
print(f"\nWilcoxon signed-rank, black vs alt (paired, per patient): p = {w.pvalue:.3e}", flush=True)
print(f"black median across patients: {pat['r_texture_black'].median():+.3f}", flush=True)
print(f"alt   median across patients: {pat['r_texture_alt'].median():+.3f}", flush=True)

for col, label in [("r_texture_black", "black-baseline"), ("r_texture_alt", "alt-baseline")]:
    x = pat[col].dropna().values
    try:
        p0 = wilcoxon(x).pvalue
    except Exception:
        p0 = np.nan
    print(f"{label}: median={np.median(x):+.3f}, Wilcoxon-vs-0 p={p0:.3e}", flush=True)
