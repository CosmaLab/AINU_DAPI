"""
Direct, non-algebraic verification of what r_localvar's sign actually means:
bin each cell's nucleus pixels into texture terciles (smooth / mid / textured) and report
the *mean signed attribution* in each bin, plus what fraction of the (unsigned) suppressing
and supporting attribution mass falls in the textured tercile. This settles the direction
question empirically instead of by reasoning about correlation-sign algebra.
"""
import glob, os
import numpy as np
import pandas as pd
from full_attr_analysis import model_frame, attr_map, nucleus_mask, local_variance, ATTR_DIR, MIN_MASK_PX

stems = sorted(os.path.basename(p).replace("_AML_occlusion.npy", "")
               for p in glob.glob(f"{ATTR_DIR}/*_AML_occlusion.npy"))

rows = []
for s in stems:
    gray = model_frame(s)
    if gray is None:
        continue
    try:
        attr = attr_map(s)
    except Exception:
        continue
    m = nucleus_mask(gray)
    if m.sum() < MIN_MASK_PX:
        continue
    ai = attr[m]
    vi = local_variance(gray)[m]
    q1, q2 = np.percentile(vi, [33.333, 66.667])
    smooth, textured = vi <= q1, vi >= q2
    pos, neg = np.clip(ai, 0, None), np.clip(-ai, 0, None)
    rows.append(dict(
        File=s,
        attr_mean_smooth=float(ai[smooth].mean()),
        attr_mean_textured=float(ai[textured].mean()),
        neg_mass_frac_textured=float(neg[textured].sum() / (neg.sum() + 1e-9)),
        neg_mass_frac_smooth=float(neg[smooth].sum() / (neg.sum() + 1e-9)),
    ))

df = pd.DataFrame(rows)
df["class"] = np.where(df.File.str.startswith("AML"), "AML", "control")
df.to_csv("verify_texture_direction_percell.csv", index=False)

print("Per-class MEDIAN of per-cell values (n cells):")
print(df.groupby("class")[["attr_mean_smooth", "attr_mean_textured",
                            "neg_mass_frac_smooth", "neg_mass_frac_textured"]].median())
print()
print("(uniform/no-relationship expectation for neg_mass_frac_* would be ~0.333 each)")
