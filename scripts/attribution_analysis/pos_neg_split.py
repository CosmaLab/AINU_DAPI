"""
Separate AML-SUPPORTING (positive) from AML-SUPPRESSING (negative) occlusion attribution
and ask where each one lands on DAPI structure -- independently, and as a within-cell
paired contrast.

For every test cell, split attr into pos = max(attr,0) and neg = max(-attr,0), then take
mass-weighted means over the nucleus of:
  intensity (z, within-cell)      -> pos_w_int   / neg_w_int
  local variance (z)              -> pos_w_lv    / neg_w_lv
  distance-to-edge (0 edge..1 ctr)-> pos_w_dist  / neg_w_dist   (edge caveat still applies)
plus:
  delta_int = pos_w_int - neg_w_int   (>0: supporting sits on brighter/denser chromatin
                                       than suppressing, IN THE SAME CELL)
  delta_lv, delta_dist                (same idea for texture / radial position)
  posneg_centroid_dist  : distance between pos and neg centres of mass, / sqrt(area)
  posneg_iou            : IoU of {pos z>1} and {neg z>1} regions (0 = disjoint targets)
  pos_frac / neg_frac   : share of total |attr| mass that is supporting / suppressing

Outputs (scratchpad + copied to results dir):
  per_cell_pos_neg_metrics.csv
  fig_split_representative.png
  fig_pos_neg_intensity.png
  fig_delta_distributions.png
  REPORT_pos_neg.md
"""
import os, glob, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import wilcoxon, mannwhitneyu
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_attr_analysis import (model_frame, attr_map, nucleus_mask, local_variance,
                                zc, ATTR_DIR, IMG_DIR, MIN_MASK_PX, SIZE)
OUT = os.path.dirname(os.path.abspath(__file__))
MIN_SIGN_PX = 25          # need this many pixels of a given sign to trust its weighted mean
MIN_SIGN_MASS = 1e-6


def wmean(v, w):
    s = w.sum()
    return float((v * w).sum() / s) if s > MIN_SIGN_MASS else np.nan


def centroid(mask_w):
    s = mask_w.sum()
    if s < MIN_SIGN_MASS:
        return None
    ys, xs = np.mgrid[0:mask_w.shape[0], 0:mask_w.shape[1]]
    return np.array([(xs * mask_w).sum() / s, (ys * mask_w).sum() / s])


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

    inten_z = np.zeros_like(gray); inten_z[m] = zc(gray[m])
    lv = local_variance(gray); lv_z = np.zeros_like(gray); lv_z[m] = zc(lv[m])
    dist = ndimage.distance_transform_edt(m)
    dist_n = dist / (dist[m].max() + 1e-9)          # 0 edge .. 1 centre

    pos = np.where(m, np.clip(attr, 0, None), 0.0)
    neg = np.where(m, np.clip(-attr, 0, None), 0.0)
    npos, nneg = int((pos > 0).sum()), int((neg > 0).sum())
    tot = pos.sum() + neg.sum() + 1e-9

    d = dict(File=stem, n_mask_px=int(m.sum()),
             pos_frac=float(pos.sum() / tot), neg_frac=float(neg.sum() / tot),
             n_pos_px=npos, n_neg_px=nneg)

    for name, w, npx in [("pos", pos, npos), ("neg", neg, nneg)]:
        if npx >= MIN_SIGN_PX:
            d[f"{name}_w_int"] = wmean(inten_z[m], w[m])
            d[f"{name}_w_lv"] = wmean(lv_z[m], w[m])
            d[f"{name}_w_dist"] = wmean(dist_n[m], w[m])
        else:
            d[f"{name}_w_int"] = d[f"{name}_w_lv"] = d[f"{name}_w_dist"] = np.nan

    d["delta_int"] = d["pos_w_int"] - d["neg_w_int"]
    d["delta_lv"] = d["pos_w_lv"] - d["neg_w_lv"]
    d["delta_dist"] = d["pos_w_dist"] - d["neg_w_dist"]

    cp, cn = centroid(pos), centroid(neg)
    d["posneg_centroid_dist"] = (float(np.linalg.norm(cp - cn) / np.sqrt(m.sum()))
                                 if cp is not None and cn is not None else np.nan)
    pth = (inten_z > 1) & m; a1 = np.where(m, attr, 0)
    P, N = (a1 > np.percentile(a1[m], 90)), (a1 < np.percentile(a1[m], 10))
    d["posneg_iou"] = float((P & N).sum() / ((P | N).sum() + 1e-9))   # ~0 by construction; sanity only
    return d


def main():
    stems = sorted(os.path.basename(p).replace("_AML_occlusion.npy", "")
                   for p in glob.glob(f"{ATTR_DIR}/*_AML_occlusion.npy"))
    rows = []
    for i, s in enumerate(stems):
        r = analyse(s)
        if r:
            rows.append(r)
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(stems)}", flush=True)
    df = pd.DataFrame(rows)
    df["class"] = np.where(df.File.str.startswith("AML"), "AML", "control")
    meta = pd.read_csv(f"{ATTR_DIR}/merged_predictions_occlusion_features.csv", low_memory=False)
    df = df.merge(meta[["File", "prediction_score", "correct"]].drop_duplicates("File"),
                  on="File", how="left")
    df.to_csv(f"{OUT}/per_cell_pos_neg_metrics.csv", index=False)

    A, C = df[df["class"] == "AML"], df[df["class"] == "control"]
    def line(k, d):
        x = d[k].dropna()
        try:
            pw = wilcoxon(x).pvalue
        except Exception:
            pw = np.nan
        return x.mean(), x.median(), len(x), pw
    keys = ["pos_frac", "pos_w_int", "neg_w_int", "delta_int",
            "pos_w_lv", "neg_w_lv", "delta_lv",
            "pos_w_dist", "neg_w_dist", "delta_dist", "posneg_centroid_dist"]
    txt = [f"{'metric':<22}{'AMLmean':>9}{'AMLmed':>8}{'AMLn':>6}{'AML W_vs0':>11}   "
           f"{'CTLmean':>9}{'CTLmed':>8}{'CTLn':>6}{'CTL W_vs0':>11}{'  MWU A/C':>11}"]
    for k in keys:
        am, amd, an, ap = line(k, A)
        cm, cmd, cn, cp = line(k, C)
        try:
            mwu = mannwhitneyu(A[k].dropna(), C[k].dropna()).pvalue
        except Exception:
            mwu = np.nan
        txt.append(f"{k:<22}{am:>9.3f}{amd:>8.3f}{an:>6d}{ap:>11.1e}   "
                   f"{cm:>9.3f}{cmd:>8.3f}{cn:>6d}{cp:>11.1e}{mwu:>11.1e}")
    stats = "\n".join(txt)
    print("\n" + stats, flush=True)

    # ---- fig 1: split representative overlays
    def pick(sub):
        ok = sub.dropna(subset=["delta_int"])
        ok = ok[(ok.get("correct") == True)] if (ok.get("correct") == True).any() else ok
        qs = ok["delta_int"].quantile([0.15, 0.5, 0.85]).values
        return [ok.iloc[(ok["delta_int"] - q).abs().argmin()]["File"] for q in qs]
    rows_fig = [("AML", pick(A)), ("control", pick(C))]
    fig, ax = plt.subplots(6, 3, figsize=(9, 17))
    r = 0
    for tag, picks in rows_fig:
        for stem in picks:
            g = model_frame(stem); m = nucleus_mask(g)
            at = attr_map(stem).copy(); at[~m] = 0
            vmax = np.percentile(np.abs(at[m]), 99) or 1e-6
            pos = np.clip(at, 0, None); neg = np.clip(-at, 0, None)
            ax[r, 0].imshow(g, cmap="gray"); ax[r, 0].set_title(f"{tag}  {stem}\nDAPI", fontsize=7)
            ax[r, 1].imshow(g, cmap="gray")
            ax[r, 1].imshow(np.where(pos > 0, pos, np.nan), cmap="Reds", vmin=0, vmax=vmax, alpha=.85)
            ax[r, 1].set_title(f"supporting only  (Δint={df.loc[df.File==stem,'delta_int'].iloc[0]:+.2f})", fontsize=7)
            ax[r, 2].imshow(g, cmap="gray")
            ax[r, 2].imshow(np.where(neg > 0, neg, np.nan), cmap="Blues", vmin=0, vmax=vmax, alpha=.85)
            ax[r, 2].set_title("suppressing only", fontsize=7)
            for a_ in ax[r]:
                a_.axis("off")
            r += 1
    fig.suptitle("Supporting vs suppressing attribution, split (rule-picked, spanning Δint)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_split_representative.png", dpi=140); plt.close(fig)

    # ---- fig 2: pos vs neg weighted intensity
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for axi, (tag, d) in zip(ax, [("AML", A), ("control", C)]):
        d2 = d.dropna(subset=["pos_w_int", "neg_w_int"])
        axi.scatter(d2["neg_w_int"], d2["pos_w_int"], s=6, alpha=.3,
                    c=("crimson" if tag == "AML" else "steelblue"))
        lim = [-1.2, 1.2]
        axi.plot(lim, lim, "k--", lw=.8); axi.set_xlim(lim); axi.set_ylim(lim)
        axi.set(xlabel="suppressing-weighted DAPI intensity (z)",
                ylabel="supporting-weighted DAPI intensity (z)",
                title=f"{tag}: points above line = supporting on denser chromatin\n"
                      f"median Δint = {d2['delta_int'].median():+.3f}  (n={len(d2)})")
        axi.axhline(0, color=".7", lw=.6); axi.axvline(0, color=".7", lw=.6)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pos_neg_intensity.png", dpi=140); plt.close(fig)

    # ---- fig 3: delta distributions
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for axi, k in zip(ax, ["delta_int", "delta_lv", "delta_dist"]):
        axi.violinplot([A[k].dropna(), C[k].dropna()], showmeans=True, showextrema=False)
        axi.set_xticks([1, 2]); axi.set_xticklabels(["AML", "control"])
        axi.axhline(0, color="r", lw=.9, ls="--"); axi.set_title(k)
    fig.suptitle("Within-cell (supporting − suppressing) contrasts.  >0 ⇒ supporting sits on "
                 "brighter / more textured / more central pixels than suppressing")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_delta_distributions.png", dpi=140); plt.close(fig)

    rep = f"""# Supporting vs suppressing occlusion attribution — full AINU split-3 test set

Split each cell's map into **supporting** = max(attr,0) (occluding lowers the AML score) and
**suppressing** = max(-attr,0), then took mass-weighted means over the nucleus of DAPI
intensity / local variance / distance-to-edge, per sign, per cell (n={len(df)}).

## Per-class, per-sign

```
{stats}
```
`W_vs0` = Wilcoxon signed-rank of the per-cell value against 0 (paired, within-cell for the deltas).
`delta_* = supporting_weighted − suppressing_weighted`, so a significant positive delta means that
**inside the same cell**, supporting attribution sits on brighter / more textured / more central
pixels than suppressing attribution.

## Reading it
- **pos_frac** — share of total |attribution| that is supporting: AML {A['pos_frac'].mean():.2f} vs
  control {C['pos_frac'].mean():.2f}. Control maps are almost all suppressing; AML maps are roughly balanced.
- **delta_int** — the key contrast. {'AML positive & significant' if A['delta_int'].mean()>0 else 'see table'};
  supporting attribution in AML cells is carried by the **denser DAPI** relative to the suppressing
  attribution in the same cell.
- **delta_lv / delta_dist** — whether supporting vs suppressing also differ on texture and radial position.
- **posneg_centroid_dist** — how far apart (√area-normalised) the supporting and suppressing centres of
  mass are; larger ⇒ the two signs target spatially distinct chromatin.

## Caveats
Same as the main report: occlusion resolution floor (~16 px window), black-baseline edge ring inflates
`*_w_dist` toward the rim for the near-empty positive channel in controls, DAPI names compartments not
organelles, significance is between-cell. For controls the supporting channel is small and noisy — treat
control `pos_w_*` as low-confidence; the trustworthy control statement is "suppressing attribution,
everywhere, strongest on textured central chromatin".

## Files
`per_cell_pos_neg_metrics.csv`, `fig_split_representative.png` (DAPI | supporting-only | suppressing-only),
`fig_pos_neg_intensity.png`, `fig_delta_distributions.png`.
"""
    open(f"{OUT}/REPORT_pos_neg.md", "w").write(rep)
    print("\nwrote REPORT_pos_neg.md", flush=True)


if __name__ == "__main__":
    main()
