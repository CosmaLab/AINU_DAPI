"""
Full-dataset occlusion-attribution vs DAPI-structure analysis for the AINU split-3
test cells (the 2072 *_AML_occlusion.npy in .../both/).

For every test cell it puts the attribution map and the raw DAPI in the model's own
128x128 frame, masks the nucleus, and measures WHERE the AML-supporting / AML-suppressing
signal falls relative to DAPI structure:

  r_int_pearson / r_int_spearman : attribution vs DAPI intensity, per-pixel, within nucleus
  r_localvar                     : attribution vs local variance (texture / puncta-ness)
  attr_bright_minus_dim          : mean attribution in the brightest DAPI tercile minus
                                   the dimmest tercile  (logit units)  -> dense chromatin vs DNA-poor
  attr_periph_minus_core         : mean attribution in the outer 25% (by distance-to-edge)
                                   minus the inner core  -> tests the "condensed peripheral
                                   heterochromatin" idea directly
  frac_pos_mass_bright/_periph   : share of total POSITIVE (AML-supporting) attribution mass
                                   landing in the bright tercile / peripheral band
  frac_neg_mass_bright/_periph   : same for negative (AML-suppressing) mass
QC: solidity, mask_frac_frame  (multiplet / debris flag)

Outputs (scratchpad):
  per_cell_attribution_metrics.csv
  summary_by_class.csv
  fig_percell_distributions.png
  fig_pooled_hexbin.png
  fig_representative_overlays.png
  fig_contactsheet_AML.png / fig_contactsheet_control.png
  REPORT.md
"""
import os, glob, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
warnings.filterwarnings("ignore")

ROOT = os.environ.get("AINU_ROOT", ".")
ATTR_DIR = f"{ROOT}/results/occlusion_analysis/ainu_split3_model_best.pth.tar/PadResize128_GD_r03_u4-32_p07_grey_blackbg/both"
IMG_DIR = f"{ROOT}/data/human_data"
OUT = os.path.dirname(os.path.abspath(__file__))
SIZE = 128
MIN_MASK_PX = 300


# ---------------------------------------------------------------- helpers
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
    return np.load(f"{ATTR_DIR}/{stem}_AML_occlusion.npy")[0]


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


def solidity(mask):
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return np.nan
    try:
        pts = np.column_stack([xs, ys]).astype(float)
        h = ConvexHull(pts)
        return float(mask.sum() / h.volume)   # 2D: .volume == area
    except Exception:
        return np.nan


def zc(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


# ---------------------------------------------------------------- per-cell metrics
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

    gi = gray[m]
    ai = attr[m]
    vi = local_variance(gray)[m]

    # distance to nuclear boundary (inside)
    dist = ndimage.distance_transform_edt(m)[m]

    # intensity terciles within nucleus
    q1, q2 = np.percentile(gi, [33.333, 66.667])
    dim, bright = gi <= q1, gi >= q2
    # peripheral / core by distance
    d25, d75 = np.percentile(dist, [25, 75])
    periph, core = dist <= d25, dist >= d75

    pos = np.clip(ai, 0, None)
    neg = np.clip(-ai, 0, None)
    pmass, nmass = pos.sum() + 1e-9, neg.sum() + 1e-9

    r_int = np.corrcoef(zc(gi), zc(ai))[0, 1]
    r_sp = spearmanr(gi, ai).correlation
    r_lv = np.corrcoef(zc(vi), zc(ai))[0, 1]

    return dict(
        File=stem,
        n_mask_px=int(m.sum()),
        mask_frac_frame=float(m.sum() / (SIZE * SIZE)),
        solidity=solidity(m),
        attr_mean=float(ai.mean()),
        attr_pos_mass=float(pmass),
        attr_neg_mass=float(nmass),
        r_int_pearson=float(r_int),
        r_int_spearman=float(r_sp),
        r_localvar=float(r_lv),
        attr_bright_minus_dim=float(ai[bright].mean() - ai[dim].mean()),
        attr_periph_minus_core=float(ai[periph].mean() - ai[core].mean()),
        frac_pos_mass_bright=float(pos[bright].sum() / pmass),
        frac_pos_mass_periph=float(pos[periph].sum() / pmass),
        frac_neg_mass_bright=float(neg[bright].sum() / nmass),
        frac_neg_mass_periph=float(neg[periph].sum() / nmass),
        # keep the z-scored pixel clouds for pooled plots (subsample to cap memory)
        _cloud=np.column_stack([zc(gi), zc(ai), zc(vi)])[
            np.random.default_rng(abs(hash(stem)) % 2**32).choice(
                len(gi), size=min(400, len(gi)), replace=False)],
    )


# ---------------------------------------------------------------- run
def main():
    stems = sorted(os.path.basename(p).replace("_AML_occlusion.npy", "")
                   for p in glob.glob(f"{ATTR_DIR}/*_AML_occlusion.npy"))
    meta = pd.read_csv(f"{ATTR_DIR}/merged_predictions_occlusion_features.csv", low_memory=False)
    meta = meta[["File", "prediction_score", "correct"]].drop_duplicates("File")
    pid = pd.read_csv(f"{IMG_DIR}/patient_image_data.csv", low_memory=False)
    keepc = [c for c in ["File", "condition", "disease_type", "source", "patientID", "new_labels"] if c in pid.columns]
    pid = pid[keepc].drop_duplicates("File")

    rows, clouds = [], {}
    for i, s in enumerate(stems):
        r = analyse(s)
        if r is not None:
            clouds[s] = r.pop("_cloud")
            rows.append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(stems)}  kept {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df["class"] = np.where(df["File"].str.startswith("AML"), "AML", "control")
    df = df.merge(meta, on="File", how="left").merge(pid, on="File", how="left")
    df.to_csv(f"{OUT}/per_cell_attribution_metrics.csv", index=False)
    print("per-cell CSV:", df.shape, flush=True)

    metrics = ["r_int_pearson", "r_int_spearman", "r_localvar",
               "attr_bright_minus_dim", "attr_periph_minus_core",
               "frac_pos_mass_bright", "frac_pos_mass_periph",
               "frac_neg_mass_bright", "frac_neg_mass_periph"]

    # ---- summary + stats
    g = df.groupby("class")
    summ = g[metrics].agg(["mean", "std", "median", "count"])
    summ.to_csv(f"{OUT}/summary_by_class.csv")

    lines = []
    A = df[df["class"] == "AML"]
    C = df[df["class"] == "control"]
    lines.append(f"{'metric':<26} {'AML mean':>12} {'ctrl mean':>12} {'MWU p':>12} "
                 f"{'AML W p vs0':>13} {'ctrl W p vs0':>13}")
    for mkey in metrics:
        a, c = A[mkey].dropna(), C[mkey].dropna()
        p_mwu = mannwhitneyu(a, c, alternative="two-sided").pvalue
        def w0(x):
            x = x[np.isfinite(x)]
            try:
                return wilcoxon(x, zero_method="wilcox").pvalue
            except Exception:
                return np.nan
        lines.append(f"{mkey:<26} {a.mean():>12.4f} {c.mean():>12.4f} {p_mwu:>12.2e} "
                     f"{w0(a.values):>13.2e} {w0(c.values):>13.2e}")
    stats_txt = "\n".join(lines)
    print(stats_txt, flush=True)

    # ---- fig 1: per-cell distributions
    show = ["r_int_pearson", "r_localvar", "attr_bright_minus_dim", "attr_periph_minus_core"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for k, axi in zip(show, ax.ravel()):
        data = [A[k].dropna().values, C[k].dropna().values]
        parts = axi.violinplot(data, showmeans=True, showextrema=False)
        axi.set_xticks([1, 2]); axi.set_xticklabels(["AML", "control"])
        axi.axhline(0, color="0.6", lw=0.8, ls="--")
        axi.set_title(k)
    fig.suptitle("Per-cell attribution–structure metrics (all test cells)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_percell_distributions.png", dpi=140); plt.close(fig)

    # ---- fig 2: pooled pixel clouds
    def pool(sub):
        arr = np.vstack([clouds[s] for s in sub["File"] if s in clouds])
        return arr
    pa, pc = pool(A), pool(C)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for j, (name, P) in enumerate([("AML", pa), ("control", pc)]):
        hb = ax[0, j].hexbin(P[:, 0], P[:, 1], gridsize=60, cmap="magma", bins="log")
        r = np.corrcoef(P[:, 0], P[:, 1])[0, 1]
        ax[0, j].set(title=f"{name}: attr vs DAPI intensity  r={r:+.3f}",
                     xlabel="DAPI intensity (z)", ylabel="attribution (z)")
        ax[0, j].axhline(0, color="c", lw=.6); ax[0, j].axvline(0, color="c", lw=.6)
        r2 = np.corrcoef(P[:, 2], P[:, 1])[0, 1]
        ax[1, j].hexbin(P[:, 2], P[:, 1], gridsize=60, cmap="magma", bins="log")
        ax[1, j].set(title=f"{name}: attr vs local variance  r={r2:+.3f}",
                     xlabel="local variance (z)", ylabel="attribution (z)")
        ax[1, j].axhline(0, color="c", lw=.6)
    fig.suptitle("Pooled within-cell pixel clouds (z-scored per cell)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pooled_hexbin.png", dpi=140); plt.close(fig)

    # ---- fig 3: rule-based representative overlays
    def repr_panel(sub, tag):
        ok = sub[(sub["correct"] == True) & (sub["solidity"] > 0.90)
                 & (sub["prediction_score"].between(0.55, 0.99))
                 & (sub["mask_frac_frame"].between(0.20, 0.75))].dropna(subset=["r_int_pearson"])
        if len(ok) < 4:
            ok = sub.dropna(subset=["r_int_pearson"])
        qs = ok["r_int_pearson"].quantile([0.1, 0.4, 0.6, 0.9]).values
        picks = [ok.iloc[(ok["r_int_pearson"] - q).abs().argmin()]["File"] for q in qs]
        return picks
    picks = {"AML": repr_panel(A, "AML"), "control": repr_panel(C, "control")}
    fig, ax = plt.subplots(2, 4, figsize=(13, 7))
    for row, tag in enumerate(["AML", "control"]):
        for col, stem in enumerate(picks[tag]):
            gray = model_frame(stem); m = nucleus_mask(gray)
            a = attr_map(stem).copy(); a[~m] = 0
            vmax = np.percentile(np.abs(a[m]), 99) or 1e-6
            ax[row, col].imshow(gray, cmap="gray")
            ax[row, col].imshow(a, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.55)
            rr = float(df.loc[df.File == stem, "r_int_pearson"].iloc[0])
            ax[row, col].set_title(f"{tag}  r_int={rr:+.2f}\n{stem}", fontsize=7)
            ax[row, col].axis("off")
    fig.suptitle("Representative overlays (rule-picked: correct, single nucleus, spanning r_int deciles)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_representative_overlays.png", dpi=140); plt.close(fig)

    # ---- fig 4/5: contact sheets
    def contact(sub, tag, n=25, seed=0):
        pick = sub[(sub["correct"] == True)].sample(min(n, len(sub)), random_state=seed)["File"].tolist()
        s = int(np.ceil(np.sqrt(len(pick))))
        fig, ax = plt.subplots(s, s, figsize=(2 * s, 2 * s))
        for axi, stem in zip(ax.ravel(), pick):
            gray = model_frame(stem); m = nucleus_mask(gray)
            a = attr_map(stem).copy(); a[~m] = 0
            vmax = np.percentile(np.abs(a[m]), 99) or 1e-6
            axi.imshow(gray, cmap="gray")
            axi.imshow(a, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.55)
            axi.set_title(stem, fontsize=5); axi.axis("off")
        for axi in ax.ravel()[len(pick):]:
            axi.axis("off")
        fig.suptitle(f"{tag} — random correct test cells (seed {seed})")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_contactsheet_{tag}.png", dpi=130); plt.close(fig)
    contact(A, "AML"); contact(C, "control")

    # ---- sign check
    sc = df.dropna(subset=["prediction_score"])
    r_sign = np.corrcoef(sc["prediction_score"], sc["attr_pos_mass"])[0, 1]

    # ---- report
    rep = f"""# Occlusion attribution vs DAPI structure — full AINU split-3 test set

**Cells analysed:** {len(df)}  ({(df['class']=='AML').sum()} AML, {(df['class']=='control').sum()} control)
skipped for tiny/failed mask: {len(stems) - len(df)} of {len(stems)}.

**Frame:** raw DAPI PNG -> LongestMaxSize(128) + centered black pad -> the model's 128x128 input.
**Attribution:** `<stem>_AML_occlusion.npy` channel 0 = Captum Occlusion toward the AML-class logit.
**Sign check:** corr(prediction_score, positive-attribution mass) = {r_sign:+.3f}
-> {'positive attribution = AML-supporting (colormap correct).' if r_sign > 0 else 'SIGN FLIPPED.'}

## Per-class summary and tests

```
{stats_txt}
```
(MWU p = Mann–Whitney AML vs control; W p vs0 = Wilcoxon signed-rank of the per-cell
value against zero, i.e. "is the effect consistent across cells".)

## How to read the key metrics
- **attr_bright_minus_dim > 0** : AML-supporting signal concentrates on the **brightest DAPI
  pixels** (dense / condensed chromatin) rather than DNA-poor regions (nucleoli, lacunae).
  **< 0** : the opposite — signal sits on dim, DNA-poor territory.
- **attr_periph_minus_core > 0** : signal concentrates in the **outer 25%** of the nucleus
  (rim / peripheral heterochromatin) vs the core.
- **r_localvar > 0** : signal follows **texture / edges / puncta** rather than flat areas.
- **frac_pos_mass_bright** vs **frac_neg_mass_bright** : whether AML-*supporting* and
  AML-*suppressing* mass land on different chromatin compartments (the whole point of
  separating the signs).

## Files
- `per_cell_attribution_metrics.csv` — one row per cell, all metrics + prediction_score/correct/source
- `summary_by_class.csv`
- `fig_percell_distributions.png`, `fig_pooled_hexbin.png`,
  `fig_representative_overlays.png`, `fig_contactsheet_AML.png`, `fig_contactsheet_control.png`
"""
    open(f"{OUT}/REPORT.md", "w").write(rep)
    print("\nwrote REPORT.md", flush=True)


if __name__ == "__main__":
    main()
