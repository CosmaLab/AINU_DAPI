"""
Extends verify_texture_direction.py (which only covered the original 2-class AML/control
cohort) to all four groups, and turns it into a proper, citable result: this is the direct,
non-algebraic evidence for what r_texture's sign means (Figure 6E), not just the correlation
coefficient itself.

For each cell, nucleus pixels are split into texture terciles (smoothest / mid / most
textured third, by local DAPI variance) and two things are measured directly, without
touching the correlation coefficient: (1) mean signed attribution in the smoothest vs. most
textured third, and (2) what share of the cell's total AML-suppressing attribution mass
falls in each third. A patient-level contrast, smooth_minus_textured = attr_mean_smooth -
attr_mean_textured, summarizes (1) as one number per patient: more negative means the
suppressing (healthy) signal concentrates more strongly on smooth chromatin.

Reuses the identical model_frame/nucleus_mask/local_variance functions already used by
full_attr_analysis.py (AML/control) and mrd_mds_texture_extraction.py (Preleukemia/MRD), and
the same group-definition logic, so cell-level results are directly comparable to the rest of
this project's 4-group analyses.

Outputs (this dir):
  table_texture_tercile_direction.csv       (long, one row per patient)
  table_texture_tercile_summary.csv / .md   (group medians, Kruskal-Wallis, pairwise BH-FDR)
  fig_6E2_texture_tercile_direction.png/.pdf  (Figure 6E, second panel)
"""
import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import kruskal, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# same style block as patient_level_texture_panel.py, so this panel matches
# Figure 6E/6F exactly instead of falling back to matplotlib defaults.
sns.set_theme(style="whitegrid", font_scale=0.95)
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "figure.dpi": 120,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
})
FW = 20 / 2.54  # full page width in inches, same convention as patient_level_texture_panel.py

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AINU_ROOT", ".")
IMG_DIR = f"{ROOT}/data/human_data"
AML_CT_DIR = f"{ROOT}/results/occlusion_analysis/ainu_split3_model_best.pth.tar/PadResize128_GD_r03_u4-32_p07_grey_blackbg/both"
MRD_MDS_DIR = f"{ROOT}/results/occlusion_analysis/ainu_split3_model_best.pth.tar/ainu_split3_MRD_MDS"

SIZE = 128
MIN_MASK_PX = 300
MIN_CELLS_PER_PATIENT = 5
GROUP_ORDER = ["Control", "Preleukemia", "MRD", "AML"]
GROUP_COLORS = {"AML": "#B21F2C", "Control": "#2367AC", "Preleukemia": "#F3A582", "MRD": "#93C5DE"}


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


def tercile_metrics(stem, attr_map_fn):
    gray = model_frame(stem)
    if gray is None:
        return None
    try:
        attr = attr_map_fn(stem)
    except Exception:
        return None
    m = nucleus_mask(gray)
    if m.sum() < MIN_MASK_PX:
        return None
    ai = attr[m]
    vi = local_variance(gray)[m]
    q1, q2 = np.percentile(vi, [33.333, 66.667])
    smooth, textured = vi <= q1, vi >= q2
    pos, neg = np.clip(ai, 0, None), np.clip(-ai, 0, None)
    return dict(
        File=stem,
        attr_mean_smooth=float(ai[smooth].mean()),
        attr_mean_textured=float(ai[textured].mean()),
        neg_mass_frac_smooth=float(neg[smooth].sum() / (neg.sum() + 1e-9)),
        neg_mass_frac_textured=float(neg[textured].sum() / (neg.sum() + 1e-9)),
    )


def bh_fdr(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) if passed.any() else -1
    reject = np.zeros(n, dtype=bool)
    if k >= 0:
        reject[order[: k + 1]] = True
    q_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty(n)
    q[order] = q_sorted
    return reject, q


def load_aml_control_cells():
    stems = sorted(os.path.basename(p).replace("_AML_occlusion.npy", "")
                   for p in glob.glob(f"{AML_CT_DIR}/*_AML_occlusion.npy"))
    pid = pd.read_csv(f"{IMG_DIR}/patient_image_data.csv", low_memory=False)
    pid = pid[["File", "patientID"]].drop_duplicates("File")

    def attr_map(stem):
        return np.load(f"{AML_CT_DIR}/{stem}_AML_occlusion.npy")[0]

    rows = []
    for i, s in enumerate(stems):
        r = tercile_metrics(s, attr_map)
        if r is not None:
            rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"  AML/control {i + 1}/{len(stems)} kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df["group"] = np.where(df["File"].str.startswith("AML"), "AML", "Control")
    df = df.merge(pid, on="File", how="left")
    return df


def load_preleukemia_mrd_cells():
    meta = pd.read_csv(f"{MRD_MDS_DIR}/MRD_MDS.csv", sep="\t", low_memory=False)
    meta = meta[["File", "patientID", "disease_type", "MRD"]].drop_duplicates("File")
    is_mds = meta["disease_type"].isin(["MDS", "CMML", "CCUS"])
    is_mrd = meta["disease_type"].isin(["de novo", "AML _post-CMML"])
    meta["group"] = np.select([is_mds, is_mrd], ["Preleukemia", "MRD"], default=None)
    meta = meta[meta["group"].notna()].reset_index(drop=True)

    stems = sorted(os.path.basename(p).replace("_AML.npy", "")
                   for p in glob.glob(f"{MRD_MDS_DIR}/*_AML.npy"))
    stems = [s for s in stems if s in set(meta["File"])]

    def attr_map(stem):
        return np.load(f"{MRD_MDS_DIR}/{stem}_AML.npy")[0]

    rows = []
    for i, s in enumerate(stems):
        r = tercile_metrics(s, attr_map)
        if r is not None:
            rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"  Preleukemia/MRD {i + 1}/{len(stems)} kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df = df.merge(meta[["File", "patientID", "group"]], on="File", how="left")
    return df


def main():
    df1 = load_aml_control_cells()
    df2 = load_preleukemia_mrd_cells()
    df = pd.concat([df1, df2], ignore_index=True)
    df["smooth_minus_textured"] = df["attr_mean_smooth"] - df["attr_mean_textured"]
    print(f"\ncell counts: {df['group'].value_counts().to_dict()}", flush=True)

    metrics = ["attr_mean_smooth", "attr_mean_textured", "smooth_minus_textured",
               "neg_mass_frac_smooth", "neg_mass_frac_textured"]
    n_cells = df.groupby("patientID").size().rename("n_cells")
    pat = (df.groupby(["patientID", "group"], observed=True)[metrics]
             .median()
             .join(n_cells)
             .reset_index())
    pat = pat[pat["n_cells"] >= MIN_CELLS_PER_PATIENT].copy()
    pat.to_csv(f"{OUT}/table_texture_tercile_direction.csv", index=False)

    groups = {g: pat[pat["group"] == g] for g in GROUP_ORDER}
    for g in GROUP_ORDER:
        print(f"  {g}: n={len(groups[g])} patients, median smooth_minus_textured="
              f"{groups[g]['smooth_minus_textured'].median():+.3f}", flush=True)

    key = "smooth_minus_textured"
    vals = [groups[g][key].dropna().values for g in GROUP_ORDER]
    kw_stat, kw_p = kruskal(*vals)
    print(f"\nKruskal-Wallis across 4 groups: H={kw_stat:.2f} p={kw_p:.3e}", flush=True)

    rows = []
    for g1, g2 in combinations(GROUP_ORDER, 2):
        v1, v2 = groups[g1][key].dropna(), groups[g2][key].dropna()
        _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        rows.append(dict(group1=g1, group2=g2, n1=len(v1), n2=len(v2),
                          median1=v1.median(), median2=v2.median(), p_raw=p))
    res = pd.DataFrame(rows)
    reject, q = bh_fdr(res["p_raw"].values)
    res["q_bh"] = q
    res["significant_q05"] = reject
    res = res.sort_values("p_raw").reset_index(drop=True)
    res.to_csv(f"{OUT}/table_texture_tercile_summary.csv", index=False)
    print(res.to_string(index=False), flush=True)

    n_by_g = {g: len(groups[g]) for g in GROUP_ORDER}
    md = f"""**Pixel-tercile direction check: does the suppressing (healthy) signal concentrate on smooth or textured chromatin?**

For each cell, nucleus pixels were split into texture terciles (smoothest/mid/most textured
third, by local DAPI variance). smooth_minus_textured = mean attribution in the smoothest
third minus the most textured third, patient-level median across cells (n = {n_by_g}). More
negative values mean the suppressing signal concentrates more strongly on smooth chromatin.
Kruskal-Wallis across 4 groups: H={kw_stat:.2f}, p={kw_p:.2e}. Pairwise Mann-Whitney U,
Benjamini-Hochberg FDR correction (q<0.05) applied jointly across all 6 tests.

{res.to_string(index=False)}
"""
    open(f"{OUT}/TABLE_texture_tercile_direction.md", "w").write(md)
    print("\nwrote table_texture_tercile_direction.csv, table_texture_tercile_summary.csv, "
          "TABLE_texture_tercile_direction.md", flush=True)

    # ---- Table S5.xlsx, same styling as Table_S1.xlsx/Table_S2.xlsx ----
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    FONT_NAME = "Arial"
    HEADER_FILL = PatternFill("solid", fgColor="1F3864")
    HEADER_FONT = Font(name=FONT_NAME, size=10, bold=True, color="FFFFFF")
    BODY_FONT = Font(name=FONT_NAME, size=10)
    TITLE_FONT = Font(name=FONT_NAME, size=11, bold=True)
    CAPTION_FONT = Font(name=FONT_NAME, size=10)
    SIG_FILL = PatternFill("solid", fgColor="D9EAD3")
    NS_FILL = PatternFill("solid", fgColor="F4F4F4")
    THIN = Side(style="thin", color="B7B7B7")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    wb = Workbook()
    ws = wb.active
    ws.title = "Table S5"
    ws.sheet_view.showGridLines = False

    title = ("Table S5. Pixel-tercile direction check: does the suppressing signal concentrate "
              "on smooth or textured chromatin?")
    caption = (f"For each cell, nucleus pixels were split into texture terciles (smoothest/mid/"
               f"most textured third, by local DAPI variance). smooth_minus_textured = mean "
               f"attribution in the smoothest third minus the most textured third, patient-level "
               f"median across cells ({', '.join(f'{g} n={n_by_g[g]}' for g in GROUP_ORDER)}). "
               f"More negative values mean the suppressing signal concentrates more strongly on "
               f"smooth chromatin. Kruskal-Wallis across four groups: H={kw_stat:.2f}, "
               f"p={kw_p:.2e} (Figure 6E, second panel). Two-sided Mann-Whitney U tests, "
               f"Benjamini-Hochberg FDR correction (q<0.05) applied jointly across all "
               f"{len(res)} pairwise tests.")

    display_cols = ["group1", "group2", "n1", "n2", "median1", "median2", "p_raw", "q_bh", "significant_q05"]
    d = res[display_cols].copy()
    ncols = len(display_cols)

    tc = ws.cell(row=1, column=1, value=title)
    tc.font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    tc.alignment = Alignment(wrap_text=True, vertical="top")

    cc = ws.cell(row=2, column=1, value=caption)
    cc.font = CAPTION_FONT
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    cc.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 16 * (3 + len(caption) // 110)

    header_row = 4
    for j, col in enumerate(display_cols, start=1):
        c = ws.cell(row=header_row, column=j, value=col)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = BORDER

    for i, (_, row) in enumerate(d.iterrows(), start=header_row + 1):
        fill = SIG_FILL if row["significant_q05"] else NS_FILL
        for j, col in enumerate(display_cols, start=1):
            v = row[col]
            cell = ws.cell(row=i, column=j)
            if col in ("median1", "median2"):
                cell.value = v
                cell.number_format = "+0.000;-0.000;0.000"
            elif col in ("p_raw", "q_bh"):
                cell.value = v
                cell.number_format = "0.00E+00"
            elif col == "significant_q05":
                cell.value = "TRUE" if v else "FALSE"
            elif col in ("n1", "n2"):
                cell.value = int(v)
            else:
                cell.value = v
            cell.font = BODY_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = BORDER
            cell.fill = fill

    for j, col in enumerate(display_cols, start=1):
        max_len = d[col].map(lambda v: len(str(v))).max()
        ws.column_dimensions[get_column_letter(j)].width = max(len(col), max_len) + 4
    ws.freeze_panes = f"A{header_row + 1}"

    wb.save(f"{OUT}/Table_S5.xlsx")
    print("wrote Table_S5.xlsx", flush=True)

    # ---- Figure 6E, second panel: smooth_minus_textured by group ----
    # same figure size, boxplot style, and title convention as the single_panels
    # loop in patient_level_texture_panel.py (Figure 6E/6F), so this sits as a
    # visually matching second panel rather than a differently-proportioned plot.
    from scipy.stats import mannwhitneyu
    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"
    _, p_ctl_aml = mannwhitneyu(groups["Control"][key], groups["AML"][key], alternative="two-sided")

    fig, ax = plt.subplots(1, 1, figsize=(FW * 0.55, FW * 0.42))
    data = [groups[g][key].dropna() for g in GROUP_ORDER]
    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5,
                     tick_labels=[f"{g}\n(n={len(groups[g])})" for g in GROUP_ORDER],
                     medianprops=dict(color="black", lw=2))
    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.6)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_title(f"Suppressing signal: smooth minus textured chromatin\n"
                 f"Kruskal-Wallis p={kw_p:.1e}  |  Control vs AML {_stars(p_ctl_aml)}",
                 fontweight="bold", fontsize=8)
    ax.set_ylabel("Attribution in smoothest third\nminus most textured third", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_6E2_texture_tercile_direction.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_6E2_texture_tercile_direction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_6E2_texture_tercile_direction.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
