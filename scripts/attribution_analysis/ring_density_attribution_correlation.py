"""
Standalone, reproducible pipeline for Figure S4B and Table S2: per-ring, per-class
Spearman correlation between "relative chromatin density" (DAPI intensity, z-scored
within patient -- pooled mean/SD across all 8 rings, so patient-level exposure/staining
differences are removed while the shape of the radial profile is preserved) and (a) net
occlusion attribution, (b) AML-supporting mass fraction, in each of the 8 rings.

This reimplements, standalone, the exact data-assembly pipeline from
occlusion_feature_analysis_AML_MDS_CT.ipynb (cells 1-18: load both CSVs, fix duplicate
_x/_y columns and the AML_CT-vs-MDS_MDR column-name mismatch, concat, split out the MRD
class, fix patientID_x for Control/AML) and cell 27's bootstrap correlation, but ALSO
records the raw bootstrap replicate distribution per test so an exact two-sided
percentile-bootstrap p-value can be computed and properly BH-FDR corrected across the
full family of tests -- the previous version only saved the 95% CI, so significance was
being called per-test at an uncorrected alpha=0.05, and no p-value existed to correct.

Outputs (this dir):
  table_S2_ring_density_correlation.csv   (long format, one row per class x ring)
  TABLE_S2.md                             (manuscript-ready markdown + caption)
  Table_S2.xlsx                           (formatted Excel, matches Table_S3/S4 style)
  fig_S4B_ring_density_correlation.png/.pdf
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("AINU_ROOT", ".")

CSV_PATH_AML_CT = (_ROOT + "/results/occlusion_analysis/"
                    "ainu_split3_model_best.pth.tar/PadResize128_GD_r03_u4-32_p07_grey_blackbg/"
                    "both/merged_predictions_AML_occlusion_features.csv")
CSV_PATH_MDS_MDR = (_ROOT + "/results/occlusion_analysis/"
                     "ainu_split3_model_best.pth.tar/ainu_split3_MRD_MDS/MDS_AML_occlusion_features.csv")
PATIENT_META_PATH = _ROOT + "/data/human_data/patient_image_data.csv"

COLORS = {"AML": "#B21F2C", "Control": "#2367AC", "Preleukemia": "#F3A582", "MRD": "#93C5DE"}
MARKER_CLASS = {"AML": "^", "Control": "o", "Preleukemia": "s", "MRD": "D"}
GROUP_ORDER = ["Control", "Preleukemia", "MRD", "AML"]

N_RINGS = 8
RING_LABELS = ["R0 (boundary)"] + [f"R{i}" for i in range(1, 7)] + ["R7 (centre)"]
ATTR_RINGS = [f"attr_ring0{i}_mean" for i in range(N_RINGS)]
ATTR_RINGS_POS = [f"attr_pos_ring0{i}_mean" for i in range(N_RINGS)]
ATTR_RINGS_NEG = [f"attr_neg_ring0{i}_mean" for i in range(N_RINGS)]
INT_RINGS = [f"radial_ring0{i}_mean_x" for i in range(N_RINGS)]
N_BOOT = 2000
BOOT_SEED = 42
FW = 20 / 2.54


def build_df():
    """Reproduce occlusion_feature_analysis_AML_MDS_CT.ipynb cells 1-18, standalone."""
    # ---- MDS_MDR source ----
    df = pd.read_csv(CSV_PATH_MDS_MDR)
    patient_meta = pd.read_csv(PATIENT_META_PATH)
    df = df.merge(patient_meta, left_on="image_name", right_on="File", how="left")
    label_col = "new_labels" if "new_labels" in df.columns else "condition"
    df["_class"] = df[label_col].str.strip()

    # drop duplicate _x/_y columns created by the merge
    for col in list(df.columns):
        if col.endswith("_x") and col[:-2] + "_y" in df.columns:
            if df[col].equals(df[col[:-2] + "_y"]):
                df = df.drop(columns=[col[:-2] + "_y"]).rename(columns={col: col[:-2]})

    # INT_RINGS only exist bare (no _x suffix) in this source -- copy across
    int_rings_bare = [c[:-2] if c.endswith("_x") else c for c in INT_RINGS]
    df[INT_RINGS] = df[int_rings_bare]
    df["source_x"] = df["source"]

    # ---- AML_CT source ----
    df_aml_ct = pd.read_csv(CSV_PATH_AML_CT)
    label_col_ct = "new_labels" if "new_labels" in df_aml_ct.columns else "condition"
    df_aml_ct["_class"] = df_aml_ct[label_col_ct].str.strip()
    for col in list(df_aml_ct.columns):
        if col.endswith("_x") and col[:-2] + "_y" in df_aml_ct.columns:
            if df_aml_ct[col].equals(df_aml_ct[col[:-2] + "_y"]):
                df_aml_ct = df_aml_ct.drop(columns=[col[:-2] + "_y"]).rename(columns={col: col[:-2]})

    # ---- concat, dedup, patient id ----
    df = pd.concat([df, df_aml_ct], ignore_index=True)
    df = df.drop_duplicates(["image_name", "_class"])
    df["patientID_x"] = df["patientID"].astype(str)

    # ---- split out MRD (any row with a non-null MRD status) ----
    df_sin_mrd = df[df["MRD"].isna()].copy()
    mrd = df[df["MRD"].notna()].copy()
    mrd["_class"] = "MRD"
    df = pd.concat([df_sin_mrd, mrd], ignore_index=True)

    # ---- patientID_x fix for Control/AML (use patientID.1, the AML_CT-source id) ----
    df.loc[df["_class"] == "Control", "patientID_x"] = df.loc[df["_class"] == "Control", "patientID.1"]
    df.loc[df["_class"] == "AML", "patientID_x"] = df.loc[df["_class"] == "AML", "patientID.1"]

    return df


def ring_bootstrap_r_and_p(x, y, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """Patient-clustered Spearman r, 95% CI, and an exact two-sided percentile-bootstrap
    p-value (fraction of bootstrap replicates crossing zero, doubled), from the SAME
    bootstrap replicate distribution used for the CI -- not a normal approximation."""
    x, y, clusters = np.asarray(x), np.asarray(y), np.asarray(clusters)
    unique_patients = np.unique(clusters)
    if len(x) < 5 or len(unique_patients) < 2:
        return dict(r=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n_patients=len(unique_patients))

    r_obs, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot_r = np.empty(n_boot)
    idx_by_patient = {p: np.where(clusters == p)[0] for p in unique_patients}
    for i in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([idx_by_patient[p] for p in sampled])
        boot_r[i], _ = spearmanr(x[idx], y[idx])

    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    # exact two-sided percentile-bootstrap p-value; floor at 1/n_boot so p is never
    # literally zero (can't resolve finer than the bootstrap resolution)
    p_boot = 2 * min((boot_r <= 0).mean(), (boot_r >= 0).mean())
    p_boot = max(p_boot, 1.0 / n_boot)
    return dict(r=float(r_obs), ci_lo=float(ci_lo), ci_hi=float(ci_hi), p=float(p_boot),
                n_patients=int(len(unique_patients)))


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


def main():
    df = build_df()
    all_classes = [c for c in GROUP_ORDER if c in df["_class"].unique()]
    print(f"classes present: {all_classes}", flush=True)
    for c in all_classes:
        print(f"  {c}: n={len(df[df['_class']==c])} cells, "
              f"{df[df['_class']==c]['patientID_x'].nunique()} patients", flush=True)

    int_rings = [c for c in INT_RINGS if c in df.columns]
    avail_rings = [c for c in ATTR_RINGS if c in df.columns]
    avail_rings_pos = [c for c in ATTR_RINGS_POS if c in df.columns]
    avail_rings_neg = [c for c in ATTR_RINGS_NEG if c in df.columns]
    assert len(int_rings) == len(avail_rings) == N_RINGS, "ring columns not all found"

    # ---- per-patient z-score of DAPI intensity, pooled mean/SD across all 8 rings ----
    pat_mu = df.groupby("patientID_x")[int_rings].transform("mean").mean(axis=1)
    pat_sig = df.groupby("patientID_x")[int_rings].transform("std").mean(axis=1).replace(0, np.nan)
    int_rings_z = [f"{c}_z" for c in int_rings]
    df[int_rings_z] = df[int_rings].sub(pat_mu, axis=0).div(pat_sig, axis=0)

    # NOTE: the positive-mass-fraction target was dropped from this analysis (kept
    # in earlier drafts of this script) -- only net attribution is reported/plotted,
    # since only net attribution is what's actually discussed in the text and cited
    # against Table S2. Keeping an uncited second target in the same table/BH-FDR
    # family would inflate the correction denominator for a claim never made.

    # ---- run the bootstrap for every class x ring (net attribution only) ----
    rows = []
    for cls in all_classes:
        sub = df[df["_class"] == cls]
        for i in range(N_RINGS):
            dapi_col, attr_col = int_rings_z[i], avail_rings[i]
            valid = sub[["patientID_x", dapi_col, attr_col]].dropna()
            res = ring_bootstrap_r_and_p(valid[dapi_col].values, valid[attr_col].values,
                                          valid["patientID_x"].values)
            rows.append(dict(cls=cls, ring=i, ring_label=RING_LABELS[i], **res))
        print(f"  done: {cls}", flush=True)

    res_df = pd.DataFrame(rows)

    # ---- BH-FDR across this family (32 tests: 4 classes x 8 rings) ----
    reject, q = bh_fdr(res_df["p"].values)
    res_df["q_bh"] = q
    res_df["significant_q05"] = reject
    res_df.to_csv(f"{OUT}/table_S2_ring_density_correlation.csv", index=False)

    n_tot = len(res_df)
    n_raw = int((res_df["p"] < 0.05).sum())
    n_bh = int(reject.sum())
    print(f"\n{n_tot} tests total | uncorrected p<0.05: {n_raw} | BH-FDR q<0.05: {n_bh}", flush=True)

    # ---- Figure S4B: single panel, point + 95% CI errorbar, per-class offsets ----
    fig, ax = plt.subplots(1, 1, figsize=(FW * 0.65, FW * 0.42))
    n_cls = len(all_classes)
    offset_step = 0.12
    res_df_sorted = res_df.sort_values("ring")
    for ci, cls in enumerate(all_classes):
        s = res_df_sorted[res_df_sorted["cls"] == cls].sort_values("ring")
        x_off = s["ring"].values + (ci - (n_cls - 1) / 2) * offset_step
        yerr = np.vstack([s["r"] - s["ci_lo"], s["ci_hi"] - s["r"]])
        ax.errorbar(x_off, s["r"], yerr=yerr, fmt=MARKER_CLASS[cls] + ":", color=COLORS[cls],
                    label=cls, lw=1, ms=5, capsize=2, elinewidth=1, zorder=3)
        # mark BH-FDR-significant points with a black dot on top
        sig = s[s["significant_q05"]]
        if len(sig):
            ax.scatter(sig["ring"].values + (ci - (n_cls - 1) / 2) * offset_step, sig["r"],
                       s=14, color="black", zorder=5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axhline(0.3, color="gray", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(-0.3, color="gray", lw=0.6, ls=":", alpha=0.6)
    ax.set_xticks(range(N_RINGS))
    ax.set_xticklabels([l.replace(" ", "\n") for l in RING_LABELS], fontsize=8)
    ax.set_xlabel("Ring")
    ax.set_ylabel("Spearman r\n(chromatin density vs net attribution)")
    ax.set_title("Relative chromatin density x net attribution,\nper ring",
                 fontweight="bold", pad=10)
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right", fontsize=7, markerscale=0.9, framealpha=0.9,
              handlelength=1.5, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_S4B_ring_density_correlation.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_S4B_ring_density_correlation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_S4B_ring_density_correlation.png/.pdf", flush=True)

    # ---- Table S2 markdown + xlsx ----
    def _md_table(d):
        cols = ["cls", "ring_label", "n_patients", "r", "ci_lo", "ci_hi", "p", "q_bh", "significant_q05"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in d.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if c in ("r", "ci_lo", "ci_hi"):
                    cells.append(f"{v:+.3f}")
                elif c in ("p", "q_bh"):
                    cells.append(f"{v:.2e}")
                elif c == "significant_q05":
                    cells.append("TRUE" if v else "FALSE")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    n_pat_by_cls = {c: df[df["_class"] == c]["patientID_x"].nunique() for c in all_classes}
    table_md = f"""**Table S2. Per-ring Spearman correlation between relative chromatin density and net attribution.**

Spearman correlations between mean DAPI intensity (z-scored within patient, pooled mean/SD
across all 8 rings) and mean net occlusion attribution, in each of 8 concentric rings,
computed separately for {", ".join(f"{c} (n={n_pat_by_cls[c]})" for c in all_classes)}.
95% confidence intervals and two-sided p-values were estimated by patient-clustered bootstrap
({N_BOOT} resamples; resampling whole patients, not individual cells). Benjamini-Hochberg FDR
correction (q<0.05) was applied jointly across all {n_tot} tests (4 classes x 8 rings).
significant_q05 = TRUE means q_bh < 0.05.

{n_tot} tests total: {n_raw} significant at uncorrected p<0.05; **{n_bh} significant after BH-FDR
correction**.

{_md_table(res_df)}
"""
    open(f"{OUT}/TABLE_S2.md", "w").write(table_md)

    # ---- xlsx ----
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
    ws.title = "Table S2"
    ws.sheet_view.showGridLines = False

    title = "Table S2. Per-ring Spearman correlation between relative chromatin density and attribution."
    caption = (f"Spearman correlations between mean DAPI intensity (z-scored within patient, pooled "
               f"mean/SD across all 8 rings) and net occlusion attribution / AML-supporting mass fraction, "
               f"per ring, per class ({', '.join(f'{c} n={n_pat_by_cls[c]}' for c in all_classes)}). "
               f"95% CI and p-values from patient-clustered bootstrap ({N_BOOT} resamples). "
               f"Benjamini-Hochberg FDR correction (q<0.05) applied jointly across all {n_tot} tests. "
               f"{n_bh} of {n_tot} survive correction (vs {n_raw} at uncorrected p<0.05).")

    display_cols = ["cls", "ring_label", "n_patients", "r", "ci_lo", "ci_hi", "p", "q_bh", "significant_q05"]
    d = res_df[display_cols].copy()

    ncols = len(display_cols)
    tc = ws.cell(row=1, column=1, value=title)
    tc.font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    tc.alignment = Alignment(wrap_text=True, vertical="top")

    cc = ws.cell(row=2, column=1, value=caption)
    cc.font = CAPTION_FONT
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    cc.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 16 * (2 + len(caption) // 110)

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
            if col in ("r", "ci_lo", "ci_hi"):
                cell.value = v
                cell.number_format = "+0.000;-0.000;0.000"
            elif col in ("p", "q_bh"):
                cell.value = v
                cell.number_format = "0.00E+00"
            elif col == "significant_q05":
                cell.value = "TRUE" if v else "FALSE"
            elif col == "n_patients":
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

    wb.save(f"{OUT}/Table_S2.xlsx")
    print("wrote TABLE_S2.md, Table_S2.xlsx, table_S2_ring_density_correlation.csv", flush=True)


if __name__ == "__main__":
    main()
