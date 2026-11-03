"""
Standalone, reproducible pipeline for Table S1: patient-level AML-supporting and
AML-suppressing attribution mass, compared across Control, Preleukemia, MRD, AML.

Reuses build_df() from ring_density_attribution_correlation.py (same 4-group, cell-level
dataframe), which already carries attr_pos_sum and attr_neg_sum (per-cell sum of
positive / |negative| occlusion attribution over the whole nucleus -- "total strength of
AML-supporting / AML-suppressing attribution" as described in the text) straight from the
merged feature CSVs. Patient-level values are the per-patient median across that patient's
test cells (avoids pseudo-replication), matching the original notebook analysis
(occlusion_feature_analysis_AML_MDS_CT.ipynb, cell 58, "Fig 6E: whole-cell AML-class
attribution masses") and reproducing its Kruskal-Wallis values exactly (supporting
p=7.115e-09, suppressing p=3.705e-06, both already quoted in the text).

Difference from the original notebook cell: BH-FDR here is applied ONCE, jointly, across
all 12 pairwise tests (2 features x 6 group pairs), matching how Table S4 pools its pairwise
family, instead of two separate 6-test families (one per feature) as the notebook did. This
changes q-values but not which comparisons were already reported as significant in the text.

Outputs (this dir):
  table_S1_attribution_mass_pairwise.csv
  TABLE_S1.md
  Table_S1.xlsx
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu

from ring_density_attribution_correlation import build_df, GROUP_ORDER

OUT = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["attr_pos_sum", "attr_neg_sum"]
FEATURE_LABEL = {"attr_pos_sum": "AML-supporting mass", "attr_neg_sum": "AML-suppressing mass"}


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


def _sig_stars(q):
    if q < 0.001: return "***"
    if q < 0.01: return "**"
    if q < 0.05: return "*"
    return "ns"


def main():
    df = build_df()
    all_classes = [c for c in GROUP_ORDER if c in df["_class"].unique()]

    pat = (df.dropna(subset=FEATURES + ["patientID_x"])
             .groupby(["patientID_x", "_class"])[FEATURES]
             .median()
             .reset_index())
    n_by_cls = {c: (pat["_class"] == c).sum() for c in all_classes}
    print("patients per group:", n_by_cls, flush=True)

    kw = {}
    for feat in FEATURES:
        groups = [pat.loc[pat["_class"] == c, feat].dropna().values for c in all_classes]
        stat, p = kruskal(*groups)
        kw[feat] = dict(H=stat, p=p)
        print(f"{feat}: Kruskal-Wallis H={stat:.2f} p={p:.3e}", flush=True)

    rows = []
    for feat in FEATURES:
        for g1, g2 in combinations(all_classes, 2):
            v1 = pat.loc[pat["_class"] == g1, feat].dropna()
            v2 = pat.loc[pat["_class"] == g2, feat].dropna()
            _, p_raw = mannwhitneyu(v1, v2, alternative="two-sided")
            rows.append(dict(feature=FEATURE_LABEL[feat], group1=g1, group2=g2,
                              n1=len(v1), n2=len(v2),
                              median1=v1.median(), median2=v2.median(), p_raw=p_raw))

    res_df = pd.DataFrame(rows)
    reject, q = bh_fdr(res_df["p_raw"].values)
    res_df["q_bh"] = q
    res_df["significant_q05"] = reject
    res_df["sig"] = res_df["q_bh"].map(_sig_stars)
    res_df = res_df.sort_values("p_raw").reset_index(drop=True)
    res_df.to_csv(f"{OUT}/table_S1_attribution_mass_pairwise.csv", index=False)

    n_bh = int(reject.sum())
    n_raw = int((res_df["p_raw"] < 0.05).sum())
    print(f"\n{len(res_df)} tests total | uncorrected p<0.05: {n_raw} | joint BH-FDR q<0.05: {n_bh}", flush=True)
    print(res_df.to_string(index=False), flush=True)

    # ---- markdown ----
    def _md(d):
        cols = ["feature", "group1", "group2", "n1", "n2", "median1", "median2", "p_raw", "q_bh", "sig"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in d.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if c in ("median1", "median2"):
                    cells.append(f"{v:.1f}")
                elif c in ("p_raw", "q_bh"):
                    cells.append(f"{v:.2e}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    n_str = ", ".join(f"{c} (n={n_by_cls[c]})" for c in all_classes)
    table_md = f"""**Table S1. Pairwise comparisons of patient-level AML-supporting and AML-suppressing attribution mass.**

Patient-level median attribution mass (sum of positive / |negative| per-pixel occlusion
attribution over the whole nucleus, median across each patient's test cells) compared
across four groups: {n_str}. Kruskal-Wallis omnibus test across all four groups: AML-supporting
mass p={kw['attr_pos_sum']['p']:.2e}, AML-suppressing mass p={kw['attr_neg_sum']['p']:.2e}
(both cited in the text; Figure 6F). Two-sided Mann-Whitney U tests compare each pair of
groups; Benjamini-Hochberg FDR correction (q<0.05) was applied jointly across all
{len(res_df)} pairwise tests (2 features x 6 group pairs). *** q<0.001, ** q<0.01, * q<0.05,
ns = not significant. n1/n2 and median1/median2 refer to group1/group2 respectively.

{n_bh} of {len(res_df)} pairwise comparisons survive BH-FDR correction ({n_raw} at
uncorrected p<0.05).

{_md(res_df)}
"""
    open(f"{OUT}/TABLE_S1.md", "w").write(table_md)

    # ---- xlsx, same styling as Table_S2.xlsx ----
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
    ws.title = "Table S1"
    ws.sheet_view.showGridLines = False

    title = "Table S1. Pairwise comparisons of patient-level AML-supporting and AML-suppressing attribution mass."
    caption = (f"Patient-level median attribution mass compared across four groups ({n_str}). "
               f"Kruskal-Wallis omnibus p: AML-supporting mass {kw['attr_pos_sum']['p']:.2e}, "
               f"AML-suppressing mass {kw['attr_neg_sum']['p']:.2e} (Figure 6F). Two-sided "
               f"Mann-Whitney U tests, Benjamini-Hochberg FDR correction (q<0.05) applied "
               f"jointly across all {len(res_df)} pairwise tests. {n_bh} of {len(res_df)} "
               f"survive correction (vs {n_raw} at uncorrected p<0.05).")

    display_cols = ["feature", "group1", "group2", "n1", "n2", "median1", "median2", "p_raw", "q_bh", "significant_q05"]
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
            if col in ("median1", "median2"):
                cell.value = v
                cell.number_format = "0.0"
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

    wb.save(f"{OUT}/Table_S1.xlsx")
    print("\nwrote TABLE_S1.md, Table_S1.xlsx, table_S1_attribution_mass_pairwise.csv", flush=True)


if __name__ == "__main__":
    main()
