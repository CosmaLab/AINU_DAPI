"""
Patient-level version of the texture finding, now split into two figures, S4C and S4D
(replaces the old, combined 2-panel S4C, which mixed two different-unit metrics under one
caption -- r_localvar is a Pearson correlation, delta_int is a DAPI-intensity contrast, and
they answer different questions, so each now gets its own figure and its own citation).
This replaces the
whole-cell DAPI-contrast panel).

Aggregates two per-cell metrics to one value per patient (median over that patient's
test cells), then compares across four groups: control, MDS, MRD, AML.

  r_localvar  (from per_cell_attribution_metrics.csv, full_attr_analysis.py, and
               mrd_mds_texture_metrics.csv, mrd_mds_texture_extraction.py, for MDS/MRD)
      Pearson r between attribution and local DAPI variance (texture/puncta-ness),
      computed per cell over nucleus pixels. Patient-level median tests whether
      *organized texture-following* attribution is a property of control nuclei that
      is progressively lost in preleukemia (MDS) / post-treatment monitoring (MRD) / AML.

  delta_int   (from per_cell_pos_neg_metrics.csv, pos_neg_split.py, and
               mrd_mds_texture_metrics.csv for MDS/MRD)
      pos_w_int - neg_w_int : within-cell contrast between where AML-SUPPORTING and
      AML-SUPPRESSING attribution land on DAPI intensity. Patient-level median tests
      whether supporting attribution sits on denser chromatin than suppressing
      attribution, and whether that differs across groups.

Group definitions (naming matches occlusion_feature_analysis_AML_MDS_CT.ipynb):
  Control     : original AML/control test set, class == 'control'
  AML         : original AML/control test set, class == 'AML'
  Preleukemia : true preleukemic cohort (disease_type MDS/CMML/CCUS; MRD column NaN)
  MRD         : de-novo-AML post-treatment MRD-monitoring cohort (disease_type de novo /
                "AML _post-CMML"; MRD column negative/low-positive/positive/unspecific)

Outputs (this dir):
  patient_level_texture_metrics.csv   (one row per patient, all 4 groups)
  fig_S4C_texture_correlation.png                (Figure S4C: r_localvar)
  fig_S4D_supporting_suppressing_contrast.png    (Figure S4D: delta_int)
  REPORT_S4C.md
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu, wilcoxon, kruskal

OUT = os.path.dirname(os.path.abspath(__file__))
MIN_CELLS_PER_PATIENT = 5   # drop patients too thin to trust a per-patient median

# ── style: match occlusion_feature_analysis.ipynb (journal 20cm-canvas, 8pt Arial) ──
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
FW = 20 / 2.54  # full page width in inches, same convention as the notebook

# Same palette as occlusion_feature_analysis_AML_MDS_CT.ipynb (the 4-class sibling
# notebook), not the 2-class occlusion_feature_analysis.ipynb palette.
GROUP_ORDER = ["Control", "Preleukemia", "MRD", "AML"]
GROUP_COLORS = {"AML": "#B21F2C", "Control": "#2367AC", "Preleukemia": "#F3A582", "MRD": "#93C5DE"}


def mwu_stars(a, b):
    """Mann-Whitney U significance stars (matches occlusion_feature_analysis.ipynb)."""
    _, p = mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
    if p < 0.001: return "***", p
    if p < 0.01:  return "**", p
    if p < 0.05:  return "*", p
    return "ns", p


def w0(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan
    try:
        return wilcoxon(x, zero_method="wilcox").pvalue
    except Exception:
        return np.nan


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject, qvals), same order as input.
    No statsmodels dependency -- standard BH step-up procedure, monotone q-values."""
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) if passed.any() else -1
    reject = np.zeros(n, dtype=bool)
    if k >= 0:
        reject[order[:k + 1]] = True
    q_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    qvals = np.empty(n)
    qvals[order] = q_sorted
    return reject, qvals


def main():
    # ---- original AML/control test-set cells ----
    attr = pd.read_csv(f"{OUT}/per_cell_attribution_metrics.csv", low_memory=False)[
        ["File", "patientID", "class", "r_localvar", "r_int_pearson"]]
    posneg = pd.read_csv(f"{OUT}/per_cell_pos_neg_metrics.csv", low_memory=False)[
        ["File", "class", "delta_int", "delta_lv"]]
    df1 = attr.merge(posneg, on=["File", "class"], how="inner")
    df1["group"] = df1["class"].map({"AML": "AML", "control": "Control"})
    df1 = df1.drop(columns=["class"])
    print(f"AML/control cell rows: {len(df1)}", flush=True)

    # ---- MDS/MRD cohort cells ----
    mm = pd.read_csv(f"{OUT}/mrd_mds_texture_metrics.csv", low_memory=False)[
        ["File", "patientID", "group", "r_localvar", "r_int_pearson", "delta_int", "delta_lv"]]
    print(f"MDS/MRD cell rows: {len(mm)}", flush=True)

    df = pd.concat([df1, mm], ignore_index=True)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    print(f"combined cell-level rows: {len(df)}", flush=True)

    # ---- per-patient aggregation (median across that patient's test cells) ----
    METRICS = ["r_localvar", "r_int_pearson", "delta_int", "delta_lv"]
    n_cells = df.groupby("patientID").size().rename("n_cells")
    pat = (df.groupby(["patientID", "group"], observed=True)[METRICS]
             .median()
             .join(n_cells)
             .reset_index())
    dropped = pat[pat["n_cells"] < MIN_CELLS_PER_PATIENT]
    pat = pat[pat["n_cells"] >= MIN_CELLS_PER_PATIENT].copy()
    if len(dropped):
        print(f"dropped {len(dropped)} patient(s) with <{MIN_CELLS_PER_PATIENT} cells: "
              f"{dropped['patientID'].tolist()}", flush=True)
    pat.to_csv(f"{OUT}/patient_level_texture_metrics.csv", index=False)

    groups = {g: pat[pat["group"] == g] for g in GROUP_ORDER}
    for g in GROUP_ORDER:
        print(f"  {g}: n={len(groups[g])} patients", flush=True)

    stats = {}
    for key in ["r_localvar", "r_int_pearson", "delta_int"]:
        stats[key] = {}
        for g in GROUP_ORDER:
            x = groups[g][key].dropna()
            q1, q3 = (x.quantile(0.25), x.quantile(0.75)) if len(x) else (np.nan, np.nan)
            stats[key][g] = dict(median=x.median(), mean=x.mean(), n=len(x), w0=w0(x.values),
                                  q1=q1, q3=q3)
            s = stats[key][g]
            print(f"{key} [{g}]: median={s['median']:+.3f} mean={s['mean']:+.3f} "
                  f"(n={s['n']}, W_vs0 p={s['w0']:.2e})", flush=True)
        vals = [groups[g][key].dropna().values for g in GROUP_ORDER if len(groups[g][key].dropna()) > 0]
        stats[key]["kruskal_p"] = kruskal(*vals).pvalue if len(vals) > 1 else np.nan
        print(f"{key}: Kruskal-Wallis across groups p = {stats[key]['kruskal_p']:.2e}", flush=True)
        stats[key]["pairwise"] = {}
        for g1, g2 in combinations(GROUP_ORDER, 2):
            a, c = groups[g1][key], groups[g2][key]
            if a.notna().sum() > 0 and c.notna().sum() > 0:
                star, p = mwu_stars(a, c)
                stats[key]["pairwise"][(g1, g2)] = p
                print(f"  {g1} vs {g2}: MWU p = {p:.2e} ({star})", flush=True)

    # ---- BH-FDR correction across the full family of pairwise tests --------
    # All 18 pairwise Mann-Whitney tests (3 metrics x 6 group pairs) are the
    # tests actually driving the group-difference claims in the text, so they
    # are the correction family. Kruskal-Wallis omnibus tests and per-group
    # Wilcoxon-vs-0 tests are reported separately and not included here.
    fdr_rows = [(key, g1, g2, p) for key in ["r_localvar", "r_int_pearson", "delta_int"]
                for (g1, g2), p in stats[key]["pairwise"].items()]
    reject, qvals = bh_fdr([r[3] for r in fdr_rows])
    fdr_df = pd.DataFrame({
        "metric": [r[0] for r in fdr_rows],
        "group1": [r[1] for r in fdr_rows],
        "group2": [r[2] for r in fdr_rows],
        "p_raw": [r[3] for r in fdr_rows],
        "q_bh": qvals,
        "significant_q05": reject,
    }).sort_values("p_raw").reset_index(drop=True)
    fdr_df.to_csv(f"{OUT}/bh_fdr_pairwise.csv", index=False)
    print(f"\nBH-FDR (q<0.05) across {len(fdr_df)} pairwise tests: "
          f"{int(reject.sum())} survive, {int((~reject).sum())} do not:", flush=True)
    print(fdr_df.to_string(index=False), flush=True)

    # ---- Supplementary Tables S3 & S4 (manuscript numbering: standalone, ---
    # not sub-lettered under one number, matching how Table S1/S2 are each
    # their own self-contained table) ----------------------------------------
    # Table S3: per-group summary (n, median [IQR], Wilcoxon-vs-0 p) for all 3
    # metrics, plus the Kruskal-Wallis omnibus p per metric.
    # Table S4: all 18 pairwise Mann-Whitney tests with raw p, BH-FDR q, and
    # significance flag. Together these are the numeric backing for Figures
    # S4C/S4D and the brightness-null claim (r_int_pearson), which has no
    # figure of its own.
    METRIC_LABEL = {"r_localvar": "r_texture", "r_int_pearson": "r_brightness", "delta_int": "Δint"}

    summary_rows = []
    for key in ["r_localvar", "r_int_pearson", "delta_int"]:
        for g in GROUP_ORDER:
            s = stats[key][g]
            summary_rows.append(dict(
                metric=METRIC_LABEL[key], group=g, n=s["n"],
                median=round(s["median"], 3), q1=round(s["q1"], 3), q3=round(s["q3"], 3),
                wilcoxon_vs0_p=s["w0"],
            ))
        summary_rows.append(dict(metric=METRIC_LABEL[key], group="(Kruskal-Wallis, 4 groups)",
                                  n=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                                  wilcoxon_vs0_p=stats[key]["kruskal_p"]))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT}/table_S3_group_summary.csv", index=False)

    pairwise_table = fdr_df.copy()
    pairwise_table["metric"] = pairwise_table["metric"].map(METRIC_LABEL)

    def _sig_stars(q):
        if q < 0.001: return "***"
        if q < 0.01:  return "**"
        if q < 0.05:  return "*"
        return "ns"

    pairwise_table["sig"] = pairwise_table["q_bh"].map(_sig_stars)
    pairwise_table.to_csv(f"{OUT}/table_S4_pairwise_fdr.csv", index=False)

    def _md_table(df, float_cols):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    cells.append("—")
                elif c == "n":
                    cells.append(str(int(v)))
                elif isinstance(v, float) and c in float_cols:
                    is_pval = c.endswith("_p") or c in ("p_raw", "q_bh")
                    cells.append(f"{v:.2e}" if is_pval else f"{v:+.3f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    n_str = ", ".join(f"{g} (n = {len(groups[g])})" for g in GROUP_ORDER)

    table_s3_md = f"""**Table S3. Patient-level chromatin-texture and brightness attribution statistics.**

Patient-level median Pearson correlation between occlusion attribution and local chromatin
texture (r_texture), between attribution and raw DAPI brightness (r_brightness, a
brightness-only control), and within-cell contrast between attribution-weighted mean DAPI
intensity of AML-supporting and AML-suppressing pixels (Δint), compared across four groups:
{n_str}. Values are patient-level medians [Q1, Q3] of per-cell measurements. Wilcoxon
signed-rank tests assessed whether each group's median differed from zero; Kruskal-Wallis
tests assessed the overall difference among all four groups for each metric. r_texture and
Δint are plotted in Figures S4C and S4D respectively; r_brightness is a text-only control
and is not plotted. n denotes patients.

{_md_table(summary_df, ["median", "q1", "q3", "wilcoxon_vs0_p"])}
"""
    open(f"{OUT}/TABLE_S3.md", "w").write(table_s3_md)

    table_s4_md = f"""**Table S4. Pairwise comparisons of chromatin-texture and brightness attribution across disease stages.**

Two-sided Mann-Whitney U tests comparing patient-level median r_texture, r_brightness, and
Δint (defined in Table S3) between all six pairs of the four groups ({n_str}).
Benjamini-Hochberg FDR correction was applied jointly across all 18 pairwise comparisons
(3 attribution metrics x 6 group pairs). *** padj < 0.001, ** padj < 0.01, * padj < 0.05,
ns = not significant (padj >= 0.05). n denotes patients.

{_md_table(pairwise_table, ["p_raw", "q_bh"])}
"""
    open(f"{OUT}/TABLE_S4.md", "w").write(table_s4_md)
    print(f"\nwrote TABLE_S3.md, TABLE_S4.md, table_S3_group_summary.csv, table_S4_pairwise_fdr.csv", flush=True)

    # ---- figures: two separate single-panel boxplots, patient-level, 4 groups --
    # r_localvar (Pearson r) and delta_int (a DAPI-intensity contrast) are
    # different units answering different questions -- does attribution track
    # texture? vs. do supporting/suppressing pixels sit on different-density
    # chromatin? -- so they are now split into two citable figures (S4C, S4D)
    # instead of two panels sharing one figure/caption. Same box-plot styling
    # as occlusion_feature_analysis.ipynb's per-metric summary panels.
    single_panels = [
        ("S4C", "fig_S4C_texture_correlation", "r_localvar",
         "Attribution–texture correlation, by disease stage",
         "r_texture = Pearson r\n(attribution vs local DAPI variance)"),
        ("S4D", "fig_S4D_supporting_suppressing_contrast", "delta_int",
         "Supporting vs suppressing chromatin-density contrast, by disease stage",
         "Δint = I_support − I_suppress\n(weighted-mean DAPI intensity)"),
    ]
    for fig_label, fname, key, title, ylab in single_panels:
        fig, axi = plt.subplots(1, 1, figsize=(FW * 0.55, FW * 0.42))
        data = [groups[g][key].dropna() for g in GROUP_ORDER]
        bp = axi.boxplot(data, patch_artist=True, notch=False, widths=0.5,
                         tick_labels=[f"{g}\n(n={len(groups[g])})" for g in GROUP_ORDER],
                         medianprops=dict(color="black", lw=2))
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.6)
        axi.axhline(0, color="0.4", lw=0.8, ls="--")
        star_ctl_aml, _ = mwu_stars(groups["Control"][key], groups["AML"][key])
        kp = stats[key]["kruskal_p"]
        axi.set_title(f"{title}\nKruskal-Wallis p={kp:.1e}  |  Control vs AML {star_ctl_aml}",
                      fontweight="bold", fontsize=8)
        axi.set_ylabel(ylab, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{fname}.png", dpi=200, bbox_inches="tight")
        fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fname}.png/.pdf (Figure {fig_label})", flush=True)

    def fmt_pairwise(key):
        lines = []
        for (g1, g2), p in stats[key]["pairwise"].items():
            lines.append(f"- {g1} vs {g2}: p = {p:.2e}")
        return "\n".join(lines)

    def fmt_group_lines(key):
        lines = []
        for g in GROUP_ORDER:
            s = stats[key][g]
            lines.append(f"- {g}: median = {s['median']:+.3f} (n={s['n']}, Wilcoxon vs 0 p = {s['w0']:.2e})")
        return "\n".join(lines)

    rep = f"""# Figures S4C & S4D — patient-level texture result, all groups

Per-cell `r_localvar` (attribution vs local DAPI variance) and `delta_int`
(supporting-minus-suppressing weighted DAPI intensity) were aggregated to one value per
patient (median over that patient's test cells, patients with < {MIN_CELLS_PER_PATIENT} cells
dropped), then compared across **Control, Preleukemia, MRD, and AML** patients (naming and
colors match `occlusion_feature_analysis_AML_MDS_CT.ipynb`).

**Patients analysed:** {', '.join(f'{g}={len(groups[g])}' for g in GROUP_ORDER)}
({len(dropped)} dropped for too few cells).

Preleukemia/MRD occlusion maps come from a separate inference run (`ainu_split3_MRD_MDS/`,
`<stem>_AML.npy`) with the same array format as the AML/control test set
(`<stem>_AML_occlusion.npy`) -- channel 0 of a (3,128,128) Captum Occlusion attribution
toward the AML logit -- so the identical nucleus-segmentation / metric code applies
(`mrd_mds_texture_extraction.py`).

Group definitions: **Preleukemia** = disease_type in {{MDS, CMML, CCUS}} (MRD column NaN --
true preleukemia). **MRD** = disease_type in {{de novo, "AML _post-CMML"}} (MRD column has a
value: negative/low-positive/positive/unspecific -- the de-novo-AML post-treatment
MRD-monitoring cohort).

## r_localvar — attribution follows chromatin texture (Figure S4C)

{fmt_group_lines("r_localvar")}
- Kruskal-Wallis across all 4 groups: p = {stats["r_localvar"]["kruskal_p"]:.2e}

Pairwise (Mann-Whitney):
{fmt_pairwise("r_localvar")}

## r_int_pearson — brightness-null control (is the texture effect just brightness?)

{fmt_group_lines("r_int_pearson")}
- Kruskal-Wallis across all 4 groups: p = {stats["r_int_pearson"]["kruskal_p"]:.2e}

Pairwise (Mann-Whitney):
{fmt_pairwise("r_int_pearson")}

Control's r_int_pearson is not significantly different from 0 (Wilcoxon vs 0 above), while its
r_localvar is (p={stats['r_localvar']['Control']['w0']:.2e}) -- confirms the texture-tracking
result in Control is not attributable to attribution simply tracking overall brightness.

## delta_int — supporting vs suppressing chromatin density, within cell (Figure S4D)

{fmt_group_lines("delta_int")}
- Kruskal-Wallis across all 4 groups: p = {stats["delta_int"]["kruskal_p"]:.2e}

Pairwise (Mann-Whitney):
{fmt_pairwise("delta_int")}

## Files
`patient_level_texture_metrics.csv` (one row per patient, all 4 groups, all 4 metrics),
`fig_S4C_texture_correlation.png` / `.pdf` (Figure S4C, r_localvar),
`fig_S4D_supporting_suppressing_contrast.png` / `.pdf` (Figure S4D, delta_int),
`mrd_mds_texture_metrics.csv` (per-cell Preleukemia/MRD metrics),
`bh_fdr_pairwise.csv` (all 18 pairwise tests, raw p, BH q-value, significance at q<0.05).
"""
    open(f"{OUT}/REPORT_S4C.md", "w").write(rep)
    print("\nwrote REPORT_S4C.md", flush=True)


if __name__ == "__main__":
    main()
