"""Build the ring-wise statistics behind the asterisks of Figure 6D (Table S1, Part A).

Rebuilds `df` with the data-preparation cells of occlusion_feature_analysis_AML_MDS_CT.ipynb (the notebook that draws Figure 6D),
checks that the ring means reproduce the saved 02a_ring_profiles_all_classes_{0,1,2}.csv, then repeats the notebook's test:
  AML vs Control, two-sided Mann-Whitney U on patient-level medians, Benjamini-Hochberg across the 8 rings within each panel.

Output  Paper Tables/Table_S1_partA_ring_stats.csv 
        Paper Tables/Table_S1_combined.xlsx         
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(os.environ.get("AINU_ROOT", "."))
NB = Path(__file__).resolve().parents[2] / "notebooks" / "occlusion_feature_analysis_AML_MDS_CT.ipynb"
TABLES = ROOT / "Paper Tables"
FIGDIR = ROOT / "results/occlusion_analysis/ainu_split3_model_best.pth.tar/MDS_AML_occlusion_analysis_figures"
OUT_CSV = TABLES / "Table_S1_partA_ring_stats.csv"
OUT_XLSX = TABLES / "Table_S1_combined.xlsx"
assert not OUT_XLSX.exists(), f"{OUT_XLSX} already exists - not overwriting"

# ---- rebuild df exactly as in the notebook (data-preparation cells only, no plotting/saving)
cells = json.load(open(NB))["cells"]
ns = {}
for i in (1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 17):
    exec("".join(cells[i]["source"]).replace("/path/to/ainu_root", str(ROOT)), ns)
df = ns["df"]
pat_col = "patientID_x"
N = 8
panels = {"Total attribution (supporting - suppressing)": [f"attr_ring0{i}_mean" for i in range(N)],
          "AML-supporting attribution": [f"attr_pos_ring0{i}_mean" for i in range(N)],
          "AML-suppressing attribution (magnitude)": [f"attr_neg_ring0{i}_mean" for i in range(N)]}
labels = ["R0 (boundary)"] + [f"R{i}" for i in range(1, 7)] + ["R7 (centre)"]

# ---- check against the saved Figure 6D data (patient means, all four groups)
for k, (name, cols) in enumerate(panels.items()):
    ref = pd.read_csv(FIGDIR / f"02a_ring_profiles_all_classes_{k}.csv")
    for cls in ["AML", "Control", "Preleukemia", "MRD"]:
        pm = df[df["_class"] == cls].dropna(subset=cols).groupby(pat_col)[cols].mean()
        mine = pm.mean().values                       # the saved csv holds the unsigned (magnitude) values
        saved = ref[ref["class"] == cls]["mean"].values
        assert len(saved) == N and np.allclose(mine, saved, atol=1e-6), (name, cls, np.abs(mine - saved).max())
print("ring means reproduce the saved Figure 6D data for all 4 groups x 3 panels")

# ---- the test behind the stars
def star(q): return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"
rows = []
for name, cols in panels.items():
    a = df[df["_class"] == "AML"].groupby(pat_col)[cols].median()
    c = df[df["_class"] == "Control"].groupby(pat_col)[cols].median()
    res = [mannwhitneyu(a[col].dropna(), c[col].dropna(), alternative="two-sided") for col in cols]
    p = np.array([r.pvalue for r in res])
    q = multipletests(p, method="fdr_bh")[1]
    for j, col in enumerate(cols):
        rows.append([name, labels[j], a[col].notna().sum(), c[col].notna().sum(), a[col].median(), c[col].median(),
                     res[j].statistic, p[j], q[j], star(q[j])])
A = pd.DataFrame(rows, columns=["panel", "ring", "n_AML", "n_Control", "median_AML", "median_Control", "U", "p_raw", "q_bh", "sig"])
A.to_csv(OUT_CSV, index=False)
pd.set_option("display.width", 220)
print(A.assign(median_AML=A.median_AML.round(3), median_Control=A.median_Control.round(3)).to_string(index=False))

# ---- workbook: Part A (new) + Part B (existing Table S1, copied as is)
src = openpyxl.load_workbook(TABLES / "Table_S1.xlsx", data_only=True).active
B = [list(r) for r in src.iter_rows(values_only=True) if any(v is not None for v in r)]
B_title, B_desc, B_header, B_rows = B[0][0], B[1][0], B[2], B[3:]
NCOL = 10
wb = Workbook(); ws = wb.active; ws.title = "Table S1"
bold = Font(bold=True); head = PatternFill("solid", fgColor="E8E8E8")
def text(t, b=False, h=None):
    r = ws.max_row + 1; ws.cell(r, 1, t).alignment = Alignment(wrap_text=True, vertical="top"); ws.cell(r, 1).font = Font(bold=b)
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=NCOL)
    if h: ws.row_dimensions[r].height = h
def header(cols):
    r = ws.max_row + 1
    for j, v in enumerate(cols, 1):
        c = ws.cell(r, j, v); c.font = bold; c.fill = head; c.alignment = Alignment(wrap_text=True, horizontal="center")
ws.append([]); ws.delete_rows(1)
text("Table S1. Ring-wise and whole-nucleus attribution statistics across disease stages, related to Figure 6 and Figure S5.", True)
text("(A) Ring-wise comparison of AML (n = 33 patients) versus Control (n = 15 patients), which provides the asterisks in Figure 6D. For each of the eight concentric nuclear rings "
     "(R0 boundary to R7 centre), the patient-level median of the mean per-pixel attribution was compared between AML and Control with two-sided Mann-Whitney U tests. "
     "Benjamini-Hochberg FDR correction was applied across the eight rings, separately for each of the three attribution types. Suppressing attribution is given as magnitude. "
     "*** q < 0.001, ** q < 0.01, * q < 0.05, ns = not significant. Pre-leukemia and MRD are not tested in Figure 6D. n denotes patients.", h=88)
ws.append([])
text("A. Ring-wise attribution, AML versus Control", True)
header(["Attribution", "Ring", "n AML", "n Control", "Median AML", "Median Control", "U", "p (raw)", "q (BH)", "Significance"])
for _, r in A.iterrows():
    row = ws.max_row + 1
    for j, (v, f) in enumerate(zip(r.tolist(), [None, None, "0", "0", "0.000", "0.000", "0.0", "0.00E+00", "0.00E+00", None]), 1):
        c = ws.cell(row, j, v)
        if f: c.number_format = f
ws.append([])
text("B. " + B_title.replace("Table S1. ", "").replace("Pairwise comparisons of", "Pairwise comparisons of") , True)
desc = B_desc.replace("(Figure 6F)", "(Figure S5E)")
text("(B) " + desc, h=75)
header(B_header)
for r in B_rows:
    row = ws.max_row + 1
    for j, v in enumerate(r, 1):
        c = ws.cell(row, j, v)
        if j in (6, 7): c.number_format = "0.0"
        if j in (8, 9): c.number_format = "0.00E+00"
for j, w in enumerate([34, 16, 8, 10, 12, 14, 8, 11, 11, 12], 1): ws.column_dimensions[get_column_letter(j)].width = w
wb.save(OUT_XLSX)
print("saved", OUT_XLSX)
