"""Build Supplementary Table S6: robustness of AINU to exclusion of the two re-annotated samples (AML12, AML14).

Part A  test performance of the five retrained AML-vs-Control split models    <- results/4-CT-AML-PB-BM/*_split{1-5}.log
Part B  per-patient % of cells predicted as AML (mean +- SD over the 5 models) <- results/patient_scoring/patient_pct_aml_new_models.csv
Output  Paper Tables/Table_S6.xlsx 
"""
import os
import re
from pathlib import Path
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(os.environ.get("AINU_ROOT", "."))
LOGS = ROOT / "results" / "4-CT-AML-PB-BM"
PATIENTS = ROOT / "results" / "patient_scoring" / "patient_pct_aml_new_models.csv"
OUT = ROOT / "Paper Tables" / "Table_S6.xlsx"
assert not OUT.exists(), f"{OUT} already exists - not overwriting"

# ---------- Part A: parse the five training logs
rows_a = []
for k in range(1, 6):
    txt = (LOGS / f"corrected_version_wo_AML12_14_split{k}.log").read_text().splitlines()
    n = {}
    for line in txt:
        m = re.match(r"(Training|Validation|Test) set: (\[.*\])\s*$", line.strip())
        if m and m.group(1) not in n:
            n[m.group(1)] = m.group(2).count("', '") + 1
    body = "\n".join(txt)
    acc = float(re.search(r"Test Accuracy: ([\d.]+)", body).group(1))
    f1w = float(re.search(r"weighted ([\d.]+)", body).group(1))
    auc = float(re.search(r"ROC AUC score: ([\d.]+)", body).group(1))
    neg = re.search(r"negative\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", body)
    pos = re.search(r"positive\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", body)
    rows_a.append([k, n["Training"], n["Validation"], n["Test"], acc, f1w, auc,
                   float(neg.group(2)), float(pos.group(2)), int(neg.group(4)), int(pos.group(4))])
A_cols = ["Split", "Training cells", "Validation cells", "Test cells", "Accuracy", "Weighted F1", "ROC AUC",
          "Control recall", "AML recall", "Test Control cells", "Test AML cells"]
A = pd.DataFrame(rows_a, columns=A_cols)

# ---------- Part B: per-patient table
P = pd.read_csv(PATIENTS)
sp = [f"pct_AML_split{k}" for k in range(1, 6)]
def status(r):
    if r["panel"] == "Fig2A_preleukemia":
        return "CEL (re-annotated)" if r["patient"] == "AML12" else r["disease_type"]
    return "MRD+ (re-annotated; category pending)" if r["patient"] == "AML14" else r["MRD"]
P["status"] = P.apply(status, axis=1)
B_cols = ["Patient", "Sample ID", "Status", "Source", "Cells", "Split 1 (%)", "Split 2 (%)", "Split 3 (%)", "Split 4 (%)", "Split 5 (%)", "Mean (%)", "SD (%)"]
def block(df):
    df = df.sort_values("pct_AML_mean", ascending=False)
    return [[r["patient"], r["patientID"], r["status"], r["source"], int(r["n_cells"])] + [r[c] for c in sp] + [r["pct_AML_mean"], r["pct_AML_sd"]] for _, r in df.iterrows()]
reann = P["patient"].isin(["AML12", "AML14"])
pre = block(P[(P["panel"] == "Fig2A_preleukemia") & ~reann])
mrd = block(P[(P["panel"] == "Fig3A_MRD") & ~reann])
rea = block(P[reann])

# ---------- workbook
wb = Workbook(); ws = wb.active; ws.title = "Table S6"
bold = Font(bold=True); head_fill = PatternFill("solid", fgColor="E8E8E8"); sub_fill = PatternFill("solid", fgColor="F5F5F5")
NCOL = max(len(A_cols), len(B_cols))
def merged_text(text, bold_=False, height=None):
    r = ws.max_row + 1
    ws.cell(r, 1, text).alignment = Alignment(wrap_text=True, vertical="top")
    ws.cell(r, 1).font = Font(bold=bold_)
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=NCOL)
    if height: ws.row_dimensions[r].height = height
def header(cols):
    r = ws.max_row + 1
    for j, c in enumerate(cols, 1):
        cell = ws.cell(r, j, c); cell.font = bold; cell.fill = head_fill; cell.alignment = Alignment(wrap_text=True, horizontal="center")
def add(row, fmts, fill=None, bold_=False):
    r = ws.max_row + 1
    for j, (v, f) in enumerate(zip(row, fmts), 1):
        cell = ws.cell(r, j, None if (isinstance(v, float) and np.isnan(v)) else v)
        if f: cell.number_format = f
        if fill: cell.fill = fill
        if bold_: cell.font = bold
def subhead(text):
    r = ws.max_row + 1
    ws.cell(r, 1, text).font = bold
    for j in range(1, len(B_cols) + 1): ws.cell(r, j).fill = sub_fill

ws.append([]); ws.delete_rows(1)
merged_text("Table S6. Robustness of AINU to exclusion of two samples re-annotated after the analysis (AML12, AML14).", True)
merged_text("Five AML-versus-control models (splits 1-5) were retrained without AML12 (chronic eosinophilic leukemia at sampling, evolved to AML four months later) "
            "and AML14 (MRD-positive sample, not diagnosis), which had been part of the AML class of the original models. These two samples are absent from the "
            "training, validation and test sets of the retrained models. (A) Test performance of each retrained model; recall = fraction of test cells of that class "
            "classified correctly; mean and s.d. are over the five splits. (B) For each patient and each retrained model, the percentage of cells predicted as AML "
            "(P(AML) > 0.5); mean and s.d. over the five models (the dot and error bar of Figure 2A and 3A). AML12 and AML14 are listed separately.", height=95)
ws.append([])
merged_text("A. AML-versus-control test performance of the retrained models", True)
header(A_cols)
fmtA = [None, "0", "0", "0", "0.000", "0.000", "0.000", "0.000", "0.000", "0", "0"]
for r in rows_a: add(r, fmtA)
num = A[["Accuracy", "Weighted F1", "ROC AUC", "Control recall", "AML recall"]]
add(["Mean", None, None, None] + list(num.mean()) + [None, None], fmtA, fill=sub_fill, bold_=True)
add(["s.d.", None, None, None] + list(num.std(ddof=1)) + [None, None], fmtA, fill=sub_fill, bold_=True)
ws.append([])
merged_text("B. Per-patient percentage of cells predicted as AML (retrained models)", True)
header(B_cols)
fmtB = [None, None, None, None, "0"] + ["0.0"] * 7
subhead("Pre-leukemia (MDS, CMML, CCUS), n = %d" % len(pre))
for r in pre: add(r, fmtB)
subhead("MRD cohort (post-treatment samples), n = %d" % len(mrd))
for r in mrd: add(r, fmtB)
subhead("Samples re-annotated after the analysis (excluded from training of these models)")
for r in rea: add(r, fmtB)
widths = [34, 30, 28, 9, 9, 12, 12, 12, 12, 12, 11, 9]
for j, w in enumerate(widths, 1): ws.column_dimensions[get_column_letter(j)].width = w
wb.save(OUT)
print("saved", OUT)
print(A.round(4).to_string(index=False))
print("mean accuracy %.4f  s.d. %.4f | mean AUC %.4f  s.d. %.4f" % (A["Accuracy"].mean(), A["Accuracy"].std(ddof=1), A["ROC AUC"].mean(), A["ROC AUC"].std(ddof=1)))
print("patients: pre-leukemia", len(pre), "| MRD", len(mrd), "| re-annotated", len(rea))
