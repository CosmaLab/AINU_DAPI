"""Per-patient % of cells predicted as AML with the 5 retrained split models (4-CT-AML-PB-BM, without AML12/AML14).

Recalculates Fig 2A (preleukemia patients) and Fig 3A (MRD cohort):
  for each patient and each split model: fraction of the patient's cells with P(AML) > 0.5
  -> dot = mean over the 5 split models, error bar = SD over the 5 split models.

Outputs (results/patient_scoring/):
  per_cell_prob_aml_new_models.csv   P(AML) of every cell under each split model
  patient_pct_aml_new_models.csv     per-patient table (per-split values, mean, SD)
  fig2A_preleukemia_new_models.(png|pdf), fig3A_MRD_new_models.(png|pdf)
"""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import albumentations as A
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("AINU_ROOT", "."))
IMG = ROOT / "data" / "human_data"
CELLS = IMG / "MRD_MDS_with_AML12_14.csv"          # MRD + MDS cells, incl. AML12 (CEL) and AML14 (MRD+); tab separated
AUG = ROOT / "augmentation" / "test" / "PadResize128_GD_r03_u4-32_p07_grey_blackbg.json"
OUTDIR = ROOT / "results" / "patient_scoring"
OUTDIR.mkdir(parents=True, exist_ok=True)
SPLITS = [1, 2, 3, 4, 5]
CK = {k: ROOT / f"results/4-CT-AML-PB-BM/corrected_version_wo_AML12_14_split{k}_model_best.pth.tar" for k in SPLITS}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cells = pd.read_csv(CELLS, sep="\t", low_memory=False)
cells = cells[[c for c in ["File", "patient", "patientID", "disease_type", "MRD", "source", "hospital"] if c in cells.columns]].copy()
print(f"{len(cells)} cells, {cells['patient'].nunique()} patients, device {device}", flush=True)

# ---- preprocess every image once (same test pipeline as in training: resize/pad to 128, /255)
transform = A.load(str(AUG))
t0 = time.time()
X = torch.stack([
    (transform(image=cv2.cvtColor(cv2.imread(str(IMG / f"{f}.png")), cv2.COLOR_BGR2RGB))["image"] / 255).half()
    for f in cells["File"]
])
print(f"cached {tuple(X.shape)} in {time.time() - t0:.0f}s", flush=True)

# ---- score with each split model
for k in SPLITS:
    t0 = time.time()
    m = models.densenet121(); m.classifier = nn.Linear(1024, 2)
    ck = torch.load(CK[k], map_location="cpu", weights_only=False)
    m.load_state_dict({n.replace("module.", ""): v for n, v in ck["state_dict"].items()})
    m = m.to(device).eval()
    p = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            p.append(torch.softmax(m(X[i:i + 64].float().to(device)), 1)[:, 1].cpu().numpy())
    cells[f"p_aml_split{k}"] = np.concatenate(p)
    print(f"split{k}: {time.time() - t0:.0f}s, % cells predicted AML overall = {(cells[f'p_aml_split{k}'] > 0.5).mean() * 100:.1f}", flush=True)
cells.to_csv(OUTDIR / "per_cell_prob_aml_new_models.csv", index=False)

# ---- per patient: % predicted AML in each split, then mean +- SD over splits
for k in SPLITS:
    cells[f"pred_aml_split{k}"] = cells[f"p_aml_split{k}"] > 0.5
agg = {f"pct_AML_split{k}": (f"pred_aml_split{k}", lambda s: s.mean() * 100) for k in SPLITS}
tab = cells.groupby("patient").agg(n_cells=("File", "size"), patientID=("patientID", "first"), disease_type=("disease_type", "first"),
                                   MRD=("MRD", "first"), source=("source", "first"), hospital=("hospital", "first"), **agg).reset_index()
split_cols = [f"pct_AML_split{k}" for k in SPLITS]
tab["pct_AML_mean"] = tab[split_cols].mean(axis=1)
tab["pct_AML_sd"] = tab[split_cols].std(axis=1, ddof=1)
tab["panel"] = np.where(tab["MRD"].notna(), "Fig3A_MRD", "Fig2A_preleukemia")
tab = tab.sort_values(["panel", "pct_AML_mean"], ascending=[True, False])
tab.round(2).to_csv(OUTDIR / "patient_pct_aml_new_models.csv", index=False)
pd.set_option("display.width", 200)
print("\n", tab[["panel", "patient", "patientID", "n_cells"] + split_cols + ["pct_AML_mean", "pct_AML_sd"]].round(1).to_string(index=False))

# ---- plots in the style of Fig 2A / 3A
markers = {"BM": "o", "PB": "^"}
hosp = sorted(tab["hospital"].dropna().unique())
cmap = dict(zip(hosp, plt.cm.tab10.colors))


def dotplot(d, title, fname):
    d = d.sort_values("pct_AML_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.2, 0.28 * len(d) + 1.4))
    for i, r in d.iterrows():
        ax.errorbar(r["pct_AML_mean"] / 100, i, xerr=r["pct_AML_sd"] / 100, fmt=markers.get(r["source"], "o"),
                    color=cmap.get(r["hospital"], "k"), ecolor="k", elinewidth=0.9, capsize=2, ms=7, zorder=3)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["patient"], fontsize=8)
    ax.axvline(0.5, color="grey", lw=0.6, ls="--", zorder=1)
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy of prediction as AML\n(mean ± SD over 5 split models)")
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(axis="x", alpha=0.3)
    h = [plt.Line2D([], [], marker="o", ls="", color=cmap[x], label=x) for x in hosp] + \
        [plt.Line2D([], [], marker=m, ls="", color="grey", label=s) for s, m in markers.items()]
    ax.legend(handles=h, fontsize=6, loc="lower right", frameon=True)
    fig.tight_layout(); fig.savefig(OUTDIR / f"{fname}.png", dpi=200); fig.savefig(OUTDIR / f"{fname}.pdf"); plt.close(fig)


dotplot(tab[tab["panel"] == "Fig2A_preleukemia"], "Control vs AML (preleukemia)", "fig2A_preleukemia_new_models")
dotplot(tab[tab["panel"] == "Fig3A_MRD"], "Control vs AML (MRD)", "fig3A_MRD_new_models")
print("\nsaved to", OUTDIR)
