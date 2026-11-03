import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import ast, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import albumentations as A
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = os.environ.get("AINU_ROOT", ".")
IMG = f"{ROOT}/data/human_data"
CK = f"{ROOT}/results/4-CT-AML-PB-BM/corrected_version_wo_AML12_14_split3_model_best.pth.tar"
LOG = f"{ROOT}/results/4-CT-AML-PB-BM/corrected_version_wo_AML12_14_split3.log"
AUG = f"{ROOT}/augmentation/test/PadResize128_GD_r03_u4-32_p07_grey_blackbg.json"
OUT = f"{ROOT}/embeddings_with_predictions_newmodel_MRD_MDS.csv"
RANDOM_STATE, PERPLEXITY, PCA_PRE_DIM, HOOK = 42, 30, 50, "norm5"   # same settings as generate_embedding_explorer.ipynb

# ---- partitions of the NEW model
parts = {}
for line in open(LOG):
    m = re.match(r"(Training|Validation|Test) set: (\[.*\])\s*$", line.strip())
    if m and m.group(1) not in parts:
        parts[m.group(1)] = set(ast.literal_eval(m.group(2)))
print({k: len(v) for k, v in parts.items()})

def partition(f):
    return "train" if f in parts["Training"] else "val" if f in parts["Validation"] else "test" if f in parts["Test"] else "not_in_lists"

# ---- cells to embed
feat = pd.read_csv(f"{ROOT}/data/human_data/cell_features.csv", low_memory=False)
mrd_mds = pd.read_csv(f"{ROOT}/data/human_data/MRD_MDS_with_AML12_14.csv", sep="\t", low_memory=False)
ref = feat[feat["File"].isin(parts["Test"])].copy()                       # new model's held-out AML + Control cells
ref["cohort"] = "Reference test (AML/Control)"
mm = mrd_mds.copy(); mm["cohort"] = "MRD/MDS set"
both = pd.concat([mm, ref], ignore_index=True)
both["also_in_reference_test"] = both["File"].isin(ref["File"])
both = both.drop_duplicates("File", keep="first").reset_index(drop=True)  # keep MRD/MDS annotation for duplicates
print("cells to embed:", len(both), both["cohort"].value_counts().to_dict())

def group(r):
    if r["cohort"].startswith("Reference"):
        return f"{r['new_labels']} (reference test)"
    if r["patientID"] == "AML12": return "CEL (AML12)"
    if r["MRD"] == "negative": return "MRD-negative AML"
    if isinstance(r["MRD"], str): return r["MRD"] if r["MRD"] != "MRD+ (category pending)" else "MRD+ AML14 (category pending)"
    return r["disease_type"]
both["group"] = both.apply(group, axis=1)
both["new_model_partition"] = both["File"].map(partition)
both["image_path"] = both["File"].map(lambda f: f"{IMG}/{f}.png")

# ---- model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.densenet121(); model.classifier = nn.Linear(1024, 2)
ck = torch.load(CK, map_location="cpu", weights_only=False)
model.load_state_dict({k.replace("module.", ""): v for k, v in ck["state_dict"].items()})
model = model.to(device).eval()
transform = A.load(AUG)
bank = []
handle = getattr(model.features, HOOK).register_forward_hook(lambda m, i, o: bank.append(o.detach().cpu()))
probs = []
names = both["File"].tolist()
with torch.no_grad():
    for i in range(0, len(names), 32):
        batch = []
        for n in names[i:i + 32]:
            im = cv2.cvtColor(cv2.imread(f"{IMG}/{n}.png"), cv2.COLOR_BGR2RGB)
            batch.append(transform(image=im)["image"] / 255)
        probs.append(torch.softmax(model(torch.stack(batch).float().to(device)), 1).cpu().numpy())
        if (i // 32) % 50 == 0: print(f"  {i}/{len(names)}", flush=True)
handle.remove()
emb = torch.cat(bank).mean(dim=(2, 3)).numpy(); probs = np.concatenate(probs)
print("embeddings:", emb.shape)

# ---- dimensionality reduction (same recipe as the notebook, 2D)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(emb)
pre = PCA(n_components=PCA_PRE_DIM, random_state=RANDOM_STATE).fit_transform(emb)
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=RANDOM_STATE, n_jobs=-1).fit_transform(pre)

# ---- assemble in the original file's column layout
out = pd.DataFrame({
    "File": both["File"], "celltype": both["celltype"], "condition": both["condition"], "new_labels": both["new_labels"],
    "split3": both["new_model_partition"], "image_path": both["image_path"],
    "tsne_1": tsne[:, 0], "tsne_2": tsne[:, 1], "pca_1": pca2[:, 0], "pca_2": pca2[:, 1],
    "pred_label": np.where(probs[:, 1] > 0.5, "AML", "Control"),
    "prob_AML": probs[:, 1], "prob_Control": probs[:, 0],
})
is_ref = (both["cohort"] == "Reference test (AML/Control)").values
truth = both["new_labels"].map({"AML": "AML", "Control": "Control"})
out["correct"] = np.where(is_ref, out["pred_label"] == truth, np.nan)
out["correct_str"] = np.where(is_ref, np.where(out["correct"] == 1, "Correct", "Misclassified"), "")
out["is_misclassified"] = np.where(is_ref, out["correct"] == 0, np.nan)
for c in ["cohort", "group", "patientID", "disease_type", "MRD", "also_in_reference_test"]:
    out[c] = both[c].values
out["new_model_partition"] = both["new_model_partition"].values
out.to_csv(OUT, index=False)
print("saved", OUT, out.shape)

# ---- sanity checks
r = out[is_ref]
print(f"\nSANITY: new model on its {len(r)} reference test cells: accuracy {r['correct'].astype(float).mean():.4f} (training log says 0.9163)")
print("\ncells per group and new-model partition:")
print(pd.crosstab(out["group"], out["new_model_partition"]).to_string())
g = out.groupby("group").agg(n=("File", "size"), pct_pred_AML=("pred_label", lambda s: (s == "AML").mean() * 100), mean_prob_AML=("prob_AML", "mean")).round(3)
print("\n", g.to_string())
