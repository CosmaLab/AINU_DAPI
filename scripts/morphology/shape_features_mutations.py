"""
AINU chromatin features vs somatic mutations in AML / pre-leukemic patients.
Two-level correlative analysis:
  (A) Per-patient: Mann-Whitney U on patient-mean features, BH-FDR, rank-biserial + Cohen's d.
  (B) Per-cell: linear mixed-effects (feature ~ mutation + age + sex + disease_type + (1|patient)).
Plus AML-score x mutation tests and full QC.
Requires: pandas, numpy, scipy, statsmodels, seaborn, matplotlib.
"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import seaborn as sns, matplotlib.pyplot as plt

# ---------- 0. CONFIG ----------
CSV = "data/human_data/cell_features.csv"
ID_COLS = ["cell_id","patient_id","hospital","age","sex","disease_type",
           "blast_percentage","AML_probability","AML_status"]
MUTATIONS = ["NPM1","FLT3","DNMT3A","TET2","TP53","RUNX1","SRSF2","ASXL1"]
PRIORITY  = ["NPM1","FLT3","DNMT3A","TET2","TP53","RUNX1","ASXL1","SRSF2"]
ALPHA = 0.05

df = pd.read_csv(CSV)

# ---------- 1. QC & FILTERING ----------
df = df.drop_duplicates(keep="first")                      # remove duplicate cell rows
feature_cols = [c for c in df.columns if c not in ID_COLS and c not in MUTATIONS]
#assert len(feature_cols) == 298, f"Expected 298 features, found {len(feature_cols)}"

# missingness reports
miss_by_feature  = df[feature_cols].isna().mean().sort_values(ascending=False)
miss_by_patient  = df.groupby("patient_id")[feature_cols].apply(lambda d: d.isna().mean().mean())
miss_by_hospital = df.groupby("hospital")[feature_cols].apply(lambda d: d.isna().mean().mean())

# AML probability sanity: must be in [0,1]; drop rows missing it
assert df["AML_probability"].between(0,1).all(), "AML_probability outside [0,1]!"
df = df.dropna(subset=["AML_probability"])

# sample-size summaries
n_cells, n_patients = len(df), df["patient_id"].nunique()
cells_per_patient = df.groupby("patient_id").size()
patients_per_mut = {m: df.groupby("patient_id")[m].first().value_counts(dropna=True).to_dict()
                    for m in MUTATIONS}

# ---------- 2. OUTCOME VARIABLES ----------
# per-patient AML score (aggregate single-cell probabilities)
pat = (df.groupby("patient_id")
         .agg(AML_mean=("AML_probability","mean"),
              AML_median=("AML_probability","median"),
              AML_sd=("AML_probability","std"),
              age=("age","first"), sex=("sex","first"),
              disease_type=("disease_type","first"),
              blast=("blast_percentage","first")))
for m in MUTATIONS:
    pat[m] = df.groupby("patient_id")[m].first()          # patient-level mutation status
pat_feat = df.groupby("patient_id")[feature_cols].mean()  # patient-mean of each feature

# ---------- 3+4a. PER-PATIENT FULL-PANEL ANALYSIS ----------
def rank_biserial(x, y):
    U,_ = mannwhitneyu(x, y, alternative="two-sided")
    return 1 - 2*U/(len(x)*len(y))           # = 2*AUC-1 form
def cohens_d(x, y):
    nx,ny=len(x),len(y); sp=np.sqrt(((nx-1)*x.var(ddof=1)+(ny-1)*y.var(ddof=1))/(nx+ny-2))
    return (x.mean()-y.mean())/sp if sp>0 else np.nan

rows=[]
for m in PRIORITY:
    grp = pat[m].dropna()
    mut_ids = grp[grp==1].index; wt_ids = grp[grp==0].index
    if len(mut_ids)<2 or len(wt_ids)<2: continue
    for f in feature_cols:
        a = pat_feat.loc[mut_ids, f].dropna(); b = pat_feat.loc[wt_ids, f].dropna()
        if len(a)<2 or len(b)<2: continue
        try: U,p = mannwhitneyu(a,b,alternative="two-sided")
        except ValueError: continue
        rows.append(dict(mutation=m, feature=f, n_mut=len(a), n_wt=len(b),
                         U=U, p_raw=p, rank_biserial=rank_biserial(a,b),
                         cohens_d=cohens_d(a,b), dir=np.sign(a.mean()-b.mean())))
pp = pd.DataFrame(rows)
pp["p_adj"] = multipletests(pp["p_raw"], method="fdr_bh")[1]   # BH across ALL tests
pp_sig = pp[pp["p_adj"]<ALPHA].sort_values("p_adj")

# ---------- 4b. PER-CELL MIXED-EFFECTS ANALYSIS ----------
cell_rows=[]
for m in PRIORITY:
    sub = df[df[m].notna()].copy()
    if sub["patient_id"].nunique()<4 or sub[m].nunique()<2: continue
    for f in feature_cols:
        d = sub[[f,m,"age","sex","disease_type","patient_id"]].dropna()
        if d["patient_id"].nunique()<4: continue
        d["_y"] = (d[f]-d[f].mean())/d[f].std(ddof=0)     # z-score outcome for comparability
        try:
            md = smf.mixedlm(f"_y ~ C({m}) + age + C(sex) + C(disease_type)",
                             d, groups=d["patient_id"]).fit(reml=False, method="lbfgs")
            coef_name = [c for c in md.params.index if c.startswith(f"C({m})")][0]
            cell_rows.append(dict(mutation=m, feature=f,
                coef=md.params[coef_name], se=md.bse[coef_name],
                t=md.tvalues[coef_name], p_raw=md.pvalues[coef_name],
                n_cells=len(d), n_patients=d["patient_id"].nunique()))
        except Exception as e:
            cell_rows.append(dict(mutation=m, feature=f, coef=np.nan, se=np.nan,
                                  t=np.nan, p_raw=np.nan, n_cells=len(d),
                                  n_patients=d["patient_id"].nunique(), error=str(e)))
pc = pd.DataFrame(cell_rows).dropna(subset=["p_raw"])
pc["p_adj"] = multipletests(pc["p_raw"], method="fdr_bh")[1]
pc_sig = pc[pc["p_adj"]<ALPHA].sort_values("p_adj")

# overlap between levels
overlap = pd.merge(pp_sig[["mutation","feature","cohens_d"]],
                   pc_sig[["mutation","feature","coef"]],
                   on=["mutation","feature"], how="inner")

# ---------- 5. AML-SCORE x MUTATION ----------
aml_rows=[]
for m in PRIORITY:                                        # per-patient
    g=pat[m].dropna(); a=pat.loc[g[g==1].index,"AML_mean"]; b=pat.loc[g[g==0].index,"AML_mean"]
    if len(a)>=2 and len(b)>=2:
        U,p=mannwhitneyu(a,b,alternative="two-sided")
        aml_rows.append(dict(mutation=m,n_mut=len(a),n_wt=len(b),
            mean_mut=a.mean(),mean_wt=b.mean(),U=U,p_raw=p))
aml=pd.DataFrame(aml_rows); aml["p_adj"]=multipletests(aml["p_raw"],method="fdr_bh")[1]

aml_cell=[]
for m in PRIORITY:                                        # per-cell mixed model
    sub=df[df[m].notna()].dropna(subset=["AML_probability","age","sex","disease_type"])
    if sub["patient_id"].nunique()<4: continue
    md=smf.mixedlm(f"AML_probability ~ C({m}) + age + C(sex) + C(disease_type)",
                   sub, groups=sub["patient_id"]).fit(reml=False, method="lbfgs")
    cn=[c for c in md.params.index if c.startswith(f"C({m})")][0]
    aml_cell.append(dict(mutation=m,coef=md.params[cn],se=md.bse[cn],
                         t=md.tvalues[cn],p_raw=md.pvalues[cn]))
amlc=pd.DataFrame(aml_cell); amlc["p_adj"]=multipletests(amlc["p_raw"],method="fdr_bh")[1]

# ---------- 6. OUTPUT TABLES & FIGURES ----------
with pd.ExcelWriter("AINU_mutation_results.xlsx") as xl:
    pp.to_excel(xl,"perpatient_all",index=False)
    pp_sig.to_excel(xl,"perpatient_sig",index=False)
    pc.to_excel(xl,"percell_all",index=False)
    pc_sig.to_excel(xl,"percell_sig",index=False)
    aml.to_excel(xl,"AMLscore_perpatient",index=False)
    amlc.to_excel(xl,"AMLscore_percell",index=False)
    overlap.to_excel(xl,"overlap",index=False)
    miss_by_feature.to_frame("frac_missing").to_excel(xl,"missing_by_feature")

# Heatmap: per-patient effect sizes (Cohen's d) for significant feature x mutation
if not pp_sig.empty:
    piv = pp_sig.pivot_table(index="feature",columns="mutation",values="cohens_d")
    plt.figure(figsize=(8,max(4,0.3*len(piv))))
    sns.heatmap(piv,cmap="RdBu_r",center=0,annot=False)
    plt.title("Per-patient: Cohen's d (p_adj<0.05)"); plt.tight_layout()
    plt.savefig("heatmap_perpatient.png",dpi=300); plt.close()

# Heatmap: per-cell coefficient sign for significant pairs
if not pc_sig.empty:
    piv2 = pc_sig.pivot_table(index="feature",columns="mutation",values="coef")
    plt.figure(figsize=(8,max(4,0.3*len(piv2))))
    sns.heatmap(piv2,cmap="RdBu_r",center=0); plt.title("Per-cell: mixed-model coef (p_adj<0.05)")
    plt.tight_layout(); plt.savefig("heatmap_percell.png",dpi=300); plt.close()

# Scatter: top-5 feature x mutation pairs (cell-level, colored by mutation status)
for i,r in pp_sig.head(5).iterrows():
    m,f=r["mutation"],r["feature"]; d=df[df[m].notna()]
    plt.figure(figsize=(5,4))
    sns.stripplot(data=d,x=m,y=f,alpha=0.3,jitter=0.25)
    sns.pointplot(data=d,x=m,y=f,color="k",errorbar="sd")
    plt.title(f"{f} vs {m}  (p_adj={r['p_adj']:.1e})"); plt.tight_layout()
    plt.savefig(f"scatter_{m}_{f}.png",dpi=300); plt.close()

# Scatter: AML score vs blast %, colored by AML status / NPM1 / TP53
for hue in ["AML_status","NPM1","TP53"]:
    if hue in df.columns:
        plt.figure(figsize=(5,4))
        sns.scatterplot(data=df,x="blast_percentage",y="AML_probability",hue=hue,alpha=0.4)
        sns.regplot(data=df,x="blast_percentage",y="AML_probability",scatter=False,color="k")
        plt.tight_layout(); plt.savefig(f"AMLscore_vs_blast_{hue}.png",dpi=300); plt.close()

# QC summary
with open("QC_summary.txt","w") as fh:
    fh.write(f"Cells after QC: {n_cells}\nPatients: {n_patients}\n")
    fh.write(f"Cells/patient: median={cells_per_patient.median()}, "
             f"range={cells_per_patient.min()}-{cells_per_patient.max()}\n")
    for m in MUTATIONS: fh.write(f"{m}: {patients_per_mut[m]}\n")