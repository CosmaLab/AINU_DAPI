# R scripts for the accuracy and probability figures

Written by Carlotta Viana. They plot results produced by the Python pipeline (trained models, per-cell probabilities, per-patient
accuracies); they do not train or run models.

| Script | Figure (as noted by the author; check against the current numbering) |
|---|---|
| `retrain_onepatout_plot_patients_summary.R` | Figure 1F (retraining of the frozen model for single patients) |
| `MDS_plot_probabilities_avg5splits.R` | Figure 2 (pre-leukemia: probability distributions averaged over the five split models) |
| `MRD_plot_probabilities_avg5splits.R` | Figure 3 (MRD: same plots) |
| `onepatout_plots.R` | leave-one-patient-out, retraining and 4-class plots (most of the remaining accuracy panels); not yet reorganised |

## Running

```
AINU_R_DATA=/path/to/input/tables Rscript MDS_plot_probabilities_avg5splits.R
```

The scripts `setwd()` to `AINU_R_DATA` (default: the current directory) and write PDFs to `figures/` there.
R packages: ggplot2, dplyr, tidyr, readr, readxl, stringr, purrr, ggrepel, ggtext, ggsci, patchwork, rstatix, ggpubr, viridis, scales.

## Input tables

The inputs are the outputs of the Python pipeline and are not included: `embeddings/{mds,mrd}/embeddings_*_sample4s{1-5}_*.csv`,
`dataset_summary_{AML,MDS,MRD}.txt`, `*_splits_metrics_summary*.txt`, `retrain_<patient>_metrics_summary.txt`,
`patient_accuracy_mean_sd_for_R.csv`.

They must use the labels of the paper: patient labels (`AML17`, `MRD10-2`, ...) in the `patient`, `patientID` and `celltype` columns,
and the site labels `Commercial`, `Hospital1`, `Hospital2`, `Hospital3` in the `hospital` column. The retraining files are named
`retrain_AML17_metrics_summary.txt` (and `retrain_AML06_seed43_metrics_summary.txt`, ...). Earlier versions of these scripts carried
code-to-label corrections for individual samples; they were removed because the inputs now carry the labels directly.
