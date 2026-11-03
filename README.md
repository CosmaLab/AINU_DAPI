# AINU: code for "Deep Learning Identifies Abnormal Chromatin Signatures in Neoplastic Cells in Acute Myeloid Leukemia"

AINU is a DenseNet121 classifier trained on single-channel DAPI confocal images of single nuclei. This repository contains the code used to
score patients and cells with the trained models, to compute occlusion-based attribution statistics, to build the embeddings, and to produce
the supplementary tables. Images, de-identified annotation tables, occlusion maps and model weights are deposited separately (see *Data*).

## Contents

```
augmentation/        Albumentations pipelines (train/, val/ and test/) used for the 128 x 128 model input
scripts/
  training/               AINU_tuning.py: training, validation, testing, ROC/PR curves and confusion matrices
  inference_and_tables/   scoring of cells and patients with trained models, embeddings, Tables S1 (part A) and S6
  occlusion/              occlusion maps (black baseline; nuclear-mean baseline for the robustness check)
  attribution_analysis/   ring-wise, texture, density and attribution-mass statistics (Figure 6, Figure S5, Tables S1-S5)
  morphology/             nuclear morphology features versus mutations
  figures_R/              R scripts for the accuracy and probability-distribution figures (Figures 1F, 2, 3; see its README)
notebooks/           figure-level analysis notebooks (outputs cleared)
requirements.txt     package versions
```

## Data

Everything below is de-identified: sample codes and hospital names are replaced by the labels of the paper (see the end of this
section), and the key that links them to the original codes is not distributed.

| Item | Where | Notes |
|---|---|---|
| Single-nucleus images (16,866 PNG)  | grouped by diagnosis (AML, Control, MRD, Preleukemia) with a `manifest.csv` |
| Annotation tables and split lists (`cell_features.csv`, `patient_image_data.csv`, `all_images.csv`, `MRD_MDS*.csv`, `sample4*.txt`, `PB_BM_AML_vs_CTRL.txt`) | Zenodo: `DOI to be added` | cells carry `split1`-`split5` columns |
| Occlusion maps (13,111 `.npy`, 2.6 GB) and the derived per-cell feature tables | Zenodo: `DOI to be added` | same cell names as the images |
| Trained model weights (`*_model_best.pth.tar`)

Rows of the original tables that belong to samples not in the released image set are not included, so a few counts in the tables
can differ from the full dataset described in the paper (see `manifest.csv`).

All scripts expect a data root with this layout; set it with the environment variable `AINU_ROOT` (default: the current directory):

```
$AINU_ROOT/
  data/human_data/                          single-nucleus images <File>.png and annotation tables
                                            (MRD_MDS_with_AML12_14.csv, patient_image_data.csv, cell_features.csv)
  augmentation/test/<pipeline>.json         copy of ./augmentation
  results/<run name>/                       training logs and checkpoints (*_model_best.pth.tar)
  results/occlusion_analysis/...            occlusion maps (.npy) and the derived per-cell feature tables
```

The image folders of the Zenodo record must be merged into `data/human_data/` (file names are unique across diagnoses).

Patients, cells and sites carry the labels used in the paper: patients are `AML01`, `CTRL01`, `MDS01`, `MRD01`, ...; cells are
`<patient>_<number>` (for example `AML01_0142`); sites are `Commercial`, `Hospital1`, `Hospital2` and `Hospital3` (Figure 1,
Figure S1G). Per-cell files such as occlusion maps (`<cell>_AML_occlusion.npy`) use the same cell names.

The notebooks contain a single placeholder path, `/path/to/ainu_root`; replace it with your data root.

## Environment

Tested with Python 3.13 on macOS (Apple-silicon `mps` or CPU; the scripts select `mps` when available, otherwise CPU).
Install with `pip install -r requirements.txt`.

## Model

- Architecture: torchvision DenseNet121 (ImageNet-pretrained), classifier replaced by `Linear(1024, 2)`; class 1 = AML, class 0 = Control.
- Input: image read with OpenCV (BGR to RGB), transformed by `augmentation/test/PadResize128_GD_r03_u4-32_p07_grey_blackbg.json`
  (longest side resized to 128 and padded to 128 x 128), divided by 255.
- Training settings recorded in the run logs of the five AML-versus-Control splits: 300 epochs, batch size 16, SGD (momentum 0.9,
  weight decay 1e-4), one-cycle learning-rate schedule (maximum learning rate 5e-4), cross-entropy loss, cell-level 80/20 train/test split
  within each patient, with a validation subset taken from the training cells, five random splits (`split1`-`split5`).
- The training code is the AINU repository of the authors; see *Training* below.

## Reproducing the analyses

Run the scripts from a directory where `AINU_ROOT` points to the data root.

| Output | Script |
|---|---|
| Per-patient percentage of cells predicted as AML, mean and s.d. over the five split models (Figures 2A and 3A) | `scripts/inference_and_tables/score_patients_all_models.py` |
| Embeddings with predictions for the t-SNE (Figure 4), followed by `notebooks/t-sne.ipynb` | `scripts/inference_and_tables/embed_newmodel_with_MRD_MDS.py` |
| Table S6: performance of the retrained models and per-patient predictions | `scripts/inference_and_tables/build_table_S6.py` |
| Table S1 part A: ring-wise AML versus Control statistics (asterisks of Figure 6D); also needs `notebooks/occlusion_feature_analysis_AML_MDS_CT.ipynb` | `scripts/inference_and_tables/build_table_S1_partA.py` |
| Table S1 part B: AML-supporting and AML-suppressing attribution mass, pairwise tests | `scripts/attribution_analysis/attribution_mass_pairwise.py` |
| Ring-wise density versus attribution (Table S2) | `scripts/attribution_analysis/ring_density_attribution_correlation.py` |
| Texture and brightness correlations, group comparisons (Figure 6E-F, Tables S3 and S4) | `scripts/attribution_analysis/patient_level_texture_panel.py`, `mrd_mds_texture_extraction.py` |
| Smooth-versus-textured tercile analysis (Table S5) | `scripts/attribution_analysis/texture_tercile_direction.py`, `verify_texture_direction.py` |
| Attribution versus DAPI structure for every test cell; positive/negative split | `scripts/attribution_analysis/full_attr_analysis.py`, `pos_neg_split.py` |
| Independence of ring-wise density and texture effects | `scripts/attribution_analysis/ring_texture_independence_check.py` |
| Baseline robustness of the texture correlation | `scripts/occlusion/occlusion_nucmean_baseline.py`, `scripts/attribution_analysis/baseline_robustness_check.py` |
| Retraining, probability-distribution and 4-class accuracy plots (Figures 1F, 2, 3 and supplementary panels; R) | `scripts/figures_R/` |
| Chromatin features versus somatic mutations | `scripts/morphology/shape_features_mutations.py` |
| Radial attribution profiles (Figure 6D) and per-ring statistics | `notebooks/occlusion_feature_analysis_AML_MDS_CT.ipynb` |
| Occlusion analysis of the preleukemia and MRD cohorts | `notebooks/MRD_MDS_Analysis_Clean.ipynb` |
| Shape-stratified analyses | `notebooks/grouped_shape_analysis.ipynb` |

## Occlusion maps

Occlusion attribution uses Captum's `Occlusion` on the 128 x 128 model input (scaled to [0, 1]): window 3 x 8 x 8 (all channels, 8 x 8
pixels), stride 3 x 4 x 4, **baseline 0 (black)**, target = output of the AML class (index 1) for every cell, whatever its class.
Positive values support the AML call and negative values oppose it. The maps of the paper come from the split-3 model of the AML-versus-Control run (`4-PB_BM_all/ainu_split3`);
`scripts/occlusion/run_occlusion_MRD_MDS.sh` is an example job. Run each script with the log file of the trained model
(`--log_file`), an output folder (`--out_path`) and `--savenpy`:

| Script | Cells | Output |
|---|---|---|
| `scripts/occlusion/occlusion_logfile.py` | test cells of the AML-versus-Control model that were classified correctly | `<cell>_<positive/negative>_AML_occlusion.npy`, PDF overlays, `predictions.csv` |
| `scripts/occlusion/occlusion_file_AML.py` | every cell of a given list (pre-leukemia, MRD), whatever the prediction | `<cell>_AML.npy`, `predictions.csv` |
| `scripts/occlusion/occlusion_logfile_baseline.py` | as `occlusion_logfile.py`, baseline = per-image nuclear mean (pixels > 0.04) | `<cell>_<class>_occlusion_AML_baseline.npy` |
| `scripts/occlusion/occlusion_nucmean_baseline.py` | MRD / pre-leukemia cells, same nuclear-mean baseline | see script |

## Training

`scripts/training/AINU_tuning.py` trains and tests the network (adapted by Davide Carnevali from the PyTorch ImageNet example; the
BSD 3-Clause notice is kept in the file). It reads a tab-separated list of cells (`File`, `celltype`, `condition`, and the columns
`split1`-`split5` with `train`/`val`/`test`), so the five splits of the paper are fixed by the lists and not drawn again. If a split
column is missing the script draws a new, unseeded random split. It writes `<name>.log`, `<name>_model_best.pth.tar` (best validation
accuracy), ROC/precision-recall curves and confusion matrices to the output folder, and it overwrites the list file with the split column.
The images are read as `<data dir>/<File>.png`, so the images must be in one folder (the cohort folders of the released image set can be
merged: file names are unique).

```
python scripts/training/AINU_tuning.py -a densenet121 /path/to/images/ \
  -L sample4.txt --display_labels "positive=AML,negative=Control" -classes 2 --pretrained \
  -sname split1 --epochs 300 -b 16 -opt SGD --lr 0.0001 --ocmaxlr 0.0005 --scheduler one_cycle_lr \
  --loss_function CrossEntropyLoss -j 2 \
  -augdir augmentation/ -aug PadResize128_GD_r03_u4-32_p07_grey_blackbg \
  -o /path/to/results/run_split1/ -n run_split1
```

Add `--only_test --test_model_path <checkpoint> --test_data_path <list> --test_images_dir <dir>` to test a trained model on other
cells (for example pre-leukemia or MRD), and `--resume <checkpoint>` with a larger `--epochs` to continue a run. The command above is
the configuration described in the Methods for the AML-versus-Control models (to be checked against the run logs); the exact command of each run is stored as `Running parameters` in
its log file.

Single-patient retraining (patient-specific list with 20 images of the patient added to the training cells, started from the
checkpoint of the model trained without the patient) uses the same script with `--resume`, for example:

```
python scripts/training/AINU_tuning.py -a densenet121 /path/to/images/ -L sample4_without_AML23_20images.txt \
  --epochs 400 -b 16 -aug PadResize128_GD_r03_u4-32_p07_grey_blackbg -augdir augmentation/ \
  -o /path/to/results/retrain/AML23_seed43/ -n sample4_without_AML23_20images_split1 -classes 2 --pretrained -opt SGD -j 2 \
  -sname split1 --lr 0.0001 --ocmaxlr 0.0005 --loss_function CrossEntropyLoss --scheduler one_cycle_lr \
  --display_labels "positive=AML,negative=Control" \
  --resume /path/to/results/retrain/AML23_seed43/sample4_without_AML23_20images_split1_checkpoint.pth.tar
```

## Code availability

The code is available at https://github.com/CosmaLab/AINU_DAPI (release `v1.0`, archived on Zenodo: `DOI to be added`). The de-identified
images, annotation tables, occlusion maps and trained weights are on Zenodo (`DOI to be added`).

## License and citation

Code license: BSD-3. If you use this code, please cite the paper above.
