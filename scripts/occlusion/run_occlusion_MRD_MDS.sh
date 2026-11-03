#!/bin/bash
# Example SLURM job: occlusion maps for the MRD and pre-leukemia cells with the split-3 model of the AML-versus-Control run
# (4-PB_BM_all). Adapt the partition, resources and the three paths below.
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

RESULTS=/path/to/results            # contains 4-PB_BM_all/ainu_split3.log and the model checkpoint named in it
CELL_LIST=/path/to/MRD_MDS.csv      # tab-separated list of cells (File, condition, ...)
OUT=/path/to/results/ainu_split3_MRD_MDS/

python scripts/occlusion/occlusion_file_AML.py \
    --log_file ${RESULTS}/4-PB_BM_all/ainu_split3.log \
    --out_path ${OUT} \
    --csv_file ${CELL_LIST} \
    --savenpy
