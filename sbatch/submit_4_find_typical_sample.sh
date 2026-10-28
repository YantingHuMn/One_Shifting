#!/bin/bash
#SBATCH --job-name=typical
#SBATCH --partition=shared
# #SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=1G
#SBATCH --output=/dcs11/hongkai/data/yhu1/One_Shifting_Results/logs/find_typical_sample.out
#SBATCH --error=/dcs11/hongkai/data/yhu1/One_Shifting_Results/logs/find_typical_sample.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH --dependency=afterok:33324229

cd /dcs10/hongkai/data/yhu1/One_Shifting
BASE_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"

module load conda_R

rm -f ${BASE_DIR}/col_reverse_representative_sample_ranking.csv
Rscript ../One_Shifting/R/typical_sample_find.R \
    "$BASE_DIR" \
    "bubble_plot_summary_col_gene_v1_raw__v2_raw__recon_revtrans_revnorm.csv"

