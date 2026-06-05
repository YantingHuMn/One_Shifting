#!/bin/bash
#SBATCH --job-name=multi_One_Shifting
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00
#SBATCH --mem=18G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/submit_average.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/submit_average.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH --dependency=afterok:31582287

BASE_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA"
TSV_FILE="/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA_input_sample_pairs.tsv"


cd /dcs10/hongkai/data/yhu1/One_Shifting

module load conda_R

# Rscript ../One_Shifting/R/run_evaluation_across_samples.R \
#     $BASE_DIR \
#     $TSV_FILE

Rscript ../One_Shifting/R/check_HCA_sparsity.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA" 

