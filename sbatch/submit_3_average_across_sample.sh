#!/bin/bash
#SBATCH --job-name=multi_One_Shifting
#SBATCH --partition=shared
#SBATCH --time=12:00:00
#SBATCH --mem=1G
#SBATCH --output=/dcs11/hongkai/data/yhu1/One_Shifting_Results/logs/submit_average.out
#SBATCH --error=/dcs11/hongkai/data/yhu1/One_Shifting_Results/logs/submit_average.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH --dependency=afterok:33324229

cd /dcs10/hongkai/data/yhu1/One_Shifting 

BASE_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"
log_dir="${BASE_DIR}/logs"

# tsv="${BASE_DIR}/${folder_name}_input_sample_pairs.tsv"
tsv="${BASE_DIR}/HCA_input_sample_pairs.tsv,${BASE_DIR}/ENCODE_input_sample_pairs.tsv,${BASE_DIR}/10x_input_sample_pairs.tsv"

log_file="${log_dir}/AAA_HCA_ENCODE_10x_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1

module load conda_R

Rscript ../One_Shifting/R/run_evaluation_across_samples_one_scale.R \
    "$BASE_DIR" \
    "$tsv" \
    "all" \
    "" \
    "col" \
    "scVI_mse" \
    "selected"
