#!/bin/bash
#SBATCH --job-name=multi_One_Shifting
#SBATCH --partition=shared
#SBATCH --time=3-00:00:00
#SBATCH --mem=8G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/submit_average.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/submit_average.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --dependency=afterok:33324229

# sample_names=("Hr3" "LGS10S" "PF_specimen")

# export folder_name="HCA"
# export dropout_keep_par=0.1

# for sample_name in "${sample_names[@]}"; do
#     export sample_name

#     echo "========================================"
#     echo "Running bubble plots for sample_name=${sample_name}"
#     echo "========================================"

#     bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
#         "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"

#     bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
#         "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"

#     echo "Done sample_name=${sample_name}"
#     echo
# done

BASE_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA"
TSV_FILE="/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA_input_sample_pairs.tsv"


cd /dcs10/hongkai/data/yhu1/One_Shifting

module load conda_R

# Rscript ../One_Shifting/R/run_evaluation_across_samples.R \
#     $BASE_DIR \
#     $TSV_FILE \
#     "1,3,7" \
#     "dropout_0p1"

Rscript ../One_Shifting/R/run_evaluation_across_samples_one_scale.R \
    $BASE_DIR \
    $TSV_FILE \
    "all" \
    "" \
    "col"


# Rscript ../One_Shifting/R/check_HCA_sparsity.R \
#     "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA" 


