#!/bin/bash
#SBATCH --job-name=10x_plot
#SBATCH --partition=shared
#SBATCH --time=12:00:00
#SBATCH --mem=2G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/10x/AAA_logs/plot_10x_%A.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/10x/AAA_logs/plot_10x_%A.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting

if [[ -z "${sample_name}" ]]; then
    echo "ERROR: sample_name is not set"
    exit 1
fi

export sample_name
export folder_name="10x"

echo "======================================"
echo "Running bubble plots"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "sample_name=${sample_name}"
echo "folder_name=${folder_name}"
echo "HOSTNAME=$(hostname)"
echo "======================================"

bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
    "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"

# bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
#     "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"