#!/bin/bash
#SBATCH --job-name=ENCODE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --array=0-7
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/ENCODE/AAA_logs/submit_ENCODE_pip_%A_%a.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/ENCODE/AAA_logs/submit_ENCODE_pip_%A_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting

if [[ -z "${sample_name}" ]]; then
    echo "ERROR: sample_name is not set"
    exit 1
fi

export sample_name
export folder_name="ENCODE"

methods=("VAE" "DCA_mse" "scVI_mse" "Transformer_denoise")

configs=(
    "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"
    "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"
)

method=${methods[$((SLURM_ARRAY_TASK_ID % 4))]}
config=${configs[$((SLURM_ARRAY_TASK_ID / 4))]}

echo "======================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "sample_name=${sample_name}"
echo "folder_name=${folder_name}"
echo "method=${method}"
echo "config=${config}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HOSTNAME=$(hostname)"
echo "======================================"

nvidia-smi

bash ../One_Shifting/inst/scripts/run_pbmc_general_mse_reverse.sh \
    "$config" \
    "$method"

# bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
#     "../One_Shifting/inst/config/config_run_multi_RNA.sh"

# bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
#     "../One_Shifting/inst/config/config_run_multi_ATAC.sh"
