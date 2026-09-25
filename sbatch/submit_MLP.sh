#!/bin/bash
#SBATCH --job-name=2MLP_One_Shifting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/logs/submit_all_steps_vae_MLP2_%A_%a.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/logs/submit_all_steps_vae_MLP2_%A_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1

method="MLP"

cd /dcs10/hongkai/data/yhu1/One_Shifting

sample_names=("10w_NR" "10w_FR")

export sample_name="${sample_names[$SLURM_ARRAY_TASK_ID]}"
export folder_name="HCA"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "sample_name=${sample_name}"
echo "folder_name=${folder_name}"
echo "method=${method}"

bash ../One_Shifting/inst/scripts/run_MLP_general.sh \
    "/dcs10/hongkai/data/yhu1/One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh" \
    "$method"

# bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
#     ../One_Shifting/inst/config/config_run_MLP_general.sh