#!/bin/bash
#SBATCH --job-name=ShiftFormer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --output=/dcs11/hongkai/data/yhu1/One_Shifting_Results/logs/submit_ShiftFormer_%A_%a.out
#SBATCH --error=/dcs11/hongkai/data/yhu1/One_Shifting_Results/logs/submit_ShiftFormer_%A_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

# Array task 0 = INPUT_RNA, 1 = INPUT_ATAC (each task only reads its own modality)
cd /dcs10/hongkai/data/yhu1/One_Shifting

if [[ -z "${sample_name}" ]]; then
    echo "ERROR: sample_name is not set"
    exit 1
fi

if [[ -z "${dropout_keep_par}" ]]; then
    echo "ERROR: dropout_keep_par is not set (use 1 for no dropout)"
    exit 1
fi

export sample_name
export folder_name
export dropout_keep_par

configs=(
    "../One_Shifting/inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"
    "../One_Shifting/inst/config/config_run_multi_ATAC_HCA_10x_ENCODE.sh"
)

task_id=${SLURM_ARRAY_TASK_ID}

if (( task_id >= ${#configs[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${task_id} exceeds total tasks=${#configs[@]}"
    exit 1
fi

config=${configs[$task_id]}

echo "======================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "sample_name=${sample_name}"
echo "folder_name=${folder_name}"
echo "dropout_keep_par=${dropout_keep_par}"
echo "config=${config}"
echo "HOSTNAME=$(hostname)"
echo "======================================"

nvidia-smi

bash ../One_Shifting/inst/scripts/run_ShiftFormer_choose_combination.sh \
    "$config" \
    "$dropout_keep_par"
