#!/bin/bash
#SBATCH --job-name=UMAP_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=/dcs10/hongkai/data/yhu1/One_Shifting_Results/UMAP/AAA_logs/train_UMAP_%A_%a.out
#SBATCH --error=/dcs10/hongkai/data/yhu1/One_Shifting_Results/UMAP/AAA_logs/train_UMAP_%A_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting || exit 1
OUTPUT_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"

if [[ -z "${sample_name:-}" ]]; then
    echo "[ERROR] sample_name is not set"
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID is not set"
    exit 1c
fi

module load conda_R

methods=(
    "VAE"
    "DCA_mse"
    "scVI_mse"
    "Transformer_denoise"
)

task_id="${SLURM_ARRAY_TASK_ID}"

if (( task_id < 0 || task_id >= ${#methods[@]} )); then
    echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID=${task_id}"
    exit 1
fi

method="${methods[$task_id]}"

if [[ "$sample_name" == "Cite_seq" ]]; then
    READ_DIR="${OUTPUT_DIR}/UMAP/Cite_seq"
    DATA_PATH="${READ_DIR}/orig_data/cbmc_rna_counts.feather"
elif [[ "$sample_name" == "Zheng_pbmcs" ]]; then
    READ_DIR="${OUTPUT_DIR}/UMAP/Zheng_pbmcs"
    DATA_PATH="${READ_DIR}/orig_data/zheng_pbmc_rna_counts.feather"
else
    echo "[ERROR] Unknown sample_name: ${sample_name}"
    exit 1
fi

echo "======================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "sample_name=${sample_name}"
echo "method=${method}"
echo "READ_DIR=${READ_DIR}"
echo "DATA_PATH=${DATA_PATH}"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "======================================"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] Input data not found: ${DATA_PATH}"
    exit 1
fi

bash ../One_Shifting/inst/scripts/run_multiomics_general_select_norm_UMAP.sh \
    "$DATA_PATH" \
    "$READ_DIR" \
    "$method"

status=$?

if [[ $status -ne 0 ]]; then
    echo "[ERROR] ${method} pipeline failed for sample ${sample_name}"
    exit $status
fi

echo "======================================"
echo "Completed method=${method}"
echo "Completed sample_name=${sample_name}"
echo "======================================"
