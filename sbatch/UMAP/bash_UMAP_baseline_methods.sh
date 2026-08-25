#!/bin/bash
#SBATCH --job-name=UMAP_plot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/dcs10/hongkai/data/yhu1/One_Shifting_Results/UMAP/AAA_logs/plot_UMAP_%j.out
#SBATCH --error=/dcs10/hongkai/data/yhu1/One_Shifting_Results/UMAP/AAA_logs/plot_UMAP_%j.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL

cd /dcs10/hongkai/data/yhu1/One_Shifting || exit 1
OUTPUT_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"

if [[ -z "${sample_name:-}" ]]; then
    echo "[ERROR] sample_name is not set"
    exit 1
fi

if [[ "$sample_name" == "Cite_seq" ]]; then
    READ_DIR="${OUTPUT_DIR}/UMAP/Cite_seq"
    DATA_PATH="${READ_DIR}/orig_data/cbmc_celltype_protein.csv"
elif [[ "$sample_name" == "Zheng_pbmcs" ]]; then
    READ_DIR="${OUTPUT_DIR}/UMAP/Zheng_pbmcs"
    DATA_PATH="${READ_DIR}/orig_data/zheng_pbmc_celltype_facs.csv"
else
    echo "[ERROR] Unknown sample_name: ${sample_name}"
    exit 1
fi

echo "======================================"
echo "Running baseline models and UMAP plots"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "sample_name=${sample_name}"
echo "READ_DIR=${READ_DIR}"
echo "cell-type DATA_PATH=${DATA_PATH}"
echo "HOSTNAME=$(hostname)"
echo "======================================"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] Cell-type reference file not found: ${DATA_PATH}"
    exit 1
fi

module load conda_R

LOSS="zinb"

echo "======================================"
echo "Running scVI, DCA, and SAVER"
echo "LOSS=${LOSS}"
echo "======================================"

bash ../One_Shifting/inst/scripts/run_UMAP_scVI_DCA_Saver.sh \
    "$READ_DIR" \
    "normed_data" \
    "$LOSS"

baseline_status=$?

if [[ $baseline_status -ne 0 ]]; then
    echo "[ERROR] Baseline pipeline failed for ${sample_name}"
    exit $baseline_status
fi

export SEURAT_LEIDEN_ENV="/dcs10/hongkai/data/yhu1/conda_envs/seurat_leiden"
export RETICULATE_PYTHON="${SEURAT_LEIDEN_ENV}/bin/python"

export LD_LIBRARY_PATH="${SEURAT_LEIDEN_ENV}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${SEURAT_LEIDEN_ENV}/lib/libstdc++.so.6:${SEURAT_LEIDEN_ENV}/lib/libgcc_s.so.1"

unset PYTHONHOME
unset PYTHONPATH

echo "======================================"
echo "Running final UMAP plot"
echo "RETICULATE_PYTHON=${RETICULATE_PYTHON}"
echo "registry=${READ_DIR}/recon_registry.csv"
echo "======================================"

if [[ ! -f "${READ_DIR}/recon_registry.csv" ]]; then
    echo "[ERROR] Reconstruction registry not found:"
    echo "        ${READ_DIR}/recon_registry.csv"
    exit 1
fi

# use own hvgs
Rscript ../One_Shifting/R/run_UMAP_plot.R \
    "$READ_DIR" \
    "$DATA_PATH" \
    "${READ_DIR}/recon_registry.csv" \
    "" \
    ""
    
# use common hvg lits
# Rscript ../One_Shifting/R/run_UMAP_plot.R \
#     "$READ_DIR" \
#     "$DATA_PATH" \
#     "${READ_DIR}/recon_registry.csv" \
#     "" \
#     "${READ_DIR}/orig_data/hvg_2000.txt"

plot_status=$?

if [[ $plot_status -ne 0 ]]; then
    echo "[ERROR] UMAP plotting failed for ${sample_name}"
    exit $plot_status
fi

echo "======================================"
echo "Completed baseline and UMAP plotting"
echo "sample_name=${sample_name}"
echo "======================================"