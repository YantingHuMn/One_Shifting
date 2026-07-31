#!/bin/bash

samples=("Cite_seq" "Zheng_pbmcs")

cd /dcs10/hongkai/data/yhu1/One_Shifting

worker_script="../One_Shifting/sbatch/UMAP/bash_UMAP_four_methods.sh"
plot_script="../One_Shifting/sbatch/UMAP/bash_UMAP_baseline_methods.sh"

OUTPUT_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"
log_dir="${OUTPUT_DIR}/UMAP/AAA_logs"

mkdir -p "$log_dir"

echo "Submitting UMAP data preparation job..."

prep_job_id=$(sbatch \
    --job-name="UMAP_data_load" \
    --partition=shared \
    --time=12:00:00 \
    --mem=128G \
    --output="${log_dir}/UMAP_data_load_%j.out" \
    --error="${log_dir}/UMAP_data_load_%j.err" \
    --wrap="
        cd /dcs10/hongkai/data/yhu1/One_Shifting || exit 1
        module load conda_R

        Rscript ../One_Shifting/R/run_UMAP_data_load.R \
            '${OUTPUT_DIR}/UMAP'
    " | awk '{print $4}')

    
if [[ -z "$prep_job_id" ]]; then
    echo "[ERROR] Failed to submit UMAP data preparation job"
    exit 1
fi

echo "Submitted data preparation job: ${prep_job_id}"
echo "--------------------------------------"

for sample_name in "${samples[@]}"; do

    echo "Submitting 4-method training array for sample_name=${sample_name}"

    train_job_id=$(sbatch \
        --job-name="UMAP_${sample_name}" \
        --array=0-3 \
        --dependency=afterok:${prep_job_id} \
        --exclude=compute-170 \
        --export=ALL,sample_name="${sample_name}" \
        --output="${log_dir}/train_UMAP_${sample_name}_%A_%a.out" \
        --error="${log_dir}/train_UMAP_${sample_name}_%A_%a.err" \
        "$worker_script" | awk '{print $4}')

    if [[ -z "$train_job_id" ]]; then
        echo "[ERROR] Failed to submit training array for ${sample_name}"
        continue
    fi

    echo "Submitted training array job: ${train_job_id}"
    echo "Submitting dependent baseline and plot job for sample_name=${sample_name}"

    baseline_job_id=$(sbatch \
        --job-name="UMAP_baseline_${sample_name}" \
        --dependency=afterok:${train_job_id} \
        --exclude=compute-170 \
        --export=ALL,sample_name="${sample_name}" \
        --output="${log_dir}/baseline_UMAP_${sample_name}_%j.out" \
        --error="${log_dir}/baseline_UMAP_${sample_name}_%j.err" \
        "$plot_script" | awk '{print $4}')

    if [[ -z "$baseline_job_id" ]]; then
        echo "[ERROR] Failed to submit baseline job for ${sample_name}"
        continue
    fi

    echo "Submitted baseline job: ${baseline_job_id}"
    echo "Dependency: afterok:${train_job_id}"
    echo "--------------------------------------"

done
