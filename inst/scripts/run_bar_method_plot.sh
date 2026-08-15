#!/bin/bash
CONFIG_FILE="$1"
filter_method="$2"

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_pipeline.sh <config_file> [method]"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    exit 1
fi

source "$CONFIG_FILE"

echo "Transformation: $trans_factor"
echo "INPUT_CATEGORY: $V1"
echo "READ_DIR: $READ_DIR"

module load conda_R

echo "=== Draw Bubble Plot ==="
for csv_file in "${READ_DIR}"/bubble_plot_summary_*.csv; do
    if [ ! -f "$csv_file" ]; then
        echo "[WARN] No bubble_plot_summary_*.csv files found in ${READ_DIR}"
        break
    fi

    if [[ "$csv_file" == *_avg_rank_perf.csv ]]; then
        continue
    fi

    if [[ "$csv_file" == *_avg_rank_resid.csv ]]; then
        continue
    fi

    if [[ "$csv_file" == *_table_df.csv ]]; then
        continue
    fi

    echo "[INFO] Plotting: $csv_file"

    Rscript ../One_Shifting/R/run_plot_summary_bar_table.R \
        "$csv_file" \
        "$filter_method" 
done