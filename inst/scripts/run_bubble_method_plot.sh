#!/bin/bash
CONFIG_FILE="$1"
filter_method="$2"
bubble_scope="${3:-all}"

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_bubble_method_plot.sh <config_file> [method] [all|selected]"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    echo "  method: optional method to exclude (e.g. scVI_mse)"
    echo "  all|selected: process all bubble summaries or only the two selected col_gene summaries"
    exit 1
fi

source "$CONFIG_FILE"

if [ "$bubble_scope" = "all" ]; then
    csv_files=("${READ_DIR}"/bubble_plot_summary_*.csv)
elif [ "$bubble_scope" = "selected" ]; then
    csv_files=(
        "${READ_DIR}/bubble_plot_summary_col_gene_v1_normF_trans__v2_normF_trans__recon_asis.csv"
        "${READ_DIR}/bubble_plot_summary_col_gene_v1_raw__v2_raw__recon_revtrans_revnorm.csv"
    )
else
    echo "[ERROR] Third argument must be 'all' or 'selected'. Got: $bubble_scope"
    exit 1
fi

echo "Transformation: $trans_factor"
echo "INPUT_CATEGORY: $V1"
echo "READ_DIR: $READ_DIR"

module load conda_R

echo "=== Draw Bubble Plot ==="
echo "Bubble scope: $bubble_scope"
for csv_file in "${csv_files[@]}"; do
    if [ ! -f "$csv_file" ]; then
        echo "[WARN] Missing: $csv_file"
        continue
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

    Rscript ../One_Shifting/R/run_plot_summary_bubble_table.R \
        "$csv_file" \
        "$filter_method" 
done
