#!/bin/bash
CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_pipeline.sh <config_file> <method>"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    exit 1
fi

source "$CONFIG_FILE"

echo "Transformation: $trans_factor"
echo "INPUT_CATEGORY: $V1"
echo "READ_DIR: $READ_DIR"

module load conda_R

echo "=== Draw Bubble Plot ==="
for gt_suffix in "" "_noGTzero"; do
    Rscript ../One_Shifting/R/run_plot_summary_bubble_table.R \
        "${READ_DIR}/bubble_plot_summary_col_gene${gt_suffix}.csv"

    Rscript ../One_Shifting/R/run_plot_summary_bubble_table.R \
        "${READ_DIR}/bubble_plot_summary_row_cell${gt_suffix}.csv"
done



