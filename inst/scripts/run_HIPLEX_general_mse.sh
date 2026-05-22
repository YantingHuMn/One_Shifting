#!/bin/bash
CONFIG_FILE="$1"
method="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$method" ]; then
    echo "Usage: bash run_pipeline.sh <config_file> <method>"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    echo "  method: VAE, DCA_mse, scVI_MSE, Transformer_denoise"
    exit 1
fi

source "$CONFIG_FILE"

echo "INPUT_FILE: $INPUT_FILE"
echo "INPUT_CATEGORY: $V1"
echo "READ_DIR: $READ_DIR"
echo "METHOD: $method"

CONDITION="given_${V2}_${V2_norm_factor}_${V2_trans_factor}"
OUTPUT_DIR="${READ_DIR}/${method}/${CONDITION}"
mkdir -p $OUTPUT_DIR

trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000 "standardize")
norm_factor_string=$(IFS=','; echo "${norm_factor[*]}")

module load conda_R

LOCK_FILE="$READ_DIR/.count_matrix_done"
LOCK_DIR="$READ_DIR/.count_matrix_lock"
MAX_WAIT=180  # 30 minutes

if [ ! -f "$LOCK_FILE" ]; then
    mkdir -p "$(dirname $LOCK_FILE)"

    while true; do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

            Rscript ../One_Shifting/R/Step1_build_count_matrix.R \
              $INPUT_FILE \
              $GROUND_TRUTH_FILE \
              "$READ_DIR" \
              "$norm_factor_string" \
              "$norm_factor_string"

            touch "$LOCK_FILE"
            rmdir "$LOCK_DIR"
            trap - EXIT
            break
        else
            echo "Lock exists, waiting..."
            WAIT_COUNT=0
            while [ ! -f "$LOCK_FILE" ]; do
                sleep 10
                WAIT_COUNT=$((WAIT_COUNT + 1))
                if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
                    echo "Timeout after 30min, removing stale lock and retrying..."
                    rmdir "$LOCK_DIR" 2>/dev/null
                    break
                fi
            done
            [ -f "$LOCK_FILE" ] && break
        fi
    done
fi
echo "Count matrix ready, proceeding..."


echo "===Finish build count matrix==="

# Rscript /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/seurat/Step_post_sparsity_avg.R \
#   /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/seurat_all_genes/RNA/rna_counts_75per.feather \
#   "${OUTPUT_DIR}" \
#   "rna"

# Rscript /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/seurat/Step_post_sparsity_avg.R \
#   /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/seurat_all_genes/ATAC/activity_counts_75per.feather \
#   "${OUTPUT_DIR}" \
#   "atac"
  
echo "=== Starting VAE pipeline ==="

rm -f "${OUTPUT_DIR}/plots_summary_pearson_col_gene.csv"
rm -f "${OUTPUT_DIR}/plots_summary_spearman_col_gene.csv"
rm -f "${OUTPUT_DIR}/plots_summary_pearson_row_cell.csv"
rm -f "${OUTPUT_DIR}/plots_summary_spearman_row_cell.csv"

rm -f "${OUTPUT_DIR}/plots_summary_pearson_col_gene_noGTzero.csv"
rm -f "${OUTPUT_DIR}/plots_summary_spearman_col_gene_noGTzero.csv"
rm -f "${OUTPUT_DIR}/plots_summary_pearson_row_cell_noGTzero.csv"
rm -f "${OUTPUT_DIR}/plots_summary_spearman_row_cell_noGTzero.csv"


# === delete bubble summary===
BUBBLE_LOCK_DIR="$READ_DIR/.bubble_reset_lock"
BUBBLE_DONE_FLAG="$READ_DIR/.bubble_reset_done"

if [ ! -f "$BUBBLE_DONE_FLAG" ]; then
    if mkdir "$BUBBLE_LOCK_DIR" 2>/dev/null; then
        trap 'rmdir "$BUBBLE_LOCK_DIR" 2>/dev/null' EXIT
        
        echo "First method ($method) - clearing old bubble summaries..."
        rm -f "${READ_DIR}"/bubble_plot_summary_*.csv
        
        touch "$BUBBLE_DONE_FLAG"
        rmdir "$BUBBLE_LOCK_DIR"
        trap - EXIT
    else
        echo "Waiting for bubble reset..."
        while [ ! -f "$BUBBLE_DONE_FLAG" ]; do
            sleep 2
        done
    fi
fi

echo "Bubble reset done, proceeding with $method..."

if [ "$method" = "VAE" ]; then
    METHOD_ARGS="--beta_grid 0 --hidden_grid1 256 --hidden_grid2 128"
elif [ "$method" = "DCA_mse" ]; then
    METHOD_ARGS="--dropout_grid 0.0 --hidden_grid1 256 --hidden_grid2 128"
elif [ "$method" = "scVI_mse" ]; then
    METHOD_ARGS="--hidden_grid 512,128,32"
elif [ "$method" = "Transformer_denoise" ]; then
    METHOD_ARGS="--n_tokens_grid 32 --d_model_grid 64 --nhead_grid 4 --num_layers_grid 1 --dim_feedforward_grid 128 --dropout_grid 0.1 --batch_size_grid 512"
fi
        
# train - find par
echo "=== Step 4: Training VAE on filtered data ==="
for this_trans_factor in "${trans_factor[@]}"; do
    OUT_DIR="$OUTPUT_DIR/$this_trans_factor" 

    mkdir -p "$OUT_DIR"

    for factor in "${norm_factor[@]}"; do

        # Skip incompatible (standardize + count+1/sqrt/log) combinations
        if [ "$factor" = "standardize" ]; then
            case "$this_trans_factor" in
                no_trans) ;;  # allow
                *) echo "[SKIP] $this_trans_factor incompatible with standardize (negative values)"
                   continue ;;
            esac
        fi

        DATA_PATH1="$READ_DIR/$V1/Count_Matrix_norm_by_$factor.feather"
        DATA_PATH2="$READ_DIR/$V2/Count_Matrix_norm_by_$V2_norm_factor.feather"
        SUMMARY_FILE="${OUT_DIR}/hyper_par_norm_by_${factor}.tsv"

        source ~/.bashrc
        conda activate vae_env2
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

        python -u ../One_Shifting/myproject/Step2_train_${method}.py \
        --data_path1 "$DATA_PATH1" \
        --data_path2 "$DATA_PATH2" \
        --out_summary "$SUMMARY_FILE" \
        --early_stop \
        --patience 5 \
        --n_splits 2 \
        --eval_metric val_loss \
        --trans1_grid "$this_trans_factor" \
        --trans2_grid "$V2_trans_factor" \
        --nonzero_weight_grid 1.0 \
        $METHOD_ARGS
        
        # reconstruct
        SAVED_DIR="${OUT_DIR}/saved_models"
        OUT_PATH="${OUT_DIR}/reconstruct_trans_by_${this_trans_factor}_norm_by_${factor}.feather"

        python -u ../One_Shifting/myproject/Step3_reconstruct_${method}.py \
        --data_path1 "$DATA_PATH1" \
        --data_path2 "$DATA_PATH2" \
        --transformed_out_dir "$OUT_DIR" \
        --saved_models_dir "$SAVED_DIR" \
        --out_path "$OUT_PATH"

        python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
        conda deactivate 

        # correlation
        echo "=== Step 7: Correlation analysis ==="
        module load conda_R

        Figure_DIR="$OUTPUT_DIR/Figures_col"
        mkdir -p "$Figure_DIR"

        Rscript ../One_Shifting/R/run_correlation_scatter.R \
          "$OUT_DIR/Count_matrix_transformed_rep2.feather" \
          "$OUT_DIR/Count_matrix_transformed_rep1.feather" \
          "$OUT_DIR/reconstruct_trans_by_${this_trans_factor}_norm_by_${factor}.feather" \
          "$Figure_DIR"  \
          "$OUT_DIR/saved_models" \
          "$factor" \
          "$V2_norm_factor" \
          "$this_trans_factor" \
          "$V2_trans_factor" \
          "${OUTPUT_DIR}" \
          "col" \
          "${V1}" \
          "${method}"             

        sleep 2
    done
done


module load conda_R
 
Rscript ../One_Shifting/R/run_combine_figures.R \
  "$OUTPUT_DIR/Figures_col"


# Rscript /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/compare_Dec_12/Step12_combine_plot_no_data.R \
#   "$OUTPUT_DIR/ROC" \
#   "roc" \
#   30

# Rscript /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/compare_Dec_12/Step12_combine_plot_no_data.R \
#   "$OUTPUT_DIR/ROC_balanced" \
#   "roc_balanced" \
#   30

# Summary scatter plots
for corr_method in pearson spearman; do
    for gt_suffix in "" "_noGTzero"; do
        Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
          "${OUTPUT_DIR}/plots_summary_${corr_method}_col_gene${gt_suffix}.csv" \
          $method \
          "log(count+2)"
    done
done

echo "=== Step 8: Bubble Plot ==="
for corr_method in pearson spearman; do
    for gt_suffix in "" "_noGTzero"; do
        Rscript ../One_Shifting/R/Step_post_summary_bubble_table.R \
            "$method" \
            "$V1" \
            "$OUTPUT_DIR" \
            "TRUE" \
            "${OUTPUT_DIR}/plots_summary_${corr_method}_col_gene${gt_suffix}.csv" \
            "${READ_DIR}/bubble_plot_summary_col_gene${gt_suffix}.csv" \
            "$corr_method" \
            "$V2_trans_factor" \
            "$V2_norm_factor" \
            "${trans_factor[@]}"
    done
done