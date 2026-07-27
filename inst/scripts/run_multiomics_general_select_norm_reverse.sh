#!/bin/bash
CONFIG_FILE="$1"
method="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$method" ]; then
    echo "Usage: bash run_pipeline.sh <config_file> <method>"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    echo "  method: VAE, DCA_mse, scVI_mse, Transformer_denoise"
    exit 1
fi

source "$CONFIG_FILE"

echo "INPUT_FILE: $INPUT_FILE"
echo "GROUND_TRUTH_FILE: $GROUND_TRUTH_FILE"
echo "READ_DIR: $READ_DIR"
echo "INPUT_CATEGORY: $V1"
echo "OUTPUT_CATEGORY: $V2"
echo "V2_norm_factor: $V2_norm_factor"
echo "V2_trans_factor: $V2_trans_factor"
echo "METHOD: $method"


CONDITION="given_${V2}_${V2_norm_factor}_${V2_trans_factor}"
OUTPUT_DIR="${READ_DIR}/${method}/${CONDITION}"
mkdir -p $OUTPUT_DIR

trans_factor=("no_trans" "count+1" "sqrt" "sqrt+1" "log2" "log2(count+2)")
# trans_factor=("log2" "count+1" "log2(count+2)")
# norm_factor=("no_norm" 1000000 100000 10000 1000 "standardize")
norm_factor=("no_norm")
norm_factor_string=$(IFS=','; echo "${norm_factor[*]}")

module load conda_R

LOCK_FILE="$READ_DIR/.count_matrix_done"
LOCK_DIR="$READ_DIR/.count_matrix_lock"
MAX_WAIT=180  # 30

if [ ! -f "$LOCK_FILE" ]; then
    mkdir -p "$READ_DIR"

    while true; do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

            Rscript ../One_Shifting/R/Step1_build_count_matrix.R \
              $INPUT_FILE \
              $GROUND_TRUTH_FILE \
              "$READ_DIR" \
              "$norm_factor_string" \
              "$norm_factor_string" \
              "TRUE" \
              "p25"

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
  
echo "=== Starting ${method} pipeline ==="

for data_mode in default v1_trans_v2_trans v1_reverse v1_trans_v2_trans_norm_100000 v1_trans_v2_trans_norm_factor; do
    if [ "$data_mode" = "default" ]; then
        mode_suffix="v1_trans_v2_no_trans"
    elif [ "$data_mode" = "v1_trans_v2_trans" ]; then
        mode_suffix="v1_trans_v2_trans"
    elif [ "$data_mode" = "v1_reverse" ]; then
        mode_suffix="v1_reverse_v2_no_trans"
    elif [ "$data_mode" = "v1_trans_v2_trans_norm_100000" ]; then
        mode_suffix="v1_trans_v2_trans_norm_100000"
    elif [ "$data_mode" = "v1_trans_v2_trans_norm_factor" ]; then
        mode_suffix="v1_trans_v2_trans_norm_factor"
    fi

    for corr_dir in col row; do
        Figure_DIR="$OUTPUT_DIR/Figures_${corr_dir}_${mode_suffix}"
        rm -f "${Figure_DIR}"/plots_summary_*.csv
    done
done


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
    METHOD_ARGS="--beta_grid 0 --hidden_grid1 4096 --hidden_grid2 1024"
elif [ "$method" = "DCA_mse" ]; then
    METHOD_ARGS="--dropout_grid 0.0 --hidden_grid1 4096 --hidden_grid2 1024"
elif [ "$method" = "scVI_mse" ]; then
    METHOD_ARGS="--beta_grid 0 --hidden_grid 512,256,128,64"
elif [ "$method" = "Transformer_denoise" ]; then
    METHOD_ARGS="--n_tokens_grid 32 --d_model_grid 64 --nhead_grid 4 --num_layers_grid 1 --dim_feedforward_grid 128 --dropout_grid 0.1"
fi
        
# train - find par
echo "=== Step 4: Training ${method} on filtered data ==="
for this_trans_factor in "${trans_factor[@]}"; do
    OUT_DIR="$OUTPUT_DIR/$this_trans_factor" 

    mkdir -p "$OUT_DIR"

    DATA_PATH1="$READ_DIR/$V1/Count_Matrix_norm_by_no_norm.feather"
    DATA_PATH2="$READ_DIR/$V2/Count_Matrix_norm_by_no_norm.feather"
    SUMMARY_FILE="${OUT_DIR}/hyper_par_report.tsv"

    source ~/.bashrc
    conda activate vae_env2
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

    python -u ../One_Shifting/myproject/Step2_3_train_${method}_norm.py \
    --data_path "$DATA_PATH1" \
    --out_summary "$SUMMARY_FILE" \
    --early_stop \
    --patience 10 \
    --n_splits 5 \
    --trans_grid "$this_trans_factor" \
    $METHOD_ARGS
    
    python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
    conda deactivate 

    # reconstruct
    SAVED_DIR="${OUT_DIR}/saved_models"

    factor=$(awk -F'\t' '$1=="# selected_norm"{print $2; exit}' "$SUMMARY_FILE")

    OUT_PATH="${OUT_DIR}/reconstruct_trans_by_${this_trans_factor}_norm_by_${factor}.feather"
    INPUT_TRANS_PATH="${OUT_DIR}/input_trans_by_${this_trans_factor}_norm_by_${factor}.feather"

    # correlation
    echo "=== Step 7: Correlation analysis ==="
    module load conda_R

    # for data_mode in default v1_trans_v2_trans v1_reverse v1_trans_v2_trans_norm_100000; do
    for data_mode in default v1_trans_v2_trans v1_trans_v2_trans_norm_100000 v1_trans_v2_trans_norm_factor; do
        if [ "$data_mode" = "default" ]; then
            mode_suffix="v1_trans_v2_no_trans"
        elif [ "$data_mode" = "v1_trans_v2_trans" ]; then
            mode_suffix="v1_trans_v2_trans"
        elif [ "$data_mode" = "v1_reverse" ]; then
            mode_suffix="v1_reverse_v2_no_trans"
        elif [ "$data_mode" = "v1_trans_v2_trans_norm_100000" ]; then
            mode_suffix="v1_trans_v2_trans_norm_100000"
        elif [ "$data_mode" = "v1_trans_v2_trans_norm_factor" ]; then
            mode_suffix="v1_trans_v2_trans_norm_factor"
        fi

        for corr_dir in col row; do
            Figure_DIR="$OUTPUT_DIR/Figures_${corr_dir}_${mode_suffix}"
            mkdir -p "$Figure_DIR"

            Rscript ../One_Shifting/R/run_correlation_scatter_reverse.R \
                "$DATA_PATH2" \
                "$INPUT_TRANS_PATH" \
                "$OUT_PATH" \
                "$Figure_DIR"  \
                "$OUT_DIR/saved_models" \
                "$factor" \
                "$V2_norm_factor" \
                "$this_trans_factor" \
                "$V2_trans_factor" \
                "${Figure_DIR}" \
                "$corr_dir" \
                "${V1}" \
                "${method}" \
                "$data_mode"
        done

        sleep 2

    done
done


# === Summary steps: run after all trans_factor x norm_factor combinations are done ===
echo "=== Post-processing: combine figures and summary ==="
module load conda_R


for data_mode in default v1_trans_v2_trans v1_reverse v1_trans_v2_trans_norm_100000 v1_trans_v2_trans_norm_factor; do
    if [ "$data_mode" = "default" ]; then
        mode_suffix="v1_trans_v2_no_trans"
    elif [ "$data_mode" = "v1_trans_v2_trans" ]; then
        mode_suffix="v1_trans_v2_trans"
    elif [ "$data_mode" = "v1_reverse" ]; then
        mode_suffix="v1_reverse_v2_no_trans"
    elif [ "$data_mode" = "v1_trans_v2_trans_norm_100000" ]; then
        mode_suffix="v1_trans_v2_trans_norm_100000"
    elif [ "$data_mode" = "v1_trans_v2_trans_norm_factor" ]; then
        mode_suffix="v1_trans_v2_trans_norm_factor"
    fi

    COL_Figure_DIR="$OUTPUT_DIR/Figures_col_${mode_suffix}"
    ROW_Figure_DIR="$OUTPUT_DIR/Figures_row_${mode_suffix}"

    for corr_dir in col row; do
        Figure_DIR="$OUTPUT_DIR/Figures_${corr_dir}_${mode_suffix}"

        Rscript ../One_Shifting/R/run_combine_figures_reverse.R \
            "${Figure_DIR}" \
            "${corr_dir}"

        if [ "$corr_dir" = "col" ]; then
            obj="gene"
        else
            obj="cell"
        fi
        for corr_method in pearson spearman; do
            if [ "$data_mode" = "default" ]; then
                mode_tag=""
            else
                mode_tag="_${data_mode}"
            fi
            Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
                "${Figure_DIR}/plots_summary_${corr_method}_${corr_dir}_${obj}${mode_tag}.csv" \
                $method \
                "log(count+2)"
        done
    done

    for corr_method in pearson spearman; do
        if [ "$data_mode" = "default" ]; then
            mode_tag=""
        else
            mode_tag="_${data_mode}"
        fi
        Rscript ../One_Shifting/R/Step_post_summary_bubble_table.R \
            "$method" \
            "$V1" \
            "$OUTPUT_DIR" \
            "TRUE" \
            "${COL_Figure_DIR}/plots_summary_${corr_method}_col_gene${mode_tag}.csv" \
            "${READ_DIR}/bubble_plot_summary_col_gene_${mode_suffix}.csv" \
            "$corr_method" \
            "$V2_trans_factor" \
            "$V2_norm_factor" \
            "${trans_factor[@]}"

        if [[ "$V1" == *RNA* && "$V2" == *ATAC* ]] || [[ "$V1" == *ATAC* && "$V2" == *RNA* ]]; then
            Rscript ../One_Shifting/R/Step_post_summary_bubble_table.R \
                "$method" \
                "$V1" \
                "$OUTPUT_DIR" \
                "FALSE" \
                "${ROW_Figure_DIR}/plots_summary_${corr_method}_row_cell${mode_tag}.csv" \
                "${READ_DIR}/bubble_plot_summary_row_cell_${mode_suffix}.csv" \
                "$corr_method" \
                "$V2_trans_factor" \
                "$V2_norm_factor" \
                "${trans_factor[@]}"
        fi
    done
done