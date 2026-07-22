#!/bin/bash
INPUT_FILE="$1"
READ_DIR="$2"
method="$3"

echo "INPUT_FILE: $INPUT_FILE"
echo "READ_DIR: $READ_DIR"
echo "METHOD: $method"


V1="normed_data"
count_matrix_dir="${READ_DIR}/${V1}"
mkdir -p $count_matrix_dir
OUTPUT_DIR="${READ_DIR}/${method}"
mkdir -p $OUTPUT_DIR

registry_file="${READ_DIR}/recon_registry.csv"
trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000 "standardize")
norm_factor_string=$(IFS=','; echo "${norm_factor[*]}")

REGISTRY_LOCK="$READ_DIR/.registry_lock"
REGISTRY_DONE="$READ_DIR/.registry_initialized"

if [ ! -f "$REGISTRY_DONE" ]; then
    if mkdir "$REGISTRY_LOCK" 2>/dev/null; then
        rm -f "$registry_file"
        touch "$REGISTRY_DONE"
        rmdir "$REGISTRY_LOCK"
    else
        echo "Waiting for registry initialization..."
        while [ ! -f "$REGISTRY_DONE" ]; do
            sleep 2
        done
    fi
fi

module load conda_R

LOCK_FILE="$count_matrix_dir/.count_matrix_done"
if [ ! -f "$LOCK_FILE" ]; then
    mkdir -p "$(dirname $LOCK_FILE)"
    if mkdir "$count_matrix_dir/.count_matrix_lock" 2>/dev/null; then
        # The job that grabs the lock is executed
        Rscript ../One_Shifting/R/Step1_build_count_matrix_UMAP.R \
          $INPUT_FILE \
          "${count_matrix_dir}" \
          "${norm_factor_string}"
        touch "$LOCK_FILE"
        rmdir "$count_matrix_dir/.count_matrix_lock"
    else
        # waiting for the job that didn't get the lock
        echo "Waiting for count matrix to be built..."
        while [ ! -f "$LOCK_FILE" ]; do
            sleep 10
        done
    fi
fi
echo "Count matrix ready, proceeding..."
echo "===Finish build count matrix==="
  
echo "=== Starting ${method} pipeline ==="

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
    METHOD_ARGS="--hidden_grid 512,256,128,64"
elif [ "$method" = "Transformer_denoise" ]; then
    METHOD_ARGS="--n_tokens_grid 32 --d_model_grid 64 --nhead_grid 4 --num_layers_grid 1 --dim_feedforward_grid 128 --dropout_grid 0.1"
fi
        
# train - find par
echo "=== Step 4: Training ${method} on filtered data ==="
for this_trans_factor in "${trans_factor[@]}"; do
    OUT_DIR="$OUTPUT_DIR/$this_trans_factor" 

    mkdir -p "$OUT_DIR"

    DATA_PATH1="$READ_DIR/$V1/Count_Matrix_norm_by_no_norm.feather"
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

    # recon
    OUT_PATH="${OUT_DIR}/reconstruct_trans_by_${this_trans_factor}_norm_by_${factor}.feather"

    # correlation
    echo "=== Step 7: Correlation analysis ==="

    if [ -f "$OUT_PATH" ]; then
        flock "$registry_file.lock" Rscript ../One_Shifting/R/run_record_UMAP_recon_path.R \
            "$registry_file" \
            "$OUT_PATH" \
            "$method" \
            "$this_trans_factor" \
            "$factor"
    else
        echo "[WARNING] Reconstruct file not found: $OUT_PATH"
    fi

    sleep 2

done
