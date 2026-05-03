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
# trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
trans_factor=("count+1" "log2(count+2)")
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
          "$norm_factor_string"
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

    for factor in "${norm_factor[@]}"; do

        # Skip incompatible (standardize + sqrt/log) combinations
        if [ "$factor" = "standardize" ]; then
            case "$this_trans_factor" in
                no_trans|count+1) ;;  # allow
                *) echo "[SKIP] $this_trans_factor incompatible with standardize (negative values)"
                   continue ;;
            esac
        fi

        DATA_PATH1="$READ_DIR/$V1/Count_Matrix_norm_by_$factor.feather"
        SUMMARY_FILE="${OUT_DIR}/hyper_par_norm_by_${factor}.tsv"

        source ~/.bashrc
        conda activate vae_env2
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

        python -u ../One_Shifting/myproject/UMAP/UMAP_Step2_train_${method}.py \
        --data_path1 "$DATA_PATH1" \
        --out_summary "$SUMMARY_FILE" \
        --early_stop \
        --patience 5 \
        --n_splits 2 \
        --eval_metric val_loss \
        --trans1_grid "$this_trans_factor" \
        --nonzero_weight_grid 1.0 \
        $METHOD_ARGS
        
        # reconstruct
        SAVED_DIR="${OUT_DIR}/saved_models"
        OUT_PATH="${OUT_DIR}/reconstruct_trans_by_${this_trans_factor}_norm_by_${factor}.feather"

        python -u ../One_Shifting/myproject/UMAP/UMAP_Step3_reconstruct_${method}.py \
        --data_path1 "$DATA_PATH1" \
        --transformed_out_dir "$OUT_DIR" \
        --saved_models_dir "$SAVED_DIR" \
        --out_path "$OUT_PATH"

        python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
        conda deactivate 

        sleep 2

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
done
