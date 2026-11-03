#!/bin/bash
# Peak-level MSE self-reconstruction for ONE sample and ONE method.
CONFIG_FILE="$1"
method="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$method" ]; then
    echo "Usage: bash run_peak_mse.sh <config_file> <method>"
    echo "  method: VAE, DCA_mse, Transformer_denoise"
    exit 1
fi

source "$CONFIG_FILE" || { echo "[FAIL] could not read config: $CONFIG_FILE"; exit 1; }

case "$method" in
    VAE|DCA_mse|Transformer_denoise) ;;
    *) echo "[FAIL] unsupported method '$method' (VAE, DCA_mse, Transformer_denoise)"; exit 1 ;;
esac

echo "=========================================="
echo "folder_name      : $folder_name"
echo "sample_name      : $sample_name"
echo "method           : $method"
echo "input_path(h5ad) : $input_path"
echo "OUT_DIR          : $OUT_DIR"
echo "READ_DIR         : $READ_DIR"
echo "V1 / V2          : $V1 / $V2"
echo "INPUT_FILE       : $INPUT_FILE"
echo "GROUND_TRUTH_FILE: $GROUND_TRUTH_FILE"
echo "genome build     : $GENOME_BUILD"
echo "=========================================="

CONDITION="given_${V2}_${V2_norm_factor}_${V2_trans_factor}"
OUTPUT_DIR="${READ_DIR}/${method}/${CONDITION}"
CACHE_DIR="${READ_DIR}/eval_cache"
SUMMARY_CSV="${READ_DIR}/peak_mse_summary_${method}.csv"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

trans_factor=("no_trans" "count+1" "sqrt" "sqrt+1" "log2" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000)
norm_string=$(IFS=','; echo "${norm_factor[*]}")

echo "=== Step 1: read the h5ad and write the q95 peak counts ==="
module load conda_R

if [ ! -f "$INPUT_FILE" ] || [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "[Step 1] extracting from $input_path"
    Rscript ../One_Shifting/R/load_rna_atac_from_h5ad.R \
        "$input_path" \
        "$sample_name" \
        "${OUT_DIR}/${folder_name}"
    if [ $? -ne 0 ]; then echo "[FAIL] Step 1: load_rna_atac_from_h5ad.R"; exit 1; fi
else
    echo "[Step 1] q95 peaks and RNA already present, not re-extracting"
fi

# The q95 block inside load_rna_atac_from_h5ad.R is wrapped in a tryCatch that only warns,
# so its absence has to be checked rather than assumed.
if [ ! -f "$INPUT_FILE" ]; then
    echo "[FAIL] Step 1: $INPUT_FILE was not produced (check the log for 'failed to save atac_peak_counts_top_q95')"
    exit 1
fi
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "[FAIL] Step 1: $GROUND_TRUTH_FILE was not produced"
    exit 1
fi

echo "=== Step 2: build the normalized peak matrices (one fixed peak set) ==="
if [ ! -f "${READ_DIR}/${V1}/peak_cell_manifest.txt" ]; then
    Rscript ../One_Shifting/R/prep_peak_q95_matrix.R \
        "$INPUT_FILE" \
        "$READ_DIR" \
        "$V1" \
        "$norm_string" \
        "$GENOME_BUILD"
    if [ $? -ne 0 ]; then echo "[FAIL] Step 2: prep_peak_q95_matrix.R"; exit 1; fi
else
    echo "[Step 2] peak matrices already built, reusing the fixed peak set"
fi

RAW_PEAK_PATH="${READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather"
LIB_SIZES_CSV="${READ_DIR}/${V1}/lib_sizes.csv"
for f in "$RAW_PEAK_PATH" "$LIB_SIZES_CSV"; do
    if [ ! -f "$f" ]; then echo "[FAIL] Step 2: missing $f"; exit 1; fi
done

if [ "$method" = "VAE" ]; then
    METHOD_ARGS="--beta_grid 0 --hidden_grid1 2048 --hidden_grid2 512 --batch_size_grid 16 --epochs_inner 100 --epochs_outer 100"
elif [ "$method" = "DCA_mse" ]; then
    METHOD_ARGS="--dropout_grid 0.0 --hidden_grid1 2048 --hidden_grid2 512 --batch_size_grid 16 --epochs_inner 100 --epochs_outer 100"
elif [ "$method" = "Transformer_denoise" ]; then
    METHOD_ARGS="--n_tokens_grid 32 --d_model_grid 64 --nhead_grid 4 --num_layers_grid 1 --dim_feedforward_grid 128 --dropout_grid 0.1 --batch_size_grid 16 --epochs_inner 100 --epochs_outer 100"
fi

echo "=== Step 3: train, reconstruct and score ${#trans_factor[@]} x ${#norm_factor[@]} combinations ==="
for this_trans_factor in "${trans_factor[@]}"; do
    for this_norm_factor in "${norm_factor[@]}"; do

        COMBO="trans=${this_trans_factor} norm=${this_norm_factor}"
        COMBO_DIR="${OUTPUT_DIR}/${this_trans_factor}/norm_${this_norm_factor}"
        mkdir -p "$COMBO_DIR"
        echo ""
        echo "---- ${COMBO} ----"

        DATA_PATH1="${READ_DIR}/${V1}/Count_Matrix_norm_by_${this_norm_factor}.feather"
        if [ ! -f "$DATA_PATH1" ]; then
            echo "[FAIL] ${COMBO}: missing $DATA_PATH1"; exit 1
        fi
        # Self-reconstruction: the loss is peaks against the same peaks. Step2's interface
        # wants a second matrix, so it gets the same peak matrix -- RNA is never passed
        # here and never reaches the objective or the model selection.
        DATA_PATH2="$DATA_PATH1"
        SUMMARY_FILE="${COMBO_DIR}/hyper_par_norm_by_${this_norm_factor}.tsv"
        SAVED_DIR="${COMBO_DIR}/saved_models"
        OUT_PATH="${COMBO_DIR}/reconstruct_trans_by_${this_trans_factor}_norm_by_${this_norm_factor}.feather"

        source ~/.bashrc
        conda activate vae_env2

        echo "[train] ${method} | ${COMBO}"
        python -u ../One_Shifting/myproject/Step2_train_${method}.py \
            --data_path1 "$DATA_PATH1" \
            --data_path2 "$DATA_PATH2" \
            --out_summary "$SUMMARY_FILE" \
            --early_stop \
            --eval_metric val_loss \
            --trans1_grid "$this_trans_factor" \
            --trans2_grid "$this_trans_factor" \
            --normalization "$this_norm_factor" \
            --nonzero_weight_grid 1.0 \
            --allow_zero_rows \
            $METHOD_ARGS
        if [ $? -ne 0 ]; then echo "[FAIL] train ${method} | ${COMBO}"; conda deactivate; exit 1; fi

        echo "[reconstruct] ${method} | ${COMBO}"
        python -u ../One_Shifting/myproject/Step3_reconstruct_${method}.py \
            --data_path1 "$DATA_PATH1" \
            --data_path2 "$DATA_PATH2" \
            --transformed_out_dir "$COMBO_DIR" \
            --saved_models_dir "$SAVED_DIR" \
            --out_path "$OUT_PATH" \
            --normalization "$this_norm_factor" \
            --allow_zero_rows
        if [ $? -ne 0 ]; then echo "[FAIL] reconstruct ${method} | ${COMBO}"; conda deactivate; exit 1; fi

        python -c "import torch; torch.cuda.empty_cache(); print('GPU cleared')"
        conda deactivate

        if [ ! -f "$OUT_PATH" ]; then
            echo "[FAIL] ${COMBO}: reconstruction not written: $OUT_PATH"; exit 1
        fi

        # Step3 writes the reconstruction in transformed + normalized peak space and does
        # not reverse anything, so the reversal happens once, here, before summarizing.
        echo "[reverse + summarize + score] ${COMBO}"
        module load conda_R
        Rscript ../One_Shifting/R/peak_recon_to_gene_activity_eval.R \
            "$OUT_PATH" \
            "$RAW_PEAK_PATH" \
            "$LIB_SIZES_CSV" \
            "$GROUND_TRUTH_FILE" \
            "$this_trans_factor" \
            "$this_norm_factor" \
            "$COMBO_DIR" \
            "$CACHE_DIR" \
            "$method" \
            "$SUMMARY_CSV"
        if [ $? -ne 0 ]; then echo "[FAIL] evaluate ${COMBO}"; exit 1; fi
    done
done

echo "=== Step 4: summary over all combinations ==="
module load conda_R
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "[FAIL] Step 4: $SUMMARY_CSV was never written"; exit 1
fi

for corr_method in pearson spearman; do
    PLOT_CSV="${READ_DIR}/peak_mse_summary_${method}_${corr_method}.csv"
    awk -F',' -v m="$corr_method" 'NR==1 || $4=="\""m"\"" || $4==m' "$SUMMARY_CSV" > "$PLOT_CSV"
    Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
        "$PLOT_CSV" \
        "$method"
    if [ $? -ne 0 ]; then echo "[FAIL] Step 4: summary plot for $corr_method"; exit 1; fi
done

echo ""
echo "=== Done: ${method} | ${sample_name} ==="
echo "summary : $SUMMARY_CSV"
echo "outputs : $OUTPUT_DIR"