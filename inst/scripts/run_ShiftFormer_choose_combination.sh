#!/bin/bash
CONFIG_FILE="$1"
dropout_keep_par="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$dropout_keep_par" ]; then
    echo "Usage: bash run_ShiftFormer_choose_combination.sh <config_file> <dropout_keep_par>"
    echo "  config_file      : e.g. inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh"
    echo "  dropout_keep_par : positive multiplier of each cell's library, e.g. 0.7, 0.9, 1, 1.2 (1 = no dropout)"
    exit 1
fi

# keep fraction: numeric, p > 0 (p > 1 resamples a larger total); canonical form so 0.90 / 0.9 and 1.0 / 1 share one folder
if ! awk -v p="$dropout_keep_par" 'BEGIN { exit !(p ~ /^[0-9]*\.?[0-9]+$/ && p + 0 > 0) }'; then
    echo "[FAIL] dropout_keep_par='$dropout_keep_par' must be a positive number"
    exit 1
fi
dropout_keep_par=$(awk -v p="$dropout_keep_par" 'BEGIN { printf "%g", p + 0 }')
export dropout_keep_par

source "$CONFIG_FILE"

DROPOUT_SEED=42
METHOD_DIR="${READ_DIR}/Transformer_denoise_choose_combination"
INPUT_DIR="${METHOD_DIR}/input"
RAW_PATH="${RAW_READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather"
RAW_DONE_FILE="${RAW_READ_DIR}/.count_matrix_done"

if [ "$dropout_keep_par" = "1" ]; then
    DROPOUT_PATH="$RAW_PATH"
else
    DROPOUT_PATH="${INPUT_DIR}/${V1}_dropout_keep_${dropout_tag}_seed${DROPOUT_SEED}_Count_Matrix_norm_by_no_norm.feather"
fi
DROPOUT_DONE_FILE="${INPUT_DIR}/.dropout_done"

echo "RAW_READ_DIR: $RAW_READ_DIR"
echo "READ_DIR: $READ_DIR"
echo "INPUT_CATEGORY: $V1"
echo "dropout_keep_par: $dropout_keep_par"
echo "TRAIN (raw): $RAW_PATH"
echo "TEST (dropout): $DROPOUT_PATH"
echo "OUTPUT_DIR: $METHOD_DIR"

# Raw matrix must already be built by Step1 (never rebuilt here; only this modality is read)
if [ ! -f "$RAW_DONE_FILE" ] || [ ! -f "$RAW_PATH" ]; then
    echo "[FAIL] raw count matrix not ready ($RAW_DONE_FILE / $RAW_PATH). Build it with Step1 first."
    exit 1
fi

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[CHECK_ONLY] paths resolved; nothing generated or trained."
    exit 0
fi

# Every submission retrains and rewrites summary.tsv / report.txt / run_log.json in METHOD_DIR
# (no done marker, no lock: nothing has to be deleted before resubmitting).
mkdir -p "$METHOD_DIR"

# === Dropout matrix: generated once from the raw matrix, reused afterwards ===
# The R script writes a temp file and renames it, so an existing file is always complete;
# a rerun with the same raw matrix, keep and seed would produce the identical matrix anyway.
if [ "$dropout_keep_par" != "1" ]; then
    mkdir -p "$INPUT_DIR"
    if [ -f "$DROPOUT_PATH" ]; then
        echo "Dropout matrix exists, reusing: $DROPOUT_PATH"
    else
        module load conda_R
        Rscript ../One_Shifting/R/run_build_dropout_from_raw.R \
            "$RAW_PATH" \
            "$DROPOUT_PATH" \
            "$dropout_keep_par" \
            "$DROPOUT_SEED" || { echo "[FATAL] dropout generation failed" >&2; exit 1; }
        printf "raw=%s\nkeep_par=%s\nseed=%s\nfunction=R/dropout_multinomial.R (cell_axis=row)\ncreated=%s\n" \
            "$RAW_PATH" "$dropout_keep_par" "$DROPOUT_SEED" "$(date -Iseconds)" > "$DROPOUT_DONE_FILE"
    fi
fi
echo "===Finish preparing inputs==="

# === ShiftFormer ===
echo "======Run ShiftFormer choose_combination======"
source ~/.bashrc
conda activate vae_env2
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

python ../One_Shifting/myproject/Step2_3_train_ShiftFormer.py \
    --train "$RAW_PATH" \
    --test "$DROPOUT_PATH" \
    --out_dir "$METHOD_DIR" \
    --pipeline_genes "$RAW_PATH"
status=$?

python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
conda deactivate

if [ $status -eq 0 ]; then
    touch "$DONE_FILE"
    echo "=== Done: ${METHOD_DIR}/report.txt ==="
elif [ $status -eq 3 ]; then
    echo "[WARNING] Some combinations failed; provisional results saved in ${METHOD_DIR}. No completion marker was created." >&2
    exit 0
else
    echo "[FATAL] ShiftFormer exited with $status (2 = stopped for user); see ${METHOD_DIR}" >&2
    exit $status
fi