#!/bin/bash
# Task 2 -- cross modality: ATAC gene activity -> RNA expression of the same cells.
#
# Input is every gene-activity feature surviving the training-cell filter (not only TFs);
# target is the non-TF RNA HVGs. Cells are paired by identifier, so the two gene sets need
# not match in content or size.
set -euo pipefail

CONFIG_FILE="${1:-}"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_MLP_cross.sh <config_file>"
    exit 1
fi
source "$CONFIG_FILE"

: "${READ_DIR:?}" "${V1:?}" "${V2:?}" "${N_HVG:=1000}" "${MIN_CELLS:=3}"
: "${TEST_FRAC:=0.2}" "${VAL_FRAC:=0.1}" "${SEED:=42}" "${EPOCHS:=100}" "${PATIENCE:=15}"
: "${HIDDEN1:=256}" "${HIDDEN2:=128}" "${LR:=0.0001}" "${BATCH:=64}" "${DROPOUT:=0.1}"
: "${REUSE_LISTS_DIR:=}" "${TARGET_NORM:=no_norm}" "${TARGET_TRANS:=no_trans}"

OUTPUT_DIR="${READ_DIR}/MLP_cross_${V1}_to_${V2}"
LISTS_DIR="${OUTPUT_DIR}/gene_lists"
COMPARE="${OUTPUT_DIR}/compare_combinations_trans_norm.csv"
mkdir -p "$LISTS_DIR"

# Labels for the summary figure. V1 / V2 are "<sample_name>_<modality>", so the sample name
# is stripped from them rather than added a second time. Checked now, not after training.
: "${sample_name:?sample_name must be set (exported by the submitter) to label the summary figure}"
case "$V1" in "${sample_name}_"*) ;; *) echo "V1='${V1}' does not start with '${sample_name}_'"; exit 1 ;; esac
case "$V2" in "${sample_name}_"*) ;; *) echo "V2='${V2}' does not start with '${sample_name}_'"; exit 1 ;; esac
MOD_LABEL="${V1#"${sample_name}_"}_to_${V2#"${sample_name}_"}"

trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000)

RAW_IN="${READ_DIR}/${V1}/Count_Matrix_norm_by_no_norm.feather"
RAW_TG="${READ_DIR}/${V2}/Count_Matrix_norm_by_no_norm.feather"

module load conda_R
if [ -n "$REUSE_LISTS_DIR" ]; then
    echo "=== Reusing split and target gene list from ${REUSE_LISTS_DIR} ==="
    cp "${REUSE_LISTS_DIR}/cell_split.csv"   "${LISTS_DIR}/cell_split.csv"
    cp "${REUSE_LISTS_DIR}/target_genes.txt" "${LISTS_DIR}/target_genes.txt"
    # the input side still has to be chosen, on this experiment's own training cells
    Rscript ../One_Shifting/R/prep_mlp_data.R \
        --mode lists --task cross \
        --input_matrix "$RAW_IN" --target_matrix "$RAW_TG" \
        --lists_dir "${LISTS_DIR}_inputonly" \
        --n_hvg "$N_HVG" --min_cells "$MIN_CELLS" \
        --test_frac "$TEST_FRAC" --val_frac "$VAL_FRAC" --seed "$SEED"
    cp "${LISTS_DIR}_inputonly/input_genes.txt" "${LISTS_DIR}/input_genes.txt"
    cp "${LISTS_DIR}_inputonly/tf_names.txt"    "${LISTS_DIR}/tf_names.txt"
    cp "${LISTS_DIR}_inputonly/hvg_target_names.txt" "${LISTS_DIR}/hvg_target_names.txt"
else
    echo "=== Fixing the split and the gene lists (training cells only) ==="
    Rscript ../One_Shifting/R/prep_mlp_data.R \
        --mode lists --task cross \
        --input_matrix "$RAW_IN" --target_matrix "$RAW_TG" \
        --lists_dir "$LISTS_DIR" \
        --n_hvg "$N_HVG" --min_cells "$MIN_CELLS" \
        --test_frac "$TEST_FRAC" --val_frac "$VAL_FRAC" --seed "$SEED"
fi

for this_trans in "${trans_factor[@]}"; do
    for this_norm in "${norm_factor[@]}"; do

        if [ "$this_norm" = "standardize" ]; then
            case "$this_trans" in
                no_trans|count+1) ;;
                *) echo "[SKIP] ${this_trans} + standardize"; continue ;;
            esac
        fi

        OUT_DIR="${OUTPUT_DIR}/${this_trans}/norm_${this_norm}"
        MAT_IN="${READ_DIR}/${V1}/Count_Matrix_norm_by_${this_norm}.feather"
        MAT_TG="${READ_DIR}/${V2}/Count_Matrix_norm_by_${TARGET_NORM}.feather"
        mkdir -p "$OUT_DIR"

        module load conda_R
        Rscript ../One_Shifting/R/prep_mlp_data.R \
            --mode extract --task cross \
            --input_matrix "$MAT_IN" --target_matrix "$MAT_TG" \
            --lists_dir "$LISTS_DIR" --out_dir "$OUT_DIR"

        source ~/.bashrc
        conda activate vae_env2

        python -u ../One_Shifting/myproject/Step2_train_MLP_cross.py \
            --input_train  "${OUT_DIR}/input_train.feather" \
            --input_val    "${OUT_DIR}/input_val.feather" \
            --target_train "${OUT_DIR}/target_train.feather" \
            --target_val   "${OUT_DIR}/target_val.feather" \
            --model_dir    "${OUT_DIR}/saved_model" \
            --out_summary  "${OUT_DIR}/mlp_results.tsv" \
            --input_trans "${this_trans}"  --target_trans "${TARGET_TRANS}" \
            --input_norm  "${this_norm}"   --target_norm  "${TARGET_NORM}" \
            --hidden_grid1 "$HIDDEN1" --hidden_grid2 "$HIDDEN2" \
            --lr_grid "$LR" --batch_size_grid "$BATCH" --dropout "$DROPOUT" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --min_delta 0 --seed "$SEED"

        python -u ../One_Shifting/myproject/Step3_predict_MLP_cross.py \
            --model_dir   "${OUT_DIR}/saved_model" \
            --input_test  "${OUT_DIR}/input_test.feather" \
            --target_test "${OUT_DIR}/target_test.feather" \
            --out_dir     "${OUT_DIR}/predictions" \
            --compare_path "$COMPARE"

        conda deactivate
    done
done

# one summary figure for the whole sweep; the target is fixed no_norm + no_trans, so the
# original_scale rows are raw counts
module load conda_R
Rscript ../One_Shifting/R/run_MLP_summary_scatter_plot.R \
    --csv "$COMPARE" \
    --sample "$sample_name" \
    --task cross \
    --modality "$MOD_LABEL" \
    --space original_scale \
    --out_dir "${OUTPUT_DIR}/figures"

echo "=== Done. Summary: ${COMPARE} ==="