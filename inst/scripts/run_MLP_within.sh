#!/bin/bash
# Task 1 -- within one modality: TF features -> non-TF highly variable genes.
#
# The modality is a parameter, not a constant: point MODALITY at the RNA folder and this
# runs RNA TF -> RNA HVG; point it at the ATAC folder and it runs gene-activity TF ->
# gene-activity HVG. Nothing below assumes RNA.
#
# The cell split and the gene lists are built ONCE, from the raw-count (no_norm) matrix,
# with selection restricted to training cells. Every (transformation, normalization)
# combination then reuses them, so the combinations stay comparable and the test cells
# never enter feature selection.
CONFIG_FILE="${1:-}"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash run_MLP_general.sh <config_file>"
    exit 1
fi
source "$CONFIG_FILE"

: "${READ_DIR:?}" "${MODALITY:?}" "${N_HVG:=1000}" "${MIN_CELLS:=3}"
: "${TEST_FRAC:=0.2}" "${VAL_FRAC:=0.1}" "${SEED:=42}" "${EPOCHS:=100}" "${PATIENCE:=15}"
: "${HIDDEN1:=256}" "${HIDDEN2:=128}" "${LR:=0.0001}" "${BATCH:=64}" "${DROPOUT:=0.1}"
: "${OUTPUT_SUFFIX=_relu_rawscale}"

OUTPUT_DIR="${READ_DIR}/MLP_within_${MODALITY}${OUTPUT_SUFFIX}"
LISTS_DIR="${OUTPUT_DIR}/gene_lists"
COMPARE="${OUTPUT_DIR}/compare_combinations_trans_norm.csv"
mkdir -p "$LISTS_DIR"

# Labels for the summary figure. MODALITY is "<sample_name>_<RNA|ATAC>", so the sample
# name is stripped from it rather than added a second time. Checked now, not after training.
: "${sample_name:?sample_name must be set (exported by the submitter) to label the summary figure}"
case "$MODALITY" in
    "${sample_name}_"*) ;;
    *) echo "MODALITY='${MODALITY}' does not start with '${sample_name}_'; cannot derive the figure label"; exit 1 ;;
esac
MOD_LABEL="${MODALITY#"${sample_name}_"}"

trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000)

RAW="${READ_DIR}/${MODALITY}/Count_Matrix_norm_by_no_norm.feather"

# fail now, not after hours of training, if the raw-scale inputs are missing
for f in "$RAW" "${READ_DIR}/${MODALITY}/lib_sizes.csv"; do
    [ -f "$f" ] || { echo "Missing required file: $f"; exit 1; }
done

module load conda_R
echo "=== Fixing the split and the gene lists (training cells only) ==="
Rscript ../One_Shifting/R/prep_mlp_data.R \
    --mode lists --task within \
    --input_matrix  "$RAW" \
    --target_matrix "$RAW" \
    --lists_dir "$LISTS_DIR" \
    --n_hvg "$N_HVG" --min_cells "$MIN_CELLS" \
    --test_frac "$TEST_FRAC" --val_frac "$VAL_FRAC" --seed "$SEED"

for this_trans in "${trans_factor[@]}"; do
    for this_norm in "${norm_factor[@]}"; do

        # standardize yields negative values, which sqrt and log cannot take
        if [ "$this_norm" = "standardize" ]; then
            case "$this_trans" in
                no_trans|count+1) ;;
                *) echo "[SKIP] ${this_trans} + standardize"; continue ;;
            esac
        fi

        OUT_DIR="${OUTPUT_DIR}/${this_trans}/norm_${this_norm}"
        MAT="${READ_DIR}/${MODALITY}/Count_Matrix_norm_by_${this_norm}.feather"
        mkdir -p "$OUT_DIR"

        echo "=== Running MLP within ${MODALITY} with ${this_trans} + ${this_norm} ==="
        module load conda_R
        Rscript ../One_Shifting/R/prep_mlp_data.R \
            --mode extract --task within \
            --input_matrix "$MAT" --target_matrix "$MAT" \
            --lists_dir "$LISTS_DIR" --out_dir "$OUT_DIR"

        source ~/.bashrc
        conda activate vae_env2
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

        echo "=== Training MLP within ${MODALITY} with ${this_trans} + ${this_norm} ==="
        python ../One_Shifting/myproject/Step2_train_MLP.py \
            --input_train  "${OUT_DIR}/input_train.feather" \
            --input_val    "${OUT_DIR}/input_val.feather" \
            --target_train "${OUT_DIR}/target_train.feather" \
            --target_val   "${OUT_DIR}/target_val.feather" \
            --model_dir    "${OUT_DIR}/saved_model" \
            --out_summary  "${OUT_DIR}/mlp_results.tsv" \
            --input_trans "${this_trans}" --target_trans "${this_trans}" \
            --input_norm  "${this_norm}"  --target_norm  "${this_norm}" \
            --hidden_grid1 "$HIDDEN1" --hidden_grid2 "$HIDDEN2" \
            --lr_grid "$LR" --batch_size_grid "$BATCH" --dropout "$DROPOUT" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --min_delta 0 --seed "$SEED"

        echo "=== Predicting MLP within ${MODALITY} with ${this_trans} + ${this_norm} ==="
        python ../One_Shifting/myproject/Step3_predict_MLP.py \
            --model_dir   "${OUT_DIR}/saved_model" \
            --input_test  "${OUT_DIR}/input_test.feather" \
            --target_test "${OUT_DIR}/target_test.feather" \
            --out_dir     "${OUT_DIR}/predictions" \
            --target_raw_matrix "${READ_DIR}/${MODALITY}/Count_Matrix_norm_by_no_norm.feather" \
            --target_lib_sizes  "${READ_DIR}/${MODALITY}/lib_sizes.csv" \
            --compare_path "$COMPARE"

        conda deactivate
    done
done

# one summary figure for the whole sweep, on the fully recovered raw-count scale
echo "=== Evaluating MLP ==="
module load conda_R
Rscript ../One_Shifting/R/run_MLP_summary_scatter_plot.R \
    --csv "$COMPARE" \
    --sample "$sample_name" \
    --task within \
    --modality "$MOD_LABEL" \
    --space raw_counts,target_space \
    --out_dir "${OUTPUT_DIR}/figures"

echo "=== Done. Summary: ${COMPARE} ==="