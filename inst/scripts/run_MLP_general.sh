#!/bin/bash
CONFIG_FILE="$1"
method="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$method" ]; then
    echo "Usage: bash run_pipeline.sh <config_file> <method>"
    echo "  config_file: path to config (e.g. configs/pbmc_atac_rna.sh)"
    echo "  method: MLP"
    exit 1
fi

source "$CONFIG_FILE"

echo "INPUT_FILE: $INPUT_FILE"
echo "GROUND_TRUTH_FILE: $GROUND_TRUTH_FILE"
echo "READ_DIR: $READ_DIR"
echo "INPUT_CATEGORY: $V1"
echo "OUTPUT_CATEGORY: $V1"
echo "METHOD: $method"

N_HVG=100
TEST_FRAC=0.2
VAL_FRAC=0.1
SEED=42

CONDITION="given_${V2}_${V2_norm_factor}_${V2_trans_factor}"
OUTPUT_DIR="${READ_DIR}/${method}/${CONDITION}"
mkdir -p $OUTPUT_DIR

trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000 "standardize")
norm_factor_string=$(IFS=','; echo "${norm_factor[*]}")

module load conda_R


LOCK_FILE="$READ_DIR/.count_matrix_done"
if [ ! -f "$LOCK_FILE" ]; then
    mkdir -p "$(dirname $LOCK_FILE)"
    if mkdir "$READ_DIR/.count_matrix_lock" 2>/dev/null; then
        # The job that grabs the lock is executed
        Rscript ../One_Shifting/R/Step1_build_count_matrix.R \
          $INPUT_FILE \
          $GROUND_TRUTH_FILE \
          "$READ_DIR" \
          "$norm_factor_string" \
          "$norm_factor_string" \
          "TRUE" \
          "p25"
        touch "$LOCK_FILE"
        rmdir "$READ_DIR/.count_matrix_lock"
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

echo "===Find TF and hvg==="

GENE_LIST_DIR="${OUTPUT_DIR}/gene_lists"
mkdir -p "$GENE_LIST_DIR"
 
Rscript ../One_Shifting/R/run_select_tf_hvg.R \
    "$READ_DIR/$V1/Count_Matrix_norm_by_no_norm.feather" \
    $GENE_LIST_DIR \
    $N_HVG
 
TF_NAMES="${GENE_LIST_DIR}/tf_names.txt"
HVG_NAMES="${GENE_LIST_DIR}/hvg_target_names.txt"

# Ground Truth RNA
DATA_PATH2="$READ_DIR/$V2/Count_Matrix_norm_by_$V2_norm_factor.feather"

Rscript ../One_Shifting/R/run_MLP_data_pre.R \
    $DATA_PATH2 \
    "${OUTPUT_DIR}/${V2}_Ground_TRUTH" \
    $TF_NAMES \
    $HVG_NAMES \
    $TEST_FRAC \
    $VAL_FRAC \
    $SEED
    
echo "=== Starting ${method} pipeline ==="


# train - find par
echo "=== Step 4: Training VAE on filtered data ==="
for this_trans_factor in "${trans_factor[@]}"; do
    for factor in "${norm_factor[@]}"; do

        # Skip incompatible (standardize + sqrt/log) combinations
        if [ "$factor" = "standardize" ]; then
            case "$this_trans_factor" in
                no_trans|count+1) ;;  # allow
                *) echo "[SKIP] $this_trans_factor incompatible with standardize (negative values)"
                   continue ;;
            esac
        fi

        OUT_DIR="$OUTPUT_DIR/${this_trans_factor}/norm_${factor}"
        mkdir -p "$OUT_DIR"

        DATA_PATH1="$READ_DIR/$V1/Count_Matrix_norm_by_$factor.feather"
        
        module load conda_R
        Rscript ../One_Shifting/R/run_MLP_data_pre.R \
            $DATA_PATH1 \
            $OUT_DIR \
            $TF_NAMES \
            $HVG_NAMES \
            $TEST_FRAC \
            $VAL_FRAC \
            $SEED

        source ~/.bashrc
        conda activate vae_env2
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

        python -u ../One_Shifting/myproject/Step2_train_MLP.py \
        --tf_train ${OUT_DIR}/tf_train.feather \
        --tf_val ${OUT_DIR}/tf_val.feather \
        --tf_test ${OUT_DIR}/tf_test.feather \
        --target_train ${OUT_DIR}/target_train.feather \
        --target_val ${OUT_DIR}/target_val.feather \
        --target_test ${OUT_DIR}/target_test.feather \
        --trans "${this_trans_factor}" \
        --epochs 60 \
        --patience 10 \
        --out_summary ${OUT_DIR}/mlp_results.tsv

        # predict
        SAVED_DIR="${OUT_DIR}/saved_models"
        OUT_PATH="${OUT_DIR}/input_predicted_trans_by_${this_trans_factor}_norm_by_${factor}.feather" # input predicted
        OUT_pre="${OUT_DIR}/input_true_trans_by_${this_trans_factor}_norm_by_${factor}.feather" # input
        OUT_GROUND_TRUTH="${OUT_DIR}/ground_truth_trans_by_${this_trans_factor}_norm_by_${factor}.feather" # ground truth

        python ../One_Shifting/myproject/Step3_predict_MLP.py \
        --model_dir ${SAVED_DIR} \
        --tf_test ${OUT_DIR}/tf_test.feather \
        --target_test ${OUT_DIR}/target_test.feather \
        --ground_test ${OUTPUT_DIR}/${V2}_Ground_TRUTH/target_test.feather \
        --out_pred ${OUT_PATH} \
        --out_truth ${OUT_pre} \
        --out_ground_truth ${OUT_GROUND_TRUTH}

        python -c "import torch; torch.cuda.empty_cache(); del torch; print('GPU cleared')"
        conda deactivate 

        # correlation
        echo "=== Step 7: Correlation analysis ==="
        module load conda_R

        Figure_DIR="$OUTPUT_DIR/Figures_col"
        mkdir -p "$Figure_DIR"

        Rscript ../One_Shifting/R/run_correlation_scatter.R \
          "$OUT_GROUND_TRUTH" \
          "$OUT_pre" \
          "$OUT_PATH" \
          "$Figure_DIR"  \
          "$SAVED_DIR" \
          "$factor" \
          "$V2_norm_factor" \
          "$this_trans_factor" \
          "$V2_trans_factor" \
          "${OUTPUT_DIR}" \
          "col" \
          "${V1}" \
          "${method}"        

        Figure_DIR="$OUTPUT_DIR/Figures_row"
        mkdir -p "$Figure_DIR"

        Rscript ../One_Shifting/R/run_correlation_scatter.R \
          "$OUT_GROUND_TRUTH" \
          "$OUT_pre" \
          "$OUT_PATH" \
          "$Figure_DIR"  \
          "$SAVED_DIR" \
          "$factor" \
          "$V2_norm_factor" \
          "$this_trans_factor" \
          "$V2_trans_factor" \
          "${OUTPUT_DIR}" \
          "row" \
          "${V1}" \
          "${method}"        

        sleep 2

    done
done


module load conda_R
 
Rscript ../One_Shifting/R/run_combine_figures.R \
  "$OUTPUT_DIR/Figures_col" 

Rscript ../One_Shifting/R/run_combine_figures.R \
  "$OUTPUT_DIR/Figures_row" \
  "row"


sleep 2

# Summary scatter plots
for corr_method in pearson spearman; do
    for gt_suffix in "" "_noGTzero"; do
        Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
          "${OUTPUT_DIR}/plots_summary_${corr_method}_col_gene${gt_suffix}.csv" \
          $method \
          "log(count+2)"
    done
done

for corr_method in pearson spearman; do
    for gt_suffix in "" "_noGTzero"; do
        Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
          "${OUTPUT_DIR}/plots_summary_${corr_method}_row_cell${gt_suffix}.csv" \
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

        if [[ "$V1" == *RNA* && "$V2" == *ATAC* ]] || [[ "$V1" == *ATAC* && "$V2" == *RNA* ]]; then
            Rscript ../One_Shifting/R/Step_post_summary_bubble_table.R \
                "$method" \
                "$V1" \
                "$OUTPUT_DIR" \
                "FALSE" \
                "${OUTPUT_DIR}/plots_summary_${corr_method}_row_cell${gt_suffix}.csv" \
                "${READ_DIR}/bubble_plot_summary_row_cell${gt_suffix}.csv" \
                "$corr_method" \
                "$V2_trans_factor" \
                "$V2_norm_factor" \
                "${trans_factor[@]}"
        fi
    done
done
