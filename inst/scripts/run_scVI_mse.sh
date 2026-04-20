#!/bin/bash
source ../One_Shifting/inst/config/config_run_scVI_mse.sh

echo "INPUT_FILE: $INPUT_FILE"
echo "INPUT_CATEGORY: $V1"
echo "READ_DIR: $READ_DIR"

method="scVI_MSE"
CONDITION="given_${V2}_${V2_norm_factor}_${V2_trans_factor}"
OUTPUT_DIR="${READ_DIR}/${method}/${CONDITION}"
mkdir -p $OUTPUT_DIR

trans_factor=("no_trans" "sqrt" "sqrt+1" "log2" "count+1" "log2(count+2)")
norm_factor=("no_norm" 1000000 100000 10000 1000 "standardize")
norm_factor_string=$(IFS=','; echo "${norm_factor[*]}")

module load conda_R

# V1: rna; V2: atac gene activity
Rscript ../One_Shifting/R/Step1_build_count_matrix.R \
  $INPUT_FILE \
  $GROUND_TRUTH_FILE \
  "$READ_DIR" \
  "$norm_factor_string" \
  "$norm_factor_string" \
  "TRUE"

# dim: 10412 1393

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

# train - find par
echo "=== Step 4: Training VAE on filtered data ==="
for this_trans_factor in "${trans_factor[@]}"; do
    OUT_DIR="$OUTPUT_DIR/$this_trans_factor" 

    mkdir -p "$OUT_DIR"

    for factor in "${norm_factor[@]}"; do
        DATA_PATH1="$READ_DIR/$V1/Count_Matrix_800_norm_by_$factor.feather"
        DATA_PATH2="$READ_DIR/$V2/Count_Matrix_800_norm_by_$V2_norm_factor.feather"
        SUMMARY_FILE="${OUT_DIR}/vae_artificial_ground_truth_hyper_par.tsv"

        source ~/.bashrc
        conda activate vae_env2
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

        python -u ../One_Shifting/myproject/Step2_train_scVI_mse.py \
        --data_path1 "$DATA_PATH1" \
        --data_path2 "$DATA_PATH2" \
        --out_summary "$SUMMARY_FILE" \
        --early_stop \
        --hidden_grid "128" \
        --patience 5 \
        --n_splits 2 \
        --eval_metric val_loss \
        --trans1_grid "$this_trans_factor" \
        --trans2_grid "$V2_trans_factor" \
        --nonzero_weight_grid 1.0

        # reconstruct
        SAVED_DIR="${OUT_DIR}/saved_models"
        OUT_PATH="${OUT_DIR}/reconstruct_trans_by_${this_trans_factor}_norm_by_${factor}.feather"

        python ../One_Shifting/myproject/Step3_reconstruct_scVI_mse.py \
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

        Figure_DIR="$OUTPUT_DIR/Figures_row"
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
          "row" \
          "${V1}" \
          "${method}"        

        sleep 2
    done
done


module load conda_R
 
Rscript ../One_Shifting/R/run_combine_figures.R \
  "$OUTPUT_DIR/Figures_col" \

Rscript ../One_Shifting/R/run_combine_figures.R \
  "$OUTPUT_DIR/Figures_row" \
  "row"

# Rscript /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/compare_Dec_12/Step12_combine_plot_no_data.R \
#   "$OUTPUT_DIR/ROC" \
#   "roc" \
#   30

# Rscript /dcs10/hongkai/data/yhu1/Autoencoder/artificial_ground_truth_compare_km/final_model_2_0/compare_Dec_12/Step12_combine_plot_no_data.R \
#   "$OUTPUT_DIR/ROC_balanced" \
#   "roc_balanced" \
#   30


Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_pearson_col_gene.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_pearson_col_gene_noGTzero.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_spearman_col_gene.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_spearman_col_gene_noGTzero.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_pearson_row_cell.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_pearson_row_cell_noGTzero.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_spearman_row_cell.csv" \
  "log(count+2)"

Rscript ../One_Shifting/R/run_summary_scatter_plot.R \
  "${OUTPUT_DIR}/plots_summary_spearman_row_cell_noGTzero.csv" \
  "log(count+2)"
