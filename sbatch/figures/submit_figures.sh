
module load conda_R

# Find the typical sample by median rank
# rm -f /dcs07/hongkai/data/yhu1/One_Shifting_Results/col_reverse_representative_sample_ranking.csv
# Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/typical_sample_find.R 

# Plot the method specific scatter plot
cd /dcs10/hongkai/data/yhu1/One_Shifting 
BASE_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"

# Figure 2
folder_name="HCA"
sample_name="MA11_heart"
norm="1000"
trans="log2(count+2)"

Rscript ../One_Shifting/R/figures/sample_scatter.R \
    "${BASE_DIR}/${folder_name}/${sample_name}_INPUT_RNA/VAE/given_${sample_name}_ATAC_no_norm_no_trans/Figures_col_v1_raw__v2_raw__recon_revtrans_revnorm/d_pearson_v1_recon_reverse_trans_norm_col_scatter_v1_trans_${trans}_norm_${norm}_v2_trans_no_trans_norm_no_norm.csv" \
    "${BASE_DIR}/Paper_Figures/One_shifting_sample_eg/${sample_name}_INPUT_RNA_VAE_col_v1_reverse_pearson_trans_${trans}_norm_${norm}_v2_trans_no_trans_norm_no_norm.png" \
    "Pearson" \
    "${norm}" \
    "${trans}" \
    "${sample_name}"

Rscript ../One_Shifting/R/figures/sample_scatter.R \
    "${BASE_DIR}/${folder_name}/${sample_name}_INPUT_RNA/VAE/given_${sample_name}_ATAC_no_norm_no_trans/Figures_col_v1_raw__v2_raw__recon_revtrans_revnorm/g_spearman_v1_recon_reverse_trans_norm_col_scatter_v1_trans_${trans}_norm_${norm}_v2_trans_no_trans_norm_no_norm.csv" \
    "${BASE_DIR}/Paper_Figures/One_shifting_sample_eg/${sample_name}_INPUT_RNA_VAE_col_v1_reverse_spearman_trans_${trans}_norm_${norm}_v2_trans_no_trans_norm_no_norm.png" \
    "Spearman" \
    "${norm}" \
    "${trans}" \
    "${sample_name}"

# Plot the scatter combination summary
Rscript ../One_Shifting/R/figures/sample_trans_norm_combination.R \
    "${BASE_DIR}/${folder_name}/${sample_name}_INPUT_RNA/VAE/given_${sample_name}_ATAC_no_norm_no_trans/Figures_col_v1_raw__v2_raw__recon_revtrans_revnorm/plots_summary_pearson_col_gene_v1_recon_reverse_trans_norm.csv" \
    "${BASE_DIR}/Paper_FiguresOne_shifting_sample_eg/${sample_name}_VAE_pearson_summary.png" \
    "Pearson" \
    "${sample_name}"    

Rscript ../One_Shifting/R/figures/sample_trans_norm_combination.R \
    "${BASE_DIR}/${folder_name}/${sample_name}_INPUT_RNA/VAE/given_${sample_name}_ATAC_no_norm_no_trans/Figures_col_v1_raw__v2_raw__recon_revtrans_revnorm/plots_summary_spearman_col_gene_v1_recon_reverse_trans_norm.csv" \
    "${BASE_DIR}/Paper_Figures/One_shifting_sample_eg/${sample_name}_VAE_spearman_summary.png" \
    "Spearman" \
    "${sample_name}"    


# Average bubble plot
tsv="${BASE_DIR}/HCA_input_sample_pairs.tsv,${BASE_DIR}/ENCODE_input_sample_pairs.tsv,${BASE_DIR}/10x_input_sample_pairs.tsv"
Rscript ../One_Shifting/R/figures/average_rna_atac_bubble_plot.R \
    "$BASE_DIR" \
    "$tsv" \
    "all" \
    "col" \
    "${BASE_DIR}/Paper_Figures/One_shifting_sample_eg"


# Figure -- baseline
BASE_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"
for modality in RNA ATAC; do
    for method in VAE DCA_mse Transformer_denoise; do
        for folder_name in HCA ENCODE 10x; do
            Rscript ../One_Shifting/R/figures/run_paper_figure_baseline_vs_method.R \
                "$BASE_DIR" \
                "$folder_name" \
                "$method" \
                "$modality" \
                "${BASE_DIR}/Paper_Figures/baseline_compare/${modality}/${method}" \
                "pearson,spearman" \
                "${BASE_DIR}/${folder_name}_input_sample_pairs.tsv" \
                || break
        done
    done
done


## Plot with baseline (single sample)
Rscript ../One_Shifting/R/figures/sample_scvi_dca_saver.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/d_pearson_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000000_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/scVI/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/DCA/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/SAVER/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_baseline_pearson_summary.png" \
    "Pearson" \
    "KTBpool11"      


Rscript ../One_Shifting/R/figures/sample_scvi_dca_saver.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/g_spearman_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/scVI/Figures_col/g_spearman_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/DCA/Figures_col/g_spearman_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/SAVER/Figures_col/g_spearman_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_baseline_spearman_summary.png" \
    "Spearman" \
    "KTBpool11"           

python /dcs10/hongkai/data/yhu1/One_Shifting/myproject/rank_curve_real_data.py \
    --data_path1 "/dcs10/hongkai/data/yhu1/One_Shifting_Results/10x/10k_PBMC_Multiome_nextgem_Chromium_Controller_INPUT_ATAC/10k_PBMC_Multiome_nextgem_Chromium_Controller_ATAC/Count_Matrix_norm_by_no_norm.feather" \
    --data_path2 "/dcs10/hongkai/data/yhu1/One_Shifting_Results/10x/10k_PBMC_Multiome_nextgem_Chromium_Controller_INPUT_ATAC/10k_PBMC_Multiome_nextgem_Chromium_Controller_RNA/Count_Matrix_norm_by_no_norm.feather" \
    --out /dcs10/hongkai/data/yhu1/One_Shifting_Results/10x/10k_PBMC_Multiome_nextgem_Chromium_Controller_INPUT_ATAC/AAA_rank_curve_ATAC_norm_no_norm.csv



# Line plot
module load conda_R
N_CORES=16 Rscript ../One_Shifting/R/check_calculate_sparsity.R \
    ${BASE_DIR}/HCA_input_sample_pairs.tsv \
    ${BASE_DIR}/ENCODE_input_sample_pairs.tsv \
    ${BASE_DIR}/10x_input_sample_pairs.tsv \
    ${BASE_DIR}/sample_sparsity.csv


N_CORES=16 Rscript ../One_Shifting/R/figures/line_plot_1_add_stats_to_sparsity.R \
    ${BASE_DIR}/HCA_input_sample_pairs.tsv \
    ${BASE_DIR}/ENCODE_input_sample_pairs.tsv \
    ${BASE_DIR}/10x_input_sample_pairs.tsv \
    ${BASE_DIR}/sample_sparsity.csv \
    VAE,DCA_mse,scVI_mse,Transformer_denoise

Methods=("VAE" "DCA_mse" "scVI_mse" "Transformer_denoise")

for Method in "${Methods[@]}"; do
    Rscript ../One_Shifting/R/figures/line_plot_2_plot_scatter.R \
        "${BASE_DIR}/sample_sparsity.csv" \
        "$Method" \
        "RNA,ATAC" \
        "pearson,spearman" \
        "performance,residual" \
        "${BASE_DIR}/Paper_Figures/sparsity"
done




