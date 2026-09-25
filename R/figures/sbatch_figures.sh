
module load conda_R

# Find the typical sample by median rank
# rm -f /dcs07/hongkai/data/yhu1/One_Shifting_Results/col_reverse_representative_sample_ranking.csv
# Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/typical_sample_find.R 

# Plot the method specific scatter plot
Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_scatter.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/d_pearson_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000000_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_INPUT_RNA_VAE_col_v1_reverse_pearson_trans_log2(count+2)_norm_1000000_v2_trans_no_trans_norm_no_norm.png" \
    "Pearson" \
    "1 million" \
    "log2(count + 2)" \
    "KTBpool11"

Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_scatter.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/g_spearman_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_INPUT_RNA_VAE_col_v1_reverse_spearman_trans_log2(count+2)_norm_1000_v2_trans_no_trans_norm_no_norm.png" \
    "Spearman" \
    "1000" \
    "log2(count + 2)" \
    "KTBpool11"

# Plot the scatter combination summary
Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_trans_norm_combination.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/plots_summary_pearson_col_gene_v1_reverse.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_VAE_pearson_summary.png" \
    "Pearson" \
    "KTBpool11"    

Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_trans_norm_combination.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/plots_summary_spearman_col_gene_v1_reverse.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_VAE_spearman_summary.png" \
    "Spearman" \
    "KTBpool11"    

# Average bubble plot
cd /dcs10/hongkai/data/yhu1/One_Shifting 
BASE_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"
tsv="${BASE_DIR}/HCA_input_sample_pairs.tsv,${BASE_DIR}/ENCODE_input_sample_pairs.tsv,/dcs10/hongkai/data/yhu1/One_Shifting_Results/10x_input_sample_pairs.tsv"
module load conda_R

Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/average_rna_atac_bubble_plot.R \
    "$BASE_DIR" \
    "$tsv" \
    "all" \
    "col" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures"


# Line plot
cd /dcs10/hongkai/data/yhu1/One_Shifting 
BASE_DIR="/dcs07/hongkai/data/yhu1/One_Shifting_Results"
module load conda_R
N_CORES=16 Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/check_calculate_sparsity.R \
    ${BASE_DIR}/HCA_input_sample_pairs.tsv \
    ${BASE_DIR}/ENCODE_input_sample_pairs.tsv \
    /dcs10/hongkai/data/yhu1/One_Shifting_Results/10x_input_sample_pairs.tsv \
    ${BASE_DIR}/sample_sparsity.csv


N_CORES=16 Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/line_plot_1_add_stats_to_sparsity.R \
    ${BASE_DIR}/HCA_input_sample_pairs.tsv \
    ${BASE_DIR}/ENCODE_input_sample_pairs.tsv \
    /dcs10/hongkai/data/yhu1/One_Shifting_Results/10x_input_sample_pairs.tsv \
    ${BASE_DIR}/sample_sparsity.csv \
    VAE,DCA_mse,scVI_mse,Transformer_denoise

Methods=("VAE" "DCA_mse" "scVI_mse" "Transformer_denoise")

for Method in "${Methods[@]}"; do
    Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/line_plot_2_plot_scatter.R \
        "${BASE_DIR}/sample_sparsity.csv" \
        "$Method" \
        "RNA,ATAC" \
        "pearson,spearman" \
        "performance,residual" \
        "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/sparsity"
done





# Plot with baseline
Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_scvi_dca_saver.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/d_pearson_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000000_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/scVI/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/DCA/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/SAVER/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_baseline_pearson_summary.png" \
    "Pearson" \
    "KTBpool11"      


Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_scvi_dca_saver.R \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/VAE/given_KTBpool11_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/g_spearman_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/scVI/Figures_col/g_spearman_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/DCA/Figures_col/g_spearman_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/SAVER/Figures_col/g_spearman_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_baseline_spearman_summary.png" \
    "Spearman" \
    "KTBpool11"      


# Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/figures/sample_scvi_dca_saver.R \
#     "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/PF_specimen_INPUT_RNA/VAE/given_PF_specimen_ATAC_no_norm_no_trans/Figures_col_v1_reverse_v2_no_trans/d_pearson_v1_reverse_col_scatter_v1_trans_log2(count+2)_norm_1000000_v2_trans_no_trans_norm_no_norm.csv" \
#     "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/scVI/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
#     "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/PF_specimen_INPUT_RNA/DCA/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
#     "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/KTBpool11_INPUT_RNA/SAVER/Figures_col/d_pearson_col_scatter_v1_trans_no_trans_norm_no_norm_v2_trans_no_trans_norm_no_norm.csv" \
#     "/dcs07/hongkai/data/yhu1/One_Shifting_Results/Paper_Figures/KTBpool11_baseline_pearson_summary.png" \
#     "Pearson" \
#     "PF_specimen"      


python /dcs10/hongkai/data/yhu1/One_Shifting/myproject/rank_curve_real_data.py \
    --data_path1 "/dcs10/hongkai/data/yhu1/One_Shifting_Results/10x/10k_PBMC_Multiome_nextgem_Chromium_Controller_INPUT_ATAC/10k_PBMC_Multiome_nextgem_Chromium_Controller_ATAC/Count_Matrix_norm_by_no_norm.feather" \
    --data_path2 "/dcs10/hongkai/data/yhu1/One_Shifting_Results/10x/10k_PBMC_Multiome_nextgem_Chromium_Controller_INPUT_ATAC/10k_PBMC_Multiome_nextgem_Chromium_Controller_RNA/Count_Matrix_norm_by_no_norm.feather" \
    --out /dcs10/hongkai/data/yhu1/One_Shifting_Results/10x/10k_PBMC_Multiome_nextgem_Chromium_Controller_INPUT_ATAC/AAA_rank_curve_ATAC_norm_no_norm.csv