
cd /dcs10/hongkai/data/yhu1/One_Shifting

module load conda_R

BASE_DIR="/dcs11/hongkai/data/yhu1/One_Shifting_Results"

for method in VAE DCA_mse Transformer_denoise; do
    for performance_type in absolute relative; do
        Rscript ../One_Shifting/R/figures/plot_dropout_sparsity_performance.R \
            "$BASE_DIR" \
            "./data/dropout_samples.tsv" \
            "$method" \
            "${BASE_DIR}/Paper_Figures/dropout_sparsity/${method}" \
            "pearson,spearman" \
            "" \
            "${performance_type}"
    done
done