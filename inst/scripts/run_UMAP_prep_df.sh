
module load conda_R

OUT_DIR="../One_Shifting_Results/UMAP"
Rscript ../One_Shifting/R/run_UMAP_data_load.R \
    $OUT_DIR
