

module load conda_R

save_dir="../One_Shifting_Results"
mkdir -p $save_dir

Rscript ../One_Shifting/R/load_seurat_pbmc.R \
    $save_dir