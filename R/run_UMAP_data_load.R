source("../One_Shifting/R/load_UMAP_cbmc.R")
source("../One_Shifting/R/load_UMAP_zheng_pbmc.R")

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1] 

# Cite-seq
save_dir <- paste0(out_dir, "/Cite_seq/orig_data")
load_UMAP_cbmc(save_dir)

# FACS-sorting
save_dir <- paste0(out_dir, "/Zheng_pbmcs/orig_data")
load_UMAP_zheng_pbmc(save_dir)

