source("../One_Shifting/R/load_UMAP_cbmc.R")
source("../One_Shifting/R/load_UMAP_zheng_duo8.R")

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1] 
dir.create(out_dir, recursive = FALSE, showWarnings = FALSE)

# Cite-seq
save_dir <- paste0(out_dir, "/Cite_seq")
load_UMAP_cbmc(save_dir, use_hvg = TRUE)

# FACS-sorting
save_dir <- paste0(out_dir, "/Zheng_pbmcs")
load_UMAP_zheng_duo8(save_dir)
