source("../One_Shifting/R/plot_multiple_umap.R")
source("../One_Shifting/R/plot_ari_bubble.R")
source("../One_Shifting/R/plot_ari_bubble_trans.R")

args <- commandArgs(trailingOnly = TRUE)
read_dir <- args[1]
cell_type_reference_path <- args[2] 
data_path_df_path <- args[3]

if (length(args) >= 4) {
    clustering_methods <- strsplit(args[4], ",")[[1]]
} else {
    clustering_methods <- c("kmeans", "louvain", "leiden")
}

celltype_df <- read.csv(cell_type_reference_path)

# registry paths
data_path_df <- read.csv(data_path_df_path)
data_paths <- data_path_df$path
data_names <- paste0(data_path_df$method, "_", data_path_df$trans, "_", data_path_df$norm_factor)

# other paths
seurat_file <- list.files(
    path = paste0(read_dir, "/orig_data/"),
    pattern = ".*seurat_norm_transposed\\.feather$",
    full.names = TRUE
)
seurat_standard_norm <- seurat_file[1]

other_paths <- c(
    paste0(read_dir, "/normed_data/Count_Matrix_norm_by_no_norm.feather"),
    paste0(read_dir, "/normed_data/Count_Matrix_norm_by_1000.feather"),
    paste0(read_dir, "/normed_data/Count_Matrix_norm_by_10000.feather"),
    paste0(read_dir, "/normed_data/Count_Matrix_norm_by_100000.feather"),
    paste0(read_dir, "/normed_data/Count_Matrix_norm_by_1000000.feather"),
    paste0(read_dir, "/normed_data/Count_Matrix_norm_by_standardize.feather"),
    seurat_standard_norm,
    paste0(read_dir, "/scVI/reconstruct_scVI_trans_by_no_trans_norm_by_no_norm.feather"),
    paste0(read_dir, "/DCA/reconstruct_DCA_trans_by_no_trans_norm_by_no_norm.feather"),
    paste0(read_dir, "/SAVER/reconstruct_SAVER_trans_by_no_trans_norm_by_no_norm.feather")
)

other_names <- c(
  "Original",
    "Original (Norm 1K)",
    "Original (Norm 10K)",
    "Original (Norm 100K)",
    "Original (Norm 1M)",
    "Original (Standardize)",
    "Original (seurat standard normalize)",
    "scVI",
    "DCA",
    "SAVER"
)

data_paths <- c(data_paths, other_paths)
data_names <- c(data_names, other_names)


output_dir <- paste0(read_dir, "/plots")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if ("kmeans" %in% clustering_methods) {
result <- plot_multiple_umap(
  data_paths = data_paths,
  data_names = data_names,
  celltype_df = celltype_df,
  output_dir = output_dir,
  n_clusters = length(unique(na.omit(celltype_df$cell_type))),
  clustering_method = "kmeans",
  ncol = 5,
  width = 30
)
rm(result)
gc()
}

if ("louvain" %in% clustering_methods) {
result <- plot_multiple_umap(
  data_paths = data_paths,
  data_names = data_names,
  celltype_df = celltype_df,
  output_dir = output_dir,
  n_clusters = length(unique(na.omit(celltype_df$cell_type))),
  clustering_method = "louvain",
  ncol = 5,
  width = 30
)
rm(result)
gc()
}

if ("leiden" %in% clustering_methods) {
result <- plot_multiple_umap(
  data_paths = data_paths,
  data_names = data_names,
  celltype_df = celltype_df,
  output_dir = output_dir,
  n_clusters = length(unique(na.omit(celltype_df$cell_type))),
  clustering_method = "leiden",
  ncol = 5,
  width = 30
)
rm(result)
gc()
}


# plot_ari_bubble(output_dir)
plot_ari_bubble_trans(output_dir)