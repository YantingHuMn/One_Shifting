
source("../One_Shifting/R/select_tf_hvg.R")
args <- commandArgs(trailingOnly = TRUE)

expr_df_path <- args[1] 
output_dir <- args[2]
n_hvg <- as.integer(args[3])

select_tf_hvg(expr_df_path = expr_df_path, output_dir = output_dir, n_hvg = n_hvg)