
source("../One_Shifting/R/split_tf_hvg_data.R")

args <- commandArgs(trailingOnly = TRUE)
expr_df_path <- args[1] 
output_dir <- args[2]
tf_names <- args[3]
hvg_names <- args[4]
test_frac <- as.numeric(args[5])
val_frac <- as.numeric(args[6])
seed <- as.integer(args[7])

split_tf_hvg_data(expr_df_path, output_dir, tf_names, hvg_names, test_frac, val_frac, seed)
