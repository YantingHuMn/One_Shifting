
source("../One_Shifting/R/run_build_norm_filter_pipeline.R")

args <- commandArgs(trailingOnly = TRUE)

path1 <- args[1]
path2 <- args[2]
out_dir <- args[3]
norm_factor1_string <- args[4]
norm_factor2_string <- args[5]
transpose <- args[6]

norm_factor1 <- unlist(strsplit(norm_factor1_string, ","))
norm_factor2 <- unlist(strsplit(norm_factor2_string, ","))

run_build_norm_filter_pipeline(path1 = path1, path2 = path2, out_dir = out_dir, norm_factor1 = norm_factor1, norm_factor2 = norm_factor2, filtered_percentile = 0.25, lib_size_min = "p25", transpose = transpose)
