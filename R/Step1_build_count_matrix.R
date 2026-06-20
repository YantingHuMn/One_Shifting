source("../One_Shifting/R/run_build_norm_filter_pipeline.R")

args <- commandArgs(trailingOnly = TRUE)

path1 <- args[1]
path2 <- if (length(args) >= 2 && !is.na(args[2]) && args[2] != "") args[2] else NULL
out_dir <- args[3]
norm_factor1_string <- args[4]
norm_factor2_string <- if (length(args) >= 5 && !is.na(args[5]) && args[5] != "") args[5] else NULL
transpose <- if (length(args) >= 6 && !is.na(args[6]) && args[6] != "") as.logical(args[6]) else FALSE
lib_size_min <- if (length(args) >= 7 && !is.na(args[7]) && args[7] != "") args[7] else NULL
dropout_keep_par <- if (length(args) >= 8 && !is.na(args[8]) && args[8] != "") as.numeric(args[8]) else NULL

norm_factor1 <- unlist(strsplit(norm_factor1_string, ","))

if (!is.null(norm_factor2_string)) {
    norm_factor2 <- unlist(strsplit(norm_factor2_string, ","))
} else {
    norm_factor2 <- NULL
}

run_build_norm_filter_pipeline(path1 = path1, path2 = path2, out_dir = out_dir, norm_factor1 = norm_factor1, norm_factor2 = norm_factor2, filtered_percentile = 0.25, lib_size_min = lib_size_min, transpose = transpose, dropout_keep_par = dropout_keep_par, dropout_target = "v1")