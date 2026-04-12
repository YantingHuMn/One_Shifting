source(file.path(dirname(sys.frame(1)$ofile), "count_matrix_function_with_qc.R"))
source(file.path(dirname(sys.frame(1)$ofile), "qc.R"))

source(file.path(dirname(sys.frame(1)$ofile), "cut_df_pct.R"))
source(file.path(dirname(sys.frame(1)$ofile), "get_filtered_samples.R"))
source(file.path(dirname(sys.frame(1)$ofile), "match_bam_files.R"))
source(file.path(dirname(sys.frame(1)$ofile), "build_and_normalize.R"))
source(file.path(dirname(sys.frame(1)$ofile), "apply_transformations.R"))


# Main Function: filter col → remove 0 row → cut col → normalize
run_pipeline <- function(path1, path2, out_dir, norm_factor1, norm_factor2, filtered_percentile = 0.25,
                         zero_pct_max = NULL, pearson_min = NULL, spearman_min = NULL, lib_size_min = NULL, histone_only = FALSE) {

    # QC based on path1
    filtered_crf <- get_filtered_samples(path1, filtered_percentile)

    if (histone_only) {
        histone_pattern <- "^H[0-9]"
        filtered_crf <- filtered_crf[sapply(filtered_crf, function(x) {
            parts <- strsplit(x, "-")[[1]]
            left <- parts[1]
            right <- paste(parts[-1], collapse = "-")
            grepl(histone_pattern, left) & grepl(histone_pattern, right)
        })]
        cat("After histone filter:", length(filtered_crf), "CRF pairs remaining\n")
    }

    # V1: Generate original count matrix and remove 0行
    cat("\n=== Processing V1 (initial) ===\n")
    v1_bam_files <- match_bam_files(path1, filtered_crf)
    v1_init <- build_and_normalize(v1_bam_files, out_dir, basename(path1), "no_norm", keep_rows = NULL, keep_cols = NULL)

    # cut columns
    keep_cols <- NULL
    if (!is.null(zero_pct_max) || !is.null(pearson_min) || !is.null(spearman_min) || !is.null(lib_size_min)) {
        cat("\n=== Processing V2 (for cut reference) ===\n")
        v2_bam_files <- match_bam_files(path2, filtered_crf)
        v2_init <- build_and_normalize(v2_bam_files, out_dir, basename(path2), "no_norm",
                                       keep_rows = v1_init$keep_rows, keep_cols = NULL)

        v1_mat <- as.matrix(v1_init$count_data[, -1])
        v2_mat <- as.matrix(v2_init$count_data[, -1])

        cat("\n=== Cutting columns ===\n")
        keep_cols <- cut_df_pct(v1_mat, v2_mat, zero_pct_max, pearson_min, spearman_min, lib_size_min)
    }

    # V1: Use keep_rows + keep_cols normalize and save again
    cat("\n=== Processing V1 (final) ===\n")
    v1_result <- build_and_normalize(v1_bam_files, out_dir, basename(path1), norm_factor1,
                                     keep_rows = v1_init$keep_rows, keep_cols = keep_cols)

    # V2: Use V1's keep_rows and keep_cols
    cat("\n=== Processing V2 (final) ===\n")
    if (is.null(keep_cols)) v2_bam_files <- match_bam_files(path2, filtered_crf)
    v2_result <- build_and_normalize(v2_bam_files, out_dir, basename(path2), norm_factor2,
                                     keep_rows = v1_init$keep_rows, keep_cols = keep_cols)

    cat("\n=== Pipeline complete ===\n")
    return(list(v1 = v1_result$count_data, v2 = v2_result$count_data))
}