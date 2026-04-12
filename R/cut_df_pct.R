cut_df_pct <- function(mat_num, mat2_num = NULL, zero_pct_max = NULL, pearson_min = NULL, spearman_min = NULL, lib_size_min = NULL) {
    keep <- rep(TRUE, ncol(mat_num))

    if (!is.null(zero_pct_max)) {
        zero_pct <- colMeans(mat_num == 0) * 100
        keep <- keep & (zero_pct <= zero_pct_max)
        cat(sprintf("Zero pct filter (<= %.2f%%): %d / %d columns kept\n",
                    zero_pct_max, sum(keep), length(keep)))
    }

    if (!is.null(lib_size_min)) {
        lib_sizes <- colSums(mat_num)
        keep <- keep & (lib_sizes >= lib_size_min)
        cat(sprintf("Library size filter (>= %.2f): %d / %d columns kept\n",
                    lib_size_min, sum(keep), length(keep)))
    }

    if (!is.null(pearson_min)) {
        if (is.null(mat2_num)) stop("pearson_min requires mat2_num")
        cor_vec <- sapply(seq_len(ncol(mat_num)), function(i) cor(mat_num[, i], mat2_num[, i], method = "pearson"))
        keep <- keep & (!is.na(cor_vec) & cor_vec >= pearson_min)
        cat(sprintf("Pearson filter (>= %.4f): %d / %d columns kept\n",
                    pearson_min, sum(keep), length(keep)))
    }

    if (!is.null(spearman_min)) {
        if (is.null(mat2_num)) stop("spearman_min requires mat2_num")
        cor_vec <- sapply(seq_len(ncol(mat_num)), function(i) cor(mat_num[, i], mat2_num[, i], method = "spearman"))
        keep <- keep & (!is.na(cor_vec) & cor_vec >= spearman_min)
        cat(sprintf("Spearman filter (>= %.4f): %d / %d columns kept\n",
                    spearman_min, sum(keep), length(keep)))
    }

    cat(sprintf("Cut columns: %d kept out of %d\n", sum(keep), length(keep)))
    return(keep)
}